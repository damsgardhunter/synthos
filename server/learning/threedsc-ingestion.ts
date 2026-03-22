/**
 * 3DSC Dataset Ingestion Pipeline
 * ================================
 * Ingests the 3DSC_MP superconductor dataset (aimat-lab/3DSC on GitHub) into
 * `supercon_external_entries`. This dataset contains ~2,000+ experimentally
 * confirmed superconductors matched to Materials Project DFT structures, with
 * rich structural features: band gap, formation energy, e_above_hull, density,
 * lattice parameters, space group, crystal system, and magnetic type.
 *
 * Sources (tried in order):
 *   1. Local CSV at THREEDSC_CSV_PATH env var
 *      (or server/learning/data/3dsc_mp.csv)
 *   2. GitHub raw download (aimat-lab/3DSC, several paths tried)
 *
 * Safety guarantees (matches jarvis-ingestion.ts / supercon-db-ingestion.ts):
 *   - Batched writes (BATCH_SIZE rows) with YIELD_MS gap between batches.
 *   - ON CONFLICT DO NOTHING — idempotent, safe to re-run.
 *   - Progress persisted in system_state table (key: '3dsc_mp_ingestion').
 *   - Single-instance guard: subsequent startThreeDSCIngestion() calls are no-ops.
 *   - Multi-line quoted fields (e.g. bibtex) handled by a streaming state-machine
 *     parser — not a simple readline split.
 */

import * as fs   from "fs";
import * as path from "path";
import { db } from "../db";
import { superconExternalEntries, systemState } from "@shared/schema";
import { eq } from "drizzle-orm";

const BATCH_SIZE = 100;
const YIELD_MS   = 80;
const STATE_KEY  = "3dsc_mp_ingestion";
const SOURCE_TAG = "3dsc-mp";

// GitHub download URLs to try. Confirmed path as of 2026-03:
//   superconductors_3D/data/final/MP/3DSC_MP.csv  (main branch only — no master)
const GITHUB_URLS = [
  "https://raw.githubusercontent.com/aimat-lab/3DSC/main/superconductors_3D/data/final/MP/3DSC_MP.csv",
  // Fallbacks in case the repo is reorganised
  "https://raw.githubusercontent.com/aimat-lab/3DSC/main/data/final/MP/3DSC_MP.csv",
  "https://raw.githubusercontent.com/aimat-lab/3DSC/main/superconductors_3D/data/final/3DSC_MP.csv",
];

let _running = false;

// ── Types ─────────────────────────────────────────────────────────────────────

interface IngestRow {
  formula:         string;
  tc:              number | null;
  isSuperconductor: boolean;
  source:          string;
  externalId:      string | null;
  spaceGroup:      string | null;
  crystalSystem:   string | null;
  family:          string | null;
  lambda:          number | null;
  pressureGpa:     number;
  rawData:         Record<string, unknown>;
}

interface IngestionState {
  status:       "idle" | "running" | "done" | "failed";
  rowsIngested: number;
  rowsTotal:    number | null;
  lastOffset:   number;   // row index (0-based, excluding header)
  finishedAt:   string | null;
  error:        string | null;
}

// ── State helpers ──────────────────────────────────────────────────────────────

async function loadState(): Promise<IngestionState> {
  try {
    const rows = await db.select().from(systemState).where(eq(systemState.key, STATE_KEY)).limit(1);
    if (rows.length > 0) return rows[0].value as IngestionState;
  } catch { /* first run */ }
  return { status: "idle", rowsIngested: 0, rowsTotal: null, lastOffset: 0, finishedAt: null, error: null };
}

async function saveState(state: IngestionState): Promise<void> {
  try {
    await db.insert(systemState).values({ key: STATE_KEY, value: state as any }).onConflictDoUpdate({
      target: systemState.key,
      set: { value: state as any, updatedAt: new Date() },
    });
  } catch { /* non-fatal */ }
}

function delay(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

// ── Streaming state-machine CSV parser ─────────────────────────────────────────
//
// The 3DSC CSV has multi-line quoted fields (e.g. bibtex citations), so a simple
// readline split would corrupt those rows. This generator reads the file in 64 KB
// chunks and emits one string[] per complete CSV record.

async function* parseCSVFile(filePath: string): AsyncGenerator<string[]> {
  const CHUNK = 64 * 1024;
  const fd    = fs.openSync(filePath, "r");
  const buf   = Buffer.allocUnsafe(CHUNK);

  // State machine variables — shared across chunks so multi-line quoted fields
  // (e.g. bibtex) are handled correctly at chunk boundaries.
  let inQuote = false;
  let fields: string[] = [];
  let current = "";

  try {
    let bytesRead: number;
    do {
      bytesRead = fs.readSync(fd, buf, 0, CHUNK, null);
      if (bytesRead === 0) break;
      const text = buf.subarray(0, bytesRead).toString("utf8");

      for (let i = 0; i < text.length; i++) {
        const ch = text[i];

        if (ch === '"') {
          if (inQuote && text[i + 1] === '"') {
            current += '"';
            i++;
          } else {
            inQuote = !inQuote;
          }
        } else if (ch === ',' && !inQuote) {
          fields.push(current);
          current = "";
        } else if (ch === '\r' && !inQuote) {
          if (text[i + 1] === '\n') i++;
          fields.push(current);
          current = "";
          if (fields.some(f => f.length > 0)) yield fields;
          fields = [];
        } else if (ch === '\n' && !inQuote) {
          fields.push(current);
          current = "";
          if (fields.some(f => f.length > 0)) yield fields;
          fields = [];
        } else {
          current += ch;
        }
      }
    } while (bytesRead === CHUNK);

    // Flush final record (file not ending with newline)
    if (current.length > 0 || fields.length > 0) {
      fields.push(current);
      if (fields.some(f => f.length > 0)) yield fields;
    }
  } finally {
    fs.closeSync(fd);
  }
}

// ── Float parsing ─────────────────────────────────────────────────────────────

const PG_REAL_MIN =  1.18e-38;
const PG_REAL_MAX =  3.4e+38;

function parseFloat_(val: string): number | null {
  const n = parseFloat(val);
  if (!isFinite(n)) return null;
  if (n !== 0 && Math.abs(n) < PG_REAL_MIN) return 0;
  if (Math.abs(n) > PG_REAL_MAX) return null;
  return n;
}

// ── Row parser ────────────────────────────────────────────────────────────────

function parse3DSCRow(
  headers: string[],
  values:  string[],
  rowIdx:  number,
): IngestRow | null {
  if (values.length < 5) return null;

  const get = (...names: string[]): string => {
    for (const n of names) {
      const idx = headers.indexOf(n);
      if (idx >= 0 && values[idx] !== undefined) return values[idx].trim();
    }
    return "";
  };

  // Skip excluded rows (e.g. bad CIF, duplicate, structural anomaly)
  const exclusion = get("reason for exclusion");
  if (exclusion && exclusion.toLowerCase() !== "nan" && exclusion !== "") return null;

  const formula = get("formula_sc");
  if (!formula || formula.length < 1) return null;

  const tcStr = get("tc");
  const tc = parseFloat_(tcStr);

  // Structural features from the MP-matched structure
  const spaceGroup   = get("spacegroup_2")    || null;
  const crystalSys   = get("crystal_system_2") || null;
  const family       = get("sc_class")         || null;
  const materialId   = get("material_id_2")    || `3DSC-L${rowIdx}`;
  const bandGap      = parseFloat_(get("band_gap_2"));
  const formE        = parseFloat_(get("formation_energy_per_atom_2"));
  const eHull        = parseFloat_(get("e_above_hull_2"));
  const density      = parseFloat_(get("density_2"));
  const cellVol      = parseFloat_(get("cell_volume_2"));
  const latA         = parseFloat_(get("lata_2"));
  const latB         = parseFloat_(get("latb_2"));
  const latC         = parseFloat_(get("latc_2"));
  const magType      = get("magnetic_type_2")  || null;
  const nSites       = parseFloat_(get("nsites_2"));
  const nElements    = parseFloat_(get("num_elements_sc"));
  const synthDoped   = get("synth_doped").toLowerCase() === "true";
  const scClass      = get("sc_class_unique_sc").toLowerCase() === "true";
  const chemComp     = get("chemical_composition_sc") || null;

  // Build rawData with rich structural features for GNN feature enrichment
  const rawData: Record<string, unknown> = {
    source_dataset: "3DSC_MP",
    material_id: materialId,
    synth_doped: synthDoped,
    sc_class_unique: scClass,
    chemical_composition: chemComp,
  };
  if (bandGap  !== null)  rawData["band_gap"]                   = bandGap;
  if (formE    !== null)  rawData["formation_energy_per_atom"]   = formE;
  if (eHull    !== null)  rawData["e_above_hull"]                = eHull;
  if (density  !== null)  rawData["density"]                     = density;
  if (cellVol  !== null)  rawData["cell_volume"]                 = cellVol;
  if (latA     !== null)  rawData["lata"]                        = latA;
  if (latB     !== null)  rawData["latb"]                        = latB;
  if (latC     !== null)  rawData["latc"]                        = latC;
  if (magType)            rawData["magnetic_type"]               = magType;
  if (nSites   !== null)  rawData["nsites"]                      = nSites;
  if (nElements !== null) rawData["num_elements"]                = nElements;

  // Crystal-system one-hot flags (useful downstream for feature engineering)
  for (const cs of ["cubic", "hexagonal", "monoclinic", "orthorhombic", "tetragonal", "triclinic", "trigonal"]) {
    const v = parseFloat_(get(cs));
    if (v !== null) rawData[`cs_${cs}`] = v;
  }

  return {
    formula,
    tc,
    isSuperconductor: tc !== null ? tc > 0 : true,
    source: SOURCE_TAG,
    externalId: materialId,
    spaceGroup,
    crystalSystem: crystalSys,
    family,
    lambda: null,   // 3DSC does not include λ
    pressureGpa: 0, // MP structures are ambient-pressure
    rawData,
  };
}

// ── Batch insert ───────────────────────────────────────────────────────────────

async function insertBatch(rows: IngestRow[]): Promise<number> {
  if (rows.length === 0) return 0;
  try {
    const values = rows.map(r => ({
      formula:          r.formula,
      tc:               r.tc,
      isSuperconductor: r.isSuperconductor,
      source:           r.source,
      externalId:       r.externalId,
      spaceGroup:       r.spaceGroup,
      crystalSystem:    r.crystalSystem,
      family:           r.family,
      lambda:           r.lambda,
      pressureGpa:      r.pressureGpa,
      rawData:          r.rawData,
    }));
    await db.insert(superconExternalEntries).values(values).onConflictDoNothing();
    return rows.length;
  } catch (err: any) {
    console.warn(`[3DSC] Batch insert failed: ${err.message?.slice(0, 120)}`);
    return 0;
  }
}

// ── File ingestion ─────────────────────────────────────────────────────────────

async function ingestFile(filePath: string, state: IngestionState): Promise<void> {
  console.log(`[3DSC] Ingesting from ${path.basename(filePath)}`);

  let headers: string[] = [];
  let batch: IngestRow[] = [];
  let rowIdx = 0;           // 0-based data row index (excludes header)
  let totalInserted = state.rowsIngested;

  for await (const fields of parseCSVFile(filePath)) {
    // First record is the header — but skip any leading comment lines (start with #)
    if (headers.length === 0) {
      if (fields[0]?.trimStart().startsWith("#")) continue;
      headers = fields.map(h => h.toLowerCase().replace(/^"|"$/g, "").trim());
      continue;
    }

    rowIdx++;
    if (rowIdx <= state.lastOffset) continue; // resume from checkpoint

    const row = parse3DSCRow(headers, fields, rowIdx);
    if (row) batch.push(row);

    if (batch.length >= BATCH_SIZE) {
      totalInserted += await insertBatch(batch);
      batch = [];
      state.rowsIngested = totalInserted;
      state.lastOffset = rowIdx;
      await saveState(state);
      await delay(YIELD_MS);
    }
  }

  // Flush remaining
  if (batch.length > 0) {
    totalInserted += await insertBatch(batch);
  }

  state.rowsIngested = totalInserted;
  state.rowsTotal    = rowIdx;
  state.lastOffset   = rowIdx;
  console.log(`[3DSC] Ingestion complete — ${totalInserted.toLocaleString()} rows inserted (${rowIdx.toLocaleString()} data rows parsed)`);
}

// ── GitHub download ────────────────────────────────────────────────────────────

async function downloadFromGitHub(destPath: string): Promise<boolean> {
  for (const url of GITHUB_URLS) {
    try {
      console.log(`[3DSC] Trying: ${url}`);
      const res = await fetch(url, {
        headers: { "User-Agent": "QuantumAlchemyEngine/1.0" },
        signal: AbortSignal.timeout(120_000),
      });
      if (!res.ok) {
        console.warn(`[3DSC] HTTP ${res.status} from ${url}`);
        continue;
      }
      const text = await res.text();
      const lines = text.split("\n").length;
      if (lines < 5) {
        console.warn(`[3DSC] Response too short (${lines} lines) from ${url}`);
        continue;
      }
      fs.mkdirSync(path.dirname(destPath), { recursive: true });
      fs.writeFileSync(destPath, text, "utf8");
      console.log(`[3DSC] Downloaded ${(text.length / 1024).toFixed(0)} KB (${lines.toLocaleString()} lines) from ${url}`);
      return true;
    } catch (err: any) {
      console.warn(`[3DSC] Fetch failed for ${url}: ${err.message}`);
    }
  }
  return false;
}

// ── Entry point ────────────────────────────────────────────────────────────────

/**
 * Start background ingestion of the 3DSC_MP dataset.
 * Fire-and-forget — safe to call multiple times; subsequent calls are no-ops
 * while ingestion is running or already completed with data.
 */
export function startThreeDSCIngestion(): void {
  if (_running) return;
  _running = true;
  void runIngestion().finally(() => { _running = false; });
}

async function runIngestion(): Promise<void> {
  const state = await loadState();

  if (state.status === "done" && (state.rowsIngested ?? 0) > 0) {
    console.log(`[3DSC] Already ingested ${state.rowsIngested.toLocaleString()} rows — skipping`);
    return;
  }

  // If previous attempt completed with 0 rows, reset and retry
  if (state.status === "done" && state.rowsIngested === 0) {
    console.log("[3DSC] Previous run completed with 0 rows — retrying");
    state.status     = "idle";
    state.lastOffset = 0;
  }

  state.status = "running";
  state.error  = null;
  await saveState(state);

  try {
    // 1. Check local paths first (env var or well-known bundled paths)
    const candidatePaths = [
      process.env.THREEDSC_CSV_PATH,
      path.join(process.cwd(), "server/learning/data/3dsc_mp.csv"),
      path.join(process.cwd(), "server/learning/3dsc_mp.csv"),
    ].filter(Boolean) as string[];

    const localPath = candidatePaths.find(p => {
      try { return fs.existsSync(p) && fs.statSync(p).size > 1024; }
      catch { return false; }
    });

    if (localPath) {
      console.log(`[3DSC] Using local file: ${localPath}`);
      await ingestFile(localPath, state);
    } else {
      // 2. Download from GitHub
      const downloadPath = path.join(process.cwd(), "server/learning/data/3dsc_mp.csv");
      const ok = await downloadFromGitHub(downloadPath);
      if (!ok) {
        throw new Error(
          "All 3DSC GitHub URLs failed. " +
          "Clone https://github.com/aimat-lab/3DSC and set THREEDSC_CSV_PATH to the CSV, " +
          "or place it at server/learning/data/3dsc_mp.csv"
        );
      }
      state.lastOffset = 0; // fresh download — start from beginning
      await ingestFile(downloadPath, state);
    }

    state.status     = "done";
    state.finishedAt = new Date().toISOString();
  } catch (err: any) {
    state.status = "failed";
    state.error  = err.message?.slice(0, 500) ?? "unknown error";
    console.error(`[3DSC] Ingestion failed: ${state.error}`);
  }

  await saveState(state);
}

/**
 * Returns the current ingestion progress for monitoring / health checks.
 */
export async function getThreeDSCIngestionStatus(): Promise<IngestionState> {
  return loadState();
}
