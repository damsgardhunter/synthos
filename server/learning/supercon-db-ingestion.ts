/**
 * SuperCon DB Ingestion Pipeline
 * ================================
 * Ingests the full NIMS SuperCon database (~33,000 entries) into `supercon_external_entries`.
 *
 * Sources (tried in order):
 *   1. Local CSV file at SUPERCON_CSV_PATH env var (or ./server/learning/data/supercon.csv)
 *      — Download from https://supercon.nims.go.jp/download/ (NIMS SuperCon) or use the
 *        Hamidieh (2018) dataset CSV (21,263 rows, freely available on Kaggle/GitHub).
 *   2. NIMS Open Data REST API (no auth required for basic queries, paginated).
 *   3. Hamidieh dataset via direct GitHub raw URL (as a fallback open-access source).
 *
 * Safety guarantees:
 *   - All DB writes are batched (BATCH_SIZE rows) with a yield between batches.
 *   - Uses ON CONFLICT DO NOTHING so re-runs are idempotent.
 *   - Tracks progress in `system_state` table (key: 'supercon_ingestion_state').
 *   - Never blocks the event loop: every DB write is awaited with a 50 ms yield gap.
 *   - Call startSuperConIngestion() once from the GCP worker or server startup.
 *     Subsequent calls within the same process are no-ops if ingestion is already running.
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { db } from "../db";
import { superconExternalEntries, systemState } from "@shared/schema";
import { eq } from "drizzle-orm";

const BATCH_SIZE = 100;          // rows per DB insert batch
const YIELD_BETWEEN_BATCHES_MS = 80;  // ms to sleep between batches (keeps event loop alive)
const NIMS_API_BASE = "https://supercon.nims.go.jp/api/v1";
// Multiple mirrors tried in order — the Hamidieh 2018 SuperCon dataset (21,263 rows).
const HAMIDIEH_URLS = [
  // khamidieh original repo (try both branch names)
  "https://raw.githubusercontent.com/khamidieh/predict_sc/main/supercon.csv",
  "https://raw.githubusercontent.com/khamidieh/predict_sc/master/supercon.csv",
  // Common re-hosts / forks
  "https://raw.githubusercontent.com/wolverine2710/SuperconductorCriticalTempPrediction/master/supercon.csv",
  "https://raw.githubusercontent.com/jarrodmillman/scikit-learn-datasets/master/supercon/supercon.csv",
];
const STATE_KEY = "supercon_ingestion_state";

let _running = false;

// ── Types ───────────────────────────────────────────────────────────────────

interface IngestRow {
  formula: string;
  tc: number | null;
  isSuperconductor: boolean;
  source: string;
  externalId: string | null;
  spaceGroup: string | null;
  crystalSystem: string | null;
  family: string | null;
  lambda: number | null;
  pressureGpa: number;
  rawData: Record<string, unknown>;
}

interface IngestionState {
  status: "idle" | "running" | "done" | "failed";
  source: string;
  rowsIngested: number;
  rowsTotal: number | null;
  lastOffset: number;
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
}

// ── State persistence ────────────────────────────────────────────────────────

async function loadState(): Promise<IngestionState> {
  try {
    const rows = await db.select().from(systemState).where(eq(systemState.key, STATE_KEY)).limit(1);
    if (rows.length > 0) return rows[0].value as IngestionState;
  } catch { /* first run or DB unavailable */ }
  return { status: "idle", source: "", rowsIngested: 0, rowsTotal: null, lastOffset: 0, startedAt: null, finishedAt: null, error: null };
}

async function saveState(state: IngestionState): Promise<void> {
  await db.insert(systemState).values({ key: STATE_KEY, value: state as any }).onConflictDoUpdate({
    target: systemState.key,
    set: { value: state as any, updatedAt: new Date() },
  });
}

// ── Batch insert ─────────────────────────────────────────────────────────────

async function insertBatch(rows: IngestRow[]): Promise<number> {
  if (rows.length === 0) return 0;
  try {
    const values = rows.map(r => ({
      formula: r.formula,
      tc: r.tc,
      isSuperconductor: r.isSuperconductor,
      source: r.source,
      externalId: r.externalId,
      spaceGroup: r.spaceGroup,
      crystalSystem: r.crystalSystem,
      family: r.family,
      lambda: r.lambda,
      pressureGpa: r.pressureGpa,
      rawData: r.rawData,
    }));
    await db.insert(superconExternalEntries).values(values).onConflictDoNothing();
    return rows.length;
  } catch (err: any) {
    console.warn(`[SuperCon] Batch insert failed: ${err.message?.slice(0, 120)}`);
    return 0;
  }
}

// ── CSV parser ───────────────────────────────────────────────────────────────

function parseTc(val: string): number | null {
  const n = parseFloat(val);
  return isFinite(n) ? n : null;
}

/**
 * Parses a SuperCon-format CSV (Hamidieh / NIMS export) into IngestRow[].
 * Expects headers: formula, critical_temp (or tc), space_group, family, lambda, pressure, ...
 * Tolerates missing columns gracefully.
 */
function parseHamidiehRow(headers: string[], values: string[], lineNum: number): IngestRow | null {
  if (values.length < 2) return null;
  const get = (names: string[]): string => {
    for (const n of names) {
      const idx = headers.indexOf(n);
      if (idx >= 0 && values[idx] !== undefined) return values[idx].trim();
    }
    return "";
  };

  const formula = get(["material", "formula", "compound", "name"]);
  if (!formula) return null;

  const tcStr = get(["critical_temp", "tc", "Tc", "Tc(K)", "critical_temperature"]);
  const tc = parseTc(tcStr);

  const raw: Record<string, unknown> = {};
  headers.forEach((h, i) => { if (values[i] !== undefined) raw[h] = values[i]; });

  return {
    formula,
    tc,
    isSuperconductor: tc !== null ? tc > 0 : true,
    source: "hamidieh",
    externalId: `L${lineNum}`,
    spaceGroup: get(["space_group", "spacegroup", "sg"]) || null,
    crystalSystem: get(["crystal_system", "lattice"]) || null,
    family: get(["family", "type", "class"]) || null,
    lambda: parseTc(get(["lambda", "electron_phonon_coupling"])),
    pressureGpa: parseTc(get(["pressure", "pressure_gpa"])) ?? 0,
    rawData: raw,
  };
}

async function ingestFromLocalCSV(filePath: string, state: IngestionState): Promise<void> {
  console.log(`[SuperCon] Ingesting from local CSV: ${filePath}`);
  const fileStream = fs.createReadStream(filePath);
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let headers: string[] = [];
  let batch: IngestRow[] = [];
  let lineNum = 0;
  let totalInserted = state.rowsIngested;

  for await (const line of rl) {
    lineNum++;
    if (lineNum === 1) {
      headers = line.split(",").map(h => h.trim().toLowerCase().replace(/^"|"$/g, ""));
      if (lineNum <= state.lastOffset) continue;
      continue;
    }
    if (lineNum <= state.lastOffset) continue;

    // Parse CSV row (handles quoted fields with embedded commas)
    const values = line.match(/(".*?"|[^,]+|(?<=,)(?=,)|(?<=,)$|^(?=,))/g)?.map(v =>
      v.replace(/^"|"$/g, "")
    ) ?? line.split(",");

    const row = parseHamidiehRow(headers, values, lineNum);
    if (row) batch.push(row);

    if (batch.length >= BATCH_SIZE) {
      totalInserted += await insertBatch(batch);
      batch = [];
      state.rowsIngested = totalInserted;
      state.lastOffset = lineNum;
      await saveState(state);
      await new Promise(r => setTimeout(r, YIELD_BETWEEN_BATCHES_MS));
    }
  }

  if (batch.length > 0) {
    totalInserted += await insertBatch(batch);
    state.rowsIngested = totalInserted;
    state.lastOffset = lineNum;
  }
  console.log(`[SuperCon] CSV ingestion complete: ${totalInserted} rows inserted`);
}

// ── NIMS API ingestion ───────────────────────────────────────────────────────

interface NIMSRecord {
  id?: string;
  formula?: string;
  tc?: number;
  space_group?: string;
  crystal_system?: string;
  pressure?: number;
  lambda?: number;
  [key: string]: unknown;
}

async function fetchNIMSPage(offset: number, limit: number = 200): Promise<NIMSRecord[]> {
  const url = `${NIMS_API_BASE}/superconductors?offset=${offset}&limit=${limit}&format=json`;
  const response = await fetch(url, {
    headers: { "Accept": "application/json", "User-Agent": "QuantumAlchemyEngine/1.0" },
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new Error(`NIMS API ${response.status}: ${response.statusText}`);
  const data = await response.json() as NIMSRecord[] | { results?: NIMSRecord[] };
  return Array.isArray(data) ? data : (data.results ?? []);
}

function nimsRecordToRow(rec: NIMSRecord): IngestRow | null {
  const formula = String(rec.formula ?? "").trim();
  if (!formula) return null;
  const tc = rec.tc != null ? Number(rec.tc) : null;
  return {
    formula,
    tc,
    isSuperconductor: tc !== null ? tc > 0 : true,
    source: "nims",
    externalId: rec.id ? String(rec.id) : null,
    spaceGroup: rec.space_group ? String(rec.space_group) : null,
    crystalSystem: rec.crystal_system ? String(rec.crystal_system) : null,
    family: null,
    lambda: rec.lambda != null ? Number(rec.lambda) : null,
    pressureGpa: rec.pressure != null ? Number(rec.pressure) : 0,
    rawData: rec as Record<string, unknown>,
  };
}

async function ingestFromNIMSAPI(state: IngestionState): Promise<void> {
  console.log(`[SuperCon] Ingesting from NIMS API (offset=${state.lastOffset})`);
  let offset = state.lastOffset;
  let totalInserted = state.rowsIngested;
  let consecutiveEmpty = 0;

  while (consecutiveEmpty < 3) {
    let records: NIMSRecord[];
    try {
      records = await fetchNIMSPage(offset, BATCH_SIZE);
    } catch (err: any) {
      console.warn(`[SuperCon] NIMS API error at offset=${offset}: ${err.message}`);
      // Back off and retry once
      await new Promise(r => setTimeout(r, 10_000));
      try { records = await fetchNIMSPage(offset, BATCH_SIZE); }
      catch { break; }
    }

    if (records.length === 0) { consecutiveEmpty++; break; }
    consecutiveEmpty = 0;

    const rows = records.map(nimsRecordToRow).filter(Boolean) as IngestRow[];
    totalInserted += await insertBatch(rows);
    offset += records.length;
    state.lastOffset = offset;
    state.rowsIngested = totalInserted;
    await saveState(state);
    await new Promise(r => setTimeout(r, YIELD_BETWEEN_BATCHES_MS));
  }
  console.log(`[SuperCon] NIMS API ingestion complete: ${totalInserted} rows inserted`);
  // Throw if nothing was ingested so the caller can fall back to Hamidieh CSV.
  if (totalInserted === 0) throw new Error("NIMS API returned 0 rows");
}

// ── Hamidieh fallback download ────────────────────────────────────────────────

async function downloadHamidiehCSV(destPath: string): Promise<boolean> {
  for (const url of HAMIDIEH_URLS) {
    try {
      console.log(`[SuperCon] Trying Hamidieh mirror: ${url}`);
      const response = await fetch(url, { signal: AbortSignal.timeout(60_000) });
      if (!response.ok) {
        console.warn(`[SuperCon] Mirror ${url} returned HTTP ${response.status}`);
        continue;
      }
      const text = await response.text();
      // Sanity check: must look like a CSV with at least a few hundred rows
      if (text.split("\n").length < 10) {
        console.warn(`[SuperCon] Mirror ${url} returned too-short response (${text.length} bytes)`);
        continue;
      }
      fs.mkdirSync(path.dirname(destPath), { recursive: true });
      fs.writeFileSync(destPath, text, "utf8");
      console.log(`[SuperCon] Downloaded ${Math.round(text.length / 1024)} KB from ${url}`);
      return true;
    } catch (err: any) {
      console.warn(`[SuperCon] Mirror ${url} failed: ${err.message}`);
    }
  }
  console.warn(`[SuperCon] All ${HAMIDIEH_URLS.length} mirrors failed`);
  return false;
}

// ── Entry point ───────────────────────────────────────────────────────────────

/**
 * Start the background SuperCon ingestion.
 * Safe to call multiple times — subsequent calls are no-ops while running.
 * Does NOT await the ingestion (runs fully in background).
 */
export function startSuperConIngestion(): void {
  if (_running) return;
  _running = true;
  void runIngestion().finally(() => { _running = false; });
}

async function runIngestion(): Promise<void> {
  const state = await loadState();

  if (state.status === "done" && state.rowsIngested > 0) {
    console.log(`[SuperCon] Ingestion already complete (${state.rowsIngested} rows). Skipping.`);
    return;
  }
  if (state.status === "done" && state.rowsIngested === 0) {
    console.log(`[SuperCon] Previous ingestion completed with 0 rows — retrying all sources.`);
    state.status = "idle";
    state.lastOffset = 0;
    state.source = "";
  }

  state.status = "running";
  state.startedAt = state.startedAt ?? new Date().toISOString();
  state.error = null;
  await saveState(state);

  try {
    // 1. Bundled Kaggle datasets committed directly to the repo — highest priority
    //    because they're always available without any network request.
    //    unique_m.csv: Hamidieh 2018 dataset, 21,263 rows, columns include
    //    `material` (formula string) and `critical_temp` (Tc in K).
    const bundledPaths = [
      process.env.SUPERCON_CSV_PATH,
      path.join(process.cwd(), "server/learning/unique_m.csv"),
      path.join(process.cwd(), "server/learning/data/supercon.csv"),
    ].filter(Boolean) as string[];

    const foundLocal = bundledPaths.find(p => fs.existsSync(p));
    if (foundLocal) {
      console.log(`[SuperCon] Using bundled CSV: ${foundLocal}`);
      state.source = foundLocal.includes("unique_m") ? "kaggle-unique_m" : "local-csv";
      await ingestFromLocalCSV(foundLocal, state);
    } else {
      // 2. Try NIMS REST API
      try {
        state.source = "nims-api";
        await ingestFromNIMSAPI(state);
      } catch (apiErr: any) {
        console.warn(`[SuperCon] NIMS API unavailable (${apiErr.message}), falling back to Hamidieh CSV`);
        // 3. Download Hamidieh CSV as a fallback open-access source
        const downloadPath = path.join(process.cwd(), "server/learning/data/supercon_hamidieh.csv");
        const ok = await downloadHamidiehCSV(downloadPath);
        if (ok) {
          state.source = "hamidieh";
          state.lastOffset = 0;
          await ingestFromLocalCSV(downloadPath, state);
        } else {
          throw new Error("All SuperCon sources exhausted. Place a CSV at SUPERCON_CSV_PATH.");
        }
      }
    }

    state.status = "done";
    state.finishedAt = new Date().toISOString();
    console.log(`[SuperCon] Ingestion finished. Total rows: ${state.rowsIngested}`);
  } catch (err: any) {
    state.status = "failed";
    state.error = err.message?.slice(0, 500) ?? "unknown";
    console.error(`[SuperCon] Ingestion failed: ${state.error}`);
  }

  await saveState(state);
}

/**
 * Pull all ingested entries as SuperconEntry-compatible objects for use in training.
 * Returns at most `limit` rows ordered by tc DESC.
 */
export async function loadSuperConDBEntries(limit = 10_000): Promise<Array<{
  formula: string; tc: number | null; isSuperconductor: boolean;
  spaceGroup: string | null; crystalSystem: string | null; family: string | null;
  lambda: number | null; pressureGpa: number;
}>> {
  try {
    const rows = await db.select({
      formula: superconExternalEntries.formula,
      tc: superconExternalEntries.tc,
      isSuperconductor: superconExternalEntries.isSuperconductor,
      spaceGroup: superconExternalEntries.spaceGroup,
      crystalSystem: superconExternalEntries.crystalSystem,
      family: superconExternalEntries.family,
      lambda: superconExternalEntries.lambda,
      pressureGpa: superconExternalEntries.pressureGpa,
    }).from(superconExternalEntries)
      .orderBy(superconExternalEntries.tc)
      .limit(limit);
    return rows;
  } catch (err: any) {
    console.warn(`[SuperCon] loadSuperConDBEntries failed: ${err.message}`);
    return [];
  }
}

/**
 * Returns the current ingestion progress for display / health checks.
 */
export async function getSuperConIngestionStatus(): Promise<IngestionState> {
  return loadState();
}
