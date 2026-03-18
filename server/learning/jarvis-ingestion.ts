/**
 * JARVIS Dataset Ingestion Pipeline
 * ==================================
 * Ingests pre-downloaded JARVIS CSV files into `supercon_external_entries`.
 *
 * CSV files are produced by scripts/download-jarvis.py and should be present
 * in server/learning/ before this runs (either locally or on GCP after git pull).
 *
 * Files consumed:
 *   jarvis_supercon_chem.csv   — 16k experimental Tc labels
 *   jarvis_supercon_3d.csv     — ~1k DFT-verified SC structures
 *   jarvis_supercon_2d.csv     — ~160 2D SC structures
 *   jarvis_dft3d_metallic.csv  — metallic inorganic materials (Tc = 0 negatives)
 *
 * Safety guarantees (same as supercon-db-ingestion.ts):
 *   - Batched writes with 80 ms yield between batches
 *   - ON CONFLICT DO NOTHING — idempotent re-runs
 *   - Progress persisted in system_state table
 *   - Single-instance guard: subsequent calls are no-ops while running
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { fileURLToPath } from "url";
import { db } from "../db";
import { superconExternalEntries, systemState } from "@shared/schema";
import { eq } from "drizzle-orm";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const BATCH_SIZE = 150;
const YIELD_MS   = 80;

const JARVIS_CSV_DIR = path.join(__dirname);  // server/learning/

const JARVIS_FILES: Array<{
  filename: string;
  stateKey: string;
  label: string;
}> = [
  { filename: "jarvis_supercon_chem.csv",  stateKey: "jarvis_supercon_chem_ingestion",  label: "JARVIS-SuperCon-Chem"  },
  { filename: "jarvis_supercon_3d.csv",    stateKey: "jarvis_supercon_3d_ingestion",    label: "JARVIS-SuperCon-3D"    },
  { filename: "jarvis_supercon_2d.csv",    stateKey: "jarvis_supercon_2d_ingestion",    label: "JARVIS-SuperCon-2D"    },
  { filename: "jarvis_dft3d_metallic.csv", stateKey: "jarvis_dft3d_metallic_ingestion", label: "JARVIS-DFT3D-Metallic" },
];

let _running = false;

// ── Types ─────────────────────────────────────────────────────────────────────

interface JarvisRow {
  formula:                  string;
  tc:                       number | null;
  isSuperconductor:         boolean;
  source:                   string;
  externalId:               string | null;
  spaceGroup:               string | null;
  crystalSystem:            string | null;
  family:                   string | null;
  lambda:                   number | null;
  pressureGpa:              number;
  rawData:                  Record<string, unknown>;
}

interface IngestionState {
  status:        "idle" | "running" | "done" | "failed";
  rowsIngested:  number;
  rowsTotal:     number | null;
  lastOffset:    number;
  finishedAt:    string | null;
  error:         string | null;
}

// ── State helpers ─────────────────────────────────────────────────────────────

async function loadState(key: string): Promise<IngestionState> {
  try {
    const rows = await db.select().from(systemState).where(eq(systemState.key, key)).limit(1);
    if (rows.length > 0) return rows[0].value as IngestionState;
  } catch { /* first run */ }
  return { status: "idle", rowsIngested: 0, rowsTotal: null, lastOffset: 0, finishedAt: null, error: null };
}

async function saveState(key: string, state: IngestionState): Promise<void> {
  try {
    await db.insert(systemState).values({ key, value: state as any }).onConflictDoUpdate({
      target: systemState.key,
      set: { value: state as any, updatedAt: new Date() },
    });
  } catch { /* non-fatal */ }
}

function delay(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

// ── CSV parsing ───────────────────────────────────────────────────────────────

function parseTc(val: string): number | null {
  const n = parseFloat(val);
  return isFinite(n) ? n : null;
}

function parseJarvisRow(
  headers: string[],
  values: string[],
  lineNum: number,
  sourceLabel: string,
): JarvisRow | null {
  if (values.length < 2) return null;

  const get = (...names: string[]): string => {
    for (const n of names) {
      const idx = headers.indexOf(n);
      if (idx >= 0 && values[idx] !== undefined) return values[idx].trim();
    }
    return "";
  };

  const formula = get("formula", "material", "composition");
  if (!formula || formula.length < 1) return null;

  const tcStr = get("tc");
  const tc    = parseTc(tcStr);
  const isSC  = get("is_superconductor").toLowerCase() !== "false";

  const raw: Record<string, unknown> = {};
  headers.forEach((h, i) => {
    if (values[i] !== undefined && values[i] !== "") {
      // raw_data column is already JSON — parse it back
      if (h === "raw_data") {
        try { raw["_raw"] = JSON.parse(values[i]); } catch { raw["_raw"] = values[i]; }
      } else {
        raw[h] = values[i];
      }
    }
  });

  // data_confidence from CSV → store in rawData so downstream can read it
  const confidence = get("data_confidence") || "experimental";
  raw["data_confidence"] = confidence;

  const bandgap = parseTc(get("bandgap_ev"));
  if (bandgap !== null) raw["bandgap_ev"] = bandgap;

  const formE = parseTc(get("formation_energy_per_atom"));
  if (formE !== null) raw["formation_energy_per_atom"] = formE;

  return {
    formula,
    tc,
    isSuperconductor: isSC,
    source: get("source") || sourceLabel,
    externalId: get("external_id") || `${sourceLabel}-L${lineNum}`,
    spaceGroup:   get("space_group") || null,
    crystalSystem: get("crystal_system") || null,
    family:       get("family") || null,
    lambda:       parseTc(get("lambda")),
    pressureGpa:  parseTc(get("pressure_gpa")) ?? 0,
    rawData:      raw,
  };
}

// ── Batch insert ──────────────────────────────────────────────────────────────

async function insertBatch(rows: JarvisRow[]): Promise<number> {
  if (rows.length === 0) return 0;
  try {
    const values = rows.map(r => ({
      formula:       r.formula,
      tc:            r.tc,
      isSuperconductor: r.isSuperconductor,
      source:        r.source,
      externalId:    r.externalId,
      spaceGroup:    r.spaceGroup,
      crystalSystem: r.crystalSystem,
      family:        r.family,
      lambda:        r.lambda,
      pressureGpa:   r.pressureGpa,
      rawData:       r.rawData,
    }));
    await db.insert(superconExternalEntries).values(values).onConflictDoNothing();
    return rows.length;
  } catch (err: any) {
    console.warn(`[JARVIS] Batch insert failed: ${err.message?.slice(0, 120)}`);
    return 0;
  }
}

// ── Single-file ingestion ─────────────────────────────────────────────────────

async function ingestFile(
  filePath:    string,
  stateKey:    string,
  label:       string,
): Promise<void> {
  const state = await loadState(stateKey);

  if (state.status === "done") {
    console.log(`[JARVIS] ${label}: already fully ingested (${state.rowsIngested.toLocaleString()} rows) — skipping`);
    return;
  }

  if (!fs.existsSync(filePath)) {
    console.warn(`[JARVIS] ${label}: CSV not found at ${filePath} — run scripts/download-jarvis.py first`);
    return;
  }

  console.log(`[JARVIS] ${label}: starting ingestion from ${path.basename(filePath)}`);
  state.status = "running";
  await saveState(stateKey, state);

  const fileStream = fs.createReadStream(filePath);
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let headers: string[] = [];
  let batch: JarvisRow[] = [];
  let lineNum = 0;
  let totalInserted = state.rowsIngested;

  try {
    for await (const line of rl) {
      lineNum++;

      if (lineNum === 1) {
        headers = line.split(",").map(h => h.trim().toLowerCase().replace(/^"|"$/g, ""));
        continue;
      }

      // Resume from last checkpoint
      if (lineNum <= state.lastOffset) continue;

      // Parse CSV line (handle quoted fields with embedded commas)
      const values = splitCsvLine(line);
      const row = parseJarvisRow(headers, values, lineNum, label);
      if (row) batch.push(row);

      if (batch.length >= BATCH_SIZE) {
        totalInserted += await insertBatch(batch);
        batch = [];
        state.rowsIngested = totalInserted;
        state.lastOffset = lineNum;
        await saveState(stateKey, state);
        await delay(YIELD_MS);
      }
    }

    // Flush remainder
    if (batch.length > 0) {
      totalInserted += await insertBatch(batch);
    }

    state.status = "done";
    state.rowsIngested = totalInserted;
    state.rowsTotal = lineNum - 1;
    state.lastOffset = lineNum;
    state.finishedAt = new Date().toISOString();
    await saveState(stateKey, state);
    console.log(`[JARVIS] ${label}: done — ${totalInserted.toLocaleString()} rows inserted`);

  } catch (err: any) {
    state.status = "failed";
    state.error = err.message;
    await saveState(stateKey, state);
    console.error(`[JARVIS] ${label}: ingestion failed — ${err.message}`);
  }
}

// ── CSV line splitter (handles quoted fields) ─────────────────────────────────

function splitCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

// ── Public entry point ────────────────────────────────────────────────────────

/**
 * Ingest all available JARVIS CSV files into the learning database.
 * Safe to call multiple times — already-completed files are skipped.
 * Designed to run as a background fire-and-forget task.
 */
export async function startJarvisIngestion(): Promise<void> {
  if (_running) {
    console.log("[JARVIS] Ingestion already in progress — skipping duplicate call");
    return;
  }
  _running = true;

  console.log("[JARVIS] Starting JARVIS dataset ingestion...");

  try {
    for (const { filename, stateKey, label } of JARVIS_FILES) {
      const filePath = path.join(JARVIS_CSV_DIR, filename);
      await ingestFile(filePath, stateKey, label);
      // Pause between files to let the event loop breathe
      await delay(500);
    }
    console.log("[JARVIS] All JARVIS files processed.");
  } finally {
    _running = false;
  }
}

/**
 * Returns ingestion status for all JARVIS datasets.
 * Used by the /api/status endpoint.
 */
export async function getJarvisIngestionStatus(): Promise<Array<{
  label: string;
  status: string;
  rowsIngested: number;
  finishedAt: string | null;
}>> {
  const results = [];
  for (const { stateKey, label } of JARVIS_FILES) {
    const state = await loadState(stateKey);
    results.push({
      label,
      status: state.status,
      rowsIngested: state.rowsIngested,
      finishedAt: state.finishedAt,
    });
  }
  return results;
}
