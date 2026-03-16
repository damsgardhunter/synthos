/**
 * Extended Training Dataset
 *
 * Augments the 513-entry SUPERCON_TRAINING_DATA with:
 *   1. ~2000-3000 Materials Project metallic compounds (including negative examples, tc=0)
 *   2. Rolling averages for formulas that appear in multiple batches
 *
 * Results are cached to disk (server/learning/cache/extended-dataset.json) with a 7-day TTL
 * so repeated startups don't re-hit the MP API.
 *
 * Usage:
 *   const extended = await getExtendedTrainingData();
 *   // Returns SuperconEntry[] ready to add to TrainingPool
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { SUPERCON_TRAINING_DATA, type SuperconEntry } from "./supercon-dataset";
import { classifyFamily } from "./utils";
import { fetchMPBatchFromAPI } from "./materials-project-client";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CACHE_DIR  = path.join(__dirname, "cache");
const CACHE_FILE = path.join(CACHE_DIR, "extended-dataset.json");
const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

// How many MP materials to pull in total (fetched in batches of 500)
const MP_TARGET_COUNT = 2500;
const MP_BATCH_SIZE   = 500;

// ─── disk cache helpers ───────────────────────────────────────────────────────

function readDiskCache(): SuperconEntry[] | null {
  try {
    if (!fs.existsSync(CACHE_FILE)) return null;
    const stat = fs.statSync(CACHE_FILE);
    if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
    const raw = fs.readFileSync(CACHE_FILE, "utf8");
    const parsed = JSON.parse(raw) as SuperconEntry[];
    if (!Array.isArray(parsed) || parsed.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeDiskCache(entries: SuperconEntry[]): void {
  try {
    if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(CACHE_FILE, JSON.stringify(entries, null, 2), "utf8");
  } catch (err: any) {
    console.warn(`[ExtDataset] Failed to write disk cache: ${err?.message}`);
  }
}

// ─── rolling-average accumulator ─────────────────────────────────────────────

interface Accumulator {
  tcSum: number;
  tcCount: number;
  pressureSum: number;
  pressureCount: number;
  family: string;
  spaceGroup: string;
  crystalSystem: string;
  isSuperconductor: boolean;
}

/** Merge multiple observations of the same formula into a single SuperconEntry. */
function accumulatorToEntry(formula: string, acc: Accumulator): SuperconEntry {
  return {
    formula,
    tc: acc.tcCount > 0 ? acc.tcSum / acc.tcCount : 0,
    family: acc.family,
    isSuperconductor: acc.isSuperconductor,
    pressureGPa: acc.pressureCount > 0 ? acc.pressureSum / acc.pressureCount : 0,
    spaceGroup: acc.spaceGroup || undefined,
    crystalSystem: acc.crystalSystem || undefined,
  };
}

// ─── main export ─────────────────────────────────────────────────────────────

let cachedResult: SuperconEntry[] | null = null;
let fetchPromise: Promise<SuperconEntry[]> | null = null;

/**
 * Returns up to ~2500 MP-derived entries suitable for augmenting XGBoost training.
 * First call fetches from the MP API (or disk cache). Subsequent calls return immediately.
 */
export async function getExtendedTrainingData(): Promise<SuperconEntry[]> {
  if (cachedResult) return cachedResult;
  if (fetchPromise) return fetchPromise;

  fetchPromise = _fetchExtendedData();
  cachedResult = await fetchPromise;
  return cachedResult;
}

/** Force-invalidates the in-memory cache (disk cache unchanged). */
export function invalidateExtendedDataCache(): void {
  cachedResult = null;
  fetchPromise = null;
}

async function _fetchExtendedData(): Promise<SuperconEntry[]> {
  // 1. Try disk cache
  const disk = readDiskCache();
  if (disk) {
    console.log(`[ExtDataset] Loaded ${disk.length} entries from disk cache`);
    return disk;
  }

  // 2. Build a set of formulas already in the base dataset (skip duplicates)
  const baseFormulas = new Set(SUPERCON_TRAINING_DATA.map(e => e.formula));

  // 3. Fetch from MP API in batches
  const accumMap = new Map<string, Accumulator>();
  let totalFetched = 0;

  const numBatches = Math.ceil(MP_TARGET_COUNT / MP_BATCH_SIZE);
  for (let batch = 0; batch < numBatches; batch++) {
    const skip = batch * MP_BATCH_SIZE;
    try {
      const records = await fetchMPBatchFromAPI(MP_BATCH_SIZE, skip);
      if (records.length === 0) {
        console.log(`[ExtDataset] Batch ${batch + 1}: empty response, stopping early`);
        break;
      }

      for (const rec of records) {
        if (!rec.formula || baseFormulas.has(rec.formula)) continue;

        // Determine superconductor status:
        // We treat ALL MP-seeded entries as non-superconductors (tc=0) unless
        // they are already in SUPERCON_TRAINING_DATA (filtered above).
        // This is intentional: MP metallic compounds without published Tc data
        // serve as high-quality *negative* training examples.
        const isSC = false;
        const tc = 0;

        const existing = accumMap.get(rec.formula);
        if (existing) {
          // Rolling average — repeated formula from a later batch
          existing.tcSum += tc;
          existing.tcCount += 1;
          existing.pressureSum += 0;
          existing.pressureCount += 1;
        } else {
          accumMap.set(rec.formula, {
            tcSum: tc,
            tcCount: 1,
            pressureSum: 0,
            pressureCount: 1,
            family: classifyFamily(rec.formula),
            spaceGroup: rec.spaceGroup ?? "",
            crystalSystem: rec.crystalSystem ?? "",
            isSuperconductor: isSC,
          });
        }
      }

      totalFetched += records.length;
      console.log(`[ExtDataset] Batch ${batch + 1}/${numBatches}: +${records.length} records (total raw: ${totalFetched})`);

      // Yield briefly between batches so we don't monopolise the event loop
      await new Promise<void>(r => setTimeout(r, 200));
    } catch (err: any) {
      console.warn(`[ExtDataset] Batch ${batch + 1} failed: ${err?.message?.slice(0, 120)}`);
      break;
    }
  }

  if (accumMap.size === 0) {
    console.log("[ExtDataset] No MP data available — using base dataset only");
    return [];
  }

  // 4. Convert accumulators → SuperconEntry array, filter insulators (high band-gap
  //    proxied by family "Other" with no relevant composition is fine; they still
  //    serve as structural-diversity negative examples).
  const entries: SuperconEntry[] = [];
  accumMap.forEach((acc, formula) => {
    entries.push(accumulatorToEntry(formula, acc));
  });

  // 5. Persist to disk
  writeDiskCache(entries);
  console.log(`[ExtDataset] Built extended dataset: ${entries.length} entries (${accumMap.size} unique formulas). Cached to disk.`);

  return entries;
}

/**
 * Returns the extended dataset entry count without triggering a fetch.
 * Returns -1 if no data is loaded yet.
 */
export function getExtendedDatasetSize(): number {
  return cachedResult?.length ?? -1;
}
