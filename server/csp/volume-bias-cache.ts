/**
 * Volume-bias feedback cache.
 *
 * F6 (CHGNet MLIP relaxation) measures how far the CSP candidate volumes
 * drift when relaxed. When a batch shows a systematic bias — e.g. "275/300
 * candidates expanded by avg 90%" — the CSP volume prior is wrong and every
 * future batch for that composition starts from the same bad volume.
 *
 * This cache persists the observed correction so the NEXT CSP batch for the
 * same formula + pressure generates structures at corrected volumes. The
 * correction compounds across runs (each run measures drift relative to the
 * already-corrected volumes), so it converges even if a single CHGNet
 * estimate is imperfect.
 */
import * as fs from "fs";
import * as path from "path";

const CACHE_PATH = path.join(process.cwd(), "logs", "volume-bias-cache.json");

interface VolumeBiasEntry {
  /** Multiply the CSP target volume-per-atom by this factor. */
  factor: number;
  updatedAt: string;
  /** How many F6 batches have contributed to this correction. */
  sampleCount: number;
}

function cacheKey(formula: string, pressureGPa: number): string {
  return `${formula.replace(/[^A-Za-z0-9]/g, "")}@${Math.round(pressureGPa)}`;
}

function readCache(): Record<string, VolumeBiasEntry> {
  try {
    const parsed = JSON.parse(fs.readFileSync(CACHE_PATH, "utf-8"));
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

/**
 * Look up the persisted volume correction for a composition at a pressure.
 * Returns 1.0 (no correction) when nothing is cached.
 */
export function loadVolumeBias(formula: string, pressureGPa: number): number {
  const entry = readCache()[cacheKey(formula, pressureGPa)];
  return entry && Number.isFinite(entry.factor) && entry.factor > 0 ? entry.factor : 1.0;
}

/**
 * Persist a volume-bias correction observed by F6. `observedFactor` is the
 * ratio CHGNet wanted relative to the volumes THIS batch already used (e.g.
 * 1.63 if candidates expanded). It compounds onto any prior correction, so
 * the stored factor is always relative to the uncorrected CSP prior.
 */
export function saveVolumeBias(formula: string, pressureGPa: number, observedFactor: number): void {
  if (!Number.isFinite(observedFactor) || observedFactor <= 0) return;
  const cache = readCache();
  const key = cacheKey(formula, pressureGPa);
  const prev = cache[key];
  const compounded = (prev?.factor ?? 1.0) * observedFactor;
  const clamped = Math.max(0.4, Math.min(3.0, compounded));
  cache[key] = {
    factor: clamped,
    updatedAt: new Date().toISOString(),
    sampleCount: (prev?.sampleCount ?? 0) + 1,
  };
  try {
    fs.mkdirSync(path.dirname(CACHE_PATH), { recursive: true });
    fs.writeFileSync(CACHE_PATH, JSON.stringify(cache, null, 2));
  } catch {
    // Non-fatal: the correction just won't carry to the next batch.
  }
}
