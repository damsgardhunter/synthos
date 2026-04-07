/**
 * Pareto Frontier Optimizer — NSGA-II style non-dominated sorting
 *
 * Three objectives (all maximized internally):
 *   f1 — predicted Tc (higher = better)
 *   f2 — formation energy stability (lower |Ef| = better → mapped to maximize)
 *   f3 — synthesis feasibility score from synthesis-planner (higher = better)
 *
 * Candidates on Pareto rank 1 are not dominated by any other candidate on all
 * three objectives simultaneously — they represent the best multi-objective trade-offs.
 */

import { estimateFormationEnergy } from "../learning/phase-diagram-engine";
import { planSynthesisRoutes } from "../synthesis/synthesis-planner";

export interface ParetoCandidate {
  formula: string;
  predictedTc: number | null;
  decompositionEnergy?: number | null;
  mlFeatures?: Record<string, any> | null;
  /** ML quality score [0,1] — used as a proxy Tc source when predictedTc is absent */
  ensembleScore?: number | null;
}

export interface ParetoObjectives {
  /** Normalized Tc in [0,1] where 1 = 400 K reference */
  tc: number;
  /** Stability: 1 - |Ef|/3 clamped to [0,1]; higher = more stable */
  stability: number;
  /** Synthesis feasibility in [0,1] from synthesis-planner bestRoute */
  synthesizability: number;
  /**
   * True when tc was derived from the ensemble-score proxy rather than the
   * physics-engine predictedTc.  Shown in the UI tooltip so users know the
   * Tc value is an estimate, not a physics calculation.
   */
  tcIsProxy?: boolean;
}

export interface ParetoResult {
  formula: string;
  rank: number;
  objectives: ParetoObjectives;
  /** true iff rank === 1 */
  isFront: boolean;
}

// ── In-memory caches ──────────────────────────────────────────────────────────

/** Latest computed Pareto results, keyed by formula */
const latestRankMap = new Map<string, number>();
/** Snapshot of the last full Pareto run */
let latestParetoResults: ParetoResult[] = [];
let lastRecomputeMs = 0;
/** Pending debounce timer for scheduled recomputes */
let recomputeTimer: ReturnType<typeof setTimeout> | null = null;

/** Lightweight synthesis-score cache so repeated Pareto runs don't re-plan */
const synthScoreCache = new Map<string, number>();
const SYNTH_CACHE_MAX = 2000;

// ── Objective computation ─────────────────────────────────────────────────────

async function getSynthScore(formula: string, formationEnergyEv?: number | null): Promise<number> {
  if (synthScoreCache.has(formula)) return synthScoreCache.get(formula)!;
  try {
    const plan = await planSynthesisRoutes(formula, {
      maxRoutes: 1,
      formationEnergy: formationEnergyEv ?? null,
    });
    const score = plan.bestRoute?.feasibilityScore ?? 0.4;
    if (synthScoreCache.size >= SYNTH_CACHE_MAX) {
      // Evict oldest 10 %
      const keys = Array.from(synthScoreCache.keys()).slice(0, Math.floor(SYNTH_CACHE_MAX * 0.1));
      for (const k of keys) synthScoreCache.delete(k);
    }
    synthScoreCache.set(formula, score);
    return score;
  } catch {
    return 0.4; // neutral default on error
  }
}

/**
 * Scale factor mapping ensembleScore=1.0 → TC_PROXY_REF_K when no physics-engine
 * Tc is available.  Ensemble scores are Tc-quality proxies, not temperature values,
 * so this is a rough calibration only; it is strictly better than treating all
 * unscored candidates as tc=0 on the Pareto front.
 */
const TC_PROXY_REF_K = 200;

export async function computeParetoObjectives(c: ParetoCandidate): Promise<ParetoObjectives> {
  // f1 — Tc: prefer the physics-engine predictedTc; fall back to an ensemble-score
  // proxy when predictedTc is null or 0 (candidate not yet through physics pipeline).
  const hasDftTc = c.predictedTc != null && c.predictedTc > 0;
  const rawTc = hasDftTc
    ? c.predictedTc!
    : (c.ensembleScore ?? 0) * TC_PROXY_REF_K;
  const tc = Math.max(0, Math.min(1, rawTc / 400));
  const tcIsProxy = !hasDftTc;

  // f2 — stability (prefer small |formation energy|)
  let ef: number;
  if (c.decompositionEnergy != null) {
    ef = c.decompositionEnergy;
  } else {
    try {
      ef = estimateFormationEnergy(c.formula);
    } catch {
      ef = -1.0; // neutral default
    }
  }
  const stability = Math.max(0, Math.min(1, 1 - Math.abs(ef) / 3.0));

  // f3 — synthesizability
  const synthesizability = await getSynthScore(c.formula, ef);

  return { tc, stability, synthesizability, tcIsProxy };
}

// ── Dominance and sorting ─────────────────────────────────────────────────────

/**
 * Returns true iff objectives A weakly dominate B on all objectives and
 * strictly dominate on at least one.  All objectives are "maximize".
 */
function dominates(a: ParetoObjectives, b: ParetoObjectives): boolean {
  const allGeq =
    a.tc >= b.tc - 1e-9 &&
    a.stability >= b.stability - 1e-9 &&
    a.synthesizability >= b.synthesizability - 1e-9;
  if (!allGeq) return false;
  return (
    a.tc > b.tc + 1e-9 ||
    a.stability > b.stability + 1e-9 ||
    a.synthesizability > b.synthesizability + 1e-9
  );
}

/**
 * NSGA-II non-dominated sorting.
 * Returns ParetoResult[] with rank assigned (1 = Pareto front, 2 = next, …).
 */
export async function nonDominatedSort(
  candidates: ParetoCandidate[],
  precomputedObjectives?: Map<string, ParetoObjectives>,
): Promise<ParetoResult[]> {
  const n = candidates.length;
  if (n === 0) return [];

  // Compute objectives with a concurrency limit so we don't fire N simultaneous
  // planSynthesisRoutes calls (each does DB queries) and exhaust the pg pool.
  // Cache-hits resolve instantly; cache-misses do DB work — limiting to 20 concurrent
  // matches the pool size and prevents cascade connection timeouts.
  const CONCURRENCY = 20;
  const objs: ParetoObjectives[] = new Array(n);
  for (let start = 0; start < n; start += CONCURRENCY) {
    const batch = candidates.slice(start, start + CONCURRENCY);
    const batchObjs = await Promise.all(
      batch.map(c =>
        precomputedObjectives?.has(c.formula)
          ? Promise.resolve(precomputedObjectives.get(c.formula)!)
          : computeParetoObjectives(c)
      )
    );
    for (let k = 0; k < batchObjs.length; k++) objs[start + k] = batchObjs[k];
  }

  // dominationCount[i]: how many candidates dominate i
  const dominationCount = new Int32Array(n);
  // dominatedSet[i]: indices dominated by i
  const dominatedSet: number[][] = Array.from({ length: n }, () => []);

  // O(n²) domination sweep — yield every 25 outer iterations so the event loop
  // stays responsive. With n capped at 200 this is ~8 yields total (negligible overhead).
  // Previously yielded every 100 which at n=500 still caused 10-120s freezes between yields.
  for (let i = 0; i < n; i++) {
    if (i > 0 && i % 25 === 0) await new Promise<void>(r => setTimeout(r, 0));
    for (let j = i + 1; j < n; j++) {
      if (dominates(objs[i], objs[j])) {
        dominatedSet[i].push(j);
        dominationCount[j]++;
      } else if (dominates(objs[j], objs[i])) {
        dominatedSet[j].push(i);
        dominationCount[i]++;
      }
    }
  }

  const ranks = new Int32Array(n);
  let currentFront: number[] = [];
  for (let i = 0; i < n; i++) {
    if (dominationCount[i] === 0) {
      ranks[i] = 1;
      currentFront.push(i);
    }
  }

  let rank = 1;
  while (currentFront.length > 0) {
    const nextFront: number[] = [];
    for (const i of currentFront) {
      for (const j of dominatedSet[i]) {
        dominationCount[j]--;
        if (dominationCount[j] === 0) {
          ranks[j] = rank + 1;
          nextFront.push(j);
        }
      }
    }
    rank++;
    currentFront = nextFront;
  }

  return candidates.map((c, i) => ({
    formula: c.formula,
    rank: ranks[i],
    objectives: objs[i],
    isFront: ranks[i] === 1,
  }));
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Compute Pareto ranks for a batch of candidates and update in-memory state.
 * Returns the full sorted result array.
 */
export async function recomputeParetoFrontier(candidates: ParetoCandidate[]): Promise<ParetoResult[]> {
  if (candidates.length === 0) return [];

  const results = await nonDominatedSort(candidates);

  // Update rank map
  latestRankMap.clear();
  for (const r of results) latestRankMap.set(r.formula, r.rank);
  latestParetoResults = results;
  lastRecomputeMs = Date.now();

  const rank1Count = results.filter(r => r.rank === 1).length;
  console.log(`[Pareto] Recomputed ${results.length} candidates: ${rank1Count} on rank-1 front`);

  return results;
}

/**
 * Schedule a debounced Pareto recompute using the provided candidate loader.
 * Multiple DFT completions in quick succession collapse into a single run.
 * Optional `onComplete` receives results so the caller can write ranks back to storage.
 */
// Maximum candidates fed into a recompute. O(n²) domination sweep with n=500
// does 125k comparisons and blocks the event loop for 10-120s even with yields.
// 200 candidates covers the meaningful Pareto front while keeping the sweep to
// ~20k comparisons — roughly 6× cheaper and stays well under 5s.
const MAX_PARETO_CANDIDATES = 200;

export function scheduleParetoRecompute(
  loadCandidates: () => Promise<ParetoCandidate[]>,
  delayMs = 8_000,
  onComplete?: (results: ParetoResult[]) => Promise<void>,
): void {
  if (recomputeTimer) clearTimeout(recomputeTimer);
  recomputeTimer = setTimeout(async () => {
    recomputeTimer = null;
    try {
      const all = await loadCandidates();
      if (all.length === 0) return;
      // Cap at MAX_PARETO_CANDIDATES — loader returns candidates sorted by Tc desc
      // so we keep the highest-Tc candidates which are the most relevant for the front.
      const candidates = all.slice(0, MAX_PARETO_CANDIDATES);
      const results = await recomputeParetoFrontier(candidates);
      if (onComplete) await onComplete(results);
    } catch (err: any) {
      console.error("[Pareto] Scheduled recompute failed:", err?.message ?? err);
    }
  }, delayMs);
}

/** Returns the latest rank map (formula → Pareto rank). Updated after each recompute. */
export function getLatestParetoRanks(): ReadonlyMap<string, number> {
  return latestRankMap;
}

/** Returns the full latest Pareto results for API consumption. */
export function getParetoFrontierData(): {
  results: ParetoResult[];
  rank1Count: number;
  totalCandidates: number;
  lastRecomputedAt: number;
} {
  return {
    results: latestParetoResults,
    rank1Count: latestParetoResults.filter(r => r.rank === 1).length,
    totalCandidates: latestParetoResults.length,
    lastRecomputedAt: lastRecomputeMs,
  };
}

/** Invalidate a single formula's synthesis score cache entry (call after synthesis update). */
export function invalidateSynthCache(formula: string): void {
  synthScoreCache.delete(formula);
}
