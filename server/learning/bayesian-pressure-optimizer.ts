import { predictPressureCurve, type PressureCurve } from "./pressure-aware-surrogate";
import { computeEnthalpyStability } from "./enthalpy-stability";
import { buildPressureResponseProfile, interpolateAtPressure } from "./pressure-property-map";
import { estimateFamilyPressure } from "./candidate-generator";
import { parseFormulaCounts } from "./physics-engine";

interface PressureObservation {
  formula: string;
  pressureGpa: number;
  tc: number;
  stable: boolean;
  enthalpy: number;
  timestamp: number;
}

interface PressureGPPrediction {
  mean: number;
  std: number;
  pressureGpa: number;
}

export interface BayesianPressureResult {
  formula: string;
  optimalPressure: number;
  predictedTcAtOptimal: number;
  unpenalizedTcAtOptimal: number;
  stableAtOptimal: boolean;
  confidence: number;
  acquisitionHistory: { pressure: number; acquisition: number }[];
  observationsUsed: number;
  method: string;
  timestamp: number;
}

export interface BayesianPressureStats {
  totalOptimizations: number;
  formulasOptimized: number;
  avgOptimalPressure: number;
  avgPredictedTc: number;
  lowPressureOptimalCount: number;
  recentResults: { formula: string; optimalPressure: number; predictedTc: number }[];
}

const observationStore = new Map<string, PressureObservation[]>();
const observationAccessOrder: string[] = [];
const resultCache = new Map<string, BayesianPressureResult>();
const MAX_OBSERVATIONS_PER_FORMULA = 100;
const MAX_FORMULAS = 5000;
const MAX_RESULTS_CACHE = 300;
const PRESSURE_SEARCH_MIN = 0;
const CACHE_TTL_MS = 10 * 60 * 1000;

interface GPCache {
  L: number[][];
  alpha: number[];
  yMean: number;
  obsCount: number;
  lengthScale: number;
  signalVar: number;
  noiseVar: number;
}

const gpCacheStore = new Map<string, GPCache>();

function dynamicPressureMax(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const hCount = counts["H"] || 0;
  const hydrogenRatio = totalAtoms > 0 ? hCount / totalAtoms : 0;

  if (hydrogenRatio >= 0.7) return 500;
  if (hydrogenRatio >= 0.5) return 450;
  if (hydrogenRatio >= 0.3) return 400;
  if (hCount > 0) return 350;

  const organicElements = ["C", "N", "O", "S"];
  const organicCount = organicElements.reduce((s, e) => s + (counts[e] || 0), 0);
  if (organicCount / totalAtoms > 0.5) return 200;

  return 350;
}

function matern52Kernel1D(x1: number, x2: number, lengthScale: number, signalVar: number): number {
  const r = Math.abs(x1 - x2) / lengthScale;
  const sqrt5r = Math.sqrt(5) * r;
  return signalVar * (1 + sqrt5r + (5 / 3) * r * r) * Math.exp(-sqrt5r);
}

function cholDecomp(K: number[][]): number[][] {
  const n = K.length;
  let jitter = 1e-8;

  for (let attempt = 0; attempt <= 4; attempt++) {
    if (attempt > 0) {
      for (let i = 0; i < n; i++) K[i][i] += jitter;
      jitter *= 10;
    }

    const L = Array.from({ length: n }, () => new Array(n).fill(0));
    let failed = false;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
        if (i === j) {
          const diag = K[i][i] - sum;
          if (diag <= 0) { failed = true; break; }
          L[i][j] = Math.sqrt(diag);
        } else {
          L[i][j] = L[j][j] > 1e-15 ? (K[i][j] - sum) / L[j][j] : 0;
        }
      }
      if (failed) break;
    }

    if (!failed) return L;
  }

  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) L[i][i] = Math.sqrt(Math.max(K[i][i], 1e-8));
  return L;
}

function cholSolve(L: number[][], b: number[]): number[] {
  const n = L.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) sum += L[i][j] * y[j];
    y[i] = L[i][i] > 1e-10 ? (b[i] - sum) / L[i][i] : 0;
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += L[j][i] * x[j];
    x[i] = L[i][i] > 1e-10 ? (y[i] - sum) / L[i][i] : 0;
  }
  return x;
}

function cholForwardSolve(L: number[][], b: number[]): number[] {
  const n = L.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) sum += L[i][j] * y[j];
    y[i] = L[i][i] > 1e-10 ? (b[i] - sum) / L[i][i] : 0;
  }
  return y;
}

function optimizeLengthScale(obs: PressureObservation[], signalVar: number, noiseVar: number): number {
  const n = obs.length;
  if (n < 3) return 30;

  const candidates = [10, 15, 20, 30, 50, 80, 120];
  let bestLS = 30;
  let bestLML = -Infinity;

  const yRaw = obs.map(o => o.tc);
  const yMean = yRaw.reduce((s, v) => s + v, 0) / n;
  const y = yRaw.map(v => v - yMean);

  for (const ls of candidates) {
    const K: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const k = matern52Kernel1D(obs[i].pressureGpa, obs[j].pressureGpa, ls, signalVar);
        K[i][j] = k;
        K[j][i] = k;
      }
      K[i][i] += noiseVar;
    }

    const L = cholDecomp(K);
    const alpha = cholSolve(L, y);

    let logDet = 0;
    for (let i = 0; i < n; i++) logDet += Math.log(Math.max(L[i][i], 1e-15));
    logDet *= 2;

    let dataFit = 0;
    for (let i = 0; i < n; i++) dataFit += y[i] * alpha[i];

    const lml = -0.5 * dataFit - 0.5 * logDet - 0.5 * n * Math.log(2 * Math.PI);

    if (Number.isFinite(lml) && lml > bestLML) {
      bestLML = lml;
      bestLS = ls;
    }
  }

  return bestLS;
}

function erf(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1.0 / (1.0 + p * Math.abs(x));
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function normCDF(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function normPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function touchLRU(formula: string): void {
  const idx = observationAccessOrder.indexOf(formula);
  if (idx !== -1) observationAccessOrder.splice(idx, 1);
  observationAccessOrder.push(formula);
}

function evictLRU(): void {
  while (observationStore.size > MAX_FORMULAS && observationAccessOrder.length > 0) {
    const oldest = observationAccessOrder.shift()!;
    observationStore.delete(oldest);
    gpCacheStore.delete(oldest);
    resultCache.delete(oldest);
  }
}

export function addPressureObservation(
  formula: string,
  pressureGpa: number,
  tc: number,
  stable: boolean,
  enthalpy: number = 0
): void {
  if (!observationStore.has(formula)) {
    observationStore.set(formula, []);
  }
  const obs = observationStore.get(formula)!;

  const existing = obs.find(o => Math.abs(o.pressureGpa - pressureGpa) < 2);
  if (existing) {
    if (tc > existing.tc) {
      existing.tc = tc;
      existing.stable = stable;
      existing.enthalpy = enthalpy;
      existing.timestamp = Date.now();
      gpCacheStore.delete(formula);
    }
    touchLRU(formula);
    return;
  }

  obs.push({ formula, pressureGpa, tc, stable, enthalpy, timestamp: Date.now() });
  gpCacheStore.delete(formula);

  if (obs.length > MAX_OBSERVATIONS_PER_FORMULA) {
    obs.sort((a, b) => b.tc - a.tc);
    obs.length = MAX_OBSERVATIONS_PER_FORMULA;
  }

  touchLRU(formula);
  evictLRU();
}

function seedFromSurrogate(formula: string): void {
  if (observationStore.has(formula) && observationStore.get(formula)!.length >= 5) {
    return;
  }

  try {
    const profile = buildPressureResponseProfile(formula);
    for (const pt of profile.tcVsPressure) {
      const stabPt = profile.stabilityVsPressure.find(s => s.pressure === pt.pressure);
      addPressureObservation(
        formula,
        pt.pressure,
        pt.tc,
        stabPt?.stable ?? false,
        stabPt?.enthalpy ?? 0
      );
    }
  } catch {}
}

function buildGPFactors(
  obs: PressureObservation[],
  lengthScale: number,
  signalVar: number,
  noiseVar: number,
  cacheKey?: string
): { L: number[][]; alpha: number[]; yMean: number } {
  if (cacheKey) {
    const cached = gpCacheStore.get(cacheKey);
    if (cached && cached.obsCount === obs.length &&
        cached.lengthScale === lengthScale &&
        cached.signalVar === signalVar &&
        cached.noiseVar === noiseVar) {
      return { L: cached.L, alpha: cached.alpha, yMean: cached.yMean };
    }
  }

  const n = obs.length;
  const yRaw = obs.map(o => o.tc);
  const yMean = yRaw.reduce((s, v) => s + v, 0) / n;
  const y = yRaw.map(v => v - yMean);

  const K: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const k = matern52Kernel1D(obs[i].pressureGpa, obs[j].pressureGpa, lengthScale, signalVar);
      K[i][j] = k;
      K[j][i] = k;
    }
    K[i][i] += noiseVar;
  }

  const L = cholDecomp(K);
  const alpha = cholSolve(L, y);

  if (cacheKey) {
    gpCacheStore.set(cacheKey, { L, alpha, yMean, obsCount: n, lengthScale, signalVar, noiseVar });
  }

  return { L, alpha, yMean };
}

function gpPredict(
  obs: PressureObservation[],
  targetPressure: number,
  lengthScale: number = 30,
  signalVar: number = 1.0,
  noiseVar: number = 0.05,
  cacheKey?: string
): PressureGPPrediction {
  const n = obs.length;
  if (n === 0) {
    return { mean: 0, std: Math.sqrt(signalVar), pressureGpa: targetPressure };
  }

  const { L, alpha, yMean } = buildGPFactors(obs, lengthScale, signalVar, noiseVar, cacheKey);

  const kStar = new Array(n);
  for (let i = 0; i < n; i++) {
    kStar[i] = matern52Kernel1D(targetPressure, obs[i].pressureGpa, lengthScale, signalVar);
  }

  let mean = yMean;
  for (let i = 0; i < n; i++) mean += kStar[i] * alpha[i];

  const v = cholForwardSolve(L, kStar);
  let variance = signalVar;
  for (let i = 0; i < v.length; i++) variance -= v[i] * v[i];
  variance = Math.max(variance, 1e-6);

  return { mean: Math.max(0, mean), std: Math.sqrt(variance), pressureGpa: targetPressure };
}

function estimateStabilityProbability(obs: PressureObservation[], targetPressure: number): number {
  if (obs.length === 0) return 0.5;

  let weightedStable = 0;
  let totalWeight = 0;
  for (const o of obs) {
    const dist = Math.abs(o.pressureGpa - targetPressure);
    const w = Math.exp(-0.5 * (dist / 30) * (dist / 30));
    weightedStable += w * (o.stable ? 1 : 0);
    totalWeight += w;
  }

  if (totalWeight < 1e-6) return 0.5;
  const proximityStab = weightedStable / totalWeight;

  let enthalpyPenalty = 0;
  let enthalpyCount = 0;
  for (const o of obs) {
    const dist = Math.abs(o.pressureGpa - targetPressure);
    if (dist < 50) {
      if (o.enthalpy > 0.1) enthalpyPenalty += Math.min(1, o.enthalpy / 0.5);
      enthalpyCount++;
    }
  }
  const avgEnthPenalty = enthalpyCount > 0 ? enthalpyPenalty / enthalpyCount : 0;

  return Math.max(0.05, proximityStab * (1 - avgEnthPenalty * 0.5));
}

function expectedImprovement(
  obs: PressureObservation[],
  targetPressure: number,
  bestTc: number,
  lengthScale: number,
  xi: number = 0.5,
  cacheKey?: string
): number {
  const pred = gpPredict(obs, targetPressure, lengthScale, 1.0, 0.05, cacheKey);
  if (pred.std < 1e-8) return 0;
  const improvement = pred.mean - bestTc - xi;
  const z = improvement / pred.std;
  const ei = improvement * normCDF(z) + pred.std * normPDF(z);
  if (!Number.isFinite(ei) || ei <= 0) return 0;
  const stabProb = estimateStabilityProbability(obs, targetPressure);
  return ei * stabProb;
}

function quasiRandomCandidates(min: number, max: number, count: number, seed: number): number[] {
  const phi = (1 + Math.sqrt(5)) / 2;
  const alpha = 1 / phi;
  const candidates: number[] = [];
  const seedOffset = (seed * 0.618033988749895) % 1;
  for (let i = 0; i < count; i++) {
    const u = ((i + 1) * alpha + seedOffset) % 1;
    candidates.push(min + (max - min) * u);
  }
  return candidates;
}

export function optimizePressureForFormula(
  formula: string,
  nIterations: number = 5,
  nCandidatesPerIter: number = 20,
  familyPressureHint?: number
): BayesianPressureResult {
  const cached = resultCache.get(formula);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
    return cached;
  }

  seedFromSurrogate(formula);

  const obs = observationStore.get(formula) ?? [];
  if (obs.length === 0) {
    return {
      formula,
      optimalPressure: 0,
      predictedTcAtOptimal: 0,
      unpenalizedTcAtOptimal: 0,
      stableAtOptimal: false,
      confidence: 0,
      acquisitionHistory: [],
      observationsUsed: 0,
      method: "no-data",
      timestamp: Date.now(),
    };
  }

  const hint = familyPressureHint ?? 0;
  const pressureMax = dynamicPressureMax(formula);
  let searchMin = PRESSURE_SEARCH_MIN;
  let searchMax = pressureMax;
  if (hint > 50) {
    searchMin = Math.max(0, hint - 80);
    searchMax = Math.min(pressureMax, hint + 100);
  } else if (hint > 0) {
    searchMin = 0;
    searchMax = Math.min(pressureMax, hint + 60);
  }

  let bestTc = Math.max(...obs.map(o => o.tc));
  const acquisitionHistory: { pressure: number; acquisition: number }[] = [];

  const tcRange = Math.max(1, bestTc);
  const normalizedSignalVar = 1.0;
  const noiseVar = 0.05;
  const normalizedObs = obs.map(o => ({
    ...o,
    tc: o.tc / tcRange,
  }));
  const normalizedBestTc = bestTc / tcRange;

  let ls = optimizeLengthScale(normalizedObs, normalizedSignalVar, noiseVar);

  for (let iter = 0; iter < nIterations; iter++) {
    let bestCandidate = 0;
    let bestEI = -Infinity;

    const candidates = quasiRandomCandidates(searchMin, searchMax, nCandidatesPerIter, iter);

    for (let c = 0; c < candidates.length; c++) {
      const candidateP = candidates[c];

      const rawEI = expectedImprovement(normalizedObs, candidateP, normalizedBestTc, ls, 0.01, `${formula}:norm`);
      const pressureFrac = candidateP / 500;
      const scaledPenalty = pressureFrac * 0.1 * Math.max(0.01, rawEI);
      const ei = rawEI - scaledPenalty;

      acquisitionHistory.push({ pressure: candidateP, acquisition: ei });

      if (ei > bestEI) {
        bestEI = ei;
        bestCandidate = candidateP;
      }
    }

    try {
      const interp = interpolateAtPressure(formula, bestCandidate);
      const newTcNorm = interp.tc / tcRange;
      normalizedObs.push({
        formula,
        pressureGpa: bestCandidate,
        tc: newTcNorm,
        stable: interp.enthalpyStable,
        enthalpy: interp.enthalpy,
        timestamp: Date.now(),
      });

      if (interp.tc > bestTc) {
        bestTc = interp.tc;
      }

      ls = optimizeLengthScale(normalizedObs, normalizedSignalVar, noiseVar);
      gpCacheStore.delete(`${formula}:norm`);

      addPressureObservation(formula, bestCandidate, interp.tc, interp.enthalpyStable, interp.enthalpy);
    } catch {}
  }

  const finalLS = optimizeLengthScale(obs, 1.0, 0.05);
  let optimalPressure = 0;
  let gpOptimalTc = 0;
  let peakUnpenalizedTc = 0;
  let peakUnpenalizedPressure = 0;

  const scanStep = 5;
  for (let p = searchMin; p <= searchMax; p += scanStep) {
    const pred = gpPredict(obs, p, finalLS, 1.0, 0.05, `${formula}:final`);
    const stabProb = estimateStabilityProbability(obs, p);
    const pressureFrac = p / 500;
    const scaledPressurePenalty = pressureFrac * Math.max(1, pred.mean);
    const penalizedTc = pred.mean * stabProb - scaledPressurePenalty * 0.1;
    if (penalizedTc > gpOptimalTc) {
      gpOptimalTc = penalizedTc;
      optimalPressure = p;
    }
    if (pred.mean > peakUnpenalizedTc) {
      peakUnpenalizedTc = pred.mean;
      peakUnpenalizedPressure = p;
    }
  }

  const optPred = gpPredict(obs, optimalPressure, finalLS, 1.0, 0.05, `${formula}:final`);
  const relStd = optPred.mean > 0 ? optPred.std / optPred.mean : 1;
  const confidence = Math.max(0, Math.min(1, 1 - relStd));
  const gpPredictedTc = Math.max(0, optPred.mean);

  let stableAtOptimal = false;
  try {
    const interp = interpolateAtPressure(formula, optimalPressure);
    stableAtOptimal = interp.enthalpyStable;
  } catch {}

  const result: BayesianPressureResult = {
    formula,
    optimalPressure: Math.round(optimalPressure),
    predictedTcAtOptimal: Math.round(gpPredictedTc * 10) / 10,
    unpenalizedTcAtOptimal: Math.round(Math.max(0, peakUnpenalizedTc) * 10) / 10,
    stableAtOptimal,
    confidence: Math.round(confidence * 1000) / 1000,
    acquisitionHistory: acquisitionHistory.slice(-20),
    observationsUsed: obs.length,
    method: "bayesian-ei-pressure-penalized",
    timestamp: Date.now(),
  };

  gpCacheStore.delete(`${formula}:norm`);
  gpCacheStore.delete(`${formula}:final`);

  if (resultCache.size >= MAX_RESULTS_CACHE) {
    const firstKey = resultCache.keys().next().value;
    if (firstKey !== undefined) resultCache.delete(firstKey);
  }
  resultCache.set(formula, result);

  return result;
}

export function batchOptimizePressure(
  formulas: string[],
  maxFormulas: number = 10
): BayesianPressureResult[] {
  return formulas.slice(0, maxFormulas).map(f => optimizePressureForFormula(f, 5, 20, estimateFamilyPressure(f)));
}

export function getBayesianPressureStats(): BayesianPressureStats {
  const results = Array.from(resultCache.values());

  if (results.length === 0) {
    return {
      totalOptimizations: 0,
      formulasOptimized: 0,
      avgOptimalPressure: 0,
      avgPredictedTc: 0,
      lowPressureOptimalCount: 0,
      recentResults: [],
    };
  }

  const avgPressure = results.reduce((s, r) => s + r.optimalPressure, 0) / results.length;
  const avgTc = results.reduce((s, r) => s + r.predictedTcAtOptimal, 0) / results.length;
  const lowPCount = results.filter(r => r.optimalPressure <= 50).length;

  const recentResults = results
    .slice(-15)
    .map(r => ({
      formula: r.formula,
      optimalPressure: r.optimalPressure,
      predictedTc: r.predictedTcAtOptimal,
    }));

  return {
    totalOptimizations: results.length,
    formulasOptimized: new Set(results.map(r => r.formula)).size,
    avgOptimalPressure: Math.round(avgPressure),
    avgPredictedTc: Math.round(avgTc * 10) / 10,
    lowPressureOptimalCount: lowPCount,
    recentResults,
  };
}

export function getObservationCount(formula: string): number {
  return observationStore.get(formula)?.length ?? 0;
}

export function clearBayesianPressureCache(): void {
  resultCache.clear();
  gpCacheStore.clear();
}
