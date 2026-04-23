import { classifyFamily } from "./utils";

export interface PredictionRealityEntry {
  formula: string;
  pressure: number;
  model_prediction: {
    Tc: number;
    stable: boolean;
    formation_energy: number | null;
    xgboost_Tc: number;
    gnn_Tc: number;
  };
  ground_truth: {
    Tc: number;
    stable: boolean;
    formation_energy: number | null;
    lambda: number | null;
    DOS_EF: number | null;
  };
  error: {
    tc_error: number;
    tc_abs_error: number;
    tc_squared_error: number;
    tc_pct_error: number | null;
    stability_correct: boolean;
    fe_error: number | null;
  };
  predicted_sigma: number | null;
  source: string;
  cycle: number;
  timestamp: number;
}

export interface PredictionRealityMetrics {
  count: number;
  rmse: number;
  mae: number;
  bias: number;
  r2: number;
  maxError: number;
  medianAbsError: number;
  stabilityAccuracy: number;
  stabilityBalancedAccuracy: number;
  stabilityF1: number;
  feMAE: number | null;
  pct_within_10K: number;
  pct_within_30K: number;
  pct_within_50K: number;
  pct_within_20_percent: number;
  smallSampleWarning: boolean;
}

export interface FamilyMetrics {
  family: string;
  count: number;
  rmse: number;
  mae: number;
  bias: number;
  smallSampleWarning: boolean;
}

export interface RetrainTriggerConfig {
  sampleThreshold: number;
  enabled: boolean;
}

export interface RetrainTriggerState {
  config: RetrainTriggerConfig;
  samplesSinceLastTrigger: number;
  totalTriggers: number;
  lastTriggerTimestamp: number;
  lastTriggerSampleCount: number;
}

const MAX_LEDGER_SIZE = 5000;
const ledger: PredictionRealityEntry[] = [];
let samplesSinceLastRetrain = 0;
let totalRetrainTriggers = 0;
let lastTriggerTimestamp = 0;
let lastTriggerSampleCount = 0;

const retrainConfig: RetrainTriggerConfig = {
  sampleThreshold: 100,
  enabled: true,
};

type LedgerListener = () => void;
const ledgerListeners: LedgerListener[] = [];

export function onLedgerEntry(listener: LedgerListener): void {
  ledgerListeners.push(listener);
}

let cachedGlobalMetrics: PredictionRealityMetrics | null = null;
let cachedGlobalMetricsDirty = true;

let _gnnModule: any = null;
let _gbModule: any = null;
let _mlModule: any = null;

export async function recordPredictionVsReality(
  formula: string,
  pressure: number,
  prediction: {
    Tc: number;
    stable: boolean;
    formation_energy: number | null;
    xgboost_Tc: number;
    gnn_Tc: number;
  },
  reality: {
    Tc: number;
    stable: boolean;
    formation_energy: number | null;
    lambda: number | null;
    DOS_EF: number | null;
  },
  source: string,
  cycle: number
): Promise<PredictionRealityEntry> {
  const tcError = prediction.Tc - reality.Tc;
  const tcAbsError = Math.abs(tcError);
  const tcSquaredError = tcError * tcError;
  const tcPctError = reality.Tc > 0 ? (tcAbsError / reality.Tc) * 100 : null;

  let feError: number | null = null;
  if (prediction.formation_energy !== null && reality.formation_energy !== null) {
    feError = prediction.formation_energy - reality.formation_energy;
  }

  let predicted_sigma: number | null = null;
  try {
    if (!_gnnModule) _gnnModule = require("./graph-neural-net");
    if (!_gbModule) _gbModule = require("./gradient-boost");
    if (!_mlModule) _mlModule = require("./ml-predictor");
    let gnnSigma = 0, xgbSigma = 0;
    try {
      const gnnRes = _gnnModule.gnnPredictWithUncertainty(formula);
      if (Number.isFinite(gnnRes.totalStd) && gnnRes.totalStd > 0) gnnSigma = gnnRes.totalStd;
    } catch {}
    try {
      const feats = await _mlModule.extractFeatures(formula);
      const xgbRes = await _gbModule.gbPredictWithUncertainty(feats, formula);
      if (Number.isFinite(xgbRes.totalStd) && xgbRes.totalStd > 0) xgbSigma = xgbRes.totalStd;
    } catch {}
    if (gnnSigma > 0 && xgbSigma > 0) {
      predicted_sigma = Math.sqrt(1 / (1 / (gnnSigma ** 2) + 1 / (xgbSigma ** 2)));
    } else if (xgbSigma > 0) {
      predicted_sigma = xgbSigma;
    } else if (gnnSigma > 0) {
      predicted_sigma = gnnSigma;
    } else {
      // No model uncertainty available — leave null so the conformal calibrator
      // uses its empirical RMSE fallback (family-specific, data-driven) instead
      // of a hardcoded heuristic. The old `Math.max(10, Tc * 0.3)` produced
      // identical ±22K CIs for all chemistries regardless of data coverage.
      predicted_sigma = null;
    }
  } catch {
    predicted_sigma = null;
  }

  const entry: PredictionRealityEntry = {
    formula,
    pressure,
    model_prediction: prediction,
    ground_truth: reality,
    error: {
      tc_error: tcError,
      tc_abs_error: tcAbsError,
      tc_squared_error: tcSquaredError,
      tc_pct_error: tcPctError,
      stability_correct: prediction.stable === reality.stable,
      fe_error: feError,
    },
    predicted_sigma,
    source,
    cycle,
    timestamp: Date.now(),
  };

  ledger.push(entry);
  if (ledger.length > MAX_LEDGER_SIZE) {
    ledger.splice(0, ledger.length - MAX_LEDGER_SIZE);
    worstHeapDirty = true;
  }

  samplesSinceLastRetrain++;
  cachedGlobalMetricsDirty = true;
  cachedCICoverageDirty = true;
  if (!worstHeapDirty) insertIntoWorstHeap(entry);

  for (const listener of [...ledgerListeners]) {
    try { listener(); } catch {}
  }

  return entry;
}

function quickSelectMedian(arr: number[]): number {
  if (arr.length === 0) return 0;
  if (arr.length === 1) return arr[0];
  const a = arr.slice();
  const n = a.length;

  function partition(lo: number, hi: number, pivotIdx: number): number {
    const pivotVal = a[pivotIdx];
    [a[pivotIdx], a[hi]] = [a[hi], a[pivotIdx]];
    let storeIdx = lo;
    for (let i = lo; i < hi; i++) {
      if (a[i] < pivotVal) {
        [a[storeIdx], a[i]] = [a[i], a[storeIdx]];
        storeIdx++;
      }
    }
    [a[storeIdx], a[hi]] = [a[hi], a[storeIdx]];
    return storeIdx;
  }

  function select(lo: number, hi: number, k: number): number {
    if (lo === hi) return a[lo];
    const pivotIdx = lo + ((hi - lo) >> 1);
    const p = partition(lo, hi, pivotIdx);
    if (k === p) return a[k];
    return k < p ? select(lo, p - 1, k) : select(p + 1, hi, k);
  }

  const mid = n >> 1;
  if (n % 2 === 1) return select(0, n - 1, mid);
  const upper = select(0, n - 1, mid);
  const lower = select(0, n - 1, mid - 1);
  return (lower + upper) / 2;
}

function computeMetricsInternal(data: PredictionRealityEntry[]): PredictionRealityMetrics {
  if (data.length === 0) {
    return {
      count: 0, rmse: 0, mae: 0, bias: 0, r2: 0, maxError: 0,
      medianAbsError: 0, stabilityAccuracy: 0, stabilityBalancedAccuracy: 0,
      stabilityF1: 0, feMAE: null,
      pct_within_10K: 0, pct_within_30K: 0, pct_within_50K: 0,
      pct_within_20_percent: 0, smallSampleWarning: true,
    };
  }

  const n = data.length;
  let sumSqErr = 0;
  let sumAbsErr = 0;
  let sumSignedErr = 0;
  let maxErr = 0;
  let feSumAbs = 0;
  let feCount = 0;
  let within10 = 0;
  let within30 = 0;
  let within50 = 0;
  let within20pct = 0;
  const absErrors: number[] = new Array(n);

  let tp = 0, fp = 0, tn = 0, fn = 0;

  const actualMean = data.reduce((s, e) => s + e.ground_truth.Tc, 0) / n;
  let ssTot = 0;

  for (let i = 0; i < n; i++) {
    const e = data[i];
    sumSqErr += e.error.tc_squared_error;
    sumAbsErr += e.error.tc_abs_error;
    sumSignedErr += e.error.tc_error;
    if (e.error.tc_abs_error > maxErr) maxErr = e.error.tc_abs_error;

    if (e.model_prediction.stable && e.ground_truth.stable) tp++;
    else if (e.model_prediction.stable && !e.ground_truth.stable) fp++;
    else if (!e.model_prediction.stable && e.ground_truth.stable) fn++;
    else tn++;

    if (e.error.fe_error !== null) {
      feSumAbs += Math.abs(e.error.fe_error);
      feCount++;
    }
    if (e.error.tc_abs_error <= 10) within10++;
    if (e.error.tc_abs_error <= 30) within30++;
    if (e.error.tc_abs_error <= 50) within50++;

    const actualTc = Math.abs(e.ground_truth.Tc);
    if (actualTc > 0.5) {
      if (e.error.tc_abs_error / actualTc <= 0.2) within20pct++;
    } else {
      if (e.error.tc_abs_error <= 1.0) within20pct++;
    }

    absErrors[i] = e.error.tc_abs_error;
    ssTot += (e.ground_truth.Tc - actualMean) ** 2;
  }

  const medianAbsError = quickSelectMedian(absErrors);
  const r2 = ssTot > 0 ? 1 - sumSqErr / ssTot : 0;

  const sensitivity = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const specificity = (tn + fp) > 0 ? tn / (tn + fp) : 0;
  const balancedAccuracy = (sensitivity + specificity) / 2;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const f1 = (precision + sensitivity) > 0
    ? 2 * (precision * sensitivity) / (precision + sensitivity)
    : 0;

  return {
    count: n,
    rmse: Math.sqrt(sumSqErr / n),
    mae: sumAbsErr / n,
    bias: sumSignedErr / n,
    r2,
    maxError: maxErr,
    medianAbsError,
    stabilityAccuracy: (tp + tn) / n,
    stabilityBalancedAccuracy: balancedAccuracy,
    stabilityF1: f1,
    feMAE: feCount > 0 ? feSumAbs / feCount : null,
    pct_within_10K: within10 / n,
    pct_within_30K: within30 / n,
    pct_within_50K: within50 / n,
    pct_within_20_percent: within20pct / n,
    smallSampleWarning: n < 10,
  };
}

export function computeMetrics(entries?: PredictionRealityEntry[]): PredictionRealityMetrics {
  if (!entries) {
    if (!cachedGlobalMetricsDirty && cachedGlobalMetrics) return cachedGlobalMetrics;
    cachedGlobalMetrics = computeMetricsInternal(ledger);
    cachedGlobalMetricsDirty = false;
    return cachedGlobalMetrics;
  }
  return computeMetricsInternal(entries);
}

export function computeRecentMetrics(windowSize: number = 100): PredictionRealityMetrics {
  return computeMetrics(ledger.slice(-windowSize));
}

export function computeMetricsByFamily(): FamilyMetrics[] {
  const families: Record<string, PredictionRealityEntry[]> = {};
  for (const e of ledger) {
    const fam = classifyFamily(e.formula);
    if (!families[fam]) families[fam] = [];
    families[fam].push(e);
  }

  const results: FamilyMetrics[] = [];
  for (const [family, entries] of Object.entries(families)) {
    if (entries.length < 2) continue;
    const n = entries.length;
    let sumSq = 0;
    let sumAbs = 0;
    let sumSigned = 0;
    for (const e of entries) {
      sumSq += e.error.tc_squared_error;
      sumAbs += e.error.tc_abs_error;
      sumSigned += e.error.tc_error;
    }
    results.push({
      family,
      count: n,
      rmse: Math.sqrt(sumSq / n),
      mae: sumAbs / n,
      bias: sumSigned / n,
      smallSampleWarning: n < 10,
    });
  }

  return results.sort((a, b) => b.mae - a.mae);
}


const WORST_HEAP_SIZE = 20;
let worstHeap: PredictionRealityEntry[] = [];
let worstHeapDirty = true;

function rebuildWorstHeap(): void {
  if (!worstHeapDirty) return;
  worstHeap = [];
  for (const e of ledger) {
    insertIntoWorstHeap(e);
  }
  worstHeapDirty = false;
}

function insertIntoWorstHeap(entry: PredictionRealityEntry): void {
  const err = entry.error.tc_abs_error;
  if (worstHeap.length < WORST_HEAP_SIZE) {
    worstHeap.push(entry);
    worstHeap.sort((a, b) => a.error.tc_abs_error - b.error.tc_abs_error);
  } else if (err > worstHeap[0].error.tc_abs_error) {
    worstHeap[0] = entry;
    worstHeap.sort((a, b) => a.error.tc_abs_error - b.error.tc_abs_error);
  }
}

export function getWorstPredictions(topN: number = 20): PredictionRealityEntry[] {
  if (topN === WORST_HEAP_SIZE) {
    rebuildWorstHeap();
    return [...worstHeap].reverse();
  }
  return [...ledger]
    .sort((a, b) => b.error.tc_abs_error - a.error.tc_abs_error)
    .slice(0, topN);
}

export function getBestPredictions(topN: number = 20): PredictionRealityEntry[] {
  return [...ledger]
    .sort((a, b) => a.error.tc_abs_error - b.error.tc_abs_error)
    .slice(0, topN);
}

export function getOverpredictions(minError: number = 30): PredictionRealityEntry[] {
  return ledger.filter(e => e.error.tc_error > minError);
}

export function getUnderpredictions(minError: number = 30): PredictionRealityEntry[] {
  return ledger.filter(e => e.error.tc_error < -minError);
}

export function checkRetrainTrigger(): {
  shouldRetrain: boolean;
  samplesSinceLastRetrain: number;
  threshold: number;
  reason: string;
} {
  if (!retrainConfig.enabled) {
    return {
      shouldRetrain: false,
      samplesSinceLastRetrain,
      threshold: retrainConfig.sampleThreshold,
      reason: "Sample-count trigger disabled",
    };
  }

  if (samplesSinceLastRetrain >= retrainConfig.sampleThreshold) {
    return {
      shouldRetrain: true,
      samplesSinceLastRetrain,
      threshold: retrainConfig.sampleThreshold,
      reason: `${samplesSinceLastRetrain} new samples >= threshold ${retrainConfig.sampleThreshold}`,
    };
  }

  return {
    shouldRetrain: false,
    samplesSinceLastRetrain,
    threshold: retrainConfig.sampleThreshold,
    reason: `${samplesSinceLastRetrain}/${retrainConfig.sampleThreshold} samples (${((samplesSinceLastRetrain / retrainConfig.sampleThreshold) * 100).toFixed(0)}%)`,
  };
}

export function acknowledgeRetrain(): void {
  totalRetrainTriggers++;
  lastTriggerTimestamp = Date.now();
  lastTriggerSampleCount = samplesSinceLastRetrain;
  samplesSinceLastRetrain = 0;
}

export function setRetrainThreshold(threshold: number): void {
  retrainConfig.sampleThreshold = Math.max(1, Math.min(1000, threshold));
}

export function setRetrainEnabled(enabled: boolean): void {
  retrainConfig.enabled = enabled;
}

export function getRetrainTriggerState(): RetrainTriggerState {
  return {
    config: { ...retrainConfig },
    samplesSinceLastRetrain,
    totalTriggers: totalRetrainTriggers,
    lastTriggerTimestamp,
    lastTriggerSampleCount,
  };
}

export interface CycleImprovementRecord {
  cycle: number;
  timestamp: number;
  count: number;
  rmse: number;
  mae: number;
  bias: number;
  r2: number;
  stabilityAccuracy: number;
  gnn_r2: number;
  gnn_rmse: number;
  xgb_r2: number;
  xgb_rmse: number;
  ciCoverage90: number;
  ciCoverage95: number;
  ciCoverage99: number;
  /** MAE computed only on ledger entries that came from a real DFT/DFPT run — the true accuracy metric */
  dftVerifiedMAE: number | null;
}

const cycleImprovementHistory: CycleImprovementRecord[] = [];
const MAX_CYCLE_HISTORY = 200;

interface CICoverageResult {
  coverage90: number;
  coverage95: number;
  coverage99: number;
}
let cachedCICoverage: CICoverageResult | null = null;
let cachedCICoverageDirty = true;

function computeCICoverageAll(entries: PredictionRealityEntry[]): CICoverageResult {
  if (!cachedCICoverageDirty && cachedCICoverage) return cachedCICoverage;

  const z90 = 1.645, z95 = 1.96, z99 = 2.576;
  let covered90 = 0, covered95 = 0, covered99 = 0;
  let total = 0;

  for (const e of entries) {
    const pred = e.model_prediction?.Tc;
    const actual = e.ground_truth?.Tc;
    if (!Number.isFinite(pred) || !Number.isFinite(actual)) continue;
    if (e.predicted_sigma == null || !Number.isFinite(e.predicted_sigma) || e.predicted_sigma <= 0) continue;
    const sigma = e.predicted_sigma;
    const hw90 = z90 * sigma;
    const hw95 = z95 * sigma;
    const hw99 = z99 * sigma;
    const err = Math.abs(actual - pred);
    if (err <= hw90) covered90++;
    if (err <= hw95) covered95++;
    if (err <= hw99) covered99++;
    total++;
  }

  const result: CICoverageResult = {
    coverage90: total >= 5 ? covered90 / total : NaN,
    coverage95: total >= 5 ? covered95 / total : NaN,
    coverage99: total >= 5 ? covered99 / total : NaN,
  };
  cachedCICoverage = result;
  cachedCICoverageDirty = false;
  return result;
}

export function recordCycleImprovement(
  cycle: number,
  gnnMetrics: { r2: number; rmse: number },
  xgbMetrics: { r2: number; rmse: number }
): CycleImprovementRecord {
  const allMetrics = computeMetrics();
  const ci = computeCICoverageAll(ledger);

  const ciCoverage90 = Number.isFinite(ci.coverage90) ? ci.coverage90 : 0.90;
  const ciCoverage95 = Number.isFinite(ci.coverage95) ? ci.coverage95 : 0.95;
  const ciCoverage99 = Number.isFinite(ci.coverage99) ? ci.coverage99 : 0.99;

  // Compute MAE restricted to ledger entries whose ground truth came from a real DFT/DFPT run.
  // This is the primary accuracy metric — ML-estimated Tc labels can self-reinforce errors.
  const dftEntries = ledger.filter(e => e.source === "dft" || e.source.startsWith("dfpt"));
  const dftVerifiedMAE = dftEntries.length > 0
    ? dftEntries.reduce((s, e) => s + e.error.tc_abs_error, 0) / dftEntries.length
    : null;

  const record: CycleImprovementRecord = {
    cycle,
    timestamp: Date.now(),
    count: allMetrics.count,
    rmse: allMetrics.rmse,
    mae: allMetrics.mae,
    bias: allMetrics.bias,
    r2: allMetrics.r2,
    stabilityAccuracy: allMetrics.stabilityAccuracy,
    gnn_r2: gnnMetrics.r2,
    gnn_rmse: gnnMetrics.rmse,
    xgb_r2: xgbMetrics.r2,
    xgb_rmse: xgbMetrics.rmse,
    ciCoverage90: Math.round(ciCoverage90 * 10000) / 10000,
    ciCoverage95: Math.round(ciCoverage95 * 10000) / 10000,
    ciCoverage99: Math.round(ciCoverage99 * 10000) / 10000,
    dftVerifiedMAE: dftVerifiedMAE != null ? Math.round(dftVerifiedMAE * 100) / 100 : null,
  };

  cycleImprovementHistory.push(record);
  if (cycleImprovementHistory.length > MAX_CYCLE_HISTORY) {
    cycleImprovementHistory.shift();
  }

  const trend = cycleImprovementHistory.length >= 2
    ? cycleImprovementHistory[cycleImprovementHistory.length - 1].rmse - cycleImprovementHistory[cycleImprovementHistory.length - 2].rmse
    : 0;
  const trendDir = trend < -1 ? "improving" : trend > 1 ? "degrading" : "stable";

  const hasCIData = Number.isFinite(ci.coverage95);
  const coverageStr = hasCIData ? `${(ciCoverage95 * 100).toFixed(1)}%` : "N/A (no predicted_sigma)";
  const coverageNote = hasCIData
    ? (ciCoverage95 < 0.90 ? " [CI95 MISCALIBRATED]" : ciCoverage95 > 0.99 ? " [CI95 OVERCONSERVATIVE]" : "")
    : "";
  const dftMaeStr = record.dftVerifiedMAE != null
    ? `, DFT-MAE=${record.dftVerifiedMAE.toFixed(2)}K (n=${dftEntries.length})`
    : "";
  console.log(
    `[Prediction Ledger] Cycle ${cycle}: Tc RMSE=${record.rmse.toFixed(2)}K, MAE=${record.mae.toFixed(2)}K${dftMaeStr}, ` +
    `R²=${record.r2.toFixed(4)}, GNN R²=${gnnMetrics.r2.toFixed(4)}, XGB R²=${xgbMetrics.r2.toFixed(4)}, ` +
    `CI95 coverage=${coverageStr} (goal: 95%) [${trendDir}]${coverageNote}`
  );

  return record;
}

export function getCycleImprovementHistory(): CycleImprovementRecord[] {
  return [...cycleImprovementHistory];
}

export function getImprovementTrend(windowSize: number = 5): {
  improving: boolean;
  rmseTrend: number[];
  maeTrend: number[];
  r2Trend: number[];
  avgRmseChange: number;
  ewmaRmseChange: number;
} {
  const recent = cycleImprovementHistory.slice(-windowSize);
  const rmseTrend = recent.map(r => r.rmse);
  const maeTrend = recent.map(r => r.mae);
  const r2Trend = recent.map(r => r.r2);

  let avgRmseChange = 0;
  let ewmaRmseChange = 0;
  if (rmseTrend.length >= 2) {
    const alpha = 2 / (rmseTrend.length + 1);
    let ewma = rmseTrend[1] - rmseTrend[0];
    let sumChanges = ewma;
    for (let i = 2; i < rmseTrend.length; i++) {
      const change = rmseTrend[i] - rmseTrend[i - 1];
      ewma = alpha * change + (1 - alpha) * ewma;
      sumChanges += change;
    }
    avgRmseChange = sumChanges / (rmseTrend.length - 1);
    ewmaRmseChange = ewma;
  }

  return {
    improving: ewmaRmseChange < 0,
    rmseTrend,
    maeTrend,
    r2Trend,
    avgRmseChange,
    ewmaRmseChange,
  };
}

export function getLedgerSize(): number {
  return ledger.length;
}

export function getLedgerSlice(offset: number, limit: number): PredictionRealityEntry[] {
  return ledger.slice(offset, offset + limit);
}

export function getMetricsForLLM(): string {
  const all = computeMetrics();
  const recent = computeRecentMetrics(50);
  const byFamily = computeMetricsByFamily();
  const trigger = checkRetrainTrigger();

  const p1 = (v: number) => (v * 100).toFixed(1);
  const k2 = (v: number) => v.toFixed(2);

  const lines: string[] = [
    "=== Prediction vs Reality Ledger ===",
    `Total entries: ${all.count}${all.smallSampleWarning ? " [SMALL SAMPLE WARNING: n<10, metrics unreliable]" : ""}`,
    `Overall: RMSE=${k2(all.rmse)}K, MAE=${k2(all.mae)}K, bias=${k2(all.bias)}K, R²=${all.r2.toFixed(4)}`,
    `Accuracy: within10K=${p1(all.pct_within_10K)}%, within30K=${p1(all.pct_within_30K)}%, within50K=${p1(all.pct_within_50K)}%, within20%=${p1(all.pct_within_20_percent)}%`,
    `Stability: accuracy=${p1(all.stabilityAccuracy)}%, balanced=${p1(all.stabilityBalancedAccuracy)}%, F1=${all.stabilityF1.toFixed(4)}`,
    `MaxError: ${k2(all.maxError)}K, MedianAbsError: ${k2(all.medianAbsError)}K`,
  ];

  if (recent.count > 0 && recent.count !== all.count) {
    lines.push(`Recent (last ${recent.count}): RMSE=${k2(recent.rmse)}K, MAE=${k2(recent.mae)}K, bias=${k2(recent.bias)}K, R²=${recent.r2.toFixed(4)}`);
  }

  if (byFamily.length > 0) {
    lines.push("Per-family error:");
    for (const f of byFamily.slice(0, 5)) {
      const warn = f.smallSampleWarning ? " [small sample]" : "";
      lines.push(`  ${f.family}: MAE=${k2(f.mae)}K, bias=${k2(f.bias)}K, RMSE=${k2(f.rmse)}K (n=${f.count})${warn}`);
    }
  }

  lines.push(`Retrain trigger: ${trigger.reason}`);

  if (cycleImprovementHistory.length > 0) {
    lines.push("Cycle improvement trend:");
    const recentCycles = cycleImprovementHistory.slice(-10);
    for (const r of recentCycles) {
      lines.push(`  cycle ${r.cycle}: RMSE=${k2(r.rmse)}K, R²=${r.r2.toFixed(4)}`);
    }
    const trend = getImprovementTrend();
    lines.push(`  Trend: ${trend.improving ? "improving" : "not improving"} (EWMA RMSE change=${k2(trend.ewmaRmseChange)}K/cycle, avg=${k2(trend.avgRmseChange)}K/cycle)`);
  }

  return lines.join("\n");
}
