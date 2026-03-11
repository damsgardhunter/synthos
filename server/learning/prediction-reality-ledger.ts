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
  feMAE: number | null;
  pct_within_10K: number;
  pct_within_30K: number;
  pct_within_50K: number;
}

export interface FamilyMetrics {
  family: string;
  count: number;
  rmse: number;
  mae: number;
  bias: number;
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

export function recordPredictionVsReality(
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
): PredictionRealityEntry {
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
    const { gnnPredictWithUncertainty } = require("./graph-neural-net");
    const { gbPredictWithUncertainty } = require("./gradient-boost");
    const { extractFeatures } = require("./ml-predictor");
    let gnnSigma = 0, xgbSigma = 0;
    try {
      const gnnRes = gnnPredictWithUncertainty(formula);
      if (Number.isFinite(gnnRes.totalStd) && gnnRes.totalStd > 0) gnnSigma = gnnRes.totalStd;
    } catch {}
    try {
      const feats = extractFeatures(formula);
      const xgbRes = gbPredictWithUncertainty(feats, formula);
      if (Number.isFinite(xgbRes.totalStd) && xgbRes.totalStd > 0) xgbSigma = xgbRes.totalStd;
    } catch {}
    if (gnnSigma > 0 && xgbSigma > 0) {
      predicted_sigma = Math.sqrt(1 / (1 / (gnnSigma ** 2) + 1 / (xgbSigma ** 2)));
    } else if (xgbSigma > 0) {
      predicted_sigma = xgbSigma;
    } else if (gnnSigma > 0) {
      predicted_sigma = gnnSigma;
    } else {
      predicted_sigma = Math.max(10, Math.abs(prediction.Tc) * 0.3);
    }
  } catch {
    predicted_sigma = Math.max(10, Math.abs(prediction.Tc) * 0.3);
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
  }

  samplesSinceLastRetrain++;

  try {
    const { notifyNewLedgerEntry } = require("./conformal-calibrator");
    notifyNewLedgerEntry();
  } catch {}

  return entry;
}

export function computeMetrics(entries?: PredictionRealityEntry[]): PredictionRealityMetrics {
  const data = entries ?? ledger;
  if (data.length === 0) {
    return {
      count: 0, rmse: 0, mae: 0, bias: 0, r2: 0, maxError: 0,
      medianAbsError: 0, stabilityAccuracy: 0, feMAE: null,
      pct_within_10K: 0, pct_within_30K: 0, pct_within_50K: 0,
    };
  }

  const n = data.length;
  let sumSqErr = 0;
  let sumAbsErr = 0;
  let sumSignedErr = 0;
  let maxErr = 0;
  let stabilityCorrect = 0;
  let feSumAbs = 0;
  let feCount = 0;
  let within10 = 0;
  let within30 = 0;
  let within50 = 0;
  const absErrors: number[] = [];

  const actualMean = data.reduce((s, e) => s + e.ground_truth.Tc, 0) / n;
  let ssTot = 0;

  for (const e of data) {
    sumSqErr += e.error.tc_squared_error;
    sumAbsErr += e.error.tc_abs_error;
    sumSignedErr += e.error.tc_error;
    if (e.error.tc_abs_error > maxErr) maxErr = e.error.tc_abs_error;
    if (e.error.stability_correct) stabilityCorrect++;
    if (e.error.fe_error !== null) {
      feSumAbs += Math.abs(e.error.fe_error);
      feCount++;
    }
    if (e.error.tc_abs_error <= 10) within10++;
    if (e.error.tc_abs_error <= 30) within30++;
    if (e.error.tc_abs_error <= 50) within50++;
    absErrors.push(e.error.tc_abs_error);
    ssTot += (e.ground_truth.Tc - actualMean) ** 2;
  }

  absErrors.sort((a, b) => a - b);
  const medianIdx = Math.floor(absErrors.length / 2);
  const medianAbsError = absErrors.length % 2 === 0
    ? (absErrors[medianIdx - 1] + absErrors[medianIdx]) / 2
    : absErrors[medianIdx];

  const r2 = ssTot > 0 ? 1 - sumSqErr / ssTot : 0;

  return {
    count: n,
    rmse: Math.sqrt(sumSqErr / n),
    mae: sumAbsErr / n,
    bias: sumSignedErr / n,
    r2,
    maxError: maxErr,
    medianAbsError,
    stabilityAccuracy: stabilityCorrect / n,
    feMAE: feCount > 0 ? feSumAbs / feCount : null,
    pct_within_10K: within10 / n,
    pct_within_30K: within30 / n,
    pct_within_50K: within50 / n,
  };
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
    });
  }

  return results.sort((a, b) => b.mae - a.mae);
}

function classifyFamily(formula: string): string {
  const f = formula.toLowerCase();
  if (f.includes("h") && /\d/.test(f)) {
    const hMatch = f.match(/h(\d+)/);
    if (hMatch && parseInt(hMatch[1]) >= 3) return "hydride";
  }
  if (f.includes("cu") && f.includes("o")) return "cuprate";
  if (f.includes("fe") && (f.includes("as") || f.includes("se"))) return "pnictide";
  if (f.includes("nb") || f.includes("v3")) return "a15";
  if (f.includes("mg") && f.includes("b")) return "mgb2-type";
  if (f.includes("bi") && f.includes("s")) return "chalcogenide";
  return "other";
}

export function getWorstPredictions(topN: number = 20): PredictionRealityEntry[] {
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
}

const cycleImprovementHistory: CycleImprovementRecord[] = [];
const MAX_CYCLE_HISTORY = 200;

function computeCICoverage(entries: PredictionRealityEntry[], zMultiplier: number): number {
  let covered = 0;
  let total = 0;
  for (const e of entries) {
    const pred = e.model_prediction?.Tc;
    const actual = e.ground_truth?.Tc;
    if (!Number.isFinite(pred) || !Number.isFinite(actual)) continue;
    let sigma: number;
    if (e.predicted_sigma != null && Number.isFinite(e.predicted_sigma) && e.predicted_sigma > 0) {
      sigma = e.predicted_sigma;
    } else {
      sigma = Math.max(Math.abs(e.error?.tc_abs_error ?? 10), 1);
    }
    const halfWidth = zMultiplier * sigma;
    if (actual >= pred - halfWidth && actual <= pred + halfWidth) covered++;
    total++;
  }
  return total > 0 ? covered / total : zMultiplier > 2 ? 0.99 : zMultiplier > 1.9 ? 0.95 : 0.90;
}

export function recordCycleImprovement(
  cycle: number,
  gnnMetrics: { r2: number; rmse: number },
  xgbMetrics: { r2: number; rmse: number }
): CycleImprovementRecord {
  const cycleEntries = ledger.filter(e => e.cycle === cycle);
  const allMetrics = computeMetrics();

  const ciCoverage90 = computeCICoverage(ledger, 1.645);
  const ciCoverage95 = computeCICoverage(ledger, 1.96);
  const ciCoverage99 = computeCICoverage(ledger, 2.576);

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
  };

  cycleImprovementHistory.push(record);
  if (cycleImprovementHistory.length > MAX_CYCLE_HISTORY) {
    cycleImprovementHistory.shift();
  }

  const trend = cycleImprovementHistory.length >= 2
    ? cycleImprovementHistory[cycleImprovementHistory.length - 1].rmse - cycleImprovementHistory[cycleImprovementHistory.length - 2].rmse
    : 0;
  const trendDir = trend < -1 ? "improving" : trend > 1 ? "degrading" : "stable";

  const coverageNote = ciCoverage95 < 0.90 ? " [CI95 MISCALIBRATED]" : ciCoverage95 > 0.99 ? " [CI95 OVERCONSERVATIVE]" : "";
  console.log(
    `[Prediction Ledger] Cycle ${cycle}: Tc RMSE=${record.rmse.toFixed(2)}K, MAE=${record.mae.toFixed(2)}K, ` +
    `R²=${record.r2.toFixed(4)}, GNN R²=${gnnMetrics.r2.toFixed(4)}, XGB R²=${xgbMetrics.r2.toFixed(4)}, ` +
    `CI95 coverage=${(ciCoverage95 * 100).toFixed(1)}% (goal: 95%) [${trendDir}]${coverageNote}`
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
} {
  const recent = cycleImprovementHistory.slice(-windowSize);
  const rmseTrend = recent.map(r => r.rmse);
  const maeTrend = recent.map(r => r.mae);
  const r2Trend = recent.map(r => r.r2);

  let avgRmseChange = 0;
  if (rmseTrend.length >= 2) {
    const changes: number[] = [];
    for (let i = 1; i < rmseTrend.length; i++) {
      changes.push(rmseTrend[i] - rmseTrend[i - 1]);
    }
    avgRmseChange = changes.reduce((s, c) => s + c, 0) / changes.length;
  }

  return {
    improving: avgRmseChange < 0,
    rmseTrend,
    maeTrend,
    r2Trend,
    avgRmseChange,
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

  const lines: string[] = [
    "=== Prediction vs Reality Ledger ===",
    `Total entries: ${all.count}`,
    `Overall: RMSE=${all.rmse.toFixed(2)}K, MAE=${all.mae.toFixed(2)}K, bias=${all.bias.toFixed(2)}K, R²=${all.r2.toFixed(4)}`,
    `Accuracy: within 10K=${(all.pct_within_10K * 100).toFixed(1)}%, within 30K=${(all.pct_within_30K * 100).toFixed(1)}%, within 50K=${(all.pct_within_50K * 100).toFixed(1)}%`,
    `Stability accuracy: ${(all.stabilityAccuracy * 100).toFixed(1)}%`,
    `Max error: ${all.maxError.toFixed(1)}K, Median abs error: ${all.medianAbsError.toFixed(1)}K`,
  ];

  if (recent.count > 0 && recent.count !== all.count) {
    lines.push(`Recent (last ${recent.count}): RMSE=${recent.rmse.toFixed(2)}K, MAE=${recent.mae.toFixed(2)}K, bias=${recent.bias.toFixed(2)}K`);
  }

  if (byFamily.length > 0) {
    lines.push("Per-family error:");
    for (const f of byFamily.slice(0, 5)) {
      lines.push(`  ${f.family}: MAE=${f.mae.toFixed(2)}K, bias=${f.bias.toFixed(2)}K (n=${f.count})`);
    }
  }

  lines.push(`Retrain trigger: ${trigger.reason}`);

  if (cycleImprovementHistory.length > 0) {
    lines.push("Cycle improvement trend:");
    const recent = cycleImprovementHistory.slice(-10);
    for (const r of recent) {
      lines.push(`  cycle ${r.cycle}: Tc RMSE=${r.rmse.toFixed(2)}K, R²=${r.r2.toFixed(4)}`);
    }
    const trend = getImprovementTrend();
    lines.push(`  Trend: ${trend.improving ? "improving" : "not improving"} (avg RMSE change=${trend.avgRmseChange.toFixed(2)}K/cycle)`);
  }

  return lines.join("\n");
}
