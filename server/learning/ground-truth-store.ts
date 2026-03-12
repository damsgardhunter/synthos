export interface ModelPrediction {
  predicted_Tc: number;
  predicted_stable: boolean;
  predicted_formation_energy: number | null;
  xgboost_Tc: number;
  gnn_Tc: number;
  ensemble_score: number;
}

export interface GroundTruthDatapoint {
  formula: string;
  structure: string | null;
  pressure: number;
  formation_energy: number | null;
  DOS_EF: number | null;
  band_gap: number | null;
  lambda: number | null;
  omega_log: number | null;
  Tc: number;
  phonon_stable: boolean;
  is_strong_coupling: boolean;
  mu_star: number | null;
  model_prediction: ModelPrediction | null;
  source: "full-dft" | "xtb" | "surrogate" | "external" | "active-learning";
  confidence: "high" | "medium" | "low";
  cycle: number;
  timestamp: number;
  evaluation_wall_time_ms: number;
}

export interface BatchCycle {
  cycleNumber: number;
  startedAt: number;
  completedAt: number;
  candidatesSubmitted: number;
  evaluationSuccesses: number;
  evaluationFailures: number;
  newDatapoints: number;
  retrainTriggered: boolean;
  preRetrainMetrics: { r2: number; mae: number; datasetSize: number } | null;
  postRetrainMetrics: { r2: number; mae: number; datasetSize: number } | null;
  r2Improvement: number;
  maeImprovement: number;
  bestTcThisCycle: number;
  avgUncertaintyBefore: number;
  avgUncertaintyAfter: number;
}

export interface GroundTruthDatasetSummary {
  totalDatapoints: number;
  totalCycles: number;
  bySource: Record<string, number>;
  byConfidence: Record<string, number>;
  avgTc: number;
  maxTc: number;
  stableCount: number;
  unstableCount: number;
  avgLambda: number;
  avgDOS_EF: number;
  pressureDistribution: { min: number; max: number; mean: number };
  cumulativeR2Improvement: number;
}

const MAX_DATAPOINTS = 5000;
const MAX_CYCLES = 200;
const datapoints: GroundTruthDatapoint[] = [];
const cycles: BatchCycle[] = [];
let currentCycleNumber = 0;

export function addGroundTruthDatapoint(dp: GroundTruthDatapoint): void {
  const existing = datapoints.findIndex(
    d => d.formula === dp.formula && Math.abs(d.pressure - dp.pressure) < 0.5
  );

  if (existing !== -1) {
    const old = datapoints[existing];
    const sourcePriority: Record<string, number> = {
      "active-learning": 0, "surrogate": 1, "xtb": 2, "external": 3, "full-dft": 4,
    };
    if ((sourcePriority[dp.source] ?? 0) >= (sourcePriority[old.source] ?? 0)) {
      datapoints[existing] = dp;
      try { require("./unified-training-dataset").invalidateUnifiedCache(); } catch {}
    }
    return;
  }

  datapoints.push(dp);
  try { require("./unified-training-dataset").invalidateUnifiedCache(); } catch {}
  if (datapoints.length > MAX_DATAPOINTS) {
    const excess = datapoints.length - MAX_DATAPOINTS;
    const prunable: number[] = [];
    const protectedSources = new Set(["external", "full-dft"]);
    for (let i = 0; i < datapoints.length && prunable.length < excess; i++) {
      if (!protectedSources.has(datapoints[i].source)) {
        prunable.push(i);
      }
    }
    if (prunable.length >= excess) {
      for (let k = prunable.length - 1; k >= 0; k--) {
        datapoints.splice(prunable[k], 1);
      }
    } else {
      datapoints.splice(0, excess);
    }
  }
}

export function addBatchFromEvaluation(
  formula: string,
  pressure: number,
  results: {
    tc: number;
    lambda: number;
    omegaLog: number;
    dosAtEF: number;
    formationEnergy: number | null;
    bandGap: number | null;
    phononStable: boolean;
    isStrongCoupling: boolean;
    muStar: number;
    tier: string;
    confidence: string;
    wallTimeMs: number;
    structure?: string | null;
  },
  cycle: number,
  prediction?: ModelPrediction | null
): GroundTruthDatapoint {
  const sourceMap: Record<string, GroundTruthDatapoint["source"]> = {
    "full-dft": "full-dft",
    "xtb": "xtb",
    "surrogate": "surrogate",
    "external": "external",
  };

  const dp: GroundTruthDatapoint = {
    formula,
    structure: results.structure ?? null,
    pressure,
    formation_energy: results.formationEnergy,
    DOS_EF: results.dosAtEF > 0 ? results.dosAtEF : null,
    band_gap: results.bandGap,
    lambda: results.lambda > 0 ? results.lambda : null,
    omega_log: results.omegaLog > 0 ? results.omegaLog : null,
    Tc: Math.max(0, results.tc),
    phonon_stable: results.phononStable,
    is_strong_coupling: results.isStrongCoupling,
    mu_star: results.muStar > 0 ? results.muStar : null,
    model_prediction: prediction ?? null,
    source: sourceMap[results.tier] ?? "active-learning",
    confidence: (results.confidence === "high" || results.confidence === "medium" || results.confidence === "low")
      ? results.confidence as any
      : "low",
    cycle,
    timestamp: Date.now(),
    evaluation_wall_time_ms: results.wallTimeMs,
  };

  addGroundTruthDatapoint(dp);
  return dp;
}

export function startNewBatchCycle(): number {
  currentCycleNumber++;
  return currentCycleNumber;
}

export function getCurrentCycleNumber(): number {
  return currentCycleNumber;
}

export function recordBatchCycle(cycle: BatchCycle): void {
  cycles.push(cycle);
  if (cycles.length > MAX_CYCLES) {
    cycles.splice(0, cycles.length - MAX_CYCLES);
  }
}

export function getGroundTruthDataset(): GroundTruthDatapoint[] {
  return [...datapoints];
}

export function getGroundTruthDatasetSlice(offset: number, limit: number): GroundTruthDatapoint[] {
  return datapoints.slice(offset, offset + limit);
}

export function getBatchCycles(): BatchCycle[] {
  return [...cycles];
}

export function getRecentBatchCycles(n: number = 10): BatchCycle[] {
  return cycles.slice(-n);
}

export function getGroundTruthSummary(): GroundTruthDatasetSummary {
  const bySource: Record<string, number> = {};
  const byConfidence: Record<string, number> = {};
  let tcSum = 0;
  let maxTc = 0;
  let stableCount = 0;
  let unstableCount = 0;
  let lambdaSum = 0;
  let lambdaCount = 0;
  let dosSum = 0;
  let dosCount = 0;
  let pressureSum = 0;
  let pressureMin = Infinity;
  let pressureMax = -Infinity;

  for (const dp of datapoints) {
    bySource[dp.source] = (bySource[dp.source] ?? 0) + 1;
    byConfidence[dp.confidence] = (byConfidence[dp.confidence] ?? 0) + 1;
    tcSum += dp.Tc;
    if (dp.Tc > maxTc) maxTc = dp.Tc;
    if (dp.phonon_stable) stableCount++;
    else unstableCount++;
    if (dp.lambda !== null) { lambdaSum += dp.lambda; lambdaCount++; }
    if (dp.DOS_EF !== null) { dosSum += dp.DOS_EF; dosCount++; }
    pressureSum += dp.pressure;
    if (dp.pressure < pressureMin) pressureMin = dp.pressure;
    if (dp.pressure > pressureMax) pressureMax = dp.pressure;
  }

  const n = datapoints.length || 1;

  let cumulativeR2 = 0;
  for (const c of cycles) {
    cumulativeR2 += c.r2Improvement;
  }

  return {
    totalDatapoints: datapoints.length,
    totalCycles: cycles.length,
    bySource,
    byConfidence,
    avgTc: tcSum / n,
    maxTc,
    stableCount,
    unstableCount,
    avgLambda: lambdaCount > 0 ? lambdaSum / lambdaCount : 0,
    avgDOS_EF: dosCount > 0 ? dosSum / dosCount : 0,
    pressureDistribution: {
      min: pressureMin === Infinity ? 0 : pressureMin,
      max: pressureMax === -Infinity ? 0 : pressureMax,
      mean: pressureSum / n,
    },
    cumulativeR2Improvement: cumulativeR2,
  };
}

export function getGroundTruthForLLM(): string {
  const summary = getGroundTruthSummary();
  const recentCycles = getRecentBatchCycles(5);

  const lines: string[] = [
    "=== Ground Truth Dataset ===",
    `Total datapoints: ${summary.totalDatapoints}`,
    `Total batch cycles: ${summary.totalCycles}`,
    `Sources: ${Object.entries(summary.bySource).map(([k, v]) => `${k}=${v}`).join(", ") || "none"}`,
    `Confidence: ${Object.entries(summary.byConfidence).map(([k, v]) => `${k}=${v}`).join(", ") || "none"}`,
    `Avg Tc: ${summary.avgTc.toFixed(1)}K, Max Tc: ${summary.maxTc.toFixed(1)}K`,
    `Stable: ${summary.stableCount}, Unstable: ${summary.unstableCount}`,
    `Avg lambda: ${summary.avgLambda.toFixed(3)}, Avg DOS(EF): ${summary.avgDOS_EF.toFixed(3)}`,
    `Pressure range: ${summary.pressureDistribution.min.toFixed(1)}-${summary.pressureDistribution.max.toFixed(1)} GPa (mean=${summary.pressureDistribution.mean.toFixed(1)})`,
    `Cumulative R² improvement: ${summary.cumulativeR2Improvement.toFixed(4)}`,
  ];

  if (recentCycles.length > 0) {
    lines.push("Recent batch cycles:");
    for (const c of recentCycles) {
      lines.push(
        `  Cycle ${c.cycleNumber}: ${c.newDatapoints} new pts, ` +
        `${c.evaluationSuccesses}/${c.candidatesSubmitted} success, ` +
        `R² ${c.preRetrainMetrics?.r2.toFixed(4) ?? "?"} -> ${c.postRetrainMetrics?.r2.toFixed(4) ?? "?"}, ` +
        `unc ${c.avgUncertaintyBefore.toFixed(3)} -> ${c.avgUncertaintyAfter.toFixed(3)}`
      );
    }
  }

  return lines.join("\n");
}

export function getDatapointsByFormula(formula: string): GroundTruthDatapoint[] {
  return datapoints.filter(d => d.formula === formula);
}

export function getDatapointsByCycle(cycle: number): GroundTruthDatapoint[] {
  return datapoints.filter(d => d.cycle === cycle);
}

export function getDatasetForTraining(): {
  formulas: string[];
  targets: number[];
  features: Record<string, (number | null)[]>;
  isSuperconductor: boolean[];
} {
  const formulas: string[] = [];
  const targets: number[] = [];
  const isSuperconductor: boolean[] = [];
  const features: Record<string, (number | null)[]> = {
    formation_energy: [],
    DOS_EF: [],
    lambda: [],
    omega_log: [],
    band_gap: [],
    pressure: [],
    mu_star: [],
  };

  for (const dp of datapoints) {
    if (dp.Tc < 0) continue;
    formulas.push(dp.formula);
    targets.push(Math.max(0, dp.Tc));
    isSuperconductor.push(dp.Tc > 0);
    features.formation_energy.push(dp.formation_energy);
    features.DOS_EF.push(dp.DOS_EF);
    features.lambda.push(dp.lambda);
    features.omega_log.push(dp.omega_log);
    features.band_gap.push(dp.band_gap);
    features.pressure.push(dp.pressure);
    features.mu_star.push(dp.mu_star);
  }

  return { formulas, targets, features, isSuperconductor };
}
