import { getGroundTruthSummary, getRecentBatchCycles, getCurrentCycleNumber, type BatchCycle } from "./ground-truth-store";
import { computeMetrics, computeMetricsByFamily, computeRecentMetrics, getImprovementTrend } from "./prediction-reality-ledger";
import { getPhysicsStoreStats } from "./physics-results-store";
import { getUnifiedDatasetStats } from "./unified-training-dataset";

function lazyGradientBoost() {
  return require("./gradient-boost") as { getXGBEnsembleStats: () => any };
}
function lazyGNN() {
  return require("./graph-neural-net") as { getGNNModelVersion: () => number; getDFTTrainingDatasetStats: () => any };
}
function lazyLambda() {
  return require("./lambda-regressor") as { getLambdaRegressorStats: () => any };
}

export interface CycleDiagnosticReport {
  cycleNumber: number;
  timestamp: number;

  datasetSize: number;
  newSamplesThisCycle: number;
  withPhysicsResults: number;
  withDerivedFeatures: number;
  sourceBreakdown: Record<string, number>;

  tcModel: {
    rmse: number;
    mae: number;
    r2: number;
    bias: number;
    sampleCount: number;
    calibrationStatus: string;
    familyBias: Record<string, { rmse: number; bias: number; count: number }>;
  };

  lambdaModel: {
    rmse: number;
    mae: number;
    r2: number;
    bias: string;
    datasetSize: number;
    retrainCount: number;
  };

  gnnStatus: {
    version: number;
    ensembleSize: number;
    datasetSize: number;
  };

  xgbStatus: {
    ensembleSize: number;
    datasetSize: number;
  };

  physicsStore: {
    totalEntries: number;
    tierBreakdown: Record<string, number>;
    avgLambda: number;
    phononStableFraction: number;
  };

  trend: {
    rmseDirection: string;
    last3CycleRMSE: number[];
    improvingModels: string[];
    degradingModels: string[];
  };

  alerts: string[];
}

const reportHistory: CycleDiagnosticReport[] = [];
const MAX_REPORTS = 200;

function computeCalibrationStatus(r2: number, rmse: number): string {
  if (r2 > 0.8 && rmse < 30) return "good";
  if (r2 > 0.6 && rmse < 60) return "acceptable";
  if (r2 > 0.3) return "needs_improvement";
  return "poor";
}

export function generateCycleDiagnostics(): CycleDiagnosticReport {
  const cycleNumber = getCurrentCycleNumber();
  const datasetStats = getUnifiedDatasetStats();
  const recentCycles = getRecentBatchCycles(1);
  const newSamples = recentCycles.length > 0 ? recentCycles[0].newDatapoints : 0;

  const tcMetrics = computeMetrics();
  const familyMetrics = computeMetricsByFamily();
  const trend = getImprovementTrend();
  const physicsStats = getPhysicsStoreStats();

  let lambdaStats = { r2: 0, mae: 0, rmse: 0, datasetSize: 0, retrainCount: 0, bias: "unknown" };
  try {
    const ls = lazyLambda().getLambdaRegressorStats();
    const lBias = ls.recentErrors && ls.recentErrors.length > 0
      ? ls.recentErrors.reduce((s: number, e: any) => s + ((e.predicted ?? 0) - (e.actual ?? 0)), 0) / ls.recentErrors.length
      : 0;
    lambdaStats = {
      r2: ls.r2 ?? 0,
      mae: ls.mae ?? 0,
      rmse: ls.rmse ?? 0,
      datasetSize: ls.datasetSize ?? 0,
      retrainCount: ls.retrainCount ?? 0,
      bias: lBias > 0.05 ? "overpredicts" : lBias < -0.05 ? "underpredicts" : "balanced",
    };
  } catch {}

  let gnnStatus = { version: 0, ensembleSize: 0, datasetSize: 0 };
  try {
    const gv = lazyGNN().getGNNModelVersion();
    const gds = lazyGNN().getDFTTrainingDatasetStats();
    gnnStatus = { version: gv, ensembleSize: 4, datasetSize: gds.totalSamples ?? 0 };
  } catch {}

  let xgbStatus = { ensembleSize: 0, datasetSize: 0 };
  try {
    const xs = lazyGradientBoost().getXGBEnsembleStats();
    xgbStatus = { ensembleSize: xs.ensembleSize ?? 5, datasetSize: xs.datasetSize ?? 0 };
  } catch {}

  const familyBias: Record<string, { rmse: number; bias: number; count: number }> = {};
  for (const [family, fm] of Object.entries(familyMetrics)) {
    familyBias[family] = {
      rmse: (fm as any).rmse ?? 0,
      bias: (fm as any).bias ?? 0,
      count: (fm as any).count ?? 0,
    };
  }

  const alerts: string[] = [];

  if (tcMetrics.rmse > 100) alerts.push("Tc model RMSE exceeds 100K — model may be stagnating");
  if (tcMetrics.r2 < 0.3) alerts.push("Tc model R2 below 0.3 — insufficient predictive power");
  if (lambdaStats.rmse > 0.5) alerts.push("Lambda model RMSE exceeds 0.5 — coupling predictions unreliable");
  if (datasetStats.withPhysicsResults < datasetStats.totalRecords * 0.1) {
    alerts.push("Less than 10% of dataset has physics results — feature coverage is low");
  }

  if (trend.status === "degrading") alerts.push("Model performance degrading over recent cycles");
  if (newSamples === 0 && cycleNumber > 0) alerts.push("No new samples added this cycle — exploration may be stuck");

  for (const [family, fb] of Object.entries(familyBias)) {
    if (Math.abs(fb.bias) < 5) continue;
    if (fb.bias > 30) alerts.push(`${family}: overpredicts Tc by ${fb.bias.toFixed(1)}K on average`);
    if (fb.bias < -30) alerts.push(`${family}: underpredicts Tc by ${Math.abs(fb.bias).toFixed(1)}K on average`);
  }

  const trendRMSEs = (trend.recentCycles ?? []).map((c: any) => c.rmse).filter((r: number) => r > 0);
  const improvingModels: string[] = [];
  const degradingModels: string[] = [];

  if (trendRMSEs.length >= 2) {
    const last = trendRMSEs[trendRMSEs.length - 1];
    const prev = trendRMSEs[trendRMSEs.length - 2];
    if (last < prev) improvingModels.push("Tc-ensemble");
    if (last > prev * 1.1) degradingModels.push("Tc-ensemble");
  }

  const report: CycleDiagnosticReport = {
    cycleNumber,
    timestamp: Date.now(),
    datasetSize: datasetStats.totalRecords,
    newSamplesThisCycle: newSamples,
    withPhysicsResults: datasetStats.withPhysicsResults,
    withDerivedFeatures: datasetStats.withDerivedFeatures,
    sourceBreakdown: datasetStats.sourceBreakdown,
    tcModel: {
      rmse: tcMetrics.rmse,
      mae: tcMetrics.mae,
      r2: tcMetrics.r2,
      bias: tcMetrics.bias,
      sampleCount: tcMetrics.sampleCount,
      calibrationStatus: computeCalibrationStatus(tcMetrics.r2, tcMetrics.rmse),
      familyBias,
    },
    lambdaModel: {
      rmse: lambdaStats.rmse,
      mae: lambdaStats.mae,
      r2: lambdaStats.r2,
      bias: lambdaStats.bias,
      datasetSize: lambdaStats.datasetSize,
      retrainCount: lambdaStats.retrainCount,
    },
    gnnStatus,
    xgbStatus,
    physicsStore: {
      totalEntries: physicsStats.totalEntries,
      tierBreakdown: physicsStats.tierBreakdown,
      avgLambda: physicsStats.avgLambda,
      phononStableFraction: physicsStats.phononStableFraction,
    },
    trend: {
      rmseDirection: trend.status ?? "unknown",
      last3CycleRMSE: trendRMSEs.slice(-3),
      improvingModels,
      degradingModels,
    },
    alerts,
  };

  reportHistory.push(report);
  if (reportHistory.length > MAX_REPORTS) {
    reportHistory.splice(0, reportHistory.length - MAX_REPORTS);
  }

  return report;
}

export function getReportHistory(): CycleDiagnosticReport[] {
  return [...reportHistory];
}

export function getLatestReport(): CycleDiagnosticReport | null {
  return reportHistory.length > 0 ? reportHistory[reportHistory.length - 1] : null;
}

export function formatDiagnosticReportText(report: CycleDiagnosticReport): string {
  const lines: string[] = [
    `=== Cycle ${report.cycleNumber} Diagnostics ===`,
    ``,
    `dataset size: ${report.datasetSize.toLocaleString()}`,
    `new samples: ${report.newSamplesThisCycle}`,
    `physics coverage: ${report.withPhysicsResults}/${report.datasetSize}`,
    `derived features: ${report.withDerivedFeatures}/${report.datasetSize}`,
    ``,
    `Tc model`,
    `  RMSE: ${report.tcModel.rmse.toFixed(1)} K`,
    `  MAE: ${report.tcModel.mae.toFixed(1)} K`,
    `  R2: ${report.tcModel.r2.toFixed(4)}`,
    `  bias: ${report.tcModel.bias > 0 ? "+" : ""}${report.tcModel.bias.toFixed(1)} K`,
    `  calibration: ${report.tcModel.calibrationStatus}`,
    `  samples: ${report.tcModel.sampleCount}`,
  ];

  const families = Object.entries(report.tcModel.familyBias);
  if (families.length > 0) {
    lines.push(`  family breakdown:`);
    for (const [fam, fb] of families) {
      const biasLabel = fb.bias > 5 ? "overpredicts" : fb.bias < -5 ? "underpredicts" : "balanced";
      lines.push(`    ${fam}: RMSE=${fb.rmse.toFixed(1)}K, bias=${biasLabel} (${fb.bias > 0 ? "+" : ""}${fb.bias.toFixed(1)}K), n=${fb.count}`);
    }
  }

  lines.push(``);
  lines.push(`lambda model`);
  lines.push(`  RMSE: ${report.lambdaModel.rmse.toFixed(3)}`);
  lines.push(`  MAE: ${report.lambdaModel.mae.toFixed(3)}`);
  lines.push(`  R2: ${report.lambdaModel.r2.toFixed(4)}`);
  lines.push(`  bias: ${report.lambdaModel.bias}`);
  lines.push(`  dataset: ${report.lambdaModel.datasetSize}`);

  lines.push(``);
  lines.push(`GNN: v${report.gnnStatus.version}, ${report.gnnStatus.ensembleSize} models, ${report.gnnStatus.datasetSize} samples`);
  lines.push(`XGBoost: ${report.xgbStatus.ensembleSize} trees, ${report.xgbStatus.datasetSize} samples`);

  lines.push(``);
  lines.push(`physics store: ${report.physicsStore.totalEntries} entries`);
  lines.push(`  tiers: ${Object.entries(report.physicsStore.tierBreakdown).map(([k, v]) => `${k}=${v}`).join(", ")}`);
  lines.push(`  avg lambda: ${report.physicsStore.avgLambda.toFixed(4)}, phonon stable: ${(report.physicsStore.phononStableFraction * 100).toFixed(1)}%`);

  lines.push(``);
  lines.push(`trend: ${report.trend.rmseDirection}`);
  if (report.trend.last3CycleRMSE.length > 0) {
    lines.push(`  recent RMSE: [${report.trend.last3CycleRMSE.map(r => r.toFixed(1)).join(", ")}]`);
  }
  if (report.trend.improvingModels.length > 0) {
    lines.push(`  improving: ${report.trend.improvingModels.join(", ")}`);
  }
  if (report.trend.degradingModels.length > 0) {
    lines.push(`  degrading: ${report.trend.degradingModels.join(", ")}`);
  }

  if (report.alerts.length > 0) {
    lines.push(``);
    lines.push(`ALERTS:`);
    for (const alert of report.alerts) {
      lines.push(`  * ${alert}`);
    }
  }

  return lines.join("\n");
}

export function getDiagnosticsForLLM(): string {
  const report = generateCycleDiagnostics();
  return formatDiagnosticReportText(report);
}
