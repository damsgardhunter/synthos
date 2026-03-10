import { getCalibrationData, getXGBEnsembleStats, getModelVersionHistory, getEvaluatedDatasetStats, getSurrogateStats } from "./gradient-boost";
import { getGNNVersionHistory, getGNNModelVersion, ENSEMBLE_SIZE, getDFTTrainingDatasetStats } from "./graph-neural-net";
import { getLambdaRegressorStats } from "./lambda-regressor";
import { getPhononSurrogateStats } from "../physics/phonon-surrogate";
import { getTBSurrogateStats } from "../physics/tb-ml-surrogate";
import { getStructurePredictorStats } from "../crystal/structure-predictor-ml";
import { getPressureStructureStats } from "../crystal/pressure-structure-model";

type HealthStatus = "green" | "yellow" | "red";

interface PredictionOutcome {
  model: string;
  formula: string;
  predicted: number;
  actual: number;
  timestamp: number;
  family?: string;
}

interface FamilyBias {
  family: string;
  count: number;
  meanError: number;
  meanAbsError: number;
  bias: "over" | "under" | "neutral";
}

interface CalibrationBin {
  binLabel: string;
  lower: number;
  upper: number;
  count: number;
  withinRange: number;
  calibrationRate: number;
}

interface ModelHealth {
  model: string;
  status: HealthStatus;
  reasons: string[];
}

interface XGBoostDiagnostics {
  r2: number;
  mae: number;
  rmse: number;
  nSamples: number;
  nTrees: number;
  featureCount: number;
  ensembleSize: number;
  ensembleTreeCounts: number[];
  predictionVariance: number;
  residualPercentiles: { p5: number; p10: number; p25: number; p50: number; p75: number; p90: number; p95: number };
  absResidualPercentiles: { p50: number; p75: number; p90: number; p95: number };
  datasetSize: number;
  modelVersion: number;
  trainedAt: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  meanResidualSign: number;
}

interface GNNDiagnostics {
  ensembleSize: number;
  modelVersion: number;
  datasetSize: number;
  trainedAt: number;
  latestR2: number;
  latestMAE: number;
  latestRMSE: number;
  predictionCount: number;
  modelStalenessMs: number;
}

interface LambdaDiagnostics {
  r2: number;
  mae: number;
  rmse: number;
  datasetSize: number;
  ensembleSize: number;
  retrainCount: number;
  totalPredictions: number;
  tierBreakdown: { "verified-dfpt": number; "ml-regression": number; "physics-engine": number };
  recentErrors: { formula: string; predicted: number; actual: number; absError: number }[];
}

interface PhononSurrogateDiagnostics {
  datasetSize: number;
  omegaLogMAE: number;
  debyeTempMAE: number;
  maxFreqMAE: number;
  stabilityAccuracy: number;
  totalPredictions: number;
  hitRate: number;
}

interface TBSurrogateDiagnostics {
  datasetSize: number;
  modelCount: number;
  predictions: number;
  trainings: number;
  avgPredictionTimeMs: number;
  treeCounts: Record<string, number>;
}

interface StructurePredictorDiagnostics {
  datasetSize: number;
  spacegroupAccuracy: number;
  crystalSystemAccuracy: number;
  prototypeAccuracy: number;
  latticeMAE: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  trainCount: number;
}

interface PressureStructureDiagnostics {
  datasetSize: number;
  predictionCount: number;
  transitionRecords: number;
  modelTrained: boolean;
}

export interface ComprehensiveModelDiagnostics {
  timestamp: number;
  xgboost: XGBoostDiagnostics;
  gnn: GNNDiagnostics;
  lambda: LambdaDiagnostics;
  phononSurrogate: PhononSurrogateDiagnostics;
  tbSurrogate: TBSurrogateDiagnostics;
  structurePredictor: StructurePredictorDiagnostics;
  pressureStructure: PressureStructureDiagnostics;
  familyBias: FamilyBias[];
  calibrationBins: CalibrationBin[];
  predictionOutcomeCount: number;
}

const MAX_OUTCOMES = 500;
let predictionOutcomes: PredictionOutcome[] = [];

function classifyFamily(formula: string): string {
  if (/H\d/.test(formula) || formula.includes("H")) {
    const hMatch = formula.match(/H(\d+)/);
    if (hMatch && parseInt(hMatch[1]) >= 4) return "hydride";
  }
  if (formula.includes("Cu") && formula.includes("O") && /Ba|Sr|La|Y|Ca|Bi|Tl|Hg/.test(formula)) return "cuprate";
  if (/Fe|Co|Ni/.test(formula) && /As|Se|Te|P|S/.test(formula)) return "pnictide";
  if (formula.includes("B") && /Mg|Nb|Ti|Zr|Hf|Ta|W|Mo|V|Cr/.test(formula)) return "boride";
  return "conventional";
}

export function recordPredictionOutcome(model: string, formula: string, predicted: number, actual: number): void {
  const family = classifyFamily(formula);
  predictionOutcomes.push({
    model,
    formula,
    predicted,
    actual,
    timestamp: Date.now(),
    family,
  });
  if (predictionOutcomes.length > MAX_OUTCOMES) {
    predictionOutcomes = predictionOutcomes.slice(-MAX_OUTCOMES);
  }
}

function computeFamilyBias(modelFilter?: string): FamilyBias[] {
  const families = ["hydride", "cuprate", "pnictide", "boride", "conventional"];
  const result: FamilyBias[] = [];

  for (const family of families) {
    const outcomes = predictionOutcomes.filter(
      o => o.family === family && (modelFilter ? o.model === modelFilter : true)
    );
    if (outcomes.length === 0) {
      result.push({ family, count: 0, meanError: 0, meanAbsError: 0, bias: "neutral" });
      continue;
    }
    const errors = outcomes.map(o => o.predicted - o.actual);
    const meanError = errors.reduce((s, e) => s + e, 0) / errors.length;
    const meanAbsError = errors.reduce((s, e) => s + Math.abs(e), 0) / errors.length;
    const bias: "over" | "under" | "neutral" = meanError > 2 ? "over" : meanError < -2 ? "under" : "neutral";
    result.push({
      family,
      count: outcomes.length,
      meanError: Math.round(meanError * 100) / 100,
      meanAbsError: Math.round(meanAbsError * 100) / 100,
      bias,
    });
  }

  return result;
}

function computeCalibrationBins(): CalibrationBin[] {
  const bins: CalibrationBin[] = [];
  const binEdges = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

  for (let i = 0; i < binEdges.length - 1; i++) {
    const lower = binEdges[i];
    const upper = binEdges[i + 1];
    const binLabel = `${lower.toFixed(1)}-${upper.toFixed(1)}`;

    const inBin = predictionOutcomes.filter(o => {
      const absDiff = Math.abs(o.predicted - o.actual);
      const maxVal = Math.max(Math.abs(o.predicted), Math.abs(o.actual), 1);
      const normalizedError = absDiff / maxVal;
      const confidence = Math.max(0, 1 - normalizedError);
      return confidence >= lower && confidence < upper;
    });

    const withinRange = inBin.filter(o => {
      const absDiff = Math.abs(o.predicted - o.actual);
      const tolerance = Math.max(5, Math.abs(o.actual) * 0.3);
      return absDiff <= tolerance;
    }).length;

    bins.push({
      binLabel,
      lower,
      upper,
      count: inBin.length,
      withinRange,
      calibrationRate: inBin.length > 0 ? Math.round((withinRange / inBin.length) * 1000) / 1000 : 0,
    });
  }

  return bins;
}

export function getComprehensiveModelDiagnostics(): ComprehensiveModelDiagnostics {
  const calibration = getCalibrationData();
  const ensembleStats = getXGBEnsembleStats();
  const versionHistory = getModelVersionHistory();
  const evalStats = getEvaluatedDatasetStats();
  const surrogateStats = getSurrogateStats();

  const latestVersion = versionHistory.latestMetrics;

  let falsePositiveRate = 0;
  let falseNegativeRate = 0;
  let meanResidualSign = 0;

  if (calibration.predictedVsActual && calibration.predictedVsActual.length > 0) {
    const fpCount = calibration.predictedVsActual.filter(p => p.predicted > 10 && p.actual < 5).length;
    const fnCount = calibration.predictedVsActual.filter(p => p.predicted < 5 && p.actual > 10).length;
    const total = calibration.predictedVsActual.length;
    falsePositiveRate = total > 0 ? Math.round((fpCount / total) * 10000) / 10000 : 0;
    falseNegativeRate = total > 0 ? Math.round((fnCount / total) * 10000) / 10000 : 0;

    const signs = calibration.predictedVsActual.map(p => p.residual > 0 ? 1 : p.residual < 0 ? -1 : 0);
    meanResidualSign = signs.length > 0 ? Math.round((signs.reduce((s: number, v: number) => s + v, 0 as number) / signs.length) * 1000) / 1000 : 0;
  }

  const xgboost: XGBoostDiagnostics = {
    r2: calibration.r2,
    mae: calibration.mae,
    rmse: calibration.rmse,
    nSamples: calibration.nSamples,
    nTrees: calibration.nTrees,
    featureCount: surrogateStats.totalFeatures,
    ensembleSize: ensembleStats.ensembleSize,
    ensembleTreeCounts: ensembleStats.modelTreeCounts,
    predictionVariance: latestVersion?.predictionVariance ?? 0,
    residualPercentiles: calibration.percentiles,
    absResidualPercentiles: calibration.absResidualPercentiles,
    datasetSize: evalStats.totalEvaluated + calibration.nSamples,
    modelVersion: versionHistory.currentVersion,
    trainedAt: calibration.computedAt,
    falsePositiveRate,
    falseNegativeRate,
    meanResidualSign,
  };

  const gnnHistory = getGNNVersionHistory();
  const latestGNN = gnnHistory.length > 0 ? gnnHistory[gnnHistory.length - 1] : null;
  const dftDatasetStats = getDFTTrainingDatasetStats();

  const gnn: GNNDiagnostics = {
    ensembleSize: ENSEMBLE_SIZE,
    modelVersion: getGNNModelVersion(),
    datasetSize: dftDatasetStats.totalSize,
    trainedAt: latestGNN?.trainedAt ?? 0,
    latestR2: latestGNN?.r2 ?? 0,
    latestMAE: latestGNN?.mae ?? 0,
    latestRMSE: latestGNN?.rmse ?? 0,
    predictionCount: dftDatasetStats.totalSize,
    modelStalenessMs: latestGNN ? Date.now() - latestGNN.trainedAt : 0,
  };

  const lambdaStats = getLambdaRegressorStats();
  const lambda: LambdaDiagnostics = {
    r2: lambdaStats.metrics.r2,
    mae: lambdaStats.metrics.mae,
    rmse: lambdaStats.metrics.rmse,
    datasetSize: lambdaStats.datasetSize,
    ensembleSize: lambdaStats.ensembleSize,
    retrainCount: lambdaStats.retrainCount,
    totalPredictions: lambdaStats.totalPredictions,
    tierBreakdown: lambdaStats.tierBreakdown,
    recentErrors: lambdaStats.recentErrors,
  };

  const phononStats = getPhononSurrogateStats();
  const phononSurrogate: PhononSurrogateDiagnostics = {
    datasetSize: phononStats.datasetSize,
    omegaLogMAE: phononStats.metrics.omegaLogMAE,
    debyeTempMAE: phononStats.metrics.debyeTempMAE,
    maxFreqMAE: phononStats.metrics.maxFreqMAE,
    stabilityAccuracy: phononStats.metrics.stabilityAccuracy,
    totalPredictions: phononStats.totalPredictions,
    hitRate: phononStats.totalPredictions > 0
      ? Math.round((phononStats.tierBreakdown.hits / Math.max(1, phononStats.tierBreakdown.hits + phononStats.tierBreakdown.misses)) * 1000) / 1000
      : 0,
  };

  const tbStats = getTBSurrogateStats();
  const tbSurrogate: TBSurrogateDiagnostics = {
    datasetSize: tbStats.datasetSize,
    modelCount: tbStats.modelCount,
    predictions: tbStats.predictions,
    trainings: tbStats.trainings,
    avgPredictionTimeMs: tbStats.avgPredictionTimeMs,
    treeCounts: tbStats.treeCounts,
  };

  const structStats = getStructurePredictorStats();
  const structurePredictor: StructurePredictorDiagnostics = {
    datasetSize: structStats.datasetSize,
    spacegroupAccuracy: structStats.metrics.spacegroupAccuracy,
    crystalSystemAccuracy: structStats.metrics.crystalSystemAccuracy,
    prototypeAccuracy: structStats.metrics.prototypeAccuracy,
    latticeMAE: structStats.metrics.latticeMAE,
    trainCount: structStats.trainCount,
  };

  const pressureStats = getPressureStructureStats();
  const pressureStructure: PressureStructureDiagnostics = {
    datasetSize: pressureStats.datasetSize,
    predictionCount: pressureStats.predictionCount,
    transitionRecords: pressureStats.transitionRecords,
    modelTrained: pressureStats.modelTrained,
  };

  return {
    timestamp: Date.now(),
    xgboost,
    gnn,
    lambda,
    phononSurrogate,
    tbSurrogate,
    structurePredictor,
    pressureStructure,
    familyBias: computeFamilyBias(),
    calibrationBins: computeCalibrationBins(),
    predictionOutcomeCount: predictionOutcomes.length,
  };
}

export function getModelDiagnosticsForLLM(): string {
  const d = getComprehensiveModelDiagnostics();
  const lines: string[] = [];

  lines.push("=== MODEL DIAGNOSTICS REPORT ===");
  lines.push("");

  lines.push("## XGBoost Tc Predictor");
  lines.push(`  R²=${d.xgboost.r2} | MAE=${d.xgboost.mae}K | RMSE=${d.xgboost.rmse}K`);
  lines.push(`  Trees=${d.xgboost.nTrees} | Features=${d.xgboost.featureCount} | Dataset=${d.xgboost.nSamples}`);
  lines.push(`  Ensemble: ${d.xgboost.ensembleSize} models, tree counts=[${d.xgboost.ensembleTreeCounts.join(",")}]`);
  lines.push(`  Prediction variance=${d.xgboost.predictionVariance}K`);
  lines.push(`  False positive rate (pred>10K,actual<5K)=${d.xgboost.falsePositiveRate}`);
  lines.push(`  False negative rate (pred<5K,actual>10K)=${d.xgboost.falseNegativeRate}`);
  lines.push(`  Prediction bias (mean residual sign)=${d.xgboost.meanResidualSign}`);
  lines.push(`  Residual p90=${d.xgboost.absResidualPercentiles.p90}K, p95=${d.xgboost.absResidualPercentiles.p95}K`);
  if (d.xgboost.r2 < 0.5) lines.push("  ** WARNING: Low R² — model may be underfitting **");
  if (d.xgboost.rmse > 30) lines.push("  ** WARNING: High RMSE — large prediction errors **");
  lines.push("");

  lines.push("## GNN Ensemble");
  lines.push(`  Version=${d.gnn.modelVersion} | Ensemble=${d.gnn.ensembleSize} models`);
  lines.push(`  R²=${d.gnn.latestR2} | MAE=${d.gnn.latestMAE}K | RMSE=${d.gnn.latestRMSE}K`);
  lines.push(`  Dataset=${d.gnn.datasetSize} | Staleness=${Math.round(d.gnn.modelStalenessMs / 60000)}min`);
  if (d.gnn.modelStalenessMs > 6 * 60 * 60 * 1000) lines.push("  ** WARNING: Model is stale (>6h) **");
  lines.push("");

  lines.push("## Lambda Regressor");
  lines.push(`  R²=${d.lambda.r2} | MAE=${d.lambda.mae} | RMSE=${d.lambda.rmse}`);
  lines.push(`  Dataset=${d.lambda.datasetSize} | Retrains=${d.lambda.retrainCount}`);
  lines.push(`  Tier breakdown: DFPT=${d.lambda.tierBreakdown["verified-dfpt"]}, ML=${d.lambda.tierBreakdown["ml-regression"]}, Physics=${d.lambda.tierBreakdown["physics-engine"]}`);
  if (d.lambda.recentErrors.length > 0) {
    const avgErr = d.lambda.recentErrors.reduce((s, e) => s + e.absError, 0) / d.lambda.recentErrors.length;
    lines.push(`  Recent avg error=${avgErr.toFixed(3)}`);
  }
  lines.push("");

  lines.push("## Phonon Surrogate");
  lines.push(`  Dataset=${d.phononSurrogate.datasetSize} | Predictions=${d.phononSurrogate.totalPredictions}`);
  lines.push(`  omegaLog MAE=${d.phononSurrogate.omegaLogMAE} | debyeTemp MAE=${d.phononSurrogate.debyeTempMAE}`);
  lines.push(`  maxFreq MAE=${d.phononSurrogate.maxFreqMAE} | Stability accuracy=${d.phononSurrogate.stabilityAccuracy}`);
  lines.push(`  Hit rate=${d.phononSurrogate.hitRate}`);
  lines.push("");

  lines.push("## TB Surrogate");
  lines.push(`  Dataset=${d.tbSurrogate.datasetSize} | Models=${d.tbSurrogate.modelCount}`);
  lines.push(`  Predictions=${d.tbSurrogate.predictions} | Trainings=${d.tbSurrogate.trainings}`);
  lines.push(`  Avg prediction time=${d.tbSurrogate.avgPredictionTimeMs}ms`);
  lines.push("");

  lines.push("## Structure Predictor ML");
  lines.push(`  Dataset=${d.structurePredictor.datasetSize} | Trains=${d.structurePredictor.trainCount}`);
  lines.push(`  Spacegroup accuracy=${d.structurePredictor.spacegroupAccuracy}`);
  lines.push(`  Crystal system accuracy=${d.structurePredictor.crystalSystemAccuracy}`);
  lines.push(`  Lattice param MAE: a=${d.structurePredictor.latticeMAE.a}, b=${d.structurePredictor.latticeMAE.b}, c=${d.structurePredictor.latticeMAE.c}`);
  lines.push("");

  lines.push("## Pressure Structure Model");
  lines.push(`  Dataset=${d.pressureStructure.datasetSize} | Predictions=${d.pressureStructure.predictionCount}`);
  lines.push(`  Trained=${d.pressureStructure.modelTrained} | Transitions=${d.pressureStructure.transitionRecords}`);
  lines.push("");

  if (d.familyBias.some(fb => fb.count > 0)) {
    lines.push("## Per-Family Bias Analysis");
    for (const fb of d.familyBias) {
      if (fb.count === 0) continue;
      lines.push(`  ${fb.family}: n=${fb.count}, mean_error=${fb.meanError}K, |error|=${fb.meanAbsError}K, bias=${fb.bias}`);
    }
    lines.push("");
  }

  if (d.calibrationBins.some(b => b.count > 0)) {
    lines.push("## Uncertainty Calibration");
    for (const bin of d.calibrationBins) {
      if (bin.count === 0) continue;
      lines.push(`  Confidence ${bin.binLabel}: n=${bin.count}, within_range=${bin.withinRange}, rate=${bin.calibrationRate}`);
    }
    lines.push("");
  }

  lines.push(`Prediction outcomes tracked: ${d.predictionOutcomeCount}`);
  lines.push(`Report generated: ${new Date(d.timestamp).toISOString()}`);

  return lines.join("\n");
}

export function getModelHealthSummary(): ModelHealth[] {
  const d = getComprehensiveModelDiagnostics();
  const health: ModelHealth[] = [];

  {
    const reasons: string[] = [];
    let status: HealthStatus = "green";
    if (d.xgboost.r2 < 0.3) { status = "red"; reasons.push(`Very low R²=${d.xgboost.r2}`); }
    else if (d.xgboost.r2 < 0.6) { status = "yellow"; reasons.push(`Low R²=${d.xgboost.r2}`); }
    if (d.xgboost.rmse > 40) { status = "red"; reasons.push(`High RMSE=${d.xgboost.rmse}K`); }
    else if (d.xgboost.rmse > 25) { if (status === "green") status = "yellow"; reasons.push(`Elevated RMSE=${d.xgboost.rmse}K`); }
    if (d.xgboost.falsePositiveRate > 0.15) { if (status === "green") status = "yellow"; reasons.push(`High false positive rate=${d.xgboost.falsePositiveRate}`); }
    if (d.xgboost.nTrees === 0) { status = "red"; reasons.push("No trees trained"); }
    if (reasons.length === 0) reasons.push("All metrics within acceptable range");
    health.push({ model: "xgboost", status, reasons });
  }

  {
    const reasons: string[] = [];
    let status: HealthStatus = "green";
    if (d.gnn.latestR2 < 0.2) { status = "red"; reasons.push(`Very low R²=${d.gnn.latestR2}`); }
    else if (d.gnn.latestR2 < 0.5) { status = "yellow"; reasons.push(`Low R²=${d.gnn.latestR2}`); }
    if (d.gnn.modelStalenessMs > 12 * 60 * 60 * 1000) { status = "red"; reasons.push("Model stale (>12h)"); }
    else if (d.gnn.modelStalenessMs > 6 * 60 * 60 * 1000) { if (status === "green") status = "yellow"; reasons.push("Model stale (>6h)"); }
    if (d.gnn.modelVersion === 0) { status = "red"; reasons.push("No model trained"); }
    if (reasons.length === 0) reasons.push("All metrics within acceptable range");
    health.push({ model: "gnn", status, reasons });
  }

  {
    const reasons: string[] = [];
    let status: HealthStatus = "green";
    if (!d.lambda.datasetSize || d.lambda.datasetSize < 5) { status = "red"; reasons.push("Insufficient training data"); }
    if (d.lambda.r2 < 0.2) { status = "red"; reasons.push(`Very low R²=${d.lambda.r2}`); }
    else if (d.lambda.r2 < 0.5) { if (status === "green") status = "yellow"; reasons.push(`Low R²=${d.lambda.r2}`); }
    if (d.lambda.rmse > 0.5) { if (status === "green") status = "yellow"; reasons.push(`High RMSE=${d.lambda.rmse}`); }
    if (reasons.length === 0) reasons.push("All metrics within acceptable range");
    health.push({ model: "lambda-regressor", status, reasons });
  }

  {
    const reasons: string[] = [];
    let status: HealthStatus = "green";
    if (d.phononSurrogate.datasetSize === 0) { status = "red"; reasons.push("No training data"); }
    else if (d.phononSurrogate.datasetSize < 10) { status = "yellow"; reasons.push("Small dataset"); }
    if (d.phononSurrogate.stabilityAccuracy < 0.5 && d.phononSurrogate.datasetSize > 0) {
      if (status === "green") status = "yellow";
      reasons.push(`Low stability accuracy=${d.phononSurrogate.stabilityAccuracy}`);
    }
    if (reasons.length === 0) reasons.push("All metrics within acceptable range");
    health.push({ model: "phonon-surrogate", status, reasons });
  }

  {
    const reasons: string[] = [];
    let status: HealthStatus = "green";
    if (!d.tbSurrogate.datasetSize) { status = "yellow"; reasons.push("No training data yet"); }
    if (d.tbSurrogate.trainings === 0) { if (status === "green") status = "yellow"; reasons.push("Never trained"); }
    if (reasons.length === 0) reasons.push("All metrics within acceptable range");
    health.push({ model: "tb-surrogate", status, reasons });
  }

  {
    const reasons: string[] = [];
    let status: HealthStatus = "green";
    if (d.structurePredictor.datasetSize === 0) { status = "red"; reasons.push("No training data"); }
    if (d.structurePredictor.crystalSystemAccuracy < 0.3 && d.structurePredictor.datasetSize > 0) {
      if (status === "green") status = "yellow";
      reasons.push(`Low crystal system accuracy=${d.structurePredictor.crystalSystemAccuracy}`);
    }
    if (reasons.length === 0) reasons.push("All metrics within acceptable range");
    health.push({ model: "structure-predictor", status, reasons });
  }

  {
    const reasons: string[] = [];
    let status: HealthStatus = "green";
    if (!d.pressureStructure.modelTrained) { status = "yellow"; reasons.push("Model not yet trained"); }
    if (d.pressureStructure.datasetSize === 0) { if (status === "green") status = "yellow"; reasons.push("No training data"); }
    if (reasons.length === 0) reasons.push("All metrics within acceptable range");
    health.push({ model: "pressure-structure", status, reasons });
  }

  return health;
}

export function getPerFamilyBias(model?: string): FamilyBias[] {
  return computeFamilyBias(model);
}
