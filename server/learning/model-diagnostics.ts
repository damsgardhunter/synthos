import { getCalibrationData, getXGBEnsembleStats, getModelVersionHistory, getEvaluatedDatasetStats, getSurrogateStats, getGlobalFeatureImportance } from "./gradient-boost";
import { getGNNVersionHistory, getGNNModelVersion, ENSEMBLE_SIZE, getDFTTrainingDatasetStats } from "./graph-neural-net";
import { getLambdaRegressorStats } from "./lambda-regressor";
import { getPhononSurrogateStats } from "../physics/phonon-surrogate";
import { getTBSurrogateStats } from "../physics/tb-ml-surrogate";
import { getStructurePredictorStats } from "../crystal/structure-predictor-ml";
import { getPressureStructureStats } from "../crystal/pressure-structure-model";
import { getCalibrationStats } from "./surrogate-fitness";
import { getFailurePatterns, getFailureDBStats, getFailureEntries, type StructureFailureEntry } from "../crystal/structure-failure-db";
import { classifyFamily } from "./utils";
import { COMPOSITION_FEATURE_NAMES } from "./composition-features";

type HealthStatus = "green" | "yellow" | "red";

const COMPOSITION_FEATURE_SET = new Set(COMPOSITION_FEATURE_NAMES);
const INFERENCE_BUFFER_SIZE = 200;
const inferenceTimings: Map<string, { buffer: number[]; head: number; count: number }> = new Map();

export function recordInferenceTime(model: string, ms: number): void {
  let tracker = inferenceTimings.get(model);
  if (!tracker) {
    tracker = { buffer: new Array(INFERENCE_BUFFER_SIZE).fill(0), head: 0, count: 0 };
    inferenceTimings.set(model, tracker);
  }
  tracker.buffer[tracker.head] = ms;
  tracker.head = (tracker.head + 1) % INFERENCE_BUFFER_SIZE;
  if (tracker.count < INFERENCE_BUFFER_SIZE) tracker.count++;
}

function getAvgInferenceMs(model: string): number {
  const defaults: Record<string, number> = { xgboost: 3, gnn: 8 };
  const tracker = inferenceTimings.get(model);
  if (!tracker || tracker.count === 0) return defaults[model] ?? 5;
  let sum = 0;
  for (let i = 0; i < tracker.count; i++) sum += tracker.buffer[i];
  return Math.round((sum / tracker.count) * 100) / 100;
}

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
  newOutcomesSinceLastTrain: number;
  stale: boolean;
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
  hitRate: number | null;
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

export interface FeatureImportanceEntry {
  name: string;
  index: number;
  importance: number;
  normalizedImportance: number;
}

export interface ErrorCluster {
  pattern: string;
  family: string;
  direction: "over" | "under";
  count: number;
  meanError: number;
  medianError: number;
  worstFormula: string;
  worstError: number;
  pressureRelated: boolean;
  suggestedAction: string;
}

export interface CrossModelInsight {
  models: string[];
  family: string;
  pattern: string;
  suggestion: string;
}

export interface ErrorAnalysisReport {
  totalOutcomes: number;
  totalErrors: number;
  largeErrors: number;
  errorClusters: ErrorCluster[];
  overallBias: "over" | "under" | "neutral";
  overallMeanError: number;
  overallRMSE: number;
  overallNRMSE: number;
  topFailures: { formula: string; predicted: number; actual: number; error: number; family: string }[];
  familyDataGaps: { family: string; sampleCount: number; needsMore: boolean }[];
  dataRequestSuggestions: string[];
  crossModelInsights: CrossModelInsight[];
}

export interface FailedMaterialSummary {
  formula: string;
  failureReason: string;
  source: string;
  formationEnergy?: number;
  imaginaryModeCount?: number;
  lowestPhononFreq?: number;
  bandGap?: number;
  details?: string;
  failedAt: number;
}

export interface RootCauseCluster {
  rootCause: "lattice_instability" | "electronic_instability" | "isolated";
  formulas: string[];
  reasons: string[];
  count: number;
  description: string;
}

export interface FailureSummaryReport {
  totalFailures: number;
  byReason: { reason: string; count: number; percentage: number }[];
  bySource: { source: string; count: number }[];
  recentFailures: FailedMaterialSummary[];
  failurePatterns: {
    topFailingElementPairs: { pair: string; count: number }[];
    topFailingCrystalSystems: { system: string; count: number }[];
  };
  predictedStableActualUnstable: FailedMaterialSummary[];
  rootCauseClusters: RootCauseCluster[];
  llmSuggestions: PrioritizedSuggestion[];
}

export interface PrioritizedSuggestion {
  text: string;
  priority: "critical" | "high" | "medium" | "low";
  impactScore: number;
  category: "retrain" | "prefilter" | "data_gap" | "architecture" | "monitoring";
}

export interface ModelVersionScorecard {
  modelName: string;
  version: number;
  trainedAt: number;
  metrics: Record<string, number>;
  hyperparameters: Record<string, number | string>;
  datasetSize: number;
  inferenceSpeedMs: number;
  computeEnvironment: string;
}

export interface ModelBenchmarkReport {
  scorecards: ModelVersionScorecard[];
  versionComparisons: VersionComparison[];
  bestVersionByMetric: Record<string, { version: number; value: number }>;
}

export interface VersionComparison {
  modelName: string;
  fromVersion: number;
  toVersion: number;
  metricDeltas: Record<string, number>;
  recommendation: string;
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
  featureImportance: FeatureImportanceEntry[];
  errorAnalysis: ErrorAnalysisReport;
  failureSummary: FailureSummaryReport;
  benchmark: ModelBenchmarkReport;
}

const MAX_OUTCOMES = 500;
const outcomeBuffer: PredictionOutcome[] = new Array(MAX_OUTCOMES);
let outcomeHead = 0;
let outcomeCount = 0;

function getOutcomes(): PredictionOutcome[] {
  if (outcomeCount < MAX_OUTCOMES) return outcomeBuffer.slice(0, outcomeCount);
  return [...outcomeBuffer.slice(outcomeHead), ...outcomeBuffer.slice(0, outcomeHead)];
}

export function recordPredictionOutcome(model: string, formula: string, predicted: number, actual: number): void {
  const family = classifyFamily(formula);
  const entry: PredictionOutcome = {
    model,
    formula,
    predicted,
    actual,
    timestamp: Date.now(),
    family,
  };
  if (outcomeCount < MAX_OUTCOMES) {
    outcomeBuffer[outcomeCount] = entry;
    outcomeCount++;
  } else {
    outcomeBuffer[outcomeHead] = entry;
    outcomeHead = (outcomeHead + 1) % MAX_OUTCOMES;
  }
  diagnosticCache = null;
}

function computeFamilyBias(modelFilter?: string): FamilyBias[] {
  const families = ["hydride", "cuprate", "pnictide", "boride", "conventional"];
  const result: FamilyBias[] = [];

  for (const family of families) {
    const outcomes = getOutcomes().filter(
      o => o.family === family && (modelFilter ? o.model === modelFilter : true)
    );
    if (outcomes.length === 0) {
      result.push({ family, count: 0, meanError: 0, meanAbsError: 0, bias: "neutral" });
      continue;
    }
    const errors = outcomes.map(o => o.predicted - o.actual);
    const meanError = errors.reduce((s, e) => s + e, 0) / errors.length;
    const meanAbsError = errors.reduce((s, e) => s + Math.abs(e), 0) / errors.length;
    const avgActual = outcomes.reduce((s, o) => s + Math.abs(o.actual), 0) / outcomes.length;
    const biasThreshold = Math.max(2, avgActual * 0.15);
    const bias: "over" | "under" | "neutral" = meanError > biasThreshold ? "over" : meanError < -biasThreshold ? "under" : "neutral";
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

    const allOutcomes = getOutcomes();
    const inBin = allOutcomes.filter(o => {
      const relError = Math.abs(o.predicted - o.actual) / Math.max(Math.abs(o.actual), 1);
      return relError >= (1 - upper) && relError < (1 - lower);
    });

    const withinRange = inBin.filter(o => {
      const binMidpoint = (lower + upper) / 2;
      const tolerance = Math.max(5, Math.abs(o.actual) * (1 - binMidpoint));
      const absDiff = Math.abs(o.predicted - o.actual);
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

function computeErrorAnalysis(): ErrorAnalysisReport {
  const allOutcomes = getOutcomes();
  const outcomes = allOutcomes.filter(o =>
    Number.isFinite(o.predicted) && Number.isFinite(o.actual)
  );
  const totalOutcomes = outcomes.length;

  if (totalOutcomes === 0) {
    return {
      totalOutcomes: 0, totalErrors: 0, largeErrors: 0,
      errorClusters: [], overallBias: "neutral", overallMeanError: 0, overallRMSE: 0,
      overallNRMSE: 0,
      topFailures: [], familyDataGaps: [], dataRequestSuggestions: [],
      crossModelInsights: [],
    };
  }

  const now = Date.now();
  const HALF_LIFE_MS = 3600_000;
  const weights = outcomes.map(o => {
    const age = now - o.timestamp;
    return Math.pow(0.5, age / HALF_LIFE_MS);
  });
  const totalWeight = weights.reduce((s, w) => s + w, 0);

  const errors = outcomes.map(o => o.predicted - o.actual);
  const absErrors = errors.map(e => Math.abs(e));
  const overallMeanError = totalWeight > 0
    ? errors.reduce((s, e, i) => s + e * weights[i], 0) / totalWeight
    : 0;
  const weightedSqErr = totalWeight > 0
    ? errors.reduce((s, e, i) => s + e * e * weights[i], 0) / totalWeight
    : 0;
  const overallRMSE = Math.sqrt(weightedSqErr);

  const actualRange = Math.max(1, Math.max(...outcomes.map(o => Math.abs(o.actual))));
  const overallNRMSE = overallRMSE / actualRange;

  const largeErrorThreshold = 30;
  const largeErrors = absErrors.filter(e => e > largeErrorThreshold).length;
  const totalErrors = absErrors.filter(e => e > 5).length;
  const avgActualAll = outcomes.reduce((s, o) => s + Math.abs(o.actual), 0) / totalOutcomes;
  const biasThreshGlobal = Math.max(5, avgActualAll * 0.15);
  const overallBias: "over" | "under" | "neutral" = overallMeanError > biasThreshGlobal ? "over" : overallMeanError < -biasThreshGlobal ? "under" : "neutral";

  const clusters: ErrorCluster[] = [];
  const families = ["hydride", "cuprate", "pnictide", "boride", "conventional"];

  for (const family of families) {
    const familyOutcomes = outcomes.filter(o => o.family === family);
    if (familyOutcomes.length < 3) continue;

    const familyErrors = familyOutcomes.map(o => o.predicted - o.actual);
    const familyAbsErrors = familyErrors.map(e => Math.abs(e));
    const meanErr = familyErrors.reduce((s, e) => s + e, 0) / familyErrors.length;
    const sortedAbsErrors = [...familyAbsErrors].sort((a, b) => a - b);
    const medianErr = sortedAbsErrors[Math.floor(sortedAbsErrors.length / 2)];

    const overCount = familyErrors.filter(e => e > 10).length;
    const underCount = familyErrors.filter(e => e < -10).length;

    if (overCount > familyOutcomes.length * 0.4 || underCount > familyOutcomes.length * 0.4) {
      const direction: "over" | "under" = overCount > underCount ? "over" : "under";
      const worstIdx = familyAbsErrors.indexOf(Math.max(...familyAbsErrors));
      const worstOutcome = familyOutcomes[worstIdx];
      const hasHighHContent = familyOutcomes.some(o => o.formula.match(/H\d{2,}/));
      const isHydride = family === "hydride" || family === "Hydrides";
      const pressureRelated = hasHighHContent;

      let suggestedAction = "";
      if (isHydride && direction === "over" && hasHighHContent) {
        suggestedAction = `Add pressure feature correction for ${family}; retrain lambda model with pressure-adjusted data`;
      } else if (isHydride && direction === "over") {
        suggestedAction = `Check anharmonicity and H zero-point motion corrections for ${family}; these can suppress Tc independently of pressure`;
      } else if (isHydride && direction === "under") {
        suggestedAction = `Review H zero-point energy and quantum nuclear effects for ${family}; anharmonic phonon softening may enhance coupling`;
      } else if (direction === "over") {
        suggestedAction = `Recalibrate ${family} predictions; add regularization or family-specific bias correction`;
      } else {
        suggestedAction = `Expand ${family} training data; model may be underexploring this region`;
      }

      clusters.push({
        pattern: `Model ${direction === "over" ? "overpredicts" : "underpredicts"} ${family}`,
        family,
        direction,
        count: familyOutcomes.length,
        meanError: Math.round(meanErr * 100) / 100,
        medianError: Math.round(medianErr * 100) / 100,
        worstFormula: worstOutcome.formula,
        worstError: Math.round((worstOutcome.predicted - worstOutcome.actual) * 100) / 100,
        pressureRelated,
        suggestedAction,
      });
    }
  }

  const highHContentOutcomes = outcomes.filter(o => o.formula.match(/H\d{2,}/));
  const lowHHydrideOutcomes = outcomes.filter(o =>
    (o.family === "hydride" || o.family === "Hydrides") && !o.formula.match(/H\d{2,}/)
  );

  if (highHContentOutcomes.length >= 3) {
    const hpErrors = highHContentOutcomes.map(o => o.predicted - o.actual);
    const hpMean = hpErrors.reduce((s, e) => s + e, 0) / hpErrors.length;
    if (Math.abs(hpMean) > 20) {
      const duplicateSpecific = clusters.find(c => c.pattern.includes("High-H-content") && c.pressureRelated);
      if (!duplicateSpecific) {
        const hpAbsErrors = hpErrors.map(e => Math.abs(e));
        const sortedHp = [...hpErrors].sort((a, b) => a - b);
        const hpMedian = sortedHp[Math.floor(sortedHp.length / 2)];
        const worstIdx = hpAbsErrors.indexOf(Math.max(...hpAbsErrors));
        clusters.push({
          pattern: `High-H-content hydride ${hpMean > 0 ? "overprediction" : "underprediction"} bias`,
          family: "hydride",
          direction: hpMean > 0 ? "over" : "under",
          count: highHContentOutcomes.length,
          meanError: Math.round(hpMean * 100) / 100,
          medianError: Math.round(hpMedian * 100) / 100,
          worstFormula: highHContentOutcomes[worstIdx].formula,
          worstError: Math.round(hpErrors[worstIdx] * 100) / 100,
          pressureRelated: true,
          suggestedAction: "Add explicit pressure features; retrain with pressure-corrected Tc data",
        });
      }
    }
  }

  if (lowHHydrideOutcomes.length >= 3) {
    const lhErrors = lowHHydrideOutcomes.map(o => o.predicted - o.actual);
    const lhMean = lhErrors.reduce((s, e) => s + e, 0) / lhErrors.length;
    if (Math.abs(lhMean) > 10) {
      const duplicateSpecific = clusters.find(c => c.pattern.includes("Low-H hydride") && !c.pressureRelated);
      if (!duplicateSpecific) {
        const lhAbsErrors = lhErrors.map(e => Math.abs(e));
        const sortedLh = [...lhErrors].sort((a, b) => a - b);
        const lhMedian = sortedLh[Math.floor(sortedLh.length / 2)];
        const worstIdx = lhAbsErrors.indexOf(Math.max(...lhAbsErrors));
        clusters.push({
          pattern: `Low-H hydride ${lhMean > 0 ? "overprediction" : "underprediction"} bias`,
          family: "hydride",
          direction: lhMean > 0 ? "over" : "under",
          count: lowHHydrideOutcomes.length,
          meanError: Math.round(lhMean * 100) / 100,
          medianError: Math.round(lhMedian * 100) / 100,
          worstFormula: lowHHydrideOutcomes[worstIdx].formula,
          worstError: Math.round(lhErrors[worstIdx] * 100) / 100,
          pressureRelated: false,
          suggestedAction: "Review anharmonicity corrections and H zero-point motion effects; these dominate over pressure for low-H-content compounds",
        });
      }
    }
  }

  clusters.sort((a, b) => Math.abs(b.meanError) - Math.abs(a.meanError));

  const topFailures = [...outcomes]
    .map(o => ({
      formula: o.formula,
      predicted: Math.round(o.predicted * 100) / 100,
      actual: Math.round(o.actual * 100) / 100,
      error: Math.round((o.predicted - o.actual) * 100) / 100,
      family: o.family || "unknown",
    }))
    .sort((a, b) => Math.abs(b.error) - Math.abs(a.error))
    .slice(0, 10);

  const calibStats = getCalibrationStats();
  const familyDataGaps: { family: string; sampleCount: number; needsMore: boolean }[] = [];
  const dataRequestSuggestions: string[] = [];

  const MIN_SAMPLES_PER_FAMILY = 20;
  for (const family of families) {
    const familyCalib = calibStats.familyCalibrations.find(fc => fc.family === family);
    const count = familyCalib?.sampleCount ?? 0;
    const needsMore = count < MIN_SAMPLES_PER_FAMILY;
    familyDataGaps.push({ family, sampleCount: count, needsMore });

    if (needsMore) {
      dataRequestSuggestions.push(
        `Generate ${MIN_SAMPLES_PER_FAMILY - count} additional ${family} structures and run TB evaluation to expand training data`
      );
    }

    if (familyCalib && count >= MIN_SAMPLES_PER_FAMILY) {
      const familyOutcomesForAvg = outcomes.filter(o => o.family === family);
      const avgActualTc = familyOutcomesForAvg.length > 0
        ? familyOutcomesForAvg.reduce((s, o) => s + Math.abs(o.actual), 0) / familyOutcomesForAvg.length
        : 50;
      const rmaeThreshold = Math.max(10, avgActualTc * 0.3);
      if (familyCalib.meanAbsError > rmaeThreshold) {
        const rmaePercent = avgActualTc > 0 ? Math.round((familyCalib.meanAbsError / avgActualTc) * 100) : 0;
        dataRequestSuggestions.push(
          `${family} has high relative error (MAE=${familyCalib.meanAbsError.toFixed(1)}K, ${rmaePercent}% of avg Tc=${avgActualTc.toFixed(0)}K) with ${count} samples — run DFT enrichment on worst-predicted ${family} candidates`
        );
      }
    }
  }

  for (const cluster of clusters) {
    if (cluster.pressureRelated) {
      dataRequestSuggestions.push(
        `${cluster.pattern}: generate pressure-variant structures for ${cluster.family} at multiple pressures (0, 50, 150, 300 GPa)`
      );
    }
  }

  const gnnHistoryForStaleness = getGNNVersionHistory();
  const latestGNNForStaleness = gnnHistoryForStaleness.length > 0 ? gnnHistoryForStaleness[gnnHistoryForStaleness.length - 1] : null;
  if (latestGNNForStaleness && (Date.now() - latestGNNForStaleness.trainedAt) > 24 * 3600_000) {
    dataRequestSuggestions.push(
      `GNN ensemble is stale (${Math.round((Date.now() - latestGNNForStaleness.trainedAt) / 3600_000)}h since last train) — trigger GNN retrain with latest DFT data`
    );
  }

  const crossModelInsights: CrossModelInsight[] = [];
  const uniqueModels = [...new Set(outcomes.map(o => o.model))];
  if (uniqueModels.length >= 2) {
    for (const family of families) {
      const modelBiases: { model: string; mean: number; count: number }[] = [];
      for (const model of uniqueModels) {
        const mfOutcomes = outcomes.filter(o => o.model === model && o.family === family);
        if (mfOutcomes.length < 2) continue;
        const mean = mfOutcomes.reduce((s, o) => s + (o.predicted - o.actual), 0) / mfOutcomes.length;
        modelBiases.push({ model, mean, count: mfOutcomes.length });
      }
      if (modelBiases.length >= 2) {
        const allOver = modelBiases.every(mb => mb.mean > 5);
        const allUnder = modelBiases.every(mb => mb.mean < -5);
        const mixed = modelBiases.some(mb => mb.mean > 5) && modelBiases.some(mb => mb.mean < -5);
        if (allOver || allUnder) {
          crossModelInsights.push({
            models: modelBiases.map(mb => mb.model),
            family,
            pattern: `All models ${allOver ? "overpredict" : "underpredict"} ${family}`,
            suggestion: `Systematic ${family} bias across models suggests feature extraction issue (e.g., lambdaProxy or DOS estimation)`,
          });
        } else if (mixed) {
          crossModelInsights.push({
            models: modelBiases.map(mb => mb.model),
            family,
            pattern: `Models disagree on ${family} direction`,
            suggestion: `Ensemble averaging healthy for ${family}; model disagreement provides natural uncertainty estimate`,
          });
        }
      }
    }
  }

  return {
    totalOutcomes,
    totalErrors,
    largeErrors,
    errorClusters: clusters,
    overallBias,
    overallMeanError: Math.round(overallMeanError * 100) / 100,
    overallRMSE: Math.round(overallRMSE * 100) / 100,
    overallNRMSE: Math.round(overallNRMSE * 10000) / 10000,
    topFailures,
    familyDataGaps,
    dataRequestSuggestions,
    crossModelInsights,
  };
}

const DIAGNOSTIC_CACHE_TTL_MS = 60_000;
let diagnosticCache: { result: ComprehensiveModelDiagnostics; timestamp: number } | null = null;

export function invalidateDiagnosticCache(): void {
  diagnosticCache = null;
}

export async function getComprehensiveModelDiagnostics(): Promise<ComprehensiveModelDiagnostics> {
  if (diagnosticCache && (Date.now() - diagnosticCache.timestamp) < DIAGNOSTIC_CACHE_TTL_MS) {
    return diagnosticCache.result;
  }

  const calibration = await getCalibrationData();
  const ensembleStats = getXGBEnsembleStats();
  const versionHistory = getModelVersionHistory();
  const evalStats = getEvaluatedDatasetStats();
  const surrogateStats = getSurrogateStats();

  const latestVersion = versionHistory.latestMetrics;

  let falsePositiveRate = 0;
  let falseNegativeRate = 0;
  let meanResidualSign = 0;

  if (calibration.predictedVsActual && calibration.predictedVsActual.length > 0) {
    const fpCount = calibration.predictedVsActual.filter(p => p.predicted > 77 && p.actual < 20).length;
    const fnCount = calibration.predictedVsActual.filter(p => p.predicted < 20 && p.actual > 77).length;
    const total = calibration.predictedVsActual.length;
    falsePositiveRate = total > 0 ? Math.round((fpCount / total) * 10000) / 10000 : 0;
    falseNegativeRate = total > 0 ? Math.round((fnCount / total) * 10000) / 10000 : 0;

    const residuals = calibration.predictedVsActual.map(p => p.residual);
    const totalMagnitude = residuals.reduce((s, r) => s + Math.abs(r), 0);
    if (totalMagnitude > 0) {
      const signedSum = residuals.reduce((s, r) => s + r, 0);
      meanResidualSign = Math.round((signedSum / totalMagnitude) * 1000) / 1000;
    }
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

  const GNN_STALENESS_THRESHOLD_MS = 24 * 3600_000;
  const gnnStalenessMs = latestGNN ? Date.now() - latestGNN.trainedAt : 0;
  const gnnIsStale = gnnStalenessMs > GNN_STALENESS_THRESHOLD_MS;

  const gnnTrainedAt = latestGNN?.trainedAt ?? 0;
  const newOutcomesSinceLastTrain = gnnTrainedAt > 0
    ? getOutcomes().filter(o => o.timestamp > gnnTrainedAt).length
    : 0;

  const gnn: GNNDiagnostics = {
    ensembleSize: ENSEMBLE_SIZE,
    modelVersion: getGNNModelVersion(),
    datasetSize: dftDatasetStats.totalSize,
    trainedAt: gnnTrainedAt,
    latestR2: latestGNN?.r2 ?? 0,
    latestMAE: latestGNN?.mae ?? 0,
    latestRMSE: latestGNN?.rmse ?? 0,
    predictionCount: dftDatasetStats.totalSize,
    modelStalenessMs: gnnStalenessMs,
    newOutcomesSinceLastTrain,
    stale: gnnIsStale,
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
    hitRate: (phononStats.tierBreakdown.hits + phononStats.tierBreakdown.misses) > 0
      ? Math.round((phononStats.tierBreakdown.hits / (phononStats.tierBreakdown.hits + phononStats.tierBreakdown.misses)) * 1000) / 1000
      : null,
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

  const featureImportance = getGlobalFeatureImportance(25);
  const errorAnalysis = computeErrorAnalysis();
  const failureSummary = computeFailureSummary();
  const benchmark = computeBenchmarkReport();

  const result: ComprehensiveModelDiagnostics = {
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
    predictionOutcomeCount: outcomeCount,
    featureImportance,
    errorAnalysis,
    failureSummary,
    benchmark,
  };

  diagnosticCache = { result, timestamp: Date.now() };
  return result;
}

function toFailedSummary(entry: StructureFailureEntry): FailedMaterialSummary {
  return {
    formula: entry.formula,
    failureReason: entry.failureReason,
    source: entry.source,
    formationEnergy: entry.formationEnergy,
    imaginaryModeCount: entry.imaginaryModeCount,
    lowestPhononFreq: entry.lowestPhononFreq,
    bandGap: entry.bandGap,
    details: entry.details,
    failedAt: entry.failedAt,
  };
}

const STABILITY_FAILURE_REASONS = new Set(["unstable_phonons", "structure_collapse", "high_formation_energy"]);
const RECENT_CAPACITY = 20;

function insertBoundedRecent(arr: FailedMaterialSummary[], item: FailedMaterialSummary): void {
  if (arr.length < RECENT_CAPACITY) {
    arr.push(item);
    let i = arr.length - 1;
    while (i > 0 && arr[i].failedAt > arr[i - 1].failedAt) {
      [arr[i], arr[i - 1]] = [arr[i - 1], arr[i]];
      i--;
    }
  } else if (item.failedAt > arr[arr.length - 1].failedAt) {
    arr[arr.length - 1] = item;
    let i = arr.length - 1;
    while (i > 0 && arr[i].failedAt > arr[i - 1].failedAt) {
      [arr[i], arr[i - 1]] = [arr[i - 1], arr[i]];
      i--;
    }
  }
}

function computeFailureSummary(): FailureSummaryReport {
  const patterns = getFailurePatterns();
  const allEntries = getFailureEntries();

  const reasonCounts = new Map<string, number>();
  const sourceCounts = new Map<string, number>();
  const recentFailures: FailedMaterialSummary[] = [];
  const recentStableUnstable: FailedMaterialSummary[] = [];
  let stableUnstableTotal = 0;

  for (const entry of allEntries) {
    reasonCounts.set(entry.failureReason, (reasonCounts.get(entry.failureReason) || 0) + 1);
    sourceCounts.set(entry.source, (sourceCounts.get(entry.source) || 0) + 1);

    const summary = toFailedSummary(entry);
    insertBoundedRecent(recentFailures, summary);

    if (STABILITY_FAILURE_REASONS.has(entry.failureReason)) {
      stableUnstableTotal++;
      insertBoundedRecent(recentStableUnstable, summary);
    }
  }

  const totalEntries = allEntries.length;
  const total = totalEntries || 1;
  const byReason = Array.from(reasonCounts.entries())
    .map(([reason, count]) => ({
      reason,
      count,
      percentage: Math.round((count / total) * 1000) / 10,
    }))
    .sort((a, b) => b.count - a.count);

  const bySource = Array.from(sourceCounts.entries())
    .map(([source, count]) => ({ source, count }))
    .sort((a, b) => b.count - a.count);

  const LATTICE_INSTABILITY_REASONS = new Set(["unstable_phonons", "structure_collapse", "high_formation_energy"]);
  const ELECTRONIC_INSTABILITY_REASONS = new Set(["non_metallic", "scf_divergence"]);
  const formulaReasons = new Map<string, Set<string>>();
  for (const entry of allEntries) {
    if (!formulaReasons.has(entry.formula)) formulaReasons.set(entry.formula, new Set());
    formulaReasons.get(entry.formula)!.add(entry.failureReason);
  }

  const rootCauseClusters: RootCauseCluster[] = [];
  const latticeFormulas: string[] = [];
  const electronicFormulas: string[] = [];
  const latticeReasonsSeen = new Set<string>();
  const electronicReasonsSeen = new Set<string>();

  for (const [formula, reasons] of formulaReasons) {
    const latticeReasons = [...reasons].filter(r => LATTICE_INSTABILITY_REASONS.has(r));
    const electronicReasons = [...reasons].filter(r => ELECTRONIC_INSTABILITY_REASONS.has(r));
    if (latticeReasons.length >= 2) {
      latticeFormulas.push(formula);
      for (const r of latticeReasons) latticeReasonsSeen.add(r);
    }
    if (electronicReasons.length >= 2 || (electronicReasons.length >= 1 && latticeReasons.length >= 1)) {
      electronicFormulas.push(formula);
      for (const r of electronicReasons) electronicReasonsSeen.add(r);
    }
  }

  if (latticeFormulas.length > 0) {
    rootCauseClusters.push({
      rootCause: "lattice_instability",
      formulas: latticeFormulas.slice(0, 20),
      reasons: [...latticeReasonsSeen],
      count: latticeFormulas.length,
      description: `${latticeFormulas.length} formulas exhibit multiple lattice instability signals (${[...latticeReasonsSeen].join(" + ")}), suggesting the generator produces unphysical lattice constants`,
    });
  }
  if (electronicFormulas.length > 0) {
    rootCauseClusters.push({
      rootCause: "electronic_instability",
      formulas: electronicFormulas.slice(0, 20),
      reasons: [...electronicReasonsSeen],
      count: electronicFormulas.length,
      description: `${electronicFormulas.length} formulas show coupled electronic instability (${[...electronicReasonsSeen].join(" + ")})`,
    });
  }

  const suggestions: PrioritizedSuggestion[] = [];
  const phononFailures = reasonCounts.get("unstable_phonons") || 0;
  const collapseFailures = reasonCounts.get("structure_collapse") || 0;
  const energyFailures = reasonCounts.get("high_formation_energy") || 0;

  const phononRate = totalEntries > 0 ? phononFailures / totalEntries : 0;
  const collapseRate = totalEntries > 0 ? collapseFailures / totalEntries : 0;
  const energyRate = totalEntries > 0 ? energyFailures / totalEntries : 0;
  const falseStableRate = totalEntries > 0 ? stableUnstableTotal / totalEntries : 0;

  if (phononRate > 0.05 && phononFailures >= 3) {
    suggestions.push({
      text: `Train phonon stability classifier — ${phononFailures}/${totalEntries} (${(phononRate * 100).toFixed(1)}%) phonon instability failures`,
      priority: phononRate > 0.2 ? "critical" : phononRate > 0.1 ? "high" : "medium",
      impactScore: Math.round(phononRate * 100) / 100,
      category: "retrain",
    });
  }
  if (collapseRate > 0.03 && collapseFailures >= 2) {
    suggestions.push({
      text: `Improve structure predictor — ${collapseFailures}/${totalEntries} (${(collapseRate * 100).toFixed(1)}%) structure collapse failures`,
      priority: collapseRate > 0.15 ? "critical" : collapseRate > 0.08 ? "high" : "medium",
      impactScore: Math.round(collapseRate * 100) / 100,
      category: "architecture",
    });
  }
  if (energyRate > 0.05 && energyFailures >= 3) {
    suggestions.push({
      text: `Tighten formation energy pre-filter — ${energyFailures}/${totalEntries} (${(energyRate * 100).toFixed(1)}%) high formation energy rejections`,
      priority: energyRate > 0.15 ? "high" : "medium",
      impactScore: Math.round(energyRate * 100) / 100,
      category: "prefilter",
    });
  }
  if (falseStableRate > 0.08 && stableUnstableTotal >= 5) {
    suggestions.push({
      text: `High false-stable rate: ${stableUnstableTotal}/${totalEntries} (${(falseStableRate * 100).toFixed(1)}%) predicted stable but actually unstable`,
      priority: falseStableRate > 0.25 ? "critical" : falseStableRate > 0.15 ? "high" : "medium",
      impactScore: Math.round(falseStableRate * 100) / 100,
      category: "retrain",
    });
  }
  if (patterns.elementPairs.length > 0) {
    const worst = patterns.elementPairs[0];
    const pairRate = totalEntries > 0 ? worst.failureCount / totalEntries : 0;
    suggestions.push({
      text: `Avoid element pair ${worst.pair} — ${worst.failureCount} failures`,
      priority: pairRate > 0.1 ? "high" : "low",
      impactScore: Math.round(pairRate * 100) / 100,
      category: "prefilter",
    });
  }
  if (rootCauseClusters.length > 0) {
    const latticeCluster = rootCauseClusters.find(c => c.rootCause === "lattice_instability");
    if (latticeCluster && latticeCluster.count >= 3) {
      const rate = totalEntries > 0 ? latticeCluster.count / totalEntries : 0;
      suggestions.push({
        text: `Root cause: ${latticeCluster.count} formulas share linked lattice instability (${latticeCluster.reasons.join(" + ")}) — generator may produce unphysical lattice constants`,
        priority: rate > 0.15 ? "critical" : rate > 0.08 ? "high" : "medium",
        impactScore: Math.round(rate * 100) / 100,
        category: "architecture",
      });
    }
  }

  suggestions.sort((a, b) => b.impactScore - a.impactScore);

  return {
    totalFailures: totalEntries,
    byReason,
    bySource,
    recentFailures,
    failurePatterns: {
      topFailingElementPairs: patterns.elementPairs.slice(0, 10).map(p => ({ pair: p.pair, count: p.failureCount })),
      topFailingCrystalSystems: patterns.crystalSystems.slice(0, 5).map(s => ({ system: s.system, count: s.failureCount })),
    },
    predictedStableActualUnstable: recentStableUnstable,
    rootCauseClusters,
    llmSuggestions: suggestions,
  };
}

function compareModelVersions(cards: ModelVersionScorecard[]): VersionComparison[] {
  const result: VersionComparison[] = [];
  for (let i = 1; i < cards.length; i++) {
    const prev = cards[i - 1];
    const curr = cards[i];
    const deltas: Record<string, number> = {};
    for (const key of Object.keys(curr.metrics)) {
      deltas[key] = Math.round((curr.metrics[key] - (prev.metrics[key] || 0)) * 10000) / 10000;
    }
    const r2Better = (deltas.r2 || 0) > 0;
    const maeBetter = (deltas.mae || 0) < 0;
    let rec = "no significant change";
    if (r2Better && maeBetter) rec = "improved accuracy — keep architecture";
    else if (r2Better) rec = "improved R² — monitor MAE";
    else if (maeBetter) rec = "improved MAE — monitor R²";
    else if ((deltas.r2 || 0) < -0.02) rec = "regression detected — consider rollback";
    if (prev.computeEnvironment !== curr.computeEnvironment) {
      rec += ` (compute env changed: ${prev.computeEnvironment} -> ${curr.computeEnvironment})`;
    }
    result.push({
      modelName: curr.modelName,
      fromVersion: prev.version,
      toVersion: curr.version,
      metricDeltas: deltas,
      recommendation: rec,
    });
  }
  return result;
}

function computeBenchmarkReport(): ModelBenchmarkReport {
  const xgbHistory = getModelVersionHistory();
  const gnnHistory = getGNNVersionHistory();
  const scorecards: ModelVersionScorecard[] = [];
  const comparisons: VersionComparison[] = [];
  const bestByMetric: Record<string, { version: number; value: number }> = {};

  const sortedXgbHistory = [...xgbHistory.history].sort((a, b) => a.version - b.version);
  for (const v of sortedXgbHistory) {
    scorecards.push({
      modelName: "xgboost",
      version: v.version,
      trainedAt: v.trainedAt,
      metrics: {
        r2: v.r2,
        mae: v.mae,
        rmse: v.rmse,
        predictionVariance: v.predictionVariance,
      },
      hyperparameters: {
        nTrees: v.nTrees,
        ensembleSize: v.ensembleSize,
      },
      datasetSize: v.datasetSize,
      inferenceSpeedMs: getAvgInferenceMs("xgboost"),
      computeEnvironment: "cpu",
    });
  }

  comparisons.push(...compareModelVersions(scorecards.filter(s => s.modelName === "xgboost")));

  const sortedGnnHistory = [...gnnHistory].sort((a, b) => a.version - b.version);
  for (const v of sortedGnnHistory) {
    scorecards.push({
      modelName: "gnn",
      version: v.version,
      trainedAt: v.trainedAt,
      metrics: {
        r2: v.r2,
        mae: v.mae,
        rmse: v.rmse,
      },
      hyperparameters: {
        ensembleSize: v.ensembleSize,
        dftSamples: v.dftSamples,
      },
      datasetSize: v.datasetSize,
      inferenceSpeedMs: getAvgInferenceMs("gnn"),
      computeEnvironment: "cpu",
    });
  }

  comparisons.push(...compareModelVersions(scorecards.filter(s => s.modelName === "gnn")));

  const HIGHER_BETTER_METRICS = new Set(["r2", "accuracy", "hitRate", "stabilityAccuracy", "f1_score"]);
  for (const card of scorecards) {
    for (const [metric, value] of Object.entries(card.metrics)) {
      const key = `${card.modelName}:${metric}`;
      const isHigherBetter = HIGHER_BETTER_METRICS.has(metric);
      const existing = bestByMetric[key];
      if (!existing ||
        (isHigherBetter && value > existing.value) ||
        (!isHigherBetter && value < existing.value)) {
        bestByMetric[key] = { version: card.version, value };
      }
    }
  }

  return {
    scorecards: scorecards.slice(-30),
    versionComparisons: comparisons.slice(-20),
    bestVersionByMetric: bestByMetric,
  };
}

export async function getModelDiagnosticsForLLM(): Promise<string> {
  const d = await getComprehensiveModelDiagnostics();
  const lines: string[] = [];

  lines.push("=== MODEL DIAGNOSTICS REPORT ===");
  lines.push("");

  lines.push("## XGBoost Tc Predictor");
  lines.push(`  R²=${d.xgboost.r2} | MAE=${d.xgboost.mae}K | RMSE=${d.xgboost.rmse}K`);
  lines.push(`  Trees=${d.xgboost.nTrees} | Features=${d.xgboost.featureCount} | Dataset=${d.xgboost.nSamples}`);
  lines.push(`  Ensemble: ${d.xgboost.ensembleSize} models, tree counts=[${d.xgboost.ensembleTreeCounts.join(",")}]`);
  lines.push(`  Prediction variance=${d.xgboost.predictionVariance}K`);
  lines.push(`  False positive rate (pred>77K,actual<20K)=${d.xgboost.falsePositiveRate}`);
  lines.push(`  False negative rate (pred<20K,actual>77K)=${d.xgboost.falseNegativeRate}`);
  lines.push(`  Prediction bias (impact-weighted sign)=${d.xgboost.meanResidualSign} (>0 = net overprediction, <0 = net underprediction)`);
  lines.push(`  Residual p90=${d.xgboost.absResidualPercentiles.p90}K, p95=${d.xgboost.absResidualPercentiles.p95}K`);
  if (d.xgboost.r2 < 0.5) lines.push("  ** WARNING: Low R² — model may be underfitting **");
  const familyTcRanges: Record<string, { label: string; avgTc: number; acceptableRMSE: number }> = {
    conventional: { label: "conventional (<10K)", avgTc: 5, acceptableRMSE: 3 },
    pnictide: { label: "pnictide (10-60K)", avgTc: 35, acceptableRMSE: 10 },
    cuprate: { label: "cuprate (30-140K)", avgTc: 90, acceptableRMSE: 20 },
    hydride: { label: "hydride (100-300K)", avgTc: 200, acceptableRMSE: 40 },
    boride: { label: "boride (1-40K)", avgTc: 20, acceptableRMSE: 8 },
  };
  for (const fb of d.familyBias) {
    if (fb.count === 0) continue;
    const range = familyTcRanges[fb.family];
    if (range && fb.meanAbsError > range.acceptableRMSE) {
      const pctError = range.avgTc > 0 ? Math.round((fb.meanAbsError / range.avgTc) * 100) : 0;
      lines.push(`  ** WARNING: ${range.label} MAE=${fb.meanAbsError}K is ${pctError}% of avg Tc=${range.avgTc}K (acceptable <${range.acceptableRMSE}K) **`);
    }
  }
  if (d.xgboost.rmse > 30 && !d.familyBias.some(fb => fb.count > 0)) {
    lines.push("  ** WARNING: High RMSE — large prediction errors **");
  }
  lines.push("");

  lines.push("## GNN Ensemble");
  lines.push(`  Version=${d.gnn.modelVersion} | Ensemble=${d.gnn.ensembleSize} models`);
  lines.push(`  R²=${d.gnn.latestR2} | MAE=${d.gnn.latestMAE}K | RMSE=${d.gnn.latestRMSE}K`);
  lines.push(`  Dataset=${d.gnn.datasetSize} | Staleness=${Math.round(d.gnn.modelStalenessMs / 60000)}min | NewOutcomes=${d.gnn.newOutcomesSinceLastTrain} | Stale=${d.gnn.stale}`);
  if (d.gnn.stale) lines.push("  ** WARNING: Model is stale (>24h) — retrain recommended **");
  if (d.gnn.newOutcomesSinceLastTrain >= 50) lines.push(`  ** WARNING: ${d.gnn.newOutcomesSinceLastTrain} new outcomes since last train — model may be operating on outdated data **`);
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
  lines.push(`  Hit rate=${d.phononSurrogate.hitRate !== null ? d.phononSurrogate.hitRate : "N/A (no evaluated predictions)"}`);
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

  if (d.featureImportance.length > 0) {
    lines.push("## Feature Importance (XGBoost top features)");
    const topFeatures = d.featureImportance.slice(0, 15);
    for (const f of topFeatures) {
      lines.push(`  ${f.name}: ${f.normalizedImportance.toFixed(3)}`);
    }
    const isCompositionFeature = (name: string) => COMPOSITION_FEATURE_SET.has(name);
    const compFeatures = topFeatures.filter(f => isCompositionFeature(f.name));
    const physicsFeatures = topFeatures.filter(f => !isCompositionFeature(f.name));
    lines.push(`  Physics features in top 15: ${physicsFeatures.length}`);
    lines.push(`  Composition features in top 15: ${compFeatures.length}`);

    const phonon = topFeatures.filter(f => /phonon|omegaLog|debye/i.test(f.name));
    const eph = topFeatures.filter(f => /lambda|coupling|alpha|massEnhancement/i.test(f.name));
    const electronic = topFeatures.filter(f => /dos|band|nesting|vanHove|fermi/i.test(f.name));
    if (phonon.length === 0) lines.push("  ** NOTE: No phonon features in top 15 — model may ignore phonon proxies **");
    if (eph.length === 0) lines.push("  ** NOTE: No e-ph coupling features in top 15 — model may miss lambda signals **");
    if (electronic.length === 0) lines.push("  ** NOTE: No electronic structure features in top 15 **");
    lines.push("");
  }

  if (d.errorAnalysis.totalOutcomes > 0) {
    lines.push("## Error Analysis");
    lines.push(`  Total evaluations: ${d.errorAnalysis.totalOutcomes}`);
    lines.push(`  Errors > 5K: ${d.errorAnalysis.totalErrors} | Errors > 30K: ${d.errorAnalysis.largeErrors}`);
    lines.push(`  Overall bias: ${d.errorAnalysis.overallBias} (mean error=${d.errorAnalysis.overallMeanError}K)`);
    lines.push(`  Overall RMSE: ${d.errorAnalysis.overallRMSE}K (NRMSE: ${(d.errorAnalysis.overallNRMSE * 100).toFixed(1)}%)`);

    if (d.errorAnalysis.crossModelInsights.length > 0) {
      lines.push("  Cross-model insights:");
      for (const ci of d.errorAnalysis.crossModelInsights) {
        lines.push(`    ${ci.pattern} [${ci.models.join(", ")}]: ${ci.suggestion}`);
      }
    }

    if (d.errorAnalysis.errorClusters.length > 0) {
      lines.push("  Error clusters:");
      for (const c of d.errorAnalysis.errorClusters) {
        lines.push(`    ${c.pattern}: n=${c.count}, mean_err=${c.meanError}K, worst=${c.worstFormula} (${c.worstError}K)`);
        lines.push(`      Action: ${c.suggestedAction}`);
      }
    }

    if (d.errorAnalysis.topFailures.length > 0) {
      lines.push("  Top prediction failures:");
      for (const f of d.errorAnalysis.topFailures.slice(0, 5)) {
        lines.push(`    ${f.formula} [${f.family}]: predicted=${f.predicted}K, actual=${f.actual}K, error=${f.error}K`);
      }
    }

    if (d.errorAnalysis.familyDataGaps.some(g => g.needsMore)) {
      lines.push("  Data gaps:");
      for (const g of d.errorAnalysis.familyDataGaps.filter(g => g.needsMore)) {
        lines.push(`    ${g.family}: only ${g.sampleCount} samples (needs >= 20)`);
      }
    }

    if (d.errorAnalysis.dataRequestSuggestions.length > 0) {
      lines.push("  Data requests:");
      for (const s of d.errorAnalysis.dataRequestSuggestions.slice(0, 5)) {
        lines.push(`    - ${s}`);
      }
    }
    lines.push("");
  }

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

  if (d.failureSummary.totalFailures > 0) {
    lines.push("## Failed Materials Analysis");
    lines.push(`  Total failures: ${d.failureSummary.totalFailures}`);
    lines.push("  By failure reason:");
    for (const r of d.failureSummary.byReason) {
      lines.push(`    ${r.reason}: ${r.count} (${r.percentage}%)`);
    }
    lines.push("  By source:");
    for (const s of d.failureSummary.bySource) {
      lines.push(`    ${s.source}: ${s.count}`);
    }
    if (d.failureSummary.predictedStableActualUnstable.length > 0) {
      lines.push(`  Predicted stable but actually unstable: ${d.failureSummary.predictedStableActualUnstable.length}`);
      for (const f of d.failureSummary.predictedStableActualUnstable.slice(0, 8)) {
        const extras: string[] = [];
        if (f.imaginaryModeCount) extras.push(`imaginary_modes=${f.imaginaryModeCount}`);
        if (f.lowestPhononFreq != null) extras.push(`lowest_freq=${f.lowestPhononFreq}`);
        if (f.formationEnergy != null) extras.push(`Ef=${f.formationEnergy}`);
        lines.push(`    ${f.formula} [${f.failureReason}] via ${f.source}${extras.length > 0 ? " | " + extras.join(", ") : ""}`);
      }
    }
    if (d.failureSummary.rootCauseClusters.length > 0) {
      lines.push("  Root cause analysis:");
      for (const rc of d.failureSummary.rootCauseClusters) {
        lines.push(`    ${rc.rootCause}: ${rc.count} formulas — ${rc.description}`);
        if (rc.formulas.length > 0) lines.push(`      examples: ${rc.formulas.slice(0, 5).join(", ")}`);
      }
    }
    if (d.failureSummary.failurePatterns.topFailingElementPairs.length > 0) {
      lines.push("  Top failing element pairs:");
      for (const p of d.failureSummary.failurePatterns.topFailingElementPairs.slice(0, 5)) {
        lines.push(`    ${p.pair}: ${p.count} failures`);
      }
    }
    if (d.failureSummary.llmSuggestions.length > 0) {
      lines.push("  Suggested actions from failure analysis:");
      for (const s of d.failureSummary.llmSuggestions) {
        lines.push(`    - [${s.priority.toUpperCase()}] (impact=${s.impactScore}, ${s.category}) ${s.text}`);
      }
    }
    lines.push("");
  }

  if (d.benchmark.scorecards.length > 0) {
    lines.push("## Model Version Benchmarks");
    const sortedCards = [...d.benchmark.scorecards].sort((a, b) => a.version - b.version);
    const latestByModel = new Map<string, ModelVersionScorecard>();
    for (const sc of sortedCards) {
      latestByModel.set(sc.modelName, sc);
    }
    for (const [model, sc] of latestByModel) {
      const metricStr = Object.entries(sc.metrics).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`).join(", ");
      const hpAll = Object.entries(sc.hyperparameters);
      const hpStr = hpAll.slice(0, 5).map(([k, v]) => `${k}=${v}`).join(", ") + (hpAll.length > 5 ? ` (+${hpAll.length - 5} more)` : "");
      lines.push(`  ${model} v${sc.version}: ${metricStr}`);
      lines.push(`    hyperparams: ${hpStr} | dataset=${sc.datasetSize} | inference=${sc.inferenceSpeedMs}ms (${sc.computeEnvironment})`);
    }
    if (d.benchmark.versionComparisons.length > 0) {
      const recentComps = d.benchmark.versionComparisons.slice(-5);
      lines.push("  Recent version comparisons:");
      for (const c of recentComps) {
        const deltaStr = Object.entries(c.metricDeltas).map(([k, v]) => `${k}=${v > 0 ? "+" : ""}${v}`).join(", ");
        lines.push(`    ${c.modelName} v${c.fromVersion}->v${c.toVersion}: ${deltaStr} | ${c.recommendation}`);
      }
    }
    lines.push("");
  }

  lines.push(`Prediction outcomes tracked: ${d.predictionOutcomeCount}`);
  lines.push(`Report generated: ${new Date(d.timestamp).toISOString()}`);

  return lines.join("\n");
}

const SEVERITY_MAP: Record<number, HealthStatus> = { 0: "green", 1: "yellow", 2: "red" };

function healthCheck(model: string, checks: { severity: number; reason: string }[]): ModelHealth {
  let maxSev = 0;
  const reasons: string[] = [];
  for (const c of checks) {
    if (c.severity > 0) {
      const tag = c.severity >= 2 ? "[RED]" : "[YELLOW]";
      reasons.push(`${tag} ${c.reason}`);
      if (c.severity > maxSev) maxSev = c.severity;
    }
  }
  if (reasons.length === 0) reasons.push("All metrics within acceptable range");
  return { model, status: SEVERITY_MAP[maxSev] || "green", reasons };
}

export async function getModelHealthSummary(): Promise<ModelHealth[]> {
  const d = await getComprehensiveModelDiagnostics();
  const health: ModelHealth[] = [];

  const nrmse = d.errorAnalysis.totalOutcomes > 0 ? d.errorAnalysis.overallNRMSE : null;
  const nrmseSeverity = nrmse !== null ? (nrmse > 0.5 ? 2 : nrmse > 0.3 ? 1 : 0) : 0;
  const nrmseReason = nrmse !== null
    ? `NRMSE=${(nrmse * 100).toFixed(1)}% — ${nrmse > 0.5 ? "predictions unusable relative to Tc scale" : "elevated error relative to Tc scale"}`
    : "";
  health.push(healthCheck("xgboost", [
    { severity: d.xgboost.r2 < 0.3 ? 2 : d.xgboost.r2 < 0.6 ? 1 : 0, reason: d.xgboost.r2 < 0.3 ? `Very low R²=${d.xgboost.r2}` : `Low R²=${d.xgboost.r2}` },
    { severity: nrmseSeverity, reason: nrmseReason },
    { severity: d.xgboost.falsePositiveRate > 0.15 ? 1 : 0, reason: `High false positive rate=${d.xgboost.falsePositiveRate}` },
    { severity: d.xgboost.nTrees === 0 ? 2 : 0, reason: "No trees trained" },
  ]));

  const gnnOutcomeSev = d.gnn.newOutcomesSinceLastTrain >= 150 ? 2 : d.gnn.newOutcomesSinceLastTrain >= 50 ? 1 : 0;
  health.push(healthCheck("gnn", [
    { severity: d.gnn.latestR2 < 0.2 ? 2 : d.gnn.latestR2 < 0.5 ? 1 : 0, reason: d.gnn.latestR2 < 0.2 ? `Very low R²=${d.gnn.latestR2}` : `Low R²=${d.gnn.latestR2}` },
    { severity: d.gnn.stale ? 2 : d.gnn.modelStalenessMs > 12 * 3600_000 ? 1 : 0, reason: d.gnn.stale ? "Model stale (>24h) — trigger retrain" : "Model aging (>12h)" },
    { severity: gnnOutcomeSev, reason: `${d.gnn.newOutcomesSinceLastTrain} new outcomes since last train — model may be operating on outdated data` },
    { severity: d.gnn.modelVersion === 0 ? 2 : 0, reason: "No model trained" },
  ]));

  health.push(healthCheck("lambda-regressor", [
    { severity: (!d.lambda.datasetSize || d.lambda.datasetSize < 5) ? 2 : 0, reason: "Insufficient training data" },
    { severity: d.lambda.r2 < 0.2 ? 2 : d.lambda.r2 < 0.5 ? 1 : 0, reason: d.lambda.r2 < 0.2 ? `Very low R²=${d.lambda.r2}` : `Low R²=${d.lambda.r2}` },
    { severity: d.lambda.rmse > 0.5 ? 1 : 0, reason: `High RMSE=${d.lambda.rmse}` },
  ]));

  const phononStabAcc = d.phononSurrogate.stabilityAccuracy;
  const phononStabSev = d.phononSurrogate.datasetSize > 0
    ? (phononStabAcc < 0.7 ? 2 : phononStabAcc < 0.85 ? 1 : 0)
    : 0;
  health.push(healthCheck("phonon-surrogate", [
    { severity: d.phononSurrogate.datasetSize === 0 ? 2 : d.phononSurrogate.datasetSize < 10 ? 1 : 0, reason: d.phononSurrogate.datasetSize === 0 ? "No training data" : "Small dataset" },
    { severity: phononStabSev, reason: phononStabAcc < 0.7 ? `Stability accuracy=${phononStabAcc} (<0.7) — near random, wastes DFT compute on unstable candidates` : `Stability accuracy=${phononStabAcc} (<0.85) — significant false-stable rate` },
  ]));

  health.push(healthCheck("tb-surrogate", [
    { severity: !d.tbSurrogate.datasetSize ? 1 : 0, reason: "No training data yet" },
    { severity: d.tbSurrogate.trainings === 0 ? 1 : 0, reason: "Never trained" },
  ]));

  health.push(healthCheck("structure-predictor", [
    { severity: d.structurePredictor.datasetSize === 0 ? 2 : 0, reason: "No training data" },
    { severity: (d.structurePredictor.crystalSystemAccuracy < 0.3 && d.structurePredictor.datasetSize > 0) ? 1 : 0, reason: `Low crystal system accuracy=${d.structurePredictor.crystalSystemAccuracy}` },
  ]));

  health.push(healthCheck("pressure-structure", [
    { severity: !d.pressureStructure.modelTrained ? 1 : 0, reason: "Model not yet trained" },
    { severity: d.pressureStructure.datasetSize === 0 ? 1 : 0, reason: "No training data" },
  ]));

  return health;
}

export function getPerFamilyBias(model?: string): FamilyBias[] {
  return computeFamilyBias(model);
}

export function getErrorAnalysis(): ErrorAnalysisReport {
  return computeErrorAnalysis();
}

export function getFeatureImportanceReport(topN: number = 25): FeatureImportanceEntry[] {
  return getGlobalFeatureImportance(topN);
}

export function getFailureSummary(): FailureSummaryReport {
  return computeFailureSummary();
}

export function getModelBenchmark(): ModelBenchmarkReport {
  return computeBenchmarkReport();
}

export function getFailedMaterialsForLLM(): string {
  const summary = computeFailureSummary();
  const lines: string[] = [];
  lines.push("=== FAILED MATERIALS REPORT ===");
  lines.push(`Total failures tracked: ${summary.totalFailures}`);
  lines.push("");

  if (summary.byReason.length > 0) {
    lines.push("Failure breakdown:");
    for (const r of summary.byReason) {
      lines.push(`  ${r.reason}: ${r.count} (${r.percentage}%)`);
    }
    lines.push("");
  }

  if (summary.predictedStableActualUnstable.length > 0) {
    lines.push(`Predicted stable but actually unstable (${summary.predictedStableActualUnstable.length} materials):`);
    for (const f of summary.predictedStableActualUnstable.slice(0, 15)) {
      const extras: string[] = [];
      if (f.imaginaryModeCount) extras.push(`imaginary_modes=${f.imaginaryModeCount}`);
      if (f.lowestPhononFreq != null) extras.push(`lowest_freq=${f.lowestPhononFreq}`);
      if (f.formationEnergy != null) extras.push(`Ef=${f.formationEnergy}`);
      lines.push(`  ${f.formula} [${f.failureReason}] via ${f.source}${extras.length > 0 ? " | " + extras.join(", ") : ""}`);
    }
    lines.push("");
  }

  if (summary.failurePatterns.topFailingElementPairs.length > 0) {
    lines.push("Top failing element pairs:");
    for (const p of summary.failurePatterns.topFailingElementPairs.slice(0, 8)) {
      lines.push(`  ${p.pair}: ${p.count}`);
    }
    lines.push("");
  }

  if (summary.llmSuggestions.length > 0) {
    lines.push("Suggested actions (sorted by impact):");
    for (const s of summary.llmSuggestions) {
      lines.push(`  - [${s.priority.toUpperCase()}] (impact=${s.impactScore}, ${s.category}) ${s.text}`);
    }
  }

  return lines.join("\n");
}

export function getBenchmarkForLLM(): string {
  const report = computeBenchmarkReport();
  const lines: string[] = [];
  lines.push("=== MODEL BENCHMARK REPORT ===");

  if (report.scorecards.length === 0) {
    lines.push("No version history available yet.");
    return lines.join("\n");
  }

  const sortedReportCards = [...report.scorecards].sort((a, b) => a.version - b.version);
  const latestByModel = new Map<string, ModelVersionScorecard[]>();
  for (const sc of sortedReportCards) {
    if (!latestByModel.has(sc.modelName)) latestByModel.set(sc.modelName, []);
    latestByModel.get(sc.modelName)!.push(sc);
  }

  for (const [model, cards] of latestByModel) {
    lines.push("");
    lines.push(`## ${model} (${cards.length} versions)`);
    const latest = cards[cards.length - 1];
    lines.push(`  Latest: v${latest.version}`);
    for (const [k, v] of Object.entries(latest.metrics)) {
      lines.push(`    ${k}: ${typeof v === "number" ? v.toFixed(4) : v}`);
    }
    lines.push(`    inference speed: ${latest.inferenceSpeedMs} ms (${latest.computeEnvironment})`);
    lines.push(`    dataset: ${latest.datasetSize} samples`);
    const hpEntries = Object.entries(latest.hyperparameters);
    const hpSummary = hpEntries.slice(0, 5).map(([k, v]) => `${k}=${v}`).join(", ");
    lines.push(`    hyperparameters: ${hpSummary}${hpEntries.length > 5 ? ` (+${hpEntries.length - 5} more)` : ""}`);
  }

  if (report.versionComparisons.length > 0) {
    lines.push("");
    lines.push("## Version Comparisons");
    for (const c of report.versionComparisons.slice(-8)) {
      const deltaStr = Object.entries(c.metricDeltas).map(([k, v]) => `${k}=${v > 0 ? "+" : ""}${v}`).join(", ");
      lines.push(`  ${c.modelName} v${c.fromVersion}->v${c.toVersion}: ${deltaStr}`);
      lines.push(`    recommendation: ${c.recommendation}`);
    }
  }

  return lines.join("\n");
}
