export interface PredictionRecord {
  formula: string;
  predictedTc: number;
  actualTc: number;
  error: number;
  absError: number;
  timestamp: number;
}

export interface RollingWindowMetrics {
  mae: number;
  rmse: number;
  r2: number;
  count: number;
}

export interface PerformanceMetrics {
  totalPredictions: number;
  overall: RollingWindowMetrics;
  last50: RollingWindowMetrics;
  last100: RollingWindowMetrics;
  last500: RollingWindowMetrics;
  theoryDiscoverySuccessRate: number;
  candidateSuccessRate: number;
  surrogateModelAccuracy: number;
  parameterDrift: number;
  parameterVersion: number;
  retrainFlagged: boolean;
  retrainReason: string | null;
  errorTrend: number[];
  theoryCount: number;
  dataQuality: DataQualityMetrics;
}

export interface DataQualityMetrics {
  missingActualCount: number;
  outlierCount: number;
  coverageFraction: number;
  averageConfidence: number;
}

interface TheoryDiscoveryRecord {
  timestamp: number;
  equationCount: number;
  bestAccuracy: number;
  success: boolean;
}

interface CandidateOutcome {
  formula: string;
  passed: boolean;
  timestamp: number;
}

const MAX_HISTORY = 2000;
const ROLLING_50 = 50;
const ROLLING_100 = 100;
const ROLLING_500 = 500;
const RETRAIN_MAE_INCREASE_THRESHOLD = 0.20;
const ERROR_TREND_BUCKETS = 20;

const predictionHistory: PredictionRecord[] = [];
const theoryDiscoveryHistory: TheoryDiscoveryRecord[] = [];
const candidateOutcomes: CandidateOutcome[] = [];

let parameterVersion = 1;
let previousParameterHash = 0;
let surrogateAccuracyEstimate = 0.5;
let retrainFlagged = false;
let retrainReason: string | null = null;

function computeWindowMetrics(records: PredictionRecord[]): RollingWindowMetrics {
  if (records.length === 0) {
    return { mae: 0, rmse: 0, r2: 0, count: 0 };
  }

  const n = records.length;
  let sumAbsError = 0;
  let sumSqError = 0;
  let sumActual = 0;

  for (const r of records) {
    sumAbsError += r.absError;
    sumSqError += r.error * r.error;
    sumActual += r.actualTc;
  }

  const mae = sumAbsError / n;
  const rmse = Math.sqrt(sumSqError / n);

  const meanActual = sumActual / n;
  let ssTot = 0;
  let ssRes = 0;
  for (const r of records) {
    ssTot += (r.actualTc - meanActual) ** 2;
    ssRes += r.error * r.error;
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

  return { mae, rmse, r2, count: n };
}

function getWindow(size: number): PredictionRecord[] {
  if (predictionHistory.length <= size) return predictionHistory;
  return predictionHistory.slice(predictionHistory.length - size);
}

export function recordPrediction(formula: string, predicted: number, actual: number): void {
  const error = predicted - actual;
  const absError = Math.abs(error);

  predictionHistory.push({
    formula,
    predictedTc: predicted,
    actualTc: actual,
    error,
    absError,
    timestamp: Date.now(),
  });

  if (predictionHistory.length > MAX_HISTORY) {
    predictionHistory.splice(0, predictionHistory.length - MAX_HISTORY);
  }

  checkRetrainTrigger();
}

export function recordTheoryDiscovery(equationCount: number, bestAccuracy: number): void {
  theoryDiscoveryHistory.push({
    timestamp: Date.now(),
    equationCount,
    bestAccuracy,
    success: bestAccuracy > 0.5 && equationCount > 0,
  });

  if (theoryDiscoveryHistory.length > 200) {
    theoryDiscoveryHistory.splice(0, theoryDiscoveryHistory.length - 200);
  }
}

export function recordCandidateOutcome(formula: string, passed: boolean): void {
  candidateOutcomes.push({
    formula,
    passed,
    timestamp: Date.now(),
  });

  if (candidateOutcomes.length > 1000) {
    candidateOutcomes.splice(0, candidateOutcomes.length - 1000);
  }
}

export function updateSurrogateAccuracy(accuracy: number): void {
  surrogateAccuracyEstimate = Math.max(0, Math.min(1, accuracy));
}

export function bumpParameterVersion(newParamHash: number): void {
  parameterVersion++;
  previousParameterHash = newParamHash;
}

function checkRetrainTrigger(): void {
  if (predictionHistory.length < ROLLING_50 * 2) {
    retrainFlagged = false;
    retrainReason = null;
    return;
  }

  const recent50 = getWindow(ROLLING_50);
  const recentMAE = computeWindowMetrics(recent50).mae;

  const older = predictionHistory.slice(
    Math.max(0, predictionHistory.length - ROLLING_100),
    predictionHistory.length - ROLLING_50
  );
  if (older.length < 10) {
    retrainFlagged = false;
    retrainReason = null;
    return;
  }
  const olderMAE = computeWindowMetrics(older).mae;

  if (olderMAE > 0) {
    const increase = (recentMAE - olderMAE) / olderMAE;
    if (increase > RETRAIN_MAE_INCREASE_THRESHOLD) {
      retrainFlagged = true;
      retrainReason = `MAE increased ${(increase * 100).toFixed(1)}% over last ${ROLLING_50} evaluations (${olderMAE.toFixed(2)} -> ${recentMAE.toFixed(2)})`;
      return;
    }
  }

  const recent100 = getWindow(ROLLING_100);
  const r2_100 = computeWindowMetrics(recent100).r2;
  if (r2_100 < 0.3 && predictionHistory.length >= ROLLING_100) {
    retrainFlagged = true;
    retrainReason = `R² dropped to ${r2_100.toFixed(3)} over last ${ROLLING_100} evaluations`;
    return;
  }

  retrainFlagged = false;
  retrainReason = null;
}

function computeErrorTrend(): number[] {
  if (predictionHistory.length < ERROR_TREND_BUCKETS) {
    return predictionHistory.map(r => r.absError);
  }

  const bucketSize = Math.floor(predictionHistory.length / ERROR_TREND_BUCKETS);
  const trend: number[] = [];

  for (let i = 0; i < ERROR_TREND_BUCKETS; i++) {
    const start = i * bucketSize;
    const end = i === ERROR_TREND_BUCKETS - 1 ? predictionHistory.length : (i + 1) * bucketSize;
    const bucket = predictionHistory.slice(start, end);
    const avgErr = bucket.reduce((s, r) => s + r.absError, 0) / bucket.length;
    trend.push(Math.round(avgErr * 100) / 100);
  }

  return trend;
}

function computeParameterDrift(): number {
  if (parameterVersion <= 1) return 0;
  return Math.min(1, (parameterVersion - 1) * 0.05);
}

function computeDataQuality(): DataQualityMetrics {
  const total = predictionHistory.length;
  if (total === 0) {
    return { missingActualCount: 0, outlierCount: 0, coverageFraction: 0, averageConfidence: 0 };
  }

  let outlierCount = 0;
  const absErrors = predictionHistory.map(r => r.absError);
  const medianError = absErrors.sort((a, b) => a - b)[Math.floor(absErrors.length / 2)] || 0;
  const iqr = absErrors.length >= 4
    ? absErrors[Math.floor(absErrors.length * 0.75)] - absErrors[Math.floor(absErrors.length * 0.25)]
    : medianError;
  const outlierThreshold = medianError + 3 * Math.max(iqr, 1);

  for (const r of predictionHistory) {
    if (r.absError > outlierThreshold) outlierCount++;
  }

  const uniqueFormulas = new Set(predictionHistory.map(r => r.formula)).size;
  const coverageFraction = Math.min(1, uniqueFormulas / Math.max(total, 1));

  const recentMetrics = computeWindowMetrics(getWindow(ROLLING_100));
  const averageConfidence = Math.max(0, Math.min(1, recentMetrics.r2));

  return {
    missingActualCount: 0,
    outlierCount,
    coverageFraction,
    averageConfidence,
  };
}

export function getPerformanceMetrics(): PerformanceMetrics {
  const overall = computeWindowMetrics(predictionHistory);
  const last50 = computeWindowMetrics(getWindow(ROLLING_50));
  const last100 = computeWindowMetrics(getWindow(ROLLING_100));
  const last500 = computeWindowMetrics(getWindow(ROLLING_500));

  const theorySuccesses = theoryDiscoveryHistory.filter(t => t.success).length;
  const theoryDiscoverySuccessRate = theoryDiscoveryHistory.length > 0
    ? theorySuccesses / theoryDiscoveryHistory.length
    : 0;

  const candidatePassed = candidateOutcomes.filter(c => c.passed).length;
  const candidateSuccessRate = candidateOutcomes.length > 0
    ? candidatePassed / candidateOutcomes.length
    : 0;

  return {
    totalPredictions: predictionHistory.length,
    overall,
    last50,
    last100,
    last500,
    theoryDiscoverySuccessRate,
    candidateSuccessRate,
    surrogateModelAccuracy: surrogateAccuracyEstimate,
    parameterDrift: computeParameterDrift(),
    parameterVersion,
    retrainFlagged,
    retrainReason,
    errorTrend: computeErrorTrend(),
    theoryCount: theoryDiscoveryHistory.length,
    dataQuality: computeDataQuality(),
  };
}

export function shouldRetrain(): boolean {
  return retrainFlagged;
}

export function getRetrainReason(): string | null {
  return retrainReason;
}

export function clearRetrainFlag(): void {
  retrainFlagged = false;
  retrainReason = null;
}

export interface ContinuousLearningResult {
  cycleTimestamp: number;
  featureDBSize: number;
  performanceSnapshot: PerformanceMetrics;
  retrainTriggered: boolean;
  theoryDiscoveryRun: boolean;
  parametersUpdated: boolean;
  actions: string[];
}

export async function runContinuousLearningCycle(featureDB: {
  getDatasetSize: () => number;
  getFeatureDataset?: () => Array<{ materialId: string; features: Record<string, number>; tc: number }>;
}): Promise<ContinuousLearningResult> {
  const actions: string[] = [];
  const metrics = getPerformanceMetrics();
  const dbSize = featureDB.getDatasetSize();

  let retrainTriggered = false;
  if (shouldRetrain()) {
    retrainTriggered = true;
    actions.push(`Retrain triggered: ${retrainReason}`);
    clearRetrainFlag();
  }

  let theoryDiscoveryRun = false;
  if (dbSize >= 30 && predictionHistory.length % 50 === 0 && predictionHistory.length > 0) {
    theoryDiscoveryRun = true;
    actions.push(`Theory discovery eligible with ${dbSize} feature records`);
  }

  let parametersUpdated = false;
  if (metrics.last50.mae > 0 && metrics.parameterDrift < 0.8) {
    if (metrics.last50.r2 < 0.5 && predictionHistory.length >= ROLLING_50) {
      parametersUpdated = true;
      actions.push(`Parameter update suggested: R²=${metrics.last50.r2.toFixed(3)} below threshold`);
    }
  }

  if (metrics.dataQuality.outlierCount > predictionHistory.length * 0.1) {
    actions.push(`Data quality warning: ${metrics.dataQuality.outlierCount} outliers detected (${(metrics.dataQuality.outlierCount / Math.max(predictionHistory.length, 1) * 100).toFixed(1)}%)`);
  }

  if (dbSize < 10) {
    actions.push(`Insufficient data: ${dbSize} records in feature DB, need at least 10 for meaningful analysis`);
  }

  return {
    cycleTimestamp: Date.now(),
    featureDBSize: dbSize,
    performanceSnapshot: metrics,
    retrainTriggered,
    theoryDiscoveryRun,
    parametersUpdated,
    actions,
  };
}

export function getPredictionHistory(limit?: number): PredictionRecord[] {
  if (limit && limit < predictionHistory.length) {
    return predictionHistory.slice(predictionHistory.length - limit);
  }
  return [...predictionHistory];
}

export function getTotalPredictionCount(): number {
  return predictionHistory.length;
}
