export interface PhysicsParameters {
  muStar: {
    screeningFactor: number;
    logRatioFloor: number;
    classBlendingRatio: number;
  };
  phononScale: {
    hydrogenDebyeScaling: number;
    maxFrequencyHeuristic: number;
    logAverageRatio: number;
  };
  anharmonicFactor: {
    gruneisenMultiplier: number;
    hFractionWeight: number;
    lambdaSuppressionCoefficient: number;
  };
  pairingWeight: {
    correlationPenaltyExponent: number;
    softModeThreshold: number;
  };
  version: number;
  lastUpdated: number;
}

const DEFAULT_PARAMETERS: PhysicsParameters = {
  muStar: {
    screeningFactor: 1.0,
    logRatioFloor: 1.1,
    classBlendingRatio: 0.5,
  },
  phononScale: {
    hydrogenDebyeScaling: 1.0,
    maxFrequencyHeuristic: 1.0,
    logAverageRatio: 0.65,
  },
  anharmonicFactor: {
    gruneisenMultiplier: 1.0,
    hFractionWeight: 1.0,
    lambdaSuppressionCoefficient: 1.0,
  },
  pairingWeight: {
    correlationPenaltyExponent: 1.5,
    softModeThreshold: 0.3,
  },
  version: 1,
  lastUpdated: Date.now(),
};

interface PredictionRecord {
  formula: string;
  predictedTc: number;
  observedTc: number;
  error: number;
  absError: number;
  features: number[];
  parameters: number[];
  timestamp: number;
}

interface ParameterSnapshot {
  version: number;
  parameters: PhysicsParameters;
  mae: number;
  rmse: number;
  nObservations: number;
  timestamp: number;
}

function parametersToVector(params: PhysicsParameters): number[] {
  return [
    params.muStar.screeningFactor,
    params.muStar.logRatioFloor,
    params.muStar.classBlendingRatio,
    params.phononScale.hydrogenDebyeScaling,
    params.phononScale.maxFrequencyHeuristic,
    params.phononScale.logAverageRatio,
    params.anharmonicFactor.gruneisenMultiplier,
    params.anharmonicFactor.hFractionWeight,
    params.anharmonicFactor.lambdaSuppressionCoefficient,
    params.pairingWeight.correlationPenaltyExponent,
    params.pairingWeight.softModeThreshold,
  ];
}

function vectorToParameters(vec: number[], base: PhysicsParameters): PhysicsParameters {
  return {
    muStar: {
      screeningFactor: clamp(vec[0], 0.1, 3.0),
      logRatioFloor: clamp(vec[1], 0.5, 5.0),
      classBlendingRatio: clamp(vec[2], 0.0, 1.0),
    },
    phononScale: {
      hydrogenDebyeScaling: clamp(vec[3], 0.1, 5.0),
      maxFrequencyHeuristic: clamp(vec[4], 0.1, 5.0),
      logAverageRatio: clamp(vec[5], 0.1, 1.0),
    },
    anharmonicFactor: {
      gruneisenMultiplier: clamp(vec[6], 0.1, 5.0),
      hFractionWeight: clamp(vec[7], 0.1, 5.0),
      lambdaSuppressionCoefficient: clamp(vec[8], 0.1, 5.0),
    },
    pairingWeight: {
      correlationPenaltyExponent: clamp(vec[9], 0.5, 4.0),
      softModeThreshold: clamp(vec[10], 0.05, 1.0),
    },
    version: base.version + 1,
    lastUpdated: Date.now(),
  };
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function rbfKernel(x1: number[], x2: number[], lengthScale: number): number {
  let sqDist = 0;
  for (let i = 0; i < x1.length; i++) {
    const d = (x1[i] ?? 0) - (x2[i] ?? 0);
    sqDist += d * d;
  }
  return Math.exp(-0.5 * sqDist / (lengthScale * lengthScale));
}

function choleskyDecompose(K: number[][]): number[][] {
  const n = K.length;
  let jitter = 1e-6;

  for (let attempt = 0; attempt <= 3; attempt++) {
    if (attempt > 0) {
      for (let i = 0; i < n; i++) K[i][i] += jitter;
      jitter *= 10;
    }

    const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    let failed = false;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
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

  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) L[i][i] = Math.sqrt(Math.max(K[i][i], 1e-6));
  return L;
}

function choleskySolve(L: number[][], b: number[]): number[] {
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

class BayesianParameterOptimizer {
  private lengthScale = 1.0;
  private signalVariance = 1.0;
  private noiseVariance = 0.01;

  predict(
    trainX: number[][],
    trainY: number[],
    testX: number[],
  ): { mean: number; std: number } {
    const n = trainX.length;
    if (n === 0) return { mean: 0, std: Math.sqrt(this.signalVariance) };

    const K: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const k = this.signalVariance * rbfKernel(trainX[i], trainX[j], this.lengthScale);
        K[i][j] = k;
        K[j][i] = k;
      }
      K[i][i] += this.noiseVariance;
    }

    const L = choleskyDecompose(K);
    const alpha = choleskySolve(L, trainY);

    const kStar = new Array(n);
    for (let i = 0; i < n; i++) {
      kStar[i] = this.signalVariance * rbfKernel(testX, trainX[i], this.lengthScale);
    }

    let mean = 0;
    for (let i = 0; i < n; i++) mean += kStar[i] * alpha[i];

    const v = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < i; j++) sum += L[i][j] * v[j];
      v[i] = L[i][i] > 1e-10 ? (kStar[i] - sum) / L[i][i] : 0;
    }
    let variance = this.signalVariance;
    for (let i = 0; i < v.length; i++) variance -= v[i] * v[i];
    variance = Math.max(variance, 1e-6);

    return { mean, std: Math.sqrt(variance) };
  }

  suggestNextPoint(
    trainX: number[][],
    trainY: number[],
    currentParams: number[],
    nCandidates: number = 20,
  ): number[] {
    const dim = currentParams.length;
    const candidates: number[][] = [];

    candidates.push([...currentParams]);

    for (let c = 0; c < nCandidates - 1; c++) {
      const candidate = currentParams.map((p, i) => {
        const perturbation = (Math.random() - 0.5) * 0.4 * Math.abs(p);
        return p + perturbation;
      });
      candidates.push(candidate);
    }

    let bestCandidate = candidates[0];
    let bestAcquisition = -Infinity;

    const bestY = trainY.length > 0 ? Math.min(...trainY) : 0;

    for (const candidate of candidates) {
      const { mean, std } = this.predict(trainX, trainY, candidate);
      const improvement = bestY - mean;
      if (std < 1e-8) continue;
      const z = improvement / std;
      const ei = improvement * normalCDF(z) + std * normalPDF(z);

      if (ei > bestAcquisition) {
        bestAcquisition = ei;
        bestCandidate = candidate;
      }
    }

    return bestCandidate;
  }
}

function normalCDF(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327;
  const p =
    d *
    Math.exp((-x * x) / 2) *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}

function normalPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

const MAX_PREDICTION_HISTORY = 1000;
const MAX_PARAMETER_HISTORY = 100;
const MIN_OBSERVATIONS_FOR_UPDATE = 10;
const UPDATE_INTERVAL_MS = 60_000;

let currentParameters: PhysicsParameters = { ...DEFAULT_PARAMETERS };
const predictionHistory: PredictionRecord[] = [];
const parameterHistory: ParameterSnapshot[] = [];
const bayesianOptimizer = new BayesianParameterOptimizer();
let lastUpdateTime = 0;

export function getPhysicsParameters(): PhysicsParameters {
  return { ...currentParameters };
}

export function updatePhysicsParameters(
  observedTc: number,
  predictedTc: number,
  features: number[],
  formula: string = "unknown",
): { updated: boolean; version: number; currentMAE: number } {
  const error = predictedTc - observedTc;
  const absError = Math.abs(error);

  const record: PredictionRecord = {
    formula,
    predictedTc,
    observedTc,
    error,
    absError,
    features: features.slice(0, 20),
    parameters: parametersToVector(currentParameters),
    timestamp: Date.now(),
  };

  predictionHistory.push(record);
  if (predictionHistory.length > MAX_PREDICTION_HISTORY) {
    predictionHistory.splice(0, predictionHistory.length - MAX_PREDICTION_HISTORY);
  }

  const currentMAE = computeMAE(predictionHistory);

  const shouldUpdate =
    predictionHistory.length >= MIN_OBSERVATIONS_FOR_UPDATE &&
    Date.now() - lastUpdateTime > UPDATE_INTERVAL_MS;

  if (!shouldUpdate) {
    return { updated: false, version: currentParameters.version, currentMAE };
  }

  const recentRecords = predictionHistory.slice(-100);

  const trainX: number[][] = [];
  const trainY: number[] = [];

  for (const rec of recentRecords) {
    trainX.push(rec.parameters);
    trainY.push(rec.absError);
  }

  if (trainX.length < MIN_OBSERVATIONS_FOR_UPDATE) {
    return { updated: false, version: currentParameters.version, currentMAE };
  }

  const currentVec = parametersToVector(currentParameters);
  const suggestedVec = bayesianOptimizer.suggestNextPoint(
    trainX,
    trainY,
    currentVec,
    30,
  );

  const blendFactor = 0.3;
  const blendedVec = currentVec.map(
    (v, i) => v * (1 - blendFactor) + suggestedVec[i] * blendFactor,
  );

  const newParams = vectorToParameters(blendedVec, currentParameters);

  const snapshot: ParameterSnapshot = {
    version: newParams.version,
    parameters: { ...newParams },
    mae: currentMAE,
    rmse: computeRMSE(recentRecords),
    nObservations: predictionHistory.length,
    timestamp: Date.now(),
  };

  parameterHistory.push(snapshot);
  if (parameterHistory.length > MAX_PARAMETER_HISTORY) {
    parameterHistory.splice(0, parameterHistory.length - MAX_PARAMETER_HISTORY);
  }

  currentParameters = newParams;
  lastUpdateTime = Date.now();

  return { updated: true, version: newParams.version, currentMAE };
}

export function getParameterHistory(): ParameterSnapshot[] {
  return [...parameterHistory];
}

export interface ModelPerformance {
  currentVersion: number;
  totalPredictions: number;
  recentMAE: number;
  recentRMSE: number;
  recentR2: number;
  allTimeMAE: number;
  allTimeRMSE: number;
  parameterDrift: number;
  lastUpdated: number;
  errorTrend: "improving" | "stable" | "degrading";
  predictionErrorHistory: Array<{ timestamp: number; absError: number }>;
}

export function getModelPerformance(): ModelPerformance {
  const recent = predictionHistory.slice(-50);
  const all = predictionHistory;

  const recentMAE = computeMAE(recent);
  const recentRMSE = computeRMSE(recent);
  const recentR2 = computeR2(recent);
  const allTimeMAE = computeMAE(all);
  const allTimeRMSE = computeRMSE(all);

  let parameterDrift = 0;
  if (parameterHistory.length >= 2) {
    const first = parameterHistory[0];
    const last = parameterHistory[parameterHistory.length - 1];
    const v1 = parametersToVector(first.parameters);
    const v2 = parametersToVector(last.parameters);
    let sqDist = 0;
    for (let i = 0; i < v1.length; i++) {
      const d = v1[i] - v2[i];
      sqDist += d * d;
    }
    parameterDrift = Math.sqrt(sqDist);
  }

  let errorTrend: "improving" | "stable" | "degrading" = "stable";
  if (recent.length >= 20) {
    const firstHalf = recent.slice(0, Math.floor(recent.length / 2));
    const secondHalf = recent.slice(Math.floor(recent.length / 2));
    const maeFirst = computeMAE(firstHalf);
    const maeSecond = computeMAE(secondHalf);
    const changeRatio = maeFirst > 0 ? (maeSecond - maeFirst) / maeFirst : 0;
    if (changeRatio < -0.1) errorTrend = "improving";
    else if (changeRatio > 0.1) errorTrend = "degrading";
  }

  const errorHistory = recent.map((r) => ({
    timestamp: r.timestamp,
    absError: r.absError,
  }));

  return {
    currentVersion: currentParameters.version,
    totalPredictions: predictionHistory.length,
    recentMAE,
    recentRMSE,
    recentR2,
    allTimeMAE,
    allTimeRMSE,
    parameterDrift: Math.round(parameterDrift * 1000) / 1000,
    lastUpdated: currentParameters.lastUpdated,
    errorTrend,
    predictionErrorHistory: errorHistory,
  };
}

function computeMAE(records: PredictionRecord[]): number {
  if (records.length === 0) return 0;
  const sum = records.reduce((s, r) => s + r.absError, 0);
  return Math.round((sum / records.length) * 100) / 100;
}

function computeRMSE(records: PredictionRecord[]): number {
  if (records.length === 0) return 0;
  const sumSq = records.reduce((s, r) => s + r.error * r.error, 0);
  return Math.round(Math.sqrt(sumSq / records.length) * 100) / 100;
}

function computeR2(records: PredictionRecord[]): number {
  if (records.length < 2) return 0;
  const meanObserved =
    records.reduce((s, r) => s + r.observedTc, 0) / records.length;
  const ssTot = records.reduce(
    (s, r) => s + (r.observedTc - meanObserved) ** 2,
    0,
  );
  const ssRes = records.reduce((s, r) => s + r.error * r.error, 0);
  if (ssTot < 1e-10) return 0;
  const r2 = 1 - ssRes / ssTot;
  return Math.round(Math.max(-1, Math.min(1, r2)) * 1000) / 1000;
}
