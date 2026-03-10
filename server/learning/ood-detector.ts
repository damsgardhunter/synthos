import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { extractFeatures, type MLFeatureVector } from "./ml-predictor";
import { computeCompositionFeatures, compositionFeatureVector } from "./composition-features";

interface OODResult {
  oodScore: number;
  mahalanobisDistance: number;
  mahalanobisPercentile: number;
  gmmLogLikelihood: number;
  gmmPercentile: number;
  gnnLatentDistance: number;
  isOOD: boolean;
  oodSigmaPenalty: number;
  oodCategory: "in-distribution" | "borderline" | "moderate-ood" | "strong-ood" | "extreme-ood";
}

interface OODStats {
  trainingSetSize: number;
  featureDim: number;
  gmmComponents: number;
  mahalanobisThreshold95: number;
  mahalanobisThreshold99: number;
  mahalanobisMedian: number;
  gmmThreshold5: number;
  gmmThreshold1: number;
  gmmMedian: number;
  lastUpdated: number;
  oodQueryCount: number;
  oodDetectedCount: number;
}

interface GMMComponent {
  mean: number[];
  precision: number[];
  weight: number;
  logDetCov: number;
}

const GMM_COMPONENTS = 5;
const OOD_PENALTY_SCALE = 2.0;
const MAX_TRAINING_SAMPLES = 500;
const FEATURE_SUBSET_SIZE = 30;

let trainingMean: number[] = [];
let trainingInvVar: number[] = [];
let trainingStd: number[] = [];
let trainingMahalanobisDistances: number[] = [];
let mahalanobisP50 = 5;
let mahalanobisP95 = 10;
let mahalanobisP99 = 20;

let gmmModel: GMMComponent[] = [];
let gmmTrainingLogLikelihoods: number[] = [];
let gmmP50 = -20;
let gmmP5 = -100;
let gmmP1 = -200;

let lastOODUpdate = 0;
let oodQueryCount = 0;
let oodDetectedCount = 0;
let featureDim = 0;

const KEY_FEATURE_INDICES = Array.from({ length: FEATURE_SUBSET_SIZE }, (_, i) => i);

function featureSubset(fullFeatures: number[]): number[] {
  return KEY_FEATURE_INDICES.map(i => i < fullFeatures.length ? fullFeatures[i] : 0);
}

function extractOODFeatureVector(formula: string): number[] {
  const features = extractFeatures(formula);
  const physics = [
    features.electronPhononLambda ?? 0,
    features.metallicity ?? 0,
    features.logPhononFreq ?? 0,
    features.debyeTemperature ?? 0,
    features.correlationStrength ?? 0,
    features.valenceElectronConcentration ?? 0,
    features.avgElectronegativity ?? 0,
    features.enSpread ?? 0,
    features.hydrogenRatio ?? 0,
    features.avgAtomicRadius ?? 0,
    features.avgBulkModulus ?? 0,
    features.numElements ?? 0,
    features.cooperPairStrength ?? 0,
    features.dimensionalityScore ?? 0,
    features.meissnerPotential ?? 0,
    features.dosAtEF ?? 0,
    features.orbitalDFraction ?? 0,
    features.nestingScore ?? 0,
    features.vanHoveProximity ?? 0,
    features.bandFlatness ?? 0,
    features.softModeScore ?? 0,
    features.connectivityIndex ?? 0,
    features.phononSpectralCentroid ?? 0,
    features.phononSpectralWidth ?? 0,
    features.bondStiffnessVariance ?? 0,
    features.chargeTransferMagnitude ?? 0,
    features.pettiforNumber ?? 0,
    features.maxAtomicMass ?? 0,
    features.avgSommerfeldGamma ?? 0,
    features.electronDensityEstimate ?? 0,
  ];
  return physics.map(v => Number.isFinite(v) ? v : 0);
}

function computeMahalanobis(x: number[]): number {
  if (trainingMean.length === 0) return 0;
  const sub = featureSubset(x);
  let dist2 = 0;
  for (let i = 0; i < sub.length && i < trainingMean.length; i++) {
    const diff = sub[i] - trainingMean[i];
    dist2 += diff * diff * trainingInvVar[i];
  }
  return Math.sqrt(Math.max(0, dist2));
}

function computeGMMLogLikelihood(x: number[]): number {
  if (gmmModel.length === 0) return 0;
  const sub = featureSubset(x);
  const logProbs: number[] = [];

  for (const comp of gmmModel) {
    let logP = Math.log(Math.max(comp.weight, 1e-10));
    logP -= 0.5 * comp.logDetCov;
    logP -= 0.5 * sub.length * Math.log(2 * Math.PI);
    let mahal = 0;
    for (let i = 0; i < sub.length && i < comp.mean.length; i++) {
      const diff = sub[i] - comp.mean[i];
      mahal += diff * diff * comp.precision[i];
    }
    logP -= 0.5 * mahal;
    logProbs.push(logP);
  }

  const maxLog = Math.max(...logProbs);
  const sumExp = logProbs.reduce((s, lp) => s + Math.exp(lp - maxLog), 0);
  return maxLog + Math.log(sumExp);
}

function percentileRank(value: number, sorted: number[]): number {
  if (sorted.length === 0) return 0.5;
  let count = 0;
  for (const s of sorted) {
    if (value <= s) break;
    count++;
  }
  return count / sorted.length;
}

function getPercentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(Math.floor(p * sorted.length), sorted.length - 1);
  return sorted[idx];
}

function fitGMM(data: number[][]): GMMComponent[] {
  const n = data.length;
  const d = data[0]?.length ?? 0;
  if (n < GMM_COMPONENTS * 2 || d === 0) return [];

  const K = GMM_COMPONENTS;
  const components: GMMComponent[] = [];

  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort(() => Math.random() - 0.5);

  for (let k = 0; k < K; k++) {
    const centerIdx = indices[k % n];
    components.push({
      mean: [...data[centerIdx]],
      precision: new Array(d).fill(1),
      weight: 1 / K,
      logDetCov: 0,
    });
  }

  for (let iter = 0; iter < 20; iter++) {
    const responsibilities: number[][] = [];
    for (let i = 0; i < n; i++) {
      const logProbs: number[] = [];
      for (let k = 0; k < K; k++) {
        let logP = Math.log(Math.max(components[k].weight, 1e-10));
        let mahal = 0;
        for (let j = 0; j < d; j++) {
          const diff = data[i][j] - components[k].mean[j];
          mahal += diff * diff * components[k].precision[j];
        }
        logP -= 0.5 * mahal;
        logP -= 0.5 * components[k].logDetCov;
        logProbs.push(logP);
      }
      const maxLog = Math.max(...logProbs);
      const expProbs = logProbs.map(lp => Math.exp(lp - maxLog));
      const sumExp = expProbs.reduce((s, v) => s + v, 0);
      responsibilities.push(expProbs.map(e => e / sumExp));
    }

    for (let k = 0; k < K; k++) {
      let nk = 0;
      const newMean = new Array(d).fill(0);
      for (let i = 0; i < n; i++) {
        nk += responsibilities[i][k];
        for (let j = 0; j < d; j++) {
          newMean[j] += responsibilities[i][k] * data[i][j];
        }
      }
      nk = Math.max(nk, 1e-6);
      for (let j = 0; j < d; j++) newMean[j] /= nk;

      const newVar = new Array(d).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
          const diff = data[i][j] - newMean[j];
          newVar[j] += responsibilities[i][k] * diff * diff;
        }
      }
      let logDet = 0;
      const precision = new Array(d).fill(0);
      for (let j = 0; j < d; j++) {
        const v = Math.max(newVar[j] / nk, 1e-6);
        precision[j] = 1 / v;
        logDet += Math.log(v);
      }

      components[k].mean = newMean;
      components[k].precision = precision;
      components[k].weight = nk / n;
      components[k].logDetCov = logDet;
    }
  }

  return components;
}

export function updateOODModel(): void {
  const trainingVectors: number[][] = [];

  const step = Math.max(1, Math.floor(SUPERCON_TRAINING_DATA.length / MAX_TRAINING_SAMPLES));
  for (let i = 0; i < SUPERCON_TRAINING_DATA.length; i += step) {
    try {
      const vec = extractOODFeatureVector(SUPERCON_TRAINING_DATA[i].formula);
      if (vec.every(v => Number.isFinite(v))) {
        trainingVectors.push(vec);
      }
    } catch {}
  }

  if (trainingVectors.length < 10) return;

  featureDim = trainingVectors[0].length;
  const n = trainingVectors.length;
  const d = Math.min(featureDim, FEATURE_SUBSET_SIZE);

  trainingMean = new Array(d).fill(0);
  trainingStd = new Array(d).fill(0);
  trainingInvVar = new Array(d).fill(0);

  for (const vec of trainingVectors) {
    const sub = featureSubset(vec);
    for (let j = 0; j < d; j++) trainingMean[j] += sub[j];
  }
  for (let j = 0; j < d; j++) trainingMean[j] /= n;

  for (const vec of trainingVectors) {
    const sub = featureSubset(vec);
    for (let j = 0; j < d; j++) {
      const diff = sub[j] - trainingMean[j];
      trainingStd[j] += diff * diff;
    }
  }
  for (let j = 0; j < d; j++) {
    trainingStd[j] = Math.sqrt(trainingStd[j] / n);
    const variance = Math.max(trainingStd[j] ** 2, 1e-6);
    trainingInvVar[j] = 1 / variance;
  }

  const normalizedVectors = trainingVectors.map(vec => {
    const sub = featureSubset(vec);
    return sub.map((v, j) => trainingStd[j] > 1e-6 ? (v - trainingMean[j]) / trainingStd[j] : 0);
  });

  gmmModel = fitGMM(normalizedVectors);

  trainingMahalanobisDistances = trainingVectors.map(v => computeMahalanobis(v));
  trainingMahalanobisDistances.sort((a, b) => a - b);
  mahalanobisP50 = getPercentile(trainingMahalanobisDistances, 0.50);
  mahalanobisP95 = getPercentile(trainingMahalanobisDistances, 0.95);
  mahalanobisP99 = getPercentile(trainingMahalanobisDistances, 0.99);

  gmmTrainingLogLikelihoods = normalizedVectors.map(v => {
    const sub = featureSubset(v);
    return computeGMMLogLikelihood(sub);
  });
  gmmTrainingLogLikelihoods.sort((a, b) => a - b);
  gmmP50 = getPercentile(gmmTrainingLogLikelihoods, 0.50);
  gmmP5 = getPercentile(gmmTrainingLogLikelihoods, 0.05);
  gmmP1 = getPercentile(gmmTrainingLogLikelihoods, 0.01);

  lastOODUpdate = Date.now();
}

export function computeOODScore(
  formulaOrFeatures: string | number[],
  gnnLatentDist?: number,
): OODResult {
  oodQueryCount++;

  let featureVec: number[];
  if (typeof formulaOrFeatures === "string") {
    featureVec = extractOODFeatureVector(formulaOrFeatures);
  } else {
    featureVec = formulaOrFeatures;
  }

  if (trainingMean.length === 0) {
    updateOODModel();
  }

  const mahaDist = computeMahalanobis(featureVec);
  const mahaPercentile = percentileRank(mahaDist, trainingMahalanobisDistances);

  const sub = featureSubset(featureVec);
  const normSub = sub.map((v, j) =>
    trainingStd[j] > 1e-6 ? (v - trainingMean[j]) / trainingStd[j] : 0
  );
  const gmmLL = gmmModel.length > 0 ? computeGMMLogLikelihood(normSub) : 0;
  const gmmPercentile = 1 - percentileRank(gmmLL, gmmTrainingLogLikelihoods);

  const latentDist = gnnLatentDist ?? 0;

  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
  const mahaScale = Math.max(mahalanobisP95 - mahalanobisP50, 0.1);
  const mahaScore = sigmoid(3 * (mahaDist - mahalanobisP95) / mahaScale);

  let gmmScore = 0;
  if (gmmModel.length > 0) {
    const gmmScale = Math.max(gmmP50 - gmmP5, 0.1);
    gmmScore = sigmoid(3 * (gmmP5 - gmmLL) / gmmScale);
  }

  const latentScore = sigmoid(6 * (latentDist - 0.3));

  const oodScore = Math.min(1, Math.max(0,
    0.4 * mahaScore + 0.35 * gmmScore + 0.25 * latentScore
  ));

  const isOOD = oodScore > 0.3;
  if (isOOD) oodDetectedCount++;

  const oodSigmaPenalty = oodScore * OOD_PENALTY_SCALE;

  let oodCategory: OODResult["oodCategory"];
  if (oodScore < 0.1) oodCategory = "in-distribution";
  else if (oodScore < 0.3) oodCategory = "borderline";
  else if (oodScore < 0.5) oodCategory = "moderate-ood";
  else if (oodScore < 0.75) oodCategory = "strong-ood";
  else oodCategory = "extreme-ood";

  return {
    oodScore: Math.round(oodScore * 10000) / 10000,
    mahalanobisDistance: Math.round(mahaDist * 1000) / 1000,
    mahalanobisPercentile: Math.round(mahaPercentile * 10000) / 10000,
    gmmLogLikelihood: Math.round(gmmLL * 100) / 100,
    gmmPercentile: Math.round(gmmPercentile * 10000) / 10000,
    gnnLatentDistance: Math.round(latentDist * 1000) / 1000,
    isOOD,
    oodSigmaPenalty: Math.round(oodSigmaPenalty * 10000) / 10000,
    oodCategory,
  };
}

export function getOODStats(): OODStats {
  return {
    trainingSetSize: trainingMahalanobisDistances.length,
    featureDim,
    gmmComponents: gmmModel.length,
    mahalanobisThreshold95: Math.round(mahalanobisP95 * 1000) / 1000,
    mahalanobisThreshold99: Math.round(mahalanobisP99 * 1000) / 1000,
    mahalanobisMedian: Math.round(mahalanobisP50 * 1000) / 1000,
    gmmThreshold5: Math.round(gmmP5 * 100) / 100,
    gmmThreshold1: Math.round(gmmP1 * 100) / 100,
    gmmMedian: Math.round(gmmP50 * 100) / 100,
    lastUpdated: lastOODUpdate,
    oodQueryCount,
    oodDetectedCount,
  };
}

export function getOODSummaryForLLM(): string {
  const stats = getOODStats();
  return `OOD Detector: ${stats.trainingSetSize} training samples, ${stats.gmmComponents}-component GMM, Mahalanobis P95=${stats.mahalanobisThreshold95}, P99=${stats.mahalanobisThreshold99}. Queries: ${stats.oodQueryCount} total, ${stats.oodDetectedCount} flagged OOD (${stats.oodQueryCount > 0 ? Math.round(100 * stats.oodDetectedCount / stats.oodQueryCount) : 0}%).`;
}
