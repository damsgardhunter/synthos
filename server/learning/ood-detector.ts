import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { extractFeatures, getCachedFeatures, type MLFeatureVector } from "./ml-predictor";
import { normalizeFormula } from "./utils";
import { computeCompositionFeatures, compositionFeatureVector } from "./composition-features";
import { getCalibrationState } from "./conformal-calibrator";

interface OODResult {
  formula: string | null;
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
const GMM_MAX_ITER = 50;
const GMM_CONVERGENCE_THRESHOLD = 1e-4;

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

let conformalLatentThreshold = 0.3;
let conformalMahaSteepness = 3.0;
let conformalGmmSteepness = 3.0;
let conformalLatentSteepness = 6.0;

const KEY_FEATURE_INDICES = Array.from({ length: FEATURE_SUBSET_SIZE }, (_, i) => i);

function yieldToEventLoop(): Promise<void> {
  return new Promise(resolve => setImmediate(resolve));
}

function normalizeVector(raw: number[]): number[] {
  const sub = featureSubset(raw);
  const result = new Array(sub.length);
  for (let j = 0; j < sub.length; j++) {
    result[j] = trainingStd[j] > 1e-6 ? (sub[j] - trainingMean[j]) / trainingStd[j] : 0;
  }
  return result;
}

function updateConformalThresholds(): void {
  try {
    const cal = getCalibrationState();
    if (cal.calibrationDatasetSize < 10) return;

    const tempScale = Math.max(cal.temperatureScale, 0.1);
    conformalLatentThreshold = Math.min(1.0, Math.max(0.1, cal.medianNonconformityScore * tempScale));
    conformalLatentSteepness = Math.min(12, Math.max(2, cal.conformalQ95 / Math.max(cal.medianNonconformityScore, 0.01)));
    conformalMahaSteepness = Math.min(8, Math.max(1.5, 1.5 + cal.conformalQ95));
    conformalGmmSteepness = Math.min(8, Math.max(1.5, 1.5 + cal.conformalQ90));
  } catch {}
}

function featureSubset(fullFeatures: number[]): number[] {
  return KEY_FEATURE_INDICES.map(i => i < fullFeatures.length ? fullFeatures[i] : 0);
}

function extractOODFeatureVectorSync(formula: string): number[] {
  const normKey = normalizeFormula(formula);
  const features = getCachedFeatures(normKey);
  if (!features) {
    return new Array(FEATURE_SUBSET_SIZE).fill(0);
  }
  return buildOODVector(features);
}

async function extractOODFeatureVectorAsync(formula: string): Promise<number[]> {
  const features = await extractFeatures(formula);
  return buildOODVector(features);
}

function buildOODVector(features: any): number[] {
  const raw = [
    features.electronPhononLambda,
    features.metallicity,
    features.logPhononFreq,
    features.debyeTemperature,
    features.correlationStrength,
    features.valenceElectronConcentration,
    features.avgElectronegativity,
    features.enSpread,
    features.hydrogenRatio,
    features.avgAtomicRadius,
    features.avgBulkModulus,
    features.numElements,
    features.cooperPairStrength,
    features.dimensionalityScore,
    features.meissnerPotential,
    features.dosAtEF,
    features.orbitalDFraction,
    features.nestingScore,
    features.vanHoveProximity,
    features.bandFlatness,
    features.softModeScore,
    features.connectivityIndex,
    features.phononSpectralCentroid,
    features.phononSpectralWidth,
    features.bondStiffnessVariance,
    features.chargeTransferMagnitude,
    features.pettiforNumber,
    features.maxAtomicMass,
    features.avgSommerfeldGamma,
    features.electronDensityEstimate,
  ];
  const result = new Array(FEATURE_SUBSET_SIZE).fill(0);
  for (let i = 0; i < FEATURE_SUBSET_SIZE; i++) {
    const v = i < raw.length ? raw[i] : undefined;
    result[i] = (v !== undefined && v !== null && Number.isFinite(v)) ? v : 0;
  }
  return result;
}

function computeMahalanobis(x: number[]): number {
  if (trainingMean.length === 0) return 0;
  const sub = featureSubset(x);
  let dist2 = 0;
  const dim = Math.min(sub.length, trainingMean.length, trainingInvVar.length);
  for (let i = 0; i < dim; i++) {
    const s = Number.isFinite(sub[i]) ? sub[i] : 0;
    const diff = s - trainingMean[i];
    const iv = Number.isFinite(trainingInvVar[i]) ? trainingInvVar[i] : 1;
    dist2 += diff * diff * iv;
  }
  return Number.isFinite(dist2) ? Math.sqrt(Math.max(0, dist2)) : 0;
}

function computeGMMLogLikelihood(x: number[]): number {
  if (gmmModel.length === 0) return 0;
  const sub = featureSubset(x);
  const logProbs: number[] = [];

  for (const comp of gmmModel) {
    const dim = Math.min(sub.length, comp.mean.length, comp.precision.length);
    let logP = Math.log(Math.max(comp.weight, 1e-10));
    logP -= 0.5 * comp.logDetCov;
    logP -= 0.5 * dim * Math.log(2 * Math.PI);
    let mahal = 0;
    for (let i = 0; i < dim; i++) {
      const s = Number.isFinite(sub[i]) ? sub[i] : 0;
      const diff = s - comp.mean[i];
      const p = Number.isFinite(comp.precision[i]) ? comp.precision[i] : 1;
      mahal += diff * diff * p;
    }
    if (!Number.isFinite(mahal)) mahal = 0;
    logP -= 0.5 * mahal;
    logProbs.push(logP);
  }

  const maxLog = Math.max(...logProbs);
  const sumExp = logProbs.reduce((s, lp) => s + Math.exp(lp - maxLog), 0);
  return maxLog + Math.log(sumExp);
}

function percentileRank(value: number, sorted: number[]): number {
  if (sorted.length === 0) return 0.5;
  let below = 0;
  let equal = 0;
  for (const s of sorted) {
    if (s < value) below++;
    else if (s === value) equal++;
    else break;
  }
  return (below + 0.5 * equal) / sorted.length;
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

  let prevLogLikelihood = -Infinity;
  let converged = false;

  for (let iter = 0; iter < GMM_MAX_ITER; iter++) {
    const responsibilities: number[][] = [];
    let totalLogLikelihood = 0;

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
      totalLogLikelihood += maxLog + Math.log(sumExp);
      responsibilities.push(expProbs.map(e => e / sumExp));
    }

    if (iter > 0) {
      const delta = Math.abs(totalLogLikelihood - prevLogLikelihood);
      const relDelta = delta / (Math.abs(prevLogLikelihood) + 1e-10);
      if (relDelta < GMM_CONVERGENCE_THRESHOLD) {
        converged = true;
        break;
      }
    }
    prevLogLikelihood = totalLogLikelihood;

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
        const rawV = newVar[j] / nk;
        const v = (Number.isFinite(rawV) && rawV > 1e-6) ? rawV : 1e-6;
        precision[j] = 1 / v;
        logDet += Math.log(v);
      }

      components[k].mean = newMean;
      components[k].precision = precision;
      components[k].weight = nk / n;
      components[k].logDetCov = logDet;
    }
  }

  if (!converged) {
    console.warn(`[OOD] GMM did not converge after ${GMM_MAX_ITER} iterations (last LL=${prevLogLikelihood.toFixed(2)})`);
  }

  return components;
}

const YIELD_BATCH_SIZE = 50;
let oodModelReady = false;

export async function updateOODModel(): Promise<void> {
  const trainingVectors: number[][] = [];

  const step = Math.max(1, Math.floor(SUPERCON_TRAINING_DATA.length / MAX_TRAINING_SAMPLES));
  let extracted = 0;
  for (let i = 0; i < SUPERCON_TRAINING_DATA.length; i += step) {
    try {
      const vec = await extractOODFeatureVectorAsync(SUPERCON_TRAINING_DATA[i].formula);
      if (vec.every(v => Number.isFinite(v))) {
        trainingVectors.push(vec);
      }
    } catch {}
    extracted++;
    if (extracted % YIELD_BATCH_SIZE === 0) {
      await yieldToEventLoop();
    }
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

  await yieldToEventLoop();

  const normalizedVectors = trainingVectors.map(vec => normalizeVector(vec));

  gmmModel = fitGMM(normalizedVectors);

  await yieldToEventLoop();

  trainingMahalanobisDistances = trainingVectors.map(v => computeMahalanobis(v));
  trainingMahalanobisDistances.sort((a, b) => a - b);
  mahalanobisP50 = getPercentile(trainingMahalanobisDistances, 0.50);
  mahalanobisP95 = getPercentile(trainingMahalanobisDistances, 0.95);
  mahalanobisP99 = getPercentile(trainingMahalanobisDistances, 0.99);

  gmmTrainingLogLikelihoods = normalizedVectors.map(v => computeGMMLogLikelihood(v));
  gmmTrainingLogLikelihoods.sort((a, b) => a - b);
  gmmP50 = getPercentile(gmmTrainingLogLikelihoods, 0.50);
  gmmP5 = getPercentile(gmmTrainingLogLikelihoods, 0.05);
  gmmP1 = getPercentile(gmmTrainingLogLikelihoods, 0.01);

  updateConformalThresholds();

  lastOODUpdate = Date.now();
  oodModelReady = true;
}

export function computeOODScore(
  formulaOrFeatures: string | number[],
  gnnLatentDist?: number,
): OODResult {
  oodQueryCount++;

  let featureVec: number[];
  if (typeof formulaOrFeatures === "string") {
    featureVec = extractOODFeatureVectorSync(formulaOrFeatures);
  } else {
    featureVec = formulaOrFeatures;
  }

  const inputFormula = typeof formulaOrFeatures === "string" ? formulaOrFeatures : null;

  if (!oodModelReady && trainingMean.length === 0) {
    return {
      formula: inputFormula,
      oodScore: 0.5,
      mahalanobisDistance: 0,
      mahalanobisPercentile: 0.5,
      gmmLogLikelihood: 0,
      gmmPercentile: 0.5,
      gnnLatentDistance: gnnLatentDist ?? 0,
      isOOD: false,
      oodSigmaPenalty: 0,
      oodCategory: "borderline",
    };
  }

  const mahaDist = computeMahalanobis(featureVec);
  const mahaPercentile = percentileRank(mahaDist, trainingMahalanobisDistances);

  const normVec = normalizeVector(featureVec);
  const gmmLL = gmmModel.length > 0 ? computeGMMLogLikelihood(normVec) : 0;
  const gmmPercentile = 1 - percentileRank(gmmLL, gmmTrainingLogLikelihoods);

  const latentDist = gnnLatentDist ?? 0;

  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
  const mahaScale = Math.max(mahalanobisP95 - mahalanobisP50, 0.1);
  const mahaScore = sigmoid(conformalMahaSteepness * (mahaDist - mahalanobisP95) / mahaScale);

  let gmmScore = 0;
  if (gmmModel.length > 0) {
    const gmmScale = Math.max(gmmP50 - gmmP5, 0.1);
    gmmScore = sigmoid(conformalGmmSteepness * (gmmP5 - gmmLL) / gmmScale);
  }

  const latentScore = sigmoid(conformalLatentSteepness * (latentDist - conformalLatentThreshold));

  const oodScore = Math.min(1, Math.max(0,
    0.4 * mahaScore + 0.35 * gmmScore + 0.25 * latentScore
  ));

  let oodCategory: OODResult["oodCategory"];
  if (oodScore < 0.1) oodCategory = "in-distribution";
  else if (oodScore < 0.3) oodCategory = "borderline";
  else if (oodScore < 0.5) oodCategory = "moderate-ood";
  else if (oodScore < 0.75) oodCategory = "strong-ood";
  else oodCategory = "extreme-ood";

  const isOOD = oodScore > 0.3;
  if (isOOD) {
    oodDetectedCount++;
    const label = inputFormula ?? "feature-vector";
    console.log(`[OOD] Flagged ${oodCategory}: "${label}" score=${oodScore.toFixed(4)} maha=${mahaDist.toFixed(2)} gmmLL=${gmmLL.toFixed(2)} latent=${latentDist.toFixed(3)} penalty=${(oodScore * OOD_PENALTY_SCALE).toFixed(4)}`);
  }

  const oodSigmaPenalty = oodScore * OOD_PENALTY_SCALE;

  return {
    formula: inputFormula,
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
