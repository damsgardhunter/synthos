import { SUPERCON_TRAINING_DATA, type SuperconEntry } from "./supercon-dataset";
import { getExtendedTrainingData } from "./extended-dataset";
import { extractFeatures, type MLFeatureVector } from "./ml-predictor";
import { computeMiedemaFormationEnergy } from "./phase-diagram-engine";
import { computeCompositionFeatures, compositionFeatureVector, COMPOSITION_FEATURE_NAMES } from "./composition-features";
import { classifyFamily } from "./utils";
import { storage } from "../storage";
import { systemMetrics } from "../../shared/schema";
import { db } from "../db";

interface HyperparamOverrides {
  nTrees?: number;
  learningRate?: number;
  maxDepth?: number;
  minSamples?: number;
  ensembleSize?: number;
  dropoutRate?: number;
  layerCount?: number;
  regularizationL2?: number;
  batchSize?: number;
  epochs?: number;
  bootstrapRatio?: number;
}

let xgboostHyperparamResolver: (() => HyperparamOverrides | undefined) | null = null;

export function registerHyperparamResolver(resolver: () => HyperparamOverrides | undefined): void {
  xgboostHyperparamResolver = resolver;
}

function getXGBoostHyperparamOverrides(): HyperparamOverrides {
  if (xgboostHyperparamResolver) {
    return xgboostHyperparamResolver() ?? {};
  }
  return {};
}

interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | number;
  right: TreeNode | number;
}

interface FlatTreeNode {
  featureIndex: number;
  threshold: number;
  leftChild: number;
  rightChild: number;
}

interface FlatTree {
  nodes: FlatTreeNode[];
  leafValues: number[];
}

function flattenTree(tree: TreeNode | number): FlatTree {
  const nodes: FlatTreeNode[] = [];
  const leafValues: number[] = [];

  function visit(node: TreeNode | number): number {
    if (typeof node === "number") {
      const leafIdx = -(leafValues.length + 1);
      leafValues.push(node);
      return leafIdx;
    }
    const idx = nodes.length;
    nodes.push({ featureIndex: node.featureIndex, threshold: node.threshold, leftChild: 0, rightChild: 0 });
    nodes[idx].leftChild = visit(node.left);
    nodes[idx].rightChild = visit(node.right);
    return idx;
  }
  visit(tree);
  return { nodes, leafValues };
}

export function predictFlat(flat: FlatTree, x: number[]): number {
  let idx = 0;
  while (idx >= 0) {
    const node = flat.nodes[idx];
    idx = x[node.featureIndex] <= node.threshold ? node.leftChild : node.rightChild;
  }
  return flat.leafValues[-(idx + 1)];
}

interface GBModel {
  trees: TreeNode[];
  flatTrees: FlatTree[];
  learningRate: number;
  basePrediction: number;
  featureNames: string[];
  featureMask?: number[];
  trainedAt: number;
  logTransformed?: boolean; // targets trained with log1p(tc), predictions need expm1
}

/** Predict a single Tc value from a GBModel (used by eval-harness CV). */
export function gbPredictFromModel(model: GBModel, x: number[]): number {
  let pred = model.basePrediction;
  for (const flat of model.flatTrees) {
    pred += model.learningRate * predictFlat(flat, x);
  }
  return model.logTransformed ? Math.expm1(Math.max(0, pred)) : Math.max(0, pred);
}

let cachedModel: GBModel | null = null;

const XGB_ENSEMBLE_SIZE = 5;
const BOOTSTRAP_SAMPLE_RATIO = 0.8;

interface GBEnsemble {
  models: GBModel[];
  trainedAt: number;
  isLogVariance?: boolean;
}

let cachedEnsembleXGB: GBEnsemble | null = null;
let cachedVarianceEnsembleXGB: GBEnsemble | null = null;
let cachedGlobalFeatureImportance: { name: string; index: number; importance: number; normalizedImportance: number }[] | null = null;

let curiosityProvider: (() => number) | null = null;

export function setCuriosityProvider(provider: () => number): void {
  curiosityProvider = provider;
}

// Formulas confirmed by a real DFPT run — receive 5× effective sample weight during XGB
// retraining by oversampling their (X, y) rows in the training matrix.
const dftVerifiedFormulas = new Set<string>();

export function registerDFTVerifiedFormula(formula: string): void {
  dftVerifiedFormulas.add(formula);
}

function getCuriosityMultiplier(): number {
  if (curiosityProvider) return curiosityProvider();
  return 1.5;
}

function bootstrapSample(X: number[][], y: number[], ratio: number = BOOTSTRAP_SAMPLE_RATIO): { X: number[][]; y: number[] } {
  const n = X.length;
  const sampleSize = Math.floor(n * ratio);
  for (let attempt = 0; attempt < 5; attempt++) {
    const sampledX: number[][] = [];
    const sampledY: number[] = [];
    for (let i = 0; i < sampleSize; i++) {
      const idx = Math.floor(Math.random() * n);
      sampledX.push(X[idx]);
      sampledY.push(y[idx]);
    }
    const uniqueY = new Set(sampledY.map(v => Math.round(v * 10)));
    if (uniqueY.size > 1 || n < 5) return { X: sampledX, y: sampledY };
  }
  const sampledX: number[][] = [];
  const sampledY: number[] = [];
  for (let i = 0; i < sampleSize; i++) {
    const idx = Math.floor(Math.random() * n);
    sampledX.push(X[idx]);
    sampledY.push(y[idx]);
  }
  return { X: sampledX, y: sampledY };
}

const ENSEMBLE_MAX_DEPTHS = [5, 6, 7, 6, 5];
const ENSEMBLE_LEARNING_RATES = [0.04, 0.05, 0.06, 0.05, 0.04];

const FEATURE_SUBSAMPLE_RATIO = 0.7;

async function trainEnsembleXGB(X: number[][], y: number[]): Promise<GBEnsemble> {
  const overrides = getXGBoostHyperparamOverrides();
  // When GCP handles full GNN training, use a lighter local surrogate to avoid blocking the event loop
  const isOffloaded = process.env.OFFLOAD_GNN_TO_GCP === "true";
  const models: GBModel[] = [];
  for (let i = 0; i < XGB_ENSEMBLE_SIZE; i++) {
    const { X: bsX, y: bsY } = bootstrapSample(X, y);
    const depth = overrides.maxDepth ?? ENSEMBLE_MAX_DEPTHS[i % ENSEMBLE_MAX_DEPTHS.length];
    const lr = overrides.learningRate ?? ENSEMBLE_LEARNING_RATES[i % ENSEMBLE_LEARNING_RATES.length];
    const nTrees = overrides.nTrees ?? (isOffloaded ? 100 : 300);
    const model = await trainGradientBoosting(bsX, bsY, nTrees, lr, depth, FEATURE_SUBSAMPLE_RATIO);
    models.push(model);
    await new Promise<void>(r => setTimeout(r, 0)); // yield between each XGB model
  }
  return { models, trainedAt: Date.now() };
}

async function trainVarianceEnsembleXGB(X: number[][], y: number[], meanEnsemble: GBEnsemble): Promise<GBEnsemble> {
  const LOG_EPSILON = 1e-6;
  const logSquaredResiduals: number[] = [];
  for (let i = 0; i < X.length; i++) {
    const meanPred = predictEnsembleXGB(meanEnsemble, X[i]).mean;
    const residual = y[i] - meanPred;
    logSquaredResiduals.push(Math.log(residual * residual + LOG_EPSILON));
  }

  const overrides = getXGBoostHyperparamOverrides();
  const isOffloaded = process.env.OFFLOAD_GNN_TO_GCP === "true";
  const models: GBModel[] = [];
  for (let i = 0; i < XGB_ENSEMBLE_SIZE; i++) {
    const { X: bsX, y: bsY } = bootstrapSample(X, logSquaredResiduals);
    const depth = overrides.maxDepth ?? ENSEMBLE_MAX_DEPTHS[i % ENSEMBLE_MAX_DEPTHS.length];
    const lr = overrides.learningRate ?? ENSEMBLE_LEARNING_RATES[i % ENSEMBLE_LEARNING_RATES.length];
    const nTrees = overrides.nTrees ?? (isOffloaded ? 70 : 200);
    const model = await trainGradientBoosting(bsX, bsY, nTrees, lr, depth, FEATURE_SUBSAMPLE_RATIO);
    models.push(model);
    await new Promise<void>(r => setTimeout(r, 0)); // yield between each variance model
  }
  return { models, trainedAt: Date.now(), isLogVariance: true };
}

function predictEnsembleXGB(ensemble: GBEnsemble, x: number[]): { mean: number; std: number; predictions: number[] } {
  const predictions = ensemble.models.map(m => predictWithModel(m, x));
  const mean = predictions.reduce((s, v) => s + v, 0) / predictions.length;
  const variance = predictions.reduce((s, v) => s + (v - mean) ** 2, 0) / predictions.length;
  return { mean: Math.max(0, mean), std: Math.sqrt(variance), predictions };
}

interface CalibrationData {
  r2: number;
  mae: number;
  mse: number;
  rmse: number;
  nSamples: number;
  nTrees: number;
  residuals: number[];
  percentiles: { p5: number; p10: number; p25: number; p50: number; p75: number; p90: number; p95: number };
  absResidualPercentiles: { p50: number; p75: number; p90: number; p95: number };
  relativeErrorPercentiles: { p50: number; p75: number; p90: number; p95: number };
  predictedVsActual: { formula: string; actual: number; predicted: number; residual: number }[];
  computedAt: number;
}

let cachedCalibration: CalibrationData | null = null;

function computePercentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (idx - lower) * (sorted[upper] - sorted[lower]);
}

async function computeCalibration(model: GBModel): Promise<CalibrationData> {
  const { X, y, formulas } = await prepareTrainingData();
  const details: { formula: string; actual: number; predicted: number; residual: number }[] = [];
  const residuals: number[] = [];
  const actualTcs: number[] = [];
  let sse = 0;
  let totalAbsError = 0;

  for (let i = 0; i < X.length; i++) {
    const pred = predictWithModel(model, X[i]);
    const residual = y[i] - pred;
    residuals.push(residual);
    actualTcs.push(y[i]);
    details.push({ formula: formulas[i], actual: y[i], predicted: Math.round(pred * 10) / 10, residual: Math.round(residual * 10) / 10 });
    sse += residual ** 2;
    totalAbsError += Math.abs(residual);
  }

  const n = details.length;
  const meanTc = n > 0 ? actualTcs.reduce((s, v) => s + v, 0) / n : 0;
  const sst = actualTcs.reduce((s, v) => s + (v - meanTc) ** 2, 0);
  const mse = n > 0 ? sse / n : 0;
  const mae = n > 0 ? totalAbsError / n : 0;
  const r2 = sst < 1e-6 ? 0 : 1 - sse / sst;
  const rmse = Math.sqrt(mse);

  const sortedResiduals = [...residuals].sort((a, b) => a - b);
  const absResiduals = residuals.map(r => Math.abs(r)).sort((a, b) => a - b);

  const relativeErrors: number[] = [];
  for (let i = 0; i < residuals.length; i++) {
    const denom = Math.max(5, actualTcs[i]);
    relativeErrors.push(Math.abs(residuals[i]) / denom);
  }
  relativeErrors.sort((a, b) => a - b);

  return {
    r2: Math.round(r2 * 10000) / 10000,
    mae: Math.round(mae * 100) / 100,
    mse: Math.round(mse * 100) / 100,
    rmse: Math.round(rmse * 100) / 100,
    nSamples: n,
    nTrees: model.trees.length,
    residuals,
    percentiles: {
      p5: Math.round(computePercentile(sortedResiduals, 5) * 100) / 100,
      p10: Math.round(computePercentile(sortedResiduals, 10) * 100) / 100,
      p25: Math.round(computePercentile(sortedResiduals, 25) * 100) / 100,
      p50: Math.round(computePercentile(sortedResiduals, 50) * 100) / 100,
      p75: Math.round(computePercentile(sortedResiduals, 75) * 100) / 100,
      p90: Math.round(computePercentile(sortedResiduals, 90) * 100) / 100,
      p95: Math.round(computePercentile(sortedResiduals, 95) * 100) / 100,
    },
    absResidualPercentiles: {
      p50: Math.round(computePercentile(absResiduals, 50) * 100) / 100,
      p75: Math.round(computePercentile(absResiduals, 75) * 100) / 100,
      p90: Math.round(computePercentile(absResiduals, 90) * 100) / 100,
      p95: Math.round(computePercentile(absResiduals, 95) * 100) / 100,
    },
    relativeErrorPercentiles: {
      p50: Math.round(computePercentile(relativeErrors, 50) * 10000) / 10000,
      p75: Math.round(computePercentile(relativeErrors, 75) * 10000) / 10000,
      p90: Math.round(computePercentile(relativeErrors, 90) * 10000) / 10000,
      p95: Math.round(computePercentile(relativeErrors, 95) * 10000) / 10000,
    },
    predictedVsActual: details,
    computedAt: Date.now(),
  };
}

const STATIC_FEATURE_MEANS: Record<string, number> = {
  electronPhononLambda: 0.5,
  metallicity: 0.5,
  logPhononFreq: 200,
  debyeTemperature: 300,
  correlationStrength: 0.3,
  valenceElectronConcentration: 4,
  avgElectronegativity: 2.0,
  enSpread: 1.0,
  hydrogenRatio: 0.0,
  pettiforNumber: 60,
  avgAtomicRadius: 130,
  avgSommerfeldGamma: 2.0,
  avgBulkModulus: 100,
  maxAtomicMass: 50,
  numElements: 2,
  cooperPairStrength: 0.5,
  dimensionalityScore: 0.5,
  electronDensityEstimate: 0.5,
  phononCouplingEstimate: 0.5,
  meissnerPotential: 0.5,
  dftConfidence: 0.5,
  orbitalCharacterCode: 0.5,
  phononSpectralCentroid: 0.5,
  phononSpectralWidth: 0.5,
  bondStiffnessVariance: 0.5,
  chargeTransferMagnitude: 0.5,
  connectivityIndex: 0.5,
  nestingScore: 0.3,
  vanHoveProximity: 0.3,
  bandFlatness: 0.3,
  softModeScore: 0.3,
  motifScore: 0.3,
  orbitalDFraction: 0.3,
  mottProximityScore: 0.3,
  topologicalBandScore: 0.3,
  dimensionalityScoreV2: 0.3,
  phononSofteningIndex: 0.3,
  spinFluctuationStrength: 0.3,
  fermiSurfaceNestingScore: 0.3,
  dosAtEF: 1.0,
  muStarEstimate: 0.13,
  pressureGpa: 0,
  optimalPressureGpa: 0,
  lambdaProxy: 0,
  alphaCouplingStrength: 0,
  phononHardness: 0.5,
  massEnhancement: 1.5,
  couplingAsymmetry: 1.0,
  bandGap: 0,
  formationEnergy: 0,
  stability: 0.5,
};

let computedFeatureMeans: Record<string, number> | null = null;
let featureMeansComputedAt = 0;
const FEATURE_MEANS_RECOMPUTE_INTERVAL_MS = 3600_000;

function getFeatureMeans(): Record<string, number> {
  if (computedFeatureMeans) {
    return computedFeatureMeans;
  }
  return STATIC_FEATURE_MEANS;
}

async function precomputeFeatureMeans(): Promise<void> {
  const now = Date.now();
  if (computedFeatureMeans && now - featureMeansComputedAt < FEATURE_MEANS_RECOMPUTE_INTERVAL_MS) {
    return;
  }
  try {
    const sums: Record<string, number> = {};
    const counts: Record<string, number> = {};
    for (const key of Object.keys(STATIC_FEATURE_MEANS)) {
      sums[key] = 0;
      counts[key] = 0;
    }
    let processed = 0;
    for (let i = 0; i < SUPERCON_TRAINING_DATA.length; i++) {
      const entry = SUPERCON_TRAINING_DATA[i];
      try {
        const features = await extractFeatures(entry.formula, { pressureGpa: entry.pressureGPa ?? 0 } as any);
        const fAny = features as any;
        for (const key of Object.keys(STATIC_FEATURE_MEANS)) {
          const v = fAny[key];
          if (v != null && Number.isFinite(v)) {
            sums[key] += v;
            counts[key]++;
          }
        }
        processed++;
      } catch {
        continue;
      }
      // Yield after every entry with setTimeout (not setImmediate) so timer callbacks
      // (heartbeat, keepalive) can fire. setImmediate starves timers in long chains.
      await new Promise<void>(r => setTimeout(r, 0));
    }
    if (processed < 10) {
      computedFeatureMeans = { ...STATIC_FEATURE_MEANS };
    } else {
      computedFeatureMeans = { ...STATIC_FEATURE_MEANS };
      for (const key of Object.keys(STATIC_FEATURE_MEANS)) {
        if (counts[key] > 10) {
          computedFeatureMeans[key] = sums[key] / counts[key];
        }
      }
    }
    featureMeansComputedAt = now;
    console.log(`[GradientBoost] Recomputed feature means from ${processed} training entries`);
  } catch {
    if (!computedFeatureMeans) computedFeatureMeans = { ...STATIC_FEATURE_MEANS };
  }
}

const FEATURE_MEANS = new Proxy(STATIC_FEATURE_MEANS, {
  get(_target, prop: string) {
    if (computedFeatureMeans) return computedFeatureMeans[prop];
    return STATIC_FEATURE_MEANS[prop];
  },
  has(_target, prop: string) {
    return prop in STATIC_FEATURE_MEANS;
  },
  ownKeys(_target) {
    return Object.keys(STATIC_FEATURE_MEANS);
  },
  getOwnPropertyDescriptor(_target, prop: string) {
    if (prop in STATIC_FEATURE_MEANS) {
      const v = computedFeatureMeans ? computedFeatureMeans[prop] : STATIC_FEATURE_MEANS[prop];
      return { configurable: true, enumerable: true, value: v };
    }
    return undefined;
  }
});

function sanitize(v: number | undefined | null, fallback?: number): number {
  if (v == null || !Number.isFinite(v)) return fallback ?? 0;
  return v;
}

function deriveMultiBandScore(f: MLFeatureVector): number {
  let score = 0;
  if ((f.nestingScore ?? 0) > 0.3) score += 0.3;
  if ((f.fermiSurfaceNestingScore ?? 0) > 0.3) score += 0.2;
  if ((f.dosAtEF ?? 0) > 2.0) score += 0.2;
  if (f.hasTransitionMetal && f.numElements >= 3) score += 0.15;
  if ((f.orbitalDFraction ?? 0) > 0.3) score += 0.15;
  return Math.min(1, score);
}

const NON_CENTROSYMMETRIC_SG_NUMBERS = new Set([
  1,
  3, 4, 5, 6, 7, 8, 9,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
  75, 76, 77, 78, 79, 80, 81, 82, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
  143, 144, 145, 146, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
  168, 169, 170, 171, 172, 173, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
  195, 196, 197, 198, 199, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
]);

const NON_CENTRO_HM_SYMBOLS = new Set([
  "P1", "P2", "P21", "C2", "Pm", "Pc", "Cm", "Cc",
  "P222", "P2221", "P21212", "P212121", "C2221", "C222", "F222", "I222", "I212121",
  "Pmm2", "Pmc21", "Pcc2", "Pma2", "Pca21", "Pnc2", "Pmn21", "Pba2", "Pna21", "Pnn2",
  "Cmm2", "Cmc21", "Ccc2", "Amm2", "Aem2", "Ama2", "Aea2",
  "Fmm2", "Fdd2", "Imm2", "Iba2", "Ima2",
  "P4", "P41", "P42", "P43", "I4", "I41",
  "P-4", "I-4",
  "P422", "P4212", "P4122", "P41212", "P4222", "P42212", "P4322", "P43212", "I422", "I4122",
  "P4mm", "P4bm", "P42cm", "P42nm", "P4cc", "P4nc", "P42mc", "P42bc",
  "I4mm", "I4cm", "I41md", "I41cd",
  "P-42m", "P-42c", "P-421m", "P-421c", "P-4m2", "P-4c2", "P-4b2", "P-4n2",
  "I-4m2", "I-4c2", "I-42m", "I-42d",
  "P3", "P31", "P32", "R3",
  "P312", "P321", "P3112", "P3121", "P3212", "P3221", "R32",
  "P3m1", "P31m", "P3c1", "P31c", "R3m", "R3c",
  "P6", "P61", "P65", "P62", "P64", "P63",
  "P622", "P6122", "P6522", "P6222", "P6422", "P6322",
  "P6mm", "P6cc", "P63cm", "P63mc",
  "P-6m2", "P-6c2", "P-62m", "P-62c",
  "P23", "F23", "I23", "P213", "I213",
  "P432", "P4232", "F432", "F4132", "I432", "P4332", "P4132", "I4132",
  "P-43m", "F-43m", "I-43m", "P-43n", "F-43c", "I-43d",
]);

function deriveNonCentrosymmetric(f: MLFeatureVector): number {
  const sym = (f as any).crystalSymmetry;
  if (!sym || typeof sym !== "string") return 0;
  const sg = sym.trim();

  const sgNumMatch = sg.match(/^(\d+)$/);
  if (sgNumMatch) {
    return NON_CENTROSYMMETRIC_SG_NUMBERS.has(parseInt(sgNumMatch[1], 10)) ? 1 : 0;
  }

  const sgNumPrefix = sg.match(/^(\d+)\s/);
  if (sgNumPrefix) {
    return NON_CENTROSYMMETRIC_SG_NUMBERS.has(parseInt(sgNumPrefix[1], 10)) ? 1 : 0;
  }

  if (NON_CENTRO_HM_SYMBOLS.has(sg)) return 1;

  const sgClean = sg.replace(/\s+/g, "");
  if (NON_CENTRO_HM_SYMBOLS.has(sgClean)) return 1;

  const normalized = sg.toLowerCase();
  if (normalized.includes("non-centrosymmetric") || normalized.includes("noncentro")) return 1;
  if (normalized.includes("polar") || normalized.includes("chiral")) return 1;
  return 0;
}

const miedemaCache = new Map<string, number>();

function getCachedMiedemaEnergy(formula: string): number {
  const cached = miedemaCache.get(formula);
  if (cached !== undefined) return cached;
  try {
    const e = computeMiedemaFormationEnergy(formula);
    const val = Number.isFinite(e) ? e : 0;
    if (miedemaCache.size > 10000) {
      const firstKey = miedemaCache.keys().next().value;
      if (firstKey !== undefined) miedemaCache.delete(firstKey);
    }
    miedemaCache.set(formula, val);
    return val;
  } catch {
    miedemaCache.set(formula, 0);
    return 0;
  }
}

const STATIC_CRYSTAL_SYM_ENCODING: Record<string, number> = {
  cubic: 35.0, hexagonal: 28.0, trigonal: 25.0,
  tetragonal: 40.0, orthorhombic: 30.0,
  monoclinic: 22.0, triclinic: 18.0,
};

let crystalSymTargetEncoding: Map<string, number> | null = null;

async function precomputeCrystalSymTargetEncoding(): Promise<void> {
  if (crystalSymTargetEncoding) return;
  crystalSymTargetEncoding = new Map();
  const symTcSums = new Map<string, { sum: number; count: number }>();
  let globalSum = 0;
  let globalCount = 0;
  for (let i = 0; i < SUPERCON_TRAINING_DATA.length; i++) {
    const entry = SUPERCON_TRAINING_DATA[i];
    try {
      const features = await extractFeatures(entry.formula, { pressureGpa: entry.pressureGPa ?? 0 } as any);
      const entrySym = (features as any).crystalSymmetry;
      if (!entrySym || typeof entrySym !== "string") continue;
      const normalized = entrySym.toLowerCase().trim();
      const SG_MAP: Record<string, string> = {
        cubic: "cubic", hexagonal: "hexagonal", trigonal: "trigonal",
        tetragonal: "tetragonal", orthorhombic: "orthorhombic",
        monoclinic: "monoclinic", triclinic: "triclinic", rhombohedral: "trigonal",
      };
      let category: string | null = null;
      if (SG_MAP[normalized]) {
        category = SG_MAP[normalized];
      } else {
        for (const [key, val] of Object.entries(SG_MAP)) {
          if (normalized.includes(key)) { category = val; break; }
        }
      }
      if (!category) continue;
      const existing = symTcSums.get(category) || { sum: 0, count: 0 };
      existing.sum += entry.tc;
      existing.count++;
      symTcSums.set(category, existing);
      globalSum += entry.tc;
      globalCount++;
    } catch { continue; }
    // Yield after every entry with setTimeout so heartbeat timers can fire
    await new Promise<void>(r => setTimeout(r, 0));
  }
  const globalMean = globalCount > 0 ? globalSum / globalCount : 20;
  const SMOOTHING = 10;
  for (const [cat, data] of Array.from(symTcSums.entries())) {
    const smoothed = (data.sum + SMOOTHING * globalMean) / (data.count + SMOOTHING);
    crystalSymTargetEncoding.set(cat, smoothed);
  }
  console.log(`[GradientBoost] Crystal symmetry target encoding: ${Array.from(crystalSymTargetEncoding.entries()).map(([k, v]) => `${k}=${v.toFixed(1)}K`).join(", ")}`);
}

function getCrystalSymTargetEncoded(sym: string): number | null {
  if (!crystalSymTargetEncoding) return null;
  const normalized = sym.toLowerCase().trim();
  const SG_LOOKUP: Record<string, string> = {
    cubic: "cubic", hexagonal: "hexagonal", trigonal: "trigonal",
    tetragonal: "tetragonal", orthorhombic: "orthorhombic",
    monoclinic: "monoclinic", triclinic: "triclinic", rhombohedral: "trigonal",
  };
  let category: string | null = null;
  if (SG_LOOKUP[normalized]) {
    category = SG_LOOKUP[normalized];
  } else {
    for (const [key, val] of Object.entries(SG_LOOKUP)) {
      if (normalized.includes(key)) { category = val; break; }
    }
  }
  if (!category) return null;
  if (crystalSymTargetEncoding) return crystalSymTargetEncoding.get(category) ?? null;
  return STATIC_CRYSTAL_SYM_ENCODING[category] ?? null;
}

function featureVectorToArray(f: MLFeatureVector, formula?: string): number[] {
  const resolvedFormula = formula || f._sourceFormula;
  const miedemaEnergy = resolvedFormula ? getCachedMiedemaEnergy(resolvedFormula) : 0;

  let pressureGpa = sanitize(f.pressureGpa, FEATURE_MEANS.pressureGpa);
  if (pressureGpa === 0 && f.hasHydrogen && f.hydrogenRatio > 0.3) {
    const maxMass = sanitize(f.maxAtomicMass, 50);
    const massScale = maxMass >= 130 ? 0.75 : maxMass >= 80 ? 0.85 : maxMass >= 40 ? 1.0 : 1.15;
    if (f.hydrogenRatio >= 8) pressureGpa = Math.round(200 * massScale);
    else if (f.hydrogenRatio >= 6) pressureGpa = Math.round(150 * massScale);
    else if (f.hydrogenRatio >= 4) pressureGpa = Math.round(100 * massScale);
  }

  const physicsFeatures = [
    sanitize(f.electronPhononLambda, FEATURE_MEANS.electronPhononLambda),
    sanitize(f.metallicity, FEATURE_MEANS.metallicity),
    sanitize(f.logPhononFreq, FEATURE_MEANS.logPhononFreq),
    sanitize(f.debyeTemperature, FEATURE_MEANS.debyeTemperature),
    sanitize(f.correlationStrength, FEATURE_MEANS.correlationStrength),
    sanitize(f.valenceElectronConcentration, FEATURE_MEANS.valenceElectronConcentration),
    sanitize(f.avgElectronegativity, FEATURE_MEANS.avgElectronegativity),
    sanitize(f.enSpread, FEATURE_MEANS.enSpread),
    sanitize(f.hydrogenRatio, FEATURE_MEANS.hydrogenRatio),
    sanitize(f.pettiforNumber, FEATURE_MEANS.pettiforNumber),
    sanitize(f.avgAtomicRadius, FEATURE_MEANS.avgAtomicRadius),
    sanitize(f.avgSommerfeldGamma, FEATURE_MEANS.avgSommerfeldGamma),
    sanitize(f.avgBulkModulus, FEATURE_MEANS.avgBulkModulus),
    sanitize(f.maxAtomicMass, FEATURE_MEANS.maxAtomicMass),
    sanitize(f.numElements, FEATURE_MEANS.numElements),
    f.hasTransitionMetal ? 1 : 0,
    f.hasRareEarth ? 1 : 0,
    f.hasHydrogen ? 1 : 0,
    f.hasChalcogen ? 1 : 0,
    f.hasPnictogen ? 1 : 0,
    sanitize(f.cooperPairStrength, FEATURE_MEANS.cooperPairStrength),
    sanitize(f.dimensionalityScore, FEATURE_MEANS.dimensionalityScore),
    sanitize(f.anharmonicityScore, 0.3),
    sanitize(f.electronDensityEstimate, FEATURE_MEANS.electronDensityEstimate),
    sanitize(f.phononCouplingEstimate, FEATURE_MEANS.phononCouplingEstimate),
    f.dWaveSymmetry ? 1 : 0,
    sanitize(f.meissnerPotential, FEATURE_MEANS.meissnerPotential),
    sanitize(f.dftConfidence, FEATURE_MEANS.dftConfidence),
    sanitize(f.orbitalCharacterCode, FEATURE_MEANS.orbitalCharacterCode),
    sanitize(f.phononSpectralCentroid, FEATURE_MEANS.phononSpectralCentroid),
    sanitize(f.phononSpectralWidth, FEATURE_MEANS.phononSpectralWidth),
    sanitize(f.bondStiffnessVariance, FEATURE_MEANS.bondStiffnessVariance),
    sanitize(f.chargeTransferMagnitude, FEATURE_MEANS.chargeTransferMagnitude),
    sanitize(f.connectivityIndex, FEATURE_MEANS.connectivityIndex),
    sanitize(f.nestingScore, FEATURE_MEANS.nestingScore),
    sanitize(f.vanHoveProximity, FEATURE_MEANS.vanHoveProximity),
    sanitize(f.bandFlatness, FEATURE_MEANS.bandFlatness),
    sanitize(f.softModeScore, FEATURE_MEANS.softModeScore),
    sanitize(f.motifScore, FEATURE_MEANS.motifScore),
    sanitize(f.orbitalDFraction, FEATURE_MEANS.orbitalDFraction),
    sanitize(f.mottProximityScore, FEATURE_MEANS.mottProximityScore),
    sanitize(f.topologicalBandScore, FEATURE_MEANS.topologicalBandScore),
    sanitize(f.dimensionalityScoreV2, FEATURE_MEANS.dimensionalityScoreV2),
    sanitize(f.phononSofteningIndex, FEATURE_MEANS.phononSofteningIndex),
    sanitize(f.spinFluctuationStrength, FEATURE_MEANS.spinFluctuationStrength),
    sanitize(f.fermiSurfaceNestingScore, FEATURE_MEANS.fermiSurfaceNestingScore),
    sanitize(f.dosAtEF, FEATURE_MEANS.dosAtEF),
    sanitize(f.muStarEstimate, FEATURE_MEANS.muStarEstimate),
    pressureGpa,
    sanitize(f.optimalPressureGpa, FEATURE_MEANS.optimalPressureGpa),
    sanitize(f.lambdaProxy, FEATURE_MEANS.lambdaProxy),
    sanitize(f.alphaCouplingStrength, FEATURE_MEANS.alphaCouplingStrength),
    sanitize(f.phononHardness, FEATURE_MEANS.phononHardness),
    sanitize(f.massEnhancement, FEATURE_MEANS.massEnhancement),
    sanitize(f.couplingAsymmetry, FEATURE_MEANS.couplingAsymmetry),
    sanitize((f as any).bandGap, FEATURE_MEANS.bandGap),
    sanitize((f as any).formationEnergy, FEATURE_MEANS.formationEnergy),
    sanitize((f as any).stability, FEATURE_MEANS.stability),
    (() => {
      const sym = (f as any).crystalSymmetry;
      if (!sym || typeof sym !== "string") return 0;
      const targetEncoded = getCrystalSymTargetEncoded(sym);
      if (targetEncoded !== null) return targetEncoded;
      return 0;
    })(),
    deriveMultiBandScore(f),
    sanitize(miedemaEnergy, 0),
    deriveNonCentrosymmetric(f),
    sanitize((f as any).dosAtEF_tb, 0),
    sanitize((f as any).bandFlatness_tb, 0),
    sanitize((f as any).lambdaProxy_tb, 0),
    sanitize((f as any).disorderVacancyFraction, 0),
    sanitize((f as any).disorderBondVariance, 0),
    sanitize((f as any).disorderLatticeStrain, 0),
    sanitize((f as any).disorderSiteMixingEntropy, 0),
    sanitize((f as any).disorderConfigEntropy, 0),
    sanitize((f as any).disorderDosSignal, 0),
    sanitize((f as any).familyBoride, 0),
    sanitize((f as any).familyA15, 0),
    sanitize((f as any).familyPnictide, 0),
    sanitize((f as any).familyChalcogenide, 0),
    sanitize((f as any).familyHeavyFermion, 0),
  ];

  let compFeatures: number[] = [];
  if (resolvedFormula) {
    try {
      const cf = computeCompositionFeatures(resolvedFormula);
      compFeatures = compositionFeatureVector(cf).map(v => Number.isFinite(v) ? v : 0);
    } catch {
      compFeatures = new Array(COMPOSITION_FEATURE_NAMES.length).fill(0);
    }
  } else {
    compFeatures = new Array(COMPOSITION_FEATURE_NAMES.length).fill(0);
  }

  return [...physicsFeatures, ...compFeatures];
}

const PHYSICS_FEATURE_NAMES = [
  "lambda", "metallicity", "omegaLog", "debyeTemp", "correlation",
  "VEC", "avgEN", "enSpread", "hRatio", "pettifor",
  "atomicRadius", "sommerfeldGamma", "bulkModulus", "maxMass", "nElements",
  "hasTM", "hasRE", "hasH", "hasChalcogen", "hasPnictogen",
  "cooperPair", "dimensionality", "anharmonic", "electronDensity",
  "phononCoupling", "dWave", "meissner", "dftConfidence",
  "orbitalChar", "phononCentroid", "phononWidth", "bondStiffVar",
  "chargeTransfer", "connectivity",
  "nestingScore", "vanHoveProx", "bandFlatness", "softModeScore",
  "motifScore", "orbitalDFrac", "mottProx", "topoScore", "dimScoreV2",
  "phononSoftening", "spinFluc", "fsNesting", "dosEF", "muStar",
  "pressureGpa", "optimalPressure",
  "lambdaProxy", "alphaCouplingStrength", "phononHardness", "massEnhancement", "couplingAsymmetry",
  "bandGap", "formationEnergy", "stability", "crystalSymmetry",
  "multiBandScore", "miedemaFormEnergy", "nonCentrosymmetric",
  "dosAtEF_tb", "bandFlatness_tb", "lambdaProxy_tb",
  "disorderVacancyFrac", "disorderBondVar", "disorderLatticeStrain",
  "disorderSiteMixEntropy", "disorderConfigEntropy", "disorderDosSignal",
  "familyBoride", "familyA15", "familyPnictide", "familyChalcogenide", "familyHeavyFermion",
];

const FEATURE_NAMES = [...PHYSICS_FEATURE_NAMES, ...COMPOSITION_FEATURE_NAMES];

const splitSortBuf = {
  sortIdx: new Int32Array(0),
  vals: new Float64Array(0),
  res: new Float64Array(0),
};

function ensureSplitBuffers(n: number): void {
  if (splitSortBuf.sortIdx.length < n) {
    const cap = Math.max(n, 1024);
    splitSortBuf.sortIdx = new Int32Array(cap);
    splitSortBuf.vals = new Float64Array(cap);
    splitSortBuf.res = new Float64Array(cap);
  }
}

function findBestSplitForSubset(
  X: number[][],
  residuals: number[],
  indices: number[],
  featureIndex: number,
  minSamples: number = 2
): { threshold: number; improvement: number; leftIndices: number[]; rightIndices: number[] } {
  const n = indices.length;
  ensureSplitBuffers(n);
  const { sortIdx, vals, res } = splitSortBuf;

  let totalSum = 0;
  for (let i = 0; i < n; i++) {
    const idx = indices[i];
    sortIdx[i] = idx;
    vals[i] = X[idx][featureIndex];
    res[i] = residuals[idx];
    totalSum += residuals[idx];
  }

  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => vals[a] - vals[b]);
  const tmpVals = new Float64Array(n);
  const tmpRes = new Float64Array(n);
  const tmpIdx = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    tmpVals[i] = vals[order[i]];
    tmpRes[i] = res[order[i]];
    tmpIdx[i] = sortIdx[order[i]];
  }
  for (let i = 0; i < n; i++) {
    vals[i] = tmpVals[i];
    res[i] = tmpRes[i];
    sortIdx[i] = tmpIdx[i];
  }

  const totalMeanSq = (totalSum * totalSum) / n;
  let bestImprovement = -Infinity;
  let bestThreshold = 0;
  let bestSplitPos = 0;
  let leftSum = 0;

  for (let i = 0; i < n - 1; i++) {
    leftSum += res[i];
    const leftCount = i + 1;
    const rightCount = n - leftCount;
    const rightSum = totalSum - leftSum;

    if (vals[i] === vals[i + 1]) continue;
    if (leftCount < minSamples || rightCount < minSamples) continue;

    const splitScore = (leftSum * leftSum) / leftCount + (rightSum * rightSum) / rightCount;
    const improvement = splitScore - totalMeanSq;

    if (improvement > bestImprovement) {
      bestImprovement = improvement;
      bestThreshold = (vals[i] + vals[i + 1]) / 2;
      bestSplitPos = i + 1;
    }
  }

  const leftIndices: number[] = new Array(bestSplitPos);
  const rightIndices: number[] = new Array(n - bestSplitPos);
  for (let i = 0; i < bestSplitPos; i++) leftIndices[i] = sortIdx[i];
  for (let i = bestSplitPos; i < n; i++) rightIndices[i - bestSplitPos] = sortIdx[i];

  return {
    threshold: bestThreshold,
    improvement: bestImprovement,
    leftIndices,
    rightIndices,
  };
}

const MAX_TREE_DEPTH = 10;

function leafValue(residuals: number[], indices: number[]): number {
  if (indices.length === 0) return 0;
  return indices.reduce((s, i) => s + residuals[i], 0) / indices.length;
}

async function buildTree(
  X: number[][],
  residuals: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamples: number,
  yieldCounter: { n: number } = { n: 0 }
): Promise<TreeNode | number> {
  // Yield every 300 recursive calls so the event loop can handle HTTP requests
  // between subtree builds. At depth 6-7 with large datasets this prevents
  // 5-15s event-loop freezes.
  yieldCounter.n++;
  if (yieldCounter.n % 300 === 0) {
    await new Promise<void>(r => setTimeout(r, 0));
  }

  const clampedMaxDepth = Math.min(maxDepth, MAX_TREE_DEPTH);
  if (depth >= clampedMaxDepth || indices.length < minSamples) {
    return leafValue(residuals, indices);
  }

  const nFeatures = X[0].length;
  let bestFeature = -1;
  let bestImprovement = -Infinity;
  let bestThreshold = 0;
  let bestLeftIdx: number[] = [];
  let bestRightIdx: number[] = [];

  const maxFeaturesPerSplit = Math.min(nFeatures, Math.max(8, Math.ceil(Math.sqrt(nFeatures))));
  const featureIndices: number[] = [];
  if (maxFeaturesPerSplit >= nFeatures) {
    for (let i = 0; i < nFeatures; i++) featureIndices.push(i);
  } else {
    const all = Array.from({ length: nFeatures }, (_, i) => i);
    for (let i = all.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [all[i], all[j]] = [all[j], all[i]];
    }
    for (let i = 0; i < maxFeaturesPerSplit; i++) featureIndices.push(all[i]);
  }

  for (const fi of featureIndices) {
    const split = findBestSplitForSubset(X, residuals, indices, fi, minSamples);
    if (split.improvement > 0 && split.improvement > bestImprovement && split.leftIndices.length >= minSamples && split.rightIndices.length >= minSamples) {
      bestImprovement = split.improvement;
      bestFeature = fi;
      bestThreshold = split.threshold;
      bestLeftIdx = split.leftIndices;
      bestRightIdx = split.rightIndices;
    }
  }

  if (bestFeature === -1) {
    return leafValue(residuals, indices);
  }

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: await buildTree(X, residuals, bestLeftIdx, depth + 1, clampedMaxDepth, minSamples, yieldCounter),
    right: await buildTree(X, residuals, bestRightIdx, depth + 1, clampedMaxDepth, minSamples, yieldCounter),
  };
}

function predictTree(tree: TreeNode | number, x: number[]): number {
  if (typeof tree === "number") return tree;
  if (x[tree.featureIndex] <= tree.threshold) {
    return predictTree(tree.left, x);
  }
  return predictTree(tree.right, x);
}

export async function trainGradientBoosting(
  X: number[][],
  y: number[],
  nEstimators: number = 200,
  learningRate: number = 0.1,
  maxDepth: number = 4,
  featureSubsampleRatio: number = 1.0
): GBModel {
  const n = X.length;
  const nFeatures = X[0]?.length ?? 0;

  let featureMask: number[] | undefined;
  let projX = X;
  if (featureSubsampleRatio < 1.0 && nFeatures > 5) {
    const keepCount = Math.max(5, Math.floor(nFeatures * featureSubsampleRatio));
    const allFeatureIdx = Array.from({ length: nFeatures }, (_, i) => i);
    for (let i = allFeatureIdx.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allFeatureIdx[i], allFeatureIdx[j]] = [allFeatureIdx[j], allFeatureIdx[i]];
    }
    featureMask = allFeatureIdx.slice(0, keepCount).sort((a, b) => a - b);
    projX = X.map(row => featureMask!.map(fi => row[fi]));
  }

  const valSize = Math.max(2, Math.floor(n * 0.15));
  const shuffledIndices = Array.from({ length: n }, (_, i) => i);
  for (let i = shuffledIndices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]];
  }
  const valIdxArr = shuffledIndices.slice(0, valSize);
  const trainIndices = shuffledIndices.slice(valSize);

  const trainX = trainIndices.map(i => projX[i]);
  // Log-transform targets: Tc spans 0–300K with median ~0.5K (heavily skewed).
  // log1p compresses the range and lets trees learn the exponential lambda→Tc relationship.
  const trainY = trainIndices.map(i => Math.log1p(Math.max(0, y[i])));
  const valX = valIdxArr.map(i => projX[i]);
  const valY = valIdxArr.map(i => Math.log1p(Math.max(0, y[i])));

  const nTrain = trainX.length;
  const allTrainIndices = Array.from({ length: nTrain }, (_, i) => i);

  const basePrediction = trainY.reduce((s, v) => s + v, 0) / nTrain;
  const trainPredictions = new Array(nTrain).fill(basePrediction);
  const valPredictions = new Array(valX.length).fill(basePrediction);
  const trees: TreeNode[] = [];

  const yVariance = trainY.reduce((s, yi) => s + (yi - basePrediction) ** 2, 0) / nTrain;
  const RESIDUAL_EPSILON = 1e-6;
  const relativeConvergenceThreshold = Math.max(RESIDUAL_EPSILON, 0.001 * yVariance);

  let bestValMSE = Infinity;
  let valIncreaseCount = 0;
  const MAX_VAL_INCREASE = 3;
  let prevTrainMSE = yVariance;

  for (let iter = 0; iter < nEstimators; iter++) {
    if (iter > 0 && iter % 10 === 0) {
      await new Promise<void>(r => setTimeout(r, 0)); // yield every 10 trees — setTimeout not setImmediate (setImmediate starves timer callbacks)
      if (iter % 20 === 0) process.stdout.write(`\r[GradientBoost] Training... tree ${iter}/${nEstimators} (${X.length} samples)`);
    }

    const residuals = trainY.map((yi, i) => yi - trainPredictions[i]);

    const residualMSE = residuals.reduce((s, r) => s + r * r, 0) / nTrain;
    if (residualMSE < RESIDUAL_EPSILON) break;

    const tree = await buildTree(trainX, residuals, allTrainIndices, 0, maxDepth, 12);
    if (typeof tree === "number") break;

    trees.push(tree);

    for (let i = 0; i < nTrain; i++) {
      trainPredictions[i] += learningRate * predictTree(tree, trainX[i]);
    }

    const trainMSE = trainY.reduce((s, yi, i) => s + (yi - trainPredictions[i]) ** 2, 0) / nTrain;
    if (trainMSE < relativeConvergenceThreshold) break;

    const mseImprovement = prevTrainMSE - trainMSE;
    if (mseImprovement >= 0 && mseImprovement < RESIDUAL_EPSILON * 10) break;
    prevTrainMSE = trainMSE;

    for (let i = 0; i < valX.length; i++) {
      valPredictions[i] += learningRate * predictTree(tree, valX[i]);
    }
    const valMSE = valY.reduce((s, yi, i) => s + (yi - valPredictions[i]) ** 2, 0) / valY.length;

    if (valMSE < bestValMSE) {
      bestValMSE = valMSE;
      valIncreaseCount = 0;
    } else {
      valIncreaseCount++;
      if (valIncreaseCount >= MAX_VAL_INCREASE) break;
    }
  }

  return {
    trees,
    flatTrees: trees.map(t => flattenTree(t)),
    learningRate,
    basePrediction,
    featureNames: FEATURE_NAMES,
    featureMask,
    trainedAt: Date.now(),
    logTransformed: true,
  };
}

function predictWithModel(model: GBModel, x: number[]): number {
  const px = model.featureMask ? model.featureMask.map(fi => x[fi]) : x;
  let prediction = model.basePrediction;
  if (model.flatTrees && model.flatTrees.length > 0) {
    for (const flat of model.flatTrees) {
      if (flat.nodes.length === 0) continue;
      const treeVal = predictFlat(flat, px);
      if (!Number.isFinite(treeVal)) continue;
      prediction += model.learningRate * treeVal;
    }
  } else {
    for (const tree of model.trees) {
      const treeVal = predictTree(tree, px);
      if (!Number.isFinite(treeVal)) continue;
      prediction += model.learningRate * treeVal;
    }
  }
  if (!Number.isFinite(prediction)) return model.logTransformed ? Math.expm1(Math.max(0, model.basePrediction)) : model.basePrediction;
  // Un-transform log1p(Tc) → Tc
  return model.logTransformed ? Math.max(0, Math.expm1(prediction)) : Math.max(0, prediction);
}

function getTreeFeatureImportance(tree: TreeNode | number): Map<number, number> {
  const imp = new Map<number, number>();
  if (typeof tree === "number") return imp;

  imp.set(tree.featureIndex, (imp.get(tree.featureIndex) || 0) + 1);
  const leftImp = getTreeFeatureImportance(tree.left);
  const rightImp = getTreeFeatureImportance(tree.right);
  for (const [k, v] of leftImp) imp.set(k, (imp.get(k) || 0) + v);
  for (const [k, v] of rightImp) imp.set(k, (imp.get(k) || 0) + v);
  return imp;
}

class TrainingPool {
  private featureCache = new Map<string, { x: number[]; tc: number }>();
  private X: number[][] = [];
  private y: number[] = [];
  private formulas: string[] = [];
  private dirty = true;

  async add(formula: string, tc: number, pressureGPa: number = 0, knownPhysics?: { lambda?: number; omegaLog?: number }): Promise<boolean> {
    const key = `${formula}|${tc}`;
    if (this.featureCache.has(key)) return false;
    try {
      const features = await extractFeatures(formula, { pressureGpa: pressureGPa } as any);
      const fArr = featureVectorToArray(features, formula);
      if (fArr.some(v => !Number.isFinite(v))) return false;
      // Override heuristic feature values with verified physics where available.
      // This is critical: extractFeatures() computes lambda from heuristics (~0.07 for MgB2),
      // but the known lambda=0.87 gives XGBoost correct signal to learn the lambda→Tc relationship.
      if (knownPhysics?.lambda != null && knownPhysics.lambda > 0) {
        const li = PHYSICS_FEATURE_NAMES.indexOf("lambda");
        if (li >= 0) fArr[li] = knownPhysics.lambda;
      }
      if (knownPhysics?.omegaLog != null && knownPhysics.omegaLog > 0) {
        const oi = PHYSICS_FEATURE_NAMES.indexOf("omegaLog");
        if (oi >= 0) fArr[oi] = knownPhysics.omegaLog;
      }
      this.featureCache.set(key, { x: fArr, tc });
      this.dirty = true;
      return true;
    } catch {
      return false;
    }
  }

  async addBatch(entries: { formula: string; tc: number; pressureGPa?: number; lambda?: number; omegaLog?: number }[]): Promise<number> {
    let added = 0;
    for (let _bi = 0; _bi < entries.length; _bi++) {
      // Yield every 20 entries — each add() calls extractFeatures (~350ms),
      // so without yields a large batch starves the event loop / timer callbacks.
      if (_bi > 0 && _bi % 20 === 0) await new Promise<void>(r => setTimeout(r, 0));
      const e = entries[_bi];
      const known = (e.lambda != null || e.omegaLog != null) ? { lambda: e.lambda, omegaLog: e.omegaLog } : undefined;
      if (await this.add(e.formula, e.tc, e.pressureGPa ?? 0, known)) added++;
    }
    return added;
  }

  getTrainingData(): { X: number[][]; y: number[]; formulas: string[] } {
    if (!this.dirty) return { X: this.X, y: this.y, formulas: this.formulas };

    this.X = [];
    this.y = [];
    this.formulas = [];
    for (const [key, cached] of this.featureCache) {
      this.X.push(cached.x);
      this.y.push(cached.tc);
      this.formulas.push(key.split("|")[0]);
    }
    this.dirty = false;
    return { X: this.X, y: this.y, formulas: this.formulas };
  }

  get size(): number {
    return this.featureCache.size;
  }

  has(formula: string, tc: number): boolean {
    return this.featureCache.has(`${formula}|${tc}`);
  }

  hasFormula(formula: string): boolean {
    for (const key of this.featureCache.keys()) {
      if (key.startsWith(formula + "|")) return true;
    }
    return false;
  }

  clear(): void {
    this.featureCache.clear();
    this.X = [];
    this.y = [];
    this.formulas = [];
    this.dirty = true;
  }
}

const trainingPool = new TrainingPool();
let poolInitialized = false;
let poolInitializing = false;
let poolInitPromise: Promise<void> | null = null;

let cachedTrainingSnapshot: { dataSize: number } | null = null;

const FALLBACK_MODEL: GBModel = {
  trees: [],
  flatTrees: [],
  learningRate: 0.1,
  basePrediction: 20,
  featureNames: FEATURE_NAMES,
  trainedAt: 0,
};

async function ensurePoolInitialized(): Promise<void> {
  if (poolInitialized && cachedTrainingSnapshot?.dataSize === SUPERCON_TRAINING_DATA.length) return;
  if (poolInitializing && poolInitPromise) {
    await poolInitPromise;
    return;
  }
  if (poolInitializing) return;
  // Pool not yet started (startPoolInit not called yet — it's deferred 75s after startup).
  // Adding ALL training data here would block for 26+ minutes (700+ entries × extractFeatures).
  // Return early and wait for the scheduled pool init to run.
  if (!poolInitialized && !poolInitializing) return;
}

export async function initPoolAsync(): Promise<void> {
  if (poolInitialized || poolInitializing) return;
  poolInitializing = true;

  // Only include confirmed superconductors (tc > 0) — non-superconductors bias Tc regression toward zero.
  // Carry known lambda/omegaLog so the feature vector uses verified values instead of heuristics.
  const allEntries = SUPERCON_TRAINING_DATA
    .filter(e => e.isSuperconductor && e.tc > 0)
    .map(e => ({ formula: e.formula, tc: e.tc, pressureGPa: e.pressureGPa ?? 0, lambda: e.lambda }));
  const startTime = Date.now();

  const PHASE1_SIZE = 15;
  const shuffled = [...allEntries].sort(() => Math.random() - 0.5);
  const phase1 = shuffled.slice(0, PHASE1_SIZE);
  const phase2 = shuffled.slice(PHASE1_SIZE);

  let added = 0;
  for (let i = 0; i < phase1.length; i++) {
    const e = phase1[i];
    if (await trainingPool.add(e.formula, e.tc, e.pressureGPa, e.lambda != null ? { lambda: e.lambda } : undefined)) added++;
    await new Promise<void>(r => setTimeout(r, 50));
  }

  poolInitialized = true;
  poolInitializing = false;
  cachedTrainingSnapshot = { dataSize: SUPERCON_TRAINING_DATA.length };
  console.log(`[TrainingPool] Phase 1: ${trainingPool.size} vectors (${added} new) in ${Date.now() - startTime}ms`);

  // Defer expensive precomputation — each function iterates over ALL SUPERCON_TRAINING_DATA
  // (~700+ entries × ~2.3s extractFeatures each = 26+ minutes of event-loop blocking).
  // Defer 10 minutes so the first learning cycles (target: 2 complete by ~2min) run undisturbed.
  // When OFFLOAD_XGB_TO_GCP=true, GCP handles full model training so local precompute is lower priority.
  setTimeout(() => {
    Promise.all([
      precomputeFeatureMeans(),
      precomputeCrystalSymTargetEncoding(),
    ]).catch(e => console.error("[GradientBoost] Precompute failed:", e));
  }, 600_000); // 10 minutes after Phase 1 completes

  await new Promise<void>(r => setTimeout(r, 50));

  if (process.env.OFFLOAD_XGB_TO_GCP !== "true") {
    try {
      const { X, y } = trainingPool.getTrainingData();
      if (X.length >= 10) {
        const trainStart = Date.now();
        console.log(`[GradientBoost] Phase 1 training — ${X.length} samples, 30 trees...`);
        cachedModel = await trainGradientBoosting(X, y, 30, 0.12, 4);
        cachedCalibration = await computeCalibration(cachedModel);
        await logModelVersion("phase1-training", X.length);
        console.log(`[GradientBoost] Phase 1 model ready in ${Date.now() - trainStart}ms`);
      }
    } catch (e) {
      console.error("[GradientBoost] Phase 1 training failed:", e);
    }
  }

  backfillPool(phase2, startTime);
}

async function backfillPool(remaining: { formula: string; tc: number; pressureGPa: number; lambda?: number }[], t0: number): Promise<void> {
  const offloadToGCP = process.env.OFFLOAD_XGB_TO_GCP === "true";
  // Always backfill the full dataset — even when offloading to GCP, we need the complete
  // feature matrix to dispatch to GCP so it trains on 500+ samples, not just the 15 from Phase 1.
  // Only the local ensemble training is skipped; dispatch happens after backfill completes.
  if (offloadToGCP) {
    console.log("[TrainingPool] OFFLOAD_XGB_TO_GCP=true — backfilling pool so full dataset is dispatched to GCP");
  }
  // Short delay to let startup I/O settle, then backfill remaining entries with event-loop yields
  await new Promise<void>(r => setTimeout(r, 30000));
  let added = 0;
  for (let i = 0; i < remaining.length; i++) {
    const e = remaining[i];
    if (await trainingPool.add(e.formula, e.tc, e.pressureGPa, e.lambda != null ? { lambda: e.lambda } : undefined)) added++;
    // Yield every entry with a real timer tick (not just setImmediate) so the
    // event loop can process keepalive pings and incoming HTTP requests between
    // each heavy extractFeatures call (~350ms each).
    await new Promise<void>(r => setTimeout(r, 0));
  }
  console.log(`[TrainingPool] Phase 2 backfill: ${trainingPool.size} total vectors (+${added}) in ${Date.now() - t0}ms`);

  // Load extended MP dataset — only include confirmed superconductors (tc > 0).
  // The tc=0 negative examples contaminate Tc regression, pulling predictions toward zero.
  // Non-SC screening is handled by the lambda/DOS prefilter, not by XGBoost Tc values.
  try {
    const extended = await getExtendedTrainingData();
    const superconductingExtended = extended.filter(e => e.isSuperconductor && e.tc > 0);
    if (superconductingExtended.length > 0) {
      const extStart = Date.now();
      let extAdded = 0;
      for (const e of superconductingExtended) {
        if (trainingPool.hasFormula(e.formula)) continue;
        if (await trainingPool.add(e.formula, e.tc, e.pressureGPa ?? 0)) extAdded++;
        await new Promise<void>(r => setTimeout(r, 0)); // yield per entry
      }
      console.log(`[TrainingPool] Extended dataset: +${extAdded} superconducting entries in ${Date.now() - extStart}ms (pool: ${trainingPool.size})`);
    }
  } catch (err: any) {
    console.warn(`[TrainingPool] Extended dataset load failed: ${err?.message?.slice(0, 120)}`);
  }

  try {
    const { X } = trainingPool.getTrainingData();
    if (X.length >= 10) {
      const featureNames = Object.keys(STATIC_FEATURE_MEANS);
      const sums: Record<string, number> = {};
      const counts: Record<string, number> = {};
      for (const key of featureNames) { sums[key] = 0; counts[key] = 0; }
      for (const row of X) {
        for (let fi = 0; fi < featureNames.length && fi < row.length; fi++) {
          const v = row[fi];
          if (Number.isFinite(v)) { sums[featureNames[fi]] += v; counts[featureNames[fi]]++; }
        }
      }
      computedFeatureMeans = { ...STATIC_FEATURE_MEANS };
      for (const key of featureNames) {
        if (counts[key] > 10) computedFeatureMeans[key] = sums[key] / counts[key];
      }
      featureMeansComputedAt = Date.now();
      console.log(`[GradientBoost] Feature means computed from ${X.length} pool vectors`);
    }
  } catch {}

  await new Promise<void>(r => setTimeout(r, 200));

  try {
    const { X, y, formulas } = trainingPool.getTrainingData();
    if (X.length >= 10) {
      const trainStart = Date.now();
      console.log(`[GradientBoost] Phase 2 (full) training — ${X.length} samples, 45 trees...`);
      cachedModel = await trainGradientBoosting(X, y, 45, 0.12, 4);
      cachedCalibration = await computeCalibration(cachedModel);
      await logModelVersion("full-training", X.length);
      console.log(`[GradientBoost] Phase 2 model ready in ${Date.now() - trainStart}ms`);

      setTimeout(async () => {
        try {
          if (process.env.OFFLOAD_XGB_TO_GCP === "true") {
            console.log(`[GradientBoost] OFFLOAD_XGB_TO_GCP=true — dispatching deferred ensemble to GCP (${X.length} samples)`);
            dispatchXGBJobToGCP(X, y, formulas).catch(err => console.warn("[GradientBoost] GCP XGB dispatch failed:", err.message));
          } else {
            const isOffloaded = process.env.OFFLOAD_GNN_TO_GCP === "true";
            console.log(`[GradientBoost] Deferred ensemble training — ${X.length} samples, ${isOffloaded ? "100" : "300"} trees × 5 models + variance ensemble...`);
            const ensStart = Date.now();
            cachedEnsembleXGB = await trainEnsembleXGB(X, y);
            console.log(`[GradientBoost] Mean ensemble done in ${Date.now() - ensStart}ms, training variance ensemble...`);
            const varStart = Date.now();
            cachedVarianceEnsembleXGB = await trainVarianceEnsembleXGB(X, y, cachedEnsembleXGB);
            console.log(`[GradientBoost] Variance ensemble done in ${Date.now() - varStart}ms — XGBoost fully ready`);
            cachedGlobalFeatureImportance = null;
            buildFeatureImportanceCache();
          }
        } catch (e) { console.error("[GradientBoost] Deferred ensemble training failed:", e); }
      }, 45000);
    }
  } catch (e) {
    console.error("[GradientBoost] Full training failed:", e);
  }
}

export function startPoolInit(): void {
  if (!poolInitPromise) {
    poolInitPromise = initPoolAsync();
  }
}

async function prepareTrainingData(): Promise<{ X: number[][]; y: number[]; formulas: string[] }> {
  await ensurePoolInitialized();
  return trainingPool.getTrainingData();
}

/** Public accessor for eval-harness CV — returns feature matrix after pool is ready. */
export async function getXGBTrainingData(): Promise<{ X: number[][]; y: number[]; formulas: string[] }> {
  await ensurePoolInitialized();
  return trainingPool.getTrainingData();
}

export async function getTrainedModel(): Promise<GBModel> {
  if (cachedModel) return cachedModel;

  if (!poolInitialized) {
    return FALLBACK_MODEL;
  }

  const { X, y } = await prepareTrainingData();

  if (X.length < 10) {
    cachedModel = {
      trees: [],
      flatTrees: [],
      learningRate: 0.1,
      basePrediction: 20,
      featureNames: FEATURE_NAMES,
      trainedAt: Date.now(),
    };
    return cachedModel;
  }

  cachedModel = await trainGradientBoosting(X, y, 80, 0.08, 5);
  cachedCalibration = await computeCalibration(cachedModel);

  if (!cachedEnsembleXGB && X.length >= 30) {
    setTimeout(async () => {
      try {
        cachedEnsembleXGB = await trainEnsembleXGB(X, y);
        cachedVarianceEnsembleXGB = await trainVarianceEnsembleXGB(X, y, cachedEnsembleXGB);
        cachedGlobalFeatureImportance = null;
        buildFeatureImportanceCache();
      } catch (e) { console.error("[GradientBoost] Deferred ensemble training failed:", e); }
    }, 30000);
  }

  await logModelVersion("initial-training", X.length);

  return cachedModel;
}

const APPLICATION_TC_TARGETS: Record<string, { target: number; range: [number, number]; label: string }> = {
  "high-temperature superconductor": { target: 293, range: [200, 350], label: "room-temperature" },
  "clathrate hydride superconductor": { target: 293, range: [200, 350], label: "room-temperature hydride" },
  "topological insulator": { target: 30, range: [5, 77], label: "cryogenic topological" },
  "heavy fermion superconductor": { target: 20, range: [2, 77], label: "heavy fermion" },
  "kagome lattice metal": { target: 77, range: [20, 150], label: "LN₂-range kagome" },
  "spin-orbit coupled material": { target: 50, range: [10, 120], label: "SOC material" },
  "ultra-hard coating material": { target: 40, range: [10, 100], label: "hard coating SC" },
  "MXene-based compound": { target: 77, range: [20, 150], label: "MXene SC" },
};

let activeApplication: string | null = null;

export function setActiveApplication(app: string | null): void {
  activeApplication = app;
}

function applicationAwareScore(tc: number, application?: string | null): number {
  const app = application || activeApplication;
  if (!app || !APPLICATION_TC_TARGETS[app]) {
    if (tc > 293) return 0.92;
    if (tc > 200) return 0.85;
    if (tc > 100) return 0.70;
    if (tc > 50) return 0.55;
    if (tc > 20) return 0.40;
    if (tc > 5) return 0.25;
    if (tc > 1) return 0.15;
    return 0.05;
  }

  const { target, range } = APPLICATION_TC_TARGETS[app];
  const [lo, hi] = range;

  if (tc >= lo && tc <= hi) {
    const distFromTarget = Math.abs(tc - target);
    const maxDist = Math.max(target - lo, hi - target);
    const proximity = 1.0 - distFromTarget / maxDist;
    return 0.70 + 0.25 * proximity;
  }

  if (tc > hi) {
    const overshoot = (tc - hi) / hi;
    return Math.max(0.50, 0.70 - overshoot * 0.3);
  }

  const undershoot = (lo - tc) / lo;
  return Math.max(0.05, 0.40 - undershoot * 0.5);
}

export async function gbPredict(features: MLFeatureVector, formula?: string): Promise<{ tcPredicted: number; score: number; reasoning: string[] }> {
  const t0 = Date.now();
  const model = await getTrainedModel();
  const x = featureVectorToArray(features, formula);
  const tcPredicted = predictWithModel(model, x);

  const reasoning: string[] = [];

  const featureImportance = new Map<number, number>();
  for (const tree of model.trees) {
    const imp = getTreeFeatureImportance(tree);
    for (const [k, v] of imp) {
      featureImportance.set(k, (featureImportance.get(k) || 0) + v);
    }
  }

  const sorted = [...featureImportance.entries()].sort((a, b) => b[1] - a[1]);
  const top3 = sorted.slice(0, 3);
  for (const [fi] of top3) {
    const name = FEATURE_NAMES[fi] || `f${fi}`;
    const val = x[fi];
    reasoning.push(`${name}=${val.toFixed(4)} (key predictor)`);
  }

  let score = applicationAwareScore(tcPredicted);

  const lambda = features.electronPhononLambda;
  if (lambda > 1.5) score += 0.10;
  else if (lambda > 0.8) score += 0.05;

  if (features.metallicity < 0.3) score -= 0.15;
  else if (features.metallicity < 0.5) score -= 0.05;

  if (features.correlationStrength > 0.85) score -= 0.10;

  score = Math.max(0.01, Math.min(0.95, score));

  const safeTc = Number.isFinite(tcPredicted) ? Math.min(350, Math.max(0, Math.round(tcPredicted * 10) / 10)) : 0;
  const elapsed = Date.now() - t0;
  try { const { recordInferenceTime } = require("./model-diagnostics"); recordInferenceTime("xgboost", elapsed); } catch {}
  return { tcPredicted: safeTc, score, reasoning };
}

export interface XGBUncertaintyResult {
  tcMean: number;
  tcStd: number;
  tcCI95: [number, number];
  epistemicStd: number;
  aleatoricStd: number;
  totalStd: number;
  normalizedUncertainty: number;
  score: number;
  perModelPredictions: number[];
  acquisitionScore: number;
  reasoning: string[];
}

export async function gbPredictWithUncertainty(features: MLFeatureVector, formula?: string): Promise<XGBUncertaintyResult> {
  await getTrainedModel();

  const resolvedFormula = formula || features._sourceFormula;
  const x = featureVectorToArray(features, resolvedFormula);

  if (!cachedEnsembleXGB) {
    const singlePred = predictWithModel(cachedModel!, x);
    const safeTc = Number.isFinite(singlePred) ? Math.max(0, singlePred) : 0;
    const nSamples = cachedTrainingSnapshot?.dataSize ?? 0;
    const coldStartUncertainty = nSamples < 5 ? 0.95 : nSamples < 15 ? 0.85 : nSamples < 30 ? 0.75 : 0.6;
    const coldStartStd = coldStartUncertainty * Math.max(1, safeTc + 10);
    return {
      tcMean: safeTc,
      tcStd: coldStartStd,
      tcCI95: [Math.max(0, safeTc - 1.96 * coldStartStd), safeTc + 1.96 * coldStartStd],
      epistemicStd: coldStartStd,
      aleatoricStd: 0,
      totalStd: coldStartStd,
      normalizedUncertainty: coldStartUncertainty,
      score: safeTc > 100 ? 0.7 : safeTc > 20 ? 0.4 : 0.1,
      perModelPredictions: [safeTc],
      acquisitionScore: safeTc / 300 + getCuriosityMultiplier() * coldStartUncertainty,
      reasoning: [`Single model (${nSamples} samples, no ensemble yet — high uncertainty)`],
    };
  }

  const result = predictEnsembleXGB(cachedEnsembleXGB, x);
  const meanTc = result.mean;

  const epistemicStd = result.std;
  const epistemicVar = epistemicStd * epistemicStd;

  let aleatoricVar = 0;
  let aleatoricNote = "";
  if (cachedVarianceEnsembleXGB) {
    const varResult = predictEnsembleXGB(cachedVarianceEnsembleXGB, x);
    if (cachedVarianceEnsembleXGB.isLogVariance) {
      aleatoricVar = Math.max(0, Math.exp(varResult.mean));
    } else {
      aleatoricVar = Math.max(0, varResult.mean);
    }

    const aleatoricStdRaw = Math.sqrt(aleatoricVar);
    const expectedMinStd = meanTc > 50 ? meanTc * 0.05 : 2.5;
    if (aleatoricStdRaw < expectedMinStd && meanTc > 20) {
      aleatoricVar = expectedMinStd * expectedMinStd;
      aleatoricNote = `Aleatoric floor applied (${expectedMinStd.toFixed(1)}K, ~5% of Tc)`;
    }

    const maxReasonableVar = meanTc > 10 ? (meanTc * 2) * (meanTc * 2) : 400;
    if (aleatoricVar > maxReasonableVar) {
      aleatoricVar = maxReasonableVar;
      aleatoricNote = `Aleatoric capped at ${Math.sqrt(maxReasonableVar).toFixed(1)}K (variance model outlier)`;
    }
  }
  const aleatoricStd = Math.sqrt(aleatoricVar);

  const totalVar = epistemicVar + aleatoricVar;
  const totalStd = Math.sqrt(totalVar);

  const ci95Lower = Math.max(0, meanTc - 1.96 * totalStd);
  const ci95Upper = meanTc + 1.96 * totalStd;

  const normalizedUncertainty = Math.min(1.0, totalStd / Math.max(1, meanTc + 10));

  let score = applicationAwareScore(meanTc);

  if (features.electronPhononLambda > 1.5) score += 0.10;
  else if (features.electronPhononLambda > 0.8) score += 0.05;
  if (features.metallicity < 0.3) score -= 0.15;
  else if (features.metallicity < 0.5) score -= 0.05;
  if (features.correlationStrength > 0.85) score -= 0.10;
  score = Math.max(0.01, Math.min(0.95, score));

  const normalizedTc = Math.min(1.0, meanTc / 300);
  const curiosity = getCuriosityMultiplier();
  const acquisitionScore = normalizedTc + curiosity * normalizedUncertainty;

  const reasoning: string[] = [];
  reasoning.push(`Ensemble: ${XGB_ENSEMBLE_SIZE} models, Tc=${meanTc.toFixed(1)}K ± ${totalStd.toFixed(1)}K`);
  reasoning.push(`Epistemic σ=${epistemicStd.toFixed(2)}K, Aleatoric σ=${aleatoricStd.toFixed(2)}K`);
  if (aleatoricNote) reasoning.push(aleatoricNote);
  reasoning.push(`95% CI: [${ci95Lower.toFixed(1)}K, ${ci95Upper.toFixed(1)}K]`);
  if (normalizedUncertainty > 0.6) reasoning.push("Very high uncertainty - priority exploration target");
  else if (normalizedUncertainty > 0.3) reasoning.push("Moderate uncertainty - good exploration candidate");
  else reasoning.push("Low uncertainty - prediction is confident");

  const safeMean = Number.isFinite(meanTc) ? Math.min(350, Math.max(0, Math.round(meanTc * 10) / 10)) : 0;

  return {
    tcMean: safeMean,
    tcStd: Math.round(totalStd * 10) / 10,
    tcCI95: [Math.round(ci95Lower * 10) / 10, Math.round(ci95Upper * 10) / 10],
    epistemicStd: Math.round(epistemicStd * 100) / 100,
    aleatoricStd: Math.round(aleatoricStd * 100) / 100,
    totalStd: Math.round(totalStd * 100) / 100,
    normalizedUncertainty: Math.round(normalizedUncertainty * 1000) / 1000,
    score,
    perModelPredictions: result.predictions.map(p => Math.round(p * 10) / 10),
    acquisitionScore: Math.round(acquisitionScore * 1000) / 1000,
    reasoning,
  };
}

function buildFeatureImportanceCache(): void {
  const model = cachedModel;
  if (!model || model.trees.length === 0) {
    cachedGlobalFeatureImportance = [];
    return;
  }

  const featureImp = new Map<number, number>();
  for (const tree of model.trees) {
    const imp = getTreeFeatureImportance(tree);
    for (const [k, v] of imp) {
      featureImp.set(k, (featureImp.get(k) || 0) + v);
    }
  }

  if (cachedEnsembleXGB) {
    for (const ensModel of cachedEnsembleXGB.models) {
      for (const tree of ensModel.trees) {
        const imp = getTreeFeatureImportance(tree);
        for (const [k, v] of imp) {
          featureImp.set(k, (featureImp.get(k) || 0) + v);
        }
      }
    }
  }

  const sorted = [...featureImp.entries()].sort((a, b) => b[1] - a[1]);
  const maxImp = sorted.length > 0 ? sorted[0][1] : 1;

  cachedGlobalFeatureImportance = sorted.map(([idx, count]) => ({
    name: FEATURE_NAMES[idx] || `feature_${idx}`,
    index: idx,
    importance: count,
    normalizedImportance: Math.round((count / maxImp) * 1000) / 1000,
  }));
}

export function getGlobalFeatureImportance(topN: number = 30): { name: string; index: number; importance: number; normalizedImportance: number }[] {
  if (!cachedGlobalFeatureImportance) {
    buildFeatureImportanceCache();
  }
  return (cachedGlobalFeatureImportance || []).slice(0, topN);
}

export function getFeatureNames(): string[] {
  return [...FEATURE_NAMES];
}

export function getXGBEnsembleStats() {
  return {
    ensembleSize: XGB_ENSEMBLE_SIZE,
    bootstrapRatio: BOOTSTRAP_SAMPLE_RATIO,
    trained: cachedEnsembleXGB !== null,
    trainedAt: cachedEnsembleXGB?.trainedAt ?? null,
    modelTreeCounts: cachedEnsembleXGB?.models.map(m => m.trees.length) ?? [],
  };
}

let cachedValidation: { mse: number; r2: number; nTrees: number; details: { formula: string; actual: number; predicted: number }[]; forVersion: number } | null = null;

export async function validateModel(): Promise<{ mse: number; r2: number; nTrees: number; details: { formula: string; actual: number; predicted: number }[] }> {
  if (cachedValidation && cachedValidation.forVersion === modelVersion && modelVersion > 0) {
    return { mse: cachedValidation.mse, r2: cachedValidation.r2, nTrees: cachedValidation.nTrees, details: cachedValidation.details };
  }

  const model = await getTrainedModel();
  const details: { formula: string; actual: number; predicted: number }[] = [];
  let sse = 0;
  let sst = 0;
  const allTc = SUPERCON_TRAINING_DATA.map(e => e.tc);
  const meanTc = allTc.reduce((s, v) => s + v, 0) / allTc.length;

  for (const entry of SUPERCON_TRAINING_DATA) {
    try {
      const features = await extractFeatures(entry.formula, { pressureGpa: entry.pressureGPa ?? 0 } as any);
      const x = featureVectorToArray(features, entry.formula);
      if (x.some(v => !Number.isFinite(v))) continue;
      const pred = predictWithModel(model, x);
      details.push({ formula: entry.formula, actual: entry.tc, predicted: Math.round(pred * 10) / 10 });
      sse += (entry.tc - pred) ** 2;
      sst += (entry.tc - meanTc) ** 2;
    } catch {
      continue;
    }
  }

  const result = {
    mse: sse / details.length,
    r2: sst < 1e-6 ? 0 : 1 - sse / sst,
    nTrees: model.trees.length,
    details,
  };

  cachedValidation = { ...result, forVersion: modelVersion };
  return result;
}

export async function getCalibrationData(): Promise<Omit<CalibrationData, 'residuals'> & { residualCount: number }> {
  if (!cachedCalibration) {
    if (!cachedModel) {
      return {
        r2: 0, mae: 0, mse: 0, rmse: 0, nSamples: 0, nTrees: 0,
        percentiles: { p5: 0, p10: 0, p25: 0, p50: 0, p75: 0, p90: 0, p95: 0 },
        absResidualPercentiles: { p50: 0, p75: 0, p90: 0, p95: 0 },
        relativeErrorPercentiles: { p50: 0, p75: 0, p90: 0, p95: 0 },
        predictedVsActual: [],
        residualCount: 0,
      };
    }
    cachedCalibration = await computeCalibration(cachedModel);
  }
  const { residuals, ...rest } = cachedCalibration;
  return { ...rest, residualCount: residuals.length };
}

export async function getConfidenceBand(predictedTc: number): Promise<{ lower: number; upper: number }> {
  if (!cachedCalibration) {
    if (!cachedModel) {
      return { lower: Math.max(0, predictedTc - 20), upper: predictedTc + 20 };
    }
    cachedCalibration = await computeCalibration(cachedModel);
  }

  const relP90 = cachedCalibration.relativeErrorPercentiles.p90;
  const absP90 = cachedCalibration.absResidualPercentiles.p90;

  const relativeMargin = predictedTc * relP90;
  const absoluteFloor = Math.min(absP90, 5);
  const errorMargin = Math.max(relativeMargin, absoluteFloor);

  return {
    lower: Math.round(Math.max(0, predictedTc - errorMargin) * 10) / 10,
    upper: Math.round((predictedTc + errorMargin) * 10) / 10,
  };
}

let failureExamples: SuperconEntry[] = [];
let successExamples: SuperconEntry[] = [];
let surrogateScreenCount = 0;
let surrogatePassCount = 0;
let surrogateRejectCount = 0;
let lastRetrainCycle = 0;

let modelVersion = 0;
const MAX_VERSION_HISTORY = 50;

interface ModelVersionRecord {
  version: number;
  trainedAt: number;
  datasetSize: number;
  nTrees: number;
  r2: number;
  mae: number;
  rmse: number;
  ensembleSize: number;
  ensembleTreeCounts: number[];
  successExamples: number;
  failureExamples: number;
  evaluatedEntries: number;
  trigger: string;
  predictionVariance: number;
}

const versionHistory: ModelVersionRecord[] = [];

async function logModelVersion(trigger: string, datasetSize: number): Promise<ModelVersionRecord> {
  modelVersion++;
  const cal = cachedCalibration ?? await computeCalibration(cachedModel!);
  const ensembleStats = getXGBEnsembleStats();

  let predVariance = 0;
  if (cachedEnsembleXGB && datasetSize >= 500 && xgboostRetrainCount >= 10) {
    const benchmarks = ["MgB2", "NbSn3", "YBa2Cu3O7", "LaH10", "FeSe"];

    const frontierFormulas: string[] = [];
    const evalKeys = [...evaluatedDataset.keys()];
    if (evalKeys.length > 0) {
      const shuffled = evalKeys.slice();
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      for (const f of shuffled.slice(0, 5)) {
        if (!benchmarks.includes(f)) frontierFormulas.push(f);
      }
    }

    const testFormulas = [...benchmarks, ...frontierFormulas];
    const variances: number[] = [];
    for (const f of testFormulas) {
      try {
        const features = await extractFeatures(f);
        const x = featureVectorToArray(features, f);
        const result = predictEnsembleXGB(cachedEnsembleXGB, x);
        variances.push(result.std);
      } catch { /* skip */ }
    }
    predVariance = variances.length > 0
      ? variances.reduce((s, v) => s + v, 0) / variances.length
      : 0;
  }

  const record: ModelVersionRecord = {
    version: modelVersion,
    trainedAt: Date.now(),
    datasetSize,
    nTrees: cachedModel?.trees.length ?? 0,
    r2: cal.r2,
    mae: cal.mae,
    rmse: cal.rmse,
    ensembleSize: ensembleStats.ensembleSize,
    ensembleTreeCounts: ensembleStats.modelTreeCounts,
    successExamples: successExamples.length,
    failureExamples: failureExamples.length,
    evaluatedEntries: evaluatedDataset.size,
    trigger,
    predictionVariance: Math.round(predVariance * 100) / 100,
  };

  versionHistory.push(record);
  if (versionHistory.length > MAX_VERSION_HISTORY) {
    versionHistory.shift();
  }

  console.log(
    `[XGBoost] v${record.version} | R²=${record.r2.toFixed(4)} | MAE=${record.mae.toFixed(2)}K | RMSE=${record.rmse.toFixed(2)}K | ` +
    `trees=${record.nTrees} | ensemble=[${record.ensembleTreeCounts.join(",")}] | ` +
    `dataset=${record.datasetSize} (${record.successExamples}S/${record.failureExamples}F/${record.evaluatedEntries}E) | ` +
    `predVar=${record.predictionVariance.toFixed(2)}K | trigger=${record.trigger}`
  );

  return record;
}

function shouldRetrainOnErrorRate(): boolean {
  if (versionHistory.length < 2) return false;
  const latest = versionHistory[versionHistory.length - 1];
  const prev = versionHistory[versionHistory.length - 2];
  if (latest.mae > prev.mae * 1.15 || latest.r2 < prev.r2 - 0.05) {
    console.log(
      `[XGBoost] Error rate degradation detected: R² ${prev.r2.toFixed(4)} -> ${latest.r2.toFixed(4)}, ` +
      `MAE ${prev.mae.toFixed(2)} -> ${latest.mae.toFixed(2)} — triggering retrain`
    );
    return true;
  }
  return false;
}

export function getModelVersionHistory() {
  return {
    currentVersion: modelVersion,
    historyLength: versionHistory.length,
    history: versionHistory.slice(-20),
    latestMetrics: versionHistory.length > 0 ? versionHistory[versionHistory.length - 1] : null,
    performanceTrend: versionHistory.length >= 3
      ? {
          r2Trend: versionHistory.slice(-5).map(v => ({ version: v.version, r2: v.r2 })),
          maeTrend: versionHistory.slice(-5).map(v => ({ version: v.version, mae: v.mae })),
          datasetGrowth: versionHistory.slice(-5).map(v => ({ version: v.version, size: v.datasetSize })),
        }
      : null,
  };
}

export interface EvaluatedEntry {
  formula: string;
  tc: number;
  formationEnergy: number | null;
  stable: boolean;
  source: "dft" | "xtb" | "external" | "active-learning";
  evaluatedAt: number;
  pressureGPa: number;
  lambda?: number;
  omegaLog?: number;
  dosAtEF?: number;
}

const evaluatedDataset = new Map<string, EvaluatedEntry>();
let xgboostRetrainCount = 0;
let lastXGBoostRetrainCycle = 0;
let totalDFTFeedback = 0;

const SOURCE_PRIORITY: Record<EvaluatedEntry["source"], number> = {
  "active-learning": 0,
  "xtb": 1,
  "dft": 2,
  "external": 3,
};

export function incorporateDFTResult(
  formula: string,
  tc: number,
  formationEnergy: number | null,
  stable: boolean,
  source: EvaluatedEntry["source"] = "dft",
  lambda?: number,
  omegaLog?: number,
  dosAtEF?: number,
  pressureGPa: number = 0
): boolean {
  totalDFTFeedback++;

  const existing = evaluatedDataset.get(formula);
  if (existing) {
    if (SOURCE_PRIORITY[source] > SOURCE_PRIORITY[existing.source]) {
      existing.tc = Math.max(0, tc);
      existing.formationEnergy = formationEnergy;
      existing.stable = stable;
      existing.source = source;
      existing.evaluatedAt = Date.now();
      existing.pressureGPa = pressureGPa;
      if (lambda != null) existing.lambda = lambda;
      if (omegaLog != null) existing.omegaLog = omegaLog;
      if (dosAtEF != null) existing.dosAtEF = dosAtEF;
      return true;
    }
    return false;
  }

  evaluatedDataset.set(formula, {
    formula,
    tc: Math.max(0, tc),
    formationEnergy,
    stable,
    source,
    evaluatedAt: Date.now(),
    pressureGPa,
    lambda,
    omegaLog,
    dosAtEF,
  });

  return true;
}

export async function retrainXGBoostFromEvaluated(cycleCount?: number): Promise<{
  retrained: boolean;
  datasetSize: number;
  newEntries: number;
}> {
  await ensurePoolInitialized();

  await trainingPool.addBatch(successExamples.map(e => ({ formula: e.formula, tc: e.tc, pressureGPa: e.pressureGPa ?? 0 })));
  await trainingPool.addBatch(failureExamples.map(e => ({ formula: e.formula, tc: e.tc, pressureGPa: e.pressureGPa ?? 0 })));

  const existingFormulas = new Set<string>();
  for (const e of SUPERCON_TRAINING_DATA) existingFormulas.add(e.formula);

  let newFromEval = 0;
  for (const [formula, entry] of evaluatedDataset) {
    if (existingFormulas.has(formula)) continue;
    existingFormulas.add(formula);
    const originalFamily = classifyFamily(entry.formula);
    const dftStatus = entry.stable ? "DFT-Evaluated" : "DFT-Failed";
    const newEntry: SuperconEntry = {
      formula: entry.formula,
      tc: entry.tc,
      family: `${originalFamily}|${dftStatus}`,
      isSuperconductor: entry.tc > 0 && entry.stable,
      pressureGPa: entry.pressureGPa,
    };
    SUPERCON_TRAINING_DATA.push(newEntry);
    await trainingPool.add(entry.formula, entry.tc, entry.pressureGPa);
    newFromEval++;
  }

  if (newFromEval > 0) {
    cachedTrainingSnapshot = null;
  }

  let { X, y, formulas } = trainingPool.getTrainingData();

  if (X.length < 10) {
    return { retrained: false, datasetSize: X.length, newEntries: newFromEval };
  }

  // Oversample DFT-verified rows to give them 5× effective weight in the
  // bootstrap ensemble.  Appending 4 extra copies is equivalent to a 5× sample
  // weight without modifying the bootstrap sampling kernel.
  if (dftVerifiedFormulas.size > 0) {
    const extraX: number[][] = [];
    const extraY: number[] = [];
    for (let i = 0; i < formulas.length; i++) {
      if (dftVerifiedFormulas.has(formulas[i])) {
        for (let k = 0; k < 4; k++) {
          extraX.push(X[i]);
          extraY.push(y[i]);
        }
      }
    }
    if (extraX.length > 0) {
      X = [...X, ...extraX];
      y = [...y, ...extraY];
      console.log(`[XGBoost] DFT-verified oversampling: +${extraX.length} rows (${extraX.length / 4} DFT formulas × 4 extra copies)`);
    }
  }

  const hp = getXGBoostHyperparamOverrides();
  const hpTrees = hp.nTrees ?? 300;
  const hpLR = hp.learningRate ?? 0.05;
  const hpDepth = hp.maxDepth ?? 6;

  const gcpXgbMode = process.env.OFFLOAD_XGB_TO_GCP === "true";

  if (gcpXgbMode) {
    // GCP handles all training — dispatch and return immediately, no local training.
    if (X.length >= 30) {
      console.log(`[XGBoost] OFFLOAD_XGB_TO_GCP=true — dispatching retrain to GCP (${X.length} samples)`);
      dispatchXGBJobToGCP(X, y, formulas).catch(err => console.warn("[XGBoost] GCP dispatch failed:", err.message));
    }
    return { retrained: true, datasetSize: X.length, newEntries: newFromEval };
  }

  invalidateModel();
  console.log(`[XGBoost] Retrain #${xgboostRetrainCount + 1} — ${X.length} samples, ${hpTrees} trees (lr=${hpLR}, depth=${hpDepth})...`);
  const retrainStart = Date.now();
  cachedModel = await trainGradientBoosting(X, y, hpTrees, hpLR, hpDepth);
  await new Promise<void>(r => setTimeout(r, 0));
  cachedCalibration = await computeCalibration(cachedModel);
  console.log(`[XGBoost] Base model done in ${Date.now() - retrainStart}ms`);

  if (X.length >= 30) {
    if (false) { // local-only path
      console.log(`[XGBoost] Training mean ensemble (${X.length} samples)...`);
      const ensStart = Date.now();
      cachedEnsembleXGB = await trainEnsembleXGB(X, y);
      console.log(`[XGBoost] Mean ensemble done in ${Date.now() - ensStart}ms, training variance ensemble...`);
      const varStart = Date.now();
      cachedVarianceEnsembleXGB = await trainVarianceEnsembleXGB(X, y, cachedEnsembleXGB);
      console.log(`[XGBoost] Variance ensemble done in ${Date.now() - varStart}ms`);
      cachedGlobalFeatureImportance = null;
      buildFeatureImportanceCache();
      await new Promise<void>(r => setTimeout(r, 0));
    }
  }

  xgboostRetrainCount++;
  if (cycleCount != null) lastXGBoostRetrainCycle = cycleCount;
  lastRetrainCycle = Date.now();

  const vr = await logModelVersion("evaluated-retrain", X.length);

  if (shouldRetrainOnErrorRate()) {
    const correctedLR = hpLR * 0.6;
    const correctedDepth = Math.max(3, hpDepth - 1);
    const correctedMinSamples = (hp.minSamples ?? 5) + 3;
    console.log(
      `[XGBoost] Error-rate correction: LR ${hpLR}->${correctedLR.toFixed(3)}, ` +
      `depth ${hpDepth}->${correctedDepth}, minSamples +3 to ${correctedMinSamples}`
    );
    invalidateModel();
    cachedModel = await trainGradientBoosting(X, y, hpTrees, correctedLR, correctedDepth);
    await new Promise<void>(r => setTimeout(r, 0));
    cachedCalibration = await computeCalibration(cachedModel);
    if (X.length >= 30) {
      if (process.env.OFFLOAD_XGB_TO_GCP === "true") {
        dispatchXGBJobToGCP(X, y, formulas).catch(() => {});
      } else {
        cachedEnsembleXGB = await trainEnsembleXGB(X, y);
        await new Promise<void>(r => setTimeout(r, 0));
        cachedVarianceEnsembleXGB = await trainVarianceEnsembleXGB(X, y, cachedEnsembleXGB);
        await new Promise<void>(r => setTimeout(r, 0));
        cachedGlobalFeatureImportance = null;
        buildFeatureImportanceCache();
        await new Promise<void>(r => setTimeout(r, 0));
      }
    }
    await logModelVersion("error-rate-correction", X.length);
  }

  try {
    const { updateOODModel } = require("./ood-detector");
    updateOODModel().catch(() => {});
  } catch {}

  return { retrained: true, datasetSize: X.length, newEntries: newFromEval };
}

export function getEvaluatedDatasetStats() {
  const entries = [...evaluatedDataset.values()];
  const n = entries.length;
  let dft = 0, xtb = 0, external = 0, activeLearning = 0, stableCount = 0, tcSum = 0;
  for (const e of entries) {
    if (e.source === "dft") dft++;
    else if (e.source === "xtb") xtb++;
    else if (e.source === "external") external++;
    else if (e.source === "active-learning") activeLearning++;
    if (e.stable) stableCount++;
    tcSum += e.tc;
  }
  return {
    totalEvaluated: n,
    totalDFTFeedback,
    xgboostRetrainCount,
    lastXGBoostRetrainCycle,
    bySource: { dft, xtb, external, activeLearning },
    stableCount,
    unstableCount: n - stableCount,
    avgTc: n > 0 ? tcSum / n : 0,
    datasetGrowthRate: n > 0 ? n / Math.max(1, xgboostRetrainCount) : 0,
  };
}

export function invalidateModel(): void {
  cachedModel = null;
  cachedCalibration = null;
  cachedEnsembleXGB = null;
  cachedVarianceEnsembleXGB = null;
  cachedTrainingSnapshot = null;
  cachedGlobalFeatureImportance = null;
  cachedValidation = null;
  poolInitialized = false;
  crystalSymTargetEncoding = null;
  miedemaCache.clear();
}

export async function surrogateScreen(formula: string, minTcThreshold: number = 5): Promise<{
  pass: boolean;
  predictedTc: number;
  score: number;
  reasoning: string[];
}> {
  surrogateScreenCount++;
  try {
    const features = await extractFeatures(formula);

    if (features.metallicity < 0.05) {
      surrogateRejectCount++;
      return { pass: false, predictedTc: 0, score: 0, reasoning: ["Deep insulator: metallicity < 0.05, not dopable"] };
    }

    const result = await gbPredict(features, formula);
    const predictedTc = Number.isFinite(result.tcPredicted) ? result.tcPredicted : 0;

    if (features.metallicity < 0.15) {
      const dopingPenalty = 0.5;
      const adjustedScore = result.score * dopingPenalty;
      const dopingReasoning = [
        ...result.reasoning,
        `Low metallicity (${features.metallicity.toFixed(3)}): parent phase is insulating — potential doping candidate, score penalized 50%`,
      ];
      if (adjustedScore < 0.1 || predictedTc < minTcThreshold) {
        surrogateRejectCount++;
        return { pass: false, predictedTc, score: adjustedScore, reasoning: dopingReasoning };
      }
      surrogatePassCount++;
      return { pass: true, predictedTc, score: adjustedScore, reasoning: dopingReasoning };
    }

    if (predictedTc < minTcThreshold) {
      surrogateRejectCount++;
      return { pass: false, predictedTc, score: result.score, reasoning: result.reasoning };
    }

    if (result.score < 0.1) {
      surrogateRejectCount++;
      return { pass: false, predictedTc, score: result.score, reasoning: result.reasoning };
    }

    surrogatePassCount++;
    maybePersistSurrogateMetrics();
    return { pass: true, predictedTc, score: result.score, reasoning: result.reasoning };
  } catch {
    surrogateRejectCount++;
    maybePersistSurrogateMetrics();
    return { pass: false, predictedTc: 0, score: 0, reasoning: ["Feature extraction failed — rejecting candidate"] };
  }
}

export function getSurrogateStats() {
  return {
    totalScreened: surrogateScreenCount,
    totalPassed: surrogatePassCount,
    totalRejected: surrogateRejectCount,
    passRate: surrogateScreenCount > 0 ? surrogatePassCount / surrogateScreenCount : 0,
    successExamples: successExamples.length,
    failureExamples: failureExamples.length,
    lastRetrainCycle,
    totalFeatures: FEATURE_NAMES.length,
    physicsFeatures: PHYSICS_FEATURE_NAMES.length,
    compositionFeatures: COMPOSITION_FEATURE_NAMES.length,
    hyperparameters: { nEstimators: 300, learningRate: 0.05, maxDepth: 6 },
    ensemble: getXGBEnsembleStats(),
    modelVersion,
    latestMetrics: versionHistory.length > 0 ? versionHistory[versionHistory.length - 1] : null,
  };
}

export async function incorporateSuccessData(formula: string, tc: number): Promise<void> {
  const existing = new Set([
    ...SUPERCON_TRAINING_DATA.map(e => e.formula),
    ...successExamples.map(e => e.formula),
  ]);
  if (existing.has(formula)) return;

  successExamples.push({
    formula,
    tc,
    family: "Discovered",
    isSuperconductor: tc > 0,
    pressureGPa: 0,
  });

  if (successExamples.length % 10 === 0) {
    await retrainWithAccumulatedData();
  }
}

export async function retrainWithAccumulatedData(): Promise<number> {
  await ensurePoolInitialized();

  await trainingPool.addBatch(successExamples.map(e => ({ formula: e.formula, tc: e.tc, pressureGPa: e.pressureGPa ?? 0 })));
  await trainingPool.addBatch(failureExamples.map(e => ({ formula: e.formula, tc: e.tc, pressureGPa: e.pressureGPa ?? 0 })));

  const { X, y } = trainingPool.getTrainingData();

  if (X.length < 10) return 0;

  invalidateModel();
  cachedModel = await trainGradientBoosting(X, y, 300, 0.05, 6);
  await new Promise<void>(r => setTimeout(r, 0));
  cachedCalibration = await computeCalibration(cachedModel);
  if (X.length >= 30) {
    cachedEnsembleXGB = await trainEnsembleXGB(X, y);
    await new Promise<void>(r => setTimeout(r, 0));
    cachedVarianceEnsembleXGB = await trainVarianceEnsembleXGB(X, y, cachedEnsembleXGB);
    await new Promise<void>(r => setTimeout(r, 0));
    cachedGlobalFeatureImportance = null;
    buildFeatureImportanceCache();
    await new Promise<void>(r => setTimeout(r, 0));
  }
  lastRetrainCycle = Date.now();

  await logModelVersion("accumulated-retrain", X.length);

  const totalNew = successExamples.length + failureExamples.length;
  return totalNew;
}

export async function incorporateFailureData(): Promise<number> {
  const failedResults = await storage.getFailedComputationalResults(500);
  if (failedResults.length === 0) return 0;

  const seenFormulas = new Set<string>();
  for (const e of SUPERCON_TRAINING_DATA) seenFormulas.add(e.formula);
  for (const e of failureExamples) seenFormulas.add(e.formula);

  let added = 0;
  for (const result of failedResults) {
    const formula = result.formula;
    if (seenFormulas.has(formula)) continue;
    seenFormulas.add(formula);
    failureExamples.push({
      formula,
      tc: 0,
      family: "Failed",
      isSuperconductor: false,
      pressureGPa: 0,
    });
    await trainingPool.add(formula, 0, 0);
    added++;
  }

  if (added > 0) {
    invalidateModel();
    await ensurePoolInitialized();

    await trainingPool.addBatch(failureExamples.map(e => ({ formula: e.formula, tc: e.tc, pressureGPa: e.pressureGPa ?? 0 })));

    const { X, y } = trainingPool.getTrainingData();

    if (X.length >= 10) {
      cachedModel = await trainGradientBoosting(X, y, 300, 0.05, 6);
      await new Promise<void>(r => setTimeout(r, 0));
      cachedCalibration = await computeCalibration(cachedModel);
      if (X.length >= 30) {
        cachedEnsembleXGB = await trainEnsembleXGB(X, y);
        await new Promise<void>(r => setTimeout(r, 0));
        cachedVarianceEnsembleXGB = await trainVarianceEnsembleXGB(X, y, cachedEnsembleXGB);
        await new Promise<void>(r => setTimeout(r, 0));
        cachedGlobalFeatureImportance = null;
        buildFeatureImportanceCache();
        await new Promise<void>(r => setTimeout(r, 0));
      }
      await logModelVersion("failure-retrain", X.length);
    }

    if (failureExamples.length >= 10 && failureExamples.length % 10 === 0) {
      try {
        const { updateOODModel } = require("./ood-detector");
        updateOODModel().then(() => {
          console.log(`[XGBoost] OOD model updated at ${failureExamples.length} failure examples`);
        }).catch(() => {});
      } catch {}
    }
  }

  return added;
}

export function getFailureExampleCount(): number {
  return failureExamples.length;
}

let lastMetricsPersistCount = 0;
const METRICS_PERSIST_INTERVAL = 50;

export async function persistSurrogateMetrics(): Promise<void> {
  if (surrogateScreenCount - lastMetricsPersistCount < METRICS_PERSIST_INTERVAL) return;
  lastMetricsPersistCount = surrogateScreenCount;
  try {
    const passRate = surrogateScreenCount > 0 ? surrogatePassCount / surrogateScreenCount : 0;
    await db.insert(systemMetrics).values([
      {
        metricName: "surrogate_screen_total",
        metricValue: surrogateScreenCount,
        metadata: {
          passed: surrogatePassCount,
          rejected: surrogateRejectCount,
          passRate: Math.round(passRate * 10000) / 10000,
          successExamples: successExamples.length,
          failureExamples: failureExamples.length,
          modelVersion,
        },
      },
    ]);
  } catch (err) {
    console.warn("[XGBoost] Failed to persist surrogate metrics:", err);
  }
}

export function maybePersistSurrogateMetrics(): void {
  if (surrogateScreenCount - lastMetricsPersistCount >= METRICS_PERSIST_INTERVAL) {
    persistSurrogateMetrics().catch(() => {});
  }
}

// ── GCP XGBoost offload ───────────────────────────────────────────────────────

async function dispatchXGBJobToGCP(X: number[][], y: number[], formulas?: string[]): Promise<void> {
  const job = await storage.insertXgbTrainingJob({
    status: "queued",
    featuresX: X as any,
    labelsY: y as any,
    datasetSize: X.length,
  });
  console.log(`[XGBoost] Dispatched training job #${job.id} to GCP (${X.length} samples)`);

  // Co-dispatch a GNN training job with the same formula+Tc pairs so both models
  // train on the same data and stay in sync. Skip if a GNN job is already queued
  // to avoid piling up duplicate jobs when XGB retrains on the same dataset.
  if (formulas && formulas.length > 0 && process.env.OFFLOAD_GNN_TO_GCP === "true") {
    storage.getQueuedGnnTrainingJob().then(existing => {
      if (existing) {
        console.log(`[XGBoost] GNN co-dispatch skipped — job #${existing.id} already queued`);
        return;
      }
      const feIdx = PHYSICS_FEATURE_NAMES.indexOf("formationEnergy");
      const lambdaIdx = PHYSICS_FEATURE_NAMES.indexOf("lambda");
      const gnnTrainingData = formulas.map((formula, i) => ({
        formula,
        tc: y[i] ?? 0,
        formationEnergy: (feIdx >= 0 && X[i]?.[feIdx] != null && Number.isFinite(X[i][feIdx]) && X[i][feIdx] !== 0)
          ? X[i][feIdx] : undefined as number | undefined,
        lambda: (lambdaIdx >= 0 && X[i]?.[lambdaIdx] != null && Number.isFinite(X[i][lambdaIdx]) && X[i][lambdaIdx] > 0)
          ? X[i][lambdaIdx] : undefined as number | undefined,
        structure: undefined as any,
        prototype: undefined as string | undefined,
      }));
      return storage.insertGnnTrainingJob({
        status: "queued",
        trainingData: gnnTrainingData as any,
        datasetSize: gnnTrainingData.length,
        dftSamples: 0,
      }).then(gnnJob => {
        const withFe = gnnTrainingData.filter(s => s.formationEnergy != null).length;
        const withLambda = gnnTrainingData.filter(s => s.lambda != null).length;
        console.log(`[XGBoost] Co-dispatched GNN job #${gnnJob.id} (${gnnTrainingData.length} samples, ${withFe} with FE, ${withLambda} with λ)`);
      });
    }).catch(err => {
      console.warn(`[XGBoost] GNN co-dispatch failed: ${err.message}`);
    });
  }
}

/**
 * Background poller: checks every 30s for completed GCP XGB jobs and applies
 * the model weights locally. Call once at engine startup when OFFLOAD_XGB_TO_GCP=true.
 */
export function startGCPXGBPoller(): void {
  if (process.env.OFFLOAD_XGB_TO_GCP !== "true") return;

  let lastAppliedJobId = 0;

  async function poll() {
    try {
      // Lightweight pre-check: fetch only the job ID to avoid loading multi-MB
      // model blobs (ensembleXGB, varianceEnsembleXGB) on every poll tick.
      const idCheck = await db.execute(
        `SELECT id FROM xgb_training_jobs WHERE status = 'done' ORDER BY completed_at DESC LIMIT 1`
      );
      const latestId: number | undefined = ((idCheck as any).rows?.[0] ?? (Array.isArray(idCheck) ? idCheck[0] : undefined))?.id;
      if (latestId && latestId > lastAppliedJobId) {
        const job = await storage.getLatestCompletedXgbJob();
        if (job && job.id > lastAppliedJobId) {
          lastAppliedJobId = job.id;
          if (job.model) {
            cachedModel = job.model as any;
            cachedCalibration = await computeCalibration(cachedModel!);
          }
          if (job.ensembleXGB) {
            cachedEnsembleXGB = job.ensembleXGB as any;
          }
          if (job.varianceEnsembleXGB) {
            cachedVarianceEnsembleXGB = job.varianceEnsembleXGB as any;
            cachedGlobalFeatureImportance = null;
            buildFeatureImportanceCache();
          }
          console.log(`[GCP-XGB-Poller] Applied XGBoost models from job #${job.id} — R²=${job.r2?.toFixed(3) ?? "?"}, N=${job.datasetSize ?? "?"}`);
        }
      }
    } catch { /* silent */ }
    setTimeout(poll, 30_000);
  }

  setTimeout(poll, 30_000);
  console.log("[GCP-XGB-Poller] Background XGBoost weight poller started (30s interval)");
}
