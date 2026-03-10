import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { getAllPhysicsResults, getPhysicsResult, type PhysicsResult } from "./physics-results-store";
import {
  getElementData,
  getCompositionWeightedProperty,
  getAverageMass,
  isTransitionMetal,
} from "./elemental-data";
import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  parseFormulaElements,
} from "./physics-engine";

interface LambdaTreeNode {
  featureIndex: number;
  threshold: number;
  left: LambdaTreeNode | number;
  right: LambdaTreeNode | number;
}

interface LambdaGBModel {
  trees: LambdaTreeNode[];
  learningRate: number;
  basePrediction: number;
  trainedAt: number;
}

interface LambdaEnsemble {
  models: LambdaGBModel[];
  trainedAt: number;
  datasetSize: number;
  metrics: { r2: number; mae: number; rmse: number };
}

export interface LambdaPrediction {
  lambda: number;
  uncertainty: number;
  tier: "verified-dfpt" | "ml-regression" | "physics-engine";
  confidence: number;
  features: Record<string, number>;
  perModelPredictions?: number[];
}

const LAMBDA_FEATURE_NAMES = [
  "dosAtEF",
  "bandFlatness",
  "vanHoveProximity",
  "nestingScore",
  "avgMass",
  "debyeTemp",
  "hopfieldEta",
  "pressure",
  "hydrogenRatio",
  "vec",
  "metallicity",
  "numElements",
  "avgBulkModulus",
  "avgSommerfeldGamma",
  "enSpread",
  "orbitalDFraction",
];

const LAMBDA_ENSEMBLE_SIZE = 3;
const MIN_RETRAIN_SAMPLES = 20;
const RETRAIN_INTERVAL_MS = 30 * 60 * 1000;

let lambdaEnsemble: LambdaEnsemble | null = null;
let lastPhysicsStoreSize = 0;
let totalPredictions = 0;
let tierBreakdown = { "verified-dfpt": 0, "ml-regression": 0, "physics-engine": 0 };
let retrainCount = 0;
let lastRetrainTime = 0;
let predictionErrors: { predicted: number; actual: number; formula: string }[] = [];

function parseFormula(formula: string): string[] {
  const matches = formula.match(/[A-Z][a-z]?/g);
  return matches ? [...new Set(matches)] : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const re = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(formula)) !== null) {
    const el = m[1];
    const n = m[2] ? parseFloat(m[2]) : 1;
    counts[el] = (counts[el] || 0) + n;
  }
  return counts;
}

function extractLambdaFeatures(formula: string, pressure: number = 0): Record<string, number> {
  const elements = parseFormula(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  const electronic = computeElectronicStructure(formula);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula);

  const avgMass = getAverageMass(counts);
  const debyeTemp = phonon.debyeTemperature;
  const hopfieldEta = getCompositionWeightedProperty(counts, "mcMillanHopfieldEta") ?? 0;
  const avgBulk = getCompositionWeightedProperty(counts, "bulkModulus") ?? 0;
  const avgGamma = getCompositionWeightedProperty(counts, "sommerfeldGamma") ?? 0;

  const enValues = elements.map(e => getElementData(e)?.paulingElectronegativity ?? 1.5);
  const enSpread = enValues.length > 1 ? Math.max(...enValues) - Math.min(...enValues) : 0;

  const hCount = counts["H"] || 0;
  const metalAtomCount = elements
    .filter(e => isTransitionMetal(e) || ["La", "Y", "Ce", "Sc", "Ba", "Sr", "Ca"].includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hydrogenRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;

  let totalVE = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
  }
  const vec = totalAtoms > 0 ? totalVE / totalAtoms : 0;

  let orbitalDFraction = 0;
  let dElectrons = 0;
  let totalElectrons = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) {
      const frac = (counts[el] || 1) / totalAtoms;
      const dEl = data.electronConfiguration?.match(/\dd(\d+)/);
      if (dEl) dElectrons += parseInt(dEl[1]) * frac;
      totalElectrons += data.atomicNumber * frac;
    }
  }
  orbitalDFraction = totalElectrons > 0 ? Math.min(1, dElectrons / Math.max(1, totalElectrons * 0.5)) : 0;

  return {
    dosAtEF: electronic.densityOfStatesAtFermi,
    bandFlatness: electronic.bandFlatness ?? 0,
    vanHoveProximity: electronic.vanHoveProximity ?? 0,
    nestingScore: electronic.nestingScore ?? 0,
    avgMass,
    debyeTemp,
    hopfieldEta,
    pressure,
    hydrogenRatio,
    vec,
    metallicity: electronic.metallicity,
    numElements: elements.length,
    avgBulkModulus: avgBulk,
    avgSommerfeldGamma: avgGamma,
    enSpread,
    orbitalDFraction,
  };
}

function featuresToArray(features: Record<string, number>): number[] {
  return LAMBDA_FEATURE_NAMES.map(name => {
    const v = features[name];
    return v != null && Number.isFinite(v) ? v : 0;
  });
}

function findBestSplit(
  X: number[][],
  residuals: number[],
  indices: number[],
  featureIndex: number
): { threshold: number; improvement: number; leftIndices: number[]; rightIndices: number[] } {
  const pairs = indices
    .map(i => ({ val: X[i][featureIndex], res: residuals[i], idx: i }))
    .sort((a, b) => a.val - b.val);

  const n = pairs.length;
  const totalSum = pairs.reduce((s, p) => s + p.res, 0);

  let bestImprovement = -Infinity;
  let bestThreshold = 0;
  let bestSplitPos = 0;
  let leftSum = 0;

  for (let i = 0; i < n - 1; i++) {
    leftSum += pairs[i].res;
    const leftCount = i + 1;
    const rightCount = n - leftCount;
    const rightSum = totalSum - leftSum;

    if (pairs[i].val === pairs[i + 1].val) continue;

    const improvement = (leftSum * leftSum) / leftCount + (rightSum * rightSum) / rightCount;
    if (improvement > bestImprovement) {
      bestImprovement = improvement;
      bestThreshold = (pairs[i].val + pairs[i + 1].val) / 2;
      bestSplitPos = i + 1;
    }
  }

  return {
    threshold: bestThreshold,
    improvement: bestImprovement,
    leftIndices: pairs.slice(0, bestSplitPos).map(p => p.idx),
    rightIndices: pairs.slice(bestSplitPos).map(p => p.idx),
  };
}

function buildTree(
  X: number[][],
  residuals: number[],
  indices: number[],
  depth: number,
  maxDepth: number
): LambdaTreeNode | number {
  if (depth >= maxDepth || indices.length < 4) {
    const sum = indices.reduce((s, i) => s + residuals[i], 0);
    return sum / indices.length;
  }

  const nFeatures = X[0].length;
  let bestFeature = -1;
  let bestImprovement = -Infinity;
  let bestThreshold = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (let fi = 0; fi < nFeatures; fi++) {
    const split = findBestSplit(X, residuals, indices, fi);
    if (split.improvement > bestImprovement && split.leftIndices.length >= 2 && split.rightIndices.length >= 2) {
      bestImprovement = split.improvement;
      bestFeature = fi;
      bestThreshold = split.threshold;
      bestLeft = split.leftIndices;
      bestRight = split.rightIndices;
    }
  }

  if (bestFeature === -1) {
    const sum = indices.reduce((s, i) => s + residuals[i], 0);
    return sum / indices.length;
  }

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildTree(X, residuals, bestLeft, depth + 1, maxDepth),
    right: buildTree(X, residuals, bestRight, depth + 1, maxDepth),
  };
}

function predictTree(tree: LambdaTreeNode | number, x: number[]): number {
  if (typeof tree === "number") return tree;
  return x[tree.featureIndex] <= tree.threshold
    ? predictTree(tree.left, x)
    : predictTree(tree.right, x);
}

function trainLambdaGBM(
  X: number[][],
  y: number[],
  nEstimators: number = 150,
  learningRate: number = 0.08,
  maxDepth: number = 4
): LambdaGBModel {
  const n = X.length;
  const basePrediction = y.reduce((s, v) => s + v, 0) / n;
  const predictions = new Array(n).fill(basePrediction);
  const trees: LambdaTreeNode[] = [];
  const allIndices = Array.from({ length: n }, (_, i) => i);

  for (let t = 0; t < nEstimators; t++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);

    const tree = buildTree(X, residuals, allIndices, 0, maxDepth);
    if (typeof tree === "number" && Math.abs(tree) < 1e-6) break;

    trees.push(typeof tree === "number" ? { featureIndex: 0, threshold: 0, left: tree, right: tree } : tree);

    for (let i = 0; i < n; i++) {
      predictions[i] += learningRate * predictTree(tree, X[i]);
    }

    const mse = y.reduce((s, yi, i) => s + (yi - predictions[i]) ** 2, 0) / n;
    if (mse < 0.001) break;
  }

  return { trees, learningRate, basePrediction, trainedAt: Date.now() };
}

function predictWithModel(model: LambdaGBModel, x: number[]): number {
  let pred = model.basePrediction;
  for (const tree of model.trees) {
    pred += model.learningRate * predictTree(tree, x);
  }
  return Math.max(0, pred);
}

function bootstrapSample(X: number[][], y: number[]): { X: number[][]; y: number[] } {
  const n = X.length;
  const size = Math.floor(n * 0.8);
  const bsX: number[][] = [];
  const bsY: number[] = [];
  for (let i = 0; i < size; i++) {
    const idx = Math.floor(Math.random() * n);
    bsX.push(X[idx]);
    bsY.push(y[idx]);
  }
  return { X: bsX, y: bsY };
}

function buildTrainingData(): { X: number[][]; y: number[]; formulas: string[] } {
  const X: number[][] = [];
  const y: number[] = [];
  const formulas: string[] = [];

  for (const entry of SUPERCON_TRAINING_DATA) {
    if (entry.lambda == null || entry.lambda <= 0) continue;
    try {
      const features = extractLambdaFeatures(entry.formula, entry.pressureGPa ?? 0);
      const vec = featuresToArray(features);
      if (vec.some(v => !Number.isFinite(v))) continue;
      X.push(vec);
      y.push(entry.lambda);
      formulas.push(entry.formula);
    } catch {
      continue;
    }
  }

  const physicsResults = getAllPhysicsResults();
  for (const result of physicsResults) {
    if (result.lambda <= 0) continue;
    const existing = formulas.indexOf(result.formula);
    if (existing !== -1 && result.tier !== "full-dft") continue;

    try {
      const features = extractLambdaFeatures(result.formula, result.pressure);
      features.dosAtEF = result.dosAtEF > 0 ? result.dosAtEF : features.dosAtEF;
      const vec = featuresToArray(features);
      if (vec.some(v => !Number.isFinite(v))) continue;

      if (existing !== -1) {
        X[existing] = vec;
        y[existing] = result.lambda;
      } else {
        X.push(vec);
        y.push(result.lambda);
        formulas.push(result.formula);
      }
    } catch {
      continue;
    }
  }

  return { X, y, formulas };
}

function computeMetrics(model: LambdaGBModel, X: number[][], y: number[]): { r2: number; mae: number; rmse: number } {
  if (X.length === 0) return { r2: 0, mae: 0, rmse: 0 };

  const meanY = y.reduce((s, v) => s + v, 0) / y.length;
  let sse = 0;
  let sst = 0;
  let totalAbsErr = 0;

  for (let i = 0; i < X.length; i++) {
    const pred = predictWithModel(model, X[i]);
    const residual = y[i] - pred;
    sse += residual ** 2;
    sst += (y[i] - meanY) ** 2;
    totalAbsErr += Math.abs(residual);
  }

  return {
    r2: sst > 0 ? Math.round((1 - sse / sst) * 10000) / 10000 : 0,
    mae: Math.round((totalAbsErr / y.length) * 10000) / 10000,
    rmse: Math.round(Math.sqrt(sse / y.length) * 10000) / 10000,
  };
}

export function trainLambdaRegressor(): void {
  const { X, y, formulas } = buildTrainingData();
  if (X.length < 5) return;

  const models: LambdaGBModel[] = [];
  for (let i = 0; i < LAMBDA_ENSEMBLE_SIZE; i++) {
    const { X: bsX, y: bsY } = bootstrapSample(X, y);
    const model = trainLambdaGBM(bsX, bsY, 150, 0.08, 4);
    models.push(model);
  }

  const metrics = computeMetrics(models[0], X, y);

  lambdaEnsemble = {
    models,
    trainedAt: Date.now(),
    datasetSize: X.length,
    metrics,
  };

  retrainCount++;
  lastRetrainTime = Date.now();
  lastPhysicsStoreSize = getAllPhysicsResults().length;
}

function shouldRetrain(): boolean {
  if (!lambdaEnsemble) return true;

  const currentPhysicsSize = getAllPhysicsResults().length;
  const newResults = currentPhysicsSize - lastPhysicsStoreSize;
  if (newResults >= MIN_RETRAIN_SAMPLES) return true;

  if (Date.now() - lastRetrainTime > RETRAIN_INTERVAL_MS && newResults > 0) return true;

  if (predictionErrors.length >= 10) {
    const recentErrors = predictionErrors.slice(-10);
    const avgError = recentErrors.reduce((s, e) => s + Math.abs(e.predicted - e.actual), 0) / recentErrors.length;
    if (avgError > 0.3) return true;
  }

  return false;
}

export function predictLambda(formula: string, pressure: number = 0): LambdaPrediction {
  totalPredictions++;

  const physicsResult = getPhysicsResult(formula);
  if (physicsResult && physicsResult.lambda > 0) {
    tierBreakdown["verified-dfpt"]++;
    return {
      lambda: physicsResult.lambda,
      uncertainty: 0.05,
      tier: "verified-dfpt",
      confidence: physicsResult.tier === "full-dft" ? 0.95 : physicsResult.tier === "xtb" ? 0.80 : 0.65,
      features: {
        dosAtEF: physicsResult.dosAtEF,
        pressure: physicsResult.pressure,
      },
    };
  }

  if (shouldRetrain()) {
    try { trainLambdaRegressor(); } catch {}
  }

  if (lambdaEnsemble && lambdaEnsemble.models.length > 0) {
    try {
      const features = extractLambdaFeatures(formula, pressure);
      const x = featuresToArray(features);

      if (!x.some(v => !Number.isFinite(v))) {
        const predictions = lambdaEnsemble.models.map(m => predictWithModel(m, x));
        const mean = predictions.reduce((s, v) => s + v, 0) / predictions.length;
        const variance = predictions.reduce((s, v) => s + (v - mean) ** 2, 0) / predictions.length;
        const std = Math.sqrt(variance);

        const lambda = Math.max(0, Math.min(4.5, mean));
        const normalizedUncertainty = lambda > 0 ? std / lambda : 1;
        const confidence = Math.max(0.1, Math.min(0.9, 1 - normalizedUncertainty));

        tierBreakdown["ml-regression"]++;
        return {
          lambda,
          uncertainty: std,
          tier: "ml-regression",
          confidence,
          features,
          perModelPredictions: predictions.map(p => Math.round(p * 10000) / 10000),
        };
      }
    } catch {}
  }

  try {
    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula);

    tierBreakdown["physics-engine"]++;
    return {
      lambda: coupling.lambda,
      uncertainty: coupling.lambda * 0.3,
      tier: "physics-engine",
      confidence: 0.4,
      features: {
        dosAtEF: electronic.densityOfStatesAtFermi,
        metallicity: electronic.metallicity,
        debyeTemp: phonon.debyeTemperature,
      },
    };
  } catch {
    tierBreakdown["physics-engine"]++;
    return {
      lambda: 0.5,
      uncertainty: 0.5,
      tier: "physics-engine",
      confidence: 0.1,
      features: {},
    };
  }
}

export function recordLambdaValidation(formula: string, predictedLambda: number, actualLambda: number): void {
  predictionErrors.push({ predicted: predictedLambda, actual: actualLambda, formula });
  if (predictionErrors.length > 100) {
    predictionErrors = predictionErrors.slice(-50);
  }
}

export function getLambdaRegressorStats() {
  const physicsResults = getAllPhysicsResults();
  const superconWithLambda = SUPERCON_TRAINING_DATA.filter(e => e.lambda != null && e.lambda > 0).length;

  return {
    modelTrained: lambdaEnsemble !== null,
    ensembleSize: lambdaEnsemble?.models.length ?? 0,
    datasetSize: lambdaEnsemble?.datasetSize ?? 0,
    superconLambdaEntries: superconWithLambda,
    physicsStoreEntries: physicsResults.length,
    physicsStoreLambdaEntries: physicsResults.filter(r => r.lambda > 0).length,
    metrics: lambdaEnsemble?.metrics ?? { r2: 0, mae: 0, rmse: 0 },
    retrainCount,
    lastRetrainTime,
    totalPredictions,
    tierBreakdown,
    featureNames: LAMBDA_FEATURE_NAMES,
    recentErrors: predictionErrors.slice(-10).map(e => ({
      formula: e.formula,
      predicted: Math.round(e.predicted * 1000) / 1000,
      actual: Math.round(e.actual * 1000) / 1000,
      absError: Math.round(Math.abs(e.predicted - e.actual) * 1000) / 1000,
    })),
    tiers: [
      { name: "Tier 3: Verified DFPT", source: "physics-results-store", cost: "expensive", fidelity: "highest", description: "Full DFPT or quantum engine pipeline lambda" },
      { name: "Tier 1: ML Regression", source: "lambda-regressor GBM", cost: "milliseconds", fidelity: "medium", description: "Trained regression from DFPT results" },
      { name: "Fallback: Physics Engine", source: "physics-engine rules", cost: "cheap", fidelity: "low", description: "Hard-coded rules (legacy)" },
    ],
  };
}

export function initLambdaRegressor(): void {
  try {
    trainLambdaRegressor();
  } catch {}
}
