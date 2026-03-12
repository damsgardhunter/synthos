import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { getAllPhysicsResults, getPhysicsResult, getPhysicsStoreSize, type PhysicsResult } from "./physics-results-store";
import {
  getElementData,
  getCompositionWeightedProperty,
  getAverageMass,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "./elemental-data";
import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  parseFormulaElements,
} from "./physics-engine";

function seededRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return (s >>> 0) / 4294967296;
  };
}

const ENSEMBLE_SEEDS = [314159, 271828, 173205];

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

const LAMBDA_ENSEMBLE_SIZE = 2;
const MIN_RETRAIN_SAMPLES = 20;
const RETRAIN_INTERVAL_MS = 30 * 60 * 1000;

let lambdaEnsemble: LambdaEnsemble | null = null;
let lastPhysicsStoreSize = 0;
let totalPredictions = 0;
let tierBreakdown: Record<string, number> = { "verified-dfpt": 0, "ml-regression": 0, "physics-engine": 0, "heuristic": 0 };
let retrainCount = 0;
let lastRetrainTime = 0;
let predictionErrors: { predicted: number; actual: number; formula: string }[] = [];
let isTraining = false;

function computeENSpread(formula: string): number {
  const elements = parseFormula(formula);
  const enValues = elements.map(e => getElementData(e)?.paulingElectronegativity ?? 1.5);
  return enValues.length > 1 ? Math.max(...enValues) - Math.min(...enValues) : 0;
}

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

const FAMILY_PHONON_DEFAULTS: Record<string, { debyeTemp: number; omegaLog: number }> = {
  hydride:     { debyeTemp: 1500, omegaLog: 800 },
  superhydride:{ debyeTemp: 2200, omegaLog: 1200 },
  cuprate:     { debyeTemp: 400,  omegaLog: 250 },
  pnictide:    { debyeTemp: 300,  omegaLog: 200 },
  boride:      { debyeTemp: 900,  omegaLog: 500 },
  carbide:     { debyeTemp: 700,  omegaLog: 400 },
  nitride:     { debyeTemp: 600,  omegaLog: 350 },
  chalcogenide:{ debyeTemp: 250,  omegaLog: 160 },
  heavyfermion:{ debyeTemp: 150,  omegaLog: 80 },
  default:     { debyeTemp: 300,  omegaLog: 200 },
};

function detectPhononFamily(elements: string[], counts: Record<string, number>, totalAtoms: number): string {
  const hCount = counts["H"] || 0;
  const hRatio = hCount / Math.max(1, totalAtoms - hCount);
  if (hRatio >= 6) return "superhydride";
  if (hCount > 0 && hRatio >= 1) return "hydride";
  if (elements.includes("Cu") && elements.includes("O")) return "cuprate";
  if (elements.some(e => ["As", "P", "Sb"].includes(e)) && elements.includes("Fe")) return "pnictide";
  if (elements.includes("B") && !elements.includes("O")) return "boride";
  if (elements.includes("C") && !elements.includes("O")) return "carbide";
  if (elements.includes("N") && !elements.includes("O")) return "nitride";
  if (elements.some(e => ["S", "Se", "Te"].includes(e))) return "chalcogenide";
  if (elements.some(e => ["Ce", "U", "Yb", "Sm"].includes(e))) return "heavyfermion";
  return "default";
}

function extractLambdaFeatures(formula: string, pressure: number = 0): Record<string, number> {
  const elements = parseFormula(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  const electronic = computeElectronicStructure(formula);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula);

  const avgMass = getAverageMass(counts);
  const family = detectPhononFamily(elements, counts, totalAtoms);
  const familyDefaults = FAMILY_PHONON_DEFAULTS[family] || FAMILY_PHONON_DEFAULTS.default;
  const debyeTemp0 = phonon.debyeTemperature > 50 ? phonon.debyeTemperature : familyDefaults.debyeTemp;
  const hopfieldEta = getCompositionWeightedProperty(counts, "mcMillanHopfieldEta") ?? 0;
  const avgBulk0 = getCompositionWeightedProperty(counts, "bulkModulus") ?? 0;
  const avgGamma = getCompositionWeightedProperty(counts, "sommerfeldGamma") ?? 0;

  const gruneisen = 1.5;
  const bPrime = 4.0;
  const pressureRatio = avgBulk0 > 0 ? pressure / avgBulk0 : 0;
  const compressionFactor = Math.pow(1 + bPrime * pressureRatio, 1 / bPrime);
  const bulkPressureScale = compressionFactor;
  const debyePressureScale = Math.pow(compressionFactor, gruneisen);
  const avgBulk = avgBulk0 * Math.max(1, bulkPressureScale);
  const debyeTemp = debyeTemp0 * Math.max(1, debyePressureScale);

  const enValues = elements.map(e => getElementData(e)?.paulingElectronegativity ?? 1.5);
  const enSpread = enValues.length > 1 ? Math.max(...enValues) - Math.min(...enValues) : 0;

  const hCount = counts["H"] || 0;
  const ALKALI_ALKALINE = new Set(["Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"]);
  const metalAtomCount = elements
    .filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e) || ALKALI_ALKALINE.has(e))
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
  let totalValenceElectrons = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) {
      const frac = (counts[el] || 1) / totalAtoms;
      const z = data.atomicNumber;
      let dCount = 0;
      if (z >= 21 && z <= 30) dCount = z - 20;
      else if (z >= 39 && z <= 48) dCount = z - 38;
      else if (z >= 57 && z <= 71) dCount = Math.min(z - 56, 10);
      else if (z >= 72 && z <= 80) dCount = z - 70;
      else if (z >= 89 && z <= 103) dCount = Math.min(z - 88, 10);
      dElectrons += dCount * frac;
      totalValenceElectrons += data.valenceElectrons * frac;
    }
  }
  orbitalDFraction = totalValenceElectrons > 0 ? Math.min(1, dElectrons / totalValenceElectrons) : 0;

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

function preSortFeatures(X: number[][], nSamples: number, nFeatures: number): number[][] {
  const sorted: number[][] = [];
  for (let fi = 0; fi < nFeatures; fi++) {
    const order = Array.from({ length: nSamples }, (_, i) => i);
    order.sort((a, b) => X[a][fi] - X[b][fi]);
    sorted.push(order);
  }
  return sorted;
}

function findBestSplit(
  X: number[][],
  residuals: number[],
  indices: number[],
  featureIndex: number,
  preSorted?: number[][]
): { threshold: number; improvement: number; leftIndices: number[]; rightIndices: number[] } {
  let pairs: { val: number; res: number; idx: number }[];

  if (preSorted && indices.length > 50) {
    const indexSet = new Set(indices);
    pairs = [];
    for (const idx of preSorted[featureIndex]) {
      if (indexSet.has(idx)) {
        pairs.push({ val: X[idx][featureIndex], res: residuals[idx], idx });
      }
    }
  } else {
    pairs = indices
      .map(i => ({ val: X[i][featureIndex], res: residuals[i], idx: i }))
      .sort((a, b) => a.val - b.val);
  }

  const n = pairs.length;
  const totalSum = pairs.reduce((s, p) => s + p.res, 0);
  const parentScore = (totalSum * totalSum) / n;

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

    const improvement = (leftSum * leftSum) / leftCount + (rightSum * rightSum) / rightCount - parentScore;
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
  maxDepth: number,
  preSorted?: number[][]
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
    const split = findBestSplit(X, residuals, indices, fi, preSorted);
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
    left: buildTree(X, residuals, bestLeft, depth + 1, maxDepth, preSorted),
    right: buildTree(X, residuals, bestRight, depth + 1, maxDepth, preSorted),
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
  maxDepth: number = 4,
  seed: number = 42
): LambdaGBModel {
  const n = X.length;
  const nFeatures = X[0]?.length ?? 0;
  const rng = seededRng(seed);
  const basePrediction = y.reduce((s, v) => s + v, 0) / n;
  const predictions = new Array(n).fill(basePrediction);
  const trees: LambdaTreeNode[] = [];
  const preSorted = nFeatures > 0 ? preSortFeatures(X, n, nFeatures) : undefined;

  const valSize = Math.max(2, Math.floor(n * 0.2));
  const shuffled = Array.from({ length: n }, (_, i) => i);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const valIdx = new Set(shuffled.slice(0, valSize));
  const trainIdx = shuffled.slice(valSize);
  const valPreds = new Array(n).fill(basePrediction);

  let bestValMSE = Infinity;
  let patience = 10;
  let staleRounds = 0;

  for (let t = 0; t < nEstimators; t++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);

    const tree = buildTree(X, residuals, trainIdx, 0, maxDepth, preSorted);
    if (typeof tree === "number") break;

    trees.push(tree);

    for (let i = 0; i < n; i++) {
      const update = learningRate * predictTree(tree, X[i]);
      predictions[i] += update;
      valPreds[i] += update;
    }

    let valSSE = 0;
    let valCount = 0;
    for (const vi of valIdx) {
      valSSE += (y[vi] - valPreds[vi]) ** 2;
      valCount++;
    }
    const valMSE = valCount > 0 ? valSSE / valCount : 0;

    if (valMSE < bestValMSE - 1e-6) {
      bestValMSE = valMSE;
      staleRounds = 0;
    } else {
      staleRounds++;
      if (staleRounds >= patience) break;
    }
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

function bootstrapSample(X: number[][], y: number[], rng: () => number): { X: number[][]; y: number[]; oobIndices: number[] } {
  const n = X.length;
  const size = Math.floor(n * 0.8);
  const bsX: number[][] = [];
  const bsY: number[] = [];
  const selectedSet = new Set<number>();
  for (let i = 0; i < size; i++) {
    const idx = Math.floor(rng() * n);
    bsX.push(X[idx]);
    bsY.push(y[idx]);
    selectedSet.add(idx);
  }
  const oobIndices: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!selectedSet.has(i)) oobIndices.push(i);
  }
  return { X: bsX, y: bsY, oobIndices };
}

async function buildTrainingDataAsync(): Promise<{ X: number[][]; y: number[]; formulas: string[] }> {
  const X: number[][] = [];
  const y: number[] = [];
  const formulas: string[] = [];
  const keyIndex = new Map<string, number>();

  let count = 0;
  for (const entry of SUPERCON_TRAINING_DATA) {
    if (entry.lambda == null || entry.lambda <= 0) continue;
    try {
      const pressure = entry.pressureGPa ?? 0;
      const features = extractLambdaFeatures(entry.formula, pressure);
      const vec = featuresToArray(features);
      if (vec.some(v => !Number.isFinite(v))) continue;
      const key = `${entry.formula}@${Math.round(pressure)}GPa`;
      const idx = X.length;
      X.push(vec);
      y.push(entry.lambda);
      formulas.push(entry.formula);
      keyIndex.set(key, idx);
    } catch {
      continue;
    }
    if (++count % 10 === 0) await new Promise<void>(r => setTimeout(r, 20));
  }

  const physicsResults = getAllPhysicsResults();
  for (const result of physicsResults) {
    if (result.lambda <= 0) continue;
    const key = `${result.formula}@${Math.round(result.pressure)}GPa`;
    const existing = keyIndex.get(key);
    if (existing !== undefined && result.tier !== "full-dft") continue;

    try {
      const features = extractLambdaFeatures(result.formula, result.pressure);
      features.dosAtEF = result.dosAtEF > 0 ? result.dosAtEF : features.dosAtEF;
      const vec = featuresToArray(features);
      if (vec.some(v => !Number.isFinite(v))) continue;

      if (existing !== undefined) {
        X[existing] = vec;
        y[existing] = result.lambda;
      } else {
        const idx = X.length;
        X.push(vec);
        y.push(result.lambda);
        formulas.push(result.formula);
        keyIndex.set(key, idx);
      }
    } catch {
      continue;
    }
    if (count % 10 === 0) await new Promise<void>(r => setTimeout(r, 20));
    count++;
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

export async function trainLambdaRegressor(): Promise<void> {
  const { X, y, formulas } = await buildTrainingDataAsync();
  if (X.length < 5) return;

  const models: LambdaGBModel[] = [];
  for (let i = 0; i < LAMBDA_ENSEMBLE_SIZE; i++) {
    const rng = seededRng(ENSEMBLE_SEEDS[i] + X.length);
    const { X: bsX, y: bsY } = bootstrapSample(X, y, rng);
    const model = trainLambdaGBM(bsX, bsY, 80, 0.10, 4, ENSEMBLE_SEEDS[i]);
    models.push(model);
    await new Promise<void>(r => setTimeout(r, 10));
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
  lastPhysicsStoreSize = getPhysicsStoreSize();
}

function shouldRetrain(): boolean {
  if (!lambdaEnsemble) return true;

  const currentPhysicsSize = getPhysicsStoreSize();
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

  if (shouldRetrain() && !isTraining && lambdaEnsemble !== null) {
    isTraining = true;
    trainLambdaRegressor().catch(() => {}).finally(() => { isTraining = false; });
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
        const mae = lambdaEnsemble.metrics?.mae ?? 0.2;
        const calibratedStd = Math.max(std, mae * 0.5);
        const confidence = Math.max(0.1, Math.min(0.9,
          1 - (calibratedStd / Math.max(mae * 3, 0.6))
        ));

        tierBreakdown["ml-regression"]++;
        return {
          lambda,
          uncertainty: calibratedStd,
          tier: "ml-regression",
          confidence,
          features,
          perModelPredictions: predictions.map(p => Math.round(p * 10000) / 10000),
        };
      }
    } catch {}
  }

  if (!lambdaEnsemble) {
    tierBreakdown["heuristic"]++;
    return {
      lambda: 0.5,
      uncertainty: 0.4,
      tier: "heuristic",
      confidence: 0.1,
      features: {},
    };
  }

  try {
    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula);

    const enSpread = computeENSpread(formula);
    const baseMultiplier = 0.3;
    const ionicPenalty = enSpread > 1.0 ? Math.min(0.25, (enSpread - 1.0) * 0.15) : 0;
    const metallicityBonus = electronic.metallicity > 0.7 ? -0.05 : 0;
    const uncertaintyMultiplier = Math.min(0.55, baseMultiplier + ionicPenalty + metallicityBonus);

    tierBreakdown["physics-engine"]++;
    return {
      lambda: coupling.lambda,
      uncertainty: coupling.lambda * uncertaintyMultiplier,
      tier: "physics-engine",
      confidence: Math.max(0.15, 0.5 - ionicPenalty * 2),
      features: {
        dosAtEF: electronic.densityOfStatesAtFermi,
        metallicity: electronic.metallicity,
        debyeTemp: phonon.debyeTemperature,
        enSpread,
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

export async function initLambdaRegressor(): Promise<void> {
  try {
    await trainLambdaRegressor();
  } catch {}
}
