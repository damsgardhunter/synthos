import { SUPERCON_TRAINING_DATA } from "../learning/supercon-dataset";
import { getAllPhysicsResults, type PhysicsResult } from "../learning/physics-results-store";
import {
  getElementData,
  getCompositionWeightedProperty,
  getAverageMass,
  isTransitionMetal,
} from "../learning/elemental-data";
import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  parseFormulaElements,
} from "../learning/physics-engine";
import { predictLambda } from "../learning/lambda-regressor";

interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | number;
  right: TreeNode | number;
}

interface GBModel {
  trees: TreeNode[];
  learningRate: number;
  basePrediction: number;
}

interface PhononSurrogateModels {
  omegaLog: GBModel;
  debyeTemp: GBModel;
  maxPhononFreq: GBModel;
  phononStability: GBModel;
  trainedAt: number;
  datasetSize: number;
  metrics: { omegaLogMAE: number; debyeTempMAE: number; maxFreqMAE: number; stabilityAccuracy: number };
}

export interface PhononSurrogatePrediction {
  omegaLog: number;
  debyeTemp: number;
  maxPhononFreq: number;
  phononStability: boolean;
  stabilityProbability: number;
  confidence: number;
  tier: "phonon-surrogate" | "physics-engine-fallback";
}

const FEATURE_NAMES = [
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
  "lambdaPrediction",
];

let surrogateModels: PhononSurrogateModels | null = null;
let surrogateDisabled = false;
let lastTrainSize = 0;
let lastKnownResultCount = 0;
let totalPredictions = 0;
let tierBreakdown = { hits: 0, misses: 0, fallbacks: 0 };

export function notifyNewPhysicsResult(): void {
  lastKnownResultCount++;
}

function parseFormula(formula: string): string[] {
  const matches = formula.match(/[A-Z][a-z]?/g);
  return matches ? Array.from(new Set(matches)) : [];
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

function extractFeatures(formula: string, pressure: number = 0): Record<string, number> {
  const elements = parseFormula(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  const electronic = computeElectronicStructure(formula);
  const phonon = computePhononSpectrum(formula, electronic);

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
  let dCount = 0;
  let totalElectrons = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) {
      const frac = (counts[el] || 1) / totalAtoms;
      if (isTransitionMetal(el)) dCount += data.valenceElectrons * frac;
      totalElectrons += data.atomicNumber * frac;
    }
  }
  orbitalDFraction = totalElectrons > 0 ? Math.min(1, dCount / Math.max(1, totalElectrons * 0.5)) : 0;

  const lambdaPred = predictLambda(formula, pressure);

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
    lambdaPrediction: lambdaPred.lambda,
  };
}

function featuresToArray(features: Record<string, number>): number[] {
  return FEATURE_NAMES.map(name => {
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
): TreeNode | number {
  if (depth >= maxDepth || indices.length < 10) {
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

function predictTree(tree: TreeNode | number, x: number[]): number {
  if (typeof tree === "number") return tree;
  return x[tree.featureIndex] <= tree.threshold
    ? predictTree(tree.left, x)
    : predictTree(tree.right, x);
}

function trainGBM(
  X: number[][],
  y: number[],
  nEstimators: number = 120,
  learningRate: number = 0.08,
  maxDepth: number = 4
): GBModel {
  const n = X.length;
  const basePrediction = y.reduce((s, v) => s + v, 0) / n;
  const predictions = new Array(n).fill(basePrediction);
  const trees: TreeNode[] = [];
  const allIndices = Array.from({ length: n }, (_, i) => i);

  for (let t = 0; t < nEstimators; t++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);

    const tree = buildTree(X, residuals, allIndices, 0, maxDepth);
    if (typeof tree === "number") break;

    trees.push(tree);

    for (let i = 0; i < n; i++) {
      predictions[i] += learningRate * predictTree(tree, X[i]);
    }

    const mse = y.reduce((s, yi, i) => s + (yi - predictions[i]) ** 2, 0) / n;
    if (mse < 0.001) break;
  }

  return { trees, learningRate, basePrediction };
}

function predictWithModel(model: GBModel, x: number[]): number {
  let pred = model.basePrediction;
  for (const tree of model.trees) {
    pred += model.learningRate * predictTree(tree, x);
  }
  return pred;
}

interface TrainingRow {
  features: number[];
  omegaLog: number;
  debyeTemp: number;
  maxPhononFreq: number;
  phononStable: number;
}

const FAMILY_DEFAULTS: Record<string, { omegaLog: number; debyeTemp: number; maxFreq: number }> = {
  superhydride: { omegaLog: 1200, debyeTemp: 2200, maxFreq: 3500 },
  hydride:      { omegaLog: 800,  debyeTemp: 1500, maxFreq: 2000 },
  cuprate:      { omegaLog: 250,  debyeTemp: 400,  maxFreq: 600 },
  pnictide:     { omegaLog: 200,  debyeTemp: 300,  maxFreq: 450 },
  boride:       { omegaLog: 500,  debyeTemp: 900,  maxFreq: 1200 },
  default:      { omegaLog: 200,  debyeTemp: 300,  maxFreq: 500 },
};

function classifyPhononFamily(formula: string): string {
  const el = parseFormula(formula);
  const counts = parseFormulaCounts(formula);
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  const hCount = counts["H"] || 0;
  const hRatio = hCount / Math.max(1, total - hCount);
  if (hRatio >= 6) return "superhydride";
  if (hCount > 0 && hRatio >= 1) return "hydride";
  if (el.includes("Cu") && el.includes("O")) return "cuprate";
  if (el.some(e => ["As", "P", "Sb"].includes(e)) && el.includes("Fe")) return "pnictide";
  if (el.includes("B") && !el.includes("O")) return "boride";
  return "default";
}

function familyImputeOmegaLog(formula: string, raw: number): number {
  if (raw > 0) return raw;
  return (FAMILY_DEFAULTS[classifyPhononFamily(formula)] || FAMILY_DEFAULTS.default).omegaLog;
}

function familyImputeDebye(formula: string, raw: number): number {
  if (raw > 50) return raw;
  return (FAMILY_DEFAULTS[classifyPhononFamily(formula)] || FAMILY_DEFAULTS.default).debyeTemp;
}

function familyImputeMaxFreq(formula: string, raw: number): number {
  if (raw > 0) return raw;
  return (FAMILY_DEFAULTS[classifyPhononFamily(formula)] || FAMILY_DEFAULTS.default).maxFreq;
}

function buildTrainingData(): TrainingRow[] {
  const rows: TrainingRow[] = [];
  const seen = new Set<string>();

  const physicsResults = getAllPhysicsResults();
  lastKnownResultCount = physicsResults.length;

  for (const result of physicsResults) {
    if (result.omegaLog <= 0 && result.tc <= 0) continue;
    if (seen.has(result.formula)) continue;
    seen.add(result.formula);

    try {
      const feats = extractFeatures(result.formula, result.pressure);
      const vec = featuresToArray(feats);
      if (vec.some(v => !Number.isFinite(v))) continue;

      rows.push({
        features: vec,
        omegaLog: familyImputeOmegaLog(result.formula, result.omegaLog),
        debyeTemp: familyImputeDebye(result.formula, feats.debyeTemp),
        maxPhononFreq: familyImputeMaxFreq(result.formula, result.omega2 > 0 ? result.omega2 * 1.5 : 0),
        phononStable: result.phononStable ? 1 : 0,
      });
    } catch {
      continue;
    }
  }

  for (const entry of SUPERCON_TRAINING_DATA) {
    if (!entry.isSuperconductor) continue;
    if (seen.has(entry.formula)) continue;
    seen.add(entry.formula);

    try {
      const feats = extractFeatures(entry.formula, entry.pressureGPa ?? 0);
      const vec = featuresToArray(feats);
      if (vec.some(v => !Number.isFinite(v))) continue;

      const electronic = computeElectronicStructure(entry.formula);
      const phonon = computePhononSpectrum(entry.formula, electronic);

      rows.push({
        features: vec,
        omegaLog: familyImputeOmegaLog(entry.formula, phonon.logAverageFrequency),
        debyeTemp: familyImputeDebye(entry.formula, phonon.debyeTemperature),
        maxPhononFreq: familyImputeMaxFreq(entry.formula, phonon.maxPhononFrequency),
        phononStable: phonon.hasImaginaryModes ? 0 : 1,
      });
    } catch {
      continue;
    }
  }

  return rows;
}

function computeMAE(model: GBModel, X: number[][], y: number[]): number {
  if (X.length === 0) return 0;
  let totalErr = 0;
  for (let i = 0; i < X.length; i++) {
    totalErr += Math.abs(y[i] - predictWithModel(model, X[i]));
  }
  return Math.round((totalErr / X.length) * 1000) / 1000;
}

function computeAccuracy(model: GBModel, X: number[][], y: number[]): number {
  if (X.length === 0) return 0;
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const pred = predictWithModel(model, X[i]) >= 0.5 ? 1 : 0;
    if (pred === Math.round(y[i])) correct++;
  }
  return Math.round((correct / X.length) * 1000) / 1000;
}

export async function trainPhononSurrogate(): Promise<void> {
  const rows = buildTrainingData();
  if (rows.length < 10) return;

  const X = rows.map(r => r.features);
  const yOmegaLog = rows.map(r => r.omegaLog);
  const yDebyeTemp = rows.map(r => r.debyeTemp);
  const yMaxFreq = rows.map(r => r.maxPhononFreq);
  const yStability = rows.map(r => r.phononStable);

  // Yield between each heavy GBM training call (60-tree models on large datasets
  // can take 1-5s each, freezing the event loop without explicit yields).
  const omegaLogModel = trainGBM(X, yOmegaLog, 60, 0.12, 4);
  await new Promise<void>(r => setTimeout(r, 0));
  const debyeTempModel = trainGBM(X, yDebyeTemp, 60, 0.12, 4);
  await new Promise<void>(r => setTimeout(r, 0));
  const maxFreqModel = trainGBM(X, yMaxFreq, 60, 0.12, 4);
  await new Promise<void>(r => setTimeout(r, 0));
  const stabilityModel = trainGBM(X, yStability, 50, 0.08, 3);
  await new Promise<void>(r => setTimeout(r, 0));

  const omegaLogMAE = computeMAE(omegaLogModel, X, yOmegaLog);
  const meanOmegaLog = yOmegaLog.reduce((s, v) => s + v, 0) / yOmegaLog.length;
  const omegaLogRelError = meanOmegaLog > 0 ? omegaLogMAE / meanOmegaLog : 1;

  if (omegaLogRelError > 0.20) {
    surrogateDisabled = true;
    surrogateModels = null;
    lastTrainSize = rows.length;
    return;
  }

  surrogateDisabled = false;
  surrogateModels = {
    omegaLog: omegaLogModel,
    debyeTemp: debyeTempModel,
    maxPhononFreq: maxFreqModel,
    phononStability: stabilityModel,
    trainedAt: Date.now(),
    datasetSize: rows.length,
    metrics: {
      omegaLogMAE,
      debyeTempMAE: computeMAE(debyeTempModel, X, yDebyeTemp),
      maxFreqMAE: computeMAE(maxFreqModel, X, yMaxFreq),
      stabilityAccuracy: computeAccuracy(stabilityModel, X, yStability),
    },
  };

  lastTrainSize = rows.length;
}

function shouldRetrain(): boolean {
  if (!surrogateModels) return true;
  if (lastKnownResultCount - lastTrainSize >= 15) return true;
  if (Date.now() - surrogateModels.trainedAt > 30 * 60 * 1000 && lastKnownResultCount > lastTrainSize) return true;
  return false;
}

export function predictPhononProperties(formula: string, pressure: number = 0): PhononSurrogatePrediction {
  totalPredictions++;

  if (shouldRetrain() && surrogateModels !== null) {
    // Fire-and-forget: trainPhononSurrogate is now async with yields between
    // heavy GBM training calls. Don't block the prediction path.
    trainPhononSurrogate().catch(() => {});
  }

  if (surrogateModels) {
    try {
      const feats = extractFeatures(formula, pressure);
      const x = featuresToArray(feats);

      if (!x.some(v => !Number.isFinite(v))) {
        const omegaLog = Math.max(1, predictWithModel(surrogateModels.omegaLog, x));
        const debyeTemp = Math.max(10, predictWithModel(surrogateModels.debyeTemp, x));
        const maxPhononFreq = Math.max(10, predictWithModel(surrogateModels.maxPhononFreq, x));
        const stabilityRaw = predictWithModel(surrogateModels.phononStability, x);
        const stabilityProbability = Math.max(0, Math.min(1, stabilityRaw));
        const phononStability = stabilityProbability >= 0.5;

        const datasetConfidence = Math.min(0.85, 0.4 + surrogateModels.datasetSize * 0.003);
        const confidence = Number((datasetConfidence * (1 - surrogateModels.metrics.omegaLogMAE / Math.max(1, omegaLog) * 0.3)).toFixed(3));

        tierBreakdown.hits++;
        return {
          omegaLog: Math.round(omegaLog * 100) / 100,
          debyeTemp: Math.round(debyeTemp * 10) / 10,
          maxPhononFreq: Math.round(maxPhononFreq * 10) / 10,
          phononStability,
          stabilityProbability: Math.round(stabilityProbability * 1000) / 1000,
          confidence: Math.max(0.1, Math.min(0.85, confidence)),
          tier: "phonon-surrogate",
        };
      }
    } catch {}
  }

  tierBreakdown.fallbacks++;
  try {
    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);

    return {
      omegaLog: Math.round(phonon.logAverageFrequency * 100) / 100,
      debyeTemp: Math.round(phonon.debyeTemperature * 10) / 10,
      maxPhononFreq: Math.round(phonon.maxPhononFrequency * 10) / 10,
      phononStability: !phonon.hasImaginaryModes,
      stabilityProbability: phonon.hasImaginaryModes ? 0.2 : 0.8,
      confidence: 0.3,
      tier: "physics-engine-fallback",
    };
  } catch {
    return {
      omegaLog: familyImputeOmegaLog(formula, 0),
      debyeTemp: familyImputeDebye(formula, 0),
      maxPhononFreq: familyImputeMaxFreq(formula, 0),
      phononStability: true,
      stabilityProbability: 0.5,
      confidence: 0.1,
      tier: "physics-engine-fallback",
    };
  }
}

export function getPhononSurrogateStats() {
  return {
    modelTrained: surrogateModels !== null,
    surrogateDisabled,
    datasetSize: surrogateModels?.datasetSize ?? 0,
    trainedAt: surrogateModels?.trainedAt ?? 0,
    metrics: surrogateModels?.metrics ?? { omegaLogMAE: 0, debyeTempMAE: 0, maxFreqMAE: 0, stabilityAccuracy: 0 },
    totalPredictions,
    tierBreakdown,
    featureNames: FEATURE_NAMES,
    targets: ["omegaLog", "debyeTemp", "maxPhononFreq", "phononStability"],
  };
}

export function initPhononSurrogate(): void {
  trainPhononSurrogate().catch(() => {});
}
