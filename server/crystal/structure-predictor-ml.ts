import { computeCompositionFeatures, compositionFeatureVector } from "../learning/composition-features";
import { getGraphFeatureVector, buildGraphFromStructure, buildGraphFromFormula } from "./crystal-graph-builder";
import { getTrainingData, getDatasetStats, type CrystalStructureEntry } from "./crystal-structure-dataset";

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

interface ClassifierModel {
  classes: string[];
  models: GBModel[];
}

interface StructureMLModels {
  spacegroupClassifier: ClassifierModel | null;
  crystalSystemClassifier: ClassifierModel | null;
  latticeRegressor: { a: GBModel; b: GBModel; c: GBModel; alpha: GBModel; beta: GBModel; gamma: GBModel } | null;
  prototypeClassifier: ClassifierModel | null;
  trainedAt: number;
  datasetSize: number;
  metrics: {
    spacegroupAccuracy: number;
    crystalSystemAccuracy: number;
    latticeMAE: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
    prototypeAccuracy: number;
  };
}

export interface StructurePredictionML {
  formula: string;
  topSpacegroups: { spacegroup: number; symbol: string; probability: number }[];
  crystalSystem: { predicted: string; probabilities: Record<string, number> };
  latticeParams: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  prototype: { predicted: string; probabilities: Record<string, number> };
  confidence: number;
  source: "ml" | "heuristic";
}

let models: StructureMLModels | null = null;
let lastTrainedDatasetSize = 0;
let trainCount = 0;

const SPACEGROUP_SYMBOLS: Record<number, string> = {
  225: "Fm-3m", 229: "Im-3m", 227: "Fd-3m", 221: "Pm-3m", 223: "Pm-3n",
  191: "P6/mmm", 194: "P6_3/mmc", 139: "I4/mmm", 129: "P4/nmm", 123: "P4/mmm",
  99: "P4mm", 141: "I4_1/amd", 62: "Pnma", 47: "Pmmm", 166: "R-3m",
  216: "F-43m", 217: "I-43m", 230: "Ia-3d", 12: "C2/m", 15: "C2/c",
  2: "P-1", 148: "R-3", 167: "R-3c",
};

function findBestSplit(
  X: number[][], residuals: number[], indices: number[], featureIndex: number
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
  X: number[][], residuals: number[], indices: number[], depth: number, maxDepth: number
): TreeNode | number {
  if (depth >= maxDepth || indices.length < 4) {
    return indices.reduce((s, i) => s + residuals[i], 0) / indices.length;
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
    return indices.reduce((s, i) => s + residuals[i], 0) / indices.length;
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

function trainGBM(X: number[][], y: number[], nEstimators = 100, lr = 0.1, maxDepth = 4): GBModel {
  const n = X.length;
  const basePrediction = y.reduce((s, v) => s + v, 0) / n;
  const predictions = new Array(n).fill(basePrediction);
  const trees: TreeNode[] = [];
  const allIndices = Array.from({ length: n }, (_, i) => i);

  for (let t = 0; t < nEstimators; t++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);
    const tree = buildTree(X, residuals, allIndices, 0, maxDepth);
    if (typeof tree === "number" && Math.abs(tree) < 1e-8) break;
    trees.push(typeof tree === "number" ? { featureIndex: 0, threshold: 0, left: tree, right: tree } : tree);

    for (let i = 0; i < n; i++) {
      predictions[i] += lr * predictTree(tree, X[i]);
    }
  }

  return { trees, learningRate: lr, basePrediction };
}

function predictGBM(model: GBModel, x: number[]): number {
  let pred = model.basePrediction;
  for (const tree of model.trees) {
    pred += model.learningRate * predictTree(tree, x);
  }
  return pred;
}

function trainOneVsAllClassifier(X: number[][], labels: string[], nEstimators = 80, lr = 0.1, maxDepth = 3): ClassifierModel {
  const classes = Array.from(new Set(labels));
  const classModels: GBModel[] = [];

  for (const cls of classes) {
    const binaryY = labels.map(l => l === cls ? 1.0 : 0.0);
    const model = trainGBM(X, binaryY, nEstimators, lr, maxDepth);
    classModels.push(model);
  }

  return { classes, models: classModels };
}

function predictClassifier(classifier: ClassifierModel, x: number[]): { predicted: string; probabilities: Record<string, number> } {
  const scores = classifier.models.map(m => predictGBM(m, x));
  const maxScore = Math.max(...scores);
  const expScores = scores.map(s => Math.exp(s - maxScore));
  const sumExp = expScores.reduce((s, v) => s + v, 0);
  const probs: Record<string, number> = {};

  for (let i = 0; i < classifier.classes.length; i++) {
    probs[classifier.classes[i]] = Math.round((expScores[i] / sumExp) * 10000) / 10000;
  }

  const bestIdx = scores.indexOf(Math.max(...scores));
  return { predicted: classifier.classes[bestIdx], probabilities: probs };
}

function extractFeatureVector(formula: string, entry?: CrystalStructureEntry): number[] {
  const compFeatures = computeCompositionFeatures(formula);
  const compVec = compositionFeatureVector(compFeatures);

  let graphVec: number[];
  if (entry && entry.atomicPositions && entry.atomicPositions.length > 0) {
    try {
      const graph = buildGraphFromStructure(entry);
      graphVec = getGraphFeatureVector(graph);
    } catch {
      const graph = buildGraphFromFormula(formula);
      graphVec = getGraphFeatureVector(graph);
    }
  } else {
    const graph = buildGraphFromFormula(formula);
    graphVec = getGraphFeatureVector(graph);
  }

  return [...compVec, ...graphVec];
}

function computeClassifierAccuracy(classifier: ClassifierModel, X: number[][], labels: string[]): number {
  if (X.length === 0) return 0;
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const pred = predictClassifier(classifier, X[i]);
    if (pred.predicted === labels[i]) correct++;
  }
  return Math.round((correct / X.length) * 10000) / 10000;
}

function computeRegressorMAE(model: GBModel, X: number[][], y: number[]): number {
  if (X.length === 0) return 0;
  let totalErr = 0;
  for (let i = 0; i < X.length; i++) {
    totalErr += Math.abs(predictGBM(model, X[i]) - y[i]);
  }
  return Math.round((totalErr / X.length) * 10000) / 10000;
}

export function trainStructurePredictor(): void {
  const dataset = getTrainingData();
  if (dataset.length < 10) return;

  const X: number[][] = [];
  const sgLabels: string[] = [];
  const csLabels: string[] = [];
  const protoLabels: string[] = [];
  const latticeA: number[] = [];
  const latticeB: number[] = [];
  const latticeC: number[] = [];
  const latticeAlpha: number[] = [];
  const latticeBeta: number[] = [];
  const latticeGamma: number[] = [];

  for (const entry of dataset) {
    try {
      const vec = extractFeatureVector(entry.formula, entry);
      if (vec.some(v => !Number.isFinite(v))) continue;

      X.push(vec);
      sgLabels.push(String(entry.spacegroup));
      csLabels.push(entry.crystalSystem);
      protoLabels.push(entry.prototype);
      latticeA.push(entry.lattice.a);
      latticeB.push(entry.lattice.b);
      latticeC.push(entry.lattice.c);
      latticeAlpha.push(entry.lattice.alpha);
      latticeBeta.push(entry.lattice.beta);
      latticeGamma.push(entry.lattice.gamma);
    } catch {
      continue;
    }
  }

  if (X.length < 10) return;

  const spacegroupClassifier = trainOneVsAllClassifier(X, sgLabels, 15, 0.15, 3);
  const crystalSystemClassifier = trainOneVsAllClassifier(X, csLabels, 20, 0.15, 3);
  const prototypeClassifier = trainOneVsAllClassifier(X, protoLabels, 15, 0.15, 3);

  const latticeRegressor = {
    a: trainGBM(X, latticeA, 25, 0.12, 3),
    b: trainGBM(X, latticeB, 25, 0.12, 3),
    c: trainGBM(X, latticeC, 25, 0.12, 3),
    alpha: trainGBM(X, latticeAlpha, 15, 0.1, 3),
    beta: trainGBM(X, latticeBeta, 15, 0.1, 3),
    gamma: trainGBM(X, latticeGamma, 15, 0.1, 3),
  };

  const sgAcc = computeClassifierAccuracy(spacegroupClassifier, X, sgLabels);
  const csAcc = computeClassifierAccuracy(crystalSystemClassifier, X, csLabels);
  const protoAcc = computeClassifierAccuracy(prototypeClassifier, X, protoLabels);

  const latticeMAE = {
    a: computeRegressorMAE(latticeRegressor.a, X, latticeA),
    b: computeRegressorMAE(latticeRegressor.b, X, latticeB),
    c: computeRegressorMAE(latticeRegressor.c, X, latticeC),
    alpha: computeRegressorMAE(latticeRegressor.alpha, X, latticeAlpha),
    beta: computeRegressorMAE(latticeRegressor.beta, X, latticeBeta),
    gamma: computeRegressorMAE(latticeRegressor.gamma, X, latticeGamma),
  };

  models = {
    spacegroupClassifier,
    crystalSystemClassifier,
    latticeRegressor,
    prototypeClassifier,
    trainedAt: Date.now(),
    datasetSize: X.length,
    metrics: {
      spacegroupAccuracy: sgAcc,
      crystalSystemAccuracy: csAcc,
      latticeMAE,
      prototypeAccuracy: protoAcc,
    },
  };

  lastTrainedDatasetSize = dataset.length;
  trainCount++;
}

function shouldRetrain(): boolean {
  if (!models) return true;
  const currentSize = getTrainingData().length;
  if (currentSize - lastTrainedDatasetSize >= 50) return true;
  return false;
}

let trainingInProgress = false;

function ensureTrained(): void {
}

export function isStructurePredictorReady(): boolean {
  return !!models;
}

export function trainStructurePredictorBackground(): void {
  if (models || trainingInProgress) return;
  trainingInProgress = true;
  setTimeout(() => {
    try {
      console.log(`[StructurePredictor] Background training started`);
      const t0 = Date.now();
      trainStructurePredictor();
      console.log(`[StructurePredictor] Background training completed in ${Date.now() - t0}ms`);
    } catch (e) {
      console.log(`[StructurePredictor] Background training failed: ${e}`);
    }
    trainingInProgress = false;
  }, 120000);
}

export function predictStructure(formula: string): StructurePredictionML {
  ensureTrained();

  if (!models || !models.spacegroupClassifier || !models.crystalSystemClassifier || !models.latticeRegressor || !models.prototypeClassifier) {
    return heuristicFallback(formula);
  }

  try {
    const vec = extractFeatureVector(formula);
    if (vec.some(v => !Number.isFinite(v))) return heuristicFallback(formula);

    const sgPred = predictClassifier(models.spacegroupClassifier, vec);
    const csPred = predictClassifier(models.crystalSystemClassifier, vec);
    const protoPred = predictClassifier(models.prototypeClassifier, vec);

    const sortedSG = Object.entries(sgPred.probabilities)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3);

    const topSpacegroups = sortedSG.map(([sg, prob]) => {
      const sgNum = parseInt(sg);
      return {
        spacegroup: sgNum,
        symbol: SPACEGROUP_SYMBOLS[sgNum] || `SG-${sgNum}`,
        probability: prob,
      };
    });

    const a = Math.max(1.0, predictGBM(models.latticeRegressor.a, vec));
    const b = Math.max(1.0, predictGBM(models.latticeRegressor.b, vec));
    const c = Math.max(1.0, predictGBM(models.latticeRegressor.c, vec));
    const alpha = Math.max(30, Math.min(150, predictGBM(models.latticeRegressor.alpha, vec)));
    const beta = Math.max(30, Math.min(150, predictGBM(models.latticeRegressor.beta, vec)));
    const gamma = Math.max(30, Math.min(150, predictGBM(models.latticeRegressor.gamma, vec)));

    const topProb = topSpacegroups[0]?.probability ?? 0;
    const csProb = Math.max(...Object.values(csPred.probabilities));
    const confidence = Math.round(((topProb + csProb) / 2) * 10000) / 10000;

    return {
      formula,
      topSpacegroups,
      crystalSystem: csPred,
      latticeParams: {
        a: Math.round(a * 1000) / 1000,
        b: Math.round(b * 1000) / 1000,
        c: Math.round(c * 1000) / 1000,
        alpha: Math.round(alpha * 100) / 100,
        beta: Math.round(beta * 100) / 100,
        gamma: Math.round(gamma * 100) / 100,
      },
      prototype: protoPred,
      confidence,
      source: "ml",
    };
  } catch {
    return heuristicFallback(formula);
  }
}

function heuristicFallback(formula: string): StructurePredictionML {
  const compFeatures = computeCompositionFeatures(formula);

  let cs = "cubic";
  if (compFeatures.enRange > 1.5) cs = "tetragonal";
  else if (compFeatures.enRange > 1.0) cs = "orthorhombic";
  else if (compFeatures.nAtoms > 10) cs = "hexagonal";

  let proto = "FCC";
  if (compFeatures.dElectronFrac > 0.5) proto = "BCC";
  else if (compFeatures.pElectronFrac > 0.3) proto = "diamond";
  else if (compFeatures.nAtoms > 5 && compFeatures.enRange > 0.5) proto = "perovskite";

  const a = compFeatures.latticeConstMean > 0 ? compFeatures.latticeConstMean : 4.0;
  const angles = cs === "hexagonal" ? { alpha: 90, beta: 90, gamma: 120 } : { alpha: 90, beta: 90, gamma: 90 };

  return {
    formula,
    topSpacegroups: [{ spacegroup: 225, symbol: "Fm-3m", probability: 0.3 }],
    crystalSystem: { predicted: cs, probabilities: { [cs]: 0.3 } },
    latticeParams: { a: Math.round(a * 1000) / 1000, b: Math.round(a * 1000) / 1000, c: Math.round(a * 1000) / 1000, ...angles },
    prototype: { predicted: proto, probabilities: { [proto]: 0.3 } },
    confidence: 0.2,
    source: "heuristic",
  };
}

export function getStructurePredictorStats() {
  const dsStats = getDatasetStats();
  return {
    modelTrained: models !== null,
    datasetSize: models?.datasetSize ?? 0,
    totalDatasetEntries: dsStats.totalCount,
    trainCount,
    trainedAt: models?.trainedAt ?? 0,
    metrics: models?.metrics ?? {
      spacegroupAccuracy: 0,
      crystalSystemAccuracy: 0,
      latticeMAE: { a: 0, b: 0, c: 0, alpha: 0, beta: 0, gamma: 0 },
      prototypeAccuracy: 0,
    },
    spacegroupClasses: models?.spacegroupClassifier?.classes.length ?? 0,
    crystalSystemClasses: models?.crystalSystemClassifier?.classes.length ?? 0,
    prototypeClasses: models?.prototypeClassifier?.classes.length ?? 0,
    datasetBreakdown: dsStats.byCrystalSystem,
    retrainThreshold: 50,
  };
}

export function initStructurePredictorML(): void {
  trainStructurePredictorBackground();
}
