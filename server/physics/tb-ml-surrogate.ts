import { computeTBProperties, type TBProperties } from "./tight-binding-engine";
import { getTrainingData } from "../crystal/crystal-structure-dataset";
import { getAllPhysicsResults } from "../learning/physics-results-store";
import { computeCompositionFeatures, compositionFeatureVector } from "../learning/composition-features";

function yieldToEventLoop(): Promise<void> {
  return new Promise<void>(resolve => setTimeout(resolve, 0));
}

interface TBSurrogateTarget {
  dosAtEF: number;
  bandFlatness: number;
  vanHoveProximity: number;
  hopfieldEta: number;
  lambdaProxy: number;
  effectiveMass: number;
  bandwidth: number;
  bandDegeneracy: number;
  metallicity: number;
}

const TARGET_NAMES: (keyof TBSurrogateTarget)[] = [
  "dosAtEF", "bandFlatness", "vanHoveProximity", "hopfieldEta",
  "lambdaProxy", "effectiveMass", "bandwidth", "bandDegeneracy", "metallicity",
];

interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | number;
  right: TreeNode | number;
}

interface GBMModel {
  trees: TreeNode[];
  learningRate: number;
  basePrediction: number;
}

interface TBSurrogateModels {
  models: Map<string, GBMModel>;
  trainedAt: number;
  datasetSize: number;
  featureDim: number;
}

let surrogateModels: TBSurrogateModels | null = null;
let newComputationCount = 0;
const RETRAIN_THRESHOLD = 50;

const surrogateStats = {
  predictions: 0,
  trainings: 0,
  datasetSize: 0,
  lastTrainedAt: 0,
  avgPredictionTimeMs: 0,
  totalPredictionTimeMs: 0,
};

function buildFeatureVector(formula: string): number[] {
  try {
    const cf = computeCompositionFeatures(formula);
    return compositionFeatureVector(cf).map(v => Number.isFinite(v) ? v : 0);
  } catch {
    return new Array(28).fill(0);
  }
}

function findBestSplit(
  X: number[][], residuals: number[], indices: number[], featureIndex: number
): { threshold: number; improvement: number; leftIndices: number[]; rightIndices: number[] } {
  const pairs = indices.map(i => ({ idx: i, val: X[i][featureIndex], res: residuals[i] }));
  pairs.sort((a, b) => a.val - b.val);
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
    if (leftCount === 0 || rightCount === 0) continue;
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

function trainGBM(X: number[][], y: number[], nTrees: number = 100, lr: number = 0.05, maxDepth: number = 4): GBMModel {
  const n = X.length;
  const allIndices = Array.from({ length: n }, (_, i) => i);
  const basePrediction = y.reduce((s, v) => s + v, 0) / n;
  const predictions = new Array(n).fill(basePrediction);
  const trees: TreeNode[] = [];

  for (let iter = 0; iter < nTrees; iter++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);
    const tree = buildTree(X, residuals, allIndices, 0, maxDepth, );
    if (typeof tree === "number") break;
    trees.push(tree);
    for (let i = 0; i < n; i++) {
      predictions[i] += lr * predictTree(tree, X[i]);
    }
  }

  return { trees, learningRate: lr, basePrediction };
}

function predictGBM(model: GBMModel, x: number[]): number {
  let pred = model.basePrediction;
  for (const tree of model.trees) {
    const v = predictTree(tree, x);
    if (Number.isFinite(v)) pred += model.learningRate * v;
  }
  return Number.isFinite(pred) ? pred : model.basePrediction;
}

async function generateTrainingData(maxFormulas = 70): Promise<{ formulas: string[]; X: number[][]; Y: Map<string, number[]> }> {
  const formulas: string[] = [];
  const X: number[][] = [];
  const Y = new Map<string, number[]>();
  for (const name of TARGET_NAMES) Y.set(name, []);

  const seedEntries = getTrainingData();
  const subsample = seedEntries.length > 50 ? seedEntries.filter((_, i) => i % Math.ceil(seedEntries.length / 50) === 0) : seedEntries;
  const seedFormulas = subsample.map(e => e.formula);

  const physicsResults = getAllPhysicsResults();
  const physicsFormulas = physicsResults.slice(0, 20).map(r => r.formula);

  // Cap total formulas to avoid long synchronous blocking. computeTBProperties
  // takes ~0.9s per formula. Yield before each call so the event loop stays
  // responsive — each computation is still ~0.9s but they are now separated.
  const allFormulas = Array.from(new Set([...seedFormulas, ...physicsFormulas])).slice(0, maxFormulas);

  for (const formula of allFormulas) {
    await yieldToEventLoop(); // prevent stacking multiple 0.9s blocks back-to-back
    try {
      const tbProps = computeTBProperties(formula, 0);
      const features = buildFeatureVector(formula);
      if (features.some(v => !Number.isFinite(v))) continue;

      formulas.push(formula);
      X.push(features);
      Y.get("dosAtEF")!.push(tbProps.dosAtEF);
      Y.get("bandFlatness")!.push(tbProps.bandFlatness);
      Y.get("vanHoveProximity")!.push(tbProps.vanHoveProximity);
      Y.get("hopfieldEta")!.push(tbProps.hopfieldEta);
      Y.get("lambdaProxy")!.push(tbProps.lambdaProxy);
      Y.get("effectiveMass")!.push(tbProps.effectiveMass);
      Y.get("bandwidth")!.push(tbProps.bandwidth);
      Y.get("bandDegeneracy")!.push(tbProps.bandDegeneracy);
      Y.get("metallicity")!.push(tbProps.metallicity);
    } catch {
      continue;
    }
  }

  return { formulas, X, Y };
}

async function trainAllModels(X: number[][], Y: Map<string, number[]>, quickInit = false): Promise<TBSurrogateModels> {
  const models = new Map<string, GBMModel>();
  // quickInit: use 8 trees (fast startup) instead of 40 (full quality).
  // Full retrains use 40 trees. Yield between targets so heartbeats can fire.
  const nTrees = quickInit ? 8 : 40;

  for (const targetName of TARGET_NAMES) {
    await yieldToEventLoop();
    const y = Y.get(targetName)!;
    if (y.length < 5) {
      models.set(targetName, { trees: [], learningRate: 0.05, basePrediction: y.length > 0 ? y.reduce((s, v) => s + v, 0) / y.length : 0 });
      continue;
    }
    const model = trainGBM(X, y, nTrees, 0.08, 3);
    models.set(targetName, model);
  }

  surrogateStats.trainings++;
  surrogateStats.lastTrainedAt = Date.now();
  surrogateStats.datasetSize = X.length;

  return {
    models,
    trainedAt: Date.now(),
    datasetSize: X.length,
    featureDim: X.length > 0 ? X[0].length : 0,
  };
}

let initDeferred = false;
let initInProgress = false;

async function trainImmediate(maxFormulas?: number): Promise<void> {
  if (initInProgress) return;
  initInProgress = true;
  try {
    const { X, Y } = await generateTrainingData(maxFormulas);
    surrogateModels = await trainAllModels(X, Y, true); // quickInit=true: 8 trees instead of 40
    console.log(`[TB-ML Surrogate] Immediate init: ${surrogateModels.datasetSize} samples`);
  } catch (e) {
    console.log(`[TB-ML Surrogate] Immediate init failed: ${e instanceof Error ? e.message : "unknown"}`);
  } finally {
    initInProgress = false;
  }
}

function deferInit(): void {
  if (initDeferred) return;
  initDeferred = true;
  // Delay 600s — let learning cycles warm the physics cache first.
  // trainImmediate is now async: each computeTBProperties call is separated by a
  // yieldToEventLoop so the event loop is never blocked more than ~0.9s at a time.
  setTimeout(() => {
    if (!surrogateModels) {
      trainImmediate(10)
        .then(() => console.log(`[TB-ML Surrogate] Deferred init complete: ${surrogateModels?.datasetSize ?? 0} samples`))
        .catch(e => console.log(`[TB-ML Surrogate] Deferred init failed: ${e instanceof Error ? e.message : "unknown"}`));
    }
  }, 600_000);
}

function ensureModels(): TBSurrogateModels | null {
  if (surrogateModels) return surrogateModels;
  // Do NOT call trainImmediate here — computeTBProperties inside generateTrainingData
  // takes ~0.9s per formula × 23 samples = 21s of synchronous blocking on first use.
  // Instead, trigger the deferred background init (10s delay) and return null (defaults).
  deferInit();
  return null;
}

export function predictTBProperties(formula: string): TBSurrogateTarget & { confidence: number; source: "surrogate" } {
  const start = Date.now();
  const models = ensureModels();

  if (!models) {
    surrogateStats.predictions++;
    return {
      dosAtEF: 0, bandFlatness: 0, vanHoveProximity: 0, hopfieldEta: 0,
      lambdaProxy: 0, effectiveMass: 1, bandwidth: 0, bandDegeneracy: 1, metallicity: 0,
      confidence: 0, source: "surrogate",
    };
  }

  const features = buildFeatureVector(formula);

  const result: any = { source: "surrogate" as const };

  for (const targetName of TARGET_NAMES) {
    const model = models.models.get(targetName);
    if (!model || model.trees.length === 0) {
      result[targetName] = model?.basePrediction ?? 0;
    } else {
      result[targetName] = Math.max(0, predictGBM(model, features));
    }
  }

  result.confidence = models.datasetSize > 50 ? 0.7 : models.datasetSize > 20 ? 0.5 : 0.3;

  const elapsed = Date.now() - start;
  surrogateStats.predictions++;
  surrogateStats.totalPredictionTimeMs += elapsed;
  surrogateStats.avgPredictionTimeMs = surrogateStats.totalPredictionTimeMs / surrogateStats.predictions;

  return result as TBSurrogateTarget & { confidence: number; source: "surrogate" };
}

export function recordNewTBComputation(): void {
  newComputationCount++;
}

function incrementalTrainGBM(existing: GBMModel, X: number[][], y: number[], additionalTrees: number = 20): GBMModel {
  const n = X.length;
  const predictions = new Array(n);
  for (let i = 0; i < n; i++) {
    predictions[i] = predictGBM(existing, X[i]);
  }

  const allIndices = Array.from({ length: n }, (_, i) => i);
  const newTrees: TreeNode[] = [];

  for (let iter = 0; iter < additionalTrees; iter++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);
    const tree = buildTree(X, residuals, allIndices, 0, 3);
    if (typeof tree === "number") break;
    newTrees.push(tree);
    for (let i = 0; i < n; i++) {
      predictions[i] += existing.learningRate * predictTree(tree, X[i]);
    }
  }

  return {
    trees: [...existing.trees, ...newTrees],
    learningRate: existing.learningRate,
    basePrediction: existing.basePrediction,
  };
}

export async function retrainTBSurrogate(): Promise<{ retrained: boolean; datasetSize: number; reason: string }> {
  if (newComputationCount < RETRAIN_THRESHOLD && surrogateModels) {
    return { retrained: false, datasetSize: surrogateModels.datasetSize, reason: `Only ${newComputationCount} new computations (need ${RETRAIN_THRESHOLD})` };
  }

  // generateTrainingData is now async and yields between each computeTBProperties call
  // (~0.9s each), so the event loop is never blocked for more than ~0.9s at a time.
  const { X, Y } = await generateTrainingData(15);
  const prevCount = newComputationCount;
  newComputationCount = 0;

  if (surrogateModels && X.length > 0) {
    const existingModels = surrogateModels.models;
    const updatedModels = new Map<string, GBMModel>();
    const additionalTrees = Math.min(20, Math.max(5, Math.floor(prevCount / 5)));

    for (const targetName of TARGET_NAMES) {
      await yieldToEventLoop();
      const y = Y.get(targetName)!;
      const existing = existingModels.get(targetName);
      if (!existing || existing.trees.length === 0 || y.length < 5) {
        updatedModels.set(targetName, trainGBM(X, y, 40, 0.08, 3));
      } else {
        const maxTrees = 120;
        if (existing.trees.length + additionalTrees > maxTrees) {
          updatedModels.set(targetName, trainGBM(X, y, 40, 0.08, 3));
        } else {
          updatedModels.set(targetName, incrementalTrainGBM(existing, X, y, additionalTrees));
        }
      }
    }

    surrogateStats.trainings++;
    surrogateStats.lastTrainedAt = Date.now();
    surrogateStats.datasetSize = X.length;

    surrogateModels = {
      models: updatedModels,
      trainedAt: Date.now(),
      datasetSize: X.length,
      featureDim: X.length > 0 ? X[0].length : 0,
    };

    return { retrained: true, datasetSize: X.length, reason: `Incremental retrain: +${additionalTrees} trees per target` };
  }

  surrogateModels = await trainAllModels(X, Y);
  return { retrained: true, datasetSize: X.length, reason: "Full retrain (no prior models)" };
}

export function getTBSurrogateStats() {
  const models = surrogateModels;
  return {
    trained: !!models,
    datasetSize: models?.datasetSize ?? 0,
    featureDim: models?.featureDim ?? 0,
    modelCount: TARGET_NAMES.length,
    targetNames: [...TARGET_NAMES],
    predictions: surrogateStats.predictions,
    trainings: surrogateStats.trainings,
    lastTrainedAt: surrogateStats.lastTrainedAt,
    avgPredictionTimeMs: Math.round(surrogateStats.avgPredictionTimeMs * 100) / 100,
    pendingNewComputations: newComputationCount,
    retrainThreshold: RETRAIN_THRESHOLD,
    treeCounts: models ? Object.fromEntries(
      TARGET_NAMES.map(name => [name, models.models.get(name)?.trees.length ?? 0])
    ) : {},
  };
}

export function initTBSurrogate(): void {
  deferInit();
  console.log(`[TB-ML Surrogate] Initialization deferred to background`);
}
