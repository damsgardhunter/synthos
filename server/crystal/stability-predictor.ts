import { extractFeatures } from "../learning/ml-predictor";
import { getGraphFeatureVector, buildGraphFromFormula } from "./crystal-graph-builder";
import { getTrainingData, type CrystalStructureEntry } from "./crystal-structure-dataset";
import { getAllPhysicsResults, type PhysicsResult } from "../learning/physics-results-store";
import { getFailureEntries, type StructureFailureEntry } from "./structure-failure-db";

interface StabilityPrediction {
  predictedFormationEnergy: number;
  phononStabilityProb: number;
  isLikelyStable: boolean;
}

interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | number;
  right: TreeNode | number;
}

interface StabilityModel {
  formationEnergyTrees: TreeNode[];
  phononStabilityTrees: TreeNode[];
  learningRate: number;
  baseFE: number;
  basePhonon: number;
  trainedAt: number;
  nSamples: number;
}

let cachedModel: StabilityModel | null = null;
let totalPredictions = 0;
let totalRejected = 0;
let totalAccepted = 0;

const FE_THRESHOLD = 0.5;
const PHONON_THRESHOLD = 0.3;

async function extractStabilityFeatures(formula: string): Promise<number[]> {
  try {
    const mlFeatures = await extractFeatures(formula);
    const compositionFeats = [
      mlFeatures.avgElectronegativity / 4,
      mlFeatures.maxAtomicMass / 250,
      mlFeatures.numElements / 8,
      mlFeatures.hasTransitionMetal ? 1 : 0,
      mlFeatures.hasRareEarth ? 1 : 0,
      mlFeatures.hasHydrogen ? 1 : 0,
      mlFeatures.hasChalcogen ? 1 : 0,
      mlFeatures.hasPnictogen ? 1 : 0,
      mlFeatures.enSpread / 3,
      mlFeatures.hydrogenRatio / 10,
      mlFeatures.valenceElectronConcentration / 8,
      mlFeatures.avgAtomicRadius / 250,
      mlFeatures.pettiforNumber / 100,
      mlFeatures.avgBulkModulus / 500,
      mlFeatures.debyeTemperature / 2000,
      mlFeatures.avgSommerfeldGamma / 10,
      mlFeatures.metallicity,
      mlFeatures.electronDensityEstimate,
      mlFeatures.cooperPairStrength,
      mlFeatures.dimensionalityScore,
      mlFeatures.correlationStrength,
      mlFeatures.electronPhononLambda / 3,
      mlFeatures.logPhononFreq / 1000,
      mlFeatures.bondStiffnessVariance,
      mlFeatures.chargeTransferMagnitude,
      mlFeatures.connectivityIndex,
      mlFeatures.formationEnergy != null ? mlFeatures.formationEnergy / 5 : 0,
      mlFeatures.bandGap != null ? mlFeatures.bandGap / 5 : 0,
    ];

    let graphFeats: number[] = [];
    try {
      const graph = buildGraphFromFormula(formula);
      graphFeats = getGraphFeatureVector(graph);
    } catch {
      graphFeats = new Array(35).fill(0);
    }

    return [...compositionFeats, ...graphFeats].map(v => Number.isFinite(v) ? v : 0);
  } catch {
    return new Array(63).fill(0);
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
  if (depth >= maxDepth || indices.length < 6) {
    const sum = indices.reduce((s, i) => s + residuals[i], 0);
    return sum / indices.length;
  }

  const nFeatures = X[0].length;
  let bestFeature = -1;
  let bestImprovement = -Infinity;
  let bestThreshold = 0;
  let bestLeftIdx: number[] = [];
  let bestRightIdx: number[] = [];

  for (let fi = 0; fi < nFeatures; fi++) {
    const split = findBestSplit(X, residuals, indices, fi);
    if (split.improvement > bestImprovement && split.leftIndices.length >= 2 && split.rightIndices.length >= 2) {
      bestImprovement = split.improvement;
      bestFeature = fi;
      bestThreshold = split.threshold;
      bestLeftIdx = split.leftIndices;
      bestRightIdx = split.rightIndices;
    }
  }

  if (bestFeature === -1) {
    const sum = indices.reduce((s, i) => s + residuals[i], 0);
    return sum / indices.length;
  }

  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildTree(X, residuals, bestLeftIdx, depth + 1, maxDepth),
    right: buildTree(X, residuals, bestRightIdx, depth + 1, maxDepth),
  };
}

function predictTree(tree: TreeNode | number, x: number[]): number {
  if (typeof tree === "number") return tree;
  if (x[tree.featureIndex] <= tree.threshold) {
    return predictTree(tree.left, x);
  }
  return predictTree(tree.right, x);
}

function trainGBM(
  X: number[][], y: number[], nEstimators: number, learningRate: number, maxDepth: number
): { trees: TreeNode[]; basePrediction: number } {
  const n = X.length;
  const basePrediction = y.reduce((s, v) => s + v, 0) / n;
  const predictions = new Array(n).fill(basePrediction);
  const trees: TreeNode[] = [];
  const allIndices = Array.from({ length: n }, (_, i) => i);

  for (let iter = 0; iter < nEstimators; iter++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);
    const tree = buildTree(X, residuals, allIndices, 0, maxDepth, );
    if (typeof tree === "number") break;
    trees.push(tree);
    for (let i = 0; i < n; i++) {
      predictions[i] += learningRate * predictTree(tree, X[i]);
    }
  }

  return { trees, basePrediction };
}

function predictWithTrees(trees: TreeNode[], basePrediction: number, learningRate: number, x: number[]): number {
  let prediction = basePrediction;
  for (const tree of trees) {
    const val = predictTree(tree, x);
    if (!Number.isFinite(val)) continue;
    prediction += learningRate * val;
  }
  return Number.isFinite(prediction) ? prediction : basePrediction;
}

async function gatherTrainingData(): Promise<{ X: number[][]; yFE: number[]; yPhonon: number[] }> {
  const X: number[][] = [];
  const yFE: number[] = [];
  const yPhonon: number[] = [];

  const seedData = getTrainingData();
  for (const entry of seedData) {
    try {
      const features = await extractStabilityFeatures(entry.formula);
      if (features.some(v => !Number.isFinite(v))) continue;
      X.push(features);
      yFE.push(entry.formationEnergy);
      yPhonon.push(1.0);
    } catch {
      continue;
    }
  }

  const physicsResults = getAllPhysicsResults();
  for (const result of physicsResults) {
    try {
      const features = await extractStabilityFeatures(result.formula);
      if (features.some(v => !Number.isFinite(v))) continue;
      X.push(features);
      yFE.push(result.formationEnergy ?? 0);
      yPhonon.push(result.phononStable ? 1.0 : 0.0);
    } catch {
      continue;
    }
  }

  const failures = getFailureEntries();
  for (const entry of failures) {
    try {
      const features = await extractStabilityFeatures(entry.formula);
      if (features.some(v => !Number.isFinite(v))) continue;
      X.push(features);
      if (entry.failureReason === "high_formation_energy") {
        yFE.push(entry.formationEnergy ?? 1.0);
        yPhonon.push(0.3);
      } else if (entry.failureReason === "unstable_phonons") {
        yFE.push(entry.formationEnergy ?? 0.2);
        yPhonon.push(0.0);
      } else if (entry.failureReason === "structure_collapse") {
        yFE.push(entry.formationEnergy ?? 0.8);
        yPhonon.push(0.1);
      } else {
        yFE.push(entry.formationEnergy ?? 0.3);
        yPhonon.push(0.2);
      }
    } catch {
      continue;
    }
  }

  return { X, yFE, yPhonon };
}

export async function trainStabilityPredictor(): Promise<void> {
  const { X, yFE, yPhonon } = await gatherTrainingData();

  if (X.length < 10) {
    cachedModel = {
      formationEnergyTrees: [],
      phononStabilityTrees: [],
      learningRate: 0.05,
      baseFE: 0,
      basePhonon: 0.5,
      trainedAt: Date.now(),
      nSamples: 0,
    };
    return;
  }

  const feResult = trainGBM(X, yFE, 150, 0.05, 4);
  const phononResult = trainGBM(X, yPhonon, 150, 0.05, 4);

  cachedModel = {
    formationEnergyTrees: feResult.trees,
    phononStabilityTrees: phononResult.trees,
    learningRate: 0.05,
    baseFE: feResult.basePrediction,
    basePhonon: phononResult.basePrediction,
    trainedAt: Date.now(),
    nSamples: X.length,
  };
}

async function getModel(): Promise<StabilityModel> {
  if (!cachedModel) {
    await trainStabilityPredictor();
  }
  return cachedModel!;
}

export async function predictStabilityScreen(formula: string): Promise<StabilityPrediction> {
  totalPredictions++;
  const model = await getModel();
  const features = await extractStabilityFeatures(formula);

  let predictedFormationEnergy: number;
  let phononStabilityProb: number;

  if (model.formationEnergyTrees.length === 0) {
    predictedFormationEnergy = 0;
    phononStabilityProb = 0.5;
  } else {
    predictedFormationEnergy = predictWithTrees(
      model.formationEnergyTrees, model.baseFE, model.learningRate, features
    );
    phononStabilityProb = predictWithTrees(
      model.phononStabilityTrees, model.basePhonon, model.learningRate, features
    );
  }

  phononStabilityProb = Math.max(0, Math.min(1, phononStabilityProb));

  const isLikelyStable = predictedFormationEnergy <= FE_THRESHOLD && phononStabilityProb >= PHONON_THRESHOLD;

  if (isLikelyStable) {
    totalAccepted++;
  } else {
    totalRejected++;
  }

  return {
    predictedFormationEnergy: Math.round(predictedFormationEnergy * 1000) / 1000,
    phononStabilityProb: Math.round(phononStabilityProb * 1000) / 1000,
    isLikelyStable,
  };
}

export function getStabilityPredictorStats(): {
  modelTrained: boolean;
  trainedAt: number | null;
  nSamples: number;
  nFETrees: number;
  nPhononTrees: number;
  totalPredictions: number;
  totalAccepted: number;
  totalRejected: number;
  rejectionRate: number;
  feThreshold: number;
  phononThreshold: number;
} {
  const model = cachedModel;
  const total = totalPredictions || 1;
  return {
    modelTrained: model != null && model.nSamples > 0,
    trainedAt: model?.trainedAt ?? null,
    nSamples: model?.nSamples ?? 0,
    nFETrees: model?.formationEnergyTrees.length ?? 0,
    nPhononTrees: model?.phononStabilityTrees.length ?? 0,
    totalPredictions,
    totalAccepted,
    totalRejected,
    rejectionRate: Math.round((totalRejected / total) * 10000) / 10000,
    feThreshold: FE_THRESHOLD,
    phononThreshold: PHONON_THRESHOLD,
  };
}
