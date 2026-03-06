import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { extractFeatures, type MLFeatureVector } from "./ml-predictor";

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
  featureNames: string[];
  trainedAt: number;
}

let cachedModel: GBModel | null = null;

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
  predictedVsActual: { formula: string; actual: number; predicted: number; residual: number }[];
  computedAt: number;
}

let cachedCalibration: CalibrationData | null = null;

function computePercentile(sorted: number[], p: number): number {
  const idx = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (idx - lower) * (sorted[upper] - sorted[lower]);
}

function computeCalibration(model: GBModel): CalibrationData {
  const details: { formula: string; actual: number; predicted: number; residual: number }[] = [];
  const residuals: number[] = [];
  let sse = 0;
  let sst = 0;
  let totalAbsError = 0;

  const allTc = SUPERCON_TRAINING_DATA.map(e => e.tc);
  const meanTc = allTc.reduce((s, v) => s + v, 0) / allTc.length;

  for (const entry of SUPERCON_TRAINING_DATA) {
    try {
      const features = extractFeatures(entry.formula);
      const x = featureVectorToArray(features);
      if (x.some(v => !Number.isFinite(v))) continue;
      const pred = predictWithModel(model, x);
      const residual = entry.tc - pred;
      residuals.push(residual);
      details.push({ formula: entry.formula, actual: entry.tc, predicted: Math.round(pred * 10) / 10, residual: Math.round(residual * 10) / 10 });
      sse += residual ** 2;
      sst += (entry.tc - meanTc) ** 2;
      totalAbsError += Math.abs(residual);
    } catch {
      continue;
    }
  }

  const n = details.length;
  const mse = sse / n;
  const mae = totalAbsError / n;
  const r2 = 1 - sse / sst;
  const rmse = Math.sqrt(mse);

  const sortedResiduals = [...residuals].sort((a, b) => a - b);
  const absResiduals = residuals.map(r => Math.abs(r)).sort((a, b) => a - b);

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
    predictedVsActual: details,
    computedAt: Date.now(),
  };
}

function featureVectorToArray(f: MLFeatureVector): number[] {
  return [
    f.electronPhononLambda,
    f.metallicity,
    f.logPhononFreq,
    f.debyeTemperature,
    f.correlationStrength,
    f.valenceElectronConcentration,
    f.avgElectronegativity,
    f.enSpread,
    f.hydrogenRatio,
    f.pettiforNumber,
    f.avgAtomicRadius,
    f.avgSommerfeldGamma,
    f.avgBulkModulus,
    f.maxAtomicMass,
    f.numElements,
    f.hasTransitionMetal ? 1 : 0,
    f.hasRareEarth ? 1 : 0,
    f.hasHydrogen ? 1 : 0,
    f.hasChalcogen ? 1 : 0,
    f.hasPnictogen ? 1 : 0,
    f.cooperPairStrength,
    f.dimensionalityScore,
    f.anharmonicityFlag ? 1 : 0,
    f.electronDensityEstimate,
    f.phononCouplingEstimate,
    f.dWaveSymmetry ? 1 : 0,
    f.meissnerPotential,
    f.dftConfidence ?? 0,
    f.orbitalCharacterCode ?? 0,
    f.phononSpectralCentroid ?? 0,
    f.phononSpectralWidth ?? 0,
    f.bondStiffnessVariance ?? 0,
    f.chargeTransferMagnitude ?? 0,
    f.connectivityIndex ?? 0.5,
  ];
}

const FEATURE_NAMES = [
  "lambda", "metallicity", "omegaLog", "debyeTemp", "correlation",
  "VEC", "avgEN", "enSpread", "hRatio", "pettifor",
  "atomicRadius", "sommerfeldGamma", "bulkModulus", "maxMass", "nElements",
  "hasTM", "hasRE", "hasH", "hasChalcogen", "hasPnictogen",
  "cooperPair", "dimensionality", "anharmonic", "electronDensity",
  "phononCoupling", "dWave", "meissner", "dftConfidence",
  "orbitalChar", "phononCentroid", "phononWidth", "bondStiffVar",
  "chargeTransfer", "connectivity",
];

function findBestSplitForSubset(
  X: number[][],
  residuals: number[],
  indices: number[],
  featureIndex: number
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
  maxDepth: number,
  minSamples: number
): TreeNode | number {
  if (depth >= maxDepth || indices.length < minSamples) {
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
    const split = findBestSplitForSubset(X, residuals, indices, fi);
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
    left: buildTree(X, residuals, bestLeftIdx, depth + 1, maxDepth, minSamples),
    right: buildTree(X, residuals, bestRightIdx, depth + 1, maxDepth, minSamples),
  };
}

function predictTree(tree: TreeNode | number, x: number[]): number {
  if (typeof tree === "number") return tree;
  if (x[tree.featureIndex] <= tree.threshold) {
    return predictTree(tree.left, x);
  }
  return predictTree(tree.right, x);
}

function trainGradientBoosting(
  X: number[][],
  y: number[],
  nEstimators: number = 200,
  learningRate: number = 0.1,
  maxDepth: number = 4
): GBModel {
  const n = X.length;
  const allIndices = Array.from({ length: n }, (_, i) => i);

  const basePrediction = y.reduce((s, v) => s + v, 0) / n;
  const predictions = new Array(n).fill(basePrediction);
  const trees: TreeNode[] = [];

  for (let iter = 0; iter < nEstimators; iter++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);

    const tree = buildTree(X, residuals, allIndices, 0, maxDepth, 5);
    if (typeof tree === "number") break;

    trees.push(tree);

    for (let i = 0; i < n; i++) {
      predictions[i] += learningRate * predictTree(tree, X[i]);
    }

    const mse = y.reduce((s, yi, i) => s + (yi - predictions[i]) ** 2, 0) / n;
    if (mse < 1.0) break;
  }

  return {
    trees,
    learningRate,
    basePrediction,
    featureNames: FEATURE_NAMES,
    trainedAt: Date.now(),
  };
}

function predictWithModel(model: GBModel, x: number[]): number {
  let prediction = model.basePrediction;
  for (const tree of model.trees) {
    prediction += model.learningRate * predictTree(tree, x);
  }
  return Math.max(0, prediction);
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

export function getTrainedModel(): GBModel {
  if (cachedModel) return cachedModel;

  const X: number[][] = [];
  const y: number[] = [];

  for (const entry of SUPERCON_TRAINING_DATA) {
    try {
      const features = extractFeatures(entry.formula);
      const fArr = featureVectorToArray(features);
      if (fArr.some(v => !Number.isFinite(v))) continue;
      X.push(fArr);
      y.push(entry.tc);
    } catch {
      continue;
    }
  }

  if (X.length < 10) {
    cachedModel = {
      trees: [],
      learningRate: 0.1,
      basePrediction: 20,
      featureNames: FEATURE_NAMES,
      trainedAt: Date.now(),
    };
    return cachedModel;
  }

  cachedModel = trainGradientBoosting(X, y, 300, 0.1, 4);
  cachedCalibration = computeCalibration(cachedModel);
  return cachedModel;
}

export function gbPredict(features: MLFeatureVector): { tcPredicted: number; score: number; reasoning: string[] } {
  const model = getTrainedModel();
  const x = featureVectorToArray(features);
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
    reasoning.push(`${name}=${val.toFixed(2)} (key predictor)`);
  }

  let score = 0;
  if (tcPredicted > 200) score = 0.85;
  else if (tcPredicted > 100) score = 0.70;
  else if (tcPredicted > 50) score = 0.55;
  else if (tcPredicted > 20) score = 0.40;
  else if (tcPredicted > 5) score = 0.25;
  else if (tcPredicted > 1) score = 0.15;
  else score = 0.05;

  const lambda = features.electronPhononLambda;
  if (lambda > 1.5) score += 0.10;
  else if (lambda > 0.8) score += 0.05;

  if (features.metallicity < 0.3) score -= 0.15;
  else if (features.metallicity < 0.5) score -= 0.05;

  if (features.correlationStrength > 0.85) score -= 0.10;

  score = Math.max(0.01, Math.min(0.95, score));

  return { tcPredicted: Math.max(0, Math.round(tcPredicted * 10) / 10), score, reasoning };
}

export function validateModel(): { mse: number; r2: number; nTrees: number; details: { formula: string; actual: number; predicted: number }[] } {
  const model = getTrainedModel();
  const details: { formula: string; actual: number; predicted: number }[] = [];
  let sse = 0;
  let sst = 0;
  const allTc = SUPERCON_TRAINING_DATA.map(e => e.tc);
  const meanTc = allTc.reduce((s, v) => s + v, 0) / allTc.length;

  for (const entry of SUPERCON_TRAINING_DATA) {
    try {
      const features = extractFeatures(entry.formula);
      const x = featureVectorToArray(features);
      if (x.some(v => !Number.isFinite(v))) continue;
      const pred = predictWithModel(model, x);
      details.push({ formula: entry.formula, actual: entry.tc, predicted: Math.round(pred * 10) / 10 });
      sse += (entry.tc - pred) ** 2;
      sst += (entry.tc - meanTc) ** 2;
    } catch {
      continue;
    }
  }

  return {
    mse: sse / details.length,
    r2: 1 - sse / sst,
    nTrees: model.trees.length,
    details,
  };
}

export function getCalibrationData(): Omit<CalibrationData, 'residuals'> & { residualCount: number } {
  getTrainedModel();
  if (!cachedCalibration) {
    const model = getTrainedModel();
    cachedCalibration = computeCalibration(model);
  }
  const { residuals, ...rest } = cachedCalibration;
  return { ...rest, residualCount: residuals.length };
}

export function getConfidenceBand(predictedTc: number): { lower: number; upper: number } {
  getTrainedModel();
  if (!cachedCalibration) {
    const model = getTrainedModel();
    cachedCalibration = computeCalibration(model);
  }
  const p90 = cachedCalibration.absResidualPercentiles.p90;
  const scaleFactor = Math.max(1, predictedTc / 50);
  const errorMargin = p90 * Math.sqrt(scaleFactor);
  return {
    lower: Math.round(Math.max(0, predictedTc - errorMargin) * 10) / 10,
    upper: Math.round((predictedTc + errorMargin) * 10) / 10,
  };
}
