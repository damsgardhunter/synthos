import { SUPERCON_TRAINING_DATA, type SuperconEntry } from "./supercon-dataset";
import { extractFeatures, type MLFeatureVector } from "./ml-predictor";
import { computeMiedemaFormationEnergy } from "./phase-diagram-engine";
import { computeCompositionFeatures, compositionFeatureVector, COMPOSITION_FEATURE_NAMES } from "./composition-features";
import { storage } from "../storage";

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
      const x = featureVectorToArray(features, entry.formula);
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
  const r2 = sst < 1e-6 ? 0 : 1 - sse / sst;
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

const FEATURE_MEANS: Record<string, number> = {
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
  bandGap: 0,
  formationEnergy: 0,
  stability: 0.5,
};

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

function deriveNonCentrosymmetric(f: MLFeatureVector): number {
  const sym = (f as any).crystalSymmetry;
  if (!sym || typeof sym !== "string") return 0;
  const sg = sym.trim();
  const NON_CENTRO_GROUPS = [
    "P4mm", "P4bm", "P42cm", "P42nm", "P4cc", "P4nc",
    "I4mm", "I4cm", "I41md", "I41cd",
    "P6mm", "P6cc", "P63mc", "P63cm",
    "R3m", "R3c",
    "P21", "P1", "Pca21", "Pna21", "Pmc21", "Pmn21",
    "Fdd2", "Aba2", "Ima2", "Cmc21",
    "F-43m", "I-4m2", "I-42d", "P-4m2", "P-42m",
    "P213", "I213",
  ];
  for (const g of NON_CENTRO_GROUPS) {
    if (sg.includes(g)) return 1;
  }
  const normalized = sg.toLowerCase();
  if (normalized.includes("non-centrosymmetric") || normalized.includes("noncentro")) return 1;
  return 0;
}

function featureVectorToArray(f: MLFeatureVector, formula?: string): number[] {
  const resolvedFormula = formula || f._sourceFormula;
  let miedemaEnergy = 0;
  if (resolvedFormula) {
    try {
      miedemaEnergy = computeMiedemaFormationEnergy(resolvedFormula);
      if (!Number.isFinite(miedemaEnergy)) miedemaEnergy = 0;
    } catch {
      miedemaEnergy = 0;
    }
  }

  let pressureGpa = sanitize(f.pressureGpa, FEATURE_MEANS.pressureGpa);
  if (pressureGpa === 0 && f.hasHydrogen && f.hydrogenRatio > 0.3) {
    if (f.hydrogenRatio >= 8) pressureGpa = 200;
    else if (f.hydrogenRatio >= 6) pressureGpa = 150;
    else if (f.hydrogenRatio >= 4) pressureGpa = 100;
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
    f.anharmonicityFlag ? 1 : 0,
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
    sanitize((f as any).bandGap, FEATURE_MEANS.bandGap),
    sanitize((f as any).formationEnergy, FEATURE_MEANS.formationEnergy),
    sanitize((f as any).stability, FEATURE_MEANS.stability),
    (() => {
      const sym = (f as any).crystalSymmetry;
      if (!sym || typeof sym !== "string") return 0;
      const normalized = sym.toLowerCase().trim();
      const SG_MAP: Record<string, number> = { cubic: 7, hexagonal: 6, tetragonal: 5, orthorhombic: 4, monoclinic: 3, triclinic: 2, trigonal: 1 };
      for (const [key, val] of Object.entries(SG_MAP)) { if (normalized.includes(key)) return val; }
      return 0;
    })(),
    deriveMultiBandScore(f),
    sanitize(miedemaEnergy, 0),
    deriveNonCentrosymmetric(f),
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
  "bandGap", "formationEnergy", "stability", "crystalSymmetry",
  "multiBandScore", "miedemaFormEnergy", "nonCentrosymmetric",
];

const FEATURE_NAMES = [...PHYSICS_FEATURE_NAMES, ...COMPOSITION_FEATURE_NAMES];

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

  const valSize = Math.max(2, Math.floor(n * 0.15));
  const shuffledIndices = Array.from({ length: n }, (_, i) => i);
  for (let i = shuffledIndices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffledIndices[i], shuffledIndices[j]] = [shuffledIndices[j], shuffledIndices[i]];
  }
  const valIdxArr = shuffledIndices.slice(0, valSize);
  const trainIndices = shuffledIndices.slice(valSize);

  const trainX = trainIndices.map(i => X[i]);
  const trainY = trainIndices.map(i => y[i]);
  const valX = valIdxArr.map(i => X[i]);
  const valY = valIdxArr.map(i => y[i]);

  const nTrain = trainX.length;
  const allTrainIndices = Array.from({ length: nTrain }, (_, i) => i);

  const basePrediction = trainY.reduce((s, v) => s + v, 0) / nTrain;
  const trainPredictions = new Array(nTrain).fill(basePrediction);
  const valPredictions = new Array(valX.length).fill(basePrediction);
  const trees: TreeNode[] = [];

  const yVariance = trainY.reduce((s, yi) => s + (yi - basePrediction) ** 2, 0) / nTrain;
  const mseThreshold = Math.max(1.0, 0.01 * yVariance);

  let bestValMSE = Infinity;
  let valIncreaseCount = 0;
  const MAX_VAL_INCREASE = 3;

  for (let iter = 0; iter < nEstimators; iter++) {
    const residuals = trainY.map((yi, i) => yi - trainPredictions[i]);

    const tree = buildTree(trainX, residuals, allTrainIndices, 0, maxDepth, 8);
    if (typeof tree === "number") break;

    trees.push(tree);

    for (let i = 0; i < nTrain; i++) {
      trainPredictions[i] += learningRate * predictTree(tree, trainX[i]);
    }

    const trainMSE = trainY.reduce((s, yi, i) => s + (yi - trainPredictions[i]) ** 2, 0) / nTrain;
    if (trainMSE < mseThreshold) break;

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
    learningRate,
    basePrediction,
    featureNames: FEATURE_NAMES,
    trainedAt: Date.now(),
  };
}

function predictWithModel(model: GBModel, x: number[]): number {
  let prediction = model.basePrediction;
  for (const tree of model.trees) {
    const treeVal = predictTree(tree, x);
    if (!Number.isFinite(treeVal)) continue;
    prediction += model.learningRate * treeVal;
  }
  if (!Number.isFinite(prediction)) return model.basePrediction;
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
      const fArr = featureVectorToArray(features, entry.formula);
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

  cachedModel = trainGradientBoosting(X, y, 300, 0.05, 6);
  cachedCalibration = computeCalibration(cachedModel);
  return cachedModel;
}

export function gbPredict(features: MLFeatureVector, formula?: string): { tcPredicted: number; score: number; reasoning: string[] } {
  const model = getTrainedModel();
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
    reasoning.push(`${name}=${val.toFixed(2)} (key predictor)`);
  }

  let score = 0;
  if (tcPredicted > 293) score = 0.92;
  else if (tcPredicted > 200) score = 0.85;
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

  const safeTc = Number.isFinite(tcPredicted) ? Math.min(350, Math.max(0, Math.round(tcPredicted * 10) / 10)) : 0;
  return { tcPredicted: safeTc, score, reasoning };
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

  return {
    mse: sse / details.length,
    r2: sst < 1e-6 ? 0 : 1 - sse / sst,
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

let failureExamples: SuperconEntry[] = [];
let successExamples: SuperconEntry[] = [];
let surrogateScreenCount = 0;
let surrogatePassCount = 0;
let surrogateRejectCount = 0;
let lastRetrainCycle = 0;

export interface EvaluatedEntry {
  formula: string;
  tc: number;
  formationEnergy: number | null;
  stable: boolean;
  source: "dft" | "xtb" | "external" | "active-learning";
  evaluatedAt: number;
}

const evaluatedDataset: EvaluatedEntry[] = [];
const evaluatedFormulas = new Set<string>();
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
  source: EvaluatedEntry["source"] = "dft"
): boolean {
  totalDFTFeedback++;

  const existing = evaluatedDataset.find(e => e.formula === formula);
  if (existing) {
    if (SOURCE_PRIORITY[source] > SOURCE_PRIORITY[existing.source]) {
      existing.tc = Math.max(0, tc);
      existing.formationEnergy = formationEnergy;
      existing.stable = stable;
      existing.source = source;
      existing.evaluatedAt = Date.now();
      return true;
    }
    return false;
  }

  evaluatedFormulas.add(formula);
  evaluatedDataset.push({
    formula,
    tc: Math.max(0, tc),
    formationEnergy,
    stable,
    source,
    evaluatedAt: Date.now(),
  });

  return true;
}

export async function retrainXGBoostFromEvaluated(cycleCount?: number): Promise<{
  retrained: boolean;
  datasetSize: number;
  newEntries: number;
}> {
  const augmentedData = [
    ...SUPERCON_TRAINING_DATA,
    ...successExamples,
    ...failureExamples,
  ];

  const seen = new Set(augmentedData.map(e => e.formula));
  let newFromEval = 0;
  for (const entry of evaluatedDataset) {
    if (seen.has(entry.formula)) continue;
    seen.add(entry.formula);
    augmentedData.push({
      formula: entry.formula,
      tc: entry.tc,
      family: entry.stable ? "DFT-Evaluated" : "DFT-Failed",
      isSuperconductor: entry.tc > 0 && entry.stable,
    });
    newFromEval++;
  }

  const X: number[][] = [];
  const y: number[] = [];

  for (const entry of augmentedData) {
    try {
      const features = extractFeatures(entry.formula);
      const fArr = featureVectorToArray(features, entry.formula);
      if (fArr.some(v => !Number.isFinite(v))) continue;
      X.push(fArr);
      y.push(entry.tc);
    } catch {
      continue;
    }
  }

  if (X.length < 10) {
    return { retrained: false, datasetSize: X.length, newEntries: newFromEval };
  }

  invalidateModel();
  cachedModel = trainGradientBoosting(X, y, 300, 0.05, 6);
  cachedCalibration = computeCalibration(cachedModel);
  xgboostRetrainCount++;
  if (cycleCount != null) lastXGBoostRetrainCycle = cycleCount;
  lastRetrainCycle = Date.now();

  console.log(`[XGBoost-AL] Retrained with ${X.length} total samples (${newFromEval} from evaluated dataset, ${successExamples.length} successes, ${failureExamples.length} failures)`);

  return { retrained: true, datasetSize: X.length, newEntries: newFromEval };
}

export function getEvaluatedDatasetStats() {
  return {
    totalEvaluated: evaluatedDataset.length,
    totalDFTFeedback,
    xgboostRetrainCount,
    lastXGBoostRetrainCycle,
    bySource: {
      dft: evaluatedDataset.filter(e => e.source === "dft").length,
      xtb: evaluatedDataset.filter(e => e.source === "xtb").length,
      external: evaluatedDataset.filter(e => e.source === "external").length,
      activeLearning: evaluatedDataset.filter(e => e.source === "active-learning").length,
    },
    stableCount: evaluatedDataset.filter(e => e.stable).length,
    unstableCount: evaluatedDataset.filter(e => !e.stable).length,
    avgTc: evaluatedDataset.length > 0
      ? evaluatedDataset.reduce((s, e) => s + e.tc, 0) / evaluatedDataset.length
      : 0,
    datasetGrowthRate: evaluatedDataset.length > 0
      ? evaluatedDataset.length / Math.max(1, xgboostRetrainCount)
      : 0,
  };
}

export function invalidateModel(): void {
  cachedModel = null;
  cachedCalibration = null;
}

export function surrogateScreen(formula: string, minTcThreshold: number = 5): {
  pass: boolean;
  predictedTc: number;
  score: number;
  reasoning: string[];
} {
  surrogateScreenCount++;
  try {
    const features = extractFeatures(formula);

    if (features.metallicity < 0.15) {
      surrogateRejectCount++;
      return { pass: false, predictedTc: 0, score: 0, reasoning: ["Insulator: metallicity too low"] };
    }

    const result = gbPredict(features, formula);
    const predictedTc = Number.isFinite(result.tcPredicted) ? result.tcPredicted : 0;

    if (predictedTc < minTcThreshold) {
      surrogateRejectCount++;
      return { pass: false, predictedTc, score: result.score, reasoning: result.reasoning };
    }

    if (result.score < 0.1) {
      surrogateRejectCount++;
      return { pass: false, predictedTc, score: result.score, reasoning: result.reasoning };
    }

    surrogatePassCount++;
    return { pass: true, predictedTc, score: result.score, reasoning: result.reasoning };
  } catch {
    surrogateRejectCount++;
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
  });

  if (successExamples.length % 10 === 0) {
    await retrainWithAccumulatedData();
  }
}

export async function retrainWithAccumulatedData(): Promise<number> {
  const augmentedData = [...SUPERCON_TRAINING_DATA, ...successExamples, ...failureExamples];

  const X: number[][] = [];
  const y: number[] = [];

  for (const entry of augmentedData) {
    try {
      const features = extractFeatures(entry.formula);
      const fArr = featureVectorToArray(features, entry.formula);
      if (fArr.some(v => !Number.isFinite(v))) continue;
      X.push(fArr);
      y.push(entry.tc);
    } catch {
      continue;
    }
  }

  if (X.length < 10) return 0;

  invalidateModel();
  cachedModel = trainGradientBoosting(X, y, 300, 0.05, 6);
  cachedCalibration = computeCalibration(cachedModel);
  lastRetrainCycle = Date.now();

  const totalNew = successExamples.length + failureExamples.length;
  console.log(`[Surrogate] Model retrained with ${totalNew} new examples (${successExamples.length} successes, ${failureExamples.length} failures). Total training size: ${X.length}`);

  return totalNew;
}

export async function incorporateFailureData(): Promise<number> {
  const failedResults = await storage.getFailedComputationalResults(500);
  if (failedResults.length === 0) return 0;

  const seenFormulas = new Set<string>(
    [...SUPERCON_TRAINING_DATA, ...failureExamples].map(e => e.formula)
  );

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
    });
    added++;
  }

  if (added > 0) {
    invalidateModel();

    const augmentedData = [...SUPERCON_TRAINING_DATA, ...failureExamples];

    const X: number[][] = [];
    const y: number[] = [];

    for (const entry of augmentedData) {
      try {
        const features = extractFeatures(entry.formula);
        const fArr = featureVectorToArray(features, entry.formula);
        if (fArr.some(v => !Number.isFinite(v))) continue;
        X.push(fArr);
        y.push(entry.tc);
      } catch {
        continue;
      }
    }

    if (X.length >= 10) {
      cachedModel = trainGradientBoosting(X, y, 300, 0.05, 6);
      cachedCalibration = computeCalibration(cachedModel);
    }

    console.log(`XGBoost model retrained with ${failureExamples.length} failure examples`);
  }

  return added;
}

export function getFailureExampleCount(): number {
  return failureExamples.length;
}
