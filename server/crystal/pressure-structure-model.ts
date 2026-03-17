import { computeCompositionFeatures, compositionFeatureVector } from "../learning/composition-features";
import { getTrainingData, type CrystalStructureEntry } from "./crystal-structure-dataset";
import { getCompositionWeightedProperty } from "../learning/elemental-data";

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

interface PressureStructurePrediction {
  crystalSystem: string;
  spacegroup: number;
  spacegroupSymbol: string;
  latticeParams: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  prototype: string;
  confidence: number;
}

interface PressurePhaseEntry {
  pressureGPa: number;
  crystalSystem: string;
  spacegroup: number;
  spacegroupSymbol: string;
  latticeParams: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  prototype: string;
  confidence: number;
}

interface PressureTransitionRecord {
  formula: string;
  pBefore: number;
  pAfter: number;
  structBefore: string;
  structAfter: string;
  timestamp: number;
}

const SPACEGROUP_SYMBOLS: Record<number, string> = {
  225: "Fm-3m", 229: "Im-3m", 227: "Fd-3m", 221: "Pm-3m", 223: "Pm-3n",
  191: "P6/mmm", 194: "P6_3/mmc", 139: "I4/mmm", 129: "P4/nmm", 123: "P4/mmm",
  99: "P4mm", 141: "I4_1/amd", 62: "Pnma", 47: "Pmmm", 166: "R-3m",
  216: "F-43m", 217: "I-43m", 230: "Ia-3d", 12: "C2/m", 15: "C2/c",
  2: "P-1", 148: "R-3", 167: "R-3c", 186: "P6_3mc", 136: "P4_2/mnm",
  14: "P2_1/c", 204: "Im-3", 176: "P6_3/m", 187: "P-6m2", 161: "R3c",
  59: "Pmmn", 71: "Immm",
};

let crystalSystemClassifier: ClassifierModel | null = null;
let spacegroupClassifier: ClassifierModel | null = null;
let prototypeClassifier: ClassifierModel | null = null;
let latticeRegressors: { a: GBModel; b: GBModel; c: GBModel; alpha: GBModel; beta: GBModel; gamma: GBModel } | null = null;
let modelTrained = false;
let trainedAt = 0;
let datasetSize = 0;
let predictionCount = 0;
let transitionRecords: PressureTransitionRecord[] = [];
const PRESSURES = [0, 50, 150, 300];

function parseFormulaCounts(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function hasHydrogen(formula: string): boolean {
  const counts = parseFormulaCounts(formula);
  return "H" in counts;
}

function hydrogenRatio(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const hCount = counts["H"] || 0;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  return totalAtoms > 0 ? hCount / totalAtoms : 0;
}

function estimateBulkModulus(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const bm = getCompositionWeightedProperty(counts, "bulkModulus");
  return bm && bm > 0 ? bm : 100;
}

function murnaghanVolumeRatio(pressure: number, K0: number, K0p: number = 4.0): number {
  if (pressure <= 0 || K0 <= 0) return 1.0;
  const pGpa = pressure;
  const eta = 1 + (K0p / K0) * pGpa;
  const ratio = Math.pow(eta, -1 / K0p);
  return Math.max(0.5, Math.min(1.0, ratio));
}

function compressLattice(
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number },
  pressure: number,
  K0: number
): { a: number; b: number; c: number; alpha: number; beta: number; gamma: number } {
  const vRatio = murnaghanVolumeRatio(pressure, K0);
  const linearScale = Math.pow(vRatio, 1 / 3);
  return {
    a: lattice.a * linearScale,
    b: lattice.b * linearScale,
    c: lattice.c * linearScale,
    alpha: lattice.alpha,
    beta: lattice.beta,
    gamma: lattice.gamma,
  };
}

function buildFeatureVector(formula: string, pressureGPa: number): number[] {
  const cf = computeCompositionFeatures(formula);
  const compVec = compositionFeatureVector(cf);
  return [...compVec, pressureGPa / 300.0];
}

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

function trainGBM(X: number[][], y: number[], nEstimators = 100, lr = 0.05, maxDepth = 4): GBModel {
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

function trainOneVsAllClassifier(X: number[][], labels: string[], nEstimators = 80, lr = 0.08, maxDepth = 3): ClassifierModel {
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

const HYDRIDE_HIGH_P_RULES: Array<{
  minPressure: number;
  crystalSystem: string;
  spacegroup: number;
  prototype: string;
}> = [
  { minPressure: 150, crystalSystem: "cubic", spacegroup: 225, prototype: "sodalite-clathrate" },
  { minPressure: 150, crystalSystem: "cubic", spacegroup: 229, prototype: "clathrate" },
  { minPressure: 100, crystalSystem: "hexagonal", spacegroup: 194, prototype: "clathrate" },
];

const METAL_HIGH_P_RULES: Array<{
  minPressure: number;
  crystalSystem: string;
  spacegroup: number;
  prototype: string;
}> = [
  { minPressure: 100, crystalSystem: "cubic", spacegroup: 225, prototype: "FCC" },
  { minPressure: 100, crystalSystem: "hexagonal", spacegroup: 194, prototype: "HCP" },
];

function generateSyntheticTrainingData(): {
  X: number[][]; csLabels: string[]; sgLabels: string[]; protoLabels: string[];
  latticeA: number[]; latticeB: number[]; latticeC: number[];
  latticeAlpha: number[]; latticeBeta: number[]; latticeGamma: number[];
} {
  const allSeedData = getTrainingData();
  const seedData = allSeedData.length > 80 ? allSeedData.filter((_, i) => i % Math.ceil(allSeedData.length / 80) === 0) : allSeedData;
  const X: number[][] = [];
  const csLabels: string[] = [];
  const sgLabels: string[] = [];
  const protoLabels: string[] = [];
  const latticeA: number[] = [];
  const latticeB: number[] = [];
  const latticeC: number[] = [];
  const latticeAlpha: number[] = [];
  const latticeBeta: number[] = [];
  const latticeGamma: number[] = [];

  for (const entry of seedData) {
    const K0 = estimateBulkModulus(entry.formula);
    const isHydride = hasHydrogen(entry.formula);
    const hRatio = hydrogenRatio(entry.formula);

    for (const pressure of PRESSURES) {
      try {
        const vec = buildFeatureVector(entry.formula, pressure);
        if (vec.some(v => !Number.isFinite(v))) continue;

        let cs = entry.crystalSystem;
        let sg = entry.spacegroup;
        let proto = entry.prototype;
        const compressed = compressLattice(entry.lattice, pressure, K0);

        if (isHydride && hRatio > 0.5 && pressure >= 150) {
          const rule = HYDRIDE_HIGH_P_RULES[Math.floor(Math.random() * HYDRIDE_HIGH_P_RULES.length)];
          cs = rule.crystalSystem;
          sg = rule.spacegroup;
          proto = rule.prototype;
        } else if (!isHydride && pressure >= 100) {
          const metalRule = METAL_HIGH_P_RULES[Math.floor(Math.random() * METAL_HIGH_P_RULES.length)];
          if (Math.random() < 0.4) {
            cs = metalRule.crystalSystem;
            sg = metalRule.spacegroup;
            proto = metalRule.prototype;
          }
        }

        X.push(vec);
        csLabels.push(cs);
        sgLabels.push(String(sg));
        protoLabels.push(proto);
        latticeA.push(compressed.a);
        latticeB.push(compressed.b);
        latticeC.push(compressed.c);
        latticeAlpha.push(compressed.alpha);
        latticeBeta.push(compressed.beta);
        latticeGamma.push(compressed.gamma);
      } catch {
        continue;
      }
    }
  }

  for (const record of transitionRecords) {
    try {
      const vec = buildFeatureVector(record.formula, record.pAfter);
      if (vec.some(v => !Number.isFinite(v))) continue;
      X.push(vec);
      csLabels.push(record.structAfter);
      sgLabels.push("225");
      protoLabels.push(record.structAfter);
      latticeA.push(4.0);
      latticeB.push(4.0);
      latticeC.push(4.0);
      latticeAlpha.push(90);
      latticeBeta.push(90);
      latticeGamma.push(90);
    } catch {
      continue;
    }
  }

  return { X, csLabels, sgLabels, protoLabels, latticeA, latticeB, latticeC, latticeAlpha, latticeBeta, latticeGamma };
}

async function trainPressureStructureModel(): Promise<void> {
  const data = generateSyntheticTrainingData();
  if (data.X.length < 20) return;

  // Yield between each synchronous trainGBM / trainOneVsAllClassifier call
  // so timer callbacks and heartbeats can fire during the ~5-8s total training.
  crystalSystemClassifier = trainOneVsAllClassifier(data.X, data.csLabels, 30, 0.1, 3);
  await new Promise<void>(r => setTimeout(r, 0));
  spacegroupClassifier = trainOneVsAllClassifier(data.X, data.sgLabels, 30, 0.1, 3);
  await new Promise<void>(r => setTimeout(r, 0));
  prototypeClassifier = trainOneVsAllClassifier(data.X, data.protoLabels, 30, 0.1, 3);
  await new Promise<void>(r => setTimeout(r, 0));

  const a = trainGBM(data.X, data.latticeA, 30, 0.08, 3);
  await new Promise<void>(r => setTimeout(r, 0));
  const b = trainGBM(data.X, data.latticeB, 30, 0.08, 3);
  await new Promise<void>(r => setTimeout(r, 0));
  const c = trainGBM(data.X, data.latticeC, 30, 0.08, 3);
  await new Promise<void>(r => setTimeout(r, 0));
  const alpha = trainGBM(data.X, data.latticeAlpha, 20, 0.08, 3);
  await new Promise<void>(r => setTimeout(r, 0));
  const beta = trainGBM(data.X, data.latticeBeta, 20, 0.08, 3);
  await new Promise<void>(r => setTimeout(r, 0));
  const gamma = trainGBM(data.X, data.latticeGamma, 20, 0.08, 3);

  latticeRegressors = { a, b, c, alpha, beta, gamma };
  modelTrained = true;
  trainedAt = Date.now();
  datasetSize = data.X.length;
}

let trainDeferred = false;

function deferTraining(): void {
  if (trainDeferred) return;
  trainDeferred = true;
  setTimeout(() => {
    trainPressureStructureModel().catch(() => {});
  }, 8000);
}

function ensureTrained(): void {
  if (!modelTrained) {
    deferTraining();
  }
}

export function predictStructureAtPressure(formula: string, pressureGPa: number): PressureStructurePrediction {
  ensureTrained();
  predictionCount++;

  if (!crystalSystemClassifier || !spacegroupClassifier || !prototypeClassifier || !latticeRegressors) {
    return fallbackPrediction(formula, pressureGPa);
  }

  try {
    const vec = buildFeatureVector(formula, pressureGPa);
    if (vec.some(v => !Number.isFinite(v))) return fallbackPrediction(formula, pressureGPa);

    const csPred = predictClassifier(crystalSystemClassifier, vec);
    const sgPred = predictClassifier(spacegroupClassifier, vec);
    const protoPred = predictClassifier(prototypeClassifier, vec);

    const sgNum = parseInt(sgPred.predicted) || 225;
    const a = Math.max(1.0, predictGBM(latticeRegressors.a, vec));
    const b = Math.max(1.0, predictGBM(latticeRegressors.b, vec));
    const c = Math.max(1.0, predictGBM(latticeRegressors.c, vec));
    const alpha = Math.max(30, Math.min(150, predictGBM(latticeRegressors.alpha, vec)));
    const beta = Math.max(30, Math.min(150, predictGBM(latticeRegressors.beta, vec)));
    const gamma = Math.max(30, Math.min(150, predictGBM(latticeRegressors.gamma, vec)));

    const csProb = Math.max(...Object.values(csPred.probabilities));
    const sgProb = Math.max(...Object.values(sgPred.probabilities));
    const confidence = Math.round(((csProb + sgProb) / 2) * 10000) / 10000;

    return {
      crystalSystem: csPred.predicted,
      spacegroup: sgNum,
      spacegroupSymbol: SPACEGROUP_SYMBOLS[sgNum] || `SG-${sgNum}`,
      latticeParams: {
        a: Math.round(a * 1000) / 1000,
        b: Math.round(b * 1000) / 1000,
        c: Math.round(c * 1000) / 1000,
        alpha: Math.round(alpha * 100) / 100,
        beta: Math.round(beta * 100) / 100,
        gamma: Math.round(gamma * 100) / 100,
      },
      prototype: protoPred.predicted,
      confidence,
    };
  } catch {
    return fallbackPrediction(formula, pressureGPa);
  }
}

function fallbackPrediction(formula: string, pressureGPa: number): PressureStructurePrediction {
  const isH = hasHydrogen(formula);
  const hR = hydrogenRatio(formula);

  let cs = "cubic";
  let sg = 225;
  let proto = "FCC";

  if (isH && hR > 0.5 && pressureGPa >= 150) {
    cs = "cubic";
    sg = hR > 0.7 ? 225 : 229;
    proto = hR > 0.7 ? "sodalite-clathrate" : "clathrate";
  } else if (pressureGPa >= 100) {
    cs = "cubic";
    sg = 225;
    proto = "FCC";
  }

  const K0 = estimateBulkModulus(formula);
  const vRatio = murnaghanVolumeRatio(pressureGPa, K0);
  const scale = Math.pow(vRatio, 1 / 3);
  const baseLattice = 4.0;

  return {
    crystalSystem: cs,
    spacegroup: sg,
    spacegroupSymbol: SPACEGROUP_SYMBOLS[sg] || `SG-${sg}`,
    latticeParams: {
      a: Math.round(baseLattice * scale * 1000) / 1000,
      b: Math.round(baseLattice * scale * 1000) / 1000,
      c: Math.round(baseLattice * scale * 1000) / 1000,
      alpha: 90,
      beta: 90,
      gamma: cs === "hexagonal" ? 120 : 90,
    },
    prototype: proto,
    confidence: 0.2,
  };
}

export function getPressurePhaseMap(formula: string): PressurePhaseEntry[] {
  const steps: number[] = [];
  for (let p = 0; p <= 350; p += 25) {
    steps.push(p);
  }
  return steps.map(p => {
    const pred = predictStructureAtPressure(formula, p);
    return {
      pressureGPa: p,
      ...pred,
    };
  });
}

export function learnPressureTransition(
  formula: string,
  pBefore: number,
  pAfter: number,
  structBefore: string,
  structAfter: string
): void {
  transitionRecords.push({
    formula,
    pBefore,
    pAfter,
    structBefore,
    structAfter,
    timestamp: Date.now(),
  });

  if (transitionRecords.length % 10 === 0) {
    modelTrained = false;
  }
}

export function getPressureStructureStats(): {
  modelTrained: boolean;
  trainedAt: number;
  datasetSize: number;
  predictionCount: number;
  transitionRecords: number;
  pressureLevels: number[];
  phaseTransitionPatterns: { pattern: string; count: number }[];
} {
  const patternMap = new Map<string, number>();
  for (const rec of transitionRecords) {
    const key = `${rec.structBefore} -> ${rec.structAfter}`;
    patternMap.set(key, (patternMap.get(key) || 0) + 1);
  }
  const patterns = Array.from(patternMap.entries())
    .map(([pattern, count]) => ({ pattern, count }))
    .sort((a, b) => b.count - a.count);

  return {
    modelTrained,
    trainedAt,
    datasetSize,
    predictionCount,
    transitionRecords: transitionRecords.length,
    pressureLevels: PRESSURES,
    phaseTransitionPatterns: patterns,
  };
}

export function initPressureStructureModel(): void {
  try {
    trainPressureStructureModel();
  } catch {}
}
