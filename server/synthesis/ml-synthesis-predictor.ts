import { ELEMENTAL_DATA, getElementData, getMeltingPoint } from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";

interface SynthesisTreeNode {
  featureIndex: number;
  threshold: number;
  left: SynthesisTreeNode | number;
  right: SynthesisTreeNode | number;
}

interface SynthesisGBModel {
  trees: SynthesisTreeNode[];
  learningRate: number;
  basePrediction: number;
  featureNames: string[];
  trainedAt: number;
}

interface SynthesisTrainingEntry {
  formula: string;
  feasible: number;
}

interface SynthesisFeasibilityResult {
  feasibility: number;
  confidence: number;
  features: Record<string, number>;
  reasoning: string[];
}

interface SynthesisPredictorStats {
  trained: boolean;
  trainingSize: number;
  featureImportance: Record<string, number>;
}

const FEATURE_NAMES = [
  "weightedAvgEN",
  "stdEN",
  "miedemaFormationEnergy",
  "coordNumberEstimate",
  "shannonEntropy",
  "numElements",
  "totalAtomCount",
  "valenceElectronConcentration",
  "atomicRadiusMismatch",
  "meltingPointRange",
  "avgMeltingPoint",
  "maxMeltingPoint",
  "avgAtomicRadius",
  "avgBulkModulus",
  "hasTransitionMetal",
  "hasRareEarth",
  "hasHydrogen",
  "hasOxygen",
  "hFraction",
  "metalFraction",
];

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function extractSynthesisFeatures(formula: string): number[] {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);

  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }

  const enValues: number[] = [];
  const radiusValues: number[] = [];
  const mpValues: number[] = [];
  const bulkValues: number[] = [];
  let totalVE = 0;
  let hasTransitionMetal = false;
  let hasRareEarth = false;
  let hasHydrogen = elements.includes("H");
  let hasOxygen = elements.includes("O");

  const TM_SET = new Set(["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"]);
  const RE_SET = new Set(["La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]);
  const NONMETALS = new Set(["H","He","C","N","O","F","Ne","P","S","Cl","Ar","Se","Br","Kr","I","Xe","B","Si","Ge","As","Sb","Te"]);

  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;

    const en = data.paulingElectronegativity ?? 1.5;
    enValues.push(en);

    radiusValues.push(data.atomicRadius);

    const mp = data.meltingPoint;
    if (mp != null) mpValues.push(mp);

    if (data.bulkModulus != null && data.bulkModulus > 0) bulkValues.push(data.bulkModulus);

    totalVE += data.valenceElectrons * (counts[el] || 1);

    if (TM_SET.has(el)) hasTransitionMetal = true;
    if (RE_SET.has(el)) hasRareEarth = true;
  }

  let weightedAvgEN = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const en = data?.paulingElectronegativity ?? 1.5;
    weightedAvgEN += en * fractions[el];
  }

  let stdEN = 0;
  if (enValues.length > 1) {
    const meanEN = enValues.reduce((s, v) => s + v, 0) / enValues.length;
    stdEN = Math.sqrt(enValues.reduce((s, v) => s + (v - meanEN) ** 2, 0) / enValues.length);
  }

  let miedemaFormationEnergy = 0;
  if (elements.length >= 2) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const dA = ELEMENTAL_DATA[elements[i]];
        const dB = ELEMENTAL_DATA[elements[j]];
        if (!dA || !dB) continue;
        const phiA = dA.miedemaPhiStar;
        const phiB = dB.miedemaPhiStar;
        const nwsA = dA.miedemaNws13;
        const nwsB = dB.miedemaNws13;
        const vA = dA.miedemaV23;
        const vB = dB.miedemaV23;
        if (phiA == null || phiB == null || nwsA == null || nwsB == null || vA == null || vB == null) continue;
        const deltaPhi = phiA - phiB;
        const deltaNws = nwsA - nwsB;
        const nwsAvgInv = 2 / (1 / nwsA + 1 / nwsB);
        const fAB = 2 * fractions[elements[i]] * fractions[elements[j]];
        const vAvg = (vA * fractions[elements[i]] + vB * fractions[elements[j]]) / (fractions[elements[i]] + fractions[elements[j]]);
        const interfaceEnergy = (-14.1 * deltaPhi * deltaPhi + 9.4 * deltaNws * deltaNws) / nwsAvgInv;
        miedemaFormationEnergy += fAB * vAvg * interfaceEnergy;
      }
    }
    miedemaFormationEnergy /= totalAtoms;
  }

  let coordNumberEstimate = 0;
  if (elements.length === 1) coordNumberEstimate = 12;
  else if (elements.length === 2) coordNumberEstimate = 8;
  else if (elements.length === 3) coordNumberEstimate = 6;
  else coordNumberEstimate = 4;

  let shannonEntropy = 0;
  for (const el of elements) {
    const frac = fractions[el];
    if (frac > 0 && frac < 1) {
      shannonEntropy -= frac * Math.log2(frac);
    }
  }

  const vec = totalAtoms > 0 ? totalVE / totalAtoms : 0;

  let atomicRadiusMismatch = 0;
  if (radiusValues.length > 1) {
    const meanR = radiusValues.reduce((s, v) => s + v, 0) / radiusValues.length;
    atomicRadiusMismatch = Math.sqrt(radiusValues.reduce((s, v) => s + (v - meanR) ** 2, 0) / radiusValues.length) / Math.max(meanR, 1);
  }

  const meltingPointRange = mpValues.length > 1 ? Math.max(...mpValues) - Math.min(...mpValues) : 0;
  const avgMeltingPoint = mpValues.length > 0 ? mpValues.reduce((s, v) => s + v, 0) / mpValues.length : 1000;
  const maxMeltingPoint = mpValues.length > 0 ? Math.max(...mpValues) : 1000;
  const avgAtomicRadius = radiusValues.length > 0 ? radiusValues.reduce((s, v) => s + v, 0) / radiusValues.length : 130;
  const avgBulkModulus = bulkValues.length > 0 ? bulkValues.reduce((s, v) => s + v, 0) / bulkValues.length : 50;

  const hFraction = hasHydrogen ? (counts["H"] || 0) / totalAtoms : 0;

  let metalCount = 0;
  for (const el of elements) {
    if (!NONMETALS.has(el)) metalCount += counts[el] || 0;
  }
  const metalFraction = totalAtoms > 0 ? metalCount / totalAtoms : 0;

  return [
    weightedAvgEN,
    stdEN,
    miedemaFormationEnergy,
    coordNumberEstimate,
    shannonEntropy,
    elements.length,
    totalAtoms,
    vec,
    atomicRadiusMismatch,
    meltingPointRange,
    avgMeltingPoint,
    maxMeltingPoint,
    avgAtomicRadius,
    avgBulkModulus,
    hasTransitionMetal ? 1 : 0,
    hasRareEarth ? 1 : 0,
    hasHydrogen ? 1 : 0,
    hasOxygen ? 1 : 0,
    hFraction,
    metalFraction,
  ];
}

const KNOWN_SYNTHESIZABLE: SynthesisTrainingEntry[] = [
  { formula: "MgB2", feasible: 1.0 },
  { formula: "NbTi", feasible: 1.0 },
  { formula: "Nb3Sn", feasible: 1.0 },
  { formula: "Nb3Ge", feasible: 1.0 },
  { formula: "Nb3Al", feasible: 1.0 },
  { formula: "YBa2Cu3O7", feasible: 1.0 },
  { formula: "Bi2Sr2CaCu2O8", feasible: 1.0 },
  { formula: "La2CuO4", feasible: 1.0 },
  { formula: "LaFeAsO", feasible: 1.0 },
  { formula: "BaFe2As2", feasible: 1.0 },
  { formula: "FeSe", feasible: 1.0 },
  { formula: "FeTe", feasible: 1.0 },
  { formula: "NbN", feasible: 1.0 },
  { formula: "NbC", feasible: 1.0 },
  { formula: "TiN", feasible: 1.0 },
  { formula: "V3Si", feasible: 1.0 },
  { formula: "V3Ga", feasible: 1.0 },
  { formula: "PbMo6S8", feasible: 0.9 },
  { formula: "LiTi2O4", feasible: 1.0 },
  { formula: "SrTiO3", feasible: 1.0 },
  { formula: "BaTiO3", feasible: 1.0 },
  { formula: "LaAlO3", feasible: 1.0 },
  { formula: "Al2O3", feasible: 1.0 },
  { formula: "TiO2", feasible: 1.0 },
  { formula: "ZrO2", feasible: 1.0 },
  { formula: "CaH2", feasible: 1.0 },
  { formula: "NaH", feasible: 1.0 },
  { formula: "LiH", feasible: 1.0 },
  { formula: "PdH", feasible: 0.9 },
  { formula: "TiH2", feasible: 1.0 },
  { formula: "ZrH2", feasible: 1.0 },
  { formula: "CeRhIn5", feasible: 0.85 },
  { formula: "CeCoIn5", feasible: 0.85 },
  { formula: "UPt3", feasible: 0.8 },
  { formula: "CeCu2Si2", feasible: 0.85 },
  { formula: "MoS2", feasible: 1.0 },
  { formula: "NbSe2", feasible: 1.0 },
  { formula: "TaS2", feasible: 1.0 },
  { formula: "CuO", feasible: 1.0 },
  { formula: "Fe2O3", feasible: 1.0 },
  { formula: "NiO", feasible: 1.0 },
  { formula: "CoO", feasible: 1.0 },
  { formula: "MnO2", feasible: 1.0 },
  { formula: "SnO2", feasible: 1.0 },
  { formula: "GaAs", feasible: 1.0 },
  { formula: "InP", feasible: 1.0 },
  { formula: "GaN", feasible: 1.0 },
  { formula: "AlN", feasible: 1.0 },
  { formula: "SiC", feasible: 1.0 },
  { formula: "WC", feasible: 1.0 },
  { formula: "TiC", feasible: 1.0 },
  { formula: "ZrB2", feasible: 1.0 },
  { formula: "HfB2", feasible: 0.9 },
  { formula: "LaB6", feasible: 1.0 },
  { formula: "CaB6", feasible: 1.0 },
  { formula: "CuAl2", feasible: 1.0 },
  { formula: "NiAl", feasible: 1.0 },
  { formula: "TiAl", feasible: 1.0 },
  { formula: "FeAl", feasible: 1.0 },
  { formula: "CoSi2", feasible: 1.0 },
];

const KNOWN_UNFEASIBLE: SynthesisTrainingEntry[] = [
  { formula: "HeNe", feasible: 0.0 },
  { formula: "ArKr", feasible: 0.0 },
  { formula: "XeF6", feasible: 0.15 },
  { formula: "AuF7", feasible: 0.05 },
  { formula: "CuF8", feasible: 0.0 },
  { formula: "FeAu5", feasible: 0.1 },
  { formula: "NaCs3Rb2", feasible: 0.15 },
  { formula: "LiNaKRbCs", feasible: 0.1 },
  { formula: "HgTl3Bi2", feasible: 0.15 },
  { formula: "OsIr3Ru2PtRh", feasible: 0.2 },
  { formula: "WMoReTcOs", feasible: 0.25 },
  { formula: "H100Fe", feasible: 0.0 },
  { formula: "ClBrIF", feasible: 0.05 },
  { formula: "NeFe", feasible: 0.0 },
  { formula: "ArNi", feasible: 0.0 },
  { formula: "HeC", feasible: 0.0 },
  { formula: "NaNa", feasible: 0.0 },
  { formula: "AuPt9Ir3Os2", feasible: 0.2 },
  { formula: "La10Ce10Pr10", feasible: 0.15 },
  { formula: "BaK3Na2Li", feasible: 0.1 },
  { formula: "TlHg4Bi3", feasible: 0.1 },
  { formula: "CdHg3Tl2", feasible: 0.1 },
  { formula: "He10O", feasible: 0.0 },
  { formula: "Fe100C100", feasible: 0.0 },
  { formula: "Ne3Si", feasible: 0.0 },
  { formula: "ArO2", feasible: 0.0 },
  { formula: "He2Fe", feasible: 0.0 },
  { formula: "Kr3Nb", feasible: 0.0 },
  { formula: "Cs20F50", feasible: 0.0 },
  { formula: "Ba50O100", feasible: 0.0 },
  { formula: "Na100Cl100", feasible: 0.05 },
  { formula: "Au20Cu80O200", feasible: 0.0 },
  { formula: "Pt10Ir10Os10Ru10", feasible: 0.15 },
  { formula: "Li50H200", feasible: 0.0 },
  { formula: "Ag7F15", feasible: 0.0 },
  { formula: "HeNeArKr", feasible: 0.0 },
];

const MARGINAL_COMPOUNDS: SynthesisTrainingEntry[] = [
  { formula: "LaH10", feasible: 0.4 },
  { formula: "YH6", feasible: 0.4 },
  { formula: "YH9", feasible: 0.35 },
  { formula: "CaH6", feasible: 0.4 },
  { formula: "H3S", feasible: 0.45 },
  { formula: "ScH12", feasible: 0.25 },
  { formula: "ThH10", feasible: 0.3 },
  { formula: "BaH12", feasible: 0.25 },
  { formula: "CsH7", feasible: 0.2 },
  { formula: "RbH9", feasible: 0.2 },
  { formula: "SrVO3", feasible: 0.7 },
  { formula: "Sr2RuO4", feasible: 0.75 },
  { formula: "Na2IrO3", feasible: 0.65 },
  { formula: "CaKFe4As4", feasible: 0.6 },
  { formula: "Tl2Ba2CaCu2O8", feasible: 0.55 },
  { formula: "HgBa2CaCu2O6", feasible: 0.5 },
];

function getAllTrainingData(): SynthesisTrainingEntry[] {
  return [...KNOWN_SYNTHESIZABLE, ...KNOWN_UNFEASIBLE, ...MARGINAL_COMPOUNDS];
}

function findBestSplit(
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

function buildSynthesisTree(
  X: number[][],
  residuals: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamples: number
): SynthesisTreeNode | number {
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
    const split = findBestSplit(X, residuals, indices, fi);
    if (split.improvement > bestImprovement && split.leftIndices.length >= 1 && split.rightIndices.length >= 1) {
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
    left: buildSynthesisTree(X, residuals, bestLeftIdx, depth + 1, maxDepth, minSamples),
    right: buildSynthesisTree(X, residuals, bestRightIdx, depth + 1, maxDepth, minSamples),
  };
}

function predictSynthesisTree(tree: SynthesisTreeNode | number, x: number[]): number {
  if (typeof tree === "number") return tree;
  if (x[tree.featureIndex] <= tree.threshold) {
    return predictSynthesisTree(tree.left, x);
  }
  return predictSynthesisTree(tree.right, x);
}

function trainSynthesisGB(
  X: number[][],
  y: number[],
  nEstimators: number = 100,
  learningRate: number = 0.1,
  maxDepth: number = 3
): SynthesisGBModel {
  const n = X.length;
  const allIndices = Array.from({ length: n }, (_, i) => i);
  const basePrediction = y.reduce((s, v) => s + v, 0) / n;
  const predictions = new Array(n).fill(basePrediction);
  const trees: SynthesisTreeNode[] = [];

  for (let iter = 0; iter < nEstimators; iter++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);
    const tree = buildSynthesisTree(X, residuals, allIndices, 0, maxDepth, 2);
    if (typeof tree === "number") break;
    trees.push(tree);
    for (let i = 0; i < n; i++) {
      predictions[i] += learningRate * predictSynthesisTree(tree, X[i]);
    }
    const mse = y.reduce((s, yi, i) => s + (yi - predictions[i]) ** 2, 0) / n;
    if (mse < 0.001) break;
  }

  return {
    trees,
    learningRate,
    basePrediction,
    featureNames: FEATURE_NAMES,
    trainedAt: Date.now(),
  };
}

function predictWithSynthesisModel(model: SynthesisGBModel, x: number[]): number {
  let prediction = model.basePrediction;
  for (const tree of model.trees) {
    const treeVal = predictSynthesisTree(tree, x);
    if (!Number.isFinite(treeVal)) continue;
    prediction += model.learningRate * treeVal;
  }
  if (!Number.isFinite(prediction)) return model.basePrediction;
  return Math.max(0, Math.min(1, prediction));
}

function getTreeFeatureImportanceSynthesis(tree: SynthesisTreeNode | number): Map<number, number> {
  const imp = new Map<number, number>();
  if (typeof tree === "number") return imp;
  imp.set(tree.featureIndex, (imp.get(tree.featureIndex) || 0) + 1);
  const leftImp = getTreeFeatureImportanceSynthesis(tree.left);
  const rightImp = getTreeFeatureImportanceSynthesis(tree.right);
  Array.from(leftImp.entries()).forEach(([k, v]) => imp.set(k, (imp.get(k) || 0) + v));
  Array.from(rightImp.entries()).forEach(([k, v]) => imp.set(k, (imp.get(k) || 0) + v));
  return imp;
}

let cachedSynthesisModel: SynthesisGBModel | null = null;
let cachedFeatureImportance: Record<string, number> | null = null;
let trainingSize = 0;

function getTrainedSynthesisModel(): SynthesisGBModel {
  if (cachedSynthesisModel) return cachedSynthesisModel;

  const data = getAllTrainingData();
  const X: number[][] = [];
  const y: number[] = [];

  for (const entry of data) {
    try {
      const features = extractSynthesisFeatures(entry.formula);
      if (features.some(v => !Number.isFinite(v))) continue;
      X.push(features);
      y.push(entry.feasible);
    } catch {
      continue;
    }
  }

  trainingSize = X.length;

  if (X.length < 5) {
    cachedSynthesisModel = {
      trees: [],
      learningRate: 0.1,
      basePrediction: 0.5,
      featureNames: FEATURE_NAMES,
      trainedAt: Date.now(),
    };
    return cachedSynthesisModel;
  }

  cachedSynthesisModel = trainSynthesisGB(X, y, 150, 0.1, 3);

  const importance = new Map<number, number>();
  for (const tree of cachedSynthesisModel.trees) {
    const imp = getTreeFeatureImportanceSynthesis(tree);
    Array.from(imp.entries()).forEach(([k, v]) => {
      importance.set(k, (importance.get(k) || 0) + v);
    });
  }

  let totalSplits = 0;
  Array.from(importance.values()).forEach(v => { totalSplits += v; });
  totalSplits = totalSplits || 1;
  cachedFeatureImportance = {};
  Array.from(importance.entries()).forEach(([idx, count]) => {
    const name = FEATURE_NAMES[idx] || `feature_${idx}`;
    cachedFeatureImportance![name] = Math.round((count / totalSplits) * 10000) / 10000;
  });

  return cachedSynthesisModel;
}

export function predictSynthesisFeasibility(
  formula: string,
  reactionType?: string
): SynthesisFeasibilityResult {
  const model = getTrainedSynthesisModel();
  const featureArray = extractSynthesisFeatures(formula);
  const rawPrediction = predictWithSynthesisModel(model, featureArray);
  const reasoning: string[] = [];

  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);

  const features: Record<string, number> = {};
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    features[FEATURE_NAMES[i]] = Math.round(featureArray[i] * 10000) / 10000;
  }

  let feasibility = rawPrediction;

  const NOBLE_GASES = new Set(["He", "Ne", "Ar", "Kr", "Xe"]);

  if (elements.length === 1) {
    if (NOBLE_GASES.has(elements[0])) {
      feasibility = 0.05;
      reasoning.push("Noble gas element - cannot form stable compounds");
    } else {
      feasibility = 1.0;
      reasoning.push("Pure element - trivially available");
    }
  } else if (elements.some(e => NOBLE_GASES.has(e))) {
    feasibility *= 0.1;
    reasoning.push("Contains noble gas element - extremely unlikely to form compounds");
  }

  const hFrac = (counts["H"] || 0) / totalAtoms;
  if (hFrac > 0.8) {
    feasibility = Math.min(feasibility, 0.2);
    reasoning.push("Extremely hydrogen-rich - requires extreme pressures for stabilization");
  } else if (hFrac > 0.6) {
    feasibility = Math.min(feasibility, 0.4);
    reasoning.push("High hydrogen content - likely requires high-pressure synthesis");
  }

  const miedemaE = featureArray[2];
  if (miedemaE < -0.5) {
    feasibility = Math.min(1, feasibility + 0.1);
    reasoning.push(`Negative Miedema formation energy (${miedemaE.toFixed(3)} eV/atom) - thermodynamically favorable`);
  } else if (miedemaE > 0.3) {
    feasibility = Math.max(0, feasibility - 0.1);
    reasoning.push(`Positive Miedema formation energy (${miedemaE.toFixed(3)} eV/atom) - thermodynamically unfavorable`);
  } else {
    reasoning.push(`Near-zero formation energy (${miedemaE.toFixed(3)} eV/atom) - marginal thermodynamic driving force`);
  }

  if (elements.length >= 5) {
    feasibility = Math.max(0, feasibility - 0.05 * (elements.length - 4));
    reasoning.push(`Complex ${elements.length}-element compound - higher synthesis difficulty`);
  }

  const family = classifyFamily(formula);
  if (reactionType) {
    reasoning.push(`Reaction type: ${reactionType}`);
  }
  reasoning.push(`Material family: ${family}`);

  const radiusMismatch = featureArray[8];
  if (radiusMismatch > 0.3) {
    feasibility = Math.max(0, feasibility - 0.05);
    reasoning.push(`Large atomic radius mismatch (${(radiusMismatch * 100).toFixed(1)}%) - structural strain expected`);
  }

  const vec = featureArray[7];
  if (vec > 0 && vec < 12) {
    reasoning.push(`Valence electron concentration: ${vec.toFixed(2)}`);
  }

  feasibility = Math.max(0, Math.min(1, feasibility));
  feasibility = Math.round(feasibility * 10000) / 10000;

  let confidence = 0.5;
  if (model.trees.length > 50) confidence += 0.2;
  else if (model.trees.length > 20) confidence += 0.1;

  if (elements.every(e => ELEMENTAL_DATA[e])) confidence += 0.15;
  else confidence -= 0.1;

  if (elements.length <= 3) confidence += 0.1;
  else if (elements.length >= 5) confidence -= 0.1;

  confidence = Math.max(0.1, Math.min(0.95, confidence));
  confidence = Math.round(confidence * 100) / 100;

  return {
    feasibility,
    confidence,
    features,
    reasoning,
  };
}

export function getSynthesisPredictorStats(): SynthesisPredictorStats {
  const model = getTrainedSynthesisModel();
  return {
    trained: model.trees.length > 0,
    trainingSize,
    featureImportance: cachedFeatureImportance || {},
  };
}
