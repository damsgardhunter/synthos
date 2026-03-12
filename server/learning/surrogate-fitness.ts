import { gnnPredictWithUncertainty, type GNNPredictionWithUncertainty } from "./graph-neural-net";
import { recordPredictionVariance } from "./uncertainty-tracker";
import { extractFeatures } from "./ml-predictor";
import { gbPredictWithUncertainty, type XGBUncertaintyResult } from "./gradient-boost";
import { classifyFamily } from "./utils";
import { computeCompositionFeatures } from "./composition-features";

export interface EvaluationRecord {
  formula: string;
  family: string;
  predictedTc: number;
  actualTc: number;
  predictedStable: boolean;
  actualStable: boolean;
  predictedFormationEnergy: number;
  actualFormationEnergy: number | null;
  timestamp: number;
  source: "dft" | "xtb" | "surrogate";
}

export interface FamilyCalibration {
  family: string;
  sampleCount: number;
  meanAbsError: number;
  meanSignedError: number;
  overestimateRatio: number;
  calibrationFactor: number;
  stabilityAccuracy: number;
}

export interface UncertaintyBreakdown {
  gnnEnsembleVariance: number;
  xgbEnsembleVariance: number;
  combined: number;
  explorationWeight: number;
  explorationBonus: number;
}

export interface SurrogateFitnessResult {
  fitness: number;
  components: {
    predictedTcNorm: number;
    stabilityScore: number;
    synthesisScore: number;
    noveltyScore: number;
    uncertaintyBonus: number;
  };
  uncertaintyBreakdown: UncertaintyBreakdown;
  calibrationFactor: number;
  rawFitness: number;
  gnnPrediction: GNNPredictionWithUncertainty | null;
  xgbPrediction: XGBUncertaintyResult | null;
  family: string;
}

export interface FeedbackLoopStats {
  totalEvaluations: number;
  globalMeanAbsError: number;
  globalOverestimateRatio: number;
  familyCalibrations: FamilyCalibration[];
  recentErrors: { formula: string; predicted: number; actual: number; error: number }[];
  fitnessWeightEvolution: { cycle: number; weights: typeof currentWeights }[];
  explorationWeight: number;
  explorationSchedule: { maxWeight: number; minWeight: number; decayHalfLife: number; currentWeight: number };
  noveltySearch: { knownCompositions: number; vectorDimensions: number };
}

export interface RawCalibrationStats {
  globalMeanAbsError: number;
  globalOverestimateRatio: number;
  explorationWeight: number;
  familyCalibrations: FamilyCalibration[];
}

const MAX_EVAL_HISTORY = 500;
const MAX_RECENT_ERRORS = 20;
const MAX_WEIGHT_HISTORY = 50;

const evaluationHistory: EvaluationRecord[] = [];
let lastRetrainEvalIndex = 0;
const familyErrorAccumulators: Map<string, { sumAbsErr: number; sumSignedErr: number; overestimates: number; stableCorrect: number; count: number }> = new Map();

let currentWeights = {
  predictedTc: 0.45,
  stability: 0.25,
  synthesis: 0.15,
  novelty: 0.10,
  uncertainty: 0.05,
};

const weightHistory: { cycle: number; weights: typeof currentWeights }[] = [];
let adaptationCycle = 0;

const seenFormulas = new Set<string>();

function getOrCreateFamilyAccumulator(family: string) {
  if (!familyErrorAccumulators.has(family)) {
    familyErrorAccumulators.set(family, { sumAbsErr: 0, sumSignedErr: 0, overestimates: 0, stableCorrect: 0, count: 0 });
  }
  return familyErrorAccumulators.get(family)!;
}

function getFamilyCalibrationFactor(family: string): number {
  const acc = familyErrorAccumulators.get(family);
  if (!acc || acc.count < 3) return 1.0;

  const meanAbsErr = acc.sumAbsErr / acc.count;
  const overestimateRatio = acc.overestimates / acc.count;

  if (overestimateRatio > 0.7 && meanAbsErr > 30) {
    return 0.6;
  } else if (overestimateRatio > 0.5 && meanAbsErr > 15) {
    return 0.75;
  } else if (meanAbsErr > 50) {
    return 0.7;
  } else if (meanAbsErr < 5 && overestimateRatio < 0.3) {
    return 1.1;
  }
  return 1.0 - Math.min(0.3, meanAbsErr / 200);
}

function getGlobalCalibrationFactor(): number {
  if (evaluationHistory.length < 5) return 1.0;

  const postRetrainCount = evaluationHistory.length - lastRetrainEvalIndex;
  const isPostRetrain = postRetrainCount > 0 && postRetrainCount < 50;

  if (isPostRetrain && postRetrainCount < 5) {
    return 1.0;
  }

  let samples: EvaluationRecord[];
  if (isPostRetrain) {
    samples = evaluationHistory.slice(lastRetrainEvalIndex);
  } else {
    samples = evaluationHistory.slice(-50);
  }

  let totalWeight = 0;
  let weightedAbsErr = 0;
  for (let i = 0; i < samples.length; i++) {
    const recency = isPostRetrain ? (1 + i) / samples.length : 1.0;
    const w = 0.5 + 0.5 * recency;
    weightedAbsErr += Math.abs(samples[i].predictedTc - samples[i].actualTc) * w;
    totalWeight += w;
  }
  const meanAbsErr = totalWeight > 0 ? weightedAbsErr / totalWeight : 0;
  return 1.0 - Math.min(0.25, meanAbsErr / 300);
}

export function notifyModelRetrain(): void {
  lastRetrainEvalIndex = evaluationHistory.length;
}

const COMP_VECTOR_KEYS = [
  "enMean", "enStd", "radiusMean", "radiusStd", "massMean", "massStd",
  "vecMean", "vecStd", "ieMean", "ieStd", "eaMean", "eaStd",
  "debyeMean", "debyeStd", "bulkModMean", "bulkModStd", "meltMean", "meltStd",
  "density", "volPerAtom", "stoner", "hopfield", "gruneisen",
  "ionic", "covalent", "metallic", "dFrac", "fFrac", "pFrac", "sFrac",
  "entropy", "pettiMean", "pettiStd",
] as const;

const MAX_KNOWN_VECTORS = 2000;
const knownVectorBuffer: { formula: string; vec: number[] }[] = new Array(MAX_KNOWN_VECTORS);
let kvHead = 0;
let kvCount = 0;
const knownVectorFormulas = new Set<string>();

const COMP_VECTOR_RANGES: { min: number; max: number }[] = [];
let _compRangesInitialized = false;

function initCompVectorRanges(): void {
  if (_compRangesInitialized) return;
  const defaultRanges: Record<string, [number, number]> = {
    enMean: [0.7, 3.5], enStd: [0, 1.5],
    radiusMean: [50, 250], radiusStd: [0, 100],
    massMean: [1, 240], massStd: [0, 120],
    vecMean: [1, 14], vecStd: [0, 6],
    ieMean: [3, 25], ieStd: [0, 10],
    eaMean: [-1, 4], eaStd: [0, 3],
    debyeMean: [50, 800], debyeStd: [0, 400],
    bulkModMean: [1, 400], bulkModStd: [0, 200],
    meltMean: [200, 4000], meltStd: [0, 2000],
    density: [0.5, 25], volPerAtom: [5, 80],
    stoner: [0, 2], hopfield: [0, 5], gruneisen: [0, 3],
    ionic: [0, 1], covalent: [0, 1], metallic: [0, 1],
    dFrac: [0, 1], fFrac: [0, 1], pFrac: [0, 1], sFrac: [0, 1],
    entropy: [0, 3], pettiMean: [0, 1], pettiStd: [0, 0.5],
  };
  for (const key of COMP_VECTOR_KEYS) {
    const range = defaultRanges[key] ?? [0, 1];
    COMP_VECTOR_RANGES.push({ min: range[0], max: range[1] });
  }
  _compRangesInitialized = true;
}

function compositionToVector(formula: string): number[] | null {
  try {
    initCompVectorRanges();
    const feats = computeCompositionFeatures(formula);
    const vec: number[] = [];
    for (let i = 0; i < COMP_VECTOR_KEYS.length; i++) {
      const key = COMP_VECTOR_KEYS[i];
      const val = (feats as any)[key];
      const raw = typeof val === "number" && isFinite(val) ? val : 0;
      const range = COMP_VECTOR_RANGES[i];
      const span = range.max - range.min;
      vec.push(span > 0 ? Math.max(0, Math.min(1, (raw - range.min) / span)) : 0);
    }
    return vec;
  } catch {
    return null;
  }
}

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = (a[i] || 0) - (b[i] || 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function addToKnownVectors(formula: string): void {
  if (knownVectorFormulas.has(formula)) return;
  const vec = compositionToVector(formula);
  if (!vec) return;

  if (kvCount >= MAX_KNOWN_VECTORS) {
    const evicted = knownVectorBuffer[kvHead];
    if (evicted) knownVectorFormulas.delete(evicted.formula);
  }

  knownVectorBuffer[kvHead] = { formula, vec };
  knownVectorFormulas.add(formula);
  kvHead = (kvHead + 1) % MAX_KNOWN_VECTORS;
  if (kvCount < MAX_KNOWN_VECTORS) kvCount++;
}

function getKnownVectors(): { formula: string; vec: number[] }[] {
  const result: { formula: string; vec: number[] }[] = [];
  for (let i = 0; i < kvCount; i++) {
    const entry = knownVectorBuffer[i];
    if (entry) result.push(entry);
  }
  return result;
}

function computeNoveltyScore(formula: string): number {
  if (seenFormulas.has(formula)) return 0.05;

  const family = classifyFamily(formula);
  const familyCount = evaluationHistory.filter(r => r.family === family).length;

  let novelty = 1.0;
  if (familyCount > 20) novelty -= 0.2;
  else if (familyCount > 10) novelty -= 0.1;

  const knownVecs = getKnownVectors();
  if (knownVecs.length >= 3) {
    const candidateVec = compositionToVector(formula);
    if (candidateVec) {
      const distances = knownVecs.map(kv => euclideanDistance(candidateVec, kv.vec));
      distances.sort((a, b) => a - b);
      const kNearest = Math.min(5, distances.length);
      const avgNearestDist = distances.slice(0, kNearest).reduce((s, d) => s + d, 0) / kNearest;
      const maxExpectedDist = 3.0;
      const distNovelty = Math.min(1.0, avgNearestDist / maxExpectedDist);
      novelty = 0.3 * novelty + 0.7 * distNovelty;
    }
  }

  return Math.max(0, Math.min(1, novelty));
}

function computeSynthesisHeuristicScore(formula: string): number {
  const elements = formula.match(/[A-Z][a-z]?/g) || [];
  const uniqueElements = [...new Set(elements)];
  let score = 0.5;
  if (uniqueElements.length <= 3) score += 0.15;
  if (uniqueElements.length <= 2) score += 0.1;
  if (uniqueElements.length > 4) score -= 0.2;

  const hasCommonElements = uniqueElements.some(e => ["Cu", "Fe", "Ni", "Ti", "Al", "Mg", "Ca", "Sr", "La", "Y", "Ba"].includes(e));
  if (hasCommonElements) score += 0.1;

  const hasRareElements = uniqueElements.some(e => ["Tc", "Pm", "At", "Fr", "Ra"].includes(e));
  if (hasRareElements) score -= 0.3;

  const hasToxicElements = uniqueElements.some(e => ["Be", "Tl", "Cd", "Hg", "Pb", "As"].includes(e));
  if (hasToxicElements) score -= 0.1;

  return Math.max(0, Math.min(1, score));
}

const EXPLORATION_WEIGHT_MAX = 0.25;
const EXPLORATION_WEIGHT_MIN = 0.03;
const EXPLORATION_DECAY_HALF_LIFE = 100;

function computeExplorationWeight(): number {
  const nEvals = evaluationHistory.length;
  if (nEvals < 5) return EXPLORATION_WEIGHT_MAX;
  const decayed = EXPLORATION_WEIGHT_MAX * Math.pow(0.5, nEvals / EXPLORATION_DECAY_HALF_LIFE);
  const recent = evaluationHistory.slice(-30);
  const recentOverestimates = recent.filter(r => {
    const margin = r.actualTc < 20 ? 5 : (r.actualTc < 77 ? r.actualTc * 0.3 : r.actualTc * 0.2);
    return r.predictedTc > r.actualTc + margin;
  }).length / recent.length;
  const recentMeanAbsErr = recent.reduce((s, r) => s + Math.abs(r.predictedTc - r.actualTc), 0) / recent.length;
  let boost = 0;
  if (recentMeanAbsErr > 25) boost += 0.05;
  if (recentOverestimates > 0.6) boost += 0.04;
  const uniqueFamilies = new Set(evaluationHistory.map(r => r.family)).size;
  const familyCoverage = Math.min(1, uniqueFamilies / 8);
  if (familyCoverage < 0.5) boost += 0.03;
  return Math.max(EXPLORATION_WEIGHT_MIN, Math.min(EXPLORATION_WEIGHT_MAX, decayed + boost));
}

export function getExplorationWeight(): number {
  return computeExplorationWeight();
}

export async function computeSurrogateFitness(formula: string, crystalPrototype?: string): Promise<SurrogateFitnessResult> {
  const family = classifyFamily(formula);

  let gnnPred: GNNPredictionWithUncertainty | null = null;
  let xgbPred: XGBUncertaintyResult | null = null;

  try {
    gnnPred = gnnPredictWithUncertainty(formula, crystalPrototype);
  } catch {}

  try {
    const features = await extractFeatures(formula);
    xgbPred = await gbPredictWithUncertainty(features, formula);
  } catch {}

  const gnnTc = gnnPred?.tc ?? 0;
  const xgbTc = xgbPred?.tcMean ?? 0;
  const predictedTc = gnnPred ? (xgbPred ? gnnTc * 0.6 + xgbTc * 0.4 : gnnTc) : xgbTc;
  const predictedTcNorm = Math.min(1.0, Math.max(0, predictedTc / 300));

  const gnnStability = gnnPred?.stabilityProbability ?? 0.5;
  const gnnPhononStable = gnnPred?.phononStability ? 1.0 : 0.0;
  const stabilityScore = gnnStability * 0.6 + gnnPhononStable * 0.4;

  const synthesisScore = computeSynthesisHeuristicScore(formula);
  const noveltyScore = computeNoveltyScore(formula);

  const gnnEnsembleVariance = gnnPred?.uncertainty ?? 0.5;
  const xgbEnsembleVariance = xgbPred?.normalizedUncertainty ?? 0.5;
  const combinedUncertainty = gnnPred ? (xgbPred ? gnnEnsembleVariance * 0.6 + xgbEnsembleVariance * 0.4 : gnnEnsembleVariance) : xgbEnsembleVariance;

  const ensembleStd = xgbPred?.tcStd ?? (gnnPred ? Math.sqrt(gnnPred.uncertainty) * predictedTc * 0.1 : 0);
  recordPredictionVariance({
    formula,
    predictedTc,
    ensembleStd,
    normalizedUncertainty: combinedUncertainty,
    confidenceLower: predictedTc - 1.645 * ensembleStd,
    confidenceUpper: predictedTc + 1.645 * ensembleStd,
    timestamp: Date.now(),
    family,
  });

  const explorationWeight = computeExplorationWeight();
  const uncertaintyBonus = Math.min(1.0, combinedUncertainty * 1.5);
  const explorationBonus = uncertaintyBonus * explorationWeight;

  const rawFitness =
    currentWeights.predictedTc * predictedTcNorm +
    currentWeights.stability * stabilityScore +
    currentWeights.synthesis * synthesisScore +
    currentWeights.novelty * noveltyScore +
    currentWeights.uncertainty * uncertaintyBonus;

  const familyCalFactor = getFamilyCalibrationFactor(family);
  const globalCalFactor = getGlobalCalibrationFactor();
  const calibrationFactor = (familyCalFactor + globalCalFactor) / 2;

  const modelDependentFitness =
    currentWeights.predictedTc * predictedTcNorm +
    currentWeights.stability * stabilityScore;

  const modelIndependentFitness =
    currentWeights.synthesis * synthesisScore +
    currentWeights.novelty * noveltyScore +
    currentWeights.uncertainty * uncertaintyBonus;

  const fitness = modelDependentFitness * calibrationFactor +
    modelIndependentFitness +
    explorationBonus;

  seenFormulas.add(formula);

  const uncertaintyBreakdown: UncertaintyBreakdown = {
    gnnEnsembleVariance,
    xgbEnsembleVariance,
    combined: combinedUncertainty,
    explorationWeight,
    explorationBonus,
  };

  return {
    fitness: Math.max(0, Math.min(1, fitness)),
    components: { predictedTcNorm, stabilityScore, synthesisScore, noveltyScore, uncertaintyBonus },
    uncertaintyBreakdown,
    calibrationFactor,
    rawFitness: Math.max(0, Math.min(1, rawFitness)),
    gnnPrediction: gnnPred,
    xgbPrediction: xgbPred,
    family,
  };
}

export function recordEvaluationResult(
  formula: string,
  predicted: { tc: number; stable: boolean; formationEnergy: number },
  actual: { tc: number; stable: boolean; formationEnergy: number | null },
  source: "dft" | "xtb" | "surrogate" = "dft"
): void {
  const family = classifyFamily(formula);

  const record: EvaluationRecord = {
    formula,
    family,
    predictedTc: predicted.tc,
    actualTc: actual.tc,
    predictedStable: predicted.stable,
    actualStable: actual.stable,
    predictedFormationEnergy: predicted.formationEnergy,
    actualFormationEnergy: actual.formationEnergy,
    timestamp: Date.now(),
    source,
  };

  evaluationHistory.push(record);
  if (evaluationHistory.length > MAX_EVAL_HISTORY) {
    evaluationHistory.splice(0, evaluationHistory.length - MAX_EVAL_HISTORY);
  }

  addToKnownVectors(formula);

  const acc = getOrCreateFamilyAccumulator(family);
  if (source !== "surrogate") {
    const absErr = Math.abs(predicted.tc - actual.tc);
    acc.sumAbsErr += absErr;
    acc.sumSignedErr += (predicted.tc - actual.tc);
    if (predicted.tc > actual.tc * 1.2) acc.overestimates++;
    if (predicted.stable === actual.stable) acc.stableCorrect++;
    acc.count++;
  }

  if (evaluationHistory.length % 20 === 0 && evaluationHistory.length >= 10) {
    adaptWeights();
  }
}

const FAMILY_TC_ERROR_THRESHOLDS: Record<string, { high: number; low: number }> = {
  Hydride: { high: 50, low: 20 },
  Cuprate: { high: 30, low: 10 },
  Pnictide: { high: 20, low: 8 },
  Intermetallic: { high: 5, low: 2 },
  HeavyFermion: { high: 3, low: 1 },
  Organic: { high: 5, low: 2 },
  Other: { high: 15, low: 5 },
};

function getFamilyTcThresholds(records: EvaluationRecord[]): { high: number; low: number } {
  const familyCounts = new Map<string, number>();
  for (const r of records) {
    familyCounts.set(r.family, (familyCounts.get(r.family) ?? 0) + 1);
  }

  let dominantFamily = "Other";
  let maxCount = 0;
  for (const [f, c] of familyCounts) {
    if (c > maxCount) { maxCount = c; dominantFamily = f; }
  }

  return FAMILY_TC_ERROR_THRESHOLDS[dominantFamily] ?? FAMILY_TC_ERROR_THRESHOLDS.Other;
}

function adaptWeights(): void {
  adaptationCycle++;

  const recent: EvaluationRecord[] = [];
  const startIdx = Math.max(0, evaluationHistory.length - 50);
  for (let i = startIdx; i < evaluationHistory.length; i++) {
    if (evaluationHistory[i].source !== "surrogate") recent.push(evaluationHistory[i]);
  }
  if (recent.length < 10) return;

  let sumTcErr = 0;
  let truePositives = 0;
  let falsePositives = 0;
  let falseNegatives = 0;
  for (const r of recent) {
    sumTcErr += Math.abs(r.predictedTc - r.actualTc);
    if (r.predictedStable && r.actualStable) truePositives++;
    else if (r.predictedStable && !r.actualStable) falsePositives++;
    else if (!r.predictedStable && r.actualStable) falseNegatives++;
  }
  const meanTcError = sumTcErr / recent.length;

  const precision = (truePositives + falsePositives) > 0
    ? truePositives / (truePositives + falsePositives) : 0.5;
  const recall = (truePositives + falseNegatives) > 0
    ? truePositives / (truePositives + falseNegatives) : 0.5;
  const f1 = (precision + recall) > 0
    ? 2 * precision * recall / (precision + recall) : 0;

  const thresholds = getFamilyTcThresholds(recent);
  const lr = 0.02;

  if (meanTcError > thresholds.high) {
    const tcReduction = Math.min(lr, currentWeights.predictedTc - 0.20);
    if (tcReduction > 0) {
      currentWeights.predictedTc -= tcReduction;
      currentWeights.uncertainty = Math.min(0.15, currentWeights.uncertainty + tcReduction * 0.4);
      currentWeights.novelty = Math.min(0.20, currentWeights.novelty + tcReduction * 0.4);
      currentWeights.stability += tcReduction * 0.2;
    }
  } else if (meanTcError < thresholds.low) {
    const tcIncrease = Math.min(lr, 0.55 - currentWeights.predictedTc);
    if (tcIncrease > 0) {
      currentWeights.predictedTc += tcIncrease;
      const shrinkPool = currentWeights.uncertainty + currentWeights.novelty;
      if (shrinkPool > 0.05) {
        const shrinkRatio = Math.min(tcIncrease, shrinkPool - 0.05) / shrinkPool;
        currentWeights.uncertainty *= (1 - shrinkRatio);
        currentWeights.novelty *= (1 - shrinkRatio);
      }
    }
  }

  if (f1 > 0.75 && precision > 0.7) {
    currentWeights.stability = Math.min(0.35, currentWeights.stability + lr * 0.3);
  } else if (precision < 0.5 || f1 < 0.4) {
    const stabReduction = Math.min(lr * 0.3, currentWeights.stability - 0.10);
    if (stabReduction > 0) {
      currentWeights.stability -= stabReduction;
      currentWeights.novelty += stabReduction * 0.5;
      currentWeights.uncertainty += stabReduction * 0.5;
    }
  }

  const total = Object.values(currentWeights).reduce((s, w) => s + w, 0);
  for (const key of Object.keys(currentWeights) as (keyof typeof currentWeights)[]) {
    currentWeights[key] /= total;
  }

  weightHistory.push({ cycle: adaptationCycle, weights: { ...currentWeights } });
  if (weightHistory.length > MAX_WEIGHT_HISTORY) {
    weightHistory.splice(0, weightHistory.length - MAX_WEIGHT_HISTORY);
  }
}

export function getCalibrationStats(): FeedbackLoopStats {
  const familyCalibrations: FamilyCalibration[] = [];
  for (const [family, acc] of familyErrorAccumulators.entries()) {
    if (acc.count === 0) continue;
    familyCalibrations.push({
      family,
      sampleCount: acc.count,
      meanAbsError: acc.sumAbsErr / acc.count,
      meanSignedError: acc.sumSignedErr / acc.count,
      overestimateRatio: acc.overestimates / acc.count,
      calibrationFactor: getFamilyCalibrationFactor(family),
      stabilityAccuracy: acc.stableCorrect / acc.count,
    });
  }

  familyCalibrations.sort((a, b) => b.sampleCount - a.sampleCount);

  const calibratedHistory = evaluationHistory.filter(r => r.source !== "surrogate");

  const globalAbsErr = calibratedHistory.length > 0
    ? calibratedHistory.reduce((s, r) => s + Math.abs(r.predictedTc - r.actualTc), 0) / calibratedHistory.length
    : 0;

  const globalOverestimate = calibratedHistory.length > 0
    ? calibratedHistory.filter(r => r.predictedTc > r.actualTc * 1.2).length / calibratedHistory.length
    : 0;

  const recentErrors = calibratedHistory.slice(-MAX_RECENT_ERRORS).map(r => ({
    formula: r.formula,
    predicted: r.predictedTc,
    actual: r.actualTc,
    error: r.predictedTc - r.actualTc,
  }));

  const expWeight = computeExplorationWeight();

  return {
    totalEvaluations: calibratedHistory.length,
    globalMeanAbsError: Math.round(globalAbsErr * 100) / 100,
    globalOverestimateRatio: Math.round(globalOverestimate * 1000) / 1000,
    familyCalibrations,
    recentErrors,
    fitnessWeightEvolution: weightHistory.slice(-20),
    explorationWeight: Math.round(expWeight * 1000) / 1000,
    explorationSchedule: {
      maxWeight: EXPLORATION_WEIGHT_MAX,
      minWeight: EXPLORATION_WEIGHT_MIN,
      decayHalfLife: EXPLORATION_DECAY_HALF_LIFE,
      currentWeight: Math.round(expWeight * 1000) / 1000,
    },
    noveltySearch: {
      knownCompositions: kvCount,
      vectorDimensions: COMP_VECTOR_KEYS.length,
    },
  };
}

export function getRawCalibrationStats(): RawCalibrationStats {
  const familyCalibrations: FamilyCalibration[] = [];
  for (const [family, acc] of familyErrorAccumulators.entries()) {
    if (acc.count === 0) continue;
    familyCalibrations.push({
      family,
      sampleCount: acc.count,
      meanAbsError: acc.sumAbsErr / acc.count,
      meanSignedError: acc.sumSignedErr / acc.count,
      overestimateRatio: acc.overestimates / acc.count,
      calibrationFactor: getFamilyCalibrationFactor(family),
      stabilityAccuracy: acc.stableCorrect / acc.count,
    });
  }

  const calibratedHistory = evaluationHistory.filter(r => r.source !== "surrogate");

  return {
    globalMeanAbsError: calibratedHistory.length > 0
      ? calibratedHistory.reduce((s, r) => s + Math.abs(r.predictedTc - r.actualTc), 0) / calibratedHistory.length
      : 0,
    globalOverestimateRatio: calibratedHistory.length > 0
      ? calibratedHistory.filter(r => r.predictedTc > r.actualTc * 1.2).length / calibratedHistory.length
      : 0,
    explorationWeight: computeExplorationWeight(),
    familyCalibrations,
  };
}

export function getCurrentFitnessWeights(): typeof currentWeights {
  return { ...currentWeights };
}

export function getEvaluationCount(): number {
  return evaluationHistory.length;
}
