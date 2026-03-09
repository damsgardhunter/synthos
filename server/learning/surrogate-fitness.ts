import { gnnPredictWithUncertainty, type GNNPredictionWithUncertainty } from "./graph-neural-net";
import { extractFeatures } from "./ml-predictor";
import { gbPredictWithUncertainty, type XGBUncertaintyResult } from "./gradient-boost";
import { classifyFamily } from "./utils";

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
  source: "dft" | "xtb";
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
}

const MAX_EVAL_HISTORY = 500;
const MAX_RECENT_ERRORS = 20;
const MAX_WEIGHT_HISTORY = 50;

const evaluationHistory: EvaluationRecord[] = [];
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
  const recent = evaluationHistory.slice(-50);
  const meanAbsErr = recent.reduce((s, r) => s + Math.abs(r.predictedTc - r.actualTc), 0) / recent.length;
  return 1.0 - Math.min(0.25, meanAbsErr / 300);
}

function computeNoveltyScore(formula: string): number {
  const family = classifyFamily(formula);
  const familyCount = evaluationHistory.filter(r => r.family === family).length;

  let novelty = 1.0;
  if (familyCount > 20) novelty -= 0.3;
  else if (familyCount > 10) novelty -= 0.15;

  if (seenFormulas.has(formula)) novelty -= 0.5;

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
  const recentOverestimates = recent.filter(r => r.predictedTc > r.actualTc * 1.2).length / recent.length;
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

export function computeSurrogateFitness(formula: string, crystalPrototype?: string): SurrogateFitnessResult {
  const family = classifyFamily(formula);

  let gnnPred: GNNPredictionWithUncertainty | null = null;
  let xgbPred: XGBUncertaintyResult | null = null;

  try {
    gnnPred = gnnPredictWithUncertainty(formula, crystalPrototype);
  } catch {}

  try {
    const features = extractFeatures(formula);
    xgbPred = gbPredictWithUncertainty(features, formula);
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

  const tcComponent = currentWeights.predictedTc * predictedTcNorm * calibrationFactor;
  const fitness = tcComponent +
    currentWeights.stability * stabilityScore +
    currentWeights.synthesis * synthesisScore +
    currentWeights.novelty * noveltyScore +
    currentWeights.uncertainty * uncertaintyBonus +
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
  source: "dft" | "xtb" = "dft"
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

  const acc = getOrCreateFamilyAccumulator(family);
  const absErr = Math.abs(predicted.tc - actual.tc);
  acc.sumAbsErr += absErr;
  acc.sumSignedErr += (predicted.tc - actual.tc);
  if (predicted.tc > actual.tc * 1.2) acc.overestimates++;
  if (predicted.stable === actual.stable) acc.stableCorrect++;
  acc.count++;

  if (evaluationHistory.length % 20 === 0 && evaluationHistory.length >= 10) {
    adaptWeights();
  }
}

function adaptWeights(): void {
  adaptationCycle++;

  const recent = evaluationHistory.slice(-50);
  if (recent.length < 10) return;

  const tcErrors = recent.map(r => Math.abs(r.predictedTc - r.actualTc));
  const meanTcError = tcErrors.reduce((s, e) => s + e, 0) / tcErrors.length;

  const stabilityCorrect = recent.filter(r => r.predictedStable === r.actualStable).length / recent.length;

  const lr = 0.02;

  if (meanTcError > 30) {
    currentWeights.predictedTc = Math.max(0.20, currentWeights.predictedTc - lr);
    currentWeights.uncertainty = Math.min(0.15, currentWeights.uncertainty + lr * 0.5);
    currentWeights.stability = Math.min(0.35, currentWeights.stability + lr * 0.5);
  } else if (meanTcError < 10) {
    currentWeights.predictedTc = Math.min(0.55, currentWeights.predictedTc + lr);
    currentWeights.uncertainty = Math.max(0.02, currentWeights.uncertainty - lr * 0.3);
  }

  if (stabilityCorrect > 0.8) {
    currentWeights.stability = Math.min(0.35, currentWeights.stability + lr * 0.3);
  } else if (stabilityCorrect < 0.5) {
    currentWeights.stability = Math.max(0.10, currentWeights.stability - lr * 0.3);
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

  const globalAbsErr = evaluationHistory.length > 0
    ? evaluationHistory.reduce((s, r) => s + Math.abs(r.predictedTc - r.actualTc), 0) / evaluationHistory.length
    : 0;

  const globalOverestimate = evaluationHistory.length > 0
    ? evaluationHistory.filter(r => r.predictedTc > r.actualTc * 1.2).length / evaluationHistory.length
    : 0;

  const recentErrors = evaluationHistory.slice(-MAX_RECENT_ERRORS).map(r => ({
    formula: r.formula,
    predicted: r.predictedTc,
    actual: r.actualTc,
    error: r.predictedTc - r.actualTc,
  }));

  const expWeight = computeExplorationWeight();

  return {
    totalEvaluations: evaluationHistory.length,
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
  };
}

export function getCurrentFitnessWeights(): typeof currentWeights {
  return { ...currentWeights };
}

export function getEvaluationCount(): number {
  return evaluationHistory.length;
}
