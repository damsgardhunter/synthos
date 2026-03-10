import { storage } from "../storage";
import type { SuperconductorCandidate } from "@shared/schema";
import type { EventEmitter } from "./engine";
import { gnnPredictWithUncertainty } from "./graph-neural-net";
import { invalidateGNNModel, trainGNNSurrogate, trainEnsembleAsync, setCachedEnsemble, ENSEMBLE_SIZE, addDFTTrainingResult, getDFTTrainingDataset, logGNNVersion, getGNNModelVersion } from "./graph-neural-net";
import { resolveDFTFeatures, describeDFTSources } from "./dft-feature-resolver";
import { extractFeatures } from "./ml-predictor";
import { gbPredict, gbPredictWithUncertainty, incorporateFailureData, incorporateDFTResult, retrainXGBoostFromEvaluated, validateModel, getEvaluatedDatasetStats } from "./gradient-boost";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { computeDiscoveryScore } from "./family-filters";
import { recordEvaluationResult } from "./surrogate-fitness";
import { generateAdaptivePressureSamples, recordPressureCoverage, findOptimalPressure, predictPressureCurve } from "./pressure-aware-surrogate";
import { detectPhaseTransitions } from "./pressure-phase-detector";
import { estimateFamilyPressure } from "./candidate-generator";
import { runQuantumEnginePipeline, getQuantumEngineStats, type QuantumEngineResult } from "../dft/quantum-engine-pipeline";
import { getElementData } from "./elemental-data";
import { scoreFormulaNovelty } from "../crystal/structure-novelty-detector";
import { computeStructureEmbedding, estimateStructureUncertainty } from "../crystal/structure-embedding";
import { computeOODScore } from "./ood-detector";
import {
  addBatchFromEvaluation, startNewBatchCycle, recordBatchCycle,
  getCurrentCycleNumber, getGroundTruthSummary, getGroundTruthForLLM,
  getGroundTruthDataset, getRecentBatchCycles, getDatapointsByCycle,
  type BatchCycle, type GroundTruthDatapoint, type ModelPrediction,
} from "./ground-truth-store";
import {
  recordPredictionVsReality, checkRetrainTrigger, acknowledgeRetrain,
  computeMetrics, computeRecentMetrics, getMetricsForLLM,
  recordCycleImprovement,
} from "./prediction-reality-ledger";

export interface ActiveLearningConvergence {
  totalDFTRuns: number;
  avgUncertaintyBefore: number;
  avgUncertaintyAfter: number;
  modelRetrains: number;
  bestTcFromLoop: number;
}

export interface DiscoveryEfficiency {
  usefulDiscoveries: number;
  totalEvaluations: number;
  efficiencyRatio: number;
  stableCount: number;
  unstableCount: number;
  highTcCount: number;
  failureBreakdown: {
    unstablePhonons: number;
    highFormationEnergy: number;
    nonMetallic: number;
    lowTc: number;
    pipelineCrash: number;
  };
}

export interface ActiveLearningCycleRecord {
  cycle: number;
  timestamp: number;
  candidatesSelected: number;
  dftSuccesses: number;
  dftFailures: number;
  avgGnnUncertainty: number;
  avgXgbUncertainty: number;
  avgCombinedUncertainty: number;
  uncertaintyAfter: number;
  uncertaintyReductionPct: number;
  gnnRetrained: boolean;
  gnnVersion: number | null;
  tierBreakdown: { bestTc: number; highUncertainty: number; randomExploration: number; pressureExploration: number; pureCuriosity: number };
  topFormula: string;
  topAcquisitionScore: number;
  bestTcThisCycle: number;
  discoveryEfficiency: DiscoveryEfficiency;
}

const cycleHistory: ActiveLearningCycleRecord[] = [];
const MAX_CYCLE_HISTORY = 100;

export function getActiveLearningCycleHistory(): ActiveLearningCycleRecord[] {
  return [...cycleHistory];
}

let convergenceStats: ActiveLearningConvergence = {
  totalDFTRuns: 0,
  avgUncertaintyBefore: 1.0,
  avgUncertaintyAfter: 1.0,
  modelRetrains: 0,
  bestTcFromLoop: 0,
};

let totalEnrichedSinceLastRetrain = 0;
let lastRetrainCycle = 0;
const RETRAIN_CYCLE_INTERVAL = 20;
const RETRAIN_DFT_THRESHOLD = 50;
const recentUncertaintyDrops: number[] = [];

const quantumEnginePipelineStats = {
  fullPipelineRuns: 0,
  fallbackRuns: 0,
  avgLambdaFromPipeline: 0,
  avgTcFromPipeline: 0,
  lambdaSum: 0,
  tcSum: 0,
  successfulPipelineRuns: 0,
};

export function getQuantumEnginePipelineStats() {
  const n = quantumEnginePipelineStats.successfulPipelineRuns || 1;
  const engineStats = getQuantumEngineStats();
  return {
    fullPipelineRuns: quantumEnginePipelineStats.fullPipelineRuns,
    fallbackRuns: quantumEnginePipelineStats.fallbackRuns,
    activeLearningAvgLambda: Number((quantumEnginePipelineStats.lambdaSum / n).toFixed(4)),
    activeLearningAvgTc: Number((quantumEnginePipelineStats.tcSum / n).toFixed(2)),
    successfulPipelineRuns: quantumEnginePipelineStats.successfulPipelineRuns,
    ...engineStats,
  };
}

export function getActiveLearningStats(): ActiveLearningConvergence {
  return { ...convergenceStats };
}

interface RankedCandidate {
  candidate: SuperconductorCandidate;
  acquisitionScore: number;
  normalizedTc: number;
  uncertainty: number;
  xgbUncertainty: number;
  selectionTier: "best-tc" | "high-uncertainty" | "random-exploration" | "pressure-exploration" | "pure-curiosity";
  targetPressureGpa?: number;
  eiScore?: number;
  ucbScore?: number;
  curiosityScore?: number;
  structuralDistance?: number;
}

export interface PressureCoverageStats {
  formulasWithCoverage: number;
  totalPressurePoints: number;
  avgPointsPerFormula: number;
  pressureTransitionsFound: number;
}

const pressureCoverageStats: PressureCoverageStats = {
  formulasWithCoverage: 0,
  totalPressurePoints: 0,
  avgPointsPerFormula: 0,
  pressureTransitionsFound: 0,
};

export function getPressureCoverageStats(): PressureCoverageStats {
  return { ...pressureCoverageStats };
}

function computeAdaptiveAlpha(): number {
  const baseAlpha = 2.0;
  const decayRate = 0.3;
  const minAlpha = 0.5;
  return Math.max(minAlpha, baseAlpha - decayRate * convergenceStats.modelRetrains);
}

function parseFormulaElements(formula: string): Record<string, number> {
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

const seenCompositions: Map<string, number[]> = new Map();

function computeCompositionVector(formula: string): number[] {
  const counts = parseFormulaElements(formula);
  const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const vec: number[] = [];
  for (const el of Object.keys(counts).sort()) {
    vec.push((counts[el] || 0) / total);
  }
  return vec;
}

function computeCompositionDistance(formula: string): number {
  const vec = computeCompositionVector(formula);
  if (seenCompositions.size === 0) return 1.0;

  let minDist = Infinity;
  seenCompositions.forEach((refVec) => {
    const maxLen = Math.max(vec.length, refVec.length);
    let sumSq = 0;
    for (let i = 0; i < maxLen; i++) {
      const diff = (vec[i] ?? 0) - (refVec[i] ?? 0);
      sumSq += diff * diff;
    }
    minDist = Math.min(minDist, Math.sqrt(sumSq));
  });
  return Math.min(1.0, minDist);
}

function computeElementRarity(formula: string): number {
  const counts = parseFormulaElements(formula);
  const elements = Object.keys(counts);
  if (elements.length === 0) return 0;

  const commonElements = new Set([
    "Fe", "Cu", "Ni", "Co", "Ti", "Al", "Mg", "Zn", "Mn", "Cr",
    "O", "N", "C", "H", "S", "Si", "B", "P", "Se", "Te",
    "La", "Y", "Ba", "Sr", "Ca", "Nb", "V", "Mo", "W", "Pb",
  ]);

  let raritySum = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const isCommon = commonElements.has(el);
    const atomicNumber = data?.atomicNumber ?? 50;
    let elRarity = isCommon ? 0.1 : 0.5;
    if (atomicNumber > 56 && atomicNumber <= 71) elRarity += 0.2;
    if (atomicNumber > 88) elRarity += 0.3;
    raritySum += Math.min(1.0, elRarity);
  }
  return Math.min(1.0, raritySum / elements.length);
}

function computeStructuralUniqueness(candidate: SuperconductorCandidate): number {
  const counts = parseFormulaElements(candidate.formula);
  const elements = Object.keys(counts);
  const nElements = elements.length;

  let uniqueness = 0;
  if (nElements >= 4) uniqueness += 0.3;
  else if (nElements >= 3) uniqueness += 0.15;

  const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const fractions = Object.values(counts).map(c => c / total);
  const entropy = -fractions.reduce((s, f) => s + (f > 0 ? f * Math.log2(f) : 0), 0);
  const maxEntropy = Math.log2(Math.max(nElements, 1)) || 1;
  uniqueness += 0.4 * (entropy / maxEntropy);

  const hasUnusualRatio = fractions.some(f => f > 0.7) || fractions.some(f => f < 0.05 && f > 0);
  if (hasUnusualRatio) uniqueness += 0.15;

  const family = (candidate as any).family ?? "";
  if (typeof family === "string" && family.length > 0) {
    const uncommonFamilies = ["skutterudite", "chevrel", "clathrate", "max-phase", "heusler"];
    if (uncommonFamilies.some(f => family.toLowerCase().includes(f))) {
      uniqueness += 0.15;
    }
  }

  return Math.min(1.0, uniqueness);
}

let trainingSetVectors: { formula: string; vec: number[] }[] | null = null;

function getTrainingSetVectors(): { formula: string; vec: number[] }[] {
  if (trainingSetVectors) return trainingSetVectors;
  const PERIODIC_ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","U"
  ];
  const elIndex = new Map<string, number>();
  PERIODIC_ELEMENTS.forEach((el, i) => elIndex.set(el, i));
  const dim = PERIODIC_ELEMENTS.length;

  trainingSetVectors = SUPERCON_TRAINING_DATA.map(entry => {
    const counts = parseFormulaElements(entry.formula);
    const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    const vec = new Array(dim).fill(0);
    for (const [el, n] of Object.entries(counts)) {
      const idx = elIndex.get(el);
      if (idx !== undefined) vec[idx] = n / total;
    }
    return { formula: entry.formula, vec };
  });
  return trainingSetVectors;
}

function computeFixedDimVector(formula: string): number[] {
  const PERIODIC_ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","U"
  ];
  const elIndex = new Map<string, number>();
  PERIODIC_ELEMENTS.forEach((el, i) => elIndex.set(el, i));
  const dim = PERIODIC_ELEMENTS.length;
  const counts = parseFormulaElements(formula);
  const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const vec = new Array(dim).fill(0);
  for (const [el, n] of Object.entries(counts)) {
    const idx = elIndex.get(el);
    if (idx !== undefined) vec[idx] = n / total;
  }
  return vec;
}

function computeStructuralDistanceFromTrainingSet(formula: string): number {
  const refVecs = getTrainingSetVectors();
  const candidateVec = computeFixedDimVector(formula);

  let minDist = Infinity;
  for (const ref of refVecs) {
    let sumSq = 0;
    for (let i = 0; i < candidateVec.length; i++) {
      const diff = candidateVec[i] - ref.vec[i];
      sumSq += diff * diff;
    }
    minDist = Math.min(minDist, Math.sqrt(sumSq));
  }
  return Math.min(1.0, minDist / 0.8);
}

function computeExpectedImprovement(
  predictedTc: number,
  sigma: number,
  bestTcSoFar: number
): number {
  if (sigma <= 1e-8) return predictedTc > bestTcSoFar ? (predictedTc - bestTcSoFar) / 300 : 0;

  const z = (predictedTc - bestTcSoFar) / sigma;
  const phi = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const PHI = 0.5 * (1 + erf(z / Math.SQRT2));

  const ei = sigma * (z * PHI + phi);
  return Math.min(1.0, ei / 300);
}

function erf(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1.0 / (1.0 + p * Math.abs(x));
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function computeUCB(
  predictedTc: number,
  sigma: number,
  kappa: number = 2.0
): number {
  return Math.min(1.0, (predictedTc + kappa * sigma) / 300);
}

function computeCuriosityScore(
  structuralDistance: number,
  uncertainty: number,
  oodScore: number,
  noveltyScore: number
): number {
  return Math.min(1.0,
    0.35 * structuralDistance +
    0.30 * uncertainty +
    0.20 * oodScore +
    0.15 * noveltyScore
  );
}

function computeNoveltyScore(candidate: SuperconductorCandidate): number {
  const compositionDist = computeCompositionDistance(candidate.formula);
  const elementRarity = computeElementRarity(candidate.formula);
  const structuralUniqueness = computeStructuralUniqueness(candidate);
  const baseNovelty = 0.4 * compositionDist + 0.3 * elementRarity + 0.3 * structuralUniqueness;

  let fingerprintNovelty = 0.5;
  try {
    const noveltyResult = scoreFormulaNovelty(candidate.formula);
    fingerprintNovelty = noveltyResult.noveltyScore;
  } catch {}

  return 0.6 * baseNovelty + 0.4 * fingerprintNovelty;
}

function computeStabilityProbability(candidate: SuperconductorCandidate, candidatePressure: number): number {
  let stabilityProb = 0.5;

  try {
    const gnnResult = gnnPredictWithUncertainty(candidate.formula, undefined, candidatePressure);
    if (gnnResult.stabilityProbability != null && Number.isFinite(gnnResult.stabilityProbability)) {
      stabilityProb = gnnResult.stabilityProbability;
    }
  } catch {}

  const candidateStability = (candidate as any).stability ?? (candidate as any).stabilityScore;
  if (candidateStability != null && Number.isFinite(candidateStability)) {
    stabilityProb = 0.6 * stabilityProb + 0.4 * Math.min(1.0, Math.max(0, candidateStability));
  }

  return Math.min(1.0, Math.max(0, stabilityProb));
}

export function selectForDFT(
  candidates: SuperconductorCandidate[],
  budget: number = 20
): RankedCandidate[] {
  const pureCuriositySlots = Math.max(1, Math.ceil(budget * 0.20));
  const pressureExplorationSlots = Math.min(2, Math.ceil(budget * 0.10));
  const bestTcSlots = Math.min(6, Math.ceil(budget * 0.25));
  const highUncertaintySlots = Math.min(6, Math.ceil(budget * 0.25));
  const randomSlots = Math.max(1, budget - bestTcSlots - highUncertaintySlots - pureCuriositySlots - pressureExplorationSlots);

  seenCompositions.clear();
  for (const c of candidates) {
    seenCompositions.set(c.formula, computeCompositionVector(c.formula));
  }

  const bestTcSoFar = convergenceStats.bestTcFromLoop > 0
    ? convergenceStats.bestTcFromLoop
    : Math.max(...candidates.map(c => c.predictedTc ?? 0), 39);

  const kappa = computeAdaptiveAlpha();

  const scored: {
    candidate: SuperconductorCandidate;
    normalizedTc: number;
    predictedTcRaw: number;
    sigmaRaw: number;
    gnnUncertainty: number;
    xgbUncertainty: number;
    combinedUncertainty: number;
    stabilityProbability: number;
    noveltyScore: number;
    acquisitionScore: number;
    eiScore: number;
    ucbScore: number;
    curiosityScore: number;
    structuralDistance: number;
    oodScore: number;
  }[] = [];

  for (const candidate of candidates) {
    const tc = candidate.predictedTc ?? 0;
    const normalizedTc = Math.min(1.0, Math.max(0, tc / 300));

    const candidatePressure = (candidate as any).pressureGpa ?? estimateFamilyPressure(candidate.formula);

    let gnnUncertainty = candidate.uncertaintyEstimate ?? 0.5;
    let gnnSigmaK = tc * 0.3;
    try {
      const gnnResult = gnnPredictWithUncertainty(candidate.formula, undefined, candidatePressure);
      gnnUncertainty = Math.max(gnnUncertainty, gnnResult.uncertainty);
      if (gnnResult.uncertaintyBreakdown) {
        gnnSigmaK = gnnResult.uncertaintyBreakdown.totalSigma ?? tc * gnnUncertainty;
      }
    } catch (e: any) { console.error("[ActiveLearning] GNN predict error:", e?.message?.slice(0, 200)); }

    let xgbUncertainty = 0.5;
    let xgbSigmaK = tc * 0.3;
    try {
      const features = extractFeatures(candidate.formula, { pressureGpa: candidatePressure } as any);
      const xgbResult = gbPredictWithUncertainty(features, candidate.formula);
      xgbUncertainty = xgbResult.normalizedUncertainty;
      xgbSigmaK = xgbResult.totalStd ?? xgbResult.tcStd ?? tc * xgbUncertainty;
    } catch (e: any) { console.error("[ActiveLearning] XGB uncertainty error:", e?.message?.slice(0, 200)); }

    let embeddingUncertainty = 0.5;
    try {
      const emb = computeStructureEmbedding(candidate.formula);
      embeddingUncertainty = estimateStructureUncertainty(emb);
    } catch {}

    const sigmaRaw = Math.sqrt(
      0.5 * gnnSigmaK * gnnSigmaK +
      0.5 * xgbSigmaK * xgbSigmaK
    );

    let baseCombinedUncertainty = 0.35 * gnnUncertainty + 0.35 * xgbUncertainty + 0.3 * embeddingUncertainty;

    let oodScoreVal = 0;
    try {
      const ood = computeOODScore(candidate.formula);
      oodScoreVal = ood.oodScore;
      if (ood.isOOD) {
        baseCombinedUncertainty = Math.min(1.0, baseCombinedUncertainty * (1 + ood.oodScore));
      }
    } catch {}

    const combinedUncertainty = baseCombinedUncertainty;

    const stabilityProbability = computeStabilityProbability(candidate, candidatePressure);

    const noveltyScore = computeNoveltyScore(candidate);

    const structuralDistance = computeStructuralDistanceFromTrainingSet(candidate.formula);

    const eiScore = computeExpectedImprovement(tc, sigmaRaw, bestTcSoFar);

    const ucbScore = computeUCB(tc, sigmaRaw, kappa);

    const curiosityScore = computeCuriosityScore(
      structuralDistance,
      combinedUncertainty,
      oodScoreVal,
      noveltyScore
    );

    const acquisitionScore =
      0.30 * eiScore +
      0.25 * ucbScore +
      0.20 * combinedUncertainty +
      0.10 * curiosityScore +
      0.10 * normalizedTc +
      0.05 * stabilityProbability;

    scored.push({
      candidate,
      normalizedTc,
      predictedTcRaw: tc,
      sigmaRaw,
      gnnUncertainty,
      xgbUncertainty,
      combinedUncertainty,
      stabilityProbability,
      noveltyScore,
      acquisitionScore,
      eiScore,
      ucbScore,
      curiosityScore,
      structuralDistance,
      oodScore: oodScoreVal,
    });
  }

  const selected: RankedCandidate[] = [];
  const seenFormulas = new Set<string>();

  const byEI = [...scored].sort((a, b) => b.eiScore - a.eiScore || b.ucbScore - a.ucbScore);
  for (const s of byEI) {
    if (selected.length >= bestTcSlots) break;
    if (seenFormulas.has(s.candidate.formula)) continue;
    seenFormulas.add(s.candidate.formula);
    selected.push({
      candidate: s.candidate,
      acquisitionScore: s.acquisitionScore,
      normalizedTc: s.normalizedTc,
      uncertainty: s.combinedUncertainty,
      xgbUncertainty: s.xgbUncertainty,
      selectionTier: "best-tc",
      eiScore: s.eiScore,
      ucbScore: s.ucbScore,
      structuralDistance: s.structuralDistance,
    });
  }

  const byUCB = [...scored].sort((a, b) => b.ucbScore - a.ucbScore || b.combinedUncertainty - a.combinedUncertainty);
  for (const s of byUCB) {
    if (selected.length >= bestTcSlots + highUncertaintySlots) break;
    if (seenFormulas.has(s.candidate.formula)) continue;
    seenFormulas.add(s.candidate.formula);
    selected.push({
      candidate: s.candidate,
      acquisitionScore: s.acquisitionScore,
      normalizedTc: s.normalizedTc,
      uncertainty: s.combinedUncertainty,
      xgbUncertainty: s.xgbUncertainty,
      selectionTier: "high-uncertainty",
      eiScore: s.eiScore,
      ucbScore: s.ucbScore,
      structuralDistance: s.structuralDistance,
    });
  }

  const byCuriosity = [...scored].sort((a, b) => b.curiosityScore - a.curiosityScore || b.structuralDistance - a.structuralDistance);
  for (const s of byCuriosity) {
    if (selected.filter(r => r.selectionTier === "pure-curiosity").length >= pureCuriositySlots) break;
    if (seenFormulas.has(s.candidate.formula)) continue;
    seenFormulas.add(s.candidate.formula);
    selected.push({
      candidate: s.candidate,
      acquisitionScore: s.acquisitionScore,
      normalizedTc: s.normalizedTc,
      uncertainty: s.combinedUncertainty,
      xgbUncertainty: s.xgbUncertainty,
      selectionTier: "pure-curiosity",
      eiScore: s.eiScore,
      ucbScore: s.ucbScore,
      curiosityScore: s.curiosityScore,
      structuralDistance: s.structuralDistance,
    });
  }

  const remaining = scored.filter(s => !seenFormulas.has(s.candidate.formula));
  for (let i = remaining.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
  }
  for (const s of remaining) {
    if (selected.length >= budget - pressureExplorationSlots) break;
    seenFormulas.add(s.candidate.formula);
    selected.push({
      candidate: s.candidate,
      acquisitionScore: s.acquisitionScore,
      normalizedTc: s.normalizedTc,
      uncertainty: s.combinedUncertainty,
      xgbUncertainty: s.xgbUncertainty,
      selectionTier: "random-exploration",
      eiScore: s.eiScore,
      ucbScore: s.ucbScore,
      structuralDistance: s.structuralDistance,
    });
  }

  const topForPressure = selected.slice(0, 5);
  let pressureAdded = 0;
  for (const ranked of topForPressure) {
    if (pressureAdded >= pressureExplorationSlots) break;
    try {
      const samples = generateAdaptivePressureSamples(ranked.candidate.formula, 2);
      for (const sample of samples) {
        if (pressureAdded >= pressureExplorationSlots || selected.length >= budget) break;
        selected.push({
          candidate: ranked.candidate,
          acquisitionScore: ranked.acquisitionScore + sample.uncertainty * 0.5,
          normalizedTc: ranked.normalizedTc,
          uncertainty: sample.uncertainty,
          xgbUncertainty: sample.uncertainty,
          selectionTier: "pressure-exploration",
          targetPressureGpa: sample.pressureGpa,
        });
        pressureAdded++;
      }
    } catch {}
  }

  return selected;
}

async function runDFTEnrichmentForCandidate(
  emit: EventEmitter,
  candidate: SuperconductorCandidate,
  pressureGpa?: number
): Promise<boolean> {
  const evalPressure = pressureGpa ?? (candidate as any).pressureGpa ?? estimateFamilyPressure(candidate.formula);

  let quantumResult: QuantumEngineResult | null = null;
  let usedQuantumPipeline = false;

  try {
    quantumResult = await runQuantumEnginePipeline(candidate.formula, evalPressure);
    usedQuantumPipeline = true;
    quantumEnginePipelineStats.fullPipelineRuns++;

    const entry = quantumResult.entry;
    const stepsSummary = quantumResult.steps
      .map(s => `${s.name}:${s.status}`)
      .join(", ");

    emit("log", {
      phase: "active-learning",
      event: "Quantum engine enrichment",
      detail: `${candidate.formula} @ ${evalPressure} GPa -- tier=${entry.tier}, lambda=${entry.lambda.toFixed(3)}, omegaLog=${entry.omegaLog.toFixed(1)}, Tc=${entry.tc.toFixed(1)}K, DOS(EF)=${entry.dosAtEF.toFixed(2)}, phonon_stable=${entry.isPhononStable}, confidence=${entry.confidence} [${stepsSummary}]`,
      dataSource: "Active Learning",
    });

    if (entry.lambda > 0 && entry.tc > 0) {
      quantumEnginePipelineStats.successfulPipelineRuns++;
      quantumEnginePipelineStats.lambdaSum += entry.lambda;
      quantumEnginePipelineStats.tcSum += entry.tc;
    }

    const features = extractFeatures(candidate.formula, undefined, undefined, undefined, undefined);
    const gb = gbPredict(features, candidate.formula);
    const nnScore = candidate.neuralNetScore ?? candidate.quantumCoherence ?? 0.3;
    const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);

    let confidence: string;
    if (entry.tier === "full-dft" && entry.confidence === "high") confidence = "high";
    else if (entry.tier === "full-dft" || entry.tier === "xtb") confidence = "medium";
    else confidence = "low";

    const quantumComputationalResults = {
      lambda: entry.lambda,
      omegaLog: entry.omegaLog,
      tc: entry.tc,
      dosAtEF: entry.dosAtEF,
      phononSpectrum: entry.phononSpectrum,
      alpha2FSummary: entry.alpha2FSummary,
      isStrongCoupling: entry.isStrongCoupling,
      gapRatio: entry.gapRatio,
      muStar: entry.muStar,
      omega2: entry.omega2,
      tcAllenDynes: entry.tcAllenDynes,
      tcEliashberg: entry.tcEliashberg,
      isotopeAlpha: entry.isotopeAlpha,
      tier: entry.tier,
      pressure: entry.pressure,
      wallTimeMs: entry.wallTimeMs,
    };

    const existingMlFeatures = (candidate.mlFeatures as Record<string, any>) || {};
    const updatedMlFeatures = {
      ...existingMlFeatures,
      quantumEngineResult: quantumComputationalResults,
    };

    const updates: any = {
      xgboostScore: gb.score,
      ensembleScore: ensemble,
      dataConfidence: confidence,
      electronPhononCoupling: entry.lambda,
      logPhononFrequency: entry.omegaLog,
      coulombPseudopotential: entry.muStar,
      mlFeatures: updatedMlFeatures,
    };

    if (entry.formationEnergy !== null) {
      updates.decompositionEnergy = entry.formationEnergy;
    }

    const reconciledTc = entry.tc > 0
      ? entry.tc * 0.7 + gb.tcPredicted * 0.3
      : gb.tcPredicted;
    if (reconciledTc > 0) {
      updates.predictedTc = Math.round(reconciledTc * 10) / 10;
    }

    await storage.updateSuperconductorCandidate(candidate.id, updates);

    const formEnergy = entry.formationEnergy;
    const isStable = entry.isPhononStable && (formEnergy === null || formEnergy < 0.5);
    const dftSource = entry.tier === "full-dft" ? "external" as const : "active-learning" as const;
    incorporateDFTResult(
      candidate.formula,
      reconciledTc,
      formEnergy,
      isStable,
      dftSource,
      entry.lambda > 0 ? entry.lambda : undefined,
      entry.omegaLog > 0 ? entry.omegaLog : undefined,
      entry.dosAtEF > 0 ? entry.dosAtEF : undefined
    );

    addDFTTrainingResult({
      formula: candidate.formula,
      tc: reconciledTc,
      formationEnergy: formEnergy,
      bandGap: entry.bandGap,
      structure: undefined,
      prototype: undefined,
      source: entry.tier === "full-dft" ? "external" : "active-learning",
      lambda: entry.lambda > 0 ? entry.lambda : undefined,
      omegaLog: entry.omegaLog > 0 ? entry.omegaLog : undefined,
      dosAtEF: entry.dosAtEF > 0 ? entry.dosAtEF : undefined,
      phononStable: entry.isPhononStable,
    });

    let gnnTcPredicted = 0;
    let gnnStablePredicted = true;
    let gnnFePredicted = 0;
    try {
      const gnnPred = gnnPredictWithUncertainty(candidate.formula);
      gnnTcPredicted = gnnPred.tc;
      gnnStablePredicted = gnnPred.phononStability;
      gnnFePredicted = gnnPred.formationEnergy;
    } catch {}

    const ensemblePredictedTc = gnnTcPredicted > 0
      ? gnnTcPredicted * 0.4 + gb.tcPredicted * 0.6
      : gb.tcPredicted;

    const modelPred: ModelPrediction = {
      predicted_Tc: ensemblePredictedTc,
      predicted_stable: gnnStablePredicted,
      predicted_formation_energy: gnnFePredicted || null,
      xgboost_Tc: gb.tcPredicted,
      gnn_Tc: gnnTcPredicted,
      ensemble_score: ensemble,
    };

    addBatchFromEvaluation(
      candidate.formula,
      evalPressure,
      {
        tc: reconciledTc,
        lambda: entry.lambda,
        omegaLog: entry.omegaLog,
        dosAtEF: entry.dosAtEF,
        formationEnergy: formEnergy,
        bandGap: entry.bandGap,
        phononStable: entry.isPhononStable,
        isStrongCoupling: entry.isStrongCoupling,
        muStar: entry.muStar,
        tier: entry.tier,
        confidence: confidence,
        wallTimeMs: entry.wallTimeMs,
      },
      getCurrentCycleNumber(),
      modelPred
    );

    recordPredictionVsReality(
      candidate.formula,
      evalPressure,
      {
        Tc: ensemblePredictedTc,
        stable: gnnStablePredicted,
        formation_energy: gnnFePredicted || null,
        xgboost_Tc: gb.tcPredicted,
        gnn_Tc: gnnTcPredicted,
      },
      {
        Tc: reconciledTc,
        stable: isStable,
        formation_energy: formEnergy,
        lambda: entry.lambda > 0 ? entry.lambda : null,
        DOS_EF: entry.dosAtEF > 0 ? entry.dosAtEF : null,
      },
      entry.tier,
      getCurrentCycleNumber()
    );

    const predictedTc = gnnTcPredicted > 0 ? gnnTcPredicted * 0.4 + reconciledTc * 0.6 : reconciledTc;
    const fidelity = entry.tier === "full-dft" ? "dft" as const : "xtb" as const;
    recordEvaluationResult(
      candidate.formula,
      { tc: predictedTc, stable: gnnStablePredicted, formationEnergy: gnnFePredicted },
      { tc: reconciledTc, stable: isStable, formationEnergy: formEnergy },
      fidelity
    );

    return true;
  } catch (quantumErr) {
    if (usedQuantumPipeline) {
      console.log(`[Active Learning] Quantum engine pipeline failed for ${candidate.formula}, falling back to DFT feature resolver: ${quantumErr instanceof Error ? quantumErr.message : String(quantumErr)}`);
    }
  }

  quantumEnginePipelineStats.fallbackRuns++;

  try {
    const dftData = await resolveDFTFeatures(candidate.formula);

    const desc = describeDFTSources(dftData);
    const hasExternalData = dftData.sources.mp || dftData.sources.aflow;
    const sourceType = hasExternalData ? "external+analytical" : "analytical";

    emit("log", {
      phase: "active-learning",
      event: "DFT enrichment (fallback)",
      detail: `${candidate.formula} -- DFT data (${sourceType}, coverage=${dftData.dftCoverage.toFixed(2)}): ${desc}`,
      dataSource: "Active Learning",
    });

    const features = extractFeatures(candidate.formula, undefined, undefined, undefined, dftData);
    const gb = gbPredict(features);
    const nnScore = candidate.neuralNetScore ?? candidate.quantumCoherence ?? 0.3;
    const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);

    const hasExternalDFT = dftData.sources.mp || dftData.sources.aflow;
    let confidence: string;
    if (dftData.dftCoverage > 0.6 && hasExternalDFT) confidence = "high";
    else if (dftData.dftCoverage > 0.2 && hasExternalDFT) confidence = "medium";
    else confidence = "low";

    const updates: any = {
      xgboostScore: gb.score,
      ensembleScore: ensemble,
      dataConfidence: confidence,
    };

    if (dftData.formationEnergy.source !== "analytical") {
      updates.formationEnergy = dftData.formationEnergy.value;
    }
    if (dftData.bandGap.source !== "analytical") {
      updates.bandGap = dftData.bandGap.value;
    }

    await storage.updateSuperconductorCandidate(candidate.id, updates);

    const formEnergy = dftData.formationEnergy?.value ?? null;
    const isStable = formEnergy !== null ? formEnergy < 0.5 : true;
    const dftSource = hasExternalDFT ? "external" as const : "active-learning" as const;
    incorporateDFTResult(
      candidate.formula,
      gb.tcPredicted,
      formEnergy,
      isStable,
      dftSource
    );

    addDFTTrainingResult({
      formula: candidate.formula,
      tc: gb.tcPredicted,
      formationEnergy: formEnergy,
      bandGap: dftData.bandGap?.value ?? null,
      structure: undefined,
      prototype: undefined,
      source: hasExternalDFT ? "external" : "active-learning",
    });

    let gnnTcPredicted = 0;
    let gnnStablePredicted = true;
    let gnnFePredicted = 0;
    try {
      const gnnPred = gnnPredictWithUncertainty(candidate.formula);
      gnnTcPredicted = gnnPred.tc;
      gnnStablePredicted = gnnPred.phononStability;
      gnnFePredicted = gnnPred.formationEnergy;
    } catch {}

    const predictedTc = gnnTcPredicted > 0 ? gnnTcPredicted * 0.6 + gb.tcPredicted * 0.4 : gb.tcPredicted;

    const fallbackModelPred: ModelPrediction = {
      predicted_Tc: predictedTc,
      predicted_stable: gnnStablePredicted,
      predicted_formation_energy: gnnFePredicted || null,
      xgboost_Tc: gb.tcPredicted,
      gnn_Tc: gnnTcPredicted,
      ensemble_score: ensemble,
    };

    addBatchFromEvaluation(
      candidate.formula,
      evalPressure,
      {
        tc: gb.tcPredicted,
        lambda: 0,
        omegaLog: 0,
        dosAtEF: 0,
        formationEnergy: formEnergy,
        bandGap: dftData.bandGap?.value ?? null,
        phononStable: isStable,
        isStrongCoupling: false,
        muStar: 0,
        tier: hasExternalDFT ? "external" : "surrogate",
        confidence: confidence,
        wallTimeMs: 0,
      },
      getCurrentCycleNumber(),
      fallbackModelPred
    );

    recordPredictionVsReality(
      candidate.formula,
      evalPressure,
      {
        Tc: predictedTc,
        stable: gnnStablePredicted,
        formation_energy: gnnFePredicted || null,
        xgboost_Tc: gb.tcPredicted,
        gnn_Tc: gnnTcPredicted,
      },
      {
        Tc: gb.tcPredicted,
        stable: isStable,
        formation_energy: formEnergy,
        lambda: null,
        DOS_EF: null,
      },
      hasExternalDFT ? "external" : "surrogate",
      getCurrentCycleNumber()
    );
    recordEvaluationResult(
      candidate.formula,
      { tc: predictedTc, stable: gnnStablePredicted, formationEnergy: gnnFePredicted },
      { tc: gb.tcPredicted, stable: isStable, formationEnergy: formEnergy },
      hasExternalDFT ? "dft" : "xtb"
    );

    return true;
  } catch (err) {
    console.log(`[Active Learning] DFT enrichment failed for ${candidate.formula}: ${err instanceof Error ? err.message : String(err)}`);

    incorporateDFTResult(
      candidate.formula,
      0,
      null,
      false,
      "active-learning"
    );

    let failGnnTc = 0;
    try {
      const gnnPred = gnnPredictWithUncertainty(candidate.formula);
      failGnnTc = gnnPred.tc;
    } catch {}
    const failPredTc = failGnnTc > 0 ? failGnnTc : (candidate.predictedTc ?? 0);
    recordEvaluationResult(
      candidate.formula,
      { tc: failPredTc, stable: true, formationEnergy: 0 },
      { tc: 0, stable: false, formationEnergy: null },
      "xtb"
    );

    return false;
  }
}

async function retrainGNNWithEnrichedData(
  emit: EventEmitter
): Promise<{ r2Before: number; maeBefore: number; r2After: number; maeAfter: number }> {
  const validationBefore = validateModel();
  const r2Before = validationBefore.r2;
  const maeBefore = Math.sqrt(validationBefore.mse);

  const trainingData = SUPERCON_TRAINING_DATA
    .filter(e => e.isSuperconductor)
    .map(e => ({
      formula: e.formula,
      tc: e.tc,
      formationEnergy: undefined as number | undefined,
      structure: undefined,
      prototype: undefined as string | undefined,
    }));

  const seenFormulas = new Set(trainingData.map(t => t.formula));

  const dftDataset = getDFTTrainingDataset();
  let dftMergeCount = 0;
  for (const dftRecord of dftDataset) {
    if (seenFormulas.has(dftRecord.formula)) continue;
    if (dftRecord.tc <= 0) continue;
    seenFormulas.add(dftRecord.formula);
    trainingData.push({
      formula: dftRecord.formula,
      tc: dftRecord.tc,
      formationEnergy: dftRecord.formationEnergy ?? undefined,
      structure: dftRecord.structure,
      prototype: dftRecord.prototype,
    });
    dftMergeCount++;
  }

  try {
    const enrichedCandidates = await storage.getSuperconductorCandidates(100);
    for (const c of enrichedCandidates) {
      if (c.dataConfidence === "high" || c.dataConfidence === "medium") {
        if (seenFormulas.has(c.formula)) continue;

        const mlf = c.mlFeatures as Record<string, any> | null;
        const hasDFTBandGap = mlf?.bandGap != null && mlf.bandGap >= 0;
        const hasDFTFormationEnergy = c.decompositionEnergy != null;
        const hasDFTValidation = hasDFTBandGap || hasDFTFormationEnergy;
        if (!hasDFTValidation) continue;

        const dftFeatures = extractFeatures(c.formula, undefined, undefined, undefined, undefined);
        const gb = gbPredict(dftFeatures);
        const dftCorrectedTc = gb.tcPredicted;

        if (dftCorrectedTc > 0) {
          seenFormulas.add(c.formula);
          trainingData.push({
            formula: c.formula,
            tc: dftCorrectedTc,
            formationEnergy: c.decompositionEnergy ?? undefined,
            structure: undefined,
            prototype: undefined,
          });
        }
      }
    }
  } catch (e: any) { console.error("[ActiveLearning] enrichment error:", e?.message?.slice(0, 200)); }

  const superconCount = SUPERCON_TRAINING_DATA.filter(e => e.isSuperconductor).length;
  const enrichedCount = trainingData.length - superconCount;
  const dftDatasetForVersion = getDFTTrainingDataset();
  const dftCount = dftDatasetForVersion.length;

  const ensembleModels = await trainEnsembleAsync(trainingData);
  invalidateGNNModel();
  setCachedEnsemble(ensembleModels, trainingData);

  logGNNVersion("active-learning-retrain", trainingData.length, dftCount, enrichedCount);

  await incorporateFailureData();

  const xgbResult = await retrainXGBoostFromEvaluated();

  const validationAfter = validateModel();
  const r2After = validationAfter.r2;
  const maeAfter = Math.sqrt(validationAfter.mse);

  const uncertaintyDrop = convergenceStats.avgUncertaintyBefore - convergenceStats.avgUncertaintyAfter;
  recentUncertaintyDrops.push(uncertaintyDrop);
  if (recentUncertaintyDrops.length > 3) recentUncertaintyDrops.shift();

  const avgRecentDrop = recentUncertaintyDrops.length >= 3
    ? recentUncertaintyDrops.reduce((s, v) => s + v, 0) / recentUncertaintyDrops.length
    : 1.0;
  const converged = avgRecentDrop < 0.1 && recentUncertaintyDrops.length >= 3;

  const evalStats = getEvaluatedDatasetStats();

  emit("log", {
    phase: "active-learning",
    event: "GNN + XGBoost retrained",
    detail: `R² ${r2Before.toFixed(4)} → ${r2After.toFixed(4)}, MAE ${maeBefore.toFixed(2)} → ${maeAfter.toFixed(2)}, GNN samples: ${trainingData.length} (${dftMergeCount} from DFT dataset, total DFT pool: ${dftDataset.length}), XGBoost: ${xgbResult.datasetSize} samples (${xgbResult.newEntries} from eval dataset), evaluated pool: ${evalStats.totalEvaluated}${converged ? ' [CONVERGED]' : ''}`,
    dataSource: "Active Learning",
  });

  return { r2Before, maeBefore, r2After, maeAfter };
}

export async function runActiveLearningCycle(
  emit: EventEmitter,
  memory: { cycleCount: number }
): Promise<ActiveLearningConvergence> {
  emit("log", {
    phase: "active-learning",
    event: "Active learning cycle started",
    detail: `Cycle ${memory.cycleCount}: selecting uncertain candidates for DFT enrichment`,
    dataSource: "Active Learning",
  });

  const allCandidates = await storage.getSuperconductorCandidates(200);

  const eligibleCandidates = allCandidates.filter(c =>
    c.dataConfidence !== "high" &&
    (c.predictedTc ?? 0) > 5
  );

  if (eligibleCandidates.length === 0) {
    emit("log", {
      phase: "active-learning",
      event: "Active learning skipped",
      detail: "No eligible candidates for DFT enrichment",
      dataSource: "Active Learning",
    });
    return convergenceStats;
  }

  const selected = selectForDFT(eligibleCandidates, 20);

  const avgUncertaintyBefore = selected.length > 0
    ? selected.reduce((sum, r) => sum + r.uncertainty, 0) / selected.length
    : 0;

  const tierCounts = {
    bestTc: selected.filter(s => s.selectionTier === "best-tc").length,
    highUncertainty: selected.filter(s => s.selectionTier === "high-uncertainty").length,
    pureCuriosity: selected.filter(s => s.selectionTier === "pure-curiosity").length,
    randomExploration: selected.filter(s => s.selectionTier === "random-exploration").length,
    pressureExploration: selected.filter(s => s.selectionTier === "pressure-exploration").length,
  };
  const avgXgbUnc = selected.length > 0
    ? selected.reduce((s, r) => s + r.xgbUncertainty, 0) / selected.length
    : 0;

  emit("log", {
    phase: "active-learning",
    event: "DFT candidates selected",
    detail: `Selected ${selected.length} candidates [${tierCounts.bestTc} EI-best, ${tierCounts.highUncertainty} UCB-uncertain, ${tierCounts.pureCuriosity} curiosity, ${tierCounts.randomExploration} random, ${tierCounts.pressureExploration} pressure] (avg unc: ${avgUncertaintyBefore.toFixed(3)}, kappa=${computeAdaptiveAlpha().toFixed(2)}, top: ${selected[0]?.candidate.formula ?? 'none'} EI=${selected[0]?.eiScore?.toFixed(3) ?? 0} UCB=${selected[0]?.ucbScore?.toFixed(3) ?? 0})`,
    dataSource: "Active Learning",
  });

  let dftSuccessCount = 0;
  let bestTcThisLoop = 0;
  let pipelineCrashCount = 0;

  for (const ranked of selected) {
    const { candidate } = ranked;
    const candidatePressure = ranked.targetPressureGpa ?? (candidate as any).pressureGpa ?? estimateFamilyPressure(candidate.formula);
    const enriched = await runDFTEnrichmentForCandidate(emit, candidate, candidatePressure);
    if (enriched) {
      dftSuccessCount++;
      convergenceStats.totalDFTRuns++;
      recordPressureCoverage(candidate.formula, candidatePressure);
    } else {
      pipelineCrashCount++;
    }
    if ((candidate.predictedTc ?? 0) > bestTcThisLoop) {
      bestTcThisLoop = candidate.predictedTc ?? 0;
    }
  }

  const cycleDatapoints = getDatapointsByCycle(getCurrentCycleNumber());
  const USEFUL_TC_THRESHOLD = 20;
  let usefulDiscoveries = 0;
  let stableCount = 0;
  let unstablePhonons = 0;
  let highFormationEnergy = 0;
  let nonMetallic = 0;
  let lowTcCount = 0;

  for (const dp of cycleDatapoints) {
    const isUseful = dp.Tc >= USEFUL_TC_THRESHOLD && dp.phonon_stable;
    if (isUseful) usefulDiscoveries++;
    if (dp.phonon_stable) stableCount++;
    if (!dp.phonon_stable) unstablePhonons++;
    if (dp.formation_energy !== null && dp.formation_energy > 0.5) highFormationEnergy++;
    if (dp.band_gap !== null && dp.band_gap > 0.5) nonMetallic++;
    if (dp.Tc < USEFUL_TC_THRESHOLD && dp.phonon_stable) lowTcCount++;
  }

  const discoveryEff: DiscoveryEfficiency = {
    usefulDiscoveries,
    totalEvaluations: selected.length,
    efficiencyRatio: selected.length > 0 ? usefulDiscoveries / selected.length : 0,
    stableCount,
    unstableCount: unstablePhonons,
    highTcCount: cycleDatapoints.filter(dp => dp.Tc >= 100).length,
    failureBreakdown: {
      unstablePhonons,
      highFormationEnergy,
      nonMetallic,
      lowTc: lowTcCount,
      pipelineCrash: pipelineCrashCount,
    },
  };

  emit("log", {
    phase: "active-learning",
    event: "Discovery efficiency",
    detail: `${usefulDiscoveries} useful materials / ${selected.length} evaluations (${(discoveryEff.efficiencyRatio * 100).toFixed(1)}%). Failures: ${unstablePhonons} unstable phonons, ${highFormationEnergy} high formation energy, ${nonMetallic} non-metallic, ${lowTcCount} low Tc, ${pipelineCrashCount} pipeline crashes`,
    dataSource: "Active Learning",
  });

  let pressureTransitionsThisCycle = 0;
  const pressureCandidates = selected.filter(s => s.selectionTier === "pressure-exploration" || (s.candidate.predictedTc ?? 0) > 50);
  for (const ranked of pressureCandidates.slice(0, 5)) {
    try {
      const transitions = detectPhaseTransitions(ranked.candidate.formula);
      pressureTransitionsThisCycle += transitions.length;
      if (transitions.length > 0) {
        emit("log", {
          phase: "active-learning",
          event: "Pressure phase transitions detected",
          detail: `${ranked.candidate.formula}: ${transitions.length} transition(s) — ${transitions.map(t => `${t.type} at ${t.pressureStart}-${t.pressureEnd} GPa (conf=${t.confidence.toFixed(2)})`).join(", ")}`,
          dataSource: "Active Learning",
        });
      }
    } catch { /* skip */ }
  }
  pressureCoverageStats.pressureTransitionsFound += pressureTransitionsThisCycle;

  for (const ranked of selected.slice(0, 3)) {
    try {
      predictPressureCurve(ranked.candidate.formula);
      const optimal = findOptimalPressure(ranked.candidate.formula);
      if (optimal.optimalPressureGpa > 0 && optimal.maxTc > 50) {
        emit("log", {
          phase: "active-learning",
          event: "Pressure curve analyzed",
          detail: `${ranked.candidate.formula}: optimal P=${optimal.optimalPressureGpa} GPa, max Tc=${optimal.maxTc.toFixed(1)}K`,
          dataSource: "Active Learning",
        });
      }
    } catch { /* skip */ }
  }

  totalEnrichedSinceLastRetrain += dftSuccessCount;

  const batchCycleNum = startNewBatchCycle();
  const validationPre = validateModel();
  const preR2 = validationPre.r2;
  const preMAE = Math.sqrt(validationPre.mse);
  const preDatasetSize = getEvaluatedDatasetStats().totalEvaluated;

  let retrainResult = { r2Before: 0, maeBefore: 0, r2After: 0, maeAfter: 0 };
  const sampleTrigger = checkRetrainTrigger();
  const shouldRetrain = dftSuccessCount > 0 || sampleTrigger.shouldRetrain;

  if (shouldRetrain) {
    const reasons: string[] = [];
    if (dftSuccessCount > 0) reasons.push(`batch-mandatory (${dftSuccessCount} new datapoints)`);
    if (sampleTrigger.shouldRetrain) reasons.push(`sample-count threshold (${sampleTrigger.reason})`);
    const retrainReason = reasons.join(" + ");
    emit("log", {
      phase: "active-learning",
      event: "Batch retrain triggered",
      detail: `Trigger: ${retrainReason}, total enriched=${totalEnrichedSinceLastRetrain}, ground-truth dataset=${getGroundTruthSummary().totalDatapoints}, ledger=${computeMetrics().count} entries`,
      dataSource: "Active Learning",
    });
    retrainResult = await retrainGNNWithEnrichedData(emit);
    convergenceStats.modelRetrains++;
    totalEnrichedSinceLastRetrain = 0;
    lastRetrainCycle = memory.cycleCount;
    if (sampleTrigger.shouldRetrain) acknowledgeRetrain();
  } else {
    emit("log", {
      phase: "active-learning",
      event: "Batch evaluation warning",
      detail: `All ${selected.length} candidates failed DFT/TB evaluation in batch cycle ${batchCycleNum}. No new data to incorporate — skipping retrain.`,
      dataSource: "Active Learning",
    });
  }

  const validationPost = validateModel();
  const postR2 = validationPost.r2;
  const postMAE = Math.sqrt(validationPost.mse);
  const postDatasetSize = getEvaluatedDatasetStats().totalEvaluated;

  if (shouldRetrain) {
    recordCycleImprovement(
      batchCycleNum,
      { r2: postR2, rmse: postMAE },
      { r2: retrainResult.r2After, rmse: retrainResult.maeAfter }
    );
  }

  let avgUncertaintyAfter = avgUncertaintyBefore;
  if (selected.length > 0) {
    let totalUncertaintyAfter = 0;
    for (const { candidate } of selected) {
      try {
        const gnnResult = gnnPredictWithUncertainty(candidate.formula);
        totalUncertaintyAfter += gnnResult.uncertainty;
      } catch {
        totalUncertaintyAfter += 0.5;
      }
    }
    avgUncertaintyAfter = totalUncertaintyAfter / selected.length;
  }

  convergenceStats.avgUncertaintyBefore = avgUncertaintyBefore;
  convergenceStats.avgUncertaintyAfter = avgUncertaintyAfter;
  if (bestTcThisLoop > convergenceStats.bestTcFromLoop) {
    convergenceStats.bestTcFromLoop = bestTcThisLoop;
  }

  const avgGnnUnc = selected.length > 0
    ? selected.reduce((s, r) => s + (r.uncertainty - r.xgbUncertainty * 0.5) * 2, 0) / selected.length
    : 0;
  const uncReductionPct = avgUncertaintyBefore > 0
    ? (avgUncertaintyBefore - avgUncertaintyAfter) / avgUncertaintyBefore * 100
    : 0;

  const cycleRecord: ActiveLearningCycleRecord = {
    cycle: memory.cycleCount,
    timestamp: Date.now(),
    candidatesSelected: selected.length,
    dftSuccesses: dftSuccessCount,
    dftFailures: selected.length - dftSuccessCount,
    avgGnnUncertainty: Math.round(avgGnnUnc * 1000) / 1000,
    avgXgbUncertainty: Math.round(avgXgbUnc * 1000) / 1000,
    avgCombinedUncertainty: Math.round(avgUncertaintyBefore * 1000) / 1000,
    uncertaintyAfter: Math.round(avgUncertaintyAfter * 1000) / 1000,
    uncertaintyReductionPct: Math.round(uncReductionPct * 10) / 10,
    gnnRetrained: shouldRetrain,
    gnnVersion: shouldRetrain ? getGNNModelVersion() : null,
    tierBreakdown: tierCounts,
    topFormula: selected[0]?.candidate.formula ?? "",
    topAcquisitionScore: Math.round((selected[0]?.acquisitionScore ?? 0) * 1000) / 1000,
    bestTcThisCycle: Math.round(bestTcThisLoop * 10) / 10,
    discoveryEfficiency: discoveryEff,
  };
  cycleHistory.push(cycleRecord);
  if (cycleHistory.length > MAX_CYCLE_HISTORY) cycleHistory.shift();

  const batchCycleRecord: BatchCycle = {
    cycleNumber: batchCycleNum,
    startedAt: cycleRecord.timestamp - 1000,
    completedAt: Date.now(),
    candidatesSubmitted: selected.length,
    evaluationSuccesses: dftSuccessCount,
    evaluationFailures: selected.length - dftSuccessCount,
    newDatapoints: dftSuccessCount,
    retrainTriggered: shouldRetrain,
    preRetrainMetrics: { r2: preR2, mae: preMAE, datasetSize: preDatasetSize },
    postRetrainMetrics: shouldRetrain ? { r2: postR2, mae: postMAE, datasetSize: postDatasetSize } : null,
    r2Improvement: shouldRetrain ? postR2 - preR2 : 0,
    maeImprovement: shouldRetrain ? preMAE - postMAE : 0,
    bestTcThisCycle: bestTcThisLoop,
    avgUncertaintyBefore,
    avgUncertaintyAfter,
  };
  recordBatchCycle(batchCycleRecord);

  const gtSummary = getGroundTruthSummary();
  emit("log", {
    phase: "active-learning",
    event: `Batch cycle ${batchCycleNum} complete`,
    detail: `${dftSuccessCount} structures evaluated, ${gtSummary.totalDatapoints} total ground-truth datapoints, R² ${preR2.toFixed(4)} -> ${postR2.toFixed(4)}, MAE ${preMAE.toFixed(2)} -> ${postMAE.toFixed(2)}, dataset size ${preDatasetSize} -> ${postDatasetSize}`,
    dataSource: "Active Learning",
  });

  for (const { candidate } of selected) {
    try {
      const existingCandidate = await storage.getSuperconductorByFormula(candidate.formula);
      if (existingCandidate) {
        const hullDist = (existingCandidate.mlFeatures as any)?.stabilityGate?.hullDistance ?? 0.05;
        const discoveryResult = computeDiscoveryScore({
          predictedTc: existingCandidate.predictedTc ?? 0,
          formula: existingCandidate.formula,
          hullDistance: hullDist,
          synthesisScore: existingCandidate.stabilityScore ?? 0.5,
          uncertaintyEstimate: (existingCandidate.mlFeatures as any)?.uncertaintyEstimate ?? 0.5,
        });
        await storage.updateSuperconductorCandidate(existingCandidate.id, {
          discoveryScore: discoveryResult.discoveryScore,
        });
      }
    } catch (e: any) { console.error("[ActiveLearning] discovery score error:", e?.message?.slice(0, 200)); }
  }

  const uncertaintyReduction = avgUncertaintyBefore > 0
    ? ((avgUncertaintyBefore - avgUncertaintyAfter) / avgUncertaintyBefore * 100).toFixed(1)
    : "0";

  emit("log", {
    phase: "active-learning",
    event: "Active learning cycle complete",
    detail: `DFT enriched: ${dftSuccessCount}/${selected.length}, uncertainty reduction: ${uncertaintyReduction}%, model retrains: ${convergenceStats.modelRetrains}, best Tc: ${convergenceStats.bestTcFromLoop.toFixed(1)}K`,
    dataSource: "Active Learning",
  });

  return convergenceStats;
}
