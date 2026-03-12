import { TargetProperties, InverseCandidate, computeTargetDistance, computeReward, CompositionBias } from "./target-schema";
import { generateInverseCandidates, refineCandidate, createInitialBias } from "./inverse-generator";
import { solveConstraints, type ConstraintSolution } from "./constraint-solver";
import { solveConstraintGraph, getConstraintGraphGuidance, type GraphSolution } from "./constraint-graph-solver";
import { checkPhysicsConstraints, createConstraintRegistry, ConstraintRegistry, type ConstraintResult } from "./physics-constraint-engine";
import { evaluatePillars, type PillarEvaluation, type SCPillarTargets } from "./sc-pillars-optimizer";
import { runDifferentiableOptimization, type DifferentiableResult } from "./differentiable-optimizer";
import { updateLearningState, deriveCompositionBias, createInitialLearningState, type InverseLearningState } from "./inverse-learning";
import { gbPredict } from "../learning/gradient-boost";
import { extractFeatures } from "../learning/ml-predictor";
import { gnnPredictWithUncertainty } from "../learning/graph-neural-net";

export interface PipelineGoal {
  targetTc: number;
  maxPressure: number;
  minLambda: number;
  maxHullDistance: number;
  metallicRequired: boolean;
  phononStable: boolean;
  preferredPrototypes?: string[];
  preferredElements?: string[];
  excludeElements?: string[];
  maxIterations: number;
  convergenceThreshold: number;
  surrogateWeight: number;
  constraintStrictness: "relaxed" | "standard" | "strict";
}

export interface PipelineCandidate {
  formula: string;
  source: "generator" | "refinement" | "gradient" | "constraint-guided";
  iteration: number;
  constraintResult: ConstraintResult | null;
  pillarEvaluation: PillarEvaluation | null;
  surrogateScores: {
    gbTc: number;
    gbScore: number;
    gnnTc: number;
    gnnConfidence: number;
    gnnLambda: number;
    ensembleTc: number;
  } | null;
  physicsValidated: boolean;
  targetDistance: number;
  reward: number;
  rank: number;
}

export interface PipelineIterationResult {
  iteration: number;
  candidatesGenerated: number;
  constraintsPassed: number;
  surrogateEvaluated: number;
  physicsValidated: number;
  bestTcThisIteration: number;
  bestFormulaThisIteration: string;
  averageTargetDistance: number;
  convergenceDelta: number;
  topCandidates: PipelineCandidate[];
  wallTimeMs: number;
}

export interface PipelineState {
  goal: PipelineGoal;
  iteration: number;
  status: "idle" | "running" | "converged" | "completed" | "paused";
  learningState: InverseLearningState;
  bias: CompositionBias;
  constraintSolution: ConstraintSolution | null;
  graphSolution: GraphSolution | null;
  bestTcOverall: number;
  bestFormulaOverall: string;
  bestDistance: number;
  convergenceHistory: number[];
  iterationHistory: PipelineIterationResult[];
  seenFormulas: Set<string>;
  totalCandidatesGenerated: number;
  totalConstraintsPassed: number;
  totalSurrogateEvaluated: number;
  totalPhysicsValidated: number;
  startedAt: number;
  lastIterationAt: number;
}

const pipelineRegistries = new Map<PipelineState, ConstraintRegistry>();

function getPipelineRegistry(state: PipelineState): ConstraintRegistry {
  let reg = pipelineRegistries.get(state);
  if (!reg) {
    reg = createConstraintRegistry();
    pipelineRegistries.set(state, reg);
  }
  return reg;
}

export interface PipelineStats {
  status: string;
  iteration: number;
  bestTc: number;
  bestFormula: string;
  bestDistance: number;
  totalGenerated: number;
  totalPassed: number;
  convergenceHistory: number[];
  surrogateAccuracy: number;
  constraintPassRate: number;
  topCandidates: { formula: string; tc: number; distance: number; source: string }[];
  iterationsPerMinute: number;
  estimatedIterationsToConverge: number | null;
}

const pipelines = new Map<string, PipelineState>();
const pipelineCompletedAt = new Map<string, number>();
const PIPELINE_TTL_MS = 30 * 60 * 1000;
let totalPipelineRuns = 0;
let totalCandidatesProcessed = 0;

function pruneCompletedPipelines(): void {
  const now = Date.now();
  for (const [id, completedTime] of pipelineCompletedAt) {
    if (now - completedTime > PIPELINE_TTL_MS) {
      pipelines.delete(id);
      pipelineCompletedAt.delete(id);
    }
  }
}

function goalToTargetProperties(goal: PipelineGoal): TargetProperties {
  return {
    targetTc: goal.targetTc,
    maxPressure: goal.maxPressure,
    minLambda: goal.minLambda,
    maxHullDistance: goal.maxHullDistance,
    metallicRequired: goal.metallicRequired,
    phononStable: goal.phononStable,
    preferredPrototypes: goal.preferredPrototypes,
    preferredElements: goal.preferredElements,
    excludeElements: goal.excludeElements,
  };
}

function createDefaultGoal(): PipelineGoal {
  return {
    targetTc: 293,
    maxPressure: 50,
    minLambda: 1.5,
    maxHullDistance: 0.05,
    metallicRequired: true,
    phononStable: true,
    maxIterations: 200,
    convergenceThreshold: 0.02,
    surrogateWeight: 0.7,
    constraintStrictness: "standard",
  };
}

export function createPipeline(id: string, goal?: Partial<PipelineGoal>): PipelineState {
  const fullGoal: PipelineGoal = { ...createDefaultGoal(), ...goal };
  const target = goalToTargetProperties(fullGoal);
  const bias = createInitialBias(target);

  let constraintSolution: ConstraintSolution | null = null;
  let graphSolution: GraphSolution | null = null;

  try {
    constraintSolution = solveConstraints(fullGoal.targetTc, 0.10);
  } catch (e: any) { console.error("[NextGenPipeline] solveConstraints error:", e?.message?.slice(0, 200)); }

  try {
    graphSolution = solveConstraintGraph(fullGoal.targetTc, 0.10);
  } catch (e: any) { console.error("[NextGenPipeline] solveConstraintGraph error:", e?.message?.slice(0, 200)); }

  const state: PipelineState = {
    goal: fullGoal,
    iteration: 0,
    status: "idle",
    learningState: createInitialLearningState(),
    bias,
    constraintSolution,
    graphSolution,
    bestTcOverall: 0,
    bestFormulaOverall: "",
    bestDistance: 1.0,
    convergenceHistory: [],
    iterationHistory: [],
    seenFormulas: new Set<string>(),
    totalCandidatesGenerated: 0,
    totalConstraintsPassed: 0,
    totalSurrogateEvaluated: 0,
    totalPhysicsValidated: 0,
    startedAt: Date.now(),
    lastIterationAt: 0,
  };

  pipelines.set(id, state);
  return state;
}

export async function runPipelineIteration(id: string): Promise<PipelineIterationResult | null> {
  const state = pipelines.get(id);
  if (!state || state.status === "converged" || state.status === "completed" || state.status === "paused") return null;

  const startTime = Date.now();
  state.status = "running";
  state.iteration++;
  state.lastIterationAt = Date.now();

  try {
  return await _runPipelineIterationInner(state, id, startTime);
  } catch (e: any) {
    console.error("[NextGenPipeline] iteration error:", e?.message?.slice(0, 300));
    state.status = "idle";
    return null;
  }
}

async function _runPipelineIterationInner(state: PipelineState, id: string, startTime: number): Promise<PipelineIterationResult> {
  const target = goalToTargetProperties(state.goal);
  const allCandidates: PipelineCandidate[] = [];

  const freshCandidates = generateInverseCandidates(
    target,
    id,
    state.iteration,
    state.bias,
    40,
  );
  for (const c of freshCandidates) {
    allCandidates.push({
      formula: c.formula,
      source: "generator",
      iteration: state.iteration,
      constraintResult: null,
      pillarEvaluation: null,
      surrogateScores: null,
      physicsValidated: false,
      targetDistance: 1.0,
      reward: 0,
      rank: 0,
    });
  }

  if (state.learningState.bestCandidates.length > 0) {
    const isEarlyStage = state.bestDistance >= 0.3;
    const topNCount = isEarlyStage ? 3 : 5;
    const perCandidateRefined = isEarlyStage ? 2 : 4;
    const topN = state.learningState.bestCandidates.slice(0, topNCount);
    for (const best of topN) {
      try {
        const refined = refineCandidate(best, target, state.bias);
        for (const r of refined.slice(0, perCandidateRefined)) {
          allCandidates.push({
            formula: r.formula,
            source: "refinement",
            iteration: state.iteration,
            constraintResult: null,
            pillarEvaluation: null,
            surrogateScores: null,
            physicsValidated: false,
            targetDistance: 1.0,
            reward: 0,
            rank: 0,
          });
        }
      } catch (e: any) { console.error("[NextGenPipeline] RL generate error:", e?.message?.slice(0, 200)); }
    }
  }

  const gradientWarmup = Math.max(8, Math.floor(state.goal.maxIterations * 0.15));
  if (state.iteration > gradientWarmup && state.bestDistance < 0.5 && state.learningState.bestCandidates.length > 0) {
    const topForGradient = state.learningState.bestCandidates
      .filter(c => c.targetDistance < 0.6)
      .slice(0, 3);
    for (const best of topForGradient) {
      try {
        const diffResult = await runDifferentiableOptimization(best.formula, target);
        if (diffResult && diffResult.optimizedFormula && diffResult.optimizedFormula !== best.formula) {
          allCandidates.push({
            formula: diffResult.optimizedFormula,
            source: "gradient",
            iteration: state.iteration,
            constraintResult: null,
            pillarEvaluation: null,
            surrogateScores: null,
            physicsValidated: false,
            targetDistance: 1.0,
            reward: 0,
            rank: 0,
          });
        }
      } catch (e: any) { console.error("[NextGenPipeline] diff optimization error:", e?.message?.slice(0, 200)); }
    }
  }

  const deduplicated = allCandidates.filter(c => {
    if (state.seenFormulas.has(c.formula)) return false;
    state.seenFormulas.add(c.formula);
    return true;
  });
  if (state.seenFormulas.size > 5000) {
    const entries = Array.from(state.seenFormulas);
    state.seenFormulas = new Set(entries.slice(-2500));
  }

  state.totalCandidatesGenerated += deduplicated.length;

  const baseThreshold = state.goal.constraintStrictness === "strict" ? 0.3
    : state.goal.constraintStrictness === "relaxed" ? 0.8 : 0.5;
  const convergenceProgress = Math.min(1, state.iteration / Math.max(1, state.goal.maxIterations));
  const constraintThreshold = baseThreshold + (1.0 - baseThreshold) * Math.pow(1 - convergenceProgress, 2);

  const pipelineRegistry = getPipelineRegistry(state);
  const constraintPassed: PipelineCandidate[] = [];
  for (const candidate of deduplicated) {
    try {
      const result = checkPhysicsConstraints(candidate.formula, { maxPressureGPa: state.goal.maxPressure, registry: pipelineRegistry });
      candidate.constraintResult = result;
      if (result.isValid || result.totalPenalty < constraintThreshold) {
        constraintPassed.push(candidate);
      } else {
        const acceptProb = Math.exp(-result.totalPenalty / Math.max(0.1, constraintThreshold));
        if (Math.random() < acceptProb * 0.3) {
          constraintPassed.push(candidate);
        }
      }
    } catch (e: any) { console.error("[NextGenPipeline] constraint check error:", e?.message?.slice(0, 200)); }
  }

  state.totalConstraintsPassed += constraintPassed.length;

  const pillarTargets: SCPillarTargets = {
    minLambda: state.constraintSolution?.requiredLambda?.optimal ?? state.goal.minLambda,
    minOmegaLogK: state.constraintSolution?.requiredOmegaLog?.optimal ?? 500,
    minDOS: state.constraintSolution?.requiredDOS?.optimalDOS ?? 3.0,
    minNesting: 0.5,
    minFlatBand: 0.6,
    minPairingGlue: 0.5,
    minInstability: 0.3,
    minHydrogenCage: 0.4,
    preferredMotifs: ["cage", "layered", "kagome"],
  };

  for (const candidate of constraintPassed) {
    try {
      candidate.pillarEvaluation = await evaluatePillars(candidate.formula, pillarTargets, { maxPressureGPa: state.goal.maxPressure });
    } catch (e: any) { console.error("[NextGenPipeline] pillar eval error:", e?.message?.slice(0, 200)); }
  }

  for (const candidate of constraintPassed) {
    try {
      const features = await extractFeatures(candidate.formula);
      const gb = await gbPredict(features);
      const gnn = gnnPredictWithUncertainty(candidate.formula);

      const ensembleTc = gb.tcPredicted * 0.35 + (gnn.tc ?? 0) * 0.65;

      candidate.surrogateScores = {
        gbTc: gb.tcPredicted,
        gbScore: gb.score,
        gnnTc: gnn.tc ?? 0,
        gnnConfidence: gnn.confidence ?? 0,
        gnnLambda: gnn.lambda ?? 0,
        ensembleTc,
      };

      const hullEstimate = Math.max(0, gnn.formationEnergy ?? 0.02);
      const pressureEstimate = candidate.constraintResult
        ? (candidate.constraintResult.totalPenalty > 0.5 ? state.goal.maxPressure * 0.8 : 0)
        : 0;
      const predicted = {
        tc: ensembleTc,
        lambda: gnn.lambda ?? gb.score * 2,
        hull: hullEstimate,
        pressure: pressureEstimate,
      };
      candidate.targetDistance = computeTargetDistance(target, predicted);
      candidate.reward = computeReward(candidate.targetDistance);
      candidate.physicsValidated = true;
    } catch (e: any) { console.error("[NextGenPipeline] surrogate eval error:", e?.message?.slice(0, 200)); }
  }

  state.totalSurrogateEvaluated += constraintPassed.filter(c => c.surrogateScores !== null).length;
  state.totalPhysicsValidated += constraintPassed.filter(c => c.physicsValidated).length;

  const scored = constraintPassed
    .filter(c => c.surrogateScores !== null)
    .sort((a, b) => {
      const aTc = a.surrogateScores!.ensembleTc;
      const bTc = b.surrogateScores!.ensembleTc;
      const pillarA = a.pillarEvaluation?.overallFitness ?? 0;
      const pillarB = b.pillarEvaluation?.overallFitness ?? 0;
      const normTcA = aTc / Math.max(1, state.goal.targetTc);
      const normTcB = bTc / Math.max(1, state.goal.targetTc);
      const compositeA = normTcA * state.goal.surrogateWeight + pillarA * (1 - state.goal.surrogateWeight);
      const compositeB = normTcB * state.goal.surrogateWeight + pillarB * (1 - state.goal.surrogateWeight);
      return compositeB - compositeA;
    });

  scored.forEach((c, i) => { c.rank = i + 1; });

  const bestTcThisIteration = scored.length > 0
    ? Math.max(...scored.map(c => c.surrogateScores!.ensembleTc))
    : 0;

  const bestFormulaThisIteration = scored.length > 0
    ? scored[0].formula
    : "";

  if (bestTcThisIteration > state.bestTcOverall) {
    state.bestTcOverall = bestTcThisIteration;
    state.bestFormulaOverall = bestFormulaThisIteration;
  }

  const avgDistance = scored.length > 0
    ? scored.reduce((s, c) => s + c.targetDistance, 0) / scored.length
    : 1.0;

  const bestDistThisIter = scored.length > 0
    ? Math.min(...scored.map(c => c.targetDistance))
    : 1.0;

  if (bestDistThisIter < state.bestDistance) {
    state.bestDistance = bestDistThisIter;
  }

  const prevBestDist = state.convergenceHistory.length > 0
    ? state.convergenceHistory[state.convergenceHistory.length - 1]
    : 1.0;
  const convergenceDelta = prevBestDist - state.bestDistance;
  state.convergenceHistory.push(state.bestDistance);
  if (state.convergenceHistory.length > 200) {
    state.convergenceHistory = state.convergenceHistory.slice(-100);
  }

  const results = scored.map(c => ({
    formula: c.formula,
    tc: c.surrogateScores!.ensembleTc,
    lambda: c.surrogateScores!.gnnLambda,
    hull: c.targetDistance < 1.0 ? Math.max(0, c.targetDistance * 0.1) : 0.02,
    pressure: c.constraintResult
      ? (c.constraintResult.totalPenalty > 0.5 ? state.goal.maxPressure * 0.8 : 0)
      : 0,
    passedPipeline: c.physicsValidated && (c.constraintResult?.isValid ?? true),
  }));

  const inverseCandidates: InverseCandidate[] = scored.map(c => ({
    formula: c.formula,
    source: "inverse" as const,
    campaignId: id,
    targetDistance: c.targetDistance,
    iteration: state.iteration,
    predictedTc: c.surrogateScores!.ensembleTc,
    predictedLambda: c.surrogateScores!.gnnLambda,
    predictedHull: 0.02,
    predictedPressure: 0,
  }));

  try {
    state.learningState = updateLearningState(
      state.learningState,
      inverseCandidates,
      results,
      target,
    );
  } catch (e: any) {
    console.error("[NextGenPipeline] learning state update error:", e?.message?.slice(0, 300));
  }

  const baseBias = createInitialBias(target);
  state.bias = deriveCompositionBias(state.learningState, baseBias);

  if (state.convergenceHistory.length >= 20) {
    const last20 = state.convergenceHistory.slice(-20);
    const stagnationRange = Math.max(...last20) - Math.min(...last20);
    if (stagnationRange < 0.005 && state.bestDistance > 0.15) {
      for (const [el, w] of state.bias.elementWeights) {
        state.bias.elementWeights.set(el, w * (0.6 + Math.random() * 0.8));
      }
      if (state.seenFormulas.size > 100) {
        const entries = Array.from(state.seenFormulas);
        state.seenFormulas = new Set(entries.slice(-50));
      }
    }
  }

  const minConvergeIters = Math.max(20, Math.floor(state.goal.maxIterations * 0.2));
  if (state.convergenceHistory.length >= 15 && state.iteration >= minConvergeIters) {
    const last15 = state.convergenceHistory.slice(-15);
    const range = Math.max(...last15) - Math.min(...last15);
    if (range < state.goal.convergenceThreshold && state.bestDistance < 0.15) {
      state.status = "converged";
      pipelineCompletedAt.set(id, Date.now());
    }
  }

  if (state.iteration >= state.goal.maxIterations) {
    state.status = "completed";
    pipelineCompletedAt.set(id, Date.now());
  }

  if (state.status !== "converged" && state.status !== "completed") {
    state.status = "running";
  }

  const iterResult: PipelineIterationResult = {
    iteration: state.iteration,
    candidatesGenerated: deduplicated.length,
    constraintsPassed: constraintPassed.length,
    surrogateEvaluated: constraintPassed.filter(c => c.surrogateScores !== null).length,
    physicsValidated: constraintPassed.filter(c => c.physicsValidated).length,
    bestTcThisIteration,
    bestFormulaThisIteration,
    averageTargetDistance: avgDistance,
    convergenceDelta,
    topCandidates: scored.slice(0, 10).map(c => ({
      ...c,
      constraintResult: null,
      pillarEvaluation: null,
    })),
    wallTimeMs: Date.now() - startTime,
  };

  state.iterationHistory.push(iterResult);
  if (state.iterationHistory.length > 50) {
    state.iterationHistory = state.iterationHistory.slice(-50);
  }

  totalPipelineRuns++;
  totalCandidatesProcessed += deduplicated.length;

  return iterResult;
}

export function getPipelineState(id: string): PipelineState | undefined {
  pruneCompletedPipelines();
  return pipelines.get(id);
}

export function getPipelineStats(id: string): PipelineStats | null {
  const state = pipelines.get(id);
  if (!state) return null;

  const elapsedMs = Date.now() - state.startedAt;
  const iterPerMin = elapsedMs > 0 ? (state.iteration / (elapsedMs / 60000)) : 0;

  let estimatedItersToConverge: number | null = null;
  if (state.convergenceHistory.length >= 5) {
    const recent = state.convergenceHistory.slice(-5);
    let weightedImprovement = 0;
    let totalWeight = 0;
    for (let i = 0; i < recent.length - 1; i++) {
      const weight = i + 1;
      weightedImprovement += (recent[i] - recent[i + 1]) * weight;
      totalWeight += weight;
    }
    const avgImprovement = totalWeight > 0 ? weightedImprovement / totalWeight : 0;
    if (avgImprovement > 0.001) {
      estimatedItersToConverge = Math.ceil(state.bestDistance / avgImprovement);
    }
  }

  const constraintPassRate = state.totalCandidatesGenerated > 0
    ? state.totalConstraintsPassed / state.totalCandidatesGenerated
    : 0;

  const surrogateAccuracy = state.totalSurrogateEvaluated > 0
    ? state.totalPhysicsValidated / state.totalSurrogateEvaluated
    : 0;

  return {
    status: state.status,
    iteration: state.iteration,
    bestTc: state.bestTcOverall,
    bestFormula: state.bestFormulaOverall,
    bestDistance: state.bestDistance,
    totalGenerated: state.totalCandidatesGenerated,
    totalPassed: state.totalPhysicsValidated,
    convergenceHistory: state.convergenceHistory.slice(-100),
    surrogateAccuracy,
    constraintPassRate,
    topCandidates: state.learningState.bestCandidates.slice(0, 10).map(c => ({
      formula: c.formula,
      tc: c.predictedTc ?? 0,
      distance: c.targetDistance,
      source: c.source || "inverse",
    })),
    iterationsPerMinute: Math.round(iterPerMin * 10) / 10,
    estimatedIterationsToConverge: estimatedItersToConverge,
  };
}

export function pausePipeline(id: string): boolean {
  const state = pipelines.get(id);
  if (!state) return false;
  state.status = "paused";
  return true;
}

export function resumePipeline(id: string): boolean {
  const state = pipelines.get(id);
  if (!state || state.status !== "paused") return false;
  state.status = "running";
  return true;
}

export function deletePipeline(id: string): boolean {
  return pipelines.delete(id);
}

export function getAllPipelines(): PipelineState[] {
  return Array.from(pipelines.values());
}

export function getNextGenPipelineStats(): {
  activePipelines: number;
  totalRuns: number;
  totalCandidatesProcessed: number;
  pipelines: { id: string; status: string; iteration: number; bestTc: number; bestDistance: number; bestFormula: string }[];
} {
  const pipelineSummaries = Array.from(pipelines.entries()).map(([id, s]) => ({
    id,
    status: s.status,
    iteration: s.iteration,
    bestTc: s.bestTcOverall,
    bestDistance: s.bestDistance,
    bestFormula: s.bestFormulaOverall,
  }));

  return {
    activePipelines: pipelineSummaries.filter(p => p.status === "running").length,
    totalRuns: totalPipelineRuns,
    totalCandidatesProcessed,
    pipelines: pipelineSummaries,
  };
}
