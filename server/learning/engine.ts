import { WebSocketServer, WebSocket } from "ws";
import type { Server } from "http";
import { storage } from "../storage";
import { fetchOQMDMaterials, fetchElementFocusedMaterials, fetchKnownMaterials, getNextOQMDOffset } from "./data-fetcher";
import { analyzeBondingPatterns, analyzePropertyPredictionPatterns } from "./nlp-engine";
import { generateNovelFormulas, setBoundaryHuntingMode, setInverseDesignMode, setChemicalSpaceExpansionMode, getGenerationModes } from "./formula-generator";
import { runSuperconductorResearch, generateInverseDesignCandidates, getInverseDesignCount } from "./superconductor-research";
import { getAllActiveCampaigns, runInverseCycle, processInverseResults, getSerializableCampaignState, getInverseDesignStats as getInverseOptimizerStats, loadCampaign, restoreCampaignsFromDB } from "../inverse/inverse-optimizer";
import { runGradientDescentCycle, getDifferentiableOptimizerStats } from "../inverse/differentiable-optimizer";
import { runStructureDiffusionCycle, getStructureDiffusionStats } from "../ai/structure-diffusion";
import { constraintGuidedGenerate, checkPhysicsConstraints, updateConstraintWeightsFromReward, getConstraintEngineStats } from "../inverse/physics-constraint-engine";
import { runPillarCycle, evaluatePillars, updatePillarWeightsFromReward, getPillarOptimizerStats } from "../inverse/sc-pillars-optimizer";
import type { InverseCandidate } from "../inverse/target-schema";
import { discoverSynthesisProcesses, discoverChemicalReactions, getNextReactionTopic } from "./synthesis-tracker";
import { runFullPhysicsAnalysis, applyAmbientTcCap, setConstraintMode, getConstraintMode, parseFormulaElements, computeElectronicStructure } from "./physics-engine";
import { runPressureAnalysis } from "./pressure-engine";
import { runStructurePredictionBatch, runGenerativeStructureDiscovery, getStructuralVariantCount, runNovelPrototypeGeneration, getNovelPrototypeCount, runEvolutionaryStructureSearch, setMutationIntensity } from "./structure-predictor";
import { runMultiFidelityPipeline } from "./multi-fidelity-pipeline";
import { evaluateInsightNovelty, requiresQuantitativeContent } from "./insight-detector";
import { analyzeAndEvolveStrategy, captureConvergenceSnapshot, trackDuplicatesSkipped } from "./strategy-analyzer";
import { checkMilestones } from "./milestone-tracker";
import { extractFeatures, physicsPredictor } from "./ml-predictor";
import type { PhysicsPrediction } from "./ml-predictor";
import { gbPredict, incorporateFailureData, getFailureExampleCount } from "./gradient-boost";
import { normalizeFormula, classifyFamily, sanitizeForbiddenWords, isValidFormula } from "./utils";
import { runMassiveGeneration, type MassiveGenerationStats } from "./candidate-generator";
import { resolveDFTFeatures, describeDFTSources } from "./dft-feature-resolver";
import type { DFTResolvedFeatures } from "./dft-feature-resolver";
import { runSynthesisReasoning } from "./synthesis-reasoning";
import { runConvexHullAnalysis, passesStabilityGate } from "./phase-diagram-engine";
import type { StabilityGateResult } from "./phase-diagram-engine";
import { invalidateGNNModel, trainGNNSurrogate } from "./graph-neural-net";
import { runStructuralMutations } from "./structural-mutator";
import { evolveRules, screenWithPatterns, getMinedRules } from "./pattern-miner";
import { findOptimalRegion, getPhaseExplorationSeedFormulas } from "./phase-explorer";
import { runFamilyAwareGeneration } from "./family-generators";
import { bayesianOptimizer } from "./bayesian-optimizer";
import { rlAgent } from "./rl-agent";
import { applyFamilyFilter, rankCandidate, computeDiscoveryScore } from "./family-filters";
import { runPrototypeGeneration, type PrototypeCandidate } from "./prototype-generator";
import { gnnPredictWithUncertainty } from "./graph-neural-net";
import { runActiveLearningCycle, getActiveLearningStats } from "./active-learning";
import { getXTBStats } from "../dft/qe-dft-engine";
import { runDiffusionGenerationCycle, getDiffusionStats } from "../ai/crystal-generator";
import { analyzeTopology, trackTopologyResult, getTopologyStats, type TopologicalAnalysis } from "../physics/topology-engine";
import { computePairingProfile, type PairingProfile } from "../physics/pairing-mechanisms";
import { encodeGenome, genomeDiversity, type MaterialGenome } from "../physics/materials-genome";
import { computeFermiSurface, type FermiSurfaceResult } from "../physics/fermi-surface-engine";
import { analyzeHydrogenNetwork, trackHydrogenNetworkResult, type HydrogenNetworkAnalysis } from "../physics/hydrogen-network-engine";
import { analyzeReactionNetwork } from "../physics/reaction-network-engine";
import { predictBandStructure, getBandSurrogateMLFeatures, type BandSurrogatePrediction } from "../physics/band-structure-surrogate";
import { passesStabilityPreFilter } from "../physics/stability-predictor";
import { detectQuantumCriticality, type QuantumCriticalAnalysis } from "../physics/quantum-criticality";
import { discoveryMemory, buildFingerprint } from "./discovery-memory";

export type EventEmitter = (type: string, data: any) => void;

function shouldContinue(): boolean {
  return state === "running";
}

type EngineState = "stopped" | "running" | "paused";
type EngineTempo = "excited" | "exploring" | "contemplating";

interface EngineStatus {
  state: EngineState;
  activeTasks: string[];
  cycleCount: number;
  lastCycleAt: string | null;
  totalMaterialsFetched: number;
  totalInsightsGenerated: number;
  totalPredictionsMade: number;
  totalSynthesisDiscovered: number;
  totalReactionsLearned: number;
  totalScCandidates: number;
  totalPhysicsComputed: number;
  totalStructuresPredicted: number;
  totalPipelineScreened: number;
  totalNovelSynthesisProposed: number;
  totalInverseDesigned: number;
  totalStructuralVariants: number;
  tempo: EngineTempo;
  statusMessage: string;
}

let wss: WebSocketServer | null = null;
let state: EngineState = "stopped";
let cycleTimer: ReturnType<typeof setTimeout> | null = null;
let cycleCount = 0;
let totalMaterialsFetched = 0;
let totalInsightsGenerated = 0;
let totalPredictionsMade = 0;
let totalSynthesisDiscovered = 0;
let totalReactionsLearned = 0;
let totalScCandidates = 0;
let totalPhysicsComputed = 0;
let totalStructuresPredicted = 0;
let totalPipelineScreened = 0;
let totalNovelSynthesisProposed = 0;
let activeTasks: Set<string> = new Set();
let lastCycleAt: string | null = null;
let allInsights: string[] = [];
let isRunningCycle = false;
let phase7Offset = 0;
let currentStrategyHint: string | null = null;
let currentStrategyFocusAreas: { area: string; priority: number }[] = [];
let currentFamilyCounts: Record<string, number> = {};
let engineTempo: EngineTempo = "exploring";
let cycleIntervalMs = 15000;
let exploitCyclesRemaining = 0;
let currentExploitFamily: string | null = null;
let totalDFTEnriched = 0;
let currentStatusMessage = "Initializing research systems";

let autonomousTotalScreened = 0;
let autonomousTotalPassed = 0;
let autonomousBestTc = 0;
let autonomousStartTime = Date.now();
let autonomousGNNRetrainCount = 0;
const alreadyScreenedFormulas = new Set<string>();
const MAX_SCREENED_CACHE_SIZE = 10000;
let lastActiveLearningCycle = 0;
let recentTcImproved = false;
let recentNewCandidates = 0;
let failuresSinceLastRetrain = 0;
let lastRetrainCycle = 0;

interface PreviousCycleMetrics {
  bestTc: number;
  bestScore: number;
  candidateCount: number;
  familyDiversity: number;
  insightCount: number;
  topFamily: string;
  pipelinePassed: number;
  pipelineTotal: number;
}
let previousCycleMetrics: PreviousCycleMetrics | null = null;
let cycleInsightsThisCycle = 0;

type ThoughtCategory = "strategy" | "discovery" | "stagnation" | "milestone";

function broadcastThought(text: string, category: ThoughtCategory) {
  broadcast("thought", { text, category });
}

function updateTempo() {
  const prevTempo = engineTempo;
  if (recentTcImproved || recentNewCandidates >= 3) {
    engineTempo = "excited";
    cycleIntervalMs = 10000;
  } else if (cyclesSinceTcImproved > 10) {
    engineTempo = "contemplating";
    cycleIntervalMs = 22000;
  } else {
    engineTempo = "exploring";
    cycleIntervalMs = 15000;
  }
  if (prevTempo !== engineTempo) {
    broadcast("tempoChange", { tempo: engineTempo, intervalMs: cycleIntervalMs });
  }
}

function generateStatusMessage(): string {
  if (state === "stopped") return "Engine offline";
  if (state === "paused") return "Research paused";

  const topFocus = currentStrategyFocusAreas[0]?.area || "";
  const tasks = Array.from(activeTasks);

  if (engineTempo === "excited") {
    if (topFocus) return `Actively pursuing ${topFocus.toLowerCase()} candidates`;
    return "Energized by recent discoveries";
  }
  if (engineTempo === "contemplating") {
    if (cyclesSinceTcImproved > 15) return "Deep analysis mode — reconsidering approach";
    return "Re-analyzing top candidates with stricter physics";
  }
  if (tasks.includes("SC Research")) return "Screening superconductor candidates";
  if (tasks.includes("Computational Physics")) return "Running physics verification";
  if (tasks.includes("Data Fetching")) return "Scanning scientific databases for new materials";
  if (topFocus) return `Exploring ${topFocus.toLowerCase()} chemical space`;
  return "Scanning chemical space for novel compositions";
}

function broadcast(type: string, data: any) {
  if (!wss) return;
  const msg = JSON.stringify({ type, data, timestamp: new Date().toISOString() });
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  });
}

const recentLogCache = new Set<string>();
const RECENT_LOG_CACHE_MAX = 100;

const DEDUP_EVENT_PATTERNS = [
  "started", "discovery started", "fetch started", "import started",
  "analysis started", "Prediction patterns discovered",
  "All top candidates have crystal structures",
];

const emit: EventEmitter = (type: string, data: any) => {
  if (type === "log" && data.event && data.phase) {
    const evt = data.event as string;
    const shouldDedup = DEDUP_EVENT_PATTERNS.some(p => evt.includes(p));
    if (shouldDedup) {
      const cacheKey = `${evt}::${data.detail || ""}`;
      if (recentLogCache.has(cacheKey)) {
        return;
      }
      recentLogCache.add(cacheKey);
      if (recentLogCache.size > RECENT_LOG_CACHE_MAX) {
        const first = recentLogCache.values().next().value;
        if (first !== undefined) recentLogCache.delete(first);
      }
    }
  }

  broadcast(type, data);

  if (type === "log" && data.event && data.phase) {
    storage.insertResearchLog({
      phase: data.phase,
      event: data.event,
      detail: data.detail || null,
      dataSource: data.dataSource || null,
    }).catch(() => {});
    if (data.event === "Novel insight discovered") {
      const detail = data.detail || "";
      const insightText = detail.replace(/^\[NOVEL \d+%\]\s*/, "");
      if (requiresQuantitativeContent(insightText)) {
        cycleInsightsThisCycle++;
      }
    }
  }
};

const PHASE_BASE_TARGETS: Record<number, number> = {
  1: 47,
  2: 118,
  3: 50,
  4: 500,
  5: 50,
  6: 200,
  7: 500,
  8: 300,
  9: 300,
  10: 200,
  11: 150,
  12: 300,
};

function dynamicTarget(phaseId: number, itemsLearned: number): number {
  const base = PHASE_BASE_TARGETS[phaseId] ?? 100;
  if (itemsLearned <= base) return base;
  return Math.ceil(itemsLearned * 1.1);
}

function computeProgress(phaseId: number, itemsLearned: number): number {
  const target = dynamicTarget(phaseId, itemsLearned);
  return Math.min(99, Math.floor((itemsLearned / target) * 100));
}

async function updatePhaseStatus(phaseId: number, status: string, progress: number, itemsLearned: number, totalItems?: number) {
  try {
    const phase = await storage.getLearningPhaseById(phaseId);
    if (!phase) return;

    const newProgress = Math.min(100, progress);
    const resolvedTotal = totalItems ?? dynamicTarget(phaseId, itemsLearned);
    const updates: any = { progress: newProgress, itemsLearned, totalItems: resolvedTotal };
    if (status === "active" && phase.status !== "active") {
      updates.startedAt = new Date();
    }
    if (status === "completed") {
      updates.completedAt = new Date();
    }
    updates.status = status;

    await storage.upsertLearningPhase({
      ...phase,
      ...updates,
    });

    broadcast("phaseUpdate", { phaseId, status, progress: newProgress, itemsLearned, totalItems: resolvedTotal });
  } catch (e) {
    console.error("updatePhaseStatus failed:", e);
  }
}

async function addInsightsToPhase(phaseId: number, newInsights: string[]) {
  if (newInsights.length === 0) return;
  try {
    const phase = await storage.getLearningPhaseById(phaseId);
    if (!phase) return;
    const existing = phase.insights ?? [];
    const combined = [...existing, ...newInsights.map(s => sanitizeForbiddenWords(s))].slice(-20);
    await storage.upsertLearningPhase({
      ...phase,
      insights: combined,
    });
  } catch (e) {
    console.error("addInsightsToPhase failed:", e);
  }
}

async function runPhase3_Bonding() {
  if (!shouldContinue()) return;
  activeTasks.add("Bonding Analysis");
  broadcast("taskStart", { task: "Bonding Analysis" });
  try {
    const mats = await storage.getMaterials(200, 0);
    if (mats.length === 0) return;

    await updatePhaseStatus(3, "active", 0, 0);
    if (!shouldContinue()) return;
    const insights = await analyzeBondingPatterns(emit, mats);
    allInsights.push(...insights);
    totalInsightsGenerated += insights.length;

    await addInsightsToPhase(3, insights);
    const formulas = mats.slice(0, 5).map(m => m.formula);
    await evaluateInsightNovelty(emit, insights, 3, "Chemical Bonding", formulas);
    const phase3 = await storage.getLearningPhaseById(3);
    const totalBondingInsights = (phase3?.insights ?? []).length;
    const progress = computeProgress(3, totalBondingInsights);
    await updatePhaseStatus(3, "active", progress, totalBondingInsights);
  } finally {
    activeTasks.delete("Bonding Analysis");
    broadcast("taskEnd", { task: "Bonding Analysis" });
  }
}

async function runPhase4_Materials() {
  if (!shouldContinue()) return;
  activeTasks.add("Data Fetching");
  broadcast("taskStart", { task: "Data Fetching" });
  try {
    await updatePhaseStatus(4, "active", 0, 0);

    const [oqmdCount, elementCount, knownCount] = await Promise.all([
      fetchOQMDMaterials(emit, 10, getNextOQMDOffset()),
      fetchElementFocusedMaterials(emit),
      fetchKnownMaterials(emit),
    ]);

    const total = oqmdCount + elementCount + knownCount;
    totalMaterialsFetched += total;

    const matCount = await storage.getMaterialCount();
    const progress = computeProgress(4, matCount);
    await updatePhaseStatus(4, "active", progress, matCount);
  } finally {
    activeTasks.delete("Data Fetching");
    broadcast("taskEnd", { task: "Data Fetching" });
  }
}

async function runPhase5_Prediction() {
  if (!shouldContinue()) return;
  activeTasks.add("Property Prediction");
  broadcast("taskStart", { task: "Property Prediction" });
  try {
    const mats = await storage.getMaterials(200, 0);
    if (mats.length === 0) return;

    await updatePhaseStatus(5, "active", 0, 0);
    if (!shouldContinue()) return;
    const insights = await analyzePropertyPredictionPatterns(emit, mats);
    allInsights.push(...insights);
    totalInsightsGenerated += insights.length;

    await addInsightsToPhase(5, insights);
    const predFormulas = mats.slice(0, 5).map(m => m.formula);
    await evaluateInsightNovelty(emit, insights, 5, "Property Prediction", predFormulas);
    const phase5 = await storage.getLearningPhaseById(5);
    const totalPredInsights = (phase5?.insights ?? []).length;
    const crCount5 = await storage.getComputationalResultCount();
    const scCount5 = await storage.getSuperconductorCount();
    const predictionWork = totalPredInsights + crCount5 + scCount5;
    const progress5 = computeProgress(5, predictionWork);
    await updatePhaseStatus(5, "active", progress5, predictionWork);
  } finally {
    activeTasks.delete("Property Prediction");
    broadcast("taskEnd", { task: "Property Prediction" });
  }
}

async function runPhase6_Discovery() {
  if (!shouldContinue()) return;
  activeTasks.add("Novel Discovery");
  broadcast("taskStart", { task: "Novel Discovery" });
  try {
    const prevPredCount = (await storage.getNovelPredictions()).length;
    const prevProgress = computeProgress(6, prevPredCount + totalPredictionsMade);
    await updatePhaseStatus(6, "active", prevProgress, prevPredCount + totalPredictionsMade);
    if (!shouldContinue()) return;

    const generated = await generateNovelFormulas(emit, allInsights.slice(-10), undefined, currentStrategyHint || undefined);
    totalPredictionsMade += generated;

    const predCount = (await storage.getNovelPredictions()).length;
    const scCount = await storage.getSuperconductorCount();
    const discoveryWork = predCount + scCount;
    const progress6 = computeProgress(6, discoveryWork);
    await updatePhaseStatus(6, "active", progress6, discoveryWork);
  } finally {
    activeTasks.delete("Novel Discovery");
    broadcast("taskEnd", { task: "Novel Discovery" });
  }
}

async function runPhase7_Superconductor() {
  if (!shouldContinue()) return;
  activeTasks.add("SC Research");
  broadcast("taskStart", { task: "SC Research" });
  try {
    await updatePhaseStatus(7, "active", 0, 0);
    if (!shouldContinue()) return;

    const matTotal = await storage.getMaterialCount();
    if (matTotal > 0) {
      phase7Offset = phase7Offset % matTotal;
    }
    const result = await runSuperconductorResearch(emit, allInsights.slice(-15), phase7Offset, {
      strategyFocusAreas: currentStrategyFocusAreas.length > 0 ? currentStrategyFocusAreas : undefined,
      familyCounts: Object.keys(currentFamilyCounts).length > 0 ? currentFamilyCounts : undefined,
      stagnationInfo: lastBestTcSeen > 0 ? { cyclesSinceImproved: cyclesSinceTcImproved, currentBestTc: lastBestTcSeen } : undefined,
    });
    phase7Offset += 200;
    totalScCandidates += result.generated;
    if (result.duplicatesSkipped > 0) {
      trackDuplicatesSkipped(result.duplicatesSkipped);
    }
    allInsights.push(...result.insights);
    totalInsightsGenerated += result.insights.length;

    await addInsightsToPhase(7, result.insights);
    await evaluateInsightNovelty(emit, result.insights, 7, "Superconductor Research");

    if (cycleCount % 5 === 0 && shouldContinue()) {
      try {
        const inverseDesigned = await generateInverseDesignCandidates(emit, allInsights);
        totalScCandidates += inverseDesigned;
      } catch (err: any) {
        emit("log", { phase: "phase-7", event: "Inverse design error", detail: err.message?.slice(0, 150), dataSource: "Inverse Design" });
      }
    }

    if (cycleCount % 8 === 0 && shouldContinue()) {
      try {
        const activeCampaigns = getAllActiveCampaigns();
        for (const campaign of activeCampaigns) {
          const inverseCandidates = runInverseCycle(campaign);
          if (inverseCandidates.length === 0) continue;

          emit("log", {
            phase: "inverse-optimizer",
            event: `Inverse design cycle ${campaign.cyclesRun}`,
            detail: `Campaign ${campaign.id}: ${inverseCandidates.length} candidates, target Tc=${campaign.target.targetTc}K`,
          });

          const inverseResults: { formula: string; tc: number; lambda: number; hull: number; pressure: number; passedPipeline: boolean }[] = [];

          for (const ic of inverseCandidates) {
            try {
              const features = extractFeatures(ic.formula);
              if (!features) continue;
              const gbResult = gbPredict(features);
              if (!gbResult || gbResult.tcPredicted < 3) continue;

              const candidateId = `inv-cand-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
              const passedStability = await insertCandidateWithStabilityCheck({
                id: candidateId,
                name: `Inverse-${ic.formula}`,
                formula: ic.formula,
                predictedTc: gbResult.tcPredicted,
                status: "theoretical",
                xgboostScore: gbResult.score,
                mlFeatures: features as any,
                notes: `Inverse design campaign ${campaign.id}, target Tc=${campaign.target.targetTc}K`,
                electronPhononCoupling: features.electronPhononLambda,
                crystalStructure: ic.prototype ?? null,
              });

              inverseResults.push({
                formula: ic.formula,
                tc: gbResult.tcPredicted,
                lambda: features.electronPhononLambda,
                hull: features.formationEnergy !== null ? Math.abs(features.formationEnergy) * 0.1 : 0.05,
                pressure: features.pressureGpa ?? 0,
                passedPipeline: passedStability,
              });

              if (passedStability) totalScCandidates++;
            } catch {}
          }

          processInverseResults(campaign, inverseResults);

          {
            const serializable = getSerializableCampaignState(campaign);
            await storage.updateInverseDesignCampaign(campaign.id, {
              cyclesRun: campaign.cyclesRun,
              bestTcAchieved: campaign.bestTcAchieved,
              bestDistance: campaign.learningState.bestDistance,
              candidatesGenerated: campaign.candidatesGenerated,
              candidatesPassedPipeline: campaign.candidatesPassedPipeline,
              status: campaign.status,
              learningState: serializable.learningState,
              convergenceHistory: serializable.convergenceHistory,
              topCandidates: serializable.topCandidates,
            });

            emit("log", {
              phase: "inverse-optimizer",
              event: `Inverse results processed`,
              detail: `Campaign ${campaign.id}: ${inverseResults.length} evaluated, ${inverseResults.filter(r => r.passedPipeline).length} passed, best distance=${campaign.learningState.bestDistance.toFixed(3)}, best Tc=${campaign.bestTcAchieved.toFixed(1)}K`,
            });
          }
        }
      } catch (err: any) {
        emit("log", { phase: "inverse-optimizer", event: "Inverse optimizer error", detail: err.message?.slice(0, 200) });
      }
    }

    if (cycleCount % 12 === 0 && shouldContinue()) {
      try {
        const activeCampaignList = getAllActiveCampaigns();
        for (const campaign of activeCampaignList) {
          const gradResult = runGradientDescentCycle(campaign.target, 4, 12);
          if (gradResult.bestTc > 10) {
            for (const r of gradResult.results) {
              if (r.finalTc > 10) {
                try {
                  const features = extractFeatures(r.finalFormula);
                  if (!features) continue;
                  const gb = gbPredict(features);
                  if (gb.tcPredicted >= 10) {
                    await insertCandidateWithStabilityCheck({
                      formula: normalizeFormula(r.finalFormula),
                      predictedTc: Math.round(gb.tcPredicted),
                      dataConfidence: "low",
                      ensembleScore: Math.min(0.9, gb.score),
                      verificationStage: 0,
                      notes: `[gradient-descent: ${r.totalSteps} steps, ${r.initialFormula}->${r.finalFormula}, improvement=${r.improvementRatio}]`,
                    });
                    totalScCandidates++;
                  }
                } catch {}
              }
            }
            emit("log", {
              phase: "gradient-optimizer",
              event: `Gradient descent cycle`,
              detail: `Campaign ${campaign.id}: best=${gradResult.bestFormula} Tc=${gradResult.bestTc.toFixed(1)}K from ${gradResult.results.length} seeds`,
            });
          }
        }
      } catch (err: any) {
        emit("log", { phase: "gradient-optimizer", event: "Gradient optimizer error", detail: err.message?.slice(0, 200) });
      }
    }

    if (cycleCount % 9 === 0 && shouldContinue()) {
      try {
        const topExisting = await storage.getSuperconductorCandidatesByTc(10);
        const existingFormulas = topExisting.map(c => c.formula);
        const targetTc = autonomousBestTc > 100 ? autonomousBestTc * 1.5 : 200;
        const pillarResult = runPillarCycle(existingFormulas, targetTc);
        let pillarInserted = 0;

        for (const formula of pillarResult.formulas) {
          if (!isValidFormula(formula)) continue;
          const normalized = normalizeFormula(formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;

          try {
            const features = extractFeatures(normalized);
            const gb = gbPredict(features);
            if (gb.tcPredicted >= 8) {
              const eval5 = pillarResult.evaluations.find(e => e.formula === formula);
              const inserted = await insertCandidateWithStabilityCheck({
                formula: normalized,
                predictedTc: Math.round(gb.tcPredicted),
                dataConfidence: "low",
                ensembleScore: Math.min(0.9, gb.score),
                verificationStage: 0,
                notes: `[5-pillar: fitness=${eval5?.compositeFitness.toFixed(2) ?? "?"}, pillars=${eval5?.satisfiedPillars ?? "?"}/5, motif=${eval5?.motifMatch ?? "?"}, weak=${eval5?.weakestPillar ?? "?"}]`,
              });
              if (inserted) {
                totalScCandidates++;
                pillarInserted++;
                if (eval5) {
                  updatePillarWeightsFromReward(gb.tcPredicted, eval5);
                }
              }
            }
          } catch {}
        }

        if (pillarResult.formulas.length > 0) {
          emit("log", {
            phase: "pillar-optimizer",
            event: "5-pillar guided generation",
            detail: `Generated ${pillarResult.evaluations.length} candidates, ${pillarResult.formulas.length} passed fitness threshold, inserted ${pillarInserted}. Best: ${pillarResult.bestFormula} (fitness=${pillarResult.bestFitness.toFixed(2)}, Tc=${pillarResult.bestTc.toFixed(1)}K)`,
            dataSource: "SC Pillars Optimizer",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "pillar-optimizer", event: "5-pillar optimizer error", detail: err.message?.slice(0, 200) });
      }
    }

    const scCount = await storage.getSuperconductorCount();
    const progress7 = computeProgress(7, scCount);
    await updatePhaseStatus(7, "active", progress7, scCount);
  } finally {
    activeTasks.delete("SC Research");
    broadcast("taskEnd", { task: "SC Research" });
  }
}

async function runPhase8_Synthesis() {
  if (!shouldContinue()) return;
  activeTasks.add("Synthesis Mapping");
  broadcast("taskStart", { task: "Synthesis Mapping" });
  try {
    await updatePhaseStatus(8, "active", 0, 0);
    if (!shouldContinue()) return;

    const mats = await storage.getMaterials(15, 0);
    const discovered = await discoverSynthesisProcesses(emit, mats);
    totalSynthesisDiscovered += discovered;

    const synthCount = await storage.getSynthesisCount();
    const progress8 = computeProgress(8, synthCount);
    await updatePhaseStatus(8, "active", progress8, synthCount);
  } finally {
    activeTasks.delete("Synthesis Mapping");
    broadcast("taskEnd", { task: "Synthesis Mapping" });
  }
}

async function runPhase9_Reactions() {
  if (!shouldContinue()) return;
  activeTasks.add("Reaction Discovery");
  broadcast("taskStart", { task: "Reaction Discovery" });
  try {
    await updatePhaseStatus(9, "active", 0, 0);
    if (!shouldContinue()) return;

    const topic = getNextReactionTopic();
    const discovered = await discoverChemicalReactions(emit, topic);
    totalReactionsLearned += discovered;

    const rxnCount = await storage.getReactionCount();
    const progress9 = computeProgress(9, rxnCount);
    await updatePhaseStatus(9, "active", progress9, rxnCount);
  } finally {
    activeTasks.delete("Reaction Discovery");
    broadcast("taskEnd", { task: "Reaction Discovery" });
  }
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

async function insertCandidateWithStabilityCheck(candidateData: Parameters<typeof storage.insertSuperconductorCandidate>[0]): Promise<boolean> {
  try {
    const preFilter = passesStabilityPreFilter(candidateData.formula);
    if (!preFilter.pass) {
      emit("log", {
        phase: "engine",
        event: "Stability pre-filter rejected",
        detail: `Fast stability screen rejected ${candidateData.formula}: ${preFilter.reason}`,
        dataSource: "Stability Predictor (GNN)",
      });
      return false;
    }

    const stabilityResult = await passesStabilityGate(candidateData.formula);

    if (!stabilityResult.pass) {
      emit("log", {
        phase: "engine",
        event: "Stability gate rejected",
        detail: `Stability gate rejected ${candidateData.formula}: ${stabilityResult.reason}`,
        dataSource: "Stability Gate",
      });
      return false;
    }

    const existingMlFeatures = (candidateData.mlFeatures as Record<string, any>) ?? {};
    const enrichedMlFeatures = {
      ...existingMlFeatures,
      stabilityGate: {
        hullDistance: stabilityResult.hullDistance,
        formationEnergy: stabilityResult.formationEnergy,
        verdict: stabilityResult.verdict,
        kineticBarrier: stabilityResult.kineticBarrier,
      },
    };

    const existingNotes = candidateData.notes || "";
    const stabilityNote = `[Stability: ${stabilityResult.verdict}, hullDist=${stabilityResult.hullDistance.toFixed(4)} eV/atom, formE=${stabilityResult.formationEnergy.toFixed(4)} eV/atom]`;

    await storage.insertSuperconductorCandidate({
      ...candidateData,
      mlFeatures: enrichedMlFeatures as any,
      notes: `${existingNotes} ${stabilityNote}`.trim(),
    });
    return true;
  } catch (err: any) {
    try {
      await storage.insertSuperconductorCandidate(candidateData);
      return true;
    } catch {
      return false;
    }
  }
}

const reEvalApplied = new Map<string, { lambda: number; omegaLog: number; muStar: number; hasCrystal: boolean }>();
let cyclesSinceTcImproved = 0;
let lastBestTcSeen = 0;
let lastBestPairingSusc = 0;
let explorationModeActive = false;
let explorationModeSavedConstraints: { allowBeyondEmpirical: boolean; empiricalPenaltyStrength: number } | null = null;

function computeEliashbergTc(lambda: number, omegaLog: number, muStar: number): number {
  if (lambda < 0.05 || omegaLog <= 0) return 0;
  const omegaLogK = omegaLog * 1.44;
  const denom = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denom) < 1e-6) return 0;
  const exponent = -1.04 * (1 + lambda) / denom;
  const tc = (omegaLogK / 1.2) * Math.exp(exponent);
  if (!Number.isFinite(tc) || tc < 0) return 0;
  return Math.round(tc);
}

async function reEvaluateTopCandidates() {
  try {
    const topByTc = await storage.getSuperconductorCandidatesByTc(50);

    const currentBestTc = Math.max(...topByTc.map(c => c.predictedTc ?? 0), 0);
    const currentBestPairing = Math.max(...topByTc.map(c => {
      const lambda = c.electronPhononCoupling ?? 0;
      const score = c.ensembleScore ?? 0;
      return lambda * 0.4 + score * 0.6;
    }), 0);

    if (currentBestTc > lastBestTcSeen + 2 || currentBestPairing > lastBestPairingSusc + 0.05) {
      cyclesSinceTcImproved = 0;
      lastBestTcSeen = Math.max(lastBestTcSeen, currentBestTc);
      lastBestPairingSusc = Math.max(lastBestPairingSusc, currentBestPairing);
    } else {
      cyclesSinceTcImproved++;
    }

    let updated = 0;
    for (const candidate of topByTc) {
      const lambda = candidate.electronPhononCoupling ?? 0;
      const omegaLog = candidate.logPhononFrequency ?? 0;
      const muStar = candidate.coulombPseudopotential ?? 0.12;

      const crystals = await storage.getCrystalStructuresByFormula(candidate.formula);
      const hasCrystal = crystals.some(c => c.synthesizability != null && c.synthesizability > 0.7);

      const prev = reEvalApplied.get(candidate.id);
      const inputsChanged = !prev ||
        (lambda > 0 && Math.abs(lambda - prev.lambda) > 0.05) ||
        (omegaLog > 0 && Math.abs(omegaLog - prev.omegaLog) > 5) ||
        (Math.abs(muStar - prev.muStar) > 0.01) ||
        (hasCrystal && !prev.hasCrystal);

      if (!inputsChanged) continue;

      reEvalApplied.set(candidate.id, { lambda, omegaLog, muStar, hasCrystal });

      let newTc = computeEliashbergTc(lambda, omegaLog, muStar);
      if (newTc <= 0) continue;

      const features = extractFeatures(candidate.formula);
      newTc = applyAmbientTcCap(newTc, lambda, candidate.pressureGpa ?? 0, features.metallicity ?? 0.5, candidate.formula);

      const currentTc = candidate.predictedTc ?? 0;
      if (newTc === currentTc) continue;

      await storage.updateSuperconductorCandidate(candidate.id, { predictedTc: newTc });
      updated++;
      emit("log", {
        phase: "engine",
        event: "Tc recomputed from physics",
        detail: `${candidate.formula}: ${currentTc}K -> ${newTc}K (lambda=${lambda.toFixed(2)}, omegaLog=${omegaLog.toFixed(0)}cm-1, mu*=${muStar.toFixed(2)})`,
        dataSource: "Learning Feedback",
      });
    }

    if (updated > 0) {
      emit("log", {
        phase: "engine",
        event: "Re-evaluation complete",
        detail: `${updated}/${topByTc.length} candidates updated. Stagnation: ${cyclesSinceTcImproved} cycles. Best Tc: ${Math.round(lastBestTcSeen)}K`,
        dataSource: "Learning Feedback",
      });
    }
  } catch (err: any) {
    emit("log", { phase: "engine", event: "Re-evaluation error", detail: err.message?.slice(0, 150) ?? "unknown", dataSource: "Learning Feedback" });
  }
}

const dftEnrichmentTracker = new Map<string, number>();

async function runDFTEnrichment() {
  if (!shouldContinue()) return;
  try {
    const candidates = await storage.getSuperconductorCandidates(100);
    const sorted = candidates
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));

    const toEnrich: typeof candidates = [];
    for (const c of sorted) {
      if (toEnrich.length >= 12) break;
      if (c.dataConfidence === "high") continue;
      toEnrich.push(c);
    }

    const analyticalCandidates = candidates
      .filter(c => !c.dataConfidence || c.dataConfidence === "analytical")
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));
    for (const c of analyticalCandidates) {
      if (toEnrich.length >= 20) break;
      if (!toEnrich.some(e => e.id === c.id)) {
        toEnrich.push(c);
      }
    }

    const stage1Candidates = candidates
      .filter(c => (c.predictedTc ?? 0) > 40 && c.dataConfidence !== "high" && c.dataConfidence !== "medium")
      .sort((a, b) => (b.predictedTc ?? 0) - (a.predictedTc ?? 0));
    for (const c of stage1Candidates) {
      if (toEnrich.length >= 25) break;
      if (!toEnrich.some(e => e.id === c.id)) {
        toEnrich.push(c);
      }
    }

    const currentCycle = cycleCount;
    const staleThreshold = 30;
    const staleMedium = candidates
      .filter(c => c.dataConfidence === "medium" && (currentCycle - (dftEnrichmentTracker.get(c.id) ?? 0)) > staleThreshold)
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0))
      .slice(0, 5);
    for (const c of staleMedium) {
      if (!toEnrich.some(e => e.id === c.id)) {
        toEnrich.push(c);
      }
    }

    if (toEnrich.length === 0) return;

    const totalCount = await storage.getSuperconductorCount();
    const highCount = candidates.filter(c => c.dataConfidence === "high").length;
    const medCount = candidates.filter(c => c.dataConfidence === "medium").length;
    const coveragePct = totalCount > 0 ? ((highCount + medCount) / totalCount * 100).toFixed(1) : "0";

    broadcastThought(
      `DFT coverage at ${coveragePct}% -- enriching next batch of ${toEnrich.length} candidates...`,
      "strategy"
    );

    let enriched = 0;
    for (const candidate of toEnrich) {
      if (!shouldContinue()) break;
      try {
        const dftData = await resolveDFTFeatures(candidate.formula);
        dftEnrichmentTracker.set(candidate.id, currentCycle);
        if (dftData.dftCoverage === 0) continue;

        const desc = describeDFTSources(dftData);
        emit("log", {
          phase: "engine",
          event: "DFT enrichment",
          detail: `${candidate.formula} -- found DFT data: ${desc}. Re-scoring...`,
          dataSource: "DFT Resolver",
        });

        const features = extractFeatures(candidate.formula, undefined, undefined, undefined, dftData);
        const gb = gbPredict(features);
        const nnScore = candidate.neuralNetScore ?? candidate.quantumCoherence ?? 0.3;
        const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);

        const updates: any = {
          xgboostScore: gb.score,
          ensembleScore: ensemble,
          dataConfidence: dftData.dftCoverage > 0.4 ? "high" : "medium",
        };

        if (dftData.formationEnergy.source !== "analytical") {
          updates.formationEnergy = dftData.formationEnergy.value;
        }
        if (dftData.bandGap.source !== "analytical") {
          updates.bandGap = dftData.bandGap.value;
        }

        if (dftData.phononStability?.hasImaginaryModes && dftData.phononStability.lowestFrequency < -2000) {
          const severeCount = dftData.phononStability.frequencies?.filter(f => f < -2000).length ?? dftData.phononStability.imaginaryModeCount;
          const penalty = Math.min(0.2, Math.max(0, severeCount - 1) * 0.08);
          if (penalty > 0) {
            updates.ensembleScore = Math.max(0.05, (updates.ensembleScore ?? ensemble) - penalty);
          }
          if (severeCount >= 3) {
            updates.dataConfidence = "low";
          }
          emit("log", {
            phase: "engine",
            event: "phonon instability",
            detail: `${candidate.formula}: xTB Hessian detected ${severeCount} severe imaginary mode(s) (lowest: ${dftData.phononStability.lowestFrequency.toFixed(0)} cm-1) — ensemble score penalized by ${penalty.toFixed(2)}`,
            dataSource: "xTB-Hessian",
          });
        } else if (dftData.phononStability?.imaginaryModeCount && dftData.phononStability.imaginaryModeCount > 0 && dftData.phononStability.lowestFrequency >= -2000) {
          emit("log", {
            phase: "engine",
            event: "phonon mild instability",
            detail: `${candidate.formula}: xTB Hessian detected ${dftData.phononStability.imaginaryModeCount} mild imaginary mode(s) (lowest: ${dftData.phononStability.lowestFrequency.toFixed(0)} cm-1) — within xTB tolerance, no penalty`,
            dataSource: "xTB-Hessian",
          });
        }

        await storage.updateSuperconductorCandidate(candidate.id, updates);
        enriched++;
        totalDFTEnriched++;
      } catch (err: any) {
        emit("log", {
          phase: "engine",
          event: "DFT enrichment error",
          detail: `${candidate.formula}: ${err.message?.slice(0, 100)}`,
          dataSource: "DFT Resolver",
        });
      }
    }

    if (enriched > 0) {
      emit("log", {
        phase: "engine",
        event: "DFT enrichment complete",
        detail: `Enriched ${enriched}/${toEnrich.length} candidates with DFT data (${totalDFTEnriched} total, coverage ~${coveragePct}%)`,
        dataSource: "DFT Resolver",
      });
    }
  } catch {}
}

async function runPhase10_Physics() {
  if (!shouldContinue()) return;
  activeTasks.add("Computational Physics");
  broadcast("taskStart", { task: "Computational Physics" });
  try {
    await updatePhaseStatus(10, "active", 0, 0);
    if (!shouldContinue()) return;

    const stage0 = await storage.getSuperconductorsByStage(0);
    const toAnalyze = shuffle(stage0).slice(0, 5);

    for (const candidate of toAnalyze) {
      if (!shouldContinue()) return;
      try {
        const result = await runFullPhysicsAnalysis(emit, candidate);
        totalPhysicsComputed++;

        const rawPhysicsTc = result.eliashberg.predictedTc;
        const physicsTc = (Number.isFinite(rawPhysicsTc) && rawPhysicsTc > 0 && rawPhysicsTc < 1000) ? rawPhysicsTc : 0;
        const currentTc = candidate.predictedTc ?? 0;
        let updatedTc = physicsTc > 0 ? Math.round(physicsTc) : currentTc;
        updatedTc = applyAmbientTcCap(updatedTc, result.coupling.lambda, candidate.pressureGpa ?? 0, result.electronicStructure.metallicity ?? 0.5, candidate.formula);

        const instProx = result.instabilityProximity;
        const existingNotes = candidate.notes || "";
        const instabilityNote = `[Instability: ${instProx.nearestBoundary}=${instProx.overallProximity.toFixed(2)}, QCP=${instProx.magneticQCP.toFixed(2)}, CDW=${instProx.cdwInstability.toFixed(2)}, MIT=${instProx.metalInsulatorTransition.toFixed(2)}]`;
        const pairingNote = `[Pairing: ${result.pairingAnalysis.dominant.mechanism} (Tc=${result.pairingAnalysis.dominant.tcEstimate.toFixed(0)}K, conf=${result.pairingAnalysis.dominant.confidence.toFixed(2)})]`;
        const newNotes = existingNotes.replace(/\[Instability:.*?\]/g, "").replace(/\[Pairing:.*?\]/g, "").trim();
        const updatedNotes = `${pairingNote} ${instabilityNote} ${newNotes}`.trim();

        const boundaryBoost = instProx.overallProximity > 0.5 ? instProx.overallProximity * 0.05 : 0;
        const currentEnsemble = candidate.ensembleScore ?? 0;
        const boostedEnsemble = Math.min(0.98, currentEnsemble + boundaryBoost);

        const existingMlFeatures = (candidate.mlFeatures as Record<string, any>) ?? {};
        const updatedMlFeatures = {
          ...existingMlFeatures,
          phononDispersion: {
            qPath: result.phononDispersion.qPath,
            branchCount: result.phononDispersion.branches.length,
            softModeQPoints: result.phononDispersion.softModeQPoints,
            imaginaryFrequencies: result.phononDispersion.imaginaryFrequencies,
            maxAcousticFreq: result.phononDispersion.maxAcousticFreq,
            minOpticalFreq: result.phononDispersion.minOpticalFreq,
            phononGap: result.phononDispersion.phononGap,
          },
          manyBodyCorrections: {
            quasiparticleWeight: result.manyBodyCorrections.quasiparticleWeight,
            gwDOSRenormalization: result.manyBodyCorrections.gwDOSRenormalization,
            gwBandwidthCorrection: result.manyBodyCorrections.gwBandwidthCorrection,
            vertexCorrectionLambda: result.manyBodyCorrections.vertexCorrectionLambda,
            correctedLambda: result.manyBodyCorrections.correctedLambda,
          },
          nestingFunction: {
            peakNestingQ: result.nestingFunction.peakNestingQ,
            peakNestingValue: result.nestingFunction.peakNestingValue,
            averageNesting: result.nestingFunction.averageNesting,
            nestingAnisotropy: result.nestingFunction.nestingAnisotropy,
            dominantInstability: result.nestingFunction.dominantInstability,
          },
          spinSusceptibility: {
            chiStaticPeak: result.spinSusceptibility.chiStaticPeak,
            chiDynamicPeak: result.spinSusceptibility.chiDynamicPeak,
            spinFluctuationEnergy: result.spinSusceptibility.spinFluctuationEnergy,
            correlationLength: result.spinSusceptibility.correlationLength,
            stonerEnhancement: result.spinSusceptibility.stonerEnhancement,
            isNearQCP: result.spinSusceptibility.isNearQCP,
          },
          phononDOS: {
            totalStates: result.phononDOS.totalStates,
            binCount: result.phononDOS.frequencies.length,
          },
          alpha2F: {
            integratedLambda: result.alpha2F.integratedLambda,
            binCount: result.alpha2F.frequencies.length,
          },
        };

        let topoAnalysis: TopologicalAnalysis | undefined;
        try {
          topoAnalysis = analyzeTopology(
            candidate.formula,
            result.electronicStructure,
            candidate.crystalStructure?.split(" ")[0],
            candidate.crystalStructure?.match(/\((\w+)\)/)?.[1]
          );
          trackTopologyResult(topoAnalysis);
          (updatedMlFeatures as any).topology = {
            topologicalScore: topoAnalysis.topologicalScore,
            z2Invariant: topoAnalysis.z2Invariant,
            chernIndicator: topoAnalysis.chernIndicator,
            mirrorSymmetryIndicator: topoAnalysis.mirrorSymmetryIndicator,
            socStrength: topoAnalysis.socStrength,
            bandInversionProbability: topoAnalysis.bandInversionProbability,
            diracNodeProbability: topoAnalysis.diracNodeProbability,
            majoranaFeasibility: topoAnalysis.majoranaFeasibility,
            topologicalClass: topoAnalysis.topologicalClass,
            indicators: topoAnalysis.indicators,
          };
          if (topoAnalysis.topologicalScore > 0.4) {
            emit("log", {
              phase: "phase-10",
              event: "Topological candidate detected",
              detail: `${candidate.formula}: class=${topoAnalysis.topologicalClass}, score=${topoAnalysis.topologicalScore}, SOC=${topoAnalysis.socStrength}, Z2=${topoAnalysis.z2Invariant}, Majorana=${topoAnalysis.majoranaFeasibility}, [${topoAnalysis.indicators.join(", ")}]`,
              dataSource: "Topology Engine",
            });
          }
        } catch {}

        let pairingProfile: PairingProfile | undefined;
        try {
          pairingProfile = computePairingProfile(candidate.formula);
          if (pairingProfile.compositePairingStrength > 0.4) {
            emit("log", {
              phase: "phase-10",
              event: "Pairing mechanism analysis",
              detail: `${candidate.formula}: dominant=${pairingProfile.dominantMechanism}, secondary=${pairingProfile.secondaryMechanism}, symmetry=${pairingProfile.pairingSymmetry}, composite=${pairingProfile.compositePairingStrength.toFixed(3)}, phonon=${pairingProfile.phonon.phononPairingStrength.toFixed(3)}, spin=${pairingProfile.spin.spinPairingStrength.toFixed(3)}, orbital=${pairingProfile.orbital.orbitalPairingStrength.toFixed(3)}`,
              dataSource: "Pairing Mechanism Simulator",
            });
          }
        } catch {}

        try {
          const reactionResult = analyzeReactionNetwork(candidate.formula);
          (updatedMlFeatures as any).reactionNetwork = {
            reactionStabilityScore: reactionResult.reactionStabilityScore,
            metastableLifetime: reactionResult.metastableLifetime,
            decompositionComplexity: reactionResult.decompositionComplexity,
            pathwayCount: reactionResult.reactionGraph?.edges?.length ?? 0,
          };
          if (reactionResult.reactionStabilityScore < 0.3) {
            emit("log", {
              phase: "phase-10",
              event: "Reaction stability warning",
              detail: `${candidate.formula}: stabilityScore=${reactionResult.reactionStabilityScore.toFixed(3)}, lifetime=${reactionResult.metastableLifetime}, complexity=${reactionResult.decompositionComplexity.toFixed(3)}, verdict=${reactionResult.stabilityVerdict}`,
              dataSource: "Reaction Network Engine",
            });
          }
        } catch {}

        let genomeResult: MaterialGenome | undefined;
        try {
          genomeResult = encodeGenome(candidate.formula);
          (updatedMlFeatures as any).genome = {
            family: genomeResult.metadata.family,
            dominantOrbital: genomeResult.metadata.dominantOrbital,
            genomeDim: genomeResult.vector.length,
          };
        } catch {}

        let fermiSurfaceAnalysis: FermiSurfaceResult | undefined;
        try {
          fermiSurfaceAnalysis = computeFermiSurface(candidate.formula);
          (updatedMlFeatures as any).fermiSurface = {
            fermiPocketCount: fermiSurfaceAnalysis.mlFeatures.fermiPocketCount,
            electronHoleBalance: fermiSurfaceAnalysis.mlFeatures.electronHoleBalance,
            fsDimensionality: fermiSurfaceAnalysis.mlFeatures.fsDimensionality,
            sigmaBandPresence: fermiSurfaceAnalysis.mlFeatures.sigmaBandPresence,
            multiBandScore: fermiSurfaceAnalysis.mlFeatures.multiBandScore,
          };
          if (fermiSurfaceAnalysis.pocketCount > 1 || fermiSurfaceAnalysis.nestingScore > 0.3) {
            emit("log", {
              phase: "phase-10",
              event: "Fermi surface reconstructed",
              detail: `${candidate.formula}: pockets=${fermiSurfaceAnalysis.pocketCount} (e=${fermiSurfaceAnalysis.electronPocketCount}, h=${fermiSurfaceAnalysis.holePocketCount}), e-h balance=${fermiSurfaceAnalysis.electronHoleBalance.toFixed(3)}, nesting=${fermiSurfaceAnalysis.nestingScore.toFixed(3)}, dim=${fermiSurfaceAnalysis.fsDimensionality}, sigma=${fermiSurfaceAnalysis.sigmaBandPresence.toFixed(3)}, multiBand=${fermiSurfaceAnalysis.multiBandScore.toFixed(3)}`,
              dataSource: "Fermi Surface Engine",
            });
          }
        } catch {}

        let bandSurrogatePrediction: BandSurrogatePrediction | undefined;
        try {
          bandSurrogatePrediction = predictBandStructure(
            candidate.formula,
            candidate.crystalStructure?.match(/\((\w+)\)/)?.[1],
          );
          (updatedMlFeatures as any).bandSurrogate = getBandSurrogateMLFeatures(bandSurrogatePrediction);
          if (bandSurrogatePrediction.flatBandScore > 0.5 || bandSurrogatePrediction.vhsProximity > 0.4 || bandSurrogatePrediction.nestingFromBands > 0.4) {
            emit("log", {
              phase: "phase-10",
              event: "Band structure surrogate prediction",
              detail: `${candidate.formula}: gap=${bandSurrogatePrediction.bandGap}eV(${bandSurrogatePrediction.bandGapType}), flatBand=${bandSurrogatePrediction.flatBandScore.toFixed(3)}, VHS=${bandSurrogatePrediction.vhsProximity.toFixed(3)}, nesting=${bandSurrogatePrediction.nestingFromBands.toFixed(3)}, DOS(EF)=${bandSurrogatePrediction.dosPredicted.toFixed(3)}, fsDim=${bandSurrogatePrediction.fsDimensionality}, multiBand=${bandSurrogatePrediction.multiBandScore.toFixed(3)}, bwMin=${bandSurrogatePrediction.bandwidthMin.toFixed(4)}, topo=${bandSurrogatePrediction.bandTopologyClass}, conf=${bandSurrogatePrediction.confidence.toFixed(2)}`,
              dataSource: "Band Structure Surrogate",
            });
          }
        } catch {}

        let qcAnalysis: QuantumCriticalAnalysis | undefined;
        try {
          qcAnalysis = detectQuantumCriticality(candidate.formula, {
            electronic: result.electronicStructure,
            coupling: result.coupling,
          });
          (updatedMlFeatures as any).quantumCriticality = {
            score: qcAnalysis.quantumCriticalScore,
            primaryQCP: qcAnalysis.primaryQCP,
            pairingBoost: qcAnalysis.pairingBoostFromQCP,
          };
          if (qcAnalysis.quantumCriticalScore > 0.5) {
            const qcBoost = 1 + qcAnalysis.pairingBoostFromQCP * 0.15;
            updatedTc = Math.min(400, updatedTc * qcBoost);
            emit("log", {
              phase: "phase-10",
              event: "Quantum criticality detected",
              detail: `${candidate.formula}: QCP=${qcAnalysis.primaryQCP}, score=${qcAnalysis.quantumCriticalScore.toFixed(3)}, dome=${qcAnalysis.domeProfile.domeCenter.toFixed(2)}, boost=${qcAnalysis.pairingBoostFromQCP.toFixed(3)}, channels=[mott=${qcAnalysis.channelScores.mott.toFixed(2)},sdw=${qcAnalysis.channelScores.sdw.toFixed(2)},cdw=${qcAnalysis.channelScores.cdw.toFixed(2)},nematic=${qcAnalysis.channelScores.nematic.toFixed(2)}]`,
              dataSource: "Quantum Criticality Detector",
            });
          }
        } catch {}

        await storage.updateSuperconductorCandidate(candidate.id, {
          electronPhononCoupling: result.coupling.lambda,
          logPhononFrequency: result.coupling.omegaLog,
          coulombPseudopotential: result.coupling.muStar,
          correlationStrength: result.correlation.ratio,
          fermiSurfaceTopology: result.electronicStructure.fermiSurfaceTopology,
          dimensionality: result.dimensionality,
          competingPhases: result.competingPhases as any,
          upperCriticalField: result.criticalFields.upperCriticalField,
          coherenceLength: result.criticalFields.coherenceLength,
          londonPenetrationDepth: result.criticalFields.londonPenetrationDepth,
          anisotropyRatio: result.criticalFields.anisotropyRatio,
          criticalCurrentDensity: result.criticalFields.criticalCurrentDensity,
          uncertaintyEstimate: result.uncertaintyEstimate,
          pairingMechanism: pairingProfile?.dominantMechanism ?? result.pairingAnalysis.dominant.mechanism,
          cooperPairMechanism: pairingProfile ? `${pairingProfile.dominantMechanism} (${pairingProfile.pairingSymmetry}), composite=${pairingProfile.compositePairingStrength.toFixed(3)}` : result.pairingAnalysis.dominant.description,
          predictedTc: updatedTc,
          verificationStage: 1,
          notes: updatedNotes,
          ensembleScore: boostedEnsemble,
          mlFeatures: updatedMlFeatures as any,
        });

        if (updatedTc !== currentTc) {
          emit("log", { phase: "phase-10", event: "Tc updated by physics", detail: `${candidate.formula}: ML estimate ${currentTc}K -> Eliashberg ${updatedTc}K (lambda=${result.coupling.lambda.toFixed(2)}, Hc2=${result.criticalFields.upperCriticalField}T, ${result.correlation.regime}, ${result.competingPhases.length} competing phases)`, dataSource: "Physics Engine" });
        }

        try {
          const trainFeatures = extractFeatures(candidate.formula);
          const hullDist = result.competingPhases.length > 0 ? 0.05 * result.competingPhases.length : 0.02;
          physicsPredictor.addTrainingSample(
            trainFeatures,
            result.coupling.lambda,
            result.electronicStructure.densityOfStatesAtFermi,
            result.coupling.omegaLog,
            hullDist
          );
        } catch {}
      } catch (err: any) {
        emit("log", { phase: "phase-10", event: "Physics analysis error", detail: `${candidate.formula}: ${err.message?.slice(0, 150)}`, dataSource: "Physics Engine" });
      }
    }

    if (shouldContinue()) {
      try {
        const allCandidates = await storage.getSuperconductorCandidates(50);
        const hydrideCandidates = allCandidates.filter(c => {
          const els = parseFormulaElements(c.formula);
          const cts: Record<string, number> = {};
          const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
          let m;
          while ((m = regex.exec(c.formula)) !== null) {
            cts[m[1]] = (cts[m[1]] || 0) + (m[2] ? parseFloat(m[2]) : 1);
          }
          const hCount = cts["H"] || 0;
          const nonmetals = ["H","He","B","C","N","O","F","Ne","Si","P","S","Cl","Ar","Ge","As","Se","Br","Kr","Te","I","Xe"];
          const metalAtoms = els.filter(e => !nonmetals.includes(e)).reduce((s, e) => s + (cts[e] || 0), 0);
          return hCount > 0 && metalAtoms > 0 && hCount / metalAtoms >= 2;
        });
        const hydrideToScan = shuffle(hydrideCandidates).slice(0, 3);
        for (const candidate of hydrideToScan) {
          if (!shouldContinue()) break;
          try {
            const pressureResult = runPressureAnalysis(emit, candidate.formula);
            const existingMlFeatures = (candidate.mlFeatures as Record<string, any>) ?? {};

            let hydrogenNetworkData: Record<string, any> = {};
            try {
              const hNetwork = analyzeHydrogenNetwork(candidate.formula);
              trackHydrogenNetworkResult(hNetwork);
              hydrogenNetworkData = {
                hydrogenNetworkDim: hNetwork.hydrogenNetworkDim,
                hydrogenCageScore: hNetwork.hydrogenCageScore,
                Hcoordination: hNetwork.Hcoordination,
                hydrogenConnectivity: hNetwork.hydrogenConnectivity,
                hydrogenPhononCouplingScore: hNetwork.hydrogenPhononCouplingScore,
                networkClass: hNetwork.networkClass,
                compositeSCScore: hNetwork.compositeSCScore,
                bondingType: hNetwork.bondingType,
              };
              if (hNetwork.compositeSCScore > 0.4) {
                emit("log", {
                  phase: "phase-10",
                  event: "Hydrogen network analysis",
                  detail: `${candidate.formula}: class=${hNetwork.networkClass}, dim=${hNetwork.hydrogenNetworkDim}, cage=${hNetwork.hydrogenCageScore.toFixed(3)}, coord=${hNetwork.Hcoordination}, phonon=${hNetwork.hydrogenPhononCouplingScore.toFixed(3)}, composite=${hNetwork.compositeSCScore.toFixed(3)}`,
                  dataSource: "Hydrogen Network Engine",
                });
              }
            } catch {}

            const updatedMlFeatures = {
              ...existingMlFeatures,
              pressureTcCurve: {
                optimalPressure: pressureResult.optimalPressure,
                maxTc: pressureResult.maxTc,
                points: pressureResult.pressureTcCurve.length,
                stablePoints: pressureResult.pressureTcCurve.filter(p => p.stable).length,
                hydridePhases: pressureResult.hydrideFormation?.stableHydrides.length ?? 0,
              },
              ...(Object.keys(hydrogenNetworkData).length > 0 ? { hydrogenNetwork: hydrogenNetworkData } : {}),
            };
            await storage.updateSuperconductorCandidate(candidate.id, {
              mlFeatures: updatedMlFeatures as any,
            });
          } catch (err: any) {
            emit("log", { phase: "phase-10", event: "Pressure scan error", detail: `${candidate.formula}: ${err.message?.slice(0, 150)}`, dataSource: "Pressure Engine" });
          }
        }

        const metallicHighTc = allCandidates.filter(c => {
          const tc = c.predictedTc ?? 0;
          const met = ((c.mlFeatures as Record<string, any>)?.metallicity) ?? 0;
          const existingPressure = ((c.mlFeatures as Record<string, any>)?.pressureTcCurve?.optimalPressure) ?? null;
          return tc >= 20 && met > 0.4 && existingPressure === null && !hydrideCandidates.some(h => h.id === c.id);
        });
        const nonHydrideToScan = shuffle(metallicHighTc).slice(0, 5);
        for (const candidate of nonHydrideToScan) {
          if (!shouldContinue()) break;
          try {
            const pressureResult = runPressureAnalysis(emit, candidate.formula);
            const ambientTc = candidate.predictedTc ?? 0;
            const pressureTc = pressureResult.maxTc;
            const existingMlFeatures = (candidate.mlFeatures as Record<string, any>) ?? {};
            const updatedMlFeatures = {
              ...existingMlFeatures,
              pressureTcCurve: {
                optimalPressure: pressureResult.optimalPressure,
                maxTc: pressureTc,
                points: pressureResult.pressureTcCurve.length,
                stablePoints: pressureResult.pressureTcCurve.filter(p => p.stable).length,
                pressureBoost: pressureTc > ambientTc,
              },
            };
            const updates: any = { mlFeatures: updatedMlFeatures as any };
            if (pressureTc > ambientTc && pressureResult.optimalPressure <= 50 && pressureResult.optimalPressure > 0) {
              updates.pressureGpa = pressureResult.optimalPressure;
              updates.optimalPressureGpa = pressureResult.optimalPressure;
              emit("log", {
                phase: "phase-10",
                event: "Pressure-enhanced Tc",
                detail: `${candidate.formula}: Tc ${ambientTc}K -> ${pressureTc.toFixed(1)}K at ${pressureResult.optimalPressure.toFixed(1)} GPa (moderate pressure, non-hydride)`,
                dataSource: "Pressure Engine",
              });
            }
            await storage.updateSuperconductorCandidate(candidate.id, updates);
          } catch (err: any) {
            emit("log", { phase: "phase-10", event: "Pressure scan error", detail: `${candidate.formula}: ${err.message?.slice(0, 150)}`, dataSource: "Pressure Engine" });
          }
        }
      } catch {}
    }

    if (cyclesSinceTcImproved > 3 && shouldContinue()) {
      const stage4 = await storage.getSuperconductorsByStage(4, 20);
      const highLambda = stage4
        .filter(c => (c.electronPhononCoupling ?? 0) > 2.0)
        .sort((a, b) => (b.predictedTc ?? 0) - (a.predictedTc ?? 0));
      const toReanalyze = shuffle(highLambda).slice(0, 2);
      for (const candidate of toReanalyze) {
        if (!shouldContinue()) break;
        try {
          const result = await runFullPhysicsAnalysis(emit, candidate);
          totalPhysicsComputed++;
          const newLambda = result.coupling.lambda ?? 0;
          const oldLambda = candidate.electronPhononCoupling ?? 0;
          if (Math.abs(newLambda - oldLambda) > 0.05) {
            const rawTc = result.eliashberg.predictedTc;
            const physicsTc = (Number.isFinite(rawTc) && rawTc > 0 && rawTc < 1000) ? rawTc : 0;
            const currentTc = candidate.predictedTc ?? 0;
            let updatedTc = physicsTc > 0 ? Math.round(physicsTc) : currentTc;
            updatedTc = applyAmbientTcCap(updatedTc, newLambda, candidate.pressureGpa ?? 0, result.electronicStructure.metallicity ?? 0.5, candidate.formula);
            await storage.updateSuperconductorCandidate(candidate.id, {
              electronPhononCoupling: newLambda,
              logPhononFrequency: result.coupling.omegaLog,
              predictedTc: updatedTc,
            });
            if (updatedTc !== currentTc) {
              emit("log", { phase: "phase-10", event: "Re-physics corrected Tc", detail: `${candidate.formula}: ${currentTc}K -> ${updatedTc}K (lambda ${oldLambda.toFixed(2)} -> ${newLambda.toFixed(2)}, Hc2=${result.criticalFields.upperCriticalField}T, ${result.correlation.regime}, ${result.competingPhases.length} competing phases)`, dataSource: "Physics Engine" });
            }
          }
        } catch {}
      }
    }

    const crCount = await storage.getComputationalResultCount();
    const progress10 = computeProgress(10, crCount);
    await updatePhaseStatus(10, "active", progress10, crCount);
  } finally {
    activeTasks.delete("Computational Physics");
    broadcast("taskEnd", { task: "Computational Physics" });
  }
}

async function runPhase11_StructurePrediction() {
  if (!shouldContinue()) return;
  activeTasks.add("Crystal Structure Prediction");
  broadcast("taskStart", { task: "Crystal Structure Prediction" });
  try {
    await updatePhaseStatus(11, "active", 0, 0);
    if (!shouldContinue()) return;

    const candidates = await storage.getSuperconductorCandidates(50);
    const uniqueFormulas = shuffle(candidates
      .map(c => c.formula)
      .filter((f, i, arr) => arr.indexOf(f) === i));

    const needsPrediction: string[] = [];
    for (const formula of uniqueFormulas) {
      if (needsPrediction.length >= 3) break;
      const existing = await storage.getCrystalStructuresByFormula(formula);
      if (existing.length === 0) {
        needsPrediction.push(formula);
      }
    }

    if (needsPrediction.length === 0) {
      emit("log", { phase: "phase-11", event: "All top candidates have crystal structures", detail: `Checked ${uniqueFormulas.length} unique formulas, all already predicted`, dataSource: "Structure Predictor" });
    }

    const predicted = needsPrediction.length > 0
      ? await runStructurePredictionBatch(emit, needsPrediction)
      : 0;
    totalStructuresPredicted += predicted;

    if (shouldContinue()) {
      for (const f of uniqueFormulas.slice(0, 5)) {
        if (!shouldContinue()) break;
        try {
          const hullResult = await runConvexHullAnalysis(emit, f);
          const matchingCandidates = candidates.filter(c => c.formula === f);
          for (const cand of matchingCandidates) {
            const existingNotes = cand.notes || "";
            const hullNote = `[ConvexHull: eAboveHull=${hullResult.energyAboveHull.toFixed(4)}, onHull=${hullResult.isOnHull}, decomp=${hullResult.decompositionProducts.join("+")}]`;
            if (!existingNotes.includes("[ConvexHull:")) {
              await storage.updateSuperconductorCandidate(cand.id, {
                notes: `${existingNotes} ${hullNote}`.trim(),
              });
            }
          }
        } catch {}
      }
    }

    if (shouldContinue() && cycleCount % 3 === 0) {
      try {
        const topCandidates = candidates
          .filter(c => (c.ensembleScore ?? 0) > 0.3)
          .map(c => ({ formula: c.formula, predictedTc: c.predictedTc ?? 0, ensembleScore: c.ensembleScore ?? 0 }));

        if (topCandidates.length > 0) {
          const variants = await runGenerativeStructureDiscovery(emit, topCandidates);

          for (const variant of variants) {
            if (!isValidFormula(variant.formula)) continue;
            variant.formula = normalizeFormula(variant.formula);
            const existingSC = await storage.getSuperconductorByFormula(variant.formula);
            if (!existingSC) {
              const features = extractFeatures(variant.formula);
              const gbResult = gbPredict(features);
              const lambdaML = features.electronPhononLambda ?? 0;
              const metallicityML = features.metallicity ?? 0.5;
              let rawTc = Math.round(lambdaML * 45 + (features.logPhononFreq ?? 200) * 0.05);
              if (rawTc > 80 && lambdaML < 1.5) {
                const penalty = lambdaML < 0.5 ? 0.15 : lambdaML < 1.0 ? 0.25 : 0.3;
                rawTc = Math.round(rawTc * penalty);
              }
              rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, variant.formula);

              const id = `sc-struct-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
              try {
                const inserted = await insertCandidateWithStabilityCheck({
                  id,
                  name: variant.formula,
                  formula: variant.formula,
                  predictedTc: rawTc,
                  pressureGpa: null,
                  meissnerEffect: false,
                  zeroResistance: false,
                  cooperPairMechanism: `Structural variant from ${variant.parentFormula} via ${variant.variationType}`,
                  crystalStructure: `${variant.spaceGroup} (${variant.crystalSystem})`,
                  quantumCoherence: variant.structuralNovelty,
                  stabilityScore: features.cooperPairStrength,
                  synthesisPath: null,
                  mlFeatures: features as any,
                  xgboostScore: gbResult.score,
                  neuralNetScore: variant.structuralNovelty,
                  ensembleScore: Math.min(0.9, (gbResult.score + variant.structuralNovelty) / 2),
                  roomTempViable: false,
                  status: "theoretical",
                  notes: `[Structural variant: ${variant.variationType}, topology=${variant.topology}, novelty=${variant.structuralNovelty.toFixed(2)}] ${variant.description}`,
                  electronPhononCoupling: features.electronPhononLambda ?? null,
                  logPhononFrequency: features.logPhononFreq ?? null,
                  coulombPseudopotential: 0.12,
                  pairingSymmetry: features.dWaveSymmetry ? "d-wave" : "s-wave",
                  pairingMechanism: "phonon-mediated",
                  correlationStrength: features.correlationStrength ?? null,
                  dimensionality: variant.dimensionality,
                  fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                  uncertaintyEstimate: 0.6,
                  verificationStage: 0,
                  dataConfidence: "low",
                });
                if (inserted) totalScCandidates++;
              } catch {}
            }
          }
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Generative structure error", detail: err.message?.slice(0, 150), dataSource: "Structure Generator" });
      }
    }

    if (shouldContinue() && cycleCount % 10 === 0) {
      try {
        const novelVariants = await runNovelPrototypeGeneration(emit);

        for (const variant of novelVariants) {
          if (!isValidFormula(variant.formula)) continue;
          const existingSC = await storage.getSuperconductorByFormula(variant.formula);
          if (!existingSC) {
            const features = extractFeatures(variant.formula);
            const gbResult = gbPredict(features);
            const lambdaML = features.electronPhononLambda ?? 0;
            const metallicityML = features.metallicity ?? 0.5;
            let rawTc = Math.round(lambdaML * 45 + (features.logPhononFreq ?? 200) * 0.05);
            rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, variant.formula);

            const id = `sc-novel-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
            try {
              const inserted = await insertCandidateWithStabilityCheck({
                id,
                name: variant.formula,
                formula: variant.formula,
                predictedTc: rawTc,
                pressureGpa: null,
                meissnerEffect: false,
                zeroResistance: false,
                cooperPairMechanism: `Novel prototype: ${variant.topology}`,
                crystalStructure: `${variant.spaceGroup} (${variant.crystalSystem})`,
                quantumCoherence: variant.structuralNovelty,
                stabilityScore: features.cooperPairStrength,
                synthesisPath: null,
                mlFeatures: features as any,
                xgboostScore: gbResult.score,
                neuralNetScore: variant.structuralNovelty,
                ensembleScore: Math.min(0.9, (gbResult.score + variant.structuralNovelty) / 2),
                roomTempViable: false,
                status: "theoretical",
                notes: `[Novel prototype: ${variant.topology}, novelty=${variant.structuralNovelty.toFixed(2)}] ${variant.description}`,
                electronPhononCoupling: features.electronPhononLambda ?? null,
                logPhononFrequency: features.logPhononFreq ?? null,
                coulombPseudopotential: 0.12,
                pairingSymmetry: features.dWaveSymmetry ? "d-wave" : "s-wave",
                pairingMechanism: "phonon-mediated",
                correlationStrength: features.correlationStrength ?? null,
                dimensionality: variant.dimensionality,
                fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                uncertaintyEstimate: 0.7,
                verificationStage: 0,
                dataConfidence: "low",
              });
              if (inserted) totalScCandidates++;
            } catch {}
          }
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Novel prototype generation error", detail: err.message?.slice(0, 150), dataSource: "Novel Prototype Generator" });
      }
    }

    if (shouldContinue() && cycleCount % 15 === 0) {
      try {
        const topForEvo = await storage.getSuperconductorCandidates(20);
        const evoInput = topForEvo.map((c: any) => ({
          formula: c.formula,
          predictedTc: c.predictedTc ?? 0,
          ensembleScore: c.ensembleScore ?? 0,
        }));
        const evoResults = await runEvolutionaryStructureSearch(evoInput, emit);
        let evoInserted = 0;
        for (const evoFormula of evoResults) {
          if (!isValidFormula(evoFormula)) continue;
          const existingSC = await storage.getSuperconductorByFormula(evoFormula);
          if (!existingSC) {
            const features = extractFeatures(evoFormula);
            const gbResult = gbPredict(features);
            const lambdaML = features.electronPhononLambda ?? 0;
            const metallicityML = features.metallicity ?? 0.5;
            let rawTc = Math.round(lambdaML * 45 + (features.logPhononFreq ?? 200) * 0.05);
            rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, evoFormula);
            const id = `sc-evo-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
            try {
              const inserted = await insertCandidateWithStabilityCheck({
                id,
                name: evoFormula,
                formula: evoFormula,
                predictedTc: rawTc,
                pressureGpa: null,
                meissnerEffect: false,
                zeroResistance: false,
                cooperPairMechanism: "Evolutionary structure search",
                crystalStructure: null,
                quantumCoherence: 0.5,
                stabilityScore: features.cooperPairStrength,
                synthesisPath: null,
                mlFeatures: features as any,
                xgboostScore: gbResult.score,
                neuralNetScore: 0.5,
                ensembleScore: Math.min(0.9, (gbResult.score + 0.5) / 2),
                roomTempViable: false,
                status: "theoretical",
                notes: `[Evolutionary structure search: mutated from top candidates]`,
                electronPhononCoupling: lambdaML || null,
                logPhononFrequency: features.logPhononFreq ?? null,
                coulombPseudopotential: 0.12,
                pairingSymmetry: features.dWaveSymmetry ? "d-wave" : "s-wave",
                pairingMechanism: "phonon-mediated",
                correlationStrength: features.correlationStrength ?? null,
                dimensionality: "3D",
                fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                uncertaintyEstimate: 0.6,
                verificationStage: 0,
                dataConfidence: "low",
              });
              if (inserted) {
                totalScCandidates++;
                evoInserted++;
              }
            } catch {}
          }
        }
        if (evoInserted > 0) {
          emit("log", {
            phase: "phase-11",
            event: "Evolutionary candidates inserted",
            detail: `${evoInserted} novel structures from evolutionary search`,
            dataSource: "Structure Evolution",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Evolutionary search error", detail: err.message?.slice(0, 150), dataSource: "Structure Evolution" });
      }
    }

    if (shouldContinue() && cycleCount % 5 === 0) {
      try {
        const diffResult = runDiffusionGenerationCycle(30);
        let diffInserted = 0;
        for (const crystal of diffResult.structures) {
          if (!isValidFormula(crystal.formula)) continue;
          const normalized = normalizeFormula(crystal.formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;

          const features = extractFeatures(normalized);
          const gbResult = gbPredict(features);
          const lambdaML = features.electronPhononLambda ?? 0;
          const metallicityML = features.metallicity ?? 0.5;
          let rawTc = Math.round(lambdaML * 45 + (features.logPhononFreq ?? 200) * 0.05);
          if (rawTc > 80 && lambdaML < 1.5) {
            const penalty = lambdaML < 0.5 ? 0.15 : lambdaML < 1.0 ? 0.25 : 0.3;
            rawTc = Math.round(rawTc * penalty);
          }
          rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, normalized);

          const id = `sc-diff-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          try {
            const inserted = await insertCandidateWithStabilityCheck({
              id,
              name: normalized,
              formula: normalized,
              predictedTc: rawTc,
              pressureGpa: null,
              meissnerEffect: false,
              zeroResistance: false,
              cooperPairMechanism: `Diffusion-generated ${crystal.spaceGroup} (${crystal.crystalSystem})`,
              crystalStructure: `${crystal.spaceGroup} (${crystal.crystalSystem})`,
              quantumCoherence: crystal.noveltyScore,
              stabilityScore: features.cooperPairStrength,
              synthesisPath: null,
              mlFeatures: features as any,
              xgboostScore: gbResult.score,
              neuralNetScore: crystal.noveltyScore,
              ensembleScore: Math.min(0.9, (gbResult.score + crystal.noveltyScore) / 2),
              roomTempViable: false,
              status: "theoretical",
              notes: `[Crystal diffusion: ${crystal.prototypeMatch || "novel"}, SG=${crystal.spaceGroup}, density=${crystal.densityGcm3} g/cm3, novelty=${crystal.noveltyScore.toFixed(2)}]`,
              electronPhononCoupling: features.electronPhononLambda ?? null,
              logPhononFrequency: features.logPhononFreq ?? null,
              coulombPseudopotential: 0.12,
              pairingSymmetry: features.dWaveSymmetry ? "d-wave" : "s-wave",
              pairingMechanism: "phonon-mediated",
              correlationStrength: features.correlationStrength ?? null,
              dimensionality: "3D",
              fermiSurfaceTopology: features.fermiSurfaceType ?? null,
              uncertaintyEstimate: crystal.prototypeMatch ? 0.55 : 0.7,
              verificationStage: 0,
              dataConfidence: "low",
            });
            if (inserted) {
              totalScCandidates++;
              diffInserted++;
              bayesianOptimizer.addObservation(normalized, rawTc, lambdaML, crystal.noveltyScore);
            }
          } catch {}
        }
        if (diffInserted > 0 || diffResult.structures.length > 0) {
          emit("log", {
            phase: "phase-11",
            event: "Crystal diffusion generation",
            detail: `Generated ${diffResult.structures.length} structures (${diffResult.stats.novel} novel), inserted ${diffInserted}. Avg novelty: ${diffResult.stats.avgNovelty}, protos: ${Object.entries(diffResult.stats.protoBreakdown).map(([k, v]) => `${k}:${v}`).join(", ")}`,
            dataSource: "Crystal Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Crystal diffusion error", detail: err.message?.slice(0, 150), dataSource: "Crystal Diffusion" });
      }
    }

    if (shouldContinue() && cycleCount % 7 === 0) {
      try {
        const structResult = runStructureDiffusionCycle(200, 3, 3);
        let structInserted = 0;
        for (const formula of structResult.formulas) {
          if (!isValidFormula(formula)) continue;
          const normalized = normalizeFormula(formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;

          try {
            const features = extractFeatures(normalized);
            const gbResult = gbPredict(features);
            if (gbResult.tcPredicted >= 10) {
              const inserted = await insertCandidateWithStabilityCheck({
                formula: normalized,
                predictedTc: Math.round(gbResult.tcPredicted),
                dataConfidence: "low",
                ensembleScore: Math.min(0.9, gbResult.score),
                verificationStage: 0,
                notes: `[structure-first: motif-designed, target=200K]`,
              });
              if (inserted) {
                totalScCandidates++;
                structInserted++;
              }
            }
          } catch {}
        }
        if (structResult.formulas.length > 0) {
          emit("log", {
            phase: "phase-11",
            event: "Structure-first design",
            detail: `Generated ${structResult.formulas.length} from motifs [${structResult.motifsUsed.join(", ")}], inserted ${structInserted}, best=${structResult.bestFormula} Tc=${structResult.bestTc.toFixed(1)}K`,
            dataSource: "Structure Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Structure-first design error", detail: err.message?.slice(0, 150), dataSource: "Structure Diffusion" });
      }
    }

    const csCount = await storage.getCrystalStructureCount();
    const progress11 = computeProgress(11, csCount);
    await updatePhaseStatus(11, "active", progress11, csCount);
  } finally {
    activeTasks.delete("Crystal Structure Prediction");
    broadcast("taskEnd", { task: "Crystal Structure Prediction" });
  }
}

async function runPhase12_MultiFidelity() {
  if (!shouldContinue()) return;
  activeTasks.add("Multi-Fidelity Screening");
  broadcast("taskStart", { task: "Multi-Fidelity Screening" });
  try {
    await updatePhaseStatus(12, "active", 0, 0);
    if (!shouldContinue()) return;

    const stage0 = await storage.getSuperconductorsByStage(0, 50);
    const stage1 = await storage.getSuperconductorsByStage(1, 50);
    const eligible = [...stage0, ...stage1].filter(c => (c.ensembleScore ?? 0) > 0.25);
    const unscreened = shuffle(eligible).slice(0, 6);

    if (unscreened.length > 0) {
      const results = await runMultiFidelityPipeline(emit, unscreened);
      totalPipelineScreened += results.length;

      const passedCount = results.filter(r => r.passed).length;
      if (passedCount > 0) {
        allInsights.push(`Multi-fidelity pipeline: ${passedCount}/${results.length} candidates passed all 5 stages`);
      }
    }

    const crCount = await storage.getComputationalResultCount();
    const progress12 = computeProgress(12, crCount);
    await updatePhaseStatus(12, "active", progress12, crCount);
  } finally {
    activeTasks.delete("Multi-Fidelity Screening");
    broadcast("taskEnd", { task: "Multi-Fidelity Screening" });
  }
}

async function runPhase13_SynthesisReasoning() {
  if (!shouldContinue()) return;
  activeTasks.add("Novel Synthesis Reasoning");
  broadcast("taskStart", { task: "Novel Synthesis Reasoning" });
  try {
    const topCandidates = await storage.getSuperconductorCandidatesByTc(20);
    const eligible = topCandidates.filter(c =>
      (c.verificationStage ?? 0) >= 2 || (c.predictedTc ?? 0) > 100
    );
    const toProcess = eligible.slice(0, 3);

    if (toProcess.length === 0) return;

    let proposed = 0;
    for (const candidate of toProcess) {
      if (!shouldContinue()) return;
      try {
        const existingPath = candidate.synthesisPath as any;
        const hasPhysicsReasoned = Array.isArray(existingPath?.routes)
          && existingPath.routes.some((r: any) => r.source === "physics-reasoned");
        if (hasPhysicsReasoned) continue;

        const routes = await runSynthesisReasoning(emit, candidate);
        if (routes && routes.length > 0) {
          const existingRoutes = Array.isArray(existingPath?.routes) ? existingPath.routes : [];
          const taggedExisting = existingRoutes.map((r: any) => ({
            ...r,
            source: r.source || "literature-based",
          }));
          await storage.updateSuperconductorCandidate(candidate.id, {
            synthesisPath: {
              routes: [...taggedExisting, ...routes],
              lastUpdated: new Date().toISOString(),
            },
          });
          proposed += routes.length;
          totalNovelSynthesisProposed += routes.length;
        }
      } catch (err: any) {
        emit("log", {
          phase: "phase-13",
          event: "Synthesis reasoning candidate error",
          detail: `${candidate.formula}: ${err.message?.slice(0, 100)}`,
          dataSource: "Synthesis Reasoning",
        });
      }
    }

    if (proposed > 0) {
      allInsights.push(`Novel synthesis reasoning: proposed ${proposed} physics-reasoned routes for ${toProcess.length} candidates`);
    }
  } finally {
    activeTasks.delete("Novel Synthesis Reasoning");
    broadcast("taskEnd", { task: "Novel Synthesis Reasoning" });
  }
}

async function runAutonomousDiscoveryCycle(formula: string): Promise<{ passed: boolean; tc: number; reason: string; physicsPred?: PhysicsPrediction }> {
  try {
    if (typeof formula !== "string") {
      formula = String(formula ?? "");
    }
    if (!formula || !isValidFormula(formula)) {
      return { passed: false, tc: 0, reason: "invalid-elements" };
    }

    const existingCandidate = await storage.getSuperconductorByFormula(formula);
    if (existingCandidate) {
      return { passed: false, tc: existingCandidate.predictedTc ?? 0, reason: "duplicate" };
    }

    const stabilityScreen = passesStabilityPreFilter(formula);
    if (!stabilityScreen.pass) {
      emit("log", {
        phase: "autonomous-loop",
        event: "Stability pre-filter rejected",
        detail: `${formula}: ${stabilityScreen.reason}`,
        dataSource: "Stability Predictor (GNN)",
      });
      return { passed: false, tc: 0, reason: `stability-prefilter: ${stabilityScreen.reason}` };
    }

    const family = classifyFamily(formula);
    const features = extractFeatures(formula);
    if (!features) {
      return { passed: false, tc: 0, reason: "feature-extraction-failed" };
    }

    const gbResult = gbPredict(features);
    if (gbResult.tcPredicted < 5) {
      return { passed: false, tc: gbResult.tcPredicted, reason: "low-gb-tc" };
    }

    let gnnResult: ReturnType<typeof gnnPredictWithUncertainty> | null = null;
    try {
      gnnResult = gnnPredictWithUncertainty(formula);
    } catch {}

    let primaryTc = gbResult.tcPredicted;
    let ensembleConfidence = 0.3;
    if (gnnResult && gnnResult.confidence > 0.3) {
      primaryTc = gnnResult.tc * 0.6 + gbResult.tcPredicted * 0.4;
      ensembleConfidence = gnnResult.confidence * 0.6 + gbResult.score * 0.3 + 0.1;
    }

    const physicsPred = physicsPredictor.predict(features);
    const preFilter = physicsPredictor.preFilter(physicsPred);
    if (!preFilter.pass) {
      return { passed: false, tc: Math.round(primaryTc), reason: `physics-prefilter: ${preFilter.reason}`, physicsPred };
    }

    const candidate = {
      formula,
      family,
      predictedTc: Math.round(primaryTc),
      confidence: "low" as const,
      source: "autonomous-loop",
      ensembleScore: Math.min(0.95, ensembleConfidence),
      pressureGpa: 0,
      verificationStage: 0,
    };

    const structureBatch = await runStructurePredictionBatch(emit, [candidate as any]);
    const structureResult = Array.isArray(structureBatch) ? structureBatch[0] : undefined;

    try {
      await runConvexHullAnalysis(emit, formula);
    } catch {}

    const physicsResult = await runFullPhysicsAnalysis(emit, candidate as any);
    const rawTc = physicsResult.eliashberg.predictedTc;
    const physicsTc = (Number.isFinite(rawTc) && rawTc > 0 && rawTc < 1000) ? rawTc : 0;
    let finalTc = physicsTc > 0 ? Math.round(physicsTc) : Math.round(gbResult.tcPredicted);
    finalTc = applyAmbientTcCap(finalTc, physicsResult.coupling.lambda, 0, physicsResult.electronicStructure.metallicity ?? 0.5, formula);

    try {
      const hullDist = physicsResult.competingPhases.length > 0 ? 0.05 * physicsResult.competingPhases.length : 0.02;
      physicsPredictor.addTrainingSample(
        features,
        physicsResult.coupling.lambda,
        physicsResult.electronicStructure.densityOfStatesAtFermi,
        physicsResult.coupling.omegaLog,
        hullDist
      );
    } catch {}

    let autonomousQC: QuantumCriticalAnalysis | undefined;
    try {
      autonomousQC = detectQuantumCriticality(formula, {
        electronic: physicsResult.electronicStructure,
        coupling: physicsResult.coupling,
      });
      if (autonomousQC.quantumCriticalScore > 0.5 && autonomousQC.pairingBoostFromQCP > 0.1) {
        const qcTcBoost = 1 + autonomousQC.pairingBoostFromQCP * 0.15;
        finalTc = Math.min(400, Math.round(finalTc * qcTcBoost));
      }
    } catch {}

    const synthesizabilityScore = structureResult?.synthesizability ?? 0.5;
    const lambda = physicsResult.coupling.lambda;
    const rawHullDist = physicsResult.competingPhases.length > 0 ? 0.05 * physicsResult.competingPhases.length : 0.02;
    const stabilityHullDist = Math.min(rawHullDist, 0.20);

    let tier: 1 | 2 | 3 | 0 = 0;
    let confidenceLevel: "high" | "medium" | "low" = "low";
    let verificationStage = 0;

    if (finalTc > 70 && lambda > 1.2 && stabilityHullDist < 0.10) {
      tier = 1;
      confidenceLevel = "high";
      verificationStage = 2;
    } else if (finalTc > 25 && lambda > 0.5 && stabilityHullDist < 0.20) {
      tier = 2;
      confidenceLevel = "medium";
      verificationStage = 1;
    } else if (finalTc > 10 && lambda > 0.3) {
      tier = 3;
      confidenceLevel = "low";
      verificationStage = 0;
    }

    if (tier === 0) {
      return { passed: false, tc: finalTc, reason: "below-tier3-thresholds" };
    }

    if (synthesizabilityScore < 0.2 && tier > 1) {
      tier = Math.min(tier + 1, 3) as 2 | 3;
      if (tier === 3) confidenceLevel = "low";
    }

    const instProx = physicsResult.instabilityProximity;
    const pairingNote = `[Pairing: ${physicsResult.pairingAnalysis.dominant.mechanism} (Tc=${physicsResult.pairingAnalysis.dominant.tcEstimate.toFixed(0)}K)]`;
    const instNote = `[Instability: ${instProx.nearestBoundary}=${instProx.overallProximity.toFixed(2)}]`;
    const tierNote = `[Tier ${tier}]`;

    const normalizedFormula = normalizeFormula(formula);
    const candidatePayload = {
      id: `sc-auto-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      name: normalizedFormula,
      formula: normalizedFormula,
      predictedTc: finalTc,
      dataConfidence: confidenceLevel,
      source: "autonomous-loop",
      family,
      electronPhononCoupling: physicsResult.coupling.lambda,
      logPhononFrequency: physicsResult.coupling.omegaLog,
      coulombPseudopotential: physicsResult.coupling.muStar,
      correlationStrength: physicsResult.correlation.ratio,
      fermiSurfaceTopology: physicsResult.electronicStructure.fermiSurfaceTopology,
      dimensionality: physicsResult.dimensionality,
      competingPhases: physicsResult.competingPhases as any,
      upperCriticalField: physicsResult.criticalFields.upperCriticalField,
      coherenceLength: physicsResult.criticalFields.coherenceLength,
      londonPenetrationDepth: physicsResult.criticalFields.londonPenetrationDepth,
      pairingMechanism: physicsResult.pairingAnalysis.dominant.mechanism,
      cooperPairMechanism: physicsResult.pairingAnalysis.dominant.description,
      ensembleScore: Math.min(0.95, (() => {
        const gnnScore = gnnResult ? (Math.min(1, gnnResult.tc > 100 ? 0.8 : gnnResult.tc > 20 ? 0.5 : 0.2) * gnnResult.confidence) : 0;
        const gbScore = gbResult.score;
        const noveltyBonus = synthesizabilityScore * 0.1;
        return gnnScore > 0 ? (gnnScore * 0.6 + gbScore * 0.3 + noveltyBonus) : (0.3 + (finalTc / 400) + (synthesizabilityScore * 0.2));
      })()),
      verificationStage,
      notes: `${pairingNote} ${instNote} ${tierNote} [autonomous-loop]${gnnResult ? ` [GNN: Tc=${gnnResult.tc}K, λ=${gnnResult.lambda}, conf=${(gnnResult.confidence * 100).toFixed(0)}%]` : ''}`,
      mlFeatures: {
        phononDOS: { totalStates: physicsResult.phononDOS.totalStates, binCount: physicsResult.phononDOS.frequencies.length },
        alpha2F: { integratedLambda: physicsResult.alpha2F.integratedLambda, binCount: physicsResult.alpha2F.frequencies.length },
        tier,
        ...(gnnResult ? { gnnTc: gnnResult.tc, gnnLambda: gnnResult.lambda, gnnUncertainty: gnnResult.uncertainty, gnnConfidence: gnnResult.confidence } : {}),
      } as any,
    };

    let autonomousInserted = false;
    if (tier === 1) {
      autonomousInserted = await insertCandidateWithStabilityCheck(candidatePayload);
    } else {
      try {
        await storage.insertSuperconductorCandidate(candidatePayload);
        autonomousInserted = true;
      } catch (insertErr: any) {
        const isDuplicate = insertErr?.message?.includes("duplicate") || insertErr?.code === "23505";
        if (isDuplicate) {
          autonomousInserted = true;
        } else {
          autonomousInserted = false;
        }
      }
    }

    if (!autonomousInserted) {
      return { passed: false, tc: finalTc, reason: `insert-failed (λ=${lambda.toFixed(2)},tier=${tier})` };
    }

    totalScCandidates++;
    recentNewCandidates++;

    try {
      const memFingerprint = buildFingerprint(formula, finalTc, {
        lambda: physicsResult.coupling.lambda,
        metallicity: physicsResult.electronicStructure.metallicity,
        nestingScore: physicsResult.electronicStructure.nestingScore,
        vanHoveProximity: physicsResult.electronicStructure.vanHoveProximity,
        dimensionality: physicsResult.dimensionality,
        correlationStrength: physicsResult.correlation.ratio,
      });
      discoveryMemory.recordDiscovery(formula, memFingerprint, finalTc);
    } catch {}

    if (finalTc > autonomousBestTc) {
      autonomousBestTc = finalTc;
    }

    return { passed: true, tc: finalTc, reason: `accepted-tier${tier}`, physicsPred };
  } catch (err: any) {
    return { passed: false, tc: 0, reason: `error: ${err.message?.slice(0, 80)}${err.stack ? ' @ ' + (err.stack.split('\n')[1] || '').trim().slice(0, 60) : ''}` };
  }
}

function generateFastPathFormulas(focusArea: string): string[] {
  const topCandidatesForGen: { formula: string; predictedTc?: number }[] = [];
  const scElements: Record<string, string[][]> = {
    Carbides: [["Nb","C"],["Ti","C"],["Mo","C"],["V","C"],["Zr","C"],["Hf","C"],["Ta","C"],["W","C"]],
    Borides: [["Nb","B"],["Ti","B"],["Zr","B"],["Mo","B"],["V","B"],["Ta","B"],["Hf","B"],["W","B"]],
    Nitrides: [["Nb","N"],["Ti","N"],["Zr","N"],["V","N"],["Mo","N"],["Ta","N"],["Hf","N"]],
    Hydrides: [["La","H"],["Y","H"],["Ca","H"],["Sr","H"],["Ba","H"],["Th","H"],["Sc","H"]],
    Intermetallics: [["Nb","Ge"],["Nb","Sn"],["V","Si"],["Nb","Al"],["Mo","Ge"],["V","Ga"]],
    Cuprates: [["Ba","Cu","O"],["La","Cu","O"],["Y","Ba","Cu","O"],["Bi","Sr","Cu","O"]],
  };
  const basePairs = scElements[focusArea] || scElements["Carbides"];
  for (const pair of basePairs) {
    topCandidatesForGen.push({ formula: pair.join(""), predictedTc: 20 });
  }
  const { formulas, stats } = runMassiveGeneration(topCandidatesForGen, focusArea);
  console.log(`Massive generation: ${stats.totalGenerated} generated, ${stats.uniqueAfterDedup} unique, ${stats.passedPreScreen} passed pre-screen`);
  return formulas;
}

async function runAutonomousFastPath() {
  if (!shouldContinue()) return;
  activeTasks.add("Autonomous Screening");
  broadcast("taskStart", { task: "Autonomous Screening" });

  try {
    const rlState = {
      bestTc: autonomousBestTc || (previousCycleMetrics?.bestTc ?? 0),
      avgRecentTc: previousCycleMetrics?.bestTc ?? 0,
      recentRewardTrend: 0,
      familyDiversity: previousCycleMetrics?.familyDiversity ?? 1,
      stagnationCycles: cyclesSinceTcImproved,
      explorationBudgetUsed: autonomousTotalScreened / Math.max(1, autonomousTotalScreened + 1000),
      elementSuccessEntropy: 0.5,
      cycleNumber: cycleCount,
    };

    const rlAction = rlAgent.selectAction(rlState);
    const rlDescription = rlAgent.getActionDescription(rlAction);

    const rlCandidates = rlAgent.generateCandidatesFromAction(rlAction, 30);

    let focusArea = currentStrategyFocusAreas[0]?.area || "Carbides";
    const EXPLORATION_FAMILIES = [
      "Pnictides", "Chalcogenides", "Cuprates", "Hydrides", "Kagome",
      "Sulfides", "Intermetallics", "Alloys", "Oxides", "Nitrides",
    ];
    const EXPLORATION_PROB = 0.15;
    if (Math.random() < EXPLORATION_PROB) {
      const explorationPool = EXPLORATION_FAMILIES.filter(f => f !== focusArea);
      focusArea = explorationPool[Math.floor(Math.random() * explorationPool.length)];
    }

    emit("log", {
      phase: "engine",
      event: "RL agent action",
      detail: `RL selected: ${rlDescription}. Generated ${rlCandidates.length} RL-directed candidates. Focus: ${focusArea}. Epsilon=${rlAgent.getStats().epsilon.toFixed(3)}, temp=${rlAgent.getStats().temperature.toFixed(3)}`,
      dataSource: "RL Agent",
    });

    let topCandidatesForGen: { formula: string; predictedTc?: number }[] = [];
    try {
      const existingTop = await storage.getSuperconductorCandidatesByTc(20);
      topCandidatesForGen = existingTop.map(c => ({ formula: c.formula, predictedTc: c.predictedTc ?? 0 }));
    } catch {}

    const shuffled = [...topCandidatesForGen].sort(() => Math.random() - 0.5);
    const { formulas: massiveCandidates, stats: genStats } = runMassiveGeneration(shuffled, focusArea);

    const boCandidatePool = [...new Set([...rlCandidates, ...massiveCandidates])];
    const boSuggestions = bayesianOptimizer.suggestNextCandidates(boCandidatePool, 50, "mixed");
    const boTopFormulas = boSuggestions.map(s => s.formula);

    const remainingMassive = massiveCandidates.filter(f => !boTopFormulas.includes(f));
    const candidates = [...boTopFormulas, ...remainingMassive];

    if (boSuggestions.length > 0) {
      const topBO = boSuggestions[0];
      emit("log", {
        phase: "engine",
        event: "Bayesian optimization ranking",
        detail: `BO ranked ${boSuggestions.length} candidates. Top: ${topBO.formula} (acq=${topBO.acquisitionValue.toFixed(2)}, mean=${topBO.predictedMean.toFixed(1)}K, std=${topBO.predictedStd.toFixed(2)}, source=${topBO.source}). GP observations: ${bayesianOptimizer.getStats().observationCount}`,
        dataSource: "Bayesian Optimizer",
      });
    }

    const constraintFiltered = constraintGuidedGenerate(candidates);
    const physicsCleanCandidates = [...constraintFiltered.valid, ...constraintFiltered.repaired];

    if (constraintFiltered.rejected.length > 0 || constraintFiltered.repaired.length > 0) {
      emit("log", {
        phase: "engine",
        event: "Physics constraint filter",
        detail: `${candidates.length} candidates → ${constraintFiltered.valid.length} valid, ${constraintFiltered.repaired.length} repaired, ${constraintFiltered.rejected.length} rejected. Violations: ${Object.entries(getConstraintEngineStats().violationCounts).map(([k, v]) => `${k}:${v}`).join(", ")}`,
        dataSource: "Physics Constraints",
      });
    }

    const novelCandidates = physicsCleanCandidates.filter(f => !alreadyScreenedFormulas.has(f));
    const rlNoveltyRatio = rlCandidates.filter(f => !alreadyScreenedFormulas.has(f)).length / Math.max(1, rlCandidates.length);
    for (const f of physicsCleanCandidates) alreadyScreenedFormulas.add(f);
    if (alreadyScreenedFormulas.size > MAX_SCREENED_CACHE_SIZE) {
      const toRemove = alreadyScreenedFormulas.size - MAX_SCREENED_CACHE_SIZE;
      const iter = alreadyScreenedFormulas.values();
      for (let i = 0; i < toRemove; i++) {
        alreadyScreenedFormulas.delete(iter.next().value as string);
      }
    }

    emit("log", {
      phase: "engine",
      event: `Massive generation: ${genStats.totalGenerated} generated, ${genStats.uniqueAfterDedup} unique, ${genStats.passedPreScreen} passed pre-screen, ${novelCandidates.length} novel`,
      detail: `Valence filter: ${genStats.passedValenceFilter}, compatibility filter: ${genStats.passedCompatibilityFilter}. Focus: ${focusArea}. Already screened cache: ${alreadyScreenedFormulas.size}. Feeding ${novelCandidates.length} novel formulas through autonomous pipeline.`,
      dataSource: "Candidate Generator",
    });

    let passed = 0;
    let bestTcThisBatch = 0;
    let bestFormulaThisBatch = "";
    const failedFormulas: { formula: string; tc: number }[] = [];

    const activeRules = getMinedRules();
    let filteredCandidates = novelCandidates;
    let patternFiltered = 0;
    if (activeRules.length > 0) {
      const patternScores = screenWithPatterns(novelCandidates);
      const scored = patternScores.sort((a, b) => b.theoryScore - a.theoryScore);
      const beforeCount = novelCandidates.length;
      filteredCandidates = scored
        .filter(s => s.theoryScore >= 0.3)
        .map(s => s.formula);
      patternFiltered = beforeCount - filteredCandidates.length;
      if (filteredCandidates.length === 0) filteredCandidates = novelCandidates;
    }

    let physicsPrefiltered = 0;
    for (const formula of filteredCandidates) {
      if (!shouldContinue()) break;
      autonomousTotalScreened++;

      const result = await runAutonomousDiscoveryCycle(formula);

      bayesianOptimizer.addObservation(formula, result.tc, result.physicsPred?.lambda ?? 0, result.passed ? 1 : 0);

      try {
        const els = parseFormulaElements(formula);
        rlAgent.recordElementOutcome(els, result.tc, result.passed);
      } catch {}

      try {
        const constraintDetail = constraintFiltered.details.find(d => d.formula === formula);
        if (constraintDetail && constraintDetail.violations.length > 0) {
          updateConstraintWeightsFromReward(formula, result.tc, constraintDetail.violations);
        }
      } catch {}

      if (result.passed) {
        passed++;
        autonomousTotalPassed++;
        if (result.tc > bestTcThisBatch) {
          bestTcThisBatch = result.tc;
          bestFormulaThisBatch = formula;
        }
        if (result.physicsPred) {
          const p = result.physicsPred;
          emit("log", {
            phase: "engine",
            event: "Physics ML prediction",
            detail: `Physics ML: lambda=${p.lambda.toFixed(2)}±${p.lambdaUncertainty.toFixed(2)}, DOS=${p.dosAtEF.toFixed(2)}±${p.dosUncertainty.toFixed(2)}, omega=${p.omegaLog.toFixed(0)}±${p.omegaUncertainty.toFixed(0)}, hull=${p.hullDistance.toFixed(3)}±${p.hullUncertainty.toFixed(3)} for ${formula}`,
            dataSource: "Physics ML",
          });
        }
        console.log(`[Autonomous] PASSED: ${formula} Tc=${result.tc}K reason=${result.reason}`);
      } else {
        failedFormulas.push({ formula, tc: result.tc });
        if (result.reason.startsWith("physics-prefilter")) physicsPrefiltered++;
        if (autonomousTotalScreened <= 200 || autonomousTotalScreened % 50 === 0) {
          console.log(`[Autonomous] REJECTED: ${formula} Tc=${result.tc}K reason=${result.reason}`);
        }
      }
    }

    if (failedFormulas.length > 0 && autonomousTotalScreened % 100 === 0) {
      try {
        const failureTrainingData = failedFormulas
          .filter(f => f.tc > 0)
          .map(f => ({ formula: f.formula, tc: f.tc, formationEnergy: undefined as number | undefined, structure: undefined as any }));
        if (failureTrainingData.length >= 5) {
          invalidateGNNModel();
          autonomousGNNRetrainCount++;
        }
      } catch {}
    }

    if (physicsPredictor.shouldRetrain(cycleCount)) {
      physicsPredictor.retrain(cycleCount);
      emit("log", {
        phase: "engine",
        event: "Physics ML retrained",
        detail: `PhysicsPredictor retrained on ${physicsPredictor.getTrainingSize()} samples at cycle ${cycleCount}`,
        dataSource: "Physics ML",
      });
    }

    let rlReward = rlAgent.computeReward(
      bestTcThisBatch,
      autonomousBestTc,
      passed > 0,
      passed / Math.max(1, filteredCandidates.length),
      rlNoveltyRatio * 0.5
    );

    const memStats = discoveryMemory.getStats();
    if (memStats.totalRecords > 5 && bestTcThisBatch > 20 && bestFormulaThisBatch) {
      try {
        const topFp = buildFingerprint(bestFormulaThisBatch, bestTcThisBatch, {});
        const memBonus = discoveryMemory.computeMemoryRewardBonus(topFp);
        rlReward += memBonus.bonus * 0.5;
      } catch {}
    }

    rlAgent.updatePolicy(rlState, rlAction, rlReward);

    const boStats = bayesianOptimizer.getStats();
    const rlStats = rlAgent.getStats();

    const patternDetail = patternFiltered > 0 ? ` Pattern filter removed ${patternFiltered}/${candidates.length} (${activeRules.length} rules active).` : "";
    const physicsDetail = physicsPrefiltered > 0 ? ` Physics pre-filter rejected ${physicsPrefiltered}/${filteredCandidates.length}.` : "";
    const memDetail = memStats.totalRecords > 0 ? ` Memory: ${memStats.totalRecords} patterns, ${memStats.clusterCount} clusters.` : "";
    emit("log", {
      phase: "engine",
      event: "Autonomous loop: " + filteredCandidates.length + " screened, " + passed + " passed" + (bestTcThisBatch > 0 ? ", best Tc = " + bestTcThisBatch + "K" : ""),
      detail: `RL+BO guided pipeline from ${focusArea} (${genStats.totalGenerated} massive + ${rlCandidates.length} RL-directed). RL reward=${rlReward.toFixed(3)}, updates=${rlStats.totalUpdates}, BO obs=${boStats.observationCount}.${patternDetail}${physicsDetail}${memDetail} Pass rate: ${(autonomousTotalPassed / Math.max(1, autonomousTotalScreened) * 100).toFixed(1)}%. Total screened: ${autonomousTotalScreened}. PhysicsML training: ${physicsPredictor.getTrainingSize()} samples.`,
      dataSource: "Autonomous Loop",
    });
  } finally {
    activeTasks.delete("Autonomous Screening");
    broadcast("taskEnd", { task: "Autonomous Screening" });
  }
}

export function getAutonomousLoopStats() {
  const elapsedHours = (Date.now() - autonomousStartTime) / 3600000;
  const alStats = getActiveLearningStats();
  const xtbStats = getXTBStats();
  return {
    totalScreened: autonomousTotalScreened,
    totalPassed: autonomousTotalPassed,
    passRate: autonomousTotalScreened > 0 ? autonomousTotalPassed / autonomousTotalScreened : 0,
    bestTc: autonomousBestTc,
    throughputPerHour: elapsedHours > 0 ? Math.round(autonomousTotalScreened / elapsedHours) : 0,
    gnnRetrainCount: autonomousGNNRetrainCount,
    activeLearning: alStats,
    realDFT: {
      method: "GFN2-xTB v6.7.1",
      runs: xtbStats.runs,
      successes: xtbStats.successes,
      cacheSize: xtbStats.cacheSize,
      successRate: xtbStats.runs > 0 ? `${(xtbStats.successes / xtbStats.runs * 100).toFixed(1)}%` : "N/A",
    },
    rlAgent: rlAgent.getStats(),
    bayesianOptimizer: bayesianOptimizer.getStats(),
    crystalDiffusion: getDiffusionStats(),
    topologyDetection: getTopologyStats(),
    inverseOptimizer: getInverseOptimizerStats(),
    differentiableOptimizer: getDifferentiableOptimizerStats(),
    structureFirstDesign: getStructureDiffusionStats(),
    physicsConstraints: getConstraintEngineStats(),
    scPillarsOptimizer: getPillarOptimizerStats(),
  };
}

async function runLearningCycle() {
  if (state !== "running" || isRunningCycle) return;
  isRunningCycle = true;

  cycleCount++;
  lastCycleAt = new Date().toISOString();
  cycleInsightsThisCycle = 0;
  recentNewCandidates = 0;
  recentTcImproved = false;
  recentLogCache.clear();
  broadcast("cycleStart", { cycle: cycleCount });

  let cycleStartDetail = "";
  if (previousCycleMetrics && currentStrategyFocusAreas.length > 0) {
    const topFocus = currentStrategyFocusAreas[0]?.area || "broad exploration";
    if (previousCycleMetrics.bestTc > 0) {
      const tcTrend = previousCycleMetrics.bestScore >= 0.9 ? "scores are strong" : "still building evidence";
      cycleStartDetail = `Cycle ${cycleCount}: Focusing on ${topFocus} (${tcTrend}, best Tc: ${Math.round(previousCycleMetrics.bestTc)}K). ${previousCycleMetrics.familyDiversity} families explored so far.`;

      if (cyclesSinceTcImproved > 5) {
        setBoundaryHuntingMode(true);
        setInverseDesignMode(true);
        broadcastThought(
          `No Tc improvement in ${cyclesSinceTcImproved} cycles. Current best: ${Math.round(previousCycleMetrics.bestTc)}K. Activating boundary hunting and inverse design modes to explore instability edges...`,
          "stagnation"
        );
      }

      if (cyclesSinceTcImproved >= 12 && cyclesSinceTcImproved < 16) {
        setMutationIntensity(2);
        emit("log", {
          phase: "engine",
          event: "Tc plateau level-2 diversification",
          detail: `Tc plateau: ${cyclesSinceTcImproved} cycles without improvement. Increasing mutation magnitude (level 2: wider element swaps).`,
          dataSource: "Engine",
        });
      } else if (cyclesSinceTcImproved >= 16 && cyclesSinceTcImproved < 20) {
        setMutationIntensity(3);
        exploitCyclesRemaining = 0;
        currentExploitFamily = null;
        emit("log", {
          phase: "engine",
          event: "Tc plateau level-3 diversification",
          detail: `Tc plateau: ${cyclesSinceTcImproved} cycles without improvement. Forcing strategy switch + exotic element substitutions (level 3).`,
          dataSource: "Engine",
        });
      } else if (cyclesSinceTcImproved >= 20) {
        setMutationIntensity(3);
        setChemicalSpaceExpansionMode(true);
        exploitCyclesRemaining = 0;
        currentExploitFamily = null;
        emit("log", {
          phase: "engine",
          event: "Tc plateau level-4 chemical space expansion",
          detail: `Tc plateau: ${cyclesSinceTcImproved} cycles without improvement. Activating chemical space expansion: novel elements + unexplored structure types.`,
          dataSource: "Engine",
        });
      } else if (cyclesSinceTcImproved <= 5) {
        setChemicalSpaceExpansionMode(false);
        setMutationIntensity(1);
      }

      if (cyclesSinceTcImproved >= 8 && cyclesSinceTcImproved % 8 === 0) {
        const savedMode = getConstraintMode();
        setConstraintMode({
          empiricalPenaltyStrength: 0.5,
          allowBeyondEmpirical: true,
        });
        explorationModeActive = true;
        explorationModeSavedConstraints = savedMode;
        broadcastThought(
          `Activating adaptive exploration mode: relaxing physics constraints for 1 cycle to search unexplored compositional space. Penalty strength reduced to 0.5, lambda caps raised 30%.`,
          "strategy"
        );
        emit("log", {
          phase: "phase-6",
          event: "Adaptive exploration activated",
          detail: `Stagnation at ${cyclesSinceTcImproved} cycles. Temporarily relaxing empirical penalties for broader search.`,
          dataSource: "Engine",
        });
      } else if (previousCycleMetrics.insightCount > 0) {
        broadcastThought(
          `Cycle ${cycleCount}: Last cycle produced ${cycleInsightsThisCycle || 0} insights. Focusing on ${topFocus} — ${previousCycleMetrics.familyDiversity} families in the search space, best Tc at ${Math.round(previousCycleMetrics.bestTc)}K.`,
          "strategy"
        );
      }
    } else {
      cycleStartDetail = `Cycle ${cycleCount}: Building knowledge base. Targeting ${topFocus} for superconductor discovery.`;
      broadcastThought(`Still building the knowledge foundation. Targeting ${topFocus} as the most promising direction.`, "strategy");
    }
  } else if (cycleCount <= 3) {
    cycleStartDetail = `Cycle ${cycleCount}: Initializing knowledge base. Gathering materials, synthesis paths, and reaction data before superconductor screening.`;
    broadcastThought(`Starting up. Gathering materials from OQMD, AFLOW, and literature databases before I can begin screening for superconductors.`, "discovery");
  } else {
    cycleStartDetail = `Cycle ${cycleCount}: Continuing exploration. Materials + synthesis + reactions first, then analysis, then SC research.`;
  }

  emit("log", {
    phase: "engine",
    event: `Learning cycle ${cycleCount} started`,
    detail: cycleStartDetail,
    dataSource: "Internal",
  });

  try {
    await Promise.allSettled([
      runPhase4_Materials(),
      runPhase8_Synthesis(),
      runPhase9_Reactions(),
    ]);

    if (state !== "running") return;

    await Promise.allSettled([
      runPhase3_Bonding(),
      runPhase5_Prediction(),
    ]);

    if (state !== "running") return;

    await runPhase6_Discovery();

    if (state !== "running") return;

    const matCount = await storage.getMaterialCount();
    const synthCount = await storage.getSynthesisCount();
    const rxnCount = await storage.getReactionCount();

    if (matCount >= 5 && synthCount >= 3 && rxnCount >= 3) {
      await runPhase7_Superconductor();
    } else {
      emit("log", {
        phase: "phase-7",
        event: "SC research deferred",
        detail: `Waiting for more knowledge: ${matCount} materials (need 5+), ${synthCount} synthesis paths (need 3+), ${rxnCount} reactions (need 3+)`,
        dataSource: "SC Research",
      });
    }

    if (state !== "running") return;

    const scCount = await storage.getSuperconductorCount();
    if (scCount >= 3) {
      await Promise.allSettled([
        runPhase10_Physics(),
        runPhase11_StructurePrediction(),
      ]);

      if (state !== "running") return;

      await runPhase12_MultiFidelity();

      if (state === "running") {
        await reEvaluateTopCandidates();
      }

      if (state === "running") {
        await runDFTEnrichment();
      }

      if (state === "running" && cycleCount % 25 === 0) {
        try {
          const failedResults = await storage.getFailedComputationalResults(500);
          const newFailures = failedResults.length - failuresSinceLastRetrain;
          if (newFailures >= 10) {
            const added = await incorporateFailureData();
            if (added > 0) {
              failuresSinceLastRetrain = failedResults.length;
              lastRetrainCycle = cycleCount;
              emit("log", {
                phase: "engine",
                event: `XGBoost model retrained with ${getFailureExampleCount()} failure examples`,
                detail: `Added ${added} new failure examples from pipeline. Total failure training data: ${getFailureExampleCount()}. Retrain triggered at cycle ${cycleCount}.`,
                dataSource: "ML Engine",
              });
            }
          }
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "XGBoost retrain error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "ML Engine",
          });
        }
      }

      if (state === "running" && cycleCount % 50 === 0) {
        try {
          await evolveRules(emit);
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "Pattern mining error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Pattern Miner",
          });
        }
      }

      if (state === "running") {
        await runPhase13_SynthesisReasoning();
      }

      const alStats0 = getActiveLearningStats();
      const alCooldown = cycleCount - lastActiveLearningCycle >= 5;
      const shouldRunAL = state === "running" && cycleCount >= 15 && alCooldown && (
        (cycleCount - lastActiveLearningCycle >= 15) ||
        (alStats0.totalDFTRuns === 0 && lastActiveLearningCycle === 0)
      );
      if (shouldRunAL) {
        try {
          console.log(`[Active Learning] Triggered at cycle ${cycleCount} (last AL cycle: ${lastActiveLearningCycle})`);
          const alStats = await runActiveLearningCycle(emit, { cycleCount });
          lastActiveLearningCycle = cycleCount;
          emit("log", {
            phase: "engine",
            event: "Active learning cycle complete",
            detail: `DFT runs: ${alStats.totalDFTRuns}, retrains: ${alStats.modelRetrains}, uncertainty: ${alStats.avgUncertaintyBefore.toFixed(3)} → ${alStats.avgUncertaintyAfter.toFixed(3)}, best Tc: ${alStats.bestTcFromLoop.toFixed(1)}K`,
            dataSource: "Active Learning",
          });
          autonomousGNNRetrainCount += alStats.modelRetrains > 0 ? 1 : 0;
        } catch (err: any) {
          lastActiveLearningCycle = cycleCount;
          console.log(`[Active Learning] Error at cycle ${cycleCount}: ${err.message?.slice(0, 150)}`);
          emit("log", {
            phase: "engine",
            event: "Active learning error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Active Learning",
          });
        }
      }

      if (state === "running" && cycleCount % 10 === 0) {
        try {
          const topForMutation = await storage.getSuperconductorCandidatesByTc(10);
          if (topForMutation.length > 0) {
            const mutationInput = topForMutation.map(c => ({
              formula: c.formula,
              predictedTc: c.predictedTc,
            }));
            const mutResult = runStructuralMutations(mutationInput, emit);

            let mutInserted = 0;
            const mutantFormulas = [
              ...mutResult.distorted.filter(d => d.energyPenalty < 0.3).map(d => d.formula),
              ...mutResult.layered.map(l => l.formula),
              ...mutResult.vacancy.map(v => v.formula),
              ...mutResult.strained.filter(s => Math.abs(s.strainPercent) < 5).map(s => s.formula),
            ];
            for (const mf of mutantFormulas.slice(0, 20)) {
              if (!shouldContinue()) break;
              if (!isValidFormula(mf)) continue;
              const existing = await storage.getSuperconductorByFormula(mf);
              if (!existing) {
                const features = extractFeatures(mf);
                const gb = gbPredict(features);
                if (gb.tcPredicted >= 10) {
                  const lambdaML = features.electronPhononLambda ?? 0;
                  const metallicityML = features.metallicity ?? 0.5;
                  let rawTc = Math.round(lambdaML * 45 + (features.logPhononFreq ?? 200) * 0.05);
                  rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, mf);
                  const id = `sc-mut-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
                  try {
                    const inserted = await insertCandidateWithStabilityCheck({
                      id,
                      name: mf,
                      formula: mf,
                      predictedTc: rawTc,
                      pressureGpa: null,
                      meissnerEffect: false,
                      zeroResistance: false,
                      cooperPairMechanism: "Structural mutation",
                      crystalStructure: null,
                      quantumCoherence: 0.5,
                      stabilityScore: features.cooperPairStrength,
                      synthesisPath: null,
                      mlFeatures: features as any,
                      xgboostScore: gb.score,
                      neuralNetScore: 0.5,
                      ensembleScore: Math.min(0.9, (gb.score + 0.5) / 2),
                      roomTempViable: false,
                      status: "theoretical",
                      notes: "[structural-mutation]",
                      electronPhononCoupling: lambdaML || null,
                      logPhononFrequency: features.logPhononFreq ?? null,
                      coulombPseudopotential: 0.12,
                      pairingSymmetry: features.dWaveSymmetry ? "d-wave" : "s-wave",
                      pairingMechanism: "phonon-mediated",
                      correlationStrength: features.correlationStrength ?? null,
                      dimensionality: "3D",
                      fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                      uncertaintyEstimate: 0.6,
                      verificationStage: 0,
                      dataConfidence: "low",
                    });
                    if (inserted) {
                      totalScCandidates++;
                      mutInserted++;
                    }
                  } catch {}
                }
              }
            }
            if (mutInserted > 0) {
              emit("log", {
                phase: "engine",
                event: "Structural mutation candidates inserted",
                detail: `${mutInserted} viable mutant candidates from ${mutResult.totalGenerated} structural variants`,
                dataSource: "Structural Mutator",
              });
            }
          }
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "Structural mutation error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Structural Mutator",
          });
        }
      }

      if (state === "running" && cycleCount >= 20 && cycleCount % 20 === 0) {
        try {
          const focusFamily = currentStrategyFocusAreas[0]?.area || "Carbides";
          const FAMILY_ELEMENT_SETS: Record<string, string[][]> = {
            Carbides: [["Nb","C"],["Ti","C"],["Mo","C"],["V","C"],["Nb","Ti","C"],["Nb","Mo","C"]],
            Borides: [["Nb","B"],["Ti","B"],["Zr","B"],["Mg","B"],["Nb","Ti","B"]],
            Nitrides: [["Nb","N"],["Ti","N"],["Zr","N"],["V","N"],["Nb","Ti","N"]],
            Hydrides: [["La","H"],["Y","H"],["Ca","H"],["Sr","H"],["La","Y","H"]],
            Intermetallics: [["Nb","Sn"],["V","Si"],["Nb","Ge"],["Nb","Al"]],
          };
          const elementSets = FAMILY_ELEMENT_SETS[focusFamily] || FAMILY_ELEMENT_SETS["Carbides"];
          const chosenSet = elementSets[cycleCount % elementSets.length];
          const optimalResults = findOptimalRegion(chosenSet, emit);
          const seedFormulas = optimalResults
            .filter(r => r.predictedTc > 10 && r.hullDistance < 0.3)
            .map(r => r.formula)
            .slice(0, 5);
          if (seedFormulas.length > 0) {
            for (const sf of seedFormulas) {
              if (!shouldContinue()) break;
              if (!isValidFormula(sf)) continue;
              const existing = await storage.getSuperconductorByFormula(sf);
              if (!existing) {
                const features = extractFeatures(sf);
                const gb = gbPredict(features);
                if (gb.tcPredicted >= 10) {
                  try {
                    const inserted = await insertCandidateWithStabilityCheck({
                      formula: normalizeFormula(sf),
                      predictedTc: Math.round(gb.tcPredicted),
                      dataConfidence: "low",
                      ensembleScore: Math.min(0.9, gb.score),
                      verificationStage: 0,
                      notes: `[phase-explorer: optimal from ${chosenSet.join("-")} scan, family=${classifyFamily(sf)}]`,
                    });
                    if (inserted) {
                      totalScCandidates++;
                      recentNewCandidates++;
                    }
                  } catch {}
                }
              }
            }
          }
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "Phase exploration error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Phase Explorer",
          });
        }
      }

      if (state === "running" && cycleCount >= 15 && cycleCount % 25 === 0) {
        try {
          const prototypeCandidates = runPrototypeGeneration();
          const prototypeCounts: Record<string, { generated: number; passed: number; inserted: number }> = {};
          let protoGenerated = 0;
          let protoPassedStability = 0;
          let protoInserted = 0;
          let bestDiscoveryScore = 0;

          const existingCandidates = await storage.getSuperconductorCandidates(500);
          const existingFormulas = new Set(existingCandidates.map(c => c.formula));
          const existingFormulaList = existingCandidates.map(c => c.formula);

          const deduped = prototypeCandidates.filter(pc => {
            const normalized = normalizeFormula(pc.formula);
            return !existingFormulas.has(normalized) && isValidFormula(pc.formula);
          });

          protoGenerated = deduped.length;

          const scored: {
            pc: PrototypeCandidate;
            normalized: string;
            features: ReturnType<typeof extractFeatures>;
            gbResult: ReturnType<typeof gbPredict>;
            gnnResult: ReturnType<typeof gnnPredictWithUncertainty>;
            discoveryScore: number;
            discoveryDetails: ReturnType<typeof computeDiscoveryScore>;
          }[] = [];

          for (const pc of deduped) {
            if (!shouldContinue()) break;
            const normalized = normalizeFormula(pc.formula);

            const features = extractFeatures(normalized);
            const gbResult = gbPredict(features);
            const gnnResult = gnnPredictWithUncertainty(normalized, pc.prototype);

            const familyMap: Record<string, string> = {
              "MAX-phase": "MAX-phase",
              "AlB2-type": "Boride",
              "Clathrate": "Hydride",
              "Sodalite": "Hydride",
              "Layered nitride": "Nitride",
            };
            const familyKey = familyMap[pc.prototype];
            if (familyKey) {
              const filterResult = applyFamilyFilter(normalized, familyKey, features);
              if (!filterResult.pass) continue;
            }

            let protoTopoScore = 0;
            try {
              const protoElectronic = computeElectronicStructure(normalized, null);
              const protoTopo = analyzeTopology(normalized, protoElectronic, undefined, pc.crystalSystem);
              protoTopoScore = protoTopo.topologicalScore;
              trackTopologyResult(protoTopo);
            } catch {}

            const discoveryDetails = computeDiscoveryScore({
              predictedTc: gbResult.tcPredicted,
              formula: normalized,
              hullDistance: null,
              synthesisScore: null,
              prototype: pc.prototype,
              existingFormulas: existingFormulaList.slice(0, 100),
              topologicalScore: protoTopoScore,
              uncertaintyEstimate: gnnResult?.uncertainty ?? 0.5,
            });

            if (!prototypeCounts[pc.prototype]) {
              prototypeCounts[pc.prototype] = { generated: 0, passed: 0, inserted: 0 };
            }
            prototypeCounts[pc.prototype].generated++;

            scored.push({
              pc,
              normalized,
              features,
              gbResult,
              gnnResult,
              discoveryScore: discoveryDetails.discoveryScore,
              discoveryDetails,
            });
          }

          scored.sort((a, b) => b.discoveryScore - a.discoveryScore);
          const topProto = scored.slice(0, 30);

          for (const entry of topProto) {
            if (!shouldContinue()) break;
            const { pc, normalized, features, gbResult, gnnResult, discoveryScore, discoveryDetails } = entry;

            const lambdaML = features.electronPhononLambda ?? 0;
            const metallicityML = features.metallicity ?? 0.5;
            const isHydride = pc.prototype === "Clathrate" || pc.prototype === "Sodalite";
            let predictedTc: number;
            if (gnnResult.confidence > 0.3 && gnnResult.tc > 0) {
              predictedTc = Math.round(gnnResult.tc * 0.6 + gbResult.tcPredicted * 0.4);
            } else {
              predictedTc = Math.round(gbResult.tcPredicted);
            }
            predictedTc = applyAmbientTcCap(predictedTc, lambdaML, isHydride ? 150 : 0, metallicityML, normalized);

            try {
              const id = `sc-proto-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
              const siteStr = Object.entries(pc.siteAssignment).map(([k, v]) => `${k}=${v.join(",")}`).join("; ");
              const inserted = await insertCandidateWithStabilityCheck({
                id,
                name: `${pc.prototype} ${normalized}`,
                formula: normalized,
                predictedTc,
                pressureGpa: isHydride ? 150 : null,
                meissnerEffect: false,
                zeroResistance: false,
                cooperPairMechanism: `${pc.prototype} prototype search`,
                crystalStructure: `${pc.spaceGroup} (${pc.crystalSystem})`,
                quantumCoherence: null,
                stabilityScore: features.cooperPairStrength ?? null,
                synthesisPath: null,
                mlFeatures: {
                  ...features as any,
                  prototype: pc.prototype,
                  spaceGroup: pc.spaceGroup,
                  crystalSystem: pc.crystalSystem,
                  dimensionality: pc.dimensionality,
                  siteAssignment: pc.siteAssignment,
                  gnnUncertainty: gnnResult.uncertainty,
                  gnnLambda: gnnResult.lambda,
                  gnnTc: gnnResult.tc,
                },
                xgboostScore: gbResult.score,
                neuralNetScore: gnnResult.confidence,
                ensembleScore: Math.min(0.9, gnnResult.confidence * 0.6 + gbResult.score * 0.3 + (discoveryScore > 0.5 ? 0.1 : 0.05)),
                roomTempViable: false,
                status: "theoretical",
                notes: `[${pc.prototype} prototype: ${pc.spaceGroup}, ${pc.crystalSystem}, ${pc.dimensionality}, sites: ${siteStr}] [Discovery: ${discoveryScore.toFixed(3)}]`,
                electronPhononCoupling: gnnResult.lambda || lambdaML || null,
                logPhononFrequency: features.logPhononFreq ?? null,
                coulombPseudopotential: isHydride ? 0.10 : 0.12,
                pairingSymmetry: features.dWaveSymmetry ? "d-wave" : "s-wave",
                pairingMechanism: "phonon-mediated",
                correlationStrength: features.correlationStrength ?? null,
                dimensionality: pc.dimensionality === "3D" ? "3D" : "2D",
                fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                uncertaintyEstimate: gnnResult.uncertainty,
                verificationStage: 0,
                dataConfidence: "low",
                discoveryScore,
              });
              bayesianOptimizer.addObservation(normalized, predictedTc, gnnResult.lambda || lambdaML, discoveryScore);

              if (inserted) {
                prototypeCounts[pc.prototype].passed++;
                protoPassedStability++;
                prototypeCounts[pc.prototype].inserted++;
                protoInserted++;
                totalScCandidates++;
                recentNewCandidates++;
                if (discoveryScore > bestDiscoveryScore) {
                  bestDiscoveryScore = discoveryScore;
                }
              }
            } catch {}
          }

          const protoSummary = Object.entries(prototypeCounts)
            .map(([p, c]) => `${c.generated} ${p}`)
            .join(", ");
          emit("log", {
            phase: "engine",
            event: "Prototype search complete",
            detail: `Prototype search: ${protoSummary}, ${protoPassedStability} passed stability, ${protoInserted} inserted (best discovery score: ${bestDiscoveryScore.toFixed(3)})`,
            dataSource: "Prototype Generator",
          });
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "Prototype search error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Prototype Generator",
          });
        }
      }

      if (state === "running" && cycleCount >= 10) {
        await runAutonomousFastPath();
      }
    } else {
      emit("log", {
        phase: "phase-10",
        event: "Physics pipeline deferred",
        detail: `Waiting for SC candidates: ${scCount} (need 3+)`,
        dataSource: "Physics Engine",
      });
    }

    if (state === "running") {
      try {
        const prevFocusAreas = currentStrategyFocusAreas.map(f => ({ ...f }));
        const strategy = await analyzeAndEvolveStrategy(emit, cycleCount);
        if (strategy) {
          const llmTopFamily = strategy.focusAreas[0]?.area || "";
          const llmTopMaxTc = (strategy.performanceSignals?.familyStats as any)?.[llmTopFamily]?.maxTc ?? 0;
          const currentMaxTc = currentExploitFamily
            ? ((strategy.performanceSignals?.familyStats as any)?.[currentExploitFamily]?.maxTc ?? 0)
            : 0;

          const exploreProbability = 0.15;
          const shouldRandomExplore = Math.random() < exploreProbability;
          const underExplored = strategy.performanceSignals?.underExplored as string[] | undefined;

          if (exploitCyclesRemaining > 0 && currentExploitFamily) {
            const tcGapJustifiesSwitch = llmTopFamily !== currentExploitFamily && llmTopMaxTc > currentMaxTc + 20;

            if (tcGapJustifiesSwitch) {
              currentExploitFamily = llmTopFamily;
              exploitCyclesRemaining = 8;
              emit("log", {
                phase: "engine",
                event: "Strategy override",
                detail: `Switching to ${llmTopFamily} (Tc ${llmTopMaxTc}K vs ${currentMaxTc}K) — large Tc gap overrides exploit window`,
                dataSource: "Strategy Analyzer",
              });
            } else if (shouldRandomExplore && underExplored && underExplored.length > 0) {
              const randomFamily = underExplored[Math.floor(Math.random() * underExplored.length)];
              const explorationAreas = [
                { area: randomFamily, priority: 0.8, reasoning: "Random exploration of under-explored family" },
                ...strategy.focusAreas.filter(f => f.area !== randomFamily).slice(0, 4),
              ];
              strategy.focusAreas = explorationAreas as any;
              emit("log", {
                phase: "engine",
                event: "Exploration probe",
                detail: `Random exploration: probing ${randomFamily} (${exploitCyclesRemaining} exploit cycles remaining for ${currentExploitFamily})`,
                dataSource: "Strategy Analyzer",
              });
            } else {
              const currentFamilyArea = strategy.focusAreas.find(f => f.area === currentExploitFamily);
              if (currentFamilyArea) {
                currentFamilyArea.priority = Math.max(currentFamilyArea.priority, 0.8);
              }
              strategy.focusAreas.sort((a, b) => {
                if (a.area === currentExploitFamily) return -1;
                if (b.area === currentExploitFamily) return 1;
                return b.priority - a.priority;
              });
            }
            exploitCyclesRemaining--;
          } else {
            currentExploitFamily = llmTopFamily;
            exploitCyclesRemaining = 8;
            emit("log", {
              phase: "engine",
              event: "Exploit window started",
              detail: `Locking onto ${llmTopFamily} for 8 cycles (Tc: ${llmTopMaxTc}K)`,
              dataSource: "Strategy Analyzer",
            });
          }

          currentStrategyHint = strategy.focusAreas
            .slice(0, 3)
            .map(f => f.area)
            .join(", ");
          currentStrategyFocusAreas = strategy.focusAreas.map(f => ({ area: f.area, priority: f.priority }));
          if (strategy.performanceSignals?.familyStats) {
            currentFamilyCounts = {};
            for (const [fam, stats] of Object.entries(strategy.performanceSignals.familyStats as Record<string, { count: number }>)) {
              currentFamilyCounts[fam] = stats.count;
            }
          }
          broadcast("strategyUpdate", {
            cycle: cycleCount,
            focusAreas: strategy.focusAreas,
            summary: strategy.summary,
          });

          for (const fa of strategy.focusAreas) {
            const prev = prevFocusAreas.find(p => p.area === fa.area);
            if (prev && Math.abs(fa.priority - prev.priority) > 0.15) {
              const dir = fa.priority > prev.priority ? "Promoting" : "Deprioritizing";
              broadcastThought(
                `${dir} ${fa.area} from ${(prev.priority * 100).toFixed(0)}% to ${(fa.priority * 100).toFixed(0)}% priority based on recent performance data.`,
                "strategy"
              );
            }
          }
          const newFamilies = strategy.focusAreas.filter(fa => !prevFocusAreas.find(p => p.area === fa.area));
          for (const nf of newFamilies) {
            broadcastThought(`New focus area: ${nf.area} added to research strategy at ${(nf.priority * 100).toFixed(0)}% priority.`, "discovery");
          }
        }
      } catch {}

      try {
        await captureConvergenceSnapshot(emit, cycleCount, currentStrategyHint || undefined);
        const snapshots = await storage.getConvergenceSnapshots(5);
        if (snapshots.length > 0) {
          broadcast("convergenceUpdate", {
            latest: snapshots[snapshots.length - 1],
            total: snapshots.length,
          });
        }
      } catch {}

      try {
        await checkMilestones(emit, broadcast, cycleCount, cycleInsightsThisCycle);
      } catch {}

      try {
        const currentCandidates = await storage.getSuperconductorCandidates(50);
        let currentBestTc = 0;
        let currentBestScore = 0;
        for (const c of currentCandidates) {
          if ((c.predictedTc ?? 0) > currentBestTc) currentBestTc = c.predictedTc ?? 0;
          if ((c.ensembleScore ?? 0) > currentBestScore) currentBestScore = c.ensembleScore ?? 0;
        }
        const { classifyFamily: classifyFam } = await import("./utils");
        const currentFamilies = new Set(currentCandidates.map(c => classifyFam(c.formula)));
        const currentDiversity = currentFamilies.size;
        const currentInsightCount = await storage.getNovelInsightCount();
        const stats = await storage.getStats();
        const pipelineTotal = stats.pipelineStages.reduce((s, p) => s + p.count, 0);
        const pipelinePassed = stats.pipelineStages.reduce((s, p) => s + p.passed, 0);

        let endSummaryParts: string[] = [];
        endSummaryParts.push(`${currentCandidates.length} total candidates`);

        if (previousCycleMetrics) {
          const tcDelta = currentBestTc - previousCycleMetrics.bestTc;
          if (tcDelta > 1) {
            endSummaryParts.push(`best Tc improved by ${Math.round(tcDelta)}K to ${Math.round(currentBestTc)}K`);
            recentTcImproved = true;
          } else if (currentBestTc > 0) {
            endSummaryParts.push(`best Tc unchanged at ${Math.round(currentBestTc)}K`);
          }

          const scoreDelta = currentBestScore - previousCycleMetrics.bestScore;
          if (scoreDelta > 0.005) {
            endSummaryParts.push(`top score rose to ${currentBestScore.toFixed(3)}`);
          }

          const diversityDelta = currentDiversity - previousCycleMetrics.familyDiversity;
          if (diversityDelta > 0) {
            endSummaryParts.push(`diversity expanded to ${currentDiversity} families (+${diversityDelta})`);
            broadcastThought(`Search space expanded to ${currentDiversity} material families. New territory to explore.`, "discovery");
          }

          recentNewCandidates = currentCandidates.length - previousCycleMetrics.candidateCount;

          if (cycleInsightsThisCycle > 0) {
            endSummaryParts.push(`${cycleInsightsThisCycle} novel insights`);
          }
        } else {
          if (currentBestTc > 0) endSummaryParts.push(`best Tc: ${Math.round(currentBestTc)}K`);
          if (cycleInsightsThisCycle > 0) endSummaryParts.push(`${cycleInsightsThisCycle} novel insights`);
        }

        emit("log", {
          phase: "engine",
          event: `Cycle ${cycleCount} complete`,
          detail: endSummaryParts.join(". ") + ".",
          dataSource: "Internal",
        });

        const narrativeParts: string[] = [`Cycle ${cycleCount}:`];
        if (recentNewCandidates > 0) narrativeParts.push(`${recentNewCandidates} new candidates discovered.`);
        if (recentTcImproved) narrativeParts.push(`Tc record improved to ${Math.round(currentBestTc)}K.`);
        else if (currentBestTc > 0) narrativeParts.push(`Best Tc holds at ${Math.round(currentBestTc)}K.`);
        if (cycleInsightsThisCycle > 0) narrativeParts.push(`${cycleInsightsThisCycle} novel insight${cycleInsightsThisCycle > 1 ? "s" : ""} discovered.`);
        const topFam = currentStrategyFocusAreas[0]?.area;
        if (topFam) narrativeParts.push(`Strategy: ${topFam} focus.`);

        storage.insertResearchLog({
          phase: "engine",
          event: "cycle-narrative",
          detail: narrativeParts.join(" "),
          dataSource: "Internal",
        }).catch(() => {});

        updateTempo();
        currentStatusMessage = generateStatusMessage();
        broadcast("statusMessage", { message: currentStatusMessage, tempo: engineTempo });

        const topFamily = currentStrategyFocusAreas[0]?.area || "";
        previousCycleMetrics = {
          bestTc: currentBestTc,
          bestScore: currentBestScore,
          candidateCount: currentCandidates.length,
          familyDiversity: currentDiversity,
          insightCount: currentInsightCount,
          topFamily,
          pipelinePassed,
          pipelineTotal,
        };
      } catch {}
    }
  } catch (err: any) {
    emit("log", {
      phase: "engine",
      event: "Cycle error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "Internal",
    });
  } finally {
    if (explorationModeActive && explorationModeSavedConstraints) {
      setConstraintMode(explorationModeSavedConstraints);
      explorationModeActive = false;
      explorationModeSavedConstraints = null;
      emit("log", {
        phase: "engine",
        event: "Adaptive exploration deactivated",
        detail: "Restored normal physics constraints after exploration cycle.",
        dataSource: "Engine",
      });
    }

    isRunningCycle = false;
    broadcast("cycleEnd", { cycle: cycleCount });

    if (state === "running") {
      cycleTimer = setTimeout(runLearningCycle, cycleIntervalMs);
    }
  }
}

export function initWebSocket(server: Server) {
  wss = new WebSocketServer({ server, path: "/ws" });

  wss.on("connection", (ws) => {
    ws.send(
      JSON.stringify({
        type: "status",
        data: getStatus(),
        timestamp: new Date().toISOString(),
      })
    );
  });

  console.log("WebSocket server initialized on /ws");
}

async function backfillGBScores() {
  try {
    let totalUpdated = 0;
    let totalFailed = 0;
    let batch: any[];
    do {
      batch = await storage.getUnscoredCandidates(200);
      if (batch.length === 0) break;

      for (const c of batch) {
        try {
          const features = extractFeatures(c.formula);
          const gb = gbPredict(features);
          const nnScore = c.quantumCoherence ?? 0.3;
          const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);
          await storage.updateSuperconductorCandidate(c.id, {
            xgboostScore: gb.score,
            neuralNetScore: nnScore,
            ensembleScore: ensemble,
          });
          totalUpdated++;
        } catch {
          try {
            await storage.updateSuperconductorCandidate(c.id, {
              xgboostScore: 0.3,
              neuralNetScore: 0.3,
              ensembleScore: 0.3,
            });
          } catch {}
          totalFailed++;
        }
      }
    } while (batch.length === 200);

    if (totalUpdated > 0 || totalFailed > 0) {
      emit("log", {
        phase: "engine",
        event: "GB score backfill complete",
        detail: `Scored ${totalUpdated} candidates with gradient boosting model${totalFailed > 0 ? `, ${totalFailed} failed (set to default)` : ''}`,
        dataSource: "Internal",
      });
    }
  } catch {}
}

const PHYSICS_VERSION = 14;

async function recalculatePhysics() {
  try {
    let totalRecalculated = 0;
    const batchSize = 200;

    while (true) {
      const needsRecalc = await storage.getCandidatesNeedingPhysicsRecalc(PHYSICS_VERSION, batchSize);
      if (needsRecalc.length === 0) break;

      for (const c of needsRecalc) {
        try {
          const features = extractFeatures(c.formula);
          const gb = gbPredict(features);
          const nnScore = c.neuralNetScore ?? c.quantumCoherence ?? 0.3;
          const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);

          const featureLambda = features.electronPhononLambda ?? 0;
          const omegaLogK = (features.logPhononFreq ?? 300) * 1.44;
          const muStar = 0.12;
          let mcMillanMax = 0;
          const denom = featureLambda - muStar * (1 + 0.62 * featureLambda);
          if (featureLambda > 0.2 && Math.abs(denom) > 1e-6) {
            const exponent = -1.04 * (1 + featureLambda) / denom;
            mcMillanMax = (omegaLogK / 1.2) * Math.exp(exponent);
            if (!Number.isFinite(mcMillanMax) || mcMillanMax < 0) mcMillanMax = 0;
          }

          const corrStr = features.correlationStrength ?? 0;
          const metalScore = features.metallicity ?? 0.5;
          const pressure = c.pressureGpa ?? 0;
          const isAmbient = pressure < 10;
          const isHighPressure = pressure >= 50;
          const pressureFactor = isHighPressure ? 1.0 : isAmbient ? 0.0 : (pressure - 10) / 40;

          const recalcFamily = classifyFamily(c.formula);
          const RECALC_FAMILY_CAPS: Record<string, { ambient: number; hp: number }> = {
            Carbides: { ambient: 45, hp: 80 },
            Nitrides: { ambient: 50, hp: 90 },
            Borides: { ambient: 55, hp: 120 },
            Oxides: { ambient: 40, hp: 70 },
          };

          let tcCap: number;
          if (metalScore < 0.3) {
            tcCap = Math.min(20, mcMillanMax * 0.1 || 10);
          } else if (metalScore < 0.5) {
            tcCap = Math.min(80, mcMillanMax * 0.3 || 40);
          } else if (corrStr > 0.85) {
            tcCap = Math.min(80, mcMillanMax * 0.3 || 30);
          } else if (corrStr > 0.7) {
            tcCap = Math.min(200, mcMillanMax * 0.5 || 80);
          } else if (featureLambda < 0.3) {
            tcCap = Math.min(50, mcMillanMax > 0 ? mcMillanMax * 2.0 : 30);
          } else if (featureLambda < 0.5) {
            tcCap = Math.min(80, mcMillanMax > 0 ? mcMillanMax * 2.0 : 50);
          } else if (featureLambda < 1.0) {
            const hpCap = Math.min(150, mcMillanMax > 0 ? mcMillanMax * 1.8 : 100);
            tcCap = 80 + (hpCap - 80) * pressureFactor;
          } else if (featureLambda < 1.5) {
            const hpCap = mcMillanMax > 0 ? Math.min(250, mcMillanMax * 1.5) : 150;
            tcCap = 120 + (hpCap - 120) * pressureFactor;
          } else if (featureLambda < 2.5) {
            const hpCap = mcMillanMax > 0 ? Math.min(350, mcMillanMax * 1.3) : 250;
            tcCap = 160 + (hpCap - 160) * pressureFactor;
          } else {
            const hpCap = mcMillanMax > 0 ? Math.min(350, mcMillanMax * 1.2) : 300;
            tcCap = 200 + (hpCap - 200) * pressureFactor;
          }
          tcCap = Math.round(tcCap);

          if (RECALC_FAMILY_CAPS[recalcFamily]) {
            const fc = RECALC_FAMILY_CAPS[recalcFamily];
            const familyCap = Math.round(fc.ambient + (fc.hp - fc.ambient) * pressureFactor);
            tcCap = Math.min(tcCap, familyCap);
          }

          let newTc = c.predictedTc;
          if (newTc != null && newTc > tcCap) {
            newTc = tcCap;
          }
          if (newTc != null) {
            newTc = applyAmbientTcCap(newTc, featureLambda, pressure, metalScore, c.formula);
          }

          const updatedFeatures = { ...features, physicsVersion: PHYSICS_VERSION };

          const isRoomTemp = (newTc ?? 0) >= 293 &&
            c.zeroResistance === true &&
            c.meissnerEffect === true &&
            (c.pressureGpa ?? 999) <= 50;

          await storage.updateSuperconductorCandidate(c.id, {
            predictedTc: newTc,
            mlFeatures: updatedFeatures as any,
            xgboostScore: gb.score,
            neuralNetScore: nnScore,
            ensembleScore: ensemble,
            electronPhononCoupling: features.electronPhononLambda ?? null,
            roomTempViable: isRoomTemp,
          });
          totalRecalculated++;
        } catch {}
      }

      if (totalRecalculated > 10000) break;
    }

    if (totalRecalculated > 0) {
      emit("log", {
        phase: "engine",
        event: "Physics recalculation complete",
        detail: `Recalculated ${totalRecalculated} candidates with corrected metallicity, lambda, and ambient-pressure Tc caps (v${PHYSICS_VERSION})`,
        dataSource: "Internal",
      });
    }
  } catch {}
}

export async function startEngine() {
  if (state === "running") return getStatus();
  state = "running";
  broadcast("engineState", { state: "running" });

  try {
    const maxCycle = await storage.getMaxConvergenceCycle();
    if (maxCycle > cycleCount) {
      cycleCount = maxCycle;
    }
  } catch {}

  await backfillGBScores();
  await recalculatePhysics();

  try {
    const dbCampaigns = await storage.getInverseDesignCampaigns();
    if (dbCampaigns.length > 0) {
      const restored = await restoreCampaignsFromDB(dbCampaigns as any);
      if (restored > 0) {
        emit("log", { phase: "inverse-optimizer", event: "Campaigns restored", detail: `${restored} inverse design campaigns restored from database` });
      }
    }
  } catch {}

  emit("log", {
    phase: "engine",
    event: "Learning engine started",
    detail: `Balanced learning: resuming from cycle ${cycleCount}. Materials/synthesis/reactions run first, then analysis, then SC research.`,
    dataSource: "Internal",
  });

  setTimeout(runLearningCycle, 2000);
  return getStatus();
}

export function stopEngine() {
  state = "stopped";
  if (cycleTimer) {
    clearTimeout(cycleTimer);
    cycleTimer = null;
  }
  broadcast("engineState", { state: "stopped" });

  emit("log", {
    phase: "engine",
    event: "Learning engine stopped",
    detail: `Completed ${cycleCount} cycles, ${totalScCandidates} SC candidates, ${totalSynthesisDiscovered} synthesis paths`,
    dataSource: "Internal",
  });

  return getStatus();
}

export function pauseEngine() {
  if (state !== "running") return getStatus();
  state = "paused";
  if (cycleTimer) {
    clearTimeout(cycleTimer);
    cycleTimer = null;
  }
  broadcast("engineState", { state: "paused" });
  return getStatus();
}

export function resumeEngine() {
  if (state !== "paused") return getStatus();
  state = "running";
  broadcast("engineState", { state: "running" });
  setTimeout(runLearningCycle, 2000);
  return getStatus();
}

export function getStatus(): EngineStatus {
  return {
    state,
    activeTasks: Array.from(activeTasks),
    cycleCount,
    lastCycleAt,
    totalMaterialsFetched,
    totalInsightsGenerated,
    totalPredictionsMade,
    totalSynthesisDiscovered,
    totalReactionsLearned,
    totalScCandidates,
    totalPhysicsComputed,
    totalStructuresPredicted,
    totalPipelineScreened,
    totalNovelSynthesisProposed,
    totalInverseDesigned: getInverseDesignCount(),
    totalStructuralVariants: getStructuralVariantCount(),
    tempo: engineTempo,
    statusMessage: currentStatusMessage,
  };
}
