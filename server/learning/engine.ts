import { WebSocketServer, WebSocket } from "ws";
import type { Server } from "http";
import { storage } from "../storage";
import { fetchOQMDMaterials, fetchElementFocusedMaterials, fetchKnownMaterials, getNextOQMDOffset } from "./data-fetcher";
import { analyzeBondingPatterns, analyzePropertyPredictionPatterns } from "./nlp-engine";
import { generateNovelFormulas } from "./formula-generator";
import { runSuperconductorResearch } from "./superconductor-research";
import { discoverSynthesisProcesses, discoverChemicalReactions, getNextReactionTopic } from "./synthesis-tracker";
import { runFullPhysicsAnalysis, applyAmbientTcCap } from "./physics-engine";
import { runStructurePredictionBatch } from "./structure-predictor";
import { runMultiFidelityPipeline } from "./multi-fidelity-pipeline";
import { evaluateInsightNovelty } from "./insight-detector";
import { analyzeAndEvolveStrategy, captureConvergenceSnapshot, trackDuplicatesSkipped } from "./strategy-analyzer";
import { checkMilestones } from "./milestone-tracker";
import { extractFeatures } from "./ml-predictor";
import { gbPredict } from "./gradient-boost";
import { resolveDFTFeatures, describeDFTSources } from "./dft-feature-resolver";
import type { DFTResolvedFeatures } from "./dft-feature-resolver";

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
let totalDFTEnriched = 0;
let currentStatusMessage = "Initializing research systems";
let recentTcImproved = false;
let recentNewCandidates = 0;

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
      cycleInsightsThisCycle++;
    }
  }
};

const PHASE_TARGETS: Record<number, number> = {
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

async function updatePhaseStatus(phaseId: number, status: string, progress: number, itemsLearned: number, totalItems?: number) {
  try {
    const phase = await storage.getLearningPhaseById(phaseId);
    if (!phase) return;

    const newProgress = Math.min(100, progress);
    const resolvedTotal = totalItems ?? PHASE_TARGETS[phaseId] ?? phase.totalItems;
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
    const combined = [...existing, ...newInsights].slice(-20);
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
    const mats = await storage.getMaterials(30, 0);
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
    const progress = Math.min(99, Math.floor((totalBondingInsights / 50) * 100));
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
    const progress = Math.min(99, Math.floor((matCount / 500) * 100));
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
    const mats = await storage.getMaterials(25, 0);
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
    const progress = Math.min(99, Math.floor((totalPredInsights / 50) * 100));
    await updatePhaseStatus(5, "active", progress, totalPredInsights);
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
    await updatePhaseStatus(6, "active", 0, 0);
    if (!shouldContinue()) return;

    const generated = await generateNovelFormulas(emit, allInsights.slice(-10), undefined, currentStrategyHint || undefined);
    totalPredictionsMade += generated;

    const predCount = (await storage.getNovelPredictions()).length;
    const progress = Math.min(99, Math.floor((predCount / 200) * 100));
    await updatePhaseStatus(6, "active", progress, predCount);
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
    phase7Offset += 50;
    totalScCandidates += result.generated;
    if (result.duplicatesSkipped > 0) {
      trackDuplicatesSkipped(result.duplicatesSkipped);
    }
    allInsights.push(...result.insights);
    totalInsightsGenerated += result.insights.length;

    await addInsightsToPhase(7, result.insights);
    await evaluateInsightNovelty(emit, result.insights, 7, "Superconductor Research");
    const scCount = await storage.getSuperconductorCount();
    const progress = Math.min(99, Math.floor((scCount / 500) * 100));
    await updatePhaseStatus(7, "active", progress, scCount);
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
    const progress = Math.min(99, Math.floor((synthCount / 300) * 100));
    await updatePhaseStatus(8, "active", progress, synthCount);
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
    const progress = Math.min(99, Math.floor((rxnCount / 300) * 100));
    await updatePhaseStatus(9, "active", progress, rxnCount);
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

const reEvalApplied = new Map<string, { lambda: number; omegaLog: number; muStar: number; hasCrystal: boolean }>();
let cyclesSinceTcImproved = 0;
let lastBestTcSeen = 0;

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
    if (currentBestTc > lastBestTcSeen + 2) {
      cyclesSinceTcImproved = 0;
      lastBestTcSeen = currentBestTc;
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
      newTc = applyAmbientTcCap(newTc, lambda, candidate.pressureGpa ?? 0, features.metallicity ?? 0.5);

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

async function runDFTEnrichment() {
  if (!shouldContinue()) return;
  try {
    const candidates = await storage.getSuperconductorCandidates(50);
    const sorted = candidates
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));

    const toEnrich: typeof candidates = [];
    for (const c of sorted) {
      if (toEnrich.length >= 5) break;
      if (c.dataConfidence === "high") continue;
      toEnrich.push(c);
    }

    if (toEnrich.length === 0) return;

    broadcastThought(
      `Cross-referencing top ${toEnrich.length} candidates against DFT databases...`,
      "strategy"
    );

    let enriched = 0;
    for (const candidate of toEnrich) {
      if (!shouldContinue()) break;
      try {
        const dftData = await resolveDFTFeatures(candidate.formula);
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
        detail: `Enriched ${enriched} candidates with DFT data (${totalDFTEnriched} total)`,
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
        updatedTc = applyAmbientTcCap(updatedTc, result.coupling.lambda, candidate.pressureGpa ?? 0, result.electronicStructure.metallicity ?? 0.5);

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
          pairingMechanism: result.correlation.ratio > 0.6 ? "unconventional" : "phonon-mediated",
          cooperPairMechanism: candidate.cooperPairMechanism ?? (result.correlation.ratio > 0.6
            ? `Unconventional pairing via spin-fluctuation exchange (U/W=${result.correlation.ratio.toFixed(2)})`
            : `Phonon-mediated BCS pairing with lambda=${result.coupling.lambda.toFixed(2)}`),
          predictedTc: updatedTc,
          verificationStage: 1,
        });

        if (updatedTc !== currentTc) {
          emit("log", { phase: "phase-10", event: "Tc updated by physics", detail: `${candidate.formula}: ML estimate ${currentTc}K -> Eliashberg ${updatedTc}K (lambda=${result.coupling.lambda.toFixed(2)}, Hc2=${result.criticalFields.upperCriticalField}T, ${result.correlation.regime}, ${result.competingPhases.length} competing phases)`, dataSource: "Physics Engine" });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-10", event: "Physics analysis error", detail: `${candidate.formula}: ${err.message?.slice(0, 150)}`, dataSource: "Physics Engine" });
      }
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
            updatedTc = applyAmbientTcCap(updatedTc, newLambda, candidate.pressureGpa ?? 0, result.electronicStructure.metallicity ?? 0.5);
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
    const progress = Math.min(99, Math.floor((crCount / 200) * 100));
    await updatePhaseStatus(10, "active", progress, crCount);
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

    const csCount = await storage.getCrystalStructureCount();
    const progress = Math.min(99, Math.floor((csCount / 150) * 100));
    await updatePhaseStatus(11, "active", progress, csCount);
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
    const progress = Math.min(99, Math.floor((crCount / 300) * 100));
    await updatePhaseStatus(12, "active", progress, crCount);
  } finally {
    activeTasks.delete("Multi-Fidelity Screening");
    broadcast("taskEnd", { task: "Multi-Fidelity Screening" });
  }
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
        broadcastThought(
          `No Tc improvement in ${cyclesSinceTcImproved} cycles. Current best: ${Math.round(previousCycleMetrics.bestTc)}K. Re-examining high-lambda candidates with stricter physics constraints...`,
          "stagnation"
        );
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

const PHYSICS_VERSION = 6;

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

          let newTc = c.predictedTc;
          if (newTc != null && newTc > tcCap) {
            newTc = tcCap;
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
    tempo: engineTempo,
    statusMessage: currentStatusMessage,
  };
}
