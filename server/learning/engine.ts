import { WebSocketServer, WebSocket } from "ws";
import type { Server } from "http";
import { storage } from "../storage";
import { fetchOQMDMaterials, fetchElementFocusedMaterials, fetchKnownMaterials, getNextOQMDOffset } from "./data-fetcher";
import { analyzeBondingPatterns, analyzePropertyPredictionPatterns } from "./nlp-engine";
import { generateNovelFormulas } from "./formula-generator";
import { runSuperconductorResearch } from "./superconductor-research";
import { discoverSynthesisProcesses, discoverChemicalReactions, getNextReactionTopic } from "./synthesis-tracker";

export type EventEmitter = (type: string, data: any) => void;

function shouldContinue(): boolean {
  return state === "running";
}

type EngineState = "stopped" | "running" | "paused";

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
let activeTasks: Set<string> = new Set();
let lastCycleAt: string | null = null;
let allInsights: string[] = [];
let isRunningCycle = false;

function broadcast(type: string, data: any) {
  if (!wss) return;
  const msg = JSON.stringify({ type, data, timestamp: new Date().toISOString() });
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  });
}

const emit: EventEmitter = (type: string, data: any) => {
  broadcast(type, data);

  if (type === "log" && data.event && data.phase) {
    storage.insertResearchLog({
      phase: data.phase,
      event: data.event,
      detail: data.detail || null,
      dataSource: data.dataSource || null,
    }).catch(() => {});
  }
};

async function updatePhaseStatus(phaseId: number, status: string, progress: number, itemsLearned: number) {
  try {
    const phase = await storage.getLearningPhaseById(phaseId);
    if (!phase) return;

    const newProgress = Math.min(100, progress);
    const updates: any = { progress: newProgress, itemsLearned };
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

    broadcast("phaseUpdate", { phaseId, status, progress: newProgress, itemsLearned });
  } catch (e) {}
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
  } catch (e) {}
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

    const generated = await generateNovelFormulas(emit, allInsights.slice(-10));
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

    const result = await runSuperconductorResearch(emit, allInsights.slice(-15));
    totalScCandidates += result.generated;
    allInsights.push(...result.insights);
    totalInsightsGenerated += result.insights.length;

    await addInsightsToPhase(7, result.insights);
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

async function runLearningCycle() {
  if (state !== "running" || isRunningCycle) return;
  isRunningCycle = true;

  cycleCount++;
  lastCycleAt = new Date().toISOString();
  broadcast("cycleStart", { cycle: cycleCount });

  emit("log", {
    phase: "engine",
    event: `Learning cycle ${cycleCount} started`,
    detail: "Priority: materials + synthesis + reactions first, then bonding + prediction, then discovery. SC research runs last after knowledge is built.",
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
      cycleTimer = setTimeout(runLearningCycle, 60000);
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

export function startEngine() {
  if (state === "running") return getStatus();
  state = "running";
  broadcast("engineState", { state: "running" });

  emit("log", {
    phase: "engine",
    event: "Learning engine started",
    detail: "Balanced learning: materials/synthesis/reactions run first, then analysis, then SC research (only after sufficient knowledge base).",
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
  };
}
