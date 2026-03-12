import {
  getComprehensiveModelDiagnostics,
  getModelDiagnosticsForLLM,
  getModelHealthSummary,
  getErrorAnalysis,
  getFeatureImportanceReport,
  recordPredictionOutcome,
  type ComprehensiveModelDiagnostics,
} from "./model-diagnostics";
import {
  proposeModelExperiments,
  executeExperiment,
  getExperimentHistory,
  getHyperparamOverrides,
  setHyperparamOverrides,
  getPendingDataRequests,
  getActiveTechnicalRequirements,
  type ExperimentProposal,
  type ExperimentRecord,
  type TechnicalRequirement,
} from "./model-experiment-controller";
import {
  runModelLLMCycle,
  getCurrentArchitecture,
  getActiveCustomFeatures,
} from "./model-llm-controller";
import { getUncertaintyForLLM, getVarianceSummary, getFrontierUncertainty } from "./uncertainty-tracker";
import { evaluateRetrainNeed, recordRetrainOutcome, getSchedulerForLLM, getSchedulerStats } from "./retrain-scheduler";
import { fetchOQMDMaterials, fetchElementFocusedMaterials, fetchKnownMaterials, getNextOQMDOffset } from "./data-fetcher";
import type { EventEmitter } from "./engine";

interface MetricTrend {
  model: string;
  metric: string;
  values: { cycle: number; value: number }[];
}

interface ImprovementStats {
  totalCycles: number;
  experimentsRun: number;
  improvementsAchieved: number;
  rollbacksPerformed: number;
  emergencyDataFetches: number;
  currentModelHealth: { model: string; status: string; reasons: string[] }[];
  trends: MetricTrend[];
  lastCycleAt: number | null;
  plateauModels: string[];
}

interface CooldownEntry {
  model: string;
  lastExperimentCycle: number;
}

const CYCLE_FREQUENCY = 5;
const COOLDOWN_CYCLES = 3;
const MAX_LLM_CALLS_PER_CYCLE = 3;
const ROLLBACK_THRESHOLD = 0.10;
const PLATEAU_WINDOW = 5;
const PLATEAU_THRESHOLD = 0.01;
const CONSECUTIVE_NO_IMPROVEMENT_EXPLORE_THRESHOLD = 5;
const LOW_DATASET_THRESHOLD = 200;
const HIGH_UNCERTAINTY_FRACTION_THRESHOLD = 0.25;

let totalImprovementCycles = 0;
let totalExperimentsRun = 0;
let totalImprovementsAchieved = 0;
let totalRollbacksPerformed = 0;
let lastImprovementCycleAt: number | null = null;
let metricTrends: MetricTrend[] = [];
let cooldowns: CooldownEntry[] = [];
let plateauModels: string[] = [];
let consecutiveNoImprovement: Map<string, number> = new Map();
let pendingImprovementTask: Promise<{ ran: boolean; experiment?: ExperimentRecord; improvement?: Record<string, number> }> | null = null;
let consecutivePhononMissing = 0;
let consecutiveLambdaMissing = 0;
const PHYSICS_FEATURE_WARNING_THRESHOLD = 5;

const NEGATIVE_R2_FETCH_LIMIT = 200;
const NEGATIVE_R2_DATASET_CAP = 1000;
const NEGATIVE_R2_COOLDOWN_CYCLES = 8;
let lastNegativeR2FetchCycle = -999;
let emergencyFetchCount = 0;

function isOnCooldown(model: string, currentCycle: number): boolean {
  const entry = cooldowns.find(c => c.model === model);
  if (!entry) return false;
  return (currentCycle - entry.lastExperimentCycle) < COOLDOWN_CYCLES;
}

function setCooldown(model: string, currentCycle: number): void {
  const existing = cooldowns.find(c => c.model === model);
  if (existing) {
    existing.lastExperimentCycle = currentCycle;
  } else {
    cooldowns.push({ model, lastExperimentCycle: currentCycle });
  }
}

function recordMetricTrend(model: string, metric: string, cycle: number, value: number): void {
  let trend = metricTrends.find(t => t.model === model && t.metric === metric);
  if (!trend) {
    trend = { model, metric, values: [] };
    metricTrends.push(trend);
    const indexed = trendIndex.get(model);
    if (indexed) {
      indexed.push(trend);
    } else {
      trendIndex.set(model, [trend]);
    }
  }
  trend.values.push({ cycle, value });
  if (trend.values.length > 50) {
    trend.values = trend.values.slice(-50);
  }
}

function coefficientOfVariation(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1);
  const sd = Math.sqrt(variance);
  const scale = Math.max(Math.abs(mean), 0.1);
  return sd / scale;
}

let trendIndex: Map<string, MetricTrend[]> = new Map();

function rebuildTrendIndex(): void {
  trendIndex.clear();
  for (const t of metricTrends) {
    const existing = trendIndex.get(t.model);
    if (existing) {
      existing.push(t);
    } else {
      trendIndex.set(t.model, [t]);
    }
  }
}

function detectPlateau(model: string): boolean {
  const modelTrends = trendIndex.get(model);
  if (!modelTrends || modelTrends.length === 0) return false;

  let plateauCount = 0;
  let checkedCount = 0;

  for (const trend of modelTrends) {
    if (trend.values.length < PLATEAU_WINDOW) continue;
    checkedCount++;
    const recent = trend.values.slice(-PLATEAU_WINDOW).map(v => v.value);
    const cv = coefficientOfVariation(recent);
    if (cv < PLATEAU_THRESHOLD) {
      plateauCount++;
    }
  }

  return checkedCount > 0 && plateauCount >= checkedCount;
}

function truncateAtWordBoundary(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  const cut = text.lastIndexOf(" ", maxLen);
  return (cut > maxLen * 0.6 ? text.slice(0, cut) : text.slice(0, maxLen)) + "...";
}

function safeString(value: unknown, fallback: string = "unavailable"): string {
  if (value === undefined || value === null) return fallback;
  return String(value);
}

function snapshotKeyMetrics(diagnostics: ComprehensiveModelDiagnostics): Record<string, Record<string, number>> {
  return {
    xgboost: { r2: diagnostics.xgboost.r2, mae: diagnostics.xgboost.mae, rmse: diagnostics.xgboost.rmse },
    gnn: { r2: diagnostics.gnn.latestR2, mae: diagnostics.gnn.latestMAE, rmse: diagnostics.gnn.latestRMSE },
    "lambda-regressor": { r2: diagnostics.lambda.r2, mae: diagnostics.lambda.mae, rmse: diagnostics.lambda.rmse },
    "phonon-surrogate": { omegaLogMAE: diagnostics.phononSurrogate.omegaLogMAE, stabilityAccuracy: diagnostics.phononSurrogate.stabilityAccuracy },
    "tb-surrogate": { datasetSize: diagnostics.tbSurrogate.datasetSize, predictions: diagnostics.tbSurrogate.predictions },
  };
}

const METRIC_WEIGHTS: Record<string, number> = {
  stabilityAccuracy: 2.0,
  r2: 1.0,
  mae: 1.5,
  omegaLogMAE: 1.5,
  debyeTempMAE: 1.0,
  maxFreqMAE: 1.0,
  rmse: 0.5,
};

const STABILITY_RELAXED_TC_THRESHOLD = 0.20;

function metricsWorsened(before: Record<string, number>, after: Record<string, number>): boolean {
  const higherIsBetter = ["r2", "stabilityAccuracy"];
  const lowerIsBetter = ["mae", "rmse", "omegaLogMAE", "debyeTempMAE", "maxFreqMAE"];

  let worseWeight = 0;
  let totalWeight = 0;

  const stabilityImproved = before["stabilityAccuracy"] != null && after["stabilityAccuracy"] != null
    && after["stabilityAccuracy"] > before["stabilityAccuracy"] + 0.02;

  const tcThreshold = stabilityImproved ? STABILITY_RELAXED_TC_THRESHOLD : ROLLBACK_THRESHOLD;

  for (const key of Object.keys(before)) {
    if (after[key] == null) continue;
    const weight = METRIC_WEIGHTS[key] ?? 1.0;
    totalWeight += weight;

    const isStabilityMetric = key === "stabilityAccuracy";
    const threshold = isStabilityMetric ? ROLLBACK_THRESHOLD : tcThreshold;

    if (higherIsBetter.includes(key)) {
      const delta = before[key] - after[key];
      const scale = Math.max(Math.abs(before[key]), 0.1);
      if (delta / scale > threshold) {
        worseWeight += weight;
      }
    } else if (lowerIsBetter.includes(key)) {
      const baseline = Math.max(before[key], 0.01);
      if (after[key] > baseline * (1 + threshold)) {
        worseWeight += weight;
      }
    }
  }

  return totalWeight > 0 && worseWeight > totalWeight / 2;
}

function needsExplorationReset(model: string): boolean {
  return (consecutiveNoImprovement.get(model) ?? 0) >= CONSECUTIVE_NO_IMPROVEMENT_EXPLORE_THRESHOLD;
}

function checkPhysicsFeatureWarnings(topFeatures: { name: string; normalizedImportance: number }[]): {
  phononMissing: boolean;
  lambdaMissing: boolean;
  shouldAutoTrigger: boolean;
  missingType: "phonon" | "lambda" | "both" | null;
} {
  const hasPhonon = topFeatures.some(f => /phonon|omegaLog|debye/i.test(f.name));
  const hasLambda = topFeatures.some(f => /lambda|coupling/i.test(f.name));

  if (hasPhonon) consecutivePhononMissing = 0; else consecutivePhononMissing++;
  if (hasLambda) consecutiveLambdaMissing = 0; else consecutiveLambdaMissing++;

  const phononTrigger = consecutivePhononMissing >= PHYSICS_FEATURE_WARNING_THRESHOLD;
  const lambdaTrigger = consecutiveLambdaMissing >= PHYSICS_FEATURE_WARNING_THRESHOLD;

  let missingType: "phonon" | "lambda" | "both" | null = null;
  if (phononTrigger && lambdaTrigger) missingType = "both";
  else if (phononTrigger) missingType = "phonon";
  else if (lambdaTrigger) missingType = "lambda";

  return {
    phononMissing: !hasPhonon,
    lambdaMissing: !hasLambda,
    shouldAutoTrigger: phononTrigger || lambdaTrigger,
    missingType,
  };
}

function buildPhysicsFeatureProposal(missingType: "phonon" | "lambda" | "both"): ExperimentProposal {
  const featureNames: string[] = [];
  const reasoning: string[] = [];

  if (missingType === "phonon" || missingType === "both") {
    featureNames.push("debyeTemperature", "omegaLogEstimate", "phononDOS_peak");
    reasoning.push(`phonon features absent from top-10 for ${consecutivePhononMissing} cycles`);
  }
  if (missingType === "lambda" || missingType === "both") {
    featureNames.push("lambdaEstimate", "ePh_coupling_proxy", "dosAtFermi");
    reasoning.push(`e-ph coupling features absent from top-10 for ${consecutiveLambdaMissing} cycles`);
  }

  return {
    model_target: "xgboost",
    experiment_type: "add_features",
    changes: {
      features: featureNames,
      reason: "stoichiometry_mining_guard",
      force_inclusion: true,
    },
    reasoning: `Possible stoichiometry mining: ${reasoning.join("; ")} — forcing electronic structure descriptors`,
    expected_improvement: "model learns physical mechanism instead of elemental correlations",
    priority: 1,
  };
}

function filterForExploration(proposals: ExperimentProposal[]): ExperimentProposal[] {
  const explorationTypes: Set<string> = new Set(["add_features", "request_data", "expand_dataset", "rebalance_training"]);
  const stuckModels = new Set<string>();
  consecutiveNoImprovement.forEach((count, model) => {
    if (count >= CONSECUTIVE_NO_IMPROVEMENT_EXPLORE_THRESHOLD) stuckModels.add(model);
  });
  if (stuckModels.size === 0) return proposals;

  const explorationProposals = proposals.filter(
    p => stuckModels.has(p.model_target) ? explorationTypes.has(p.experiment_type) : true
  );
  if (explorationProposals.length > 0) return explorationProposals;
  return proposals;
}

interface NegativeR2Check {
  triggered: boolean;
  models: { name: string; r2: number; datasetSize: number }[];
  totalDatasetSize: number;
}

function detectNegativeR2(diagnostics: ComprehensiveModelDiagnostics): NegativeR2Check {
  const negativeModels: { name: string; r2: number; datasetSize: number }[] = [];

  if (diagnostics.xgboost.r2 < 0) {
    negativeModels.push({ name: "xgboost", r2: diagnostics.xgboost.r2, datasetSize: diagnostics.xgboost.datasetSize });
  }
  if (diagnostics.gnn.latestR2 < 0) {
    negativeModels.push({ name: "gnn", r2: diagnostics.gnn.latestR2, datasetSize: diagnostics.gnn.datasetSize });
  }
  if (diagnostics.lambda.r2 < 0) {
    negativeModels.push({ name: "lambda-regressor", r2: diagnostics.lambda.r2, datasetSize: diagnostics.lambda.datasetSize });
  }

  const totalDatasetSize = negativeModels.length > 0
    ? Math.min(...negativeModels.map(m => m.datasetSize))
    : Math.max(diagnostics.xgboost.datasetSize, diagnostics.gnn.datasetSize, diagnostics.lambda.datasetSize);

  return {
    triggered: negativeModels.length > 0,
    models: negativeModels,
    totalDatasetSize,
  };
}

async function runEmergencyDataFetch(
  emit: EventEmitter,
  negR2: NegativeR2Check,
  currentCycle: number
): Promise<boolean> {
  if (!negR2.triggered) return false;
  if (negR2.totalDatasetSize >= NEGATIVE_R2_DATASET_CAP) {
    emit("log", {
      phase: "engine",
      event: "Negative R² detected but dataset already large",
      detail: `Models with R²<0: ${negR2.models.map(m => `${m.name}(R²=${m.r2.toFixed(3)})`).join(", ")}. Dataset=${negR2.totalDatasetSize} (cap=${NEGATIVE_R2_DATASET_CAP}). Skipping emergency fetch — issue is likely model architecture, not data scarcity.`,
      dataSource: "Model Improvement Loop",
    });
    return false;
  }
  if (currentCycle - lastNegativeR2FetchCycle < NEGATIVE_R2_COOLDOWN_CYCLES) {
    emit("log", {
      phase: "engine",
      event: "Negative R² emergency fetch on cooldown",
      detail: `Last fetch was cycle ${lastNegativeR2FetchCycle}, cooldown=${NEGATIVE_R2_COOLDOWN_CYCLES} cycles. Current=${currentCycle}.`,
      dataSource: "Model Improvement Loop",
    });
    return false;
  }

  const modelSummary = negR2.models.map(m => `${m.name}(R²=${m.r2.toFixed(3)}, n=${m.datasetSize})`).join(", ");
  emit("log", {
    phase: "engine",
    event: "Negative R² emergency data fetch triggered",
    detail: `[Improvement-Loop] Low R² detected: ${modelSummary}. Triggering emergency data fetch (target: ${NEGATIVE_R2_FETCH_LIMIT} new materials).`,
    dataSource: "Model Improvement Loop",
  });

  let totalFetched = 0;

  try {
    const oqmdOffset = getNextOQMDOffset();
    const oqmdCount = await fetchOQMDMaterials(emit, NEGATIVE_R2_FETCH_LIMIT, oqmdOffset);
    totalFetched += oqmdCount;
    if (oqmdCount > 0) {
      emit("log", {
        phase: "engine",
        event: "Emergency fetch: OQMD batch complete",
        detail: `Fetched ${oqmdCount} DFT-computed materials from OQMD (offset=${oqmdOffset})`,
        dataSource: "Model Improvement Loop",
      });
    }
  } catch (e) {
    emit("log", {
      phase: "engine",
      event: "Emergency fetch: OQMD failed",
      detail: e instanceof Error ? e.message : "unknown error",
      dataSource: "Model Improvement Loop",
    });
  }

  try {
    const elementCount = await fetchElementFocusedMaterials(emit);
    totalFetched += elementCount;
  } catch (e) {
    emit("log", {
      phase: "engine",
      event: "Emergency fetch: element-focused failed",
      detail: e instanceof Error ? e.message : "unknown error",
      dataSource: "Model Improvement Loop",
    });
  }

  try {
    const knownCount = await fetchKnownMaterials(emit);
    totalFetched += knownCount;
  } catch (e) {
    emit("log", {
      phase: "engine",
      event: "Emergency fetch: known materials failed",
      detail: e instanceof Error ? e.message : "unknown error",
      dataSource: "Model Improvement Loop",
    });
  }

  if (totalFetched > 0) {
    lastNegativeR2FetchCycle = currentCycle;
    emergencyFetchCount++;
    emit("log", {
      phase: "engine",
      event: `Emergency data fetch complete`,
      detail: `[Improvement-Loop] Dataset expanded: ${negR2.totalDatasetSize} -> ${negR2.totalDatasetSize + totalFetched} (fetched ${totalFetched} new materials, fetch #${emergencyFetchCount}). DFT validation will auto-process via qe-worker.`,
      dataSource: "Model Improvement Loop",
    });
  } else {
    emit("log", {
      phase: "engine",
      event: "Emergency data fetch: no materials fetched",
      detail: "All data sources returned 0 materials. Will retry next eligible cycle without cooldown.",
      dataSource: "Model Improvement Loop",
    });
  }

  return totalFetched > 0;
}

function shouldRunGreenExpansion(diagnostics: ComprehensiveModelDiagnostics): string | null {
  const variance = getVarianceSummary();
  if (variance.highUncertaintyFraction < HIGH_UNCERTAINTY_FRACTION_THRESHOLD) return null;

  const smallModels: { model: string; size: number }[] = [];
  if (diagnostics.xgboost.datasetSize < LOW_DATASET_THRESHOLD) smallModels.push({ model: "xgboost", size: diagnostics.xgboost.datasetSize });
  if (diagnostics.gnn.datasetSize < LOW_DATASET_THRESHOLD) smallModels.push({ model: "gnn", size: diagnostics.gnn.datasetSize });
  if (diagnostics.lambda.datasetSize < LOW_DATASET_THRESHOLD) smallModels.push({ model: "lambda-regressor", size: diagnostics.lambda.datasetSize });

  if (smallModels.length === 0) return null;
  smallModels.sort((a, b) => a.size - b.size);
  return smallModels[0].model;
}

async function runImprovementCycleInner(
  emit: EventEmitter,
  currentCycle: number
): Promise<{ ran: boolean; experiment?: ExperimentRecord; improvement?: Record<string, number> }> {
  totalImprovementCycles++;
  lastImprovementCycleAt = Date.now();

  const diagnosticsBefore = getComprehensiveModelDiagnostics();
  const health = getModelHealthSummary();
  const beforeKeyMetrics = snapshotKeyMetrics(diagnosticsBefore);

  const negR2Check = detectNegativeR2(diagnosticsBefore);
  if (negR2Check.triggered) {
    await runEmergencyDataFetch(emit, negR2Check, currentCycle);
  }

  const topFeatures = getFeatureImportanceReport(10);
  let physicsAutoTriggered = false;
  if (topFeatures.length > 0) {
    const physicsCheck = checkPhysicsFeatureWarnings(topFeatures);
    if (physicsCheck.shouldAutoTrigger && physicsCheck.missingType) {
      const physicsProposal = buildPhysicsFeatureProposal(physicsCheck.missingType);
      emit("log", {
        phase: "engine",
        event: "Model improvement: stoichiometry mining guard",
        detail: truncateAtWordBoundary(physicsProposal.reasoning, 200),
        dataSource: "Model Improvement Loop",
      });
      try {
        const record = await executeExperiment(physicsProposal);
        totalExperimentsRun++;
        physicsAutoTriggered = true;
        if (physicsCheck.missingType === "phonon" || physicsCheck.missingType === "both") consecutivePhononMissing = 0;
        if (physicsCheck.missingType === "lambda" || physicsCheck.missingType === "both") consecutiveLambdaMissing = 0;
        for (const [model, metrics] of Object.entries(beforeKeyMetrics)) {
          for (const [metric, value] of Object.entries(metrics)) {
            recordMetricTrend(model, metric, currentCycle, value);
          }
        }
        return { ran: true, experiment: record, improvement: record.improvement };
      } catch (e) {
        emit("log", {
          phase: "engine",
          event: "Model improvement: physics feature injection failed",
          detail: truncateAtWordBoundary(e instanceof Error ? e.message : "unknown", 200),
          dataSource: "Model Improvement Loop",
        });
      }
    }
  }

  const allGreen = health.every(h => h.status === "green");
  if (allGreen) {
    const expansionTarget = shouldRunGreenExpansion(diagnosticsBefore);
    if (expansionTarget) {
      const variance = getVarianceSummary();
      emit("log", {
        phase: "engine",
        event: "Model improvement: green expansion",
        detail: `All models green but high uncertainty (${(variance.highUncertaintyFraction * 100).toFixed(1)}%) — requesting data for ${expansionTarget}`,
        dataSource: "Model Improvement Loop",
      });
      const expansionProposal: ExperimentProposal = {
        model_target: expansionTarget,
        experiment_type: "expand_dataset",
        changes: { reason: "high_uncertainty_frontier", target_dataset_growth: 50 },
        reasoning: `Models healthy but ${(variance.highUncertaintyFraction * 100).toFixed(0)}% of predictions have high uncertainty — expand training data`,
        expected_improvement: "reduced uncertainty in frontier regions",
        priority: 3,
      };
      try {
        const record = await executeExperiment(expansionProposal);
        totalExperimentsRun++;
        for (const [model, metrics] of Object.entries(beforeKeyMetrics)) {
          for (const [metric, value] of Object.entries(metrics)) {
            recordMetricTrend(model, metric, currentCycle, value);
          }
        }
        return { ran: true, experiment: record, improvement: record.improvement };
      } catch (_e) {}
    }

    emit("log", {
      phase: "engine",
      event: "Model improvement skipped",
      detail: "All models healthy (green status) — no improvement needed",
      dataSource: "Model Improvement Loop",
    });

    for (const [model, metrics] of Object.entries(beforeKeyMetrics)) {
      for (const [metric, value] of Object.entries(metrics)) {
        recordMetricTrend(model, metric, currentCycle, value);
      }
    }

    return { ran: false };
  }

  const unhealthyModels = health
    .filter(h => h.status === "red" || h.status === "yellow")
    .filter(h => !isOnCooldown(h.model, currentCycle))
    .sort((a, b) => {
      if (a.status === "red" && b.status !== "red") return -1;
      if (a.status !== "red" && b.status === "red") return 1;
      return 0;
    });

  if (unhealthyModels.length === 0) {
    emit("log", {
      phase: "engine",
      event: "Model improvement deferred",
      detail: "Unhealthy models are on cooldown",
      dataSource: "Model Improvement Loop",
    });
    return { ran: false };
  }

  const report = getModelDiagnosticsForLLM();

  let proposals: ExperimentProposal[] = [];
  try {
    proposals = await proposeModelExperiments(report);
  } catch (e) {
    emit("log", {
      phase: "engine",
      event: "Model improvement LLM error",
      detail: truncateAtWordBoundary(e instanceof Error ? e.message : "unknown", 200),
      dataSource: "Model Improvement Loop",
    });
    return { ran: true };
  }

  if (proposals.length === 0) {
    emit("log", {
      phase: "engine",
      event: "Model improvement: no proposals",
      detail: "LLM did not propose any experiments",
      dataSource: "Model Improvement Loop",
    });
    return { ran: true };
  }

  proposals = filterForExploration(proposals);

  const eligibleProposals = proposals
    .filter(p => !isOnCooldown(p.model_target, currentCycle))
    .sort((a, b) => a.priority - b.priority);

  if (eligibleProposals.length === 0) {
    emit("log", {
      phase: "engine",
      event: "Model improvement: all targets on cooldown",
      detail: `Proposed targets: ${proposals.map(p => p.model_target).join(", ")}`,
      dataSource: "Model Improvement Loop",
    });
    return { ran: true };
  }

  const topExperiment = eligibleProposals[0];

  const allTechReqs = eligibleProposals.flatMap(p => p.technicalRequirements ?? []);
  if (allTechReqs.length > 0) {
    emit("log", {
      phase: "engine",
      event: "Model improvement: technical requirements for strategy",
      detail: allTechReqs.map(r => `${r.urgency.toUpperCase()}: ${r.detail}`).join("; ").slice(0, 200),
      dataSource: "Model Improvement Loop",
    });
  }

  if (needsExplorationReset(topExperiment.model_target)) {
    emit("log", {
      phase: "engine",
      event: "Model improvement: exploration reset",
      detail: `${topExperiment.model_target} has ${consecutiveNoImprovement.get(topExperiment.model_target)} consecutive no-improvement cycles — forcing exploration`,
      dataSource: "Model Improvement Loop",
    });
  }

  const preExperimentOverrides = getHyperparamOverrides(topExperiment.model_target);
  const overridesSnapshot = preExperimentOverrides ? JSON.parse(JSON.stringify(preExperimentOverrides)) : null;

  const beforeModelMetrics = beforeKeyMetrics[topExperiment.model_target] ?? {};

  emit("log", {
    phase: "engine",
    event: "Model improvement: executing experiment",
    detail: `${topExperiment.experiment_type} on ${topExperiment.model_target}: ${truncateAtWordBoundary(topExperiment.reasoning, 150)}`,
    dataSource: "Model Improvement Loop",
  });

  let record: ExperimentRecord;
  try {
    record = await executeExperiment(topExperiment);
    totalExperimentsRun++;
  } catch (e) {
    emit("log", {
      phase: "engine",
      event: "Model improvement: experiment failed",
      detail: truncateAtWordBoundary(e instanceof Error ? e.message : "unknown", 200),
      dataSource: "Model Improvement Loop",
    });
    return { ran: true };
  }

  setCooldown(topExperiment.model_target, currentCycle);

  const afterModelMetrics = record.after_metrics;

  if (metricsWorsened(beforeModelMetrics, afterModelMetrics)) {
    totalRollbacksPerformed++;
    emit("log", {
      phase: "engine",
      event: "Model improvement: ROLLBACK",
      detail: `${topExperiment.model_target} metrics worsened >10% — reverting changes`,
      dataSource: "Model Improvement Loop",
    });

    try {
      if (topExperiment.experiment_type === "adjust_hyperparameters" && overridesSnapshot != null) {
        setHyperparamOverrides(topExperiment.model_target, overridesSnapshot);
        console.log(`[Model Improvement Loop] Rolled back ${topExperiment.model_target} overrides to pre-experiment state`);
      }
    } catch (_e) {}

    const noImpCount = (consecutiveNoImprovement.get(topExperiment.model_target) ?? 0) + 1;
    consecutiveNoImprovement.set(topExperiment.model_target, noImpCount);
  } else {
    let improved = false;
    const improvementDetails: string[] = [];

    for (const key of Object.keys(beforeModelMetrics)) {
      if (afterModelMetrics[key] == null) continue;
      const before = beforeModelMetrics[key];
      const after = afterModelMetrics[key];
      const delta = after - before;

      if (["r2", "stabilityAccuracy"].includes(key) && delta > 0.001) {
        improved = true;
        improvementDetails.push(`${key}: ${before.toFixed(4)} -> ${after.toFixed(4)}`);
      } else if (["mae", "rmse", "omegaLogMAE"].includes(key) && delta < -0.001) {
        improved = true;
        improvementDetails.push(`${key}: ${before.toFixed(4)} -> ${after.toFixed(4)}`);
      }
    }

    if (improved) {
      totalImprovementsAchieved++;
      consecutiveNoImprovement.set(topExperiment.model_target, 0);
      emit("log", {
        phase: "engine",
        event: `Model improvement: ${topExperiment.model_target} improved`,
        detail: improvementDetails.join(", "),
        dataSource: "Model Improvement Loop",
      });
    } else {
      const noImpCount = (consecutiveNoImprovement.get(topExperiment.model_target) ?? 0) + 1;
      consecutiveNoImprovement.set(topExperiment.model_target, noImpCount);
    }
  }

  const updatedKeyMetrics = { ...beforeKeyMetrics };
  if (record.status === "completed" && Object.keys(afterModelMetrics).length > 0) {
    updatedKeyMetrics[topExperiment.model_target] = afterModelMetrics;
  }
  for (const [model, metrics] of Object.entries(updatedKeyMetrics)) {
    for (const [metric, value] of Object.entries(metrics)) {
      recordMetricTrend(model, metric, currentCycle, value);
    }
  }

  plateauModels = [];
  for (const model of ["xgboost", "gnn", "lambda-regressor", "phonon-surrogate", "tb-surrogate"]) {
    if (detectPlateau(model)) {
      plateauModels.push(model);
    }
  }

  return {
    ran: true,
    experiment: record,
    improvement: record.improvement,
  };
}

export async function runModelImprovementCycle(
  emit: EventEmitter,
  currentCycle: number
): Promise<{ ran: boolean; experiment?: ExperimentRecord; improvement?: Record<string, number> }> {
  if (currentCycle % CYCLE_FREQUENCY !== 0) {
    return { ran: false };
  }

  if (pendingImprovementTask != null) {
    emit("log", {
      phase: "engine",
      event: "Model improvement: previous cycle still running",
      detail: "Skipping to avoid blocking main loop",
      dataSource: "Model Improvement Loop",
    });
    return { ran: false };
  }

  pendingImprovementTask = runImprovementCycleInner(emit, currentCycle);
  try {
    const result = await pendingImprovementTask;
    return result;
  } finally {
    pendingImprovementTask = null;
  }
}

export function getModelImprovementStats(): ImprovementStats {
  const health = getModelHealthSummary();

  return {
    totalCycles: totalImprovementCycles,
    experimentsRun: totalExperimentsRun,
    improvementsAchieved: totalImprovementsAchieved,
    rollbacksPerformed: totalRollbacksPerformed,
    emergencyDataFetches: emergencyFetchCount,
    currentModelHealth: health.map(h => ({ model: h.model, status: h.status, reasons: h.reasons })),
    trends: metricTrends,
    lastCycleAt: lastImprovementCycleAt,
    plateauModels,
  };
}

export function getModelImprovementTrends(): MetricTrend[] {
  return metricTrends;
}

export function shouldRunModelImprovement(currentCycle: number): boolean {
  return currentCycle % CYCLE_FREQUENCY === 0;
}

export function recordCyclePredictionOutcomes(
  outcomes: { formula: string; predicted: number; actual: number; model?: string }[]
): void {
  for (const o of outcomes) {
    recordPredictionOutcome(o.model || "xgboost", o.formula, o.predicted, o.actual);
  }
}

export function getModelDiagnosticsSummaryForStrategy(): string {
  const health = getModelHealthSummary();
  const unhealthy = health.filter(h => h.status !== "green");

  const lines: string[] = ["\n## ML Model Health Summary"];

  if (unhealthy.length === 0) {
    lines.push("[All models green — no critical issues]");
  } else {
    for (const h of unhealthy) {
      lines.push(`  ${h.model}: ${h.status.toUpperCase()} — ${h.reasons.join("; ")}`);
    }
  }

  const diagnostics = getComprehensiveModelDiagnostics();
  lines.push(`  XGBoost: R²=${diagnostics.xgboost.r2}, MAE=${diagnostics.xgboost.mae}K, RMSE=${diagnostics.xgboost.rmse}K`);
  lines.push(`  GNN: R²=${diagnostics.gnn.latestR2}, MAE=${diagnostics.gnn.latestMAE}K`);
  lines.push(`  Lambda: R²=${diagnostics.lambda.r2}, MAE=${diagnostics.lambda.mae}`);

  const topFeatures = getFeatureImportanceReport(10);
  if (topFeatures.length > 0) {
    lines.push("  Top features: " + topFeatures.slice(0, 5).map(f => `${f.name}(${f.normalizedImportance})`).join(", "));
    const hasPhonon = topFeatures.some(f => /phonon|omegaLog|debye/i.test(f.name));
    const hasLambda = topFeatures.some(f => /lambda|coupling/i.test(f.name));
    if (!hasPhonon) {
      lines.push(`  ** Model ignores phonon features (${consecutivePhononMissing} consecutive cycles) — ${consecutivePhononMissing >= PHYSICS_FEATURE_WARNING_THRESHOLD ? "AUTO-TRIGGER pending" : "consider adding Debye temperature emphasis"} **`);
    }
    if (!hasLambda) {
      lines.push(`  ** Model ignores e-ph coupling (${consecutiveLambdaMissing} consecutive cycles) — ${consecutiveLambdaMissing >= PHYSICS_FEATURE_WARNING_THRESHOLD ? "AUTO-TRIGGER pending" : "lambda features underused"} **`);
    }
  }

  const negR2Models: string[] = [];
  if (diagnostics.xgboost.r2 < 0) negR2Models.push(`XGBoost(R²=${diagnostics.xgboost.r2})`);
  if (diagnostics.gnn.latestR2 < 0) negR2Models.push(`GNN(R²=${diagnostics.gnn.latestR2})`);
  if (diagnostics.lambda.r2 < 0) negR2Models.push(`Lambda(R²=${diagnostics.lambda.r2})`);
  if (negR2Models.length > 0) {
    lines.push(`  ** NEGATIVE R² DETECTED: ${negR2Models.join(", ")} — emergency data fetch ${emergencyFetchCount > 0 ? `active (${emergencyFetchCount} fetches performed)` : "pending"} **`);
  }

  const errorAnalysis = getErrorAnalysis();
  if (errorAnalysis.totalOutcomes > 0) {
    lines.push(`  Error analysis: ${errorAnalysis.totalErrors} errors > 5K, bias=${errorAnalysis.overallBias}, RMSE=${errorAnalysis.overallRMSE}K`);
    for (const c of errorAnalysis.errorClusters.slice(0, 3)) {
      lines.push(`    ${c.pattern} (n=${c.count}, mean_err=${c.meanError}K)`);
    }
    for (const g of errorAnalysis.familyDataGaps.filter(g => g.needsMore)) {
      lines.push(`    Data gap: ${g.family} (${g.sampleCount} samples, needs >= 20)`);
    }
  }

  if (plateauModels.length > 0) {
    lines.push(`  Plateau detected: ${plateauModels.join(", ")} — consider different experiment types`);
  }

  const dataRequests = getPendingDataRequests();
  if (dataRequests.length > 0) {
    lines.push(`  Pending data requests: ${dataRequests.length}`);
    for (const dr of dataRequests.slice(0, 3)) {
      lines.push(`    ${dr.family}: ${dr.count} structures via ${dr.method}`);
    }
  }

  const techReqs = getActiveTechnicalRequirements();
  if (techReqs.length > 0) {
    lines.push("  ## Model Constraints for Strategy:");
    for (const req of techReqs) {
      const urgTag = req.urgency === "high" ? "[HIGH]" : req.urgency === "medium" ? "[MED]" : "[LOW]";
      lines.push(`    ${urgTag} ${req.type}${req.family ? ` (${req.family})` : ""}: ${req.detail}`);
    }
  }

  const history = getExperimentHistory();
  const recentCompleted = history.filter(h => h.status === "completed").slice(0, 3);
  if (recentCompleted.length > 0) {
    lines.push("  Recent experiments (Model LLM):");
    for (const exp of recentCompleted) {
      const impStr = Object.entries(exp.improvement)
        .map(([k, v]) => `${k}:${v > 0 ? "+" : ""}${v.toFixed(4)}`)
        .join(", ");
      lines.push(`    ${exp.target_model} (${exp.type}): ${impStr || "no change"}`);
    }
  }

  try {
    const activeFeatures = getActiveCustomFeatures();
    if (activeFeatures && activeFeatures.length > 0) {
      lines.push(`  Active computed features (Model LLM): ${activeFeatures.map(f => f.name).join(", ")}`);
    }
  } catch (_e) { lines.push("  Active computed features: not yet initialized"); }

  try {
    const architecture = getCurrentArchitecture();
    if (architecture) {
      lines.push(`  Architecture (Model LLM): ${safeString(architecture.primaryModel)} (switch=${safeString(architecture.switchRecommended)})`);
      if (architecture.modelConfigs) {
        for (const mc of architecture.modelConfigs) {
          lines.push(`    ${safeString(mc.model)}: weight=${safeString(mc.weight)}`);
        }
      }
    }
  } catch (_e) { lines.push("  Architecture: not yet initialized"); }

  try {
    const variance = getVarianceSummary();
    const frontier = getFrontierUncertainty();
    lines.push(`  Uncertainty (global): mean_var=${variance.meanVariance.toFixed(4)}, high_unc=${(variance.highUncertaintyFraction * 100).toFixed(1)}%, source=${safeString(variance.decomposition?.dominantSource, "unknown")}`);
    if (frontier.frontierCount > 0) {
      lines.push(`  Uncertainty (frontier, Tc>100K): n=${frontier.frontierCount}, mean_var=${frontier.frontierMeanVariance.toFixed(4)}, high_unc=${(frontier.frontierHighUncFraction * 100).toFixed(1)}%, mean_norm=${frontier.frontierMeanNormalized.toFixed(3)}`);
      if (frontier.frontierWorstFormulas.length > 0) {
        const worstStr = frontier.frontierWorstFormulas.slice(0, 3).map(f => `${f.formula}(Tc=${f.predictedTc.toFixed(0)}K, unc=${f.normalizedUncertainty.toFixed(3)})`).join(", ");
        lines.push(`    Highest frontier uncertainty: ${worstStr}`);
      }
      if (frontier.frontierHighUncFraction > frontier.globalHighUncFraction * 1.5) {
        lines.push("    ** Frontier uncertainty significantly exceeds global — high-Tc predictions unreliable **");
      }
    } else {
      lines.push("  Uncertainty (frontier): no predictions above 100K yet");
    }
  } catch (_e) { lines.push("  Uncertainty: not yet computed"); }

  try {
    const scheduler = getSchedulerStats();
    lines.push(`  Retrain scheduler: ${scheduler.state.totalRetrainsScheduled} retrains, ${scheduler.state.totalRetrainsSkipped} skips, ~${scheduler.state.totalComputeSaved}s saved`);
  } catch (_e) { lines.push("  Retrain scheduler: not yet initialized"); }

  return lines.join("\n");
}

export async function runCombinedModelLLMCycle(
  emit: EventEmitter,
  currentCycle: number
): Promise<void> {
  if (currentCycle % 10 !== 7) return;

  try {
    const report = await runModelLLMCycle(currentCycle);

    if (report.featureProposals.length > 0) {
      emit("log", {
        phase: "engine",
        event: "Model LLM: features proposed",
        detail: `Enabled features: ${report.featureProposals.map(p => p.name).join(", ")}`,
        dataSource: "Model LLM Controller",
      });
    }

    if (report.architectureRecommendation) {
      emit("log", {
        phase: "engine",
        event: "Model LLM: architecture assessed",
        detail: `Recommended: ${report.architectureRecommendation.primaryModel} — ${truncateAtWordBoundary(report.architectureRecommendation.reasoning, 150)}`,
        dataSource: "Model LLM Controller",
      });
    }

    if (report.uncertaintyProposals.length > 0) {
      emit("log", {
        phase: "engine",
        event: "Model LLM: uncertainty improvements proposed",
        detail: report.uncertaintyProposals.map(p => `${p.type} (priority ${p.priority})`).join(", "),
        dataSource: "Uncertainty Tracker",
      });
    }

    if (report.retrainDecision) {
      const rd = report.retrainDecision;
      emit("log", {
        phase: "engine",
        event: `Retrain scheduler: ${rd.shouldRetrain ? "RETRAIN" : "SKIP"} (${rd.urgency})`,
        detail: truncateAtWordBoundary(rd.reasoning, 180),
        dataSource: "Retrain Scheduler",
      });
    }
  } catch (e) {
    emit("log", {
      phase: "engine",
      event: "Model LLM cycle error",
      detail: truncateAtWordBoundary(e instanceof Error ? e.message : "unknown", 200),
      dataSource: "Model LLM Controller",
    });
  }
}
