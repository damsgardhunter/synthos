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
import { getUncertaintyForLLM, getVarianceSummary } from "./uncertainty-tracker";
import { evaluateRetrainNeed, recordRetrainOutcome, getSchedulerForLLM, getSchedulerStats } from "./retrain-scheduler";
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

function detectPlateau(model: string): boolean {
  const modelTrends = metricTrends.filter(t => t.model === model);
  if (modelTrends.length === 0) return false;

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

function filterForExploration(proposals: ExperimentProposal[]): ExperimentProposal[] {
  const explorationTypes: Set<string> = new Set(["add_features", "request_data", "expand_dataset", "rebalance_training"]);
  const stuckModels = new Set<string>();
  for (const [model, count] of consecutiveNoImprovement) {
    if (count >= CONSECUTIVE_NO_IMPROVEMENT_EXPLORE_THRESHOLD) stuckModels.add(model);
  }
  if (stuckModels.size === 0) return proposals;

  const explorationProposals = proposals.filter(
    p => stuckModels.has(p.model_target) ? explorationTypes.has(p.experiment_type) : true
  );
  if (explorationProposals.length > 0) return explorationProposals;
  return proposals;
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
      detail: e instanceof Error ? e.message.slice(0, 150) : "unknown",
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
    detail: `${topExperiment.experiment_type} on ${topExperiment.model_target}: ${topExperiment.reasoning.slice(0, 100)}`,
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
      detail: e instanceof Error ? e.message.slice(0, 150) : "unknown",
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
    if (!hasPhonon) lines.push("  ** Model ignores phonon features — consider adding Debye temperature emphasis **");
    if (!hasLambda) lines.push("  ** Model ignores e-ph coupling — lambda features underused **");
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

  const activeFeatures = getActiveCustomFeatures();
  if (activeFeatures.length > 0) {
    lines.push(`  Active computed features (Model LLM): ${activeFeatures.map(f => f.name).join(", ")}`);
  }

  const architecture = getCurrentArchitecture();
  if (architecture) {
    lines.push(`  Architecture (Model LLM): ${architecture.primaryModel} (switch=${architecture.switchRecommended})`);
    for (const mc of architecture.modelConfigs) {
      lines.push(`    ${mc.model}: weight=${mc.weight}`);
    }
  }

  const variance = getVarianceSummary();
  lines.push(`  Uncertainty: mean_var=${variance.meanVariance.toFixed(4)}, high_unc=${(variance.highUncertaintyFraction * 100).toFixed(1)}%, source=${variance.decomposition.dominantSource}`);

  const scheduler = getSchedulerStats();
  lines.push(`  Retrain scheduler: ${scheduler.state.totalRetrainsScheduled} retrains, ${scheduler.state.totalRetrainsSkipped} skips, ~${scheduler.state.totalComputeSaved}s saved`);

  return lines.join("\n");
}

export async function runCombinedModelLLMCycle(
  emit: EventEmitter,
  currentCycle: number
): Promise<void> {
  if (currentCycle % 10 !== 0) return;

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
        detail: `Recommended: ${report.architectureRecommendation.primaryModel} — ${report.architectureRecommendation.reasoning.slice(0, 100)}`,
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
        detail: rd.reasoning.slice(0, 120),
        dataSource: "Retrain Scheduler",
      });
    }
  } catch (e) {
    emit("log", {
      phase: "engine",
      event: "Model LLM cycle error",
      detail: e instanceof Error ? e.message.slice(0, 150) : "unknown",
      dataSource: "Model LLM Controller",
    });
  }
}
