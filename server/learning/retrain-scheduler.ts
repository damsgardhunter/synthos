import OpenAI from "openai";
import { getModelVersionHistory, getEvaluatedDatasetStats, getCalibrationData, getXGBEnsembleStats } from "./gradient-boost";
import { getComprehensiveModelDiagnostics } from "./model-diagnostics";
import { getVarianceSummary, computeCalibrationCurve } from "./uncertainty-tracker";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export interface RetrainTrigger {
  type: "dataset_growth" | "error_spike" | "calibration_drift" | "uncertainty_increase" | "scheduled" | "llm_decision";
  reasoning: string;
  severity: "low" | "medium" | "high";
  metrics: Record<string, number>;
}

export interface RetrainDecision {
  shouldRetrain: boolean;
  triggers: RetrainTrigger[];
  reasoning: string;
  urgency: "immediate" | "soon" | "deferred" | "skip";
  estimatedBenefit: string;
  timestamp: number;
}

export interface RetrainHistory {
  decision: RetrainDecision;
  actuallyRetrained: boolean;
  preRetrainMetrics: { r2: number; mae: number; datasetSize: number };
  postRetrainMetrics?: { r2: number; mae: number; datasetSize: number };
  computeTimeSaved?: number;
}

interface SchedulerState {
  lastRetrainDatasetSize: number;
  lastRetrainTimestamp: number;
  lastRetrainR2: number;
  lastRetrainMAE: number;
  lastDecisionTimestamp: number;
  consecutiveSkips: number;
  totalRetrainsScheduled: number;
  totalRetrainsSkipped: number;
  totalComputeSaved: number;
}

const MAX_DECISION_HISTORY = 100;
const decisionHistory: RetrainHistory[] = [];
const DECISION_COOLDOWN_MS = 2 * 60 * 1000;
const MAX_SKIP_STREAK = 10;

const state: SchedulerState = {
  lastRetrainDatasetSize: 0,
  lastRetrainTimestamp: 0,
  lastRetrainR2: 0,
  lastRetrainMAE: Infinity,
  lastDecisionTimestamp: 0,
  consecutiveSkips: 0,
  totalRetrainsScheduled: 0,
  totalRetrainsSkipped: 0,
  totalComputeSaved: 0,
};

function checkDatasetGrowthTrigger(): RetrainTrigger | null {
  const evalStats = getEvaluatedDatasetStats();
  const currentSize = evalStats.totalEvaluated;

  if (state.lastRetrainDatasetSize === 0) {
    return {
      type: "dataset_growth",
      reasoning: `Initial baseline: dataset has ${currentSize} samples — triggering first retrain to establish ground truth`,
      severity: "high",
      metrics: { previousSize: 0, currentSize, growthRatio: Infinity },
    };
  }

  const growthRatio = currentSize / Math.max(1, state.lastRetrainDatasetSize);
  if (growthRatio >= 1.2) {
    return {
      type: "dataset_growth",
      reasoning: `Dataset grew ${((growthRatio - 1) * 100).toFixed(1)}% since last retrain (${state.lastRetrainDatasetSize} -> ${currentSize})`,
      severity: growthRatio >= 1.5 ? "high" : "medium",
      metrics: { previousSize: state.lastRetrainDatasetSize, currentSize, growthRatio },
    };
  }
  return null;
}

function checkErrorSpikeTrigger(): RetrainTrigger | null {
  const diagnostics = getComprehensiveModelDiagnostics();
  const currentR2 = diagnostics.xgboost.r2;
  const currentMAE = diagnostics.xgboost.mae;

  if (state.lastRetrainMAE === Infinity || state.lastRetrainR2 === 0) {
    return null;
  }

  const r2Drop = state.lastRetrainR2 - currentR2;
  const maeIncrease = currentMAE / Math.max(0.01, state.lastRetrainMAE);

  if (r2Drop > 0.05 || maeIncrease > 1.15) {
    return {
      type: "error_spike",
      reasoning: `Model degradation: R² dropped ${r2Drop.toFixed(4)} (${state.lastRetrainR2.toFixed(4)} -> ${currentR2.toFixed(4)}), MAE ratio ${maeIncrease.toFixed(3)}`,
      severity: r2Drop > 0.1 || maeIncrease > 1.3 ? "high" : "medium",
      metrics: { r2Drop, maeIncrease, currentR2, currentMAE },
    };
  }
  return null;
}

function checkCalibrationDriftTrigger(): RetrainTrigger | null {
  const calibration = computeCalibrationCurve();
  if (calibration.totalPredictions < 20) return null;

  if (calibration.ece > 0.15) {
    return {
      type: "calibration_drift",
      reasoning: `Calibration ECE=${calibration.ece.toFixed(4)} exceeds threshold (0.15). ${calibration.overconfidentBins} overconfident bins`,
      severity: calibration.ece > 0.25 ? "high" : "medium",
      metrics: { ece: calibration.ece, mce: calibration.mce, overconfidentBins: calibration.overconfidentBins },
    };
  }
  return null;
}

function checkUncertaintyIncreaseTrigger(): RetrainTrigger | null {
  const variance = getVarianceSummary();
  if (variance.highUncertaintyFraction > 0.4) {
    return {
      type: "uncertainty_increase",
      reasoning: `${(variance.highUncertaintyFraction * 100).toFixed(1)}% of predictions have high uncertainty (threshold: 40%). Dominant source: ${variance.decomposition.dominantSource}`,
      severity: variance.highUncertaintyFraction > 0.6 ? "high" : "medium",
      metrics: {
        highUncFraction: variance.highUncertaintyFraction,
        meanVariance: variance.meanVariance,
        epistemic: variance.decomposition.epistemic,
      },
    };
  }
  return null;
}

const GNN_STALE_WARN_MS = 6 * 3600_000;
const GNN_STALE_CRITICAL_MS = 24 * 3600_000;
const GNN_NEW_OUTCOMES_WARN = 50;
const GNN_NEW_OUTCOMES_CRITICAL = 150;

function checkGNNStalenessTrigger(): RetrainTrigger | null {
  const diagnostics = getComprehensiveModelDiagnostics();
  const stalenessMs = diagnostics.gnn.modelStalenessMs;
  const newOutcomes = diagnostics.gnn.newOutcomesSinceLastTrain;

  if (newOutcomes >= GNN_NEW_OUTCOMES_CRITICAL) {
    return {
      type: "scheduled",
      reasoning: `GNN has ${newOutcomes} unevaluated outcomes since last train (>${GNN_NEW_OUTCOMES_CRITICAL}) — model operating on severely outdated data`,
      severity: "high",
      metrics: { newOutcomes, stalenessHours: Math.round(stalenessMs / 3600_000) },
    };
  }
  if (stalenessMs > GNN_STALE_CRITICAL_MS) {
    return {
      type: "scheduled",
      reasoning: `GNN ensemble stale for ${Math.round(stalenessMs / 3600_000)}h (>24h), ${newOutcomes} new outcomes — active learning degraded without fresh model`,
      severity: "high",
      metrics: { stalenessHours: Math.round(stalenessMs / 3600_000), thresholdHours: 24, newOutcomes },
    };
  }
  if (newOutcomes >= GNN_NEW_OUTCOMES_WARN) {
    return {
      type: "scheduled",
      reasoning: `GNN has ${newOutcomes} new outcomes since last train (>${GNN_NEW_OUTCOMES_WARN}) — consider retraining to incorporate recent data`,
      severity: "medium",
      metrics: { newOutcomes, stalenessHours: Math.round(stalenessMs / 3600_000) },
    };
  }
  if (stalenessMs > GNN_STALE_WARN_MS) {
    return {
      type: "scheduled",
      reasoning: `GNN ensemble aging: ${Math.round(stalenessMs / 3600_000)}h since last retrain (warn at 6h), ${newOutcomes} new outcomes`,
      severity: "medium",
      metrics: { stalenessHours: Math.round(stalenessMs / 3600_000), thresholdHours: 6, newOutcomes },
    };
  }
  return null;
}

function collectTriggers(): RetrainTrigger[] {
  const triggers: RetrainTrigger[] = [];

  const datasetTrigger = checkDatasetGrowthTrigger();
  if (datasetTrigger) triggers.push(datasetTrigger);

  const errorTrigger = checkErrorSpikeTrigger();
  if (errorTrigger) triggers.push(errorTrigger);

  const calibrationTrigger = checkCalibrationDriftTrigger();
  if (calibrationTrigger) triggers.push(calibrationTrigger);

  const uncertaintyTrigger = checkUncertaintyIncreaseTrigger();
  if (uncertaintyTrigger) triggers.push(uncertaintyTrigger);

  const gnnStalenessTrigger = checkGNNStalenessTrigger();
  if (gnnStalenessTrigger) triggers.push(gnnStalenessTrigger);

  if (state.consecutiveSkips >= MAX_SKIP_STREAK) {
    triggers.push({
      type: "scheduled",
      reasoning: `${state.consecutiveSkips} consecutive retrain skips — forcing periodic retrain to prevent staleness`,
      severity: "medium",
      metrics: { consecutiveSkips: state.consecutiveSkips },
    });
  }

  return triggers;
}

export async function evaluateRetrainNeed(): Promise<RetrainDecision> {
  if (Date.now() - state.lastDecisionTimestamp < DECISION_COOLDOWN_MS && state.lastDecisionTimestamp > 0) {
    return {
      shouldRetrain: false,
      triggers: [],
      reasoning: "Cooldown period active — skipping evaluation",
      urgency: "skip",
      estimatedBenefit: "none",
      timestamp: Date.now(),
    };
  }

  state.lastDecisionTimestamp = Date.now();
  const triggers = collectTriggers();

  if (triggers.length === 0) {
    state.consecutiveSkips++;
    state.totalRetrainsSkipped++;
    state.totalComputeSaved += 15;
    return {
      shouldRetrain: false,
      triggers: [],
      reasoning: "No retrain triggers detected — model performance is stable",
      urgency: "skip",
      estimatedBenefit: "Saved ~15s compute by skipping unnecessary retrain",
      timestamp: Date.now(),
    };
  }

  const hasHighSeverity = triggers.some(t => t.severity === "high");
  if (hasHighSeverity) {
    state.consecutiveSkips = 0;
    state.totalRetrainsScheduled++;
    return {
      shouldRetrain: true,
      triggers,
      reasoning: `High-severity trigger detected: ${triggers.filter(t => t.severity === "high").map(t => t.type).join(", ")}`,
      urgency: "immediate",
      estimatedBenefit: "Critical model correction needed to prevent prediction drift",
      timestamp: Date.now(),
    };
  }

  const diagnostics = getComprehensiveModelDiagnostics();
  const evalStats = getEvaluatedDatasetStats();

  try {
    const triggersStr = triggers
      .map(t => `${t.type} (${t.severity}): ${t.reasoning}`)
      .join("\n");

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.2,
      max_tokens: 400,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `You are an ML operations scheduler deciding whether to retrain a superconductor Tc prediction model.
Retraining costs ~15 seconds of compute. Only retrain when the benefit outweighs the cost.

Current model state:
- XGBoost R²=${diagnostics.xgboost.r2.toFixed(4)}, MAE=${diagnostics.xgboost.mae.toFixed(2)}K
- Dataset: ${evalStats.totalEvaluated} evaluated samples
- Last retrain dataset size: ${state.lastRetrainDatasetSize}
- Consecutive skips: ${state.consecutiveSkips}
- Total retrains scheduled: ${state.totalRetrainsScheduled}
- Total compute saved: ~${state.totalComputeSaved}s

Active triggers:
${triggersStr}

Decision criteria:
- dataset_growth >= 20%: Almost always retrain (new data available)
- error_spike with R² drop > 0.05: Retrain urgently
- calibration_drift with ECE > 0.15: Retrain soon
- uncertainty_increase > 40%: Retrain if epistemic (not aleatoric)
- Multiple medium triggers together: Usually retrain

Respond JSON:
{
  "shouldRetrain": true/false,
  "urgency": "immediate"|"soon"|"deferred",
  "reasoning": "explanation",
  "estimatedBenefit": "what improvement to expect"
}`,
        },
        { role: "user", content: "Should we retrain the model now?" },
      ],
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error("No response");

    const parsed = JSON.parse(content);
    const shouldRetrain = Boolean(parsed.shouldRetrain);

    if (shouldRetrain) {
      state.consecutiveSkips = 0;
      state.totalRetrainsScheduled++;
    } else {
      state.consecutiveSkips++;
      state.totalRetrainsSkipped++;
      state.totalComputeSaved += 15;
    }

    return {
      shouldRetrain,
      triggers,
      reasoning: String(parsed.reasoning || "LLM decision"),
      urgency: (["immediate", "soon", "deferred"].includes(parsed.urgency) ? parsed.urgency : "soon") as any,
      estimatedBenefit: String(parsed.estimatedBenefit || ""),
      timestamp: Date.now(),
    };
  } catch (e) {
    console.log(`[RetrainScheduler] LLM evaluation failed: ${e instanceof Error ? e.message : "unknown"} — using rule-based fallback`);

    const mediumCount = triggers.filter(t => t.severity === "medium").length;
    const shouldRetrain = mediumCount >= 2;

    if (shouldRetrain) {
      state.consecutiveSkips = 0;
      state.totalRetrainsScheduled++;
    } else {
      state.consecutiveSkips++;
      state.totalRetrainsSkipped++;
      state.totalComputeSaved += 15;
    }

    return {
      shouldRetrain,
      triggers,
      reasoning: `Fallback: ${mediumCount} medium triggers ${shouldRetrain ? ">= 2, retraining" : "< 2, skipping"}`,
      urgency: shouldRetrain ? "soon" : "deferred",
      estimatedBenefit: shouldRetrain ? "Multi-trigger correction" : `Saved ~15s compute`,
      timestamp: Date.now(),
    };
  }
}

export function recordRetrainOutcome(
  decision: RetrainDecision,
  actuallyRetrained: boolean,
  preMetrics: { r2: number; mae: number; datasetSize: number },
  postMetrics?: { r2: number; mae: number; datasetSize: number }
): void {
  const entry: RetrainHistory = {
    decision,
    actuallyRetrained,
    preRetrainMetrics: preMetrics,
    postRetrainMetrics: postMetrics,
  };

  if (!actuallyRetrained) {
    entry.computeTimeSaved = 15;
  }

  if (actuallyRetrained) {
    state.lastRetrainDatasetSize = postMetrics?.datasetSize ?? preMetrics.datasetSize;
    state.lastRetrainTimestamp = Date.now();
    state.lastRetrainR2 = postMetrics?.r2 ?? preMetrics.r2;
    state.lastRetrainMAE = postMetrics?.mae ?? preMetrics.mae;
  }

  decisionHistory.push(entry);
  if (decisionHistory.length > MAX_DECISION_HISTORY) {
    decisionHistory.splice(0, 1);
  }
}

export function getSchedulerStats(): {
  state: SchedulerState;
  recentDecisions: RetrainHistory[];
  efficiency: {
    retrainRate: number;
    skipRate: number;
    avgBenefitPerRetrain: number;
    totalComputeSaved: number;
  };
} {
  const total = state.totalRetrainsScheduled + state.totalRetrainsSkipped;
  const retrained = decisionHistory.filter(d => d.actuallyRetrained);
  let avgBenefit = 0;
  if (retrained.length > 0) {
    const benefits = retrained
      .filter(d => d.postRetrainMetrics)
      .map(d => (d.postRetrainMetrics!.r2 - d.preRetrainMetrics.r2));
    avgBenefit = benefits.length > 0 ? benefits.reduce((a, b) => a + b, 0) / benefits.length : 0;
  }

  return {
    state: { ...state },
    recentDecisions: decisionHistory.slice(-10),
    efficiency: {
      retrainRate: total > 0 ? state.totalRetrainsScheduled / total : 0,
      skipRate: total > 0 ? state.totalRetrainsSkipped / total : 0,
      avgBenefitPerRetrain: avgBenefit,
      totalComputeSaved: state.totalComputeSaved,
    },
  };
}

export function getSchedulerForLLM(): string {
  const stats = getSchedulerStats();
  const lines: string[] = [
    "=== Retrain Scheduler ===",
    `Total decisions: ${stats.state.totalRetrainsScheduled + stats.state.totalRetrainsSkipped}`,
    `Retrains: ${stats.state.totalRetrainsScheduled}, Skips: ${stats.state.totalRetrainsSkipped}`,
    `Skip streak: ${stats.state.consecutiveSkips}`,
    `Compute saved: ~${stats.state.totalComputeSaved}s`,
    `Last retrain dataset: ${stats.state.lastRetrainDatasetSize}`,
    `Efficiency: retrain ${(stats.efficiency.retrainRate * 100).toFixed(1)}%, skip ${(stats.efficiency.skipRate * 100).toFixed(1)}%`,
  ];

  if (stats.recentDecisions.length > 0) {
    const last = stats.recentDecisions[stats.recentDecisions.length - 1];
    lines.push(`Last decision: ${last.decision.shouldRetrain ? "RETRAIN" : "SKIP"} (${last.decision.urgency}) — ${last.decision.reasoning.slice(0, 80)}`);
  }

  return lines.join("\n");
}
