import OpenAI from "openai";
import {
  getComprehensiveModelDiagnostics,
  getModelDiagnosticsForLLM,
  type ComprehensiveModelDiagnostics,
} from "./model-diagnostics";
import { retrainXGBoostFromEvaluated } from "./gradient-boost";
import { trainLambdaRegressor } from "./lambda-regressor";
import { trainPhononSurrogate } from "../physics/phonon-surrogate";
import { retrainTBSurrogate } from "../physics/tb-ml-surrogate";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export type ExperimentType =
  | "retrain_model"
  | "adjust_hyperparameters"
  | "expand_dataset"
  | "add_features"
  | "adjust_architecture"
  | "recalibrate_uncertainty"
  | "rebalance_training";

export type ExperimentStatus = "proposed" | "running" | "completed" | "failed";

export interface ExperimentProposal {
  model_target: string;
  experiment_type: ExperimentType;
  changes: Record<string, any>;
  reasoning: string;
  expected_improvement: string;
  priority: number;
}

export interface ExperimentRecord {
  id: string;
  timestamp: number;
  type: ExperimentType;
  target_model: string;
  changes: Record<string, any>;
  before_metrics: Record<string, number>;
  after_metrics: Record<string, number>;
  improvement: Record<string, number>;
  status: ExperimentStatus;
  llm_reasoning: string;
  completedAt?: number;
  error?: string;
}

export interface ModelHyperparamOverrides {
  nTrees?: number;
  learningRate?: number;
  maxDepth?: number;
  minSamples?: number;
  ensembleSize?: number;
  dropoutRate?: number;
}

const MAX_EXPERIMENT_RECORDS = 50;
let experimentRecords: ExperimentRecord[] = [];
let experimentIdCounter = 0;
const hyperparamOverrides: Map<string, ModelHyperparamOverrides> = new Map();

function generateExperimentId(): string {
  experimentIdCounter++;
  return `exp-${Date.now()}-${experimentIdCounter}`;
}

function snapshotModelMetrics(diagnostics: ComprehensiveModelDiagnostics, modelTarget: string): Record<string, number> {
  switch (modelTarget) {
    case "xgboost":
      return {
        r2: diagnostics.xgboost.r2,
        mae: diagnostics.xgboost.mae,
        rmse: diagnostics.xgboost.rmse,
        datasetSize: diagnostics.xgboost.datasetSize,
        falsePositiveRate: diagnostics.xgboost.falsePositiveRate,
        falseNegativeRate: diagnostics.xgboost.falseNegativeRate,
      };
    case "gnn":
      return {
        r2: diagnostics.gnn.latestR2,
        mae: diagnostics.gnn.latestMAE,
        rmse: diagnostics.gnn.latestRMSE,
        datasetSize: diagnostics.gnn.datasetSize,
      };
    case "lambda-regressor":
      return {
        r2: diagnostics.lambda.r2,
        mae: diagnostics.lambda.mae,
        rmse: diagnostics.lambda.rmse,
        datasetSize: diagnostics.lambda.datasetSize,
      };
    case "phonon-surrogate":
      return {
        omegaLogMAE: diagnostics.phononSurrogate.omegaLogMAE,
        debyeTempMAE: diagnostics.phononSurrogate.debyeTempMAE,
        maxFreqMAE: diagnostics.phononSurrogate.maxFreqMAE,
        stabilityAccuracy: diagnostics.phononSurrogate.stabilityAccuracy,
        datasetSize: diagnostics.phononSurrogate.datasetSize,
      };
    case "tb-surrogate":
      return {
        datasetSize: diagnostics.tbSurrogate.datasetSize,
        predictions: diagnostics.tbSurrogate.predictions,
        trainings: diagnostics.tbSurrogate.trainings,
      };
    default:
      return {};
  }
}

function computeImprovement(before: Record<string, number>, after: Record<string, number>): Record<string, number> {
  const improvement: Record<string, number> = {};
  for (const key of Object.keys(before)) {
    if (after[key] != null) {
      improvement[key] = Math.round((after[key] - before[key]) * 10000) / 10000;
    }
  }
  return improvement;
}

export async function proposeModelExperiments(diagnosticsReport?: string): Promise<ExperimentProposal[]> {
  const report = diagnosticsReport ?? getModelDiagnosticsForLLM();

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.4,
      max_tokens: 1200,
      messages: [
        {
          role: "system",
          content: `You are a machine learning model improvement advisor for a superconductor discovery system.
Given a diagnostics report of multiple ML models, propose 1-3 concrete experiments to improve model performance.

Each experiment must be one of these types:
- retrain_model: Retrain a specific model with current data
- adjust_hyperparameters: Change nTrees, learningRate, maxDepth, minSamples
- expand_dataset: Add more training data from available sources
- add_features: Enable/disable feature groups
- adjust_architecture: Change ensemble size, dropout rate
- recalibrate_uncertainty: Adjust uncertainty scaling
- rebalance_training: Adjust sampling ratios

Valid model targets: xgboost, gnn, lambda-regressor, phonon-surrogate, tb-surrogate

Respond ONLY with a JSON array of experiments:
[{
  "model_target": "string",
  "experiment_type": "string",
  "changes": {"key": "value"},
  "reasoning": "string",
  "expected_improvement": "string",
  "priority": 1
}]

Priority 1 = highest. Focus on models with worst metrics or red/yellow health status.
Be specific about what parameters to change and why.`,
        },
        {
          role: "user",
          content: report,
        },
      ],
    });

    const content = response.choices[0]?.message?.content ?? "[]";
    const jsonMatch = content.match(/\[[\s\S]*\]/);
    if (!jsonMatch) return [];

    const parsed = JSON.parse(jsonMatch[0]);
    if (!Array.isArray(parsed)) return [];

    const validTypes: ExperimentType[] = [
      "retrain_model", "adjust_hyperparameters", "expand_dataset",
      "add_features", "adjust_architecture", "recalibrate_uncertainty",
      "rebalance_training",
    ];
    const validModels = ["xgboost", "gnn", "lambda-regressor", "phonon-surrogate", "tb-surrogate"];

    return parsed
      .filter((p: any) =>
        p.model_target && validModels.includes(p.model_target) &&
        p.experiment_type && validTypes.includes(p.experiment_type) &&
        p.changes && typeof p.changes === "object" &&
        p.reasoning && typeof p.reasoning === "string"
      )
      .slice(0, 3)
      .map((p: any) => ({
        model_target: p.model_target,
        experiment_type: p.experiment_type as ExperimentType,
        changes: p.changes,
        reasoning: p.reasoning,
        expected_improvement: p.expected_improvement ?? "unknown",
        priority: Math.max(1, Math.min(3, Number(p.priority) || 2)),
      }));
  } catch (e) {
    console.log(`[Model Experiment Controller] LLM proposal failed: ${e instanceof Error ? e.message : "unknown"}`);
    return [];
  }
}

export async function executeExperiment(experiment: ExperimentProposal): Promise<ExperimentRecord> {
  const id = generateExperimentId();
  const diagnosticsBefore = getComprehensiveModelDiagnostics();
  const beforeMetrics = snapshotModelMetrics(diagnosticsBefore, experiment.model_target);

  const record: ExperimentRecord = {
    id,
    timestamp: Date.now(),
    type: experiment.experiment_type,
    target_model: experiment.model_target,
    changes: experiment.changes,
    before_metrics: beforeMetrics,
    after_metrics: {},
    improvement: {},
    status: "running",
    llm_reasoning: experiment.reasoning,
  };

  experimentRecords.push(record);
  if (experimentRecords.length > MAX_EXPERIMENT_RECORDS) {
    experimentRecords = experimentRecords.slice(-MAX_EXPERIMENT_RECORDS);
  }

  try {
    if (experiment.experiment_type === "adjust_hyperparameters") {
      const overrides = hyperparamOverrides.get(experiment.model_target) ?? {};
      if (experiment.changes.nTrees != null) overrides.nTrees = experiment.changes.nTrees;
      if (experiment.changes.learningRate != null) overrides.learningRate = experiment.changes.learningRate;
      if (experiment.changes.maxDepth != null) overrides.maxDepth = experiment.changes.maxDepth;
      if (experiment.changes.minSamples != null) overrides.minSamples = experiment.changes.minSamples;
      if (experiment.changes.ensembleSize != null) overrides.ensembleSize = experiment.changes.ensembleSize;
      if (experiment.changes.dropoutRate != null) overrides.dropoutRate = experiment.changes.dropoutRate;
      hyperparamOverrides.set(experiment.model_target, overrides);
    }

    switch (experiment.model_target) {
      case "xgboost":
        await retrainXGBoostFromEvaluated();
        break;
      case "lambda-regressor":
        trainLambdaRegressor();
        break;
      case "phonon-surrogate":
        trainPhononSurrogate();
        break;
      case "tb-surrogate":
        retrainTBSurrogate();
        break;
      case "gnn":
        break;
      default:
        break;
    }

    const diagnosticsAfter = getComprehensiveModelDiagnostics();
    const afterMetrics = snapshotModelMetrics(diagnosticsAfter, experiment.model_target);
    const improvement = computeImprovement(beforeMetrics, afterMetrics);

    record.after_metrics = afterMetrics;
    record.improvement = improvement;
    record.status = "completed";
    record.completedAt = Date.now();

    console.log(`[Model Experiment Controller] Experiment ${id} completed for ${experiment.model_target}: ${JSON.stringify(improvement)}`);
  } catch (e) {
    record.status = "failed";
    record.error = e instanceof Error ? e.message : "unknown error";
    record.completedAt = Date.now();
    console.log(`[Model Experiment Controller] Experiment ${id} failed: ${record.error}`);
  }

  return record;
}

export function getHyperparamOverrides(model: string): ModelHyperparamOverrides | undefined {
  return hyperparamOverrides.get(model);
}

export function setHyperparamOverrides(model: string, overrides: ModelHyperparamOverrides): void {
  hyperparamOverrides.set(model, overrides);
}

export function getExperimentHistory(): ExperimentRecord[] {
  return [...experimentRecords].reverse();
}

export function getActiveExperiments(): ExperimentRecord[] {
  return experimentRecords.filter(r => r.status === "running");
}

export function getExperimentStats(): {
  totalExperiments: number;
  completed: number;
  failed: number;
  running: number;
  proposed: number;
  avgImprovementByModel: Record<string, Record<string, number>>;
  recentExperiments: ExperimentRecord[];
  hyperparamOverrides: Record<string, ModelHyperparamOverrides>;
} {
  const completed = experimentRecords.filter(r => r.status === "completed");
  const failed = experimentRecords.filter(r => r.status === "failed");
  const running = experimentRecords.filter(r => r.status === "running");
  const proposed = experimentRecords.filter(r => r.status === "proposed");

  const avgImprovementByModel: Record<string, Record<string, number>> = {};
  for (const record of completed) {
    if (!avgImprovementByModel[record.target_model]) {
      avgImprovementByModel[record.target_model] = {};
    }
    for (const [key, val] of Object.entries(record.improvement)) {
      if (!avgImprovementByModel[record.target_model][key]) {
        avgImprovementByModel[record.target_model][key] = 0;
      }
      avgImprovementByModel[record.target_model][key] += val;
    }
  }
  for (const model of Object.keys(avgImprovementByModel)) {
    const modelCompleted = completed.filter(r => r.target_model === model).length;
    if (modelCompleted > 0) {
      for (const key of Object.keys(avgImprovementByModel[model])) {
        avgImprovementByModel[model][key] = Math.round((avgImprovementByModel[model][key] / modelCompleted) * 10000) / 10000;
      }
    }
  }

  const overridesObj: Record<string, ModelHyperparamOverrides> = {};
  hyperparamOverrides.forEach((val, key) => {
    overridesObj[key] = val;
  });

  return {
    totalExperiments: experimentRecords.length,
    completed: completed.length,
    failed: failed.length,
    running: running.length,
    proposed: proposed.length,
    avgImprovementByModel,
    recentExperiments: experimentRecords.slice(-10).reverse(),
    hyperparamOverrides: overridesObj,
  };
}
