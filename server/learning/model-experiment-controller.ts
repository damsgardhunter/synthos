import OpenAI from "openai";
import {
  getComprehensiveModelDiagnostics,
  getModelDiagnosticsForLLM,
  getFailedMaterialsForLLM,
  getBenchmarkForLLM,
  type ComprehensiveModelDiagnostics,
} from "./model-diagnostics";
import { retrainXGBoostFromEvaluated, registerHyperparamResolver } from "./gradient-boost";
import { trainLambdaRegressor } from "./lambda-regressor";
import { trainPhononSurrogate } from "../physics/phonon-surrogate";
import { retrainTBSurrogate } from "../physics/tb-ml-surrogate";
import { enableBuiltinFeature, selectArchitecture } from "./model-llm-controller";

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
  | "rebalance_training"
  | "request_data";

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
  layerCount?: number;
  regularizationL2?: number;
  batchSize?: number;
  epochs?: number;
  bootstrapRatio?: number;
}

export interface DataRequest {
  id: string;
  timestamp: number;
  family: string;
  count: number;
  method: string;
  pressureVariants: boolean;
  pressures: number[];
  status: "pending" | "in_progress" | "completed" | "failed";
  requestedBy: string;
  completedAt?: number;
  generatedCount?: number;
}

const MODEL_TARGET_ALIASES: Record<string, string> = {
  "xgboost": "xgboost",
  "xgb": "xgboost",
  "gradient_boost": "xgboost",
  "gradient-boost": "xgboost",
  "gnn": "gnn",
  "graph_neural_net": "gnn",
  "graph-neural-net": "gnn",
  "lambda-regressor": "lambda-regressor",
  "lambda_regressor": "lambda-regressor",
  "lambda_predictor": "lambda-regressor",
  "lambda-predictor": "lambda-regressor",
  "phonon-surrogate": "phonon-surrogate",
  "phonon_surrogate": "phonon-surrogate",
  "phonon": "phonon-surrogate",
  "tb-surrogate": "tb-surrogate",
  "tb_surrogate": "tb-surrogate",
  "tb-predictor": "tb-surrogate",
  "tb_predictor": "tb-surrogate",
};

function normalizeModelTarget(raw: string): string | null {
  return MODEL_TARGET_ALIASES[raw.toLowerCase().trim()] ?? null;
}

function repairTruncatedJSONArray(raw: string): any[] | null {
  try {
    const arrayMatch = raw.match(/\[\s*\{/);
    if (!arrayMatch) return null;
    const arrayStart = raw.indexOf("[", arrayMatch.index!);
    let depth = 0;
    let lastCompleteObj = -1;
    for (let i = arrayStart; i < raw.length; i++) {
      if (raw[i] === "{") depth++;
      if (raw[i] === "}") {
        depth--;
        if (depth === 0) lastCompleteObj = i;
      }
    }
    if (lastCompleteObj > arrayStart) {
      const repaired = raw.substring(arrayStart, lastCompleteObj + 1) + "]";
      const parsed = JSON.parse(repaired);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

const MAX_EXPERIMENT_RECORDS = 50;
const MAX_DATA_REQUESTS = 30;
let experimentRecords: ExperimentRecord[] = [];
let experimentIdCounter = 0;
const hyperparamOverrides: Map<string, ModelHyperparamOverrides> = new Map();
let pendingDataRequests: DataRequest[] = [];

registerHyperparamResolver(() => hyperparamOverrides.get("xgboost"));

function generateExperimentId(): string {
  experimentIdCounter++;
  return `exp-${Date.now()}-${experimentIdCounter}`;
}

function snapshotModelMetrics(diagnostics: ComprehensiveModelDiagnostics, modelTarget: string): Record<string, number> {
  const safeNum = (v: any): number => (typeof v === "number" && Number.isFinite(v)) ? v : 0;
  switch (modelTarget) {
    case "xgboost":
      return {
        r2: safeNum(diagnostics.xgboost?.r2),
        mae: safeNum(diagnostics.xgboost?.mae),
        rmse: safeNum(diagnostics.xgboost?.rmse),
        datasetSize: safeNum(diagnostics.xgboost?.datasetSize),
        falsePositiveRate: safeNum(diagnostics.xgboost?.falsePositiveRate),
        falseNegativeRate: safeNum(diagnostics.xgboost?.falseNegativeRate),
      };
    case "gnn":
      return {
        r2: safeNum(diagnostics.gnn?.latestR2),
        mae: safeNum(diagnostics.gnn?.latestMAE),
        rmse: safeNum(diagnostics.gnn?.latestRMSE),
        datasetSize: safeNum(diagnostics.gnn?.datasetSize),
      };
    case "lambda-regressor":
      return {
        r2: safeNum(diagnostics.lambda?.r2),
        mae: safeNum(diagnostics.lambda?.mae),
        rmse: safeNum(diagnostics.lambda?.rmse),
        datasetSize: safeNum(diagnostics.lambda?.datasetSize),
      };
    case "phonon-surrogate":
      return {
        omegaLogMAE: safeNum(diagnostics.phononSurrogate?.omegaLogMAE),
        debyeTempMAE: safeNum(diagnostics.phononSurrogate?.debyeTempMAE),
        maxFreqMAE: safeNum(diagnostics.phononSurrogate?.maxFreqMAE),
        stabilityAccuracy: safeNum(diagnostics.phononSurrogate?.stabilityAccuracy),
        datasetSize: safeNum(diagnostics.phononSurrogate?.datasetSize),
      };
    case "tb-surrogate":
      return {
        datasetSize: safeNum(diagnostics.tbSurrogate?.datasetSize),
        predictions: safeNum(diagnostics.tbSurrogate?.predictions),
        trainings: safeNum(diagnostics.tbSurrogate?.trainings),
        modelCount: safeNum(diagnostics.tbSurrogate?.modelCount),
        avgPredictionTimeMs: safeNum(diagnostics.tbSurrogate?.avgPredictionTimeMs),
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
          content: `You are the MODEL LLM for a superconductor discovery system.
Your SOLE responsibility is controlling LEARNING: training datasets, features, architectures, and hyperparameters.
You do NOT control exploration strategy, material families, or search direction — that is handled by a separate Strategy LLM.

Given a diagnostics report of multiple ML models (including failed materials and version benchmarks), propose 1-3 concrete experiments to improve model performance.

Each experiment must be one of these types:
- retrain_model: Retrain a specific model with current data
- adjust_hyperparameters: Tune training parameters. Available parameters per model:
  xgboost: nTrees (10-500), learningRate (0.001-0.3), maxDepth (3-12), minSamples (2-32), ensembleSize (1-10), bootstrapRatio (0.5-1.0)
  gnn: layerCount (2-8), learningRate (0.0001-0.01), dropoutRate (0.0-0.5), epochs (5-50), ensembleSize (1-8)
  lambda-regressor: learningRate (0.0001-0.01), layerCount (2-8), dropoutRate (0.0-0.5), regularizationL2 (0.0001-0.1)
  phonon-surrogate: nTrees (10-200), learningRate (0.01-0.2), maxDepth (3-10)
  tb-surrogate: nTrees (10-200), learningRate (0.01-0.2), maxDepth (3-10)
  Example: {"model": "lambda-regressor", "learning_rate": 0.0005, "layers": 5, "dropout": 0.2}
- expand_dataset: Add more training data from available sources
- add_features: Enable computed features like van_hove_distance, lambda_over_omega, effective_coupling_strength, dos_lambda_product, etc.
  Use changes like {"enable_features": ["van_hove_distance", "dos_lambda_product"]}
- adjust_architecture: Select between model architectures based on dataset size:
  dataset < 100: use xgboost (gradient boosting works better with small data)
  dataset 100-500: use xgboost with larger ensemble
  dataset > 500 with graph data: consider gnn or ensemble weighting
  Use changes like {"primary_model": "xgboost", "ensemble_weights": {"xgboost": 0.7, "gnn": 0.3}}
- recalibrate_uncertainty: Adjust uncertainty scaling
- rebalance_training: Adjust sampling ratios for different material families
- request_data: Request generation of new training data for specific families or material types.
  Use changes like {"family": "boride", "count": 30, "method": "tb_evaluation"} or {"family": "hydride", "pressure_variants": true, "pressures": [0,50,150,300]}

Valid model targets: xgboost, gnn, lambda-regressor, phonon-surrogate, tb-surrogate

YOUR SCOPE (learning decisions only):
- Training datasets: which data to use, sampling ratios
- Features: which computed features to enable/disable
- Architecture: which model to use (XGBoost vs GNN vs ensemble)
- Hyperparameters: learning rate, layer count, regularization, dropout, etc.

OUT OF SCOPE (handled by Strategy LLM):
- Material family priorities
- Pressure range exploration
- Search direction (explore vs exploit)

The report includes:
- Feature importance rankings (which features the model uses most/ignores)
- Error analysis with failure clusters (which families are overpredicted/underpredicted)
- Data gaps (families with too few training samples)
- Failed materials data (predicted stable but actually unstable, phonon failures, etc.)
- Model version benchmarks with scorecards and version comparisons

Use failed materials data to propose targeted classifiers or filters.
If many materials fail with unstable phonons, propose training a phonon stability classifier.
Use version benchmarks to assess whether recent changes helped or regressed.
Use feature importance to identify missing or underused features.
Use error clusters to propose targeted improvements.

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
          content: report + "\n\n" + getFailedMaterialsForLLM() + "\n\n" + getBenchmarkForLLM(),
        },
      ],
    });

    const content = response.choices[0]?.message?.content ?? "[]";
    const jsonMatch = content.match(/\[[\s\S]*\]/);
    let parsed: any[];
    if (jsonMatch) {
      try {
        const attempt = JSON.parse(jsonMatch[0]);
        parsed = Array.isArray(attempt) ? attempt : [];
      } catch {
        parsed = repairTruncatedJSONArray(content) ?? [];
      }
    } else {
      parsed = repairTruncatedJSONArray(content) ?? [];
    }
    if (parsed.length === 0) return [];

    const validTypes: ExperimentType[] = [
      "retrain_model", "adjust_hyperparameters", "expand_dataset",
      "add_features", "adjust_architecture", "recalibrate_uncertainty",
      "rebalance_training", "request_data",
    ];

    return parsed
      .map((p: any) => {
        if (!p.model_target || !p.experiment_type || !p.changes || typeof p.changes !== "object" || !p.reasoning) return null;
        const normalizedTarget = normalizeModelTarget(p.model_target);
        if (!normalizedTarget) return null;
        if (!validTypes.includes(p.experiment_type)) return null;
        return {
          model_target: normalizedTarget,
          experiment_type: p.experiment_type as ExperimentType,
          changes: p.changes,
          reasoning: typeof p.reasoning === "string" ? p.reasoning : String(p.reasoning),
          expected_improvement: p.expected_improvement ?? "unknown",
          priority: Math.max(1, Math.min(3, Number(p.priority) || 2)),
        };
      })
      .filter((p): p is ExperimentProposal => p !== null)
      .slice(0, 3);
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
      const c = experiment.changes;
      if (c.nTrees != null) overrides.nTrees = Math.max(10, Math.min(500, c.nTrees));
      if (c.learningRate != null || c.learning_rate != null) overrides.learningRate = Math.max(0.0001, Math.min(0.3, c.learningRate ?? c.learning_rate));
      if (c.maxDepth != null) overrides.maxDepth = Math.max(3, Math.min(12, c.maxDepth));
      if (c.minSamples != null) overrides.minSamples = Math.max(2, Math.min(32, c.minSamples));
      if (c.ensembleSize != null) overrides.ensembleSize = Math.max(1, Math.min(10, c.ensembleSize));
      if (c.dropoutRate != null || c.dropout != null) overrides.dropoutRate = Math.max(0, Math.min(0.5, c.dropoutRate ?? c.dropout));
      if (c.layerCount != null || c.layers != null) overrides.layerCount = Math.max(2, Math.min(8, c.layerCount ?? c.layers));
      if (c.regularizationL2 != null || c.regularization != null) overrides.regularizationL2 = Math.max(0.0001, Math.min(0.1, c.regularizationL2 ?? c.regularization));
      if (c.batchSize != null) overrides.batchSize = Math.max(4, Math.min(128, c.batchSize));
      if (c.epochs != null) overrides.epochs = Math.max(5, Math.min(50, c.epochs));
      if (c.bootstrapRatio != null) overrides.bootstrapRatio = Math.max(0.5, Math.min(1.0, c.bootstrapRatio));
      hyperparamOverrides.set(experiment.model_target, overrides);
      console.log(`[Model Experiment Controller] Hyperparams updated for ${experiment.model_target}: ${JSON.stringify(overrides)}`);
    }

    if (experiment.experiment_type === "request_data") {
      const request: DataRequest = {
        id: `dr-${Date.now()}`,
        timestamp: Date.now(),
        family: experiment.changes.family ?? "unknown",
        count: Math.min(experiment.changes.count ?? 20, 50),
        method: experiment.changes.method ?? "tb_evaluation",
        pressureVariants: experiment.changes.pressure_variants ?? false,
        pressures: experiment.changes.pressures ?? [0],
        status: "pending",
        requestedBy: id,
      };
      pendingDataRequests.push(request);
      if (pendingDataRequests.length > MAX_DATA_REQUESTS) {
        pendingDataRequests = pendingDataRequests.slice(-MAX_DATA_REQUESTS);
      }
      console.log(`[Model Experiment Controller] Data request queued: ${request.count} ${request.family} structures via ${request.method}`);
    }

    if (experiment.experiment_type === "add_features") {
      const featuresToEnable: string[] = experiment.changes.enable_features ?? [];
      let enabled = 0;
      for (const featureName of featuresToEnable) {
        if (enableBuiltinFeature(featureName)) {
          enabled++;
        }
      }
      console.log(`[Model Experiment Controller] Enabled ${enabled}/${featuresToEnable.length} computed features`);
    }

    if (experiment.experiment_type === "adjust_architecture") {
      try {
        const archResult = await selectArchitecture();
        record.changes = {
          ...record.changes,
          architecture_result: {
            primaryModel: archResult.primaryModel,
            modelConfigs: archResult.modelConfigs,
            switchRecommended: archResult.switchRecommended,
          },
        };
        console.log(`[Model Experiment Controller] Architecture selected: ${archResult.primaryModel} (switch=${archResult.switchRecommended})`);
      } catch (e) {
        console.log(`[Model Experiment Controller] Architecture selection failed: ${e instanceof Error ? e.message : "unknown"}`);
      }
    }

    switch (experiment.model_target) {
      case "xgboost":
        if (experiment.experiment_type !== "request_data") await retrainXGBoostFromEvaluated();
        break;
      case "lambda-regressor":
        if (experiment.experiment_type !== "request_data") trainLambdaRegressor();
        break;
      case "phonon-surrogate":
        if (experiment.experiment_type !== "request_data") trainPhononSurrogate();
        break;
      case "tb-surrogate":
        if (experiment.experiment_type !== "request_data") retrainTBSurrogate();
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

export function getPendingDataRequests(): DataRequest[] {
  return pendingDataRequests.filter(r => r.status === "pending");
}

export function getAllDataRequests(): DataRequest[] {
  return [...pendingDataRequests].reverse();
}

export function completeDataRequest(id: string, generatedCount: number): void {
  const request = pendingDataRequests.find(r => r.id === id);
  if (request) {
    request.status = "completed";
    request.completedAt = Date.now();
    request.generatedCount = generatedCount;
  }
}

export function markDataRequestInProgress(id: string): void {
  const request = pendingDataRequests.find(r => r.id === id);
  if (request) {
    request.status = "in_progress";
  }
}
