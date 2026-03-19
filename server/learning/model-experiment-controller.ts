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
import { enableBuiltinFeature, selectArchitecture, getAvailableFeatureDefinitions, getActiveCustomFeatures } from "./model-llm-controller";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  timeout: 60_000,
  maxRetries: 0, // Connection errors do not self-resolve; avoid 3x retry amplification
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

export interface TechnicalRequirement {
  type: "data_need" | "feature_gap" | "architecture_change";
  family?: string;
  detail: string;
  urgency: "low" | "medium" | "high";
}

export interface ExperimentProposal {
  model_target: string;
  experiment_type: ExperimentType;
  changes: Record<string, any>;
  reasoning: string;
  expected_improvement: string;
  priority: number;
  technicalRequirements?: TechnicalRequirement[];
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

const HYPERPARAM_RANGES: Record<string, Record<string, [number, number]>> = {
  xgboost: {
    nTrees: [10, 500], learningRate: [0.001, 0.3], maxDepth: [3, 12],
    minSamples: [2, 32], ensembleSize: [1, 10], bootstrapRatio: [0.5, 1.0],
  },
  gnn: {
    layerCount: [2, 8], learningRate: [0.0001, 0.01], dropoutRate: [0, 0.5],
    epochs: [5, 50], ensembleSize: [1, 8],
  },
  "lambda-regressor": {
    learningRate: [0.0001, 0.01], layerCount: [2, 8], dropoutRate: [0, 0.5],
    regularizationL2: [0.0001, 0.1],
  },
  "phonon-surrogate": { nTrees: [10, 200], learningRate: [0.01, 0.2], maxDepth: [3, 10] },
  "tb-surrogate": { nTrees: [10, 200], learningRate: [0.01, 0.2], maxDepth: [3, 10] },
};

const HYPERPARAM_ALIASES: Record<string, string> = {
  learning_rate: "learningRate", lr: "learningRate",
  n_trees: "nTrees", num_trees: "nTrees", trees: "nTrees",
  max_depth: "maxDepth", depth: "maxDepth",
  min_samples: "minSamples",
  ensemble_size: "ensembleSize",
  dropout: "dropoutRate", dropout_rate: "dropoutRate",
  layers: "layerCount", layer_count: "layerCount", num_layers: "layerCount",
  regularization: "regularizationL2", l2: "regularizationL2", regularization_l2: "regularizationL2",
  batch_size: "batchSize",
  bootstrap_ratio: "bootstrapRatio",
};

function clampHyperparameters(modelTarget: string, changes: Record<string, any>, datasetSize?: number): { clamped: Record<string, any>; warnings: string[] } {
  const ranges = HYPERPARAM_RANGES[modelTarget];
  if (!ranges) return { clamped: { ...changes }, warnings: [] };

  const dynamicRanges: Record<string, [number, number]> = {};
  for (const [k, v] of Object.entries(ranges)) dynamicRanges[k] = [...v];
  if (datasetSize != null && datasetSize > 0 && dynamicRanges["nTrees"]) {
    const safeMax = Math.min(dynamicRanges["nTrees"][1], Math.max(dynamicRanges["nTrees"][0], datasetSize * 10));
    if (safeMax < dynamicRanges["nTrees"][1]) {
      dynamicRanges["nTrees"] = [dynamicRanges["nTrees"][0], safeMax];
    }
  }
  if (datasetSize != null && datasetSize > 0 && dynamicRanges["maxDepth"]) {
    const safeMaxDepth = Math.min(dynamicRanges["maxDepth"][1], Math.max(dynamicRanges["maxDepth"][0], Math.floor(Math.log2(datasetSize))));
    if (safeMaxDepth < dynamicRanges["maxDepth"][1]) {
      dynamicRanges["maxDepth"] = [dynamicRanges["maxDepth"][0], safeMaxDepth];
    }
  }

  const clamped: Record<string, any> = {};
  const warnings: string[] = [];
  const seenCanonical = new Map<string, string>();
  for (const [rawKey, val] of Object.entries(changes)) {
    const canonKey = HYPERPARAM_ALIASES[rawKey] ?? rawKey;
    if (seenCanonical.has(canonKey)) {
      warnings.push(`${rawKey} collides with ${seenCanonical.get(canonKey)} (both map to ${canonKey}), keeping first`);
      continue;
    }
    seenCanonical.set(canonKey, rawKey);
    const range = dynamicRanges[canonKey];
    if (range && typeof val === "number") {
      const [lo, hi] = range;
      if (val < lo || val > hi) {
        warnings.push(`${rawKey}=${val} clamped to [${lo},${hi}]${datasetSize != null ? ` (dataset=${datasetSize})` : ""}`);
        clamped[canonKey] = Math.max(lo, Math.min(hi, val));
      } else {
        clamped[canonKey] = val;
      }
    } else {
      clamped[rawKey] = val;
    }
  }
  return { clamped, warnings };
}

function isDuplicateExperiment(proposal: ExperimentProposal): boolean {
  const recentWindow = experimentRecords.slice(-5);
  for (const past of recentWindow) {
    if (past.target_model !== proposal.model_target) continue;
    if (past.type !== proposal.experiment_type) continue;
    if (past.status === "completed" && Object.values(past.improvement).every(v => Math.abs(v) < 0.001)) {
      if (changesMatch(past.changes, proposal.changes)) return true;
    }
    if (past.status === "failed") {
      if (changesMatch(past.changes, proposal.changes)) return true;
    }
    if (past.status === "running") {
      if (changesMatch(past.changes, proposal.changes)) return true;
    }
  }
  return false;
}

function changesMatch(a: Record<string, any>, b: Record<string, any>): boolean {
  const keysA = Object.keys(a).sort();
  const keysB = Object.keys(b).sort();
  if (keysA.length !== keysB.length) return false;
  for (let i = 0; i < keysA.length; i++) {
    if (keysA[i] !== keysB[i]) return false;
    const va = a[keysA[i]], vb = b[keysB[i]];
    if (typeof va === "number" && typeof vb === "number") {
      if (Math.abs(va - vb) > 0.0001) return false;
    } else if (JSON.stringify(va) !== JSON.stringify(vb)) {
      return false;
    }
  }
  return true;
}

function extractTechnicalRequirements(proposal: ExperimentProposal): TechnicalRequirement[] {
  const reqs: TechnicalRequirement[] = [];
  if (proposal.experiment_type === "request_data") {
    const family = proposal.changes.family ?? "unknown";
    const count = proposal.changes.count ?? 20;
    reqs.push({
      type: "data_need",
      family,
      detail: `Model requires ${count} more ${family} training samples via ${proposal.changes.method ?? "tb_evaluation"}`,
      urgency: count >= 40 ? "high" : count >= 20 ? "medium" : "low",
    });
  }
  if (proposal.experiment_type === "rebalance_training") {
    for (const [family, ratio] of Object.entries(proposal.changes)) {
      if (typeof ratio === "number" && ratio > 0.3) {
        reqs.push({
          type: "data_need",
          family,
          detail: `Model needs higher sampling weight (${(ratio * 100).toFixed(0)}%) for ${family}`,
          urgency: ratio > 0.5 ? "high" : "medium",
        });
      }
    }
  }
  if (proposal.experiment_type === "add_features") {
    const features = proposal.changes.enable_features ?? [];
    if (features.length > 0) {
      reqs.push({
        type: "feature_gap",
        detail: `Enabling ${features.length} features: ${features.join(", ")}`,
        urgency: features.length > 3 ? "high" : "low",
      });
    }
  }
  if (proposal.experiment_type === "adjust_architecture") {
    const primary = proposal.changes.primary_model;
    if (primary) {
      reqs.push({
        type: "architecture_change",
        detail: `Model recommends switching to ${primary}`,
        urgency: "medium",
      });
    }
  }
  return reqs;
}

function buildFeatureContext(): string {
  const available = getAvailableFeatureDefinitions();
  const active = getActiveCustomFeatures();
  const activeNames = new Set(active.map(f => f.name));
  const enabledList = available.filter(f => activeNames.has(f.name));
  const disabledList = available.filter(f => !activeNames.has(f.name));
  const lines: string[] = [];
  if (disabledList.length > 0) {
    lines.push("Currently DISABLED (can be enabled via add_features):");
    for (const f of disabledList) lines.push(`  ${f.name}: ${f.description}`);
  }
  if (enabledList.length > 0) {
    lines.push("Currently ENABLED:");
    for (const f of enabledList) lines.push(`  ${f.name}: ${f.description}`);
  }
  return lines.join("\n");
}

function buildFamilyArchitectureContext(diagnostics: ComprehensiveModelDiagnostics): string {
  const lines: string[] = [];
  lines.push("Architecture guidance by family (use familyBias and error data to decide):");
  lines.push("  Hydrides: Eliashberg-mapped physics — GNN effective even with small data if graph features available; XGBoost needs explicit pressure features");
  lines.push("  Cuprates: Unconventional pairing — XGBoost with physical descriptors safer (d-wave, nesting, Mott proximity); GNN may overfit without domain features");
  lines.push("  Pnictides: Multi-band physics — ensemble (XGBoost+GNN) recommended; single model misses cross-band coupling");
  lines.push("  Borides: Sparse data, conventional-adjacent — XGBoost preferred; ensure phonon features are weighted");
  lines.push("  Conventional: Well-understood BCS — either model works, prioritize phonon and lambda features");
  const datasetSize = diagnostics.xgboost.datasetSize;
  if (datasetSize < 100) lines.push(`  Current dataset (${datasetSize} samples): Small — prefer XGBoost unless hydride-focused`);
  else if (datasetSize < 500) lines.push(`  Current dataset (${datasetSize} samples): Medium — XGBoost with larger ensemble, or begin GNN if graph data available`);
  else lines.push(`  Current dataset (${datasetSize} samples): Large — GNN or weighted ensemble viable`);
  return lines.join("\n");
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
  const report = diagnosticsReport ?? await getModelDiagnosticsForLLM();

  try {
    const currentDiagnostics = await getComprehensiveModelDiagnostics();
    const featureContext = buildFeatureContext();
    const archContext = buildFamilyArchitectureContext(currentDiagnostics);
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
- add_features: Enable computed features from the available list below. ONLY use feature names from this list.
  Use changes like {"enable_features": ["van_hove_distance", "dos_lambda_product"]}
  ${featureContext}
- adjust_architecture: Select model architecture based on family composition and dataset characteristics.
  ${archContext}
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

    const proposals = parsed
      .map((p: any) => {
        if (!p.model_target || !p.experiment_type || !p.changes || typeof p.changes !== "object" || !p.reasoning) return null;
        const normalizedTarget = normalizeModelTarget(p.model_target);
        if (!normalizedTarget) return null;
        if (!validTypes.includes(p.experiment_type)) return null;

        let finalChanges = p.changes;
        if (p.experiment_type === "adjust_hyperparameters") {
          const dsSize = normalizedTarget === "xgboost" ? currentDiagnostics.xgboost?.datasetSize
            : normalizedTarget === "gnn" ? currentDiagnostics.gnn?.datasetSize
            : normalizedTarget === "lambda-regressor" ? currentDiagnostics.lambda?.datasetSize
            : normalizedTarget === "phonon-surrogate" ? currentDiagnostics.phononSurrogate?.datasetSize
            : normalizedTarget === "tb-surrogate" ? currentDiagnostics.tbSurrogate?.datasetSize
            : undefined;
          const { clamped, warnings } = clampHyperparameters(normalizedTarget, p.changes, dsSize);
          if (warnings.length > 0) {
            console.log(`[Model Experiment Controller] Hyperparams clamped for ${normalizedTarget}: ${warnings.join("; ")}`);
          }
          finalChanges = clamped;
        }

        const proposal: ExperimentProposal = {
          model_target: normalizedTarget,
          experiment_type: p.experiment_type as ExperimentType,
          changes: finalChanges,
          reasoning: typeof p.reasoning === "string" ? p.reasoning : String(p.reasoning),
          expected_improvement: p.expected_improvement ?? "unknown",
          priority: Math.max(1, Math.min(3, Number(p.priority) || 2)),
        };

        proposal.technicalRequirements = extractTechnicalRequirements(proposal);
        return proposal;
      })
      .filter((p): p is ExperimentProposal => p !== null)
      .slice(0, 3);

    const deduped = proposals.filter(p => {
      if (isDuplicateExperiment(p)) {
        console.log(`[Model Experiment Controller] Skipped duplicate experiment: ${p.experiment_type} on ${p.model_target}`);
        return false;
      }
      return true;
    });

    return deduped;
  } catch (e) {
    console.log(`[Model Experiment Controller] LLM proposal failed: ${e instanceof Error ? e.message : "unknown"}`);
    return [];
  }
}

export async function executeExperiment(experiment: ExperimentProposal): Promise<ExperimentRecord> {
  const id = generateExperimentId();
  const diagnosticsBefore = await getComprehensiveModelDiagnostics();
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
      const execDatasetSize = beforeMetrics.datasetSize ?? undefined;
      const { clamped, warnings } = clampHyperparameters(experiment.model_target, experiment.changes, execDatasetSize);
      if (warnings.length > 0) {
        console.log(`[Model Experiment Controller] Execute-time clamp: ${warnings.join("; ")}`);
      }
      const overrides = hyperparamOverrides.get(experiment.model_target) ?? {};
      for (const [key, val] of Object.entries(clamped)) {
        if (typeof val === "number") {
          (overrides as any)[key] = val;
        }
      }
      hyperparamOverrides.set(experiment.model_target, overrides);
      console.log(`[Model Experiment Controller] Hyperparams updated for ${experiment.model_target}: ${JSON.stringify(overrides)}`);
    }

    if (experiment.experiment_type === "request_data") {
      const method = experiment.changes.method ?? "tb_evaluation";
      const maxCount = method === "qe_dft" ? 30 : method === "tb_evaluation" ? 200 : 50;
      const request: DataRequest = {
        id: `dr-${Date.now()}`,
        timestamp: Date.now(),
        family: experiment.changes.family ?? "unknown",
        count: Math.min(Math.max(1, experiment.changes.count ?? 20), maxCount),
        method,
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
      const availableNames = new Set(getAvailableFeatureDefinitions().map(f => f.name));
      let enabled = 0;
      const rejected: string[] = [];
      for (const featureName of featuresToEnable) {
        if (!availableNames.has(featureName)) {
          rejected.push(featureName);
          continue;
        }
        if (enableBuiltinFeature(featureName)) {
          enabled++;
        }
      }
      if (rejected.length > 0) {
        console.log(`[Model Experiment Controller] Rejected unknown features: ${rejected.join(", ")}`);
      }
      console.log(`[Model Experiment Controller] Enabled ${enabled}/${featuresToEnable.length} computed features (${rejected.length} rejected as unknown)`);
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
        const errMsg = e instanceof Error ? e.message : "unknown";
        record.changes = {
          ...record.changes,
          architecture_error: errMsg,
          architecture_result: null,
        };
        console.log(`[Model Experiment Controller] Architecture selection failed: ${errMsg}`);
      }
    }

    if (experiment.experiment_type !== "request_data") {
      switch (experiment.model_target) {
        case "xgboost":
          await retrainXGBoostFromEvaluated();
          break;
        case "lambda-regressor":
          await Promise.resolve(trainLambdaRegressor());
          break;
        case "phonon-surrogate":
          await Promise.resolve(trainPhononSurrogate());
          break;
        case "tb-surrogate":
          await Promise.resolve(retrainTBSurrogate());
          break;
        case "gnn":
          break;
        default:
          break;
      }
    }

    const diagnosticsAfter = await getComprehensiveModelDiagnostics();
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

  const improvementSums: Record<string, Record<string, number>> = {};
  const improvementCounts: Record<string, Record<string, number>> = {};
  for (const record of completed) {
    if (!improvementSums[record.target_model]) {
      improvementSums[record.target_model] = {};
      improvementCounts[record.target_model] = {};
    }
    for (const [key, val] of Object.entries(record.improvement)) {
      improvementSums[record.target_model][key] = (improvementSums[record.target_model][key] ?? 0) + val;
      improvementCounts[record.target_model][key] = (improvementCounts[record.target_model][key] ?? 0) + 1;
    }
  }
  const avgImprovementByModel: Record<string, Record<string, number>> = {};
  for (const model of Object.keys(improvementSums)) {
    avgImprovementByModel[model] = {};
    for (const key of Object.keys(improvementSums[model])) {
      const count = improvementCounts[model][key];
      avgImprovementByModel[model][key] = count > 0
        ? Math.round((improvementSums[model][key] / count) * 10000) / 10000
        : 0;
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
    recentExperiments: experimentRecords.slice(-10).reverse().map(r => ({
      id: r.id,
      timestamp: r.timestamp,
      type: r.type,
      target_model: r.target_model,
      status: r.status,
      improvement: r.improvement,
      completedAt: r.completedAt,
      error: r.error,
    })) as ExperimentRecord[],
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

export function getActiveTechnicalRequirements(): TechnicalRequirement[] {
  const recent = experimentRecords.slice(-10);
  const allReqs: TechnicalRequirement[] = [];
  const seen = new Set<string>();
  for (const rec of recent) {
    if (rec.status !== "completed" && rec.status !== "running") continue;
    const matchingProposal = experimentRecords.find(
      r => r.id === rec.id
    );
    if (!matchingProposal) continue;
  }
  for (const rec of recent) {
    if (rec.type === "request_data" && (rec.status === "completed" || rec.status === "running")) {
      const family = rec.changes.family ?? "unknown";
      const key = `data_need:${family}`;
      if (!seen.has(key)) {
        seen.add(key);
        allReqs.push({
          type: "data_need",
          family,
          detail: `Model requested ${rec.changes.count ?? 20} ${family} samples`,
          urgency: (rec.changes.count ?? 20) >= 40 ? "high" : "medium",
        });
      }
    }
    if (rec.type === "rebalance_training" && rec.status === "completed") {
      for (const [family, ratio] of Object.entries(rec.changes)) {
        if (typeof ratio === "number" && ratio > 0.3) {
          const key = `data_need:rebalance:${family}`;
          if (!seen.has(key)) {
            seen.add(key);
            allReqs.push({
              type: "data_need",
              family,
              detail: `Rebalancing requests ${(ratio * 100).toFixed(0)}% weight for ${family}`,
              urgency: ratio > 0.5 ? "high" : "medium",
            });
          }
        }
      }
    }
  }
  return allReqs;
}
