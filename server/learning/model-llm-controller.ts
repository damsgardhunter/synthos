import OpenAI from "openai";
import {
  getComprehensiveModelDiagnostics,
  getModelDiagnosticsForLLM,
  getFailedMaterialsForLLM,
  getBenchmarkForLLM,
  getFailureSummary,
  getModelBenchmark,
  type ModelVersionScorecard,
} from "./model-diagnostics";
import { getGlobalFeatureImportance } from "./gradient-boost";
import { extractFeatures, type MLFeatureVector } from "./ml-predictor";
import { getUncertaintyForLLM, proposeUncertaintyImprovements, type UncertaintyProposal } from "./uncertainty-tracker";
import { evaluateRetrainNeed, getSchedulerForLLM, type RetrainDecision } from "./retrain-scheduler";
import { getGroundTruthForLLM } from "./ground-truth-store";
import { getMetricsForLLM as getPredictionLedgerForLLM } from "./prediction-reality-ledger";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export interface FeatureProposal {
  name: string;
  formula: string;
  physicsRationale: string;
  inputFeatures: string[];
  computeLogic: string;
  expectedImpact: string;
  priority: number;
}

export interface ComputedFeatureDefinition {
  name: string;
  formula: string;
  computeFn: (features: MLFeatureVector) => number;
  createdAt: number;
  proposedBy: string;
  enabled: boolean;
  trainImpact?: { r2Delta: number; maeDelta: number };
}

export type ArchitectureRecommendation = {
  primaryModel: "xgboost" | "gnn" | "ensemble";
  reasoning: string;
  datasetCharacteristics: {
    size: number;
    featureCount: number;
    graphDataAvailable: boolean;
    diversityScore: number;
  };
  modelConfigs: {
    model: string;
    weight: number;
    rationale: string;
  }[];
  switchRecommended: boolean;
};

export interface ModelLLMReport {
  role: "model_llm";
  timestamp: number;
  diagnosticsSummary: string;
  failedMaterialsSummary: string;
  benchmarkSummary: string;
  uncertaintySummary: string;
  schedulerSummary: string;
  groundTruthSummary: string;
  featureProposals: FeatureProposal[];
  architectureRecommendation: ArchitectureRecommendation | null;
  uncertaintyProposals: UncertaintyProposal[];
  retrainDecision: RetrainDecision | null;
  activeCustomFeatures: string[];
}

const MAX_CUSTOM_FEATURES = 20;
const customFeatures: ComputedFeatureDefinition[] = [];
let architectureState: ArchitectureRecommendation | null = null;
let lastArchitectureCheck = 0;
const ARCHITECTURE_CHECK_INTERVAL_MS = 10 * 60 * 1000;
const ARCHITECTURE_SIZE_THRESHOLDS = [128, 256, 512, 1024, 2048, 4096];
let lastArchitectureDatasetBucket = -1;

function getDatasetSizeBucket(size: number): number {
  for (let i = ARCHITECTURE_SIZE_THRESHOLDS.length - 1; i >= 0; i--) {
    if (size >= ARCHITECTURE_SIZE_THRESHOLDS[i]) return i;
  }
  return -1;
}

function isUnconventionalProxy(f: MLFeatureVector): boolean {
  if (f.correlationStrength > 0.5 && f.mottProximityScore > 0.3) return true;
  if (f.dWaveSymmetry) return true;
  if (f.orbitalDFraction > 0.4 && f.spinFluctuationStrength > 0.3) return true;
  if (f.layeredStructure && f.correlationStrength > 0.3 && f.chargeTransferMagnitude > 0.3) return true;
  return false;
}

function pressureDistanceToOptimal(f: MLFeatureVector): number {
  const optimal = f.optimalPressureGpa;
  const current = f.pressureGpa;
  if (!Number.isFinite(optimal) || optimal <= 0) {
    return current > 0 ? Math.log1p(current / 50) : 0;
  }
  const distance = Math.abs(current - optimal);
  const normalizedDist = distance / Math.max(optimal, 1);
  return Math.exp(-normalizedDist);
}

const COMPUTABLE_FEATURES: Record<string, {
  compute: (f: MLFeatureVector) => number;
  description: string;
}> = {
  van_hove_distance: {
    compute: (f) => {
      const raw = f.vanHoveProximity;
      if (raw === undefined || raw === null || !Number.isFinite(raw)) return 999;
      return Math.abs(raw);
    },
    description: "|E_van_hove - E_F| distance to van Hove singularity",
  },
  lambda_over_omega: {
    compute: (f) => f.logPhononFreq > 0 ? f.electronPhononLambda / f.logPhononFreq : 0,
    description: "electron-phonon coupling per phonon frequency unit",
  },
  phonon_anharmonic_coupling: {
    compute: (f) => (f.anharmonicityFlag ? 1 : 0) * f.phononCouplingEstimate,
    description: "anharmonicity-weighted phonon coupling",
  },
  effective_coupling_strength: {
    compute: (f) => f.electronPhononLambda * (1 + f.massEnhancement) * (1 - f.muStarEstimate),
    description: "net pairing strength: lambda * (1+mass_enhancement) * (1-mu*)",
  },
  dos_lambda_product: {
    compute: (f) => f.dosAtEF * f.electronPhononLambda,
    description: "DOS(EF) * lambda: joint electronic-phonon indicator",
  },
  nesting_topology_score: {
    compute: (f) => f.nestingScore * (1 + f.topologicalBandScore),
    description: "Fermi surface nesting weighted by topological character",
  },
  phonon_softening_ratio: {
    compute: (f) => f.phononSofteningIndex > 0 ? f.softModeScore / Math.max(0.01, f.phononSofteningIndex) : 0,
    description: "soft mode score normalized by phonon softening index",
  },
  dimensionality_coupling: {
    compute: (f) => f.dimensionalityScoreV2 * f.cooperPairStrength,
    description: "dimensionality * Cooper pair strength: captures layered SC",
  },
  orbital_nesting_product: {
    compute: (f) => f.orbitalDFraction * f.fermiSurfaceNestingScore,
    description: "d-orbital fraction * FS nesting: d-wave pairing indicator",
  },
  pressure_lambda_interaction: {
    compute: (f) => f.electronPhononLambda * (1 + pressureDistanceToOptimal(f)),
    description: "lambda scaled by proximity to optimal pressure (Bayesian): peaks near P_opt, decays away",
  },
  charge_transfer_coupling: {
    compute: (f) => f.chargeTransferMagnitude * f.phononCouplingEstimate,
    description: "charge transfer * phonon coupling: ionic SC indicator",
  },
  mott_hubbard_proximity: {
    compute: (f) => f.mottProximityScore * f.correlationStrength,
    description: "Mott proximity * correlation: captures unconventional SC near Mott transition",
  },
  spin_phonon_competition: {
    compute: (f) => {
      if (f.spinFluctuationStrength <= 0) return f.electronPhononLambda;
      if (isUnconventionalProxy(f)) {
        return f.electronPhononLambda + f.spinFluctuationStrength * f.correlationStrength;
      }
      return f.electronPhononLambda / (1 + f.spinFluctuationStrength);
    },
    description: "family-aware: spin fluctuations boost unconventional SC (cuprate/heavy-fermion) but suppress conventional BCS",
  },
  band_flatness_dos_product: {
    compute: (f) => f.bandFlatness * f.dosAtEF,
    description: "flat band * DOS(EF): high-DOS flat band indicator",
  },
};

const DISTANCE_FEATURES = new Set(["van_hove_distance", "mott_hubbard_proximity"]);
const DISTANCE_PENALTY = 999;

function featureNaNFallback(name: string): number {
  return DISTANCE_FEATURES.has(name) ? DISTANCE_PENALTY : 0;
}

export function computeCustomFeature(name: string, features: MLFeatureVector): number | null {
  const builtin = COMPUTABLE_FEATURES[name];
  if (builtin) {
    try {
      const val = builtin.compute(features);
      return Number.isFinite(val) ? val : featureNaNFallback(name);
    } catch { return featureNaNFallback(name); }
  }

  const custom = customFeatures.find(cf => cf.name === name && cf.enabled);
  if (custom) {
    try {
      const val = custom.computeFn(features);
      return Number.isFinite(val) ? val : 0;
    } catch { return 0; }
  }

  return null;
}

export function getCustomFeatureValues(features: MLFeatureVector): Record<string, number> {
  const result: Record<string, number> = {};
  for (const cf of customFeatures.filter(c => c.enabled)) {
    result[cf.name] = computeCustomFeature(cf.name, features) ?? 0;
  }
  return result;
}

export function getActiveCustomFeatures(): ComputedFeatureDefinition[] {
  return customFeatures.filter(c => c.enabled);
}

export function getAvailableFeatureDefinitions(): { name: string; description: string }[] {
  return Object.entries(COMPUTABLE_FEATURES).map(([name, def]) => ({
    name,
    description: def.description,
  }));
}

let featureMutexLocked = false;

export function enableBuiltinFeature(name: string): boolean {
  if (featureMutexLocked) {
    console.log(`[Model LLM] Skipped feature enable for ${name}: concurrent modification in progress`);
    return false;
  }
  featureMutexLocked = true;
  try {
    const builtin = COMPUTABLE_FEATURES[name];
    if (!builtin) return false;

    const existing = customFeatures.find(cf => cf.name === name);
    if (existing) {
      existing.enabled = true;
      return true;
    }

    if (customFeatures.length >= MAX_CUSTOM_FEATURES) {
      const disabledIdx = customFeatures.findIndex(cf => !cf.enabled);
      if (disabledIdx !== -1) {
        const removed = customFeatures[disabledIdx];
        customFeatures.splice(disabledIdx, 1);
        console.log(`[Model LLM] Evicted disabled feature: ${removed.name} to make room for ${name}`);
      } else {
        return false;
      }
    }

    customFeatures.push({
      name,
      formula: builtin.description,
      computeFn: builtin.compute,
      createdAt: Date.now(),
      proposedBy: "model_llm",
      enabled: true,
    });

    console.log(`[Model LLM] Enabled computed feature: ${name} — ${builtin.description}`);
    return true;
  } finally {
    featureMutexLocked = false;
  }
}

export function disableCustomFeature(name: string): boolean {
  const feature = customFeatures.find(cf => cf.name === name);
  if (feature) {
    feature.enabled = false;
    return true;
  }
  return false;
}

export async function proposeNewFeatures(): Promise<FeatureProposal[]> {
  const featureImportance = getGlobalFeatureImportance(20);
  const failureSummary = getFailureSummary();
  const diagnostics = getComprehensiveModelDiagnostics();

  const availableFeatures = Object.entries(COMPUTABLE_FEATURES)
    .map(([name, def]) => `${name}: ${def.description}`)
    .join("\n");

  const activeFeatureList = customFeatures.filter(c => c.enabled);
  const activeFeatureNames = activeFeatureList.map(c => c.name);

  const activeFeatureImportances: string[] = [];
  for (const af of activeFeatureList) {
    const imp = featureImportance.find(f => f.name === af.name);
    const impStr = imp ? `importance=${imp.normalizedImportance.toFixed(3)}` : "importance=0.000 (NOT IN TOP-20)";
    const ageMinutes = Math.round((Date.now() - af.createdAt) / 60000);
    activeFeatureImportances.push(`  ${af.name}: ${impStr}, age=${ageMinutes}min`);
  }
  const activeFeatureReport = activeFeatureImportances.length > 0
    ? activeFeatureImportances.join("\n")
    : "  none active";

  const topFeatureStr = featureImportance.slice(0, 15)
    .map(f => `${f.name}: importance=${f.normalizedImportance.toFixed(3)}`)
    .join("\n");

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.3,
      max_tokens: 800,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `You are a materials science ML feature engineer for superconductor discovery.
Your job is to propose computed features that combine existing physics features to improve Tc prediction.

Current top features by importance:
${topFeatureStr}

Currently active custom features with their measured importance:
${activeFeatureReport}
NOTE: If an active feature has importance=0.000 or is NOT IN TOP-20, it is not helping. You should propose DISABLING it and enabling a different feature instead.

Available computable features (can be enabled):
${availableFeatures}

Model performance: XGBoost R²=${diagnostics.xgboost.r2}, MAE=${diagnostics.xgboost.mae}K
Failed materials: ${failureSummary.totalFailures} total, ${failureSummary.predictedStableActualUnstable.length} false-stable predictions
Top failing element pairs: ${failureSummary.failurePatterns.topFailingElementPairs.slice(0, 3).map(p => p.pair).join(", ") || "none"}

Propose 1-3 actions. Each action is either:
- "enable": enable a feature from the available list
- "disable": disable an active feature that has low/zero importance (replace with a better one)
Focus on:
- Features that combine underused physics signals
- Features that could help with the failure patterns observed
- Disabling features with zero measured importance before enabling new ones

Respond in JSON:
{
  "proposals": [
    {
      "action": "enable" or "disable",
      "name": "feature_name",
      "physicsRationale": "why enable/disable this feature",
      "expectedImpact": "what improvement to expect",
      "priority": 1
    }
  ]
}`,
        },
        {
          role: "user",
          content: "Propose features to enable or disable based on the current model state and failure patterns.",
        },
      ],
    });

    const content = response.choices[0]?.message?.content ?? "{}";
    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      console.log(`[Model LLM] Feature proposal JSON parse failed, content length=${content.length}`);
      return [];
    }
    const rawProposals = Array.isArray(parsed) ? parsed : Array.isArray(parsed.proposals) ? parsed.proposals : [];
    if (rawProposals.length === 0) return [];

    for (const p of rawProposals) {
      if (p.action === "disable" && p.name && activeFeatureNames.includes(p.name)) {
        disableCustomFeature(p.name);
        console.log(`[Model LLM] Disabled underperforming feature: ${p.name} — ${p.physicsRationale || "low importance"}`);
      }
    }

    const enableProposals = rawProposals.filter((p: any) => (p.action === "enable" || !p.action) && p.name && COMPUTABLE_FEATURES[p.name]);

    return enableProposals
      .slice(0, 3)
      .map((p: any) => ({
        name: p.name,
        formula: COMPUTABLE_FEATURES[p.name].description,
        physicsRationale: String(p.physicsRationale || ""),
        inputFeatures: [],
        computeLogic: COMPUTABLE_FEATURES[p.name].description,
        expectedImpact: String(p.expectedImpact || "unknown"),
        priority: Math.max(1, Math.min(3, Number(p.priority) || 2)),
      }));
  } catch (e) {
    console.log(`[Model LLM] Feature proposal failed: ${e instanceof Error ? e.message : "unknown"}`);
    return [];
  }
}

export async function selectArchitecture(): Promise<ArchitectureRecommendation> {
  const diagnostics = getComprehensiveModelDiagnostics();
  const benchmark = getModelBenchmark();

  const xgbDataset = diagnostics.xgboost.nSamples;
  const gnnDataset = diagnostics.gnn.datasetSize;
  const featureCount = diagnostics.xgboost.featureCount;
  const graphDataAvailable = gnnDataset > 0;

  const xgbCards = benchmark.scorecards.filter(s => s.modelName === "xgboost");
  const gnnCards = benchmark.scorecards.filter(s => s.modelName === "gnn");
  const latestXgb = xgbCards.length > 0 ? xgbCards[xgbCards.length - 1] : null;
  const latestGnn = gnnCards.length > 0 ? gnnCards[gnnCards.length - 1] : null;

  const familyDiversity = new Set(
    diagnostics.familyBias.filter(f => f.count > 0).map(f => f.family)
  ).size;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.2,
      max_tokens: 600,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `You are a model architecture advisor for ML-based superconductor discovery.
Select the best model architecture based on dataset characteristics and performance.

Available architectures:
1. xgboost: Gradient boosting ensemble. Best for tabular features, small-medium datasets (<1000 samples). Fast inference (~3ms).
2. gnn: Graph Neural Network ensemble. Best for crystal structure data, large datasets (>200 samples). Captures topology. Slower inference (~8ms).
3. ensemble: Weighted combination of both models. Best when both have adequate data. Most robust but slowest.

Current state:
- XGBoost: ${xgbDataset} samples, ${featureCount} features, R²=${diagnostics.xgboost.r2}, MAE=${diagnostics.xgboost.mae}K
- GNN: ${gnnDataset} samples, R²=${diagnostics.gnn.latestR2}, MAE=${diagnostics.gnn.latestMAE}K
- Family diversity: ${familyDiversity} families
- Graph data available: ${graphDataAvailable}
${latestXgb ? `- XGBoost latest v${latestXgb.version}: ${JSON.stringify(latestXgb.metrics)}` : ""}
${latestGnn ? `- GNN latest v${latestGnn.version}: ${JSON.stringify(latestGnn.metrics)}` : ""}

Rules:
- dataset < 100 samples: prefer xgboost (GNN needs more data)
- dataset > 500 samples with graph data: consider gnn or ensemble
- If one model has much better R² than the other, weight it higher in ensemble
- If both models have R² > 0.8, recommend ensemble with balanced weights

Respond in JSON:
{
  "primaryModel": "xgboost" | "gnn" | "ensemble",
  "reasoning": "why this architecture",
  "modelConfigs": [{"model": "xgboost", "weight": 0.6, "rationale": "..."}, ...],
  "switchRecommended": true/false
}`,
        },
        {
          role: "user",
          content: "Select the best architecture for the current dataset and model state.",
        },
      ],
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error("No response");

    const parsed = JSON.parse(content);
    const recommendation: ArchitectureRecommendation = {
      primaryModel: (["xgboost", "gnn", "ensemble"].includes(parsed.primaryModel) ? parsed.primaryModel : "xgboost") as any,
      reasoning: String(parsed.reasoning || ""),
      datasetCharacteristics: {
        size: xgbDataset,
        featureCount,
        graphDataAvailable,
        diversityScore: familyDiversity,
      },
      modelConfigs: Array.isArray(parsed.modelConfigs) ? parsed.modelConfigs.slice(0, 3).map((c: any) => ({
        model: String(c.model || "xgboost"),
        weight: Math.max(0, Math.min(1, Number(c.weight) || 0.5)),
        rationale: String(c.rationale || ""),
      })) : [{ model: "xgboost", weight: 1.0, rationale: "default" }],
      switchRecommended: Boolean(parsed.switchRecommended),
    };

    architectureState = recommendation;
    lastArchitectureCheck = Date.now();
    lastArchitectureDatasetBucket = getDatasetSizeBucket(xgbDataset);

    console.log(`[Model LLM] Architecture recommendation: ${recommendation.primaryModel} (switch=${recommendation.switchRecommended}, bucket=${lastArchitectureDatasetBucket})`);
    return recommendation;
  } catch (e) {
    console.log(`[Model LLM] Architecture selection failed: ${e instanceof Error ? e.message : "unknown"}`);
    const fallback: ArchitectureRecommendation = {
      primaryModel: xgbDataset > 500 && gnnDataset > 200 ? "ensemble" : "xgboost",
      reasoning: "Fallback: using dataset size heuristic",
      datasetCharacteristics: { size: xgbDataset, featureCount, graphDataAvailable, diversityScore: familyDiversity },
      modelConfigs: [{ model: "xgboost", weight: 0.7, rationale: "default primary" }],
      switchRecommended: false,
    };
    architectureState = fallback;
    lastArchitectureCheck = Date.now();
    lastArchitectureDatasetBucket = getDatasetSizeBucket(xgbDataset);
    return fallback;
  }
}

export function getCurrentArchitecture(): ArchitectureRecommendation | null {
  return architectureState;
}

export function shouldReassessArchitecture(): boolean {
  if (Date.now() - lastArchitectureCheck < ARCHITECTURE_CHECK_INTERVAL_MS) return false;

  try {
    const diagnostics = getComprehensiveModelDiagnostics();
    const totalSize = Math.max(diagnostics.xgboost.nSamples, diagnostics.gnn.datasetSize);
    const currentBucket = getDatasetSizeBucket(totalSize);
    if (currentBucket > lastArchitectureDatasetBucket) {
      console.log(`[Model LLM] Dataset crossed threshold: bucket ${lastArchitectureDatasetBucket} -> ${currentBucket} (size=${totalSize}, threshold=${ARCHITECTURE_SIZE_THRESHOLDS[currentBucket]})`);
      return true;
    }
  } catch (_e) {}

  if (architectureState === null) return true;

  return false;
}

export async function runModelLLMCycle(currentCycle: number): Promise<ModelLLMReport> {
  const diagnosticsSummary = getModelDiagnosticsForLLM();
  const failedMaterialsSummary = getFailedMaterialsForLLM();
  const benchmarkSummary = getBenchmarkForLLM();
  const uncertaintySummary = getUncertaintyForLLM();
  const schedulerSummary = getSchedulerForLLM();
  const groundTruthSummary = getGroundTruthForLLM();
  const predictionLedgerSummary = getPredictionLedgerForLLM();

  let featureProposals: FeatureProposal[] = [];
  try {
    featureProposals = await proposeNewFeatures();
    for (const proposal of featureProposals) {
      enableBuiltinFeature(proposal.name);
    }
  } catch (e) {
    console.log(`[Model LLM] Feature proposal cycle failed: ${e instanceof Error ? e.message : "unknown"}`);
  }

  let architectureRecommendation: ArchitectureRecommendation | null = null;
  if (shouldReassessArchitecture()) {
    try {
      architectureRecommendation = await selectArchitecture();
    } catch (e) {
      console.log(`[Model LLM] Architecture selection cycle failed: ${e instanceof Error ? e.message : "unknown"}`);
    }
  }

  let uncertaintyProposals: UncertaintyProposal[] = [];
  try {
    uncertaintyProposals = await proposeUncertaintyImprovements();
  } catch (e) {
    console.log(`[Model LLM] Uncertainty proposal cycle failed: ${e instanceof Error ? e.message : "unknown"}`);
  }

  let retrainDecision: RetrainDecision | null = null;
  try {
    retrainDecision = await evaluateRetrainNeed();
  } catch (e) {
    console.log(`[Model LLM] Retrain scheduling failed: ${e instanceof Error ? e.message : "unknown"}`);
  }

  return {
    role: "model_llm",
    timestamp: Date.now(),
    diagnosticsSummary,
    failedMaterialsSummary,
    benchmarkSummary,
    uncertaintySummary,
    schedulerSummary,
    groundTruthSummary,
    predictionLedgerSummary,
    featureProposals,
    architectureRecommendation,
    uncertaintyProposals,
    retrainDecision,
    activeCustomFeatures: customFeatures.filter(c => c.enabled).map(c => c.name),
  };
}

export function getModelLLMStatus(): {
  activeFeatures: { name: string; formula: string; createdAt: number; enabled: boolean }[];
  availableFeatures: { name: string; description: string }[];
  currentArchitecture: ArchitectureRecommendation | null;
  lastArchitectureCheck: number;
  uncertaintySummary: string;
  schedulerSummary: string;
} {
  return {
    activeFeatures: customFeatures.map(cf => ({
      name: cf.name,
      formula: cf.formula,
      createdAt: cf.createdAt,
      enabled: cf.enabled,
    })),
    availableFeatures: getAvailableFeatureDefinitions(),
    currentArchitecture: architectureState,
    lastArchitectureCheck,
    uncertaintySummary: getUncertaintyForLLM(),
    schedulerSummary: getSchedulerForLLM(),
  };
}
