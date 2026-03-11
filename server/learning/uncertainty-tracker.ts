import OpenAI from "openai";
import { getCalibrationData, getXGBEnsembleStats, getModelVersionHistory } from "./gradient-boost";
import { getComprehensiveModelDiagnostics } from "./model-diagnostics";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export interface PredictionVarianceEntry {
  formula: string;
  predictedTc: number;
  ensembleStd: number;
  normalizedUncertainty: number;
  confidenceLower: number;
  confidenceUpper: number;
  timestamp: number;
  family?: string;
  referenceTc?: number | null;
}

export interface CalibrationPoint {
  binCenter: number;
  expectedFrequency: number;
  observedFrequency: number;
  count: number;
  gap: number;
}

export interface CalibrationCurve {
  points: CalibrationPoint[];
  ece: number;
  mce: number;
  overconfidentBins: number;
  underconfidentBins: number;
  totalPredictions: number;
  timestamp: number;
}

export interface UncertaintyDecomposition {
  aleatoric: number;
  epistemic: number;
  total: number;
  dominantSource: "aleatoric" | "epistemic" | "balanced";
}

export interface UncertaintyProposal {
  type: "ensemble_expansion" | "mc_dropout" | "calibration_adjustment" | "data_augmentation" | "heteroscedastic_output";
  reasoning: string;
  expectedImprovement: string;
  priority: number;
  parameters: Record<string, number | string>;
}

export interface UncertaintyReport {
  calibrationCurve: CalibrationCurve;
  varianceSummary: {
    meanVariance: number;
    maxVariance: number;
    highUncertaintyCount: number;
    highUncertaintyFraction: number;
    decomposition: UncertaintyDecomposition;
  };
  proposals: UncertaintyProposal[];
  timestamp: number;
}

const MAX_VARIANCE_HISTORY = 2000;
const varianceBuffer: PredictionVarianceEntry[] = new Array(MAX_VARIANCE_HISTORY);
let vbHead = 0;
let vbSize = 0;

function vbPush(entry: PredictionVarianceEntry): void {
  varianceBuffer[vbHead] = entry;
  vbHead = (vbHead + 1) % MAX_VARIANCE_HISTORY;
  if (vbSize < MAX_VARIANCE_HISTORY) vbSize++;
}

function vbRecent(n: number): PredictionVarianceEntry[] {
  const count = Math.min(n, vbSize);
  const result: PredictionVarianceEntry[] = new Array(count);
  for (let i = 0; i < count; i++) {
    const idx = (vbHead - count + i + MAX_VARIANCE_HISTORY) % MAX_VARIANCE_HISTORY;
    result[i] = varianceBuffer[idx];
  }
  return result;
}

const calibrationHistory: CalibrationCurve[] = [];
let lastProposalTime = 0;
const PROPOSAL_INTERVAL_MS = 15 * 60 * 1000;

export function recordPredictionVariance(entry: PredictionVarianceEntry): void {
  vbPush(entry);
}

export function recordBatchVariance(entries: PredictionVarianceEntry[]): void {
  for (const e of entries) {
    vbPush(e);
  }
}

export function computeCalibrationCurve(): CalibrationCurve {
  const recent = vbRecent(500);
  const nBins = 10;
  const bins: { predicted: number[]; actual: number[] }[] = Array.from({ length: nBins }, () => ({ predicted: [], actual: [] }));

  for (const entry of recent) {
    const conf = 1 - Math.min(1, entry.normalizedUncertainty);
    const binIdx = Math.min(nBins - 1, Math.floor(conf * nBins));
    bins[binIdx].predicted.push(conf);

    let actual: number;
    if (entry.referenceTc != null && Number.isFinite(entry.referenceTc)) {
      const error = Math.abs(entry.predictedTc - entry.referenceTc);
      const threshold = Math.max(Math.abs(entry.referenceTc * 0.15), 5);
      actual = error < threshold ? 1 : 0;
    } else {
      const ciHalfWidth = 1.645 * entry.ensembleStd;
      actual = ciHalfWidth > 0 && entry.normalizedUncertainty < 0.5 ? 1 : 0;
    }
    bins[binIdx].actual.push(actual);
  }

  const points: CalibrationPoint[] = [];
  let eceSum = 0;
  let mce = 0;
  let overconfident = 0;
  let underconfident = 0;

  for (let i = 0; i < nBins; i++) {
    const binCenter = (i + 0.5) / nBins;
    const count = bins[i].predicted.length;
    if (count === 0) {
      points.push({ binCenter, expectedFrequency: binCenter, observedFrequency: 0, count: 0, gap: 0 });
      continue;
    }

    const expectedFreq = bins[i].predicted.reduce((a, b) => a + b, 0) / count;
    const observedFreq = bins[i].actual.reduce((a, b) => a + b, 0) / count;
    const gap = Math.abs(expectedFreq - observedFreq);

    eceSum += gap * (count / recent.length);
    mce = Math.max(mce, gap);

    if (expectedFreq > observedFreq + 0.05) overconfident++;
    else if (observedFreq > expectedFreq + 0.05) underconfident++;

    points.push({ binCenter, expectedFrequency: expectedFreq, observedFrequency: observedFreq, count, gap });
  }

  const curve: CalibrationCurve = {
    points,
    ece: eceSum,
    mce,
    overconfidentBins: overconfident,
    underconfidentBins: underconfident,
    totalPredictions: recent.length,
    timestamp: Date.now(),
  };

  calibrationHistory.push(curve);
  if (calibrationHistory.length > 50) calibrationHistory.splice(0, 1);

  return curve;
}

export function getVarianceSummary(): {
  meanVariance: number;
  maxVariance: number;
  highUncertaintyCount: number;
  highUncertaintyFraction: number;
  decomposition: UncertaintyDecomposition;
} {
  const recent = vbRecent(200);
  if (recent.length === 0) {
    return {
      meanVariance: 0,
      maxVariance: 0,
      highUncertaintyCount: 0,
      highUncertaintyFraction: 0,
      decomposition: { aleatoric: 0, epistemic: 0, total: 0, dominantSource: "balanced" },
    };
  }

  const variances = recent.map(e => e.ensembleStd ** 2);
  const meanVar = variances.reduce((a, b) => a + b, 0) / variances.length;
  const maxVar = Math.max(...variances);

  const highUncThreshold = 0.3;
  let highUncCount = 0;
  for (const e of recent) {
    if (e.normalizedUncertainty > highUncThreshold) highUncCount++;
  }

  const MIN_FAMILY_SIZE = 5;
  const familyGroups: Record<string, number[]> = {};
  for (const e of recent) {
    const fam = e.family || "unknown";
    if (fam === "other" || fam === "unknown") continue;
    if (!familyGroups[fam]) familyGroups[fam] = [];
    familyGroups[fam].push(e.ensembleStd ** 2);
  }

  let withinGroupVar = 0;
  let totalWeights = 0;
  for (const fam of Object.values(familyGroups)) {
    if (fam.length < MIN_FAMILY_SIZE) continue;
    const famMean = fam.reduce((a, b) => a + b, 0) / fam.length;
    const famVar = fam.reduce((a, b) => a + (b - famMean) ** 2, 0) / fam.length;
    withinGroupVar += famVar * fam.length;
    totalWeights += fam.length;
  }

  const aleatoric = totalWeights > 0 ? withinGroupVar / totalWeights : meanVar * 0.5;
  const epistemic = Math.max(0, meanVar - aleatoric);
  const total = Math.max(aleatoric + epistemic, 1e-12);

  let dominantSource: "aleatoric" | "epistemic" | "balanced" = "balanced";
  if (aleatoric / total > 0.65) dominantSource = "aleatoric";
  else if (epistemic / total > 0.65) dominantSource = "epistemic";

  return {
    meanVariance: meanVar,
    maxVariance: maxVar,
    highUncertaintyCount: highUncCount,
    highUncertaintyFraction: highUncCount / recent.length,
    decomposition: { aleatoric, epistemic, total, dominantSource },
  };
}

export function getHighUncertaintyPredictions(topN: number = 20): PredictionVarianceEntry[] {
  return vbRecent(500)
    .sort((a, b) => b.normalizedUncertainty - a.normalizedUncertainty)
    .slice(0, topN);
}

export function getVarianceByFamily(): Record<string, { count: number; meanStd: number; meanNormalized: number }> {
  const groups: Record<string, PredictionVarianceEntry[]> = {};
  for (const e of vbRecent(500)) {
    const fam = e.family || "unknown";
    if (!groups[fam]) groups[fam] = [];
    groups[fam].push(e);
  }

  const result: Record<string, { count: number; meanStd: number; meanNormalized: number }> = {};
  for (const [fam, entries] of Object.entries(groups)) {
    result[fam] = {
      count: entries.length,
      meanStd: entries.reduce((a, e) => a + e.ensembleStd, 0) / entries.length,
      meanNormalized: entries.reduce((a, e) => a + e.normalizedUncertainty, 0) / entries.length,
    };
  }
  return result;
}

export async function proposeUncertaintyImprovements(): Promise<UncertaintyProposal[]> {
  if (Date.now() - lastProposalTime < PROPOSAL_INTERVAL_MS && lastProposalTime > 0) {
    return [];
  }

  const calibration = computeCalibrationCurve();
  const variance = getVarianceSummary();
  const ensembleStats = getXGBEnsembleStats();
  const diagnostics = getComprehensiveModelDiagnostics();
  const familyVariance = getVarianceByFamily();

  const familyLines = Object.entries(familyVariance)
    .sort((a, b) => b[1].meanNormalized - a[1].meanNormalized)
    .slice(0, 5)
    .map(([f, s]) => `  ${f}: mean_std=${s.meanStd.toFixed(2)}, normalized=${s.meanNormalized.toFixed(3)}, n=${s.count}`)
    .join("\n");

  const calLines = calibration.points
    .filter(p => p.count > 0)
    .map(p => `  bin=${p.binCenter.toFixed(1)}: expected=${p.expectedFrequency.toFixed(3)}, observed=${p.observedFrequency.toFixed(3)}, gap=${p.gap.toFixed(3)}, n=${p.count}`)
    .join("\n");

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.3,
      max_tokens: 700,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `You are an ML uncertainty quantification specialist for superconductor Tc prediction.
Analyze the uncertainty metrics and propose improvements.

Current ensemble: ${ensembleStats.ensembleSize} models, bootstrap ratio=${ensembleStats.bootstrapRatio}
Model R²=${diagnostics.xgboost.r2}, MAE=${diagnostics.xgboost.mae}K, dataset=${diagnostics.xgboost.nSamples} samples

Calibration (ECE=${calibration.ece.toFixed(4)}, MCE=${calibration.mce.toFixed(4)}):
${calLines || "  No calibration data"}
Overconfident bins: ${calibration.overconfidentBins}, Underconfident bins: ${calibration.underconfidentBins}

Variance summary:
  Mean variance: ${variance.meanVariance.toFixed(4)}
  High-uncertainty fraction: ${(variance.highUncertaintyFraction * 100).toFixed(1)}%
  Aleatoric: ${variance.decomposition.aleatoric.toFixed(4)}, Epistemic: ${variance.decomposition.epistemic.toFixed(4)}
  Dominant source: ${variance.decomposition.dominantSource}

Per-family variance:
${familyLines || "  No family data"}

Available improvement types:
1. ensemble_expansion: Increase ensemble size (current=${ensembleStats.ensembleSize}). Parameters: new_size (5-20)
2. mc_dropout: Add MC dropout passes for epistemic uncertainty. Parameters: n_passes (5-30), dropout_rate (0.05-0.3)
3. calibration_adjustment: Apply temperature scaling or Platt scaling. Parameters: method ("temperature"|"platt"), temperature (0.5-2.0)
4. data_augmentation: Add noise-augmented training to reduce aleatoric uncertainty. Parameters: noise_fraction (0.01-0.1), n_augmentations (2-10)
5. heteroscedastic_output: Add a second output head to predict per-sample variance. Parameters: variance_weight (0.1-1.0)

Respond with JSON:
{
  "proposals": [
    {
      "type": "ensemble_expansion"|"mc_dropout"|"calibration_adjustment"|"data_augmentation"|"heteroscedastic_output",
      "reasoning": "why this improvement",
      "expectedImprovement": "what metric improves",
      "priority": 1-3,
      "parameters": {}
    }
  ]
}

Propose 1-3 improvements. Focus on the dominant uncertainty source. If ECE > 0.1, prioritize calibration.`,
        },
        { role: "user", content: "Propose uncertainty improvements based on the current model state." },
      ],
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return [];

    const parsed = JSON.parse(content);
    if (!Array.isArray(parsed.proposals)) return [];

    lastProposalTime = Date.now();

    const validTypes = ["ensemble_expansion", "mc_dropout", "calibration_adjustment", "data_augmentation", "heteroscedastic_output"];
    return parsed.proposals
      .filter((p: any) => validTypes.includes(p.type))
      .slice(0, 3)
      .map((p: any) => ({
        type: p.type,
        reasoning: String(p.reasoning || ""),
        expectedImprovement: String(p.expectedImprovement || ""),
        priority: Math.max(1, Math.min(3, Number(p.priority) || 2)),
        parameters: typeof p.parameters === "object" && p.parameters !== null ? p.parameters : {},
      }));
  } catch (e) {
    console.log(`[Uncertainty] LLM proposal failed: ${e instanceof Error ? e.message : "unknown"}`);
    return [];
  }
}

export function getUncertaintyReport(): UncertaintyReport {
  const calibrationCurve = computeCalibrationCurve();
  const varianceSummary = getVarianceSummary();

  return {
    calibrationCurve,
    varianceSummary,
    proposals: [],
    timestamp: Date.now(),
  };
}

export async function getFullUncertaintyReport(): Promise<UncertaintyReport> {
  const calibrationCurve = computeCalibrationCurve();
  const varianceSummary = getVarianceSummary();
  const proposals = await proposeUncertaintyImprovements();

  return {
    calibrationCurve,
    varianceSummary,
    proposals,
    timestamp: Date.now(),
  };
}

export function getUncertaintyForLLM(): string {
  const variance = getVarianceSummary();
  const calibration = calibrationHistory.length > 0 ? calibrationHistory[calibrationHistory.length - 1] : null;
  const byFamily = getVarianceByFamily();

  const lines: string[] = [
    "=== Uncertainty Tracking ===",
    `Predictions tracked: ${vbSize}`,
    `Mean variance: ${variance.meanVariance.toFixed(4)}`,
    `High uncertainty fraction: ${(variance.highUncertaintyFraction * 100).toFixed(1)}%`,
    `Decomposition: aleatoric=${variance.decomposition.aleatoric.toFixed(4)}, epistemic=${variance.decomposition.epistemic.toFixed(4)} (${variance.decomposition.dominantSource})`,
  ];

  if (calibration) {
    lines.push(`Calibration ECE: ${calibration.ece.toFixed(4)}, MCE: ${calibration.mce.toFixed(4)}`);
    lines.push(`Overconfident: ${calibration.overconfidentBins} bins, Underconfident: ${calibration.underconfidentBins} bins`);
  }

  const sortedFamilies = Object.entries(byFamily)
    .sort((a, b) => b[1].meanNormalized - a[1].meanNormalized)
    .slice(0, 5);
  if (sortedFamilies.length > 0) {
    lines.push("Top uncertain families:");
    for (const [fam, s] of sortedFamilies) {
      lines.push(`  ${fam}: std=${s.meanStd.toFixed(2)}, n=${s.count}`);
    }
  }

  return lines.join("\n");
}

export function getUncertaintyStatus(): {
  totalTracked: number;
  varianceSummary: ReturnType<typeof getVarianceSummary>;
  latestCalibration: CalibrationCurve | null;
  byFamily: ReturnType<typeof getVarianceByFamily>;
  highUncertaintyPredictions: PredictionVarianceEntry[];
} {
  return {
    totalTracked: vbSize,
    varianceSummary: getVarianceSummary(),
    latestCalibration: calibrationHistory.length > 0 ? calibrationHistory[calibrationHistory.length - 1] : null,
    byFamily: getVarianceByFamily(),
    highUncertaintyPredictions: getHighUncertaintyPredictions(10),
  };
}
