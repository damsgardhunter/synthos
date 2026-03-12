import { getLedgerSlice, getLedgerSize, onLedgerEntry, type PredictionRealityEntry } from "./prediction-reality-ledger";
import { gnnPredictWithUncertainty } from "./graph-neural-net";
import { gbPredictWithUncertainty } from "./gradient-boost";
import { extractFeatures } from "./ml-predictor";

export interface ConformalInterval {
  lower: number;
  upper: number;
  coverage: number;
  quantile: number;
  temperatureScaledSigma: number;
  rawSigma: number;
  method: "conformal" | "fallback";
}

export interface FamilyQuantiles {
  family: string;
  q90: number;
  q95: number;
  q99: number;
  count: number;
  ece: number;
}

export interface CalibrationState {
  temperatureScale: number;
  conformalQ90: number;
  conformalQ95: number;
  conformalQ99: number;
  calibrationDatasetSize: number;
  eceBefore: number;
  eceAfter: number;
  mce: number;
  lastCalibrationTimestamp: number;
  lastCalibrationLedgerSize: number;
  perFamily: FamilyQuantiles[];
  coverageAtQ95: number;
  meanNonconformityScore: number;
  medianNonconformityScore: number;
}

interface CalibrationEntry {
  formula: string;
  family: string;
  predictedTc: number;
  actualTc: number;
  predictedSigma: number;
  nonconformityScore: number;
  absError: number;
}

const MIN_CALIBRATION_SAMPLES = 10;
const RECALIBRATION_INTERVAL = 20;

let temperatureScale = 1.0;
let conformalQ90 = 1.645;
let conformalQ95 = 1.96;
let conformalQ99 = 2.576;
let calibrationDataset: CalibrationEntry[] = [];
let eceBefore = 0;
let eceAfter = 0;
let mceValue = 0;
let lastCalibrationTimestamp = 0;
let lastCalibrationLedgerSize = 0;
let familyQuantiles = new Map<string, { q90: number; q95: number; q99: number; count: number; ece: number }>();
let coverageAtQ95 = 0.95;
let meanNCS = 0;
let medianNCS = 0;
let samplesSinceLastCalibration = 0;

function classifyFamily(formula: string): string {
  const f = formula.toLowerCase();
  if (f.includes("h") && /\d/.test(f)) {
    const hMatch = f.match(/h(\d+)/);
    if (hMatch && parseInt(hMatch[1]) >= 3) return "hydride";
  }
  if (f.includes("cu") && f.includes("o")) return "cuprate";
  if (f.includes("fe") && (f.includes("as") || f.includes("se"))) return "pnictide";
  if (f.includes("nb") || f.includes("v3")) return "a15";
  if (f.includes("mg") && f.includes("b")) return "mgb2-type";
  if (f.includes("bi") && f.includes("s")) return "chalcogenide";
  return "other";
}

function computeQuantile(sortedValues: number[], alpha: number): number {
  const n = sortedValues.length;
  if (n === 0) return 1.96;
  const idx = Math.ceil((n + 1) * (1 - alpha)) - 1;
  const clampedIdx = Math.max(0, Math.min(n - 1, idx));
  return sortedValues[clampedIdx];
}

function computeECE(entries: CalibrationEntry[], tempScale: number, nBins = 10): number {
  if (entries.length < 5) return 1.0;

  const coverageLevels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
  let eceSum = 0;

  const scaledNCS = entries
    .map(e => e.absError / Math.max(1e-4, e.predictedSigma * tempScale))
    .sort((a, b) => a - b);

  for (const targetCoverage of coverageLevels) {
    const confQ = computeQuantile(scaledNCS, 1 - targetCoverage);
    let covered = 0;
    for (const e of entries) {
      const sigma = Math.max(1e-4, e.predictedSigma * tempScale);
      if (Math.abs(e.predictedTc - e.actualTc) <= confQ * sigma) {
        covered++;
      }
    }
    const observedCoverage = covered / entries.length;
    eceSum += Math.abs(observedCoverage - targetCoverage);
  }

  return eceSum / coverageLevels.length;
}

function normalQuantile(p: number): number {
  if (p <= 0.5) return -normalQuantile(1 - p);
  if (p >= 0.999) return 3.09;
  if (p >= 0.995) return 2.576;
  if (p >= 0.99) return 2.326;
  if (p >= 0.975) return 1.96;
  if (p >= 0.95) return 1.645;
  if (p >= 0.9) return 1.282;
  if (p >= 0.85) return 1.036;
  if (p >= 0.8) return 0.842;
  if (p >= 0.75) return 0.674;
  if (p >= 0.7) return 0.524;
  if (p >= 0.6) return 0.253;
  return 0;
}

function getPredictedSigmaForEntry(entry: PredictionRealityEntry): number {
  if (entry.predicted_sigma != null && Number.isFinite(entry.predicted_sigma) && entry.predicted_sigma > 0) {
    return entry.predicted_sigma;
  }

  try {
    const formula = entry.formula;

    let gnnSigma = 0;
    let xgbSigma = 0;

    try {
      const gnnResult = gnnPredictWithUncertainty(formula);
      if (Number.isFinite(gnnResult.totalStd) && gnnResult.totalStd > 0) {
        gnnSigma = gnnResult.totalStd;
      }
    } catch {}

    try {
      const features = extractFeatures(formula);
      const xgbResult = gbPredictWithUncertainty(features, formula);
      if (Number.isFinite(xgbResult.totalStd) && xgbResult.totalStd > 0) {
        xgbSigma = xgbResult.totalStd;
      }
    } catch {}

    if (gnnSigma > 0 && xgbSigma > 0) {
      const wGnn = 1 / (gnnSigma ** 2);
      const wXgb = 1 / (xgbSigma ** 2);
      return Math.sqrt(1 / (wGnn + wXgb));
    }
    if (xgbSigma > 0) return xgbSigma;
    if (gnnSigma > 0) return gnnSigma;

    return Math.max(10, Math.abs(entry.model_prediction.Tc) * 0.3);
  } catch {
    return Math.max(10, Math.abs(entry.model_prediction.Tc) * 0.3);
  }
}

const MAX_CALIBRATION_WINDOW = 1000;

function buildCalibrationDataset(): CalibrationEntry[] {
  const ledgerSize = getLedgerSize();
  if (ledgerSize === 0) return [];

  const startIdx = Math.max(0, ledgerSize - MAX_CALIBRATION_WINDOW);
  const entries = getLedgerSlice(startIdx, ledgerSize);
  const calibEntries: CalibrationEntry[] = [];

  for (const entry of entries) {
    if (!Number.isFinite(entry.model_prediction.Tc) || !Number.isFinite(entry.ground_truth.Tc)) continue;

    const predictedSigma = getPredictedSigmaForEntry(entry);
    if (!Number.isFinite(predictedSigma) || predictedSigma <= 0) continue;

    const absError = Math.abs(entry.model_prediction.Tc - entry.ground_truth.Tc);
    const ncs = absError / Math.max(1e-4, predictedSigma);

    calibEntries.push({
      formula: entry.formula,
      family: classifyFamily(entry.formula),
      predictedTc: entry.model_prediction.Tc,
      actualTc: entry.ground_truth.Tc,
      predictedSigma,
      nonconformityScore: ncs,
      absError,
    });
  }

  return calibEntries;
}

function fitTemperatureScale(entries: CalibrationEntry[]): number {
  if (entries.length < MIN_CALIBRATION_SAMPLES) return 1.0;

  let bestT = 1.0;
  let bestNLL = Infinity;

  for (let t = 0.1; t <= 10.0; t += 0.1) {
    let nll = 0;
    for (const e of entries) {
      const sigma = Math.max(1e-8, e.predictedSigma * t);
      const z = (e.predictedTc - e.actualTc) / sigma;
      nll += 0.5 * z * z + Math.log(sigma);
    }
    nll /= entries.length;

    if (nll < bestNLL) {
      bestNLL = nll;
      bestT = t;
    }
  }

  return Math.round(bestT * 100) / 100;
}

export function recalibrateFromLedger(): CalibrationState {
  const entries = buildCalibrationDataset();
  calibrationDataset = entries;

  if (entries.length < MIN_CALIBRATION_SAMPLES) {
    lastCalibrationTimestamp = Date.now();
    lastCalibrationLedgerSize = getLedgerSize();
    samplesSinceLastCalibration = 0;
    return getCalibrationState();
  }

  eceBefore = computeECE(entries, 1.0);

  temperatureScale = fitTemperatureScale(entries);

  const SIGMA_FLOOR = 1e-4;
  const scaledScores = entries.map(e => e.absError / Math.max(SIGMA_FLOOR, e.predictedSigma * temperatureScale));
  scaledScores.sort((a, b) => a - b);

  conformalQ90 = computeQuantile(scaledScores, 0.10);
  conformalQ95 = computeQuantile(scaledScores, 0.05);
  conformalQ99 = computeQuantile(scaledScores, 0.01);

  eceAfter = computeECE(entries, temperatureScale);

  let covered95 = 0;
  for (const e of entries) {
    const sigma = Math.max(SIGMA_FLOOR, e.predictedSigma * temperatureScale);
    if (Math.abs(e.predictedTc - e.actualTc) <= conformalQ95 * sigma) {
      covered95++;
    }
  }
  coverageAtQ95 = entries.length > 0 ? covered95 / entries.length : 0.95;

  meanNCS = scaledScores.reduce((s, v) => s + v, 0) / scaledScores.length;
  const midIdx = Math.floor(scaledScores.length / 2);
  medianNCS = scaledScores.length % 2 === 0
    ? (scaledScores[midIdx - 1] + scaledScores[midIdx]) / 2
    : scaledScores[midIdx];

  let maxCalGap = 0;
  const binCount = 10;
  for (let b = 0; b < binCount; b++) {
    const targetCov = (b + 1) / binCount;
    const confQ = computeQuantile(scaledScores, 1 - targetCov);
    let covered = 0;
    for (const e of entries) {
      const sigma = Math.max(SIGMA_FLOOR, e.predictedSigma * temperatureScale);
      if (Math.abs(e.predictedTc - e.actualTc) <= confQ * sigma) covered++;
    }
    const obsCov = covered / entries.length;
    maxCalGap = Math.max(maxCalGap, Math.abs(obsCov - targetCov));
  }
  mceValue = maxCalGap;

  familyQuantiles.clear();
  const byFamily = new Map<string, CalibrationEntry[]>();
  for (const e of entries) {
    const arr = byFamily.get(e.family) || [];
    arr.push(e);
    byFamily.set(e.family, arr);
  }

  const MIN_FAMILY_SAMPLES = 20;
  for (const [family, famEntries] of byFamily) {
    if (famEntries.length < MIN_FAMILY_SAMPLES) continue;
    const famScores = famEntries
      .map(e => e.absError / Math.max(SIGMA_FLOOR, e.predictedSigma * temperatureScale))
      .sort((a, b) => a - b);

    const famECE = computeECE(famEntries, temperatureScale);

    familyQuantiles.set(family, {
      q90: computeQuantile(famScores, 0.10),
      q95: computeQuantile(famScores, 0.05),
      q99: computeQuantile(famScores, 0.01),
      count: famEntries.length,
      ece: famECE,
    });
  }

  lastCalibrationTimestamp = Date.now();
  lastCalibrationLedgerSize = getLedgerSize();
  samplesSinceLastCalibration = 0;

  console.log(`[Conformal] Calibrated on ${entries.length} samples: T=${temperatureScale}, Q95=${conformalQ95.toFixed(3)}, ECE=${eceBefore.toFixed(4)}->${eceAfter.toFixed(4)}, coverage@95%=${(coverageAtQ95 * 100).toFixed(1)}%`);

  return getCalibrationState();
}

export function getConformalInterval(
  predictedTc: number,
  predictedSigma: number,
  coverage: number = 0.95,
  family?: string
): ConformalInterval {
  if (!Number.isFinite(predictedTc) || !Number.isFinite(predictedSigma) || predictedSigma <= 0) {
    const fallbackSigma = Math.max(10, Math.abs(predictedTc || 0) * 0.3);
    return {
      lower: Math.max(0, (predictedTc || 0) - 1.96 * fallbackSigma),
      upper: (predictedTc || 0) + 1.96 * fallbackSigma,
      coverage,
      quantile: 1.96,
      temperatureScaledSigma: fallbackSigma,
      rawSigma: fallbackSigma,
      method: "fallback",
    };
  }

  const tempSigma = predictedSigma * temperatureScale;

  let q: number;
  if (family && familyQuantiles.has(family)) {
    const fq = familyQuantiles.get(family)!;
    if (coverage >= 0.99) q = fq.q99;
    else if (coverage >= 0.95) q = fq.q95;
    else q = fq.q90;
  } else {
    if (coverage >= 0.99) q = conformalQ99;
    else if (coverage >= 0.95) q = conformalQ95;
    else q = conformalQ90;
  }

  const halfWidth = q * tempSigma;

  return {
    lower: Math.max(0, Math.round((predictedTc - halfWidth) * 10) / 10),
    upper: Math.round((predictedTc + halfWidth) * 10) / 10,
    coverage,
    quantile: Math.round(q * 1000) / 1000,
    temperatureScaledSigma: Math.round(tempSigma * 100) / 100,
    rawSigma: Math.round(predictedSigma * 100) / 100,
    method: calibrationDataset.length >= MIN_CALIBRATION_SAMPLES ? "conformal" : "fallback",
  };
}

export function notifyNewLedgerEntry(): void {
  samplesSinceLastCalibration++;
  if (samplesSinceLastCalibration >= RECALIBRATION_INTERVAL) {
    recalibrateFromLedger();
  }
}

onLedgerEntry(notifyNewLedgerEntry);

export function getECE(): { before: number; after: number; improvement: number; mce: number } {
  return {
    before: Math.round(eceBefore * 10000) / 10000,
    after: Math.round(eceAfter * 10000) / 10000,
    improvement: Math.round((eceBefore - eceAfter) * 10000) / 10000,
    mce: Math.round(mceValue * 10000) / 10000,
  };
}

export function getFamilyConformalQuantiles(): FamilyQuantiles[] {
  const result: FamilyQuantiles[] = [];
  for (const [family, fq] of familyQuantiles) {
    result.push({
      family,
      q90: Math.round(fq.q90 * 1000) / 1000,
      q95: Math.round(fq.q95 * 1000) / 1000,
      q99: Math.round(fq.q99 * 1000) / 1000,
      count: fq.count,
      ece: Math.round(fq.ece * 10000) / 10000,
    });
  }
  return result.sort((a, b) => b.count - a.count);
}

export function getCalibrationState(): CalibrationState {
  return {
    temperatureScale,
    conformalQ90: Math.round(conformalQ90 * 1000) / 1000,
    conformalQ95: Math.round(conformalQ95 * 1000) / 1000,
    conformalQ99: Math.round(conformalQ99 * 1000) / 1000,
    calibrationDatasetSize: calibrationDataset.length,
    eceBefore: Math.round(eceBefore * 10000) / 10000,
    eceAfter: Math.round(eceAfter * 10000) / 10000,
    mce: Math.round(mceValue * 10000) / 10000,
    lastCalibrationTimestamp,
    lastCalibrationLedgerSize,
    perFamily: getFamilyConformalQuantiles(),
    coverageAtQ95: Math.round(coverageAtQ95 * 1000) / 1000,
    meanNonconformityScore: Math.round(meanNCS * 1000) / 1000,
    medianNonconformityScore: Math.round(medianNCS * 1000) / 1000,
  };
}

export interface IntervalValidationResult {
  expectedCoverage: number;
  observedCoverage: number;
  nSamples: number;
  isMiscalibrated: boolean;
  recommendation: string;
  perLevel: { level: number; expected: number; observed: number; covered: number; total: number }[];
}

export function validateIntervalsCoverage(): IntervalValidationResult {
  const totalSize = getLedgerSize();
  const valStart = Math.max(0, totalSize - MAX_CALIBRATION_WINDOW);
  const entries = getLedgerSlice(valStart, totalSize);
  const coverageLevels = [0.90, 0.95, 0.99];
  const perLevel: IntervalValidationResult["perLevel"] = [];

  for (const level of coverageLevels) {
    let covered = 0;
    let total = 0;
    for (const entry of entries) {
      const predTc = entry.model_prediction?.Tc;
      const actualTc = entry.ground_truth?.Tc;
      if (!Number.isFinite(predTc) || !Number.isFinite(actualTc)) continue;
      let sigma: number;
      if (entry.predicted_sigma != null && Number.isFinite(entry.predicted_sigma) && entry.predicted_sigma > 0) {
        sigma = entry.predicted_sigma;
      } else {
        sigma = Math.max(Math.abs(entry.error?.tc_abs_error ?? 10), 1);
      }
      const ci = getConformalInterval(predTc, sigma, level);
      if (actualTc >= ci.lower && actualTc <= ci.upper) covered++;
      total++;
    }
    const obs = total > 0 ? covered / total : level;
    perLevel.push({ level, expected: level, observed: Math.round(obs * 10000) / 10000, covered, total });
  }

  const primary = perLevel.find(p => p.level === 0.95) ?? perLevel[0];
  const nSamples = primary?.total ?? 0;
  const observedCov = primary?.observed ?? 0.95;
  const isMisc = nSamples >= 10 && observedCov < 0.90;

  let recommendation = "Intervals appear well-calibrated";
  if (nSamples < 10) {
    recommendation = "Insufficient data for reliable interval validation (need at least 10 samples)";
  } else if (observedCov < 0.80) {
    recommendation = "Intervals are significantly too narrow — force recalibration with more data";
  } else if (observedCov < 0.90) {
    recommendation = "Intervals are moderately too narrow — recalibration recommended";
  } else if (observedCov > 0.99) {
    recommendation = "Intervals are overly conservative — could tighten for efficiency";
  }

  return {
    expectedCoverage: 0.95,
    observedCoverage: observedCov,
    nSamples,
    isMiscalibrated: isMisc,
    recommendation,
    perLevel,
  };
}

export function getCalibrationSummaryForLLM(): string {
  const state = getCalibrationState();
  const lines = [
    "=== Conformal Calibration State ===",
    `Dataset: ${state.calibrationDatasetSize} samples`,
    `Temperature scale: T=${state.temperatureScale} (raw sigma * T = calibrated sigma)`,
    `Conformal quantiles: Q90=${state.conformalQ90}, Q95=${state.conformalQ95}, Q99=${state.conformalQ99}`,
    `ECE: ${state.eceBefore} -> ${state.eceAfter} (improvement: ${(state.eceBefore - state.eceAfter).toFixed(4)})`,
    `MCE (max calibration error): ${state.mce}`,
    `Actual coverage at Q95: ${(state.coverageAtQ95 * 100).toFixed(1)}%`,
    `Mean nonconformity score: ${state.meanNonconformityScore}`,
  ];

  if (state.perFamily.length > 0) {
    lines.push("Per-family quantiles:");
    for (const f of state.perFamily) {
      lines.push(`  ${f.family}: Q95=${f.q95}, ECE=${f.ece}, n=${f.count}`);
    }
  }

  return lines.join("\n");
}
