/**
 * ML Evaluation Harness
 *
 * Provides rigorous model evaluation that is independent of the training loop:
 *   1. 20% of DFT-verified samples (quantum_engine_dataset tier=full-dft/xtb) are
 *      held out by deterministic formula hash — excluded from all model training.
 *   2. Test-set MAE, RMSE, R² for Tc (XGBoost), λ (lambda-regressor), Eform (Miedema).
 *   3. 5-fold stratified CV on the XGBoost training matrix.
 *   4. Uncertainty calibration audit: are predicted 50/75/90/95 % confidence intervals
 *      actually hitting their nominal coverage rates?
 */

import { db } from "../db";
import { quantumEngineDataset } from "@shared/schema";
import { gt, inArray } from "drizzle-orm";
import { extractFeatures } from "./ml-predictor";
import { gbPredict, gbPredictWithUncertainty, trainGradientBoosting, getXGBTrainingData, gbPredictFromModel } from "./gradient-boost";
import { predictLambda } from "./lambda-regressor";
import { computeMiedemaFormationEnergy } from "./phase-diagram-engine";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";

// ── Constants ────────────────────────────────────────────────────────────────

const DFT_TEST_FRACTION = 0.20; // 20% held-out test set

// Standard-normal z-scores for coverage levels
const COVERAGE_LEVELS = [0.50, 0.75, 0.90, 0.95] as const;
const Z_SCORES: Record<number, number> = { 0.50: 0.674, 0.75: 1.150, 0.90: 1.645, 0.95: 1.960 };

// ── Types ────────────────────────────────────────────────────────────────────

export interface TargetMetrics {
  mae: number;
  rmse: number;
  r2: number;
  bias: number;    // mean(predicted − actual) — positive = systematic over-prediction
  n: number;
}

export interface CalibrationAudit {
  nominalCoverages: number[];
  actualCoverages: number[];
  ece: number;           // expected calibration error = mean |nominal − actual|
  sharpness: number;     // mean predicted std — lower is more confident
  n: number;
  wellCalibrated: boolean; // ECE < 0.08
}

export interface CVFoldResult {
  fold: number;
  mae: number;
  rmse: number;
  r2: number;
  trainSize: number;
  valSize: number;
}

export interface CVReport {
  k: number;
  meanMAE: number;
  stdMAE: number;
  meanRMSE: number;
  stdRMSE: number;
  meanR2: number;
  stdR2: number;
  folds: CVFoldResult[];
  trainingSetSize: number;
  updatedAt: number;
}

export interface EvalReport {
  testSetSize: number;
  tc: TargetMetrics;
  lambda: TargetMetrics;
  eform: TargetMetrics;
  calibration: CalibrationAudit;
  cv: CVReport | null;
  note: string;
  updatedAt: number;
}

// ── Internal state ────────────────────────────────────────────────────────────

interface EvalSample {
  formula: string;
  tier: "full-dft" | "xtb";
  tc: number;
  lambda: number | null;
  eform: number | null;
  features: Awaited<ReturnType<typeof extractFeatures>> | null;
}

// Formulas in SUPERCON static dataset.
// Primary test set excludes these (they are in training).
// However, when the DFT-verified pool is too small (< MIN_DFT_FOR_PURE_TEST), we
// augment the test set with the 20 % hash-selected slice of SUPERCON data so the
// harness always has meaningful held-out samples even during early training.
const superconFormulaSet = new Set(SUPERCON_TRAINING_DATA.map(e => e.formula));
const MIN_DFT_FOR_PURE_TEST = 5; // below this, augment with SUPERCON static data

const testSamples = new Map<string, EvalSample>(); // formula → sample
let testSetLoaded = false;
let featuresWarmedUp = false;
let lastReport: EvalReport | null = null;
let lastCVReport: CVReport | null = null;
let isRunning = false;

// ── Deterministic split ───────────────────────────────────────────────────────

function fnv32(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/**
 * Returns true if this formula belongs to the held-out test set.
 * Purely deterministic — same answer on every call, every process restart.
 * Uses the normalised formula so case/whitespace differences don't shift the split.
 */
export function isEvalTestFormula(formula: string): boolean {
  const norm = formula.trim().replace(/\s+/g, "");
  return (fnv32(norm) % 100) < Math.round(DFT_TEST_FRACTION * 100);
}

// ── Regression helpers ────────────────────────────────────────────────────────

function regressionMetrics(actuals: number[], preds: number[]): TargetMetrics {
  const n = actuals.length;
  if (n === 0) return { mae: 0, rmse: 0, r2: 0, bias: 0, n: 0 };

  const meanActual = actuals.reduce((s, v) => s + v, 0) / n;
  let sse = 0, sst = 0, sumAbs = 0, sumDiff = 0;
  for (let i = 0; i < n; i++) {
    const diff = preds[i] - actuals[i];
    sumDiff += diff;
    sumAbs += Math.abs(diff);
    sse += diff * diff;
    sst += (actuals[i] - meanActual) ** 2;
  }
  return {
    mae: Math.round((sumAbs / n) * 100) / 100,
    rmse: Math.round(Math.sqrt(sse / n) * 100) / 100,
    r2: sst < 1e-9 ? 0 : Math.round((1 - sse / sst) * 10000) / 10000,
    bias: Math.round((sumDiff / n) * 100) / 100,
    n,
  };
}

// ── Test-set loading ──────────────────────────────────────────────────────────

async function loadTestSet(): Promise<void> {
  if (testSetLoaded) return;
  testSetLoaded = true;
  try {
    const rows = await db.select({
      material: quantumEngineDataset.material,
      tc:               quantumEngineDataset.tc,
      lambda:           quantumEngineDataset.lambda,
      formationEnergy:  quantumEngineDataset.formationEnergy,
      tier:             quantumEngineDataset.tier,
    }).from(quantumEngineDataset)
      .where(gt(quantumEngineDataset.tc, 0))
      .limit(10_000);

    let added = 0;
    for (const row of rows) {
      const formula = row.material;
      if (!formula) continue;
      if (row.tier !== "full-dft" && row.tier !== "xtb") continue;
      if (!isEvalTestFormula(formula)) continue;
      if (superconFormulaSet.has(formula)) continue; // already in static training set
      if (testSamples.has(formula)) continue;
      testSamples.set(formula, {
        formula,
        tier: row.tier as "full-dft" | "xtb",
        tc: Number(row.tc) || 0,
        lambda: row.lambda != null ? Number(row.lambda) : null,
        eform:  row.formationEnergy != null ? Number(row.formationEnergy) : null,
        features: null,
      });
      added++;
    }

    // Fallback: when DFT-verified pool is too small for a meaningful 20 % test set,
    // augment with the hash-selected 20 % slice of the SUPERCON static dataset.
    // These are real experimental Tc values and give the harness meaningful
    // held-out samples during early training when DFT results are sparse.
    if (added < MIN_DFT_FOR_PURE_TEST) {
      let superconAdded = 0;
      for (const entry of SUPERCON_TRAINING_DATA) {
        if (!entry.formula || !(entry.tc > 0)) continue;
        if (!isEvalTestFormula(entry.formula)) continue; // only the hash-selected 20%
        if (testSamples.has(entry.formula)) continue;
        testSamples.set(entry.formula, {
          formula: entry.formula,
          tier: "xtb" as const, // label as xtb-tier (experimental, not raw DFT)
          tc: entry.tc,
          lambda: (entry as any).lambda ?? null,
          eform:  (entry as any).formationEnergy ?? null,
          features: null,
        });
        superconAdded++;
      }
      if (superconAdded > 0) {
        console.log(
          `[EvalHarness] DFT pool too small (${added} samples) — augmented test set with ` +
          `${superconAdded} SUPERCON static samples (in-sample eval, clearly noted in report)`,
        );
      }
    }

    console.log(`[EvalHarness] Test set loaded: ${testSamples.size} samples (${added} DFT-verified, held-out ${(DFT_TEST_FRACTION * 100).toFixed(0)}%)`);
  } catch (e: any) {
    testSetLoaded = false;
    console.warn(`[EvalHarness] Test set load failed: ${e?.message?.slice(0, 100)}`);
  }
}

/**
 * Extract and cache features for all test samples that don't have them yet.
 * Rate-limited: 50 ms between each extractFeatures call to avoid blocking.
 */
async function warmTestFeatures(): Promise<void> {
  const toWarm = [...testSamples.values()].filter(s => s.features === null);
  if (toWarm.length === 0) return;
  console.log(`[EvalHarness] Warming features for ${toWarm.length} test samples…`);
  let done = 0;
  for (const sample of toWarm) {
    try {
      sample.features = await extractFeatures(sample.formula);
      done++;
    } catch { /* skip */ }
    await new Promise<void>(r => setTimeout(r, 50));
  }
  featuresWarmedUp = true;
  console.log(`[EvalHarness] Feature warm-up complete: ${done}/${toWarm.length} samples ready`);
}

// ── Test-set evaluation ───────────────────────────────────────────────────────

async function evaluateTestSet(): Promise<{
  tc: TargetMetrics;
  lambda: TargetMetrics;
  eform: TargetMetrics;
  calibration: CalibrationAudit;
  n: number;
}> {
  const ready = [...testSamples.values()].filter(s => s.features !== null && s.tc > 0);

  // ── Tc via XGBoost ──────────────────────────────────────────────────────────
  const tcActuals: number[] = [];
  const tcPreds:   number[] = [];
  const tcStds:    number[] = [];

  for (const sample of ready) {
    try {
      const res = await gbPredictWithUncertainty(sample.features!, sample.formula);
      tcActuals.push(sample.tc);
      tcPreds.push(res.tcMean);
      tcStds.push(res.totalStd);
    } catch { /* skip */ }
    await new Promise<void>(r => setTimeout(r, 0)); // yield between samples
  }

  // ── λ via lambda-regressor ──────────────────────────────────────────────────
  const lambdaActuals: number[] = [];
  const lambdaPreds:   number[] = [];
  for (const sample of ready) {
    if (sample.lambda == null || sample.lambda <= 0) continue;
    try {
      const pred = predictLambda(sample.formula, 0);
      lambdaActuals.push(sample.lambda);
      lambdaPreds.push(pred.lambda);
    } catch { /* skip */ }
  }

  // ── Eform: Miedema analytical vs DFT ───────────────────────────────────────
  const eformActuals: number[] = [];
  const eformPreds:   number[] = [];
  for (const sample of ready) {
    if (sample.eform == null) continue;
    try {
      const miedema = computeMiedemaFormationEnergy(sample.formula);
      if (miedema != null && Number.isFinite(miedema)) {
        eformActuals.push(sample.eform);
        eformPreds.push(miedema);
      }
    } catch { /* skip */ }
  }

  // ── Calibration audit ───────────────────────────────────────────────────────
  let calibration: CalibrationAudit;
  if (tcActuals.length < 5) {
    calibration = {
      nominalCoverages: [...COVERAGE_LEVELS],
      actualCoverages:  COVERAGE_LEVELS.map(() => 0),
      ece: 1,
      sharpness: 0,
      n: 0,
      wellCalibrated: false,
    };
  } else {
    const actualCoverages = COVERAGE_LEVELS.map(level => {
      const z = Z_SCORES[level];
      let covered = 0;
      for (let i = 0; i < tcActuals.length; i++) {
        if (Math.abs(tcActuals[i] - tcPreds[i]) <= z * tcStds[i]) covered++;
      }
      return covered / tcActuals.length;
    });
    const ece = actualCoverages.reduce((s, ac, i) => s + Math.abs(COVERAGE_LEVELS[i] - ac), 0)
                / COVERAGE_LEVELS.length;
    const sharpness = tcStds.length > 0
      ? tcStds.reduce((s, v) => s + v, 0) / tcStds.length : 0;
    calibration = {
      nominalCoverages: [...COVERAGE_LEVELS],
      actualCoverages:  actualCoverages.map(v => Math.round(v * 1000) / 1000),
      ece:   Math.round(ece * 10000) / 10000,
      sharpness: Math.round(sharpness * 10) / 10,
      n:     tcActuals.length,
      wellCalibrated: ece < 0.08,
    };
  }

  return {
    tc:     regressionMetrics(tcActuals, tcPreds),
    lambda: regressionMetrics(lambdaActuals, lambdaPreds),
    eform:  regressionMetrics(eformActuals, eformPreds),
    calibration,
    n:      ready.length,
  };
}

// ── 5-fold CV ─────────────────────────────────────────────────────────────────

export async function runCrossValidation(k = 5): Promise<CVReport> {
  const { X, y, formulas } = await getXGBTrainingData();
  const n = X.length;

  if (n < k * 3) {
    return {
      k,
      meanMAE: 0, stdMAE: 0, meanRMSE: 0, stdRMSE: 0, meanR2: 0, stdR2: 0,
      folds: [],
      trainingSetSize: n,
      updatedAt: Date.now(),
    };
  }

  // Stratified fold assignment: sort by Tc, assign folds round-robin
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => y[a] - y[b]);
  const foldAssignment = new Array<number>(n);
  order.forEach((idx, rank) => { foldAssignment[idx] = rank % k; });

  const results: CVFoldResult[] = [];

  for (let fold = 0; fold < k; fold++) {
    const trainIdx: number[] = [];
    const valIdx:   number[] = [];
    for (let i = 0; i < n; i++) {
      if (foldAssignment[i] === fold) valIdx.push(i);
      else trainIdx.push(i);
    }

    const Xtr = trainIdx.map(i => X[i]);
    const ytr = trainIdx.map(i => y[i]);
    const Xval = valIdx.map(i => X[i]);
    const yval = valIdx.map(i => y[i]);

    // Lightweight model per fold: 100 trees (not production 300)
    const foldModel = await trainGradientBoosting(Xtr, ytr, 100, 0.08, 5);

    const preds = Xval.map(x => gbPredictFromModel(foldModel, x));

    results.push({
      fold,
      trainSize: trainIdx.length,
      valSize:   valIdx.length,
      ...regressionMetrics(yval, preds),
    });

    // Yield between folds so the event loop stays responsive
    await new Promise<void>(r => setTimeout(r, 0));
  }

  const maes  = results.map(r => r.mae);
  const rmses = results.map(r => r.rmse);
  const r2s   = results.map(r => r.r2);
  const mean  = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;
  const std   = (arr: number[]) => {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
  };

  const report: CVReport = {
    k,
    meanMAE:  Math.round(mean(maes)  * 100) / 100,
    stdMAE:   Math.round(std(maes)   * 100) / 100,
    meanRMSE: Math.round(mean(rmses) * 100) / 100,
    stdRMSE:  Math.round(std(rmses)  * 100) / 100,
    meanR2:   Math.round(mean(r2s)   * 10000) / 10000,
    stdR2:    Math.round(std(r2s)    * 10000) / 10000,
    folds: results,
    trainingSetSize: n,
    updatedAt: Date.now(),
  };

  lastCVReport = report;
  console.log(
    `[EvalHarness] ${k}-fold CV — MAE=${report.meanMAE}±${report.stdMAE}K  ` +
    `RMSE=${report.meanRMSE}±${report.stdRMSE}K  R²=${report.meanR2}±${report.stdR2}  N=${n}`
  );
  return report;
}

// ── Full evaluation run ───────────────────────────────────────────────────────

/**
 * Run a full evaluation pass: load test set → warm features → evaluate all targets
 * → 5-fold CV. Results are cached and returned by getEvalReport().
 *
 * Safe to call after every AL cycle; skips if already running.
 */
export async function runEvaluation(includeCv = true): Promise<EvalReport> {
  if (isRunning) return lastReport ?? buildEmptyReport();
  isRunning = true;
  try {
    await loadTestSet();
    await warmTestFeatures();

    const { tc, lambda, eform, calibration, n } = await evaluateTestSet();

    let cv: CVReport | null = lastCVReport;
    if (includeCv) {
      try {
        cv = await runCrossValidation(5);
      } catch (e: any) {
        console.warn(`[EvalHarness] CV failed: ${e?.message?.slice(0, 80)}`);
      }
    }

    const dftOnly = [...testSamples.values()].filter(s => s.tier === "full-dft").length;
    const xtbOnly = [...testSamples.values()].filter(s => s.tier === "xtb").length;

    const report: EvalReport = {
      testSetSize: n,
      tc,
      lambda,
      eform,
      calibration,
      cv,
      note: `Test set: ${n} samples with features (${dftOnly} full-DFT, ${xtbOnly} xTB/static). ` +
            `Held-out by formula hash (${(DFT_TEST_FRACTION * 100).toFixed(0)}% split). ` +
            (dftOnly < MIN_DFT_FOR_PURE_TEST
              ? `WARNING: DFT pool < ${MIN_DFT_FOR_PURE_TEST} — SUPERCON static data used for augmentation (in-sample eval).`
              : "Clean out-of-sample evaluation."),
      updatedAt: Date.now(),
    };

    lastReport = report;
    console.log(
      `[EvalHarness] Test set — Tc: MAE=${tc.mae}K R²=${tc.r2}  ` +
      `λ: MAE=${lambda.mae} R²=${lambda.r2}  ` +
      `Eform: MAE=${eform.mae} R²=${eform.r2}  ` +
      `Cal-ECE=${calibration.ece} (${calibration.wellCalibrated ? "✓ calibrated" : "✗ miscalibrated"})`
    );
    return report;
  } finally {
    isRunning = false;
  }
}

// ── Public getters ────────────────────────────────────────────────────────────

export function getEvalReport(): EvalReport | null {
  return lastReport;
}

export function getCVReport(): CVReport | null {
  return lastCVReport;
}

export function getTestSetSize(): number {
  return testSamples.size;
}

/**
 * Called at server startup: load test set in background (no feature warm-up yet).
 * Feature extraction is deferred until the first explicit runEvaluation() call.
 */
export function initEvalHarness(): void {
  setTimeout(async () => {
    try {
      await loadTestSet();
    } catch { /* silent */ }
  }, 90_000); // 90s after startup — let models initialize first
  console.log("[EvalHarness] Deferred test-set load scheduled (T+90s)");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function buildEmptyReport(): EvalReport {
  const emptyMetrics: TargetMetrics = { mae: 0, rmse: 0, r2: 0, bias: 0, n: 0 };
  return {
    testSetSize: 0,
    tc: emptyMetrics,
    lambda: emptyMetrics,
    eform: emptyMetrics,
    calibration: {
      nominalCoverages: [...COVERAGE_LEVELS],
      actualCoverages: COVERAGE_LEVELS.map(() => 0),
      ece: 1, sharpness: 0, n: 0, wellCalibrated: false,
    },
    cv: null,
    note: "No evaluation data available yet.",
    updatedAt: Date.now(),
  };
}
