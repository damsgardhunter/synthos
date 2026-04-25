/**
 * ML Weight Loader for QE Worker
 *
 * Loads the latest XGBoost ensemble weights from the Neon DB so the QE worker
 * can run inference locally without needing the GCP service. The GCP service
 * trains the model and writes weights to the DB; this module reads them.
 *
 * Architecture:
 * - GCP service trains XGBoost → stores weights in xgb_training_jobs.ensemble_xgb
 * - QE worker calls loadLatestXGBWeights() at startup or on cache miss
 * - Predictions use the local TypeScript gbPredictFromModel() (no Python needed)
 * - Weights are refreshed every 6 hours to pick up retraining updates
 */

import { db } from "../db";
import { xgbTrainingJobs } from "@shared/schema";
import { desc, eq } from "drizzle-orm";
import { gbPredictFromModel } from "../learning/gradient-boost";
import { extractFeatures } from "../learning/ml-predictor";

// ---------------------------------------------------------------------------
// Types (matching gradient-boost.ts GBModel/GBEnsemble interfaces)
// ---------------------------------------------------------------------------

interface GBModel {
  trees: any[];
  flatTrees: any[];
  learningRate: number;
  basePrediction: number;
  featureNames: string[];
  featureMask?: number[];
  trainedAt: number;
  logTransformed?: boolean;
}

interface GBEnsemble {
  models: GBModel[];
  trainedAt: number;
  isLogVariance?: boolean;
}

// ---------------------------------------------------------------------------
// Cached state
// ---------------------------------------------------------------------------

let cachedEnsemble: GBEnsemble | null = null;
let cachedVarianceEnsemble: GBEnsemble | null = null;
let lastLoadTime = 0;
const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface LocalMLPrediction {
  tc: number;
  tcStd: number;
  tcCI95: [number, number];
  stability: number;       // P(stable) proxy from prediction confidence
  confidence: number;
  source: "local-xgb";
  nModels: number;
}

/**
 * Load the latest XGBoost ensemble weights from the Neon DB.
 * Returns true if weights were loaded successfully.
 */
export async function loadLatestXGBWeights(): Promise<boolean> {
  if (cachedEnsemble && (Date.now() - lastLoadTime) < CACHE_TTL_MS) {
    return true; // Cache still fresh
  }

  try {
    const rows = await db
      .select({
        ensembleXGB: xgbTrainingJobs.ensembleXGB,
        varianceEnsembleXGB: xgbTrainingJobs.varianceEnsembleXGB,
        r2: xgbTrainingJobs.r2,
        completedAt: xgbTrainingJobs.completedAt,
      })
      .from(xgbTrainingJobs)
      .where(eq(xgbTrainingJobs.status, "done"))
      .orderBy(desc(xgbTrainingJobs.completedAt))
      .limit(1);

    if (rows.length === 0 || !rows[0].ensembleXGB) {
      console.log("[ML-Weights] No completed XGB training jobs found in DB");
      return false;
    }

    const row = rows[0];
    cachedEnsemble = row.ensembleXGB as unknown as GBEnsemble;
    cachedVarianceEnsemble = row.varianceEnsembleXGB as unknown as GBEnsemble | null;
    lastLoadTime = Date.now();

    const nModels = cachedEnsemble?.models?.length ?? 0;
    console.log(`[ML-Weights] Loaded XGB ensemble from DB: ${nModels} models, R²=${row.r2?.toFixed(4) ?? "N/A"}, trained=${row.completedAt}`);
    return nModels > 0;
  } catch (err: any) {
    console.log(`[ML-Weights] Failed to load XGB weights from DB: ${err.message?.slice(0, 150)}`);
    return false;
  }
}

/**
 * Run XGBoost prediction locally using DB-loaded weights.
 * Extracts features from the formula and runs inference on the ensemble.
 *
 * @param formula Chemical formula (e.g., "BiGeSb")
 * @param pressureGpa Pressure in GPa (default 0)
 * @returns Prediction result or null if weights not available
 */
export async function predictWithLocalXGB(
  formula: string,
  pressureGpa: number = 0,
): Promise<LocalMLPrediction | null> {
  // Ensure weights are loaded
  if (!cachedEnsemble) {
    const loaded = await loadLatestXGBWeights();
    if (!loaded || !cachedEnsemble) return null;
  }

  try {
    // Extract features using the existing ML pipeline
    const features = await extractFeatures(formula);
    if (!features) return null;

    // Convert MLFeatureVector to flat array matching XGB training features
    const featureArray = featureVectorToArray(features);

    // Add pressure as the last feature (matches GCP training convention)
    featureArray.push(pressureGpa);

    // Run ensemble prediction
    const predictions: number[] = [];
    for (const model of cachedEnsemble.models) {
      try {
        const pred = gbPredictFromModel(model, featureArray);
        if (Number.isFinite(pred)) predictions.push(pred);
      } catch {
        // Skip broken model in ensemble
      }
    }

    if (predictions.length === 0) return null;

    // Compute ensemble statistics
    const tc = predictions.reduce((s, p) => s + p, 0) / predictions.length;
    const variance = predictions.reduce((s, p) => s + (p - tc) ** 2, 0) / predictions.length;
    const std = Math.sqrt(variance);

    // CI95 from variance ensemble if available, else from ensemble spread
    let varStd = std;
    if (cachedVarianceEnsemble && cachedVarianceEnsemble.models.length > 0) {
      const varPreds: number[] = [];
      for (const model of cachedVarianceEnsemble.models) {
        try {
          const pred = gbPredictFromModel(model, featureArray);
          if (Number.isFinite(pred) && pred > 0) varPreds.push(pred);
        } catch {}
      }
      if (varPreds.length > 0) {
        const meanVar = varPreds.reduce((s, v) => s + v, 0) / varPreds.length;
        varStd = Math.sqrt(meanVar);
      }
    }

    const totalStd = Math.sqrt(std ** 2 + varStd ** 2);
    const ci95Lo = Math.max(0, tc - 1.96 * totalStd);
    const ci95Hi = tc + 1.96 * totalStd;

    // Stability proxy: if Tc > 0 and std is reasonable relative to mean, mark stable
    const stability = tc > 1 ? Math.min(1, 0.3 + 0.7 * Math.exp(-totalStd / Math.max(tc, 1))) : 0.1;

    return {
      tc: Math.max(0, tc),
      tcStd: totalStd,
      tcCI95: [ci95Lo, ci95Hi],
      stability,
      confidence: predictions.length / cachedEnsemble.models.length,
      source: "local-xgb",
      nModels: predictions.length,
    };
  } catch (err: any) {
    console.log(`[ML-Weights] Local XGB prediction failed for ${formula}: ${err.message?.slice(0, 100)}`);
    return null;
  }
}

/**
 * Check if local XGB weights are available (loaded in cache).
 */
export function hasLocalXGBWeights(): boolean {
  return cachedEnsemble !== null && cachedEnsemble.models.length > 0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Convert an MLFeatureVector object to a flat number array.
 * The order must match what the XGB model was trained on.
 */
function featureVectorToArray(features: any): number[] {
  // The feature vector from extractFeatures() is a flat object with numeric properties.
  // XGB expects them in the same order as featureNames in the model.
  // We extract in a canonical order matching the training pipeline.
  const keys = [
    "avgElectronegativity", "maxAtomicMass", "numElements",
    "hasTransitionMetal", "hasRareEarth", "hasHydrogen", "hasChalcogen", "hasPnictogen",
    "bandGap", "formationEnergy", "stability", "crystalSymmetry",
    "electronDensityEstimate", "phononCouplingEstimate", "dWaveSymmetry",
    "layeredStructure", "cooperPairStrength", "meissnerPotential",
    "correlationStrength", "fermiSurfaceType", "dimensionalityScore",
    "anharmonicityFlag", "anharmonicityScore", "electronPhononLambda",
    "logPhononFreq", "debyeTemperature", "phononSpectralCentroid",
    "phononSpectralWidth", "phononSofteningIndex",
    "upperCriticalField", "metallicity", "avgAtomicRadius",
    "pettiforNumber", "valenceElectronConcentration", "enSpread",
    "hydrogenRatio", "avgSommerfeldGamma", "avgBulkModulus",
    "dftConfidence", "dosAtEF", "muStarEstimate",
    "dosAtEF_tb", "bandFlatness_tb", "derivedBandwidth",
    "orbitalCharacterCode", "bondStiffnessVariance", "chargeTransferMagnitude",
    "connectivityIndex", "nestingScore", "vanHoveProximity",
    "bandFlatness", "softModeScore", "motifScore",
    "orbitalDFraction", "mottProximityScore", "topologicalBandScore",
    "spinFluctuationStrength", "fermiSurfaceNestingScore",
    "lambdaProxy", "alphaCouplingStrength", "phononHardness",
    "massEnhancement", "couplingAsymmetry",
    "derivedVanHoveDistance", "derivedBondStiffness", "derivedEPCDensity",
    "derivedSpectralWeight", "derivedAnharmonicRatio", "derivedCouplingEfficiency",
    "disorderVacancyFraction", "disorderBondVariance", "disorderLatticeStrain",
    "disorderSiteMixingEntropy", "disorderConfigEntropy", "disorderDosSignal",
    "dopingCarrierDensity", "dopingLatticeStrain", "dopingBondVariance",
    "dopantValenceDiff",
    "familyBoride", "familyA15", "familyPnictide", "familyChalcogenide", "familyHeavyFermion",
    "pressureGpa", "optimalPressureGpa", "feasibilityScore", "lightElementFraction",
  ];

  const arr: number[] = [];
  for (const key of keys) {
    const val = (features as any)[key];
    arr.push(typeof val === "number" && Number.isFinite(val) ? val : 0);
  }
  return arr;
}
