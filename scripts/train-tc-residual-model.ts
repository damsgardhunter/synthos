/**
 * Phase 4: ML-Physics Hybrid — Residual Correction Model
 *
 * Trains a lightweight model to learn: correction(features) = Tc_ref / Tc_AD
 * Then: Tc_final = Tc_AD × correction(features)
 *
 * Uses leave-one-out cross-validation on VERIFIED_COMPOUNDS to evaluate.
 * The model learns residual patterns that pure physics (AD formula) misses:
 *   - f-electron effects (Ce compounds)
 *   - CDW competition (NbSe2)
 *   - correlation-enhanced EPC (BKBO)
 *   - multi-band effects (borocarbides)
 */

import { VERIFIED_COMPOUNDS, allenDynesTcUncalibrated } from "../server/learning/physics-engine";
import { extractFeatures, type MLFeatureVector } from "../server/learning/ml-predictor";

interface CompoundData {
  formula: string;
  tcRef: number;
  tcAD: number;
  correction: number; // tcRef / tcAD
  features: number[];
  family: string;
}

function classifyFamily(compound: string, pressure: number): string {
  if (/H\d/.test(compound) && pressure >= 100) return "superhydride";
  if (/H\d/.test(compound) && pressure >= 30) return "hydride-hp";
  if (/H\d/.test(compound)) return "hydride-lp";
  if (/^(Nb|V)3/.test(compound)) return "A15";
  if (/C60/.test(compound) || /^(K3|Rb3|Cs3)C/.test(compound)) return "fullerene";
  if (/Ni.*B.*C/.test(compound)) return "borocarbide";
  if (/Mo.*S/.test(compound) && /Pb/.test(compound)) return "chevrel";
  if (/Se2|S2/.test(compound)) return "tmd";
  if (/BKBO|BiO/.test(compound)) return "bismuthate";
  if (/^[A-Z][a-z]?$/.test(compound)) return "elemental";
  return "conventional";
}

// Simple gradient-boosted stump for residual correction
interface Stump {
  featureIdx: number;
  threshold: number;
  leftValue: number;
  rightValue: number;
}

function trainStump(X: number[][], y: number[], weights: number[]): Stump {
  const n = X.length;
  const nFeatures = X[0].length;
  let bestLoss = Infinity;
  let bestStump: Stump = { featureIdx: 0, threshold: 0, leftValue: 0, rightValue: 0 };

  for (let f = 0; f < nFeatures; f++) {
    const vals = X.map((x, i) => ({ v: x[f], y: y[i], w: weights[i] })).sort((a, b) => a.v - b.v);
    for (let split = 1; split < n; split++) {
      if (vals[split].v === vals[split - 1].v) continue;
      const threshold = (vals[split].v + vals[split - 1].v) / 2;

      let leftSum = 0, leftW = 0, rightSum = 0, rightW = 0;
      for (let i = 0; i < split; i++) { leftSum += vals[i].y * vals[i].w; leftW += vals[i].w; }
      for (let i = split; i < n; i++) { rightSum += vals[i].y * vals[i].w; rightW += vals[i].w; }

      const leftMean = leftW > 0 ? leftSum / leftW : 0;
      const rightMean = rightW > 0 ? rightSum / rightW : 0;

      let loss = 0;
      for (let i = 0; i < split; i++) loss += vals[i].w * (vals[i].y - leftMean) ** 2;
      for (let i = split; i < n; i++) loss += vals[i].w * (vals[i].y - rightMean) ** 2;

      if (loss < bestLoss) {
        bestLoss = loss;
        bestStump = { featureIdx: f, threshold, leftValue: leftMean, rightValue: rightMean };
      }
    }
  }
  return bestStump;
}

function predictStump(stump: Stump, x: number[]): number {
  return x[stump.featureIdx] <= stump.threshold ? stump.leftValue : stump.rightValue;
}

function trainGBResidual(X: number[][], y: number[], nTrees = 30, lr = 0.1): Stump[] {
  const n = X.length;
  const weights = new Array(n).fill(1 / n);
  const predictions = new Array(n).fill(0);
  const stumps: Stump[] = [];

  for (let t = 0; t < nTrees; t++) {
    const residuals = y.map((yi, i) => yi - predictions[i]);
    const stump = trainStump(X, residuals, weights);
    stumps.push(stump);
    for (let i = 0; i < n; i++) {
      predictions[i] += lr * predictStump(stump, X[i]);
    }
  }
  return stumps;
}

function predictGB(stumps: Stump[], x: number[], lr = 0.1): number {
  let pred = 0;
  for (const s of stumps) pred += lr * predictStump(s, x);
  return pred;
}

async function main() {
  console.log("=== Phase 4: ML-Physics Hybrid — Residual Correction Model ===\n");

  // Step 1: Collect data for all AD-applicable verified compounds with tcRef > 0
  const data: CompoundData[] = [];
  for (const [formula, v] of Object.entries(VERIFIED_COMPOUNDS)) {
    if (v.adApplicable === false) continue;
    if (v.tcRef <= 0) continue;

    const isHydride = formula.includes("H") && v.lambda > 1.2;
    const tcAD = allenDynesTcUncalibrated(v.lambda, v.omegaLog, v.muStar, v.omegaLog * v.omegaLog * 1.2, isHydride, formula);
    if (tcAD <= 0) continue;

    const correction = v.tcRef / tcAD;
    const family = classifyFamily(formula, v.pressureGpa);

    // Extract features — use a subset that's available without DB
    const features = [
      v.lambda,
      v.omegaLog,
      v.muStar,
      v.pressureGpa,
      v.lambda * v.omegaLog,  // coupling * phonon frequency
      v.lambda / (v.lambda - v.muStar * (1 + 0.62 * v.lambda) + 0.001), // denominator proximity
      isHydride ? 1 : 0,
      v.lambda > 1.5 ? 1 : 0,  // strong coupling flag
      v.pressureGpa > 100 ? 1 : 0, // high pressure flag
      Math.log(v.omegaLog + 1), // log phonon freq
      v.lambda * v.lambda, // lambda squared
      v.muStar * (1 + 0.62 * v.lambda), // effective mu*
    ];

    data.push({ formula, tcRef: v.tcRef, tcAD, correction, features, family });
  }

  console.log(`Compounds loaded: ${data.length}`);
  console.log(`Families: ${[...new Set(data.map(d => d.family))].join(", ")}\n`);

  // Step 2: Baseline (AD only)
  const baselineErrors = data.map(d => Math.abs(d.tcAD - d.tcRef) / d.tcRef);
  const baselineMAE = baselineErrors.reduce((s, e) => s + e, 0) / baselineErrors.length;
  console.log(`Baseline AD MAE: ${(baselineMAE * 100).toFixed(1)}%\n`);

  // Step 3: Leave-one-out cross-validation
  const loocvErrors: number[] = [];
  const loocvResults: { formula: string; tcRef: number; tcAD: number; tcML: number; errAD: number; errML: number; family: string }[] = [];

  const X = data.map(d => d.features);
  const y = data.map(d => Math.log(d.correction)); // predict log(correction) for stability

  for (let i = 0; i < data.length; i++) {
    // Leave one out
    const trainX = X.filter((_, j) => j !== i);
    const trainY = y.filter((_, j) => j !== i);
    const testX = X[i];

    // Train on N-1 compounds
    const stumps = trainGBResidual(trainX, trainY, 20, 0.08);

    // Predict correction for held-out compound
    const predLogCorrection = predictGB(stumps, testX, 0.08);
    const predCorrection = Math.exp(predLogCorrection);

    // Clamp correction to [0.5, 2.0] to prevent wild extrapolation
    const clampedCorrection = Math.max(0.5, Math.min(2.0, predCorrection));

    const tcML = data[i].tcAD * clampedCorrection;
    const errML = Math.abs(tcML - data[i].tcRef) / data[i].tcRef;
    const errAD = Math.abs(data[i].tcAD - data[i].tcRef) / data[i].tcRef;

    loocvErrors.push(errML);
    loocvResults.push({
      formula: data[i].formula,
      tcRef: data[i].tcRef,
      tcAD: data[i].tcAD,
      tcML: Math.round(tcML * 10) / 10,
      errAD: Math.round(errAD * 1000) / 10,
      errML: Math.round(errML * 1000) / 10,
      family: data[i].family,
    });
  }

  const mlMAE = loocvErrors.reduce((s, e) => s + e, 0) / loocvErrors.length;
  console.log(`ML+Physics MAE (LOOCV): ${(mlMAE * 100).toFixed(1)}%`);
  console.log(`Improvement: ${((baselineMAE - mlMAE) * 100).toFixed(1)} percentage points\n`);

  // Step 4: Show improvements for worst AD compounds
  loocvResults.sort((a, b) => b.errAD - a.errAD);
  console.log("Top improvements (worst AD compounds):");
  console.log("Formula           Family          AD_err  ML_err  Direction");
  console.log("─".repeat(70));
  for (const r of loocvResults.slice(0, 15)) {
    const dir = r.errML < r.errAD ? "✓ IMPROVED" : "✗ worse";
    const pad = (s: string, n: number) => s.padEnd(n);
    console.log(`${pad(r.formula, 18)}${pad(r.family, 16)}${(r.errAD + "%").padStart(7)}  ${(r.errML + "%").padStart(7)}  ${dir}`);
  }

  // Step 5: Summary by family
  console.log("\nBy family:");
  const families = [...new Set(loocvResults.map(r => r.family))];
  for (const fam of families) {
    const famResults = loocvResults.filter(r => r.family === fam);
    const famAD = famResults.reduce((s, r) => s + r.errAD, 0) / famResults.length;
    const famML = famResults.reduce((s, r) => s + r.errML, 0) / famResults.length;
    const dir = famML < famAD ? "↓" : "↑";
    console.log(`  ${fam.padEnd(16)} n=${famResults.length}  AD: ${famAD.toFixed(1)}%  ML: ${famML.toFixed(1)}%  ${dir}`);
  }

  // Write results
  const report = {
    timestamp: new Date().toISOString(),
    baselineMAE: Math.round(baselineMAE * 1000) / 10,
    mlMAE: Math.round(mlMAE * 1000) / 10,
    improvement: Math.round((baselineMAE - mlMAE) * 1000) / 10,
    compoundsEvaluated: data.length,
    loocvResults,
  };

  const fs = await import("fs");
  fs.writeFileSync("logs/tc-residual-model-report.json", JSON.stringify(report, null, 2));
  console.log("\nReport written to logs/tc-residual-model-report.json");
}

main().catch(console.error);
