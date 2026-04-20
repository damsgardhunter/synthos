/**
 * Benchmark script for computePhysicsTcUQ against VERIFIED_COMPOUNDS.
 * Measures MAE, MAPE, and identifies worst offenders.
 */
import { computePhysicsTcUQ, VERIFIED_COMPOUNDS } from "../server/learning/physics-engine";

interface BenchmarkResult {
  formula: string;
  tcRef: number;
  tcPred: number;
  absErr: number;
  relErr: number; // percentage
  lambda: number;
  omegaLog: number;
  pressure: number;
}

const results: BenchmarkResult[] = [];
let skipped = 0;

for (const [formula, data] of Object.entries(VERIFIED_COMPOUNDS)) {
  // Skip AD-inapplicable compounds
  if (data.adApplicable === false) {
    skipped++;
    continue;
  }
  // Skip Tc=0 compounds (can't compute relative error)
  if (data.tcRef <= 0) {
    skipped++;
    continue;
  }

  try {
    const result = computePhysicsTcUQ(formula, data.pressureGpa);
    const tcPred = result.mean;
    const absErr = Math.abs(tcPred - data.tcRef);
    const relErr = (absErr / data.tcRef) * 100;

    results.push({
      formula,
      tcRef: data.tcRef,
      tcPred: Math.round(tcPred * 10) / 10,
      absErr: Math.round(absErr * 10) / 10,
      relErr: Math.round(relErr * 10) / 10,
      lambda: data.lambda,
      omegaLog: data.omegaLog,
      pressure: data.pressureGpa,
    });
  } catch (err: any) {
    console.error(`ERROR: ${formula}: ${err.message}`);
  }
}

// Sort by relative error descending
results.sort((a, b) => b.relErr - a.relErr);

// Aggregate stats
const maeK = results.reduce((s, r) => s + r.absErr, 0) / results.length;
const mape = results.reduce((s, r) => s + r.relErr, 0) / results.length;
const sortedRelErr = [...results].sort((a, b) => a.relErr - b.relErr);
const medianApe = sortedRelErr[Math.floor(sortedRelErr.length / 2)]?.relErr ?? 0;
const maxErrCompound = results[0];

console.log("=".repeat(100));
console.log("  computePhysicsTcUQ Benchmark vs VERIFIED_COMPOUNDS");
console.log("=".repeat(100));
console.log(`  Compounds tested: ${results.length} (${skipped} skipped: AD-inapplicable or Tc=0)`);
console.log(`  MAE:       ${maeK.toFixed(1)} K`);
console.log(`  MAPE:      ${mape.toFixed(1)}%`);
console.log(`  Median APE: ${medianApe.toFixed(1)}%`);
console.log(`  Worst:     ${maxErrCompound?.formula} (${maxErrCompound?.relErr}% error)`);
console.log("=".repeat(100));

// Top 15 worst offenders
console.log("\n  TOP 15 WORST OFFENDERS (by relative error):");
console.log("  " + "-".repeat(96));
console.log(`  ${"Formula".padEnd(12)} ${"Tc_ref(K)".padStart(10)} ${"Tc_pred(K)".padStart(11)} ${"AbsErr(K)".padStart(10)} ${"RelErr(%)".padStart(10)} ${"lambda".padStart(8)} ${"omegaLog".padStart(10)} ${"P(GPa)".padStart(8)}`);
console.log("  " + "-".repeat(96));
for (const r of results.slice(0, 15)) {
  console.log(`  ${r.formula.padEnd(12)} ${r.tcRef.toFixed(1).padStart(10)} ${r.tcPred.toFixed(1).padStart(11)} ${r.absErr.toFixed(1).padStart(10)} ${r.relErr.toFixed(1).padStart(10)} ${r.lambda.toFixed(2).padStart(8)} ${r.omegaLog.toFixed(0).padStart(10)} ${r.pressure.toFixed(0).padStart(8)}`);
}

// All compounds sorted by formula
console.log("\n  ALL RESULTS (sorted by relative error):");
console.log("  " + "-".repeat(96));
for (const r of results) {
  const flag = r.relErr > 15 ? " !!!" : r.relErr > 10 ? " !!" : "";
  console.log(`  ${r.formula.padEnd(12)} ${r.tcRef.toFixed(1).padStart(10)} ${r.tcPred.toFixed(1).padStart(11)} ${r.absErr.toFixed(1).padStart(10)} ${r.relErr.toFixed(1).padStart(10)}${flag}`);
}

console.log(`\nBENCHMARK: MAE=${maeK.toFixed(1)}K, MAPE=${mape.toFixed(1)}%, compounds=${results.length}, target=<10%`);
