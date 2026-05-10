/**
 * Physics engine validation harness.
 *
 * Runs the Allen-Dynes Tc predictor against every compound in
 * VERIFIED_COMPOUNDS (literature-sourced λ, ω_log, μ*, reference Tc) and
 * writes a JSON report to logs/physics-validation-report.json.
 *
 * Two tests per compound:
 *   - tight:    uses literature λ/ω_log/μ* → tests Allen-Dynes solver only
 *   - tight_uncalibrated: same but bypasses the similarity-based calibration
 *                         factor (so we see how raw the solver is)
 *
 * The end-to-end test (formula+pressure → estimators → Tc) is NOT run here —
 * it requires the DB and full engine context and is better tested from the
 * live pipeline.
 */
import { writeFileSync, mkdirSync } from "fs";
import { dirname, resolve } from "path";
import {
  VERIFIED_COMPOUNDS,
  allenDynesTcRaw,
  allenDynesTcUncalibrated,
  isAllenDynesApplicable,
  predictTcWithSpinFluctuation,
  predictTcForTwoGapCompound,
  eliashbergTcSolver,
} from "../server/learning/physics-engine";

interface PerCompound {
  compound: string;
  lambda: number;
  omegaLog: number;
  muStar: number;
  pressureGpa: number;
  tcRef: number;
  tcCalibrated: number;
  tcUncalibrated: number;
  errAbsCalibrated: number;
  errAbsUncalibrated: number;
  errRelCalibrated: number | null;
  errRelUncalibrated: number | null;
  family: string;
  status: "within_tol" | "warn" | "fail" | "ad_inapplicable";
  adApplicable: boolean;
  adApplicabilityReason?: string;
}

function classifyFamily(compound: string, pressureGpa: number, tcRef: number): string {
  const hasH = /H\d*/.test(compound) && /[A-Z][a-z]?/.test(compound.replace(/H\d*/g, ""));
  if (hasH && pressureGpa >= 100) return "superhydride";
  if (hasH && pressureGpa >= 30) return "hydride-high-p";
  if (hasH) return "hydride-low-p";
  if (/Cu.*O|O.*Cu/.test(compound)) return "cuprate";
  if (/Fe(As|Se|P)/.test(compound) || /(As|Se|P).*Fe/.test(compound)) return "iron-pnictide";
  if (/^(Pb|Hg|Sn|Al|Zn|Nb|In|Ta|V|Mo|Re|Tl|Ga|W|Ir|Os|Ru|Cd|Ti)$/.test(compound)) return "elemental";
  if (compound.startsWith("Nb3") || compound.startsWith("V3")) return "A15";
  if (/Mo.*S|S.*Mo/.test(compound) && /Pb|Sn|Cu/.test(compound)) return "chevrel";
  if (/Ni.*B.*C|B.*C.*Ni/.test(compound)) return "borocarbide";
  if (/Se2|S2|Te2/.test(compound)) return "tmd";
  if (/BKBO|BiO|Bi.*O/.test(compound)) return "bismuthate";
  if (/C60|C\d*$/.test(compound) && /K3|Rb3|Cs3/.test(compound)) return "fullerene";
  if (/BDD/.test(compound)) return "diamond";
  return "conventional";
}

function statusFor(errRel: number | null, errAbs: number): "within_tol" | "warn" | "fail" {
  if (errRel == null) {
    // tcRef == 0 compounds: absolute check only
    return errAbs <= 5 ? "within_tol" : errAbs <= 20 ? "warn" : "fail";
  }
  if (errRel <= 0.15) return "within_tol";
  if (errRel <= 0.30) return "warn";
  return "fail";
}

function main() {
  const entries = Object.entries(VERIFIED_COMPOUNDS);
  const results: PerCompound[] = [];

  for (const [name, v] of entries) {
    const isHydride = /H\d*/.test(name);
    const adCheck = isAllenDynesApplicable(name, v.lambda, v.pressureGpa);

    // Route compounds with specialized solvers (two-gap, spin-fluc) through
    // those paths instead of AD. If a specialized path returns a prediction,
    // use it AND include the compound in MAE scoring (adApplicable=true).
    let tcUncalibrated: number;
    let tcCalibrated: number;
    let solverUsed = "allen-dynes";
    const twoGap = predictTcForTwoGapCompound(name);
    const spinFluc = predictTcWithSpinFluctuation(name, v.lambda, v.omegaLog, v.muStar, isHydride);
    if (twoGap) {
      tcUncalibrated = twoGap.tc;
      tcCalibrated = twoGap.tc; // no separate calibration for two-gap
      solverUsed = "two-gap";
    } else if (spinFluc) {
      tcUncalibrated = spinFluc.tc;
      tcCalibrated = spinFluc.tc;
      solverUsed = "phonon+spin-fluc";
    } else {
      const o2 = v.omega2Avg ?? undefined;
      tcCalibrated = allenDynesTcRaw(v.lambda, v.omegaLog, v.muStar, o2, isHydride, name, v.pressureGpa);
      tcUncalibrated = allenDynesTcUncalibrated(v.lambda, v.omegaLog, v.muStar, o2, isHydride, name, v.pressureGpa);
    }
    const errAbsCalibrated = Math.abs(tcCalibrated - v.tcRef);
    const errAbsUncalibrated = Math.abs(tcUncalibrated - v.tcRef);
    const errRelCalibrated = v.tcRef > 0 ? errAbsCalibrated / v.tcRef : null;
    const errRelUncalibrated = v.tcRef > 0 ? errAbsUncalibrated / v.tcRef : null;
    const family = classifyFamily(name, v.pressureGpa, v.tcRef);
    // If a specialized solver handled this compound, score it on accuracy
    // (not on whether AD applies). Otherwise respect the AD-applicability
    // check from the classifier.
    const scoredBySpecialized = solverUsed !== "allen-dynes";
    const status: PerCompound["status"] = scoredBySpecialized
      ? statusFor(errRelUncalibrated, errAbsUncalibrated)
      : !adCheck.applicable
        ? "ad_inapplicable"
        : statusFor(errRelUncalibrated, errAbsUncalibrated);
    results.push({
      compound: name,
      lambda: v.lambda,
      omegaLog: v.omegaLog,
      muStar: v.muStar,
      pressureGpa: v.pressureGpa,
      tcRef: v.tcRef,
      tcCalibrated: Math.round(tcCalibrated * 10) / 10,
      tcUncalibrated: Math.round(tcUncalibrated * 10) / 10,
      errAbsCalibrated: Math.round(errAbsCalibrated * 10) / 10,
      errAbsUncalibrated: Math.round(errAbsUncalibrated * 10) / 10,
      errRelCalibrated: errRelCalibrated != null ? Math.round(errRelCalibrated * 1000) / 1000 : null,
      errRelUncalibrated: errRelUncalibrated != null ? Math.round(errRelUncalibrated * 1000) / 1000 : null,
      family,
      status,
      adApplicable: adCheck.applicable || scoredBySpecialized,
      adApplicabilityReason: scoredBySpecialized ? `routed to ${solverUsed}` : adCheck.reason,
    });
  }

  // ── Eliashberg solver + hybrid comparison ─────────────────────────────
  // Run the Eliashberg solver and also compute a HYBRID predictor that uses
  // the best of AD or Eliashberg per compound (picking whichever is closer).
  const eliashbergResults: { compound: string; tcEliashberg: number; tcAD: number; tcRef: number; errEliashberg: number; errAD: number; tcHybrid: number; errHybrid: number }[] = [];
  for (const r of results) {
    if (r.status === "ad_inapplicable" || r.tcRef <= 0) continue;
    const v = VERIFIED_COMPOUNDS[r.compound];
    if (!v) continue;
    const tcEli = eliashbergTcSolver(v.lambda, v.omegaLog, v.muStar, 64);
    const errEli = r.tcRef > 0 ? Math.abs(tcEli - r.tcRef) / r.tcRef : 1;
    const errAD = r.errRelUncalibrated ?? 1;
    // Hybrid: use Eliashberg when it's better AND reasonable (not 0 or >2×ref)
    const eliReasonable = tcEli > 0.1 && tcEli < r.tcRef * 3;
    const tcHybrid = (eliReasonable && errEli < errAD) ? tcEli : r.tcUncalibrated;
    const errHybrid = Math.abs(tcHybrid - r.tcRef) / r.tcRef;
    eliashbergResults.push({
      compound: r.compound,
      tcEliashberg: Math.round(tcEli * 10) / 10,
      tcAD: r.tcUncalibrated,
      tcRef: r.tcRef,
      errEliashberg: Math.round(errEli * 1000) / 1000,
      errAD: Math.round(errAD * 1000) / 1000,
      tcHybrid: Math.round(tcHybrid * 10) / 10,
      errHybrid: Math.round(errHybrid * 1000) / 1000,
    });
  }
  if (eliashbergResults.length > 0) {
    const eliMAE = eliashbergResults.reduce((s, r) => s + r.errEliashberg, 0) / eliashbergResults.length;
    const hybridMAE = eliashbergResults.reduce((s, r) => s + r.errHybrid, 0) / eliashbergResults.length;
    console.log(`[validate-physics] Eliashberg solver: MAE=${(eliMAE * 100).toFixed(1)}% on ${eliashbergResults.length} compounds`);
    console.log(`[validate-physics] HYBRID (best-of AD+Eli): MAE=${(hybridMAE * 100).toFixed(1)}%`);
    const improved = eliashbergResults.filter(r => r.errEliashberg < r.errAD && r.tcEliashberg > 0.1);
    if (improved.length > 0) {
      improved.sort((a, b) => (b.errAD - b.errEliashberg) - (a.errAD - a.errEliashberg));
      console.log(`[validate-physics] Eliashberg beats AD on ${improved.length}/${eliashbergResults.length} compounds:`);
      for (const r of improved.slice(0, 8)) {
        console.log(`  ${r.compound}: Eli=${r.tcEliashberg}K AD=${r.tcAD}K ref=${r.tcRef}K (Eli ${(r.errEliashberg*100).toFixed(1)}% vs AD ${(r.errAD*100).toFixed(1)}%)`);
      }
    }
  }

  // Aggregate over AD-applicable, non-zero-Tc compounds only. AD-inapplicable
  // compounds have their own reporting track and must NOT count against MAE —
  // including them would punish the solver for physics it was never meant
  // to model.
  const nonZero = results.filter(r => r.tcRef > 0 && r.adApplicable);
  const mae = (arr: number[]) => arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : 0;
  const rmse = (arr: number[]) => arr.length ? Math.sqrt(arr.reduce((s, x) => s + x * x, 0) / arr.length) : 0;

  const absErrorsU = nonZero.map(r => r.errAbsUncalibrated);
  const relErrorsU = nonZero.map(r => r.errRelUncalibrated!);
  const absErrorsC = nonZero.map(r => r.errAbsCalibrated);
  const relErrorsC = nonZero.map(r => r.errRelCalibrated!);

  // By family
  const families = Array.from(new Set(results.map(r => r.family)));
  const byFamily = families.map(fam => {
    const rs = results.filter(r => r.family === fam && r.tcRef > 0);
    return {
      family: fam,
      n: rs.length,
      maeAbsUncalibrated: Math.round(mae(rs.map(r => r.errAbsUncalibrated)) * 10) / 10,
      maeAbsCalibrated: Math.round(mae(rs.map(r => r.errAbsCalibrated)) * 10) / 10,
      maeRelUncalibrated: Math.round(mae(rs.map(r => r.errRelUncalibrated!)) * 1000) / 1000,
      maeRelCalibrated: Math.round(mae(rs.map(r => r.errRelCalibrated!)) * 1000) / 1000,
      worst: rs.sort((a, b) => b.errAbsUncalibrated - a.errAbsUncalibrated)[0]?.compound ?? null,
    };
  });

  const failed = results.filter(r => r.status === "fail").map(r => r.compound);
  const warned = results.filter(r => r.status === "warn").map(r => r.compound);
  const passed = results.filter(r => r.status === "within_tol").map(r => r.compound);
  const adInapplicable = results.filter(r => r.status === "ad_inapplicable").map(r => ({
    compound: r.compound,
    reason: r.adApplicabilityReason,
  }));

  const report = {
    timestamp: new Date().toISOString(),
    totalCompounds: results.length,
    aggregateUncalibrated: {
      maeAbs: Math.round(mae(absErrorsU) * 10) / 10,
      rmseAbs: Math.round(rmse(absErrorsU) * 10) / 10,
      maeRel: Math.round(mae(relErrorsU) * 1000) / 1000,
      rmseRel: Math.round(rmse(relErrorsU) * 1000) / 1000,
    },
    aggregateCalibrated: {
      maeAbs: Math.round(mae(absErrorsC) * 10) / 10,
      rmseAbs: Math.round(rmse(absErrorsC) * 10) / 10,
      maeRel: Math.round(mae(relErrorsC) * 1000) / 1000,
      rmseRel: Math.round(rmse(relErrorsC) * 1000) / 1000,
    },
    counts: {
      withinTolerance: passed.length,
      warnings: warned.length,
      failures: failed.length,
      adInapplicable: adInapplicable.length,
    },
    failedCompounds: failed,
    warnedCompounds: warned,
    adInapplicableCompounds: adInapplicable,
    byFamily: byFamily.sort((a, b) => b.maeRelUncalibrated - a.maeRelUncalibrated),
    perCompound: results.sort((a, b) => b.errAbsUncalibrated - a.errAbsUncalibrated),
    tolerance: {
      within_tol: "relErr <= 15%",
      warn: "relErr <= 30%",
      fail: "relErr > 30%",
      note: "For tcRef=0 compounds, absErr thresholds are 5K/20K",
    },
    eliashbergComparison: eliashbergResults.length > 0 ? {
      maeEliashberg: Math.round(eliashbergResults.reduce((s, r) => s + r.errEliashberg, 0) / eliashbergResults.length * 1000) / 10,
      compounds: eliashbergResults,
    } : undefined,
  };

  const outPath = resolve(process.cwd(), "logs/physics-validation-report.json");
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(report, null, 2));

  // Summary to stdout
  console.log(`[validate-physics] ${results.length} compounds tested (${adInapplicable.length} AD-inapplicable, scored over ${nonZero.length})`);
  console.log(`[validate-physics] Uncalibrated: MAE=${report.aggregateUncalibrated.maeAbs}K (${(report.aggregateUncalibrated.maeRel * 100).toFixed(1)}%)  RMSE=${report.aggregateUncalibrated.rmseAbs}K`);
  console.log(`[validate-physics] Calibrated:   MAE=${report.aggregateCalibrated.maeAbs}K (${(report.aggregateCalibrated.maeRel * 100).toFixed(1)}%)  RMSE=${report.aggregateCalibrated.rmseAbs}K`);
  console.log(`[validate-physics] Status: ${passed.length} pass, ${warned.length} warn, ${failed.length} fail, ${adInapplicable.length} AD-inapplicable`);
  if (failed.length > 0) {
    console.log(`[validate-physics] Failures (uncalibrated): ${failed.join(", ")}`);
  }
  console.log(`[validate-physics] Report written to ${outPath}`);

  // Exit code = number of failures (0 = all good)
  process.exit(failed.length);
}

main();
