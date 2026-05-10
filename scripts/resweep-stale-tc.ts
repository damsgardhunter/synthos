/**
 * Sweep stored superconductor_candidates for predictions whose Tc would now
 * be different under the gated allenDynesTcRaw (isAllenDynesApplicable +
 * applyHydrideSanityGate embedded). Dry-run by default — pass --commit to
 * write the new Tc back to the DB.
 *
 * What it catches:
 *   - MoH6N-class: H-rich + clathrate-incapable framework metal → cap to 50K
 *   - Ambient H-rich Tc>100K → cap to 40K
 *   - Anomalous-tunneling hydrides (Pd/Nb/Ta) at low coupling → 0
 *   - Two-gap / spin-fluc compounds routed elsewhere → 0 from AD
 */
import { db } from "../server/db";
import { superconductorCandidates } from "../shared/schema";
import { gt, isNotNull } from "drizzle-orm";
import {
  allenDynesTcRaw,
  isAllenDynesApplicable,
} from "../server/learning/physics-engine";

const COMMIT = process.argv.includes("--commit");
const TC_THRESHOLD = 50; // Re-check candidates with stored Tc > 50K (catches borderline gate cases)

interface DriftRow {
  id: string;
  formula: string;
  pressureGpa: number;
  storedTc: number;
  newTc: number;
  delta: number;
  reason: string;
  lambda: number;
  omegaLog: number;
}

async function main() {
  console.log(`[resweep] mode=${COMMIT ? "COMMIT" : "DRY-RUN"} threshold=Tc>${TC_THRESHOLD}K`);
  const rows = await db
    .select()
    .from(superconductorCandidates)
    .where(gt(superconductorCandidates.predictedTc, TC_THRESHOLD));
  console.log(`[resweep] ${rows.length} candidates above threshold`);

  const drifts: DriftRow[] = [];
  let unchanged = 0;
  let missingInputs = 0;

  for (const r of rows) {
    const lambda = r.electronPhononCoupling ?? 0;
    const omegaLog = r.logPhononFrequency ?? 0;
    const muStar = r.coulombPseudopotential ?? 0.10;
    const pressureGpa = r.pressureGpa ?? 0;
    const storedTc = r.predictedTc ?? 0;

    if (lambda <= 0 || omegaLog <= 0) {
      missingInputs++;
      continue;
    }

    const isHydride = /H\d/.test(r.formula);
    const gatedTc = allenDynesTcRaw(lambda, omegaLog, muStar, undefined, isHydride, r.formula, pressureGpa);
    const ungatedTc = allenDynesTcRaw(lambda, omegaLog, muStar, undefined, isHydride);
    const adCheck = isAllenDynesApplicable(r.formula, lambda, pressureGpa);

    // True gate firing = ungated and gated AD diverge by >5K with formula provided.
    // (Calibration alone can't change between calls — only the gates can.)
    const gateFired = Math.abs(ungatedTc - gatedTc) > 5 || !adCheck.applicable;
    if (!gateFired) {
      unchanged++;
      continue;
    }

    const newTc = gatedTc;
    const delta = Math.abs(newTc - storedTc);
    const reason = !adCheck.applicable
      ? `AD inapplicable: ${adCheck.reason}`
      : `hydride sanity gate (ungated AD=${ungatedTc.toFixed(0)}K → gated=${gatedTc.toFixed(0)}K)`;

    drifts.push({
      id: r.id,
      formula: r.formula,
      pressureGpa,
      storedTc: Math.round(storedTc * 10) / 10,
      newTc: Math.round(newTc * 10) / 10,
      delta: Math.round(delta * 10) / 10,
      reason,
      lambda: Math.round(lambda * 100) / 100,
      omegaLog: Math.round(omegaLog),
    });
  }

  drifts.sort((a, b) => b.delta - a.delta);

  console.log(`\n[resweep] ${drifts.length} candidates with Tc drift ≥5K (sorted by largest drop):\n`);
  console.table(drifts.slice(0, 50));

  console.log(`\n[resweep] summary: ${drifts.length} drift, ${unchanged} unchanged, ${missingInputs} missing-inputs (skipped)`);

  if (COMMIT && drifts.length > 0) {
    console.log(`\n[resweep] COMMITTING ${drifts.length} updates…`);
    let ok = 0, fail = 0;
    for (const d of drifts) {
      try {
        const noteSuffix = `\n[gate-resweep ${new Date().toISOString().slice(0,10)}] Tc revised ${d.storedTc}K → ${d.newTc}K — ${d.reason}`;
        const existing = await db
          .select({ notes: superconductorCandidates.notes })
          .from(superconductorCandidates)
          .where(eq(superconductorCandidates.id, d.id));
        const newNotes = (existing[0]?.notes ?? "") + noteSuffix;
        await db
          .update(superconductorCandidates)
          .set({ predictedTc: d.newTc, notes: newNotes })
          .where(eq(superconductorCandidates.id, d.id));
        ok++;
      } catch (err) {
        console.warn(`[resweep] failed to update ${d.formula}:`, (err as Error).message);
        fail++;
      }
    }
    console.log(`[resweep] commit complete: ${ok} updated, ${fail} failed`);
  } else if (drifts.length > 0) {
    console.log(`\n[resweep] DRY-RUN — re-run with --commit to apply.`);
  }

  process.exit(0);
}

import { eq } from "drizzle-orm";

main().catch(err => {
  console.error("[resweep] fatal:", err);
  process.exit(1);
});
