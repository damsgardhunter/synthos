/**
 * Standalone physics recalculation script.
 * Updates all candidates to PHYSICS_VERSION=25 without competing with the engine.
 * Run: OPENAI_API_KEY=sk-dummy npx tsx --env-file=.env scripts/recalc-physics-v25.ts
 */
import { db } from "../server/db";
import { computePhysicsTcUQ } from "../server/learning/physics-engine";

const PHYSICS_VERSION = 26; // Force re-run even if v25 tag was set without updating Tc
const BATCH_SIZE = 50;

async function run() {
  // Count total candidates
  const countResult = await db.execute(
    `SELECT COUNT(*) as count FROM superconductor_candidates`
  );
  const totalNeeded = Number((countResult as any).rows?.[0]?.count ?? 0);
  console.log(`[Recalc] ${totalNeeded} total candidates — recalculating ALL to v${PHYSICS_VERSION}`);

  if (totalNeeded === 0) {
    console.log("[Recalc] All candidates up to date!");
    process.exit(0);
  }

  let updated = 0;
  let errors = 0;
  const startMs = Date.now();

  while (true) {
    // Fetch next batch — use physicsVersion < 26 to get ALL that haven't been
    // recalculated by THIS script run (v26 = forced full recalc)
    const rows = await db.execute(
      `SELECT id, formula, predicted_tc, pressure_gpa, ml_features
       FROM superconductor_candidates
       WHERE (ml_features->>'physicsVersion')::int IS DISTINCT FROM ${PHYSICS_VERSION}
       ORDER BY predicted_tc DESC NULLS LAST
       LIMIT ${BATCH_SIZE}`
    );
    const candidates: any[] = (rows as any).rows ?? [];
    if (candidates.length === 0) break;

    for (const c of candidates) {
      try {
        const pressure = c.pressure_gpa ?? 0;
        const uq = computePhysicsTcUQ(c.formula, pressure);
        const newTc = uq.mean > 0 ? Math.round(uq.mean * 10) / 10 : 0;
        const oldTc = c.predicted_tc;

        // Update the candidate
        const features = typeof c.ml_features === 'string' ? JSON.parse(c.ml_features) : (c.ml_features ?? {});
        features.physicsVersion = PHYSICS_VERSION;

        await db.execute(
          `UPDATE superconductor_candidates
           SET predicted_tc = ${newTc},
               ml_features = '${JSON.stringify(features).replace(/'/g, "''")}'::jsonb
           WHERE id = '${c.id}'`
        );

        updated++;
        const change = oldTc ? `${oldTc}K → ${newTc}K` : `→ ${newTc}K`;
        if (Math.abs((oldTc ?? 0) - newTc) > 10) {
          console.log(`  [${updated}/${totalNeeded}] ${c.formula} @ ${pressure}GPa: ${change} (significant change)`);
        }
      } catch (err: any) {
        errors++;
        console.warn(`  ERROR ${c.formula}: ${err.message?.slice(0, 80)}`);
      }

      // Yield every 5 candidates to prevent event loop starvation
      if (updated % 5 === 0) {
        await new Promise(r => setTimeout(r, 10));
      }
    }

    const elapsed = ((Date.now() - startMs) / 1000).toFixed(0);
    const rate = (updated / Math.max(1, (Date.now() - startMs) / 1000)).toFixed(1);
    console.log(`[Recalc] Progress: ${updated}/${totalNeeded} (${rate}/s, ${elapsed}s elapsed, ${errors} errors)`);

    // Warm DB connection between batches
    try { await db.execute("SELECT 1"); } catch { await new Promise(r => setTimeout(r, 2000)); }
  }

  const elapsed = ((Date.now() - startMs) / 1000).toFixed(1);
  console.log(`\n[Recalc] Done! Updated ${updated} candidates in ${elapsed}s (${errors} errors)`);
  process.exit(0);
}

run().catch(err => {
  console.error("[Recalc] Fatal:", err);
  process.exit(1);
});
