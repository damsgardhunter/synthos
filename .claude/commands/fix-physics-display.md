# Fix Physics Display — Unified Tc Prediction Loop

You are fixing a critical bug: the stored `predictedTc` does NOT match the "Physics (lambda)" value on the consensus panel. Your job is to make them identical by fixing `computePhysicsTcUQ` to match our corrected physics engine, with proper pressure support.

Use `/loop /fix-physics-display` to run continuously.

---

## CONTEXT

Two physics calculations exist and DISAGREE:
1. **"Physics (lambda): 190.1K"** — consensus panel, from `computeUnifiedCI` → `computePhysicsTcUQ(formula)` at pressure=0
2. **Stored `predictedTc`: 313K** — from `computePhysicsTcUQ(formula, 170)` at stored pressure

The problem: `computePhysicsTcUQ` gives wildly different results at different pressures because it uses the HEURISTIC lambda from `computeElectronPhononCoupling` which doesn't properly account for pressure in many cases. Meanwhile, our corrected `allenDynesTcUncalibrated` with 10 corrections IS accurate (8.3% MAE on 56 verified compounds).

## PHASE 1: Trace Both Code Paths

Read these files and document EXACTLY how each value is computed:

1. `server/learning/ml-predictor.ts` — `computeUnifiedCI()` function, specifically the `physicsUQ` field
2. `server/learning/physics-engine.ts` — `computePhysicsTcUQ()` function  
3. `server/learning/physics-engine.ts` — `computeElectronPhononCoupling()` — where lambda comes from
4. `server/learning/physics-engine.ts` — `allenDynesTcUncalibrated()` — our corrected equation

Document:
- What lambda value does `computePhysicsTcUQ` use vs what our corrected equation uses for VERIFIED compounds?
- How does pressure affect the heuristic lambda?
- Why does Gd2H6 get 190K at P=0 but 313K at P=170?

## PHASE 2: Add Pressure to VERIFIED_COMPOUNDS

For every compound in VERIFIED_COMPOUNDS that has `pressureGpa > 0`:
- Verify the pressure is correct from literature
- Add pressure-dependent omega_log if available (phonon hardening at high P)
- Ensure the corrected AD equation uses pressure properly via the `pressureGpa` parameter

Search literature for pressure-dependent EPC parameters for key hydrides:
- LaH10 at 170 GPa
- H3S at 155 GPa  
- CaH6 at 172 GPa
- YH6 at 166 GPa

## PHASE 3: Fix computePhysicsTcUQ

Fix `computePhysicsTcUQ` so it:
1. Uses the SAME corrected `allenDynesTcUncalibrated` with all 10 corrections
2. Properly passes pressure to ALL computations
3. For VERIFIED_COMPOUNDS: uses the verified lambda/omegaLog/muStar (not heuristic)
4. For unknown compounds: uses the heuristic lambda BUT with pressure-aware corrections
5. Returns the SAME Tc regardless of whether called from unified-ci or recalculatePhysics

The fix should ensure:
- `computePhysicsTcUQ("LaH10", 170).mean` ≈ 224K (matching our verified prediction)
- `computePhysicsTcUQ("Gd2H6", 170).mean` = `computePhysicsTcUQ("Gd2H6", 0).mean` adjusted by pressure
- The "Physics (lambda)" on the consensus panel matches `predictedTc` in the database

## PHASE 4: Update Recalculation

Make `recalculatePhysics()` in engine.ts call `computePhysicsTcUQ(formula, pressure)` with the candidate's stored pressure, AND ensure the result matches what the frontend shows.

Update the unified-ci endpoint to ALSO pass pressure when computing physicsUQ.

## PHASE 5: Verify and Recalculate

Run validation harness to confirm 8.3% MAE still holds.
Bump PHYSICS_VERSION, mark all candidates, restart server.
Monitor until LaH10 is the top SC candidate.

## VERIFICATION CRITERIA

The task is NOT complete until:
- [ ] LaH10 is the #1 SC candidate by predictedTc
- [ ] For any compound, "Physics (lambda): X K" on consensus panel = predictedTc on profile page
- [ ] Pressure is included in predictions
- [ ] Validation harness still passes (0 failures)

## RULES

- **ASK before making changes** if you're unsure about the approach
- **DO NOT remove pressure** from predictions
- **DO NOT change the corrected allenDynesTcUncalibrated** — it's validated at 8.3% MAE
- **ONE change per iteration** — verify before moving on
- **Read before editing** — always read the current state first
- **Re-run validation after every change**
