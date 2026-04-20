# Tc-UQ Physics Estimator Tuner — One Iteration

You are tuning `computePhysicsTcUQ` and its callees (`allenDynesTcUncalibrated`, `computeElectronPhononCoupling`, `computeTcWithUncertainty`) in `server/learning/physics-engine.ts` to achieve **< 10% MAE** (target ~8%) on the VERIFIED_COMPOUNDS benchmark set.

## What this function does
`computePhysicsTcUQ(formula, pressureGpa)` is the physics-only Tc predictor. It:
1. Calls `computeElectronicStructure(formula)` — heuristic electronic structure
2. Calls `computePhononSpectrum(formula, electronic, pressureGpa)` — heuristic phonon spectrum
3. Calls `computeElectronPhononCoupling(electronic, phonon, formula, pressureGpa)` — returns lambda, omegaLog, muStar
4. Calls `computeTcWithUncertainty(...)` which calls `allenDynesTcUncalibrated(...)` — the core Allen-Dynes Tc formula with 10+ physics corrections

For VERIFIED_COMPOUNDS, step 3 short-circuits and returns the literature lambda/omegaLog/muStar directly. The Tc error comes from `allenDynesTcUncalibrated` mapping those parameters to Tc.

## Each iteration you MUST:

### Step 1: Write and run a benchmark script
Create/update `scripts/benchmark-tc-uq.ts` that:
- Imports `computePhysicsTcUQ` and `VERIFIED_COMPOUNDS` from `server/learning/physics-engine.ts`
- For each compound in VERIFIED_COMPOUNDS where `adApplicable !== false`:
  - Calls `computePhysicsTcUQ(formula, compound.pressureGpa)`
  - Computes: predicted Tc (the `.mean`), reference Tc (`tcRef`), absolute error, relative error (%)
- Computes aggregate stats: **MAE (K)**, **MAPE (%)**, **median APE (%)**, **max error compound**
- Prints a sorted table of the WORST offenders (highest absolute % error) — top 15
- Also prints a summary line: `BENCHMARK: MAE=X.X K, MAPE=X.X%, compounds=N, target=<10%`

Run it with: `npx tsx scripts/benchmark-tc-uq.ts`

### Step 2: Analyze the errors
Look at the benchmark output. Identify:
- Which compounds have > 15% error? Why?
- Are errors systematic (e.g. all hydrides over-predicted, all weak-coupling under-predicted)?
- Which correction factor in `allenDynesTcUncalibrated` is misbehaving?
- Are there missing compounds in VERIFIED_COMPOUNDS that would help calibrate? Add them with DOI citations.

### Step 3: Make targeted fixes
Based on the error analysis, make ONE focused change per iteration. Examples:
- Adjust an anharmonic correction coefficient
- Fix a pressure-dependent scaling
- Add a missing physics correction
- Add verified compounds with pressure data (especially for families with high error)
- Fix the heuristic lambda/omegaLog computation for non-verified compounds
- Calibrate against known experimental data

**IMPORTANT RULES:**
- Do NOT add hard caps or suppressions — magnetism, gaps, etc. are FEATURES not gates
- Do NOT over-fit to one compound at the expense of others — check ALL compounds after each change
- Keep changes small and physically motivated — cite the physics reason
- The function should work for BOTH verified compounds (literature params) AND novel ones (heuristic params)

### Step 4: Re-run benchmark
Run the benchmark again after your fix. Compare before/after MAE.
- If MAE decreased: good, report the improvement
- If MAE increased: revert and try a different approach
- If MAE < 10%: report success!

### Step 5: Report
Print a summary:
```
=== Tc-UQ Tuner Iteration Report ===
Before: MAE=X.X K, MAPE=X.X%
After:  MAE=X.X K, MAPE=X.X%
Change: [description of what you changed and why]
Worst offenders remaining: [top 3]
Target (<10% MAPE): [MET / NOT MET — continue looping]
```

## Key files
- `server/learning/physics-engine.ts` — main file with all functions
  - `VERIFIED_COMPOUNDS` (~line 1899-2241) — reference data
  - `computeElectronPhononCoupling` (~line 2243) — lambda/omegaLog/muStar
  - `allenDynesTcUncalibrated` (~line 2819) — core Tc formula
  - `computeTcWithUncertainty` (~line 3450) — UQ wrapper
  - `computePhysicsTcUQ` (~line 3541) — top-level entry point
- `server/physics/tc-formulas.ts` — McMillan, SISSO, Xie formulas

## Context on H3Sm
The user reported H3Sm (samarium trihydride) predicts 309K which is wildly wrong. SmH3 is a rare-earth trihydride at ambient pressure — Tc should be very low or zero. The heuristic `computeElectronPhononCoupling` is computing absurdly high lambda for it. This is an example of a non-verified compound where the heuristic fails. Fix the heuristic for low-stoichiometry rare-earth hydrides (H:metal ratio < 4) as part of your tuning.
