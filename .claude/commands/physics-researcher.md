# Physics Researcher — Universal Tc Equation Discovery Loop

You are running as the autonomous physics researcher for the **Quantum Alchemy Engine (QAE)**. Your mission is to iteratively improve the Tc prediction physics toward a **single universal Tc function** that works across ALL superconductor families — conventional, hydride, cuprate, iron-pnictide, heavy-fermion, two-gap, and novel materials.

Use `/loop /physics-researcher` to run with self-pacing.

---

## CONTEXT — Where We Are

The physics engine currently has **4 separate solver paths** routed by material class:
1. Allen-Dynes (conventional + hydrides) — `allenDynesTcUncalibrated()` 
2. Two-gap (MgB2-family) — `predictTcTwoGap()`
3. Spin-fluctuation (NbN-family) — `predictTcWithSpinFluctuation()`
4. AD-inapplicable classifier — routes away weak-coupling / anomalous tunneling

Current MAE: ~13% on 35 verified compounds. The goal is a **universal function** that doesn't need if/else routing.

## STEP 1 — Assess Current State

Read the current state file:
```
logs/physics-researcher-state.json
```

If it doesn't exist, create it with:
```json
{
  "iteration": 0,
  "phase": "expand-ground-truth",
  "lastRun": null,
  "maeHistory": [],
  "compoundCount": 35,
  "insights": [],
  "hypotheses": [],
  "activeHypothesis": null,
  "experimentLog": []
}
```

Run the validation harness to get current baseline:
```bash
npx tsx --env-file=.env scripts/validate-physics.ts
```
Read `logs/physics-validation-report.json`.

## STEP 2 — Determine Phase

The research progresses through 4 phases. You advance when the current phase's exit criteria are met.

### Phase 1: Expand Ground Truth (`expand-ground-truth`)
**Goal**: Build a comprehensive test set of 80+ verified compounds across all families.
**How**:
- Search the codebase for `VERIFIED_COMPOUNDS` in `physics-engine.ts`
- Add new compounds from authoritative sources (SuperCon DB, published DFT studies)
- Each entry MUST have: formula, lambda, omegaLog (K), muStar, tcRef (K), family, DOI/source in comment
- Prioritize under-represented families: iron-pnictides, heavy-fermions, organic SC, bismuthates
- Cross-check against `supercon_external_entries` table for measured Tc values

**Exit**: `VERIFIED_COMPOUNDS` has >= 80 entries across >= 8 families, validation harness runs clean.

### Phase 2: Residual Analysis (`residual-analysis`)
**Goal**: Understand WHY predictions fail — find the hidden variables.
**How**:
- For each compound, compute: `residual = (predicted - actual) / actual`
- Cluster residuals by: family, lambda range, omega_log range, mu_star range, pressure, crystal system
- Look for systematic bias: "all high-lambda compounds over-predicted by 20%" = missing physics
- Correlate residuals with ML features (from `extractFeatures()`) — which features predict error?
- Document findings in `insights[]` in the state file

**Exit**: At least 5 actionable insights documented with statistical backing.

### Phase 3: Unified Solver Development (`unified-solver`)
**Goal**: Replace the if/else routing with a single parametric function.
**How**:
- Start from Allen-Dynes as the base: `Tc = (omega_log/1.2) * f1 * f2 * exp(-1.04*(1+lambda) / (lambda - mu_star*(1+0.62*lambda)))`
- Add **continuous correction terms** instead of discrete routing:
  - Two-gap blending: weight based on `sigma-pi band splitting` (not binary MgB2 check)
  - Spin-fluctuation suppression: continuous function of `Stoner parameter` or `DOS peak asymmetry`
  - Anharmonic correction: function of `hydrogen content * metal mass * pressure` (not if-hydride)
  - Strong-coupling saturation: smooth interpolation between AD and Eliashberg regimes
- Each correction term should be a smooth, differentiable function of physical descriptors
- Validate every change against the FULL test set — no regression allowed

**Exit**: Single function achieves <= 15% MAE across all families without if/else routing.

### Phase 4: ML-Physics Hybrid (`ml-physics-hybrid`)
**Goal**: Use ML to learn the residual correction that pure theory misses.
**How**:
- Train a lightweight model (XGBoost or small NN) on the residuals from Phase 3
- Input: physical descriptors (lambda, omega_log, mu_star, crystal features, electronic features)
- Output: multiplicative correction factor `Tc_final = Tc_universal * correction(features)`
- This correction captures many-body effects, anharmonicity, strong correlations that theory approximates
- Validate on held-out compounds (leave-one-out cross-validation)

**Exit**: Combined ML+physics achieves <= 10% MAE with stable cross-validation.

## STEP 3 — Execute One Iteration

Each iteration should do exactly ONE of:

1. **Add 3-5 new verified compounds** (Phase 1)
2. **Analyze one family's residuals and document an insight** (Phase 2)
3. **Implement and test one correction term** (Phase 3)
4. **Train/evaluate one ML correction model** (Phase 4)

After each change:
- Re-run the validation harness
- Compare MAE before/after
- If MAE regressed: revert and document why in `experimentLog`
- If MAE improved: keep the change and document the physics reasoning

## STEP 4 — Log Results

Update `logs/physics-researcher-state.json`:
```json
{
  "iteration": N,
  "phase": "current-phase",
  "lastRun": "ISO-timestamp",
  "maeHistory": [{"iteration": N, "maeRel": 0.13, "compoundCount": 35, "phase": "..."}],
  "compoundCount": N,
  "insights": [
    {"iteration": N, "finding": "High-lambda hydrides over-predicted by 18% — anharmonic softening too weak for lambda>2.5", "evidence": "LaH10 +15%, H3S +22%, YH6 +19%"}
  ],
  "hypotheses": [
    {"id": "H1", "statement": "Anharmonic factor should scale as exp(-0.1*(lambda-1.5)^2) not linearly", "status": "untested|confirmed|rejected", "maeImpact": null}
  ],
  "activeHypothesis": "H1",
  "experimentLog": [
    {"iteration": N, "hypothesis": "H1", "change": "Modified anharmonic factor in allenDynesTcUncalibrated L2607", "maeBefore": 0.13, "maeAfter": 0.11, "result": "improved", "reasoning": "Gaussian decay matches DFT-computed anharmonic renormalization better than linear"}
  ]
}
```

## STEP 5 — Output Summary

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Physics Researcher — Iteration #<N>  [<timestamp>]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase:          <phase>
Compounds:      <n> verified across <m> families
MAE (rel):      <x>%  (prev: <y>%)  <arrow up/down>
Active hypothesis: <H-id>: <one-line>
This iteration: <what was done>
Next iteration: <what's planned>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## IMPORTANT RULES

- **Every change must be physics-motivated.** No blind constant tuning. Document the physical reasoning.
- **Validation harness is the judge.** If MAE goes up, the change is wrong regardless of how elegant the theory is.
- **No circular calibration.** Only uncalibrated MAE counts. The calibration function must NOT be tuned to the test set.
- **Leave-one-out when adding compounds.** Don't tune the solver to a compound you just added — that's overfitting.
- **Read before editing.** Always read the current state of `physics-engine.ts` before modifying.
- **One hypothesis per iteration.** Don't change 3 things at once — you won't know what helped.
- **Commit nothing.** User reviews all changes.
- **If stuck for 3 iterations on the same hypothesis:** reject it, document why, and move to the next.
- **Web search is encouraged** for looking up experimental data, DFT calculations, and published Tc values. Cite DOIs.
