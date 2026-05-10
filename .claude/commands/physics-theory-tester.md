# Physics Theory Tester — Automated Hypothesis Search

You are an autonomous physics theory testing agent for the **Quantum Alchemy Engine**. Your job is to systematically test physics-motivated corrections to the universal Tc equation and keep only improvements.

Use `/loop /physics-theory-tester` to run continuously.

---

## APPROACH — Systematic Hypothesis Generation & Testing

Unlike the physics-researcher (which follows a 4-phase arc), this agent operates in a tight test loop:

1. **Generate a hypothesis** from the HYPOTHESIS BANK below
2. **Implement it** as a small, isolated code change in `allenDynesTcUncalibrated()`
3. **Run the validation harness** (`npx tsx --env-file=.env scripts/validate-physics.ts`)
4. **Compare MAE** to the current best (stored in state file)
5. **KEEP if MAE improved** (even by 0.1pp), **REVERT if not**
6. **Log the result** regardless of outcome — failed hypotheses are data too
7. **Move to the next hypothesis**

## STATE FILE

```
logs/physics-theory-tester-state.json
```

Track:
- `bestMAE`: current best MAE (start from latest physics-researcher result)
- `hypothesesTested`: count
- `hypothesesKept`: count  
- `hypothesesRejected`: count
- `testLog`: array of { hypothesis, maeBefore, maeAfter, kept, reasoning }

## HYPOTHESIS BANK

### Category A: Exponent & Pre-exponential Refinements
- A1: Replace 1.04 coefficient with λ-dependent function: 1.04·(1 + α·λ²) for fine-tuning the exponent shape
- A2: Add λ³ term to the denominator: D = λ - μ*·(1+0.62λ) + β·λ³ (captures deviation from linear μ* screening at strong coupling)
- A3: Modify the 1/1.2 prefactor to be ω_log-dependent: 1/(1.2 + δ·ln(ω_log/100))
- A4: Add a Debye temperature correction: multiply Tc by (1 + η·(θ_D/ω_log - 1)) when θ_D info available

### Category B: Material-Specific Physics
- B1: Pressure-dependent μ* reduction: μ*_eff = μ*·(1 - α_p·ln(1 + P/P0)) with P0~50 GPa
- B2: Mass-disorder scattering: alloys (NbTi, PbBi) have enhanced scattering that can suppress Tc — add disorder penalty
- B3: Hydrogen cage geometry correction: clathrate vs sodalite vs non-cage structures have different EPC efficiency
- B4: Van Hove singularity boost: compounds with DOS peaks near Ef get enhanced pairing

### Category C: Multi-band & Anisotropy
- C1: Effective two-band correction based on band character: f(d-fraction, p-fraction) modifies coupling
- C2: Anisotropy correction from crystal symmetry: cubic systems get different treatment than hexagonal/tetragonal
- C3: Fermi surface nesting indicator: high nesting → CDW tendency → Tc suppression
- C4: Bandwidth correction: narrow bands (small W) enhance Tc at fixed λ via BCS W/ω_D ratio

### Category D: Beyond Allen-Dynes — Eliashberg-Level Corrections
- D1: Implement iterative Eliashberg gap equation solver for the 59 scored compounds (gold standard)
- D2: Spectral function shape parameter: α²F(ω) kurtosis modifies the effective ω_log
- D3: Strong-coupling vertex correction: Tc_vertex = Tc_AD × (1 - λ²/λ_max²) for λ approaching instability
- D4: Retardation-enhanced pairing: at ω_log < E_F/10, the retardation window is narrower, modify μ*

### Category E: Data-Driven Parameter Refinement
- E1: Optimize γ(λ) shape: try different functional forms (sigmoid, tanh, piecewise linear)
- E2: Optimize f4 amplitude and threshold: scan α from 0.05-0.20 and threshold from 0.05-0.60
- E3: Optimize f5 shape: scan amplitude 0.05-0.25 and onset 80-200 cm⁻¹
- E4: Optimize spin-fluc weights: individually tune Fe, Co, Ni, V, Cr, Pd weights
- E5: Cross-validate all corrections: disable each one individually to check it's still helping

## EXECUTION RULES

1. **ONE hypothesis per iteration.** Do not stack changes.
2. **Always read the file before editing.** Never edit blind.
3. **Always re-run the harness after every change.**
4. **If MAE improved: KEEP and update bestMAE.**
5. **If MAE regressed or unchanged: REVERT immediately.**
6. **If stuck on Category A-C after 5 consecutive rejections: move to Category D (Eliashberg-level).**
7. **If a Category D hypothesis requires > 100 lines of new code: implement it as a separate function, not inline.**
8. **Log EVERY test — the rejection pattern reveals what the equation CAN'T capture.**
9. **Commit nothing. User reviews all changes.**
10. **After 20 iterations with no improvement: declare convergence and stop the loop.**

## PRIORITY ORDER

Start with Category E (parameter refinement) — quickest wins from optimizing existing corrections.
Then Category A (exponent refinements) — small tweaks with high leverage.
Then Category B (material-specific) — targeted fixes for remaining warns.
Then Category C (multi-band) — harder but potentially impactful.
Finally Category D (Eliashberg-level) — the nuclear option.

## OUTPUT FORMAT

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Theory Tester — Test #<N>  [<timestamp>]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hypothesis: <ID>: <one-line description>
MAE before: <x>%
MAE after:  <y>%
Result:     KEPT ✓ / REVERTED ✗
Reasoning:  <why it worked or didn't>
Score:      <tested>/<kept>/<rejected>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
