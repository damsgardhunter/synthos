# Physics Engine Validator — One Iteration

You are running as the autonomous physics validator for the **Quantum Alchemy Engine (QAE)**. Your job is to run the Allen-Dynes Tc predictor against literature reference compounds and fix the physics engine code until predictions converge to within ±15% of measured Tc for all compounds.

Use `/loop 10m /physics-validator` to run on a schedule.

---

## STEP 1 — Run the Validation Harness

```bash
npx tsx --env-file=.env scripts/validate-physics.ts
```

This writes `logs/physics-validation-report.json` with:
- `aggregateUncalibrated.maeAbs` / `maeRel` — primary metric (calibrated numbers are circular)
- `counts.failures` — compounds with relErr > 30%
- `byFamily` — family-grouped MAE, sorted worst-first
- `perCompound` — per-compound detail, sorted by absolute error

## STEP 2 — Read State

```
logs/physics-validation-state.json
```

Fields:
- `cleanRuns` — consecutive runs with `maeRel <= targetMaeRel` AND `failures = 0`
- `fixHistory` — last 20 fixes (cycle, file, line, hypothesis, MAE before/after)

## STEP 3 — Check Stopping Condition

If `cleanRuns >= 10` output:
```
╔══════════════════════════════════════════════════════════╗
║  PHYSICS VALIDATION COMPLETE                             ║
║  10 consecutive runs with MAE <= target and 0 failures.  ║
║  Physics engine is validated. You can stop the loop.     ║
╚══════════════════════════════════════════════════════════╝
```
Bump `lastRun` timestamp and exit.

## STEP 4 — Decide What to Fix

Priority order:

1. **Systematic family bias** — if an entire family (e.g. `conventional`, `A15`, `superhydride`) has `maeRel > 0.25`, the solver or calibration is wrong for that class. This is ONE fix that clears many compounds.

2. **Single-compound outliers** — a compound with `errRel > 0.50` while its family is otherwise fine. Could be:
   - Wrong literature value in `VERIFIED_COMPOUNDS` → verify against primary source
   - Edge case in solver (e.g. very soft phonons like Hg)
   - Heavy-atom / clathrate gate mis-firing

3. **Calibrated-MAE blowup** — if `aggregateCalibrated.maeRel > aggregateUncalibrated.maeRel` the calibration function in `getCalibrationFactor` is broken. Investigate `computeCalibrationFactors` in `physics-engine.ts`.

**Do NOT** blindly tweak constants to move one compound closer if it will move others farther. Every change must be justified by physics and measured by running the harness again.

## STEP 5 — Free Investigation Mode

**Banned phrases in `openIssues`:** "Holding", "Monitoring", "Need more data", "Will check next cycle".

On the 3rd consecutive run with the same failure pattern and no fix applied, you must do ONE of:

1. **Apply a code fix.** Read the source, find the actual cause, edit, re-run validation. Record in `fixHistory`.

2. **Revert an earlier wrong fix.** If MAE went UP after your last edit, revert that edit and document why (e.g. "reverted λ-cap change — it suppressed LaH10 from 250→180K").

3. **Correct a reference value.** If you can cite a peer-reviewed source for a different literature value, update `VERIFIED_COMPOUNDS` and note the source in a comment on that line.

4. **Escalate.** Write `NEEDS USER DECISION: <one-line question>` in `openIssues`. Do not add further analysis until the user responds.

## STEP 6 — Common Fix Locations

| Symptom | Likely Location |
|---------|-----------------|
| All superhydrides under-predicted | `hydrideStrongCouplingTc` / `predictTcEliashberg` λ>1 branch in `physics-engine.ts` |
| All conventional metals over-predicted | Allen-Dynes `f1`/`f2` Lambda2 formula (`physics-engine.ts:~2265`) |
| Hg (ω_log=36) way off | Low-ω_log edge: denominator clamp / strongCouplingCorrection at λ>1.5 |
| MgB2 off by >50% | `omega2Avg` not set — solver defaults to `omegaRatio=1` which ignores two-gap spectrum |
| Calibrated MAE > Uncalibrated MAE | `computeCalibrationFactors` / `getCalibrationFactor` — the similarity kernel is mis-weighted |
| New `applyHydrideSanityGate` clamping legitimate compound | Check `CLATHRATE_CAPABLE_METALS` list — missing Th, Lu, etc. |
| New `LAMBDA_HARD_CAP` killing LaH10 | LaH10 λ=3.41 is right at cap=3.5; verify cap isn't firing spuriously |

## STEP 7 — Update State and Report

Update `logs/physics-validation-state.json`:

```json
{
  "cleanRuns": <inc if pass, else 0>,
  "totalRuns": <inc by 1>,
  "lastRun": "<ISO>",
  "lastStatus": "clean | fixed | issues_remain",
  "lastMaeRelUncalibrated": <from report>,
  "lastFailureCount": <from report>,
  "openIssues": [...],
  "fixHistory": [
    { "run": N, "timestamp": "...", "file": "physics-engine.ts", "line": 2150, "hypothesis": "...", "maeBeforeRel": X, "maeAfterRel": Y, "compoundsFixed": [...] },
    ...last 20
  ]
}
```

Output the cycle summary:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Physics Validator — Run #<N>  [<timestamp>]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clean streak:   <cleanRuns> / 10
MAE (rel, unc): <x>%   (target: 15%)
MAE (abs, unc): <x> K
Failures:       <n>
Warnings:       <n>
Worst family:   <family> @ <mae>%

<bullet: what was fixed this run, or "No fix this run">
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## IMPORTANT RULES

- **DO NOT** change a `tcRef` or `lambda` in `VERIFIED_COMPOUNDS` without citing a specific DOI in the commit-ready comment.
- **DO NOT** add compounds to `VERIFIED_COMPOUNDS` during this loop — that's a separate task handled by the dataset expansion flow.
- **DO NOT** use the calibration factor to "fix" MAE — calibration uses the same reference set, so it's circular. Only the `uncalibrated` MAE counts.
- **DO NOT** edit the validation script to widen tolerances. Tolerances are ±15% / ±30% / fail; that's the standard.
- **Read the source file before editing** — never edit blind.
- **Re-run the harness after every edit** to confirm the fix helped.
- **Commit nothing** — user reviews all changes.
