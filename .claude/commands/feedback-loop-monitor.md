# Feedback Loop MAE Monitor — One Iteration

You are monitoring the **Feedback Loop** of the Quantum Alchemy Engine to verify that the Mean Absolute Error (MAE) is trending downward toward the <10K target.

## Context
The feedback loop compares ML-predicted Tc against ground-truth Tc from DFT/external data. Recent fixes removed the broken local TS GNN from ensemble predictions and added multiplicative bias correction to the conformal calibrator. This monitor checks whether those fixes are working.

## STEP 1 — Fetch current feedback loop stats

```bash
curl -s http://localhost:4000/api/surrogate-fitness/stats 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    mae = d.get('globalMeanAbsError', 'N/A')
    overest = d.get('globalOverestimateRatio', 'N/A')
    evals = d.get('totalEvaluations', 0)
    families = len(d.get('familyCalibrations', []))
    pillars = d.get('pillarDFTFeedback', [])
    pillar_validated = sum(1 for p in pillars if p.get('accuracy', 0) > 0)
    explore = d.get('explorationWeight', 'N/A')
    print(f'Evaluations: {evals}')
    print(f'MAE: {mae}K (target: <10K)')
    print(f'Overestimate Rate: {overest}')
    print(f'Families Tracked: {families}')
    print(f'Exploration Weight: {explore}')
    print(f'Pillars Validated: {pillar_validated}/{len(pillars)}')
    for fc in d.get('familyCalibrations', [])[:5]:
        print(f'  {fc[\"family\"]}: MAE={fc[\"meanAbsError\"]:.1f}K, overest={fc[\"overestimateRatio\"]:.0%}, cal={fc[\"calibrationFactor\"]:.2f}')
    if mae != 'N/A' and mae < 10:
        print('SUCCESS: MAE is below 10K target!')
    elif mae != 'N/A' and mae < 30:
        print('IMPROVING: MAE is below 30K, trending toward target')
    elif mae != 'N/A':
        print(f'NEEDS WORK: MAE={mae}K still above target')
except Exception as e:
    print(f'Error parsing response: {e}')
" || echo "Server not reachable at localhost:5174"
```

## STEP 2 — Fetch calibration state

```bash
curl -s http://localhost:4000/api/calibration/status 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'Temperature Scale: {d.get(\"temperatureScale\", \"N/A\")}')
    print(f'Dataset Size: {d.get(\"calibrationDatasetSize\", 0)}')
    print(f'ECE: {d.get(\"eceBefore\", \"N/A\")} -> {d.get(\"eceAfter\", \"N/A\")}')
    print(f'Coverage@95%: {d.get(\"coverageAtQ95\", \"N/A\")}')
    for f in d.get('perFamily', []):
        print(f'  {f[\"family\"]}: Q95={f[\"q95\"]}, ECE={f[\"ece\"]}, n={f[\"count\"]}')
except Exception as e:
    print(f'Error: {e}')
" || echo "Calibration endpoint not reachable"
```

## STEP 3 — Check prediction ledger metrics

```bash
curl -s http://localhost:5174/api/ml-calibration 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'Ledger entries: {d.get(\"count\", 0)}')
    print(f'Recent MAE: {d.get(\"recentMAE\", \"N/A\")}')
    print(f'Recent bias: {d.get(\"recentBias\", \"N/A\")}')
    print(f'R2: {d.get(\"r2\", \"N/A\")}')
except Exception as e:
    print(f'Error: {e}')
" || echo "ML calibration endpoint not reachable"
```

## STEP 4 — Analyze and report

Based on the data collected:

1. **Report current MAE** and whether it is improving compared to the 76K baseline
2. **Check overestimate rate** — should be dropping from 89% toward 50%
3. **Verify family tracking** — should be >1 family tracked
4. **Check pillar validation** — should start showing >0% accuracy
5. **If MAE > 30K still**: Check server logs for whether local GNN is still being used in ensemble predictions. Look for `[Conformal] Calibrated` log entries showing the bias multiplier.

```bash
grep -i "\[Conformal\] Calibrated" logs/server-latest.log 2>/dev/null | tail -3
grep -i "biasMultiplier" logs/server-latest.log 2>/dev/null | tail -3
```

6. **Report a clear summary**: MAE trend, what's working, what still needs attention.
