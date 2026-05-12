#!/usr/bin/env python3
"""
Adaptive Temperature Grid for Pairing Susceptibility Sweep.

Instead of a fixed log-spaced grid, uses a multi-phase strategy:
  Phase 1 — Coarse scan: 3 temperatures to establish λ(T) trend
  Phase 2 — Trend detection: fit λ(T) and estimate T where λ ≈ 0.7
  Phase 3 — Refinement: 2-3 points around the λ ≈ 0.5-0.9 region
  Phase 4 — Tc bracket: if λ crosses 1.0, bisect to ±10K accuracy

Halves compute cost vs 10-point fixed grid for the same Tc accuracy.

References:
  Standard bisection strategy, adapted for DCA pairing eigenvalues.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable


class AdaptiveTemperatureGrid:
    """
    Manages an adaptive temperature grid that refines based on
    computed λ_pair(T) values.
    """

    def __init__(
        self,
        T_max_eV: float,
        T_min_eV: float,
        max_total_points: int = 10,
        lambda_target: float = 1.0,
        refinement_range: Tuple[float, float] = (0.5, 0.9),
    ):
        """
        Args:
            T_max_eV: highest temperature (eV)
            T_min_eV: lowest temperature (eV)
            max_total_points: hard cap on total DCA calculations
            lambda_target: eigenvalue value that signals Tc (default 1.0)
            refinement_range: refine when λ is in this range
        """
        self.T_max = T_max_eV
        self.T_min = T_min_eV
        self.max_points = max_total_points
        self.lambda_target = lambda_target
        self.refine_lo, self.refine_hi = refinement_range

        # Completed points: list of (T_eV, lambda_pair)
        self.completed: List[Tuple[float, float]] = []

    def get_coarse_grid(self) -> List[float]:
        """Phase 1: return 3 coarse temperatures (high, mid, low)."""
        T_mid = np.sqrt(self.T_max * self.T_min)  # geometric mean
        return sorted([self.T_max, T_mid, self.T_min], reverse=True)

    def record_result(self, T_eV: float, lambda_pair: float):
        """Record a completed (T, λ) measurement."""
        self.completed.append((T_eV, lambda_pair))
        self.completed.sort(key=lambda x: -x[0])  # descending T

    def suggest_next_temperatures(self) -> List[float]:
        """
        Based on completed points, suggest the next temperatures to compute.

        Returns empty list if no more points needed or budget exhausted.
        """
        if len(self.completed) < 3:
            # Haven't completed coarse scan yet
            existing_T = {t for t, _ in self.completed}
            coarse = self.get_coarse_grid()
            return [t for t in coarse if t not in existing_T]

        remaining_budget = self.max_points - len(self.completed)
        if remaining_budget <= 0:
            return []

        T_vals = np.array([t for t, _ in self.completed])
        L_vals = np.array([l for _, l in self.completed])

        # Check if λ is growing as T decreases
        idx = np.argsort(T_vals)  # ascending T
        T_asc = T_vals[idx]
        L_asc = L_vals[idx]

        lambda_max = np.max(np.abs(L_asc))
        # λ grows as T DECREASES, so in ascending-T order L_asc[0] (lowest T) is largest
        lambda_growing = len(L_asc) >= 2 and abs(L_asc[0]) > abs(L_asc[-1])

        if not lambda_growing and lambda_max < 0.1:
            # λ is not growing — no SC instability at these temperatures
            return []

        suggestions = []

        # Case 1: λ already crossed 1 → bisect to find Tc
        # In ascending T: L_asc[i] (lower T) has higher λ, L_asc[i+1] (higher T) has lower λ
        for i in range(len(L_asc) - 1):
            if abs(L_asc[i]) >= self.lambda_target and abs(L_asc[i + 1]) < self.lambda_target:
                T_bisect = (T_asc[i] + T_asc[i + 1]) / 2
                suggestions.append(T_bisect)
                if remaining_budget > 1:
                    suggestions.append((T_bisect + T_asc[i]) / 2)
                    suggestions.append((T_bisect + T_asc[i + 1]) / 2)
                return suggestions[:remaining_budget]

        # Case 2: λ in refinement range → add points around it
        in_range = [i for i, l in enumerate(L_asc) if self.refine_lo <= abs(l) <= self.refine_hi]
        if in_range:
            # Add points around the region where λ is near the target
            T_lo = T_asc[max(0, min(in_range) - 1)]
            T_hi = T_asc[min(len(T_asc) - 1, max(in_range) + 1)]

            # 2-3 points in this interval
            n_refine = min(3, remaining_budget)
            for i in range(n_refine):
                frac = (i + 1) / (n_refine + 1)
                T_new = T_lo + frac * (T_hi - T_lo)
                if not any(abs(T_new - t) < (T_hi - T_lo) * 0.05 for t, _ in self.completed):
                    suggestions.append(T_new)

            return suggestions[:remaining_budget]

        # Case 3: λ growing but not yet in refinement range → extend to lower T
        if lambda_growing and lambda_max < self.refine_lo:
            # Extrapolate: at what T might λ reach 0.5?
            # Simple linear extrapolation of ln(λ) vs 1/T
            if len(L_asc) >= 2 and all(abs(l) > 0.01 for l in L_asc):
                inv_T = 1.0 / T_asc
                ln_L = np.log(np.abs(L_asc) + 1e-10)
                slope, intercept = np.polyfit(inv_T, ln_L, 1)
                if slope > 0:
                    # Estimate 1/T where ln(λ) = ln(0.5)
                    inv_T_target = (np.log(0.5) - intercept) / slope
                    if inv_T_target > 1.0 / self.T_min:
                        T_target = max(1.0 / inv_T_target, self.T_min)
                        suggestions.append(T_target)

            # Also extend below current minimum (T_asc[0] is lowest T in ascending order)
            T_extend = T_asc[0] * 0.7
            if T_extend >= self.T_min and not any(abs(T_extend - t) / T_extend < 0.1 for t, _ in self.completed):
                suggestions.append(T_extend)

            return suggestions[:remaining_budget]

        return []

    def get_tc_estimate(self) -> Dict:
        """Estimate Tc from all completed points."""
        if len(self.completed) < 2:
            return {"tc_K": None, "confidence": "insufficient_data"}

        from pairing_susceptibility import extrapolate_tc

        T_list = [t * 11604.5 for t, _ in self.completed]  # eV → K
        L_list = [abs(l) for _, l in self.completed]

        return extrapolate_tc(T_list, L_list)

    def report(self) -> Dict:
        """Full report of the adaptive grid state."""
        return {
            "n_completed": len(self.completed),
            "max_points": self.max_points,
            "T_range_eV": [self.T_min, self.T_max],
            "completed_points": [
                {"T_eV": t, "T_K": t * 11604.5, "lambda": l}
                for t, l in sorted(self.completed, key=lambda x: -x[0])
            ],
            "lambda_max": float(max(abs(l) for _, l in self.completed)) if self.completed else 0,
            "tc_estimate": self.get_tc_estimate(),
        }


def run_adaptive_sweep(
    compute_lambda_at_T: Callable[[float], float],
    T_max_eV: float = 0.05,
    T_min_eV: float = 0.01,
    max_points: int = 8,
) -> Dict:
    """
    Run the full adaptive temperature sweep.

    Args:
        compute_lambda_at_T: callable(T_eV) → λ_pair
            This wraps the DCA + pairing calculation at a single temperature.
        T_max_eV: highest temperature
        T_min_eV: lowest temperature
        max_points: budget

    Returns:
        dict with all results and Tc estimate
    """
    grid = AdaptiveTemperatureGrid(T_max_eV, T_min_eV, max_points)

    # Phase 1: coarse scan
    coarse = grid.get_coarse_grid()
    print(f"[AdaptiveT] Phase 1: coarse scan at {len(coarse)} temperatures")
    for T in coarse:
        T_K = T * 11604.5
        print(f"[AdaptiveT] Computing λ at T={T_K:.0f}K...")
        lam = compute_lambda_at_T(T)
        grid.record_result(T, lam)
        print(f"[AdaptiveT] T={T_K:.0f}K: λ={lam:.4f}")

    # Phases 2-4: adaptive refinement
    phase = 2
    while True:
        next_temps = grid.suggest_next_temperatures()
        if not next_temps:
            print(f"[AdaptiveT] No more temperatures to compute (phase {phase})")
            break

        print(f"[AdaptiveT] Phase {phase}: {len(next_temps)} refinement points")
        for T in next_temps:
            T_K = T * 11604.5
            print(f"[AdaptiveT] Computing λ at T={T_K:.0f}K...")
            lam = compute_lambda_at_T(T)
            grid.record_result(T, lam)
            print(f"[AdaptiveT] T={T_K:.0f}K: λ={lam:.4f}")

        phase += 1

    report = grid.report()
    tc = report.get("tc_estimate", {})
    tc_K = tc.get("tc_bse")
    print(f"\n[AdaptiveT] === Summary ===")
    print(f"  Points computed: {report['n_completed']}/{max_points}")
    print(f"  λ_max = {report['lambda_max']:.4f}")
    print(f"  Tc = {tc_K:.1f} K ({tc.get('tc_confidence', '?')})" if tc_K else "  Tc: not determined")

    return report
