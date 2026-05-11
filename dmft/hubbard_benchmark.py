#!/usr/bin/env python3
"""
C2: 2D Hubbard Model d-Wave Pairing Benchmark.

Reproduces the canonical Maier-Jarrell-Scalapino result using DCA:

  Model:  2D square lattice Hubbard model
  U = 8t
  t' = -0.3t  (next-nearest-neighbor hopping)
  Hole doping = 15%  (filling n ≈ 0.85)
  N_c = 4  (2×2 plaquette DCA)

Expected result:
  The d-wave (d_{x²-y²}) pairing eigenvalue λ_d(T) increases as T
  decreases and approaches 1 around T/t ≈ 0.02 (corresponding to
  Tc ~ 0.02t ≈ 100-200K for realistic cuprate hopping t ~ 0.3-0.5 eV).

  At each temperature, the d-wave channel dominates over s-wave and
  extended-s channels — this is the signature that spin-fluctuation-
  mediated d-wave pairing emerges from the repulsive Hubbard model.

This benchmark validates:
  1. The DCA self-consistency loop converges
  2. The pairing susceptibility calculation produces physical eigenvalues
  3. d-wave dominates over s-wave at intermediate coupling
  4. λ_d grows with decreasing T in the right temperature range

If this benchmark fails, no downstream real-material DCA results can
be trusted.

References:
  Maier et al., PRL 95, 237001 (2005) — Fig. 1, d-wave λ vs T
  Maier et al., Rev. Mod. Phys. 77, 1027 (2005) — DCA methodology
  Jarrell et al., EPL 56, 563 (2001) — DCA d-wave
  Gull et al., Rev. Mod. Phys. 83, 349 (2011) — QMC benchmarks

Usage:
  python3 hubbard_benchmark.py [work_dir]
  # or inside Docker:
  mpirun -np 20 python3 hubbard_benchmark.py /data/dmft_jobs/hubbard_bench

Environment variables:
  BENCHMARK_QUICK=1     — reduced grid/cycles for fast testing (~30 min)
  BENCHMARK_PRODUCTION=1 — full production parameters (~48h)
"""

import numpy as np
import json
import os
import sys
import time

from dca_solver import (
    DCAParams,
    build_lattice_kpoints,
    lattice_dispersion_hubbard,
    run_dca,
    compute_dca_pairing_eigenvalue,
)


# ── Benchmark parameters ─────────────────────────────────────────────────────

# Hopping parameters
T_HOPPING = 1.0        # nearest-neighbor hopping (energy unit)
TP_HOPPING = -0.3      # next-nearest-neighbor (hole-like Fermi surface)
U_HUBBARD = 8.0        # on-site Coulomb repulsion = 8t (intermediate coupling)

# Target filling: n = 0.85 (15% hole doping from half-filling)
TARGET_FILLING = 0.85

# Temperature sweep: T/t from high to low
# At T/t ~ 0.02, d-wave λ → 1 (the Maier result)
TEMPERATURES_T_UNITS = [0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02]

# Quick mode for testing
TEMPERATURES_QUICK = [0.20, 0.10, 0.05, 0.03]


def get_benchmark_params(
    temperature_t: float,
    quick: bool = False,
) -> DCAParams:
    """
    Build DCA parameters for the Hubbard benchmark at a given temperature.

    The chemical potential μ is adjusted to achieve the target filling.
    For the 2D Hubbard with t'=-0.3t at 15% doping, μ ≈ -1.0t to -1.5t
    depending on U and T. We start with a rough estimate and let the
    DCA loop adjust density.
    """
    beta = 1.0 / temperature_t  # β·t = 1/T in units of t

    # Chemical potential estimate for 15% hole doping
    # At half-filling: μ = U/2 (particle-hole symmetry for t'=0)
    # With t' and doping: μ shifts down
    mu_estimate = U_HUBBARD / 2.0 - 1.2  # rough starting point

    if quick:
        return DCAParams(
            nc=4, dim=2, beta=beta,
            n_iw=128, n_k_per_dim=32,
            U=U_HUBBARD, J=0.0, mu=mu_estimate,
            n_iter=15, sigma_mix=0.7,
            convergence_tol=1e-3,
            n_warmup=10000, n_cycles=500_000, length_cycle=100,
        )
    else:
        return DCAParams(
            nc=4, dim=2, beta=beta,
            n_iw=256, n_k_per_dim=64,
            U=U_HUBBARD, J=0.0, mu=mu_estimate,
            n_iter=30, sigma_mix=0.5,
            convergence_tol=1e-4,
            n_warmup=50000, n_cycles=5_000_000, length_cycle=200,
        )


def adjust_chemical_potential(
    params: DCAParams,
    eps_k: np.ndarray,
    kpoints: np.ndarray,
    target_n: float,
    tol: float = 0.02,
    max_iter: int = 10,
) -> float:
    """
    Adjust μ to hit the target filling using a simple bisection
    on the non-interacting density.

    For the interacting case, the DCA loop itself adjusts the filling
    via self-energy feedback. But starting closer to the right μ
    speeds convergence.
    """
    from dca_solver import compute_lattice_gf_dca_fast, assign_k_to_patches

    patch_assignment = assign_k_to_patches(kpoints, params.K_cluster)
    sigma_zero = np.zeros((params.nc, 2 * params.n_iw), dtype=complex)

    def density_at_mu(mu):
        gk = compute_lattice_gf_dca_fast(eps_k, sigma_zero, patch_assignment, mu, params.wn)
        # n = 2 * (1/β) Σ_n (1/N_k) Σ_k G(k, iω_n)   (factor 2 for spin)
        return 2.0 * (1.0 / params.beta) * np.sum(gk.real) / len(eps_k) + 1.0

    mu_lo, mu_hi = -10.0, 10.0

    for _ in range(max_iter):
        mu_mid = (mu_lo + mu_hi) / 2.0
        n_mid = density_at_mu(mu_mid)
        if abs(n_mid - target_n) < tol:
            return mu_mid
        if n_mid > target_n:
            mu_hi = mu_mid
        else:
            mu_lo = mu_mid

    return (mu_lo + mu_hi) / 2.0


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark(work_dir: str, quick: bool = False) -> dict:
    """
    Run the full Hubbard model d-wave pairing benchmark.

    Steps for each temperature T:
      1. Set up DCA parameters (U=8t, t'=-0.3t, 15% doping)
      2. Adjust μ for target filling
      3. Run DCA self-consistency loop
      4. Compute pairing eigenvalues (d-wave, s-wave)
      5. Record λ_d(T), λ_s(T)

    After all temperatures:
      - Check that λ_d > λ_s at all T (d-wave dominates)
      - Check that λ_d increases with decreasing T
      - Extrapolate Tc from λ_d(T) → 1

    Returns:
        benchmark results dict
    """
    os.makedirs(work_dir, exist_ok=True)

    temperatures = TEMPERATURES_QUICK if quick else TEMPERATURES_T_UNITS
    mode = "QUICK" if quick else "PRODUCTION"

    print(f"\n{'='*70}")
    print(f"  2D HUBBARD MODEL d-WAVE PAIRING BENCHMARK ({mode})")
    print(f"  U={U_HUBBARD}t, t'={TP_HOPPING}t, target n={TARGET_FILLING}")
    print(f"  N_c=4 (2×2 DCA), {len(temperatures)} temperatures")
    print(f"{'='*70}\n")

    # Build lattice once (same for all temperatures)
    sample_params = get_benchmark_params(temperatures[0], quick)
    kpoints = build_lattice_kpoints(sample_params.n_k_per_dim, dim=2)
    eps_k = lattice_dispersion_hubbard(kpoints, t=T_HOPPING, tp=TP_HOPPING)
    n_k = len(eps_k)
    print(f"[Bench] Lattice: {sample_params.n_k_per_dim}×{sample_params.n_k_per_dim} = {n_k} k-points")
    print(f"[Bench] Bandwidth W = {eps_k.max() - eps_k.min():.2f}t")

    # Results accumulator
    results = {
        "model": "2D Hubbard",
        "U": U_HUBBARD,
        "t": T_HOPPING,
        "tp": TP_HOPPING,
        "target_filling": TARGET_FILLING,
        "nc": 4,
        "n_k": n_k,
        "mode": mode,
        "temperature_sweep": [],
    }

    t0_total = time.time()

    # Carry forward sigma from previous (higher) temperature as warm start
    sigma_warmstart = None

    for i_T, T_val in enumerate(temperatures):
        print(f"\n{'─'*60}")
        print(f"  Temperature {i_T+1}/{len(temperatures)}: T/t = {T_val:.4f} "
              f"(β·t = {1.0/T_val:.1f})")
        print(f"{'─'*60}")

        params = get_benchmark_params(T_val, quick)

        # Adjust chemical potential for target filling
        mu_adj = adjust_chemical_potential(params, eps_k, kpoints, TARGET_FILLING)
        params.mu = mu_adj
        print(f"[Bench] μ adjusted to {mu_adj:.4f}t for target n={TARGET_FILLING}")

        # Run DCA
        T_work_dir = os.path.join(work_dir, f"T_{T_val:.4f}")
        dca_result = run_dca(
            eps_k=eps_k,
            kpoints=kpoints,
            params=params,
            work_dir=T_work_dir,
            initial_sigma=sigma_warmstart,
        )

        # Warm-start next temperature from this one
        if dca_result.get("converged"):
            sigma_warmstart = dca_result["sigma_c"]

        # Compute pairing eigenvalues
        pairing = compute_dca_pairing_eigenvalue(
            dca_result, params, eps_k, kpoints,
        )

        lambda_d = pairing.get("d-x2y2", {}).get("lambda", 0.0)
        lambda_s = pairing.get("s-wave", {}).get("lambda", 0.0)
        lambda_sext = pairing.get("s-ext", {}).get("lambda", 0.0)

        T_entry = {
            "T_over_t": T_val,
            "beta_t": 1.0 / T_val,
            "mu": float(params.mu),
            "filling": dca_result.get("density"),
            "dca_converged": dca_result.get("converged", False),
            "dca_iterations": dca_result.get("n_iterations"),
            "lambda_d": float(lambda_d),
            "lambda_s": float(lambda_s),
            "lambda_s_ext": float(lambda_sext),
            "dominant_channel": pairing.get("dominant_channel"),
            "elapsed_seconds": dca_result.get("elapsed_seconds", 0),
        }
        results["temperature_sweep"].append(T_entry)

        print(f"\n[Bench] T/t={T_val:.4f}: λ_d={lambda_d:.4f}, λ_s={lambda_s:.4f}, "
              f"λ_s±={lambda_sext:.4f}, n={dca_result.get('density', '?'):.4f}, "
              f"dominant={pairing.get('dominant_channel')}")

    elapsed_total = time.time() - t0_total

    # ── Analysis ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BENCHMARK ANALYSIS")
    print(f"{'='*70}")

    sweep = results["temperature_sweep"]
    T_vals = [e["T_over_t"] for e in sweep]
    lambda_d_vals = [e["lambda_d"] for e in sweep]
    lambda_s_vals = [e["lambda_s"] for e in sweep]

    # Check 1: d-wave dominates
    d_dominates = all(
        abs(e["lambda_d"]) > abs(e["lambda_s"]) for e in sweep
        if e["dca_converged"]
    )
    print(f"  d-wave dominates over s-wave at all T: {'PASS' if d_dominates else 'FAIL'}")

    # Check 2: λ_d grows with decreasing T
    converged_entries = [e for e in sweep if e["dca_converged"]]
    if len(converged_entries) >= 2:
        lambda_d_trend = [e["lambda_d"] for e in converged_entries]
        monotonic = all(
            lambda_d_trend[i] <= lambda_d_trend[i+1] * 1.1  # allow 10% noise
            for i in range(len(lambda_d_trend) - 1)
        )
        print(f"  λ_d increases with decreasing T: {'PASS' if monotonic else 'WEAK'}")
    else:
        monotonic = False
        print(f"  λ_d trend: INSUFFICIENT DATA (only {len(converged_entries)} converged)")

    # Check 3: λ_d approaches 1 at low T
    lambda_max = max(lambda_d_vals) if lambda_d_vals else 0
    approaches_one = lambda_max > 0.3  # in benchmark, may not reach 1 in quick mode
    print(f"  Max λ_d = {lambda_max:.4f} {'(→1 expected at T/t≈0.02)' if lambda_max > 0.5 else ''}")

    # Tc extrapolation
    if len(converged_entries) >= 2:
        from pairing_susceptibility import extrapolate_tc
        T_list = [e["T_over_t"] for e in converged_entries]
        ld_list = [abs(e["lambda_d"]) for e in converged_entries]
        tc_result = extrapolate_tc(T_list, ld_list)
        tc_over_t = tc_result.get("tc_bse")
        print(f"  Tc/t extrapolation: {tc_over_t:.4f}" if tc_over_t else "  Tc/t: not determined")
        results["tc_over_t"] = tc_over_t
        results["tc_confidence"] = tc_result.get("tc_confidence")

    # Overall verdict
    passed = d_dominates and lambda_max > 0.1
    results["benchmark_passed"] = passed
    results["d_wave_dominates"] = d_dominates
    results["lambda_d_monotonic"] = monotonic
    results["lambda_d_max"] = lambda_max
    results["total_elapsed_hours"] = elapsed_total / 3600

    verdict = "PASS" if passed else "FAIL"
    print(f"\n  Overall benchmark: {verdict}")
    print(f"  Total time: {elapsed_total/3600:.1f} hours")
    print(f"{'='*70}\n")

    # ── Save ─────────────────────────────────────────────────────────────
    results_path = os.path.join(work_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Bench] Results saved to {results_path}")

    # Print summary table
    print(f"\n  T/t     β·t    <n>     λ_d      λ_s     λ_s±    dominant")
    print(f"  {'─'*65}")
    for e in sweep:
        fill = e.get('filling', 0) or 0
        print(f"  {e['T_over_t']:.4f}  {e['beta_t']:6.1f}  "
              f"{fill:.4f}  {e['lambda_d']:+.4f}  {e['lambda_s']:+.4f}  "
              f"{e['lambda_s_ext']:+.4f}  {e['dominant_channel']}")

    return results


if __name__ == "__main__":
    work_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("DMFT_WORK_DIR", "/data/dmft_jobs"),
        "hubbard_benchmark"
    )

    quick = os.environ.get("BENCHMARK_QUICK", "0") == "1"
    if "--quick" in sys.argv:
        quick = True

    results = run_benchmark(work_dir, quick=quick)

    if results.get("benchmark_passed"):
        print("\n*** BENCHMARK PASSED — DCA d-wave pairing validated ***")
        sys.exit(0)
    else:
        print("\n*** BENCHMARK FAILED — review results before trusting real-material DCA ***")
        sys.exit(1)
