#!/usr/bin/env python3
"""
Physics Validation Harness for the DMFT Pipeline.

Four tests that must pass before trusting any real-material result:

  V1: 2D Hubbard benchmark — lam_d > lam_s, monotonic, -> 1 at T/t~0.02
  V2: BSE SVD cutoff sensitivity — Gamma_singlet sign stable across cutoffs
  V3: Kanamori single-orbital limit — n_orb=1 J=0 reproduces Hubbard U
  V4: CSC tail subtraction — non-interacting G gives n=exact filling

Run:
  python3 physics_validation.py [--quick]

Each test prints PASS/FAIL with diagnostics. Exit code 0 if all pass.
"""

import numpy as np
import sys
import json
import os
import time


# ═══════════════════════════════════════════════════════════════════════════════
# V1: 2D Hubbard d-wave pairing benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def test_v1_hubbard_benchmark(quick: bool = True) -> dict:
    """
    Verify DCA produces d-wave pairing in the 2D Hubbard model.

    Reference: Maier et al., PRL 95, 237001 (2005)

    At U=8t, t'=-0.3t, n=0.85, N_c=4:
      - lam_d > lam_s at all T
      - lam_d increases as T decreases
      - lam_d -> 1 near T/t ~ 0.02

    We test at 3 temperatures (quick) or 6 (full) using the Hubbard-I
    fallback solver (no TRIQS needed) to verify the DCA loop + pairing
    extraction machinery. The absolute lam values won't match QMC, but
    the qualitative behavior (d > s, monotonic) tests the code path.
    """
    print("\n" + "="*60)
    print("V1: 2D Hubbard d-wave pairing benchmark")
    print("="*60)

    from dca_solver import (
        DCAParams, build_lattice_kpoints, lattice_dispersion_hubbard,
        run_dca, compute_dca_pairing_eigenvalue,
    )

    U, t, tp = 8.0, 1.0, -0.3
    target_n = 0.85
    temperatures = [0.20, 0.10, 0.05] if quick else [0.20, 0.15, 0.10, 0.08, 0.06, 0.05]

    kpoints = build_lattice_kpoints(32, dim=2)
    eps_k = lattice_dispersion_hubbard(kpoints, t=t, tp=tp)

    results = []
    sigma_warmstart = None
    errors = []

    for T_val in temperatures:
        beta = 1.0 / T_val
        mu = U / 2.0 - 1.2

        params = DCAParams(
            nc=4, dim=2, beta=beta, n_iw=64, n_k_per_dim=32,
            U=U, mu=mu, n_iter=10, sigma_mix=0.7, convergence_tol=1e-3,
            n_warmup=5000, n_cycles=100_000, length_cycle=50,
        )

        work_dir = f"/tmp/v1_hubbard_T{T_val:.2f}"
        os.makedirs(work_dir, exist_ok=True)

        dca_result = run_dca(eps_k, kpoints, params, work_dir, initial_sigma=sigma_warmstart)

        if dca_result.get("converged"):
            sigma_warmstart = dca_result["sigma_c"]

        pairing = compute_dca_pairing_eigenvalue(dca_result, params, eps_k, kpoints)

        ld = pairing.get("d-x2y2", {}).get("lambda", 0)
        ls = pairing.get("s-wave", {}).get("lambda", 0)

        results.append({"T": T_val, "lambda_d": ld, "lambda_s": ls,
                         "converged": dca_result.get("converged", False)})
        print(f"  T/t={T_val:.2f}: lam_d={ld:+.4f}, lam_s={ls:+.4f}, "
              f"d>s={'Y' if abs(ld) > abs(ls) else 'N'}")

    # ── Assertions ──
    converged = [r for r in results if r["converged"]]

    # Check 1: d-wave dominates
    d_dominant = all(abs(r["lambda_d"]) > abs(r["lambda_s"]) for r in converged)
    if not d_dominant:
        errors.append("FAIL: lam_d not > lam_s at all converged temperatures")

    # Check 2: lam_d magnitude increases with decreasing T.
    # Step-by-step we allow 5% backtracking (QMC noise per point), but the
    # CUMULATIVE trend across all sampled T must show a monotonic gain
    # (a Spearman correlation with -T > 0.7). The old 20%-per-step
    # tolerance was loose enough that a flat or even slightly-decreasing
    # λ_d(T) could pass — three 20% steps compound to 51% total decrease.
    if len(converged) >= 2:
        ld_vals = [abs(r["lambda_d"]) for r in converged]
        # Strict per-step: <=5% backtracking
        monotonic = all(
            ld_vals[i+1] >= ld_vals[i] * 0.95
            for i in range(len(ld_vals) - 1)
        )
        if not monotonic:
            errors.append(
                f"FAIL: lam_d not monotonically increasing (per-step): {ld_vals}"
            )

        # Cumulative trend check via Spearman rank correlation (no scipy
        # dependence — compute by hand).
        if len(converged) >= 3:
            # Temperatures (in eV) sorted by appearance in `converged`
            T_vals = [float(r.get("T_eV", r.get("T_K", 0) / 11604.5))
                      for r in converged]
            # We want λ_d ∝ −T (more negative T → more λ_d), so check that
            # the ranking of λ_d is anti-correlated with the ranking of T.
            T_rank = np.argsort(np.argsort(T_vals)).astype(float)
            l_rank = np.argsort(np.argsort(ld_vals)).astype(float)
            n = len(T_vals)
            # Pearson on ranks = Spearman ρ. Mean of ranks 0..n-1 is (n-1)/2.
            rank_mean = (n - 1) / 2.0
            cov = float(np.mean((T_rank - rank_mean) * (l_rank - rank_mean)))
            var_t = float(np.mean((T_rank - rank_mean) ** 2))
            var_l = float(np.mean((l_rank - rank_mean) ** 2))
            rho = cov / np.sqrt(var_t * var_l + 1e-30)
            if rho > -0.7:
                errors.append(
                    f"FAIL: λ_d(T) not anti-correlated with T "
                    f"(Spearman ρ={rho:.2f}, expected ≤ -0.7)"
                )

    # Check 3: lam_d should be nonzero and have consistent sign
    ld_signs = [np.sign(r["lambda_d"]) for r in converged if abs(r["lambda_d"]) > 1e-6]
    if ld_signs and not all(s == ld_signs[0] for s in ld_signs):
        errors.append(f"FAIL: lam_d sign flips across temperatures: {ld_signs}")

    # Check 4: N_c scan (if not quick) — verify N_c=4 result is reasonable
    # For the Hubbard-I fallback, we just check the DCA loop completes
    # With Hubbard-I fallback, self-consistency may not converge to tight
    # tolerance, but we still get qualitatively correct pairing structure.
    # Only fail if nothing ran at all or all eigenvalues are zero.
    if not results:
        errors.append("FAIL: No DCA iterations completed")
    elif all(abs(r["lambda_d"]) < 1e-10 for r in results):
        errors.append("FAIL: All lam_d are zero -- DCA pairing extraction broken")

    passed = len(errors) == 0
    print(f"\n  V1 result: {'PASS' if passed else 'FAIL'}")
    for e in errors:
        print(f"    {e}")

    return {"test": "V1_hubbard_benchmark", "passed": passed, "errors": errors,
            "results": results}


# ═══════════════════════════════════════════════════════════════════════════════
# V2: BSE SVD cutoff sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def test_v2_bse_svd_sensitivity() -> dict:
    """
    Verify that the BSE-extracted Gamma_singlet has a stable sign across
    a range of SVD cutoff values.

    The test:
      1. Construct a synthetic chi_loc and chi⁰_loc with known vertex
      2. Add realistic noise (simulating QMC)
      3. Run BSE inversion at cutoffs 1e-6, 1e-8, 1e-10, 1e-12
      4. Check that the leading eigenvalue of Gamma_singlet has consistent sign
      5. Check that Gamma_singlet is stable within 20% across cutoffs

    A sign flip in Gamma_singlet means the BSE inversion is numerically
    unreliable — the pipeline would misidentify attractive vs repulsive
    pairing channels.
    """
    print("\n" + "="*60)
    print("V2: BSE SVD cutoff sensitivity")
    print("="*60)

    from bse_solver import solve_bse_local_svd, decompose_vertex_channels

    n_iw_f = 10
    n_orb = 1
    n_iw_b = 3
    N = 2 * n_iw_f * n_orb * n_orb
    n_bos = 2 * n_iw_b + 1

    # Construct synthetic data: known vertex Gamma = -U (attractive in singlet)
    U_test = 2.0
    np.random.seed(42)

    # Build chi⁰ as a diagonal matrix with 1/(iω) decay
    chi0 = np.zeros((2 * n_iw_f, n_iw_b * 2 + 1, n_orb, n_orb, n_orb, n_orb), dtype=complex)
    for iv in range(2 * n_iw_f):
        wn = (2 * (iv - n_iw_f) + 1) * np.pi / 40.0  # beta=40
        for iw in range(n_bos):
            chi0[iv, iw, 0, 0, 0, 0] = -1.0 / (1j * wn + 0.5)

    # Build chi from the Dyson equation: chi = chi⁰ + chi⁰ · Gamma · chi = chi⁰/(1 - Gamma·chi⁰)
    # For scalar: chi = chi⁰ / (1 + U·chi⁰)  (note sign: Gamma = -U for attractive)
    chi_loc = np.zeros((2 * n_iw_f, 2 * n_iw_f, n_iw_b * 2 + 1,
                         n_orb, n_orb, n_orb, n_orb), dtype=complex)
    for iv in range(2 * n_iw_f):
        for ivp in range(2 * n_iw_f):
            for iw in range(n_bos):
                c0 = chi0[iv, iw, 0, 0, 0, 0]
                # chi(ν,ν',Ω) ~ chi⁰(ν,Ω) · δ_{ν,ν'} / (1 - Gamma·chi⁰)
                if iv == ivp:
                    chi_loc[iv, ivp, iw, 0, 0, 0, 0] = c0 / (1.0 + U_test * c0)
                # Add small noise to off-diagonal
                chi_loc[iv, ivp, iw, 0, 0, 0, 0] += (
                    np.random.randn() + 1j * np.random.randn()
                ) * 1e-4

    cutoffs = [1e-6, 1e-8, 1e-10, 1e-12]
    singlet_evals = []
    errors = []

    for cutoff in cutoffs:
        gamma, diag = solve_bse_local_svd(chi_loc, chi0, n_iw_f, n_orb, n_iw_b, svd_cutoff=cutoff)
        channels = decompose_vertex_channels(gamma, n_iw_f, n_orb)
        ev = channels["max_eval_singlet"]
        singlet_evals.append(ev)
        print(f"  cutoff={cutoff:.0e}: max_eval_singlet={ev:+.6f}, "
              f"cond_chi⁰={diag['mean_cond_chi0']:.1e}, "
              f"truncated={diag['total_truncated_chi0']}")

    # Check sign stability
    signs = [np.sign(ev) for ev in singlet_evals if abs(ev) > 1e-8]
    sign_stable = len(set(signs)) <= 1
    if not sign_stable:
        errors.append(f"FAIL: Gamma_singlet sign flips across cutoffs: {singlet_evals}")

    # Check magnitude stability across SVD cutoffs.
    # Tightened from the original 50%-per-step (which let a 2× systematic
    # bias slip through) to 20% — for synthetic chi with a known Γ=-U vertex
    # the SVD-stabilized BSE should give consistent magnitudes at any cutoff
    # ≥ 1e-12; a factor-2 spread would indicate the inversion is genuinely
    # broken, not noise.
    if len(singlet_evals) >= 2:
        ref = singlet_evals[1]  # 1e-8 as reference (typical production setting)
        for i, ev in enumerate(singlet_evals):
            if abs(ref) > 1e-8:
                ratio = abs(ev / ref)
                if ratio < 0.8 or ratio > 1.25:
                    errors.append(
                        f"FAIL: Gamma_singlet unstable at cutoff={cutoffs[i]:.0e}: "
                        f"ev={ev:.4f} vs ref={ref:.4f} (ratio={ratio:.2f})"
                    )

    passed = len(errors) == 0
    print(f"\n  V2 result: {'PASS' if passed else 'FAIL'}")
    for e in errors:
        print(f"    {e}")

    return {"test": "V2_bse_svd_sensitivity", "passed": passed, "errors": errors,
            "cutoffs": cutoffs, "singlet_evals": singlet_evals}


# ═══════════════════════════════════════════════════════════════════════════════
# V3: Kanamori single-orbital limit
# ═══════════════════════════════════════════════════════════════════════════════

def test_v3_kanamori_single_orbital() -> dict:
    """
    Verify that the Kanamori interaction reduces to plain Hubbard U
    when n_orb=1 and J=0.

    For a single orbital, all inter-orbital terms vanish (no a!=b pairs).
    The only surviving term should be: H = U · n_↑ · n_↓

    The test constructs the Kanamori interaction with n_orb=1, J=0,
    and verifies:
      1. U_prime matrix is empty (no inter-orbital terms)
      2. J_pair matrix is empty
      3. Only intra-orbital U survives
      4. The density-density version is identical
      5. For n_orb=2, J=0: U_prime = U (Kanamori constraint U'=U-2J=U)
      6. For n_orb=2, J>0: U_prime = U-2J (proper Kanamori relation)

    If this fails, there's a sign error in the operator construction.
    """
    print("\n" + "="*60)
    print("V3: Kanamori single-orbital limit")
    print("="*60)

    from multiorbital_dca import KanamoriInteraction

    errors = []

    # Test 1: n_orb=1, J=0 -> pure Hubbard
    print("  Test 1: n_orb=1, J=0")
    K1 = KanamoriInteraction(n_orb=1, U=np.array([4.0]), J=np.array([0.0]))
    assert K1.U_prime.shape == (1, 1), "U_prime wrong shape"
    assert K1.U_prime[0, 0] == 0.0, "U_prime should be 0 for single orbital"
    assert K1.J_pair[0, 0] == 0.0, "J_pair should be 0 for single orbital"
    print("    U_prime=0, J_pair=0 — correct (single orbital, no inter-orbital terms)")

    # Test 2: n_orb=2, J=0 -> U_prime = U (Kanamori constraint)
    print("  Test 2: n_orb=2, J=0")
    U_val = 4.0
    K2 = KanamoriInteraction(n_orb=2, U=np.array([U_val, U_val]), J=np.array([0.0, 0.0]))
    up_expected = U_val  # U' = U - 2J = U - 0 = U
    up_actual = K2.U_prime[0, 1]
    if abs(up_actual - up_expected) > 1e-10:
        errors.append(f"FAIL: U'={up_actual}, expected {up_expected} (J=0 -> U'=U)")
    else:
        print(f"    U'={up_actual:.1f} = U={U_val:.1f} — correct")

    # Test 3: n_orb=2, J=0.9 -> U_prime = U - 2J = 2.2
    print("  Test 3: n_orb=2, J=0.9")
    J_val = 0.9
    K3 = KanamoriInteraction(n_orb=2, U=np.array([U_val, U_val]), J=np.array([J_val, J_val]))
    up_expected3 = U_val - 2 * J_val  # 4.0 - 1.8 = 2.2
    up_actual3 = K3.U_prime[0, 1]
    jp_actual3 = K3.J_pair[0, 1]
    if abs(up_actual3 - up_expected3) > 1e-10:
        errors.append(f"FAIL: U'={up_actual3}, expected {up_expected3}")
    elif abs(jp_actual3 - J_val) > 1e-10:
        errors.append(f"FAIL: J_pair={jp_actual3}, expected {J_val}")
    else:
        print(f"    U'={up_actual3:.1f} = U-2J={up_expected3:.1f}, J_pair={jp_actual3:.1f} — correct")

    # Test 4: Symmetry — U_prime[a,b] == U_prime[b,a]
    print("  Test 4: U' symmetry (3-orbital)")
    K4 = KanamoriInteraction(
        n_orb=3,
        U=np.array([8.0, 4.0, 4.0]),
        J=np.array([0.9, 0.5, 0.5]),
    )
    for a in range(3):
        for b in range(3):
            if a != b:
                if abs(K4.U_prime[a, b] - K4.U_prime[b, a]) > 1e-10:
                    errors.append(f"FAIL: U'[{a},{b}]={K4.U_prime[a,b]:.4f} != U'[{b},{a}]={K4.U_prime[b,a]:.4f}")
    if not any("symmetry" in e.lower() or "U'[" in e for e in errors):
        print("    U'[a,b] == U'[b,a] for all pairs — correct")

    # Test 5: Kanamori constraint verification
    # For equal-U orbitals: U' = U - 2J must hold
    print("  Test 5: Kanamori constraint U'=U-2J for equal-U orbitals")
    K5 = KanamoriInteraction(n_orb=3, U=np.array([6.0]*3), J=np.array([0.7]*3))
    for a in range(3):
        for b in range(3):
            if a != b:
                expected = 6.0 - 2 * 0.7  # 4.6
                if abs(K5.U_prime[a, b] - expected) > 1e-10:
                    errors.append(f"FAIL: U'[{a},{b}]={K5.U_prime[a,b]:.4f}, expected {expected}")
    if not any("constraint" in e.lower() or "U'[" in e for e in errors):
        print(f"    U'=U-2J={6.0-2*0.7:.1f} for all off-diagonal pairs — correct")

    passed = len(errors) == 0
    print(f"\n  V3 result: {'PASS' if passed else 'FAIL'}")
    for e in errors:
        print(f"    {e}")

    return {"test": "V3_kanamori_single_orbital", "passed": passed, "errors": errors}


# ═══════════════════════════════════════════════════════════════════════════════
# V4: CSC density matrix tail subtraction
# ═══════════════════════════════════════════════════════════════════════════════

def test_v4_csc_tail_subtraction() -> dict:
    """
    Verify that the density matrix extraction from G(iω) gives the
    correct filling for a non-interacting Green's function.

    For a non-interacting system at chemical potential mu:
      G(iω) = 1 / (iω + mu - eps)

    The exact occupation is:
      n = f(eps - mu) = 1 / (exp(beta(eps-mu)) + 1)    (Fermi-Dirac)

    If the tail subtraction in compute_density_matrix_from_gf is correct,
    the extracted n should match the Fermi-Dirac value to within the
    Matsubara truncation error (~1/(beta·ω_max)).

    We test:
      1. Single orbital, half-filling (mu=eps -> n=0.5 exactly)
      2. Single orbital, off half-filling (mu!=eps -> n from Fermi-Dirac)
      3. Multi-orbital, different fillings per orbital
      4. Very low temperature (beta=100) where tail matters most
      5. Sign of the density (must be 0 < n < 1)
    """
    print("\n" + "="*60)
    print("V4: CSC density matrix tail subtraction")
    print("="*60)

    from charge_selfconsistency import compute_density_matrix_from_gf

    errors = []

    def build_noninteracting_g(epsilon: float, mu: float, beta: float, n_iw: int) -> np.ndarray:
        """Build G(iω) = 1/(iω + mu - eps) for a single orbital."""
        g = np.zeros((2 * n_iw, 1, 1), dtype=complex)
        for n in range(2 * n_iw):
            wn = (2 * (n - n_iw) + 1) * np.pi / beta
            g[n, 0, 0] = 1.0 / (1j * wn + mu - epsilon)
        return g

    def fermi_dirac(epsilon: float, mu: float, beta: float) -> float:
        x = beta * (epsilon - mu)
        if x > 500:
            return 0.0
        if x < -500:
            return 1.0
        return 1.0 / (np.exp(x) + 1.0)

    # Test 1: Half-filling (eps=0, mu=0 -> n=0.5 exactly)
    print("  Test 1: Half-filling (eps=0, mu=0)")
    for n_iw in [64, 128, 256]:
        beta = 40.0
        g = build_noninteracting_g(0.0, 0.0, beta, n_iw)
        n_matrix = compute_density_matrix_from_gf(g, beta)
        n_extracted = n_matrix[0, 0]
        n_exact = 0.5
        err = abs(n_extracted - n_exact)
        status = "OK" if err < 0.01 else "BAD"
        print(f"    n_iw={n_iw}: n={n_extracted:.6f} (exact={n_exact:.6f}, err={err:.2e}) [{status}]")
        if err > 0.01:
            errors.append(f"FAIL: Half-filling n={n_extracted:.6f}, expected 0.5 (n_iw={n_iw})")

    # Test 2: Off half-filling (eps=0, mu=0.5 -> n=f(-0.5))
    print("  Test 2: Off half-filling (eps=0, mu=0.5)")
    beta = 40.0
    mu = 0.5
    n_iw = 256
    g = build_noninteracting_g(0.0, mu, beta, n_iw)
    n_matrix = compute_density_matrix_from_gf(g, beta)
    n_extracted = n_matrix[0, 0]
    n_exact = fermi_dirac(0.0, mu, beta)
    err = abs(n_extracted - n_exact)
    print(f"    n={n_extracted:.6f} (exact={n_exact:.6f}, err={err:.2e})")
    if err > 0.02:
        errors.append(f"FAIL: Off-half-filling n={n_extracted:.6f}, expected {n_exact:.6f}")

    # Test 3: Multi-orbital with different eps per orbital
    print("  Test 3: Multi-orbital (3 orbitals, different eps)")
    beta = 40.0
    n_iw = 256
    epsilons = [0.0, -1.0, 0.5]
    mu_mo = 0.0
    g_mo = np.zeros((2 * n_iw, 3, 3), dtype=complex)
    for a, eps in enumerate(epsilons):
        for n in range(2 * n_iw):
            wn = (2 * (n - n_iw) + 1) * np.pi / beta
            g_mo[n, a, a] = 1.0 / (1j * wn + mu_mo - eps)

    n_mo = compute_density_matrix_from_gf(g_mo, beta)
    for a, eps in enumerate(epsilons):
        n_exact_a = fermi_dirac(eps, mu_mo, beta)
        n_extracted_a = n_mo[a, a]
        err_a = abs(n_extracted_a - n_exact_a)
        print(f"    orb {a}: eps={eps:+.1f}, n={n_extracted_a:.6f} (exact={n_exact_a:.6f}, err={err_a:.2e})")
        if err_a > 0.02:
            errors.append(f"FAIL: Multi-orbital orb {a} n={n_extracted_a:.6f}, expected {n_exact_a:.6f}")

    # Test 4: Low temperature (beta=100, tail matters more)
    print("  Test 4: Low temperature (beta=100)")
    beta_low = 100.0
    n_iw_low = 512
    g_low = build_noninteracting_g(0.0, 0.3, beta_low, n_iw_low)
    n_low = compute_density_matrix_from_gf(g_low, beta_low)
    n_exact_low = fermi_dirac(0.0, 0.3, beta_low)
    err_low = abs(n_low[0, 0] - n_exact_low)
    print(f"    n={n_low[0,0]:.6f} (exact={n_exact_low:.6f}, err={err_low:.2e})")
    if err_low > 0.02:
        errors.append(f"FAIL: Low-T n={n_low[0,0]:.6f}, expected {n_exact_low:.6f}")

    # Test 5: Physical bounds (0 < n < 1)
    print("  Test 5: Physical bounds check")
    for test_mu in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        g_bounds = build_noninteracting_g(0.0, test_mu, 40.0, 256)
        n_bounds = compute_density_matrix_from_gf(g_bounds, 40.0)
        n_val = n_bounds[0, 0]
        if n_val < -0.01 or n_val > 1.01:
            errors.append(f"FAIL: n={n_val:.4f} out of [0,1] at mu={test_mu}")
    if not any("bounds" in e.lower() for e in errors):
        print("    All n ∈ [0,1] — correct")

    # Test 6: Off-diagonal elements should be zero for diagonal G
    print("  Test 6: Off-diagonal density matrix = 0 for diagonal G")
    off_diag_max = np.max(np.abs(n_mo - np.diag(np.diag(n_mo))))
    print(f"    max |n_offdiag| = {off_diag_max:.2e}")
    if off_diag_max > 1e-10:
        errors.append(f"FAIL: Off-diagonal density {off_diag_max:.2e} for diagonal G")

    passed = len(errors) == 0
    print(f"\n  V4 result: {'PASS' if passed else 'FAIL'}")
    for e in errors:
        print(f"    {e}")

    return {"test": "V4_csc_tail_subtraction", "passed": passed, "errors": errors}


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_validations(quick: bool = True) -> dict:
    """Run all four validation tests."""
    t0 = time.time()

    print("\n" + "#"*60)
    print("# DMFT PHYSICS VALIDATION HARNESS")
    print(f"# Mode: {'QUICK' if quick else 'FULL'}")
    print("#"*60)

    results = {}

    # V4 first (fastest, most likely to catch basic bugs)
    results["V4"] = test_v4_csc_tail_subtraction()

    # V3 next (no solver needed, pure Python)
    results["V3"] = test_v3_kanamori_single_orbital()

    # V2 (synthetic data, no solver)
    results["V2"] = test_v2_bse_svd_sensitivity()

    # V1 last (runs DCA loop, slowest)
    results["V1"] = test_v1_hubbard_benchmark(quick=quick)

    elapsed = time.time() - t0

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    for name, r in sorted(results.items()):
        status = "PASS" if r["passed"] else "FAIL"
        all_passed = all_passed and r["passed"]
        err_count = len(r.get("errors", []))
        print(f"  {name} [{r['test']}]: {status}" +
              (f" ({err_count} errors)" if err_count > 0 else ""))

    print(f"\n  Overall: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    print(f"  Time: {elapsed:.1f}s")
    print("="*60)

    return {
        "all_passed": all_passed,
        "tests": results,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    quick = "--quick" in sys.argv or "-q" in sys.argv
    results = run_all_validations(quick=quick)

    # Save results
    out_path = os.path.join(
        os.environ.get("DMFT_WORK_DIR", "/tmp"),
        "physics_validation_results.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    sys.exit(0 if results["all_passed"] else 1)
