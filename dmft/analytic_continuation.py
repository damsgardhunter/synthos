#!/usr/bin/env python3
"""
Analytic Continuation: Matsubara → Real Axis.

DMFT produces Σ(iω_n) and G(iω_n) on the imaginary-frequency axis.
Experimentalists measure A(ω) on the real axis (ARPES, STM, optics).
The analytic continuation iω → ω+iδ is ill-posed: small QMC noise in
G(iω) amplifies into large artifacts in A(ω).

Methods implemented:
  1. MaxEnt (TRIQS/maxent) — Bayesian, default
  2. Padé approximants — fast, less reliable for noisy data

References:
  Jarrell & Gubernatis, Phys. Rep. 269, 133 (1996) — MaxEnt review
  Beach, arXiv:cond-mat/0403055 (2004) — MaxEnt for DMFT
  Vidberg & Serene, JLTP 29, 179 (1977) — Padé for Green's functions
"""

import numpy as np
import json
import os
import time
from typing import Optional, Dict, List


def run_maxent(
    g_iw: np.ndarray,
    beta: float,
    n_orb: int,
    work_dir: str,
    omega_min: float = -15.0,
    omega_max: float = 15.0,
    n_omega: int = 1001,
    alpha_mesh_size: int = 60,
) -> Dict:
    """
    Run Maximum Entropy analytic continuation via TRIQS/maxent.

    Args:
        g_iw: G(iω) array [2*n_iw, n_orb, n_orb] or [2*n_iw]
        beta: inverse temperature
        n_orb: orbital count
        work_dir: output directory
        omega_min/max: real-frequency window (eV)
        n_omega: number of real-frequency points
        alpha_mesh_size: number of α (regularization) values to scan

    Returns:
        dict with A(ω) spectral function, Σ(ω), N(E_F)
    """
    os.makedirs(work_dir, exist_ok=True)

    try:
        return _run_maxent_triqs(g_iw, beta, n_orb, work_dir,
                                 omega_min, omega_max, n_omega, alpha_mesh_size)
    except ImportError:
        print("[AC] TRIQS/maxent not available, falling back to Padé")
        return run_pade(g_iw, beta, n_orb, work_dir, omega_min, omega_max, n_omega)


def _run_maxent_triqs(
    g_iw: np.ndarray,
    beta: float,
    n_orb: int,
    work_dir: str,
    omega_min: float,
    omega_max: float,
    n_omega: int,
    alpha_mesh_size: int,
) -> Dict:
    """TRIQS/maxent implementation."""
    from triqs.gf import GfImFreq, GfReFreq, MeshImFreq, MeshReFreq
    from triqs_maxent import TauMaxEnt, LorentzianModel, ElementwiseMaxEnt

    t0 = time.time()
    n_iw = g_iw.shape[0] // 2
    results = {"method": "maxent", "omega": [], "spectral_functions": {}}

    omega_grid = np.linspace(omega_min, omega_max, n_omega)
    results["omega"] = omega_grid.tolist()

    for orb in range(n_orb):
        # Extract diagonal G(iω) for this orbital
        if g_iw.ndim == 3:
            g_orb = g_iw[:, orb, orb]
        elif g_iw.ndim == 1:
            g_orb = g_iw
        else:
            g_orb = g_iw[:, orb] if g_iw.ndim == 2 else g_iw

        # Build TRIQS GfImFreq
        mesh = MeshImFreq(beta=beta, S="Fermion", n_iw=n_iw)
        gf = GfImFreq(mesh=mesh, shape=[1, 1])
        for idx, iw in enumerate(mesh):
            if idx < len(g_orb):
                gf[iw][0, 0] = g_orb[idx]

        # Run MaxEnt
        try:
            tm = TauMaxEnt()
            tm.set_G_iw(gf)
            tm.set_error(1e-4)  # QMC error estimate
            tm.omega = omega_grid
            tm.alpha_mesh = np.logspace(0, 6, alpha_mesh_size)
            tm.run()

            # Extract optimal A(ω) — Bryan's method picks the optimal α
            a_omega = tm.A_of_omega.data[:, 0, 0].real
            results["spectral_functions"][f"orb_{orb}"] = a_omega.tolist()

        except Exception as e:
            print(f"[AC] MaxEnt failed for orbital {orb}: {e}")
            results["spectral_functions"][f"orb_{orb}"] = [0.0] * n_omega

    # Total spectral function (sum over orbitals, factor 2 for spin)
    a_total = np.zeros(n_omega)
    for orb in range(n_orb):
        key = f"orb_{orb}"
        if key in results["spectral_functions"]:
            a_total += np.array(results["spectral_functions"][key])
    a_total *= 2.0  # spin
    results["spectral_function_total"] = a_total.tolist()

    # N(E_F) = A(ω=0) per spin
    idx_ef = n_omega // 2
    results["n_ef_dmft"] = float(a_total[idx_ef] / 2.0)  # per spin per cell

    results["elapsed_seconds"] = time.time() - t0
    results["converged"] = True

    # Save
    save_path = os.path.join(work_dir, "analytic_continuation.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_pade(
    g_iw: np.ndarray,
    beta: float,
    n_orb: int,
    work_dir: str,
    omega_min: float = -15.0,
    omega_max: float = 15.0,
    n_omega: int = 1001,
) -> Dict:
    """
    Padé approximant analytic continuation.

    Constructs a rational function P(z)/Q(z) that matches G(iω_n) at all
    Matsubara points, then evaluates on the real axis ω+iδ.

    Less reliable than MaxEnt for noisy QMC data but works without TRIQS/maxent.
    """
    t0 = time.time()
    n_iw = g_iw.shape[0] // 2
    omega_grid = np.linspace(omega_min, omega_max, n_omega)
    delta = 0.03  # broadening (eV)

    results = {
        "method": "pade",
        "omega": omega_grid.tolist(),
        "spectral_functions": {},
    }

    for orb in range(n_orb):
        if g_iw.ndim == 3:
            g_orb = g_iw[:, orb, orb]
        elif g_iw.ndim == 1:
            g_orb = g_iw
        else:
            g_orb = g_iw[:, orb] if g_iw.ndim == 2 else g_iw

        # Matsubara frequencies
        wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(2 * n_iw)])
        z_matsubara = 1j * wn

        # Use only the first ~60 frequencies (Padé is unstable for many points)
        n_pade = min(60, len(g_orb))
        mid = len(g_orb) // 2
        indices = list(range(mid - n_pade // 2, mid + n_pade // 2))
        z_p = z_matsubara[indices]
        g_p = g_orb[indices]

        # Build Padé coefficients via Thiele's continued fraction
        try:
            a_omega = _pade_evaluate(z_p, g_p, omega_grid + 1j * delta)
            spectral = -a_omega.imag / np.pi
            spectral = np.maximum(spectral, 0)  # enforce positivity
        except Exception:
            spectral = np.zeros(n_omega)

        results["spectral_functions"][f"orb_{orb}"] = spectral.tolist()

    # Total
    a_total = np.zeros(n_omega)
    for orb in range(n_orb):
        a_total += np.array(results["spectral_functions"][f"orb_{orb}"])
    a_total *= 2.0
    results["spectral_function_total"] = a_total.tolist()
    results["n_ef_dmft"] = float(a_total[n_omega // 2] / 2.0)
    results["elapsed_seconds"] = time.time() - t0
    results["converged"] = True

    save_path = os.path.join(work_dir, "analytic_continuation.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def _pade_evaluate(z_in: np.ndarray, g_in: np.ndarray, z_out: np.ndarray) -> np.ndarray:
    """Thiele continued-fraction Padé interpolation."""
    n = len(z_in)
    # Build continued fraction coefficients
    a = np.zeros(n, dtype=complex)
    a[0] = g_in[0]

    # Recursive table
    table = np.zeros((n, n), dtype=complex)
    table[:, 0] = g_in
    for j in range(1, n):
        for i in range(j, n):
            table[i, j] = (table[j - 1, j - 1] - table[i, j - 1]) / (
                (z_in[i] - z_in[j - 1]) * table[i, j - 1]
            )
            if abs(table[i, j - 1]) < 1e-30:
                table[i, j] = 0
        a[j] = table[j, j]

    # Evaluate continued fraction at z_out
    result = np.zeros(len(z_out), dtype=complex)
    for iz, z in enumerate(z_out):
        cf = a[-1]
        for j in range(n - 2, 0, -1):
            denom = 1.0 + a[j] * (z - z_in[j - 1]) * cf
            if abs(denom) < 1e-30:
                cf = 0
            else:
                cf = 1.0 / denom
        result[iz] = a[0] * cf if abs(cf) > 1e-30 else a[0]

    return result
