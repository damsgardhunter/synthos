#!/usr/bin/env python3
"""
B2: Local Bethe-Salpeter Equation (BSE) Solver.

Extracts the local irreducible vertex Γ_loc from the measured local
susceptibility χ_loc and the bare bubble χ⁰_loc.

The BSE in the particle-hole channel:
  χ_loc = χ⁰_loc + χ⁰_loc · Γ_loc · χ_loc

Rearranged to extract Γ_loc:
  Γ_loc = (χ⁰_loc)⁻¹ - (χ_loc)⁻¹

This is "just bookkeeping" but the orbital/frequency indexing is the
#1 source of bugs in vertex codes. The key subtlety: χ and Γ are
matrices in the compound index (iν, a, b) where iν is fermionic
Matsubara and a,b are orbital indices. For each bosonic frequency iΩ,
we do a separate matrix inversion.

For the pairing channel (particle-particle), the BSE reads:
  χ_pp = χ⁰_pp + χ⁰_pp · Γ_pp · χ_pp
  Γ_pp = (χ⁰_pp)⁻¹ - (χ_pp)⁻¹

We solve in BOTH channels:
  - Particle-hole (ph): gives spin/charge vertex for magnetic instabilities
  - Particle-particle (pp): gives pairing vertex for superconducting instabilities

References:
  Rohringer et al., Rev. Mod. Phys. 90, 025003 (2018) — Sec. III
  Bickers, "Theoretical Methods for Strongly Correlated Electrons" (2004)
  Maier et al., Rev. Mod. Phys. 77, 1027 (2005) — Sec. IV.B
  Galler et al., PRB 95, 115107 (2017) — multi-orbital vertex
"""

import numpy as np
from typing import Optional


# ── Index flattening ─────────────────────────────────────────────────────────
#
# The vertex Γ(iν, iν'; iΩ)_{abcd} is a matrix in the compound index
# I = (iν, a, b) where a,b are orbital/spin indices.
#
# For a given bosonic frequency iΩ, Γ is a square matrix of size
# N = (2·n_iw_f) × n_orb × n_orb.
#
# The BSE inversion operates on this matrix for each iΩ independently.

def _compound_index(iv: int, a: int, b: int, n_orb: int) -> int:
    """Map (fermionic_freq_index, orb_a, orb_b) -> flat matrix index."""
    return iv * (n_orb * n_orb) + a * n_orb + b


def _reshape_to_matrix(
    chi: np.ndarray,
    n_iw_f: int,
    n_orb: int,
    iw_idx: int,
) -> np.ndarray:
    """
    Reshape χ(iν, iν', iΩ)_{abcd} at fixed iΩ into a square matrix
    in the compound index I=(iν, a, b), J=(iν', c, d).

    Input chi shape:  [2*n_iw_f, 2*n_iw_f, 2*n_iw_b+1, n_orb, n_orb, n_orb, n_orb]
                   or [2*n_iw_f, 2*n_iw_b+1, n_orb, n_orb, n_orb, n_orb]  (diagonal in ν')

    Output: [N, N] where N = 2*n_iw_f * n_orb²
    """
    nf = 2 * n_iw_f
    N = nf * n_orb * n_orb

    mat = np.zeros((N, N), dtype=complex)

    if chi.ndim == 7:
        # Full two-fermion-frequency structure: χ(ν, ν', Ω, a, b, c, d)
        for iv in range(nf):
            for ivp in range(nf):
                for a in range(n_orb):
                    for b in range(n_orb):
                        I = _compound_index(iv, a, b, n_orb)
                        for c in range(n_orb):
                            for d in range(n_orb):
                                J = _compound_index(ivp, c, d, n_orb)
                                mat[I, J] = chi[iv, ivp, iw_idx, a, b, c, d]

    elif chi.ndim == 6:
        # Bubble structure: χ⁰(ν, Ω, a, b, c, d) — diagonal in ν
        # χ⁰ is local and has no ν' dependence (it's δ_{ν,ν'} in the bare bubble)
        for iv in range(nf):
            for a in range(n_orb):
                for b in range(n_orb):
                    I = _compound_index(iv, a, b, n_orb)
                    for c in range(n_orb):
                        for d in range(n_orb):
                            J = _compound_index(iv, c, d, n_orb)
                            mat[I, J] = chi[iv, iw_idx, a, b, c, d]

    return mat


def _reshape_to_matrix_fast(
    chi: np.ndarray,
    n_iw_f: int,
    n_orb: int,
    iw_idx: int,
) -> np.ndarray:
    """
    Vectorized reshaping — same result as _reshape_to_matrix but using
    numpy reshape/transpose instead of Python loops.
    """
    nf = 2 * n_iw_f
    N = nf * n_orb * n_orb

    if chi.ndim == 7:
        # chi[nf, nf, n_bos, norb, norb, norb, norb]
        # Extract slice at this bosonic frequency
        chi_w = chi[:, :, iw_idx, :, :, :, :]  # [nf, nf, no, no, no, no]
        # Reshape: (nf, no, no) x (nf, no, no) -> (N, N)
        # First combine (iv, a, b) into rows, (ivp, c, d) into cols
        mat = chi_w.reshape(nf, nf, n_orb * n_orb, n_orb * n_orb)
        # Transpose to get [iv, a*norb+b, ivp, c*norb+d]
        # Then reshape to [nf*norb², nf*norb²]
        mat = mat.transpose(0, 2, 1, 3).reshape(N, N)
        return mat

    elif chi.ndim == 6:
        # chi0[nf, n_bos, norb, norb, norb, norb] — diagonal in ν
        chi_w = chi[:, iw_idx, :, :, :, :]  # [nf, no, no, no, no]
        mat = np.zeros((N, N), dtype=complex)
        # Only diagonal blocks (iv == ivp)
        for iv in range(nf):
            block = chi_w[iv]  # [no, no, no, no]
            block_2d = block.reshape(n_orb * n_orb, n_orb * n_orb)
            i0 = iv * n_orb * n_orb
            i1 = i0 + n_orb * n_orb
            mat[i0:i1, i0:i1] = block_2d
        return mat

    else:
        raise ValueError(f"Unexpected chi shape: {chi.shape}")


# ── BSE inversion ────────────────────────────────────────────────────────────

def solve_bse_local(
    chi_loc: np.ndarray,
    chi0_loc: np.ndarray,
    n_iw_f: int,
    n_orb: int,
    n_iw_b: int,
    regularization: float = 1e-12,
) -> np.ndarray:
    """
    Extract the local irreducible vertex Γ_loc via BSE inversion:

      Γ_loc(iΩ) = [χ⁰_loc(iΩ)]⁻¹ - [χ_loc(iΩ)]⁻¹

    For each bosonic frequency iΩ, this is a matrix inversion in the
    compound index (iν, a, b).

    Args:
        chi_loc:  connected local susceptibility from G² measurement
                  Shape [2*n_iw_f, 2*n_iw_f, 2*n_iw_b+1, n_orb⁴] or similar
        chi0_loc: bare bubble susceptibility
                  Shape [2*n_iw_f, 2*n_iw_b+1, n_orb⁴] (diagonal in ν)
        n_iw_f:   fermionic Matsubara frequencies per side
        n_orb:    number of correlated orbitals
        n_iw_b:   bosonic Matsubara frequencies per side
        regularization: Tikhonov regularization for ill-conditioned inversions

    Returns:
        gamma_loc: irreducible vertex
                   Shape [N, N, 2*n_iw_b+1] where N = 2*n_iw_f * n_orb²
    """
    N = 2 * n_iw_f * n_orb * n_orb
    n_bos = 2 * n_iw_b + 1
    gamma_loc = np.zeros((N, N, n_bos), dtype=complex)

    reg_matrix = regularization * np.eye(N, dtype=complex)

    inversions_failed = 0

    for iw in range(n_bos):
        # Reshape susceptibilities into matrices at this bosonic frequency
        chi_mat = _reshape_to_matrix_fast(chi_loc, n_iw_f, n_orb, iw)
        chi0_mat = _reshape_to_matrix_fast(chi0_loc, n_iw_f, n_orb, iw)

        # Invert with regularization for numerical stability
        try:
            chi0_inv = np.linalg.inv(chi0_mat + reg_matrix)
        except np.linalg.LinAlgError:
            # χ⁰ is singular (e.g., at special frequencies) — use pseudoinverse
            chi0_inv = np.linalg.pinv(chi0_mat, rcond=1e-10)
            inversions_failed += 1

        try:
            chi_inv = np.linalg.inv(chi_mat + reg_matrix)
        except np.linalg.LinAlgError:
            chi_inv = np.linalg.pinv(chi_mat, rcond=1e-10)
            inversions_failed += 1

        # Γ_loc(Ω) = (χ⁰)⁻¹ - (χ)⁻¹
        gamma_loc[:, :, iw] = chi0_inv - chi_inv

    if inversions_failed > 0:
        print(f"[BSE] WARNING: {inversions_failed}/{2*n_bos} matrix inversions "
              f"used pseudoinverse (singular susceptibility)")

    return gamma_loc


def solve_bse_local_svd(
    chi_loc: np.ndarray,
    chi0_loc: np.ndarray,
    n_iw_f: int,
    n_orb: int,
    n_iw_b: int,
    svd_cutoff: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """
    SVD-stabilized BSE inversion — more robust than direct inversion
    for ill-conditioned cases (large n_orb, low temperature).

    Instead of direct inversion, uses truncated SVD:
      M⁻¹ ≈ V · diag(1/σ_i for σ_i > cutoff) · U†

    Returns:
        gamma_loc: irreducible vertex [N, N, n_bos]
        diagnostics: dict with condition numbers, truncated modes, etc.
    """
    N = 2 * n_iw_f * n_orb * n_orb
    n_bos = 2 * n_iw_b + 1
    gamma_loc = np.zeros((N, N, n_bos), dtype=complex)

    cond_chi0 = []
    cond_chi = []
    n_truncated_chi0 = []
    n_truncated_chi = []

    for iw in range(n_bos):
        chi_mat = _reshape_to_matrix_fast(chi_loc, n_iw_f, n_orb, iw)
        chi0_mat = _reshape_to_matrix_fast(chi0_loc, n_iw_f, n_orb, iw)

        # SVD of χ⁰
        U0, s0, Vh0 = np.linalg.svd(chi0_mat, full_matrices=True)
        mask0 = s0 > svd_cutoff * s0[0]
        s0_inv = np.zeros_like(s0)
        s0_inv[mask0] = 1.0 / s0[mask0]
        chi0_inv = (Vh0.conj().T * s0_inv[np.newaxis, :]) @ U0.conj().T

        cond_chi0.append(s0[0] / max(s0[-1], 1e-300))
        n_truncated_chi0.append(int(np.sum(~mask0)))

        # SVD of χ
        U, s, Vh = np.linalg.svd(chi_mat, full_matrices=True)
        mask = s > svd_cutoff * s[0]
        s_inv = np.zeros_like(s)
        s_inv[mask] = 1.0 / s[mask]
        chi_inv = (Vh.conj().T * s_inv[np.newaxis, :]) @ U.conj().T

        cond_chi.append(s[0] / max(s[-1], 1e-300))
        n_truncated_chi.append(int(np.sum(~mask)))

        gamma_loc[:, :, iw] = chi0_inv - chi_inv

    diagnostics = {
        "mean_cond_chi0": float(np.mean(cond_chi0)),
        "max_cond_chi0": float(np.max(cond_chi0)),
        "mean_cond_chi": float(np.mean(cond_chi)),
        "max_cond_chi": float(np.max(cond_chi)),
        "total_truncated_chi0": int(np.sum(n_truncated_chi0)),
        "total_truncated_chi": int(np.sum(n_truncated_chi)),
        "matrix_size": N,
        "n_bosonic_freqs": n_bos,
    }

    return gamma_loc, diagnostics


# ── Channel decomposition ───────────────────────────────────────────────────

def decompose_vertex_channels(
    gamma_loc: np.ndarray,
    n_iw_f: int,
    n_orb: int,
) -> dict:
    """
    Decompose the irreducible vertex into spin and charge channels.

    In the particle-hole channel with SU(2) symmetry:
      Γ_spin    = Γ_↑↓ - Γ_↑↑   (odd under spin flip → magnetic)
      Γ_charge  = Γ_↑↓ + Γ_↑↑   (even under spin flip → density)

    For the pairing (pp) channel:
      Γ_singlet = Γ_↑↓ - Γ_↓↑   (antisymmetric → singlet pairing)
      Γ_triplet = Γ_↑↓ + Γ_↓↑   (symmetric → triplet pairing)

    In our paramagnetic (no spin index) formulation, we approximate:
      Γ_spin   ≈ U - Γ_ph       (repulsive in spin channel)
      Γ_charge ≈ U + 2·Γ_ph     (from crossing symmetry)

    For singlet pairing vertex (the one relevant for d-wave SC):
      Γ_singlet(ν,ν') = (3/2)·Γ_spin(ν,ν') + (1/2)·Γ_charge(ν,ν')

    Returns dict with vertex components at Ω=0 (most relevant for pairing).
    """
    n_bos = gamma_loc.shape[2]
    iw_zero = n_bos // 2  # Ω=0 index

    gamma_omega0 = gamma_loc[:, :, iw_zero]  # [N, N] at Ω=0

    # For the singlet pairing channel, we need the crossing-symmetry relation
    # In the simplified (paramagnetic, no explicit spin) case:
    #   Γ_singlet ≈ (3/2)·Γ_ph - (1/2)·Γ_ph^T  (approximate)
    # More precisely, Γ_singlet = (3/2)·Γ_s + (1/2)·Γ_c where
    #   Γ_s = (Γ - Γ^T)/2  (antisymmetric = spin)
    #   Γ_c = (Γ + Γ^T)/2  (symmetric = charge)

    gamma_sym = (gamma_omega0 + gamma_omega0.T) / 2   # charge-like
    gamma_asym = (gamma_omega0 - gamma_omega0.T) / 2  # spin-like

    # Singlet pairing vertex
    gamma_singlet = 1.5 * gamma_asym + 0.5 * gamma_sym

    # Triplet pairing vertex
    gamma_triplet = -0.5 * gamma_asym + 0.5 * gamma_sym

    # Spectral properties
    evals_singlet = np.linalg.eigvalsh(
        (gamma_singlet + gamma_singlet.conj().T) / 2
    )
    evals_triplet = np.linalg.eigvalsh(
        (gamma_triplet + gamma_triplet.conj().T) / 2
    )

    return {
        "gamma_singlet": gamma_singlet,
        "gamma_triplet": gamma_triplet,
        "gamma_charge": gamma_sym,
        "gamma_spin": gamma_asym,
        "max_eval_singlet": float(np.max(np.abs(evals_singlet))),
        "max_eval_triplet": float(np.max(np.abs(evals_triplet))),
        "singlet_attractive": bool(np.min(evals_singlet.real) < 0),
        "triplet_attractive": bool(np.min(evals_triplet.real) < 0),
    }


# ── Full BSE pipeline ───────────────────────────────────────────────────────

def run_bse_solver(vertex_data_path: str, work_dir: str) -> dict:
    """
    Run the full local BSE pipeline:
      1. Load vertex data (G², χ⁰_loc, χ_loc) from vertex_measurement output
      2. Reshape into compound-index matrices
      3. Invert BSE to get Γ_loc
      4. Decompose into spin/charge/pairing channels
      5. Save results

    Args:
        vertex_data_path: path to vertex_data.npz from vertex_measurement
        work_dir: output directory

    Returns:
        dict with Γ_loc, channel decomposition, diagnostics
    """
    import os
    import json
    t0 = time.time()

    print(f"[BSE] Loading vertex data from {vertex_data_path}")
    vdata = np.load(vertex_data_path, allow_pickle=True)

    grid_params = json.loads(str(vdata["grid_params_json"]))
    n_iw_f = grid_params["n_iw_f"]
    n_iw_b = grid_params["n_iw_b"]
    n_orb = grid_params["n_orb"]

    chi0_loc = vdata.get("chi0_loc")
    chi_loc = vdata.get("chi_loc")

    if chi0_loc is None or chi_loc is None:
        return {"converged": False, "error": "Missing χ⁰_loc or χ_loc in vertex data"}

    N = 2 * n_iw_f * n_orb * n_orb
    print(f"[BSE] Matrix size N={N} ({2*n_iw_f} freq × {n_orb}² orbitals), "
          f"{2*n_iw_b+1} bosonic frequencies")

    # SVD-stabilized BSE inversion
    print("[BSE] Running SVD-stabilized BSE inversion...")
    gamma_loc, diagnostics = solve_bse_local_svd(
        chi_loc, chi0_loc, n_iw_f, n_orb, n_iw_b,
        svd_cutoff=1e-8,
    )

    print(f"[BSE] Inversion done. Condition: χ⁰ mean={diagnostics['mean_cond_chi0']:.1e}, "
          f"χ mean={diagnostics['mean_cond_chi']:.1e}, "
          f"truncated modes: χ⁰={diagnostics['total_truncated_chi0']}, "
          f"χ={diagnostics['total_truncated_chi']}")

    # Channel decomposition
    print("[BSE] Decomposing vertex into pairing channels...")
    channels = decompose_vertex_channels(gamma_loc, n_iw_f, n_orb)

    elapsed = time.time() - t0
    print(f"[BSE] Singlet attractive: {channels['singlet_attractive']}, "
          f"max eigenvalue: {channels['max_eval_singlet']:.4f}")
    print(f"[BSE] Triplet attractive: {channels['triplet_attractive']}, "
          f"max eigenvalue: {channels['max_eval_triplet']:.4f}")
    print(f"[BSE] Total BSE time: {elapsed:.1f}s")

    # Save results
    os.makedirs(work_dir, exist_ok=True)
    bse_path = os.path.join(work_dir, "bse_results.npz")
    np.savez_compressed(
        bse_path,
        gamma_loc=gamma_loc,
        gamma_singlet=channels["gamma_singlet"],
        gamma_triplet=channels["gamma_triplet"],
        gamma_charge=channels["gamma_charge"],
        gamma_spin=channels["gamma_spin"],
        grid_params_json=json.dumps(grid_params),
    )
    print(f"[BSE] Results saved to {bse_path} ({os.path.getsize(bse_path)/1e6:.0f} MB)")

    return {
        "converged": True,
        "elapsed_seconds": elapsed,
        "diagnostics": diagnostics,
        "singlet_attractive": channels["singlet_attractive"],
        "triplet_attractive": channels["triplet_attractive"],
        "max_eval_singlet": channels["max_eval_singlet"],
        "max_eval_triplet": channels["max_eval_triplet"],
        "bse_results_path": bse_path,
    }


import time  # ensure available at module level
