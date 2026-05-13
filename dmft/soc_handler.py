#!/usr/bin/env python3
"""
Spin-Orbit Coupling (SOC) handler for DMFT pipeline.

When the DFT calculation includes SOC, the Hamiltonian H(k) is a 2n_orb × 2n_orb
matrix where the basis is doubled: [|orb_0,↑⟩, ..., |orb_{n_orb-1},↑⟩,
|orb_0,↓⟩, ..., |orb_{n_orb-1},↓⟩]. The off-diagonal spin blocks (↑↓, ↓↑)
contain the SOC mixing.

This module provides:
  - Helper functions to detect SOC from bundle data
  - Block-extraction utilities for spin-resolved DMFT solving
  - Reconstruction of full SOC-included G(k,ω) from solved blocks

The standard approach for DMFT+SOC:
  1. Diagonalize H(k) in the spin×orbital basis → get "natural" SOC basis
  2. The Hubbard U is applied in the ORBITAL basis (not natural basis)
  3. CTHYB sees the full 2n_orb × 2n_orb Green's function
  4. Σ(iω) is generally non-block-diagonal in spin (SOC mixes spins)

For paramagnetic systems with weak SOC, the spin mixing in Σ is small and
DMFT can be performed with block-diagonal approximation (treat ↑↓ blocks
of Σ as zero). The current pipeline supports this APPROXIMATE SOC mode.

For strong SOC (Pt, Bi, U), the full off-diagonal Σ matters and needs
the full 2n_orb solver.

References:
  Aichhorn et al., PRB 84, 054529 (2011) — TRIQS DFT+DMFT with SOC
  Werner & Gull, PRB 86, 085114 (2012) — CTHYB with SOC
"""

import numpy as np
from typing import Optional, Dict


def has_soc_metadata(bundle_data: dict) -> bool:
    """
    Check if the bundle indicates SOC was included in the DFT calculation.

    The bundle's metadata should contain a "soc" or "spin_orbit" flag,
    or have H(k) of shape [n_k, 2*n_orb, 2*n_orb] (doubled in spin).
    """
    if bundle_data.get("soc", False) or bundle_data.get("spin_orbit", False):
        return True
    meta = bundle_data.get("metadata", {})
    if isinstance(meta, dict):
        if meta.get("soc", False) or meta.get("spin_orbit", False):
            return True
    # Heuristic: check if H(k) shape suggests doubled basis
    hk = bundle_data.get("hk")
    if hk is not None and hk.ndim == 3:
        n_orb_hk = hk.shape[1]
        n_orb_corr = sum(s.get("dim", 0) for s in bundle_data.get("corr_shells", []))
        if n_orb_corr > 0 and n_orb_hk == 2 * n_orb_corr:
            return True
    return False


def extract_spin_blocks(hk_soc: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split a SOC Hamiltonian H(k) of shape [n_k, 2*n_orb, 2*n_orb] into spin blocks.

    Basis ordering: [orb_0↑, ..., orb_{N-1}↑, orb_0↓, ..., orb_{N-1}↓]

    Returns:
        {
          "uu": H_{↑↑}(k) [n_k, n_orb, n_orb] — same-spin up
          "dd": H_{↓↓}(k) [n_k, n_orb, n_orb] — same-spin down
          "ud": H_{↑↓}(k) [n_k, n_orb, n_orb] — spin-flip
          "du": H_{↓↑}(k) [n_k, n_orb, n_orb] — spin-flip (conjugate)
        }
    """
    n_total = hk_soc.shape[1]
    if n_total % 2 != 0:
        raise ValueError(f"SOC H(k) must have even orbital count, got {n_total}")
    n_orb = n_total // 2
    return {
        "uu": hk_soc[:, :n_orb, :n_orb],
        "ud": hk_soc[:, :n_orb, n_orb:],
        "du": hk_soc[:, n_orb:, :n_orb],
        "dd": hk_soc[:, n_orb:, n_orb:],
    }


def assemble_soc_matrix(blocks: Dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of extract_spin_blocks: combine spin blocks back into 2n_orb matrix."""
    uu = blocks["uu"]
    n_k = uu.shape[0]
    n_orb = uu.shape[1]
    full = np.zeros((n_k, 2 * n_orb, 2 * n_orb), dtype=complex)
    full[:, :n_orb, :n_orb] = uu
    full[:, :n_orb, n_orb:] = blocks.get("ud", np.zeros_like(uu))
    full[:, n_orb:, :n_orb] = blocks.get("du", np.zeros_like(uu))
    full[:, n_orb:, n_orb:] = blocks.get("dd", uu)  # default: SU(2)
    return full


def soc_strength(hk_soc: np.ndarray) -> float:
    """
    Estimate the SOC mixing strength as the ratio of ||H_ud||_F / ||H_uu||_F.

    Returns 0 for no SOC, ~0.1 for moderate (e.g., 3d transition metals),
    ~0.5 for strong (4d/5d), >1 possible for very heavy elements.
    """
    blocks = extract_spin_blocks(hk_soc)
    norm_uu = np.linalg.norm(blocks["uu"]) + 1e-30
    norm_ud = np.linalg.norm(blocks["ud"])
    return float(norm_ud / norm_uu)


def compute_lattice_gf_soc(
    hk_soc: np.ndarray,
    sigma_iw: np.ndarray,
    mu: float,
    beta: float,
    n_iw: int,
) -> np.ndarray:
    """
    Compute lattice Green's function with SOC included.

    G(k, iω) = [iω + μ - H(k) - Σ(iω)]⁻¹

    where all matrices are 2n_orb × 2n_orb in the spin×orbital basis.

    Args:
        hk_soc: [n_k, 2*n_orb, 2*n_orb] SOC Hamiltonian
        sigma_iw: [2*n_iw, 2*n_orb, 2*n_orb] self-energy with SOC structure
                  (off-diagonal spin blocks allowed for strong SOC)
        mu: chemical potential
        beta: inverse temperature
        n_iw: half the number of Matsubara frequencies

    Returns:
        gk_iw: [n_k, 2*n_iw, 2*n_orb, 2*n_orb]
    """
    n_k = hk_soc.shape[0]
    n_total = hk_soc.shape[1]
    nw = 2 * n_iw
    wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(nw)])
    eye = np.eye(n_total, dtype=complex)

    gk_iw = np.zeros((n_k, nw, n_total, n_total), dtype=complex)
    for iw in range(nw):
        M = (1j * wn[iw] + mu) * eye[np.newaxis, :, :] - hk_soc - sigma_iw[iw][np.newaxis, :, :]
        gk_iw[:, iw, :, :] = np.linalg.inv(M)
    return gk_iw


def soc_gf_struct(n_orb: int) -> list:
    """
    Build the TRIQS gf_struct for a SOC calculation.

    With SOC, spin is no longer a good quantum number, so we use a single
    "ud" block of size 2*n_orb (instead of separate "up" and "down" blocks
    of size n_orb).

    Returns: [("ud", 2*n_orb)]
    """
    return [("ud", 2 * n_orb)]


def expand_kanamori_to_soc(
    interaction,  # KanamoriInteraction
    n_orb: int,
) -> "Operator":
    """
    Build the Kanamori interaction Hamiltonian for the SOC basis.

    In the SOC basis, the compound index runs over (spin, orbital) pairs:
      I = spin * n_orb + orb  (spin ∈ {0=↑, 1=↓})

    The Hubbard U is applied in the ORIGINAL orbital basis (before SOC).
    For each orbital a, we have intra-orbital U: U n_{a↑} n_{a↓}.
    Inter-orbital terms work the same way.

    Returns a TRIQS Operator on the "ud" block.
    """
    from triqs.operators import c, c_dag, n, Operator

    H = Operator()

    # Helper: compound index for (spin, orbital)
    def idx(spin, orb):
        # spin: 0=up, 1=down. Index runs over spin first then orbital
        # to match TRIQS convention for SOC blocks (ud block of size 2*n_orb)
        return spin * n_orb + orb

    # Intra-orbital: U n_{a↑} n_{a↓}
    for a in range(n_orb):
        H += interaction.U[a] * n("ud", idx(0, a)) * n("ud", idx(1, a))

    # Inter-orbital
    for a in range(n_orb):
        for b in range(n_orb):
            if a == b:
                continue
            Uprime = interaction.U_prime[a, b]
            Jpair = interaction.J_pair[a, b]

            # Opposite-spin density-density
            H += Uprime * n("ud", idx(0, a)) * n("ud", idx(1, b))

            # Same-spin density-density (a<b to avoid double-counting)
            if a < b:
                H += (Uprime - Jpair) * n("ud", idx(0, a)) * n("ud", idx(0, b))
                H += (Uprime - Jpair) * n("ud", idx(1, a)) * n("ud", idx(1, b))

            # Spin-flip
            H += -Jpair * (
                c_dag("ud", idx(0, a)) * c("ud", idx(1, a)) *
                c_dag("ud", idx(1, b)) * c("ud", idx(0, b))
            )

            # Pair-hopping: sum over all a≠b for rotational invariance
            H += Jpair * (
                c_dag("ud", idx(0, a)) * c_dag("ud", idx(1, a)) *
                c("ud", idx(1, b)) * c("ud", idx(0, b))
            )

    return H


def estimate_soc_sign_problem_penalty(
    soc_strength_val: float,
    n_orb: int,
    beta: float,
    max_penalty: float = None,
) -> float:
    """
    Estimate the additional sign problem penalty from including SOC.

    The CTHYB sign problem scales as exp(β·U·N_off), where N_off is the
    number of off-diagonal Σ elements. SOC roughly doubles N_off compared
    to a non-SOC calculation, so the sign penalty is approximately:

      penalty ~ exp(β·U·n_orb·soc_strength) for moderate SOC

    Returns a multiplicative factor for QMC cycles (>= 1).

    Args:
        max_penalty: upper bound on the multiplier. Default reads
            DMFT_SOC_PENALTY_MAX env var (fallback 50). The previous hard
            cap of 10 silently undersampled heavy-element materials (Bi,
            Hg, U, Pu) where the SOC sign problem is genuinely severe.
            Setting this to a higher value lets the QMC actually resolve
            the sign at the cost of walltime.
    """
    import os
    if max_penalty is None:
        try:
            max_penalty = float(os.environ.get("DMFT_SOC_PENALTY_MAX", "50"))
        except (TypeError, ValueError):
            max_penalty = 50.0

    # Empirical: penalty ~ 1 + 5·soc_strength·β/100·n_orb
    penalty = 1.0 + 5.0 * soc_strength_val * (beta / 100.0) * n_orb
    return float(max(1.0, min(max_penalty, penalty)))
