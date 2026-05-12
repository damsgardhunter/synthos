#!/usr/bin/env python3
"""
DCA Self-Energy Periodization — Σ(K) → Σ(k) on the full BZ.

DCA gives Σ at N_c cluster momenta K. For Fermi surface plots, ARPES
comparison, and real-axis spectroscopy we need Σ(k) on the full BZ.

Three standard methods:
  1. Nearest-patch: Σ(k) = Σ(K(k)) — discontinuous but causal
  2. Self-energy periodization: Σ(k) = Σ_R Σ(R)·exp(ik·R) — smooth but can violate causality
  3. Cumulant periodization: M(k) = Σ_R M(R)·exp(ik·R) where M=Σ/(1+Σ·G₀) — smooth AND causal

Method 3 (cumulant) is the default.

References:
  Maier et al., Rev. Mod. Phys. 77, 1027 (2005) — Sec. II.D
  Biroli et al., PRB 65, 155112 (2002) — cumulant periodization
  Stanescu & Kotliar, PRB 74, 125110 (2006) — comparison of methods
"""

import numpy as np
from typing import Optional, Dict
import os
import json


def periodize_self_energy(
    sigma_K: np.ndarray,
    K_cluster: np.ndarray,
    kpoints: np.ndarray,
    method: str = "cumulant",
    g0_K: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolate cluster self-energy Σ(K) to full BZ Σ(k).

    Args:
        sigma_K: [nc, 2*n_iw] or [nc, 2*n_iw, n_orb, n_orb] cluster Σ
        K_cluster: [nc, dim_cluster] cluster momenta in π/a units
        kpoints: [n_k, dim_k] target k-points in π/a units
                 (if dim_k > dim_cluster, extra dimensions are ignored)
        method: "nearest", "self_energy", or "cumulant"
        g0_K: [nc, 2*n_iw, ...] bare cluster G⁰ (needed for cumulant)

    Returns:
        sigma_k: [n_k, 2*n_iw, ...] interpolated self-energy
    """
    # Truncate kpoints to match cluster dimension (e.g., 3D bundle → 2D DCA)
    dim_cluster = K_cluster.shape[1]
    if kpoints.shape[1] > dim_cluster:
        kpoints = kpoints[:, :dim_cluster]

    if method == "nearest":
        return _periodize_nearest(sigma_K, K_cluster, kpoints)
    elif method == "self_energy":
        return _periodize_fourier(sigma_K, K_cluster, kpoints)
    elif method == "cumulant":
        if g0_K is None:
            print("[Periodize] WARNING: cumulant needs G⁰(K), falling back to self-energy")
            return _periodize_fourier(sigma_K, K_cluster, kpoints)
        return _periodize_cumulant(sigma_K, K_cluster, kpoints, g0_K)
    else:
        raise ValueError(f"Unknown periodization method: {method}")


def _compute_cluster_R(K_cluster, nc, dim):
    """Find real-space vectors R satisfying DCA FT orthogonality with K_cluster."""
    from itertools import combinations

    L_c = int(round(nc ** (1.0 / dim)))
    if L_c ** dim == nc:
        # Square cluster: simple integer grid
        if dim == 2:
            return np.array([[i, j] for i in range(L_c) for j in range(L_c)], dtype=float)
        else:
            return np.array([[i, j, k] for i in range(L_c) for j in range(L_c) for k in range(L_c)], dtype=float)

    # Non-square: search over integer points for FT-orthogonal set
    box = int(np.ceil(np.sqrt(nc))) + 1
    if dim == 2:
        candidates = [(i, j) for i in range(box + 1) for j in range(box + 1)]
    else:
        candidates = [(i, j, k) for i in range(box + 1) for j in range(box + 1) for k in range(box + 1)]

    for combo in combinations(range(len(candidates)), nc):
        R = np.array([candidates[c] for c in combo], dtype=float)
        F = np.exp(1j * np.pi * K_cluster @ R.T)
        FtF = F.conj().T @ F / nc
        if np.max(np.abs(FtF - np.eye(nc))) < 1e-8:
            return R

    # Fallback: K_cluster itself (may not be orthogonal for non-square clusters)
    return K_cluster.copy()


def _periodize_nearest(sigma_K, K_cluster, kpoints):
    """Nearest-patch: step function, discontinuous but always causal."""
    from dca_solver import assign_k_to_patches
    assignment = assign_k_to_patches(kpoints, K_cluster)
    return sigma_K[assignment]


def _periodize_fourier(sigma_K, K_cluster, kpoints):
    """
    Self-energy Fourier periodization:
      1. FT cluster → real space: Σ(R) = (1/N_c) Σ_K Σ(K) exp(-iK·R)
      2. Interpolate to BZ: Σ(k) = Σ_R Σ(R) exp(ik·R)

    Smooth but can violate causality (Im[Σ(k,ω)] > 0 at some k).
    """
    nc = sigma_K.shape[0]
    shape_rest = sigma_K.shape[1:]
    dim = K_cluster.shape[1]

    # Cluster real-space vectors R must satisfy the DCA orthogonality:
    #   (1/N_c) Σ_R exp(i(K-K')·R·π) = δ_{K,K'}
    # Computed via brute-force search over integer lattice points.
    R_cluster = _compute_cluster_R(K_cluster, nc, dim)

    # FT to real space: Σ(R) = (1/N_c) Σ_K Σ(K) exp(-iK·R·π)
    sigma_R = np.zeros_like(sigma_K)
    for ir in range(nc):
        for ik in range(nc):
            phase = np.exp(-1j * np.pi * np.dot(K_cluster[ik], R_cluster[ir]))
            sigma_R[ir] += sigma_K[ik] * phase / nc

    # Interpolate: Σ(k) = Σ_R Σ(R) exp(ik·R·π)
    n_k = kpoints.shape[0]
    sigma_k = np.zeros((n_k,) + shape_rest, dtype=complex)
    for ir in range(nc):
        # phase[n_k] = exp(ik·R·π) for all k-points
        phases = np.exp(1j * np.pi * (kpoints @ R_cluster[ir]))
        # Broadcast phase over frequency/orbital dimensions
        if len(shape_rest) == 1:
            sigma_k += sigma_R[ir][np.newaxis, :] * phases[:, np.newaxis]
        elif len(shape_rest) == 3:
            sigma_k += sigma_R[ir][np.newaxis, :, :, :] * phases[:, np.newaxis, np.newaxis, np.newaxis]
        else:
            for ik in range(n_k):
                sigma_k[ik] += sigma_R[ir] * phases[ik]

    return sigma_k


def _periodize_cumulant(sigma_K, K_cluster, kpoints, g0_K):
    """
    Cumulant periodization — smooth AND causal.

    Instead of periodizing Σ directly, periodize the cumulant M:
      M(K) = Σ(K) / (1 + Σ(K) · G⁰(K))
      M(k) = Σ_R M(R) · exp(ik·R)   (Fourier interpolation)
      Σ(k) = M(k) / (1 - M(k) · G⁰_interp(k))

    The cumulant M has better analytic properties than Σ, so its
    Fourier interpolation preserves causality.
    """
    nc = sigma_K.shape[0]
    shape_rest = sigma_K.shape[1:]
    is_multiorbital = len(shape_rest) == 3  # [n_w, n_orb, n_orb]

    # Compute cumulant M(K) = [1 + Σ(K)·G⁰(K)]⁻¹ · Σ(K)  (Stanescu & Kotliar PRB 74, Eq. 4)
    M_K = np.zeros_like(sigma_K)

    if is_multiorbital:
        n_w = shape_rest[0]
        n_orb = shape_rest[1]
        for ic in range(nc):
            for iw in range(n_w):
                SG = sigma_K[ic, iw] @ g0_K[ic, iw]
                denom = np.eye(n_orb) + SG
                M_K[ic, iw] = np.linalg.solve(denom, sigma_K[ic, iw])
    else:
        # Single-orbital
        n_w = shape_rest[0]
        for ic in range(nc):
            for iw in range(n_w):
                denom = 1.0 + sigma_K[ic, iw] * g0_K[ic, iw]
                M_K[ic, iw] = sigma_K[ic, iw] / denom if abs(denom) > 1e-20 else 0.0

    # Fourier-interpolate M(K) → M(k)
    M_k = _periodize_fourier.__wrapped__(M_K, K_cluster, kpoints) if hasattr(_periodize_fourier, '__wrapped__') else _periodize_fourier(M_K, K_cluster, kpoints)

    # Recover Σ(k) from M(k)
    # Need G⁰(k) — approximate with nearest-patch G⁰
    g0_k = _periodize_nearest(g0_K, K_cluster, kpoints)

    n_k = kpoints.shape[0]
    sigma_k = np.zeros((n_k,) + shape_rest, dtype=complex)

    if is_multiorbital:
        n_orb = shape_rest[1]
        for ik in range(n_k):
            for iw in range(n_w):
                MG = M_k[ik, iw] @ g0_k[ik, iw]
                denom = np.eye(n_orb) - MG
                try:
                    sigma_k[ik, iw] = np.linalg.solve(denom, M_k[ik, iw])
                except np.linalg.LinAlgError:
                    sigma_k[ik, iw] = M_k[ik, iw]
    else:
        for ik in range(n_k):
            for iw in range(n_w):
                denom = 1.0 - M_k[ik, iw] * g0_k[ik, iw]
                sigma_k[ik, iw] = M_k[ik, iw] / denom if abs(denom) > 1e-20 else M_k[ik, iw]

    return sigma_k


def compute_fermi_surface(
    hk: np.ndarray,
    sigma_k: np.ndarray,
    mu: float,
    kpoints: np.ndarray,
    beta: float,
) -> Dict:
    """
    Compute Fermi surface from A(k, ω=0).

    A(k, ω=0) = -(1/π) Im[G(k, ω=0+iδ)]
              ≈ -(1/π) Im[1 / (iω_0 + μ - H(k) - Σ(k, iω_0))]

    where ω_0 is the lowest positive Matsubara frequency.
    """
    n_k = hk.shape[0]
    n_iw = sigma_k.shape[1] // 2
    omega_0 = np.pi / beta

    # Use lowest Matsubara frequency as proxy for ω≈0
    a_kf = np.zeros(n_k)

    if hk.ndim == 3:
        n_orb = hk.shape[1]
        eye = np.eye(n_orb, dtype=complex)
        for ik in range(n_k):
            M = (1j * omega_0 + mu) * eye - hk[ik] - sigma_k[ik, n_iw]
            G = np.linalg.inv(M)
            a_kf[ik] = -np.trace(G).imag / np.pi
    else:
        for ik in range(n_k):
            G = 1.0 / (1j * omega_0 + mu - hk[ik] - sigma_k[ik, n_iw])
            a_kf[ik] = -G.imag / np.pi

    return {
        "kpoints": kpoints.tolist(),
        "spectral_weight_at_ef": a_kf.tolist(),
        "max_weight": float(np.max(a_kf)),
        "fermi_surface_volume": float(np.sum(a_kf > np.max(a_kf) * 0.3) / n_k),
    }
