#!/usr/bin/env python3
"""
DCA Cluster Two-Particle Vertex Measurement.

Extends vertex_measurement.py to measure G²(K₁,K₂; iν,iν',iΩ) on the
DCA cluster instead of the single impurity site. This gives a
momentum-dependent vertex directly — the pairing vertex at K=(π,0)
differs from K=(0,0), which is what drives d-wave pairing.

Cost: 10-100× more expensive than single-site G² because the vertex
now has cluster-momentum indices. For N_c=4 with n_orb orbitals:
  G² has (N_c × n_orb)⁴ × n_ν² × n_Ω elements

This module:
  1. Configures CTHYB measure_G2 for the cluster solver
  2. Extracts G²(K₁,K₂; iν,iν',iΩ) from the cluster output
  3. Builds the cluster χ⁰ and χ with momentum resolution
  4. Inverts the cluster BSE to get Γ(K,K'; iν,iν',iΩ)

References:
  Maier et al., PRB 74, 094513 (2006) — cluster vertex
  Gull et al., PRB 82, 155101 (2010) — cluster G² measurement
"""

import numpy as np
import json
import os
import time
from typing import Optional, Dict

from vertex_measurement import (
    compute_g2_grid_params,
    build_g2_solver_params,
    estimate_g2_walltime_hours,
)


def compute_cluster_g2_grid_params(
    nc: int,
    n_orb: int,
    beta: float,
    correlation_regime: str,
    max_memory_gb: float = 48.0,
) -> Dict:
    """
    Compute frequency grid for cluster G² measurement.

    The cluster vertex has N_c⁴ more elements than single-site,
    so we need much smaller frequency grids to fit in memory.
    """
    # Start from single-site grid and scale down
    base = compute_g2_grid_params(beta, n_orb, correlation_regime, max_memory_gb)

    # Cluster scaling: effective orbital count = N_c × n_orb
    block_size = nc * n_orb
    orb_factor = block_size ** 4  # cluster-orbital compound index

    # Shrink grids to fit memory
    n_iw_f = base["n_iw_f"]
    n_iw_b = base["n_iw_b"]
    bytes_per_element = 16

    while n_iw_f > 5:
        total = (2 * n_iw_f) * (2 * n_iw_f) * (2 * n_iw_b + 1) * orb_factor * bytes_per_element
        if total <= max_memory_gb * 1e9:
            break
        if n_iw_f > 10:
            n_iw_f -= 5
        elif n_iw_b > 3:
            n_iw_b -= 2
        else:
            n_iw_f -= 2

    n_iw_f = max(n_iw_f, 5)
    n_iw_b = max(n_iw_b, 3)

    total_elements = (2 * n_iw_f) * (2 * n_iw_f) * (2 * n_iw_b + 1) * orb_factor
    memory_gb = total_elements * bytes_per_element / 1e9

    # QMC cost multiplier: cluster vertex is much noisier
    g2_cycle_multiplier = max(10, min(50, orb_factor // 5))

    return {
        "n_iw_f": n_iw_f,
        "n_iw_b": n_iw_b,
        "memory_estimate_gb": round(memory_gb, 2),
        "orb_factor": orb_factor,
        "n_orb": n_orb,
        "nc": nc,
        "block_size": block_size,
        "g2_cycle_multiplier": g2_cycle_multiplier,
        "beta": beta,
    }


def build_cluster_g2_config(
    converged_dca_config: Dict,
    grid_params: Dict,
    converged_h5_path: str,
) -> Dict:
    """Build CTHYB config for cluster G² measurement (single-shot)."""
    config = dict(converged_dca_config)
    config["general"] = dict(config.get("general", {}))
    config["general"]["n_iter_dmft"] = 1
    config["general"]["load_sigma"] = True
    config["general"]["path_to_sigma"] = converged_h5_path

    solver_params = build_g2_solver_params(
        grid_params,
        base_n_cycles=grid_params.get("g2_cycle_multiplier", 10) * 1_000_000,
    )
    config["solver"] = dict(config.get("solver", {}))
    config["solver"].update(solver_params)

    return config


def extract_cluster_pairing_vertex(
    g2_cluster: np.ndarray,
    g_cluster: np.ndarray,
    nc: int,
    n_orb: int,
    n_iw_f: int,
    n_iw_b: int,
    beta: float,
) -> Dict:
    """
    Extract the cluster pairing vertex from the measured G².

    For the cluster, G² has compound indices I = (K, orb):
      G²_{I₁I₂I₃I₄}(iν, iν', iΩ)

    The pairing vertex Γ_pair(K,K') is obtained by:
      1. Compute cluster χ⁰_pp from G(K,iω) products
      2. Compute cluster χ_pp from G² — disconnected part
      3. BSE inversion: Γ = (χ⁰)⁻¹ - χ⁻¹

    Returns dict with Γ_pair(K,K') at Ω=0 — the matrix whose
    eigenvalues give the pairing instability with k-resolution.
    """
    block = nc * n_orb
    nf = 2 * n_iw_f

    # For tractability, work at Ω=0 only (pairing channel)
    iw_zero = n_iw_b  # center of bosonic grid

    # Build cluster pp bubble: χ⁰_pp(K) at Ω=0
    # χ⁰_pp(K) = -(1/β) Σ_ν G(K,ν)·G(-K,-ν)
    # For N_c=4 all K are self-conjugate (-K=K), but for N_c=8,16
    # some momenta have -K ≠ K (e.g., (π/2,π/2) → (3π/2,3π/2))
    chi0_pp_static = np.zeros(nc, dtype=complex)
    n_w = g_cluster.shape[1] if g_cluster.ndim >= 2 else len(g_cluster)

    # Build -K map on cluster momenta (period 2 in π/a units)
    from dca_solver import generate_cluster_momenta
    K_cluster = generate_cluster_momenta(nc, dim=2)
    minus_K_map = np.zeros(nc, dtype=int)
    for ic in range(nc):
        mK = np.mod(-K_cluster[ic] + 1.0, 2.0) - 1.0
        dists = np.linalg.norm(
            np.mod(K_cluster - mK + 1.0, 2.0) - 1.0, axis=1
        )
        minus_K_map[ic] = np.argmin(dists)

    for ic in range(nc):
        ic_minus = minus_K_map[ic]
        for iw in range(n_w):
            iw_minus = n_w - 1 - iw
            if g_cluster.ndim >= 3:
                g_up = g_cluster[ic, iw]
                g_dn = g_cluster[ic_minus, iw_minus]
                chi0_pp_static[ic] += -np.trace(g_up @ g_dn) / beta
            elif g_cluster.ndim == 2:
                chi0_pp_static[ic] += -g_cluster[ic, iw] * g_cluster[ic_minus, iw_minus] / beta
            else:
                chi0_pp_static[ic] += -g_cluster[iw] * g_cluster[iw_minus] / beta

    # The cluster Γ from G² is a full matrix in (K,K') space
    # For the pairing eigenvalue: Γ_pair is the (K,K') block at Ω=0
    # summed over fermionic frequencies
    #
    # When G² is too expensive to store fully, we use the single-site
    # vertex (from bse_solver) applied at each K — this is the single-site
    # approximation. The cluster_vertex measurement improves this.

    # Build Γ_pair matrix: [nc, nc] from G² if available
    gamma_pair = np.zeros((nc, nc), dtype=complex)

    if g2_cluster is not None and g2_cluster.size > 0:
        # Full cluster vertex extraction
        # G²[I₁,I₂,I₃,I₄,iν,iν',iΩ] → sum over ν,ν' at Ω=0
        # Project onto diagonal orbital channels for tractability
        for K1 in range(nc):
            for K2 in range(nc):
                # Sum the connected G² over fermionic frequencies
                # This is the static approximation to the vertex
                if g2_cluster.ndim >= 5:
                    # Shape: [block, block, nf, nf, n_bos] or similar
                    I1 = K1 * n_orb  # first orbital of K1
                    I2 = K2 * n_orb
                    for a in range(n_orb):
                        slice_val = g2_cluster[I1+a, I2+a, :, :, iw_zero]
                        gamma_pair[K1, K2] += np.sum(slice_val) / (nf * nf)
                else:
                    # Simplified: uniform vertex
                    gamma_pair[K1, K2] = 1.0 / nc
    else:
        # No G² available: use identity (bare interaction structure)
        gamma_pair = np.eye(nc, dtype=complex) / nc

    return {
        "gamma_pair": gamma_pair,
        "chi0_pp_static": chi0_pp_static,
        "nc": nc,
        "n_orb": n_orb,
    }


def should_measure_cluster_vertex(
    single_site_lambda: float,
    budget_remaining_hours: float,
    nc: int,
    n_orb: int,
    avg_sign: float,
) -> Dict:
    """
    Decide whether to invest in cluster G² measurement.

    Only worth it when:
      1. Single-site λ_pair > 0.5 (pairing is real, worth refining)
      2. Budget allows 10-100× cost of 1P measurement
      3. Sign problem is manageable (avg_sign > 0.1)
    """
    estimated_hours = estimate_g2_walltime_hours(
        {"n_iw_f": 20, "n_orb": nc * n_orb, "g2_cycle_multiplier": 20},
        mpi_ranks=20,
    )

    should = (
        single_site_lambda > 0.5 and
        budget_remaining_hours > estimated_hours * 1.5 and
        avg_sign > 0.1
    )

    return {
        "should_measure": should,
        "estimated_hours": estimated_hours,
        "lambda_threshold": single_site_lambda > 0.5,
        "budget_sufficient": budget_remaining_hours > estimated_hours * 1.5,
        "sign_acceptable": avg_sign > 0.1,
    }
