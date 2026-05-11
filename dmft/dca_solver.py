#!/usr/bin/env python3
"""
Dynamical Cluster Approximation (DCA) Solver.

DCA is a cluster extension of DMFT that preserves translational symmetry
by working in momentum space. Instead of mapping the lattice onto a single
impurity site, DCA maps it onto a cluster of N_c sites in momentum space.

The Brillouin zone is partitioned into N_c patches centered on cluster
momenta K. Within each patch, the self-energy is approximated as constant:
  Σ(k, iω) ≈ Σ(K, iω)  for k ∈ patch(K)

This allows d-wave pairing to emerge because the self-energy at K=(0,0)
differs from K=(π,0), breaking the isotropic s-wave constraint of
single-site DMFT.

DCA self-consistency loop:
  1. Start with Σ_c(K, iω) = 0 (or from single-site DMFT)
  2. Compute coarse-grained lattice G:
       Ḡ(K, iω) = (N_k/N_c) Σ_{k∈patch(K)} [iω + μ - ε(k) - Σ_c(K, iω)]⁻¹
  3. Compute cluster bare G via Dyson:
       G⁰_c(K, iω) = [Ḡ(K, iω)⁻¹ + Σ_c(K, iω)]⁻¹
  4. Solve the N_c-site cluster problem with G⁰_c as bath → get G_c(K, iω)
  5. Extract new Σ_c via Dyson:
       Σ_c(K, iω) = G⁰_c(K, iω)⁻¹ - G_c(K, iω)⁻¹
  6. Mix: Σ_new = α·Σ_old + (1-α)·Σ_extracted
  7. Check convergence: ||Σ_new - Σ_old|| < tol → stop, else go to 2

For the 2×2 plaquette (N_c=4), the cluster momenta are:
  K₀ = (0, 0)    — Γ point
  K₁ = (π, 0)    — X point
  K₂ = (0, π)    — Y point
  K₃ = (π, π)    — M point

References:
  Maier et al., Rev. Mod. Phys. 77, 1027 (2005) — DCA review
  Hettler et al., PRB 58, R7475 (1998) — original DCA
  Jarrell et al., PRB 64, 195130 (2001) — DCA formalism
  Maier et al., PRL 95, 237001 (2005) — d-wave pairing in DCA
"""

import numpy as np
import json
import os
import time
from typing import Optional


# ── Cluster geometry ─────────────────────────────────────────────────────────

def generate_cluster_momenta(nc: int, dim: int = 2) -> np.ndarray:
    """
    Generate cluster momenta K for a DCA cluster of size N_c.

    For N_c = 4 (2×2 in 2D):
      K = {(0,0), (π,0), (0,π), (π,π)}

    For N_c = 8 (2×2×2 in 3D or tilted 2D):
      More complex tiling; we use the standard square cluster.

    Args:
        nc: number of cluster sites (must be a perfect power: 4, 8, 16, ...)
        dim: spatial dimension (2 or 3)

    Returns:
        K_points: [nc, dim] array of cluster momenta in units of π/a
                  (i.e., (1,0) means (π/a, 0))
    """
    if dim == 2:
        if nc == 4:
            # 2×2 plaquette
            return np.array([
                [0.0, 0.0],   # Γ
                [1.0, 0.0],   # X = (π, 0)
                [0.0, 1.0],   # Y = (0, π)
                [1.0, 1.0],   # M = (π, π)
            ])
        elif nc == 8:
            # 2√2 × 2√2 tilted cluster (Betts cluster)
            return np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [1.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5],
            ])
        elif nc == 16:
            # 4×4
            pts = []
            for i in range(4):
                for j in range(4):
                    pts.append([i * 0.5, j * 0.5])
            return np.array(pts)
        else:
            raise ValueError(f"Unsupported N_c={nc} for 2D DCA")
    elif dim == 3:
        if nc == 4:
            # Simple 2×2×1 slab
            return np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ])
        else:
            raise ValueError(f"Unsupported N_c={nc} for 3D DCA")

    raise ValueError(f"Unsupported dim={dim}")


def assign_k_to_patches(
    kpoints: np.ndarray,
    K_cluster: np.ndarray,
) -> np.ndarray:
    """
    Assign each lattice k-point to its nearest cluster momentum K.

    In DCA, the BZ is partitioned into N_c Voronoi cells centered on the
    cluster momenta. Each k is assigned to the patch of its nearest K
    (with periodic boundary conditions on the BZ).

    Args:
        kpoints: [n_k, dim] lattice k-points in units of π/a
        K_cluster: [nc, dim] cluster momenta in units of π/a

    Returns:
        patch_assignment: [n_k] int array, mapping k-index → cluster-index
    """
    n_k = kpoints.shape[0]
    nc = K_cluster.shape[0]
    dim = kpoints.shape[1]

    # BZ periodicity: 2 in units of π/a (i.e., BZ is [-1,1] in π/a units)
    bz_period = 2.0

    assignment = np.zeros(n_k, dtype=int)

    for ik in range(n_k):
        min_dist = np.inf
        for ic in range(nc):
            # Distance with periodic BCs
            dk = kpoints[ik] - K_cluster[ic]
            # Wrap to [-1, 1] in π/a units
            dk = dk - bz_period * np.round(dk / bz_period)
            dist = np.sum(dk ** 2)
            if dist < min_dist:
                min_dist = dist
                assignment[ik] = ic

    return assignment


# ── Coarse-grained Green's function ─────────────────────────────────────────

def coarse_grain_gf(
    gk_iw: np.ndarray,
    patch_assignment: np.ndarray,
    nc: int,
) -> np.ndarray:
    """
    Coarse-grain the lattice Green's function over DCA patches:

      Ḡ(K, iω) = (1/N_patch) Σ_{k∈patch(K)} G(k, iω)

    Args:
        gk_iw: lattice G(k, iω), shape [n_k, n_iw] or [n_k, n_iw, n_orb, n_orb]
        patch_assignment: [n_k] patch indices
        nc: number of cluster sites

    Returns:
        g_bar: coarse-grained G, shape [nc, n_iw] or [nc, n_iw, n_orb, n_orb]
    """
    shape_rest = gk_iw.shape[1:]
    g_bar = np.zeros((nc,) + shape_rest, dtype=complex)
    counts = np.zeros(nc, dtype=int)

    for ik in range(gk_iw.shape[0]):
        ic = patch_assignment[ik]
        g_bar[ic] += gk_iw[ik]
        counts[ic] += 1

    for ic in range(nc):
        if counts[ic] > 0:
            g_bar[ic] /= counts[ic]

    return g_bar


# ── DCA self-consistency ─────────────────────────────────────────────────────

class DCAParams:
    """Parameters for a DCA calculation."""

    def __init__(
        self,
        nc: int = 4,
        dim: int = 2,
        beta: float = 40.0,
        n_iw: int = 256,
        n_k_per_dim: int = 64,
        U: float = 0.0,
        J: float = 0.0,
        mu: float = 0.0,
        n_iter: int = 30,
        sigma_mix: float = 0.5,
        convergence_tol: float = 1e-4,
        # CTHYB solver params
        n_warmup: int = 50000,
        n_cycles: int = 5_000_000,
        length_cycle: int = 200,
    ):
        self.nc = nc
        self.dim = dim
        self.beta = beta
        self.n_iw = n_iw
        self.n_k_per_dim = n_k_per_dim
        self.U = U
        self.J = J
        self.mu = mu
        self.n_iter = n_iter
        self.sigma_mix = sigma_mix
        self.convergence_tol = convergence_tol
        self.n_warmup = n_warmup
        self.n_cycles = n_cycles
        self.length_cycle = length_cycle

        # Derived
        self.K_cluster = generate_cluster_momenta(nc, dim)
        self.wn = np.array([
            (2 * (n - n_iw) + 1) * np.pi / beta for n in range(2 * n_iw)
        ])


def build_lattice_kpoints(n_per_dim: int, dim: int = 2) -> np.ndarray:
    """
    Generate a uniform lattice k-mesh in units of π/a.

    For a 2D lattice with n_per_dim=64:
      k ∈ [-1, 1) × [-1, 1) in units of π/a → 4096 points.
    """
    grid_1d = np.linspace(-1.0, 1.0, n_per_dim, endpoint=False)

    if dim == 2:
        kx, ky = np.meshgrid(grid_1d, grid_1d, indexing="ij")
        return np.column_stack([kx.ravel(), ky.ravel()])
    elif dim == 3:
        kx, ky, kz = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
        return np.column_stack([kx.ravel(), ky.ravel(), kz.ravel()])
    else:
        raise ValueError(f"Unsupported dim={dim}")


def lattice_dispersion_hubbard(
    kpoints: np.ndarray,
    t: float = 1.0,
    tp: float = 0.0,
) -> np.ndarray:
    """
    Tight-binding dispersion for the 2D square lattice Hubbard model:

      ε(k) = -2t [cos(kx·π) + cos(ky·π)] - 4t' cos(kx·π) cos(ky·π)

    Note: kpoints are in units of π/a, so cos(kx·π) = cos(kx * π).

    Args:
        kpoints: [n_k, 2] in units of π/a
        t: nearest-neighbor hopping
        tp: next-nearest-neighbor hopping

    Returns:
        eps_k: [n_k] dispersion values
    """
    kx = kpoints[:, 0] * np.pi
    ky = kpoints[:, 1] * np.pi
    return -2.0 * t * (np.cos(kx) + np.cos(ky)) - 4.0 * tp * np.cos(kx) * np.cos(ky)


def compute_lattice_gf_dca(
    eps_k: np.ndarray,
    sigma_c: np.ndarray,
    patch_assignment: np.ndarray,
    mu: float,
    wn: np.ndarray,
) -> np.ndarray:
    """
    Compute single-band lattice G(k, iω) using the DCA self-energy:

      G(k, iω_n) = 1 / [iω_n + μ - ε(k) - Σ_c(K(k), iω_n)]

    where K(k) is the cluster momentum assigned to k.

    Args:
        eps_k: [n_k] lattice dispersion
        sigma_c: [nc, 2*n_iw] cluster self-energy
        patch_assignment: [n_k] patch assignment
        mu: chemical potential
        wn: [2*n_iw] Matsubara frequencies

    Returns:
        gk_iw: [n_k, 2*n_iw] lattice Green's function
    """
    n_k = eps_k.shape[0]
    n_w = wn.shape[0]

    gk_iw = np.zeros((n_k, n_w), dtype=complex)

    for iw in range(n_w):
        iw_plus_mu = 1j * wn[iw] + mu
        for ik in range(n_k):
            ic = patch_assignment[ik]
            gk_iw[ik, iw] = 1.0 / (iw_plus_mu - eps_k[ik] - sigma_c[ic, iw])

    return gk_iw


def compute_lattice_gf_dca_fast(
    eps_k: np.ndarray,
    sigma_c: np.ndarray,
    patch_assignment: np.ndarray,
    mu: float,
    wn: np.ndarray,
) -> np.ndarray:
    """Vectorized version of DCA lattice Green's function."""
    n_k = eps_k.shape[0]
    n_w = wn.shape[0]

    # sigma_c at each k-point: sigma_k[ik, iw] = sigma_c[patch[ik], iw]
    sigma_k = sigma_c[patch_assignment, :]  # [n_k, n_w]

    # G(k, iω) = 1 / (iω + μ - ε_k - Σ(K(k), iω))
    iw_grid = 1j * wn[np.newaxis, :] + mu  # [1, n_w]
    eps_grid = eps_k[:, np.newaxis]          # [n_k, 1]

    return 1.0 / (iw_grid - eps_grid - sigma_k)


def solve_cluster_cthyb(
    g0_iw_cluster: np.ndarray,
    params: DCAParams,
    work_dir: str,
    iteration: int,
) -> dict:
    """
    Solve the N_c-site cluster problem using TRIQS/CTHYB.

    The cluster impurity problem:
      - Bath Green's function G⁰_c(K, iω) defines the hybridization
      - On-site interaction U on each cluster site
      - CTHYB solves for the interacting cluster G_c(K, iω)

    For single-band Hubbard (the benchmark case), the cluster has
    N_c orbitals with intra-orbital U and no J.

    Args:
        g0_iw_cluster: [nc, 2*n_iw] bare cluster Green's function in K-space
        params: DCA parameters
        work_dir: working directory for this iteration
        iteration: iteration number

    Returns:
        dict with:
          g_c: [nc, 2*n_iw] interacting cluster Green's function
          avg_sign: float, average QMC sign (1.0 = no sign problem)
    """
    try:
        return _solve_cluster_triqs(g0_iw_cluster, params, work_dir, iteration)
    except ImportError:
        print("[DCA] TRIQS not available, using Hubbard-I atomic limit as fallback")
        g_c = _solve_cluster_hubbard_i(g0_iw_cluster, params)
        return {"g_c": g_c, "avg_sign": 1.0}


def _solve_cluster_triqs(
    g0_iw_cluster: np.ndarray,
    params: DCAParams,
    work_dir: str,
    iteration: int,
) -> np.ndarray:
    """
    TRIQS/CTHYB cluster solver.

    Converts the DCA cluster problem into the format CTHYB expects:
      - Block structure: "up" and "down" blocks, each of size N_c
      - G⁰(iω) as a TRIQS BlockGf
      - Interaction Hamiltonian: U Σ_i n_{i↑} n_{i↓}
    """
    from triqs.gf import GfImFreq, BlockGf, MeshImFreq, inverse
    from triqs.operators import c, c_dag, n
    from triqs_cthyb import Solver

    nc = params.nc
    n_iw = params.n_iw
    beta = params.beta

    # Create the CTHYB solver with cluster block structure
    # For single-band Hubbard: "up" block of size N_c, "down" block of size N_c
    gf_struct = [("up", nc), ("down", nc)]
    S = Solver(
        beta=beta,
        gf_struct=gf_struct,
        n_iw=n_iw,
        n_tau=10001,
    )

    # Fill G⁰(iω) from the DCA coarse-grained input
    # G⁰ is diagonal in the cluster momentum K basis (DCA preserves translation)
    for spin in ["up", "down"]:
        for i in range(nc):
            for iw_idx, iw_val in enumerate(S.G0_iw[spin].mesh):
                # Map TRIQS mesh index to our array index
                n_triqs = iw_idx
                if n_triqs < 2 * n_iw:
                    S.G0_iw[spin][iw_val][i, i] = g0_iw_cluster[i, n_triqs]

    # Build the interaction Hamiltonian: H_int = U Σ_i n_{i↑} n_{i↓}
    H_int = 0
    for i in range(nc):
        H_int += params.U * n("up", i) * n("down", i)

    # Solve
    S.solve(
        h_int=H_int,
        n_warmup_cycles=params.n_warmup,
        n_cycles=params.n_cycles // 20,  # per-rank; total = this × MPI_ranks
        length_cycle=params.length_cycle,
        move_double=True,
        measure_density_matrix=True,
    )

    # Extract G_c(K, iω) from solver output
    g_c = np.zeros((nc, 2 * n_iw), dtype=complex)
    for i in range(nc):
        for iw_idx, iw_val in enumerate(S.G_iw["up"].mesh):
            if iw_idx < 2 * n_iw:
                g_c[i, iw_idx] = S.G_iw["up"][iw_val][i, i]

    # Extract average sign
    avg_sign = 1.0
    try:
        avg_sign = float(S.average_sign)
    except Exception:
        pass

    # Save solver output for diagnostics
    iter_dir = os.path.join(work_dir, f"iter_{iteration:03d}")
    os.makedirs(iter_dir, exist_ok=True)
    np.savez(
        os.path.join(iter_dir, "cluster_gf.npz"),
        g_c=g_c,
        g0_c=g0_iw_cluster,
        avg_sign=avg_sign,
        density_up=np.array([S.G_iw["up"].density()[i, i].real for i in range(nc)]),
        density_dn=np.array([S.G_iw["down"].density()[i, i].real for i in range(nc)]),
    )

    return {"g_c": g_c, "avg_sign": avg_sign}


def _solve_cluster_hubbard_i(
    g0_iw_cluster: np.ndarray,
    params: DCAParams,
) -> np.ndarray:
    """
    Hubbard-I (atomic limit) approximation for the cluster.
    Used as fallback when TRIQS is not available.

    Σ_HI(iω) = U·n / (1 - U·n / (iω))  (simplified)

    This won't capture d-wave pairing but provides a self-consistent
    test of the DCA loop machinery.
    """
    nc = params.nc
    n_w = g0_iw_cluster.shape[1]

    # Rough estimate of filling from G⁰
    filling = 0.5  # half-filling approximation for fallback

    sigma_atomic = np.zeros((nc, n_w), dtype=complex)
    for ic in range(nc):
        for iw in range(n_w):
            w = params.wn[iw]
            # Hubbard-I self-energy
            sigma_atomic[ic, iw] = params.U * filling + (
                params.U ** 2 * filling * (1 - filling) / (1j * w)
            )

    # G_c = (G⁰⁻¹ - Σ)⁻¹
    g_c = np.zeros_like(g0_iw_cluster)
    for ic in range(nc):
        for iw in range(n_w):
            g0_inv = 1.0 / g0_iw_cluster[ic, iw] if abs(g0_iw_cluster[ic, iw]) > 1e-20 else 0.0
            g_c[ic, iw] = 1.0 / (g0_inv - sigma_atomic[ic, iw])

    return g_c


# ── DCA self-consistency loop ────────────────────────────────────────────────

def run_dca(
    eps_k: np.ndarray,
    kpoints: np.ndarray,
    params: DCAParams,
    work_dir: str,
    initial_sigma: Optional[np.ndarray] = None,
) -> dict:
    """
    Run the full DCA self-consistency loop.

    Args:
        eps_k: [n_k] lattice dispersion
        kpoints: [n_k, dim] k-points in π/a units
        params: DCA parameters
        work_dir: output directory
        initial_sigma: [nc, 2*n_iw] initial self-energy (None = zero)

    Returns:
        dict with converged sigma, Green's functions, convergence history
    """
    os.makedirs(work_dir, exist_ok=True)
    nc = params.nc
    n_w = 2 * params.n_iw
    wn = params.wn

    t0 = time.time()

    # Assign k-points to DCA patches
    patch_assignment = assign_k_to_patches(kpoints, params.K_cluster)
    patch_counts = np.bincount(patch_assignment, minlength=nc)
    print(f"[DCA] N_c={nc}, n_k={len(eps_k)}, patches: {patch_counts}")
    print(f"[DCA] Cluster momenta (π/a): {params.K_cluster.tolist()}")

    # Initialize self-energy
    sigma_c = initial_sigma if initial_sigma is not None else np.zeros((nc, n_w), dtype=complex)

    convergence_history = []
    density_history = []
    sign_history = []

    for iteration in range(params.n_iter):
        t_iter = time.time()

        # Step 1: Lattice Green's function with current Σ_c
        gk_iw = compute_lattice_gf_dca_fast(eps_k, sigma_c, patch_assignment, params.mu, wn)

        # Step 2: Coarse-grain → Ḡ(K, iω)
        g_bar = coarse_grain_gf(gk_iw, patch_assignment, nc)

        # Step 3: Cluster bare G⁰ via Dyson equation
        #   G⁰_c(K, iω) = [Ḡ(K, iω)⁻¹ + Σ_c(K, iω)]⁻¹
        g0_c = np.zeros((nc, n_w), dtype=complex)
        for ic in range(nc):
            for iw in range(n_w):
                g_bar_inv = 1.0 / g_bar[ic, iw] if abs(g_bar[ic, iw]) > 1e-20 else 0.0
                g0_c[ic, iw] = 1.0 / (g_bar_inv + sigma_c[ic, iw])

        # Step 4: Solve cluster problem
        solver_result = solve_cluster_cthyb(g0_c, params, work_dir, iteration)
        g_c = solver_result["g_c"]
        avg_sign = solver_result["avg_sign"]
        sign_history.append(float(avg_sign))

        # Check sign problem — abort early if fatal
        if avg_sign < 0.05:
            print(f"[DCA] FATAL: average sign {avg_sign:.4f} < 0.05 — aborting DCA loop")
            break
        elif avg_sign < 0.2:
            print(f"[DCA] WARNING: low average sign {avg_sign:.4f}")

        # Step 5: Extract new Σ via Dyson
        sigma_new = np.zeros((nc, n_w), dtype=complex)
        for ic in range(nc):
            for iw in range(n_w):
                g0_inv = 1.0 / g0_c[ic, iw] if abs(g0_c[ic, iw]) > 1e-20 else 0.0
                gc_inv = 1.0 / g_c[ic, iw] if abs(g_c[ic, iw]) > 1e-20 else 0.0
                sigma_new[ic, iw] = g0_inv - gc_inv

        # Check for NaN/Inf in new self-energy
        if np.any(np.isnan(sigma_new)) or np.any(np.isinf(sigma_new)):
            print(f"[DCA] FATAL: NaN/Inf in self-energy at iteration {iteration+1}")
            break

        # Step 6: Mix
        sigma_mixed = params.sigma_mix * sigma_c + (1.0 - params.sigma_mix) * sigma_new

        # Step 7: Check convergence
        diff = np.max(np.abs(sigma_mixed - sigma_c))
        sigma_c = sigma_mixed

        # Compute density (with tail correction)
        density = 0.0
        for ic in range(nc):
            density += 1.0 + (2.0 / params.beta) * np.sum(g_c[ic].real)
        density *= 2.0 / nc

        elapsed_iter = time.time() - t_iter
        convergence_history.append(float(diff))
        density_history.append(float(density))

        print(f"[DCA] iter {iteration+1}/{params.n_iter}: "
              f"||ΔΣ||={diff:.2e}, <n>={density:.4f}, "
              f"<sign>={avg_sign:.4f}, time={elapsed_iter:.1f}s")

        if diff < params.convergence_tol and iteration > 2:
            print(f"[DCA] Converged after {iteration+1} iterations")
            break

    elapsed = time.time() - t0

    # Save final state
    np.savez(
        os.path.join(work_dir, "dca_converged.npz"),
        sigma_c=sigma_c,
        g_bar=g_bar,
        g_c=g_c,
        g0_c=g0_c,
        K_cluster=params.K_cluster,
        patch_assignment=patch_assignment,
        wn=wn,
        sign_history=np.array(sign_history),
    )

    converged = convergence_history[-1] < params.convergence_tol if convergence_history else False
    sign_problem = len(sign_history) > 0 and sign_history[-1] < 0.05

    return {
        "converged": converged,
        "sign_problem": sign_problem,
        "n_iterations": len(convergence_history),
        "final_diff": convergence_history[-1] if convergence_history else None,
        "density": density_history[-1] if density_history else None,
        "avg_sign": sign_history[-1] if sign_history else 1.0,
        "convergence_history": convergence_history,
        "density_history": density_history,
        "sign_history": sign_history,
        "elapsed_seconds": elapsed,
        "sigma_c": sigma_c,
        "g_bar": g_bar,
        "g_c": g_c,
        "g0_c": g0_c,
        "K_cluster": params.K_cluster,
        "patch_assignment": patch_assignment,
    }


# ── DCA pairing susceptibility ───────────────────────────────────────────────

def compute_dca_pairing_eigenvalue(
    dca_result: dict,
    params: DCAParams,
    eps_k: np.ndarray,
    kpoints: np.ndarray,
    channel: str = "singlet",
) -> dict:
    """
    Compute the DCA pairing eigenvalue from the converged cluster solution.

    In DCA, the pairing susceptibility is computed in the cluster momentum
    basis. The particle-particle bubble:

      χ⁰_pp(K, iν) = -(1/N_k) Σ_{k∈patch(K)} G(k, iν) G(-k, -iν)

    The pairing eigenvalue equation:
      λ_d · φ(K) = -Σ_{K'} Γ_pp(K, K') · χ⁰_pp(K') · φ(K')

    For d-wave: φ_d(K) = cos(Kx) - cos(Ky) is the expected eigenvector.

    For the 2×2 cluster:
      φ_d(Γ) = cos(0)-cos(0) = 0
      φ_d(X) = cos(π)-cos(0) = -2
      φ_d(Y) = cos(0)-cos(π) = +2
      φ_d(M) = cos(π)-cos(π) = 0

    So d-wave lives entirely on the X and Y patches (the antinodal regions).
    """
    nc = params.nc
    n_w = 2 * params.n_iw
    wn = params.wn
    beta = params.beta
    sigma_c = dca_result["sigma_c"]
    patch_assignment = dca_result["patch_assignment"]

    # Build the -k map
    from pairing_susceptibility import _build_minus_k_map
    minus_k_map = _build_minus_k_map(kpoints)

    # Compute particle-particle bubble χ⁰_pp(K, iν)
    # χ⁰_pp(K, iν) = -(T/N_K) Σ_{k∈patch(K)} G(k,iν) · G(-k,-iν)
    gk_iw = compute_lattice_gf_dca_fast(eps_k, sigma_c, patch_assignment, params.mu, wn)

    chi0_pp = np.zeros((nc, n_w), dtype=complex)
    T = 1.0 / beta

    for ik in range(len(eps_k)):
        ic = patch_assignment[ik]
        ik_minus = minus_k_map[ik]
        for iw in range(n_w):
            iw_minus = n_w - 1 - iw  # -ν index
            chi0_pp[ic, iw] += -T * gk_iw[ik, iw] * gk_iw[ik_minus, iw_minus]

    # Normalize by patch size
    patch_counts = np.bincount(patch_assignment, minlength=nc).astype(float)
    for ic in range(nc):
        if patch_counts[ic] > 0:
            chi0_pp[ic] /= patch_counts[ic]

    # Sum over Matsubara frequencies for the static (ν-summed) susceptibility
    # This gives the leading contribution at Ω=0
    chi0_pp_static = np.sum(chi0_pp, axis=1).real  # [nc]

    # For the bare vertex, use Γ_pp = U (contact interaction in Hubbard model)
    # The eigenvalue equation becomes:
    #   λ · φ(K) = -U · Σ_{K'} χ⁰_pp(K') · φ(K') / N_c
    # For d-wave: project onto cos(Kx)-cos(Ky)

    # d-wave form factor on cluster momenta
    Kx = params.K_cluster[:, 0] * np.pi
    Ky = params.K_cluster[:, 1] * np.pi
    phi_d = np.cos(Kx) - np.cos(Ky)

    # s-wave form factor
    phi_s = np.ones(nc)

    # Extended s-wave
    phi_sext = np.cos(Kx) + np.cos(Ky)

    # Compute eigenvalues for each channel
    # For a general vertex matrix Γ(K,K'), the eigenvalue problem is:
    #   λ · φ(K) = -Σ_{K'} Γ(K,K') · χ⁰_pp(K') · φ(K') / N_c
    #
    # With contact U: Γ(K,K') = U for all K,K'
    # So: λ = -U · Σ_K χ⁰_pp(K) · φ(K)² / (N_c · Σ_K φ(K)²)
    #
    # But for the CLUSTER vertex (which we get from G²), the vertex
    # IS K-dependent. For the benchmark without G², we use the bare U.

    results = {}

    for name, phi in [("d-x2y2", phi_d), ("s-wave", phi_s), ("s-ext", phi_sext)]:
        phi_norm_sq = np.sum(phi ** 2)
        if phi_norm_sq < 1e-10:
            results[name] = {"lambda": 0.0, "phi": phi.tolist()}
            continue

        # λ = -(U/N_c) · Σ_K χ⁰_pp(K) · φ(K)² / Σ_K φ(K)²
        lambda_val = -(params.U / nc) * np.sum(chi0_pp_static * phi ** 2) / phi_norm_sq
        results[name] = {
            "lambda": float(lambda_val),
            "phi": phi.tolist(),
            "chi0_pp_static": chi0_pp_static.tolist(),
        }

    # For the full frequency-dependent version (more accurate):
    # Build N_c × N_c matrix: M_{KK'} = -(U/N_c) · χ⁰_pp(K') · δ_{KK'} (for contact U)
    # Eigenvalues of this matrix give all pairing channels simultaneously
    M_pair = np.zeros((nc, nc), dtype=complex)
    for ic in range(nc):
        M_pair[ic, ic] = -(params.U / nc) * np.sum(chi0_pp[ic])  # sum over ν

    evals = np.linalg.eigvals(M_pair)
    evals_sorted = np.sort(evals.real)[::-1]

    results["eigenvalues_all"] = evals_sorted.tolist()
    results["lambda_leading"] = float(evals_sorted[0]) if len(evals_sorted) > 0 else 0.0

    # Identify which channel the leading eigenvalue corresponds to
    if abs(results.get("d-x2y2", {}).get("lambda", 0)) > abs(results.get("s-wave", {}).get("lambda", 0)):
        results["dominant_channel"] = "d-x2y2"
    else:
        results["dominant_channel"] = "s-wave"

    results["beta"] = beta
    results["temperature_K"] = 11604.5 / beta
    results["nc"] = nc
    results["U"] = params.U
    results["filling"] = dca_result.get("density", None)

    return results
