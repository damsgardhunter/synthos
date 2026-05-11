#!/usr/bin/env python3
"""
D1: Multi-Orbital DCA Cluster Solver.

Extends the single-band DCA (Stage C) to multi-orbital systems:
  - 3-band Emery model: Cu-d_{x²-y²} + O-p_x + O-p_y (cuprates)
  - 5-band d-shell: all five d orbitals (pnictides, nickelates)
  - Arbitrary n_orb from Wannier downfolding (real materials)

The key complications vs single-band:
  1. Sign problem worsens exponentially: cost ~ exp(β·U·n_orb·N_c)
     For 3-band, 4-site cluster: ~100× more expensive than single-band
  2. Interaction is Kanamori (U, U', J) not just Hubbard U
  3. Green's function is matrix-valued: G(K,iω)_{ab} for orbitals a,b
  4. Off-diagonal hybridization between orbitals within a cluster site

The DCA self-consistency loop is the same as single-band but with
matrix-valued quantities at every step:
  G(k,iω) → [iω+μ - H(k) - Σ(K(k),iω)]⁻¹    (matrix inversion)
  Ḡ(K,iω) → patch-average of G(k,iω)           (matrix average)
  G⁰_c(K,iω) → [Ḡ⁻¹ + Σ]⁻¹                    (matrix Dyson)

References:
  Maier et al., PRB 74, 094513 (2006) — multi-orbital DCA
  Werner et al., PRL 97, 076405 (2006) — CTHYB for multi-orbital
  Gull et al., PRB 82, 155101 (2010) — sign problem in cluster QMC
  Emery, PRL 58, 2794 (1987) — 3-band CuO₂ model
"""

import numpy as np
import json
import os
import time
from typing import Optional, List, Dict

from dca_solver import (
    DCAParams,
    generate_cluster_momenta,
    assign_k_to_patches,
    coarse_grain_gf,
    build_lattice_kpoints,
)


# ── Multi-orbital Hamiltonian builders ───────────────────────────────────────

def build_emery_hamiltonian(
    kpoints: np.ndarray,
    tpd: float = 1.3,
    tpp: float = 0.65,
    ep_minus_ed: float = 3.6,
) -> np.ndarray:
    """
    Build the 3-band Emery (Cu-O) Hamiltonian H(k) for the CuO₂ plane.

    Basis: |d_{x²-y²}⟩, |p_x⟩, |p_y⟩  (one Cu + two O per unit cell)

    H(k) = | ε_d           2it_{pd}sin(kx/2)    -2it_{pd}sin(ky/2)  |
            | -2it_{pd}sin(kx/2)    ε_p         -4t_{pp}sin(kx/2)sin(ky/2) |
            | 2it_{pd}sin(ky/2)  -4t_{pp}sin(kx/2)sin(ky/2)   ε_p   |

    where ε_d = 0, ε_p = Δ_{pd} = ε_p - ε_d.

    Standard parameters (eV, from Andersen et al. 1995):
      t_{pd} = 1.3 eV (Cu-O hopping)
      t_{pp} = 0.65 eV (O-O hopping)
      Δ_{pd} = 3.6 eV (charge-transfer energy)

    Args:
        kpoints: [n_k, 2] in units of π/a
        tpd: Cu-O hopping (eV)
        tpp: O-O hopping (eV)
        ep_minus_ed: charge-transfer energy Δ_{pd} (eV)

    Returns:
        hk: [n_k, 3, 3] Hamiltonian matrices
    """
    n_k = kpoints.shape[0]
    hk = np.zeros((n_k, 3, 3), dtype=complex)

    kx = kpoints[:, 0] * np.pi  # Convert to radians
    ky = kpoints[:, 1] * np.pi

    sx = np.sin(kx / 2)
    sy = np.sin(ky / 2)

    # Diagonal: on-site energies
    hk[:, 0, 0] = 0.0               # ε_d (set as energy zero)
    hk[:, 1, 1] = ep_minus_ed        # ε_p (O-px)
    hk[:, 2, 2] = ep_minus_ed        # ε_p (O-py)

    # Cu-O hybridization
    hk[:, 0, 1] = 2j * tpd * sx     # d - px
    hk[:, 1, 0] = -2j * tpd * sx    # px - d (hermitian conjugate)
    hk[:, 0, 2] = -2j * tpd * sy    # d - py
    hk[:, 2, 0] = 2j * tpd * sy     # py - d

    # O-O hybridization
    hk[:, 1, 2] = -4 * tpp * sx * sy  # px - py
    hk[:, 2, 1] = -4 * tpp * sx * sy  # py - px

    return hk


def build_multiorbital_hamiltonian_from_hr(
    hr_data: np.ndarray,
    kpoints: np.ndarray,
    n_orb: int,
    degeneracies: np.ndarray,
) -> np.ndarray:
    """
    Build H(k) from Wannier90 _hr.dat data for arbitrary orbital count.

    This is the general-purpose builder for real materials from the
    DMFT bundle.

    Args:
        hr_data: [n_entries, 7] — Rx, Ry, Rz, i, j, Re(H), Im(H)
        kpoints: [n_k, dim] in fractional coordinates
        n_orb: number of Wannier orbitals
        degeneracies: [n_rpts] R-vector degeneracies

    Returns:
        hk: [n_k, n_orb, n_orb] complex Hamiltonian
    """
    n_k = kpoints.shape[0]
    hk = np.zeros((n_k, n_orb, n_orb), dtype=complex)

    # Group entries by R-vector
    r_vecs = {}
    for row in hr_data:
        rx, ry, rz = int(row[0]), int(row[1]), int(row[2])
        i, j = int(row[3]) - 1, int(row[4]) - 1  # 1-based → 0-based
        val = complex(row[5], row[6])
        key = (rx, ry, rz)
        if key not in r_vecs:
            r_vecs[key] = []
        r_vecs[key].append((i, j, val))

    r_list = sorted(r_vecs.keys())

    # Fourier transform
    for ik in range(n_k):
        for ri, R in enumerate(r_list):
            phase = np.exp(2j * np.pi * np.dot(kpoints[ik], R))
            deg = degeneracies[ri] if ri < len(degeneracies) else 1
            for (i, j, val) in r_vecs[R]:
                if i < n_orb and j < n_orb:
                    hk[ik, i, j] += phase * val / deg

    return hk


# ── Multi-orbital interaction ────────────────────────────────────────────────

class KanamoriInteraction:
    """
    Kanamori interaction for multi-orbital systems.

    H_int = U Σ_a n_{a↑} n_{a↓}                              (intra-orbital)
          + U' Σ_{a≠b} n_{a↑} n_{b↓}                          (inter-orbital, opposite spin)
          + (U'-J) Σ_{a<b,σ} n_{aσ} n_{bσ}                   (inter-orbital, same spin)
          - J Σ_{a≠b} c†_{a↑} c_{a↓} c†_{b↓} c_{b↑}         (spin-flip)
          + J Σ_{a≠b} c†_{a↑} c†_{a↓} c_{b↓} c_{b↑}         (pair-hopping)

    Rotationally invariant: U' = U - 2J (Kanamori constraint).

    For the 3-band Emery model:
      U_d ~ 8-10 eV  (on Cu-d)
      U_p ~ 4-6 eV   (on O-p, often neglected)
      J_d ~ 0.9-1.0 eV (Hund's on Cu)
      Charge-transfer: Δ_{pd} ~ 3.6 eV

    References:
      Kanamori, Prog. Theor. Phys. 30, 275 (1963)
      Georges et al., ARCMP 4, 137 (2013) — Hund's metals
    """

    def __init__(
        self,
        n_orb: int,
        U: np.ndarray,
        J: np.ndarray,
        orbital_labels: Optional[List[str]] = None,
    ):
        """
        Args:
            n_orb: number of orbitals
            U: [n_orb] intra-orbital Coulomb repulsion per orbital (eV)
            J: [n_orb] Hund's coupling per orbital (eV)
                For inter-orbital pairs, we use the average J.
            orbital_labels: e.g., ["d_x2y2", "p_x", "p_y"]
        """
        self.n_orb = n_orb
        self.U = np.asarray(U, dtype=float)
        self.J = np.asarray(J, dtype=float)
        self.orbital_labels = orbital_labels or [f"orb_{i}" for i in range(n_orb)]

        # Inter-orbital U' = U - 2J (from Kanamori constraint)
        # For different orbitals with different U, use geometric mean
        self.U_prime = np.zeros((n_orb, n_orb))
        self.J_pair = np.zeros((n_orb, n_orb))
        for a in range(n_orb):
            for b in range(n_orb):
                if a != b:
                    U_avg = np.sqrt(self.U[a] * self.U[b])
                    J_avg = (self.J[a] + self.J[b]) / 2
                    self.U_prime[a, b] = U_avg - 2 * J_avg
                    self.J_pair[a, b] = J_avg

    def build_triqs_h_int(self):
        """
        Build the TRIQS operator for H_int.

        Returns a TRIQS Operator object for use with CTHYB.
        """
        from triqs.operators import c, c_dag, n, Operator

        H = Operator()
        no = self.n_orb

        for a in range(no):
            # Intra-orbital: U_a n_{a↑} n_{a↓}
            H += self.U[a] * n("up", a) * n("down", a)

        for a in range(no):
            for b in range(no):
                if a == b:
                    continue
                Uprime = self.U_prime[a, b]
                Jpair = self.J_pair[a, b]

                # Inter-orbital opposite spin: U' n_{a↑} n_{b↓}
                H += Uprime * n("up", a) * n("down", b)

                # Inter-orbital same spin: (U'-J) n_{aσ} n_{bσ} for a<b
                if a < b:
                    H += (Uprime - Jpair) * n("up", a) * n("up", b)
                    H += (Uprime - Jpair) * n("down", a) * n("down", b)

                # Spin-flip: -J c†_{a↑} c_{a↓} c†_{b↓} c_{b↑}
                H += -Jpair * c_dag("up", a) * c("down", a) * c_dag("down", b) * c("up", b)

                # Pair-hopping: J c†_{a↑} c†_{a↓} c_{b↓} c_{b↑}
                # Sum over all a≠b (not a<b) — same convention as spin-flip.
                # Georges et al. ARCMP 4, 137 (2013) Eq. 1 sums over all a≠b.
                H += Jpair * c_dag("up", a) * c_dag("down", a) * c("down", b) * c("up", b)

        return H

    def build_density_density(self):
        """
        Build density-density-only interaction (no spin-flip, no pair-hop).

        Less accurate but dramatically reduces the sign problem.
        Used when full Kanamori makes the QMC intractable.
        """
        from triqs.operators import n, Operator

        H = Operator()
        no = self.n_orb

        for a in range(no):
            H += self.U[a] * n("up", a) * n("down", a)

        for a in range(no):
            for b in range(no):
                if a == b:
                    continue
                Uprime = self.U_prime[a, b]
                Jpair = self.J_pair[a, b]
                H += Uprime * n("up", a) * n("down", b)
                if a < b:
                    H += (Uprime - Jpair) * n("up", a) * n("up", b)
                    H += (Uprime - Jpair) * n("down", a) * n("down", b)

        return H


# ── Multi-orbital DCA solver ────────────────────────────────────────────────

class MultiOrbitalDCAParams(DCAParams):
    """Extended DCA parameters for multi-orbital systems."""

    def __init__(
        self,
        n_orb: int = 3,
        interaction: Optional[KanamoriInteraction] = None,
        use_density_density: bool = False,
        correlated_orbitals: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_orb = n_orb
        self.interaction = interaction
        self.use_density_density = use_density_density
        # Which orbitals get the interaction (e.g., only Cu-d in 3-band)
        self.correlated_orbitals = correlated_orbitals or list(range(n_orb))


def compute_lattice_gf_multiorbital(
    hk: np.ndarray,
    sigma_c: np.ndarray,
    patch_assignment: np.ndarray,
    mu: float,
    wn: np.ndarray,
) -> np.ndarray:
    """
    Compute multi-orbital lattice G(k, iω) with DCA self-energy:

      G(k, iω)_{ab} = [(iω+μ)·δ_{ab} - H(k)_{ab} - Σ_c(K(k), iω)_{ab}]⁻¹

    Args:
        hk: [n_k, n_orb, n_orb] lattice Hamiltonian
        sigma_c: [nc, 2*n_iw, n_orb, n_orb] cluster self-energy
        patch_assignment: [n_k] patch indices
        mu: chemical potential
        wn: [2*n_iw] Matsubara frequencies

    Returns:
        gk_iw: [n_k, 2*n_iw, n_orb, n_orb] lattice Green's function
    """
    n_k = hk.shape[0]
    n_orb = hk.shape[1]
    n_w = wn.shape[0]
    eye = np.eye(n_orb, dtype=complex)

    gk_iw = np.zeros((n_k, n_w, n_orb, n_orb), dtype=complex)

    for iw in range(n_w):
        iw_mu = (1j * wn[iw] + mu) * eye
        for ik in range(n_k):
            ic = patch_assignment[ik]
            M = iw_mu - hk[ik] - sigma_c[ic, iw]
            gk_iw[ik, iw] = np.linalg.inv(M)

    return gk_iw


def compute_lattice_gf_multiorbital_fast(
    hk: np.ndarray,
    sigma_c: np.ndarray,
    patch_assignment: np.ndarray,
    mu: float,
    wn: np.ndarray,
) -> np.ndarray:
    """Vectorized multi-orbital lattice G — batch inversion over k-points."""
    n_k = hk.shape[0]
    n_orb = hk.shape[1]
    n_w = wn.shape[0]
    eye = np.eye(n_orb, dtype=complex)

    gk_iw = np.zeros((n_k, n_w, n_orb, n_orb), dtype=complex)

    # sigma at each k-point: [n_k, n_w, n_orb, n_orb]
    sigma_k = sigma_c[patch_assignment]  # [n_k, n_w, n_orb, n_orb]

    for iw in range(n_w):
        M = (1j * wn[iw] + mu) * eye[np.newaxis, :, :] - hk - sigma_k[:, iw, :, :]
        gk_iw[:, iw] = np.linalg.inv(M)

    return gk_iw


def solve_multiorbital_cluster_cthyb(
    g0_iw_cluster: np.ndarray,
    params: MultiOrbitalDCAParams,
    work_dir: str,
    iteration: int,
) -> dict:
    """
    Solve the multi-orbital N_c-site cluster problem with TRIQS/CTHYB.

    Block structure: "up" block of size N_c × n_orb, "down" same.
    The cluster-orbital compound index: I = K_index * n_orb + orb_index.

    Args:
        g0_iw_cluster: [nc, 2*n_iw, n_orb, n_orb] bare cluster G
        params: multi-orbital DCA parameters
        work_dir: working directory
        iteration: iteration number

    Returns:
        g_c: [nc, 2*n_iw, n_orb, n_orb] interacting cluster G
    """
    try:
        return _solve_multiorbital_triqs(g0_iw_cluster, params, work_dir, iteration)
    except ImportError:
        print("[DCA-MO] TRIQS not available, using Hartree fallback")
        g_c = _solve_multiorbital_hartree(g0_iw_cluster, params)
        return {"g_c": g_c, "avg_sign": 1.0}


def _solve_multiorbital_triqs(
    g0_iw_cluster: np.ndarray,
    params: MultiOrbitalDCAParams,
    work_dir: str,
    iteration: int,
) -> dict:
    """Full TRIQS/CTHYB solver for multi-orbital cluster."""
    from triqs.gf import GfImFreq, BlockGf, MeshImFreq, inverse
    from triqs.operators import c, c_dag, n, Operator
    from triqs_cthyb import Solver

    nc = params.nc
    no = params.n_orb
    n_iw = params.n_iw
    beta = params.beta

    # Compound index size: N_c sites × n_orb orbitals
    block_size = nc * no

    gf_struct = [("up", block_size), ("down", block_size)]
    S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_tau=10001)

    # Fill G⁰(iω) — compound index I = K*n_orb + orb
    for spin in ["up", "down"]:
        for iw_idx, iw_val in enumerate(S.G0_iw[spin].mesh):
            if iw_idx >= 2 * n_iw:
                break
            for ic in range(nc):
                for a in range(no):
                    I = ic * no + a
                    for b in range(no):
                        J = ic * no + b
                        S.G0_iw[spin][iw_val][I, J] = g0_iw_cluster[ic, iw_idx, a, b]

    # Build interaction Hamiltonian
    if params.interaction is not None:
        # Apply Kanamori interaction on each cluster site
        H_int = Operator()
        for ic in range(nc):
            # Only on correlated orbitals
            corr = params.correlated_orbitals
            for a_idx, a in enumerate(corr):
                I_a = ic * no + a
                H_int += params.interaction.U[a] * n("up", I_a) * n("down", I_a)

                for b_idx, b in enumerate(corr):
                    if a == b:
                        continue
                    I_b = ic * no + b
                    Uprime = params.interaction.U_prime[a, b]
                    Jpair = params.interaction.J_pair[a, b]

                    H_int += Uprime * n("up", I_a) * n("down", I_b)
                    if a < b:
                        H_int += (Uprime - Jpair) * n("up", I_a) * n("up", I_b)
                        H_int += (Uprime - Jpair) * n("down", I_a) * n("down", I_b)

                    if not params.use_density_density:
                        # Spin-flip
                        H_int += -Jpair * (
                            c_dag("up", I_a) * c("down", I_a) *
                            c_dag("down", I_b) * c("up", I_b)
                        )
                        # Pair-hopping
                        if a < b:
                            H_int += Jpair * (
                                c_dag("up", I_a) * c_dag("down", I_a) *
                                c("down", I_b) * c("up", I_b)
                            )
    else:
        # Fallback: simple Hubbard U on all orbitals
        H_int = Operator()
        for ic in range(nc):
            for a in range(no):
                I = ic * no + a
                H_int += params.U * n("up", I) * n("down", I)

    # QMC cycles: scale down for sign problem
    # Multi-orbital sign problem is exponentially worse
    sign_penalty = max(1, no // 2)
    effective_cycles = max(params.n_cycles // sign_penalty, 500_000)

    S.solve(
        h_int=H_int,
        n_warmup_cycles=params.n_warmup,
        n_cycles=effective_cycles // 20,
        length_cycle=params.length_cycle,
        move_double=True,
        measure_density_matrix=True,
    )

    # Extract G_c(K, iω)_{ab}
    g_c = np.zeros((nc, 2 * n_iw, no, no), dtype=complex)
    for iw_idx, iw_val in enumerate(S.G_iw["up"].mesh):
        if iw_idx >= 2 * n_iw:
            break
        for ic in range(nc):
            for a in range(no):
                I = ic * no + a
                for b in range(no):
                    J = ic * no + b
                    g_c[ic, iw_idx, a, b] = S.G_iw["up"][iw_val][I, J]

    # Save
    iter_dir = os.path.join(work_dir, f"iter_{iteration:03d}")
    os.makedirs(iter_dir, exist_ok=True)

    # Average sign
    avg_sign = 1.0
    try:
        avg_sign = S.average_sign
    except Exception:
        pass

    avg_sign = float(avg_sign)
    np.savez(os.path.join(iter_dir, "cluster_gf.npz"), g_c=g_c, avg_sign=avg_sign)
    print(f"[DCA-MO] iter {iteration}: average sign = {avg_sign:.4f}")

    return {"g_c": g_c, "avg_sign": avg_sign}


def _solve_multiorbital_hartree(
    g0_iw_cluster: np.ndarray,
    params: MultiOrbitalDCAParams,
) -> np.ndarray:
    """
    Hartree (static mean-field) fallback for multi-orbital cluster.
    Tests loop machinery without requiring TRIQS.
    """
    nc, n_w, no, _ = g0_iw_cluster.shape

    filling = np.full(no, 0.5)  # assume half-filling per orbital
    sigma_hartree = np.zeros((no, no), dtype=complex)
    for a in range(no):
        sigma_hartree[a, a] = params.U * filling[a]

    g_c = np.zeros_like(g0_iw_cluster)
    for ic in range(nc):
        for iw in range(n_w):
            g0_inv = np.linalg.inv(g0_iw_cluster[ic, iw])
            g_c[ic, iw] = np.linalg.inv(g0_inv - sigma_hartree)

    return g_c


# ── Multi-orbital DCA self-consistency loop ──────────────────────────────────

def run_multiorbital_dca(
    hk: np.ndarray,
    kpoints: np.ndarray,
    params: MultiOrbitalDCAParams,
    work_dir: str,
    initial_sigma: Optional[np.ndarray] = None,
) -> dict:
    """
    Run the multi-orbital DCA self-consistency loop.

    Same structure as single-band run_dca but with matrix-valued quantities.

    Args:
        hk: [n_k, n_orb, n_orb] lattice Hamiltonian
        kpoints: [n_k, dim] k-points
        params: multi-orbital DCA parameters
        work_dir: output directory
        initial_sigma: [nc, 2*n_iw, n_orb, n_orb] initial self-energy

    Returns:
        dict with converged sigma, Green's functions, sign history
    """
    os.makedirs(work_dir, exist_ok=True)
    nc = params.nc
    no = params.n_orb
    n_w = 2 * params.n_iw
    wn = params.wn

    t0 = time.time()

    patch_assignment = assign_k_to_patches(kpoints, params.K_cluster)
    print(f"[DCA-MO] N_c={nc}, n_orb={no}, block_size={nc*no}, n_k={len(kpoints)}")

    # Initialize
    sigma_c = initial_sigma if initial_sigma is not None else np.zeros(
        (nc, n_w, no, no), dtype=complex
    )

    convergence_history = []
    sign_history = []

    for iteration in range(params.n_iter):
        t_iter = time.time()

        # 1. Lattice G(k, iω) with current Σ
        gk_iw = compute_lattice_gf_multiorbital_fast(hk, sigma_c, patch_assignment, params.mu, wn)

        # 2. Coarse-grain
        g_bar = coarse_grain_gf(gk_iw, patch_assignment, nc)

        # 3. Cluster bare G⁰ via Dyson
        g0_c = np.zeros((nc, n_w, no, no), dtype=complex)
        for ic in range(nc):
            for iw in range(n_w):
                g_bar_inv = np.linalg.inv(g_bar[ic, iw])
                g0_c[ic, iw] = np.linalg.inv(g_bar_inv + sigma_c[ic, iw])

        # 4. Solve cluster
        solver_result = solve_multiorbital_cluster_cthyb(g0_c, params, work_dir, iteration)
        g_c = solver_result["g_c"]
        avg_sign = solver_result["avg_sign"]
        sign_history.append(float(avg_sign))

        # Check sign problem
        if avg_sign < 0.05:
            print(f"[DCA-MO] FATAL: average sign {avg_sign:.4f} < 0.05 — aborting")
            break
        elif avg_sign < 0.2:
            print(f"[DCA-MO] WARNING: low average sign {avg_sign:.4f}")

        # 5. Extract Σ
        sigma_new = np.zeros((nc, n_w, no, no), dtype=complex)
        for ic in range(nc):
            for iw in range(n_w):
                g0_inv = np.linalg.inv(g0_c[ic, iw])
                gc_inv = np.linalg.inv(g_c[ic, iw])
                sigma_new[ic, iw] = g0_inv - gc_inv

        # Check for NaN/Inf
        if np.any(np.isnan(sigma_new)) or np.any(np.isinf(sigma_new)):
            print(f"[DCA-MO] FATAL: NaN/Inf in self-energy at iteration {iteration+1}")
            break

        # 6. Mix
        sigma_mixed = params.sigma_mix * sigma_c + (1 - params.sigma_mix) * sigma_new

        # 7. Convergence check
        diff = np.max(np.abs(sigma_mixed - sigma_c))
        sigma_c = sigma_mixed

        # Density per orbital
        density = np.zeros(no)
        for a in range(no):
            for ic in range(nc):
                density[a] += 1.0 + (2.0 / params.beta) * np.sum(g_c[ic, :, a, a].real)
            density[a] *= 2.0 / nc

        elapsed_iter = time.time() - t_iter
        convergence_history.append(float(diff))

        dens_str = ", ".join(f"{d:.3f}" for d in density)
        print(f"[DCA-MO] iter {iteration+1}: ||ΔΣ||={diff:.2e}, "
              f"n=[{dens_str}], <sign>={avg_sign:.4f}, time={elapsed_iter:.1f}s")

        if diff < params.convergence_tol and iteration > 2:
            print(f"[DCA-MO] Converged after {iteration+1} iterations")
            break

    elapsed = time.time() - t0
    converged = convergence_history[-1] < params.convergence_tol if convergence_history else False

    sign_problem = len(sign_history) > 0 and sign_history[-1] < 0.05

    np.savez(
        os.path.join(work_dir, "dca_mo_converged.npz"),
        sigma_c=sigma_c, g_bar=g_bar, g_c=g_c, g0_c=g0_c,
        K_cluster=params.K_cluster, patch_assignment=patch_assignment, wn=wn,
        sign_history=np.array(sign_history),
    )

    return {
        "converged": converged,
        "sign_problem": sign_problem,
        "avg_sign": sign_history[-1] if sign_history else 1.0,
        "n_iterations": len(convergence_history),
        "final_diff": convergence_history[-1] if convergence_history else None,
        "convergence_history": convergence_history,
        "sign_history": sign_history,
        "elapsed_seconds": elapsed,
        "sigma_c": sigma_c,
        "g_bar": g_bar,
        "g_c": g_c,
        "g0_c": g0_c,
        "K_cluster": params.K_cluster,
        "patch_assignment": patch_assignment,
        "n_orb": no,
    }
