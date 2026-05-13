#!/usr/bin/env python3
"""
B3: Momentum-Resolved Pairing Susceptibility & Tc Determination.

With the local irreducible vertex Γ_loc from the BSE solver, we compute
the lattice pairing susceptibility on a momentum mesh:

  χ_pair(k, T) = Σ_q G(k+q, iω) G(-k-q, -iω) [1 - Γ_loc · χ(q, iω)]⁻¹

The leading eigenvalue λ_pair(T) of the pairing matrix tells us the
pairing instability:
  - λ_pair → 1 from below as T → Tc  ⟹  superconducting transition
  - Eigenvector symmetry ⟹  s-wave, d-wave, p-wave, etc.

For d-wave (cuprate-like):
  Δ(k) ∝ cos(kx) - cos(ky)
  The eigenvector has this cos(kx)-cos(ky) angular dependence on the FS.

For s±-wave (pnictide-like):
  Δ(k) changes sign between electron and hole pockets
  The eigenvector has opposite sign on Γ-centered and M-centered sheets.

Temperature sweep: run at multiple β values (each requires a separate DMFT
run or analytic continuation) and track λ_pair(T) → 1 extrapolation.

References:
  Maier et al., PRL 95, 237001 (2005) — DCA d-wave pairing
  Scalapino, Rev. Mod. Phys. 84, 1383 (2012) — pairing from repulsion
  Gull et al., Rev. Mod. Phys. 83, 349 (2011) — DCA methods
  Rohringer et al., Rev. Mod. Phys. 90, 025003 (2018) — Sec. V
  Monthoux et al., Nature 450, 1177 (2007) — overview
"""

import numpy as np
import json
import os
import time
from typing import Optional


# ── Lattice Green's function ─────────────────────────────────────────────────

def compute_lattice_gf(
    hk: np.ndarray,
    sigma_iw: np.ndarray,
    mu: float,
    beta: float,
    n_iw: int,
) -> np.ndarray:
    """
    Compute the lattice Green's function G(k, iω_n) from H(k) and Σ(iω_n).

    G(k, iω_n) = [iω_n + μ - H(k) - Σ(iω_n)]⁻¹

    Args:
        hk:       H(k) array [n_k, n_orb, n_orb]
        sigma_iw: self-energy Σ(iω_n) [2*n_iw, n_orb, n_orb]
        mu:       chemical potential (eV)
        beta:     inverse temperature (eV⁻¹)
        n_iw:     number of Matsubara frequencies per side

    Returns:
        gk_iw: G(k, iω_n) array [n_k, 2*n_iw, n_orb, n_orb]
    """
    n_k = hk.shape[0]
    n_orb = hk.shape[1]
    nw = 2 * n_iw

    # Matsubara frequencies: ω_n = (2n+1)π/β
    wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(nw)])

    gk_iw = np.zeros((n_k, nw, n_orb, n_orb), dtype=complex)
    eye = np.eye(n_orb, dtype=complex)

    for ik in range(n_k):
        for iw in range(nw):
            # [iω_n + μ]·I - H(k) - Σ(iω_n)
            mat = (1j * wn[iw] + mu) * eye - hk[ik] - sigma_iw[iw]
            gk_iw[ik, iw] = np.linalg.inv(mat)

    return gk_iw


def compute_lattice_gf_fast(
    hk: np.ndarray,
    sigma_iw: np.ndarray,
    mu: float,
    beta: float,
    n_iw: int,
) -> np.ndarray:
    """
    Vectorized lattice Green's function — batch-inverts over k-points.

    ~10× faster than the loop version for large k-meshes.
    """
    n_k = hk.shape[0]
    n_orb = hk.shape[1]
    nw = 2 * n_iw

    wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(nw)])
    eye = np.eye(n_orb, dtype=complex)

    gk_iw = np.zeros((n_k, nw, n_orb, n_orb), dtype=complex)

    for iw in range(nw):
        # Build all k-point matrices at this frequency simultaneously
        # M(k) = (iω + μ)·I - H(k) - Σ(iω)
        # Shape: [n_k, n_orb, n_orb]
        M = (1j * wn[iw] + mu) * eye[np.newaxis, :, :] - hk - sigma_iw[iw][np.newaxis, :, :]
        # Batch inversion
        gk_iw[:, iw, :, :] = np.linalg.inv(M)

    return gk_iw


# ── Pairing kernel construction ──────────────────────────────────────────────

def construct_pairing_matrix(
    gk_iw: np.ndarray,
    gamma_singlet: np.ndarray,
    kpoints: np.ndarray,
    n_iw_f: int,
    n_orb: int,
    beta: float,
) -> np.ndarray:
    """
    Construct the linearized Eliashberg (pairing) eigenvalue matrix.

    The pairing equation in the singlet channel:

      λ · Δ(k, iν) = -(T/N_k) Σ_{k',ν'} Γ_singlet(ν,ν') ·
                       G(k', iν') · G(-k', -iν') · Δ(k', iν')

    We diagonalize the kernel matrix:
      K_{(k,ν,a,b), (k',ν',c,d)} =
        -(T/N_k) · Γ_singlet_{(ν,a,b),(ν',c,d)} · G_{ac}(k',ν') · G_{bd}(-k',-ν')

    The largest eigenvalue λ_pair → 1 signals the SC transition.

    For computational tractability, we work with the k-point mesh
    and the fermionic frequency grid from the vertex measurement.
    The matrix size is n_k × 2·n_iw_f × n_orb² which can be large;
    we use Lanczos iteration for the leading eigenvalues.

    Args:
        gk_iw:          G(k, iω) [n_k, 2*n_iw_gf, n_orb, n_orb]
        gamma_singlet:  singlet pairing vertex [N_vertex, N_vertex]
                        where N_vertex = 2*n_iw_f * n_orb²
        kpoints:        k-points [n_k, 3] fractional
        n_iw_f:         fermionic frequencies in vertex
        n_orb:          orbital count
        beta:           inverse temperature

    Returns:
        kernel: pairing kernel matrix [n_k * N_vertex, n_k * N_vertex]
                (or a LinearOperator for large systems)
    """
    n_k = gk_iw.shape[0]
    nw_gf = gk_iw.shape[1]
    nw_vertex = 2 * n_iw_f
    N_v = nw_vertex * n_orb * n_orb
    T = 1.0 / beta
    prefactor = -T / n_k

    # For the pairing kernel we need G(k', ν') and G(-k', -ν')
    # G(-k, -iω) = G(k, iω)* for time-reversal-invariant systems
    # (paramagnetic, no SOC). This is the anomalous pairing.
    #
    # In practice: G(-k, -ν_n) at index (n_k - ik, nw - 1 - iw)
    # But for a uniform mesh with inversion symmetry, we can use
    # the k-point mapping.

    # Build the minus-k index map
    # For a uniform Gamma-centered mesh, -k maps to the point
    # closest to 1-k (mod 1) in the mesh
    minus_k_map = _build_minus_k_map(kpoints)

    # Restrict G to the vertex frequency window
    # The vertex has 2*n_iw_f frequencies; G may have more.
    # Center the window on the middle of the G frequency grid.
    iw_offset = (nw_gf - nw_vertex) // 2
    if iw_offset < 0:
        raise ValueError(f"G has fewer frequencies ({nw_gf}) than vertex ({nw_vertex})")

    # For large systems, build a sparse LinearOperator instead of dense matrix.
    # Threshold: if n_k * N_v > 5000, use iterative eigensolvers.
    total_dim = n_k * N_v
    use_sparse = total_dim > 5000

    if use_sparse:
        return _build_pairing_operator(
            gk_iw, gamma_singlet, minus_k_map,
            n_iw_f, n_orb, iw_offset, prefactor,
        )

    # Dense construction for small systems
    kernel = np.zeros((total_dim, total_dim), dtype=complex)

    for ikp in range(n_k):
        ik_minus = minus_k_map[ikp]
        for iv in range(nw_vertex):
            iv_gf = iv + iw_offset
            iv_minus = nw_gf - 1 - iv_gf  # -ν index in G
            for ivp in range(nw_vertex):
                ivp_gf = ivp + iw_offset
                ivp_minus = nw_gf - 1 - ivp_gf

                for a in range(n_orb):
                    for b in range(n_orb):
                        I_v = iv * n_orb * n_orb + a * n_orb + b
                        for c in range(n_orb):
                            for d in range(n_orb):
                                J_v = ivp * n_orb * n_orb + c * n_orb + d

                                gamma_val = gamma_singlet[I_v, J_v]
                                if abs(gamma_val) < 1e-15:
                                    continue

                                # G_{ac}(k', ν') · G_{bd}(-k', -ν')
                                g_kp = gk_iw[ikp, ivp_gf, a, c]
                                g_mkp = gk_iw[ik_minus, ivp_minus, b, d]

                                val = prefactor * gamma_val * g_kp * g_mkp

                                # Fill for ALL k-points in the row
                                # (vertex is k-independent in single-site DMFT)
                                for ik in range(n_k):
                                    row = ik * N_v + I_v
                                    col = ikp * N_v + J_v
                                    kernel[row, col] += val

    return kernel


def _build_minus_k_map(kpoints: np.ndarray) -> np.ndarray:
    """Map each k-point index to its -k partner in the mesh.

    Auto-detects BZ period from mesh range:
      - kpoints in [0,1) fractional coords → period 1
      - kpoints in [-1,1) π/a units → period 2
    """
    n_k = kpoints.shape[0]

    # Detect BZ period: if any k < -0.01, mesh is [-1,1) with period 2
    period = 2.0 if np.any(kpoints < -0.01) else 1.0
    half = period / 2.0

    # Wrap -k into the mesh range
    k_min = kpoints.min(axis=0)
    minus_k = np.mod(-kpoints - k_min, period) + k_min

    # Find nearest neighbor for each -k in the mesh
    mapping = np.zeros(n_k, dtype=int)
    for ik in range(n_k):
        dists = np.linalg.norm(
            np.mod(kpoints - minus_k[ik] + half, period) - half, axis=1
        )
        mapping[ik] = np.argmin(dists)

    return mapping


def _build_pairing_operator(
    gk_iw, gamma_singlet, minus_k_map,
    n_iw_f, n_orb, iw_offset, prefactor,
):
    """
    Build a scipy LinearOperator for the pairing kernel.
    Used for large systems where the dense matrix won't fit in memory.
    The operator implements matrix-vector product K·x without storing K.
    """
    from scipy.sparse.linalg import LinearOperator

    n_k = gk_iw.shape[0]
    nw_gf = gk_iw.shape[1]
    nw_v = 2 * n_iw_f
    N_v = nw_v * n_orb * n_orb
    total_dim = n_k * N_v

    def matvec(x):
        """K · x where x is [n_k * N_v] complex vector."""
        x_reshape = x.reshape(n_k, N_v)
        result = np.zeros_like(x_reshape)

        for ikp in range(n_k):
            ik_minus = minus_k_map[ikp]

            # Build the G-weighted input for this k'
            # weighted_x[J_v] = Σ_J G(k',ν')_{ac} · G(-k',-ν')_{bd} · x(k', J_v)
            for ivp in range(nw_v):
                ivp_gf = ivp + iw_offset
                ivp_minus = nw_gf - 1 - ivp_gf

                for c in range(n_orb):
                    for d in range(n_orb):
                        J_v = ivp * n_orb * n_orb + c * n_orb + d
                        x_val = x_reshape[ikp, J_v]
                        if abs(x_val) < 1e-20:
                            continue

                        for a in range(n_orb):
                            for b in range(n_orb):
                                g_kp = gk_iw[ikp, ivp_gf, a, c]
                                g_mkp = gk_iw[ik_minus, ivp_minus, b, d]

                                # Apply vertex: Γ[(iv,a,b),(ivp,c,d)] · G_ac · G_bd · x_{cd}
                                for iv in range(nw_v):
                                    Iv = iv * n_orb * n_orb + a * n_orb + b
                                    gamma_val = gamma_singlet[Iv, J_v]
                                    if abs(gamma_val) < 1e-15:
                                        continue
                                    contribution = prefactor * gamma_val * g_kp * g_mkp * x_val
                                    # k-independent vertex: same for all k
                                    result[:, Iv] += contribution

        return result.ravel()

    return LinearOperator(
        shape=(total_dim, total_dim),
        matvec=matvec,
        dtype=complex,
    )


# ── Eigenvalue solver ────────────────────────────────────────────────────────

def solve_pairing_eigenvalue(
    kernel,
    n_k: int,
    n_iw_f: int,
    n_orb: int,
    kpoints: np.ndarray,
    n_evals: int = 10,
) -> dict:
    """
    Find the leading eigenvalues of the pairing kernel.

    λ_pair → 1 from below signals the superconducting instability.
    The corresponding eigenvector gives the gap function Δ(k, iν).

    Args:
        kernel:   pairing kernel matrix [D, D] or LinearOperator
        n_k:      number of k-points
        n_iw_f:   fermionic frequencies in vertex
        n_orb:    orbital dimension
        kpoints:  k-points [n_k, 3]
        n_evals:  number of leading eigenvalues to compute

    Returns:
        dict with eigenvalues, gap symmetry classification, Tc estimate
    """
    from scipy.sparse.linalg import LinearOperator, eigs
    N_v = 2 * n_iw_f * n_orb * n_orb
    total_dim = n_k * N_v

    n_evals = min(n_evals, total_dim - 2)

    print(f"[Pairing] Solving eigenvalue problem: dim={total_dim}, "
          f"requesting {n_evals} eigenvalues...")

    t0 = time.time()

    if isinstance(kernel, LinearOperator):
        # Iterative solver for large systems
        try:
            eigenvalues, eigenvectors = eigs(
                kernel, k=n_evals, which="LR",  # largest real part
                maxiter=1000, tol=1e-6,
            )
        except Exception as e:
            print(f"[Pairing] WARNING: eigs failed ({e}), trying with fewer eigenvalues")
            eigenvalues, eigenvectors = eigs(
                kernel, k=min(3, n_evals), which="LR",
                maxiter=2000, tol=1e-4,
            )
    else:
        # Dense eigensolver — compute all and take largest
        eigenvalues_all = np.linalg.eigvals(kernel)
        idx = np.argsort(-eigenvalues_all.real)[:n_evals]
        eigenvalues = eigenvalues_all[idx]
        # For eigenvectors of top eigenvalues, use eigh on hermitianized kernel
        kernel_h = (kernel + kernel.conj().T) / 2
        evals_h, evecs_h = np.linalg.eigh(kernel_h)
        idx_h = np.argsort(-evals_h)[:n_evals]
        eigenvectors = evecs_h[:, idx_h]

    elapsed = time.time() - t0
    print(f"[Pairing] Eigenvalue solve took {elapsed:.1f}s")

    # Sort by real part (descending)
    idx = np.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Leading eigenvalue
    lambda_pair = eigenvalues[0].real
    print(f"[Pairing] Leading eigenvalue: λ_pair = {lambda_pair:.6f}")
    print(f"[Pairing] Top 5 eigenvalues: {[f'{ev.real:.4f}' for ev in eigenvalues[:5]]}")

    # Classify gap symmetry from the leading eigenvector
    gap_symmetry = classify_gap_symmetry(
        eigenvectors[:, 0], n_k, n_iw_f, n_orb, kpoints,
    )

    return {
        "lambda_pair": float(lambda_pair),
        "eigenvalues": [complex(ev) for ev in eigenvalues[:n_evals]],
        "gap_symmetry": gap_symmetry,
        "elapsed_seconds": elapsed,
        "total_dim": total_dim,
    }


# ── Gap symmetry classification ──────────────────────────────────────────────

# Basis functions for common pairing symmetries on a 2D square lattice
# (generalized to 3D via kz-dependence patterns)

SYMMETRY_BASIS = {
    "s-wave":       lambda kx, ky, kz: np.ones_like(kx),
    "s±-wave":      lambda kx, ky, kz: np.cos(kx) + np.cos(ky),
    "d-x2y2-wave":  lambda kx, ky, kz: np.cos(kx) - np.cos(ky),
    "d-xy-wave":    lambda kx, ky, kz: np.sin(kx) * np.sin(ky),
    "p-x-wave":     lambda kx, ky, kz: np.sin(kx),
    "p-y-wave":     lambda kx, ky, kz: np.sin(ky),
    "g-wave":       lambda kx, ky, kz: np.sin(kx) * np.sin(ky) * (np.cos(kx) - np.cos(ky)),
}

# Singlet (even-parity, k → -k symmetric) basis: spin-anti-symmetric pair
SINGLET_BASIS = {
    "s-wave":       lambda kx, ky, kz: np.ones_like(kx),
    "s±-wave":      lambda kx, ky, kz: np.cos(kx) + np.cos(ky),
    "extended-s":   lambda kx, ky, kz: np.cos(kx) * np.cos(ky),
    "d-x2y2-wave":  lambda kx, ky, kz: np.cos(kx) - np.cos(ky),
    "d-xy-wave":    lambda kx, ky, kz: np.sin(kx) * np.sin(ky),
    "d-3z2r2":      lambda kx, ky, kz: 2 * np.cos(kz) - np.cos(kx) - np.cos(ky),
    "g-wave":       lambda kx, ky, kz: np.sin(kx) * np.sin(ky) * (np.cos(kx) - np.cos(ky)),
}

# Triplet (odd-parity, k → -k antisymmetric) basis: spin-symmetric pair
# These form the components of the d-vector d(k) such that Δ(k) = i (d·σ) σ_y.
# Real solid-state SCs of interest:
#   p-x, p-y, p-z: simple p-wave (3D)
#   p+ip (chiral): UPt3 A1 phase, Sr2RuO4 candidate
#   helical p:    UTe2, B-phase of 3He
#   f-wave:       UPt3 E1u, possibly UCoGe
TRIPLET_BASIS = {
    "p-x":          lambda kx, ky, kz: np.sin(kx),
    "p-y":          lambda kx, ky, kz: np.sin(ky),
    "p-z":          lambda kx, ky, kz: np.sin(kz),
    "p+ip (chiral)": lambda kx, ky, kz: np.sin(kx) + 1j * np.sin(ky),
    "f-wave-x":     lambda kx, ky, kz: np.sin(kx) * (np.cos(kx) - np.cos(ky)),
    "f-wave-y":     lambda kx, ky, kz: np.sin(ky) * (np.cos(kx) - np.cos(ky)),
    "f-wave-z":     lambda kx, ky, kz: np.sin(kz) * (np.cos(kx) - np.cos(ky)),
}


def classify_gap_symmetry(
    eigenvector: np.ndarray,
    n_k: int,
    n_iw_f: int,
    n_orb: int,
    kpoints: np.ndarray,
) -> dict:
    """
    Classify the gap function symmetry from the leading pairing eigenvector.

    The eigenvector Δ(k, iν, a, b) lives in the compound space.
    We project onto the lowest Matsubara frequency (ν_0, most physical)
    and trace over orbitals to get Δ(k), then compute overlap with
    symmetry basis functions.

    The symmetry with the largest |<Δ|basis>|² / <Δ|Δ> wins.
    """
    N_v = 2 * n_iw_f * n_orb * n_orb

    # Reshape eigenvector: [n_k, 2*n_iw_f, n_orb, n_orb]
    delta = eigenvector.reshape(n_k, 2 * n_iw_f, n_orb, n_orb)

    # Extract at the lowest positive Matsubara frequency (index n_iw_f = center)
    delta_v0 = delta[:, n_iw_f, :, :]  # [n_k, n_orb, n_orb]

    # Trace over orbitals: Δ(k) = Tr[Δ(k, ν_0)]
    # For singlet: Δ is antisymmetric in orbital indices
    delta_k = np.trace(delta_v0, axis1=1, axis2=2)  # [n_k]

    # Normalize
    norm = np.sqrt(np.sum(np.abs(delta_k) ** 2))
    if norm < 1e-15:
        return {"symmetry": "unknown", "overlaps": {}, "nodes": "unknown"}
    delta_k_norm = delta_k / norm

    # Convert k-points to angular coordinates: kx, ky, kz in [-π, π]
    kx = 2 * np.pi * kpoints[:, 0]
    ky = 2 * np.pi * kpoints[:, 1]
    kz = 2 * np.pi * kpoints[:, 2]

    # Compute overlap with each symmetry basis
    overlaps = {}
    for name, basis_fn in SYMMETRY_BASIS.items():
        basis = basis_fn(kx, ky, kz)
        basis_norm = np.sqrt(np.sum(np.abs(basis) ** 2))
        if basis_norm < 1e-15:
            overlaps[name] = 0.0
            continue
        basis_normalized = basis / basis_norm
        overlap = np.abs(np.sum(delta_k_norm.conj() * basis_normalized)) ** 2
        overlaps[name] = float(overlap)

    # Winner: highest overlap
    best_symmetry = max(overlaps, key=overlaps.get)
    best_overlap = overlaps[best_symmetry]

    # Node structure: count sign changes of Re(Δ(k)) along high-symmetry directions
    sign_changes = _count_sign_changes(delta_k.real, kpoints)

    # Classify nodal structure
    if sign_changes == 0:
        nodes = "nodeless"
    elif sign_changes <= 2:
        nodes = "line_nodes" if best_symmetry.startswith("d") else "point_nodes"
    else:
        nodes = "complex_nodes"

    return {
        "symmetry": best_symmetry,
        "overlap": best_overlap,
        "all_overlaps": overlaps,
        "nodes": nodes,
        "sign_changes": sign_changes,
        "is_unconventional": best_symmetry not in ("s-wave",),
    }


def classify_pairing_full(
    eigenvector: np.ndarray,
    n_k: int,
    n_iw_f: int,
    n_orb: int,
    kpoints: np.ndarray,
    parity_threshold: float = 0.7,
) -> dict:
    """
    Full singlet+triplet classification of a pairing eigenvector.

    Procedure:
      1. Form Δ(k) = Tr[Δ(k, ν_0)] (orbital-symmetric scalar projection)
      2. Test parity: even (Δ(k)+Δ(-k))/2 vs odd (Δ(k)-Δ(-k))/2
      3. Project onto SINGLET basis (even functions) and TRIPLET basis (odd
         functions)
      4. Classify as singlet/triplet based on which has dominant overlap
      5. Sub-classify by best basis function

    For triplet, the eigenvector is interpreted as the dominant d-vector
    component along z (S_z=0 triplet) by default. Full d-vector reconstruction
    requires spin-resolved data not stored in the current vertex.

    Args:
        eigenvector: leading eigenvector of the pairing kernel
        n_k:    number of k-points
        n_iw_f: fermionic Matsubara frequencies per side
        n_orb:  orbital count
        kpoints: [n_k, 3] k-points (units of π/a)
        parity_threshold: fraction of norm in even/odd part to call it
            singlet/triplet (>=0.7 → unambiguous; <0.7 → mixed parity)

    Returns:
        dict {
            channel: "singlet" | "triplet" | "mixed",
            symmetry: best basis function name,
            overlap: float (squared overlap with dominant basis),
            d_vector_component: "z" (for S_z=0 default) or "x/y/z" if resolved,
            even_fraction: how much of Δ is k-even,
            odd_fraction:  how much of Δ is k-odd,
            singlet_overlaps: {basis_name: overlap},
            triplet_overlaps: {basis_name: overlap},
            nodes: ...,
        }
    """
    # Reshape: [n_k, 2*n_iw_f, n_orb, n_orb]
    delta = eigenvector.reshape(n_k, 2 * n_iw_f, n_orb, n_orb)
    # Scalar gap on lowest positive Matsubara
    delta_v0 = delta[:, n_iw_f, :, :]
    delta_k = np.trace(delta_v0, axis1=1, axis2=2)

    # Decompose into even + odd in k → -k
    # Build mapping k → -k by finding nearest neighbor in kpoints
    kp = np.asarray(kpoints)
    # For each k, find index of -k via L2 distance
    neg_k_idx = np.zeros(n_k, dtype=int)
    for ik in range(n_k):
        # nearest point to -k (mod BZ): kpoints expected in [-1, 1) in π/a units
        target = -kp[ik]
        # Wrap to [-1, 1)
        target = ((target + 1.0) % 2.0) - 1.0
        d2 = np.sum((kp - target[np.newaxis, :]) ** 2, axis=1)
        neg_k_idx[ik] = int(np.argmin(d2))

    delta_neg_k = delta_k[neg_k_idx]
    delta_even = 0.5 * (delta_k + delta_neg_k)
    delta_odd = 0.5 * (delta_k - delta_neg_k)

    n_total = float(np.sum(np.abs(delta_k) ** 2))
    n_even = float(np.sum(np.abs(delta_even) ** 2))
    n_odd = float(np.sum(np.abs(delta_odd) ** 2))

    if n_total < 1e-30:
        return {
            "channel": "unknown",
            "symmetry": "unknown",
            "even_fraction": 0.0,
            "odd_fraction": 0.0,
        }
    even_frac = n_even / n_total
    odd_frac = n_odd / n_total

    # k-space coordinates in [-π, π]
    kx = np.pi * kp[:, 0]
    ky = np.pi * kp[:, 1]
    kz = np.pi * kp[:, 2] if kp.shape[1] >= 3 else np.zeros(n_k)

    # Compute overlaps with both bases
    def _overlap(delta_proj, basis_fn):
        b = basis_fn(kx, ky, kz)
        n_b = np.sqrt(np.sum(np.abs(b) ** 2))
        n_d = np.sqrt(np.sum(np.abs(delta_proj) ** 2))
        if n_b < 1e-15 or n_d < 1e-15:
            return 0.0
        return float(np.abs(np.sum(delta_proj.conj() * b)) ** 2 / (n_b ** 2 * n_d ** 2))

    singlet_overlaps = {name: _overlap(delta_even, fn) for name, fn in SINGLET_BASIS.items()}
    triplet_overlaps = {name: _overlap(delta_odd, fn) for name, fn in TRIPLET_BASIS.items()}

    # Pick the channel based on parity fraction
    if even_frac >= parity_threshold:
        channel = "singlet"
        best = max(singlet_overlaps, key=singlet_overlaps.get)
        overlap = singlet_overlaps[best]
    elif odd_frac >= parity_threshold:
        channel = "triplet"
        best = max(triplet_overlaps, key=triplet_overlaps.get)
        overlap = triplet_overlaps[best]
    else:
        channel = "mixed"
        # Pick whichever basis has higher overlap
        best_s = max(singlet_overlaps, key=singlet_overlaps.get)
        best_t = max(triplet_overlaps, key=triplet_overlaps.get)
        if singlet_overlaps[best_s] > triplet_overlaps[best_t]:
            best, overlap = best_s, singlet_overlaps[best_s]
        else:
            best, overlap = best_t, triplet_overlaps[best_t]

    # Node classification (rough)
    sign_changes = _count_sign_changes(delta_k.real, kp)
    if channel == "singlet" and sign_changes == 0:
        nodes = "nodeless"
    elif channel == "triplet" and sign_changes <= 2:
        nodes = "point_nodes"
    elif sign_changes <= 2:
        nodes = "line_nodes"
    else:
        nodes = "complex_nodes"

    return {
        "channel": channel,
        "symmetry": best,
        "overlap": overlap,
        "d_vector_component": "z",  # S_z=0 triplet assumed; needs spin-resolved data for full reconstruction
        "even_fraction": even_frac,
        "odd_fraction": odd_frac,
        "singlet_overlaps": singlet_overlaps,
        "triplet_overlaps": triplet_overlaps,
        "nodes": nodes,
        "sign_changes": sign_changes,
        "is_unconventional": channel != "singlet" or best != "s-wave",
    }


def _count_sign_changes(delta_real: np.ndarray, kpoints: np.ndarray) -> int:
    """Count sign changes of the gap function along high-symmetry directions.

    Checks BOTH the kx-axis (catches p_x-wave nodes at kx=0) AND the diagonal
    kx=ky (catches d-wave nodes along the diagonal). Returns the max sign
    changes across these directions.
    """
    counts = []
    # Direction 1: kx axis at ky≈0, kz≈0 (p-wave nodes here)
    tol = 0.05
    # Use wrapped distance to also catch ky near 1.0 (≡ 0 by periodicity)
    ky_frac = kpoints[:, 1] % 1
    kz_frac = kpoints[:, 2] % 1
    near_axis = (
        (np.minimum(ky_frac, 1 - ky_frac) < tol) &
        (np.minimum(kz_frac, 1 - kz_frac) < tol)
    )

    def _sign_changes_along(mask, kx_coord):
        if np.sum(mask) < 3:
            return 0
        kx_line = kpoints[mask, kx_coord]
        delta_line = delta_real[mask]
        idx = np.argsort(kx_line)
        delta_sorted = delta_line[idx]
        # Filter out near-zero values to avoid counting numerical noise as
        # sign changes (e.g., d-wave gap is exactly 0 along the diagonal)
        max_abs = np.max(np.abs(delta_sorted))
        if max_abs < 1e-12:
            return 0
        thresh = 0.01 * max_abs  # 1% of max magnitude
        significant = delta_sorted[np.abs(delta_sorted) > thresh]
        if len(significant) < 2:
            return 0
        signs = np.sign(significant)
        return int(np.sum(np.abs(np.diff(signs)) > 0))

    counts.append(_sign_changes_along(near_axis, 0))

    # Direction 2: diagonal kx=ky (d-wave nodes here)
    # Δ_{x²-y²}(k) = cos(kx)-cos(ky) = 0 along kx=ky
    kx_frac = kpoints[:, 0] % 1
    diff_xy = np.abs((kx_frac - ky_frac + 0.5) % 1 - 0.5)  # wrapped |kx-ky|
    near_diag = (diff_xy < tol) & (np.minimum(kz_frac, 1 - kz_frac) < tol)
    counts.append(_sign_changes_along(near_diag, 0))

    return max(counts) if counts else 0


# ── Temperature sweep & Tc extrapolation ─────────────────────────────────────

def extrapolate_tc(
    temperatures: list[float],
    lambda_pairs: list[float],
) -> dict:
    """
    Extrapolate Tc from λ_pair(T) data.

    λ_pair increases as T decreases. When λ_pair(Tc) = 1, the system
    becomes superconducting.

    Methods:
      1. Linear extrapolation of 1 - λ(T) → 0
      2. If λ never reaches 1 in our data, extrapolate the trend
      3. If λ > 1 already at our lowest T, Tc > T_min

    Returns:
        dict with Tc estimate and confidence
    """
    if len(temperatures) < 2:
        return {
            "tc_bse": None,
            "tc_confidence": "insufficient_data",
            "lambda_at_lowest_T": float(lambda_pairs[-1]) if lambda_pairs else None,
        }

    temps = np.array(temperatures)
    lambdas = np.array(lambda_pairs)

    # Sort by temperature (ascending)
    idx = np.argsort(temps)
    temps = temps[idx]
    lambdas = lambdas[idx]

    lambda_max = lambdas[0]  # lowest T has highest lambda
    t_min = temps[0]

    # Case 1: λ already > 1 at lowest T
    if lambda_max >= 1.0:
        # Tc is above our lowest T — we can only give a lower bound
        # Interpolate between the two points bracketing λ=1
        for i in range(len(lambdas) - 1):
            if lambdas[i] >= 1.0 and lambdas[i + 1] < 1.0:
                # Linear interpolation
                t_lo, t_hi = temps[i], temps[i + 1]
                l_lo, l_hi = lambdas[i], lambdas[i + 1]
                tc = t_lo + (1.0 - l_lo) * (t_hi - t_lo) / (l_hi - l_lo)
                return {
                    "tc_bse": float(tc),
                    "tc_confidence": "interpolated",
                    "lambda_at_lowest_T": float(lambda_max),
                }

        return {
            "tc_bse": float(t_min),
            "tc_confidence": "lower_bound",
            "lambda_at_lowest_T": float(lambda_max),
        }

    # Case 2: λ < 1 at all temperatures — extrapolate
    # Fit 1/λ(T) = a + b·T (linear in T near Tc)
    # Tc ≈ (1 - a) / b
    if lambda_max > 0.3:
        inv_lambda = 1.0 / lambdas
        # Linear fit: 1/λ = a + b·T
        coeffs = np.polyfit(temps, inv_lambda, 1)
        b, a = coeffs
        if b > 0:
            tc_extrap = (1.0 - a) / b
            if tc_extrap > 0 and tc_extrap < temps[-1] * 2:
                return {
                    "tc_bse": float(tc_extrap),
                    "tc_confidence": "extrapolated",
                    "lambda_at_lowest_T": float(lambda_max),
                    "extrapolation_slope": float(b),
                }

    # Case 3: λ too small — no SC instability at accessible temperatures
    return {
        "tc_bse": 0.0,
        "tc_confidence": "no_instability" if lambda_max < 0.1 else "weak_instability",
        "lambda_at_lowest_T": float(lambda_max),
    }


# ── Full pairing susceptibility pipeline ─────────────────────────────────────

def run_pairing_susceptibility(
    bse_results_path: str,
    dmft_h5_path: str,
    bundle_data: dict,
    work_dir: str,
    beta: float = 40.0,
) -> dict:
    """
    Full pairing susceptibility pipeline:
      1. Load Γ_singlet from BSE results
      2. Load Σ(iω) from converged DMFT
      3. Compute lattice G(k, iω)
      4. Build pairing kernel
      5. Solve eigenvalue problem
      6. Classify gap symmetry
      7. Return results

    Args:
        bse_results_path: path to bse_results.npz
        dmft_h5_path:     path to converged DMFT HDF5
        bundle_data:      bundle dict with hk, kpoints, corr_shells, etc.
        work_dir:         output directory
        beta:             inverse temperature for this calculation

    Returns:
        dict with lambda_pair, gap_symmetry, Tc estimate, diagnostics
    """
    import h5py

    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()

    # 1. Load BSE results
    print(f"[Pairing] Loading BSE results from {bse_results_path}")
    bse = np.load(bse_results_path, allow_pickle=True)
    gamma_singlet = bse["gamma_singlet"]
    grid_params = json.loads(str(bse["grid_params_json"]))
    n_iw_f = grid_params["n_iw_f"]
    n_orb = grid_params["n_orb"]

    print(f"[Pairing] Γ_singlet shape: {gamma_singlet.shape}, "
          f"n_iw_f={n_iw_f}, n_orb={n_orb}")

    # 2. Load self-energy from converged DMFT
    sigma_iw = None
    mu = bundle_data.get("fermi_energy", 0.0)

    try:
        with h5py.File(dmft_h5_path, "r") as f:
            dmft_grp = f.get("DMFT_results")
            if dmft_grp:
                iter_keys = sorted([k for k in dmft_grp.keys() if k.startswith("it_")])
                if iter_keys:
                    last = dmft_grp[iter_keys[-1]]
                    for path in ["Sigma_iw", "solver/Sigma_iw"]:
                        if path in last:
                            for block in last[path].keys():
                                if "data" in last[path][block]:
                                    sigma_iw = last[path][block]["data"][()]
                                    break
                            break
                    if "mu" in last:
                        mu = float(last["mu"][()])
    except Exception as e:
        print(f"[Pairing] WARNING: Failed to load Σ(iω): {e}")

    if sigma_iw is None:
        # Fallback: zero self-energy (non-interacting limit)
        print("[Pairing] WARNING: No self-energy found, using Σ=0 (non-interacting)")
        sigma_iw = np.zeros((2 * n_iw_f, n_orb, n_orb), dtype=complex)

    # 3. Compute lattice Green's function
    hk = bundle_data["hk"]
    kpoints = bundle_data["kpoints"]
    n_k = hk.shape[0]

    print(f"[Pairing] Computing G(k, iω) on {n_k} k-points...")
    # Match sigma frequency count to n_iw_f
    n_iw_sigma = sigma_iw.shape[0] // 2
    n_iw_gf = max(n_iw_f, n_iw_sigma)
    # Pad or truncate sigma to match n_iw_gf
    if n_iw_sigma < n_iw_gf:
        sigma_padded = np.zeros((2 * n_iw_gf, n_orb, n_orb), dtype=complex)
        offset = n_iw_gf - n_iw_sigma
        sigma_padded[offset:offset + 2 * n_iw_sigma] = sigma_iw
        sigma_iw = sigma_padded
    elif n_iw_sigma > n_iw_gf:
        offset = (n_iw_sigma - n_iw_gf)
        sigma_iw = sigma_iw[offset:offset + 2 * n_iw_gf]

    gk_iw = compute_lattice_gf_fast(hk, sigma_iw, mu, beta, n_iw_gf)
    print(f"[Pairing] G(k, iω) shape: {gk_iw.shape}")

    # 4. Build pairing kernel
    print("[Pairing] Constructing pairing matrix...")
    kernel = construct_pairing_matrix(
        gk_iw, gamma_singlet, kpoints, n_iw_f, n_orb, beta,
    )

    # 5. Solve eigenvalue problem
    n_evals = min(10, n_k * 2 * n_iw_f * n_orb * n_orb - 2)
    n_evals = max(n_evals, 3)

    eigen_result = solve_pairing_eigenvalue(
        kernel, n_k, n_iw_f, n_orb, kpoints, n_evals=n_evals,
    )

    elapsed = time.time() - t0

    # 6. Compile results
    results = {
        "converged": True,
        "lambda_pair": eigen_result["lambda_pair"],
        "gap_symmetry": eigen_result["gap_symmetry"]["symmetry"],
        "gap_overlap": eigen_result["gap_symmetry"]["overlap"],
        "gap_all_overlaps": eigen_result["gap_symmetry"]["all_overlaps"],
        "gap_nodes": eigen_result["gap_symmetry"]["nodes"],
        "is_unconventional": eigen_result["gap_symmetry"]["is_unconventional"],
        "eigenvalues_real": [ev.real for ev in eigen_result["eigenvalues"]],
        "beta": beta,
        "temperature_K": 11604.5 / beta,  # 1 eV = 11604.5 K
        "n_k": n_k,
        "n_iw_f": n_iw_f,
        "n_orb": n_orb,
        "total_dim": eigen_result["total_dim"],
        "elapsed_seconds": elapsed,
    }

    # Save
    results_path = os.path.join(work_dir, "pairing_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Pairing] Results saved to {results_path}")

    return results
