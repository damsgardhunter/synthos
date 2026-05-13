#!/usr/bin/env python3
"""
D2: Charge Self-Consistent DFT+DMFT Loop.

In standard DFT+DMFT, the DFT Hamiltonian H(k) is computed once and
then DMFT runs on top. But the DMFT self-energy changes the electronic
density, which should feed back into the DFT charge density and update
H(k). This loop:

  DFT → H(k) → DMFT → ρ_DMFT(r) → DFT(ρ_new) → H'(k) → DMFT → ...

is called charge self-consistency (CSC). It matters when:
  - DMFT significantly redistributes charge between orbitals
  - The structural/electronic properties depend on the charge density
    (e.g., Mott transition, orbital polarization in nickelates)
  - You want quantitatively accurate Tc predictions

Implementation strategy:
  1. Run QE SCF with current charge density → H(k) via Wannier90
  2. Run DMFT with H(k) → Σ(K,iω), new occupations n_DMFT
  3. Construct updated density matrix from DMFT Green's function
  4. Feed back to QE as starting charge density
  5. Check density convergence: ||ρ_new - ρ_old|| < tol → stop
  6. Mix densities: ρ_next = α·ρ_old + (1-α)·ρ_new

The outer loop typically converges in 5-15 iterations.
Each iteration costs one QE SCF (~minutes) + one DMFT (~hours).

solid_dmft has experimental CSC support, but we implement our own
loop for tighter control over the QE↔DMFT interface and to reuse
the existing QE worker infrastructure.

References:
  Savrasov et al., PRL 87, 216405 (2001) — first CSC DFT+DMFT
  Haule et al., PRB 81, 195107 (2010) — eDMFT CSC
  Aichhorn et al., PRB 84, 054529 (2011) — CSC with TRIQS
  Park et al., PRB 89, 245133 (2014) — CSC for nickelates
"""

import numpy as np
import json
import os
import time
from typing import Optional, Dict

# ── Density matrix from DMFT Green's function ───────────────────────────────


def compute_density_matrix_from_gf(
    g_iw: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Compute the orbital density matrix from the Matsubara Green's function:

      n_{ab} = <c†_a c_b> = lim_{τ→0⁻} G_{ab}(τ)
             = (1/β) Σ_n G_{ab}(iω_n) · e^{iω_n·0⁺}
             = (1/β) Σ_n G_{ab}(iω_n) + tail correction

    The tail of G(iω) → 1/iω for large ω, so we need to sum carefully:
      n_{ab} = δ_{ab}/2 + (1/β) Σ_n [G_{ab}(iω_n) - δ_{ab}/(iω_n)]

    Args:
        g_iw: G(iω), shape [2*n_iw, n_orb, n_orb] or [nc, 2*n_iw, n_orb, n_orb]
        beta: inverse temperature

    Returns:
        density_matrix: [n_orb, n_orb] or [nc, n_orb, n_orb] real
    """
    if g_iw.ndim == 4:
        # Cluster: [nc, 2*n_iw, n_orb, n_orb]
        nc = g_iw.shape[0]
        n_iw = g_iw.shape[1] // 2
        n_orb = g_iw.shape[2]
        wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(2 * n_iw)])

        dens = np.zeros((nc, n_orb, n_orb))
        for ic in range(nc):
            eye = np.eye(n_orb)
            tail_sum = np.zeros((n_orb, n_orb), dtype=complex)
            for iw in range(2 * n_iw):
                tail_sum += g_iw[ic, iw] - eye / (1j * wn[iw])
            dens[ic] = (eye / 2.0 + (tail_sum / beta).real)
        return dens

    elif g_iw.ndim == 3:
        # Single-site: [2*n_iw, n_orb, n_orb]
        n_iw = g_iw.shape[0] // 2
        n_orb = g_iw.shape[1]
        wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(2 * n_iw)])

        eye = np.eye(n_orb)
        tail_sum = np.zeros((n_orb, n_orb), dtype=complex)
        for iw in range(2 * n_iw):
            tail_sum += g_iw[iw] - eye / (1j * wn[iw])
        return (eye / 2.0 + (tail_sum / beta).real)

    elif g_iw.ndim == 2:
        # Single-orbital cluster: [nc, 2*n_iw] — DCA single-band output
        nc = g_iw.shape[0]
        n_iw = g_iw.shape[1] // 2
        wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(2 * n_iw)])

        # Return [nc, 1, 1] for consistency with multi-orbital cluster
        dens = np.zeros((nc, 1, 1))
        for ic in range(nc):
            tail_sum = 0.0 + 0.0j
            for iw in range(2 * n_iw):
                tail_sum += g_iw[ic, iw] - 1.0 / (1j * wn[iw])
            dens[ic, 0, 0] = 0.5 + (tail_sum / beta).real
        return dens

    elif g_iw.ndim == 1:
        # Single-orbital single-site: [2*n_iw]
        n_iw = g_iw.shape[0] // 2
        wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(2 * n_iw)])

        tail_sum = 0.0 + 0.0j
        for iw in range(2 * n_iw):
            tail_sum += g_iw[iw] - 1.0 / (1j * wn[iw])
        # Return [1, 1] for consistency
        return np.array([[0.5 + (tail_sum / beta).real]])

    else:
        raise ValueError(f"Unexpected g_iw shape: {g_iw.shape}")


# ── High-order Matsubara tail handling ──────────────────────────────────────

def fit_gf_tail_moments(
    g_iw: np.ndarray,
    beta: float,
    n_fit: int = 30,
) -> tuple:
    """
    Fit the high-frequency tail of G(iω) to extract moments c1, c2, c3:

      G(iω) ≈ c1/iω + c2/(iω)² + c3/(iω)³ + O(1/ω⁴)

    Physical interpretation:
      c1 = 1                        (canonical anticommutator)
      c2 = ε_imp + Σ_∞ - μ          (effective level position)
      c3 = c2² + <Σ²>               (second-moment of spectrum)

    For impurity G, c1=1 always; we still fit it as a sanity check.

    Args:
        g_iw: Green's function, shape [..., 2*n_iw] where last axis is frequency
              (any leading orbital/cluster axes are preserved).
        beta: inverse temperature
        n_fit: number of highest-|ω| Matsubara points used for the fit on each side

    Returns:
        (c1, c2, c3) each with shape g_iw.shape[:-1]
        For multi-orbital G_{ab}(iω), use this on diagonal entries.
    """
    # g_iw expected shape: [..., 2*n_iw]; we operate on last axis
    n_iw = g_iw.shape[-1] // 2
    wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(2 * n_iw)])

    # Use highest |ω_n| frequencies on both ends
    n_fit = min(n_fit, n_iw - 1)
    idx_top = list(range(2 * n_iw - n_fit, 2 * n_iw))  # large positive ω
    idx_bot = list(range(0, n_fit))                     # large negative ω
    idx = np.array(idx_bot + idx_top)
    w = wn[idx]                                         # real frequencies
    G_high = g_iw[..., idx]                             # values to fit

    # G(iω_n) ≈ c1/iω_n + c2/(iω_n)² + c3/(iω_n)³
    # In real arithmetic (ω real):
    #   Re G(iω) ≈ -c2/ω²   (c1, c3 are imaginary-only contributions)
    #   Im G(iω) ≈ -c1/ω + c3/ω³
    # Fit Re G ∝ -1/ω² → c2
    # Fit Im G against (-1/ω, +1/ω³) → c1, c3
    inv_w2 = 1.0 / w**2
    inv_w = 1.0 / w
    inv_w3 = 1.0 / w**3

    # c2 from Re part (least squares: Re G = -c2 inv_w2)
    re = G_high.real
    num = np.sum(-re * inv_w2[(None,) * (g_iw.ndim - 1) + (slice(None),)], axis=-1)
    den = np.sum(inv_w2 ** 2)
    c2 = num / (den + 1e-30)

    # c1, c3 from Im part: Im G = -c1·inv_w + c3·inv_w3
    # Solve linear least squares with two basis vectors
    A = np.stack([-inv_w, inv_w3], axis=-1)  # [2*n_fit, 2]
    AtA = A.T @ A                              # [2, 2]
    AtA_inv = np.linalg.inv(AtA + 1e-12 * np.eye(2))
    im = G_high.imag
    # Project: coeffs = AtA_inv @ A.T @ im (per orbital)
    AtIm = np.einsum("nb,...n->...b", A, im)  # [..., 2]
    coeffs = np.einsum("ab,...b->...a", AtA_inv, AtIm)
    c1 = coeffs[..., 0]
    c3 = coeffs[..., 1]

    return c1, c2, c3


def density_per_spin_with_tail(
    g_diag: np.ndarray,
    beta: float,
    fit_n: int = 20,
) -> float:
    """
    Compute n_↑ = <c†_a c_a> for a single diagonal orbital entry of G(iω) using
    a 2nd-order analytical tail correction.

    Standard 1st-order formula:
        n_↑ = 1/2 + (1/β) Σ_n Re G(iω_n)
    is exact only if the Matsubara sum extends to infinity. For finite n_iw,
    higher tail moments c2/(iω)², c3/(iω)³, ... leak in.

    The 2nd-order correction adds the c2/(iω)² piece exactly:
        n_↑ = 1/2 + (1/β) Σ_n Re[G(iω_n) - c2/(iω_n)²] - c2 · β/4
    where c2 = ε_imp + Σ_∞ - μ is fitted from the high-|ω| tail of G.

    The exact analytic identity used:
        Σ_{n=-∞}^∞ 1/(iω_n)² = -β²/4   (fermionic ω_n)

    Typical accuracy gain: 3-10% error → <1% for far-from-Fermi orbitals.

    Args:
        g_diag: 1-D array of G_{aa}(iω_n) values, shape [2*n_iw]
        beta:   inverse temperature
        fit_n:  number of high-|ω| points (per side) used to fit c2

    Returns:
        n_↑ as a float (real density per spin)
    """
    g_diag = np.asarray(g_diag).ravel()
    n_iw_total = g_diag.shape[0]
    n_iw = n_iw_total // 2
    wn = np.array([(2 * (n - n_iw) + 1) * np.pi / beta for n in range(n_iw_total)])

    # 1) Fit c2 from Re G(iω) ≈ -c2/ω² at large |ω|
    n_fit = max(2, min(fit_n, n_iw - 1))
    idx = np.concatenate([np.arange(n_fit), np.arange(n_iw_total - n_fit, n_iw_total)])
    w = wn[idx]
    re = g_diag[idx].real
    inv_w2 = 1.0 / w**2
    c2 = -float(np.sum(re * inv_w2) / np.sum(inv_w2 ** 2))

    # 2) Tail-corrected density. Derivation:
    #    n_↑ = 1/2 + (1/β) Σ_n Re G(iω_n) + c2 · [(1/β) Σ_{|n|≤n_iw} 1/ω_n² - β/4]
    #    The first part is the existing simple formula; the c2 term is the
    #    correction from the missing |n|>n_iw tail of c2/(iω_n)².
    naive = 0.5 + np.sum(g_diag.real) / beta
    finite_sum = float(np.sum(1.0 / wn**2)) / beta     # (1/β) Σ_{|n|≤n_iw} 1/ω_n²
    exact_tail = beta / 4.0                              # (1/β) Σ_{n=-∞}^∞ 1/ω_n² = β/4
    c2_correction = c2 * (finite_sum - exact_tail)

    return float(naive + c2_correction)


def density_per_spin_dlr(
    g_diag: np.ndarray,
    beta: float,
    wmax: float = 8.0,
    eps: float = 1e-7,
) -> float:
    """
    Density extraction via Discrete Lehmann Representation (DLR/IR basis).

    Fits G(iω_n) into the IR basis, then evaluates G(τ=β⁻) which directly
    gives n_↑ = -G(β⁻) WITHOUT any Matsubara truncation error or tail-fit
    approximation. The IR basis spans the full analytic structure of a
    physical fermionic Green's function.

    This is the "gold-standard" density calculation when sparse_ir is
    available. Falls back to density_per_spin_with_tail if it's not.

    Args:
        g_diag: 1-D array of G_{aa}(iω_n) on the canonical fermionic
            Matsubara grid: ω_n = (2(n-n_iw)+1)π/β for n=0..2*n_iw-1
        beta:   inverse temperature
        wmax:   spectral support cutoff (must contain all of G's spectral weight)
        eps:    IR basis accuracy (smaller → larger basis, more precise)

    Returns:
        n_↑ as a float.
    """
    try:
        from dlr_basis import DLRBasis, is_available
    except ImportError:
        return density_per_spin_with_tail(g_diag, beta)

    if not is_available():
        return density_per_spin_with_tail(g_diag, beta)

    g_diag = np.asarray(g_diag).ravel()
    n_iw_total = g_diag.shape[0]
    n_iw = n_iw_total // 2
    matsu_idx = np.array(
        [2 * (n - n_iw) + 1 for n in range(n_iw_total)], dtype=int
    )

    try:
        dlr = DLRBasis(beta=beta, wmax=wmax, eps=eps)
        n_up = dlr.density_from_matsubara(g_diag, matsu_idx)
        return float(n_up)
    except Exception:
        # Any DLR failure → fall back to the analytical tail correction
        return density_per_spin_with_tail(g_diag, beta)


# ── QE density update ───────────────────────────────────────────────────────

def split_density_by_corr_shells(
    density_matrix: np.ndarray,
    corr_shells: list,
) -> list:
    """
    Split a global [n_orb, n_orb] density matrix into per-correlated-atom blocks.

    Given corr_shells = [{atom: 0, dim: 5}, {atom: 1, dim: 5}, ...] and a global
    density matrix of total size sum(dim), this returns a list of per-shell
    [dim_shell, dim_shell] sub-matrices.

    Args:
        density_matrix: [n_orb_total, n_orb_total] complex/real
        corr_shells: list of {"atom": int, "dim": int, "sort": int, "l": int}

    Returns:
        list of [dim_i, dim_i] density blocks, one per shell
    """
    blocks = []
    offset = 0
    for shell in corr_shells:
        dim = int(shell.get("dim", 5))
        blocks.append(density_matrix[offset:offset + dim, offset:offset + dim])
        offset += dim
    return blocks


def write_qe_density_correction(
    density_matrix_dmft: np.ndarray,
    density_matrix_dft: np.ndarray,
    orbital_labels: list,
    output_path: str,
    corr_shells: list = None,
):
    """
    Write the density correction Δn = n_DMFT - n_DFT for QE to read.

    QE can read an external occupation matrix via the starting_ns_eigenvalue
    card (for DFT+U) or via a custom density restart. We write both formats.

    The correction is applied to the correlated subspace occupations.
    Non-correlated orbitals keep their DFT values.

    Args:
        density_matrix_dmft: [n_orb, n_orb] from DMFT Green's function
        density_matrix_dft:  [n_orb, n_orb] from DFT (Wannier projection)
        orbital_labels: e.g., ["d_x2y2", "p_x", "p_y"]
        output_path: where to write the correction file
        corr_shells: optional list of correlated shells for multi-atom CSC.
                     If provided, the JSON output includes per-atom blocks.
    """
    delta_n = density_matrix_dmft - density_matrix_dft
    n_orb = delta_n.shape[0]

    def _to_serializable(arr):
        """Convert complex ndarray to JSON-safe nested list of [re, im] pairs."""
        a = np.asarray(arr)
        if np.iscomplexobj(a):
            return [[[float(v.real), float(v.imag)] for v in row] for row in a]
        return a.tolist()

    data = {
        "n_orb": n_orb,
        "orbital_labels": orbital_labels,
        "density_matrix_dmft": _to_serializable(density_matrix_dmft),
        "density_matrix_dft": _to_serializable(density_matrix_dft),
        "delta_n": _to_serializable(delta_n),
        "trace_dmft": float(np.trace(density_matrix_dmft).real),
        "trace_dft": float(np.trace(density_matrix_dft).real),
        "max_correction": float(np.max(np.abs(delta_n))),
    }

    # Per-atom blocks for multi-atom CSC (e.g., YBCO bilayer with 2 Cu sites)
    if corr_shells is not None and len(corr_shells) > 0:
        per_atom_blocks = []
        offset = 0
        for shell in corr_shells:
            dim = int(shell.get("dim", 5))
            atom = int(shell.get("atom", 0))
            block = np.asarray(
                density_matrix_dmft[offset:offset + dim, offset:offset + dim]
            )
            per_atom_blocks.append({
                "atom": atom,
                "shell_l": int(shell.get("l", 2)),  # 2=d, 3=f
                "dim": dim,
                "occupation_matrix": _to_serializable(block),
                "occupations_eigenvalues": np.linalg.eigvalsh(
                    0.5 * (block + block.conj().T)
                ).real.tolist(),
            })
            offset += dim
        data["corr_shells"] = [dict(s) for s in corr_shells]
        data["per_atom_density"] = per_atom_blocks

    # Write JSON for the QAE pipeline to read
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Write QE starting_ns_eigenvalue format for DFT+U restart
    occ_path = output_path.replace(".json", "_qe_occ.dat")
    with open(occ_path, "w") as f:
        f.write(f"# DMFT orbital occupations for QE restart\n")
        f.write(f"# n_orb = {n_orb}\n")
        evals = np.linalg.eigvalsh(density_matrix_dmft)
        for i, ev in enumerate(evals):
            f.write(f"  {ev:.8f}\n")


def generate_qe_scf_with_dmft_density(
    base_scf_input: str,
    density_correction_path: str,
    iteration: int,
) -> str:
    """
    Modify a QE SCF input to incorporate DMFT density feedback.

    Strategy:
      1. Set startingpot = 'file' to read density from previous SCF
      2. Update starting_magnetization if DMFT changes spin polarization
      3. Ensure disk_io = 'high' so we can extract the new H(k)
      4. Set electron_maxstep higher for CSC stability

    Args:
        base_scf_input: the original QE SCF input string
        density_correction_path: path to the density correction JSON
        iteration: CSC iteration number

    Returns:
        Modified QE SCF input string
    """
    lines = base_scf_input.split("\n")
    modified = []

    # Load DMFT density correction if available
    dmft_occupations = None
    if density_correction_path and os.path.exists(density_correction_path):
        try:
            with open(density_correction_path) as f:
                dc = json.load(f)
            dmft_occupations = dc.get("density_matrix_dmft")
        except Exception:
            pass

    for line in lines:
        stripped = line.strip().lower()

        # For iterations > 0, read density from previous run
        if "startingpot" in stripped:
            if iteration > 0:
                modified.append("  startingpot = 'file',")
            else:
                modified.append(line)
        elif "electron_maxstep" in stripped:
            modified.append("  electron_maxstep = 200,")
        elif "conv_thr" in stripped and "forc_conv" not in stripped:
            # Tighter convergence for CSC
            modified.append("  conv_thr = 1.0d-8,")
        else:
            modified.append(line)

    # Add disk_io if not present
    if not any("disk_io" in l.lower() for l in modified):
        for i, line in enumerate(modified):
            if "&CONTROL" in line.upper():
                modified.insert(i + 1, "  disk_io = 'high',")
                break

    # Inject DMFT occupations via starting_ns_eigenvalue (DFT+U/Hubbard framework)
    # This is the standard CSC feedback mechanism: DMFT orbital occupations
    # override the DFT starting guess for the correlated subspace.
    #
    # Multi-atom support: if the density_correction JSON contains per_atom_density,
    # emit a separate starting_ns_eigenvalue block per correlated atom (1-based atom
    # index in QE). Otherwise fall back to single-atom mode (atom=1).
    per_atom_density = None
    if density_correction_path and os.path.exists(density_correction_path):
        try:
            with open(density_correction_path) as f:
                dc = json.load(f)
            per_atom_density = dc.get("per_atom_density")
        except Exception:
            pass

    if dmft_occupations is not None and iteration > 0:
        # Ensure lda_plus_u is enabled (required for starting_ns_eigenvalue)
        has_hubbard = any("lda_plus_u" in l.lower() for l in modified)
        if not has_hubbard:
            for i, line in enumerate(modified):
                if "&SYSTEM" in line.upper():
                    modified.insert(i + 1, "  lda_plus_u = .true.,")
                    break

        # Build starting_ns_eigenvalue cards. Use per-atom data if available,
        # else single-atom (atom=1) fallback for backward compatibility.
        ns_lines = []

        if per_atom_density and len(per_atom_density) > 0:
            # Multi-atom mode: emit one block per correlated atom.
            # QE uses 1-based atom indexing.
            for shell in per_atom_density:
                atom_idx = int(shell.get("atom", 0)) + 1  # 0-based → 1-based
                dim = int(shell.get("dim", 5))
                evals = shell.get("occupations_eigenvalues", [])
                for m in range(min(dim, len(evals))):
                    occ = float(np.clip(evals[m], 0.0, 1.0))
                    # Format: starting_ns_eigenvalue(m, ispin, atom_index)
                    ns_lines.append(
                        f"  starting_ns_eigenvalue({m+1},1,{atom_idx}) = {occ:.6f},"
                    )
                    ns_lines.append(
                        f"  starting_ns_eigenvalue({m+1},2,{atom_idx}) = {occ:.6f},"
                    )
        else:
            # Single-atom fallback (backward compat)
            n_orb = len(dmft_occupations)
            occ_evals = np.linalg.eigvalsh(
                0.5 * (np.array(dmft_occupations) + np.array(dmft_occupations).conj().T)
            )
            for m in range(n_orb):
                occ = float(np.clip(occ_evals[m], 0.0, 1.0))
                ns_lines.append(f"  starting_ns_eigenvalue({m+1},1,1) = {occ:.6f},")
                ns_lines.append(f"  starting_ns_eigenvalue({m+1},2,1) = {occ:.6f},")

        # Insert before the closing / of &SYSTEM
        for i in range(len(modified) - 1, -1, -1):
            if modified[i].strip() == "/" and i > 0:
                # Check if this / closes &SYSTEM (find the preceding & block)
                for j in range(i - 1, -1, -1):
                    if "&SYSTEM" in modified[j].upper():
                        for k, ns_line in enumerate(ns_lines):
                            modified.insert(i + k, ns_line)
                        break
                break

    return "\n".join(modified)


# ── CSC outer loop ───────────────────────────────────────────────────────────

class CSCParams:
    """Parameters for the charge self-consistency loop."""

    def __init__(
        self,
        max_iterations: int = 15,
        density_mix: float = 0.3,
        density_tol: float = 1e-4,
        # Whether to do Wannier90 re-projection at each step
        # (expensive but captures H(k) changes beyond occupations)
        reproject_wannier: bool = True,
        # Whether to do full DCA or single-site DMFT in the inner loop
        use_dca: bool = False,
        dca_nc: int = 4,
    ):
        self.max_iterations = max_iterations
        self.density_mix = density_mix
        self.density_tol = density_tol
        self.reproject_wannier = reproject_wannier
        self.use_dca = use_dca
        self.dca_nc = dca_nc


def run_csc_loop(
    bundle_data: dict,
    dmft_config: dict,
    csc_params: CSCParams,
    work_dir: str,
    qe_callback=None,
    wannier_callback=None,
) -> dict:
    """
    Run the charge self-consistent DFT+DMFT outer loop.

    Outer loop:
      1. QE SCF with current density → charge density, band structure
      2. Wannier90 projection → H(k) for DMFT
      3. DMFT (single-site or DCA) → Σ(iω), G(iω), new density matrix
      4. Compute density correction Δn = n_DMFT - n_DFT
      5. Update charge density: ρ_next = mix(ρ_old, ρ_DMFT)
      6. Check convergence → repeat or stop

    Args:
        bundle_data: DMFT bundle dict (hk, corr_shells, etc.)
        dmft_config: solid_dmft config dict
        csc_params: CSC parameters
        work_dir: output directory
        qe_callback: async function(scf_input, work_dir) → scf_output
            Runs QE SCF. If None, H(k) is reused without QE update.
        wannier_callback: async function(scf_dir) → hk_new
            Re-runs Wannier90 projection. If None, H(k) unchanged.

    Returns:
        dict with convergence history, final density, final Σ
    """
    os.makedirs(work_dir, exist_ok=True)

    t0 = time.time()
    n_orb = bundle_data["hk"].shape[1]
    beta = dmft_config.get("general", {}).get("beta", 40.0)

    hk_current = bundle_data["hk"].copy()
    sigma_current = None  # Warm-start from previous iteration

    convergence_history = []
    density_history = []
    density_matrix_prev = None

    # Reference DFT density matrix for the double-counting correction
    # Δn = n_DMFT - n_DFT. We keep this FIXED across iterations so the
    # diagnostic delta_n actually measures the DMFT-vs-DFT discrepancy
    # (not the iteration-to-iteration mixing residual).
    #
    # Priority for the reference:
    #   1. bundle_data["density_matrix_dft"] if provided (best — from
    #      DFT/Wannier projection of the initial SCF)
    #   2. Identity * (filling/2) as a non-informative fallback
    #
    # If the bundle didn't carry an explicit DFT density (older bundles),
    # we capture the FIRST iteration's DMFT density as the reference — this
    # is not the true DFT density but it stabilizes the diagnostic; the
    # zeroth iteration's DMFT-from-DFT-Σ₀ is the closest available proxy.
    density_matrix_dft_ref = bundle_data.get("density_matrix_dft")
    if density_matrix_dft_ref is not None:
        density_matrix_dft_ref = np.asarray(density_matrix_dft_ref)
        print(f"[CSC] Using bundle DFT density as reference for Δn (trace={float(np.trace(density_matrix_dft_ref).real):.3f})")
    else:
        print("[CSC] No DFT density in bundle — will capture iteration-0 DMFT density as proxy reference")

    print(f"\n{'='*60}")
    print(f"[CSC] Charge Self-Consistent DFT+DMFT Loop")
    print(f"[CSC] n_orb={n_orb}, max_iter={csc_params.max_iterations}, "
          f"mix={csc_params.density_mix}, tol={csc_params.density_tol}")
    print(f"{'='*60}")

    for iteration in range(csc_params.max_iterations):
        iter_dir = os.path.join(work_dir, f"csc_iter_{iteration:03d}")
        os.makedirs(iter_dir, exist_ok=True)
        t_iter = time.time()

        print(f"\n[CSC] === Iteration {iteration+1}/{csc_params.max_iterations} ===")

        # ── Step 1: QE SCF (if callback available) ───────────────────
        if qe_callback is not None and iteration > 0:
            print(f"[CSC] Running QE SCF with updated density...")
            try:
                qe_result = qe_callback(iter_dir)
                if qe_result.get("converged"):
                    print(f"[CSC] QE SCF converged: E={qe_result.get('energy', '?')}")
                else:
                    print(f"[CSC] QE SCF did not converge — using previous H(k)")
            except Exception as e:
                print(f"[CSC] QE SCF failed: {e} — using previous H(k)")

        # ── Step 2: Wannier90 re-projection (if callback available) ──
        if wannier_callback is not None and csc_params.reproject_wannier and iteration > 0:
            print(f"[CSC] Re-projecting Wannier90...")
            try:
                hk_new = wannier_callback(iter_dir)
                if hk_new is not None and hk_new.shape == hk_current.shape:
                    hk_current = hk_new
                    print(f"[CSC] H(k) updated from new Wannier projection")
                else:
                    print(f"[CSC] Wannier re-projection returned incompatible H(k)")
            except Exception as e:
                print(f"[CSC] Wannier re-projection failed: {e}")

        # ── Step 3: Run DMFT with current H(k) ──────────────────────
        bundle_data_iter = dict(bundle_data)
        bundle_data_iter["hk"] = hk_current

        if csc_params.use_dca:
            from multiorbital_dca import (
                MultiOrbitalDCAParams, KanamoriInteraction,
                run_multiorbital_dca, compute_lattice_gf_multiorbital_fast,
                assign_k_to_patches,
            )

            U_vals = bundle_data.get("U_values", np.ones(n_orb) * 4.0)
            J_vals = bundle_data.get("J_values", np.ones(n_orb) * 0.7)

            interaction = KanamoriInteraction(n_orb, U_vals, J_vals)
            dca_params = MultiOrbitalDCAParams(
                n_orb=n_orb,
                nc=csc_params.dca_nc,
                interaction=interaction,
                beta=beta,
                mu=bundle_data.get("fermi_energy", 0.0),
                n_iw=128,
                n_k_per_dim=32,
                n_iter=20,
                sigma_mix=0.5,
            )

            from dca_solver import build_lattice_kpoints
            kpoints = build_lattice_kpoints(dca_params.n_k_per_dim, dim=2)
            patch_assignment = assign_k_to_patches(kpoints, dca_params.K_cluster)

            dca_result = run_multiorbital_dca(
                hk=hk_current, kpoints=kpoints,
                params=dca_params, work_dir=os.path.join(iter_dir, "dca"),
                initial_sigma=sigma_current,
            )

            g_c = dca_result.get("g_c")  # [nc, n_w, n_orb, n_orb]
            sigma_current = dca_result.get("sigma_c")
            dmft_converged = dca_result.get("converged", False)

            # Average density matrix over patches
            density_matrix_dmft = compute_density_matrix_from_gf(g_c, beta)
            density_matrix_avg = np.mean(density_matrix_dmft, axis=0)

        else:
            # Single-site DMFT via run-dmft pipeline
            from run_dmft import build_solid_dmft_config, run_solid_dmft, write_solid_dmft_inputs

            config = build_solid_dmft_config(bundle_data_iter, iter_dir)
            write_solid_dmft_inputs(bundle_data_iter, config, iter_dir)
            dmft_result = run_solid_dmft(iter_dir, config)
            dmft_converged = dmft_result.get("converged", False)

            # Extract G(iω) from DMFT output
            import h5py
            seedname = config["general"]["seedname"]
            h5_path = os.path.join(iter_dir, f"{seedname}.h5")
            g_iw = None
            try:
                with h5py.File(h5_path, "r") as f:
                    dmft_grp = f.get("DMFT_results")
                    if dmft_grp:
                        iters = sorted([k for k in dmft_grp.keys() if k.startswith("it_")])
                        if iters:
                            last = dmft_grp[iters[-1]]
                            for path in ["G_iw", "solver/G_iw", "Gimp_iw"]:
                                if path in last:
                                    for block in last[path].keys():
                                        if "data" in last[path][block]:
                                            g_iw = last[path][block]["data"][()]
                                            break
                                    break
            except Exception as e:
                print(f"[CSC] Failed to extract G(iω): {e}")

            if g_iw is not None:
                density_matrix_avg = compute_density_matrix_from_gf(g_iw, beta)
            else:
                density_matrix_avg = np.eye(n_orb) * 0.5
                print(f"[CSC] WARNING: Could not extract density matrix, using default")

        # ── Step 4: Compute density correction and mix ────────────────
        if density_matrix_prev is not None:
            delta_density = np.max(np.abs(density_matrix_avg - density_matrix_prev))
        else:
            delta_density = float("inf")

        convergence_history.append(float(delta_density))
        density_history.append(density_matrix_avg.tolist())

        # Mix densities FIRST — the mixed density is what gets fed back to QE
        if density_matrix_prev is not None:
            density_matrix_mixed = (
                csc_params.density_mix * density_matrix_prev +
                (1 - csc_params.density_mix) * density_matrix_avg
            )
        else:
            density_matrix_mixed = density_matrix_avg

        density_matrix_prev = density_matrix_mixed

        # Capture the iteration-0 DMFT density as DFT-proxy reference if the
        # bundle didn't supply a true DFT density (older bundles).
        if density_matrix_dft_ref is None:
            density_matrix_dft_ref = density_matrix_avg.copy()

        # Save MIXED density correction for the NEXT iteration's QE callback
        # (QE reads this via starting_ns_eigenvalue to seed the correlated occupations)
        # Pass corr_shells for multi-atom CSC: each correlated atom gets its own
        # starting_ns_eigenvalue block instead of all collapsing to atom=1.
        # The DFT reference is fixed across iterations so Δn = n_DMFT - n_DFT
        # is a meaningful diagnostic of double-counting drift (not just the
        # iteration mixing residual).
        write_qe_density_correction(
            density_matrix_mixed,
            density_matrix_dft_ref,
            [f"orb_{i}" for i in range(n_orb)],
            os.path.join(iter_dir, "density_correction.json"),
            corr_shells=bundle_data.get("corr_shells"),
        )

        elapsed_iter = time.time() - t_iter
        trace = np.trace(density_matrix_avg)
        print(f"[CSC] iter {iteration+1}: ||Δn||={delta_density:.2e}, "
              f"Tr(n)={trace:.4f}, DMFT_converged={dmft_converged}, "
              f"time={elapsed_iter:.0f}s")

        # ── Step 6: Check convergence ────────────────────────────────
        if delta_density < csc_params.density_tol and iteration > 0:
            print(f"[CSC] Density converged after {iteration+1} iterations")
            break

    elapsed_total = time.time() - t0
    converged = (convergence_history[-1] < csc_params.density_tol
                 if convergence_history else False)

    # Save final state
    results = {
        "converged": converged,
        "n_iterations": len(convergence_history),
        "convergence_history": convergence_history,
        "final_density_matrix": density_matrix_avg.tolist() if density_matrix_avg is not None else None,
        "final_trace": float(np.trace(density_matrix_avg)) if density_matrix_avg is not None else None,
        "elapsed_seconds": elapsed_total,
    }

    results_path = os.path.join(work_dir, "csc_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[CSC] Results saved to {results_path}")

    return results
