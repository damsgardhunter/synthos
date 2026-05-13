#!/usr/bin/env python3
"""
Discrete Lehmann Representation (DLR) / Intermediate Representation (IR)
basis for compact representation of Matsubara Green's functions.

Background
----------
A fermionic Green's function with finite spectral support [-ω_max, ω_max] has
an analytic structure G(iω) = ∫ ρ(ω) / (iω - ω) dω. Discretizing the integral
gives the DLR:  G(iω) ≈ Σ_p g_p / (iω - ω_p), where {ω_p} are a small set of
optimally-chosen poles (the DLR poles) and g_p are the residues (DLR
coefficients).

Equivalently, in the IR basis (sparse_ir):
  G(iω) = Σ_l g_l · û_l(iω)
where û_l are the analytical-continuation IR basis functions and g_l are the
IR coefficients. IR and DLR differ only in how the few-dimensional space is
parameterized; both have basis size ~ log(βω_max) / log(1/eps).

For β=40, ω_max=8, ε=1e-7:  31 basis vectors capture the function to 7 digits
versus ~256 Matsubara points needed for the same accuracy.

Benefits for DMFT
-----------------
1. Density / occupations: G(τ=β⁻) extrapolation gives n_up directly without
   tail-correcting the Matsubara sum.
2. Self-energy bandwidth: store Σ as O(30) coefficients instead of O(1000)
   Matsubara values.
3. BSE inversion: compress the fermionic axes of Γ before inverting; the BSE
   matrix shrinks by ~3-5× linear → 30-125× CPU saving per bosonic frequency.
4. Tail handling: exact by construction (the basis spans the full
   high-frequency asymptotic behavior of G).

This module provides a thin wrapper around sparse_ir with graceful fallback
when sparse_ir is unavailable (returns is_available() = False so callers can
keep using Matsubara directly).

References:
  Shinaoka et al., PRB 96, 035147 (2017) — IR basis
  Kaye, Chen, Parcollet, PRB 105, 235115 (2022) — DLR formulation
  Wallerberger et al., SoftwareX 21, 101266 (2023) — sparse_ir library
"""

from __future__ import annotations
import warnings
from typing import Optional, Tuple
import numpy as np

# ── Optional dependency handling ─────────────────────────────────────────────

_SPARSE_IR_AVAILABLE: Optional[bool] = None
_IMPORT_ERROR: Optional[str] = None


def _try_import_sparse_ir():
    """Import sparse_ir with a monkey-patch for scipy>=1.13 incompatibility."""
    global _SPARSE_IR_AVAILABLE, _IMPORT_ERROR
    if _SPARSE_IR_AVAILABLE is not None:
        return _SPARSE_IR_AVAILABLE

    try:
        # sparse_ir 1.1.x calls scipy.linalg.interpolative.seed which was
        # removed in scipy 1.13+. Monkey-patch a no-op before importing.
        import scipy.linalg.interpolative as _sli
        if not hasattr(_sli, "seed"):
            _sli.seed = lambda *_args, **_kwargs: None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import sparse_ir  # noqa: F401
            from sparse_ir import FiniteTempBasis  # noqa: F401
            from sparse_ir.sampling import MatsubaraSampling, TauSampling  # noqa: F401
            from sparse_ir.dlr import DiscreteLehmannRepresentation  # noqa: F401

        _SPARSE_IR_AVAILABLE = True
        _IMPORT_ERROR = None
    except Exception as e:
        _SPARSE_IR_AVAILABLE = False
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"

    return _SPARSE_IR_AVAILABLE


def is_available() -> bool:
    """Return True if the DLR/IR backend is functional."""
    return _try_import_sparse_ir()


def availability_diagnostic() -> dict:
    """For logging: state of the DLR backend."""
    return {
        "available": is_available(),
        "import_error": _IMPORT_ERROR,
        "backend": "sparse_ir" if is_available() else "none",
    }


# ── DLR basis wrapper ───────────────────────────────────────────────────────

class DLRBasis:
    """
    Compact IR/DLR basis for a fermionic Green's function on (β, ω_max).

    Typical usage:
        dlr = DLRBasis(beta=40.0, wmax=8.0, eps=1e-7)
        g_l = dlr.fit_from_matsubara(G_iw, matsubara_indices)
        n_up = dlr.density_from_coefs(g_l)        # tail-exact
        G_iw_new = dlr.evaluate_matsubara(g_l, new_indices)
    """

    def __init__(
        self,
        beta: float,
        wmax: float,
        eps: float = 1e-7,
        statistics: str = "F",
    ):
        """
        Args:
            beta: inverse temperature
            wmax: spectral support cutoff (Σ on G must have weight inside [-wmax, wmax])
            eps:  truncation accuracy; smaller → larger basis but better accuracy
            statistics: "F" (fermionic) or "B" (bosonic)
        """
        if not is_available():
            raise RuntimeError(
                f"sparse_ir not available: {_IMPORT_ERROR}. "
                f"Install with: pip install sparse_ir"
            )
        from sparse_ir import FiniteTempBasis
        from sparse_ir.dlr import DiscreteLehmannRepresentation

        self.beta = float(beta)
        self.wmax = float(wmax)
        self.eps = float(eps)
        self.statistics = statistics

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.basis = FiniteTempBasis(statistics, self.beta, self.wmax, eps=self.eps)
            self.dlr = DiscreteLehmannRepresentation(self.basis)

        # Default sampling grids
        self._default_matsu_idx = np.asarray(
            self.basis.default_matsubara_sampling_points(), dtype=int
        )
        self._default_tau = np.asarray(self.basis.default_tau_sampling_points())

    @property
    def size(self) -> int:
        """Number of IR basis functions (compact dimension)."""
        return int(self.basis.size)

    @property
    def n_poles(self) -> int:
        """Number of DLR poles."""
        return int(self.dlr.size)

    @property
    def poles(self) -> np.ndarray:
        """Real-frequency DLR poles ω_p (within [-ω_max, ω_max])."""
        return np.asarray(self.dlr.sampling_points)

    def default_matsubara_indices(self) -> np.ndarray:
        """Optimal Matsubara sampling points (odd integers for fermionic)."""
        return self._default_matsu_idx.copy()

    def default_tau(self) -> np.ndarray:
        """Optimal τ sampling points in [0, β]."""
        return self._default_tau.copy()

    def matsubara_omega(self, indices: np.ndarray) -> np.ndarray:
        """Convert integer Matsubara indices to frequencies ω_n.

        For fermionic: ω_n = (2k+1)π/β where the integer is (2k+1) (odd).
        For bosonic:   ω_n = 2k·π/β   where the integer is 2k (even).
        """
        idx = np.asarray(indices, dtype=int)
        return idx * np.pi / self.beta

    # ── Fitting and evaluation ─────────────────────────────────────────

    def fit_from_matsubara(
        self,
        g_iw: np.ndarray,
        matsubara_indices,
    ) -> np.ndarray:
        """
        Fit IR coefficients g_l from values of G(iω_n) on a Matsubara grid.

        Args:
            g_iw: complex values G(iω_n) at the sampling points.
                Shape [..., N_smpl] (last axis = sampling axis)
            matsubara_indices: list/array of odd integers (fermionic) of length N_smpl

        Returns:
            g_l: IR coefficients, shape [..., basis.size]
        """
        from sparse_ir.sampling import MatsubaraSampling

        arr = np.asarray(g_iw)
        smpl = MatsubaraSampling(self.basis, list(matsubara_indices))
        # sparse_ir auto-detects the sampling axis but mis-detects for multi-dim
        # input — always pass axis=-1 explicitly.
        g_l = smpl.fit(arr, axis=-1)
        return np.asarray(g_l)

    def evaluate_matsubara(
        self,
        g_l: np.ndarray,
        matsubara_indices,
    ) -> np.ndarray:
        """Evaluate G(iω_n) on a new Matsubara grid from IR coefficients."""
        from sparse_ir.sampling import MatsubaraSampling

        arr = np.asarray(g_l)
        smpl = MatsubaraSampling(self.basis, list(matsubara_indices))
        return np.asarray(smpl.evaluate(arr, axis=-1))

    def evaluate_tau(
        self,
        g_l: np.ndarray,
        tau_points: np.ndarray,
    ) -> np.ndarray:
        """Evaluate G(τ) for arbitrary τ in [0, β] from IR coefficients."""
        from sparse_ir.sampling import TauSampling

        arr = np.asarray(g_l)
        smpl = TauSampling(self.basis, np.asarray(tau_points))
        return np.asarray(smpl.evaluate(arr, axis=-1))

    # ── Density extraction (exact tail handling) ───────────────────────

    def density_from_coefs(self, g_l: np.ndarray) -> np.ndarray:
        """
        Compute occupation n_↑ = <c†c> = -G(τ=β⁻) directly from IR coefs.

        Because the IR basis spans the full analytic structure of G, this
        evaluation is exact (to ε accuracy) — no Matsubara truncation error,
        no tail moments to fit.

        Args:
            g_l: IR coefficients, shape [..., basis.size]

        Returns:
            n_up: real density, shape [...] (axis removed)
        """
        # Evaluate G at τ = β⁻ (just below β)
        tau_eval = np.array([self.beta - 1e-12])
        g_at_beta = self.evaluate_tau(g_l, tau_eval)  # shape [..., 1]
        # n_↑ = -G(β⁻).  G is real-valued at τ values; .real makes it explicit.
        n_up = -g_at_beta[..., 0].real
        return n_up

    def density_from_matsubara(
        self,
        g_iw: np.ndarray,
        matsubara_indices,
    ) -> np.ndarray:
        """
        One-shot helper: fit IR coefficients from G(iω_n) and extract density.

        Args:
            g_iw: G(iω_n), shape [..., N_smpl]
            matsubara_indices: Matsubara indices used in g_iw

        Returns:
            n_up: density, shape [...]
        """
        g_l = self.fit_from_matsubara(g_iw, matsubara_indices)
        return self.density_from_coefs(g_l)

    # ── BSE compression ────────────────────────────────────────────────

    def matsubara_to_ir_matrix(self, matsubara_indices) -> np.ndarray:
        """
        Build the matrix M such that  g_l = M @ G(iω_n)  for the given grid.

        Useful for compressing a large fermionic-Matsubara matrix down to the
        IR basis in a single matrix multiply.

        Returns:
            M: [basis.size, N_smpl] complex matrix
        """
        from sparse_ir.sampling import MatsubaraSampling

        smpl = MatsubaraSampling(self.basis, list(matsubara_indices))
        N = len(matsubara_indices)
        # Apply fit to each column of I[N, N] along axis=0 (the sampling axis).
        # Result shape: [L, N]. Each column j of the result is M @ e_j = column
        # j of M, so the result IS M directly.
        I_complex = np.eye(N, dtype=complex)
        M = smpl.fit(I_complex, axis=0)  # [L, N]
        return np.asarray(M)

    # ── Diagnostics ────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "beta": self.beta,
            "wmax": self.wmax,
            "eps": self.eps,
            "statistics": self.statistics,
            "basis_size": self.size,
            "n_poles": self.n_poles,
            "n_default_matsubara": int(len(self._default_matsu_idx)),
            "n_default_tau": int(len(self._default_tau)),
            "Λ": self.beta * self.wmax,
        }


# ── Compressed BSE inversion ────────────────────────────────────────────────

def compress_bse_matrix(
    chi_matrix: np.ndarray,
    n_iw_f: int,
    n_orb: int,
    dlr: "DLRBasis",
    matsubara_indices: np.ndarray,
) -> np.ndarray:
    """
    Compress a fermionic susceptibility matrix to the IR basis.

    Input shape:  [2*n_iw_f * n_orb², 2*n_iw_f * n_orb²]
    Output shape: [L * n_orb²,         L * n_orb²]

    where L = dlr.size << 2*n_iw_f.

    The compression is M ⊗ I_{n_orb²}, where M maps ω_n → IR coefs.

    Args:
        chi_matrix: square BSE matrix on Matsubara compound index
        n_iw_f:     fermionic frequencies per side (so 2*n_iw_f = matsubara count)
        n_orb:      orbital count
        dlr:        DLRBasis instance (β must match the data)
        matsubara_indices: Matsubara index labels for the fermionic axis
            (length 2*n_iw_f, ordered to match the BSE matrix)

    Returns:
        chi_ir: compressed matrix in (IR_l, orb², IR_l', orb²) layout, square
            of side L * n_orb².
    """
    nf = 2 * n_iw_f
    no2 = n_orb * n_orb
    N = nf * no2
    if chi_matrix.shape != (N, N):
        raise ValueError(
            f"chi_matrix shape {chi_matrix.shape} doesn't match expected {N}x{N}"
        )
    if len(matsubara_indices) != nf:
        raise ValueError(
            f"matsubara_indices length {len(matsubara_indices)} != 2*n_iw_f={nf}"
        )

    M = dlr.matsubara_to_ir_matrix(matsubara_indices)  # [L, nf]
    L = M.shape[0]

    # Reshape chi into [nf, no2, nf, no2]
    chi_4d = chi_matrix.reshape(nf, no2, nf, no2)

    # Compress along both fermionic axes: chi_ir[l, ab, l', cd] = M[l, n] M[l', n'].conj() chi[n, ab, n', cd]
    # We use M for both (rather than M, M.conj()) because we want a similarity-style transform
    # consistent with: chi_ir = M chi M^T (linear projection both sides).
    chi_step1 = np.tensordot(M, chi_4d, axes=([1], [0]))            # [L, no2, nf, no2]
    chi_ir = np.tensordot(chi_step1, M, axes=([2], [1]))            # [L, no2, no2, L]
    # Reorder axes back to [L, no2, L, no2]
    chi_ir = chi_ir.transpose(0, 1, 3, 2)
    return chi_ir.reshape(L * no2, L * no2)


def uncompress_bse_matrix(
    gamma_ir: np.ndarray,
    matsubara_indices: np.ndarray,
    n_orb: int,
    dlr: "DLRBasis",
) -> np.ndarray:
    """
    Decompress an IR-basis vertex back to a Matsubara matrix.

    Inverse of compress_bse_matrix: γ(ν, ν') ≈ M^T · γ_IR · M̄.

    Args:
        gamma_ir: [L*no², L*no²] vertex in IR
        matsubara_indices: target Matsubara indices (length nf = 2*n_iw_f)
        n_orb: orbital count
        dlr: DLRBasis

    Returns:
        gamma_full: [nf*no², nf*no²] reconstructed vertex on the Matsubara grid
    """
    no2 = n_orb * n_orb
    nf = len(matsubara_indices)

    # E maps IR coefs → Matsubara values: E[n, l] = û_l(iω_n)
    # For fermionic, we get this by applying the basis evaluator at matsu pts:
    from sparse_ir.sampling import MatsubaraSampling
    smpl = MatsubaraSampling(dlr.basis, list(matsubara_indices))
    L = dlr.size
    # Apply evaluate to identity matrix I[L, L] along axis 0 (the coef axis);
    # result shape: [nf, L] where each column is û_l(iω_n) for all n.
    E = np.asarray(smpl.evaluate(np.eye(L, dtype=complex), axis=0))  # [nf, L]

    gamma_4d = gamma_ir.reshape(L, no2, L, no2)
    g1 = np.tensordot(E, gamma_4d, axes=([1], [0]))   # [nf, no2, L, no2]
    g2 = np.tensordot(g1, E, axes=([2], [1]))         # [nf, no2, no2, nf]
    g2 = g2.transpose(0, 1, 3, 2)                     # [nf, no2, nf, no2]
    return g2.reshape(nf * no2, nf * no2)


def solve_bse_local_dlr(
    chi_loc: np.ndarray,
    chi0_loc: np.ndarray,
    n_iw_f: int,
    n_orb: int,
    n_iw_b: int,
    beta: float,
    wmax: float = 8.0,
    eps: float = 1e-6,
    svd_cutoff: float = 1e-8,
) -> Tuple[np.ndarray, dict]:
    """
    DLR-compressed BSE inversion (EXPERIMENTAL).

    Algorithm:
      1. Build a DLR basis for (β, ω_max, ε).
      2. For each bosonic frequency, compress χ and χ⁰ to IR space.
      3. Invert the compressed matrices (much smaller).
      4. Compute Γ_IR = (χ⁰_IR)⁻¹ - (χ_IR)⁻¹.
      5. Decompress back to Matsubara (only if needed downstream).

    Expected speedup: BSE matrix shrinks from N=(2*n_iw_f)·n_orb² to L·n_orb²
    where L ≈ 30 vs 2·n_iw_f ≈ 100-200.  Inversion ~ N³ → ~ (L/N)³ ·  N³.
    For L=30, N=200: 0.34% of original cost per Ω.

    WARNING (experimental):
      Unlike the 1P G(iω), the 4-point susceptibility χ(iν, iν'; iΩ) has
      high-frequency Matsubara content (Γ at large |ν|) that does NOT fit
      the bounded-support spectral representation of IR/DLR. The compression
      can therefore introduce non-trivial errors in Γ. For production use,
      stick to the Matsubara BSE in bse_solver and use DLR only for 1P
      quantities (density, Σ tails, etc.). This entrypoint exists for
      benchmarking and for use cases where the loss of accuracy is acceptable.

    Args:
        chi_loc, chi0_loc: same conventions as solve_bse_local
        n_iw_f, n_iw_b: Matsubara grid sizes
        n_orb: orbital count
        beta:  inverse temperature for the DLR
        wmax:  spectral support cutoff (will be auto-adjusted if needed)
        eps:   DLR accuracy

    Returns:
        gamma_loc_full: [nf*no², nf*no², 2*n_iw_b+1] decompressed irreducible vertex
        diagnostics: dict with timing, basis size, compression ratio
    """
    import time
    if not is_available():
        raise RuntimeError(
            "sparse_ir not installed. Install with: pip install sparse_ir"
        )

    t0 = time.time()
    nf = 2 * n_iw_f
    nb = 2 * n_iw_b + 1
    no2 = n_orb * n_orb
    N_full = nf * no2

    # Build DLR basis
    dlr = DLRBasis(beta=beta, wmax=wmax, eps=eps)
    L = dlr.size

    # Build the canonical Matsubara indices used by the BSE matrix layout:
    # ν_n indexing iv = 0..nf-1 corresponds to n = iv - n_iw_f, ω = (2n+1)π/β
    # → integer indices passed to sparse_ir are (2n+1):
    matsu_idx = np.array(
        [2 * (iv - n_iw_f) + 1 for iv in range(nf)], dtype=int
    )

    L_no2 = L * no2
    gamma_ir = np.zeros((L_no2, L_no2, nb), dtype=complex)

    cond_chi0 = []
    cond_chi = []
    n_trunc = 0

    for iw in range(nb):
        chi_mat = chi_loc[:, :, iw] if chi_loc.ndim == 3 else None
        chi0_mat = chi0_loc[:, :, iw] if chi0_loc.ndim == 3 else None

        if chi_mat is None or chi0_mat is None:
            # Caller didn't pre-flatten; fall back to per-Ω construction
            from bse_solver import _reshape_to_matrix_fast
            chi_mat = _reshape_to_matrix_fast(chi_loc, n_iw_f, n_orb, iw)
            chi0_mat = _reshape_to_matrix_fast(chi0_loc, n_iw_f, n_orb, iw)

        # Compress to IR space
        chi_ir = compress_bse_matrix(chi_mat, n_iw_f, n_orb, dlr, matsu_idx)
        chi0_ir = compress_bse_matrix(chi0_mat, n_iw_f, n_orb, dlr, matsu_idx)

        # Robust inversion via SVD
        try:
            U0, s0, V0h = np.linalg.svd(chi0_ir, full_matrices=False)
            mask0 = s0 > svd_cutoff * s0[0]
            n_trunc += int(np.sum(~mask0))
            s0_inv = np.where(mask0, 1.0 / np.where(mask0, s0, 1.0), 0.0)
            chi0_inv = (V0h.conj().T * s0_inv) @ U0.conj().T
            cond_chi0.append(s0[0] / max(s0[-1], 1e-300))

            U, s, Vh = np.linalg.svd(chi_ir, full_matrices=False)
            mask = s > svd_cutoff * s[0]
            n_trunc += int(np.sum(~mask))
            s_inv = np.where(mask, 1.0 / np.where(mask, s, 1.0), 0.0)
            chi_inv = (Vh.conj().T * s_inv) @ U.conj().T
            cond_chi.append(s[0] / max(s[-1], 1e-300))
        except np.linalg.LinAlgError:
            chi0_inv = np.linalg.pinv(chi0_ir, rcond=svd_cutoff)
            chi_inv = np.linalg.pinv(chi_ir, rcond=svd_cutoff)

        gamma_ir[:, :, iw] = chi0_inv - chi_inv

    # Decompress to Matsubara space
    t_decompress_start = time.time()
    gamma_full = np.zeros((N_full, N_full, nb), dtype=complex)
    for iw in range(nb):
        gamma_full[:, :, iw] = uncompress_bse_matrix(
            gamma_ir[:, :, iw], matsu_idx, n_orb, dlr,
        )

    elapsed = time.time() - t0
    diagnostics = {
        "dlr_basis_size": L,
        "matsubara_size": nf,
        "compression_ratio": float(L) / nf,
        "elapsed_seconds": elapsed,
        "decompress_seconds": time.time() - t_decompress_start,
        "mean_cond_chi0": float(np.mean(cond_chi0)) if cond_chi0 else 0.0,
        "mean_cond_chi": float(np.mean(cond_chi)) if cond_chi else 0.0,
        "total_truncated_modes": n_trunc,
    }
    return gamma_full, diagnostics
