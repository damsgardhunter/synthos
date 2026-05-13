#!/usr/bin/env python3
"""
DMFT-Corrected Tc for Phonon-Mediated Superconductors.

DFT overestimates N(E_F) in correlated metals because it misses the
quasiparticle renormalization Z = m/m*. DMFT fixes this:

  N_DMFT(E_F) = A(ω=0) = Z × N_DFT(E_F)  (for a Fermi liquid)

Since the Allen-Dynes Tc depends on λ ∝ N(E_F):
  λ_corrected = λ_DFT × [N_DMFT(E_F) / N_DFT(E_F)]

This one-shot correction is cheap (uses existing DMFT output) and has
big impact for A15 compounds (Nb₃Sn, V₃Si), SrTiO₃, and nickelates
where Z ~ 0.3-0.7.

References:
  Yin et al., Nat. Phys. 10, 845 (2014) — DMFT for Fe-SC
  Mandal et al., PRB 89, 220502 (2014) — DMFT λ correction
"""

import numpy as np
import json
import os
from typing import Optional, Dict


def correct_tc_with_dmft(
    n_ef_dft: float,
    n_ef_dmft: float,
    lambda_dft: float,
    omega_log: float,
    mu_star: float = 0.10,
    sigma_iw: Optional[np.ndarray] = None,
    beta: Optional[float] = None,
) -> Dict:
    """
    Compute DMFT-corrected Tc from Allen-Dynes with renormalized λ.

    Args:
        n_ef_dft: DFT density of states at Fermi level (states/eV/spin/cell)
        n_ef_dmft: DMFT spectral weight at ω=0 (from analytic continuation)
        lambda_dft: electron-phonon coupling from DFPT/EPW
        omega_log: logarithmic average phonon frequency (meV)
        mu_star: Coulomb pseudopotential
        sigma_iw: self-energy on Matsubara axis [2*n_iw, n_orb, n_orb] for Z extraction
        beta: inverse temperature (needed for Z from Σ)

    Returns:
        dict with corrected λ, Tc, Z, mass enhancement
    """
    if n_ef_dft <= 0 or n_ef_dmft <= 0:
        return {
            "error": "Non-positive N(E_F)",
            "n_ef_dft": n_ef_dft,
            "n_ef_dmft": n_ef_dmft,
        }

    # DOS ratio
    dos_ratio = n_ef_dmft / n_ef_dft

    # Quasiparticle weight from self-energy. Returns None if Z is unphysical
    # (out of (0, 1]) — caller treats those as out-of-distribution rather than
    # silently clamping to a "safe" value (would falsify ML training data).
    Z = None
    mass_enhancement = None
    z_out_of_range = False
    if sigma_iw is not None and beta is not None:
        Z = _extract_quasiparticle_weight(sigma_iw, beta)
        if Z is not None and Z > 0:
            mass_enhancement = 1.0 / Z
        else:
            z_out_of_range = True

    # Corrected λ
    lambda_corrected = lambda_dft * dos_ratio

    # Allen-Dynes Tc with corrected λ
    tc_corrected = _allen_dynes(lambda_corrected, omega_log, mu_star)
    tc_uncorrected = _allen_dynes(lambda_dft, omega_log, mu_star)

    return {
        "lambda_dft": float(lambda_dft),
        "lambda_dmft_corrected": float(lambda_corrected),
        "n_ef_dft": float(n_ef_dft),
        "n_ef_dmft": float(n_ef_dmft),
        "dos_ratio": float(dos_ratio),
        "quasiparticle_weight_Z": float(Z) if Z is not None else None,
        "z_out_of_range": bool(z_out_of_range),
        "mass_enhancement": float(mass_enhancement) if mass_enhancement is not None else None,
        "tc_uncorrected_K": float(tc_uncorrected),
        "tc_dmft_corrected_K": float(tc_corrected),
        "tc_shift_K": float(tc_corrected - tc_uncorrected),
        "tc_shift_percent": float((tc_corrected - tc_uncorrected) / max(tc_uncorrected, 0.1) * 100),
        "omega_log_meV": float(omega_log),
        "mu_star": float(mu_star),
    }


def _allen_dynes(lam: float, omega_log_meV: float, mu_star: float,
                  omega2_meV: float = None) -> float:
    """Allen-Dynes Tc formula with strong-coupling corrections f1, f2.

    Tc = (ω_log / 1.2) · f1 · f2 · exp[-1.04(1+λ)/(λ - μ*(1+0.62λ))]

    f1 = [1 + (λ/Λ_1)^(3/2)]^(1/3)      (strong-coupling enhancement)
    f2 = 1 + (ω_2/ω_log - 1)·λ²/(λ² + Λ_2²)
    Λ_1 = 2.46(1+3.8μ*)
    Λ_2 = 1.82(1+6.3μ*)(ω_2/ω_log)

    For λ < 1.5, f1≈f2≈1 and this reduces to McMillan. For strong-coupling
    hydrides (λ > 2), f1, f2 give 20-50% enhancement.

    Reference: Allen & Dynes, PRB 12, 905 (1975).
    """
    if lam <= mu_star * (1 + 0.62 * lam) or omega_log_meV <= 0:
        return 0.0
    omega_log_K = omega_log_meV * 11.6045  # meV → K
    exponent = -1.04 * (1 + lam) / (lam - mu_star * (1 + 0.62 * lam))
    if exponent < -50:
        return 0.0

    # Strong-coupling corrections
    if omega2_meV is None or omega2_meV <= 0:
        omega2_meV = omega_log_meV * 1.3  # typical ω_2/ω_log ratio
    omega2_ratio = max(1.0, omega2_meV / omega_log_meV)

    Lambda_1 = 2.46 * (1 + 3.8 * mu_star)
    Lambda_2 = 1.82 * (1 + 6.3 * mu_star) * omega2_ratio
    f1 = (1 + (lam / Lambda_1) ** 1.5) ** (1.0 / 3.0)
    f2 = 1 + (omega2_ratio - 1) * lam ** 2 / (lam ** 2 + Lambda_2 ** 2)

    tc = (omega_log_K / 1.2) * f1 * f2 * np.exp(exponent)
    return float(tc) if np.isfinite(tc) and tc > 0 else 0.0


def _extract_quasiparticle_weight(
    sigma_iw: np.ndarray,
    beta: float,
) -> Optional[float]:
    """
    Extract Z = [1 - ∂Im(Σ)/∂ω |_{ω→0}]⁻¹ from the lowest Matsubara frequencies.

    Z = 1 / (1 - Im[Σ(iω_0)] / ω_0)

    where ω_0 = π/β is the lowest positive Matsubara frequency.

    Physical regimes that produce Z outside (0, 1]:
      Z > 1: dΣ/dω > 0 — typically QMC noise; can also indicate non-Fermi-liquid
             behavior with anomalous self-energy slope.
      Z ≤ 0: dΣ/dω ≥ 1/ω_0 — happens near magnetic instability or for spectral
             features pinned to ω=0; Z=0 is the Mott insulator limit.

    We do NOT clamp these to a "safe" value: clamping silently falsifies ML
    training data and obscures real correlation physics. Instead we return
    None for unphysical values so downstream code can flag the calculation
    as out-of-distribution.

    Multi-orbital returns the arithmetic mean of Z values that fall in (0, 1].
    If NO orbital gives a physical Z, returns None and the caller decides
    (e.g., skip mass renormalization or refuse Tc estimate).
    """
    n_iw_total = sigma_iw.shape[0]
    n_iw = n_iw_total // 2
    omega_0 = np.pi / beta

    def _z_from_sigma_0(sigma_0_val: complex) -> Optional[float]:
        dSigma_dw = float(sigma_0_val.imag) / omega_0
        denom = 1.0 - dSigma_dw
        if abs(denom) < 1e-12:
            # Singular: dΣ/dω ≈ 1; physically Z → ∞ which is unphysical
            return None
        return 1.0 / denom

    if sigma_iw.ndim >= 3:
        n_orb = sigma_iw.shape[1]
        Z_phys = []
        for a in range(n_orb):
            z = _z_from_sigma_0(sigma_iw[n_iw, a, a])
            if z is not None and 0 < z <= 1:
                Z_phys.append(z)
        if Z_phys:
            return float(np.mean(Z_phys))
        # No physical Z: caller treats as out-of-distribution
        return None

    # Single-orbital
    sigma_0 = sigma_iw[n_iw] if sigma_iw.ndim == 1 else sigma_iw[n_iw, 0]
    if np.isscalar(sigma_0):
        sigma_0 = complex(sigma_0)
    z = _z_from_sigma_0(sigma_0)
    return float(z) if z is not None and 0 < z <= 1 else None
