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

    # Quasiparticle weight from self-energy
    Z = None
    mass_enhancement = None
    if sigma_iw is not None and beta is not None:
        Z = _extract_quasiparticle_weight(sigma_iw, beta)
        mass_enhancement = 1.0 / Z if Z > 0 else None

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
        "quasiparticle_weight_Z": float(Z) if Z else None,
        "mass_enhancement": float(mass_enhancement) if mass_enhancement else None,
        "tc_uncorrected_K": float(tc_uncorrected),
        "tc_dmft_corrected_K": float(tc_corrected),
        "tc_shift_K": float(tc_corrected - tc_uncorrected),
        "tc_shift_percent": float((tc_corrected - tc_uncorrected) / max(tc_uncorrected, 0.1) * 100),
        "omega_log_meV": float(omega_log),
        "mu_star": float(mu_star),
    }


def _allen_dynes(lam: float, omega_log_meV: float, mu_star: float) -> float:
    """Allen-Dynes Tc formula. Returns Tc in Kelvin."""
    if lam <= mu_star * (1 + 0.62 * lam) or omega_log_meV <= 0:
        return 0.0
    omega_log_K = omega_log_meV * 11.6045  # meV → K
    exponent = -1.04 * (1 + lam) / (lam - mu_star * (1 + 0.62 * lam))
    return (omega_log_K / 1.2) * np.exp(exponent)


def _extract_quasiparticle_weight(
    sigma_iw: np.ndarray,
    beta: float,
) -> float:
    """
    Extract Z = [1 - ∂Im(Σ)/∂ω |_{ω→0}]⁻¹ from the lowest Matsubara frequencies.

    Z = 1 / (1 - Im[Σ(iω_0)] / ω_0)

    where ω_0 = π/β is the lowest positive Matsubara frequency.
    """
    n_iw_total = sigma_iw.shape[0]
    n_iw = n_iw_total // 2

    # Lowest positive Matsubara frequency
    omega_0 = np.pi / beta

    if sigma_iw.ndim >= 3:
        # Multi-orbital: average Z over diagonal elements
        n_orb = sigma_iw.shape[1]
        Z_vals = []
        for a in range(n_orb):
            sigma_0 = sigma_iw[n_iw, a, a]  # first positive frequency
            dSigma_dw = sigma_0.imag / omega_0
            z = 1.0 / (1.0 - dSigma_dw)
            if 0 < z < 1:
                Z_vals.append(z)
        return float(np.mean(Z_vals)) if Z_vals else 0.5
    else:
        # Single-orbital
        sigma_0 = sigma_iw[n_iw] if sigma_iw.ndim == 1 else sigma_iw[n_iw, 0]
        if np.isscalar(sigma_0):
            dSigma_dw = float(np.imag(sigma_0)) / omega_0
        else:
            dSigma_dw = float(sigma_0.imag) / omega_0
        z = 1.0 / (1.0 - dSigma_dw)
        return float(z) if 0 < z < 1 else 0.5
