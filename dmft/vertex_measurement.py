#!/usr/bin/env python3
"""
B1: Two-Particle Vertex Measurement via CTHYB measure_G2.

CTHYB can measure the two-particle Green's function G²(iν, iν', iΩ) on the
impurity during the QMC sampling. This is the local two-particle correlator:

  G²_loc(iν, iν', iΩ) = <T c(iν+iΩ) c†(iν) c(iν') c†(iν'+iΩ)>

From G² we extract:
  - χ_loc(iν, iν', iΩ)  = G²_loc - G_loc(iν)·G_loc(iν')·β·δ(Ω)
  - χ⁰_loc(iν, iΩ)      = -β · G_loc(iν) · G_loc(iν+iΩ)
  - Γ_loc via BSE inversion (see bse_solver.py)

The G² measurement is 10-100× more expensive than single-particle because:
  - Three-frequency object: O(n_iw² × n_iW) matrix elements
  - Each orbital pair contributes: O(n_orb⁴) entries
  - Memory: ~(2·n_iw_f)² × n_iw_b × n_orb⁴ × 16 bytes (complex128)

Budget: 4-24 additional hours per material on top of 1P DMFT.

References:
  Boehnke et al., PRB 84, 075145 (2011) — compact representation
  Hafermann et al., EPL 85, 27007 (2009) — improved estimators
  Gunacker et al., PRB 92, 155102 (2015) — efficient G² measurement
  Rohringer et al., Rev. Mod. Phys. 90, 025003 (2018) — diagrammatic extensions
"""

import numpy as np
import os
import json
import time


# ── Frequency grid sizing ────────────────────────────────────────────────────

# The fermionic Matsubara frequencies: ν_n = (2n+1)π/β
# The bosonic Matsubara frequencies:   Ω_m = 2mπ/β
#
# For superconducting pairing the relevant bosonic frequency is Ω=0
# (Cooper pair has zero total energy in the center-of-mass frame).
# We need enough fermionic frequencies to capture the vertex structure
# around ν=0, which decays as ~1/ν² for the irreducible vertex.

def compute_g2_grid_params(
    beta: float,
    n_orb: int,
    correlation_regime: str,
    max_memory_gb: float = 32.0,
    enable_proper_crossing: bool = False,
) -> dict:
    """
    Compute optimal frequency grid sizes for G² measurement.

    The grid must balance:
      1. Enough fermionic frequencies to resolve vertex structure (~30-60)
      2. Enough bosonic frequencies for momentum-dependent susceptibility (~15-30)
      3. Memory budget: G² is n_iw_f² × n_iw_b × n_orb⁴ × 16 bytes
      4. QMC noise: more frequencies = more bins = more noise per bin

    Args:
        enable_proper_crossing: if True, enforce n_iw_b ≥ 2·n_iw_f-1 to support
            exact SU(2) ph→pp crossing in decompose_vertex_channels. Typically
            doubles memory cost. Use only when measuring G² in ph channel and
            wanting the rigorous crossing (alternative: use pp channel directly).

    Returns dict with parameters for CTHYB measure_G2 configuration.
    """
    # Memory estimate: complex128 = 16 bytes
    # G²[n_f, n_f, n_b, orb, orb, orb, orb] but we store particle-hole
    # channel which is [2*n_f, 2*n_f, 2*n_b+1, n_orb, n_orb, n_orb, n_orb]
    # In practice TRIQS stores it more efficiently but this gives the upper bound.

    bytes_per_element = 16  # complex128
    orb_factor = n_orb ** 4

    # Start with generous grids, shrink to fit memory
    n_iw_f = 60   # fermionic frequencies (each side of zero)
    n_iw_b = 30   # bosonic frequencies (each side of zero)

    # For strongly correlated: need more frequencies (vertex has slow decay)
    if correlation_regime in ("Mott-proximate", "strongly-correlated"):
        n_iw_f = min(80, n_iw_f + 20)
        n_iw_b = min(40, n_iw_b + 10)

    # Proper SU(2) crossing requires n_iw_b ≥ 2*n_iw_f - 1 so that
    # Ω = ν - ν' (which ranges over 2*n_iw_f-1 values) is always within
    # the bosonic grid. Adaptively expand n_iw_b at the cost of shrinking
    # n_iw_f if needed to fit memory.
    if enable_proper_crossing:
        n_iw_b = max(n_iw_b, 2 * n_iw_f - 1)

    # Shrink to fit memory budget
    max_bytes = max_memory_gb * 1e9
    while n_iw_f > 10:
        total_elements = (2 * n_iw_f) * (2 * n_iw_f) * (2 * n_iw_b + 1) * orb_factor
        total_bytes = total_elements * bytes_per_element
        if total_bytes <= max_bytes:
            break
        # Reduce fermionic grid first (quadratic cost), then bosonic.
        # Under proper-crossing mode we keep n_iw_b ≥ 2*n_iw_f-1 as we shrink.
        if n_iw_f > 20:
            n_iw_f -= 5
            if enable_proper_crossing:
                n_iw_b = max(n_iw_b, 2 * n_iw_f - 1)
        elif n_iw_b > 10:
            n_iw_b -= 5
        else:
            n_iw_f -= 5
            if enable_proper_crossing:
                n_iw_b = max(n_iw_b, 2 * n_iw_f - 1)

    # Ensure minimum grid
    n_iw_f = max(n_iw_f, 10)
    n_iw_b = max(n_iw_b, 5)
    if enable_proper_crossing:
        n_iw_b = max(n_iw_b, 2 * n_iw_f - 1)

    total_elements = (2 * n_iw_f) * (2 * n_iw_f) * (2 * n_iw_b + 1) * orb_factor
    memory_gb = total_elements * bytes_per_element / 1e9

    # QMC parameters: need ~10× more cycles for G² than G¹
    # The variance scales as 1/N_cycles, but G² has O(n_orb⁴) bins
    g2_cycle_multiplier = max(5, min(20, orb_factor // 10))

    return {
        "n_iw_f": n_iw_f,
        "n_iw_b": n_iw_b,
        "memory_estimate_gb": round(memory_gb, 2),
        "orb_factor": orb_factor,
        "n_orb": n_orb,
        "g2_cycle_multiplier": g2_cycle_multiplier,
        "beta": beta,
        "supports_proper_crossing": n_iw_b >= 2 * n_iw_f - 1,
    }


def estimate_g2_walltime_hours(grid_params: dict, mpi_ranks: int = 20) -> float:
    """
    Estimate wall time for G² measurement in hours.

    Empirical scaling from TRIQS/CTHYB benchmarks:
      - 5-orbital d-shell, 30 fermionic freq, 20 MPI: ~8h
      - 3-orbital t2g, 40 fermionic freq, 20 MPI: ~4h
      - 1-orbital (single-band Hubbard), 60 freq, 20 MPI: ~1h
    """
    n_orb = grid_params["n_orb"]
    n_iw_f = grid_params["n_iw_f"]
    multiplier = grid_params["g2_cycle_multiplier"]

    # Base: 1 hour for single-orbital, 30 freq, 20 ranks
    base_hours = 1.0
    orb_scaling = (n_orb / 1.0) ** 3  # roughly cubic in n_orb
    freq_scaling = (n_iw_f / 30.0) ** 1.5
    cycle_scaling = multiplier / 5.0
    rank_scaling = 20.0 / max(mpi_ranks, 1)

    return base_hours * orb_scaling * freq_scaling * cycle_scaling * rank_scaling


# ── CTHYB G² configuration builder ──────────────────────────────────────────

def build_g2_solver_params(
    grid_params: dict,
    base_n_cycles: int = 5_000_000,
    base_length_cycle: int = 200,
    channel: str = "ph",
) -> dict:
    """
    Build CTHYB solver parameters for the two-particle measurement run.

    This is a SEPARATE run from the converged 1P DMFT — we load the
    converged self-energy and measure G² without further self-consistency.

    Args:
        grid_params: frequency grid configuration from compute_g2_grid_params
        base_n_cycles: base QMC cycles per orbital
        base_length_cycle: base length of QMC cycle
        channel: "ph" (particle-hole, for spin/charge) or "pp" (particle-particle,
                 for direct pairing vertex extraction). "pp" eliminates the
                 ph→pp crossing approximation in the BSE.

    Returns a dict of solver parameters to merge into the CTHYB config.
    """
    multiplier = grid_params["g2_cycle_multiplier"]

    if channel not in ("ph", "pp"):
        raise ValueError(f"Unknown G² channel: {channel} (expected 'ph' or 'pp')")

    params = {
        "measure_G2_n_fermionic": grid_params["n_iw_f"],
        "measure_G2_n_bosonic": grid_params["n_iw_b"],

        # Increased sampling for G² noise reduction
        "n_cycles_tot": base_n_cycles * multiplier,
        "length_cycle": base_length_cycle,
        "n_warmup_cycles": 100_000,

        # Keep 1P measurements for consistency check
        "measure_G_tau": True,
        "measure_density_matrix": True,

        # Performance: G² measurement is memory-bound, keep cycle length moderate
        "move_double": True,
        "perform_tail_fit": True,
    }

    # Enable the requested channel measurement.
    # CTHYB supports both simultaneously, but each ~doubles the QMC cost.
    if channel == "ph":
        params["measure_G2_iw_ph"] = True
    elif channel == "pp":
        params["measure_G2_iw_pp"] = True

    return params


def build_g2_config(
    converged_dmft_config: dict,
    grid_params: dict,
    converged_h5_path: str,
    channel: str = "ph",
) -> dict:
    """
    Build a complete solid_dmft configuration for the G² measurement pass.

    Key difference from the 1P run:
      - calc_mode = "DMFT" but n_iter_dmft = 1 (single shot, no self-consistency)
      - Sigma is loaded from the converged run (sigma_mix = 0 effectively)
      - G² measurement flags are enabled
      - QMC cycles are increased 5-20×

    Args:
        channel: "ph" (particle-hole) or "pp" (particle-particle, direct pairing)
    """
    config = dict(converged_dmft_config)  # shallow copy top level

    # Override general settings for measurement pass
    config["general"] = dict(config.get("general", {}))
    config["general"]["n_iter_dmft"] = 1  # single-shot measurement
    config["general"]["calc_mode"] = "DMFT"
    seedname = config["general"].get("seedname", "material")
    config["general"]["jobname"] = f"{seedname}_g2_{channel}_measurement"

    # Merge G² solver params
    solver_params = build_g2_solver_params(grid_params, channel=channel)
    config["solver"] = dict(config.get("solver", {}))
    config["solver"].update(solver_params)

    # Point to converged self-energy
    config["general"]["load_sigma"] = True
    config["general"]["path_to_sigma"] = converged_h5_path

    return config


# ── G² data extraction ──────────────────────────────────────────────────────

def extract_g2_from_h5(h5_path: str, shell_index: int = 0,
                        channel: str = "ph") -> dict:
    """
    Extract the measured G²(iν, iν', iΩ) from the CTHYB output HDF5.

    TRIQS/CTHYB stores G² as a Block2Gf in:
      /DMFT_results/it_001/solver/G2_iw_{ph,pp}

    Args:
        h5_path: path to the HDF5 from a G² measurement run
        shell_index: which correlated shell to extract (for multi-shell systems)
        channel: "ph" (particle-hole) or "pp" (particle-particle, pairing)

    Returns a dict with:
      - g2_ph or g2_pp: numpy array [2*n_f, 2*n_f, 2*n_b+1, n_orb, n_orb, n_orb, n_orb]
      - channel: the channel that was extracted
      - g_iw:  single-particle G(iω) [2*n_f, n_orb, n_orb]
      - beta:  inverse temperature
      - n_iw_f, n_iw_b: grid sizes
    """
    import h5py

    result = {"channel": channel}

    with h5py.File(h5_path, "r") as f:
        # Navigate to the measurement iteration
        dmft_grp = f.get("DMFT_results")
        if dmft_grp is None:
            raise ValueError("No DMFT_results in HDF5")

        iter_keys = sorted([k for k in dmft_grp.keys() if k.startswith("it_")])
        if not iter_keys:
            raise ValueError("No iteration data found")

        last_iter = dmft_grp[iter_keys[-1]]

        # Try to load G² from the solver output
        # TRIQS stores it in various possible locations depending on version
        if channel == "pp":
            g2_paths = [f"solver/G2_iw_pp", f"G2_iw_pp"]
        else:
            g2_paths = [
                f"solver/G2_iw_ph",
                f"G2_iw_ph",
                f"solver/G2_iw",
            ]

        g2_data = None
        for gpath in g2_paths:
            if gpath in last_iter:
                g2_grp = last_iter[gpath]
                # TRIQS Block2Gf stored as nested groups per block pair.
                # Prefer same-spin blocks ("up_up" or "up" × "up") which are
                # the relevant ones for the BSE in the ph channel. Fall back
                # to whatever block has data.
                # Iterate blocks: try same-spin first (key contains "up_up" or "ud")
                preferred_keys = [k for k in g2_grp.keys()
                                  if "up_up" in k or "uu" == k or "ud" in k]
                ordered_keys = preferred_keys + [k for k in g2_grp.keys()
                                                  if k not in preferred_keys]
                for block_key in ordered_keys:
                    if "data" in g2_grp[block_key]:
                        g2_data = g2_grp[block_key]["data"][()]
                        break
                if g2_data is not None:
                    break

        if g2_data is None:
            flag = "measure_G2_iw_pp" if channel == "pp" else "measure_G2_iw_ph"
            raise ValueError(f"G² data not found in HDF5 — check {flag} was enabled")

        # Store under channel-specific key (g2_ph or g2_pp)
        result[f"g2_{channel}"] = g2_data

        # Load single-particle G(iω) for χ⁰ construction
        g_iw_paths = ["solver/G_iw", "G_iw", "solver/Gimp_iw"]
        for gpath in g_iw_paths:
            if gpath in last_iter:
                g_iw_grp = last_iter[gpath]
                for block_key in g_iw_grp.keys():
                    if "data" in g_iw_grp[block_key]:
                        result["g_iw"] = g_iw_grp[block_key]["data"][()]
                        break
                break

        # Beta from mesh
        if "solver" in last_iter and "G_iw" in last_iter["solver"]:
            for block_key in last_iter["solver"]["G_iw"].keys():
                mesh_grp = last_iter["solver"]["G_iw"][block_key].get("mesh")
                if mesh_grp is not None and "beta" in mesh_grp.attrs:
                    result["beta"] = float(mesh_grp.attrs["beta"])
                    break

    # Infer grid sizes from data shape
    if "g2_ph" in result:
        shape = result["g2_ph"].shape
        # Shape is typically [2*n_f, 2*n_f, 2*n_b+1, ...] or similar
        result["n_iw_f"] = shape[0] // 2
        result["n_iw_b"] = (shape[2] - 1) // 2 if len(shape) > 2 else 0

    return result


def compute_chi0_loc(g_iw: np.ndarray, beta: float, n_iw_b: int) -> np.ndarray:
    """
    Compute the bare (bubble) local susceptibility χ⁰_loc(iν, iΩ).

    χ⁰_loc(iν, iΩ) = -β · G_loc(iν) · G_loc(iν + iΩ)

    In the particle-hole channel with orbital indices:
      χ⁰_loc(iν, iΩ)_{abcd} = -β · G_loc(iν)_{da} · G_loc(iν + iΩ)_{bc}

    For the pairing (particle-particle) channel:
      χ⁰_pp(iν, iΩ)_{abcd} = β · G_loc(iν)_{ac} · G_loc(iΩ - iν)_{bd}

    Args:
        g_iw: G(iω) array, shape [2*n_iw, n_orb, n_orb]
        beta: inverse temperature
        n_iw_b: number of bosonic frequencies per side

    Returns:
        chi0: shape [2*n_iw_f, 2*n_iw_b+1, n_orb, n_orb, n_orb, n_orb]
               where n_iw_f = n_iw - n_iw_b (to allow shifting)
    """
    n_iw_total = g_iw.shape[0]
    n_orb = g_iw.shape[1]
    n_iw = n_iw_total // 2

    # Fermionic grid: restrict to range where ν+Ω stays within G(iω) data
    n_iw_f = n_iw - n_iw_b
    if n_iw_f < 5:
        raise ValueError(f"Insufficient fermionic frequencies: n_iw={n_iw}, n_iw_b={n_iw_b}")

    # Bosonic frequencies: Ω_m = 2mπ/β for m in [-n_iw_b, n_iw_b]
    # Fermionic: ν_n mapped to array index n + n_iw (centered)

    chi0 = np.zeros(
        (2 * n_iw_f, 2 * n_iw_b + 1, n_orb, n_orb, n_orb, n_orb),
        dtype=complex,
    )

    for iv in range(2 * n_iw_f):
        # Fermionic index in the full G array
        iv_full = iv + (n_iw - n_iw_f)
        for iw in range(2 * n_iw_b + 1):
            # Bosonic shift: ν + Ω maps fermionic index → fermionic index + bosonic index
            # Since ν_n + Ω_m = (2n+1)π/β + 2mπ/β = (2(n+m)+1)π/β = ν_{n+m}
            m = iw - n_iw_b  # bosonic index centered at 0
            iv_shifted = iv_full + m
            if 0 <= iv_shifted < n_iw_total:
                g_v = g_iw[iv_full]      # [n_orb, n_orb]
                g_vw = g_iw[iv_shifted]  # [n_orb, n_orb]
                # χ⁰_{abcd}(ν, Ω) = -β · G_{da}(ν) · G_{bc}(ν+Ω)
                for a in range(n_orb):
                    for b in range(n_orb):
                        for c in range(n_orb):
                            for d in range(n_orb):
                                chi0[iv, iw, a, b, c, d] = -beta * g_v[d, a] * g_vw[b, c]

    return chi0


def compute_chi0_loc_fast(g_iw: np.ndarray, beta: float, n_iw_b: int) -> np.ndarray:
    """
    Vectorized version of χ⁰_loc computation using einsum.

    Same physics as compute_chi0_loc but ~100× faster for multi-orbital systems.
    """
    n_iw_total = g_iw.shape[0]
    n_orb = g_iw.shape[1]
    n_iw = n_iw_total // 2
    n_iw_f = n_iw - n_iw_b

    if n_iw_f < 5:
        raise ValueError(f"Insufficient fermionic frequencies: n_iw={n_iw}, n_iw_b={n_iw_b}")

    chi0 = np.zeros(
        (2 * n_iw_f, 2 * n_iw_b + 1, n_orb, n_orb, n_orb, n_orb),
        dtype=complex,
    )

    offset = n_iw - n_iw_f

    for iw in range(2 * n_iw_b + 1):
        m = iw - n_iw_b
        # Range of valid g_iw indices for g_v: [iv_start, iv_end)
        # Constraints: iv_start >= offset (within fermionic window),
        #   iv_start + m >= 0 (shifted index valid), iv_end <= offset + 2*n_iw_f,
        #   iv_end + m <= n_iw_total
        iv_start = max(offset, -m)
        iv_end = min(offset + 2 * n_iw_f, n_iw_total - m)

        if iv_start >= iv_end:
            continue

        sl = slice(iv_start, iv_end)
        sl_shifted = slice(iv_start + m, iv_end + m)
        sl_out = slice(iv_start - offset, iv_end - offset)

        g_v = g_iw[sl]           # [nv, n_orb, n_orb]
        g_vw = g_iw[sl_shifted]  # [nv, n_orb, n_orb]

        # χ⁰_{abcd}(ν, Ω) = -β · G_{da}(ν) · G_{bc}(ν+Ω)
        # Using einsum: contract over nothing, just outer product in orbital space
        chi0[sl_out, iw, :, :, :, :] = -beta * np.einsum(
            "vda,vbc->vabcd", g_v, g_vw
        )

    return chi0


def compute_chi0_loc_pp_fast(g_iw: np.ndarray, beta: float, n_iw_b: int) -> np.ndarray:
    """
    Particle-particle channel χ⁰_loc bubble.

      χ⁰_pp_{abcd}(ν, Ω) = -β · G_{ac}(ν) · G_{bd}(Ω - ν)

    The pp bubble describes the bare (non-interacting) pair susceptibility.
    For uncorrelated Cooper pairs at total energy Ω, one electron has
    Matsubara frequency ν and the other has Ω-ν.

    This is the DIRECT pairing bubble — using it with the BSE-extracted
    pp vertex Γ_pp gives the singlet pairing susceptibility without the
    leading-order crossing approximation needed in the ph channel.

    Args:
        g_iw: single-particle G(iω), shape [2*n_iw, n_orb, n_orb]
        beta: inverse temperature
        n_iw_b: number of bosonic frequencies per side

    Returns:
        chi0_pp: shape [2*n_iw_f, 2*n_iw_b+1, n_orb, n_orb, n_orb, n_orb]
    """
    n_iw_total = g_iw.shape[0]
    n_orb = g_iw.shape[1]
    n_iw = n_iw_total // 2
    n_iw_f = n_iw - n_iw_b

    if n_iw_f < 5:
        raise ValueError(f"Insufficient fermionic frequencies: n_iw={n_iw}, n_iw_b={n_iw_b}")

    chi0 = np.zeros(
        (2 * n_iw_f, 2 * n_iw_b + 1, n_orb, n_orb, n_orb, n_orb),
        dtype=complex,
    )

    offset = n_iw - n_iw_f

    # For pp: shifted index iv' = (n_iw_total - 1 - iv_full) + m  ↔  Ω - ν
    # Derivation: Ω-ν = (2m)π/β - (2n+1)π/β = (2(m-n-1)+1)π/β,
    #   so n' = m - n - 1, and iv' = n' + n_iw = m - (iv_full - n_iw) - 1 + n_iw
    #         = m + (n_iw_total - 1) - iv_full
    for iw in range(2 * n_iw_b + 1):
        m = iw - n_iw_b
        # iv runs over the vertex window [offset, offset + 2*n_iw_f)
        # iv_shifted = (n_iw_total - 1) - iv + m must lie in [0, n_iw_total)
        # → iv ∈ [m + n_iw_total - 1 - (n_iw_total-1), m + n_iw_total - 1]
        #       = [m, m + n_iw_total - 1]
        iv_start = max(offset, m)
        iv_end = min(offset + 2 * n_iw_f, m + n_iw_total)

        if iv_start >= iv_end:
            continue

        # Build the shifted index array
        iv_range = np.arange(iv_start, iv_end)
        iv_shifted_range = (n_iw_total - 1) - iv_range + m

        # Filter to valid shifted indices
        valid = (iv_shifted_range >= 0) & (iv_shifted_range < n_iw_total)
        if not np.any(valid):
            continue
        iv_range = iv_range[valid]
        iv_shifted_range = iv_shifted_range[valid]

        g_v = g_iw[iv_range]           # G(ν)
        g_omega_minus_v = g_iw[iv_shifted_range]  # G(Ω-ν)

        sl_out = iv_range - offset

        # χ⁰_pp_{abcd}(ν, Ω) = -β · G_{ac}(ν) · G_{bd}(Ω-ν)
        chi0[sl_out, iw, :, :, :, :] = -beta * np.einsum(
            "vac,vbd->vabcd", g_v, g_omega_minus_v
        )

    return chi0


def compute_chi_loc_from_g2(
    g2_ph: np.ndarray,
    g_iw: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Extract the connected local susceptibility from G²:

      χ_loc(iν, iν', iΩ) = G²_loc(iν, iν', iΩ) - β·G(iν)·G(iν')·δ_{Ω,0}

    The disconnected part is the product of two single-particle propagators.
    Subtracting it gives the connected (interaction-driven) part.

    Args:
        g2_ph: G²(iν, iν', iΩ) from CTHYB measurement
        g_iw: G(iω) single-particle Green's function
        beta: inverse temperature

    Returns:
        chi_loc: connected susceptibility, same shape as g2_ph
    """
    # Ensure complex dtype (g2_ph from QMC may be real)
    chi_loc = g2_ph.astype(complex, copy=True)

    # Find the Ω=0 bosonic index (center of the bosonic axis)
    n_b_total = g2_ph.shape[2]
    iw_zero = n_b_total // 2

    # Subtract disconnected part at Ω=0
    # G²_disc(iν, iν', Ω=0) = β · G(iν) · G(iν') × (orbital structure)
    n_f = g2_ph.shape[0]  # 2*n_iw_f (vertex window)
    n_iw_total = g_iw.shape[0]  # 2*n_iw (full G window, typically larger)
    n_orb = g_iw.shape[1] if g_iw.ndim >= 2 else 1

    # Map vertex frequency index → G frequency index (centered window)
    # vertex iv=0 corresponds to G index iv + offset where offset centers the window
    offset = (n_iw_total - n_f) // 2

    if g_iw.ndim >= 2 and g2_ph.ndim >= 5:
        # Multi-orbital: need to handle index structure carefully
        # For particle-hole: disc_{abcd}(ν, ν'; Ω=0) = β · G_{da}(ν) · G_{bc}(ν')
        for iv in range(n_f):
            iv_g = iv + offset
            if not (0 <= iv_g < n_iw_total):
                continue
            for ivp in range(n_f):
                ivp_g = ivp + offset
                if not (0 <= ivp_g < n_iw_total):
                    continue
                if g2_ph.ndim == 7:
                    for a in range(n_orb):
                        for b in range(n_orb):
                            for c in range(n_orb):
                                for d in range(n_orb):
                                    chi_loc[iv, ivp, iw_zero, a, b, c, d] -= (
                                        beta * g_iw[iv_g, d, a] * g_iw[ivp_g, b, c]
                                    )
    else:
        # Single-orbital or flat array: simpler structure
        # disc(ν, ν', Ω=0) = β · G(ν) · G(ν')
        g_flat = g_iw.ravel() if g_iw.ndim > 1 else g_iw
        for iv in range(n_f):
            iv_g = iv + offset
            if not (0 <= iv_g < len(g_flat)):
                continue
            for ivp in range(n_f):
                ivp_g = ivp + offset
                if not (0 <= ivp_g < len(g_flat)):
                    continue
                chi_loc[iv, ivp, iw_zero] -= beta * g_flat[iv_g] * g_flat[ivp_g]

    return chi_loc


def compute_chi_loc_from_g2_pp(
    g2_pp: np.ndarray,
    g_iw: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Extract connected χ_loc^pp from particle-particle G²:

      χ_pp(ν,ν';Ω) = G²_pp(ν,ν';Ω) - β·G(ν)·G(Ω-ν)·δ_{ν,ν'}

    The pp disconnected has δ_{ν,ν'} structure (not δ_{Ω,0} like ph),
    so it contributes at ALL bosonic frequencies, only on the diagonal
    in ν,ν'.

    Args:
        g2_pp: G²(iν,iν';iΩ) from CTHYB measure_G2_iw_pp
        g_iw: G(iω) single-particle
        beta: inverse temperature

    Returns:
        chi_pp: connected pp susceptibility, same shape as g2_pp
    """
    chi_loc = g2_pp.astype(complex, copy=True)

    n_f = g2_pp.shape[0]
    n_b_total = g2_pp.shape[2]
    n_iw_b = (n_b_total - 1) // 2
    n_iw_total = g_iw.shape[0]
    n_orb = g_iw.shape[1] if g_iw.ndim >= 2 else 1
    offset = (n_iw_total - n_f) // 2

    if g_iw.ndim >= 2 and g2_pp.ndim >= 5:
        # Multi-orbital: disconnected_pp{abcd}(ν, ν=ν'; Ω) = β·G_{ac}(ν)·G_{bd}(Ω-ν)
        for iw in range(n_b_total):
            m = iw - n_iw_b
            for iv in range(n_f):
                iv_g = iv + offset
                # iv_g_shifted ↔ Ω-ν: index (n_iw_total-1) - iv_g + m
                iv_g_shifted = (n_iw_total - 1) - iv_g + m
                if not (0 <= iv_g < n_iw_total and 0 <= iv_g_shifted < n_iw_total):
                    continue
                # Diagonal in ν,ν' only (δ_{ν,ν'})
                ivp = iv  # ν' = ν
                if g2_pp.ndim == 7:
                    for a in range(n_orb):
                        for b in range(n_orb):
                            for c in range(n_orb):
                                for d in range(n_orb):
                                    chi_loc[iv, ivp, iw, a, b, c, d] -= (
                                        beta * g_iw[iv_g, a, c] * g_iw[iv_g_shifted, b, d]
                                    )
    else:
        # Single-orbital
        g_flat = g_iw.ravel() if g_iw.ndim > 1 else g_iw
        for iw in range(n_b_total):
            m = iw - n_iw_b
            for iv in range(n_f):
                iv_g = iv + offset
                iv_g_shifted = (len(g_flat) - 1) - iv_g + m
                if not (0 <= iv_g < len(g_flat) and 0 <= iv_g_shifted < len(g_flat)):
                    continue
                chi_loc[iv, iv, iw] -= beta * g_flat[iv_g] * g_flat[iv_g_shifted]

    return chi_loc


# ── Entrypoint for standalone testing ────────────────────────────────────────

def run_vertex_measurement(
    work_dir: str,
    converged_h5_path: str,
    converged_config: dict,
    data: dict,
    mpi_ranks: int = 20,
    channel: str = "ph",
    enable_proper_crossing: bool = False,
) -> dict:
    """
    Run the full G² measurement pass:
      1. Compute grid parameters based on orbital count and memory
      2. Build G² measurement config
      3. Execute CTHYB measurement (MPI)
      4. Extract G², compute χ⁰_loc and χ_loc
      5. Return all vertex data

    Args:
        work_dir: directory for this measurement run
        converged_h5_path: path to converged 1P DMFT HDF5
        converged_config: the solid_dmft config dict from the 1P run
        data: bundle data dict (for n_orb, correlation_regime, etc.)
        mpi_ranks: number of MPI ranks for CTHYB
        channel: "ph" (default, particle-hole) or "pp" (particle-particle,
                 direct pairing vertex — avoids ph→pp crossing approximation)
        enable_proper_crossing: if True (and channel="ph"), grow the bosonic
                 grid to n_iw_b ≥ 2·n_iw_f-1 so the proper Ω=ν-ν' crossing
                 can be applied in the BSE decomposition. Doubles memory.
                 Ignored for channel="pp" (no crossing needed).

    Returns:
        dict with g2_{ph,pp}, chi0_loc, chi_loc, grid_params, timing
    """
    import subprocess

    os.makedirs(work_dir, exist_ok=True)

    # Determine n_orb from correlated shells
    n_orb = max((sh["dim"] for sh in data["corr_shells"]), default=5)
    beta = converged_config.get("general", {}).get("beta", 40.0)

    t0 = time.time()

    # Proper crossing is only meaningful for the ph channel
    use_proper = enable_proper_crossing and channel == "ph"

    # 1. Grid parameters
    grid_params = compute_g2_grid_params(
        beta=beta,
        n_orb=n_orb,
        correlation_regime=data.get("correlation_regime", "moderately-correlated"),
        max_memory_gb=min(32.0, 64.0 / max(1, n_orb)),
        enable_proper_crossing=use_proper,
    )
    estimated_hours = estimate_g2_walltime_hours(grid_params, mpi_ranks)
    crossing_note = ""
    if use_proper:
        crossing_note = f", proper crossing={'OK' if grid_params['supports_proper_crossing'] else 'INSUFFICIENT'}"
    print(f"[G2] Grid: n_iw_f={grid_params['n_iw_f']}, n_iw_b={grid_params['n_iw_b']}, "
          f"memory={grid_params['memory_estimate_gb']:.1f} GB, "
          f"estimated={estimated_hours:.1f}h{crossing_note}")

    # 2. Build measurement config
    g2_config = build_g2_config(converged_config, grid_params, converged_h5_path,
                                 channel=channel)

    # 3. Write config and run
    import toml
    config_path = os.path.join(work_dir, "dmft_config_g2.toml")
    with open(config_path, "w") as f:
        toml.dump(g2_config, f)

    timeout_s = int(max(estimated_hours * 3600 * 2, 14400))  # 2× estimate, min 4h
    cmd = [
        "mpirun", "-np", str(mpi_ranks),
        "--oversubscribe", "--allow-run-as-root",
        "python3", "-m", "solid_dmft", config_path,
    ]

    print(f"[G2] Starting CTHYB G² measurement ({mpi_ranks} ranks, timeout {timeout_s}s)")
    proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, timeout=timeout_s)

    measurement_time = time.time() - t0
    print(f"[G2] Measurement finished in {measurement_time/3600:.1f}h, exit={proc.returncode}")

    if proc.returncode != 0:
        return {
            "converged": False,
            "error": proc.stderr[-1000:],
            "measurement_hours": measurement_time / 3600,
            "grid_params": grid_params,
        }

    # 4. Extract G² data
    seedname = g2_config["general"].get("seedname", "material")
    g2_h5_path = os.path.join(work_dir, f"{seedname}.h5")

    try:
        g2_data = extract_g2_from_h5(g2_h5_path, channel=channel)
    except Exception as e:
        return {
            "converged": False,
            "error": f"G² extraction failed: {e}",
            "measurement_hours": measurement_time / 3600,
            "grid_params": grid_params,
        }

    # 5. Compute χ⁰_loc and χ_loc (channel-specific)
    g_iw = g2_data.get("g_iw")
    g2_arr = g2_data.get(f"g2_{channel}")
    meas_beta = g2_data.get("beta", beta)

    chi0_loc = None
    chi_loc = None
    if g_iw is not None:
        try:
            if channel == "pp":
                chi0_loc = compute_chi0_loc_pp_fast(g_iw, meas_beta, grid_params["n_iw_b"])
            else:
                chi0_loc = compute_chi0_loc_fast(g_iw, meas_beta, grid_params["n_iw_b"])
        except Exception as e:
            print(f"[G2] WARNING: χ⁰_loc computation failed: {e}")

    if g2_arr is not None:
        try:
            if channel == "pp":
                chi_loc = compute_chi_loc_from_g2_pp(g2_arr, g_iw, meas_beta)
            else:
                chi_loc = compute_chi_loc_from_g2(g2_arr, g_iw, meas_beta)
        except Exception as e:
            print(f"[G2] WARNING: χ_loc computation failed: {e}")

    # Save intermediate results
    results = {
        "converged": True,
        "channel": channel,
        "measurement_hours": measurement_time / 3600,
        "grid_params": grid_params,
        "n_orb": n_orb,
        "beta": meas_beta,
        "g2_shape": list(g2_arr.shape) if g2_arr is not None else None,
        "chi0_available": chi0_loc is not None,
        "chi_loc_available": chi_loc is not None,
    }

    # Save numpy arrays for BSE solver
    np_path = os.path.join(work_dir, "vertex_data.npz")
    save_dict = {
        "grid_params_json": json.dumps(grid_params),
        "channel": channel,
    }
    if g2_arr is not None:
        save_dict[f"g2_{channel}"] = g2_arr
    if g_iw is not None:
        save_dict["g_iw"] = g_iw
    if chi0_loc is not None:
        save_dict["chi0_loc"] = chi0_loc
    if chi_loc is not None:
        save_dict["chi_loc"] = chi_loc
    np.savez_compressed(np_path, **save_dict)
    results["vertex_data_path"] = np_path
    print(f"[G2] Vertex data ({channel} channel) saved to {np_path} "
          f"({os.path.getsize(np_path)/1e6:.0f} MB)")

    return results
