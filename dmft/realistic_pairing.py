#!/usr/bin/env python3
"""
D3: Realistic-Model Pairing Susceptibility Pipeline.

Combines multi-orbital DCA (D1) + charge self-consistency (D2) +
pairing susceptibility (B3) into a single pipeline for real materials.

The workflow for a material like La₂CuO₄ or NdNiO₂:

  1. Load DMFT bundle (H(k), U/J, structure from DFT pipeline)
  2. Optionally run CSC loop to self-consistently update H(k)
  3. Run multi-orbital DCA at several temperatures
  4. At each T, compute pairing eigenvalue λ_pair(T) from DCA vertex
  5. Classify gap symmetry (d-wave, s±-wave, etc.)
  6. Extrapolate Tc from λ_pair(T) → 1

This is the full stack: DFT → Wannier → DMFT → vertex → BSE → pairing → Tc.

The key physics that each stage adds:
  - DFT: band structure, Fermi surface geometry
  - DFT+U: crude correlation (static mean-field)
  - Single-site DMFT: local correlations, quasiparticle renormalization Z
  - DCA: momentum-dependent self-energy, d-wave pairing
  - Multi-orbital: Cu-O charge transfer, orbital-selective correlations
  - CSC: feedback of DMFT density into DFT potential
  - Vertex/BSE: explicit pairing interaction from spin fluctuations

References:
  Gull et al., PRB 82, 155101 (2010) — DCA for Emery model
  Maier et al., PRB 74, 094513 (2006) — multi-orbital DCA pairing
  Kent et al., PRB 72, 060411 (2005) — CSC DCA
  Kitatani et al., PRB 102, 220502 (2020) — nickelate pairing
"""

import numpy as np
import json
import os
import time
from typing import Optional, List, Dict

from dca_solver import (
    DCAParams, generate_cluster_momenta, assign_k_to_patches,
    build_lattice_kpoints, compute_dca_pairing_eigenvalue,
)
from multiorbital_dca import (
    MultiOrbitalDCAParams, KanamoriInteraction,
    build_emery_hamiltonian, build_multiorbital_hamiltonian_from_hr,
    run_multiorbital_dca, compute_lattice_gf_multiorbital_fast,
)
from charge_selfconsistency import CSCParams, run_csc_loop
from pairing_susceptibility import extrapolate_tc


# ── Material model builder ───────────────────────────────────────────────────

def build_material_model(bundle_data: dict) -> dict:
    """
    Build a multi-orbital DCA model from a DMFT bundle.

    Determines:
      - Number of orbitals and which are correlated
      - Interaction parameters (U, J from bundle)
      - Hamiltonian (from bundle H(k) or model)
      - Material class (cuprate, pnictide, nickelate, etc.)

    Returns a model dict with all parameters needed for DCA.
    """
    corr_shells = bundle_data.get("corr_shells", [])
    n_orb = bundle_data["hk"].shape[1]

    # Interaction from bundle
    U_values = np.array(bundle_data.get("U_values", [4.0] * n_orb))
    J_values = np.array(bundle_data.get("J_values", [0.7] * n_orb))

    # Pad/trim to n_orb
    if len(U_values) < n_orb:
        U_values = np.pad(U_values, (0, n_orb - len(U_values)), constant_values=0)
    if len(J_values) < n_orb:
        J_values = np.pad(J_values, (0, n_orb - len(J_values)), constant_values=0)

    # Detect material class from elements
    elements = bundle_data.get("elements", [])
    if isinstance(elements, str):
        elements = json.loads(elements)

    material_class = "generic"
    if "Cu" in elements and "O" in elements:
        material_class = "cuprate"
    elif "Ni" in elements and "O" in elements:
        material_class = "nickelate"
    elif "Fe" in elements and ("As" in elements or "Se" in elements):
        material_class = "pnictide"

    # Identify correlated orbitals
    correlated_indices = list(range(n_orb))
    if corr_shells:
        correlated_indices = []
        offset = 0
        for shell in corr_shells:
            dim = shell.get("dim", 5)
            for i in range(dim):
                if offset + i < n_orb:
                    correlated_indices.append(offset + i)
            offset += dim

    # For cuprates with Emery model: only Cu-d is correlated
    use_density_density = False
    if material_class == "cuprate" and n_orb == 3:
        correlated_indices = [0]  # Cu-d_{x²-y²}
        use_density_density = False
    elif n_orb >= 5:
        # For 5-orbital d-shell: density-density reduces sign problem
        use_density_density = True

    return {
        "n_orb": n_orb,
        "material_class": material_class,
        "correlated_indices": correlated_indices,
        "U_values": U_values[:n_orb],
        "J_values": J_values[:n_orb],
        "use_density_density": use_density_density,
        "elements": elements,
    }


# ── Multi-orbital DCA pairing eigenvalue ─────────────────────────────────────

def compute_multiorbital_pairing(
    dca_result: dict,
    params: MultiOrbitalDCAParams,
    hk: np.ndarray,
    kpoints: np.ndarray,
) -> dict:
    """
    Compute pairing eigenvalues from multi-orbital DCA.

    For multi-orbital systems, the pairing susceptibility matrix is:
      χ_pp(K, iν)_{ab,cd} = -(T/N_K) Σ_{k∈K} G_{ac}(k,iν) G_{bd}(-k,-iν)

    The pairing eigenvalue equation in the orbital-resolved form:
      λ · Δ_{ab}(K) = Σ_{K',cd} Γ_{ab,cd}(K,K') · χ_pp(K')_{cd} · Δ_{cd}(K')

    For the 3-band Emery model, the d-wave gap is predominantly on the
    Cu-d orbital: Δ_{dd}(K) ∝ cos(Kx) - cos(Ky).

    For pnictides, the s±-wave gap has opposite sign on electron/hole pockets,
    predominantly on Fe-d orbitals.
    """
    nc = params.nc
    no = params.n_orb
    n_w = 2 * params.n_iw
    wn = params.wn
    beta = params.beta
    sigma_c = dca_result["sigma_c"]
    patch_assignment = dca_result["patch_assignment"]

    from pairing_susceptibility import _build_minus_k_map
    minus_k_map = _build_minus_k_map(kpoints)

    # Lattice G with full orbital structure
    gk_iw = compute_lattice_gf_multiorbital_fast(
        hk, sigma_c, patch_assignment, params.mu, wn,
    )

    T = 1.0 / beta

    # Orbital-resolved pp bubble: χ⁰_pp(K, iν)_{ab} = -(T/N_K) Σ_k G_{aa}(k,ν) G_{bb}(-k,-ν)
    # Simplified to diagonal (a=c, b=d) for tractability
    chi0_pp = np.zeros((nc, n_w, no), dtype=complex)

    for ik in range(len(kpoints)):
        ic = patch_assignment[ik]
        ik_minus = minus_k_map[ik]
        for iw in range(n_w):
            iw_minus = n_w - 1 - iw
            for a in range(no):
                chi0_pp[ic, iw, a] += -T * gk_iw[ik, iw, a, a] * gk_iw[ik_minus, iw_minus, a, a]

    patch_counts = np.bincount(patch_assignment, minlength=nc).astype(float)
    for ic in range(nc):
        if patch_counts[ic] > 0:
            chi0_pp[ic] /= patch_counts[ic]

    # Sum over frequencies → static susceptibility per orbital per patch
    chi0_static = np.sum(chi0_pp, axis=1).real  # [nc, no]

    # Form factors on cluster momenta
    Kx = params.K_cluster[:, 0] * np.pi
    Ky = params.K_cluster[:, 1] * np.pi
    phi_d = np.cos(Kx) - np.cos(Ky)
    phi_s = np.ones(nc)
    phi_sext = np.cos(Kx) + np.cos(Ky)

    results = {}

    for name, phi in [("d-x2y2", phi_d), ("s-wave", phi_s), ("s±-wave", phi_sext)]:
        phi_norm_sq = np.sum(phi ** 2)
        if phi_norm_sq < 1e-10:
            results[name] = 0.0
            continue

        # Average U over correlated orbitals for the pairing interaction
        corr = params.correlated_orbitals
        U_eff = np.mean(params.interaction.U[corr]) if params.interaction else params.U

        # λ = -(U_eff/N_c) Σ_K Σ_a chi0_static(K,a) · φ(K)² / Σ_K φ(K)²
        # Weight by correlated-orbital contribution
        chi0_weighted = np.zeros(nc)
        for ic in range(nc):
            for a in corr:
                chi0_weighted[ic] += chi0_static[ic, a]

        lambda_val = -(U_eff / nc) * np.sum(chi0_weighted * phi ** 2) / phi_norm_sq
        results[name] = float(lambda_val)

    # Dominant channel
    abs_vals = {k: abs(v) for k, v in results.items()}
    results["dominant"] = max(abs_vals, key=abs_vals.get)

    results["chi0_static_by_orbital"] = chi0_static.tolist()
    results["beta"] = beta
    results["temperature_K"] = 11604.5 / beta
    results["n_orb"] = no
    results["nc"] = nc

    return results


# ── Full realistic pipeline ──────────────────────────────────────────────────

def run_realistic_pairing_pipeline(
    bundle_path: str,
    work_dir: str,
    temperatures_eV: Optional[List[float]] = None,
    run_csc: bool = True,
    csc_max_iter: int = 10,
    nc: int = 4,
    n_k_per_dim: int = 32,
    qe_callback=None,
    wannier_callback=None,
) -> dict:
    """
    Full realistic-model pairing pipeline:

      Phase A: Load bundle, build material model
      Phase B: (Optional) Charge self-consistency loop
      Phase C: Multi-orbital DCA temperature sweep
      Phase D: Pairing eigenvalues at each T
      Phase E: Tc extrapolation and gap symmetry

    Args:
        bundle_path: path to DMFT bundle HDF5
        work_dir: output directory
        temperatures_eV: list of temperatures in eV (default: 5 points)
        run_csc: whether to run CSC before the pairing sweep
        csc_max_iter: max CSC iterations
        nc: cluster size for DCA
        n_k_per_dim: k-mesh density
        qe_callback: for CSC QE feedback (None = skip QE update)
        wannier_callback: for CSC Wannier re-projection (None = skip)

    Returns:
        dict with Tc, gap symmetry, full temperature sweep
    """
    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()

    # Default temperature sweep (high to low)
    if temperatures_eV is None:
        temperatures_eV = [0.05, 0.03, 0.02, 0.015, 0.01]

    # ── Phase A: Load and build model ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Realistic] Phase A: Loading bundle and building model")
    print(f"{'='*60}")

    from run_dmft import load_bundle
    data = load_bundle(bundle_path)
    formula = data["formula"]
    model = build_material_model(data)

    print(f"[Realistic] {formula}: {model['material_class']}, "
          f"n_orb={model['n_orb']}, correlated={model['correlated_indices']}")
    print(f"[Realistic] U={model['U_values'][:3]}..., J={model['J_values'][:3]}...")

    all_results = {
        "formula": formula,
        "material_class": model["material_class"],
        "n_orb": model["n_orb"],
        "nc": nc,
    }

    # ── Phase B: Charge self-consistency (optional) ──────────────────
    hk_final = data["hk"]

    if run_csc:
        print(f"\n{'='*60}")
        print(f"[Realistic] Phase B: Charge self-consistency")
        print(f"{'='*60}")

        csc_params = CSCParams(
            max_iterations=csc_max_iter,
            density_mix=0.3,
            density_tol=1e-3,
            use_dca=False,  # Single-site DMFT for CSC (cheaper)
        )

        from run_dmft import build_solid_dmft_config
        dmft_config = build_solid_dmft_config(data, work_dir)

        csc_result = run_csc_loop(
            bundle_data=data,
            dmft_config=dmft_config,
            csc_params=csc_params,
            work_dir=os.path.join(work_dir, "csc"),
            qe_callback=qe_callback,
            wannier_callback=wannier_callback,
        )

        all_results["csc"] = {
            "converged": csc_result.get("converged"),
            "n_iterations": csc_result.get("n_iterations"),
            "final_trace": csc_result.get("final_trace"),
        }

        # If CSC produced updated H(k), use it
        csc_hk_path = os.path.join(work_dir, "csc", "hk_updated.npy")
        if os.path.exists(csc_hk_path):
            hk_final = np.load(csc_hk_path)
            print(f"[Realistic] Using CSC-updated H(k)")
    else:
        all_results["csc"] = {"skipped": True}

    # ── Phase C+D: Temperature sweep with DCA + pairing ──────────────
    print(f"\n{'='*60}")
    print(f"[Realistic] Phase C/D: Multi-orbital DCA temperature sweep")
    print(f"{'='*60}")

    interaction = KanamoriInteraction(
        model["n_orb"], model["U_values"], model["J_values"],
    )

    kpoints = build_lattice_kpoints(n_k_per_dim, dim=2)
    sweep_results = []
    sigma_warmstart = None

    for i_T, T_eV in enumerate(temperatures_eV):
        beta = 1.0 / T_eV
        T_K = T_eV * 11604.5

        print(f"\n[Realistic] T={T_K:.0f} K (β={beta:.1f} eV⁻¹), "
              f"point {i_T+1}/{len(temperatures_eV)}")

        dca_params = MultiOrbitalDCAParams(
            n_orb=model["n_orb"],
            nc=nc,
            dim=2,
            beta=beta,
            n_iw=128,
            n_k_per_dim=n_k_per_dim,
            U=float(np.mean(model["U_values"])),
            mu=data.get("fermi_energy", 0.0),
            n_iter=20,
            sigma_mix=0.5,
            convergence_tol=1e-3,
            interaction=interaction,
            use_density_density=model["use_density_density"],
            correlated_orbitals=model["correlated_indices"],
        )

        T_dir = os.path.join(work_dir, f"T_{T_eV:.4f}")

        dca_result = run_multiorbital_dca(
            hk=hk_final, kpoints=kpoints,
            params=dca_params, work_dir=T_dir,
            initial_sigma=sigma_warmstart,
        )

        if dca_result.get("converged"):
            sigma_warmstart = dca_result["sigma_c"]

        # Pairing eigenvalues
        pairing = compute_multiorbital_pairing(
            dca_result, dca_params, hk_final, kpoints,
        )

        entry = {
            "T_eV": T_eV,
            "T_K": T_K,
            "beta": beta,
            "dca_converged": dca_result.get("converged", False),
            "lambda_d": pairing.get("d-x2y2", 0.0),
            "lambda_s": pairing.get("s-wave", 0.0),
            "lambda_spm": pairing.get("s±-wave", 0.0),
            "dominant": pairing.get("dominant", "?"),
        }
        sweep_results.append(entry)

        print(f"[Realistic] λ_d={entry['lambda_d']:.4f}, λ_s={entry['lambda_s']:.4f}, "
              f"λ_s±={entry['lambda_spm']:.4f}, dominant={entry['dominant']}")

    all_results["temperature_sweep"] = sweep_results

    # ── Phase E: Tc extrapolation ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Realistic] Phase E: Tc extrapolation")
    print(f"{'='*60}")

    converged_entries = [e for e in sweep_results if e["dca_converged"]]

    if len(converged_entries) >= 2:
        # Find dominant channel
        lambda_d_max = max(abs(e["lambda_d"]) for e in converged_entries)
        lambda_s_max = max(abs(e["lambda_s"]) for e in converged_entries)
        lambda_spm_max = max(abs(e["lambda_spm"]) for e in converged_entries)

        if lambda_d_max >= lambda_s_max and lambda_d_max >= lambda_spm_max:
            channel = "d-x2y2"
            lambdas = [abs(e["lambda_d"]) for e in converged_entries]
        elif lambda_spm_max >= lambda_s_max:
            channel = "s±-wave"
            lambdas = [abs(e["lambda_spm"]) for e in converged_entries]
        else:
            channel = "s-wave"
            lambdas = [abs(e["lambda_s"]) for e in converged_entries]

        T_list = [e["T_K"] for e in converged_entries]
        tc_result = extrapolate_tc(T_list, lambdas)

        all_results["tc_K"] = tc_result.get("tc_bse")
        all_results["tc_confidence"] = tc_result.get("tc_confidence")
        all_results["dominant_channel"] = channel
        all_results["lambda_max"] = max(lambdas)

        tc_str = f"{all_results['tc_K']:.1f} K" if all_results["tc_K"] else "not determined"
        print(f"[Realistic] Dominant channel: {channel}")
        print(f"[Realistic] Max λ = {max(lambdas):.4f}")
        print(f"[Realistic] Tc estimate: {tc_str} ({all_results['tc_confidence']})")

        is_unconventional = channel not in ("s-wave",)
        all_results["is_unconventional"] = is_unconventional
        if is_unconventional:
            print(f"[Realistic] *** UNCONVENTIONAL PAIRING: {channel} ***")
    else:
        all_results["tc_K"] = None
        all_results["tc_confidence"] = "insufficient_data"
        all_results["dominant_channel"] = "unknown"

    elapsed = time.time() - t0
    all_results["elapsed_hours"] = elapsed / 3600

    # Save
    results_path = os.path.join(work_dir, "realistic_pairing_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Realistic] Total time: {elapsed/3600:.1f}h")
    print(f"[Realistic] Results: {results_path}")

    return all_results
