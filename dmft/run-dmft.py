#!/usr/bin/env python3
"""
DMFT calculation runner — takes a QAE DMFT bundle (HDF5) and runs
solid_dmft with TRIQS/CTHYB.

Usage:
  mpirun -np $DMFT_MPI_RANKS python3 run-dmft.py /data/dmft_bundles/LaH10_200GPa.h5

The bundle HDF5 contains:
  /hamiltonian/hk          — H(k) on uniform k-mesh [nk, norb, norb] complex
  /hamiltonian/kpoints     — k-points [nk, 3] fractional
  /hamiltonian/kmesh       — mesh dimensions [3] int
  /correlated_subspace/
    corr_shells            — JSON array of {atom, l, dim, sort}
    proj_mat               — projector matrices [nk, nshells, dim, norb] complex
    corr_to_inequiv        — mapping correlated -> inequivalent shells
  /interaction/
    U_values               — per-shell Hubbard U (eV)
    J_values               — per-shell Hund's J (eV)
    hubbard_kind           — 0=density-density, 1=Kanamori, 2=Slater
  /structure/
    lattice_vectors        — [3,3] Angstrom
    positions              — [natom, 3] fractional
    elements               — element symbols
    formula                — chemical formula string
    pressure_gpa           — pressure in GPa
  /electronic/
    fermi_energy           — eV
    n_electrons            — total electrons in correlated subspace
    magnetic_ordering      — "NM" / "FM" / "AFM-*"
    correlation_regime     — from Hubbard workflow
  /metadata/
    source                 — "qae-dmft-bundle-v1"
    created_at             — ISO timestamp
    qe_quality_tier        — from QEFullResult
"""

import sys
import os
import json
import time
import numpy as np
import h5py

def load_bundle(bundle_path: str) -> dict:
    """Load a QAE DMFT bundle HDF5 into a dict."""
    data = {}
    with h5py.File(bundle_path, "r") as f:
        # Hamiltonian
        data["hk"] = f["/hamiltonian/hk"][()]
        data["kpoints"] = f["/hamiltonian/kpoints"][()]
        data["kmesh"] = tuple(f["/hamiltonian/kmesh"][()])

        # Correlated subspace
        data["corr_shells"] = json.loads(f["/correlated_subspace/corr_shells"][()])
        data["proj_mat"] = f["/correlated_subspace/proj_mat"][()]
        data["corr_to_inequiv"] = f["/correlated_subspace/corr_to_inequiv"][()]

        # Interaction
        data["U_values"] = f["/interaction/U_values"][()]
        data["J_values"] = f["/interaction/J_values"][()]
        data["hubbard_kind"] = int(f["/interaction/hubbard_kind"][()])

        # Structure
        data["formula"] = f["/structure/formula"][()].decode() if isinstance(f["/structure/formula"][()], bytes) else str(f["/structure/formula"][()])
        data["pressure_gpa"] = float(f["/structure/pressure_gpa"][()])

        # Electronic
        data["fermi_energy"] = float(f["/electronic/fermi_energy"][()])
        data["n_electrons"] = float(f["/electronic/n_electrons"][()])
        data["magnetic_ordering"] = f["/electronic/magnetic_ordering"][()].decode() if isinstance(f["/electronic/magnetic_ordering"][()], bytes) else str(f["/electronic/magnetic_ordering"][()])
        data["correlation_regime"] = f["/electronic/correlation_regime"][()].decode() if isinstance(f["/electronic/correlation_regime"][()], bytes) else str(f["/electronic/correlation_regime"][()])

    return data


def build_solid_dmft_config(data: dict, work_dir: str) -> dict:
    """
    Build solid_dmft configuration from bundle data.

    Returns a dict that maps to solid_dmft's dmft_config.toml structure.
    """
    n_shells = len(data["corr_shells"])
    n_inequiv = len(set(data["corr_to_inequiv"].tolist()))

    # Interaction: Kanamori for multi-orbital d shells, density-density fallback
    interaction_type = "kanamori" if data["hubbard_kind"] >= 1 else "density_density"

    # CTHYB solver parameters — tuned for production on 20 MPI ranks
    n_warmup = int(os.environ.get("CTHYB_N_WARMUP", 50000))
    n_cycles = int(os.environ.get("CTHYB_N_CYCLES", 5000000))
    length_cycle = int(os.environ.get("CTHYB_LENGTH_CYCLE", 200))

    # Temperature: start at 300K (beta ~ 38.7 eV^-1), can be lowered
    beta = 40.0  # eV^-1 (~300K)

    # Double-counting: "around_mean_field" for moderately correlated,
    # "fully_localized_limit" for Mott-proximate
    if data["correlation_regime"] in ("Mott-proximate", "strongly-correlated"):
        dc_type = "cFLL"  # fully localized limit (Czyzyk-Sawatzky)
    else:
        dc_type = "cAMF"  # around mean field

    config = {
        "general": {
            "seedname": data["formula"],
            "jobname": f"{data['formula']}_{int(data['pressure_gpa'])}GPa_dmft",
            "enforce_off_diag": True if n_inequiv > 1 else False,
            "beta": beta,
            "n_iter_dmft": 30,
            "dc_type": dc_type,
            "dc_dmft": True,
            "calc_mode": "DMFT",
            "mu_mix": 0.5,
            "sigma_mix": 0.5,
            "prec_mu": 0.001,
        },
        "solver": {
            "type": "cthyb",
            "n_warmup_cycles": n_warmup,
            "n_cycles_tot": n_cycles,
            "length_cycle": length_cycle,
            "measure_density_matrix": True,
            "move_double": True,
        },
        "advanced": {
            "dc_U": [float(u) for u in data["U_values"]],
            "dc_J": [float(j) for j in data["J_values"]],
        },
    }

    return config


def write_solid_dmft_inputs(data: dict, config: dict, work_dir: str):
    """
    Write all files solid_dmft needs into work_dir:
      - dmft_config.toml
      - seedname.h5 (TRIQS DFTTools format with H(k) and corr_shells)
    """
    import toml

    os.makedirs(work_dir, exist_ok=True)

    # Write config
    config_path = os.path.join(work_dir, "dmft_config.toml")
    with open(config_path, "w") as f:
        toml.dump(config, f)
    print(f"[DMFT] Wrote config: {config_path}")

    # Write DFTTools HDF5 (the format SumkDFT reads)
    seedname = config["general"]["seedname"]
    h5_path = os.path.join(work_dir, f"{seedname}.h5")
    write_dfttools_h5(data, h5_path)
    print(f"[DMFT] Wrote DFTTools H5: {h5_path}")


def write_dfttools_h5(data: dict, h5_path: str):
    """
    Write H(k) and correlated subspace in TRIQS/DFTTools HDF5 format.

    The format expected by SumkDFT:
      /dft_input/
        hopping            — H(k) [n_k, SP, n_orbitals, n_orbitals]
        n_k                — number of k-points
        SP                 — spin-polarization (0=no, 1=yes)
        SO                 — spin-orbit (0=no, 1=yes)
        n_shells           — number of atomic shells
        n_corr_shells      — number of correlated shells
        n_inequiv_shells   — number of inequivalent correlated shells
        corr_shells        — list of corr_shell dicts
        n_orbitals         — max orbital dimension
        proj_mat           — projector matrices
        bz_weights         — k-point weights (uniform)
        ...
    """
    nk = data["hk"].shape[0]
    norb = data["hk"].shape[1]
    corr_shells = data["corr_shells"]
    n_shells = len(corr_shells)
    n_inequiv = len(set(data["corr_to_inequiv"].tolist()))

    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("dft_input")

        # H(k) with spin index: shape [nk, 1, norb, norb] for non-spin-polarized
        hk_4d = data["hk"][:, np.newaxis, :, :]
        grp.create_dataset("hopping", data=hk_4d)

        grp.create_dataset("n_k", data=nk)
        grp.create_dataset("SP", data=0)  # non-spin-polarized for now
        grp.create_dataset("SO", data=0)  # no SOC in DMFT (handled by DFT)
        grp.create_dataset("n_shells", data=n_shells)
        grp.create_dataset("n_corr_shells", data=n_shells)
        grp.create_dataset("n_inequiv_shells", data=n_inequiv)
        grp.create_dataset("n_orbitals", data=norb)

        # Correlated shells info
        for i, sh in enumerate(corr_shells):
            sh_grp = grp.create_group(f"corr_shells/{i}")
            sh_grp.create_dataset("atom", data=sh["atom"])
            sh_grp.create_dataset("l", data=sh["l"])
            sh_grp.create_dataset("dim", data=sh["dim"])
            sh_grp.create_dataset("sort", data=sh.get("sort", i))
            sh_grp.create_dataset("SO", data=0)

        # corr_to_inequiv mapping
        grp.create_dataset("corr_to_inequiv", data=data["corr_to_inequiv"])

        # Projectors: [nk, n_shells, max_dim, norb]
        grp.create_dataset("proj_mat", data=data["proj_mat"][:, np.newaxis, :, :, :])

        # Uniform BZ weights
        bz_weights = np.full(nk, 1.0 / nk)
        grp.create_dataset("bz_weights", data=bz_weights)

        # Chemical potential
        grp.create_dataset("chemical_potential", data=data["fermi_energy"])

        # Metadata
        grp.attrs["source"] = "qae-dmft-bundle"
        grp.attrs["formula"] = data["formula"]
        grp.attrs["pressure_gpa"] = data["pressure_gpa"]


def run_solid_dmft(work_dir: str, config: dict) -> dict:
    """
    Execute solid_dmft. Returns a result dict.

    solid_dmft is MPI-aware and will use all available ranks when launched
    via mpirun. The CTHYB solver distributes Monte Carlo samples across ranks.
    """
    import subprocess

    seedname = config["general"]["seedname"]
    config_path = os.path.join(work_dir, "dmft_config.toml")

    mpi_ranks = int(os.environ.get("DMFT_MPI_RANKS", 20))

    t0 = time.time()
    print(f"[DMFT] Starting solid_dmft: {seedname}, {mpi_ranks} MPI ranks")

    cmd = [
        "mpirun", "-np", str(mpi_ranks),
        "--oversubscribe",
        "--allow-run-as-root",
        "python3", "-m", "solid_dmft",
        config_path,
    ]

    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=86400,  # 24h hard cap
    )

    elapsed = time.time() - t0
    print(f"[DMFT] solid_dmft finished in {elapsed:.0f}s, exit={result.returncode}")

    if result.returncode != 0:
        print(f"[DMFT] STDERR: {result.stderr[-2000:]}")
        return {
            "converged": False,
            "error": result.stderr[-500:],
            "elapsed_seconds": elapsed,
        }

    # Parse results from the output HDF5
    return parse_dmft_results(work_dir, seedname, elapsed)


def parse_dmft_results(work_dir: str, seedname: str, elapsed: float) -> dict:
    """
    Parse solid_dmft output HDF5 for key observables.

    solid_dmft writes results to {seedname}.h5 under /DMFT_results/
    """
    h5_path = os.path.join(work_dir, f"{seedname}.h5")

    result = {
        "converged": False,
        "elapsed_seconds": elapsed,
        "self_energy_available": False,
        "spectral_function_available": False,
    }

    if not os.path.exists(h5_path):
        result["error"] = "Output HDF5 not found"
        return result

    try:
        with h5py.File(h5_path, "r") as f:
            if "DMFT_results" not in f:
                result["error"] = "No DMFT_results group in output"
                return result

            dmft_grp = f["DMFT_results"]

            # Check convergence from the last iteration
            last_iter = sorted([k for k in dmft_grp.keys() if k.startswith("it_")])[-1]
            iter_grp = dmft_grp[last_iter]

            # Orbital occupations
            if "orb_occ" in iter_grp:
                result["orbital_occupations"] = iter_grp["orb_occ"][()].tolist()

            # Self-energy (Matsubara) — key DMFT output
            if "Sigma_iw" in iter_grp:
                result["self_energy_available"] = True

            # Chemical potential convergence
            if "mu" in iter_grp:
                result["chemical_potential"] = float(iter_grp["mu"][()])

            # Quasiparticle weight Z = (1 - dSigma/dw|w=0)^-1
            if "Z" in iter_grp:
                result["quasiparticle_weight"] = iter_grp["Z"][()].tolist()

            # Check if converged (chemical potential change < threshold)
            if "convergence_obs" in dmft_grp:
                conv = dmft_grp["convergence_obs"]
                if "d_mu" in conv and len(conv["d_mu"]) > 0:
                    last_dmu = abs(float(conv["d_mu"][-1]))
                    result["converged"] = last_dmu < 0.01  # eV

            result["n_iterations"] = len([k for k in dmft_grp.keys() if k.startswith("it_")])

    except Exception as e:
        result["error"] = f"Failed to parse output: {str(e)}"

    return result


def run_full_pipeline(
    bundle_path: str,
    work_dir: str,
    skip_vertex: bool = False,
    mpi_ranks: int = 20,
    vertex_channel: str = "ph",
) -> dict:
    """
    Run the complete DMFT + vertex + BSE + pairing pipeline:

      Phase 1: Single-particle DMFT (self-consistent Σ)
      Phase 2: Two-particle G² measurement (CTHYB measure_G2)
      Phase 3: Local BSE inversion (extract Γ_loc)
      Phase 4: Pairing susceptibility (λ_pair eigenvalue + gap symmetry)

    Args:
        bundle_path: path to QAE DMFT bundle HDF5
        work_dir:    working directory for all calculations
        skip_vertex: if True, only run Phase 1 (single-particle DMFT)
        mpi_ranks:   MPI ranks for CTHYB
        vertex_channel: "ph" (default) or "pp" — pp eliminates ph→pp crossing
                        approximation but ~2× the QMC cost

    Returns:
        Combined results dict with all phases
    """
    os.makedirs(work_dir, exist_ok=True)

    print(f"[DMFT] Loading bundle: {bundle_path}")
    data = load_bundle(bundle_path)
    formula = data["formula"]
    print(f"[DMFT] Formula: {formula}, P={data['pressure_gpa']} GPa")
    print(f"[DMFT] H(k) shape: {data['hk'].shape}, corr_shells: {len(data['corr_shells'])}")
    print(f"[DMFT] Magnetic ordering: {data['magnetic_ordering']}, regime: {data['correlation_regime']}")

    all_results = {"formula": formula, "phases": {}}

    # ── Phase 1: Single-particle DMFT ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[DMFT] Phase 1: Single-particle DMFT self-consistency")
    print(f"{'='*60}")

    config = build_solid_dmft_config(data, work_dir)
    write_solid_dmft_inputs(data, config, work_dir)
    dmft_results = run_solid_dmft(work_dir, config)
    all_results["phases"]["dmft_1p"] = dmft_results

    if not dmft_results.get("converged"):
        print(f"[DMFT] Phase 1 did not converge — stopping pipeline")
        all_results["converged"] = False
        all_results["stopped_at"] = "dmft_1p"
        _save_results(all_results, work_dir)
        return all_results

    if skip_vertex:
        print(f"[DMFT] Vertex measurement skipped (skip_vertex=True)")
        all_results["converged"] = True
        all_results["stopped_at"] = "dmft_1p_only"
        _save_results(all_results, work_dir)
        return all_results

    # ── Phase 2: Two-particle vertex measurement ─────────────────────────
    print(f"\n{'='*60}")
    print(f"[DMFT] Phase 2: Two-particle G² measurement")
    print(f"{'='*60}")

    try:
        from vertex_measurement import run_vertex_measurement

        seedname = config["general"]["seedname"]
        converged_h5 = os.path.join(work_dir, f"{seedname}.h5")
        vertex_work_dir = os.path.join(work_dir, "vertex")

        vertex_results = run_vertex_measurement(
            work_dir=vertex_work_dir,
            converged_h5_path=converged_h5,
            converged_config=config,
            data=data,
            mpi_ranks=mpi_ranks,
            channel=vertex_channel,
        )
        all_results["phases"]["vertex_g2"] = vertex_results

        if not vertex_results.get("converged"):
            print(f"[DMFT] Phase 2 failed — stopping pipeline")
            all_results["converged"] = False
            all_results["stopped_at"] = "vertex_g2"
            _save_results(all_results, work_dir)
            return all_results

    except ImportError as e:
        print(f"[DMFT] Phase 2 skipped (import error): {e}")
        all_results["phases"]["vertex_g2"] = {"skipped": True, "error": str(e)}
        all_results["stopped_at"] = "vertex_g2_import"
        _save_results(all_results, work_dir)
        return all_results

    # ── Phase 3: Local BSE inversion ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[DMFT] Phase 3: Local Bethe-Salpeter equation")
    print(f"{'='*60}")

    try:
        from bse_solver import run_bse_solver

        vertex_data_path = vertex_results.get("vertex_data_path")
        if not vertex_data_path:
            raise ValueError("No vertex_data_path from Phase 2")

        bse_work_dir = os.path.join(work_dir, "bse")
        bse_results = run_bse_solver(vertex_data_path, bse_work_dir)
        all_results["phases"]["bse"] = bse_results

        if not bse_results.get("converged"):
            print(f"[DMFT] Phase 3 failed — stopping pipeline")
            all_results["converged"] = False
            all_results["stopped_at"] = "bse"
            _save_results(all_results, work_dir)
            return all_results

    except (ImportError, Exception) as e:
        print(f"[DMFT] Phase 3 failed: {e}")
        all_results["phases"]["bse"] = {"converged": False, "error": str(e)}
        all_results["stopped_at"] = "bse"
        _save_results(all_results, work_dir)
        return all_results

    # ── Phase 4: Pairing susceptibility ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[DMFT] Phase 4: Pairing susceptibility eigenvalue problem")
    print(f"{'='*60}")

    try:
        from pairing_susceptibility import run_pairing_susceptibility

        bse_results_path = bse_results.get("bse_results_path")
        if not bse_results_path:
            raise ValueError("No bse_results_path from Phase 3")

        seedname = config["general"]["seedname"]
        dmft_h5 = os.path.join(work_dir, f"{seedname}.h5")
        beta = config["general"].get("beta", 40.0)

        pairing_work_dir = os.path.join(work_dir, "pairing")
        pairing_results = run_pairing_susceptibility(
            bse_results_path=bse_results_path,
            dmft_h5_path=dmft_h5,
            bundle_data=data,
            work_dir=pairing_work_dir,
            beta=beta,
        )
        all_results["phases"]["pairing"] = pairing_results

        # Summary
        lp = pairing_results.get("lambda_pair", 0)
        sym = pairing_results.get("gap_symmetry", "unknown")
        T_K = pairing_results.get("temperature_K", 0)
        print(f"\n[DMFT] === Pairing Summary ===")
        print(f"  λ_pair = {lp:.4f} at T = {T_K:.0f} K")
        print(f"  Gap symmetry: {sym}")
        print(f"  Unconventional: {pairing_results.get('is_unconventional', False)}")
        if lp >= 1.0:
            print(f"  *** SUPERCONDUCTING INSTABILITY at T = {T_K:.0f} K ***")
        elif lp > 0.5:
            print(f"  Strong pairing tendency — Tc likely below {T_K:.0f} K")

    except (ImportError, Exception) as e:
        print(f"[DMFT] Phase 4 failed: {e}")
        all_results["phases"]["pairing"] = {"converged": False, "error": str(e)}

    all_results["converged"] = True
    _save_results(all_results, work_dir)
    return all_results


def _save_results(results: dict, work_dir: str):
    """Save combined pipeline results to JSON."""
    results_path = os.path.join(work_dir, "dmft_full_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[DMFT] Full results written to {results_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mpirun -np N python3 run-dmft.py <bundle.h5> [work_dir] [--skip-vertex]")
        sys.exit(1)

    bundle_path = sys.argv[1]
    work_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else os.path.join(
        os.environ.get("DMFT_WORK_DIR", "/data/dmft_jobs"),
        os.path.splitext(os.path.basename(bundle_path))[0]
    )
    skip_vertex = "--skip-vertex" in sys.argv

    mpi_ranks = int(os.environ.get("DMFT_MPI_RANKS", 20))

    results = run_full_pipeline(
        bundle_path=bundle_path,
        work_dir=work_dir,
        skip_vertex=skip_vertex,
        mpi_ranks=mpi_ranks,
    )

    print(f"\n[DMFT] === Final Results ===")
    # Print summary without the large arrays
    summary = {k: v for k, v in results.items() if k != "phases"}
    for phase_name, phase_data in results.get("phases", {}).items():
        if isinstance(phase_data, dict):
            summary[phase_name] = {
                k: v for k, v in phase_data.items()
                if not isinstance(v, (np.ndarray, list)) or (isinstance(v, list) and len(v) < 20)
            }
    print(json.dumps(summary, indent=2, default=str))
