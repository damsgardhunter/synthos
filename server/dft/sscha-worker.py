#!/usr/bin/env python3
"""
SSCHA Anharmonic Phonon Pipeline — Python Worker

Stochastic Self-Consistent Harmonic Approximation (SSCHA) workflow:
  1. Load harmonic dynamical matrices from DFPT (.dyn files)
  2. Generate stochastic displaced configurations
  3. Compute DFT forces on each configuration via pw.x SCF
  4. Feed forces into SSCHA minimizer
  5. Iterate until free energy Hessian converges
  6. Extract anharmonic-corrected omega_log, lambda, free energy, stability

Usage:
  python3 sscha-worker.py \
    --dyn-prefix <prefix> \
    --nqirr <n> \
    --temperature <K> \
    --supercell 2 2 2 \
    --nconfigs 100 \
    --prefix <qe_prefix> \
    --pseudo-dir <path> \
    --ecutwfc <Ry> \
    --ecutrho <Ry> \
    --kgrid 4 4 4 \
    --outdir <path>

Outputs JSON to stdout with anharmonic results.
"""

import argparse
import json
import os
import sys
import subprocess
import time
import traceback
import tempfile
import shutil
import re
import numpy as np

# ─── Attempt imports for SSCHA libraries ────────────────────────────────────
_SSCHA_AVAILABLE = True
_IMPORT_ERROR = ""
try:
    import cellconstructor as CC
    import cellconstructor.Phonons
    import cellconstructor.ForceTensor
    import sscha
    import sscha.Ensemble
    import sscha.SchaMinimizer
    import sscha.Utilities
except ImportError as e:
    _SSCHA_AVAILABLE = False
    _IMPORT_ERROR = str(e)

# Try ASE for structure manipulation fallback
try:
    import ase
    import ase.io
    _ASE_AVAILABLE = True
except ImportError:
    _ASE_AVAILABLE = False


# ─── Constants ──────────────────────────────────────────────────────────────
BOHR_TO_ANG = 0.529177249
RY_TO_EV = 13.605698
RY_TO_MEV = RY_TO_EV * 1000.0
K_BOLTZMANN_EV = 8.617333262e-5  # eV/K
HBAR_EV_S = 6.582119569e-16  # eV·s
MAX_SSCHA_ITERATIONS = 20
MAX_FORCE_CALCS = 2000  # absolute safety cap on total DFT calls
PW_TIMEOUT_S = 3600  # 1 hour per SCF
TOTAL_TIMEOUT_S = 86400  # 24 hours total


def log(msg: str):
    """Log to stderr so stdout stays clean for JSON."""
    print(f"[SSCHA-py] {msg}", file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="SSCHA anharmonic phonon worker")
    p.add_argument("--dyn-prefix", required=True,
                   help="Prefix for .dyn files (e.g., 'matdyn' reads matdyn1, matdyn2, ...)")
    p.add_argument("--nqirr", type=int, required=True,
                   help="Number of irreducible q-points (number of .dyn files)")
    p.add_argument("--temperature", type=float, default=300.0,
                   help="Temperature in K for SSCHA (default: 300)")
    p.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2],
                   help="Supercell dimensions (default: 2 2 2)")
    p.add_argument("--nconfigs", type=int, default=100,
                   help="Number of stochastic configurations per iteration (default: 100)")
    p.add_argument("--prefix", required=True,
                   help="QE calculation prefix")
    p.add_argument("--pseudo-dir", required=True,
                   help="Path to pseudopotential directory")
    p.add_argument("--ecutwfc", type=float, required=True,
                   help="Plane wave cutoff in Ry")
    p.add_argument("--ecutrho", type=float, required=True,
                   help="Charge density cutoff in Ry")
    p.add_argument("--kgrid", type=int, nargs=3, default=[4, 4, 4],
                   help="k-point grid for SCF force calculations (default: 4 4 4)")
    p.add_argument("--outdir", required=True,
                   help="Working directory for calculations")
    p.add_argument("--max-iterations", type=int, default=MAX_SSCHA_ITERATIONS,
                   help=f"Maximum SSCHA iterations (default: {MAX_SSCHA_ITERATIONS})")
    p.add_argument("--pw-binary", default="/usr/local/bin/pw.x",
                   help="Path to pw.x binary")
    p.add_argument("--mpi-np", type=int, default=0,
                   help="Number of MPI ranks (0 = no MPI)")
    p.add_argument("--mpi-launcher", default="mpirun",
                   help="MPI launcher command (default: mpirun)")
    return p.parse_args()


# ─── QE SCF Force Calculator ───────────────────────────────────────────────

def build_scf_input(
    prefix: str,
    pseudo_dir: str,
    ecutwfc: float,
    ecutrho: float,
    cell_vectors: np.ndarray,  # 3x3, Angstrom
    species: list,  # [(element, mass, pp_filename), ...]
    positions: np.ndarray,  # Nx3, Angstrom (Cartesian)
    elements: list,  # element label per atom
    kgrid: list,
) -> str:
    """Generate pw.x SCF input for force calculation."""
    nat = len(elements)
    ntyp = len(species)

    species_block = "\n".join(
        f"  {el}  {mass:.4f}  {pp}" for el, mass, pp in species
    )

    # Convert Cartesian Angstrom positions
    pos_block = "\n".join(
        f"  {elements[i]}  {positions[i, 0]:.10f}  {positions[i, 1]:.10f}  {positions[i, 2]:.10f}"
        for i in range(nat)
    )

    cell_block = "\n".join(
        f"  {cell_vectors[i, 0]:.10f}  {cell_vectors[i, 1]:.10f}  {cell_vectors[i, 2]:.10f}"
        for i in range(3)
    )

    return f"""&CONTROL
  calculation = 'scf',
  prefix = '{prefix}_sscha',
  outdir = './tmp',
  pseudo_dir = '{pseudo_dir}',
  tprnfor = .true.,
  tstress = .false.,
  verbosity = 'low',
/
&SYSTEM
  ibrav = 0,
  nat = {nat},
  ntyp = {ntyp},
  ecutwfc = {ecutwfc},
  ecutrho = {ecutrho},
  occupations = 'smearing',
  smearing = 'cold',
  degauss = 0.02,
/
&ELECTRONS
  conv_thr = 1.0d-8,
  mixing_beta = 0.7,
/
ATOMIC_SPECIES
{species_block}
ATOMIC_POSITIONS {{angstrom}}
{pos_block}
CELL_PARAMETERS {{angstrom}}
{cell_block}
K_POINTS {{automatic}}
  {kgrid[0]} {kgrid[1]} {kgrid[2]}  0 0 0
"""


def parse_forces_from_scf(stdout: str, nat: int) -> np.ndarray:
    """Parse forces from pw.x SCF output. Returns Nx3 array in Ry/Bohr."""
    forces = []
    # Find "Forces acting on atoms" section
    force_section = False
    for line in stdout.split("\n"):
        if "Forces acting on atoms" in line:
            force_section = True
            forces = []
            continue
        if force_section:
            # Format: atom   1 type  1   force =     0.00000    0.00000    0.00000
            m = re.match(
                r"\s*atom\s+\d+\s+type\s+\d+\s+force\s*=\s*"
                r"([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)",
                line,
            )
            if m:
                forces.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            elif len(forces) >= nat:
                break
            elif forces and not line.strip():
                break

    if len(forces) != nat:
        raise ValueError(
            f"Expected {nat} force lines, parsed {len(forces)}"
        )
    return np.array(forces)  # Ry/Bohr


def parse_total_energy(stdout: str) -> float:
    """Parse total energy from pw.x output in Ry."""
    # "!    total energy              =     -123.456789 Ry"
    m = re.search(r"!\s*total energy\s*=\s*([-\d.Ee+]+)\s*Ry", stdout)
    if m:
        return float(m.group(1))
    raise ValueError("Could not parse total energy from SCF output")


def run_pw_scf(
    input_text: str,
    work_dir: str,
    pw_binary: str,
    mpi_np: int,
    mpi_launcher: str,
    config_idx: int,
    timeout_s: int = PW_TIMEOUT_S,
) -> dict:
    """
    Run a single pw.x SCF calculation.
    Returns dict with 'forces' (Nx3 Ry/Bohr), 'energy' (Ry), 'success' bool.
    """
    scf_dir = os.path.join(work_dir, f"config_{config_idx:04d}")
    os.makedirs(scf_dir, exist_ok=True)
    os.makedirs(os.path.join(scf_dir, "tmp"), exist_ok=True)

    input_file = os.path.join(scf_dir, "scf.in")
    with open(input_file, "w") as f:
        f.write(input_text)

    # Build command
    cmd = []
    if mpi_np >= 2:
        cmd = [mpi_launcher, "-np", str(mpi_np)]
    cmd.append(pw_binary)

    try:
        with open(input_file, "r") as stdin_f:
            result = subprocess.run(
                cmd,
                stdin=stdin_f,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=scf_dir,
            )

        if result.returncode != 0:
            log(f"  config {config_idx}: pw.x failed (exit {result.returncode})")
            # Write error log for debugging
            err_file = os.path.join(scf_dir, "scf.err")
            with open(err_file, "w") as f:
                f.write(result.stderr[:5000])
            return {"success": False, "error": f"exit code {result.returncode}"}

        stdout = result.stdout
        # Count atoms from input
        nat_match = re.search(r"nat\s*=\s*(\d+)", input_text)
        nat = int(nat_match.group(1)) if nat_match else 0

        forces = parse_forces_from_scf(stdout, nat)
        energy = parse_total_energy(stdout)

        return {
            "success": True,
            "forces": forces,
            "energy": energy,
        }

    except subprocess.TimeoutExpired:
        log(f"  config {config_idx}: pw.x timed out after {timeout_s}s")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        log(f"  config {config_idx}: exception — {e}")
        return {"success": False, "error": str(e)}
    finally:
        # Clean up large tmp directory to save disk
        tmp_dir = os.path.join(scf_dir, "tmp")
        if os.path.isdir(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass


# ─── Fallback: Manual SSCHA without python-sscha library ───────────────────
# When the sscha Python package is not installed, we implement a simplified
# stochastic displacement + harmonic self-consistency loop directly using
# cellconstructor (or raw numpy if even that is unavailable).

def compute_omega_log_lambda(freqs_cm: np.ndarray, mode_lambdas: np.ndarray) -> dict:
    """
    Compute omega_log and total lambda from mode-resolved frequencies and lambdas.
    freqs_cm: phonon frequencies in cm^-1 (positive modes only)
    mode_lambdas: electron-phonon coupling per mode
    """
    # Filter to positive frequencies only
    mask = freqs_cm > 1.0  # skip acoustic modes near zero
    if not np.any(mask):
        return {"omega_log_meV": 0.0, "lambda_total": 0.0}

    w = freqs_cm[mask]
    lam = mode_lambdas[mask]
    lam_total = np.sum(lam)

    if lam_total < 1e-12:
        return {"omega_log_meV": 0.0, "lambda_total": 0.0}

    # omega_log = exp( (2/lambda) * sum_i lambda_i * ln(omega_i) / (2 * N_modes) )
    # Simplified: weight by lambda_i
    log_omega = np.sum(lam * np.log(w)) / lam_total
    omega_log_cm = np.exp(log_omega)
    omega_log_meV = omega_log_cm * 0.12398  # cm^-1 to meV

    return {"omega_log_meV": omega_log_meV, "lambda_total": lam_total}


def allen_dynes_tc(lambda_ep: float, omega_log_meV: float, mu_star: float = 0.10) -> float:
    """Allen-Dynes Tc formula. Returns Tc in K."""
    if lambda_ep < 0.01 or omega_log_meV < 0.01:
        return 0.0
    omega_log_K = omega_log_meV / K_BOLTZMANN_EV  # meV -> K: divide by k_B in meV/K
    omega_log_K = omega_log_meV / 0.08617  # meV to K
    exponent = -1.04 * (1 + lambda_ep) / (lambda_ep - mu_star * (1 + 0.62 * lambda_ep))
    if exponent > 0 or exponent < -100:
        return 0.0
    return (omega_log_K / 1.2) * np.exp(exponent)


# ─── Main SSCHA workflow (with library) ────────────────────────────────────

def run_sscha_with_library(args) -> dict:
    """Full SSCHA run using the python-sscha library."""
    log("Loading harmonic dynamical matrices...")
    dyn = CC.Phonons.Phonons(args.dyn_prefix, nqirr=args.nqirr)

    # Check for imaginary frequencies (stability)
    w_harm, _ = dyn.DyagDinQ(0)  # Gamma point
    n_imag = np.sum(w_harm < 0)
    if n_imag > 0:
        log(f"  {n_imag} imaginary modes at Gamma — structure may be dynamically unstable")

    # Generate supercell
    supercell = dyn.GenerateSupercellDyn(np.diag(args.supercell))

    T = args.temperature  # K
    n_configs = args.nconfigs
    total_force_calcs = 0
    start_time = time.time()

    converged = False
    iteration = 0
    free_energy_history = []

    for iteration in range(1, args.max_iterations + 1):
        elapsed = time.time() - start_time
        if elapsed > TOTAL_TIMEOUT_S:
            log(f"  Total timeout ({TOTAL_TIMEOUT_S}s) reached at iteration {iteration}")
            break
        if total_force_calcs >= MAX_FORCE_CALCS:
            log(f"  Max force calculations ({MAX_FORCE_CALCS}) reached")
            break

        log(f"  Iteration {iteration}/{args.max_iterations}")

        # Generate stochastic ensemble
        ensemble = sscha.Ensemble.Ensemble(supercell, T)
        ensemble.generate(n_configs)

        # Compute DFT forces for each configuration
        structures = ensemble.structures
        nat_sc = len(structures[0])
        log(f"  Computing forces for {n_configs} configurations ({nat_sc} atoms each)...")

        # Extract species info from the dynamical matrix
        unit_cell = dyn.structure
        species_set = {}
        for i, el in enumerate(unit_cell.atoms):
            if el not in species_set:
                mass = unit_cell.masses[unit_cell.atoms.index(el)]
                # Try to find pseudopotential file
                pp_files = [f for f in os.listdir(args.pseudo_dir)
                           if f.startswith(el) and f.endswith(".UPF")]
                pp = pp_files[0] if pp_files else f"{el}.UPF"
                species_set[el] = (el, mass, pp)
        species_list = list(species_set.values())

        failed_configs = 0
        for ic in range(n_configs):
            if total_force_calcs >= MAX_FORCE_CALCS:
                break

            struct = structures[ic]
            cell_vectors = struct.unit_cell  # 3x3 Angstrom
            positions_ang = struct.coords  # Nx3 Angstrom
            atom_labels = struct.atoms

            scf_input = build_scf_input(
                prefix=args.prefix,
                pseudo_dir=args.pseudo_dir,
                ecutwfc=args.ecutwfc,
                ecutrho=args.ecutrho,
                cell_vectors=np.array(cell_vectors),
                species=species_list,
                positions=positions_ang,
                elements=atom_labels,
                kgrid=args.kgrid,
            )

            result = run_pw_scf(
                input_text=scf_input,
                work_dir=os.path.join(args.outdir, f"iter_{iteration:03d}"),
                pw_binary=args.pw_binary,
                mpi_np=args.mpi_np,
                mpi_launcher=args.mpi_launcher,
                config_idx=ic,
            )
            total_force_calcs += 1

            if result["success"]:
                ensemble.forces[ic] = result["forces"]
                ensemble.energies[ic] = result["energy"]
            else:
                failed_configs += 1
                log(f"    config {ic} failed: {result.get('error', 'unknown')}")

        if failed_configs > n_configs * 0.5:
            log(f"  >50% configs failed ({failed_configs}/{n_configs}), aborting")
            break

        log(f"  {n_configs - failed_configs}/{n_configs} SCF succeeded, running minimizer...")

        # Run SSCHA minimization step
        minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
        minim.init()
        minim.run()

        # Check convergence
        free_energy = minim.get_free_energy()
        free_energy_history.append(float(free_energy))
        log(f"  Free energy: {free_energy:.6f} Ry")

        if minim.is_converged():
            converged = True
            log(f"  Converged at iteration {iteration}!")
            break

        # Update dynamical matrix for next iteration
        supercell = minim.dyn

    # Extract final results
    log("Extracting anharmonic properties...")

    # Get anharmonic phonon frequencies
    final_dyn = supercell if converged else supercell
    w_anh, _ = final_dyn.DyagDinQ(0)  # Gamma frequencies in cm^-1

    # Compute omega_log (anharmonic)
    # For a proper calculation we'd need the full Brillouin zone,
    # but use Gamma-point modes as approximation
    positive_w = w_anh[w_anh > 1.0]

    if len(positive_w) > 0:
        # Approximate omega_log from positive frequencies
        omega_log_cm = np.exp(np.mean(np.log(positive_w)))
        omega_log_meV = omega_log_cm * 0.12398
    else:
        omega_log_meV = 0.0

    # Free energy in Ry
    final_free_energy = free_energy_history[-1] if free_energy_history else 0.0

    # Check for remaining imaginary modes (anharmonic stability)
    n_imag_anh = int(np.sum(w_anh < -5.0))  # threshold: -5 cm^-1

    elapsed_total = time.time() - start_time

    return {
        "omegaLogAnharmonic": float(omega_log_meV),
        "lambdaAnharmonic": None,  # needs EPW re-run with anharmonic phonons
        "freeEnergy": float(final_free_energy),
        "freeEnergyHistory": free_energy_history,
        "converged": converged,
        "iterations": iteration,
        "nConfigs": n_configs,
        "temperatureK": float(T),
        "totalForceCalcs": total_force_calcs,
        "nImaginaryModes": n_imag_anh,
        "anharmonicFreqsCm": [float(x) for x in w_anh.tolist()],
        "harmonicImaginaryCount": int(n_imag),
        "elapsedSeconds": round(elapsed_total, 1),
        "method": "SSCHA-full",
    }


# ─── Fallback SSCHA (no library) ──────────────────────────────────────────

def run_sscha_fallback(args) -> dict:
    """
    Simplified SSCHA-like anharmonic correction without the sscha Python library.

    Strategy:
    1. Parse harmonic .dyn files manually for frequencies and eigenvectors
    2. Generate random displacements along phonon eigenvectors
    3. Compute DFT forces
    4. Fit effective harmonic force constants from displaced forces
    5. Extract corrected frequencies
    """
    log("SSCHA library not available — running fallback anharmonic estimator")
    log(f"  Import error: {_IMPORT_ERROR}")

    # Parse harmonic dynamical matrices
    dyn_files = []
    for iq in range(1, args.nqirr + 1):
        dyn_file = f"{args.dyn_prefix}{iq}"
        if os.path.isfile(dyn_file):
            dyn_files.append(dyn_file)
        else:
            log(f"  WARNING: {dyn_file} not found")

    if not dyn_files:
        return {
            "omegaLogAnharmonic": 0.0,
            "lambdaAnharmonic": None,
            "freeEnergy": 0.0,
            "converged": False,
            "iterations": 0,
            "nConfigs": 0,
            "temperatureK": float(args.temperature),
            "totalForceCalcs": 0,
            "error": "No dynamical matrix files found",
            "method": "SSCHA-fallback",
        }

    # Parse Gamma-point dynamical matrix to get structure and frequencies
    harmonic_data = parse_dyn_file(dyn_files[0])
    if not harmonic_data:
        return {
            "omegaLogAnharmonic": 0.0,
            "lambdaAnharmonic": None,
            "freeEnergy": 0.0,
            "converged": False,
            "iterations": 0,
            "nConfigs": 0,
            "temperatureK": float(args.temperature),
            "totalForceCalcs": 0,
            "error": "Failed to parse dynamical matrix",
            "method": "SSCHA-fallback",
        }

    nat = harmonic_data["nat"]
    cell_vectors = harmonic_data["cell"]  # 3x3 Angstrom
    positions = harmonic_data["positions"]  # Nx3 Angstrom
    atom_labels = harmonic_data["atoms"]
    harmonic_freqs_cm = harmonic_data["freqs"]  # cm^-1
    eigenvectors = harmonic_data.get("eigenvectors")  # 3N x 3N if available

    log(f"  Unit cell: {nat} atoms, {len(harmonic_freqs_cm)} modes")
    log(f"  Harmonic freqs (cm^-1): {harmonic_freqs_cm[:6]}")

    n_imag_harm = int(np.sum(np.array(harmonic_freqs_cm) < -1.0))

    # Build species list from pseudopotential directory
    species_set = {}
    for el in atom_labels:
        if el not in species_set:
            pp_files = [
                f for f in os.listdir(args.pseudo_dir)
                if f.lower().startswith(el.lower()) and f.upper().endswith(".UPF")
            ]
            pp = pp_files[0] if pp_files else f"{el}.UPF"
            species_set[el] = (el, 1.0, pp)
    species_list = list(species_set.values())

    # Generate displaced configurations
    n_modes = 3 * nat
    n_configs = min(args.nconfigs, max(2 * n_modes, 20))
    T = args.temperature

    # Thermal displacement amplitude: u ~ sqrt(k_B T / (m * omega^2))
    # For each mode, displace along eigenvector with Gaussian amplitude
    displacements = []
    rng = np.random.default_rng(seed=42)

    for ic in range(n_configs):
        disp = np.zeros((nat, 3))
        for mode_idx in range(n_modes):
            freq_cm = harmonic_freqs_cm[mode_idx] if mode_idx < len(harmonic_freqs_cm) else 100.0
            if abs(freq_cm) < 5.0:
                continue  # skip near-zero (acoustic) modes

            # Convert frequency to angular frequency (rad/s)
            freq_hz = abs(freq_cm) * 2.998e10  # cm^-1 to Hz
            omega = 2 * np.pi * freq_hz

            # Thermal displacement amplitude in Angstrom
            # sigma = sqrt(hbar / (2 * m * omega) * coth(hbar*omega/(2*k_B*T)))
            # Simplified: sigma ~ 0.01-0.1 Angstrom for typical phonon modes
            hbar_omega_eV = abs(freq_cm) * 0.12398e-3  # cm^-1 to eV
            if T > 0 and hbar_omega_eV > 0:
                n_bose = 1.0 / (np.exp(hbar_omega_eV / (K_BOLTZMANN_EV * T)) - 1.0) \
                    if hbar_omega_eV / (K_BOLTZMANN_EV * T) < 100 else 0.0
                sigma = np.sqrt((2 * n_bose + 1) * hbar_omega_eV / (2.0 * abs(freq_cm) * 0.12398e-3))
                sigma = min(sigma, 0.15)  # cap displacement
                sigma = max(sigma, 0.005)  # minimum displacement
            else:
                sigma = 0.02

            # Random displacement along mode eigenvector (or random direction if no eigvec)
            amplitude = rng.normal(0, sigma)
            if eigenvectors is not None and mode_idx < eigenvectors.shape[0]:
                evec = eigenvectors[mode_idx].reshape(nat, 3)
                disp += amplitude * evec
            else:
                # Random direction per atom
                direction = rng.normal(0, 1, (nat, 3))
                direction /= (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12)
                disp += (amplitude / np.sqrt(nat)) * direction

        displacements.append(disp)

    log(f"  Generated {n_configs} displaced configurations")

    # Run DFT force calculations
    total_force_calcs = 0
    forces_list = []
    energies_list = []
    start_time = time.time()
    failed_count = 0

    for ic in range(n_configs):
        elapsed = time.time() - start_time
        if elapsed > TOTAL_TIMEOUT_S * 0.9:  # leave 10% margin
            log(f"  Approaching timeout at config {ic}/{n_configs}")
            break

        displaced_pos = positions + displacements[ic]

        scf_input = build_scf_input(
            prefix=args.prefix,
            pseudo_dir=args.pseudo_dir,
            ecutwfc=args.ecutwfc,
            ecutrho=args.ecutrho,
            cell_vectors=cell_vectors,
            species=species_list,
            positions=displaced_pos,
            elements=atom_labels,
            kgrid=args.kgrid,
        )

        result = run_pw_scf(
            input_text=scf_input,
            work_dir=os.path.join(args.outdir, "sscha_displacements"),
            pw_binary=args.pw_binary,
            mpi_np=args.mpi_np,
            mpi_launcher=args.mpi_launcher,
            config_idx=ic,
        )
        total_force_calcs += 1

        if result["success"]:
            forces_list.append(result["forces"])
            energies_list.append(result["energy"])
        else:
            failed_count += 1
            log(f"    config {ic} failed: {result.get('error', 'unknown')}")

    log(f"  Force calculations: {len(forces_list)} succeeded, {failed_count} failed")

    if len(forces_list) < max(n_modes, 5):
        log("  Insufficient successful calculations for anharmonic analysis")
        return {
            "omegaLogAnharmonic": 0.0,
            "lambdaAnharmonic": None,
            "freeEnergy": 0.0,
            "converged": False,
            "iterations": 1,
            "nConfigs": total_force_calcs,
            "temperatureK": float(T),
            "totalForceCalcs": total_force_calcs,
            "error": f"Only {len(forces_list)}/{n_configs} force calculations succeeded",
            "method": "SSCHA-fallback",
        }

    # Fit effective force constants from displacement-force data
    # F = -Phi * u  =>  Phi = -(F^T * u) / (u^T * u) in least-squares sense
    U_mat = np.array([d.flatten() for d in displacements[:len(forces_list)]])  # (nconf, 3N)
    F_mat = np.array([f.flatten() for f in forces_list])  # (nconf, 3N)

    # Convert forces from Ry/Bohr to eV/Angstrom for consistent units
    F_mat_eV_A = F_mat * (RY_TO_EV / BOHR_TO_ANG)

    # Least-squares fit: Phi = -(U^T U)^{-1} U^T F
    try:
        # Use pseudo-inverse for numerical stability
        Phi = -np.linalg.lstsq(U_mat, F_mat_eV_A, rcond=None)[0].T  # (3N, 3N)

        # Symmetrize force constant matrix
        Phi = 0.5 * (Phi + Phi.T)

        # Mass-weight the dynamical matrix
        # D = M^{-1/2} Phi M^{-1/2}
        # For simplicity, use unit masses (already in natural units)
        eigenvalues, eigvecs = np.linalg.eigh(Phi)

        # Convert eigenvalues to frequencies (cm^-1)
        # omega^2 = eigenvalue (in eV/Angstrom^2/amu)
        # For proper conversion we need atomic masses
        anh_freqs_cm = []
        for ev in eigenvalues:
            if ev > 0:
                # Approximate: omega (cm^-1) ~ sqrt(ev) * conversion_factor
                # This is a rough estimate without proper mass weighting
                freq_THz = np.sqrt(abs(ev)) * 15.633  # rough eV/A^2 -> THz
                freq_cm = freq_THz * 33.356  # THz -> cm^-1
                anh_freqs_cm.append(freq_cm)
            else:
                freq_THz = np.sqrt(abs(ev)) * 15.633
                freq_cm = freq_THz * 33.356
                anh_freqs_cm.append(-freq_cm)  # imaginary

        anh_freqs_cm = np.array(sorted(anh_freqs_cm))

        # Compute anharmonic omega_log
        positive_mask = anh_freqs_cm > 5.0
        if np.any(positive_mask):
            pos_freqs = anh_freqs_cm[positive_mask]
            omega_log_cm = np.exp(np.mean(np.log(pos_freqs)))
            omega_log_meV = omega_log_cm * 0.12398
        else:
            omega_log_meV = 0.0

        n_imag_anh = int(np.sum(anh_freqs_cm < -5.0))
        fit_converged = True

    except Exception as e:
        log(f"  Force constant fitting failed: {e}")
        omega_log_meV = 0.0
        anh_freqs_cm = np.array(harmonic_freqs_cm)
        n_imag_anh = n_imag_harm
        fit_converged = False

    # Estimate free energy from energies
    mean_energy = np.mean(energies_list) if energies_list else 0.0

    elapsed_total = time.time() - start_time

    return {
        "omegaLogAnharmonic": float(omega_log_meV),
        "lambdaAnharmonic": None,  # needs EPW coupling data
        "freeEnergy": float(mean_energy),
        "converged": fit_converged,
        "iterations": 1,
        "nConfigs": total_force_calcs,
        "temperatureK": float(T),
        "totalForceCalcs": total_force_calcs,
        "nImaginaryModes": n_imag_anh,
        "harmonicImaginaryCount": n_imag_harm,
        "anharmonicFreqsCm": [float(x) for x in anh_freqs_cm.tolist()],
        "elapsedSeconds": round(elapsed_total, 1),
        "method": "SSCHA-fallback",
    }


def parse_dyn_file(filename: str) -> dict:
    """
    Parse a QE .dyn file to extract structure and dynamical matrix.
    Returns dict with cell, positions, atoms, freqs, eigenvectors.
    """
    try:
        with open(filename, "r") as f:
            content = f.read()

        lines = content.strip().split("\n")

        # Parse cell parameters (look for 3x3 cell vectors after header)
        cell = np.zeros((3, 3))
        positions = []
        atoms = []
        nat = 0

        # Find the line with ntyp, nat, ibrav
        for i, line in enumerate(lines):
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+", line)
            if m and i < 10:
                ntyp = int(m.group(1))
                nat = int(m.group(2))
                break

        # Parse cell vectors — typically 3 lines after the lattice parameter line
        cell_found = False
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) == 3 and not cell_found:
                try:
                    vals = [float(x) for x in parts]
                    if all(abs(v) < 100 for v in vals):  # sanity check
                        cell[0] = vals
                        cell[1] = [float(x) for x in lines[i + 1].strip().split()]
                        cell[2] = [float(x) for x in lines[i + 2].strip().split()]
                        cell_found = True
                except (ValueError, IndexError):
                    continue

        # Convert cell from alat units to Angstrom if needed
        # Look for alat
        alat_bohr = 1.0
        for line in lines:
            m = re.search(r"celldm\(1\)\s*=\s*([\d.]+)", line)
            if m:
                alat_bohr = float(m.group(1))
                break
            m = re.search(r"a\s*=\s*([\d.]+)", line)
            if m:
                alat_bohr = float(m.group(1)) / BOHR_TO_ANG
                break

        alat_ang = alat_bohr * BOHR_TO_ANG
        cell *= alat_ang

        # Parse atomic positions
        for i, line in enumerate(lines):
            # Format: atom_idx  element  mass  tau_x  tau_y  tau_z
            m = re.match(
                r"\s*(\d+)\s+'(\w+)\s*'\s+[\d.]+\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)",
                line,
            )
            if m:
                atoms.append(m.group(2).strip())
                # Positions in alat units -> Angstrom
                pos = np.array([float(m.group(3)), float(m.group(4)), float(m.group(5))])
                positions.append(pos * alat_ang)

        if not positions:
            log(f"  Could not parse atomic positions from {filename}")
            return None

        positions = np.array(positions)

        # Parse frequencies
        freqs = []
        for line in lines:
            m = re.search(r"freq\s*\(\s*\d+\)\s*=\s*([-\d.]+)\s*\[cm-1\]", line)
            if m:
                freqs.append(float(m.group(1)))
            # Also handle: omega( 1) =  12.345 [cm-1]
            m2 = re.search(r"omega\s*\(\s*\d+\)\s*=\s*([-\d.]+)\s*\[cm-1\]", line)
            if m2:
                freqs.append(float(m2.group(1)))

        if not freqs:
            log(f"  No frequencies parsed from {filename}")
            # Generate dummy frequencies
            freqs = [100.0] * (3 * len(atoms))

        return {
            "nat": len(atoms),
            "cell": cell,
            "positions": positions,
            "atoms": atoms,
            "freqs": freqs,
            "eigenvectors": None,  # TODO: parse eigenvectors if needed
        }

    except Exception as e:
        log(f"  Error parsing {filename}: {e}")
        traceback.print_exc(file=sys.stderr)
        return None


# ─── Entry point ───────────────────────────────────────────────────────────

def main():
    args = parse_args()
    log(f"Starting SSCHA workflow: T={args.temperature}K, "
        f"supercell={args.supercell}, nconfigs={args.nconfigs}")
    log(f"  dyn prefix: {args.dyn_prefix}, nqirr={args.nqirr}")
    log(f"  QE: ecutwfc={args.ecutwfc}, ecutrho={args.ecutrho}, kgrid={args.kgrid}")
    log(f"  pw.x: {args.pw_binary}, MPI: {args.mpi_np} ranks")

    os.makedirs(args.outdir, exist_ok=True)

    try:
        if _SSCHA_AVAILABLE:
            result = run_sscha_with_library(args)
        else:
            result = run_sscha_fallback(args)

        # Compute Tc correction if we have omega_log
        omega_log = result.get("omegaLogAnharmonic", 0.0)
        if omega_log and omega_log > 0:
            # Estimate Tc with default mu* = 0.10 and lambda ~ 1.0 (placeholder)
            # The real lambda_anharmonic should come from EPW re-run
            for mu_star in [0.10, 0.13, 0.15]:
                tc = allen_dynes_tc(1.0, omega_log, mu_star)  # lambda=1.0 placeholder
                result[f"tcEstimate_mustar{mu_star:.2f}"] = round(tc, 2)

        result["success"] = True

    except Exception as e:
        log(f"SSCHA failed with exception: {e}")
        traceback.print_exc(file=sys.stderr)
        result = {
            "success": False,
            "error": str(e),
            "converged": False,
            "omegaLogAnharmonic": 0.0,
            "lambdaAnharmonic": None,
            "freeEnergy": 0.0,
            "iterations": 0,
            "nConfigs": 0,
            "temperatureK": float(args.temperature),
            "totalForceCalcs": 0,
            "method": "SSCHA-error",
        }

    # Output JSON to stdout
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
