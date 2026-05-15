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
    import cellconstructor.Units
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
# CODATA 2018 — kept in sync with qe-worker.ts and other TS DFT files
BOHR_TO_ANG = 0.529177210903
# CODATA 2018 — Rydberg constant times hc → eV: R_∞ × hc = 13.605693122994 eV
# Previous value 13.605698 was the older (pre-2018) value, off by 5 ppm.
RY_TO_EV = 13.605693122994
RY_TO_MEV = RY_TO_EV * 1000.0
K_BOLTZMANN_EV = 8.617333262e-5  # eV/K
HBAR_EV_S = 6.582119569e-16  # eV·s
MAX_SSCHA_ITERATIONS = 20
MAX_FORCE_CALCS = 2000  # absolute safety cap on total DFT calls
PW_TIMEOUT_S = 3600  # 1 hour per SCF
TOTAL_TIMEOUT_S = 86400  # 24 hours total

# Atomic masses in amu (for SSCHA fallback ZPE amplitude calculation).
ATOMIC_MASSES_AMU = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "Si": 28.086, "P": 30.974, "S": 32.065, "Cl": 35.453, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.380, "Ga": 69.723, "Ge": 72.640,
    "As": 74.922, "Se": 78.960, "Br": 79.904, "Rb": 85.468, "Sr": 87.620, "Y": 88.906,
    "Zr": 91.224, "Nb": 92.906, "Mo": 95.960, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76,
    "Te": 127.60, "I": 126.90, "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12,
    "Pr": 140.91, "Nd": 144.24, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.04, "Lu": 174.97,
    "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22,
    "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
    "Th": 232.04, "Pa": 231.04, "U": 238.03,
}

# Quantum harmonic oscillator amplitude prefactor for σ²[Å²] = K_QHO·(2n+1)/(m_amu·ω_cm).
# Derivation: σ²[m²] = ℏ·(2n+1)/(2·m·ω) with m in kg, ω = 2πc·ω_cm in rad/s.
#   ℏ = 1.0546e-34 J·s
#   1 amu = 1.6605e-27 kg
#   2πc[m/s] = 2π × 2.998e8 m/s = 1.883e9
#   ω[rad/s] = 2πc[m/s] × ω_cm[1/m] = 1.883e9 × 100 × ω_cm[1/cm] = 1.883e11 × ω_cm
#   σ²[m²] = 1.0546e-34·(2n+1)/(2·m_amu·1.6605e-27·1.883e11·ω_cm)
#          = (2n+1)·1.687e-19 / (m_amu·ω_cm) [m²]
#          = (2n+1)·16.87 / (m_amu·ω_cm) [Å²]  (×10^20)
# Sanity check: H @ 3000 cm⁻¹, T=0 → σ² = 16.87/(1.008·3000) = 5.58e-3 Å² → σ ≈ 0.075 Å ✓
K_QHO_AMPLITUDE_AA2_CM = 16.87


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
    p.add_argument("--lambda-harmonic", type=float, default=None,
                   help="Harmonic electron-phonon coupling λ from EPW (used as "
                        "fallback for anharmonic Tc estimate when SSCHA doesn't "
                        "re-run EPW). Without it, Tc uses λ=1.0 as a placeholder.")
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


def omega_log_inv_weighted(freqs_cm: np.ndarray) -> float:
    """
    Allen-Dynes omega_log estimated from phonon frequencies alone, for the case
    where per-mode lambda is not available (SSCHA anharmonic frequencies before
    an EPW re-run). The Allen-Dynes omega_log is the alpha2F/omega-weighted log
    average; under the DOS-only approximation alpha2F(w) ~ F(w) this reduces to
    the 1/omega-weighted log average:
        omega_log = exp[ sum_i (1/w_i) ln(w_i) / sum_i (1/w_i) ].
    An unweighted geometric mean exp(mean(ln(w))) overweights stiff optical
    modes and underweights the low-frequency acoustic modes that physically
    dominate omega_log, systematically overestimating it -> overestimating the
    Allen-Dynes Tc. Returns omega_log in cm^-1, or 0.0 if no positive modes.
    """
    w = freqs_cm[freqs_cm > 1.0]
    if w.size == 0:
        return 0.0
    inv = 1.0 / w
    return float(np.exp(np.sum(inv * np.log(w)) / np.sum(inv)))


def allen_dynes_tc(lambda_ep: float, omega_log_meV: float, mu_star: float = 0.10) -> float:
    """Allen-Dynes Tc formula with strong-coupling f1 correction. Returns Tc in K."""
    if lambda_ep < 0.01 or omega_log_meV < 0.01:
        return 0.0
    # k_B = 0.08617 meV/K, so omega_log[meV] / k_B[meV/K] = omega_log[K].
    # (The earlier line `omega_log_K = omega_log_meV / K_BOLTZMANN_EV` was
    # both wrong by a factor of 1000 — K_BOLTZMANN_EV is in eV/K, not meV/K —
    # AND dead code, overwritten on the next line. Now consolidated.)
    omega_log_K = omega_log_meV / 0.08617
    denom = lambda_ep - mu_star * (1 + 0.62 * lambda_ep)
    if denom <= 0:
        return 0.0
    exponent = -1.04 * (1 + lambda_ep) / denom
    if exponent > 0 or exponent < -100:
        return 0.0
    # Allen-Dynes f1 strong-coupling correction (Allen & Dynes, PRB 12, 905
    # (1975), eq 3.3). SSCHA candidates are strong-coupling hydrides where
    # lambda >~ 1.5 — without f1 the McMillan-only formula under-predicts Tc
    # by 10-15%. f2 (spectral-moment correction) requires <omega^2> which
    # is not available at this call site; f2 -> 1 is the safe fallback.
    lambda_bar = 2.46 * (1 + 3.8 * mu_star)
    f1 = (1 + (lambda_ep / lambda_bar) ** 1.5) ** (1 / 3)
    return (omega_log_K / 1.2) * f1 * np.exp(exponent)


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
                # Find pseudopotential file. The element symbol must be
                # exactly matched as the prefix BEFORE any extension/suffix:
                # for element "C" we must NOT pick "Ca.UPF" / "Cu.UPF" / etc.
                # Match patterns: "C.UPF", "C.pbe-...-kjpaw_psl.1.0.0.UPF",
                # "C_PBE_OPTHARDER.UPF" — i.e., element followed by "." or "_".
                pp_files = [
                    f for f in os.listdir(args.pseudo_dir)
                    if f.endswith(".UPF")
                    and (f == f"{el}.UPF" or f.startswith(f"{el}.") or f.startswith(f"{el}_"))
                ]
                # Prefer the exact-match "{el}.UPF" if multiple variants exist
                pp_files.sort(key=lambda f: 0 if f == f"{el}.UPF" else 1)
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
            # Capture the CONVERGED dynamical matrix. Without this the
            # converged-break skipped the `supercell = minim.dyn` update
            # below, so the final anharmonic frequencies / omega_log were
            # extracted from the second-to-last (pre-convergence) dyn —
            # i.e. the converged result was silently one iteration stale.
            supercell = minim.dyn
            log(f"  Converged at iteration {iteration}!")
            break

        # Update dynamical matrix for next iteration
        supercell = minim.dyn

    # Extract final results
    log("Extracting anharmonic properties...")

    # Get anharmonic phonon frequencies. `supercell` now always holds the
    # latest minimizer dyn — the converged one when converged (captured in
    # the is_converged() branch above), or the last iteration's otherwise.
    final_dyn = supercell
    # cellconstructor's DyagDinQ returns Gamma frequencies in RYDBERG atomic
    # units (per its docstring), NOT cm^-1. Every consumer below — the >1.0
    # positive-mode filter, the <-5.0 imaginary count, omega_log_inv_weighted,
    # and the *0.12398 cm^-1->meV conversion — assumes cm^-1. Without this
    # rescaling w_anh stays ~0.009 Ry for a real ~1000 cm^-1 phonon, so
    # positive_w is always empty and omegaLogAnharmonic is silently always 0.
    w_anh, _ = final_dyn.DyagDinQ(0)
    w_anh = np.asarray(w_anh) * CC.Units.RY_TO_CM

    # Compute omega_log (anharmonic)
    # For a proper calculation we'd need the full Brillouin zone,
    # but use Gamma-point modes as approximation
    positive_w = w_anh[w_anh > 1.0]

    if len(positive_w) > 0:
        # Allen-Dynes omega_log: 1/omega-weighted log average (DOS-only
        # approximation), NOT an unweighted geometric mean.
        omega_log_cm = omega_log_inv_weighted(positive_w)
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

    # Build species list from pseudopotential directory. Two bugs in the
    # previous version:
    #   1. `f.lower().startswith(el.lower())` matched "Ca.UPF" / "Cl.UPF" /
    #      "Co.UPF" etc. when looking for "C.UPF" — wrong PP assigned to
    #      Carbon atoms based on filesystem listing order.
    #   2. Mass hardcoded to 1.0 amu — pw.x's ATOMIC_SPECIES card got
    #      wrong masses, which affects phonon eigenvector mass-weighting
    #      if ph.x is run on this SCF.
    species_set = {}
    for el in atom_labels:
        if el not in species_set:
            # Exact element match: filename must start with "{el}." or "{el}_"
            # so "C" doesn't match "Ca.UPF". Case-sensitive — QE elements
            # are capitalized.
            pp_files = [
                f for f in os.listdir(args.pseudo_dir)
                if f.endswith(".UPF")
                and (f == f"{el}.UPF" or f.startswith(f"{el}.") or f.startswith(f"{el}_"))
            ]
            pp_files.sort(key=lambda f: 0 if f == f"{el}.UPF" else 1)
            pp = pp_files[0] if pp_files else f"{el}.UPF"
            mass = ATOMIC_MASSES_AMU.get(el, 50.0)
            species_set[el] = (el, mass, pp)
    species_list = list(species_set.values())

    # Generate displaced configurations
    n_modes = 3 * nat
    n_configs = min(args.nconfigs, max(2 * n_modes, 20))
    T = args.temperature

    # Per-atom mass in amu (used to scale ZPE amplitude per atom — light atoms
    # like H vibrate with larger amplitude than heavy atoms like U).
    atom_masses_amu = np.array(
        [ATOMIC_MASSES_AMU.get(el, 50.0) for el in atom_labels],
        dtype=float,
    )
    # Effective "reduced" mass for a mode where we don't have an eigenvector:
    # use the harmonic mean of atom masses (closer to the lightest atom, which
    # dominates the optical-mode amplitude).
    inv_mean_mass = float(np.mean(1.0 / np.maximum(atom_masses_amu, 0.1)))
    reduced_mass_amu = 1.0 / inv_mean_mass

    # Quantum-harmonic-oscillator displacement amplitude per mode (Å):
    #   σ_mode²[Å²] = K_QHO · (2 n_bose + 1) / (m_eff · ω[cm⁻¹])
    # The previous formula `sqrt((2n+1)·ℏω_eV / (2·ω·0.124e-3))` algebraically
    # reduced to `sqrt((2n+1)/2)` — a constant ~0.7 with no mass or frequency
    # dependence — then was clamped to [0.005, 0.15] Å. That gave the SAME
    # amplitude to every mode regardless of m or ω, completely wrong for
    # H-rich hydrides where the light-H modes need much larger amplitudes
    # than the metal-sublattice modes.
    displacements = []
    rng = np.random.default_rng(seed=42)

    for ic in range(n_configs):
        disp = np.zeros((nat, 3))
        for mode_idx in range(n_modes):
            freq_cm = harmonic_freqs_cm[mode_idx] if mode_idx < len(harmonic_freqs_cm) else 100.0
            abs_freq = abs(freq_cm)
            if abs_freq < 5.0:
                continue  # skip near-zero (acoustic) modes

            # Bose occupation at temperature T
            hbar_omega_eV = abs_freq * 0.12398e-3  # cm^-1 to eV
            if T > 0 and hbar_omega_eV > 0:
                x = hbar_omega_eV / (K_BOLTZMANN_EV * T)
                n_bose = 1.0 / (np.exp(x) - 1.0) if x < 50 else 0.0
            else:
                n_bose = 0.0

            # Mode amplitude using reduced mass (mass-aware fallback when
            # eigenvector is unavailable). σ in Å.
            sigma_mode_sq = K_QHO_AMPLITUDE_AA2_CM * (2 * n_bose + 1) / (reduced_mass_amu * abs_freq)
            sigma_mode = float(np.sqrt(max(sigma_mode_sq, 0.0)))
            # Sanity bounds: prevent pathological values for soft modes (<5 cm⁻¹
            # already skipped) and for huge mass underestimates.
            sigma_mode = min(sigma_mode, 0.20)  # 0.2 Å absolute ceiling per mode
            sigma_mode = max(sigma_mode, 0.001)

            amplitude = rng.normal(0, sigma_mode)
            if eigenvectors is not None and mode_idx < eigenvectors.shape[0]:
                evec = eigenvectors[mode_idx].reshape(nat, 3)
                disp += amplitude * evec
            else:
                # Random unit-direction per atom. Per-atom amplitude must scale
                # as 1/sqrt(m_atom) to recover the quantum amplitude in real
                # space (the eigenvector branch handles this implicitly via
                # mass-weighted normalization).
                direction = rng.normal(0, 1, (nat, 3))
                direction /= (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12)
                per_atom_scale = np.sqrt(reduced_mass_amu / atom_masses_amu)[:, None]
                disp += (amplitude / np.sqrt(nat)) * direction * per_atom_scale

        displacements.append(disp)

    log(f"  Generated {n_configs} displaced configurations")

    # Run DFT force calculations
    total_force_calcs = 0
    forces_list = []
    energies_list = []
    used_displacements = []
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
            used_displacements.append(displacements[ic])
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
    U_mat = np.array([d.flatten() for d in used_displacements])  # (nconf, 3N) — paired with successful configs only
    F_mat = np.array([f.flatten() for f in forces_list])  # (nconf, 3N)

    # Convert forces from Ry/Bohr to eV/Angstrom for consistent units
    F_mat_eV_A = F_mat * (RY_TO_EV / BOHR_TO_ANG)

    # Least-squares fit: Phi = -(U^T U)^{-1} U^T F
    try:
        # Use pseudo-inverse for numerical stability
        Phi = -np.linalg.lstsq(U_mat, F_mat_eV_A, rcond=None)[0].T  # (3N, 3N)

        # Symmetrize force constant matrix
        Phi = 0.5 * (Phi + Phi.T)

        # Mass-weight the dynamical matrix: D_ij = Phi_ij / sqrt(m_i · m_j).
        # Without this, the diagonalization assumes m=1 amu for every atom —
        # heavy elements end up with ω too high by sqrt(m_amu): Li by 2.6×,
        # Fe by 7.5×, U by 15.4×. Critical for any non-H-only system.
        masses_per_atom = np.array(
            [ATOMIC_MASSES_AMU.get(el, 50.0) for el in atom_labels],
            dtype=float,
        )
        # Expand to 3N components (each atom has x, y, z)
        masses_3N = np.repeat(masses_per_atom, 3)
        sqrt_m = np.sqrt(masses_3N)
        # D = Phi / (sqrt_m_i * sqrt_m_j) via outer-product division
        D = Phi / np.outer(sqrt_m, sqrt_m)

        eigenvalues, eigvecs = np.linalg.eigh(D)

        # Convert eigenvalues to frequencies (cm^-1):
        #   D has units eV/(Å²·amu) ⇒ sqrt(D) is angular frequency in units
        #   where the conversion factor 15.633 maps to THz when m is in amu
        #   and K in eV/Å². Then 33.356 cm⁻¹/THz to reach the standard unit.
        # See: https://en.wikipedia.org/wiki/Reciprocal_centimetre#Spectroscopy
        anh_freqs_cm = []
        for ev in eigenvalues:
            if ev > 0:
                freq_THz = np.sqrt(ev) * 15.633   # eV/(Å²·amu) → THz
                freq_cm = freq_THz * 33.356        # THz → cm⁻¹
                anh_freqs_cm.append(freq_cm)
            else:
                # Imaginary mode — record as negative frequency by convention
                freq_THz = np.sqrt(abs(ev)) * 15.633
                freq_cm = -freq_THz * 33.356
                anh_freqs_cm.append(freq_cm)

        anh_freqs_cm = np.array(sorted(anh_freqs_cm))

        # Compute anharmonic omega_log — Allen-Dynes 1/omega-weighted log
        # average (DOS-only approximation), NOT an unweighted geometric mean.
        positive_mask = anh_freqs_cm > 5.0
        if np.any(positive_mask):
            pos_freqs = anh_freqs_cm[positive_mask]
            omega_log_cm = omega_log_inv_weighted(pos_freqs)
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

        # Find the line with ntyp, nat, ibrav, celldm(1..6). A QE .dyn file
        # writes these as bare numbers (no "celldm(1) =" syntax), so celldm(1)
        # — the alat in Bohr — is the 4th token here. The .dyn cell vectors and
        # atomic positions are in units of this alat; without it they default
        # to alat=1 Bohr, collapsing the cell to ~0.5 Å and failing every SCF.
        alat_bohr = 1.0
        for i, line in enumerate(lines):
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([-\d.eE+]+)", line)
            if m and i < 10:
                ntyp = int(m.group(1))
                nat = int(m.group(2))
                try:
                    alat_bohr = float(m.group(4))
                except ValueError:
                    alat_bohr = 1.0
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

        # Convert cell from alat units to Angstrom. alat_bohr is taken from the
        # .dyn header (celldm(1)) above; the explicit-syntax search below is a
        # harmless override for the rare case a dyn file embeds it that way.
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

        # Parse frequencies. QE writes dyn-file frequencies in dual-unit form:
        #   freq (    1) =     -2.345 [THz] =    -78.12 [cm-1]
        #   omega( 1) =     -2.345 [THz] =    -78.12 [cm-1]
        #   omega(1-3) = ...    (QE 7.x range form for degenerate modes)
        # The cm⁻¹ value is the SECOND number on the line — we must skip past
        # the THz value first. Without this, the regex captured the THz value
        # and silently stored frequencies that were 33× too small. The
        # `(?:\s*-\s*\d+)?` clause handles QE 7.x range form `omega(1-3)`
        # emitted for degenerate modes in cubic/high-symmetry crystals.
        freqs = []
        for line in lines:
            # Dual-unit form (standard since QE 6.x)
            m = re.search(
                r"(?:freq|omega)\s*\(\s*\d+(?:\s*-\s*\d+)?\s*\)\s*=\s*[-\d.eE+]+\s*\[THz\]\s*=\s*([-\d.eE+]+)\s*\[\s*cm[-^]?-?1\s*\]",
                line,
            )
            if m:
                try:
                    freqs.append(float(m.group(1)))
                except ValueError:
                    pass
                continue
            # Fallback: cm⁻¹-only form (rare; some older QE versions)
            m_simple = re.search(
                r"(?:freq|omega)\s*\(\s*\d+(?:\s*-\s*\d+)?\s*\)\s*=\s*([-\d.eE+]+)\s*\[\s*cm[-^]?-?1\s*\]",
                line,
            )
            if m_simple:
                try:
                    freqs.append(float(m_simple.group(1)))
                except ValueError:
                    pass

        if not freqs:
            # A dyn file written by ph.x always contains the diagonalized
            # frequency block. An empty parse means the file is truncated or
            # corrupted. The old behavior fabricated placeholder frequencies
            # (100 cm^-1 per mode), which silently produced a SSCHA result
            # with a plausible-looking omega_log and Tc that is pure fiction.
            # Signal a genuine parse failure instead — the caller already
            # handles a None return with a proper "Failed to parse dynamical
            # matrix" error result.
            log(f"  No frequencies parsed from {filename} — treating as parse failure")
            return None

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

        # Compute Tc correction if we have omega_log.
        # Use the harmonic λ from EPW as the fallback for the anharmonic-Tc
        # estimate. The proper value is λ_anh from a re-run EPW with the
        # anharmonic phonons; absent that, harmonic λ is a far better proxy
        # than the old hardcoded 1.0 (which underestimated Tc for strong-
        # coupling hydrides by ~2-3×).
        omega_log = result.get("omegaLogAnharmonic", 0.0)
        if omega_log and omega_log > 0:
            lambda_for_tc = args.lambda_harmonic if args.lambda_harmonic is not None else 1.0
            if args.lambda_harmonic is None:
                log("[SSCHA] WARNING: --lambda-harmonic not provided; using λ=1.0 "
                    "placeholder for Tc estimate. For strong-coupling systems "
                    "(hydrides) this can under-predict Tc by 2-3×. Pass the EPW "
                    "harmonic λ via --lambda-harmonic for a better estimate.")
            for mu_star in [0.10, 0.13, 0.15]:
                tc = allen_dynes_tc(lambda_for_tc, omega_log, mu_star)
                result[f"tcEstimate_mustar{mu_star:.2f}"] = round(tc, 2)
            result["lambdaUsedForTcEstimate"] = lambda_for_tc
            result["lambdaUsedSource"] = "harmonic-from-EPW" if args.lambda_harmonic is not None else "placeholder-1.0"

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
