#!/usr/bin/env python3
"""
CSC Callback Implementations — QE SCF and Wannier90 Re-projection.

These callbacks are invoked by charge_selfconsistency.run_csc_loop() at
each outer iteration to update the DFT charge density and re-extract H(k).

Two execution modes:
  1. Local: QE + Wannier90 binaries available on the same machine
     (e.g., running CSC on the DFT worker VM)
  2. Remote: call back to the DFT worker's HTTP API
     (e.g., running CSC on the gnn-training VM via the DMFT service)

For production, mode 1 is preferred — the DFT worker has QE installed
and the CSC loop is triggered after a DFT run completes, so all the
SCF outputs (.save directory, pseudopotentials) are already present.
"""

import numpy as np
import subprocess
import json
import os
import time
from typing import Optional, Dict


def make_qe_callback(
    base_scf_input: str,
    pseudo_dir: str,
    qe_pw_binary: str = "/usr/local/bin/pw.x",
    timeout_seconds: int = 3600,
    mpi_ranks: int = 4,
) -> callable:
    """
    Create a QE SCF callback for the CSC loop.

    The callback:
      1. Writes a modified SCF input (startingpot='file' for iter>0)
      2. Copies the previous .save directory for density restart
      3. Runs pw.x
      4. Parses the SCF output for energy, forces, Fermi energy

    Args:
        base_scf_input: the original QE SCF input string (from the DFT run)
        pseudo_dir: pseudopotential directory
        qe_pw_binary: path to pw.x
        timeout_seconds: SCF timeout
        mpi_ranks: MPI parallelism for pw.x

    Returns:
        A callable(work_dir: str) -> dict with SCF results
    """
    iteration_counter = [0]

    def qe_callback(work_dir: str) -> dict:
        iteration = iteration_counter[0]
        iteration_counter[0] += 1

        os.makedirs(work_dir, exist_ok=True)
        scf_dir = os.path.join(work_dir, "scf")
        os.makedirs(scf_dir, exist_ok=True)

        # Modify SCF input for density restart
        from charge_selfconsistency import generate_qe_scf_with_dmft_density
        density_correction = os.path.join(work_dir, "..", f"csc_iter_{iteration-1:03d}",
                                          "density_correction.json") if iteration > 0 else ""
        scf_input = generate_qe_scf_with_dmft_density(
            base_scf_input, density_correction, iteration,
        )

        # Write input
        input_path = os.path.join(scf_dir, "scf.in")
        with open(input_path, "w") as f:
            f.write(scf_input)

        # Copy .save from previous iteration for density restart
        if iteration > 0:
            prev_save = os.path.join(work_dir, "..", f"csc_iter_{iteration-1:03d}",
                                     "scf", "tmp")
            curr_tmp = os.path.join(scf_dir, "tmp")
            if os.path.exists(prev_save) and not os.path.exists(curr_tmp):
                try:
                    subprocess.run(["cp", "-r", prev_save, curr_tmp],
                                   timeout=120, check=False)
                except Exception:
                    pass

        # Run pw.x
        t0 = time.time()
        cmd = [qe_pw_binary]
        if mpi_ranks > 1:
            cmd = ["mpirun", "-np", str(mpi_ranks), "--oversubscribe",
                   "--allow-run-as-root"] + cmd

        try:
            result = subprocess.run(
                cmd,
                input=scf_input,
                capture_output=True,
                text=True,
                cwd=scf_dir,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"converged": False, "error": "SCF timeout"}

        elapsed = time.time() - t0

        if result.returncode != 0:
            return {
                "converged": False,
                "error": f"pw.x exit {result.returncode}",
                "stderr": result.stderr[-500:],
                "elapsed_seconds": elapsed,
            }

        # Parse SCF output
        stdout = result.stdout
        converged = "convergence has been achieved" in stdout.lower()
        energy = None
        fermi = None

        import re
        e_match = re.search(r"!\s*total energy\s*=\s*([-\d.]+)\s*Ry", stdout)
        if e_match:
            energy = float(e_match.group(1))

        f_match = re.search(r"the Fermi energy is\s+([-\d.]+)\s*ev", stdout, re.IGNORECASE)
        if f_match:
            fermi = float(f_match.group(1))

        return {
            "converged": converged,
            "energy": energy,
            "fermi_energy": fermi,
            "elapsed_seconds": elapsed,
            "scf_dir": scf_dir,
        }

    return qe_callback


def make_wannier_callback(
    prefix: str,
    win_template: str,
    n_orb: int,
    wannier90_binary: str = "/usr/local/bin/wannier90.x",
    pw2wannier90_binary: str = "/usr/local/bin/pw2wannier90.x",
    timeout_seconds: int = 1800,
) -> callable:
    """
    Create a Wannier90 re-projection callback for the CSC loop.

    The callback:
      1. Runs wannier90.x -pp to generate .nnkp
      2. Runs pw2wannier90.x to compute overlap matrices
      3. Runs wannier90.x full minimization
      4. Parses _hr.dat to get updated H(k)
      5. Returns H(k) as numpy array

    Args:
        prefix: Wannier90 seedname (e.g., "LaCuO4")
        win_template: the .win file content (DMFT projector mode)
        n_orb: number of Wannier orbitals
        wannier90_binary: path to wannier90.x
        pw2wannier90_binary: path to pw2wannier90.x
        timeout_seconds: per-step timeout

    Returns:
        A callable(work_dir: str) -> np.ndarray of H(k) [n_k, n_orb, n_orb]
    """

    def wannier_callback(work_dir: str) -> Optional[np.ndarray]:
        wan_dir = os.path.join(work_dir, "wannier")
        os.makedirs(wan_dir, exist_ok=True)

        # Write .win file
        win_path = os.path.join(wan_dir, f"{prefix}.win")
        with open(win_path, "w") as f:
            f.write(win_template)

        # Copy SCF output (.save) from the QE callback
        scf_save = os.path.join(work_dir, "scf", "tmp")
        wan_tmp = os.path.join(wan_dir, "tmp")
        if os.path.exists(scf_save) and not os.path.exists(wan_tmp):
            try:
                subprocess.run(["cp", "-r", scf_save, wan_tmp],
                               timeout=120, check=False)
            except Exception:
                pass

        # Step 1: wannier90 -pp
        r1 = subprocess.run(
            [wannier90_binary, "-pp", prefix],
            cwd=wan_dir, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
        if r1.returncode != 0:
            print(f"[CSC-Wannier] -pp failed: {r1.stderr[-300:]}")
            return None

        # Step 2: pw2wannier90
        pw2w_input = f"""&INPUTPP
  outdir = './tmp',
  prefix = '{prefix}',
  seedname = '{prefix}',
  spin_component = 'none',
  write_mmn = .true.,
  write_amn = .true.,
  write_unk = .false.,
/
"""
        pw2w_path = os.path.join(wan_dir, f"{prefix}_pw2wan.in")
        with open(pw2w_path, "w") as f:
            f.write(pw2w_input)

        r2 = subprocess.run(
            [pw2wannier90_binary, "-i", pw2w_path],
            cwd=wan_dir, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
        if r2.returncode != 0:
            print(f"[CSC-Wannier] pw2wannier90 failed: {r2.stderr[-300:]}")
            return None

        # Step 3: wannier90 full
        r3 = subprocess.run(
            [wannier90_binary, prefix],
            cwd=wan_dir, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
        if r3.returncode != 0:
            print(f"[CSC-Wannier] full wannier90 failed: {r3.stderr[-300:]}")
            return None

        # Step 4: Parse _hr.dat → H(k)
        hr_path = os.path.join(wan_dir, f"{prefix}_hr.dat")
        if not os.path.exists(hr_path):
            print(f"[CSC-Wannier] _hr.dat not found at {hr_path}")
            return None

        try:
            from dmft_bundle_exporter_utils import parse_hr_and_fourier_transform
            hk = parse_hr_and_fourier_transform(hr_path, win_path, n_orb)
            print(f"[CSC-Wannier] H(k) updated: shape={hk.shape}")
            return hk
        except Exception as e:
            print(f"[CSC-Wannier] H(k) extraction failed: {e}")
            # Fallback: try numpy-based parser
            return _parse_hr_simple(hr_path, win_path, n_orb)

    return wannier_callback


def _parse_hr_simple(hr_path: str, win_path: str, n_orb: int) -> Optional[np.ndarray]:
    """Simple _hr.dat parser + Fourier transform as fallback."""
    try:
        with open(hr_path) as f:
            lines = f.readlines()

        num_wann = int(lines[1].strip())
        n_rpts = int(lines[2].strip())
        deg_lines = (n_rpts + 14) // 15
        degeneracies = []
        for i in range(deg_lines):
            degeneracies.extend(int(x) for x in lines[3 + i].split())

        # Parse H(R) entries
        r_vecs = {}
        data_start = 3 + deg_lines
        for ln in range(data_start, len(lines)):
            parts = lines[ln].split()
            if len(parts) < 7:
                continue
            rx, ry, rz = int(parts[0]), int(parts[1]), int(parts[2])
            i, j = int(parts[3]) - 1, int(parts[4]) - 1
            re_h, im_h = float(parts[5]), float(parts[6])
            key = (rx, ry, rz)
            if key not in r_vecs:
                r_vecs[key] = np.zeros((num_wann, num_wann), dtype=complex)
            r_vecs[key][i, j] = complex(re_h, im_h)

        # Get k-mesh from .win
        kmesh = [8, 8, 8]
        with open(win_path) as f:
            import re
            mp_match = re.search(r"mp_grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)", f.read())
            if mp_match:
                kmesh = [int(mp_match.group(i)) for i in (1, 2, 3)]

        # Fourier transform — iterate R in INSERTION order (matches degeneracy
        # array from Wannier90 _hr.dat, NOT lexicographic sort)
        nk = kmesh[0] * kmesh[1] * kmesh[2]
        hk = np.zeros((nk, num_wann, num_wann), dtype=complex)

        kidx = 0
        for ik1 in range(kmesh[0]):
            for ik2 in range(kmesh[1]):
                for ik3 in range(kmesh[2]):
                    kfrac = np.array([ik1 / kmesh[0], ik2 / kmesh[1], ik3 / kmesh[2]])
                    for ri, (R, H_R) in enumerate(r_vecs.items()):
                        phase = np.exp(2j * np.pi * np.dot(kfrac, R))
                        deg = degeneracies[ri] if ri < len(degeneracies) else 1
                        hk[kidx] += phase * H_R / deg
                    kidx += 1

        return hk

    except Exception as e:
        print(f"[CSC-Wannier] Simple H(k) parser failed: {e}")
        return None
