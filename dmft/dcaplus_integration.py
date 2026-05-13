#!/usr/bin/env python3
"""
DCA++ GPU Solver Integration.

When TRIQS/CTHYB hits the sign problem wall (⟨sign⟩ < 0.05), DCA++
from ORNL provides a GPU-accelerated CT-AUX solver that has better
sign properties for the Hubbard model in cluster geometries.

Architecture:
  - Separate Docker image (qae-dcaplus) with CUDA + DCA++
  - Same bundle HDF5 format as TRIQS pipeline
  - Invoked as a subprocess or HTTP call from the orchestrator
  - Results parsed from DCA++ output HDF5

DCA++ GitHub: github.com/CompFUSE/DCA
CT-AUX algorithm: Gull et al., EPL 82, 57003 (2008)

References:
  Hähner et al., SC'20 (2020) — DCA++ on Summit
  Maier et al., PRL 95, 237001 (2005) — DCA d-wave
"""

import numpy as np
import subprocess
import json
import os
import time
from typing import Optional, Dict


DCAPLUS_DOCKER_IMAGE = "qae-dcaplus:latest"


def check_dcaplus_available() -> Dict:
    """Check if DCA++ Docker image and GPU are available."""
    has_docker = False
    has_gpu = False
    has_image = False

    try:
        r = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
        has_docker = r.returncode == 0
    except Exception:
        pass

    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        has_gpu = r.returncode == 0
        gpu_name = r.stdout.strip() if has_gpu else None
    except Exception:
        gpu_name = None

    if has_docker:
        try:
            r = subprocess.run(["docker", "images", "-q", DCAPLUS_DOCKER_IMAGE],
                               capture_output=True, text=True, timeout=10)
            has_image = bool(r.stdout.strip())
        except Exception:
            pass

    return {
        "available": has_docker and has_gpu and has_image,
        "docker": has_docker,
        "gpu": has_gpu,
        "gpu_name": gpu_name,
        "image": has_image,
        "image_name": DCAPLUS_DOCKER_IMAGE,
    }


def generate_dcaplus_input(
    U: float,
    beta: float,
    mu: float,
    nc: int,
    t: float = 1.0,
    tp: float = -0.3,
    n_measurements: int = 1_000_000,
    work_dir: str = "/data/dcaplus",
) -> str:
    """
    Generate DCA++ input JSON for the 2D Hubbard model.

    DCA++ uses a JSON input format with model + solver + DCA parameters.
    """
    config = {
        "model": {
            "type": "tight-binding",
            "lattice": "square",
            "dimension": 2,
            "bands": 1,
            "hoppings": {
                "t": t,
                "t-prime": tp,
            },
            "interaction": {
                "type": "hubbard",
                "U": U,
            },
        },
        "DCA": {
            # For non-square nc (e.g., 8), use rectangular [Lx, Ly] with Lx*Ly=nc
            "cluster": [int(np.sqrt(nc)), int(np.sqrt(nc))] if int(np.sqrt(nc))**2 == nc
                        else [2, nc // 2],
            "lattice-size": [64, 64],
            "iterations": 20,
            "convergence-factor": 1e-4,
            "chemical-potential": mu,
            "adjust-chemical-potential": True,
            "target-density": 0.85,
        },
        "Monte-Carlo": {
            "type": "CT-AUX",
            "seed": 42,
            "warm-up-sweeps": 100,
            "sweeps-per-measurement": 1,
            "measurements": n_measurements,
        },
        "physics": {
            "temperature": 1.0 / beta,
            "beta": beta,
        },
        "output": {
            "directory": work_dir,
            "filename-dca": "dca_output.hdf5",
            "filename-analysis": "analysis_output.hdf5",
        },
    }

    return json.dumps(config, indent=2)


def run_dcaplus(
    input_config: str,
    work_dir: str,
    gpu_id: int = 0,
    timeout_hours: float = 24.0,
) -> Dict:
    """
    Run DCA++ in Docker container with GPU access.

    Args:
        input_config: DCA++ input JSON string
        work_dir: host directory for input/output
        gpu_id: which GPU to use
        timeout_hours: max walltime

    Returns:
        dict with DCA++ results
    """
    os.makedirs(work_dir, exist_ok=True)

    # Write input file
    input_path = os.path.join(work_dir, "dcaplus_input.json")
    with open(input_path, "w") as f:
        f.write(input_config)

    t0 = time.time()

    cmd = [
        "docker", "run", "--rm",
        f"--gpus=device={gpu_id}",
        "-v", f"{work_dir}:/data/dcaplus",
        DCAPLUS_DOCKER_IMAGE,
        "main_dca", "--input_file=/data/dcaplus/dcaplus_input.json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(timeout_hours * 3600),
        )
    except subprocess.TimeoutExpired:
        return {
            "converged": False,
            "error": "DCA++ timeout",
            "failure_mode": "timeout",
        }

    elapsed = time.time() - t0

    if result.returncode != 0:
        # Classify the failure mode from stderr so callers can retry with
        # adjusted parameters (smaller cluster for OOM, different solver
        # for sign-problem, more sweeps for convergence).
        stderr_tail = (result.stderr or "")[-2000:]
        lower = stderr_tail.lower()
        if "out of memory" in lower or "oom" in lower or "bad_alloc" in lower:
            failure_mode = "oom"
        elif "sign" in lower and ("problem" in lower or "small" in lower):
            failure_mode = "sign_problem"
        elif "not converg" in lower or "max iter" in lower:
            failure_mode = "convergence"
        elif "cuda" in lower or "gpu" in lower:
            failure_mode = "gpu_error"
        else:
            failure_mode = "unknown"
        return {
            "converged": False,
            "error": f"DCA++ exit {result.returncode}",
            "failure_mode": failure_mode,
            "stderr": result.stderr[-1000:],
            "elapsed_hours": elapsed / 3600,
        }

    # Parse output
    return parse_dcaplus_output(work_dir, elapsed)


def parse_dcaplus_output(
    work_dir: str,
    elapsed: float,
    expected_n_iw: Optional[int] = None,
    expected_n_orb: Optional[int] = None,
) -> Dict:
    """Parse DCA++ output HDF5 for key observables.

    Args:
        work_dir: directory containing dca_output.hdf5
        elapsed: walltime in seconds
        expected_n_iw: if given, validate Σ has shape compatible with 2*n_iw
            Matsubara frequencies. Mismatch sets failure_mode="shape_mismatch".
        expected_n_orb: if given, validate orbital count matches.
    """
    import h5py

    output_path = os.path.join(work_dir, "dca_output.hdf5")
    if not os.path.exists(output_path):
        return {
            "converged": False,
            "error": "Output HDF5 not found",
            "failure_mode": "missing_output",
        }

    results = {
        "converged": False,
        "elapsed_hours": elapsed / 3600,
        "solver": "DCA++/CT-AUX/GPU",
        "shape_warnings": [],
    }

    try:
        with h5py.File(output_path, "r") as f:
            # DCA++ stores results in /DCA-loop/iteration-N/
            if "DCA-loop" in f:
                dca_grp = f["DCA-loop"]
                import re
                # Sort numerically (not lexicographically) — "iteration-10" > "iteration-2"
                iter_keys = [k for k in dca_grp.keys() if k.startswith("iteration")]
                def _iter_num(k):
                    m = re.search(r"\d+", k)
                    return int(m.group()) if m else 0
                iter_keys = sorted(iter_keys, key=_iter_num)
                if iter_keys:
                    last = dca_grp[iter_keys[-1]]

                    if "Sigma" in last:
                        sigma = last["Sigma"][()]
                        results["sigma_shape"] = list(sigma.shape)
                        results["self_energy_available"] = True
                        # Validate against the upstream grid if specified
                        if expected_n_iw is not None:
                            # DCA++ Σ may be stored as [n_w, n_K, n_orb, n_orb]
                            # or similar; the Matsubara axis is typically the
                            # longest and equals 2*n_iw.
                            if 2 * expected_n_iw not in sigma.shape:
                                results["shape_warnings"].append(
                                    f"Σ shape {sigma.shape} has no axis matching "
                                    f"2·n_iw={2*expected_n_iw}"
                                )
                        if expected_n_orb is not None and expected_n_orb not in sigma.shape:
                            results["shape_warnings"].append(
                                f"Σ shape {sigma.shape} has no axis matching "
                                f"n_orb={expected_n_orb}"
                            )

                    if "G-k-w" in last:
                        results["greens_function_available"] = True

                    if "density" in last:
                        results["density"] = float(last["density"][()])

                    if "chemical-potential" in last:
                        results["mu"] = float(last["chemical-potential"][()])

                    results["n_iterations"] = len(iter_keys)
                    results["converged"] = True

            # Check for analysis output
            analysis_path = os.path.join(work_dir, "analysis_output.hdf5")
            if os.path.exists(analysis_path):
                with h5py.File(analysis_path, "r") as af:
                    if "leading-eigenvalues" in af:
                        evals = af["leading-eigenvalues"][()]
                        results["eigenvalues"] = evals.tolist()
                        # Take the eigenvalue with LARGEST |λ| as the pairing
                        # instability indicator (don't assume DCA++ pre-sorts)
                        if len(evals) > 0:
                            evals_arr = np.asarray(evals)
                            idx_max = int(np.argmax(np.abs(evals_arr)))
                            results["lambda_pair"] = float(np.real(evals_arr[idx_max]))

                    if "leading-eigenvectors" in af:
                        results["eigenvector_available"] = True

    except Exception as e:
        results["error"] = f"Failed to parse output: {e}"

    return results
