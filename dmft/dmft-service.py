#!/usr/bin/env python3
"""
DMFT HTTP service — accepts DMFT bundle submissions, runs solid_dmft,
returns results. Runs inside the qae-dmft Docker container.

Endpoints:
  GET  /health          — service + TRIQS installation health
  POST /submit          — submit a DMFT bundle HDF5, returns job_id
  GET  /status/<job_id> — check job status
  GET  /result/<job_id> — fetch completed DMFT results
  GET  /capabilities    — report installed TRIQS components + MPI config

The service queues jobs and runs them sequentially (DMFT is MPI-parallel
and uses all allocated cores for one job at a time).
"""

import os
import sys
import json
import uuid
import time
import threading
import traceback
from pathlib import Path
from collections import OrderedDict

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# Gap symmetry labels supported by the pairing susceptibility classifier
SYMMETRY_LABELS = [
    "s-wave", "s±-wave", "d-x2y2-wave", "d-xy-wave",
    "p-x-wave", "p-y-wave", "g-wave",
]

# ── Configuration ────────────────────────────────────────────────────────────
BUNDLE_DIR = os.environ.get("DMFT_BUNDLE_DIR", "/data/dmft_bundles")
WORK_DIR = os.environ.get("DMFT_WORK_DIR", "/data/dmft_jobs")
RESULTS_DIR = os.environ.get("DMFT_RESULTS_DIR", "/data/dmft_results")

for d in [BUNDLE_DIR, WORK_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Job tracking ─────────────────────────────────────────────────────────────
# Simple in-memory job queue. For production, back with Redis or DB.
jobs: OrderedDict[str, dict] = OrderedDict()
job_queue: list[str] = []
current_job: str | None = None
worker_lock = threading.Lock()


def worker_loop():
    """Background thread that processes DMFT jobs sequentially."""
    global current_job
    while True:
        job_id = None
        with worker_lock:
            if job_queue:
                job_id = job_queue.pop(0)
                current_job = job_id

        if job_id is None:
            time.sleep(5)
            continue

        try:
            job = jobs[job_id]
            job["status"] = "running"
            job["started_at"] = time.time()
            print(f"[DMFT-Service] Starting job {job_id}: {job.get('formula', '?')}")

            # Import and run
            from importlib import import_module
            run_mod = import_module("run-dmft")

            bundle_path = job["bundle_path"]
            job_work_dir = os.path.join(WORK_DIR, job_id)

            skip_vertex = job.get("skip_vertex", False)
            mpi_ranks = int(os.environ.get("DMFT_MPI_RANKS", 2))

            results = run_mod.run_full_pipeline(
                bundle_path=bundle_path,
                work_dir=job_work_dir,
                skip_vertex=skip_vertex,
                mpi_ranks=mpi_ranks,
            )

            job["status"] = "completed" if results.get("converged") else "completed_unconverged"
            job["results"] = results
            job["completed_at"] = time.time()

            # Copy results to results dir
            result_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"[DMFT-Service] Job {job_id} finished: converged={results.get('converged')}")

        except Exception as e:
            job = jobs.get(job_id, {})
            job["status"] = "failed"
            job["error"] = str(e)
            job["traceback"] = traceback.format_exc()
            job["completed_at"] = time.time()
            print(f"[DMFT-Service] Job {job_id} failed: {e}")

        finally:
            with worker_lock:
                current_job = None


# Worker thread started below after benchmark-aware version is defined


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Check TRIQS installation and service health."""
    checks = {}
    try:
        import triqs
        checks["triqs"] = triqs.version.version
    except Exception as e:
        checks["triqs"] = f"ERROR: {e}"

    try:
        import triqs_cthyb
        checks["cthyb"] = "ok"
    except Exception as e:
        checks["cthyb"] = f"ERROR: {e}"

    try:
        import triqs_dft_tools
        checks["dft_tools"] = "ok"
    except Exception as e:
        checks["dft_tools"] = f"ERROR: {e}"

    try:
        import solid_dmft
        checks["solid_dmft"] = "ok"
    except Exception as e:
        checks["solid_dmft"] = f"ERROR: {e}"

    healthy = all("ERROR" not in str(v) for v in checks.values())

    return jsonify({
        "status": "healthy" if healthy else "degraded",
        "components": checks,
        "jobs_pending": len(job_queue),
        "current_job": current_job,
        "total_completed": sum(1 for j in jobs.values() if j["status"].startswith("completed")),
    }), 200 if healthy else 503


@app.route("/capabilities", methods=["GET"])
def capabilities():
    """Report DMFT service capabilities."""
    mpi_ranks = int(os.environ.get("DMFT_MPI_RANKS", 2))
    return jsonify({
        "service": "qae-dmft",
        "version": "3.0.0",
        "mpi_ranks": mpi_ranks,
        "solvers": ["cthyb"],
        "interaction_types": ["density_density", "kanamori", "slater"],
        "double_counting": ["cFLL", "cAMF"],
        "analytic_continuation": ["maxent", "pade"],
        "bundle_format": "qae-dmft-bundle-v1",
        "max_correlated_orbitals": 14,
        "pipeline_phases": [
            "dmft_1p (single-particle self-consistency)",
            "vertex_g2 (two-particle G² measurement via CTHYB measure_G2)",
            "bse (local Bethe-Salpeter equation → Γ_loc)",
            "pairing (momentum-resolved pairing susceptibility → λ_pair, gap symmetry)",
            "dca (cluster DMFT with DCA, N_c=4/8/16)",
            "multiorbital_dca (3-band Emery / 5-band d-shell cluster)",
            "csc (charge self-consistent DFT+DMFT outer loop)",
            "realistic_pairing (CSC + multi-orbital DCA → Tc with gap symmetry)",
        ],
        "gap_symmetry_classification": list(SYMMETRY_LABELS),
        "submit_options": {
            "mode": "'orchestrated' (default, production) | 'single_site' (1P DMFT only) | 'research' (manual phases)",
            "skip_vertex": "bool — skip G²/BSE/pairing in research mode (default: false)",
            "run_realistic_pairing": "bool — run D1-D3 realistic pipeline in research mode (default: false)",
            "run_csc": "bool — include charge self-consistency (default: true)",
            "dca_nc": "int — force DCA cluster size (default: auto-selected)",
            "max_walltime_hours": "float — total budget (default: 72)",
        },
    })


@app.route("/submit", methods=["POST"])
def submit():
    """
    Submit a DMFT bundle for processing.

    Accepts either:
      - multipart/form-data with 'bundle' file field
      - JSON with 'bundle_path' pointing to a file already on disk
    """
    job_id = str(uuid.uuid4())[:12]

    if request.content_type and "multipart" in request.content_type:
        # File upload
        if "bundle" not in request.files:
            return jsonify({"error": "No 'bundle' file in request"}), 400
        f = request.files["bundle"]
        bundle_path = os.path.join(BUNDLE_DIR, f"{job_id}.h5")
        f.save(bundle_path)
    elif request.is_json:
        data = request.get_json()
        bundle_path = data.get("bundle_path")
        if not bundle_path or not os.path.exists(bundle_path):
            return jsonify({"error": f"bundle_path not found: {bundle_path}"}), 400
    else:
        return jsonify({"error": "Send multipart/form-data with 'bundle' file or JSON with 'bundle_path'"}), 400

    # Quick validation: check it's a valid HDF5 with expected groups
    try:
        import h5py
        with h5py.File(bundle_path, "r") as hf:
            required = ["/hamiltonian/hk", "/correlated_subspace/corr_shells", "/interaction/U_values"]
            missing = [k for k in required if k not in hf]
            if missing:
                return jsonify({"error": f"Bundle missing required datasets: {missing}"}), 400
            formula = hf["/structure/formula"][()].decode() if isinstance(hf["/structure/formula"][()], bytes) else str(hf["/structure/formula"][()])
    except Exception as e:
        return jsonify({"error": f"Invalid HDF5 bundle: {e}"}), 400

    # Parse options
    skip_vertex = False
    run_realistic = False
    run_csc = True
    dca_nc = None  # auto-select
    mode = "orchestrated"
    max_walltime_hours = 72.0
    if request.is_json:
        opts = request.get_json()
        mode = opts.get("mode", "orchestrated")
        skip_vertex = opts.get("skip_vertex", False)
        run_realistic = opts.get("run_realistic_pairing", False)
        run_csc = opts.get("run_csc", True)
        dca_nc = opts.get("dca_nc", None)
        max_walltime_hours = opts.get("max_walltime_hours", 72.0)

    job = {
        "job_id": job_id,
        "formula": formula,
        "bundle_path": bundle_path,
        "mode": mode,
        "skip_vertex": skip_vertex,
        "run_realistic": run_realistic,
        "run_csc": run_csc,
        "dca_nc": dca_nc,
        "max_walltime_hours": max_walltime_hours,
        "status": "queued",
        "submitted_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "results": None,
        "error": None,
    }
    jobs[job_id] = job

    with worker_lock:
        job_queue.append(job_id)

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "position": len(job_queue),
        "formula": formula,
    }), 202


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    """Check job status."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    elapsed = None
    if job["started_at"]:
        end = job["completed_at"] or time.time()
        elapsed = end - job["started_at"]

    return jsonify({
        "job_id": job_id,
        "formula": job["formula"],
        "status": job["status"],
        "elapsed_seconds": elapsed,
        "error": job.get("error"),
    })


@app.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    """Fetch completed DMFT results."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if not job["status"].startswith("completed"):
        return jsonify({"error": f"Job not complete, status={job['status']}"}), 409

    return jsonify({
        "job_id": job_id,
        "formula": job["formula"],
        "results": job["results"],
    })


@app.route("/benchmark", methods=["POST"])
def benchmark():
    """
    Run the 2D Hubbard d-wave pairing benchmark.

    This validates that the DCA solver correctly produces d-wave pairing
    before trusting it on real materials. Accepts JSON body:
      {"quick": true}  — fast mode (~30 min, fewer temperatures)
      {"quick": false}  — production mode (~48h, full T sweep)
    """
    quick = True
    if request.is_json:
        quick = request.get_json().get("quick", True)

    job_id = f"bench_{str(uuid.uuid4())[:8]}"
    bench_dir = os.path.join(WORK_DIR, job_id)

    job = {
        "job_id": job_id,
        "formula": "Hubbard_2D_benchmark",
        "bundle_path": None,
        "skip_vertex": True,
        "status": "queued",
        "submitted_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "results": None,
        "error": None,
        "is_benchmark": True,
        "benchmark_quick": quick,
        "benchmark_dir": bench_dir,
    }
    jobs[job_id] = job

    with worker_lock:
        job_queue.append(job_id)

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "mode": "quick" if quick else "production",
        "position": len(job_queue),
    }), 202


# Patch worker_loop to handle benchmarks
_original_worker_loop = worker_loop

def worker_loop_with_benchmark():
    """Extended worker that handles both DMFT jobs and benchmarks."""
    global current_job
    while True:
        job_id = None
        with worker_lock:
            if job_queue:
                job_id = job_queue.pop(0)
                current_job = job_id

        if job_id is None:
            time.sleep(5)
            continue

        try:
            job = jobs[job_id]
            job["status"] = "running"
            job["started_at"] = time.time()

            if job.get("is_benchmark"):
                print(f"[DMFT-Service] Starting benchmark {job_id}")
                from hubbard_benchmark import run_benchmark
                results = run_benchmark(
                    job["benchmark_dir"],
                    quick=job.get("benchmark_quick", True),
                )
                job["status"] = "completed" if results.get("benchmark_passed") else "completed_failed"
                job["results"] = {
                    k: v for k, v in results.items()
                    if k != "temperature_sweep" or (isinstance(v, list) and len(v) < 50)
                }
            elif job.get("mode") == "orchestrated":
                print(f"[DMFT-Service] Starting orchestrated pipeline {job_id}")
                from pipeline_orchestrator import run_orchestrated_pipeline, ResourceBudget
                bundle_path = job["bundle_path"]
                job_work_dir = os.path.join(WORK_DIR, job_id)
                budget = ResourceBudget(
                    max_walltime_hours=job.get("max_walltime_hours", 72.0),
                    max_memory_gb=float(os.environ.get("DMFT_MAX_MEMORY_GB", 10)),
                    max_cores=int(os.environ.get("DMFT_MPI_RANKS", 2)),
                )
                results = run_orchestrated_pipeline(
                    bundle_path=bundle_path,
                    work_dir=job_work_dir,
                    budget=budget,
                    run_csc=job.get("run_csc", True),
                    force_nc=job.get("dca_nc"),
                )
                has_tc = results.get("tc_K") is not None and results.get("tc_K", 0) > 0
                job["status"] = "completed" if has_tc else "completed_no_tc"
                job["results"] = results
            elif job.get("run_realistic"):
                print(f"[DMFT-Service] Starting realistic pairing pipeline {job_id}")
                from realistic_pairing import run_realistic_pairing_pipeline
                bundle_path = job["bundle_path"]
                job_work_dir = os.path.join(WORK_DIR, job_id)
                results = run_realistic_pairing_pipeline(
                    bundle_path=bundle_path,
                    work_dir=job_work_dir,
                    run_csc=job.get("run_csc", True),
                    nc=job.get("dca_nc", 4),
                )
                has_tc = results.get("tc_K") is not None and results.get("tc_K", 0) > 0
                job["status"] = "completed" if has_tc else "completed_no_tc"
                job["results"] = results
            else:
                from importlib import import_module
                run_mod = import_module("run-dmft")
                bundle_path = job["bundle_path"]
                job_work_dir = os.path.join(WORK_DIR, job_id)
                skip_vertex = job.get("skip_vertex", False)
                mpi_ranks = int(os.environ.get("DMFT_MPI_RANKS", 2))
                results = run_mod.run_full_pipeline(
                    bundle_path=bundle_path,
                    work_dir=job_work_dir,
                    skip_vertex=skip_vertex,
                    mpi_ranks=mpi_ranks,
                )
                job["status"] = "completed" if results.get("converged") else "completed_unconverged"
                job["results"] = results

            job["completed_at"] = time.time()
            result_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
            with open(result_path, "w") as f:
                json.dump(job.get("results", {}), f, indent=2, default=str)
            print(f"[DMFT-Service] Job {job_id} finished")

        except Exception as e:
            job = jobs.get(job_id, {})
            job["status"] = "failed"
            job["error"] = str(e)
            job["traceback"] = traceback.format_exc()
            job["completed_at"] = time.time()
            print(f"[DMFT-Service] Job {job_id} failed: {e}")

        finally:
            with worker_lock:
                current_job = None

# Replace the worker thread
worker_thread = threading.Thread(target=worker_loop_with_benchmark, daemon=True)
worker_thread.start()


if __name__ == "__main__":
    port = int(os.environ.get("DMFT_SERVICE_PORT", 8780))
    app.run(host="0.0.0.0", port=port, debug=False)
