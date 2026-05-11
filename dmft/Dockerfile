# ==============================================================================
# TRIQS 3.3.x + solid_dmft — DMFT solver for Quantum Alchemy Engine
#
# Base: official TRIQS Docker image with CTHYB, DFTTools, maxent pre-installed.
# Adds: solid_dmft, h5py, numpy/scipy matching TRIQS ABI, Wannier90 converter
#       utilities, and the QAE DMFT bridge scripts.
#
# Build:
#   docker build -t qae-dmft:latest -f dmft/Dockerfile .
#
# Run (on gnn-training VM):
#   docker compose -f dmft/docker-compose.yml up -d
# ==============================================================================

FROM triqs/triqs:3.3 AS base

LABEL maintainer="QAE <damsgardhunter@gmail.com>"
LABEL description="TRIQS 3.3 + solid_dmft for QAE DMFT pipeline"

# ── System deps for MPI + HDF5 ──────────────────────────────────────────────
USER root
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      openssh-client \
      libhdf5-mpi-dev \
      wget \
      jq \
    && rm -rf /var/lib/apt/lists/*

# ── Install solid_dmft from source (latest stable) ──────────────────────────
# solid_dmft reads Wannier90 + QE outputs and drives the CTHYB impurity solver
RUN cd /opt && \
    git clone --depth 1 --branch stable https://github.com/flatironinstitute/solid_dmft.git && \
    cd solid_dmft && \
    pip install --no-cache-dir . && \
    cd / && rm -rf /opt/solid_dmft/.git

# ── Additional Python packages for the QAE bridge ──────────────────────────
RUN pip install --no-cache-dir \
    flask==3.1.* \
    gunicorn==23.* \
    watchdog==6.*

# ── Create working directories ──────────────────────────────────────────────
RUN mkdir -p /data/dmft_jobs /data/dmft_results /data/dmft_bundles /app

# ── Additional Python packages for vertex/BSE pipeline ──────────────────────
RUN pip install --no-cache-dir \
    scipy==1.14.* \
    toml==0.10.*

# ── Copy QAE DMFT bridge service ────────────────────────────────────────────
COPY dmft/convert-bundle.py /app/convert-bundle.py
COPY dmft/dmft-service.py /app/dmft-service.py
COPY dmft/run-dmft.py /app/run-dmft.py
COPY dmft/triqs-healthcheck.py /app/triqs-healthcheck.py
COPY dmft/vertex_measurement.py /app/vertex_measurement.py
COPY dmft/bse_solver.py /app/bse_solver.py
COPY dmft/pairing_susceptibility.py /app/pairing_susceptibility.py
COPY dmft/dca_solver.py /app/dca_solver.py
COPY dmft/hubbard_benchmark.py /app/hubbard_benchmark.py
COPY dmft/multiorbital_dca.py /app/multiorbital_dca.py
COPY dmft/charge_selfconsistency.py /app/charge_selfconsistency.py
COPY dmft/realistic_pairing.py /app/realistic_pairing.py
COPY dmft/pipeline_orchestrator.py /app/pipeline_orchestrator.py
COPY dmft/csc_callbacks.py /app/csc_callbacks.py
COPY dmft/analytic_continuation.py /app/analytic_continuation.py
COPY dmft/dmft_tc_correction.py /app/dmft_tc_correction.py
COPY dmft/dca_periodization.py /app/dca_periodization.py
COPY dmft/cluster_vertex.py /app/cluster_vertex.py
COPY dmft/dcaplus_integration.py /app/dcaplus_integration.py
COPY dmft/adaptive_temperature.py /app/adaptive_temperature.py
COPY dmft/physics_validation.py /app/physics_validation.py

# ── MPI tuning for g2-standard-4 (4 vCPUs, 16GB, 1x L4 GPU) ────────────────
# GNN training + predictions use the L4 GPU + ~1-2 CPU cores.
# DMFT gets 2 cores — sufficient for N_c=4 single-band or small multi-orbital.
# Override via docker-compose.yml or DMFT_MPI_RANKS env var.
ENV DMFT_MPI_RANKS=2
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DMFT_WORK_DIR=/data/dmft_jobs
ENV DMFT_RESULTS_DIR=/data/dmft_results
ENV DMFT_BUNDLE_DIR=/data/dmft_bundles
ENV TRIQS_PYTHON=python3

EXPOSE 8780

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python3 /app/triqs-healthcheck.py

WORKDIR /app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8780", "--timeout", "7200", "dmft-service:app"]
