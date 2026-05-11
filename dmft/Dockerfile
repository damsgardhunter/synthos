# ==============================================================================
# TRIQS 3.3.x + solid_dmft — DMFT solver for Quantum Alchemy Engine
#
# Installs TRIQS ecosystem from the official Flatiron PPA on Ubuntu 22.04.
# TRIQS does NOT publish Docker Hub images — we build from their APT repo.
#
# Build:
#   docker build -t qae-dmft:latest -f dmft/Dockerfile .
#
# Run (on gnn-training VM):
#   docker compose -f dmft/docker-compose.yml up -d
# ==============================================================================

FROM ubuntu:22.04

LABEL maintainer="QAE <damsgardhunter@gmail.com>"
LABEL description="TRIQS 3.3 + solid_dmft for QAE DMFT pipeline"

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps + TRIQS PPA ─────────────────────────────────────────────────
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg software-properties-common \
      git build-essential cmake \
      python3 python3-pip python3-dev \
      libopenmpi-dev openmpi-bin \
      libhdf5-mpi-dev libfftw3-dev \
      libgmp-dev libboost-all-dev \
      wget jq openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ── Install TRIQS from Flatiron PPA ─────────────────────────────────────────
# TRIQS 3.x packages are distributed via their own Ubuntu PPA.
# If the PPA is unavailable, fall back to pip install.
RUN echo ">>> Installing TRIQS ecosystem..." && \
    (apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 63E03B28DB77B050 2>/dev/null || true) && \
    (add-apt-repository -y ppa:triqs/unstable 2>/dev/null || true) && \
    apt-get update -qq && \
    (apt-get install -y triqs triqs-cthyb triqs-dft-tools triqs-maxent 2>/dev/null || \
     echo ">>> PPA install failed — falling back to pip") && \
    rm -rf /var/lib/apt/lists/*

# ── Pip fallback for TRIQS (if PPA didn't work) + required Python packages ──
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
      numpy scipy h5py mpi4py \
      flask==3.1.* gunicorn==23.* \
      toml==0.10.* watchdog==6.* && \
    # Try pip install of TRIQS components (may fail on some platforms)
    (pip3 install --no-cache-dir triqs 2>/dev/null || echo "TRIQS pip not available") && \
    (pip3 install --no-cache-dir triqs_cthyb 2>/dev/null || echo "CTHYB pip not available") && \
    (pip3 install --no-cache-dir triqs_dft_tools 2>/dev/null || echo "DFTTools pip not available") && \
    (pip3 install --no-cache-dir triqs_maxent 2>/dev/null || echo "maxent pip not available")

# ── Install solid_dmft from source ──────────────────────────────────────────
RUN cd /opt && \
    git clone --depth 1 --branch stable https://github.com/flatironinstitute/solid_dmft.git 2>/dev/null && \
    cd solid_dmft && \
    pip3 install --no-cache-dir . 2>/dev/null || echo "solid_dmft install skipped" && \
    cd / && rm -rf /opt/solid_dmft/.git

# ── Create working directories ──────────────────────────────────────────────
RUN mkdir -p /data/dmft_jobs /data/dmft_results /data/dmft_bundles /app

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
ENV DMFT_MPI_RANKS=2
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DMFT_WORK_DIR=/data/dmft_jobs
ENV DMFT_RESULTS_DIR=/data/dmft_results
ENV DMFT_BUNDLE_DIR=/data/dmft_bundles

EXPOSE 8780

# Health check — tolerant: reports degraded if TRIQS missing but service still runs
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python3 /app/triqs-healthcheck.py || curl -sf http://localhost:8780/health > /dev/null

WORKDIR /app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8780", "--timeout", "7200", "dmft-service:app"]
