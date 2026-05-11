# ==============================================================================
# TRIQS 3.3.x + solid_dmft — DMFT solver for Quantum Alchemy Engine
#
# Builds TRIQS and all solvers from source (C++ compilation, ~30-60 min).
# TRIQS does NOT publish pip wheels or Docker Hub images.
#
# Build (first time ~45 min, cached rebuilds ~2 min):
#   docker build -t qae-dmft:latest -f dmft/Dockerfile .
#
# Run (on gnn-training VM):
#   docker compose -f dmft/docker-compose.yml up -d
# ==============================================================================

FROM ubuntu:22.04

LABEL maintainer="QAE <damsgardhunter@gmail.com>"
LABEL description="TRIQS 3.3 + solid_dmft for QAE DMFT pipeline"

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps for TRIQS C++ build ─────────────────────────────────────────
# TRIQS 3.3.x requires GCC >= 12. Ubuntu 22.04 ships GCC 11.
# Install GCC 12 from the Ubuntu toolchain PPA and set as default.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg software-properties-common \
      git build-essential cmake gfortran \
      python3 python3-pip python3-dev python3-setuptools \
      libopenmpi-dev openmpi-bin \
      libhdf5-mpi-dev libfftw3-dev \
      libgmp-dev libnfft3-dev \
      libboost-all-dev \
      libclang-dev \
      libblas-dev liblapack-dev \
      wget jq openssh-client \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update -qq \
    && apt-get install -y gcc-12 g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-12 120 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-12 120 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps needed before TRIQS build ────────────────────────────────────
# --ignore-installed: Ubuntu 22.04 blinker conflicts with Flask
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --ignore-installed \
      numpy scipy h5py mpi4py mako \
      sphinx sphinx-rtd-theme \
      matplotlib \
      flask==3.1.* gunicorn==23.* \
      toml==0.10.* watchdog==6.*

# ── Build TRIQS core from source ────────────────────────────────────────────
# TRIQS 3.3.x — the core Green's function library
ENV TRIQS_INSTALL_DIR=/opt/triqs
ENV CXXFLAGS="-std=c++20"

RUN cd /tmp && \
    git clone --depth 1 --branch 3.3.x https://github.com/TRIQS/triqs.git && \
    mkdir triqs/build && cd triqs/build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=${TRIQS_INSTALL_DIR} \
      -DCMAKE_BUILD_TYPE=Release \
      -DBuild_Tests=OFF \
      -DBuild_Documentation=OFF \
    && make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/triqs

ENV PATH="${TRIQS_INSTALL_DIR}/bin:${PATH}"
ENV PYTHONPATH="${TRIQS_INSTALL_DIR}/lib/python3/dist-packages:${PYTHONPATH}"
ENV CMAKE_PREFIX_PATH="${TRIQS_INSTALL_DIR}:${CMAKE_PREFIX_PATH}"
ENV LD_LIBRARY_PATH="${TRIQS_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"

# ── Build TRIQS/CTHYB (CT-HYB impurity solver) ─────────────────────────────
RUN cd /tmp && \
    git clone --depth 1 --branch 3.3.x https://github.com/TRIQS/cthyb.git && \
    mkdir cthyb/build && cd cthyb/build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=${TRIQS_INSTALL_DIR} \
      -DCMAKE_BUILD_TYPE=Release \
      -DBuild_Tests=OFF \
      -DBuild_Documentation=OFF \
    && make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/cthyb

# ── Build TRIQS/DFTTools (Wannier90 converter, SumkDFT) ────────────────────
RUN cd /tmp && \
    git clone --depth 1 --branch 3.3.x https://github.com/TRIQS/dft_tools.git && \
    mkdir dft_tools/build && cd dft_tools/build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=${TRIQS_INSTALL_DIR} \
      -DCMAKE_BUILD_TYPE=Release \
      -DBuild_Tests=OFF \
      -DBuild_Documentation=OFF \
    && make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/dft_tools

# ── Build TRIQS/maxent (analytic continuation) ──────────────────────────────
RUN cd /tmp && \
    git clone --depth 1 --branch 3.3.x https://github.com/TRIQS/maxent.git && \
    mkdir maxent/build && cd maxent/build && \
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=${TRIQS_INSTALL_DIR} \
      -DCMAKE_BUILD_TYPE=Release \
      -DBuild_Tests=OFF \
      -DBuild_Documentation=OFF \
    && make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/maxent

# ── Install solid_dmft ──────────────────────────────────────────────────────
RUN cd /opt && \
    git clone --depth 1 --branch stable https://github.com/flatironinstitute/solid_dmft.git && \
    cd solid_dmft && \
    pip3 install --no-cache-dir . && \
    cd / && rm -rf /opt/solid_dmft/.git

# ── Verify TRIQS installation ───────────────────────────────────────────────
RUN python3 -c "import triqs; print(f'TRIQS {triqs.version.version} installed')" && \
    python3 -c "import triqs_cthyb; print('CTHYB installed')" && \
    python3 -c "import triqs_dft_tools; print('DFTTools installed')" && \
    (python3 -c "import triqs_maxent; print('maxent installed')" || echo "maxent: skipped") && \
    (python3 -c "import solid_dmft; print('solid_dmft installed')" || echo "solid_dmft: skipped")

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

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python3 /app/triqs-healthcheck.py

WORKDIR /app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8780", "--timeout", "7200", "dmft-service:app"]
