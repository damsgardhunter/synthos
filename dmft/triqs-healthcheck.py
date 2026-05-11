#!/usr/bin/env python3
"""Health check for TRIQS + solid_dmft installation."""
import sys

def check():
    errors = []

    # TRIQS core
    try:
        import triqs.gf  # noqa: F401
    except ImportError as e:
        errors.append(f"triqs.gf: {e}")

    # CTHYB impurity solver
    try:
        import triqs_cthyb  # noqa: F401
    except ImportError as e:
        errors.append(f"triqs_cthyb: {e}")

    # DFTTools (Wannier90 converter, SumkDFT)
    try:
        import triqs_dft_tools  # noqa: F401
    except ImportError as e:
        errors.append(f"triqs_dft_tools: {e}")

    # maxent (analytic continuation)
    try:
        import triqs_maxent  # noqa: F401
    except ImportError as e:
        errors.append(f"triqs_maxent: {e}")

    # solid_dmft
    try:
        import solid_dmft  # noqa: F401
    except ImportError as e:
        errors.append(f"solid_dmft: {e}")

    # h5py for HDF5 bundle I/O
    try:
        import h5py  # noqa: F401
    except ImportError as e:
        errors.append(f"h5py: {e}")

    # MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        _ = comm.Get_size()
    except ImportError as e:
        errors.append(f"mpi4py: {e}")

    if errors:
        print(f"UNHEALTHY: {'; '.join(errors)}", file=sys.stderr)
        sys.exit(1)

    print("OK: triqs, cthyb, dft_tools, maxent, solid_dmft, h5py, mpi4py")
    sys.exit(0)

if __name__ == "__main__":
    check()
