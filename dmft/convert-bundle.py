#!/usr/bin/env python3
"""
Convert a QAE DMFT bundle from JSON to HDF5 format.

Usage:
  python3 dmft/convert-bundle.py input.json [output.h5]

If output path is omitted, replaces .json with .h5.
"""

import json
import sys
import os
import numpy as np
import h5py


def convert(json_path: str, h5_path: str):
    with open(json_path) as f:
        d = json.load(f)

    with h5py.File(h5_path, "w") as hf:
        # ── Hamiltonian ──────────────────────────────────────────────────
        if d.get("hamiltonian"):
            h = d["hamiltonian"]
            hg = hf.create_group("hamiltonian")
            hr = np.array(h["hr_data"])
            hg.create_dataset("hr_data", data=hr)
            hg.create_dataset("num_wann", data=h["num_wann"])
            hg.create_dataset("n_rpts", data=h["n_rpts"])
            hg.create_dataset("degeneracies", data=np.array(h["degeneracies"]))
            hg.create_dataset("kmesh", data=np.array(h["kmesh"]))

            # Fourier transform H(R) -> H(k)
            nw = h["num_wann"]
            km = h["kmesh"]
            nk = km[0] * km[1] * km[2]
            hk = np.zeros((nk, nw, nw), dtype=complex)

            # Build R-vec → H(R) mapping, preserving INSERTION ORDER (matches
            # the degeneracy array order from Wannier90 _hr.dat).
            r_vecs = {}
            r_first_idx = {}  # R-vec → original row index for degeneracy lookup
            for idx, row in enumerate(h["hr_data"]):
                rx, ry, rz = int(row[0]), int(row[1]), int(row[2])
                i, j = int(row[3]) - 1, int(row[4]) - 1  # 1-based -> 0-based
                re, im = row[5], row[6]
                key = (rx, ry, rz)
                if key not in r_vecs:
                    r_vecs[key] = np.zeros((nw, nw), dtype=complex)
                    # Each unique R appears nw² consecutive rows; the R-index
                    # is idx // nw² (preserves Wannier90 ordering)
                    r_first_idx[key] = len(r_first_idx)
                r_vecs[key][i, j] = complex(re, im)

            degs = h["degeneracies"]
            # Iterate R-vectors in INSERTION ORDER (matches degeneracy array)
            kidx = 0
            for ik1 in range(km[0]):
                for ik2 in range(km[1]):
                    for ik3 in range(km[2]):
                        kfrac = np.array([ik1 / km[0], ik2 / km[1], ik3 / km[2]])
                        for R, H_R in r_vecs.items():
                            ri = r_first_idx[R]
                            phase = np.exp(2j * np.pi * np.dot(kfrac, R))
                            deg = degs[ri] if ri < len(degs) else 1
                            hk[kidx] += phase * H_R / deg
                        kidx += 1

            hg.create_dataset("hk", data=hk)
            kpts = []
            for ik1 in range(km[0]):
                for ik2 in range(km[1]):
                    for ik3 in range(km[2]):
                        kpts.append([ik1 / km[0], ik2 / km[1], ik3 / km[2]])
            hg.create_dataset("kpoints", data=np.array(kpts))
            print(f"  H(k): {nk} k-points, {nw} Wannier functions")

        # ── Correlated subspace ──────────────────────────────────────────
        cs = d["correlated_subspace"]
        cg = hf.create_group("correlated_subspace")
        cg.create_dataset("corr_shells", data=json.dumps(cs["corr_shells"]))
        if cs["corr_shells"]:
            max_dim = max(s["dim"] for s in cs["corr_shells"])
            nk_total = int(np.prod(d["hamiltonian"]["kmesh"])) if d.get("hamiltonian") else 1
            nw_total = d["hamiltonian"]["num_wann"] if d.get("hamiltonian") else max_dim
            n_sh = len(cs["corr_shells"])
            proj = np.zeros((nk_total, n_sh, max_dim, nw_total), dtype=complex)
            orb_offset = 0
            for si, sh in enumerate(cs["corr_shells"]):
                for m in range(sh["dim"]):
                    if orb_offset + m < nw_total:
                        proj[:, si, m, orb_offset + m] = 1.0
                orb_offset += sh["dim"]
            cg.create_dataset("proj_mat", data=proj)
        cg.create_dataset("corr_to_inequiv", data=np.array(cs["corr_to_inequiv"]))
        print(f"  Correlated shells: {len(cs['corr_shells'])}, inequivalent: {cs['n_inequiv_shells']}")

        # ── Interaction ──────────────────────────────────────────────────
        ig = hf.create_group("interaction")
        ig.create_dataset("U_values", data=np.array(d["interaction"]["U_values"]))
        ig.create_dataset("J_values", data=np.array(d["interaction"]["J_values"]))
        ig.create_dataset("hubbard_kind", data=d["interaction"]["hubbard_kind"])

        # ── Structure ────────────────────────────────────────────────────
        sg = hf.create_group("structure")
        sg.create_dataset("formula", data=d["structure"]["formula"])
        sg.create_dataset("elements", data=json.dumps(d["structure"]["elements"]))
        sg.create_dataset("lattice_vectors", data=np.array(d["structure"]["lattice_vectors"]))
        pos = d["structure"]["positions"]
        sg.create_dataset("positions", data=np.array([[p["x"], p["y"], p["z"]] for p in pos]))
        sg.create_dataset("pressure_gpa", data=d["structure"]["pressure_gpa"])

        # ── Electronic ───────────────────────────────────────────────────
        eg = hf.create_group("electronic")
        eg.create_dataset("fermi_energy", data=d["electronic"]["fermi_energy"])
        eg.create_dataset("n_electrons", data=d["electronic"]["n_electrons"])
        eg.create_dataset("magnetic_ordering", data=d["electronic"]["magnetic_ordering"])
        eg.create_dataset("correlation_regime", data=d["electronic"]["correlation_regime"])

        # ── Metadata ─────────────────────────────────────────────────────
        mg = hf.create_group("metadata")
        for k, v in d["metadata"].items():
            mg.create_dataset(k, data=str(v))

    print(f"  Written: {h5_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert-bundle.py input.json [output.h5]")
        sys.exit(1)

    json_path = sys.argv[1]
    h5_path = sys.argv[2] if len(sys.argv) > 2 else json_path.replace(".json", ".h5")

    print(f"Converting {json_path} -> {h5_path}")
    convert(json_path, h5_path)
    print("Done.")
