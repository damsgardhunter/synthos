"""
JARVIS Dataset Downloader for Quantum Alchemy Engine
=====================================================
Downloads the four most relevant JARVIS datasets and converts them to
CSV files in server/learning/ that jarvis-ingestion.ts will pick up.

Datasets downloaded:
  1. supercon_chem  (16,414 rows) — chemical formulas + Tc labels
  2. supercon_3d    (1,058 rows)  — DFT-verified SC structures + Tc
  3. supercon_2d    (161 rows)    — 2D SC structures + Tc
  4. dft_3d         (75,993 rows) — inorganic materials, filtered to metallic
                                    subset → used as Tc=0 negative examples

Install requirements:
  pip install jarvis-tools pandas

Run from repo root:
  python scripts/download-jarvis.py

Output files written to server/learning/:
  jarvis_supercon_chem.csv
  jarvis_supercon_3d.csv
  jarvis_supercon_2d.csv
  jarvis_dft3d_metallic.csv
"""

import os
import sys
import csv
import json
import traceback

# ── Install check ─────────────────────────────────────────────────────────────

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    os.system(f"{sys.executable} -m pip install pandas")
    import pandas as pd

try:
    from jarvis.db.figshare import data as jarvis_data
except ImportError:
    print("Installing jarvis-tools...")
    os.system(f"{sys.executable} -m pip install jarvis-tools")
    from jarvis.db.figshare import data as jarvis_data

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
OUT_DIR     = os.path.join(REPO_ROOT, "server", "learning")

# Metallic filter for dft_3d: only inorganic materials with very small bandgap.
# These are plausible non-SC metals — safe to label Tc=0.
DFT3D_BANDGAP_CUTOFF_EV = 0.05   # eV  — essentially zero-gap (metallic)
# Elements that indicate an organic framework — skip these rows in dft_3d
ORGANIC_ELEMENTS = {"C", "N", "S", "P", "Cl", "Br", "I", "F"}
# Skip if formula contains mostly organic (>40% organic element fraction)
ORGANIC_FRACTION_CUTOFF = 0.40

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(val, default=None):
    try:
        v = float(val)
        return v if v == v else default  # NaN check
    except (TypeError, ValueError):
        return default

def parse_formula_elements(formula: str) -> set:
    """Very basic element extractor — uppercase letter + optional lowercase."""
    import re
    return set(re.findall(r"[A-Z][a-z]?", str(formula)))

def organic_fraction(formula: str) -> float:
    """Fraction of distinct element types that are 'organic'."""
    els = parse_formula_elements(formula)
    if not els:
        return 0.0
    return len(els & ORGANIC_ELEMENTS) / len(els)

def get_key(row: dict, *names, default=None):
    """Try multiple key names, return first match."""
    for n in names:
        if n in row and row[n] is not None and row[n] != "na" and row[n] != "":
            return row[n]
    return default

def crystal_system_from_spg(spg_num: int) -> str:
    """Map spacegroup number to crystal system."""
    if spg_num is None:
        return ""
    n = int(spg_num)
    if n <= 2:   return "triclinic"
    if n <= 15:  return "monoclinic"
    if n <= 74:  return "orthorhombic"
    if n <= 142: return "tetragonal"
    if n <= 167: return "trigonal"
    if n <= 194: return "hexagonal"
    return "cubic"

def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):,} rows → {path}")

# ── Dataset 1: supercon_chem ──────────────────────────────────────────────────

def download_supercon_chem():
    print("\n[1/4] Downloading supercon_chem (~16k rows)...")
    d = jarvis_data("supercon_chem")
    print(f"  Raw entries: {len(d)}")
    if d:
        print(f"  Keys: {list(d[0].keys())}")

    rows = []
    for i, entry in enumerate(d):
        formula = str(get_key(entry, "formula", "material", "composition", default=""))
        if not formula:
            continue

        # Tc — try common key names
        tc = safe_float(get_key(entry, "Tc", "tc", "critical_temp", "critical_temperature", "Tc_K"))

        spg = str(get_key(entry, "spg", "spacegroup", "space_group", "sg", default="") or "")
        spg_num = safe_float(get_key(entry, "spg_number", "spacegroup_number", default=None))
        crystal_sys = crystal_system_from_spg(int(spg_num)) if spg_num else \
                      str(get_key(entry, "crystal_system", "lattice", default="") or "")
        family = str(get_key(entry, "type", "family", "class", "material_type", default="") or "")
        pressure = safe_float(get_key(entry, "pressure", "pressure_gpa", default=0)) or 0.0

        rows.append({
            "formula": formula,
            "tc": tc if tc is not None else "",
            "is_superconductor": "true" if (tc is not None and tc > 0) else "true",
            "source": "jarvis-supercon-chem",
            "external_id": f"jsc-{i}",
            "space_group": spg,
            "crystal_system": crystal_sys,
            "family": family,
            "lambda": "",
            "pressure_gpa": pressure,
            "bandgap_ev": "",
            "formation_energy_per_atom": "",
            "data_confidence": "experimental",
            "raw_data": json.dumps({k: str(v) for k, v in entry.items() if k not in ("atoms",)}),
        })

    path = os.path.join(OUT_DIR, "jarvis_supercon_chem.csv")
    write_csv(path, rows, [
        "formula", "tc", "is_superconductor", "source", "external_id",
        "space_group", "crystal_system", "family", "lambda", "pressure_gpa",
        "bandgap_ev", "formation_energy_per_atom", "data_confidence", "raw_data",
    ])
    print(f"  Superconductors with Tc > 0: {sum(1 for r in rows if safe_float(r['tc'], 0) > 0):,}")

# ── Dataset 2+3: supercon_3d and supercon_2d ──────────────────────────────────

def download_supercon_dft(dataset_name: str, filename: str, label: str):
    print(f"\n[2-3/4] Downloading {dataset_name}...")
    d = jarvis_data(dataset_name)
    print(f"  Raw entries: {len(d)}")
    if d:
        print(f"  Keys: {list(d[0].keys())}")

    rows = []
    for i, entry in enumerate(d):
        formula = str(get_key(entry, "formula", "composition", "material", default=""))
        if not formula:
            # Try extracting from atoms dict
            atoms = entry.get("atoms", {})
            if isinstance(atoms, dict):
                formula = str(atoms.get("composition", ""))
        if not formula:
            continue

        tc = safe_float(get_key(entry, "Tc", "tc", "critical_temp", "Tc_supercon", "Tc_K"))
        spg_num = safe_float(get_key(entry, "spg_number", "spg", "spacegroup_number"))
        spg = str(get_key(entry, "spacegroup", "space_group", default="") or
                  (str(int(spg_num)) if spg_num else ""))
        crystal_sys = crystal_system_from_spg(int(spg_num)) if spg_num else \
                      str(get_key(entry, "crys", "crystal_system", default="") or "")

        bandgap = safe_float(get_key(
            entry, "optb88vdw_bandgap", "bandgap", "bandgap_pbe", "gap"
        ))
        formation_e = safe_float(get_key(
            entry, "formation_energy_peratom", "formation_energy_per_atom",
            "ef_per_atom", "delta_e"
        ))
        lambda_val = safe_float(get_key(
            entry, "lambda_electron_phonon", "lambda", "el_ph_coupling"
        ))
        pressure = safe_float(get_key(entry, "pressure", "pressure_gpa", default=0)) or 0.0
        jid = str(get_key(entry, "jid", "id", default=f"j3d-{i}"))

        rows.append({
            "formula": formula,
            "tc": tc if tc is not None else "",
            "is_superconductor": "true",
            "source": f"jarvis-{dataset_name}",
            "external_id": jid,
            "space_group": spg,
            "crystal_system": crystal_sys,
            "family": "",
            "lambda": lambda_val if lambda_val is not None else "",
            "pressure_gpa": pressure,
            "bandgap_ev": bandgap if bandgap is not None else "",
            "formation_energy_per_atom": formation_e if formation_e is not None else "",
            "data_confidence": "dft-verified",
            "raw_data": json.dumps({
                k: str(v) for k, v in entry.items()
                if k not in ("atoms", "wannier_bands", "phonon_bandstructure")
            }),
        })

    path = os.path.join(OUT_DIR, filename)
    write_csv(path, rows, [
        "formula", "tc", "is_superconductor", "source", "external_id",
        "space_group", "crystal_system", "family", "lambda", "pressure_gpa",
        "bandgap_ev", "formation_energy_per_atom", "data_confidence", "raw_data",
    ])
    with_tc = sum(1 for r in rows if safe_float(r["tc"], 0) > 0)
    print(f"  {label} with Tc: {with_tc:,} / {len(rows):,}")

# ── Dataset 4: dft_3d (metallic negative examples) ───────────────────────────

def download_dft3d_metallic():
    print("\n[4/4] Downloading dft_3d (~76k rows) — extracting metallic subset...")
    d = jarvis_data("dft_3d")
    print(f"  Total entries: {len(d):,}")
    if d:
        print(f"  Keys: {list(d[0].keys())}")

    rows = []
    skipped_organic = 0
    skipped_insulator = 0
    skipped_noformula = 0

    for i, entry in enumerate(d):
        formula = str(get_key(entry, "formula", "composition", default=""))
        if not formula:
            skipped_noformula += 1
            continue

        # Skip entries with organic elements (they're not SC-relevant metals)
        if organic_fraction(formula) >= ORGANIC_FRACTION_CUTOFF:
            skipped_organic += 1
            continue

        # Only keep metallic entries (very small or zero bandgap)
        bandgap = safe_float(get_key(
            entry, "optb88vdw_bandgap", "bandgap", "mbj_bandgap", "gap", default=None
        ))
        if bandgap is None or bandgap > DFT3D_BANDGAP_CUTOFF_EV:
            skipped_insulator += 1
            continue

        spg_num = safe_float(get_key(entry, "spg_number", "spg", default=None))
        spg = str(get_key(entry, "spacegroup", "space_group", default="") or
                  (str(int(spg_num)) if spg_num else ""))
        crystal_sys = crystal_system_from_spg(int(spg_num)) if spg_num else \
                      str(get_key(entry, "crys", "crystal_system", default="") or "")
        formation_e = safe_float(get_key(
            entry, "formation_energy_peratom", "formation_energy_per_atom",
            "ef_per_atom", "delta_e"
        ))
        jid = str(get_key(entry, "jid", "id", default=f"dft3d-{i}"))

        # Check if this entry already has a SC label in JARVIS
        tc_jarvis = safe_float(get_key(entry, "Tc_supercon", "Tc", "tc", default=None))
        if tc_jarvis and tc_jarvis > 0:
            # JARVIS itself flags this as a superconductor — keep as SC entry
            is_sc = "true"
            tc_out = tc_jarvis
            confidence = "dft-verified"
        else:
            is_sc = "false"
            tc_out = 0.0
            confidence = "computed"

        rows.append({
            "formula": formula,
            "tc": tc_out,
            "is_superconductor": is_sc,
            "source": "jarvis-dft3d",
            "external_id": jid,
            "space_group": spg,
            "crystal_system": crystal_sys,
            "family": "",
            "lambda": "",
            "pressure_gpa": 0.0,
            "bandgap_ev": bandgap,
            "formation_energy_per_atom": formation_e if formation_e is not None else "",
            "data_confidence": confidence,
            "raw_data": json.dumps({
                k: str(v) for k, v in entry.items()
                if k not in ("atoms", "wannier_bands")
            }),
        })

    path = os.path.join(OUT_DIR, "jarvis_dft3d_metallic.csv")
    write_csv(path, rows, [
        "formula", "tc", "is_superconductor", "source", "external_id",
        "space_group", "crystal_system", "family", "lambda", "pressure_gpa",
        "bandgap_ev", "formation_energy_per_atom", "data_confidence", "raw_data",
    ])
    print(f"  Skipped (insulator/semiconductor): {skipped_insulator:,}")
    print(f"  Skipped (organic): {skipped_organic:,}")
    print(f"  Skipped (no formula): {skipped_noformula:,}")
    sc_in_metallic = sum(1 for r in rows if r["is_superconductor"] == "true")
    print(f"  SC entries found in dft_3d: {sc_in_metallic:,}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  JARVIS Downloader — Quantum Alchemy Engine")
    print(f"  Output directory: {OUT_DIR}")
    print("=" * 60)

    datasets = [
        ("supercon_chem", download_supercon_chem),
        ("supercon_3d",   lambda: download_supercon_dft("supercon_3d", "jarvis_supercon_3d.csv", "SC structures")),
        ("supercon_2d",   lambda: download_supercon_dft("supercon_2d", "jarvis_supercon_2d.csv", "2D SC structures")),
        ("dft_3d",        download_dft3d_metallic),
    ]

    failed = []
    for name, fn in datasets:
        try:
            fn()
        except Exception as e:
            print(f"\n  ERROR downloading {name}: {e}")
            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"  Failed: {failed}")
        print("  Partial download complete — re-run to retry failed datasets.")
    else:
        print("  All datasets downloaded successfully.")
    print("""
  Next steps:
    1. git pull on GCP to get the new CSV files (or scp them)
    2. The GCP worker will auto-ingest them via jarvis-ingestion.ts on next restart
    3. Or call: startJarvisIngestion() manually from gcp-worker/index.ts
""")
