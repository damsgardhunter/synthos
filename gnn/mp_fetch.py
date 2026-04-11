"""
Materials Project graph fetcher for the GNN training pipeline.

Mirrors Colab Cell 6 exactly: pulls cuprates (elements=Cu+O, metal,
energy_above_hull ≤ 0.1) and hydrides (elements=H, metal,
energy_above_hull ≤ 0.15) from the Materials Project, builds
SuperconGraph instances via build_crystal_graph(), and applies
KNOWN_TC + HYDRIDE_PRESSURE_GPA labels in the same way Colab Cell 8
does (only matched formulas get target_tc; unmatched graphs go in as
target_tc=None multi-task anchors via formation_energy / bandgap).

Why this exists:
  Colab's training corpus includes ~hundreds of cuprate and hydride
  graphs that come from a live MP fetch in Cell 6. The GCP server
  was getting its training data exclusively from
  loadSuperconExternalSamples() in TS-land, so those MP graphs were
  missing — meaning the encoder had ~100-500 fewer examples of the
  hydride/cuprate manifold than Colab. This module closes that gap.

Caching:
  Hitting MP on every /train call would add ~30-60s of network I/O
  per training run, which compounds when startup training retries.
  Results are cached to MP_CACHE_PATH as a pickle of (cuprate_payload,
  hydride_payload). Cache TTL is 7 days. Set MP_FETCH_FORCE_REFRESH=1
  in the environment to bypass the cache for one run.

Failure modes:
  If MP_API_KEY is unset, the fetch is skipped silently and an empty
  list is returned. This makes the helper safe to call in environments
  where MP credentials aren't available — training proceeds without
  the MP augmentation rather than crashing.

  If mp-api is not installed, the import error is caught at module
  import time, fetch_mp_graphs() returns ([], []) immediately, and a
  warning is logged on first call.

  Network timeouts and MP API errors are caught per-call and produce
  ([], []) — never propagate to the trainer.
"""

import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("gnn.mp_fetch")

# Try to import mp-api at module load. If unavailable, fetch_mp_graphs()
# becomes a no-op that returns empty lists — training proceeds without
# the MP augmentation.
try:
    from mp_api.client import MPRester  # type: ignore
    _HAS_MP_API = True
except ImportError:
    MPRester = None  # type: ignore
    _HAS_MP_API = False

# Local SuperconGraph builder. This is the same module Colab Cell 6
# imports — graph construction is identical on both sides.
try:
    from graph_builder import build_crystal_graph  # type: ignore
except ImportError:
    build_crystal_graph = None  # type: ignore

# Curated label tables — same source the /train endpoint uses for the
# DB samples, so labeling logic stays consistent across data sources.
try:
    from training_data import KNOWN_TC, HYDRIDE_PRESSURE_GPA
except ImportError:
    KNOWN_TC = {}
    HYDRIDE_PRESSURE_GPA = {}

# ── Cache configuration ──────────────────────────────────────────────
# Cache lives in /opt/qae/gnn_weights on the GCP worker (same dir as
# the trained ensembles, already mounted/writable). Falls back to
# /tmp if that path doesn't exist (e.g. local dev).
_DEFAULT_CACHE_DIR = Path(os.environ.get("MP_FETCH_CACHE_DIR", "/opt/qae/gnn_weights"))
if not _DEFAULT_CACHE_DIR.exists():
    _DEFAULT_CACHE_DIR = Path("/tmp")
MP_CACHE_PATH = _DEFAULT_CACHE_DIR / "mp_fetch_cache.pkl"
MP_CACHE_TTL_S = 7 * 24 * 3600  # 7 days

# ── Filters matching Colab Cell 6 verbatim ───────────────────────────
def _is_cuprate(formula: str) -> bool:
    """Colab line ~245: bool(re.search('Cu', f) and re.search('O', f))."""
    return bool(re.search(r"Cu", formula) and re.search(r"O", formula))


def _is_hydride(formula: str) -> bool:
    """Colab line ~248: bool(re.search('H', f))."""
    return bool(re.search(r"H", formula))


# ── Cell 6 fetch + Cell 8 labeling, in one function per family ────────
def _fetch_cuprates(mpr) -> List[dict]:
    """
    Mirror Colab Cell 6 cuprate block. Returns a list of *raw payloads*
    (not built graphs), so we can cache the network result and rebuild
    graphs cheaply on each load — graph_builder may have changed schema
    between cache writes and the trainer needs to see the live schema.
    """
    docs = mpr.materials.summary.search(
        elements=["Cu", "O"],
        is_metal=True,
        fields=[
            "formula_pretty", "formation_energy_per_atom",
            "band_gap", "energy_above_hull", "symmetry",
        ],
        num_chunks=20, chunk_size=500,
    )
    log.info(f"[mp_fetch] cuprate docs from MP: {len(docs)}")
    payloads: List[dict] = []
    for doc in docs:
        formula = doc.formula_pretty
        if not _is_cuprate(formula):
            continue
        if doc.energy_above_hull is not None and doc.energy_above_hull > 0.1:
            continue
        sg = cs = None
        if doc.symmetry:
            sg = str(doc.symmetry.symbol) if doc.symmetry.symbol else None
            cs = str(doc.symmetry.crystal_system).lower() if doc.symmetry.crystal_system else None
        payloads.append({
            "formula":          formula,
            "formation_energy": doc.formation_energy_per_atom,
            "band_gap":         doc.band_gap,
            "space_group":      sg,
            "crystal_system":   cs,
        })
    return payloads


def _fetch_hydrides(mpr) -> List[dict]:
    """Mirror Colab Cell 6 hydride block (energy_above_hull cap is 0.15, looser than cuprates)."""
    docs = mpr.materials.summary.search(
        elements=["H"],
        is_metal=True,
        fields=[
            "formula_pretty", "formation_energy_per_atom",
            "band_gap", "energy_above_hull", "symmetry",
        ],
        num_chunks=20, chunk_size=500,
    )
    log.info(f"[mp_fetch] hydride docs from MP: {len(docs)}")
    payloads: List[dict] = []
    for doc in docs:
        formula = doc.formula_pretty
        if not _is_hydride(formula):
            continue
        if doc.energy_above_hull is not None and doc.energy_above_hull > 0.15:
            continue
        sg = cs = None
        if doc.symmetry:
            sg = str(doc.symmetry.symbol) if doc.symmetry.symbol else None
            cs = str(doc.symmetry.crystal_system).lower() if doc.symmetry.crystal_system else None
        payloads.append({
            "formula":          formula,
            "formation_energy": doc.formation_energy_per_atom,
            "band_gap":         doc.band_gap,
            "space_group":      sg,
            "crystal_system":   cs,
        })
    return payloads


def _build_graph_from_payload(payload: dict, family: str):
    """
    Rebuild a SuperconGraph from a cached MP payload.

    Mirrors Colab Cell 6 graph construction, then applies the
    Cell 8 labeling step (which lives in this same function so MP
    graphs always carry their final labels regardless of caller).
    """
    if build_crystal_graph is None:
        return None
    formula = payload["formula"]
    hints = {}
    if payload.get("formation_energy") is not None:
        hints["formation_energy"] = payload["formation_energy"]
    structure = None
    if payload.get("space_group") or payload.get("crystal_system"):
        structure = {
            "spaceGroup":    payload.get("space_group"),
            "crystalSystem": payload.get("crystal_system"),
        }
    try:
        g = build_crystal_graph(
            formula=formula,
            structure=structure,
            physics_hints=hints,
            pressure_gpa=0.0,
        )
    except Exception as e:
        log.debug(f"[mp_fetch] graph build failed for {formula}: {e}")
        return None

    g.target_fe  = payload.get("formation_energy")
    g.target_bg  = payload.get("band_gap")
    g.formula    = formula

    # Default: unlabeled multi-task anchor (matches Colab Cell 6).
    g.target_tc  = None
    g.target_psc = None

    # Cell 8 labeling: cuprates get TC if known; hydrides get pressure
    # injection then TC if known. Same formula-key match as Colab.
    if family == "hydride" and formula in HYDRIDE_PRESSURE_GPA:
        g.pressure_gpa = HYDRIDE_PRESSURE_GPA[formula]
    if formula in KNOWN_TC:
        tc = KNOWN_TC[formula]
        g.target_tc  = tc if tc > 0 else None
        g.target_psc = 1.0 if tc > 0 else 0.0

    return g


def _load_cache() -> Optional[Tuple[List[dict], List[dict]]]:
    """Return (cuprates, hydrides) raw payloads if cache exists and is fresh."""
    if not MP_CACHE_PATH.exists():
        return None
    if os.environ.get("MP_FETCH_FORCE_REFRESH") == "1":
        log.info("[mp_fetch] MP_FETCH_FORCE_REFRESH=1 — bypassing cache")
        return None
    age_s = time.time() - MP_CACHE_PATH.stat().st_mtime
    if age_s > MP_CACHE_TTL_S:
        log.info(f"[mp_fetch] cache expired ({age_s/3600:.1f}h old, TTL={MP_CACHE_TTL_S/3600:.0f}h)")
        return None
    try:
        with MP_CACHE_PATH.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 2:
            log.info(
                f"[mp_fetch] loaded cache ({age_s/3600:.1f}h old): "
                f"{len(data[0])} cuprate + {len(data[1])} hydride payloads"
            )
            return data
    except Exception as e:
        log.warning(f"[mp_fetch] cache load failed: {e}")
    return None


def _save_cache(cuprates: List[dict], hydrides: List[dict]) -> None:
    try:
        MP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MP_CACHE_PATH.open("wb") as f:
            pickle.dump((cuprates, hydrides), f)
        log.info(f"[mp_fetch] cached {len(cuprates) + len(hydrides)} payloads to {MP_CACHE_PATH}")
    except Exception as e:
        log.warning(f"[mp_fetch] cache save failed: {e}")


# ── Public entry point ────────────────────────────────────────────────
def fetch_mp_graphs() -> Tuple[List, List]:
    """
    Returns (cuprate_graphs, hydride_graphs) — lists of SuperconGraph
    objects ready to merge into the training set.

    Safe to call from any environment: returns ([], []) when MP_API_KEY
    is unset, mp-api is not installed, build_crystal_graph is missing,
    or any network error occurs. Never raises.

    Caches the raw MP payloads (not graphs) so a stale cache survives
    schema changes in graph_builder — graphs are always rebuilt fresh.
    """
    if build_crystal_graph is None:
        log.warning("[mp_fetch] graph_builder not importable — skipping MP fetch")
        return [], []

    # Try cache first
    cached = _load_cache()
    if cached is not None:
        cuprate_payloads, hydride_payloads = cached
    else:
        if not _HAS_MP_API:
            log.warning("[mp_fetch] mp-api not installed — skipping MP fetch (pip install mp-api)")
            return [], []
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            log.warning("[mp_fetch] MP_API_KEY not set — skipping MP fetch")
            return [], []
        try:
            t0 = time.perf_counter()
            with MPRester(api_key) as mpr:
                cuprate_payloads = _fetch_cuprates(mpr)
                hydride_payloads = _fetch_hydrides(mpr)
            log.info(
                f"[mp_fetch] fetch complete in {time.perf_counter()-t0:.1f}s: "
                f"{len(cuprate_payloads)} cuprates + {len(hydride_payloads)} hydrides"
            )
            _save_cache(cuprate_payloads, hydride_payloads)
        except Exception as e:
            log.warning(f"[mp_fetch] MP fetch failed: {type(e).__name__}: {e}")
            return [], []

    # Rebuild graphs fresh from cached payloads — graph_builder schema
    # may have changed since the cache was written.
    cuprates: List = []
    for p in cuprate_payloads:
        g = _build_graph_from_payload(p, family="cuprate")
        if g is not None:
            cuprates.append(g)

    hydrides: List = []
    for p in hydride_payloads:
        g = _build_graph_from_payload(p, family="hydride")
        if g is not None:
            hydrides.append(g)

    labeled_cuprates = sum(1 for g in cuprates if g.target_tc is not None)
    labeled_hydrides = sum(1 for g in hydrides if g.target_tc is not None)
    log.info(
        f"[mp_fetch] built {len(cuprates)} cuprate + {len(hydrides)} hydride graphs "
        f"(labeled: {labeled_cuprates} cuprates, {labeled_hydrides} hydrides)"
    )
    return cuprates, hydrides
