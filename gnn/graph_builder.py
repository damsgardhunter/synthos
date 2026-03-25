"""
graph_builder.py
================
Converts a TrainingSample (formula string + optional structure hints) into a
SuperconGraph tensor object consumable by the PyTorch GNN.

Mirrors buildCrystalGraph() / buildEdgeFeatures() / buildGlobalFeatures()
from server/learning/graph-neural-net.ts.

NODE_DIM = 32  (must match superconductor_gnn.py constant)
EDGE_DIM = 40  (N_GAUSSIAN_BASIS)
GLOBAL_COMP_DIM = 23
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from superconductor_gnn import (
    SuperconGraph,
    build_rbf_features,
    NODE_DIM, N_GAUSSIAN_BASIS, GLOBAL_COMP_DIM,
    GAUSSIAN_START, GAUSSIAN_END, GAUSSIAN_STEP, GAUSSIAN_WIDTH,
    COSINE_CUTOFF_R,
)

# ── Elemental data ────────────────────────────────────────────────────────────
# Each entry: (Z, period, group, EN, radius_pm, valence, mass_u,
#              debye_K, FIE_eV, EA_eV, mp_K, density_gcc)
# None → use group/period defaults; missing EA → 0.0
ELEMENTAL_DATA: Dict[str, Tuple] = {
    #      Z  per grp  EN    rad  val   mass   dby   fie   ea    mp     dens
    "H":  (1,  1,  1, 2.20,  53,  1,   1.008, 110, 13.60, 0.75, 14,    0.09),
    "He": (2,  1, 18, 0.00,  31,  0,   4.003, 22,  24.59, 0.00, 1,     0.16),
    "Li": (3,  2,  1, 0.98, 167,  1,   6.941, 400,  5.39, 0.62, 454,   0.53),
    "Be": (4,  2,  2, 1.57, 112,  2,   9.012, 1000, 9.32, 0.00, 1560,  1.85),
    "B":  (5,  2, 13, 2.04,  87,  3,  10.811, 1250, 8.30, 0.28, 2349,  2.34),
    "C":  (6,  2, 14, 2.55,  77,  4,  12.011, 2230,11.26, 1.26, 3823,  2.26),
    "N":  (7,  2, 15, 3.04,  75,  5,  14.007, 70,  14.53, 0.00, 63,    1.15),
    "O":  (8,  2, 16, 3.44,  73,  6,  15.999, 70,  13.62, 1.46, 55,    1.33),
    "F":  (9,  2, 17, 3.98,  71,  7,  18.998, 70,  17.42, 3.40, 53,    1.70),
    "Ne": (10, 2, 18, 0.00,  69,  0,  20.180, 70,  21.56, 0.00, 25,    1.20),
    "Na": (11, 3,  1, 0.93, 190,  1,  22.990, 158,  5.14, 0.55, 371,   0.97),
    "Mg": (12, 3,  2, 1.31, 160,  2,  24.305, 318,  7.65, 0.00, 923,   1.74),
    "Al": (13, 3, 13, 1.61, 143,  3,  26.982, 394,  5.99, 0.43, 933,   2.70),
    "Si": (14, 3, 14, 1.90, 117,  4,  28.086, 625,  8.15, 1.39, 1687,  2.33),
    "P":  (15, 3, 15, 2.19, 110,  5,  30.974, 100, 10.49, 0.75, 317,   1.82),
    "S":  (16, 3, 16, 2.58, 104,  6,  32.065, 70,  10.36, 2.08, 388,   2.07),
    "Cl": (17, 3, 17, 3.16,  99,  7,  35.453, 70,  12.97, 3.61, 172,   1.56),
    "Ar": (18, 3, 18, 0.00,  97,  0,  39.948, 70,  15.76, 0.00, 84,    1.40),
    "K":  (19, 4,  1, 0.82, 243,  1,  39.098, 100,  4.34, 0.50, 337,   0.86),
    "Ca": (20, 4,  2, 1.00, 194,  2,  40.078, 230,  6.11, 0.02, 1115,  1.55),
    "Sc": (21, 4,  3, 1.36, 184,  3,  44.956, 360,  6.54, 0.19, 1814,  2.99),
    "Ti": (22, 4,  4, 1.54, 176,  4,  47.867, 420,  6.83, 0.08, 1941,  4.51),
    "V":  (23, 4,  5, 1.63, 171,  5,  50.942, 380,  6.75, 0.53, 2183,  6.11),
    "Cr": (24, 4,  6, 1.66, 166,  6,  51.996, 606,  6.77, 0.67, 2180,  7.19),
    "Mn": (25, 4,  7, 1.55, 161,  7,  54.938, 400,  7.43, 0.00, 1519,  7.43),
    "Fe": (26, 4,  8, 1.83, 156,  8,  55.845, 470,  7.90, 0.15, 1811,  7.87),
    "Co": (27, 4,  9, 1.88, 152,  9,  58.933, 385,  7.88, 0.66, 1768,  8.90),
    "Ni": (28, 4, 10, 1.91, 149, 10,  58.693, 450,  7.64, 1.16, 1728,  8.91),
    "Cu": (29, 4, 11, 1.90, 145, 11,  63.546, 315,  7.73, 1.24, 1358,  8.96),
    "Zn": (30, 4, 12, 1.65, 142, 12,  65.380, 327,  9.39, 0.00, 693,   7.13),
    "Ga": (31, 4, 13, 1.81, 136,  3,  69.723, 240,  5.99, 0.43, 303,   5.91),
    "Ge": (32, 4, 14, 2.01, 125,  4,  72.631, 374,  7.90, 1.23, 1211,  5.32),
    "As": (33, 4, 15, 2.18, 121,  5,  74.922, 285,  9.79, 0.81, 1090,  5.73),
    "Se": (34, 4, 16, 2.55, 116,  6,  78.960, 90,   9.75, 2.02, 494,   4.82),
    "Br": (35, 4, 17, 2.96, 114,  7,  79.904, 70,  11.81, 3.37, 266,   3.12),
    "Kr": (36, 4, 18, 0.00, 112,  0,  83.798, 70,  14.00, 0.00, 116,   2.16),
    "Rb": (37, 5,  1, 0.82, 265,  1,  85.468, 56,   4.18, 0.49, 312,   1.53),
    "Sr": (38, 5,  2, 0.95, 219,  2,  87.620, 147,  5.69, 0.05, 1050,  2.64),
    "Y":  (39, 5,  3, 1.22, 212,  3,  88.906, 280,  6.22, 0.31, 1799,  4.47),
    "Zr": (40, 5,  4, 1.33, 206,  4,  91.224, 291,  6.63, 0.43, 2128,  6.52),
    "Nb": (41, 5,  5, 1.60, 198,  5,  92.906, 275,  6.76, 0.89, 2750,  8.57),
    "Mo": (42, 5,  6, 2.16, 190,  6,  95.960, 450,  7.09, 0.75, 2896, 10.22),
    "Tc": (43, 5,  7, 1.90, 183,  7,  98.000, 453,  7.28, 0.55, 2430, 11.50),
    "Ru": (44, 5,  8, 2.20, 178,  8, 101.070, 600,  7.36, 1.05, 2607, 12.37),
    "Rh": (45, 5,  9, 2.28, 173,  9, 102.906, 480,  7.46, 1.14, 2237, 12.41),
    "Pd": (46, 5, 10, 2.20, 169, 10, 106.420, 274,  8.34, 0.56, 1828, 12.02),
    "Ag": (47, 5, 11, 1.93, 165, 11, 107.868, 225,  7.58, 1.30, 1235, 10.50),
    "Cd": (48, 5, 12, 1.69, 161, 12, 112.411, 209,  8.99, 0.00, 594,   8.65),
    "In": (49, 5, 13, 1.78, 156,  3, 114.818, 108,  5.79, 0.30, 430,   7.31),
    "Sn": (50, 5, 14, 1.96, 145,  4, 118.710, 170,  7.34, 1.20, 505,   7.29),
    "Sb": (51, 5, 15, 2.05, 143,  5, 121.760, 200,  8.61, 1.07, 904,   6.69),
    "Te": (52, 5, 16, 2.10, 135,  6, 127.600, 153,  9.01, 1.97, 723,   6.24),
    "I":  (53, 5, 17, 2.66, 133,  7, 126.904, 70,  10.45, 3.06, 387,   4.93),
    "Xe": (54, 5, 18, 0.00, 130,  0, 131.293, 70,  12.13, 0.00, 165,   3.06),
    "Cs": (55, 6,  1, 0.79, 298,  1, 132.905, 38,   3.89, 0.47, 302,   1.87),
    "Ba": (56, 6,  2, 0.89, 253,  2, 137.327, 110,  5.21, 0.14, 1000,  3.59),
    "La": (57, 6,  0, 1.10, 250,  3, 138.905, 142,  5.58, 0.47, 1193,  6.15),
    "Ce": (58, 6,  0, 1.12, 248,  3, 140.116, 179,  5.54, 0.50, 1068,  6.77),
    "Pr": (59, 6,  0, 1.13, 247,  3, 140.908, 152,  5.47, 0.50, 1208,  6.77),
    "Nd": (60, 6,  0, 1.14, 206,  3, 144.242, 157,  5.53, 0.50, 1297,  7.01),
    "Sm": (62, 6,  0, 1.17, 238,  3, 150.360, 166,  5.64, 0.50, 1345,  7.52),
    "Eu": (63, 6,  0, 1.20, 235,  3, 151.964, 127,  5.67, 0.50, 1095,  5.24),
    "Gd": (64, 6,  0, 1.20, 233,  3, 157.250, 176,  6.15, 0.50, 1585,  7.90),
    "Tb": (65, 6,  0, 1.22, 225,  3, 158.925, 181,  5.86, 0.50, 1629,  8.23),
    "Dy": (66, 6,  0, 1.23, 228,  3, 162.500, 186,  5.94, 0.50, 1685,  8.55),
    "Ho": (67, 6,  0, 1.24, 226,  3, 164.930, 190,  6.02, 0.50, 1747,  8.80),
    "Er": (68, 6,  0, 1.24, 226,  3, 167.259, 188,  6.11, 0.50, 1802,  9.07),
    "Tm": (69, 6,  0, 1.25, 222,  3, 168.934, 200,  6.18, 0.50, 1818,  9.32),
    "Yb": (70, 6,  0, 1.10, 222,  3, 173.045, 120,  6.25, 0.50, 1097,  6.90),
    "Lu": (71, 6,  0, 1.27, 217,  3, 174.967, 210,  5.43, 0.50, 1936,  9.84),
    "Hf": (72, 6,  4, 1.30, 208,  4, 178.490, 252,  6.83, 0.00, 2506, 13.31),
    "Ta": (73, 6,  5, 1.50, 200,  5, 180.948, 240,  7.89, 0.32, 3290, 16.65),
    "W":  (74, 6,  6, 2.36, 193,  6, 183.840, 400,  7.98, 0.82, 3695, 19.25),
    "Re": (75, 6,  7, 1.90, 188,  7, 186.207, 430,  7.88, 0.15, 3459, 21.02),
    "Os": (76, 6,  8, 2.20, 185,  8, 190.230, 500,  8.44, 1.10, 3306, 22.59),
    "Ir": (77, 6,  9, 2.20, 180,  9, 192.217, 420,  8.97, 1.57, 2719, 22.56),
    "Pt": (78, 6, 10, 2.28, 177, 10, 195.084, 234,  8.96, 2.13, 2041, 21.45),
    "Au": (79, 6, 11, 2.54, 174, 11, 196.967, 165,  9.23, 2.31, 1337, 19.30),
    "Hg": (80, 6, 12, 2.00, 171, 12, 200.592, 100, 10.44, 0.00, 234,  13.53),
    "Tl": (81, 6, 13, 1.62, 156,  3, 204.383, 96,   6.11, 0.20, 577,  11.85),
    "Pb": (82, 6, 14, 2.33, 154,  4, 207.200, 88,   7.42, 0.36, 601,  11.34),
    "Bi": (83, 6, 15, 2.02, 143,  5, 208.980, 120,  7.29, 0.95, 544,   9.79),
    "Po": (84, 6, 16, 2.00, 135,  6, 209.000, 90,   8.42, 1.90, 527,   9.20),
    "Th": (90, 7,  0, 1.30, 237,  4, 232.038, 163,  6.31, 0.00, 2023, 11.72),
    "U":  (92, 7,  0, 1.38, 196,  6, 238.029, 207,  6.19, 0.00, 1408, 19.05),
}

# ── Normalisation stats (from FEAT_NORM in graph-neural-net.ts) ──────────────
_NORM = {
    "Z":    (47.5,  27.2),
    "en":   (1.84,  0.73),
    "rad":  (145.0, 50.0),
    "val":  (4.0,   2.4),
    "mass": (100.0, 72.0),
    "dby":  (360.0, 285.0),
    "fie":  (8.3,   3.5),
    "ea":   (0.65,  0.85),
    "mp":   (1400.0,900.0),
    "dens": (6.0,   5.5),
    "per":  (4.0,   2.0),
    "grp":  (9.5,   5.4),
    "vec":  (4.0,   2.4),
    "mis":  (0.06,  0.05),
    "nel":  (2.5,   1.5),
    "std_en":(0.40, 0.35),
    "max_en":(0.90, 0.65),
    "mmass":(100.0, 72.0),
    "srad": (15.0,  18.0),
}

def _z(val: float, key: str) -> float:
    m, s = _NORM[key]
    return (val - m) / s

def _default_elem(symbol: str) -> Tuple:
    """Return a reasonable fallback for unknown elements."""
    z = 50
    return (z, 5, 9, 1.5, 140, 4, 100, 300, 7.0, 0.5, 1200, 7.0)


# ── Formula parser ────────────────────────────────────────────────────────────

def parse_formula(formula: str) -> Dict[str, float]:
    """
    Parse a chemical formula string into element → count mapping.
    Handles: Fe2O3, YBa2Cu3O7, (Cu2O)3, LaH10, etc.
    """
    formula = formula.strip()
    # Remove spaces and expand simple parentheses (single nesting level)
    def expand(f: str) -> str:
        pat = re.compile(r'\(([A-Za-z0-9]+)\)(\d*)')
        while '(' in f:
            def rep(m):
                inner = m.group(1)
                mult  = int(m.group(2)) if m.group(2) else 1
                # re-parse inner, multiply counts
                inner_counts = _raw_parse(inner)
                return ''.join(f'{el}{int(c*mult)}' for el, c in inner_counts.items())
            f = pat.sub(rep, f)
        return f

    return _raw_parse(expand(formula))


def _raw_parse(formula: str) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', formula)
    for el, cnt in tokens:
        if not el:
            continue
        n = float(cnt) if cnt else 1.0
        counts[el] = counts.get(el, 0.0) + n
    return counts


# ── Node feature builder ──────────────────────────────────────────────────────

def _d_orbital_occupancy(Z: int) -> float:
    if 21 <= Z <= 30: return min((Z - 20) / 10.0, 1.0)
    if 39 <= Z <= 48: return min((Z - 38) / 10.0, 1.0)
    if 72 <= Z <= 80: return min((Z - 71) / 10.0, 1.0)
    if 57 <= Z <= 71 or 89 <= Z <= 103: return 0.1
    return 0.0

def _f_occupancy(Z: int) -> float:
    if 57 <= Z <= 71: return (Z - 56) / 15.0
    if 89 <= Z <= 103: return (Z - 88) / 15.0
    return 0.0

def build_node_features(element: str, multiplicity: float = 1.0) -> List[float]:
    """
    Build 32-dimensional node feature vector for an atom.
    Must match NODE_DIM = 32 in superconductor_gnn.py.
    """
    d = ELEMENTAL_DATA.get(element, _default_elem(element))
    Z, per, grp, en, rad, val, mass, dby, fie, ea, mp, dens = d

    d_occ  = _d_orbital_occupancy(Z)
    f_occ  = _f_occupancy(Z)

    # Category flags
    is_tm  = 1.0 if (21<=Z<=30 or 39<=Z<=48 or 72<=Z<=80) else 0.0
    is_lan = 1.0 if 57<=Z<=71 else 0.0
    is_act = 1.0 if 89<=Z<=103 else 0.0
    is_alk = 1.0 if grp==1 and Z>1 else 0.0
    is_ae  = 1.0 if grp==2 else 0.0
    is_hal = 1.0 if grp==17 else 0.0
    is_chal= 1.0 if grp==16 else 0.0
    is_ng  = 1.0 if grp==18 else 0.0
    is_met = 1.0 if (is_tm or grp in (1,2,11,12,13) and Z not in (5,14,32,33,51,52) and not is_ng) else 0.0

    # 12 z-scored physical features
    feat = [
        _z(Z,    "Z"),
        _z(per,  "per"),
        _z(max(grp, 0.1), "grp"),
        _z(en,   "en"),
        _z(rad,  "rad"),
        _z(val,  "val"),
        _z(mass, "mass"),
        _z(dby,  "dby"),
        _z(fie,  "fie"),
        _z(ea,   "ea"),
        _z(mp,   "mp"),
        _z(dens, "dens"),
        # 8 category flags
        d_occ,
        f_occ,
        is_tm,
        is_lan,
        is_act,
        is_alk,
        is_ae,
        is_hal,
        # 4 more flags
        is_chal,
        is_ng,
        is_met,
        # 9 additional continuous features
        en / 4.0,                              # raw EN scaled
        val / 18.0,                            # raw valence scaled
        rad / 300.0,                           # raw radius scaled
        math.log1p(multiplicity) / 4.0,        # log multiplicity
        (Z % 18) / 18.0,                       # group position cycle
        min(dby / 1000.0, 1.0),               # Debye scaled
        (fie - ea) / 20.0,                     # charge transfer proxy
        en * en / 16.0,                        # EN squared (nonlinearity)
        min(mass / 200.0, 1.0),               # mass scaled
    ]
    assert len(feat) == NODE_DIM, f"Expected {NODE_DIM} features, got {len(feat)}"
    return feat


# ── Edge / distance builder ───────────────────────────────────────────────────

def estimate_bond_distance(
    elem_i: str, elem_j: str,
    pressure_gpa: float = 0.0,
) -> float:
    """
    Estimate interatomic distance from sum of covalent radii (pm → Å).
    Pressure compresses distances via Murnaghan EOS approximation.
    Matches pressureDistanceScale() in graph-neural-net.ts.
    """
    ri = ELEMENTAL_DATA.get(elem_i, _default_elem(elem_i))[4]
    rj = ELEMENTAL_DATA.get(elem_j, _default_elem(elem_j))[4]
    dist_pm = (ri + rj) * 0.85   # covalent radius sum × bond factor
    dist_ang = dist_pm / 100.0   # pm → Å

    if pressure_gpa > 0:
        B0 = 150.0; Bp = 4.0
        ratio = 1.0 + (Bp * pressure_gpa) / B0
        scale = ratio ** (-1.0 / Bp)
        scale = scale ** (1.0 / 3.0)   # linear compression
        dist_ang *= scale

    return max(1.0, min(dist_ang, COSINE_CUTOFF_R - 0.1))


def build_edges(
    elements: List[str],
    multiplicities: List[float],
    pressure_gpa: float = 0.0,
    cutoff_ang: float = COSINE_CUTOFF_R,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Build all edges within cutoff radius.
    Returns: (edge_pairs, distances_ang)
    Bidirectional: i→j and j→i both present.
    """
    N = len(elements)
    edge_pairs: List[Tuple[int, int]] = []
    distances:  List[float] = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = estimate_bond_distance(elements[i], elements[j], pressure_gpa)
            if d < cutoff_ang:
                edge_pairs.append((i, j))
                distances.append(d)

    # Ensure at least one edge per node (fully connected if small graph)
    if not edge_pairs and N > 1:
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = estimate_bond_distance(elements[i], elements[j], pressure_gpa)
                    edge_pairs.append((i, j))
                    distances.append(d)

    return edge_pairs, distances


# ── Global composition features ───────────────────────────────────────────────

def build_global_features(
    elem_counts:    Dict[str, float],
    physics_hints:  Optional[Dict] = None,
    structure_info: Optional[Dict] = None,
    pressure_gpa:   float = 0.0,
) -> List[float]:
    """
    Build 23-dimensional global feature vector.
    Matches buildGlobalFeatures() / the 23-dim vector in graph-neural-net.ts.

    physics_hints keys: lambda, omega_log_k, dos_at_ef, formation_energy,
                        has_lambda_measured, has_omega_measured,
                        source_tag, is_cuprate, is_iron_based
    """
    if physics_hints is None:
        physics_hints = {}

    elems = list(elem_counts.keys())
    counts = list(elem_counts.values())
    total  = sum(counts)
    if total == 0:
        return [0.0] * GLOBAL_COMP_DIM

    fracs = [c / total for c in counts]

    # Get elemental data for each element
    data_list = [ELEMENTAL_DATA.get(el, _default_elem(el)) for el in elems]

    # Weighted mean EN
    mean_en = sum(f * d[3] for f, d in zip(fracs, data_list))
    # Mean d-orbital filling
    mean_d  = sum(f * _d_orbital_occupancy(d[0]) for f, d in zip(fracs, data_list))
    # VEC (valence electron concentration) — strongest SC predictor
    vec = sum(f * d[5] for f, d in zip(fracs, data_list))
    # Mean Debye temperature
    mean_dby = sum(f * d[7] for f, d in zip(fracs, data_list))
    # Atomic size mismatch δ
    mean_rad = sum(f * d[4] for f, d in zip(fracs, data_list))
    rad_var  = sum(f * ((d[4] - mean_rad) ** 2) for f, d in zip(fracs, data_list))
    mismatch = math.sqrt(max(rad_var, 0)) / max(mean_rad, 1)
    # EN heterogeneity (std)
    en_var = sum(f * (d[3] - mean_en) ** 2 for f, d in zip(fracs, data_list))
    std_en = math.sqrt(max(en_var, 0))
    # Max pairwise EN diff
    ens = [d[3] for d in data_list]
    max_en_diff = max(abs(a - b) for a in ens for b in ens) if len(ens) > 1 else 0.0
    # Mean mass
    mean_mass = sum(f * d[6] for f, d in zip(fracs, data_list))
    # Std of radius
    std_rad = math.sqrt(max(sum(f * (d[4] - mean_rad) ** 2 for f, d in zip(fracs, data_list)), 0))
    # Hydrogen flag
    has_H = 1.0 if "H" in elem_counts else 0.0
    # TM fraction
    tm_frac = sum(fracs[i] for i, d in enumerate(data_list)
                  if 21<=d[0]<=30 or 39<=d[0]<=48 or 72<=d[0]<=80)
    # Mixing entropy (HEA indicator)
    entropy = -sum(f * math.log(max(f, 1e-10)) for f in fracs) / math.log(max(len(fracs), 2))
    # n_elements
    n_el = len(elems)
    # Cuprate flag: has Cu and O
    is_cuprate  = 1.0 if "Cu" in elem_counts and "O" in elem_counts else 0.0
    # Iron-based SC flag: Fe + pnictogen/chalcogen
    pnictogen_chalcogen = {"As", "P", "Se", "Te", "S"}
    is_iron     = 1.0 if "Fe" in elem_counts and any(p in elem_counts for p in pnictogen_chalcogen) else 0.0
    # MgB2-type: Mg:B ≈ 1:2
    is_mgb2 = 0.0
    if "Mg" in elem_counts and "B" in elem_counts:
        ratio = elem_counts["B"] / max(elem_counts["Mg"], 1e-6)
        is_mgb2 = 1.0 if 1.5 <= ratio <= 2.5 else 0.0
    # Heavy fermion hint: Ce, U, Yb, Pr, Sm
    hf_set = {"Ce", "U", "Yb", "Pr", "Sm"}
    is_hf = 1.0 if any(e in elem_counts for e in hf_set) else 0.0
    # Mean d-electron count per TM atom
    tm_els = [d for d in data_list if 21<=d[0]<=30 or 39<=d[0]<=48 or 72<=d[0]<=80]
    mean_d_count = (sum(d[5] for d in tm_els) / len(tm_els)) if tm_els else 0.0

    # Physics hints (with normalization matching TS)
    lam      = physics_hints.get("lambda", None)
    omega    = physics_hints.get("omega_log_k", None)
    dos      = physics_hints.get("dos_at_ef", None)
    fe       = physics_hints.get("formation_energy", None)

    lam_feat   = min(lam / 3.0, 1.0) if lam is not None else 0.0
    omega_feat = (math.log(max(omega, 1)) / math.log(2000.0)) if omega is not None else 0.0
    dos_feat   = min(dos / 5.0, 1.0) if dos is not None else 0.0
    fe_feat    = max(-1.0, min(1.0, fe / 5.0)) if fe is not None else 0.0

    # Pressure feature (log-normalized)
    p_feat = math.log1p(pressure_gpa) / math.log1p(300.0) if pressure_gpa > 0 else 0.0

    feats = [
        _z(mean_en, "en"),            # [0]  mean EN
        mean_d,                        # [1]  mean d-orbital filling
        _z(vec, "vec"),               # [2]  VEC (strongest SC predictor)
        _z(mean_dby, "dby"),          # [3]  mean Debye temp
        _z(mismatch, "mis"),          # [4]  atomic size mismatch δ
        _z(n_el, "nel"),              # [5]  n_elements
        _z(std_en, "std_en"),         # [6]  EN std (cuprate/hydride signal)
        has_H,                         # [7]  hydrogen flag
        tm_frac,                       # [8]  TM fraction
        entropy,                       # [9]  mixing entropy (HEA)
        _z(max_en_diff, "max_en"),    # [10] max EN diff
        _z(mean_mass, "mmass"),       # [11] mean atomic mass
        _z(std_rad, "srad"),          # [12] std atomic radius
        lam_feat,                      # [13] λ hint (0 if unknown)
        omega_feat,                    # [14] log ω_log hint
        dos_feat,                      # [15] DOS hint
        fe_feat,                       # [16] formation energy hint
        is_cuprate,                    # [17] cuprate flag
        is_iron,                       # [18] iron-based flag
        p_feat,                        # [19] pressure (log)
        is_mgb2,                       # [20] MgB2-type flag
        is_hf,                         # [21] heavy-fermion flag
        min(mean_d_count / 10.0, 1.0),# [22] mean d-electron count per TM
    ]
    assert len(feats) == GLOBAL_COMP_DIM
    return feats


# ── Three-body feature builder ────────────────────────────────────────────────

def build_three_body(
    elements:    List[str],
    edge_pairs:  List[Tuple[int, int]],
    distances:   List[float],
    max_triples: int = 200,
) -> Tuple[List[Tuple[int,int,int]], List[float], List[float]]:
    """
    Build three-body (angle) features for (center, nb1, nb2) triples.
    Returns: (index_triples, d1_list, d2_list)
    Uses law of cosines; distances between neighbours estimated from element pairs.
    """
    # Build adjacency: center → [(neighbor, edge_idx), ...]
    adj: Dict[int, List[Tuple[int,int]]] = {}
    for eidx, (i, j) in enumerate(edge_pairs):
        adj.setdefault(i, []).append((j, eidx))

    triples: List[Tuple[int,int,int]] = []
    d1_list: List[float] = []
    d2_list: List[float] = []

    for center, neighbors in adj.items():
        for a_idx, (nb1, eidx1) in enumerate(neighbors):
            for nb2, eidx2 in neighbors[a_idx+1:]:
                if nb1 == nb2:
                    continue
                triples.append((center, nb1, nb2))
                d1_list.append(distances[eidx1])
                d2_list.append(distances[eidx2])
                if len(triples) >= max_triples:
                    break
            if len(triples) >= max_triples:
                break
        if len(triples) >= max_triples:
            break

    return triples, d1_list, d2_list


# ── Main graph builder ────────────────────────────────────────────────────────

MAX_NODES_PER_ELEMENT = 8

def build_crystal_graph(
    formula:        str,
    structure:      Optional[Dict]  = None,
    physics_hints:  Optional[Dict]  = None,
    pressure_gpa:   float = 0.0,
    build_3body:    bool  = True,
) -> SuperconGraph:
    """
    Convert a formula string → SuperconGraph ready for the PyTorch GNN.

    structure:     dict with optional keys: spaceGroup, crystalSystem
    physics_hints: dict with optional keys: lambda, omega_log_k, dos_at_ef,
                   formation_energy, source_tag, etc.
    """
    raw_counts = parse_formula(formula)
    if not raw_counts:
        raise ValueError(f"Could not parse formula: {formula!r}")

    # Reduce counts to reasonable node counts (mirrors normalizeFormulaCounts)
    rounded: Dict[str, int] = {el: max(1, round(c)) for el, c in raw_counts.items()}
    # GCD reduction
    from math import gcd
    from functools import reduce
    g = reduce(gcd, rounded.values())
    reduced = {el: max(1, v // g) for el, v in rounded.items()}

    elements: List[str] = []
    multiplicities: List[float] = []
    atom_z_list: List[int] = []

    for el, count in reduced.items():
        n_nodes = min(count, MAX_NODES_PER_ELEMENT)
        mult    = count / n_nodes   # each node represents mult atoms
        for _ in range(n_nodes):
            elements.append(el)
            multiplicities.append(mult)
            atom_z_list.append(ELEMENTAL_DATA.get(el, _default_elem(el))[0])

    N = len(elements)
    if N == 0:
        raise ValueError(f"Empty graph for formula: {formula!r}")

    # Build edges
    edge_pairs, distances = build_edges(elements, multiplicities, pressure_gpa)
    E = len(edge_pairs)

    # Node features
    node_feat_list = [
        build_node_features(elements[i], multiplicities[i])
        for i in range(N)
    ]
    x = torch.tensor(node_feat_list, dtype=torch.float32)   # [N, NODE_DIM]

    # Edge tensors
    if E > 0:
        edge_index = torch.tensor(
            [[p[0] for p in edge_pairs],
             [p[1] for p in edge_pairs]], dtype=torch.long
        )   # [2, E]
        dist_t = torch.tensor(distances, dtype=torch.float32)
        edge_attr = build_rbf_features(dist_t)   # [E, N_GAUSSIAN_BASIS]
    else:
        # Self-loops as fallback for single-atom graphs
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        dist_t = torch.tensor([2.5], dtype=torch.float32)
        edge_attr = build_rbf_features(dist_t)

    nm  = torch.tensor(multiplicities, dtype=torch.float32)  # [N]
    az  = torch.tensor(atom_z_list,    dtype=torch.long)     # [N]

    # Global features
    gf_raw = build_global_features(raw_counts, physics_hints, structure, pressure_gpa)
    gf     = torch.tensor(gf_raw, dtype=torch.float32)       # [GLOBAL_COMP_DIM]

    # Three-body features (optional)
    tb_index = tb_d1 = tb_d2 = None
    if build_3body and E > 0:
        triples, d1s, d2s = build_three_body(elements, edge_pairs, distances)
        if triples:
            tb_index = torch.tensor(triples, dtype=torch.long)
            tb_d1    = torch.tensor(d1s, dtype=torch.float32)
            tb_d2    = torch.tensor(d2s, dtype=torch.float32)

    return SuperconGraph(
        node_features    = x,
        edge_index       = edge_index,
        edge_attr        = edge_attr,
        node_mult        = nm,
        global_features  = gf,
        atom_z           = az,
        three_body_index = tb_index,
        three_body_angle = None,   # computed in ThreeBodyLayer from d1/d2
        three_body_d1    = tb_d1,
        three_body_d2    = tb_d2,
        pressure_gpa     = pressure_gpa,
        formula          = formula,
    )
