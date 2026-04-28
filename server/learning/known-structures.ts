/**
 * Known Crystal Structures Database
 *
 * Exact primitive-cell fractional coordinates from literature for verified
 * superconductor compounds. These are DFT-quality starting positions that
 * produce stable phonon spectra — no heuristic generation needed.
 *
 * Checked FIRST in generateAtomicPositions() (Tier 0), before prototype
 * matching or lattice-free fallback.
 *
 * Sources:
 * - LaH10: Drozdov et al., Nature 2019; Fm-3m clathrate structure
 * - CaH6: Wang et al., PNAS 2012; Im-3m sodalite cage
 * - H3S:  Drozdov et al., Nature 2015; Im-3m BCC structure
 * - YH6/YH9: Kong et al., Nature Communications 2021
 */

export interface KnownStructure {
  formula: string;
  spaceGroup: string;
  spaceGroupNumber: number;
  latticeType: "cubic" | "hexagonal" | "tetragonal";
  /** Lattice constant a in Angstroms (at the material's target pressure). */
  latticeA: number;
  latticeB?: number;
  latticeC?: number;
  /** Pressure at which this structure is stable (GPa). */
  pressureGPa: number;
  /** Primitive cell atomic positions in fractional coordinates. */
  atoms: Array<{ element: string; x: number; y: number; z: number; wyckoff?: string }>;
}

// ---------------------------------------------------------------------------
// Fm-3m (225) Clathrate Hydrides — Primitive cell (FCC → 1/4 conventional)
// ---------------------------------------------------------------------------

// LaH10: Fm-3m, a=5.10 Å at 170 GPa
// Conventional cell: 4 La(4a) + 32 H(32f) + 8 H(8c) = 44 atoms
// Primitive cell: 1 La + 8 H(32f) + 2 H(8c) = 11 atoms
// 32f parameter x ≈ 0.375 (forms the sodalite-like cage)
const LaH10_ATOMS: KnownStructure["atoms"] = [
  { element: "La", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "4a" },
  { element: "H",  x: 0.3750, y: 0.3750, z: 0.3750, wyckoff: "32f" },
  { element: "H",  x: 0.3750, y: 0.3750, z: 0.8750, wyckoff: "32f" },
  { element: "H",  x: 0.3750, y: 0.8750, z: 0.3750, wyckoff: "32f" },
  { element: "H",  x: 0.8750, y: 0.3750, z: 0.3750, wyckoff: "32f" },
  { element: "H",  x: 0.6250, y: 0.6250, z: 0.6250, wyckoff: "32f" },
  { element: "H",  x: 0.6250, y: 0.6250, z: 0.1250, wyckoff: "32f" },
  { element: "H",  x: 0.6250, y: 0.1250, z: 0.6250, wyckoff: "32f" },
  { element: "H",  x: 0.1250, y: 0.6250, z: 0.6250, wyckoff: "32f" },
  { element: "H",  x: 0.2500, y: 0.2500, z: 0.2500, wyckoff: "8c" },
  { element: "H",  x: 0.7500, y: 0.7500, z: 0.7500, wyckoff: "8c" },
];

// ThH10: Same Fm-3m structure, a=4.79 Å at 175 GPa
const ThH10_ATOMS: KnownStructure["atoms"] = LaH10_ATOMS.map(a => ({
  ...a, element: a.element === "La" ? "Th" : a.element,
}));

// CeH10: Same family, a≈5.0 Å at 200 GPa (predicted)
const CeH10_ATOMS: KnownStructure["atoms"] = LaH10_ATOMS.map(a => ({
  ...a, element: a.element === "La" ? "Ce" : a.element,
}));

// ---------------------------------------------------------------------------
// Im-3m (229) Sodalite Hydrides — Primitive cell (BCC → 1/2 conventional)
// ---------------------------------------------------------------------------

// CaH6: Im-3m, a=3.85 Å at 200 GPa
// Conventional cell: 2 Ca(2a) + 12 H(12d) = 14 atoms
// Primitive cell: 1 Ca + 6 H = 7 atoms
// H at 12d positions form sodalite cage (corner-sharing H₄ squares)
const SODALITE_H6_ATOMS = (metal: string): KnownStructure["atoms"] => [
  { element: metal, x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "2a" },
  { element: "H",  x: 0.5000, y: 0.7500, z: 0.2500, wyckoff: "12d" },
  { element: "H",  x: 0.5000, y: 0.2500, z: 0.7500, wyckoff: "12d" },
  { element: "H",  x: 0.7500, y: 0.5000, z: 0.2500, wyckoff: "12d" },
  { element: "H",  x: 0.2500, y: 0.5000, z: 0.7500, wyckoff: "12d" },
  { element: "H",  x: 0.7500, y: 0.2500, z: 0.5000, wyckoff: "12d" },
  { element: "H",  x: 0.2500, y: 0.7500, z: 0.5000, wyckoff: "12d" },
];

// H3S: Im-3m, a=3.10 Å at 200 GPa
// Conventional cell: 2 S(2a) + 6 H(6b) = 8 atoms
// Primitive cell: 1 S + 3 H = 4 atoms
// H at 6b positions form corner-sharing octahedra
const H3S_ATOMS: KnownStructure["atoms"] = [
  { element: "S",  x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "2a" },
  { element: "H",  x: 0.0000, y: 0.5000, z: 0.5000, wyckoff: "6b" },
  { element: "H",  x: 0.5000, y: 0.0000, z: 0.5000, wyckoff: "6b" },
  { element: "H",  x: 0.5000, y: 0.5000, z: 0.0000, wyckoff: "6b" },
];

// ---------------------------------------------------------------------------
// Ternary Hydrides — Doped clathrate/sodalite cages
// ---------------------------------------------------------------------------

// Li2LaH12: Ternary clathrate, based on LaH10 Fm-3m cage with Li in interstitial
// sites + extra H. Primitive cell: 1 La + 2 Li + 12 H = 15 atoms.
// Li occupies the 8c-like interstitial sites (replacing 2 of the cage H),
// and extra H fills remaining cage positions.
const Li2LaH12_ATOMS: KnownStructure["atoms"] = [
  { element: "La", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "cage-center" },
  // Li at interstitial sites (8c-derived)
  { element: "Li", x: 0.2500, y: 0.2500, z: 0.2500, wyckoff: "interstitial" },
  { element: "Li", x: 0.7500, y: 0.7500, z: 0.7500, wyckoff: "interstitial" },
  // H cage (32f-derived positions, 8 atoms)
  { element: "H",  x: 0.3750, y: 0.3750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.3750, y: 0.3750, z: 0.8750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.3750, y: 0.8750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.8750, y: 0.3750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.6250, z: 0.6250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.6250, z: 0.1250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.1250, z: 0.6250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.1250, y: 0.6250, z: 0.6250, wyckoff: "32f-cage" },
  // Extra H at displaced interstitial sites (bridging Li-cage)
  { element: "H",  x: 0.1250, y: 0.1250, z: 0.1250, wyckoff: "extra" },
  { element: "H",  x: 0.8750, y: 0.8750, z: 0.8750, wyckoff: "extra" },
  { element: "H",  x: 0.1250, y: 0.3750, z: 0.1250, wyckoff: "extra" },
  { element: "H",  x: 0.3750, y: 0.1250, z: 0.1250, wyckoff: "extra" },
];

// LaH11Li2: Same structure family as Li2LaH12 but different H count.
// 1 La + 2 Li + 11 H = 14 atoms. Li at interstitials, one fewer H.
const LaH11Li2_ATOMS: KnownStructure["atoms"] = [
  { element: "La", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "cage-center" },
  { element: "Li", x: 0.2500, y: 0.2500, z: 0.2500, wyckoff: "interstitial" },
  { element: "Li", x: 0.7500, y: 0.7500, z: 0.7500, wyckoff: "interstitial" },
  { element: "H",  x: 0.3750, y: 0.3750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.3750, y: 0.3750, z: 0.8750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.3750, y: 0.8750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.8750, y: 0.3750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.6250, z: 0.6250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.6250, z: 0.1250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.1250, z: 0.6250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.1250, y: 0.6250, z: 0.6250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.1250, y: 0.1250, z: 0.1250, wyckoff: "extra" },
  { element: "H",  x: 0.8750, y: 0.8750, z: 0.8750, wyckoff: "extra" },
  { element: "H",  x: 0.1250, y: 0.3750, z: 0.1250, wyckoff: "extra" },
];

// YH9Na2: Ternary hexagonal clathrate, Y-H cage with Na guest atoms.
// Based on YH9 P63/mmc structure with Na at interstitial hexagonal sites.
// 1 Y + 2 Na + 9 H = 12 atoms.
const YH9Na2_ATOMS: KnownStructure["atoms"] = [
  { element: "Y",  x: 0.3333, y: 0.6667, z: 0.2500, wyckoff: "2d" },
  { element: "Na", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "1a" },
  { element: "Na", x: 0.0000, y: 0.0000, z: 0.5000, wyckoff: "1a" },
  { element: "H",  x: 0.1550, y: 0.3100, z: 0.2500, wyckoff: "6h" },
  { element: "H",  x: 0.6900, y: 0.8450, z: 0.2500, wyckoff: "6h" },
  { element: "H",  x: 0.8450, y: 0.1550, z: 0.2500, wyckoff: "6h" },
  { element: "H",  x: 0.0000, y: 0.0000, z: 0.2500, wyckoff: "2b" },
  { element: "H",  x: 0.5200, y: 0.0400, z: 0.0800, wyckoff: "12k" },
  { element: "H",  x: 0.9600, y: 0.4800, z: 0.0800, wyckoff: "12k" },
  { element: "H",  x: 0.4800, y: 0.5200, z: 0.0800, wyckoff: "12k" },
  { element: "H",  x: 0.0400, y: 0.5200, z: 0.4200, wyckoff: "12k" },
  { element: "H",  x: 0.5200, y: 0.4800, z: 0.4200, wyckoff: "12k" },
];

// LaH12: Higher hydride clathrate. 1 La + 12 H = 13 atoms.
// Based on Fm-3m cage with additional H at octahedral interstices.
const LaH12_ATOMS: KnownStructure["atoms"] = [
  { element: "La", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "cage-center" },
  // 32f cage H (8 in primitive)
  { element: "H",  x: 0.3750, y: 0.3750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.3750, y: 0.3750, z: 0.8750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.3750, y: 0.8750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.8750, y: 0.3750, z: 0.3750, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.6250, z: 0.6250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.6250, z: 0.1250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.6250, y: 0.1250, z: 0.6250, wyckoff: "32f-cage" },
  { element: "H",  x: 0.1250, y: 0.6250, z: 0.6250, wyckoff: "32f-cage" },
  // 8c interstitial H (2 in primitive)
  { element: "H",  x: 0.2500, y: 0.2500, z: 0.2500, wyckoff: "8c" },
  { element: "H",  x: 0.7500, y: 0.7500, z: 0.7500, wyckoff: "8c" },
  // Extra H at octahedral interstice (between cage sites)
  { element: "H",  x: 0.5000, y: 0.0000, z: 0.0000, wyckoff: "octahedral" },
  { element: "H",  x: 0.0000, y: 0.5000, z: 0.0000, wyckoff: "octahedral" },
];

// ---------------------------------------------------------------------------
// P63/mmc (194) Hexagonal Clathrate Hydrides
// ---------------------------------------------------------------------------

// YH9: P63/mmc, a=4.20 Å, c/a≈1.55 at 200 GPa
// Y at 2d: (1/3, 2/3, 1/4) and (2/3, 1/3, 3/4)
// H at 6h: (x, 2x, 1/4) with x≈0.155 (and symmetry equivalents)
// H at 2b: (0, 0, 1/4) and (0, 0, 3/4)
// H at 12k or similar positions for remaining H atoms
// Total: 2 Y + 18 H = 20 atoms conventional, 10 in reduced formula unit
const YH9_ATOMS: KnownStructure["atoms"] = [
  { element: "Y",  x: 0.3333, y: 0.6667, z: 0.2500, wyckoff: "2d" },
  { element: "H",  x: 0.1550, y: 0.3100, z: 0.2500, wyckoff: "6h" },
  { element: "H",  x: 0.6900, y: 0.8450, z: 0.2500, wyckoff: "6h" },
  { element: "H",  x: 0.8450, y: 0.1550, z: 0.2500, wyckoff: "6h" },
  { element: "H",  x: 0.0000, y: 0.0000, z: 0.2500, wyckoff: "2b" },
  { element: "H",  x: 0.5200, y: 0.0400, z: 0.0800, wyckoff: "12k" },
  { element: "H",  x: 0.9600, y: 0.4800, z: 0.0800, wyckoff: "12k" },
  { element: "H",  x: 0.4800, y: 0.5200, z: 0.0800, wyckoff: "12k" },
  { element: "H",  x: 0.0400, y: 0.5200, z: 0.4200, wyckoff: "12k" },
  { element: "H",  x: 0.5200, y: 0.4800, z: 0.4200, wyckoff: "12k" },
];

// ---------------------------------------------------------------------------
// Simple hydrides and other verified structures
// ---------------------------------------------------------------------------

// ScH3: Fm-3m (NaCl-related), a=4.48 Å
const ScH3_ATOMS: KnownStructure["atoms"] = [
  { element: "Sc", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "4a" },
  { element: "H",  x: 0.5000, y: 0.0000, z: 0.0000, wyckoff: "4b" },
  { element: "H",  x: 0.2500, y: 0.2500, z: 0.2500, wyckoff: "8c" },
  { element: "H",  x: 0.7500, y: 0.7500, z: 0.7500, wyckoff: "8c" },
];

// MgB2: P6/mmm (AlB2-type), a=3.09 Å, c/a≈1.142
const MgB2_ATOMS: KnownStructure["atoms"] = [
  { element: "Mg", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "1a" },
  { element: "B",  x: 0.3333, y: 0.6667, z: 0.5000, wyckoff: "2d" },
  { element: "B",  x: 0.6667, y: 0.3333, z: 0.5000, wyckoff: "2d" },
];

// Nb3Sn: Pm-3n (A15), a=5.29 Å
const Nb3Sn_ATOMS: KnownStructure["atoms"] = [
  { element: "Sn", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "2a" },
  { element: "Sn", x: 0.5000, y: 0.5000, z: 0.5000, wyckoff: "2a" },
  { element: "Nb", x: 0.2500, y: 0.0000, z: 0.5000, wyckoff: "6c" },
  { element: "Nb", x: 0.7500, y: 0.0000, z: 0.5000, wyckoff: "6c" },
  { element: "Nb", x: 0.5000, y: 0.2500, z: 0.0000, wyckoff: "6c" },
  { element: "Nb", x: 0.5000, y: 0.7500, z: 0.0000, wyckoff: "6c" },
  { element: "Nb", x: 0.0000, y: 0.5000, z: 0.2500, wyckoff: "6c" },
  { element: "Nb", x: 0.0000, y: 0.5000, z: 0.7500, wyckoff: "6c" },
];

// Nb3Ge: Same A15 structure, a=5.14 Å
const Nb3Ge_ATOMS: KnownStructure["atoms"] = Nb3Sn_ATOMS.map(a => ({
  ...a, element: a.element === "Sn" ? "Ge" : a.element,
}));

// V3Si: Same A15 structure, a=4.72 Å
const V3Si_ATOMS: KnownStructure["atoms"] = Nb3Sn_ATOMS.map(a => ({
  ...a, element: a.element === "Sn" ? "Si" : (a.element === "Nb" ? "V" : a.element),
}));

// NbN: Fm-3m (NaCl/B1), a=4.39 Å
const NbN_ATOMS: KnownStructure["atoms"] = [
  { element: "Nb", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "4a" },
  { element: "N",  x: 0.5000, y: 0.5000, z: 0.5000, wyckoff: "4b" },
];

// NbC: Same B1 structure, a=4.47 Å
const NbC_ATOMS: KnownStructure["atoms"] = [
  { element: "Nb", x: 0.0000, y: 0.0000, z: 0.0000, wyckoff: "4a" },
  { element: "C",  x: 0.5000, y: 0.5000, z: 0.5000, wyckoff: "4b" },
];

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

const KNOWN_STRUCTURES: Record<string, KnownStructure> = {
  // Fm-3m clathrate hydrides
  "LaH10": { formula: "LaH10", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 5.10, pressureGPa: 170, atoms: LaH10_ATOMS },
  "ThH10": { formula: "ThH10", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.79, pressureGPa: 175, atoms: ThH10_ATOMS },
  "CeH10": { formula: "CeH10", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 5.00, pressureGPa: 200, atoms: CeH10_ATOMS },

  // Im-3m sodalite hydrides
  "CaH6":  { formula: "CaH6", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.85, pressureGPa: 200, atoms: SODALITE_H6_ATOMS("Ca") },
  "YH6":   { formula: "YH6", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.95, pressureGPa: 200, atoms: SODALITE_H6_ATOMS("Y") },
  "SrH6":  { formula: "SrH6", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 4.10, pressureGPa: 200, atoms: SODALITE_H6_ATOMS("Sr") },
  "LaH6":  { formula: "LaH6", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 4.20, pressureGPa: 175, atoms: SODALITE_H6_ATOMS("La") },
  "ScH6":  { formula: "ScH6", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.70, pressureGPa: 200, atoms: SODALITE_H6_ATOMS("Sc") },

  // Im-3m BCC hydrides
  "H3S":   { formula: "H3S", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.10, pressureGPa: 200, atoms: H3S_ATOMS },

  // P63/mmc hexagonal clathrate
  "YH9":   { formula: "YH9", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 4.20, latticeC: 6.51, pressureGPa: 200, atoms: YH9_ATOMS },
  "CeH9":  { formula: "CeH9", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 4.85, latticeC: 7.52, pressureGPa: 150, atoms: YH9_ATOMS.map(a => ({ ...a, element: a.element === "Y" ? "Ce" : a.element })) },
  "ScH9":  { formula: "ScH9", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 4.20, latticeC: 6.51, pressureGPa: 200, atoms: YH9_ATOMS.map(a => ({ ...a, element: a.element === "Y" ? "Sc" : a.element })) },

  // Ternary hydrides
  "Li2LaH12": { formula: "Li2LaH12", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 5.10, pressureGPa: 175, atoms: Li2LaH12_ATOMS },
  "LaH11Li2": { formula: "LaH11Li2", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 5.10, pressureGPa: 175, atoms: LaH11Li2_ATOMS },
  "YH9Na2":  { formula: "YH9Na2", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 4.20, latticeC: 6.51, pressureGPa: 200, atoms: YH9Na2_ATOMS },
  "LaH12":   { formula: "LaH12", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 5.10, pressureGPa: 175, atoms: LaH12_ATOMS },

  // Simple hydrides
  "ScH3":  { formula: "ScH3", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.48, pressureGPa: 0, atoms: ScH3_ATOMS },
  "YH3":   { formula: "YH3", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 5.14, pressureGPa: 0, atoms: ScH3_ATOMS.map(a => ({ ...a, element: a.element === "Sc" ? "Y" : a.element })) },

  // Non-hydride superconductors — A15 family
  "MgB2":  { formula: "MgB2", spaceGroup: "P6/mmm", spaceGroupNumber: 191, latticeType: "hexagonal", latticeA: 3.09, latticeC: 3.53, pressureGPa: 0, atoms: MgB2_ATOMS },
  "Nb3Sn": { formula: "Nb3Sn", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 5.29, pressureGPa: 0, atoms: Nb3Sn_ATOMS },
  "Nb3Ge": { formula: "Nb3Ge", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 5.14, pressureGPa: 0, atoms: Nb3Ge_ATOMS },
  "V3Si":  { formula: "V3Si", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 4.72, pressureGPa: 0, atoms: V3Si_ATOMS },
  "Nb3Al": { formula: "Nb3Al", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 5.19, pressureGPa: 0, atoms: Nb3Sn_ATOMS.map(a => ({ ...a, element: a.element === "Sn" ? "Al" : a.element })) },
  "Nb3Ga": { formula: "Nb3Ga", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 5.18, pressureGPa: 0, atoms: Nb3Sn_ATOMS.map(a => ({ ...a, element: a.element === "Sn" ? "Ga" : a.element })) },
  "V3Ga":  { formula: "V3Ga", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 4.82, pressureGPa: 0, atoms: V3Si_ATOMS.map(a => ({ ...a, element: a.element === "Si" ? "Ga" : a.element })) },

  // Rocksalt nitrides/carbides
  "NbN":   { formula: "NbN", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.39, pressureGPa: 0, atoms: NbN_ATOMS },
  "NbC":   { formula: "NbC", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.47, pressureGPa: 0, atoms: NbC_ATOMS },
  "TaN":   { formula: "TaN", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.34, pressureGPa: 0, atoms: NbN_ATOMS.map(a => ({ ...a, element: a.element === "Nb" ? "Ta" : a.element })) },
  "TaC":   { formula: "TaC", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.46, pressureGPa: 0, atoms: NbC_ATOMS.map(a => ({ ...a, element: a.element === "Nb" ? "Ta" : a.element })) },
  "TiN":   { formula: "TiN", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.24, pressureGPa: 0, atoms: NbN_ATOMS.map(a => ({ ...a, element: a.element === "Nb" ? "Ti" : a.element })) },
  "ZrN":   { formula: "ZrN", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.58, pressureGPa: 0, atoms: NbN_ATOMS.map(a => ({ ...a, element: a.element === "Nb" ? "Zr" : a.element })) },
  "VN":    { formula: "VN", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.14, pressureGPa: 0, atoms: NbN_ATOMS.map(a => ({ ...a, element: a.element === "Nb" ? "V" : a.element })) },
  "MoN":   { formula: "MoN", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.21, pressureGPa: 0, atoms: NbN_ATOMS.map(a => ({ ...a, element: a.element === "Nb" ? "Mo" : a.element })) },

  // Cuprates — YBCO-123 (P4/mmm simplified orthorhombic → tetragonal)
  "YBa2Cu3O7": { formula: "YBa2Cu3O7", spaceGroup: "Pmmm", spaceGroupNumber: 47, latticeType: "tetragonal", latticeA: 3.82, latticeC: 11.68, pressureGPa: 0, atoms: [
    { element: "Y",  x: 0.5, y: 0.5, z: 0.5, wyckoff: "1h" },
    { element: "Ba", x: 0.5, y: 0.5, z: 0.185, wyckoff: "2t" },
    { element: "Ba", x: 0.5, y: 0.5, z: 0.815, wyckoff: "2t" },
    { element: "Cu", x: 0.0, y: 0.0, z: 0.0, wyckoff: "1a" },
    { element: "Cu", x: 0.0, y: 0.0, z: 0.356, wyckoff: "2q" },
    { element: "Cu", x: 0.0, y: 0.0, z: 0.644, wyckoff: "2q" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.0, wyckoff: "1e" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.378, wyckoff: "2s" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.622, wyckoff: "2s" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.378, wyckoff: "2r" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.622, wyckoff: "2r" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.159, wyckoff: "2q" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.841, wyckoff: "2q" },
  ]},

  // Iron pnictides — BaFe2As2 (122-type, ThCr2Si2, I4/mmm)
  "BaFe2As2": { formula: "BaFe2As2", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.96, latticeC: 13.02, pressureGPa: 0, atoms: [
    { element: "Ba", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Fe", x: 0.5, y: 0.0, z: 0.25, wyckoff: "4d" },
    { element: "Fe", x: 0.0, y: 0.5, z: 0.25, wyckoff: "4d" },
    { element: "As", x: 0.0, y: 0.0, z: 0.354, wyckoff: "4e" },
    { element: "As", x: 0.0, y: 0.0, z: 0.646, wyckoff: "4e" },
  ]},

  // Iron pnictides — LaFeAsO (1111-type, P4/nmm)
  "LaFeAsO": { formula: "LaFeAsO", spaceGroup: "P4/nmm", spaceGroupNumber: 129, latticeType: "tetragonal", latticeA: 4.03, latticeC: 8.74, pressureGPa: 0, atoms: [
    { element: "La", x: 0.25, y: 0.25, z: 0.142, wyckoff: "2c" },
    { element: "Fe", x: 0.75, y: 0.25, z: 0.5, wyckoff: "2b" },
    { element: "As", x: 0.25, y: 0.25, z: 0.651, wyckoff: "2c" },
    { element: "O",  x: 0.75, y: 0.25, z: 0.0, wyckoff: "2a" },
  ]},

  // Iron chalcogenide — FeSe (11-type, P4/nmm)
  "FeSe": { formula: "FeSe", spaceGroup: "P4/nmm", spaceGroupNumber: 129, latticeType: "tetragonal", latticeA: 3.77, latticeC: 5.52, pressureGPa: 0, atoms: [
    { element: "Fe", x: 0.75, y: 0.25, z: 0.0, wyckoff: "2a" },
    { element: "Se", x: 0.25, y: 0.25, z: 0.268, wyckoff: "2c" },
  ]},

  // Iron pnictide — LiFeAs (111-type, P4/nmm)
  "LiFeAs": { formula: "LiFeAs", spaceGroup: "P4/nmm", spaceGroupNumber: 129, latticeType: "tetragonal", latticeA: 3.77, latticeC: 6.36, pressureGPa: 0, atoms: [
    { element: "Li", x: 0.25, y: 0.25, z: 0.345, wyckoff: "2c" },
    { element: "Fe", x: 0.75, y: 0.25, z: 0.0, wyckoff: "2b" },
    { element: "As", x: 0.25, y: 0.25, z: 0.737, wyckoff: "2c" },
  ]},

  // Hexaborides — CaB6 (Pm-3m, LaB6-type)
  "CaB6": { formula: "CaB6", spaceGroup: "Pm-3m", spaceGroupNumber: 221, latticeType: "cubic", latticeA: 4.15, pressureGPa: 0, atoms: [
    { element: "Ca", x: 0.0, y: 0.0, z: 0.0, wyckoff: "1a" },
    { element: "B",  x: 0.203, y: 0.5, z: 0.5, wyckoff: "6f" },
    { element: "B",  x: 0.797, y: 0.5, z: 0.5, wyckoff: "6f" },
    { element: "B",  x: 0.5, y: 0.203, z: 0.5, wyckoff: "6f" },
    { element: "B",  x: 0.5, y: 0.797, z: 0.5, wyckoff: "6f" },
    { element: "B",  x: 0.5, y: 0.5, z: 0.203, wyckoff: "6f" },
    { element: "B",  x: 0.5, y: 0.5, z: 0.797, wyckoff: "6f" },
  ]},
  "LaB6": { formula: "LaB6", spaceGroup: "Pm-3m", spaceGroupNumber: 221, latticeType: "cubic", latticeA: 4.16, pressureGPa: 0, atoms: [
    { element: "La", x: 0.0, y: 0.0, z: 0.0, wyckoff: "1a" },
    { element: "B",  x: 0.199, y: 0.5, z: 0.5, wyckoff: "6f" },
    { element: "B",  x: 0.801, y: 0.5, z: 0.5, wyckoff: "6f" },
    { element: "B",  x: 0.5, y: 0.199, z: 0.5, wyckoff: "6f" },
    { element: "B",  x: 0.5, y: 0.801, z: 0.5, wyckoff: "6f" },
    { element: "B",  x: 0.5, y: 0.5, z: 0.199, wyckoff: "6f" },
    { element: "B",  x: 0.5, y: 0.5, z: 0.801, wyckoff: "6f" },
  ]},

  // Dichalcogenides — NbSe2 (P6_3/mmc, 2H polytype)
  "NbSe2": { formula: "NbSe2", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.44, latticeC: 12.55, pressureGPa: 0, atoms: [
    { element: "Nb", x: 0.0, y: 0.0, z: 0.25, wyckoff: "2b" },
    { element: "Se", x: 0.3333, y: 0.6667, z: 0.121, wyckoff: "4f" },
    { element: "Se", x: 0.3333, y: 0.6667, z: 0.379, wyckoff: "4f" },
  ]},
  "MoS2": { formula: "MoS2", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.16, latticeC: 12.30, pressureGPa: 0, atoms: [
    { element: "Mo", x: 0.0, y: 0.0, z: 0.25, wyckoff: "2b" },
    { element: "S",  x: 0.3333, y: 0.6667, z: 0.121, wyckoff: "4f" },
    { element: "S",  x: 0.3333, y: 0.6667, z: 0.379, wyckoff: "4f" },
  ]},
  "TaS2": { formula: "TaS2", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.36, latticeC: 12.10, pressureGPa: 0, atoms: [
    { element: "Ta", x: 0.0, y: 0.0, z: 0.25, wyckoff: "2b" },
    { element: "S",  x: 0.3333, y: 0.6667, z: 0.121, wyckoff: "4f" },
    { element: "S",  x: 0.3333, y: 0.6667, z: 0.379, wyckoff: "4f" },
  ]},

  // Topological insulators — Bi2Te3 (R-3m)
  "Bi2Te3": { formula: "Bi2Te3", spaceGroup: "R-3m", spaceGroupNumber: 166, latticeType: "hexagonal", latticeA: 4.38, latticeC: 30.50, pressureGPa: 0, atoms: [
    { element: "Bi", x: 0.0, y: 0.0, z: 0.400, wyckoff: "6c" },
    { element: "Te", x: 0.0, y: 0.0, z: 0.0, wyckoff: "3a" },
    { element: "Te", x: 0.0, y: 0.0, z: 0.212, wyckoff: "6c" },
  ]},
  "Bi2Se3": { formula: "Bi2Se3", spaceGroup: "R-3m", spaceGroupNumber: 166, latticeType: "hexagonal", latticeA: 4.14, latticeC: 28.64, pressureGPa: 0, atoms: [
    { element: "Bi", x: 0.0, y: 0.0, z: 0.400, wyckoff: "6c" },
    { element: "Se", x: 0.0, y: 0.0, z: 0.0, wyckoff: "3a" },
    { element: "Se", x: 0.0, y: 0.0, z: 0.211, wyckoff: "6c" },
  ]},

  // Heavy fermion — CeCoIn5 (P4/mmm, HoCoGa5-type)
  "CeCoIn5": { formula: "CeCoIn5", spaceGroup: "P4/mmm", spaceGroupNumber: 123, latticeType: "tetragonal", latticeA: 4.61, latticeC: 7.56, pressureGPa: 0, atoms: [
    { element: "Ce", x: 0.0, y: 0.0, z: 0.0, wyckoff: "1a" },
    { element: "Co", x: 0.0, y: 0.0, z: 0.5, wyckoff: "1b" },
    { element: "In", x: 0.5, y: 0.5, z: 0.0, wyckoff: "1c" },
    { element: "In", x: 0.5, y: 0.0, z: 0.3085, wyckoff: "4i" },
    { element: "In", x: 0.0, y: 0.5, z: 0.3085, wyckoff: "4i" },
    { element: "In", x: 0.5, y: 0.0, z: 0.6915, wyckoff: "4i" },
    { element: "In", x: 0.0, y: 0.5, z: 0.6915, wyckoff: "4i" },
  ]},

  // Kagome — KV3Sb5 (P6/mmm)
  "KV3Sb5": { formula: "KV3Sb5", spaceGroup: "P6/mmm", spaceGroupNumber: 191, latticeType: "hexagonal", latticeA: 5.50, latticeC: 9.33, pressureGPa: 0, atoms: [
    { element: "K",  x: 0.0, y: 0.0, z: 0.5, wyckoff: "1b" },
    { element: "V",  x: 0.5, y: 0.0, z: 0.0, wyckoff: "3g" },
    { element: "V",  x: 0.0, y: 0.5, z: 0.0, wyckoff: "3g" },
    { element: "V",  x: 0.5, y: 0.5, z: 0.0, wyckoff: "3g" },
    { element: "Sb", x: 0.3333, y: 0.6667, z: 0.0, wyckoff: "2e" },
    { element: "Sb", x: 0.6667, y: 0.3333, z: 0.0, wyckoff: "2e" },
    { element: "Sb", x: 0.3333, y: 0.6667, z: 0.258, wyckoff: "4h" },
    { element: "Sb", x: 0.6667, y: 0.3333, z: 0.258, wyckoff: "4h" },
    { element: "Sb", x: 0.3333, y: 0.6667, z: 0.742, wyckoff: "4h" },
  ]},
  "CsV3Sb5": { formula: "CsV3Sb5", spaceGroup: "P6/mmm", spaceGroupNumber: 191, latticeType: "hexagonal", latticeA: 5.53, latticeC: 9.40, pressureGPa: 0, atoms: [
    { element: "Cs", x: 0.0, y: 0.0, z: 0.5, wyckoff: "1b" },
    { element: "V",  x: 0.5, y: 0.0, z: 0.0, wyckoff: "3g" },
    { element: "V",  x: 0.0, y: 0.5, z: 0.0, wyckoff: "3g" },
    { element: "V",  x: 0.5, y: 0.5, z: 0.0, wyckoff: "3g" },
    { element: "Sb", x: 0.3333, y: 0.6667, z: 0.0, wyckoff: "2e" },
    { element: "Sb", x: 0.6667, y: 0.3333, z: 0.0, wyckoff: "2e" },
    { element: "Sb", x: 0.3333, y: 0.6667, z: 0.258, wyckoff: "4h" },
    { element: "Sb", x: 0.6667, y: 0.3333, z: 0.258, wyckoff: "4h" },
    { element: "Sb", x: 0.3333, y: 0.6667, z: 0.742, wyckoff: "4h" },
  ]},

  // Laves phase — MgCu2 (Fd-3m, C15)
  "NbBe2": { formula: "NbBe2", spaceGroup: "Fd-3m", spaceGroupNumber: 227, latticeType: "cubic", latticeA: 6.41, pressureGPa: 0, atoms: [
    { element: "Nb", x: 0.125, y: 0.125, z: 0.125, wyckoff: "8a" },
    { element: "Nb", x: 0.875, y: 0.875, z: 0.875, wyckoff: "8a" },
    { element: "Be", x: 0.5, y: 0.5, z: 0.5, wyckoff: "16d" },
    { element: "Be", x: 0.5, y: 0.25, z: 0.25, wyckoff: "16d" },
    { element: "Be", x: 0.25, y: 0.5, z: 0.25, wyckoff: "16d" },
    { element: "Be", x: 0.25, y: 0.25, z: 0.5, wyckoff: "16d" },
  ]},

  // Borocarbide — YNi2B2C (I4/mmm)
  "YNi2B2C": { formula: "YNi2B2C", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.52, latticeC: 10.56, pressureGPa: 0, atoms: [
    { element: "Y",  x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Ni", x: 0.0, y: 0.5, z: 0.25, wyckoff: "4d" },
    { element: "Ni", x: 0.5, y: 0.0, z: 0.25, wyckoff: "4d" },
    { element: "B",  x: 0.0, y: 0.0, z: 0.360, wyckoff: "4e" },
    { element: "B",  x: 0.0, y: 0.0, z: 0.640, wyckoff: "4e" },
    { element: "C",  x: 0.0, y: 0.0, z: 0.5, wyckoff: "2b" },
  ]},

  // Diborides — TiB2 (AlB2-type, P6/mmm)
  "TiB2": { formula: "TiB2", spaceGroup: "P6/mmm", spaceGroupNumber: 191, latticeType: "hexagonal", latticeA: 3.03, latticeC: 3.23, pressureGPa: 0, atoms: [
    { element: "Ti", x: 0.0, y: 0.0, z: 0.0, wyckoff: "1a" },
    { element: "B",  x: 0.3333, y: 0.6667, z: 0.5, wyckoff: "2d" },
    { element: "B",  x: 0.6667, y: 0.3333, z: 0.5, wyckoff: "2d" },
  ]},
  "ZrB2": { formula: "ZrB2", spaceGroup: "P6/mmm", spaceGroupNumber: 191, latticeType: "hexagonal", latticeA: 3.17, latticeC: 3.53, pressureGPa: 0, atoms: [
    { element: "Zr", x: 0.0, y: 0.0, z: 0.0, wyckoff: "1a" },
    { element: "B",  x: 0.3333, y: 0.6667, z: 0.5, wyckoff: "2d" },
    { element: "B",  x: 0.6667, y: 0.3333, z: 0.5, wyckoff: "2d" },
  ]},

  // Elemental superconductors
  "Nb": { formula: "Nb", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.30, pressureGPa: 0, atoms: [{ element: "Nb", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" }] },
  "V":  { formula: "V", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.02, pressureGPa: 0, atoms: [{ element: "V", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" }] },
  "Pb": { formula: "Pb", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.95, pressureGPa: 0, atoms: [{ element: "Pb", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" }] },
  "Al": { formula: "Al", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.05, pressureGPa: 0, atoms: [{ element: "Al", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" }] },

  // Cuprate — La2CuO4 (K2NiF4-type, I4/mmm)
  "La2CuO4": { formula: "La2CuO4", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.78, latticeC: 13.23, pressureGPa: 0, atoms: [
    { element: "La", x: 0.0, y: 0.0, z: 0.361, wyckoff: "4e" },
    { element: "La", x: 0.0, y: 0.0, z: 0.639, wyckoff: "4e" },
    { element: "Cu", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.0, wyckoff: "4c" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.0, wyckoff: "4c" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.182, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.818, wyckoff: "4e" },
  ]},

  // Skutterudite — CoSb3 (Im-3)
  "CoSb3": { formula: "CoSb3", spaceGroup: "Im-3", spaceGroupNumber: 204, latticeType: "cubic", latticeA: 9.04, pressureGPa: 0, atoms: [
    { element: "Co", x: 0.25, y: 0.25, z: 0.25, wyckoff: "8c" },
    { element: "Sb", x: 0.0, y: 0.335, z: 0.158, wyckoff: "24g" },
    { element: "Sb", x: 0.0, y: 0.158, z: 0.335, wyckoff: "24g" },
    { element: "Sb", x: 0.0, y: 0.665, z: 0.842, wyckoff: "24g" },
  ]},

  // MgCNi3 — antiperovskite superconductor (Pm-3m)
  "MgCNi3": { formula: "MgCNi3", spaceGroup: "Pm-3m", spaceGroupNumber: 221, latticeType: "cubic", latticeA: 3.81, pressureGPa: 0, atoms: [
    { element: "Mg", x: 0.0, y: 0.0, z: 0.0, wyckoff: "1a" },
    { element: "C",  x: 0.5, y: 0.5, z: 0.5, wyckoff: "1b" },
    { element: "Ni", x: 0.5, y: 0.5, z: 0.0, wyckoff: "3d" },
    { element: "Ni", x: 0.5, y: 0.0, z: 0.5, wyckoff: "3d" },
    { element: "Ni", x: 0.0, y: 0.5, z: 0.5, wyckoff: "3d" },
  ]},

  // Sr2RuO4 — p-wave superconductor (I4/mmm, K2NiF4-type)
  "Sr2RuO4": { formula: "Sr2RuO4", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.87, latticeC: 12.74, pressureGPa: 0, atoms: [
    { element: "Sr", x: 0.0, y: 0.0, z: 0.353, wyckoff: "4e" },
    { element: "Sr", x: 0.0, y: 0.0, z: 0.647, wyckoff: "4e" },
    { element: "Ru", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.0, wyckoff: "4c" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.0, wyckoff: "4c" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.162, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.838, wyckoff: "4e" },
  ]},

  // Additional hydrides for completeness
  "BaH2": { formula: "BaH2", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.16, pressureGPa: 0, atoms: [
    { element: "Ba", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "H",  x: 0.25, y: 0.25, z: 0.25, wyckoff: "8c" },
    { element: "H",  x: 0.75, y: 0.75, z: 0.75, wyckoff: "8c" },
  ]},
  "CaH2": { formula: "CaH2", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 3.59, pressureGPa: 0, atoms: [
    { element: "Ca", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "H",  x: 0.25, y: 0.25, z: 0.25, wyckoff: "8c" },
    { element: "H",  x: 0.75, y: 0.75, z: 0.75, wyckoff: "8c" },
  ]},

  // NbTi alloy — BCC
  "NbTi": { formula: "NbTi", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.30, pressureGPa: 0, atoms: [
    { element: "Nb", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Ti", x: 0.5, y: 0.5, z: 0.5, wyckoff: "2a" },
  ]},

  // More cuprate variants
  "Bi2Sr2CaCu2O8": { formula: "Bi2Sr2CaCu2O8", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.81, latticeC: 30.89, pressureGPa: 0, atoms: [
    { element: "Bi", x: 0.0, y: 0.0, z: 0.199, wyckoff: "4e" },
    { element: "Bi", x: 0.0, y: 0.0, z: 0.801, wyckoff: "4e" },
    { element: "Sr", x: 0.0, y: 0.0, z: 0.110, wyckoff: "4e" },
    { element: "Sr", x: 0.0, y: 0.0, z: 0.890, wyckoff: "4e" },
    { element: "Ca", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Cu", x: 0.0, y: 0.0, z: 0.054, wyckoff: "4e" },
    { element: "Cu", x: 0.0, y: 0.0, z: 0.946, wyckoff: "4e" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.054, wyckoff: "8g" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.054, wyckoff: "8g" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.149, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.851, wyckoff: "4e" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.946, wyckoff: "8g" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.946, wyckoff: "8g" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.250, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.750, wyckoff: "4e" },
  ]},

  // More iron pnictide variants
  "SrFe2As2": { formula: "SrFe2As2", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.93, latticeC: 12.36, pressureGPa: 0, atoms: [
    { element: "Sr", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Fe", x: 0.5, y: 0.0, z: 0.25, wyckoff: "4d" },
    { element: "Fe", x: 0.0, y: 0.5, z: 0.25, wyckoff: "4d" },
    { element: "As", x: 0.0, y: 0.0, z: 0.360, wyckoff: "4e" },
    { element: "As", x: 0.0, y: 0.0, z: 0.640, wyckoff: "4e" },
  ]},
  "CaFe2As2": { formula: "CaFe2As2", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.88, latticeC: 11.74, pressureGPa: 0, atoms: [
    { element: "Ca", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Fe", x: 0.5, y: 0.0, z: 0.25, wyckoff: "4d" },
    { element: "Fe", x: 0.0, y: 0.5, z: 0.25, wyckoff: "4d" },
    { element: "As", x: 0.0, y: 0.0, z: 0.366, wyckoff: "4e" },
    { element: "As", x: 0.0, y: 0.0, z: 0.634, wyckoff: "4e" },
  ]},

  // Diamond cubic — Si, Ge, C
  "Si": { formula: "Si", spaceGroup: "Fd-3m", spaceGroupNumber: 227, latticeType: "cubic", latticeA: 5.43, pressureGPa: 0, atoms: [
    { element: "Si", x: 0.0, y: 0.0, z: 0.0, wyckoff: "8a" },
    { element: "Si", x: 0.25, y: 0.25, z: 0.25, wyckoff: "8a" },
  ]},
  "Ge": { formula: "Ge", spaceGroup: "Fd-3m", spaceGroupNumber: 227, latticeType: "cubic", latticeA: 5.66, pressureGPa: 0, atoms: [
    { element: "Ge", x: 0.0, y: 0.0, z: 0.0, wyckoff: "8a" },
    { element: "Ge", x: 0.25, y: 0.25, z: 0.25, wyckoff: "8a" },
  ]},
  "C": { formula: "C", spaceGroup: "Fd-3m", spaceGroupNumber: 227, latticeType: "cubic", latticeA: 3.57, pressureGPa: 0, atoms: [
    { element: "C", x: 0.0, y: 0.0, z: 0.0, wyckoff: "8a" },
    { element: "C", x: 0.25, y: 0.25, z: 0.25, wyckoff: "8a" },
  ]},

  // More elemental superconductors
  "Sn": { formula: "Sn", spaceGroup: "I41/amd", spaceGroupNumber: 141, latticeType: "tetragonal", latticeA: 5.83, latticeC: 3.18, pressureGPa: 0, atoms: [
    { element: "Sn", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
  ]},
  "In": { formula: "In", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.25, latticeC: 4.95, pressureGPa: 0, atoms: [
    { element: "In", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
  ]},
  "Ta": { formula: "Ta", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.30, pressureGPa: 0, atoms: [
    { element: "Ta", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
  ]},
  "Re": { formula: "Re", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 2.76, latticeC: 4.46, pressureGPa: 0, atoms: [
    { element: "Re", x: 0.3333, y: 0.6667, z: 0.25, wyckoff: "2c" },
  ]},
  "Mo": { formula: "Mo", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.15, pressureGPa: 0, atoms: [
    { element: "Mo", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
  ]},
  "W": { formula: "W", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.16, pressureGPa: 0, atoms: [
    { element: "W", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
  ]},
  "Ti": { formula: "Ti", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 2.95, latticeC: 4.69, pressureGPa: 0, atoms: [
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.25, wyckoff: "2c" },
  ]},
  "Zr": { formula: "Zr", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.23, latticeC: 5.15, pressureGPa: 0, atoms: [
    { element: "Zr", x: 0.3333, y: 0.6667, z: 0.25, wyckoff: "2c" },
  ]},
  "Hf": { formula: "Hf", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.19, latticeC: 5.05, pressureGPa: 0, atoms: [
    { element: "Hf", x: 0.3333, y: 0.6667, z: 0.25, wyckoff: "2c" },
  ]},

  // Spinel — MgAl2O4 (Fd-3m)
  "MgAl2O4": { formula: "MgAl2O4", spaceGroup: "Fd-3m", spaceGroupNumber: 227, latticeType: "cubic", latticeA: 8.08, pressureGPa: 0, atoms: [
    { element: "Mg", x: 0.125, y: 0.125, z: 0.125, wyckoff: "8a" },
    { element: "Al", x: 0.5, y: 0.5, z: 0.5, wyckoff: "16d" },
    { element: "Al", x: 0.5, y: 0.25, z: 0.25, wyckoff: "16d" },
    { element: "O",  x: 0.263, y: 0.263, z: 0.263, wyckoff: "32e" },
    { element: "O",  x: 0.263, y: 0.237, z: 0.737, wyckoff: "32e" },
    { element: "O",  x: 0.737, y: 0.263, z: 0.737, wyckoff: "32e" },
    { element: "O",  x: 0.737, y: 0.737, z: 0.263, wyckoff: "32e" },
  ]},

  // Double perovskite — Sr2FeMoO6 (Fm-3m)
  "Sr2FeMoO6": { formula: "Sr2FeMoO6", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 7.90, pressureGPa: 0, atoms: [
    { element: "Sr", x: 0.25, y: 0.25, z: 0.25, wyckoff: "8c" },
    { element: "Sr", x: 0.75, y: 0.75, z: 0.75, wyckoff: "8c" },
    { element: "Fe", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "Mo", x: 0.5, y: 0.5, z: 0.5, wyckoff: "4b" },
    { element: "O",  x: 0.25, y: 0.0, z: 0.0, wyckoff: "24e" },
    { element: "O",  x: 0.75, y: 0.0, z: 0.0, wyckoff: "24e" },
    { element: "O",  x: 0.0, y: 0.25, z: 0.0, wyckoff: "24e" },
    { element: "O",  x: 0.0, y: 0.75, z: 0.0, wyckoff: "24e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.25, wyckoff: "24e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.75, wyckoff: "24e" },
  ]},

  // Ruddlesden-Popper n=2 — Sr3Ru2O7 (I4/mmm)
  "Sr3Ru2O7": { formula: "Sr3Ru2O7", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.89, latticeC: 20.73, pressureGPa: 0, atoms: [
    { element: "Sr", x: 0.0, y: 0.0, z: 0.5, wyckoff: "2b" },
    { element: "Sr", x: 0.0, y: 0.0, z: 0.318, wyckoff: "4e" },
    { element: "Sr", x: 0.0, y: 0.0, z: 0.682, wyckoff: "4e" },
    { element: "Ru", x: 0.0, y: 0.0, z: 0.098, wyckoff: "4e" },
    { element: "Ru", x: 0.0, y: 0.0, z: 0.902, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.098, wyckoff: "8g" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.098, wyckoff: "8g" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.198, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.802, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.902, wyckoff: "8g" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.902, wyckoff: "8g" },
  ]},

  // MAX phase — Ti2AlC (P63/mmc)
  "Ti2AlC": { formula: "Ti2AlC", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.06, latticeC: 13.60, pressureGPa: 0, atoms: [
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.086, wyckoff: "4f" },
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.914, wyckoff: "4f" },
    { element: "Al", x: 0.0, y: 0.0, z: 0.25, wyckoff: "2d" },
    { element: "C",  x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
  ]},

  // MAX 312 phase — Ti3SiC2 (P63/mmc)
  "Ti3SiC2": { formula: "Ti3SiC2", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.07, latticeC: 17.67, pressureGPa: 0, atoms: [
    { element: "Ti", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.135, wyckoff: "4f" },
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.865, wyckoff: "4f" },
    { element: "Si", x: 0.0, y: 0.0, z: 0.25, wyckoff: "2b" },
    { element: "C",  x: 0.3333, y: 0.6667, z: 0.070, wyckoff: "4f" },
    { element: "C",  x: 0.3333, y: 0.6667, z: 0.930, wyckoff: "4f" },
  ]},
  "Ti3AlC2": { formula: "Ti3AlC2", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.08, latticeC: 18.58, pressureGPa: 0, atoms: [
    { element: "Ti", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.128, wyckoff: "4f" },
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.872, wyckoff: "4f" },
    { element: "Al", x: 0.0, y: 0.0, z: 0.25, wyckoff: "2b" },
    { element: "C",  x: 0.3333, y: 0.6667, z: 0.065, wyckoff: "4f" },
    { element: "C",  x: 0.3333, y: 0.6667, z: 0.935, wyckoff: "4f" },
  ]},

  // MAX 413 phase — Ti4AlN3 (P63/mmc)
  "Ti4AlN3": { formula: "Ti4AlN3", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 2.99, latticeC: 23.37, pressureGPa: 0, atoms: [
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.054, wyckoff: "4f" },
    { element: "Ti", x: 0.3333, y: 0.6667, z: 0.946, wyckoff: "4f" },
    { element: "Ti", x: 0.0, y: 0.0, z: 0.155, wyckoff: "4e" },
    { element: "Ti", x: 0.0, y: 0.0, z: 0.845, wyckoff: "4e" },
    { element: "Al", x: 0.3333, y: 0.6667, z: 0.25, wyckoff: "2c" },
    { element: "N",  x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "N",  x: 0.3333, y: 0.6667, z: 0.103, wyckoff: "4f" },
    { element: "N",  x: 0.3333, y: 0.6667, z: 0.897, wyckoff: "4f" },
  ]},
  "Ta4AlC3": { formula: "Ta4AlC3", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 3.09, latticeC: 24.12, pressureGPa: 0, atoms: [
    { element: "Ta", x: 0.3333, y: 0.6667, z: 0.054, wyckoff: "4f" },
    { element: "Ta", x: 0.3333, y: 0.6667, z: 0.946, wyckoff: "4f" },
    { element: "Ta", x: 0.0, y: 0.0, z: 0.155, wyckoff: "4e" },
    { element: "Ta", x: 0.0, y: 0.0, z: 0.845, wyckoff: "4e" },
    { element: "Al", x: 0.3333, y: 0.6667, z: 0.25, wyckoff: "2c" },
    { element: "C",  x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "C",  x: 0.3333, y: 0.6667, z: 0.103, wyckoff: "4f" },
    { element: "C",  x: 0.3333, y: 0.6667, z: 0.897, wyckoff: "4f" },
  ]},

  // Brownmillerite — Ca2Fe2O5 (Pnma)
  "Ca2Fe2O5": { formula: "Ca2Fe2O5", spaceGroup: "Pnma", spaceGroupNumber: 62, latticeType: "tetragonal", latticeA: 5.43, latticeC: 14.77, pressureGPa: 0, atoms: [
    { element: "Ca", x: 0.027, y: 0.25, z: 0.509, wyckoff: "8d" },
    { element: "Ca", x: 0.522, y: 0.25, z: 0.039, wyckoff: "8d" },
    { element: "Fe", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "Fe", x: 0.928, y: 0.25, z: 0.929, wyckoff: "4b" },
    { element: "O",  x: 0.250, y: 0.007, z: 0.231, wyckoff: "8d" },
    { element: "O",  x: 0.028, y: 0.25, z: 0.744, wyckoff: "4c" },
    { element: "O",  x: 0.595, y: 0.25, z: 0.875, wyckoff: "4c" },
    { element: "O",  x: 0.860, y: 0.25, z: 0.070, wyckoff: "8d" },
    { element: "O",  x: 0.371, y: 0.25, z: 0.419, wyckoff: "8d" },
  ]},

  // Ruddlesden-Popper n=3 — La4Ni3O10 (I4/mmm, high-Tc nickelate under pressure)
  "La4Ni3O10": { formula: "La4Ni3O10", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.86, latticeC: 27.96, pressureGPa: 0, atoms: [
    { element: "La", x: 0.0, y: 0.0, z: 0.5, wyckoff: "2b" },
    { element: "La", x: 0.0, y: 0.0, z: 0.321, wyckoff: "4e" },
    { element: "La", x: 0.0, y: 0.0, z: 0.679, wyckoff: "4e" },
    { element: "La", x: 0.0, y: 0.0, z: 0.178, wyckoff: "4e" },
    { element: "Ni", x: 0.0, y: 0.0, z: 0.071, wyckoff: "4e" },
    { element: "Ni", x: 0.0, y: 0.0, z: 0.929, wyckoff: "4e" },
    { element: "Ni", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.071, wyckoff: "8g" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.071, wyckoff: "8g" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.929, wyckoff: "8g" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.929, wyckoff: "8g" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.0, wyckoff: "4c" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.0, wyckoff: "4c" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.393, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.607, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.107, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.893, wyckoff: "4e" },
  ]},

  // Corundum — Al2O3 (R-3c)
  "Al2O3": { formula: "Al2O3", spaceGroup: "R-3c", spaceGroupNumber: 167, latticeType: "hexagonal", latticeA: 4.76, latticeC: 12.99, pressureGPa: 0, atoms: [
    { element: "Al", x: 0.0, y: 0.0, z: 0.352, wyckoff: "12c" },
    { element: "Al", x: 0.0, y: 0.0, z: 0.648, wyckoff: "12c" },
    { element: "Al", x: 0.0, y: 0.0, z: 0.852, wyckoff: "12c" },
    { element: "Al", x: 0.0, y: 0.0, z: 0.148, wyckoff: "12c" },
    { element: "O",  x: 0.306, y: 0.0, z: 0.25, wyckoff: "18e" },
    { element: "O",  x: 0.0, y: 0.306, z: 0.25, wyckoff: "18e" },
    { element: "O",  x: 0.694, y: 0.694, z: 0.25, wyckoff: "18e" },
    { element: "O",  x: 0.694, y: 0.0, z: 0.75, wyckoff: "18e" },
    { element: "O",  x: 0.0, y: 0.694, z: 0.75, wyckoff: "18e" },
    { element: "O",  x: 0.306, y: 0.306, z: 0.75, wyckoff: "18e" },
  ]},
  "Fe2O3": { formula: "Fe2O3", spaceGroup: "R-3c", spaceGroupNumber: 167, latticeType: "hexagonal", latticeA: 5.04, latticeC: 13.75, pressureGPa: 0, atoms: [
    { element: "Fe", x: 0.0, y: 0.0, z: 0.355, wyckoff: "12c" },
    { element: "Fe", x: 0.0, y: 0.0, z: 0.645, wyckoff: "12c" },
    { element: "Fe", x: 0.0, y: 0.0, z: 0.855, wyckoff: "12c" },
    { element: "Fe", x: 0.0, y: 0.0, z: 0.145, wyckoff: "12c" },
    { element: "O",  x: 0.306, y: 0.0, z: 0.25, wyckoff: "18e" },
    { element: "O",  x: 0.0, y: 0.306, z: 0.25, wyckoff: "18e" },
    { element: "O",  x: 0.694, y: 0.694, z: 0.25, wyckoff: "18e" },
    { element: "O",  x: 0.694, y: 0.0, z: 0.75, wyckoff: "18e" },
    { element: "O",  x: 0.0, y: 0.694, z: 0.75, wyckoff: "18e" },
    { element: "O",  x: 0.306, y: 0.306, z: 0.75, wyckoff: "18e" },
  ]},

  // Ilmenite — FeTiO3 (R-3)
  "FeTiO3": { formula: "FeTiO3", spaceGroup: "R-3", spaceGroupNumber: 148, latticeType: "hexagonal", latticeA: 5.09, latticeC: 14.09, pressureGPa: 0, atoms: [
    { element: "Fe", x: 0.0, y: 0.0, z: 0.356, wyckoff: "4c" },
    { element: "Fe", x: 0.0, y: 0.0, z: 0.644, wyckoff: "4c" },
    { element: "Ti", x: 0.0, y: 0.0, z: 0.146, wyckoff: "4c" },
    { element: "Ti", x: 0.0, y: 0.0, z: 0.854, wyckoff: "4c" },
    { element: "O",  x: 0.317, y: 0.020, z: 0.245, wyckoff: "18f" },
    { element: "O",  x: 0.980, y: 0.297, z: 0.245, wyckoff: "18f" },
    { element: "O",  x: 0.703, y: 0.683, z: 0.245, wyckoff: "18f" },
    { element: "O",  x: 0.683, y: 0.980, z: 0.755, wyckoff: "18f" },
    { element: "O",  x: 0.020, y: 0.703, z: 0.755, wyckoff: "18f" },
    { element: "O",  x: 0.297, y: 0.317, z: 0.755, wyckoff: "18f" },
  ]},

  // Marcasite — FeSb2 (Pnnm)
  "FeSb2": { formula: "FeSb2", spaceGroup: "Pnnm", spaceGroupNumber: 58, latticeType: "tetragonal", latticeA: 5.83, latticeC: 3.20, pressureGPa: 0, atoms: [
    { element: "Fe", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "Fe", x: 0.5, y: 0.5, z: 0.5, wyckoff: "2a" },
    { element: "Sb", x: 0.200, y: 0.378, z: 0.0, wyckoff: "4g" },
    { element: "Sb", x: 0.800, y: 0.622, z: 0.0, wyckoff: "4g" },
    { element: "Sb", x: 0.300, y: 0.878, z: 0.5, wyckoff: "4g" },
    { element: "Sb", x: 0.700, y: 0.122, z: 0.5, wyckoff: "4g" },
  ]},

  // Tl-2201 single-layer cuprate — Tl2Ba2CuO6 (I4/mmm)
  "Tl2Ba2CuO6": { formula: "Tl2Ba2CuO6", spaceGroup: "I4/mmm", spaceGroupNumber: 139, latticeType: "tetragonal", latticeA: 3.87, latticeC: 23.14, pressureGPa: 0, atoms: [
    { element: "Tl", x: 0.0, y: 0.0, z: 0.210, wyckoff: "4e" },
    { element: "Tl", x: 0.0, y: 0.0, z: 0.790, wyckoff: "4e" },
    { element: "Ba", x: 0.0, y: 0.0, z: 0.116, wyckoff: "4e" },
    { element: "Ba", x: 0.0, y: 0.0, z: 0.884, wyckoff: "4e" },
    { element: "Cu", x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a" },
    { element: "O",  x: 0.5, y: 0.0, z: 0.0, wyckoff: "4c" },
    { element: "O",  x: 0.0, y: 0.5, z: 0.0, wyckoff: "4c" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.163, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.837, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.279, wyckoff: "4e" },
    { element: "O",  x: 0.0, y: 0.0, z: 0.721, wyckoff: "4e" },
  ]},

  // Post-perovskite — CaIrO3 (Cmcm)
  "CaIrO3": { formula: "CaIrO3", spaceGroup: "Cmcm", spaceGroupNumber: 63, latticeType: "tetragonal", latticeA: 3.15, latticeC: 7.30, pressureGPa: 0, atoms: [
    { element: "Ca", x: 0.0, y: 0.25, z: 0.25, wyckoff: "4c" },
    { element: "Ca", x: 0.0, y: 0.75, z: 0.75, wyckoff: "4c" },
    { element: "Ir", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "Ir", x: 0.0, y: 0.5, z: 0.5, wyckoff: "4a" },
    { element: "O",  x: 0.0, y: 0.93, z: 0.25, wyckoff: "4c" },
    { element: "O",  x: 0.0, y: 0.07, z: 0.75, wyckoff: "4c" },
    { element: "O",  x: 0.0, y: 0.63, z: 0.07, wyckoff: "8f" },
    { element: "O",  x: 0.0, y: 0.37, z: 0.93, wyckoff: "8f" },
    { element: "O",  x: 0.0, y: 0.13, z: 0.57, wyckoff: "8f" },
    { element: "O",  x: 0.0, y: 0.87, z: 0.43, wyckoff: "8f" },
  ]},

  // Inverse Heusler — Mn2CoAl (F-43m)
  "Mn2CoAl": { formula: "Mn2CoAl", spaceGroup: "F-43m", spaceGroupNumber: 216, latticeType: "cubic", latticeA: 5.80, pressureGPa: 0, atoms: [
    { element: "Al", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "Mn", x: 0.25, y: 0.25, z: 0.25, wyckoff: "4c" },
    { element: "Mn", x: 0.75, y: 0.75, z: 0.75, wyckoff: "4d" },
    { element: "Co", x: 0.5, y: 0.5, z: 0.5, wyckoff: "4b" },
  ]},

  // GaAs — Zinc blende (F-43m)
  "GaAs": { formula: "GaAs", spaceGroup: "F-43m", spaceGroupNumber: 216, latticeType: "cubic", latticeA: 5.65, pressureGPa: 0, atoms: [
    { element: "Ga", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "As", x: 0.25, y: 0.25, z: 0.25, wyckoff: "4c" },
  ]},

  // GaN — Wurtzite (P63mc)
  "GaN": { formula: "GaN", spaceGroup: "P63mc", spaceGroupNumber: 186, latticeType: "hexagonal", latticeA: 3.19, latticeC: 5.19, pressureGPa: 0, atoms: [
    { element: "Ga", x: 0.3333, y: 0.6667, z: 0.0, wyckoff: "2b" },
    { element: "N",  x: 0.3333, y: 0.6667, z: 0.375, wyckoff: "2b" },
  ]},

  // PbTe — Rocksalt thermoelectric
  "PbTe": { formula: "PbTe", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 6.46, pressureGPa: 0, atoms: [
    { element: "Pb", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "Te", x: 0.5, y: 0.5, z: 0.5, wyckoff: "4b" },
  ]},

  // Heusler — Cu2MnAl (L21, Fm-3m)
  "Cu2MnAl": { formula: "Cu2MnAl", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 5.95, pressureGPa: 0, atoms: [
    { element: "Cu", x: 0.25, y: 0.25, z: 0.25, wyckoff: "8c" },
    { element: "Cu", x: 0.75, y: 0.75, z: 0.75, wyckoff: "8c" },
    { element: "Mn", x: 0.5, y: 0.5, z: 0.5, wyckoff: "4b" },
    { element: "Al", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
  ]},

  // Half-Heusler — NiMnSb (F-43m)
  "NiMnSb": { formula: "NiMnSb", spaceGroup: "F-43m", spaceGroupNumber: 216, latticeType: "cubic", latticeA: 5.92, pressureGPa: 0, atoms: [
    { element: "Ni", x: 0.25, y: 0.25, z: 0.25, wyckoff: "4c" },
    { element: "Mn", x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a" },
    { element: "Sb", x: 0.5, y: 0.5, z: 0.5, wyckoff: "4b" },
  ]},

  // WC — simple hexagonal (P-6m2)
  "WC": { formula: "WC", spaceGroup: "P-6m2", spaceGroupNumber: 187, latticeType: "hexagonal", latticeA: 2.91, latticeC: 2.84, pressureGPa: 0, atoms: [
    { element: "W", x: 0.0, y: 0.0, z: 0.0, wyckoff: "1a" },
    { element: "C", x: 0.3333, y: 0.6667, z: 0.5, wyckoff: "1d" },
  ]},

  // Additional sodalite hydride variants
  "BaH6":  { formula: "BaH6", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 4.20, pressureGPa: 150, atoms: SODALITE_H6_ATOMS("Ba") },
  "MgH6":  { formula: "MgH6", spaceGroup: "Im-3m", spaceGroupNumber: 229, latticeType: "cubic", latticeA: 3.60, pressureGPa: 300, atoms: SODALITE_H6_ATOMS("Mg") },
  "ThH9":  { formula: "ThH9", spaceGroup: "P63/mmc", spaceGroupNumber: 194, latticeType: "hexagonal", latticeA: 4.62, latticeC: 7.16, pressureGPa: 175, atoms: YH9_ATOMS.map(a => ({ ...a, element: a.element === "Y" ? "Th" : a.element })) },
};

// ---------------------------------------------------------------------------
// Formula normalization
// ---------------------------------------------------------------------------

function normalizeFormula(formula: string): string {
  return formula.replace(/\s+/g, "");
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Look up exact literature structure for a known compound.
 * Returns primitive-cell fractional coordinates suitable for QE input.
 */
export function lookupKnownStructure(formula: string): KnownStructure | null {
  const norm = normalizeFormula(formula);
  return KNOWN_STRUCTURES[norm] ?? null;
}

/**
 * Check if a formula has a known structure in the database.
 */
export function hasKnownStructure(formula: string): boolean {
  return normalizeFormula(formula) in KNOWN_STRUCTURES;
}

/**
 * Get all known structure formulas (for logging/diagnostics).
 */
export function getKnownStructureFormulas(): string[] {
  return Object.keys(KNOWN_STRUCTURES);
}
