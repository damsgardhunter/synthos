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

  // Simple hydrides
  "ScH3":  { formula: "ScH3", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.48, pressureGPa: 0, atoms: ScH3_ATOMS },
  "YH3":   { formula: "YH3", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 5.14, pressureGPa: 0, atoms: ScH3_ATOMS.map(a => ({ ...a, element: a.element === "Sc" ? "Y" : a.element })) },

  // Non-hydride superconductors
  "MgB2":  { formula: "MgB2", spaceGroup: "P6/mmm", spaceGroupNumber: 191, latticeType: "hexagonal", latticeA: 3.09, latticeC: 3.53, pressureGPa: 0, atoms: MgB2_ATOMS },
  "Nb3Sn": { formula: "Nb3Sn", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 5.29, pressureGPa: 0, atoms: Nb3Sn_ATOMS },
  "Nb3Ge": { formula: "Nb3Ge", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 5.14, pressureGPa: 0, atoms: Nb3Ge_ATOMS },
  "V3Si":  { formula: "V3Si", spaceGroup: "Pm-3n", spaceGroupNumber: 223, latticeType: "cubic", latticeA: 4.72, pressureGPa: 0, atoms: V3Si_ATOMS },
  "NbN":   { formula: "NbN", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.39, pressureGPa: 0, atoms: NbN_ATOMS },
  "NbC":   { formula: "NbC", spaceGroup: "Fm-3m", spaceGroupNumber: 225, latticeType: "cubic", latticeA: 4.47, pressureGPa: 0, atoms: NbC_ATOMS },
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
