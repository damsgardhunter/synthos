import { fetchSummary, isApiAvailable as isMPAvailable } from "../learning/materials-project-client";

export interface AtomicPosition {
  element: string;
  x: number;
  y: number;
  z: number;
}

export interface CrystalStructureEntry {
  formula: string;
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  atomicPositions: AtomicPosition[];
  spacegroup: number;
  spacegroupSymbol: string;
  crystalSystem: string;
  prototype: string;
  formationEnergy: number;
  volume: number;
  density: number;
  nsites: number;
  source: string;
}

const dataset: CrystalStructureEntry[] = [];

function e(formula: string, a: number, b: number, c: number, alpha: number, beta: number, gamma: number, positions: AtomicPosition[], sg: number, sgSym: string, cs: string, proto: string, fE: number, vol: number, dens: number): CrystalStructureEntry {
  return { formula, lattice: { a, b, c, alpha, beta, gamma }, atomicPositions: positions, spacegroup: sg, spacegroupSymbol: sgSym, crystalSystem: cs, prototype: proto, formationEnergy: fE, volume: vol, density: dens, nsites: positions.length, source: "seed" };
}

const SEED_DATA: CrystalStructureEntry[] = [
  e("Fe", 2.87, 2.87, 2.87, 90, 90, 90, [{ element: "Fe", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0.5, z: 0.5 }], 229, "Im-3m", "cubic", "BCC", 0, 23.55, 7.87),
  e("Cu", 3.61, 3.61, 3.61, 90, 90, 90, [{ element: "Cu", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0.5, y: 0.5, z: 0 }, { element: "Cu", x: 0.5, y: 0, z: 0.5 }, { element: "Cu", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 11.81, 8.96),
  e("Al", 4.05, 4.05, 4.05, 90, 90, 90, [{ element: "Al", x: 0, y: 0, z: 0 }, { element: "Al", x: 0.5, y: 0.5, z: 0 }, { element: "Al", x: 0.5, y: 0, z: 0.5 }, { element: "Al", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 16.60, 2.70),
  e("Si", 5.43, 5.43, 5.43, 90, 90, 90, [{ element: "Si", x: 0, y: 0, z: 0 }, { element: "Si", x: 0.25, y: 0.25, z: 0.25 }, { element: "Si", x: 0.5, y: 0.5, z: 0 }, { element: "Si", x: 0.75, y: 0.75, z: 0.25 }, { element: "Si", x: 0.5, y: 0, z: 0.5 }, { element: "Si", x: 0.75, y: 0.25, z: 0.75 }, { element: "Si", x: 0, y: 0.5, z: 0.5 }, { element: "Si", x: 0.25, y: 0.75, z: 0.75 }], 227, "Fd-3m", "cubic", "diamond", 0, 20.02, 2.33),
  e("Nb", 3.30, 3.30, 3.30, 90, 90, 90, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "Nb", x: 0.5, y: 0.5, z: 0.5 }], 229, "Im-3m", "cubic", "BCC", 0, 17.97, 8.57),
  e("W", 3.16, 3.16, 3.16, 90, 90, 90, [{ element: "W", x: 0, y: 0, z: 0 }, { element: "W", x: 0.5, y: 0.5, z: 0.5 }], 229, "Im-3m", "cubic", "BCC", 0, 15.78, 19.25),
  e("Ti", 2.95, 2.95, 4.69, 90, 90, 120, [{ element: "Ti", x: 0.333, y: 0.667, z: 0.25 }, { element: "Ti", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 17.65, 4.51),
  e("Ni", 3.52, 3.52, 3.52, 90, 90, 90, [{ element: "Ni", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0.5, y: 0.5, z: 0 }, { element: "Ni", x: 0.5, y: 0, z: 0.5 }, { element: "Ni", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 10.94, 8.91),
  e("Cr", 2.88, 2.88, 2.88, 90, 90, 90, [{ element: "Cr", x: 0, y: 0, z: 0 }, { element: "Cr", x: 0.5, y: 0.5, z: 0.5 }], 229, "Im-3m", "cubic", "BCC", 0, 11.94, 7.19),
  e("V", 3.02, 3.02, 3.02, 90, 90, 90, [{ element: "V", x: 0, y: 0, z: 0 }, { element: "V", x: 0.5, y: 0.5, z: 0.5 }], 229, "Im-3m", "cubic", "BCC", 0, 13.77, 6.11),
  e("Mo", 3.15, 3.15, 3.15, 90, 90, 90, [{ element: "Mo", x: 0, y: 0, z: 0 }, { element: "Mo", x: 0.5, y: 0.5, z: 0.5 }], 229, "Im-3m", "cubic", "BCC", 0, 15.62, 10.22),
  e("Ta", 3.30, 3.30, 3.30, 90, 90, 90, [{ element: "Ta", x: 0, y: 0, z: 0 }, { element: "Ta", x: 0.5, y: 0.5, z: 0.5 }], 229, "Im-3m", "cubic", "BCC", 0, 17.97, 16.69),
  e("Pb", 4.95, 4.95, 4.95, 90, 90, 90, [{ element: "Pb", x: 0, y: 0, z: 0 }, { element: "Pb", x: 0.5, y: 0.5, z: 0 }, { element: "Pb", x: 0.5, y: 0, z: 0.5 }, { element: "Pb", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 30.32, 11.34),
  e("Ag", 4.09, 4.09, 4.09, 90, 90, 90, [{ element: "Ag", x: 0, y: 0, z: 0 }, { element: "Ag", x: 0.5, y: 0.5, z: 0 }, { element: "Ag", x: 0.5, y: 0, z: 0.5 }, { element: "Ag", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 17.12, 10.49),
  e("Au", 4.08, 4.08, 4.08, 90, 90, 90, [{ element: "Au", x: 0, y: 0, z: 0 }, { element: "Au", x: 0.5, y: 0.5, z: 0 }, { element: "Au", x: 0.5, y: 0, z: 0.5 }, { element: "Au", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 17.01, 19.32),
  e("Pt", 3.92, 3.92, 3.92, 90, 90, 90, [{ element: "Pt", x: 0, y: 0, z: 0 }, { element: "Pt", x: 0.5, y: 0.5, z: 0 }, { element: "Pt", x: 0.5, y: 0, z: 0.5 }, { element: "Pt", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 15.10, 21.45),
  e("Zn", 2.66, 2.66, 4.95, 90, 90, 120, [{ element: "Zn", x: 0.333, y: 0.667, z: 0.25 }, { element: "Zn", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 15.21, 7.13),
  e("Mg", 3.21, 3.21, 5.21, 90, 90, 120, [{ element: "Mg", x: 0.333, y: 0.667, z: 0.25 }, { element: "Mg", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 23.24, 1.74),
  e("Sn", 5.83, 5.83, 3.18, 90, 90, 90, [{ element: "Sn", x: 0, y: 0, z: 0 }, { element: "Sn", x: 0.5, y: 0.5, z: 0.5 }, { element: "Sn", x: 0, y: 0.5, z: 0.25 }, { element: "Sn", x: 0.5, y: 0, z: 0.75 }], 141, "I4_1/amd", "tetragonal", "beta-Sn", 0, 27.05, 7.29),

  e("NaCl", 5.64, 5.64, 5.64, 90, 90, 90, [{ element: "Na", x: 0, y: 0, z: 0 }, { element: "Na", x: 0.5, y: 0.5, z: 0 }, { element: "Na", x: 0.5, y: 0, z: 0.5 }, { element: "Na", x: 0, y: 0.5, z: 0.5 }, { element: "Cl", x: 0.5, y: 0, z: 0 }, { element: "Cl", x: 0, y: 0.5, z: 0 }, { element: "Cl", x: 0, y: 0, z: 0.5 }, { element: "Cl", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -3.32, 22.43, 2.16),
  e("NbN", 4.39, 4.39, 4.39, 90, 90, 90, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "Nb", x: 0.5, y: 0.5, z: 0 }, { element: "Nb", x: 0.5, y: 0, z: 0.5 }, { element: "Nb", x: 0, y: 0.5, z: 0.5 }, { element: "N", x: 0.5, y: 0, z: 0 }, { element: "N", x: 0, y: 0.5, z: 0 }, { element: "N", x: 0, y: 0, z: 0.5 }, { element: "N", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -2.34, 10.56, 8.47),
  e("TiN", 4.24, 4.24, 4.24, 90, 90, 90, [{ element: "Ti", x: 0, y: 0, z: 0 }, { element: "Ti", x: 0.5, y: 0.5, z: 0 }, { element: "Ti", x: 0.5, y: 0, z: 0.5 }, { element: "Ti", x: 0, y: 0.5, z: 0.5 }, { element: "N", x: 0.5, y: 0, z: 0 }, { element: "N", x: 0, y: 0.5, z: 0 }, { element: "N", x: 0, y: 0, z: 0.5 }, { element: "N", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -3.43, 9.55, 5.40),
  e("VN", 4.14, 4.14, 4.14, 90, 90, 90, [{ element: "V", x: 0, y: 0, z: 0 }, { element: "V", x: 0.5, y: 0.5, z: 0 }, { element: "V", x: 0.5, y: 0, z: 0.5 }, { element: "V", x: 0, y: 0.5, z: 0.5 }, { element: "N", x: 0.5, y: 0, z: 0 }, { element: "N", x: 0, y: 0.5, z: 0 }, { element: "N", x: 0, y: 0, z: 0.5 }, { element: "N", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -2.15, 8.85, 6.13),
  e("TaC", 4.46, 4.46, 4.46, 90, 90, 90, [{ element: "Ta", x: 0, y: 0, z: 0 }, { element: "Ta", x: 0.5, y: 0.5, z: 0 }, { element: "Ta", x: 0.5, y: 0, z: 0.5 }, { element: "Ta", x: 0, y: 0.5, z: 0.5 }, { element: "C", x: 0.5, y: 0, z: 0 }, { element: "C", x: 0, y: 0.5, z: 0 }, { element: "C", x: 0, y: 0, z: 0.5 }, { element: "C", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -1.48, 11.08, 14.48),
  e("ZrN", 4.59, 4.59, 4.59, 90, 90, 90, [{ element: "Zr", x: 0, y: 0, z: 0 }, { element: "Zr", x: 0.5, y: 0.5, z: 0 }, { element: "Zr", x: 0.5, y: 0, z: 0.5 }, { element: "Zr", x: 0, y: 0.5, z: 0.5 }, { element: "N", x: 0.5, y: 0, z: 0 }, { element: "N", x: 0, y: 0.5, z: 0 }, { element: "N", x: 0, y: 0, z: 0.5 }, { element: "N", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -3.29, 12.12, 7.09),
  e("HfN", 4.52, 4.52, 4.52, 90, 90, 90, [{ element: "Hf", x: 0, y: 0, z: 0 }, { element: "Hf", x: 0.5, y: 0.5, z: 0 }, { element: "Hf", x: 0.5, y: 0, z: 0.5 }, { element: "Hf", x: 0, y: 0.5, z: 0.5 }, { element: "N", x: 0.5, y: 0, z: 0 }, { element: "N", x: 0, y: 0.5, z: 0 }, { element: "N", x: 0, y: 0, z: 0.5 }, { element: "N", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -3.65, 11.57, 13.84),
  e("NbC", 4.47, 4.47, 4.47, 90, 90, 90, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "Nb", x: 0.5, y: 0.5, z: 0 }, { element: "Nb", x: 0.5, y: 0, z: 0.5 }, { element: "Nb", x: 0, y: 0.5, z: 0.5 }, { element: "C", x: 0.5, y: 0, z: 0 }, { element: "C", x: 0, y: 0.5, z: 0 }, { element: "C", x: 0, y: 0, z: 0.5 }, { element: "C", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -1.29, 11.16, 7.82),
  e("TaN", 4.34, 4.34, 4.34, 90, 90, 90, [{ element: "Ta", x: 0, y: 0, z: 0 }, { element: "Ta", x: 0.5, y: 0.5, z: 0 }, { element: "Ta", x: 0.5, y: 0, z: 0.5 }, { element: "Ta", x: 0, y: 0.5, z: 0.5 }, { element: "N", x: 0.5, y: 0, z: 0 }, { element: "N", x: 0, y: 0.5, z: 0 }, { element: "N", x: 0, y: 0, z: 0.5 }, { element: "N", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -2.58, 10.22, 15.60),
  e("MgO", 4.21, 4.21, 4.21, 90, 90, 90, [{ element: "Mg", x: 0, y: 0, z: 0 }, { element: "Mg", x: 0.5, y: 0.5, z: 0 }, { element: "Mg", x: 0.5, y: 0, z: 0.5 }, { element: "Mg", x: 0, y: 0.5, z: 0.5 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }, { element: "O", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -6.13, 9.33, 3.58),
  e("CaO", 4.81, 4.81, 4.81, 90, 90, 90, [{ element: "Ca", x: 0, y: 0, z: 0 }, { element: "Ca", x: 0.5, y: 0.5, z: 0 }, { element: "Ca", x: 0.5, y: 0, z: 0.5 }, { element: "Ca", x: 0, y: 0.5, z: 0.5 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }, { element: "O", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "NaCl", -6.37, 13.94, 3.35),

  e("SrTiO3", 3.905, 3.905, 3.905, 90, 90, 90, [{ element: "Sr", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ti", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.36, 59.55, 5.12),
  e("BaTiO3", 4.00, 4.00, 4.04, 90, 90, 90, [{ element: "Ba", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ti", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 99, "P4mm", "tetragonal", "perovskite", -3.30, 64.64, 6.02),
  e("LaAlO3", 3.79, 3.79, 3.79, 90, 90, 90, [{ element: "La", x: 0.5, y: 0.5, z: 0.5 }, { element: "Al", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.89, 54.44, 6.52),
  e("KNbO3", 4.01, 4.01, 4.06, 90, 90, 90, [{ element: "K", x: 0.5, y: 0.5, z: 0.5 }, { element: "Nb", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 99, "P4mm", "tetragonal", "perovskite", -2.95, 65.29, 4.62),
  e("NaNbO3", 3.89, 3.89, 3.92, 90, 90, 90, [{ element: "Na", x: 0.5, y: 0.5, z: 0.5 }, { element: "Nb", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.10, 59.29, 4.56),
  e("CaTiO3", 5.38, 5.44, 7.64, 90, 90, 90, [{ element: "Ca", x: 0.007, y: 0.036, z: 0.25 }, { element: "Ti", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.290, y: 0.289, z: 0.040 }, { element: "O", x: 0.710, y: 0.711, z: 0.960 }, { element: "O", x: 0.488, y: 0.075, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -3.56, 223.33, 4.04),
  e("LaNiO3", 3.84, 3.84, 3.84, 90, 90, 90, [{ element: "La", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ni", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -2.38, 56.62, 7.11),
  e("SrRuO3", 5.57, 5.53, 7.85, 90, 90, 90, [{ element: "Sr", x: 0.012, y: 0.053, z: 0.25 }, { element: "Ru", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.225, y: 0.278, z: 0.033 }, { element: "O", x: 0.775, y: 0.722, z: 0.967 }, { element: "O", x: 0.527, y: 0.052, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -2.69, 241.87, 6.49),
  e("BaZrO3", 4.19, 4.19, 4.19, 90, 90, 90, [{ element: "Ba", x: 0.5, y: 0.5, z: 0.5 }, { element: "Zr", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.39, 73.56, 6.21),
  e("CsSnI3", 6.22, 6.22, 6.22, 90, 90, 90, [{ element: "Cs", x: 0.5, y: 0.5, z: 0.5 }, { element: "Sn", x: 0, y: 0, z: 0 }, { element: "I", x: 0.5, y: 0, z: 0 }, { element: "I", x: 0, y: 0.5, z: 0 }, { element: "I", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -1.26, 240.64, 4.50),

  e("MgB2", 3.09, 3.09, 3.52, 90, 90, 120, [{ element: "Mg", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.15, 29.09, 2.63),
  e("AlB2", 3.01, 3.01, 3.26, 90, 90, 120, [{ element: "Al", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.09, 25.60, 3.19),
  e("ZrB2", 3.17, 3.17, 3.53, 90, 90, 120, [{ element: "Zr", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -1.02, 30.72, 6.09),
  e("TiB2", 3.03, 3.03, 3.23, 90, 90, 120, [{ element: "Ti", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -1.22, 25.69, 4.52),
  e("NbB2", 3.11, 3.11, 3.26, 90, 90, 120, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.56, 27.32, 6.98),
  e("TaB2", 3.10, 3.10, 3.22, 90, 90, 120, [{ element: "Ta", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.60, 26.82, 12.62),
  e("HfB2", 3.14, 3.14, 3.47, 90, 90, 120, [{ element: "Hf", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -1.10, 29.62, 11.20),
  e("CrB2", 2.97, 2.97, 3.07, 90, 90, 120, [{ element: "Cr", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.23, 23.45, 5.20),
  e("VB2", 3.00, 3.00, 3.05, 90, 90, 120, [{ element: "V", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.45, 23.79, 5.10),
  e("MoB2", 3.04, 3.04, 3.11, 90, 90, 120, [{ element: "Mo", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.25, 24.88, 7.70),
  e("WB2", 3.02, 3.02, 3.07, 90, 90, 120, [{ element: "W", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.28, 24.27, 12.76),
  e("ScB2", 3.15, 3.15, 3.52, 90, 90, 120, [{ element: "Sc", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "B", x: 0.667, y: 0.333, z: 0.5 }], 191, "P6/mmm", "hexagonal", "AlB2", -0.85, 30.25, 3.36),

  e("Nb3Sn", 5.29, 5.29, 5.29, 90, 90, 90, [{ element: "Nb", x: 0.25, y: 0, z: 0.5 }, { element: "Nb", x: 0.75, y: 0, z: 0.5 }, { element: "Nb", x: 0.5, y: 0.25, z: 0 }, { element: "Nb", x: 0.5, y: 0.75, z: 0 }, { element: "Nb", x: 0, y: 0.5, z: 0.25 }, { element: "Nb", x: 0, y: 0.5, z: 0.75 }, { element: "Sn", x: 0, y: 0, z: 0 }, { element: "Sn", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.42, 148.04, 8.91),
  e("V3Si", 4.72, 4.72, 4.72, 90, 90, 90, [{ element: "V", x: 0.25, y: 0, z: 0.5 }, { element: "V", x: 0.75, y: 0, z: 0.5 }, { element: "V", x: 0.5, y: 0.25, z: 0 }, { element: "V", x: 0.5, y: 0.75, z: 0 }, { element: "V", x: 0, y: 0.5, z: 0.25 }, { element: "V", x: 0, y: 0.5, z: 0.75 }, { element: "Si", x: 0, y: 0, z: 0 }, { element: "Si", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.41, 105.15, 5.75),
  e("Nb3Ge", 5.17, 5.17, 5.17, 90, 90, 90, [{ element: "Nb", x: 0.25, y: 0, z: 0.5 }, { element: "Nb", x: 0.75, y: 0, z: 0.5 }, { element: "Nb", x: 0.5, y: 0.25, z: 0 }, { element: "Nb", x: 0.5, y: 0.75, z: 0 }, { element: "Nb", x: 0, y: 0.5, z: 0.25 }, { element: "Nb", x: 0, y: 0.5, z: 0.75 }, { element: "Ge", x: 0, y: 0, z: 0 }, { element: "Ge", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.35, 138.19, 8.49),
  e("Nb3Al", 5.19, 5.19, 5.19, 90, 90, 90, [{ element: "Nb", x: 0.25, y: 0, z: 0.5 }, { element: "Nb", x: 0.75, y: 0, z: 0.5 }, { element: "Nb", x: 0.5, y: 0.25, z: 0 }, { element: "Nb", x: 0.5, y: 0.75, z: 0 }, { element: "Nb", x: 0, y: 0.5, z: 0.25 }, { element: "Nb", x: 0, y: 0.5, z: 0.75 }, { element: "Al", x: 0, y: 0, z: 0 }, { element: "Al", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.38, 139.80, 7.26),
  e("V3Ga", 4.82, 4.82, 4.82, 90, 90, 90, [{ element: "V", x: 0.25, y: 0, z: 0.5 }, { element: "V", x: 0.75, y: 0, z: 0.5 }, { element: "V", x: 0.5, y: 0.25, z: 0 }, { element: "V", x: 0.5, y: 0.75, z: 0 }, { element: "V", x: 0, y: 0.5, z: 0.25 }, { element: "V", x: 0, y: 0.5, z: 0.75 }, { element: "Ga", x: 0, y: 0, z: 0 }, { element: "Ga", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.38, 112.09, 6.63),
  e("Nb3Ga", 5.18, 5.18, 5.18, 90, 90, 90, [{ element: "Nb", x: 0.25, y: 0, z: 0.5 }, { element: "Nb", x: 0.75, y: 0, z: 0.5 }, { element: "Nb", x: 0.5, y: 0.25, z: 0 }, { element: "Nb", x: 0.5, y: 0.75, z: 0 }, { element: "Nb", x: 0, y: 0.5, z: 0.25 }, { element: "Nb", x: 0, y: 0.5, z: 0.75 }, { element: "Ga", x: 0, y: 0, z: 0 }, { element: "Ga", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.35, 139.0, 8.53),
  e("Cr3Si", 4.56, 4.56, 4.56, 90, 90, 90, [{ element: "Cr", x: 0.25, y: 0, z: 0.5 }, { element: "Cr", x: 0.75, y: 0, z: 0.5 }, { element: "Cr", x: 0.5, y: 0.25, z: 0 }, { element: "Cr", x: 0.5, y: 0.75, z: 0 }, { element: "Cr", x: 0, y: 0.5, z: 0.25 }, { element: "Cr", x: 0, y: 0.5, z: 0.75 }, { element: "Si", x: 0, y: 0, z: 0 }, { element: "Si", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.25, 94.82, 6.45),
  e("Mo3Re", 4.96, 4.96, 4.96, 90, 90, 90, [{ element: "Mo", x: 0.25, y: 0, z: 0.5 }, { element: "Mo", x: 0.75, y: 0, z: 0.5 }, { element: "Mo", x: 0.5, y: 0.25, z: 0 }, { element: "Mo", x: 0.5, y: 0.75, z: 0 }, { element: "Mo", x: 0, y: 0.5, z: 0.25 }, { element: "Mo", x: 0, y: 0.5, z: 0.75 }, { element: "Re", x: 0, y: 0, z: 0 }, { element: "Re", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.15, 122.04, 12.62),

  e("LaH10", 5.10, 5.10, 5.10, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0 }, { element: "La", x: 0.5, y: 0.5, z: 0.5 }, { element: "La", x: 0.5, y: 0, z: 0.5 }, { element: "La", x: 0, y: 0.5, z: 0.5 }, { element: "H", x: 0.12, y: 0.12, z: 0.12 }, { element: "H", x: 0.88, y: 0.88, z: 0.88 }, { element: "H", x: 0.38, y: 0.38, z: 0.38 }, { element: "H", x: 0.62, y: 0.62, z: 0.62 }, { element: "H", x: 0.25, y: 0, z: 0 }, { element: "H", x: 0, y: 0.25, z: 0 }, { element: "H", x: 0, y: 0, z: 0.25 }, { element: "H", x: 0.75, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "sodalite-clathrate", -0.03, 132.65, 5.66),
  e("CaH6", 3.54, 3.54, 3.54, 90, 90, 90, [{ element: "Ca", x: 0, y: 0, z: 0 }, { element: "H", x: 0.165, y: 0.165, z: 0.165 }, { element: "H", x: 0.835, y: 0.835, z: 0.835 }, { element: "H", x: 0.165, y: 0.835, z: 0.5 }, { element: "H", x: 0.835, y: 0.165, z: 0.5 }, { element: "H", x: 0.5, y: 0.165, z: 0.835 }, { element: "H", x: 0.5, y: 0.835, z: 0.165 }], 229, "Im-3m", "cubic", "clathrate", -0.02, 44.36, 3.02),
  e("YH9", 3.61, 3.61, 5.39, 90, 90, 120, [{ element: "Y", x: 0, y: 0, z: 0 }, { element: "H", x: 0.333, y: 0.667, z: 0.12 }, { element: "H", x: 0.667, y: 0.333, z: 0.88 }, { element: "H", x: 0.167, y: 0.333, z: 0.5 }, { element: "H", x: 0.833, y: 0.667, z: 0.5 }, { element: "H", x: 0.12, y: 0.06, z: 0.25 }, { element: "H", x: 0.88, y: 0.94, z: 0.75 }, { element: "H", x: 0.5, y: 0.25, z: 0.25 }, { element: "H", x: 0.5, y: 0.75, z: 0.75 }, { element: "H", x: 0.25, y: 0.5, z: 0.25 }], 194, "P6_3/mmc", "hexagonal", "clathrate", -0.05, 60.91, 4.10),
  e("YH6", 3.66, 3.66, 3.66, 90, 90, 90, [{ element: "Y", x: 0, y: 0, z: 0 }, { element: "H", x: 0.165, y: 0.165, z: 0.165 }, { element: "H", x: 0.835, y: 0.835, z: 0.835 }, { element: "H", x: 0.165, y: 0.835, z: 0.5 }, { element: "H", x: 0.835, y: 0.165, z: 0.5 }, { element: "H", x: 0.5, y: 0.165, z: 0.835 }, { element: "H", x: 0.5, y: 0.835, z: 0.165 }], 229, "Im-3m", "cubic", "clathrate", -0.04, 49.03, 3.20),
  e("H3S", 3.09, 3.09, 3.09, 90, 90, 90, [{ element: "S", x: 0, y: 0, z: 0 }, { element: "H", x: 0.5, y: 0, z: 0 }, { element: "H", x: 0, y: 0.5, z: 0 }, { element: "H", x: 0, y: 0, z: 0.5 }], 229, "Im-3m", "cubic", "clathrate", -0.10, 29.50, 3.80),
  e("ThH10", 5.14, 5.14, 5.14, 90, 90, 90, [{ element: "Th", x: 0, y: 0, z: 0 }, { element: "Th", x: 0.5, y: 0.5, z: 0.5 }, { element: "Th", x: 0.5, y: 0, z: 0.5 }, { element: "Th", x: 0, y: 0.5, z: 0.5 }, { element: "H", x: 0.12, y: 0.12, z: 0.12 }, { element: "H", x: 0.88, y: 0.88, z: 0.88 }, { element: "H", x: 0.38, y: 0.38, z: 0.38 }, { element: "H", x: 0.62, y: 0.62, z: 0.62 }, { element: "H", x: 0.25, y: 0, z: 0 }, { element: "H", x: 0, y: 0.25, z: 0 }, { element: "H", x: 0, y: 0, z: 0.25 }, { element: "H", x: 0.75, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "sodalite-clathrate", -0.03, 135.80, 6.48),
  e("CeH9", 3.72, 3.72, 5.51, 90, 90, 120, [{ element: "Ce", x: 0, y: 0, z: 0 }, { element: "H", x: 0.333, y: 0.667, z: 0.12 }, { element: "H", x: 0.667, y: 0.333, z: 0.88 }, { element: "H", x: 0.167, y: 0.333, z: 0.5 }, { element: "H", x: 0.833, y: 0.667, z: 0.5 }, { element: "H", x: 0.12, y: 0.06, z: 0.25 }, { element: "H", x: 0.88, y: 0.94, z: 0.75 }, { element: "H", x: 0.5, y: 0.25, z: 0.25 }, { element: "H", x: 0.5, y: 0.75, z: 0.75 }, { element: "H", x: 0.25, y: 0.5, z: 0.25 }], 194, "P6_3/mmc", "hexagonal", "clathrate", -0.04, 66.02, 4.88),

  e("La2CuO4", 3.78, 3.78, 13.23, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0.361 }, { element: "La", x: 0, y: 0, z: 0.639 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.182 }, { element: "O", x: 0, y: 0, z: 0.818 }], 139, "I4/mmm", "tetragonal", "K2NiF4", -2.89, 189.10, 7.07),
  e("YBa2Cu3O7", 3.82, 3.89, 11.68, 90, 90, 90, [{ element: "Y", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.184 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.816 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0, y: 0, z: 0.356 }, { element: "Cu", x: 0, y: 0, z: 0.644 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.158 }, { element: "O", x: 0, y: 0, z: 0.842 }, { element: "O", x: 0.5, y: 0, z: 0.378 }, { element: "O", x: 0, y: 0.5, z: 0.378 }, { element: "O", x: 0.5, y: 0, z: 0.622 }, { element: "O", x: 0, y: 0.5, z: 0.622 }], 47, "Pmmm", "orthorhombic", "YBCO", -2.54, 173.56, 6.38),
  e("Bi2Sr2CaCu2O8", 5.41, 5.41, 30.89, 90, 90, 90, [{ element: "Bi", x: 0, y: 0, z: 0.05 }, { element: "Bi", x: 0.5, y: 0.5, z: 0.95 }, { element: "Sr", x: 0, y: 0, z: 0.14 }, { element: "Sr", x: 0.5, y: 0.5, z: 0.86 }, { element: "Ca", x: 0, y: 0, z: 0.25 }, { element: "Cu", x: 0, y: 0, z: 0.20 }, { element: "Cu", x: 0.5, y: 0.5, z: 0.80 }, { element: "O", x: 0, y: 0, z: 0.12 }, { element: "O", x: 0.5, y: 0.5, z: 0.88 }], 139, "I4/mmm", "tetragonal", "BSCCO", -2.10, 903.72, 6.50),
  e("HgBa2Ca2Cu3O8", 3.85, 3.85, 15.85, 90, 90, 90, [{ element: "Hg", x: 0, y: 0, z: 0 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.085 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.915 }, { element: "Ca", x: 0.5, y: 0.5, z: 0.268 }, { element: "Ca", x: 0.5, y: 0.5, z: 0.732 }, { element: "Cu", x: 0, y: 0, z: 0.17 }, { element: "Cu", x: 0, y: 0, z: 0.5 }, { element: "Cu", x: 0, y: 0, z: 0.83 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }], 123, "P4/mmm", "tetragonal", "Hg-cuprate", -1.85, 234.89, 7.20),
  e("Tl2Ba2CaCu2O8", 3.86, 3.86, 29.32, 90, 90, 90, [{ element: "Tl", x: 0, y: 0, z: 0.04 }, { element: "Tl", x: 0.5, y: 0.5, z: 0.96 }, { element: "Ba", x: 0, y: 0, z: 0.12 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.88 }, { element: "Ca", x: 0, y: 0, z: 0.25 }, { element: "Cu", x: 0, y: 0, z: 0.19 }, { element: "Cu", x: 0.5, y: 0.5, z: 0.81 }, { element: "O", x: 0.5, y: 0, z: 0.19 }, { element: "O", x: 0, y: 0.5, z: 0.19 }], 139, "I4/mmm", "tetragonal", "Tl-cuprate", -2.20, 436.62, 7.80),

  e("LaFeAsO", 4.03, 4.03, 8.74, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0.142 }, { element: "Fe", x: 0.5, y: 0, z: 0.5 }, { element: "As", x: 0, y: 0, z: 0.651 }, { element: "O", x: 0.5, y: 0.5, z: 0 }], 129, "P4/nmm", "tetragonal", "ZrCuSiAs", -2.15, 141.81, 6.15),
  e("BaFe2As2", 3.96, 3.96, 13.02, 90, 90, 90, [{ element: "Ba", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0, z: 0.25 }, { element: "Fe", x: 0, y: 0.5, z: 0.25 }, { element: "As", x: 0, y: 0, z: 0.354 }, { element: "As", x: 0, y: 0, z: 0.646 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2", -1.48, 204.12, 6.12),
  e("FeSe", 3.77, 3.77, 5.52, 90, 90, 90, [{ element: "Fe", x: 0.75, y: 0.25, z: 0 }, { element: "Se", x: 0.25, y: 0.25, z: 0.267 }], 129, "P4/nmm", "tetragonal", "PbO-type", -0.48, 78.50, 4.87),
  e("LiFeAs", 3.79, 3.79, 6.36, 90, 90, 90, [{ element: "Li", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0, z: 0.5 }, { element: "As", x: 0, y: 0, z: 0.737 }], 129, "P4/nmm", "tetragonal", "PbFCl-type", -1.25, 91.33, 5.30),
  e("NdFeAsO", 3.97, 3.97, 8.59, 90, 90, 90, [{ element: "Nd", x: 0, y: 0, z: 0.136 }, { element: "Fe", x: 0.5, y: 0, z: 0.5 }, { element: "As", x: 0, y: 0, z: 0.656 }, { element: "O", x: 0.5, y: 0.5, z: 0 }], 129, "P4/nmm", "tetragonal", "ZrCuSiAs", -2.25, 135.31, 6.78),
  e("SmFeAsO", 3.94, 3.94, 8.50, 90, 90, 90, [{ element: "Sm", x: 0, y: 0, z: 0.135 }, { element: "Fe", x: 0.5, y: 0, z: 0.5 }, { element: "As", x: 0, y: 0, z: 0.657 }, { element: "O", x: 0.5, y: 0.5, z: 0 }], 129, "P4/nmm", "tetragonal", "ZrCuSiAs", -2.30, 131.88, 7.12),
  e("CaKFe4As4", 3.87, 3.87, 12.88, 90, 90, 90, [{ element: "Ca", x: 0, y: 0, z: 0.5 }, { element: "K", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0, z: 0.25 }, { element: "Fe", x: 0, y: 0.5, z: 0.25 }, { element: "As", x: 0, y: 0, z: 0.338 }, { element: "As", x: 0, y: 0, z: 0.662 }, { element: "Fe", x: 0.5, y: 0, z: 0.75 }, { element: "Fe", x: 0, y: 0.5, z: 0.75 }, { element: "As", x: 0, y: 0, z: 0.838 }, { element: "As", x: 0, y: 0, z: 0.162 }], 123, "P4/mmm", "tetragonal", "1144-type", -1.32, 192.98, 5.89),

  e("MgZn2", 5.22, 5.22, 8.57, 90, 90, 120, [{ element: "Mg", x: 0.333, y: 0.667, z: 0.063 }, { element: "Mg", x: 0.667, y: 0.333, z: 0.937 }, { element: "Mg", x: 0.333, y: 0.667, z: 0.563 }, { element: "Mg", x: 0.667, y: 0.333, z: 0.437 }, { element: "Zn", x: 0.833, y: 0.667, z: 0.25 }, { element: "Zn", x: 0.167, y: 0.333, z: 0.75 }, { element: "Zn", x: 0, y: 0, z: 0.25 }, { element: "Zn", x: 0, y: 0, z: 0.75 }, { element: "Zn", x: 0.333, y: 0.167, z: 0.25 }, { element: "Zn", x: 0.667, y: 0.833, z: 0.75 }, { element: "Zn", x: 0.667, y: 0.333, z: 0.25 }, { element: "Zn", x: 0.333, y: 0.667, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "Laves-C14", -0.14, 201.98, 5.27),
  e("CaCu5", 5.08, 5.08, 4.07, 90, 90, 120, [{ element: "Ca", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0.333, y: 0.667, z: 0 }, { element: "Cu", x: 0.667, y: 0.333, z: 0 }, { element: "Cu", x: 0.5, y: 0, z: 0.5 }, { element: "Cu", x: 0, y: 0.5, z: 0.5 }, { element: "Cu", x: 0.5, y: 0.5, z: 0.5 }], 191, "P6/mmm", "hexagonal", "CaCu5", -0.17, 90.98, 6.23),
  e("ZrV2", 7.44, 7.44, 7.44, 90, 90, 90, [{ element: "Zr", x: 0.125, y: 0.125, z: 0.125 }, { element: "Zr", x: 0.875, y: 0.875, z: 0.875 }, { element: "Zr", x: 0.375, y: 0.375, z: 0.375 }, { element: "Zr", x: 0.625, y: 0.625, z: 0.625 }, { element: "V", x: 0, y: 0, z: 0 }, { element: "V", x: 0.5, y: 0.5, z: 0 }, { element: "V", x: 0.5, y: 0, z: 0.5 }, { element: "V", x: 0, y: 0.5, z: 0.5 }, { element: "V", x: 0.25, y: 0.25, z: 0.25 }, { element: "V", x: 0.75, y: 0.75, z: 0.25 }, { element: "V", x: 0.75, y: 0.25, z: 0.75 }, { element: "V", x: 0.25, y: 0.75, z: 0.75 }], 227, "Fd-3m", "cubic", "Laves-C15", -0.38, 411.83, 6.41),
  e("HfV2", 7.37, 7.37, 7.37, 90, 90, 90, [{ element: "Hf", x: 0.125, y: 0.125, z: 0.125 }, { element: "Hf", x: 0.875, y: 0.875, z: 0.875 }, { element: "V", x: 0, y: 0, z: 0 }, { element: "V", x: 0.5, y: 0.5, z: 0 }, { element: "V", x: 0.5, y: 0, z: 0.5 }, { element: "V", x: 0, y: 0.5, z: 0.5 }], 227, "Fd-3m", "cubic", "Laves-C15", -0.42, 399.07, 9.82),

  e("Cu2MnAl", 5.95, 5.95, 5.95, 90, 90, 90, [{ element: "Cu", x: 0.25, y: 0.25, z: 0.25 }, { element: "Cu", x: 0.75, y: 0.75, z: 0.75 }, { element: "Mn", x: 0, y: 0, z: 0 }, { element: "Al", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.32, 210.03, 6.29),
  e("Ni2MnGa", 5.83, 5.83, 5.83, 90, 90, 90, [{ element: "Ni", x: 0.25, y: 0.25, z: 0.25 }, { element: "Ni", x: 0.75, y: 0.75, z: 0.75 }, { element: "Mn", x: 0, y: 0, z: 0 }, { element: "Ga", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.35, 197.93, 8.07),
  e("Co2MnSi", 5.65, 5.65, 5.65, 90, 90, 90, [{ element: "Co", x: 0.25, y: 0.25, z: 0.25 }, { element: "Co", x: 0.75, y: 0.75, z: 0.75 }, { element: "Mn", x: 0, y: 0, z: 0 }, { element: "Si", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.55, 180.36, 7.29),

  e("CoSb3", 9.04, 9.04, 9.04, 90, 90, 90, [{ element: "Co", x: 0.25, y: 0.25, z: 0.25 }, { element: "Sb", x: 0, y: 0.335, z: 0.158 }, { element: "Sb", x: 0, y: 0.665, z: 0.842 }, { element: "Sb", x: 0.335, y: 0.158, z: 0 }, { element: "Sb", x: 0.665, y: 0.842, z: 0 }], 204, "Im-3", "cubic", "skutterudite", -0.18, 738.76, 7.64),
  e("IrSb3", 9.26, 9.26, 9.26, 90, 90, 90, [{ element: "Ir", x: 0.25, y: 0.25, z: 0.25 }, { element: "Sb", x: 0, y: 0.338, z: 0.156 }, { element: "Sb", x: 0, y: 0.662, z: 0.844 }, { element: "Sb", x: 0.338, y: 0.156, z: 0 }, { element: "Sb", x: 0.662, y: 0.844, z: 0 }], 204, "Im-3", "cubic", "skutterudite", -0.15, 793.17, 9.20),

  e("PbMo6S8", 6.55, 6.55, 6.55, 89.2, 89.2, 89.2, [{ element: "Pb", x: 0, y: 0, z: 0 }, { element: "Mo", x: 0.40, y: 0.22, z: 0.07 }, { element: "Mo", x: 0.07, y: 0.40, z: 0.22 }, { element: "Mo", x: 0.22, y: 0.07, z: 0.40 }, { element: "S", x: 0.26, y: 0.37, z: 0.62 }, { element: "S", x: 0.62, y: 0.26, z: 0.37 }, { element: "S", x: 0.37, y: 0.62, z: 0.26 }], 148, "R-3", "rhombohedral", "Chevrel", -0.65, 281.01, 7.16),

  e("CeCoIn5", 4.61, 4.61, 7.56, 90, 90, 90, [{ element: "Ce", x: 0, y: 0, z: 0 }, { element: "Co", x: 0.5, y: 0.5, z: 0 }, { element: "In", x: 0.5, y: 0, z: 0.305 }, { element: "In", x: 0, y: 0.5, z: 0.305 }, { element: "In", x: 0.5, y: 0, z: 0.695 }, { element: "In", x: 0, y: 0.5, z: 0.695 }, { element: "In", x: 0, y: 0, z: 0.5 }], 123, "P4/mmm", "tetragonal", "HoCoGa5", -0.54, 160.72, 8.56),
  e("CeCu2Si2", 4.10, 4.10, 9.93, 90, 90, 90, [{ element: "Ce", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0, y: 0.5, z: 0.25 }, { element: "Cu", x: 0.5, y: 0, z: 0.25 }, { element: "Si", x: 0, y: 0, z: 0.371 }, { element: "Si", x: 0, y: 0, z: 0.629 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2", -0.73, 166.86, 7.19),
  e("UTe2", 4.16, 6.14, 13.97, 90, 90, 90, [{ element: "U", x: 0, y: 0.25, z: 0.137 }, { element: "U", x: 0, y: 0.75, z: 0.863 }, { element: "Te", x: 0, y: 0.25, z: 0.586 }, { element: "Te", x: 0, y: 0.75, z: 0.414 }], 71, "Immm", "orthorhombic", "marcasite", -0.78, 356.84, 11.20),

  e("NiTi", 3.01, 3.01, 3.01, 90, 90, 90, [{ element: "Ni", x: 0, y: 0, z: 0 }, { element: "Ti", x: 0.5, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "CsCl-B2", -0.45, 27.27, 6.45),

  e("Bi2Se3", 4.14, 4.14, 28.64, 90, 90, 120, [{ element: "Bi", x: 0, y: 0, z: 0.399 }, { element: "Se", x: 0, y: 0, z: 0 }, { element: "Se", x: 0, y: 0, z: 0.211 }], 166, "R-3m", "rhombohedral", "Bi2Se3-TI", -0.38, 425.34, 6.82),
  e("Bi2Te3", 4.39, 4.39, 30.50, 90, 90, 120, [{ element: "Bi", x: 0, y: 0, z: 0.400 }, { element: "Te", x: 0, y: 0, z: 0 }, { element: "Te", x: 0, y: 0, z: 0.212 }], 166, "R-3m", "rhombohedral", "Bi2Te3", -0.23, 508.72, 7.86),
  e("Sb2Te3", 4.26, 4.26, 30.44, 90, 90, 120, [{ element: "Sb", x: 0, y: 0, z: 0.399 }, { element: "Te", x: 0, y: 0, z: 0 }, { element: "Te", x: 0, y: 0, z: 0.212 }], 166, "R-3m", "rhombohedral", "Bi2Te3", -0.19, 478.54, 6.50),

  e("GaAs", 5.65, 5.65, 5.65, 90, 90, 90, [{ element: "Ga", x: 0, y: 0, z: 0 }, { element: "Ga", x: 0.5, y: 0.5, z: 0 }, { element: "Ga", x: 0.5, y: 0, z: 0.5 }, { element: "Ga", x: 0, y: 0.5, z: 0.5 }, { element: "As", x: 0.25, y: 0.25, z: 0.25 }, { element: "As", x: 0.75, y: 0.75, z: 0.25 }, { element: "As", x: 0.75, y: 0.25, z: 0.75 }, { element: "As", x: 0.25, y: 0.75, z: 0.75 }], 216, "F-43m", "cubic", "zincblende", -0.74, 45.17, 5.32),
  e("GaN", 3.19, 3.19, 5.19, 90, 90, 120, [{ element: "Ga", x: 0.333, y: 0.667, z: 0 }, { element: "N", x: 0.333, y: 0.667, z: 0.385 }], 186, "P6_3mc", "hexagonal", "wurtzite", -1.24, 22.85, 6.15),
  e("ZnO", 3.25, 3.25, 5.21, 90, 90, 120, [{ element: "Zn", x: 0.333, y: 0.667, z: 0 }, { element: "O", x: 0.333, y: 0.667, z: 0.382 }], 186, "P6_3mc", "hexagonal", "wurtzite", -3.60, 23.81, 5.61),
  e("SiC", 4.36, 4.36, 4.36, 90, 90, 90, [{ element: "Si", x: 0, y: 0, z: 0 }, { element: "Si", x: 0.5, y: 0.5, z: 0 }, { element: "Si", x: 0.5, y: 0, z: 0.5 }, { element: "Si", x: 0, y: 0.5, z: 0.5 }, { element: "C", x: 0.25, y: 0.25, z: 0.25 }, { element: "C", x: 0.75, y: 0.75, z: 0.25 }, { element: "C", x: 0.75, y: 0.25, z: 0.75 }, { element: "C", x: 0.25, y: 0.75, z: 0.75 }], 216, "F-43m", "cubic", "zincblende", -0.72, 20.75, 3.21),
  e("InP", 5.87, 5.87, 5.87, 90, 90, 90, [{ element: "In", x: 0, y: 0, z: 0 }, { element: "P", x: 0.25, y: 0.25, z: 0.25 }], 216, "F-43m", "cubic", "zincblende", -0.46, 50.64, 4.81),
  e("CdTe", 6.48, 6.48, 6.48, 90, 90, 90, [{ element: "Cd", x: 0, y: 0, z: 0 }, { element: "Te", x: 0.25, y: 0.25, z: 0.25 }], 216, "F-43m", "cubic", "zincblende", -0.40, 68.10, 5.85),
  e("ZnS", 5.41, 5.41, 5.41, 90, 90, 90, [{ element: "Zn", x: 0, y: 0, z: 0 }, { element: "S", x: 0.25, y: 0.25, z: 0.25 }], 216, "F-43m", "cubic", "zincblende", -1.97, 39.60, 4.09),
  e("AlN", 3.11, 3.11, 4.98, 90, 90, 120, [{ element: "Al", x: 0.333, y: 0.667, z: 0 }, { element: "N", x: 0.333, y: 0.667, z: 0.385 }], 186, "P6_3mc", "hexagonal", "wurtzite", -3.28, 20.85, 3.26),
  e("BN", 2.50, 2.50, 6.66, 90, 90, 120, [{ element: "B", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "N", x: 0, y: 0, z: 0.5 }, { element: "N", x: 0.333, y: 0.667, z: 0 }], 194, "P6_3/mmc", "hexagonal", "graphite-BN", -2.80, 36.02, 2.10),
  e("Ge", 5.66, 5.66, 5.66, 90, 90, 90, [{ element: "Ge", x: 0, y: 0, z: 0 }, { element: "Ge", x: 0.25, y: 0.25, z: 0.25 }, { element: "Ge", x: 0.5, y: 0.5, z: 0 }, { element: "Ge", x: 0.75, y: 0.75, z: 0.25 }], 227, "Fd-3m", "cubic", "diamond", 0, 22.63, 5.32),

  e("Al2O3", 4.76, 4.76, 12.99, 90, 90, 120, [{ element: "Al", x: 0, y: 0, z: 0.352 }, { element: "O", x: 0.306, y: 0, z: 0.25 }], 167, "R-3c", "rhombohedral", "corundum", -3.47, 254.80, 3.99),
  e("TiO2", 4.59, 4.59, 2.96, 90, 90, 90, [{ element: "Ti", x: 0, y: 0, z: 0 }, { element: "Ti", x: 0.5, y: 0.5, z: 0.5 }, { element: "O", x: 0.305, y: 0.305, z: 0 }, { element: "O", x: 0.695, y: 0.695, z: 0 }, { element: "O", x: 0.805, y: 0.195, z: 0.5 }, { element: "O", x: 0.195, y: 0.805, z: 0.5 }], 136, "P4_2/mnm", "tetragonal", "rutile", -4.95, 62.45, 4.23),
  e("ZrO2", 5.15, 5.21, 5.32, 90, 99.2, 90, [{ element: "Zr", x: 0.276, y: 0.040, z: 0.208 }, { element: "O", x: 0.070, y: 0.332, z: 0.345 }, { element: "O", x: 0.442, y: 0.755, z: 0.479 }], 14, "P2_1/c", "monoclinic", "baddeleyite", -5.98, 140.44, 5.68),
  e("SnO2", 4.74, 4.74, 3.19, 90, 90, 90, [{ element: "Sn", x: 0, y: 0, z: 0 }, { element: "Sn", x: 0.5, y: 0.5, z: 0.5 }, { element: "O", x: 0.307, y: 0.307, z: 0 }, { element: "O", x: 0.693, y: 0.693, z: 0 }, { element: "O", x: 0.807, y: 0.193, z: 0.5 }, { element: "O", x: 0.193, y: 0.807, z: 0.5 }], 136, "P4_2/mnm", "tetragonal", "rutile", -2.98, 71.66, 6.99),
  e("Fe2O3", 5.04, 5.04, 13.75, 90, 90, 120, [{ element: "Fe", x: 0, y: 0, z: 0.355 }, { element: "O", x: 0.306, y: 0, z: 0.25 }], 167, "R-3c", "rhombohedral", "corundum", -2.78, 301.73, 5.26),
  e("Cr2O3", 4.96, 4.96, 13.59, 90, 90, 120, [{ element: "Cr", x: 0, y: 0, z: 0.347 }, { element: "O", x: 0.306, y: 0, z: 0.25 }], 167, "R-3c", "rhombohedral", "corundum", -3.81, 289.60, 5.22),
  e("WO3", 7.31, 7.54, 7.69, 90, 90.9, 90, [{ element: "W", x: 0.25, y: 0.028, z: 0.283 }, { element: "O", x: 0.0, y: 0.037, z: 0.213 }, { element: "O", x: 0.26, y: 0.25, z: 0.284 }, { element: "O", x: 0.28, y: 0.035, z: 0.0 }], 14, "P2_1/c", "monoclinic", "WO3", -2.92, 424.17, 7.16),
  e("V2O5", 11.51, 3.56, 4.37, 90, 90, 90, [{ element: "V", x: 0.148, y: 0.25, z: 0.892 }, { element: "O", x: 0.105, y: 0.25, z: 0.531 }, { element: "O", x: 0.072, y: 0.25, z: 0.003 }, { element: "O", x: 0.319, y: 0.25, z: 0.0 }], 59, "Pmmn", "orthorhombic", "V2O5-layered", -3.06, 179.11, 3.36),
  e("MoS2", 3.16, 3.16, 12.30, 90, 90, 120, [{ element: "Mo", x: 0.333, y: 0.667, z: 0.25 }, { element: "S", x: 0.333, y: 0.667, z: 0.621 }, { element: "S", x: 0.333, y: 0.667, z: 0.879 }], 194, "P6_3/mmc", "hexagonal", "2H-MoS2", -1.27, 106.40, 5.06),
  e("WS2", 3.15, 3.15, 12.36, 90, 90, 120, [{ element: "W", x: 0.333, y: 0.667, z: 0.25 }, { element: "S", x: 0.333, y: 0.667, z: 0.622 }, { element: "S", x: 0.333, y: 0.667, z: 0.878 }], 194, "P6_3/mmc", "hexagonal", "2H-MoS2", -0.90, 106.08, 7.50),

  e("MgAl2O4", 8.08, 8.08, 8.08, 90, 90, 90, [{ element: "Mg", x: 0.125, y: 0.125, z: 0.125 }, { element: "Al", x: 0.5, y: 0.5, z: 0.5 }, { element: "Al", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0.263, y: 0.263, z: 0.263 }], 227, "Fd-3m", "cubic", "spinel", -4.55, 527.51, 3.58),
  e("Fe3O4", 8.40, 8.40, 8.40, 90, 90, 90, [{ element: "Fe", x: 0.125, y: 0.125, z: 0.125 }, { element: "Fe", x: 0.5, y: 0.5, z: 0.5 }, { element: "Fe", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0.255, y: 0.255, z: 0.255 }], 227, "Fd-3m", "cubic", "spinel-inverse", -2.83, 592.70, 5.18),
  e("NiFe2O4", 8.34, 8.34, 8.34, 90, 90, 90, [{ element: "Ni", x: 0.125, y: 0.125, z: 0.125 }, { element: "Fe", x: 0.5, y: 0.5, z: 0.5 }, { element: "Fe", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0.257, y: 0.257, z: 0.257 }], 227, "Fd-3m", "cubic", "spinel-inverse", -2.55, 580.12, 5.37),
  e("CoFe2O4", 8.39, 8.39, 8.39, 90, 90, 90, [{ element: "Co", x: 0.125, y: 0.125, z: 0.125 }, { element: "Fe", x: 0.5, y: 0.5, z: 0.5 }, { element: "Fe", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0.256, y: 0.256, z: 0.256 }], 227, "Fd-3m", "cubic", "spinel-inverse", -2.60, 590.59, 5.29),

  e("CaF2", 5.46, 5.46, 5.46, 90, 90, 90, [{ element: "Ca", x: 0, y: 0, z: 0 }, { element: "Ca", x: 0.5, y: 0.5, z: 0 }, { element: "Ca", x: 0.5, y: 0, z: 0.5 }, { element: "Ca", x: 0, y: 0.5, z: 0.5 }, { element: "F", x: 0.25, y: 0.25, z: 0.25 }, { element: "F", x: 0.75, y: 0.75, z: 0.25 }, { element: "F", x: 0.75, y: 0.25, z: 0.75 }, { element: "F", x: 0.25, y: 0.75, z: 0.75 }, { element: "F", x: 0.25, y: 0.75, z: 0.25 }, { element: "F", x: 0.75, y: 0.25, z: 0.25 }, { element: "F", x: 0.25, y: 0.25, z: 0.75 }, { element: "F", x: 0.75, y: 0.75, z: 0.75 }], 225, "Fm-3m", "cubic", "fluorite", -6.47, 40.71, 3.18),
  e("BaF2", 6.20, 6.20, 6.20, 90, 90, 90, [{ element: "Ba", x: 0, y: 0, z: 0 }, { element: "F", x: 0.25, y: 0.25, z: 0.25 }, { element: "F", x: 0.75, y: 0.75, z: 0.25 }], 225, "Fm-3m", "cubic", "fluorite", -6.10, 59.58, 4.89),
  e("CeO2", 5.41, 5.41, 5.41, 90, 90, 90, [{ element: "Ce", x: 0, y: 0, z: 0 }, { element: "O", x: 0.25, y: 0.25, z: 0.25 }, { element: "O", x: 0.75, y: 0.75, z: 0.25 }], 225, "Fm-3m", "cubic", "fluorite", -5.46, 39.63, 7.22),
  e("UO2", 5.47, 5.47, 5.47, 90, 90, 90, [{ element: "U", x: 0, y: 0, z: 0 }, { element: "O", x: 0.25, y: 0.25, z: 0.25 }, { element: "O", x: 0.75, y: 0.75, z: 0.25 }], 225, "Fm-3m", "cubic", "fluorite", -5.71, 40.88, 10.97),

  e("NdNiO2", 3.92, 3.92, 3.28, 90, 90, 90, [{ element: "Nd", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ni", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }], 123, "P4/mmm", "tetragonal", "infinite-layer", -1.95, 50.42, 7.95),
  e("LaNiO2", 3.96, 3.96, 3.38, 90, 90, 90, [{ element: "La", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ni", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }], 123, "P4/mmm", "tetragonal", "infinite-layer", -1.90, 53.0, 7.72),
  e("PrNiO2", 3.93, 3.93, 3.30, 90, 90, 90, [{ element: "Pr", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ni", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }], 123, "P4/mmm", "tetragonal", "infinite-layer", -1.92, 50.97, 8.10),

  e("NbSe2", 3.44, 3.44, 12.55, 90, 90, 120, [{ element: "Nb", x: 0.333, y: 0.667, z: 0.25 }, { element: "Se", x: 0.333, y: 0.667, z: 0.622 }, { element: "Se", x: 0.333, y: 0.667, z: 0.878 }], 194, "P6_3/mmc", "hexagonal", "2H-NbSe2", -0.88, 128.66, 6.33),
  e("TaS2", 3.31, 3.31, 12.10, 90, 90, 120, [{ element: "Ta", x: 0.333, y: 0.667, z: 0.25 }, { element: "S", x: 0.333, y: 0.667, z: 0.621 }, { element: "S", x: 0.333, y: 0.667, z: 0.879 }], 194, "P6_3/mmc", "hexagonal", "2H-TaS2", -0.75, 114.81, 8.20),

  e("K3C60", 14.24, 14.24, 14.24, 90, 90, 90, [{ element: "K", x: 0.25, y: 0.25, z: 0.25 }, { element: "K", x: 0.5, y: 0, z: 0 }, { element: "K", x: 0, y: 0.5, z: 0 }, { element: "C", x: 0, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "fulleride", -0.05, 2887.0, 1.95),

  e("LiCoO2", 2.82, 2.82, 14.05, 90, 90, 120, [{ element: "Li", x: 0, y: 0, z: 0.5 }, { element: "Co", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.260 }], 166, "R-3m", "rhombohedral", "alpha-NaFeO2", -2.29, 96.73, 5.05),
  e("LiFePO4", 10.33, 6.01, 4.69, 90, 90, 90, [{ element: "Li", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.282, y: 0.25, z: 0.974 }, { element: "P", x: 0.095, y: 0.25, z: 0.418 }, { element: "O", x: 0.097, y: 0.25, z: 0.743 }, { element: "O", x: 0.457, y: 0.25, z: 0.206 }, { element: "O", x: 0.166, y: 0.046, z: 0.285 }], 62, "Pnma", "orthorhombic", "olivine", -3.72, 291.39, 3.60),
  e("LiMn2O4", 8.25, 8.25, 8.25, 90, 90, 90, [{ element: "Li", x: 0.125, y: 0.125, z: 0.125 }, { element: "Mn", x: 0.5, y: 0.5, z: 0.5 }, { element: "Mn", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0.263, y: 0.263, z: 0.263 }], 227, "Fd-3m", "cubic", "spinel", -3.05, 561.52, 4.28),

  e("YNi2B2C", 3.53, 3.53, 10.54, 90, 90, 90, [{ element: "Y", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0, y: 0.5, z: 0.25 }, { element: "Ni", x: 0.5, y: 0, z: 0.25 }, { element: "B", x: 0, y: 0, z: 0.355 }, { element: "B", x: 0, y: 0, z: 0.645 }, { element: "C", x: 0, y: 0, z: 0.5 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2-like", -0.72, 131.36, 6.92),
  e("LuNi2B2C", 3.47, 3.47, 10.63, 90, 90, 90, [{ element: "Lu", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0, y: 0.5, z: 0.25 }, { element: "Ni", x: 0.5, y: 0, z: 0.25 }, { element: "B", x: 0, y: 0, z: 0.354 }, { element: "B", x: 0, y: 0, z: 0.646 }, { element: "C", x: 0, y: 0, z: 0.5 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2-like", -0.68, 128.0, 7.85),

  e("MgCu2", 7.04, 7.04, 7.04, 90, 90, 90, [{ element: "Mg", x: 0.125, y: 0.125, z: 0.125 }, { element: "Mg", x: 0.875, y: 0.875, z: 0.875 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0.5, y: 0.5, z: 0 }, { element: "Cu", x: 0.5, y: 0, z: 0.5 }, { element: "Cu", x: 0, y: 0.5, z: 0.5 }], 227, "Fd-3m", "cubic", "Laves-C15", -0.08, 348.72, 5.25),

  e("NiAs", 3.62, 3.62, 5.03, 90, 90, 120, [{ element: "Ni", x: 0, y: 0, z: 0 }, { element: "As", x: 0.333, y: 0.667, z: 0.25 }], 194, "P6_3/mmc", "hexagonal", "NiAs", -0.42, 57.07, 7.60),
  e("FeS", 3.44, 3.44, 5.88, 90, 90, 120, [{ element: "Fe", x: 0, y: 0, z: 0 }, { element: "S", x: 0.333, y: 0.667, z: 0.25 }], 194, "P6_3/mmc", "hexagonal", "NiAs", -0.67, 60.24, 4.84),

  e("VO2", 5.75, 4.52, 5.38, 90, 122.6, 90, [{ element: "V", x: 0.239, y: 0, z: 0.023 }, { element: "O", x: 0.1, y: 0.21, z: 0.2 }], 14, "P2_1/c", "monoclinic", "VO2-M1", -3.42, 116.38, 4.57),
  e("ReO3", 3.75, 3.75, 3.75, 90, 90, 90, [{ element: "Re", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "ReO3", -1.95, 52.73, 6.90),

  e("PbTiO3", 3.90, 3.90, 4.15, 90, 90, 90, [{ element: "Pb", x: 0, y: 0, z: 0 }, { element: "Ti", x: 0.5, y: 0.5, z: 0.53 }, { element: "O", x: 0.5, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0.62 }, { element: "O", x: 0, y: 0.5, z: 0.62 }], 99, "P4mm", "tetragonal", "perovskite", -2.60, 63.14, 8.06),
  e("BiFeO3", 5.58, 5.58, 13.87, 90, 90, 120, [{ element: "Bi", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0, y: 0, z: 0.221 }, { element: "O", x: 0.443, y: 0.013, z: 0.952 }], 161, "R3c", "rhombohedral", "perovskite-distorted", -2.30, 373.97, 8.34),

  e("NiO", 4.18, 4.18, 4.18, 90, 90, 90, [{ element: "Ni", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -2.49, 18.26, 6.72),
  e("CoO", 4.26, 4.26, 4.26, 90, 90, 90, [{ element: "Co", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -2.40, 19.30, 6.44),
  e("MnO", 4.44, 4.44, 4.44, 90, 90, 90, [{ element: "Mn", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -3.80, 21.87, 5.43),
  e("FeO", 4.33, 4.33, 4.33, 90, 90, 90, [{ element: "Fe", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -2.82, 20.30, 5.88),

  e("WC", 2.91, 2.91, 2.84, 90, 90, 120, [{ element: "W", x: 0, y: 0, z: 0 }, { element: "C", x: 0.333, y: 0.667, z: 0.5 }], 187, "P-6m2", "hexagonal", "WC", -0.39, 20.84, 15.63),
  e("TiC", 4.33, 4.33, 4.33, 90, 90, 90, [{ element: "Ti", x: 0, y: 0, z: 0 }, { element: "C", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -1.89, 20.30, 4.93),
  e("HfC", 4.64, 4.64, 4.64, 90, 90, 90, [{ element: "Hf", x: 0, y: 0, z: 0 }, { element: "C", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -2.15, 24.95, 12.67),
  e("ZrC", 4.70, 4.70, 4.70, 90, 90, 90, [{ element: "Zr", x: 0, y: 0, z: 0 }, { element: "C", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -2.01, 25.93, 6.73),
  e("VC", 4.17, 4.17, 4.17, 90, 90, 90, [{ element: "V", x: 0, y: 0, z: 0 }, { element: "C", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -1.10, 18.14, 5.77),
  e("MoC", 4.27, 4.27, 4.27, 90, 90, 90, [{ element: "Mo", x: 0, y: 0, z: 0 }, { element: "C", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -0.50, 19.47, 9.18),
  e("B4C", 5.16, 5.16, 12.07, 90, 90, 120, [{ element: "B", x: 0, y: 0, z: 0 }, { element: "B", x: 0.333, y: 0.667, z: 0.5 }, { element: "C", x: 0, y: 0, z: 0.38 }], 166, "R-3m", "rhombohedral", "B4C", -0.50, 278.47, 2.52),

  e("Si3N4", 7.75, 7.75, 5.62, 90, 90, 120, [{ element: "Si", x: 0.174, y: 0.769, z: 0.25 }, { element: "N", x: 0.031, y: 0.332, z: 0.25 }, { element: "N", x: 0.333, y: 0.667, z: 0.25 }], 176, "P6_3/m", "hexagonal", "beta-Si3N4", -3.07, 292.52, 3.18),

  e("CsCl", 4.12, 4.12, 4.12, 90, 90, 90, [{ element: "Cs", x: 0, y: 0, z: 0 }, { element: "Cl", x: 0.5, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "CsCl", -4.26, 69.93, 4.02),
  e("KBr", 6.60, 6.60, 6.60, 90, 90, 90, [{ element: "K", x: 0, y: 0, z: 0 }, { element: "Br", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -3.58, 35.96, 2.75),
  e("LiF", 4.03, 4.03, 4.03, 90, 90, 90, [{ element: "Li", x: 0, y: 0, z: 0 }, { element: "F", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -6.25, 16.34, 2.64),
  e("KCl", 6.29, 6.29, 6.29, 90, 90, 90, [{ element: "K", x: 0, y: 0, z: 0 }, { element: "Cl", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -4.36, 31.14, 1.98),
  e("NaF", 4.63, 4.63, 4.63, 90, 90, 90, [{ element: "Na", x: 0, y: 0, z: 0 }, { element: "F", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -5.76, 24.83, 2.79),
  e("BaO", 5.52, 5.52, 5.52, 90, 90, 90, [{ element: "Ba", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -5.45, 42.09, 5.72),
  e("SrO", 5.16, 5.16, 5.16, 90, 90, 90, [{ element: "Sr", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -6.00, 34.33, 4.70),

  e("NbTi", 3.29, 3.29, 3.29, 90, 90, 90, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "Ti", x: 0.5, y: 0.5, z: 0.5 }], 229, "Im-3m", "cubic", "BCC-alloy", -0.22, 17.81, 6.56),
  e("TiAl", 3.98, 3.98, 4.07, 90, 90, 90, [{ element: "Ti", x: 0, y: 0, z: 0 }, { element: "Al", x: 0.5, y: 0.5, z: 0.5 }], 123, "P4/mmm", "tetragonal", "L10", -0.42, 64.46, 3.76),
  e("FeAl", 2.91, 2.91, 2.91, 90, 90, 90, [{ element: "Fe", x: 0, y: 0, z: 0 }, { element: "Al", x: 0.5, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "CsCl-B2", -0.36, 24.64, 5.59),

  e("Cd2Re2O7", 10.23, 10.23, 10.23, 90, 90, 90, [{ element: "Cd", x: 0.125, y: 0.125, z: 0.125 }, { element: "Re", x: 0.5, y: 0.5, z: 0.5 }, { element: "O", x: 0.325, y: 0.125, z: 0.125 }, { element: "O", x: 0.375, y: 0.375, z: 0.375 }], 227, "Fd-3m", "cubic", "pyrochlore", -1.87, 1069.42, 8.51),

  e("LaB6", 4.16, 4.16, 4.16, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0 }, { element: "B", x: 0.199, y: 0.5, z: 0.5 }, { element: "B", x: 0.801, y: 0.5, z: 0.5 }, { element: "B", x: 0.5, y: 0.199, z: 0.5 }, { element: "B", x: 0.5, y: 0.801, z: 0.5 }, { element: "B", x: 0.5, y: 0.5, z: 0.199 }, { element: "B", x: 0.5, y: 0.5, z: 0.801 }], 221, "Pm-3m", "cubic", "CaB6", -0.85, 72.0, 4.72),
  e("CaB6", 4.15, 4.15, 4.15, 90, 90, 90, [{ element: "Ca", x: 0, y: 0, z: 0 }, { element: "B", x: 0.20, y: 0.5, z: 0.5 }, { element: "B", x: 0.80, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "CaB6", -0.78, 71.47, 2.46),
  e("SmB6", 4.13, 4.13, 4.13, 90, 90, 90, [{ element: "Sm", x: 0, y: 0, z: 0 }, { element: "B", x: 0.198, y: 0.5, z: 0.5 }, { element: "B", x: 0.802, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "CaB6", -0.90, 70.44, 5.07),

  e("Y2O3", 10.60, 10.60, 10.60, 90, 90, 90, [{ element: "Y", x: 0.25, y: 0.25, z: 0.25 }, { element: "O", x: 0.391, y: 0.152, z: 0.381 }], 206, "Ia-3", "cubic", "bixbyite", -4.78, 1191.02, 5.01),
  e("Li7La3Zr2O12", 12.97, 12.97, 12.97, 90, 90, 90, [{ element: "Li", x: 0.375, y: 0, z: 0.25 }, { element: "La", x: 0.125, y: 0, z: 0.25 }, { element: "Zr", x: 0, y: 0, z: 0 }, { element: "O", x: 0.281, y: 0.101, z: 0.197 }], 230, "Ia-3d", "cubic", "garnet", -3.92, 2181.48, 5.10),

  e("FeS2", 5.42, 5.42, 5.42, 90, 90, 90, [{ element: "Fe", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0.5, z: 0 }, { element: "Fe", x: 0.5, y: 0, z: 0.5 }, { element: "Fe", x: 0, y: 0.5, z: 0.5 }, { element: "S", x: 0.385, y: 0.385, z: 0.385 }, { element: "S", x: 0.615, y: 0.615, z: 0.615 }, { element: "S", x: 0.115, y: 0.885, z: 0.615 }, { element: "S", x: 0.885, y: 0.115, z: 0.385 }], 205, "Pa-3", "cubic", "pyrite", -0.53, 39.83, 5.02),
  e("Cu2O", 4.27, 4.27, 4.27, 90, 90, 90, [{ element: "Cu", x: 0.25, y: 0.25, z: 0.25 }, { element: "Cu", x: 0.75, y: 0.75, z: 0.75 }, { element: "O", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0.5, z: 0.5 }], 224, "Pn-3m", "cubic", "cuprite", -0.86, 77.83, 6.11),

  e("CdS", 5.83, 5.83, 5.83, 90, 90, 90, [{ element: "Cd", x: 0, y: 0, z: 0 }, { element: "S", x: 0.25, y: 0.25, z: 0.25 }], 216, "F-43m", "cubic", "zincblende", -1.20, 49.55, 4.82),
  e("PbS", 5.94, 5.94, 5.94, 90, 90, 90, [{ element: "Pb", x: 0, y: 0, z: 0 }, { element: "S", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -1.02, 52.43, 7.60),
  e("PbTe", 6.46, 6.46, 6.46, 90, 90, 90, [{ element: "Pb", x: 0, y: 0, z: 0 }, { element: "Te", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -0.55, 67.47, 8.16),
  e("PbSe", 6.12, 6.12, 6.12, 90, 90, 90, [{ element: "Pb", x: 0, y: 0, z: 0 }, { element: "Se", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -0.68, 57.33, 8.10),
  e("SnTe", 6.31, 6.31, 6.31, 90, 90, 90, [{ element: "Sn", x: 0, y: 0, z: 0 }, { element: "Te", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -0.42, 62.77, 6.45),

  e("CuInSe2", 5.78, 5.78, 11.62, 90, 90, 90, [{ element: "Cu", x: 0, y: 0, z: 0 }, { element: "In", x: 0, y: 0, z: 0.5 }, { element: "Se", x: 0.25, y: 0.25, z: 0.125 }, { element: "Se", x: 0.25, y: 0.75, z: 0.375 }], 122, "I-42d", "tetragonal", "chalcopyrite", -0.82, 388.31, 5.77),
  e("CuGaSe2", 5.61, 5.61, 11.01, 90, 90, 90, [{ element: "Cu", x: 0, y: 0, z: 0 }, { element: "Ga", x: 0, y: 0, z: 0.5 }, { element: "Se", x: 0.25, y: 0.25, z: 0.125 }, { element: "Se", x: 0.25, y: 0.75, z: 0.375 }], 122, "I-42d", "tetragonal", "chalcopyrite", -0.95, 346.30, 5.57),
  e("ZnSnP2", 5.65, 5.65, 11.30, 90, 90, 90, [{ element: "Zn", x: 0, y: 0, z: 0 }, { element: "Sn", x: 0, y: 0, z: 0.5 }, { element: "P", x: 0.25, y: 0.25, z: 0.125 }, { element: "P", x: 0.25, y: 0.75, z: 0.375 }], 122, "I-42d", "tetragonal", "chalcopyrite", -0.72, 360.66, 4.15),

  e("GdBa2Cu3O7", 3.83, 3.89, 11.71, 90, 90, 90, [{ element: "Gd", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.184 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.816 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0, y: 0, z: 0.356 }, { element: "Cu", x: 0, y: 0, z: 0.644 }, { element: "O", x: 0, y: 0.5, z: 0 }], 47, "Pmmm", "orthorhombic", "YBCO", -2.52, 174.31, 6.82),
  e("SmBa2Cu3O7", 3.84, 3.90, 11.73, 90, 90, 90, [{ element: "Sm", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.184 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.816 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0, y: 0, z: 0.356 }, { element: "Cu", x: 0, y: 0, z: 0.644 }, { element: "O", x: 0, y: 0.5, z: 0 }], 47, "Pmmm", "orthorhombic", "YBCO", -2.50, 175.69, 6.60),
  e("NdBa2Cu3O7", 3.86, 3.91, 11.76, 90, 90, 90, [{ element: "Nd", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.184 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.816 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0, y: 0, z: 0.356 }, { element: "Cu", x: 0, y: 0, z: 0.644 }, { element: "O", x: 0, y: 0.5, z: 0 }], 47, "Pmmm", "orthorhombic", "YBCO", -2.51, 177.52, 6.55),
  e("EuBa2Cu3O7", 3.83, 3.89, 11.70, 90, 90, 90, [{ element: "Eu", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.184 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.816 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "Cu", x: 0, y: 0, z: 0.356 }, { element: "Cu", x: 0, y: 0, z: 0.644 }, { element: "O", x: 0, y: 0.5, z: 0 }], 47, "Pmmm", "orthorhombic", "YBCO", -2.53, 174.16, 6.95),

  e("MgSiO3", 5.33, 5.18, 7.33, 90, 90, 90, [{ element: "Mg", x: 0.25, y: 0.016, z: 0.25 }, { element: "Si", x: 0.5, y: 0.5, z: 0 }, { element: "O", x: 0.15, y: 0.55, z: 0.16 }], 62, "Pnma", "orthorhombic", "perovskite", -3.89, 202.02, 3.22),

  e("BaSnO3", 4.12, 4.12, 4.12, 90, 90, 90, [{ element: "Ba", x: 0.5, y: 0.5, z: 0.5 }, { element: "Sn", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.10, 69.93, 7.24),
  e("SrVO3", 3.84, 3.84, 3.84, 90, 90, 90, [{ element: "Sr", x: 0.5, y: 0.5, z: 0.5 }, { element: "V", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.15, 56.62, 5.10),
  e("LaCoO3", 3.83, 3.83, 3.83, 90, 90, 90, [{ element: "La", x: 0.5, y: 0.5, z: 0.5 }, { element: "Co", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -2.75, 56.18, 7.28),
  e("LaMnO3", 5.54, 5.74, 7.72, 90, 90, 90, [{ element: "La", x: 0.05, y: 0.02, z: 0.25 }, { element: "Mn", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.49, y: 0.07, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -3.45, 245.05, 6.52),
  e("BaMnO3", 5.67, 5.67, 4.72, 90, 90, 120, [{ element: "Ba", x: 0.333, y: 0.667, z: 0.25 }, { element: "Mn", x: 0, y: 0, z: 0 }, { element: "O", x: 0.167, y: 0.333, z: 0 }], 194, "P6_3/mmc", "hexagonal", "perovskite-hexagonal", -3.10, 131.30, 5.80),

  e("MgSiN2", 5.27, 6.47, 4.98, 90, 90, 90, [{ element: "Mg", x: 0.079, y: 0.123, z: 0.25 }, { element: "Si", x: 0.422, y: 0.143, z: 0.25 }, { element: "N", x: 0.360, y: 0.353, z: 0.25 }, { element: "N", x: 0.128, y: 0.400, z: 0.25 }], 33, "Pna2_1", "orthorhombic", "ortho-MgSiN2", -2.34, 169.82, 3.16),
  e("Zn3N2", 9.77, 9.77, 9.77, 90, 90, 90, [{ element: "Zn", x: 0.067, y: 0.067, z: 0.067 }, { element: "N", x: 0.25, y: 0.25, z: 0.25 }], 206, "Ia-3", "cubic", "anti-bixbyite", -0.56, 932.35, 6.22),
  e("Cu3N", 3.82, 3.82, 3.82, 90, 90, 90, [{ element: "Cu", x: 0.25, y: 0, z: 0.5 }, { element: "Cu", x: 0, y: 0.5, z: 0.25 }, { element: "Cu", x: 0.5, y: 0.25, z: 0 }, { element: "N", x: 0, y: 0, z: 0 }], 221, "Pm-3m", "cubic", "anti-ReO3", -0.11, 55.74, 5.85),
  e("Ta2O5", 6.20, 3.66, 7.79, 90, 90, 90, [{ element: "Ta", x: 0.068, y: 0, z: 0.258 }, { element: "O", x: 0.174, y: 0.5, z: 0.115 }, { element: "O", x: 0.042, y: 0, z: 0.006 }], 25, "Pmm2", "orthorhombic", "beta-Ta2O5", -4.19, 176.45, 8.20),

  e("Sc2O3", 9.85, 9.85, 9.85, 90, 90, 90, [{ element: "Sc", x: 0.25, y: 0.25, z: 0.25 }, { element: "O", x: 0.391, y: 0.152, z: 0.381 }], 206, "Ia-3", "cubic", "bixbyite", -4.95, 955.67, 3.86),
  e("In2O3", 10.12, 10.12, 10.12, 90, 90, 90, [{ element: "In", x: 0.25, y: 0.25, z: 0.25 }, { element: "O", x: 0.389, y: 0.154, z: 0.383 }], 206, "Ia-3", "cubic", "bixbyite", -3.06, 1036.43, 7.18),
  e("Ga2O3", 12.23, 3.04, 5.80, 90, 103.8, 90, [{ element: "Ga", x: 0.090, y: 0, z: 0.794 }, { element: "Ga", x: 0.341, y: 0, z: 0.686 }, { element: "O", x: 0.163, y: 0, z: 0.109 }, { element: "O", x: 0.496, y: 0, z: 0.256 }, { element: "O", x: 0.828, y: 0, z: 0.436 }], 12, "C2/m", "monoclinic", "beta-Ga2O3", -3.44, 209.42, 5.88),

  e("Pd", 3.89, 3.89, 3.89, 90, 90, 90, [{ element: "Pd", x: 0, y: 0, z: 0 }, { element: "Pd", x: 0.5, y: 0.5, z: 0 }, { element: "Pd", x: 0.5, y: 0, z: 0.5 }, { element: "Pd", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 14.72, 12.02),
  e("Rh", 3.80, 3.80, 3.80, 90, 90, 90, [{ element: "Rh", x: 0, y: 0, z: 0 }, { element: "Rh", x: 0.5, y: 0.5, z: 0 }, { element: "Rh", x: 0.5, y: 0, z: 0.5 }, { element: "Rh", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 13.71, 12.41),
  e("Ir", 3.84, 3.84, 3.84, 90, 90, 90, [{ element: "Ir", x: 0, y: 0, z: 0 }, { element: "Ir", x: 0.5, y: 0.5, z: 0 }, { element: "Ir", x: 0.5, y: 0, z: 0.5 }, { element: "Ir", x: 0, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "FCC", 0, 14.15, 22.56),
  e("Co", 2.51, 2.51, 4.07, 90, 90, 120, [{ element: "Co", x: 0.333, y: 0.667, z: 0.25 }, { element: "Co", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 11.07, 8.90),
  e("Mn", 8.91, 8.91, 8.91, 90, 90, 90, [{ element: "Mn", x: 0.317, y: 0.317, z: 0.317 }], 217, "I-43m", "cubic", "alpha-Mn", 0, 354.10, 7.21),
  e("Zr", 3.23, 3.23, 5.15, 90, 90, 120, [{ element: "Zr", x: 0.333, y: 0.667, z: 0.25 }, { element: "Zr", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 23.28, 6.52),
  e("Hf", 3.19, 3.19, 5.05, 90, 90, 120, [{ element: "Hf", x: 0.333, y: 0.667, z: 0.25 }, { element: "Hf", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 22.16, 13.31),
  e("Re", 2.76, 2.76, 4.46, 90, 90, 120, [{ element: "Re", x: 0.333, y: 0.667, z: 0.25 }, { element: "Re", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 14.71, 21.02),
  e("Os", 2.73, 2.73, 4.32, 90, 90, 120, [{ element: "Os", x: 0.333, y: 0.667, z: 0.25 }, { element: "Os", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 13.96, 22.59),
  e("Ru", 2.71, 2.71, 4.28, 90, 90, 120, [{ element: "Ru", x: 0.333, y: 0.667, z: 0.25 }, { element: "Ru", x: 0.667, y: 0.333, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "HCP", 0, 13.57, 12.37),
  e("SrFeO3", 3.85, 3.85, 3.85, 90, 90, 90, [{ element: "Sr", x: 0.5, y: 0.5, z: 0.5 }, { element: "Fe", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -2.90, 57.07, 5.45),
  e("BaHfO3", 4.17, 4.17, 4.17, 90, 90, 90, [{ element: "Ba", x: 0.5, y: 0.5, z: 0.5 }, { element: "Hf", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.50, 72.51, 8.42),
  e("Li2O", 4.61, 4.61, 4.61, 90, 90, 90, [{ element: "Li", x: 0.25, y: 0.25, z: 0.25 }, { element: "Li", x: 0.75, y: 0.75, z: 0.75 }, { element: "O", x: 0, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "anti-fluorite", -6.09, 24.50, 2.01),
  e("Na2O", 5.56, 5.56, 5.56, 90, 90, 90, [{ element: "Na", x: 0.25, y: 0.25, z: 0.25 }, { element: "Na", x: 0.75, y: 0.75, z: 0.75 }, { element: "O", x: 0, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "anti-fluorite", -4.33, 42.99, 2.27),
];

for (const entry of SEED_DATA) {
  dataset.push(entry);
}

export function getDatasetStats(): {
  totalCount: number;
  byCrystalSystem: Record<string, number>;
  byPrototype: Record<string, number>;
  bySource: Record<string, number>;
} {
  const byCrystalSystem: Record<string, number> = {};
  const byPrototype: Record<string, number> = {};
  const bySource: Record<string, number> = {};

  for (const entry of dataset) {
    byCrystalSystem[entry.crystalSystem] = (byCrystalSystem[entry.crystalSystem] || 0) + 1;
    byPrototype[entry.prototype] = (byPrototype[entry.prototype] || 0) + 1;
    bySource[entry.source] = (bySource[entry.source] || 0) + 1;
  }

  return { totalCount: dataset.length, byCrystalSystem, byPrototype, bySource };
}

export function getTrainingData(): CrystalStructureEntry[] {
  return [...dataset];
}

export function getEntriesByPrototype(prototype: string): CrystalStructureEntry[] {
  return dataset.filter(e => e.prototype.toLowerCase().includes(prototype.toLowerCase()));
}

export function getEntriesBySystem(crystalSystem: string): CrystalStructureEntry[] {
  return dataset.filter(e => e.crystalSystem.toLowerCase() === crystalSystem.toLowerCase());
}

export function getEntryByFormula(formula: string): CrystalStructureEntry | undefined {
  return dataset.find(e => e.formula === formula);
}

export async function fetchMPStructures(count: number = 50): Promise<number> {
  if (!isMPAvailable()) return 0;

  const formulasToTry = [
    "SrTiO3", "BaTiO3", "LaAlO3", "MgB2", "NaCl", "LiF", "CaF2",
    "ZnO", "TiO2", "Fe2O3", "Al2O3", "SiO2", "CeO2", "SnO2",
    "GaAs", "InP", "GaN", "SiC", "CdTe", "ZnS",
    "NbN", "TiN", "VN", "TaC", "ZrN", "HfN",
    "Nb3Sn", "V3Si", "MgZn2", "CaCu5",
    "LiCoO2", "LiFePO4", "LiMn2O4",
    "BaZrO3", "PbTiO3", "BiFeO3",
    "Bi2Se3", "Bi2Te3", "CoSb3",
    "Fe3O4", "MgAl2O4", "WC", "TiC",
    "FeS2", "Cu2O", "PbS", "PbTe",
    "NiTi", "FeAl", "TiAl",
  ];

  let added = 0;
  for (const formula of formulasToTry.slice(0, count)) {
    if (dataset.some(d => d.formula === formula)) continue;

    try {
      const summary = await fetchSummary(formula);
      if (!summary) continue;

      const entry: CrystalStructureEntry = {
        formula,
        lattice: { a: 4.0, b: 4.0, c: 4.0, alpha: 90, beta: 90, gamma: 90 },
        atomicPositions: [{ element: formula.match(/[A-Z][a-z]*/)?.[0] || "X", x: 0, y: 0, z: 0 }],
        spacegroup: 0,
        spacegroupSymbol: summary.spaceGroup || "",
        crystalSystem: summary.crystalSystem || "cubic",
        prototype: "MP-imported",
        formationEnergy: summary.formationEnergyPerAtom || 0,
        volume: summary.volume || 0,
        density: summary.density || 0,
        nsites: summary.nsites || 1,
        source: "Materials Project",
      };

      dataset.push(entry);
      added++;
    } catch {
    }
  }
  return added;
}

export async function fetchOQMDStructures(count: number = 50): Promise<number> {
  let added = 0;
  try {
    const url = `http://oqmd.org/oqmdapi/formationenergy?fields=name,entry_id,spacegroup,band_gap,stability,delta_e,composition&limit=${count}&offset=0&format=json`;
    const resp = await fetch(url, { signal: AbortSignal.timeout(20000) });
    if (!resp.ok) return 0;

    const data = await resp.json() as any;
    const entries = data?.data ?? [];

    for (const oe of entries) {
      if (!oe.name || !oe.composition) continue;
      if (dataset.some(d => d.formula === oe.composition)) continue;

      const entry: CrystalStructureEntry = {
        formula: oe.composition || oe.name,
        lattice: { a: 4.0, b: 4.0, c: 4.0, alpha: 90, beta: 90, gamma: 90 },
        atomicPositions: [{ element: (oe.composition || oe.name).match(/[A-Z][a-z]*/)?.[0] || "X", x: 0, y: 0, z: 0 }],
        spacegroup: 0,
        spacegroupSymbol: oe.spacegroup || "",
        crystalSystem: "unknown",
        prototype: "OQMD-imported",
        formationEnergy: oe.delta_e ?? 0,
        volume: 0,
        density: 0,
        nsites: 1,
        source: "OQMD",
      };

      dataset.push(entry);
      added++;
    }
  } catch {
  }
  return added;
}
