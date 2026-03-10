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

  e("ScH10", 4.92, 4.92, 4.92, 90, 90, 90, [{ element: "Sc", x: 0, y: 0, z: 0 }, { element: "H", x: 0.12, y: 0.12, z: 0.12 }, { element: "H", x: 0.88, y: 0.88, z: 0.88 }, { element: "H", x: 0.38, y: 0.38, z: 0.38 }, { element: "H", x: 0.62, y: 0.62, z: 0.62 }, { element: "H", x: 0.25, y: 0, z: 0 }, { element: "H", x: 0, y: 0.25, z: 0 }, { element: "H", x: 0, y: 0, z: 0.25 }], 225, "Fm-3m", "cubic", "sodalite-clathrate", -0.04, 119.10, 3.10),
  e("AcH10", 5.30, 5.30, 5.30, 90, 90, 90, [{ element: "Ac", x: 0, y: 0, z: 0 }, { element: "H", x: 0.12, y: 0.12, z: 0.12 }, { element: "H", x: 0.88, y: 0.88, z: 0.88 }, { element: "H", x: 0.38, y: 0.38, z: 0.38 }, { element: "H", x: 0.62, y: 0.62, z: 0.62 }, { element: "H", x: 0.25, y: 0, z: 0 }, { element: "H", x: 0, y: 0.25, z: 0 }], 225, "Fm-3m", "cubic", "sodalite-clathrate", -0.02, 148.88, 5.20),
  e("SrH6", 3.68, 3.68, 3.68, 90, 90, 90, [{ element: "Sr", x: 0, y: 0, z: 0 }, { element: "H", x: 0.165, y: 0.165, z: 0.165 }, { element: "H", x: 0.835, y: 0.835, z: 0.835 }, { element: "H", x: 0.165, y: 0.835, z: 0.5 }, { element: "H", x: 0.835, y: 0.165, z: 0.5 }, { element: "H", x: 0.5, y: 0.165, z: 0.835 }, { element: "H", x: 0.5, y: 0.835, z: 0.165 }], 229, "Im-3m", "cubic", "clathrate", -0.03, 49.84, 3.80),
  e("BaH6", 3.82, 3.82, 3.82, 90, 90, 90, [{ element: "Ba", x: 0, y: 0, z: 0 }, { element: "H", x: 0.165, y: 0.165, z: 0.165 }, { element: "H", x: 0.835, y: 0.835, z: 0.835 }, { element: "H", x: 0.165, y: 0.835, z: 0.5 }, { element: "H", x: 0.835, y: 0.165, z: 0.5 }, { element: "H", x: 0.5, y: 0.165, z: 0.835 }, { element: "H", x: 0.5, y: 0.835, z: 0.165 }], 229, "Im-3m", "cubic", "clathrate", -0.02, 55.74, 4.40),
  e("MgH6", 3.35, 3.35, 3.35, 90, 90, 90, [{ element: "Mg", x: 0, y: 0, z: 0 }, { element: "H", x: 0.165, y: 0.165, z: 0.165 }, { element: "H", x: 0.835, y: 0.835, z: 0.835 }, { element: "H", x: 0.165, y: 0.835, z: 0.5 }, { element: "H", x: 0.835, y: 0.165, z: 0.5 }, { element: "H", x: 0.5, y: 0.165, z: 0.835 }], 229, "Im-3m", "cubic", "clathrate", -0.01, 37.60, 2.20),
  e("ScH9", 3.55, 3.55, 5.30, 90, 90, 120, [{ element: "Sc", x: 0, y: 0, z: 0 }, { element: "H", x: 0.333, y: 0.667, z: 0.12 }, { element: "H", x: 0.667, y: 0.333, z: 0.88 }, { element: "H", x: 0.167, y: 0.333, z: 0.5 }, { element: "H", x: 0.833, y: 0.667, z: 0.5 }, { element: "H", x: 0.12, y: 0.06, z: 0.25 }, { element: "H", x: 0.88, y: 0.94, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "clathrate", -0.03, 57.82, 3.50),
  e("ZrH10", 5.05, 5.05, 5.05, 90, 90, 90, [{ element: "Zr", x: 0, y: 0, z: 0 }, { element: "H", x: 0.12, y: 0.12, z: 0.12 }, { element: "H", x: 0.88, y: 0.88, z: 0.88 }, { element: "H", x: 0.38, y: 0.38, z: 0.38 }, { element: "H", x: 0.62, y: 0.62, z: 0.62 }, { element: "H", x: 0.25, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "sodalite-clathrate", -0.04, 128.79, 4.50),
  e("HfH10", 4.98, 4.98, 4.98, 90, 90, 90, [{ element: "Hf", x: 0, y: 0, z: 0 }, { element: "H", x: 0.12, y: 0.12, z: 0.12 }, { element: "H", x: 0.88, y: 0.88, z: 0.88 }, { element: "H", x: 0.38, y: 0.38, z: 0.38 }, { element: "H", x: 0.62, y: 0.62, z: 0.62 }], 225, "Fm-3m", "cubic", "sodalite-clathrate", -0.03, 123.51, 7.20),

  e("Nb3Ir", 5.15, 5.15, 5.15, 90, 90, 90, [{ element: "Nb", x: 0.25, y: 0, z: 0.5 }, { element: "Nb", x: 0.75, y: 0, z: 0.5 }, { element: "Nb", x: 0.5, y: 0.25, z: 0 }, { element: "Nb", x: 0.5, y: 0.75, z: 0 }, { element: "Nb", x: 0, y: 0.5, z: 0.25 }, { element: "Nb", x: 0, y: 0.5, z: 0.75 }, { element: "Ir", x: 0, y: 0, z: 0 }, { element: "Ir", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.40, 136.59, 12.30),
  e("V3Ge", 4.77, 4.77, 4.77, 90, 90, 90, [{ element: "V", x: 0.25, y: 0, z: 0.5 }, { element: "V", x: 0.75, y: 0, z: 0.5 }, { element: "V", x: 0.5, y: 0.25, z: 0 }, { element: "V", x: 0.5, y: 0.75, z: 0 }, { element: "V", x: 0, y: 0.5, z: 0.25 }, { element: "V", x: 0, y: 0.5, z: 0.75 }, { element: "Ge", x: 0, y: 0, z: 0 }, { element: "Ge", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.33, 108.53, 6.85),
  e("Ti3Ir", 5.06, 5.06, 5.06, 90, 90, 90, [{ element: "Ti", x: 0.25, y: 0, z: 0.5 }, { element: "Ti", x: 0.75, y: 0, z: 0.5 }, { element: "Ti", x: 0.5, y: 0.25, z: 0 }, { element: "Ti", x: 0.5, y: 0.75, z: 0 }, { element: "Ti", x: 0, y: 0.5, z: 0.25 }, { element: "Ti", x: 0, y: 0.5, z: 0.75 }, { element: "Ir", x: 0, y: 0, z: 0 }, { element: "Ir", x: 0.5, y: 0.5, z: 0.5 }], 223, "Pm-3n", "cubic", "A15", -0.50, 129.55, 10.80),

  e("SrFe2As2", 3.93, 3.93, 12.36, 90, 90, 90, [{ element: "Sr", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0, z: 0.25 }, { element: "Fe", x: 0, y: 0.5, z: 0.25 }, { element: "As", x: 0, y: 0, z: 0.360 }, { element: "As", x: 0, y: 0, z: 0.640 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2", -1.42, 190.92, 5.98),
  e("CaFe2As2", 3.88, 3.88, 11.74, 90, 90, 90, [{ element: "Ca", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0, z: 0.25 }, { element: "Fe", x: 0, y: 0.5, z: 0.25 }, { element: "As", x: 0, y: 0, z: 0.366 }, { element: "As", x: 0, y: 0, z: 0.634 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2", -1.35, 176.84, 5.64),
  e("KFe2As2", 3.84, 3.84, 13.87, 90, 90, 90, [{ element: "K", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0, z: 0.25 }, { element: "Fe", x: 0, y: 0.5, z: 0.25 }, { element: "As", x: 0, y: 0, z: 0.352 }, { element: "As", x: 0, y: 0, z: 0.648 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2", -1.28, 204.42, 4.72),
  e("CeFeAsO", 4.00, 4.00, 8.65, 90, 90, 90, [{ element: "Ce", x: 0, y: 0, z: 0.140 }, { element: "Fe", x: 0.5, y: 0, z: 0.5 }, { element: "As", x: 0, y: 0, z: 0.653 }, { element: "O", x: 0.5, y: 0.5, z: 0 }], 129, "P4/nmm", "tetragonal", "ZrCuSiAs", -2.20, 138.40, 6.45),
  e("GdFeAsO", 3.92, 3.92, 8.48, 90, 90, 90, [{ element: "Gd", x: 0, y: 0, z: 0.138 }, { element: "Fe", x: 0.5, y: 0, z: 0.5 }, { element: "As", x: 0, y: 0, z: 0.660 }, { element: "O", x: 0.5, y: 0.5, z: 0 }], 129, "P4/nmm", "tetragonal", "ZrCuSiAs", -2.35, 130.28, 7.30),
  e("FeSe0.5Te0.5", 3.80, 3.80, 6.10, 90, 90, 90, [{ element: "Fe", x: 0.75, y: 0.25, z: 0 }, { element: "Se", x: 0.25, y: 0.25, z: 0.255 }, { element: "Te", x: 0.25, y: 0.25, z: 0.280 }], 129, "P4/nmm", "tetragonal", "PbO-type", -0.40, 88.12, 5.90),
  e("NaFeAs", 3.95, 3.95, 7.04, 90, 90, 90, [{ element: "Na", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.5, y: 0, z: 0.5 }, { element: "As", x: 0, y: 0, z: 0.208 }], 129, "P4/nmm", "tetragonal", "PbFCl-type", -1.18, 109.82, 4.62),

  e("La2CuO4Sr", 3.76, 3.76, 13.15, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0.361 }, { element: "Sr", x: 0, y: 0, z: 0.639 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.182 }, { element: "O", x: 0, y: 0, z: 0.818 }], 139, "I4/mmm", "tetragonal", "K2NiF4", -2.85, 185.88, 6.95),
  e("Tl2Ba2Ca2Cu3O10", 3.85, 3.85, 35.60, 90, 90, 90, [{ element: "Tl", x: 0, y: 0, z: 0.035 }, { element: "Tl", x: 0.5, y: 0.5, z: 0.965 }, { element: "Ba", x: 0, y: 0, z: 0.105 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.895 }, { element: "Ca", x: 0, y: 0, z: 0.21 }, { element: "Ca", x: 0.5, y: 0.5, z: 0.79 }, { element: "Cu", x: 0, y: 0, z: 0.165 }, { element: "Cu", x: 0.5, y: 0.5, z: 0.835 }, { element: "Cu", x: 0, y: 0, z: 0.285 }], 139, "I4/mmm", "tetragonal", "Tl-cuprate", -2.15, 527.43, 7.50),
  e("Bi2Sr2Ca2Cu3O10", 5.41, 5.41, 37.15, 90, 90, 90, [{ element: "Bi", x: 0, y: 0, z: 0.04 }, { element: "Bi", x: 0.5, y: 0.5, z: 0.96 }, { element: "Sr", x: 0, y: 0, z: 0.12 }, { element: "Sr", x: 0.5, y: 0.5, z: 0.88 }, { element: "Ca", x: 0, y: 0, z: 0.20 }, { element: "Ca", x: 0.5, y: 0.5, z: 0.80 }, { element: "Cu", x: 0, y: 0, z: 0.16 }, { element: "Cu", x: 0.5, y: 0.5, z: 0.84 }, { element: "Cu", x: 0, y: 0, z: 0.25 }], 139, "I4/mmm", "tetragonal", "BSCCO", -2.05, 1088.0, 6.35),
  e("HgBa2CuO4", 3.88, 3.88, 9.51, 90, 90, 90, [{ element: "Hg", x: 0, y: 0, z: 0 }, { element: "Ba", x: 0.5, y: 0.5, z: 0.293 }, { element: "Cu", x: 0, y: 0, z: 0.5 }, { element: "O", x: 0, y: 0.5, z: 0.5 }, { element: "O", x: 0.5, y: 0.5, z: 0 }], 123, "P4/mmm", "tetragonal", "Hg-cuprate", -1.78, 143.28, 7.55),

  e("SnMo6S8", 6.52, 6.52, 6.52, 89.3, 89.3, 89.3, [{ element: "Sn", x: 0, y: 0, z: 0 }, { element: "Mo", x: 0.40, y: 0.22, z: 0.07 }, { element: "Mo", x: 0.07, y: 0.40, z: 0.22 }, { element: "Mo", x: 0.22, y: 0.07, z: 0.40 }, { element: "S", x: 0.26, y: 0.37, z: 0.62 }, { element: "S", x: 0.62, y: 0.26, z: 0.37 }, { element: "S", x: 0.37, y: 0.62, z: 0.26 }], 148, "R-3", "rhombohedral", "Chevrel", -0.60, 277.43, 7.05),
  e("CuMo6S8", 6.48, 6.48, 6.48, 89.1, 89.1, 89.1, [{ element: "Cu", x: 0, y: 0, z: 0 }, { element: "Mo", x: 0.40, y: 0.22, z: 0.07 }, { element: "Mo", x: 0.07, y: 0.40, z: 0.22 }, { element: "Mo", x: 0.22, y: 0.07, z: 0.40 }, { element: "S", x: 0.26, y: 0.37, z: 0.62 }, { element: "S", x: 0.62, y: 0.26, z: 0.37 }, { element: "S", x: 0.37, y: 0.62, z: 0.26 }], 148, "R-3", "rhombohedral", "Chevrel", -0.58, 271.51, 6.85),
  e("TlMo6Se8", 6.88, 6.88, 6.88, 89.4, 89.4, 89.4, [{ element: "Tl", x: 0, y: 0, z: 0 }, { element: "Mo", x: 0.40, y: 0.22, z: 0.07 }, { element: "Mo", x: 0.07, y: 0.40, z: 0.22 }, { element: "Mo", x: 0.22, y: 0.07, z: 0.40 }, { element: "Se", x: 0.26, y: 0.37, z: 0.62 }, { element: "Se", x: 0.62, y: 0.26, z: 0.37 }, { element: "Se", x: 0.37, y: 0.62, z: 0.26 }], 148, "R-3", "rhombohedral", "Chevrel", -0.52, 325.66, 8.30),

  e("CeIrIn5", 4.69, 4.69, 7.51, 90, 90, 90, [{ element: "Ce", x: 0, y: 0, z: 0 }, { element: "Ir", x: 0.5, y: 0.5, z: 0 }, { element: "In", x: 0.5, y: 0, z: 0.308 }, { element: "In", x: 0, y: 0.5, z: 0.308 }, { element: "In", x: 0, y: 0, z: 0.5 }], 123, "P4/mmm", "tetragonal", "HoCoGa5", -0.50, 165.11, 9.12),
  e("CeRhIn5", 4.66, 4.66, 7.54, 90, 90, 90, [{ element: "Ce", x: 0, y: 0, z: 0 }, { element: "Rh", x: 0.5, y: 0.5, z: 0 }, { element: "In", x: 0.5, y: 0, z: 0.306 }, { element: "In", x: 0, y: 0.5, z: 0.306 }, { element: "In", x: 0, y: 0, z: 0.5 }], 123, "P4/mmm", "tetragonal", "HoCoGa5", -0.52, 163.74, 8.75),
  e("PuCoGa5", 4.24, 4.24, 6.73, 90, 90, 90, [{ element: "Pu", x: 0, y: 0, z: 0 }, { element: "Co", x: 0.5, y: 0.5, z: 0 }, { element: "Ga", x: 0.5, y: 0, z: 0.312 }, { element: "Ga", x: 0, y: 0.5, z: 0.312 }, { element: "Ga", x: 0, y: 0, z: 0.5 }], 123, "P4/mmm", "tetragonal", "HoCoGa5", -0.48, 120.90, 10.50),

  e("CeRu2Si2", 4.13, 4.13, 9.80, 90, 90, 90, [{ element: "Ce", x: 0, y: 0, z: 0 }, { element: "Ru", x: 0, y: 0.5, z: 0.25 }, { element: "Ru", x: 0.5, y: 0, z: 0.25 }, { element: "Si", x: 0, y: 0, z: 0.375 }, { element: "Si", x: 0, y: 0, z: 0.625 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2", -0.80, 166.95, 7.52),
  e("URu2Si2", 4.13, 4.13, 9.58, 90, 90, 90, [{ element: "U", x: 0, y: 0, z: 0 }, { element: "Ru", x: 0, y: 0.5, z: 0.25 }, { element: "Ru", x: 0.5, y: 0, z: 0.25 }, { element: "Si", x: 0, y: 0, z: 0.371 }, { element: "Si", x: 0, y: 0, z: 0.629 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2", -0.85, 163.48, 10.10),
  e("UPt3", 5.76, 5.76, 4.90, 90, 90, 120, [{ element: "U", x: 0.333, y: 0.667, z: 0.25 }, { element: "Pt", x: 0.167, y: 0.333, z: 0.25 }, { element: "Pt", x: 0.333, y: 0.167, z: 0.75 }, { element: "Pt", x: 0.5, y: 0, z: 0.25 }], 194, "P6_3/mmc", "hexagonal", "Ni3Sn", -0.35, 140.80, 19.40),

  e("Bi2Rh3S2", 8.48, 8.48, 5.78, 90, 90, 90, [{ element: "Bi", x: 0.333, y: 0.667, z: 0 }, { element: "Rh", x: 0.145, y: 0.290, z: 0.25 }, { element: "Rh", x: 0.855, y: 0.710, z: 0.75 }, { element: "S", x: 0.333, y: 0.667, z: 0.5 }], 194, "P6_3/mmc", "hexagonal", "parkerite", -0.62, 360.0, 8.95),
  e("Bi4Rh", 8.81, 8.81, 4.10, 90, 90, 90, [{ element: "Bi", x: 0.137, y: 0.344, z: 0 }, { element: "Bi", x: 0.414, y: 0.165, z: 0.5 }, { element: "Rh", x: 0, y: 0, z: 0 }], 123, "P4/mmm", "tetragonal", "Bi4Rh-topo", -0.28, 318.17, 10.70),

  e("Cd3As2", 12.63, 12.63, 25.43, 90, 90, 90, [{ element: "Cd", x: 0.071, y: 0.071, z: 0.071 }, { element: "As", x: 0.125, y: 0.125, z: 0.125 }], 142, "I4_1/acd", "tetragonal", "Dirac-semimetal", -0.22, 4057.0, 6.21),
  e("Na3Bi", 5.45, 5.45, 9.66, 90, 90, 120, [{ element: "Na", x: 0, y: 0, z: 0 }, { element: "Na", x: 0.333, y: 0.667, z: 0.25 }, { element: "Bi", x: 0.333, y: 0.667, z: 0.75 }], 194, "P6_3/mmc", "hexagonal", "Dirac-semimetal", -0.41, 248.50, 4.82),
  e("TaAs", 3.44, 3.44, 11.64, 90, 90, 90, [{ element: "Ta", x: 0, y: 0, z: 0 }, { element: "As", x: 0, y: 0.5, z: 0.417 }], 109, "I4_1md", "tetragonal", "Weyl-semimetal", -0.75, 137.65, 11.35),
  e("NbAs", 3.45, 3.45, 11.68, 90, 90, 90, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "As", x: 0, y: 0.5, z: 0.417 }], 109, "I4_1md", "tetragonal", "Weyl-semimetal", -0.60, 139.02, 7.52),
  e("TaP", 3.32, 3.32, 11.35, 90, 90, 90, [{ element: "Ta", x: 0, y: 0, z: 0 }, { element: "P", x: 0, y: 0.5, z: 0.417 }], 109, "I4_1md", "tetragonal", "Weyl-semimetal", -0.82, 125.10, 11.08),
  e("NbP", 3.33, 3.33, 11.37, 90, 90, 90, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "P", x: 0, y: 0.5, z: 0.417 }], 109, "I4_1md", "tetragonal", "Weyl-semimetal", -0.65, 126.08, 7.15),

  e("RuO2", 4.49, 4.49, 3.11, 90, 90, 90, [{ element: "Ru", x: 0, y: 0, z: 0 }, { element: "Ru", x: 0.5, y: 0.5, z: 0.5 }, { element: "O", x: 0.305, y: 0.305, z: 0 }, { element: "O", x: 0.695, y: 0.695, z: 0 }], 136, "P4_2/mnm", "tetragonal", "rutile", -1.68, 62.73, 6.97),
  e("IrO2", 4.51, 4.51, 3.16, 90, 90, 90, [{ element: "Ir", x: 0, y: 0, z: 0 }, { element: "Ir", x: 0.5, y: 0.5, z: 0.5 }, { element: "O", x: 0.307, y: 0.307, z: 0 }, { element: "O", x: 0.693, y: 0.693, z: 0 }], 136, "P4_2/mnm", "tetragonal", "rutile", -1.52, 64.20, 11.68),
  e("MnO2", 4.40, 4.40, 2.87, 90, 90, 90, [{ element: "Mn", x: 0, y: 0, z: 0 }, { element: "Mn", x: 0.5, y: 0.5, z: 0.5 }, { element: "O", x: 0.303, y: 0.303, z: 0 }, { element: "O", x: 0.697, y: 0.697, z: 0 }], 136, "P4_2/mnm", "tetragonal", "rutile", -2.51, 55.51, 5.03),

  e("LiNbO3", 5.15, 5.15, 13.86, 90, 90, 120, [{ element: "Li", x: 0, y: 0, z: 0.282 }, { element: "Nb", x: 0, y: 0, z: 0 }, { element: "O", x: 0.048, y: 0.344, z: 0.063 }], 161, "R3c", "rhombohedral", "LiNbO3-ferroelectric", -3.15, 318.33, 4.64),
  e("LiTaO3", 5.15, 5.15, 13.78, 90, 90, 120, [{ element: "Li", x: 0, y: 0, z: 0.280 }, { element: "Ta", x: 0, y: 0, z: 0 }, { element: "O", x: 0.050, y: 0.346, z: 0.065 }], 161, "R3c", "rhombohedral", "LiNbO3-ferroelectric", -3.45, 316.14, 7.46),

  e("ScN", 4.50, 4.50, 4.50, 90, 90, 90, [{ element: "Sc", x: 0, y: 0, z: 0 }, { element: "N", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -4.12, 11.39, 4.27),
  e("CrN", 4.15, 4.15, 4.15, 90, 90, 90, [{ element: "Cr", x: 0, y: 0, z: 0 }, { element: "N", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -1.68, 8.93, 6.14),
  e("MoN", 4.21, 4.21, 4.21, 90, 90, 90, [{ element: "Mo", x: 0, y: 0, z: 0 }, { element: "N", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -0.82, 9.35, 9.20),
  e("WN", 4.13, 4.13, 4.13, 90, 90, 90, [{ element: "W", x: 0, y: 0, z: 0 }, { element: "N", x: 0.5, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "NaCl", -0.60, 8.82, 15.72),
  e("AlN2", 3.07, 3.07, 4.91, 90, 90, 120, [{ element: "Al", x: 0.333, y: 0.667, z: 0 }, { element: "N", x: 0.333, y: 0.667, z: 0.38 }], 186, "P6_3mc", "hexagonal", "wurtzite", -3.40, 20.12, 3.30),

  e("Fe2B", 5.11, 5.11, 4.25, 90, 90, 90, [{ element: "Fe", x: 0.172, y: 0.672, z: 0.25 }, { element: "Fe", x: 0.828, y: 0.328, z: 0.75 }, { element: "B", x: 0, y: 0, z: 0 }], 140, "I4/mcm", "tetragonal", "CuAl2-type", -0.30, 110.93, 7.42),
  e("Ni3B", 5.23, 6.62, 4.39, 90, 90, 90, [{ element: "Ni", x: 0.176, y: 0.063, z: 0.25 }, { element: "Ni", x: 0.040, y: 0.319, z: 0.25 }, { element: "B", x: 0.389, y: 0.438, z: 0.25 }], 62, "Pnma", "orthorhombic", "cementite-like", -0.21, 152.05, 8.60),
  e("Co2B", 5.02, 5.02, 4.22, 90, 90, 90, [{ element: "Co", x: 0.170, y: 0.670, z: 0.25 }, { element: "Co", x: 0.830, y: 0.330, z: 0.75 }, { element: "B", x: 0, y: 0, z: 0 }], 140, "I4/mcm", "tetragonal", "CuAl2-type", -0.35, 106.35, 8.05),

  e("Ni2MnSn", 6.05, 6.05, 6.05, 90, 90, 90, [{ element: "Ni", x: 0.25, y: 0.25, z: 0.25 }, { element: "Ni", x: 0.75, y: 0.75, z: 0.75 }, { element: "Mn", x: 0, y: 0, z: 0 }, { element: "Sn", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.28, 221.45, 8.35),
  e("Fe2VAl", 5.76, 5.76, 5.76, 90, 90, 90, [{ element: "Fe", x: 0.25, y: 0.25, z: 0.25 }, { element: "Fe", x: 0.75, y: 0.75, z: 0.75 }, { element: "V", x: 0, y: 0, z: 0 }, { element: "Al", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.40, 191.10, 6.55),
  e("Co2TiSn", 6.07, 6.07, 6.07, 90, 90, 90, [{ element: "Co", x: 0.25, y: 0.25, z: 0.25 }, { element: "Co", x: 0.75, y: 0.75, z: 0.75 }, { element: "Ti", x: 0, y: 0, z: 0 }, { element: "Sn", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.45, 223.67, 7.88),
  e("Ni2MnIn", 6.07, 6.07, 6.07, 90, 90, 90, [{ element: "Ni", x: 0.25, y: 0.25, z: 0.25 }, { element: "Ni", x: 0.75, y: 0.75, z: 0.75 }, { element: "Mn", x: 0, y: 0, z: 0 }, { element: "In", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.30, 223.67, 8.25),
  e("Co2MnGe", 5.75, 5.75, 5.75, 90, 90, 90, [{ element: "Co", x: 0.25, y: 0.25, z: 0.25 }, { element: "Co", x: 0.75, y: 0.75, z: 0.75 }, { element: "Mn", x: 0, y: 0, z: 0 }, { element: "Ge", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.52, 190.11, 8.12),

  e("CoSi", 4.44, 4.44, 4.44, 90, 90, 90, [{ element: "Co", x: 0.144, y: 0.144, z: 0.144 }, { element: "Si", x: 0.844, y: 0.844, z: 0.844 }], 198, "P2_13", "cubic", "FeSi-chiral", -0.62, 87.53, 6.59),
  e("RhSi", 4.69, 4.69, 4.69, 90, 90, 90, [{ element: "Rh", x: 0.146, y: 0.146, z: 0.146 }, { element: "Si", x: 0.846, y: 0.846, z: 0.846 }], 198, "P2_13", "cubic", "FeSi-chiral", -0.72, 103.16, 7.78),
  e("FeSi", 4.49, 4.49, 4.49, 90, 90, 90, [{ element: "Fe", x: 0.137, y: 0.137, z: 0.137 }, { element: "Si", x: 0.842, y: 0.842, z: 0.842 }], 198, "P2_13", "cubic", "FeSi-chiral", -0.55, 90.52, 6.32),

  e("MoSe2", 3.29, 3.29, 12.93, 90, 90, 120, [{ element: "Mo", x: 0.333, y: 0.667, z: 0.25 }, { element: "Se", x: 0.333, y: 0.667, z: 0.628 }, { element: "Se", x: 0.333, y: 0.667, z: 0.872 }], 194, "P6_3/mmc", "hexagonal", "2H-MoS2", -0.95, 121.08, 6.90),
  e("WSe2", 3.28, 3.28, 12.96, 90, 90, 120, [{ element: "W", x: 0.333, y: 0.667, z: 0.25 }, { element: "Se", x: 0.333, y: 0.667, z: 0.629 }, { element: "Se", x: 0.333, y: 0.667, z: 0.871 }], 194, "P6_3/mmc", "hexagonal", "2H-MoS2", -0.72, 120.73, 9.32),
  e("NbS2", 3.33, 3.33, 11.95, 90, 90, 120, [{ element: "Nb", x: 0.333, y: 0.667, z: 0.25 }, { element: "S", x: 0.333, y: 0.667, z: 0.618 }, { element: "S", x: 0.333, y: 0.667, z: 0.882 }], 194, "P6_3/mmc", "hexagonal", "2H-NbS2", -0.82, 114.70, 4.50),
  e("TaSe2", 3.48, 3.48, 12.72, 90, 90, 120, [{ element: "Ta", x: 0.333, y: 0.667, z: 0.25 }, { element: "Se", x: 0.333, y: 0.667, z: 0.624 }, { element: "Se", x: 0.333, y: 0.667, z: 0.876 }], 194, "P6_3/mmc", "hexagonal", "2H-TaS2", -0.68, 133.45, 8.75),
  e("TiS2", 3.41, 3.41, 5.70, 90, 90, 120, [{ element: "Ti", x: 0, y: 0, z: 0 }, { element: "S", x: 0.333, y: 0.667, z: 0.25 }, { element: "S", x: 0.667, y: 0.333, z: 0.75 }], 164, "P-3m1", "hexagonal", "1T-TiS2", -1.15, 57.43, 3.22),
  e("TiSe2", 3.54, 3.54, 6.01, 90, 90, 120, [{ element: "Ti", x: 0, y: 0, z: 0 }, { element: "Se", x: 0.333, y: 0.667, z: 0.25 }, { element: "Se", x: 0.667, y: 0.333, z: 0.75 }], 164, "P-3m1", "hexagonal", "1T-TiS2", -0.88, 65.20, 4.50),
  e("ZrSe2", 3.77, 3.77, 6.13, 90, 90, 120, [{ element: "Zr", x: 0, y: 0, z: 0 }, { element: "Se", x: 0.333, y: 0.667, z: 0.25 }, { element: "Se", x: 0.667, y: 0.333, z: 0.75 }], 164, "P-3m1", "hexagonal", "1T-TiS2", -1.10, 75.46, 5.62),
  e("HfS2", 3.63, 3.63, 5.85, 90, 90, 120, [{ element: "Hf", x: 0, y: 0, z: 0 }, { element: "S", x: 0.333, y: 0.667, z: 0.25 }, { element: "S", x: 0.667, y: 0.333, z: 0.75 }], 164, "P-3m1", "hexagonal", "1T-TiS2", -1.25, 66.77, 6.30),

  e("SrRuO3Thin", 3.93, 3.93, 3.93, 90, 90, 90, [{ element: "Sr", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ru", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -2.70, 60.70, 6.40),
  e("CaMnO3", 5.28, 5.27, 7.46, 90, 90, 90, [{ element: "Ca", x: 0.032, y: 0.058, z: 0.25 }, { element: "Mn", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.490, y: 0.060, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -3.65, 207.53, 4.18),
  e("YAlO3", 5.18, 5.33, 7.37, 90, 90, 90, [{ element: "Y", x: 0.019, y: 0.052, z: 0.25 }, { element: "Al", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.470, y: 0.085, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -4.22, 203.47, 5.35),
  e("GdScO3", 5.45, 5.74, 7.93, 90, 90, 90, [{ element: "Gd", x: 0.020, y: 0.055, z: 0.25 }, { element: "Sc", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.460, y: 0.090, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -3.95, 248.03, 6.72),
  e("DyScO3", 5.44, 5.72, 7.89, 90, 90, 90, [{ element: "Dy", x: 0.022, y: 0.055, z: 0.25 }, { element: "Sc", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.462, y: 0.088, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -3.98, 245.68, 6.95),
  e("NdGaO3", 5.43, 5.50, 7.71, 90, 90, 90, [{ element: "Nd", x: 0.018, y: 0.048, z: 0.25 }, { element: "Ga", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.475, y: 0.082, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -3.52, 230.18, 7.10),

  e("Sr2RuO4", 3.87, 3.87, 12.74, 90, 90, 90, [{ element: "Sr", x: 0, y: 0, z: 0.353 }, { element: "Sr", x: 0, y: 0, z: 0.647 }, { element: "Ru", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.161 }, { element: "O", x: 0, y: 0, z: 0.839 }], 139, "I4/mmm", "tetragonal", "K2NiF4", -2.55, 190.67, 5.95),
  e("Sr3Ru2O7", 3.89, 3.89, 20.73, 90, 90, 90, [{ element: "Sr", x: 0, y: 0, z: 0.5 }, { element: "Sr", x: 0, y: 0, z: 0.312 }, { element: "Ru", x: 0, y: 0, z: 0.097 }, { element: "O", x: 0, y: 0.5, z: 0.097 }, { element: "O", x: 0, y: 0, z: 0.198 }, { element: "O", x: 0, y: 0, z: 0 }], 139, "I4/mmm", "tetragonal", "Ruddlesden-Popper", -2.42, 313.56, 5.88),
  e("Ca3Ti2O7", 5.41, 5.41, 19.52, 90, 90, 90, [{ element: "Ca", x: 0, y: 0, z: 0.5 }, { element: "Ca", x: 0, y: 0, z: 0.311 }, { element: "Ti", x: 0, y: 0, z: 0.098 }, { element: "O", x: 0, y: 0.5, z: 0.098 }, { element: "O", x: 0, y: 0, z: 0.199 }, { element: "O", x: 0, y: 0, z: 0 }], 139, "I4/mmm", "tetragonal", "Ruddlesden-Popper", -3.72, 571.25, 3.98),
  e("La2NiO4", 3.86, 3.86, 12.68, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0.360 }, { element: "La", x: 0, y: 0, z: 0.640 }, { element: "Ni", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.175 }], 139, "I4/mmm", "tetragonal", "K2NiF4", -2.45, 188.88, 7.15),

  e("Sr2IrO4", 3.89, 3.89, 12.90, 90, 90, 90, [{ element: "Sr", x: 0, y: 0, z: 0.353 }, { element: "Sr", x: 0, y: 0, z: 0.647 }, { element: "Ir", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.165 }], 139, "I4/mmm", "tetragonal", "K2NiF4", -2.30, 195.21, 8.05),
  e("Ba2IrO4", 3.98, 3.98, 13.25, 90, 90, 90, [{ element: "Ba", x: 0, y: 0, z: 0.357 }, { element: "Ba", x: 0, y: 0, z: 0.643 }, { element: "Ir", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.170 }], 139, "I4/mmm", "tetragonal", "K2NiF4", -2.15, 209.82, 8.90),

  e("NbSi2", 4.81, 4.81, 6.59, 90, 90, 120, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "Si", x: 0.333, y: 0, z: 0.25 }, { element: "Si", x: 0, y: 0.333, z: 0.25 }], 180, "P6_222", "hexagonal", "C40-NbSi2", -0.65, 132.10, 5.68),
  e("TaSi2", 4.78, 4.78, 6.57, 90, 90, 120, [{ element: "Ta", x: 0, y: 0, z: 0 }, { element: "Si", x: 0.333, y: 0, z: 0.25 }, { element: "Si", x: 0, y: 0.333, z: 0.25 }], 180, "P6_222", "hexagonal", "C40-NbSi2", -0.70, 129.98, 9.14),
  e("MoSi2", 3.20, 3.20, 7.86, 90, 90, 90, [{ element: "Mo", x: 0, y: 0, z: 0 }, { element: "Si", x: 0, y: 0, z: 0.333 }, { element: "Si", x: 0, y: 0, z: 0.667 }], 139, "I4/mmm", "tetragonal", "C11b-MoSi2", -0.55, 80.46, 6.26),
  e("WSi2", 3.21, 3.21, 7.83, 90, 90, 90, [{ element: "W", x: 0, y: 0, z: 0 }, { element: "Si", x: 0, y: 0, z: 0.333 }, { element: "Si", x: 0, y: 0, z: 0.667 }], 139, "I4/mmm", "tetragonal", "C11b-MoSi2", -0.52, 80.73, 9.86),
  e("TiSi2", 8.27, 4.80, 8.55, 90, 90, 90, [{ element: "Ti", x: 0.373, y: 0.25, z: 0.456 }, { element: "Si", x: 0.066, y: 0.25, z: 0.272 }, { element: "Si", x: 0.327, y: 0.25, z: 0.766 }], 63, "Cmcm", "orthorhombic", "C54-TiSi2", -0.60, 339.30, 4.04),
  e("CrSi2", 4.43, 4.43, 6.37, 90, 90, 120, [{ element: "Cr", x: 0, y: 0, z: 0 }, { element: "Si", x: 0.333, y: 0, z: 0.25 }, { element: "Si", x: 0, y: 0.333, z: 0.25 }], 180, "P6_222", "hexagonal", "C40-NbSi2", -0.55, 108.25, 4.91),

  e("Nd2CuO4", 3.94, 3.94, 12.17, 90, 90, 90, [{ element: "Nd", x: 0, y: 0, z: 0.351 }, { element: "Nd", x: 0, y: 0, z: 0.649 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.175 }], 139, "I4/mmm", "tetragonal", "Nd2CuO4-type", -2.82, 188.94, 7.45),
  e("Pr2CuO4", 3.96, 3.96, 12.22, 90, 90, 90, [{ element: "Pr", x: 0, y: 0, z: 0.352 }, { element: "Pr", x: 0, y: 0, z: 0.648 }, { element: "Cu", x: 0, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0, z: 0.176 }], 139, "I4/mmm", "tetragonal", "Nd2CuO4-type", -2.78, 191.58, 7.30),

  e("LaOBiS2", 4.07, 4.07, 13.85, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0.090 }, { element: "O", x: 0.5, y: 0.5, z: 0 }, { element: "Bi", x: 0, y: 0, z: 0.380 }, { element: "S", x: 0, y: 0.5, z: 0.290 }, { element: "S", x: 0.5, y: 0, z: 0.145 }], 129, "P4/nmm", "tetragonal", "BiS2-layer", -1.85, 229.30, 6.12),
  e("CeOBiS2", 4.05, 4.05, 13.52, 90, 90, 90, [{ element: "Ce", x: 0, y: 0, z: 0.092 }, { element: "O", x: 0.5, y: 0.5, z: 0 }, { element: "Bi", x: 0, y: 0, z: 0.382 }, { element: "S", x: 0, y: 0.5, z: 0.288 }, { element: "S", x: 0.5, y: 0, z: 0.143 }], 129, "P4/nmm", "tetragonal", "BiS2-layer", -1.80, 221.82, 6.38),
  e("NdOBiS2", 4.01, 4.01, 13.40, 90, 90, 90, [{ element: "Nd", x: 0, y: 0, z: 0.091 }, { element: "O", x: 0.5, y: 0.5, z: 0 }, { element: "Bi", x: 0, y: 0, z: 0.381 }, { element: "S", x: 0, y: 0.5, z: 0.289 }, { element: "S", x: 0.5, y: 0, z: 0.144 }], 129, "P4/nmm", "tetragonal", "BiS2-layer", -1.82, 215.52, 6.55),

  e("MgNi3C", 3.81, 3.81, 3.81, 90, 90, 90, [{ element: "Mg", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0.5, y: 0.5, z: 0 }, { element: "Ni", x: 0.5, y: 0, z: 0.5 }, { element: "Ni", x: 0, y: 0.5, z: 0.5 }, { element: "C", x: 0.5, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "anti-perovskite-SC", -0.25, 55.31, 6.82),
  e("SrPd3O", 3.93, 3.93, 3.93, 90, 90, 90, [{ element: "Sr", x: 0, y: 0, z: 0 }, { element: "Pd", x: 0.5, y: 0.5, z: 0 }, { element: "Pd", x: 0.5, y: 0, z: 0.5 }, { element: "Pd", x: 0, y: 0.5, z: 0.5 }, { element: "O", x: 0.5, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "anti-perovskite-SC", -0.30, 60.70, 7.95),

  e("RbC60", 14.38, 14.38, 14.38, 90, 90, 90, [{ element: "Rb", x: 0.25, y: 0.25, z: 0.25 }, { element: "Rb", x: 0.5, y: 0, z: 0 }, { element: "Rb", x: 0, y: 0.5, z: 0 }, { element: "C", x: 0, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "fulleride", -0.04, 2975.0, 2.30),
  e("CsRb2C60", 14.55, 14.55, 14.55, 90, 90, 90, [{ element: "Cs", x: 0.25, y: 0.25, z: 0.25 }, { element: "Rb", x: 0.5, y: 0, z: 0 }, { element: "Rb", x: 0, y: 0.5, z: 0 }, { element: "C", x: 0, y: 0, z: 0 }], 225, "Fm-3m", "cubic", "fulleride", -0.03, 3081.0, 2.50),

  e("YB6", 4.10, 4.10, 4.10, 90, 90, 90, [{ element: "Y", x: 0, y: 0, z: 0 }, { element: "B", x: 0.198, y: 0.5, z: 0.5 }, { element: "B", x: 0.802, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "CaB6", -0.82, 68.92, 3.72),
  e("BaB6", 4.27, 4.27, 4.27, 90, 90, 90, [{ element: "Ba", x: 0, y: 0, z: 0 }, { element: "B", x: 0.202, y: 0.5, z: 0.5 }, { element: "B", x: 0.798, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "CaB6", -0.72, 77.79, 3.14),
  e("EuB6", 4.18, 4.18, 4.18, 90, 90, 90, [{ element: "Eu", x: 0, y: 0, z: 0 }, { element: "B", x: 0.200, y: 0.5, z: 0.5 }, { element: "B", x: 0.800, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "CaB6", -0.68, 73.03, 4.92),

  e("Mn3GaN", 3.90, 3.90, 3.90, 90, 90, 90, [{ element: "Mn", x: 0, y: 0.5, z: 0.5 }, { element: "Mn", x: 0.5, y: 0, z: 0.5 }, { element: "Mn", x: 0.5, y: 0.5, z: 0 }, { element: "Ga", x: 0, y: 0, z: 0 }, { element: "N", x: 0.5, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "anti-perovskite", -0.38, 59.32, 6.92),
  e("Mn3SnN", 3.99, 3.99, 3.99, 90, 90, 90, [{ element: "Mn", x: 0, y: 0.5, z: 0.5 }, { element: "Mn", x: 0.5, y: 0, z: 0.5 }, { element: "Mn", x: 0.5, y: 0.5, z: 0 }, { element: "Sn", x: 0, y: 0, z: 0 }, { element: "N", x: 0.5, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "anti-perovskite", -0.32, 63.52, 7.18),

  e("YPd2Sn", 6.62, 6.62, 6.62, 90, 90, 90, [{ element: "Y", x: 0, y: 0, z: 0 }, { element: "Pd", x: 0.25, y: 0.25, z: 0.25 }, { element: "Pd", x: 0.75, y: 0.75, z: 0.75 }, { element: "Sn", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.55, 290.32, 8.42),
  e("LaPt2In", 6.58, 6.58, 6.58, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0 }, { element: "Pt", x: 0.25, y: 0.25, z: 0.25 }, { element: "Pt", x: 0.75, y: 0.75, z: 0.75 }, { element: "In", x: 0.5, y: 0.5, z: 0.5 }], 225, "Fm-3m", "cubic", "Heusler-L21", -0.48, 284.66, 12.60),

  e("SrTiO3Thin", 3.91, 3.91, 3.91, 90, 90, 90, [{ element: "Sr", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ti", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.35, 59.78, 5.13),
  e("KTaO3", 3.99, 3.99, 3.99, 90, 90, 90, [{ element: "K", x: 0.5, y: 0.5, z: 0.5 }, { element: "Ta", x: 0, y: 0, z: 0 }, { element: "O", x: 0.5, y: 0, z: 0 }, { element: "O", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0, y: 0, z: 0.5 }], 221, "Pm-3m", "cubic", "perovskite", -3.52, 63.52, 7.02),

  e("NiS2", 5.69, 5.69, 5.69, 90, 90, 90, [{ element: "Ni", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0.5, y: 0.5, z: 0 }, { element: "S", x: 0.395, y: 0.395, z: 0.395 }, { element: "S", x: 0.605, y: 0.605, z: 0.605 }], 205, "Pa-3", "cubic", "pyrite", -0.38, 46.04, 4.44),
  e("CoS2", 5.54, 5.54, 5.54, 90, 90, 90, [{ element: "Co", x: 0, y: 0, z: 0 }, { element: "Co", x: 0.5, y: 0.5, z: 0 }, { element: "S", x: 0.390, y: 0.390, z: 0.390 }, { element: "S", x: 0.610, y: 0.610, z: 0.610 }], 205, "Pa-3", "cubic", "pyrite", -0.42, 42.50, 4.27),
  e("MnS2", 6.10, 6.10, 6.10, 90, 90, 90, [{ element: "Mn", x: 0, y: 0, z: 0 }, { element: "Mn", x: 0.5, y: 0.5, z: 0 }, { element: "S", x: 0.400, y: 0.400, z: 0.400 }, { element: "S", x: 0.600, y: 0.600, z: 0.600 }], 205, "Pa-3", "cubic", "pyrite", -0.28, 56.80, 3.46),

  e("Nb2O5", 6.17, 29.18, 3.94, 90, 90, 90, [{ element: "Nb", x: 0.08, y: 0.09, z: 0 }, { element: "O", x: 0.40, y: 0.20, z: 0 }], 30, "Pnc2", "orthorhombic", "T-Nb2O5", -3.82, 710.15, 4.60),
  e("MoO3", 3.96, 13.86, 3.70, 90, 90, 90, [{ element: "Mo", x: 0.092, y: 0.101, z: 0.25 }, { element: "O", x: 0.500, y: 0.085, z: 0.25 }, { element: "O", x: 0.080, y: 0.215, z: 0.25 }, { element: "O", x: 0.430, y: 0.002, z: 0.25 }], 62, "Pnma", "orthorhombic", "MoO3-layered", -2.58, 203.10, 4.70),

  e("CrSb2", 6.03, 6.87, 3.27, 90, 90, 90, [{ element: "Cr", x: 0, y: 0, z: 0 }, { element: "Sb", x: 0.185, y: 0.353, z: 0 }], 58, "Pnnm", "orthorhombic", "marcasite", -0.38, 135.46, 7.15),
  e("FeSb2", 5.83, 6.54, 3.20, 90, 90, 90, [{ element: "Fe", x: 0, y: 0, z: 0 }, { element: "Sb", x: 0.187, y: 0.355, z: 0 }], 58, "Pnnm", "orthorhombic", "marcasite", -0.32, 121.94, 7.42),

  e("YB2C2", 3.54, 3.54, 7.60, 90, 90, 90, [{ element: "Y", x: 0, y: 0, z: 0 }, { element: "B", x: 0, y: 0, z: 0.35 }, { element: "B", x: 0, y: 0, z: 0.65 }, { element: "C", x: 0, y: 0, z: 0.5 }, { element: "C", x: 0.5, y: 0.5, z: 0 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2-like", -0.95, 95.14, 4.28),
  e("ScNiBC", 3.48, 3.48, 7.42, 90, 90, 90, [{ element: "Sc", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0, y: 0.5, z: 0.25 }, { element: "B", x: 0, y: 0, z: 0.352 }, { element: "C", x: 0, y: 0, z: 0.5 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2-like", -0.70, 89.86, 5.15),

  e("La3Ni2O7", 3.84, 3.84, 20.52, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0.5 }, { element: "La", x: 0, y: 0, z: 0.312 }, { element: "La", x: 0, y: 0, z: 0.098 }, { element: "Ni", x: 0, y: 0, z: 0.196 }, { element: "Ni", x: 0, y: 0, z: 0.392 }, { element: "O", x: 0, y: 0.5, z: 0.196 }, { element: "O", x: 0, y: 0, z: 0.098 }], 139, "I4/mmm", "tetragonal", "Ruddlesden-Popper", -2.38, 302.44, 6.92),
  e("La4Ni3O10", 3.83, 3.83, 28.02, 90, 90, 90, [{ element: "La", x: 0, y: 0, z: 0 }, { element: "La", x: 0, y: 0, z: 0.143 }, { element: "Ni", x: 0, y: 0, z: 0.071 }, { element: "Ni", x: 0, y: 0, z: 0.214 }, { element: "O", x: 0, y: 0.5, z: 0.071 }, { element: "O", x: 0, y: 0, z: 0.143 }], 139, "I4/mmm", "tetragonal", "Ruddlesden-Popper", -2.32, 411.15, 6.85),

  e("CaC6", 4.33, 4.33, 13.58, 90, 90, 120, [{ element: "Ca", x: 0, y: 0, z: 0 }, { element: "C", x: 0.333, y: 0.667, z: 0.5 }, { element: "C", x: 0.167, y: 0.833, z: 0.5 }, { element: "C", x: 0.833, y: 0.167, z: 0.5 }, { element: "C", x: 0.667, y: 0.333, z: 0.5 }, { element: "C", x: 0.5, y: 0.5, z: 0.5 }, { element: "C", x: 0.5, y: 0, z: 0.5 }], 166, "R-3m", "rhombohedral", "graphite-intercalated", -0.12, 220.15, 2.55),
  e("YbC6", 4.32, 4.32, 13.32, 90, 90, 120, [{ element: "Yb", x: 0, y: 0, z: 0 }, { element: "C", x: 0.333, y: 0.667, z: 0.5 }, { element: "C", x: 0.167, y: 0.833, z: 0.5 }, { element: "C", x: 0.833, y: 0.167, z: 0.5 }], 166, "R-3m", "rhombohedral", "graphite-intercalated", -0.10, 215.42, 4.62),

  e("Nb2SnC", 3.22, 3.22, 13.80, 90, 90, 120, [{ element: "Nb", x: 0.333, y: 0.667, z: 0.083 }, { element: "Nb", x: 0.667, y: 0.333, z: 0.417 }, { element: "Sn", x: 0, y: 0, z: 0.25 }, { element: "C", x: 0, y: 0, z: 0 }], 194, "P6_3/mmc", "hexagonal", "MAX-phase", -0.60, 123.88, 7.85),
  e("Ti3AlC2", 3.07, 3.07, 18.58, 90, 90, 120, [{ element: "Ti", x: 0.333, y: 0.667, z: 0.125 }, { element: "Ti", x: 0.333, y: 0.667, z: 0.375 }, { element: "Al", x: 0, y: 0, z: 0.25 }, { element: "C", x: 0, y: 0, z: 0.0 }, { element: "C", x: 0.333, y: 0.667, z: 0.5 }], 194, "P6_3/mmc", "hexagonal", "MAX-phase", -1.95, 151.48, 4.25),
  e("Ti2AlC", 3.05, 3.05, 13.64, 90, 90, 120, [{ element: "Ti", x: 0.333, y: 0.667, z: 0.083 }, { element: "Ti", x: 0.667, y: 0.333, z: 0.417 }, { element: "Al", x: 0, y: 0, z: 0.25 }, { element: "C", x: 0, y: 0, z: 0 }], 194, "P6_3/mmc", "hexagonal", "MAX-phase", -1.52, 109.82, 4.11),
  e("V2AlC", 2.91, 2.91, 13.19, 90, 90, 120, [{ element: "V", x: 0.333, y: 0.667, z: 0.083 }, { element: "V", x: 0.667, y: 0.333, z: 0.417 }, { element: "Al", x: 0, y: 0, z: 0.25 }, { element: "C", x: 0, y: 0, z: 0 }], 194, "P6_3/mmc", "hexagonal", "MAX-phase", -1.30, 96.75, 4.88),
  e("Cr2AlC", 2.86, 2.86, 12.83, 90, 90, 120, [{ element: "Cr", x: 0.333, y: 0.667, z: 0.083 }, { element: "Cr", x: 0.667, y: 0.333, z: 0.417 }, { element: "Al", x: 0, y: 0, z: 0.25 }, { element: "C", x: 0, y: 0, z: 0 }], 194, "P6_3/mmc", "hexagonal", "MAX-phase", -1.18, 90.85, 5.24),

  e("Nb2Pd3Se8", 6.92, 6.92, 13.85, 90, 90, 90, [{ element: "Nb", x: 0.25, y: 0.25, z: 0 }, { element: "Pd", x: 0, y: 0, z: 0.25 }, { element: "Pd", x: 0.5, y: 0, z: 0.25 }, { element: "Se", x: 0.15, y: 0.35, z: 0.12 }], 139, "I4/mmm", "tetragonal", "layered-chalcogenide", -0.55, 663.35, 7.28),
  e("Ta2NiSe5", 3.50, 12.83, 15.64, 90, 90, 90, [{ element: "Ta", x: 0.068, y: 0.382, z: 0.25 }, { element: "Ni", x: 0, y: 0, z: 0 }, { element: "Se", x: 0.225, y: 0.128, z: 0.25 }, { element: "Se", x: 0.410, y: 0.412, z: 0.25 }, { element: "Se", x: 0.125, y: 0.258, z: 0.25 }], 63, "Cmcm", "orthorhombic", "excitonic-insulator", -0.72, 702.08, 8.25),

  e("NdNiO3", 5.39, 5.38, 7.61, 90, 90, 90, [{ element: "Nd", x: 0.020, y: 0.053, z: 0.25 }, { element: "Ni", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.490, y: 0.070, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -2.42, 220.43, 7.62),
  e("SmNiO3", 5.33, 5.43, 7.57, 90, 90, 90, [{ element: "Sm", x: 0.018, y: 0.050, z: 0.25 }, { element: "Ni", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.492, y: 0.068, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -2.50, 219.08, 7.85),
  e("EuNiO3", 5.28, 5.46, 7.53, 90, 90, 90, [{ element: "Eu", x: 0.019, y: 0.051, z: 0.25 }, { element: "Ni", x: 0, y: 0.5, z: 0 }, { element: "O", x: 0.491, y: 0.069, z: 0.25 }], 62, "Pnma", "orthorhombic", "perovskite", -2.48, 216.92, 7.98),

  e("Ca2N", 3.62, 3.62, 19.10, 90, 90, 120, [{ element: "Ca", x: 0, y: 0, z: 0.067 }, { element: "Ca", x: 0.333, y: 0.667, z: 0.133 }, { element: "N", x: 0, y: 0, z: 0 }], 166, "R-3m", "rhombohedral", "anti-CdCl2-electride", -0.85, 216.82, 2.35),
  e("Sr2N", 3.80, 3.80, 20.05, 90, 90, 120, [{ element: "Sr", x: 0, y: 0, z: 0.065 }, { element: "Sr", x: 0.333, y: 0.667, z: 0.130 }, { element: "N", x: 0, y: 0, z: 0 }], 166, "R-3m", "rhombohedral", "anti-CdCl2-electride", -0.78, 250.72, 3.72),

  e("ZrNiSn", 6.11, 6.11, 6.11, 90, 90, 90, [{ element: "Zr", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0.25, y: 0.25, z: 0.25 }, { element: "Sn", x: 0.5, y: 0.5, z: 0.5 }], 216, "F-43m", "cubic", "half-Heusler", -0.50, 227.98, 7.58),
  e("TiCoSb", 5.88, 5.88, 5.88, 90, 90, 90, [{ element: "Ti", x: 0, y: 0, z: 0 }, { element: "Co", x: 0.25, y: 0.25, z: 0.25 }, { element: "Sb", x: 0.5, y: 0.5, z: 0.5 }], 216, "F-43m", "cubic", "half-Heusler", -0.55, 203.30, 7.25),
  e("HfNiSn", 6.08, 6.08, 6.08, 90, 90, 90, [{ element: "Hf", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0.25, y: 0.25, z: 0.25 }, { element: "Sn", x: 0.5, y: 0.5, z: 0.5 }], 216, "F-43m", "cubic", "half-Heusler", -0.52, 224.68, 10.55),
  e("NbFeSb", 5.95, 5.95, 5.95, 90, 90, 90, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "Fe", x: 0.25, y: 0.25, z: 0.25 }, { element: "Sb", x: 0.5, y: 0.5, z: 0.5 }], 216, "F-43m", "cubic", "half-Heusler", -0.48, 210.68, 7.82),

  e("Bi2Te2Se", 4.37, 4.37, 29.75, 90, 90, 120, [{ element: "Bi", x: 0, y: 0, z: 0.400 }, { element: "Te", x: 0, y: 0, z: 0 }, { element: "Se", x: 0, y: 0, z: 0.210 }], 166, "R-3m", "rhombohedral", "Bi2Te3", -0.22, 492.15, 7.52),
  e("Bi2Se2Te", 4.20, 4.20, 29.50, 90, 90, 120, [{ element: "Bi", x: 0, y: 0, z: 0.399 }, { element: "Se", x: 0, y: 0, z: 0 }, { element: "Te", x: 0, y: 0, z: 0.211 }], 166, "R-3m", "rhombohedral", "Bi2Te3", -0.20, 450.82, 7.32),

  e("CuAlO2", 2.86, 2.86, 16.95, 90, 90, 120, [{ element: "Cu", x: 0, y: 0, z: 0 }, { element: "Al", x: 0, y: 0, z: 0.352 }, { element: "O", x: 0.333, y: 0.667, z: 0.556 }], 166, "R-3m", "rhombohedral", "delafossite", -2.85, 120.42, 5.10),
  e("PtCoO2", 2.83, 2.83, 17.74, 90, 90, 120, [{ element: "Pt", x: 0, y: 0, z: 0 }, { element: "Co", x: 0, y: 0, z: 0.350 }, { element: "O", x: 0.333, y: 0.667, z: 0.555 }], 166, "R-3m", "rhombohedral", "delafossite", -1.55, 123.08, 10.25),
  e("PdCoO2", 2.83, 2.83, 17.68, 90, 90, 120, [{ element: "Pd", x: 0, y: 0, z: 0 }, { element: "Co", x: 0, y: 0, z: 0.350 }, { element: "O", x: 0.333, y: 0.667, z: 0.555 }], 166, "R-3m", "rhombohedral", "delafossite", -1.50, 122.65, 8.12),

  e("VGa3", 4.16, 4.16, 4.16, 90, 90, 90, [{ element: "V", x: 0, y: 0, z: 0 }, { element: "Ga", x: 0.5, y: 0.5, z: 0 }, { element: "Ga", x: 0.5, y: 0, z: 0.5 }, { element: "Ga", x: 0, y: 0.5, z: 0.5 }], 221, "Pm-3m", "cubic", "L12-Cu3Au", -0.32, 72.0, 6.42),
  e("Ni3Al", 3.57, 3.57, 3.57, 90, 90, 90, [{ element: "Ni", x: 0.5, y: 0.5, z: 0 }, { element: "Ni", x: 0.5, y: 0, z: 0.5 }, { element: "Ni", x: 0, y: 0.5, z: 0.5 }, { element: "Al", x: 0, y: 0, z: 0 }], 221, "Pm-3m", "cubic", "L12-Cu3Au", -0.47, 45.50, 7.50),
  e("Co3V", 3.60, 3.60, 3.60, 90, 90, 90, [{ element: "Co", x: 0.5, y: 0.5, z: 0 }, { element: "Co", x: 0.5, y: 0, z: 0.5 }, { element: "Co", x: 0, y: 0.5, z: 0.5 }, { element: "V", x: 0, y: 0, z: 0 }], 221, "Pm-3m", "cubic", "L12-Cu3Au", -0.38, 46.66, 8.15),

  e("BaNb2O6", 12.46, 12.46, 3.99, 90, 90, 90, [{ element: "Ba", x: 0.172, y: 0.328, z: 0 }, { element: "Nb", x: 0.074, y: 0.213, z: 0.5 }, { element: "O", x: 0.343, y: 0.006, z: 0.5 }], 127, "P4/mbm", "tetragonal", "tungsten-bronze", -3.25, 619.42, 5.52),
  e("KNb3O8", 8.90, 21.16, 3.80, 90, 90, 90, [{ element: "K", x: 0.340, y: 0.085, z: 0.25 }, { element: "Nb", x: 0.060, y: 0.105, z: 0.25 }, { element: "O", x: 0.170, y: 0.020, z: 0.25 }], 62, "Pnma", "orthorhombic", "layered-niobate", -3.10, 715.28, 3.82),

  e("TlBiSe2", 4.24, 4.24, 22.16, 90, 90, 120, [{ element: "Tl", x: 0, y: 0, z: 0 }, { element: "Bi", x: 0, y: 0, z: 0.245 }, { element: "Se", x: 0, y: 0, z: 0.120 }, { element: "Se", x: 0, y: 0, z: 0.370 }], 166, "R-3m", "rhombohedral", "layered-TI", -0.35, 345.22, 7.82),
  e("TlBiTe2", 4.42, 4.42, 23.18, 90, 90, 120, [{ element: "Tl", x: 0, y: 0, z: 0 }, { element: "Bi", x: 0, y: 0, z: 0.248 }, { element: "Te", x: 0, y: 0, z: 0.118 }, { element: "Te", x: 0, y: 0, z: 0.368 }], 166, "R-3m", "rhombohedral", "layered-TI", -0.28, 392.15, 8.55),

  e("Li2IrO3", 5.17, 8.96, 5.12, 90, 109.8, 90, [{ element: "Li", x: 0, y: 0.167, z: 0.25 }, { element: "Ir", x: 0, y: 0.333, z: 0.25 }, { element: "O", x: 0.27, y: 0.06, z: 0.01 }], 15, "C2/c", "monoclinic", "honeycomb-iridate", -2.12, 225.42, 7.95),
  e("Na2IrO3", 5.43, 9.40, 5.60, 90, 109.1, 90, [{ element: "Na", x: 0, y: 0.167, z: 0.25 }, { element: "Ir", x: 0, y: 0.333, z: 0.25 }, { element: "O", x: 0.28, y: 0.07, z: 0.02 }], 15, "C2/c", "monoclinic", "honeycomb-iridate", -1.95, 270.35, 6.72),
  e("RuCl3", 5.98, 10.35, 6.05, 90, 108.8, 90, [{ element: "Ru", x: 0, y: 0.333, z: 0.25 }, { element: "Cl", x: 0.25, y: 0.167, z: 0.25 }, { element: "Cl", x: 0.75, y: 0.167, z: 0.25 }], 15, "C2/c", "monoclinic", "honeycomb-Kitaev", -1.42, 355.68, 3.88),

  e("NbRh2B2", 3.16, 3.16, 12.05, 90, 90, 90, [{ element: "Nb", x: 0, y: 0, z: 0 }, { element: "Rh", x: 0, y: 0.5, z: 0.25 }, { element: "Rh", x: 0.5, y: 0, z: 0.25 }, { element: "B", x: 0, y: 0, z: 0.355 }, { element: "B", x: 0, y: 0, z: 0.645 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2-like", -0.65, 120.22, 8.88),
  e("ThNi2B2C", 3.52, 3.52, 10.72, 90, 90, 90, [{ element: "Th", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0, y: 0.5, z: 0.25 }, { element: "Ni", x: 0.5, y: 0, z: 0.25 }, { element: "B", x: 0, y: 0, z: 0.356 }, { element: "B", x: 0, y: 0, z: 0.644 }, { element: "C", x: 0, y: 0, z: 0.5 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2-like", -0.64, 132.80, 9.42),
  e("ErNi2B2C", 3.50, 3.50, 10.56, 90, 90, 90, [{ element: "Er", x: 0, y: 0, z: 0 }, { element: "Ni", x: 0, y: 0.5, z: 0.25 }, { element: "Ni", x: 0.5, y: 0, z: 0.25 }, { element: "B", x: 0, y: 0, z: 0.353 }, { element: "B", x: 0, y: 0, z: 0.647 }, { element: "C", x: 0, y: 0, z: 0.5 }], 139, "I4/mmm", "tetragonal", "ThCr2Si2-like", -0.66, 129.36, 8.52),

  e("MgB4", 5.46, 5.46, 4.43, 90, 90, 90, [{ element: "Mg", x: 0, y: 0, z: 0 }, { element: "B", x: 0.175, y: 0.365, z: 0.5 }, { element: "B", x: 0.365, y: 0.175, z: 0.5 }], 127, "P4/mbm", "tetragonal", "UB4-type", -0.45, 132.06, 2.48),
  e("CaB4", 5.58, 5.58, 7.19, 90, 90, 90, [{ element: "Ca", x: 0, y: 0, z: 0 }, { element: "B", x: 0.175, y: 0.365, z: 0.25 }, { element: "B", x: 0.365, y: 0.175, z: 0.25 }], 127, "P4/mbm", "tetragonal", "UB4-type", -0.52, 223.82, 2.72),

  e("Mo3Al2C", 6.87, 6.87, 6.87, 90, 90, 90, [{ element: "Mo", x: 0.25, y: 0.25, z: 0.25 }, { element: "Al", x: 0.375, y: 0, z: 0.25 }, { element: "C", x: 0, y: 0, z: 0 }], 227, "Fd-3m", "cubic", "beta-Mn-type", -0.42, 324.24, 7.15),
  e("W3Al2C", 6.82, 6.82, 6.82, 90, 90, 90, [{ element: "W", x: 0.25, y: 0.25, z: 0.25 }, { element: "Al", x: 0.375, y: 0, z: 0.25 }, { element: "C", x: 0, y: 0, z: 0 }], 227, "Fd-3m", "cubic", "beta-Mn-type", -0.38, 317.21, 12.25),
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
