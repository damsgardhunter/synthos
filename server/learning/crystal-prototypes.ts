import { getElementData, isTransitionMetal, isRareEarth } from "./elemental-data";
import { parseFormulaCounts as parseFormulaCountsCanonical } from "./utils";

export interface PrototypeTemplate {
  name: string;
  spaceGroup: string;
  latticeType: "cubic" | "hexagonal" | "tetragonal";
  cOverA: number;
  sites: { label: string; x: number; y: number; z: number; role: string }[];
  stoichiometryRatio: number[];
  coordination: number[];
  chemistryRules: (elements: string[]) => boolean;
}

export interface FilledPrototype {
  templateName: string;
  siteMap: Record<string, string>;
  atoms: { element: string; x: number; y: number; z: number }[];
  latticeParam: number;
}

export interface PreFilterResult {
  stable: boolean;
  score: number;
  reasons: string[];
}

export const IONIC_RADII: Record<string, number> = {
  H: 0.25, Li: 0.76, Be: 0.45, B: 0.27, C: 0.16, N: 1.46, O: 1.40, F: 1.33,
  Na: 1.02, Mg: 0.72, Al: 0.54, Si: 0.40, P: 0.38, S: 1.84, Cl: 1.81,
  K: 1.38, Ca: 1.00, Sc: 0.75, Ti: 0.61, V: 0.54, Cr: 0.62, Mn: 0.67,
  Fe: 0.65, Co: 0.61, Ni: 0.69, Cu: 0.73, Zn: 0.74, Ga: 0.62, Ge: 0.53,
  As: 0.46, Se: 1.98, Br: 1.96, Rb: 1.52, Sr: 1.18, Y: 0.90, Zr: 0.72,
  Nb: 0.64, Mo: 0.59, Ru: 0.62, Rh: 0.67, Pd: 0.86, Ag: 1.15, Cd: 0.95,
  In: 0.80, Sn: 0.69, Sb: 0.76, Te: 2.21, I: 2.20, Cs: 1.67, Ba: 1.35,
  La: 1.03, Ce: 1.01, Pr: 0.99, Nd: 0.98, Sm: 0.96, Gd: 0.94,
  Tc: 0.65, Hf: 0.71, Ta: 0.64, W: 0.60, Re: 0.63, Os: 0.63, Ir: 0.63, Pt: 0.63,
  Au: 0.85, Hg: 1.02, Tl: 1.50, Pb: 1.19, Bi: 1.03,
};

const OXIDATION_STATES: Record<string, number[]> = {
  H: [1, -1], Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
  Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
  Sc: [3], Y: [3], La: [3], Ce: [3, 4], Pr: [3], Nd: [3], Sm: [3], Gd: [3],
  Ti: [4, 3], Zr: [4, 2, 3], Hf: [4], V: [5, 4, 3, 2], Nb: [5, 3], Ta: [5],
  Cr: [3, 6], Mo: [6, 4], W: [6, 4], Mn: [2, 4, 7], Re: [4, 7],
  Fe: [2, 3], Ru: [3, 4], Os: [4], Co: [2, 3], Rh: [3], Ir: [3, 4],
  Ni: [2], Pd: [2], Pt: [2, 4], Cu: [2, 1], Ag: [1], Au: [3, 1],
  Zn: [2], Cd: [2], Hg: [2, 1],
  B: [3], Al: [3], Ga: [3], In: [3], Tl: [1, 3],
  C: [4, -4], Si: [4, -4], Ge: [4], Sn: [4, 2], Pb: [2, 4],
  N: [-3, 3, 5], P: [-3, 5], As: [-3, 5], Sb: [-3, 5], Bi: [3],
  O: [-2], S: [-2, 6], Se: [-2], Te: [-2],
  F: [-1], Cl: [-1], Br: [-1], I: [-1],
};

const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57,
  Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02,
  K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39,
  Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20,
  As: 1.19, Se: 1.20, Br: 1.20, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75,
  Nb: 1.64, Mo: 1.54, Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44,
  In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39, Cs: 2.44, Ba: 2.15,
  La: 2.07, Ce: 2.04, Pr: 2.03, Nd: 2.01, Sm: 1.98, Gd: 1.96,
  Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36,
  Au: 1.36, Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48,
};

const PACKING_FACTORS: Record<string, number> = {
  "A15": 0.68,
  "AlB2": 0.74,
  "ThCr2Si2": 0.68,
  "Perovskite": 0.74,
  "MX2": 0.52,
  "Anti-perovskite": 0.74,
  "BiS2-type": 0.58,
  "CaBe2Ge2": 0.58,
  "FeSe-11": 0.52,
  "K2NiF4-214": 0.52,
  "1111-Type": 0.58,
  "Infinite-layer": 0.58,
  "T-prime": 0.52,
  // High-pressure hydride prototypes
  "Bcc-MH3": 0.68,
  "Sodalite-MH6": 0.62,
  "Hex-Clathrate-MH9": 0.58,
  "Clathrate-Fm3m-MH10": 0.60,
  "Ternary-Clathrate-A2MHn": 0.58,
  "Ternary-Hex-Clathrate-A2MH9": 0.56,
  // Additional families
  "YBCO-123": 0.55,
  "Hexaboride-MB6": 0.60,
  "Diamond-Fd3m": 0.34,
  "111-Pnictide-LiFeAs": 0.62,
  "Delafossite-ABO2": 0.58,
  "LayeredOxide-AMO2": 0.58,
  "Chalcopyrite-ABX2": 0.52,
  "Olivine-A2BO4": 0.60,
  "DoublePerovskite-A2BBO6": 0.74,
  "RP-n2-A3B2O7": 0.58,
  "Bi2212-A2B2CB2O8": 0.55,
  "InverseHeusler-XA2B": 0.68,
  "MAX312-M3AX2": 0.68,
  "MAX413-M4AX3": 0.68,
  "Brownmillerite-A2B2O5": 0.62,
  "RP-n3-A4B3O10": 0.55,
};

const DEFAULT_PACKING_FACTOR = 0.68;

function getCovalentRadius(el: string): number {
  return COVALENT_RADII[el] ?? 1.3;
}

function isIonicCompound(elements: string[], templateName: string): boolean {
  const ionicTemplates = new Set([
    "Perovskite", "Anti-perovskite", "NaCl-B1", "K2NiF4-214",
    "Infinite-layer", "T-prime", "YBCO-123", "1111-Type", "BiS2-type"
  ]);
  if (ionicTemplates.has(templateName)) return true;
  return elements.some(e => ANIONS.has(e)) && elements.some(e => !ANIONS.has(e));
}

function getEffectiveRadius(el: string, useIonic: boolean): number {
  if (useIonic && IONIC_RADII[el] !== undefined) {
    return IONIC_RADII[el];
  }
  return getCovalentRadius(el);
}

export function estimateLatticeConstant(
  elements: string[],
  counts: Record<string, number>,
  template: PrototypeTemplate
): { a: number; c: number } {
  const useIonic = isIonicCompound(elements, template.name);
  let totalVolume = 0;
  for (const el of elements) {
    const n = Math.round(counts[el] || 1);
    const r = getEffectiveRadius(el, useIonic);
    totalVolume += n * (4.0 / 3.0) * Math.PI * r * r * r;
  }

  const packingFactor = PACKING_FACTORS[template.name] ?? DEFAULT_PACKING_FACTOR;
  let cellVolume = totalVolume / packingFactor;

  const totalAtoms = elements.reduce((s, el) => s + Math.round(counts[el] || 1), 0);
  const hasH = elements.includes("H");
  const minVolPerAtom = hasH ? 5.0 : 8.0;
  const minCellVolume = totalAtoms * minVolPerAtom;
  if (cellVolume < minCellVolume) {
    cellVolume = minCellVolume;
  }

  let a: number;
  let c: number;

  if (template.latticeType === "cubic") {
    a = Math.pow(cellVolume, 1.0 / 3.0);
    c = a;
  } else if (template.latticeType === "hexagonal") {
    const hexFactor = (Math.sqrt(3) / 2) * template.cOverA;
    a = Math.pow(cellVolume / hexFactor, 1.0 / 3.0);
    c = a * template.cOverA;
  } else {
    a = Math.pow(cellVolume / template.cOverA, 1.0 / 3.0);
    c = a * template.cOverA;
  }

  return { a, c };
}

const ANIONS = new Set(["O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P", "As"]);
const CATIONS_LARGE = new Set(["K", "Rb", "Cs", "Ba", "Sr", "Ca", "La", "Ce", "Pr", "Nd", "Y", "Na", "Pb", "Bi", "Tl"]);
const CATIONS_SMALL_TM = new Set(["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Nb", "Mo", "W", "Ta", "Ru", "Rh", "Ir", "Pd", "Pt", "Re", "Os", "Hf", "Zr", "Sc"]);

export const PROTOTYPE_TEMPLATES: PrototypeTemplate[] = [
  {
    name: "A15",
    spaceGroup: "Pm-3n",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "body-center" },
      { label: "B", x: 0.25, y: 0.0, z: 0.5, role: "chain" },
      { label: "B", x: 0.75, y: 0.0, z: 0.5, role: "chain" },
      { label: "B", x: 0.5, y: 0.25, z: 0.0, role: "chain" },
      { label: "B", x: 0.5, y: 0.75, z: 0.0, role: "chain" },
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "chain" },
      { label: "B", x: 0.0, y: 0.5, z: 0.75, role: "chain" },
    ],
    stoichiometryRatio: [1, 3],
    coordination: [12, 14],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => isTransitionMetal(e));
      return hasTM;
    },
  },
  {
    name: "AlB2",
    spaceGroup: "P6/mmm",
    latticeType: "hexagonal",
    cOverA: 1.08,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal-layer" },
      { label: "B", x: 0.333, y: 0.667, z: 0.5, role: "honeycomb" },
      { label: "B", x: 0.667, y: 0.333, z: 0.5, role: "honeycomb" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [12, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => isTransitionMetal(e) || isRareEarth(e) || ["Mg", "Al", "Ca", "Sr", "Ba"].includes(e));
    },
  },
  {
    name: "ThCr2Si2",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 2.5,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "spacer" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "spacer" },
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "TM-layer" },
      { label: "B", x: 0.5, y: 0.0, z: 0.25, role: "TM-layer" },
      { label: "B", x: 0.0, y: 0.5, z: 0.75, role: "TM-layer" },
      { label: "B", x: 0.5, y: 0.0, z: 0.75, role: "TM-layer" },
      { label: "C", x: 0.0, y: 0.0, z: 0.35, role: "pnictogen" },
      { label: "C", x: 0.0, y: 0.0, z: 0.65, role: "pnictogen" },
      { label: "C", x: 0.5, y: 0.5, z: 0.85, role: "pnictogen" },
      { label: "C", x: 0.5, y: 0.5, z: 0.15, role: "pnictogen" },
    ],
    stoichiometryRatio: [1, 2, 2],
    coordination: [8, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasSpacer = elements.some(e => CATIONS_LARGE.has(e));
      const hasTM = elements.some(e => isTransitionMetal(e));
      const hasPn = elements.some(e => ["As", "P", "Sb", "Si", "Ge", "Se", "Te"].includes(e));
      return hasSpacer && hasTM && hasPn;
    },
  },
  {
    name: "NaCl-B1",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation" },
      { label: "A", x: 0.5, y: 0.5, z: 0.0, role: "cation" },
      { label: "A", x: 0.5, y: 0.0, z: 0.5, role: "cation" },
      { label: "A", x: 0.0, y: 0.5, z: 0.5, role: "cation" },
      { label: "B", x: 0.5, y: 0.0, z: 0.0, role: "anion" },
      { label: "B", x: 0.0, y: 0.5, z: 0.0, role: "anion" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "anion" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "anion" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTMorRE = elements.some(e => isTransitionMetal(e) || isRareEarth(e) || ["Mg", "Ca", "Sr", "Ba", "Pb"].includes(e));
      const hasAnion = elements.some(e => ["N", "C", "O", "S", "Se", "Te", "F", "Cl", "Br", "I"].includes(e));
      return hasTMorRE && hasAnion;
    },
  },
  {
    name: "Perovskite",
    spaceGroup: "Pm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "B-site" },
      { label: "C", x: 0.5, y: 0.5, z: 0.0, role: "anion" },
      { label: "C", x: 0.5, y: 0.0, z: 0.5, role: "anion" },
      { label: "C", x: 0.0, y: 0.5, z: 0.5, role: "anion" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [12, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasLargeCation = elements.some(e => CATIONS_LARGE.has(e));
      const hasSmallCation = elements.some(e => CATIONS_SMALL_TM.has(e) || ["Al", "Ga", "In", "Ge", "Sn", "Mg"].includes(e));
      const hasAnion = elements.some(e => ANIONS.has(e));
      return hasLargeCation && (hasSmallCation || elements.some(e => isTransitionMetal(e))) && hasAnion;
    },
  },
  {
    name: "MX2",
    spaceGroup: "P6_3/mmc",
    latticeType: "hexagonal",
    cOverA: 3.9,
    sites: [
      { label: "M", x: 0.333, y: 0.667, z: 0.25, role: "metal" },
      { label: "X", x: 0.333, y: 0.667, z: 0.621, role: "chalcogen-top" },
      { label: "X", x: 0.333, y: 0.667, z: 0.879, role: "chalcogen-bot" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Mo", "W", "Re"].includes(e));
      const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
      return hasTM && hasChalcogen;
    },
  },
  {
    name: "Anti-perovskite",
    spaceGroup: "Pm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "body-center-anion" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "corner-cation" },
      { label: "X", x: 0.5, y: 0.5, z: 0.0, role: "face-center" },
      { label: "X", x: 0.5, y: 0.0, z: 0.5, role: "face-center" },
      { label: "X", x: 0.0, y: 0.5, z: 0.5, role: "face-center" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [6, 12, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasAnion = elements.some(e => ["C", "N", "B", "O"].includes(e));
      const hasMetal = elements.some(e => isTransitionMetal(e) || CATIONS_LARGE.has(e));
      return hasAnion && hasMetal;
    },
  },
  // ── High-pressure hydride prototypes ──────────────────────────────────
  // These must appear BEFORE generic templates to match first for hydrides.
  // Wyckoff positions are for the PRIMITIVE cell, computed from the
  // conventional cell symmetry operations.

  // H3S-type: Im-3m BCC, 1 M + 3 H (MH3 stoichiometry)
  // S at 2a → 1 in primitive; H at 6b → 3 in primitive
  // For: H3S, ScH3, YH3 (and any MH3 binary with H + chalcogen/metal)
  {
    name: "Bcc-MH3",
    spaceGroup: "Im-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "metal-center" },
      { label: "H", x: 0.0, y: 0.5, z: 0.5, role: "octahedral" },
      { label: "H", x: 0.5, y: 0.0, z: 0.5, role: "octahedral" },
      { label: "H", x: 0.5, y: 0.5, z: 0.0, role: "octahedral" },
    ],
    stoichiometryRatio: [1, 3],
    coordination: [6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // H3S (H is majority, S/Se/Te is minority) or ScH3/YH3 (metal + H)
      return elements.includes("H") || elements.some(e => ["S", "Se", "Te"].includes(e));
    },
  },
  // Sodalite-type: Im-3m, 1 M + 6 H (MH6 stoichiometry)
  // M at 2a → 1 in primitive; H at 12d → 6 in primitive
  // For: CaH6, YH6, SrH6, LaH6, ScH6, MgH6, BaH6
  {
    name: "Sodalite-MH6",
    spaceGroup: "Im-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "cage-center" },
      { label: "H", x: 0.50, y: 0.75, z: 0.25, role: "sodalite-cage" },
      { label: "H", x: 0.50, y: 0.25, z: 0.75, role: "sodalite-cage" },
      { label: "H", x: 0.75, y: 0.50, z: 0.25, role: "sodalite-cage" },
      { label: "H", x: 0.25, y: 0.50, z: 0.75, role: "sodalite-cage" },
      { label: "H", x: 0.75, y: 0.25, z: 0.50, role: "sodalite-cage" },
      { label: "H", x: 0.25, y: 0.75, z: 0.50, role: "sodalite-cage" },
    ],
    stoichiometryRatio: [1, 6],
    coordination: [24, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.includes("H") && elements.some(e =>
        isRareEarth(e) || ["Ca", "Sr", "Ba", "Y", "Sc", "Mg", "Th"].includes(e)
      );
    },
  },
  // Hexagonal clathrate: P63/mmc, 1 M + 9 H (MH9 stoichiometry)
  // For: YH9, CeH9, ScH9
  {
    name: "Hex-Clathrate-MH9",
    spaceGroup: "P63/mmc",
    latticeType: "hexagonal",
    cOverA: 1.55,
    sites: [
      { label: "M", x: 0.3333, y: 0.6667, z: 0.25, role: "cage-center" },
      { label: "H", x: 0.155, y: 0.310, z: 0.25, role: "hex-cage-6h" },
      { label: "H", x: 0.690, y: 0.845, z: 0.25, role: "hex-cage-6h" },
      { label: "H", x: 0.845, y: 0.155, z: 0.25, role: "hex-cage-6h" },
      { label: "H", x: 0.0, y: 0.0, z: 0.25, role: "hex-cage-2b" },
      { label: "H", x: 0.520, y: 0.040, z: 0.08, role: "hex-cage-12k" },
      { label: "H", x: 0.960, y: 0.480, z: 0.08, role: "hex-cage-12k" },
      { label: "H", x: 0.480, y: 0.520, z: 0.08, role: "hex-cage-12k" },
      { label: "H", x: 0.040, y: 0.520, z: 0.42, role: "hex-cage-12k" },
      { label: "H", x: 0.520, y: 0.480, z: 0.42, role: "hex-cage-12k" },
    ],
    stoichiometryRatio: [1, 9],
    coordination: [29, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.includes("H") && elements.some(e =>
        isRareEarth(e) || ["Y", "Sc", "Ca", "Sr", "Ba", "Th"].includes(e)
      );
    },
  },
  // Clathrate Fm-3m: 1 M + 10 H (MH10 stoichiometry)
  // La at 4a → 1 in primitive; H at 32f → 8 in primitive; H at 8c → 2 in primitive
  // For: LaH10, ThH10, CeH10 and novel RE-H10 predictions
  {
    name: "Clathrate-Fm3m-MH10",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "cage-center" },
      { label: "H", x: 0.375, y: 0.375, z: 0.375, role: "32f-cage" },
      { label: "H", x: 0.375, y: 0.375, z: 0.875, role: "32f-cage" },
      { label: "H", x: 0.375, y: 0.875, z: 0.375, role: "32f-cage" },
      { label: "H", x: 0.875, y: 0.375, z: 0.375, role: "32f-cage" },
      { label: "H", x: 0.625, y: 0.625, z: 0.625, role: "32f-cage" },
      { label: "H", x: 0.625, y: 0.625, z: 0.125, role: "32f-cage" },
      { label: "H", x: 0.625, y: 0.125, z: 0.625, role: "32f-cage" },
      { label: "H", x: 0.125, y: 0.625, z: 0.625, role: "32f-cage" },
      { label: "H", x: 0.25, y: 0.25, z: 0.25, role: "8c-interstitial" },
      { label: "H", x: 0.75, y: 0.75, z: 0.75, role: "8c-interstitial" },
    ],
    stoichiometryRatio: [1, 10],
    coordination: [32, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.includes("H") && elements.some(e =>
        isRareEarth(e) || ["Ca", "Sr", "Ba", "Y", "Sc", "Th"].includes(e)
      );
    },
  },
  // Ternary clathrate hydride: A2MHn (e.g., Li2LaH12, Li2LaH11)
  // Based on Fm-3m cage with alkali metal (Li/Na/K) at interstitial 8c sites
  // and rare earth at cage center. 3-element hydride template.
  // Sites: 1 M(RE) + 2 A(alkali) + variable H
  {
    name: "Ternary-Clathrate-A2MHn",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "cage-center" },
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "interstitial" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "interstitial" },
      { label: "H", x: 0.375, y: 0.375, z: 0.375, role: "32f-cage" },
      { label: "H", x: 0.375, y: 0.375, z: 0.875, role: "32f-cage" },
      { label: "H", x: 0.375, y: 0.875, z: 0.375, role: "32f-cage" },
      { label: "H", x: 0.875, y: 0.375, z: 0.375, role: "32f-cage" },
      { label: "H", x: 0.625, y: 0.625, z: 0.625, role: "32f-cage" },
      { label: "H", x: 0.625, y: 0.625, z: 0.125, role: "32f-cage" },
      { label: "H", x: 0.625, y: 0.125, z: 0.625, role: "32f-cage" },
      { label: "H", x: 0.125, y: 0.625, z: 0.625, role: "32f-cage" },
      { label: "H", x: 0.125, y: 0.125, z: 0.125, role: "extra" },
      { label: "H", x: 0.875, y: 0.875, z: 0.875, role: "extra" },
      { label: "H", x: 0.125, y: 0.375, z: 0.125, role: "extra" },
      { label: "H", x: 0.375, y: 0.125, z: 0.125, role: "extra" },
    ],
    stoichiometryRatio: [1, 2, 12],
    coordination: [32, 8, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasH = elements.includes("H");
      const hasAlkali = elements.some(e => ["Li", "Na", "K", "Rb", "Cs"].includes(e));
      const hasRE = elements.some(e => isRareEarth(e) || ["Y", "Sc", "Ca", "Sr", "Ba", "Th"].includes(e));
      return hasH && hasAlkali && hasRE;
    },
  },
  // Ternary hexagonal hydride: A2MH9 (e.g., YH9Na2)
  // Based on P63/mmc cage with alkali at interstices.
  {
    name: "Ternary-Hex-Clathrate-A2MH9",
    spaceGroup: "P63/mmc",
    latticeType: "hexagonal",
    cOverA: 1.55,
    sites: [
      { label: "M", x: 0.3333, y: 0.6667, z: 0.25, role: "cage-center" },
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "interstice" },
      { label: "A", x: 0.0, y: 0.0, z: 0.5, role: "interstice" },
      { label: "H", x: 0.155, y: 0.310, z: 0.25, role: "hex-cage" },
      { label: "H", x: 0.690, y: 0.845, z: 0.25, role: "hex-cage" },
      { label: "H", x: 0.845, y: 0.155, z: 0.25, role: "hex-cage" },
      { label: "H", x: 0.0, y: 0.0, z: 0.25, role: "hex-cage" },
      { label: "H", x: 0.520, y: 0.040, z: 0.08, role: "hex-cage" },
      { label: "H", x: 0.960, y: 0.480, z: 0.08, role: "hex-cage" },
      { label: "H", x: 0.480, y: 0.520, z: 0.08, role: "hex-cage" },
      { label: "H", x: 0.040, y: 0.520, z: 0.42, role: "hex-cage" },
      { label: "H", x: 0.520, y: 0.480, z: 0.42, role: "hex-cage" },
    ],
    stoichiometryRatio: [1, 2, 9],
    coordination: [29, 8, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasH = elements.includes("H");
      const hasAlkali = elements.some(e => ["Li", "Na", "K", "Rb", "Cs"].includes(e));
      const hasRE = elements.some(e => isRareEarth(e) || ["Y", "Sc", "Ca", "Sr", "Ba", "Th"].includes(e));
      return hasH && hasAlkali && hasRE;
    },
  },
  // ── End of hydride prototypes ──────────────────────────────────────────

  // ── Additional structure families ──────────────────────────────────────

  // YBCO-123 cuprate: ABa2Cu3O7 (Pmmm, orthorhombic simplified as tetragonal)
  // For: YBa2Cu3O7, NdBa2Cu3O7, GdBa2Cu3O7, SmBa2Cu3O7, etc.
  {
    name: "YBCO-123",
    spaceGroup: "Pmmm",
    latticeType: "tetragonal",
    cOverA: 3.06,
    sites: [
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "rare-earth" },
      { label: "B", x: 0.5, y: 0.5, z: 0.185, role: "Ba-site" },
      { label: "B", x: 0.5, y: 0.5, z: 0.815, role: "Ba-site" },
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "Cu-chain" },
      { label: "C", x: 0.0, y: 0.0, z: 0.356, role: "Cu-plane" },
      { label: "C", x: 0.0, y: 0.0, z: 0.644, role: "Cu-plane" },
      { label: "D", x: 0.0, y: 0.5, z: 0.0, role: "O-chain" },
      { label: "D", x: 0.5, y: 0.0, z: 0.378, role: "O-plane" },
      { label: "D", x: 0.5, y: 0.0, z: 0.622, role: "O-plane" },
      { label: "D", x: 0.0, y: 0.5, z: 0.378, role: "O-plane" },
      { label: "D", x: 0.0, y: 0.5, z: 0.622, role: "O-plane" },
      { label: "D", x: 0.0, y: 0.0, z: 0.159, role: "O-apical" },
      { label: "D", x: 0.0, y: 0.0, z: 0.841, role: "O-apical" },
    ],
    stoichiometryRatio: [1, 2, 3, 7],
    coordination: [8, 10, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasBa = elements.includes("Ba");
      const hasCu = elements.includes("Cu");
      const hasRE = elements.some(e => isRareEarth(e) || ["Y", "Bi", "Tl", "Hg"].includes(e));
      return hasO && hasBa && hasCu && hasRE;
    },
  },
  // Hexaboride: Pm-3m, 1 M + 6 B (MB6 stoichiometry)
  // B forms octahedral cage. For: CaB6, LaB6, CeB6, SmB6, etc.
  {
    name: "Hexaboride-MB6",
    spaceGroup: "Pm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "cage-center" },
      { label: "B", x: 0.2, y: 0.5, z: 0.5, role: "B-octahedron" },
      { label: "B", x: 0.8, y: 0.5, z: 0.5, role: "B-octahedron" },
      { label: "B", x: 0.5, y: 0.2, z: 0.5, role: "B-octahedron" },
      { label: "B", x: 0.5, y: 0.8, z: 0.5, role: "B-octahedron" },
      { label: "B", x: 0.5, y: 0.5, z: 0.2, role: "B-octahedron" },
      { label: "B", x: 0.5, y: 0.5, z: 0.8, role: "B-octahedron" },
    ],
    stoichiometryRatio: [1, 6],
    coordination: [24, 5],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.includes("B") && elements.some(e =>
        isRareEarth(e) || ["Ca", "Sr", "Ba", "Y", "Sc", "Th", "Eu", "Sm"].includes(e)
      );
    },
  },
  // Diamond cubic: Fd-3m, elemental (Si, Ge, C)
  // 2 atoms in primitive cell at (0,0,0) and (0.25,0.25,0.25)
  {
    name: "Diamond-Fd3m",
    spaceGroup: "Fd-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "site-1" },
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "site-2" },
    ],
    stoichiometryRatio: [1],
    coordination: [4],
    chemistryRules: (elements) => {
      if (elements.length !== 1) return false;
      return ["C", "Si", "Ge", "Sn", "Pb"].includes(elements[0]);
    },
  },
  // 111-type iron pnictide: P4/nmm, LiFeAs-type (3 elements, 1:1:1)
  {
    name: "111-Pnictide-LiFeAs",
    spaceGroup: "P4/nmm",
    latticeType: "tetragonal",
    cOverA: 1.69,
    sites: [
      { label: "A", x: 0.25, y: 0.25, z: 0.345, role: "alkali" },
      { label: "B", x: 0.75, y: 0.25, z: 0.0, role: "Fe-site" },
      { label: "C", x: 0.25, y: 0.25, z: 0.737, role: "pnictogen" },
    ],
    stoichiometryRatio: [1, 1, 1],
    coordination: [4, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasAlkali = elements.some(e => ["Li", "Na", "K", "Cu"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Co", "Ni", "Mn", "Cr", "Ru"].includes(e));
      const hasPn = elements.some(e => ["As", "P", "Sb", "Bi"].includes(e));
      return hasAlkali && hasTM && hasPn;
    },
  },
  // Delafossite: R-3m, ABO2 (CuFeO2-type)
  // For: CuAlO2, CuFeO2, PdCoO2, PtCoO2 — topological and thermoelectric
  {
    name: "Delafossite-ABO2",
    spaceGroup: "R-3m",
    latticeType: "hexagonal",
    cOverA: 5.6,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "linear-coord" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "octahedral" },
      { label: "C", x: 0.0, y: 0.0, z: 0.11, role: "oxygen" },
      { label: "C", x: 0.0, y: 0.0, z: 0.89, role: "oxygen" },
    ],
    stoichiometryRatio: [1, 1, 2],
    coordination: [2, 6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasNoble = elements.some(e => ["Cu", "Ag", "Pd", "Pt"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Al", "Cr", "Co", "Ga", "In", "Rh", "Ir"].includes(e));
      const hasO = elements.includes("O");
      return hasNoble && hasTM && hasO;
    },
  },
  // Layered oxide: R-3m, AMO2 (LiCoO2-type, battery cathode)
  // For: LiCoO2, LiNiO2, NaCoO2, LiMnO2, etc.
  {
    name: "LayeredOxide-AMO2",
    spaceGroup: "R-3m",
    latticeType: "hexagonal",
    cOverA: 4.9,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.5, role: "alkali-layer" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "TM-layer" },
      { label: "C", x: 0.0, y: 0.0, z: 0.26, role: "O-layer" },
      { label: "C", x: 0.0, y: 0.0, z: 0.74, role: "O-layer" },
    ],
    stoichiometryRatio: [1, 1, 2],
    coordination: [6, 6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasAlkali = elements.some(e => ["Li", "Na", "K"].includes(e));
      const hasTM = elements.some(e => ["Co", "Ni", "Mn", "Fe", "V", "Cr", "Ti"].includes(e));
      const hasO = elements.includes("O");
      return hasAlkali && hasTM && hasO;
    },
  },
  // Chalcopyrite: I-42d, ABX2 (CuFeS2-type)
  // For: CuFeS2, CuGaS2, CuInSe2, AgGaSe2 — photovoltaic/thermoelectric
  {
    name: "Chalcopyrite-ABX2",
    spaceGroup: "I-42d",
    latticeType: "tetragonal",
    cOverA: 1.97,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation-1" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "cation-2" },
      { label: "C", x: 0.25, y: 0.125, z: 0.625, role: "anion" },
      { label: "C", x: 0.75, y: 0.125, z: 0.875, role: "anion" },
    ],
    stoichiometryRatio: [1, 1, 2],
    coordination: [4, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasCu = elements.some(e => ["Cu", "Ag"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Ga", "In", "Al"].includes(e));
      const hasCh = elements.some(e => ["S", "Se", "Te"].includes(e));
      return hasCu && hasTM && hasCh;
    },
  },
  // Olivine: Pnma, A2BO4 (LiFePO4-type, battery cathode)
  // For: LiFePO4, LiMnPO4, LiCoPO4
  {
    name: "Olivine-A2BO4",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 0.47,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "octahedral-M1" },
      { label: "A", x: 0.28, y: 0.25, z: 0.97, role: "octahedral-M2" },
      { label: "B", x: 0.09, y: 0.25, z: 0.42, role: "tetrahedral" },
      { label: "C", x: 0.10, y: 0.25, z: 0.74, role: "O1" },
      { label: "C", x: 0.45, y: 0.25, z: 0.22, role: "O2" },
      { label: "C", x: 0.16, y: 0.04, z: 0.28, role: "O3" },
      { label: "C", x: 0.16, y: 0.46, z: 0.28, role: "O3" },
    ],
    stoichiometryRatio: [2, 1, 4],
    coordination: [6, 4, 3],
    chemistryRules: (elements) => {
      if (elements.length < 3) return false;
      const hasO = elements.includes("O");
      const hasP = elements.some(e => ["P", "Si", "Ge"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Mn", "Co", "Ni", "Mg", "Li", "Na", "Ca"].includes(e));
      return hasO && hasP && hasTM;
    },
  },
  // Double perovskite: Fm-3m, A2BB'O6
  // For: Sr2FeMoO6, Ba2CoWO6, La2NiMnO6
  {
    name: "DoublePerovskite-A2BBO6",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "A-site" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "A-site" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-site" },
      { label: "C", x: 0.5, y: 0.5, z: 0.5, role: "B'-site" },
      { label: "D", x: 0.25, y: 0.0, z: 0.0, role: "O-site" },
      { label: "D", x: 0.75, y: 0.0, z: 0.0, role: "O-site" },
      { label: "D", x: 0.0, y: 0.25, z: 0.0, role: "O-site" },
      { label: "D", x: 0.0, y: 0.75, z: 0.0, role: "O-site" },
      { label: "D", x: 0.0, y: 0.0, z: 0.25, role: "O-site" },
      { label: "D", x: 0.0, y: 0.0, z: 0.75, role: "O-site" },
    ],
    stoichiometryRatio: [2, 1, 1, 6],
    coordination: [12, 6, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasAE = elements.some(e => ["Sr", "Ba", "Ca", "La", "Nd", "Pr", "Y"].includes(e));
      const tmCount = elements.filter(e => isTransitionMetal(e)).length;
      return hasO && hasAE && tmCount >= 2;
    },
  },
  // Ruddlesden-Popper n=2: I4/mmm, A3B2O7
  // For: Sr3Ru2O7, Ca3Mn2O7, La3Ni2O7
  {
    name: "RP-n2-A3B2O7",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 5.3,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.5, role: "A-rock-salt" },
      { label: "A", x: 0.0, y: 0.0, z: 0.318, role: "A-perovskite" },
      { label: "A", x: 0.0, y: 0.0, z: 0.682, role: "A-perovskite" },
      { label: "B", x: 0.0, y: 0.0, z: 0.1, role: "B-site" },
      { label: "B", x: 0.0, y: 0.0, z: 0.9, role: "B-site" },
      { label: "C", x: 0.0, y: 0.5, z: 0.1, role: "O-equatorial" },
      { label: "C", x: 0.5, y: 0.0, z: 0.1, role: "O-equatorial" },
      { label: "C", x: 0.0, y: 0.0, z: 0.2, role: "O-apical" },
      { label: "C", x: 0.0, y: 0.0, z: 0.8, role: "O-apical" },
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "O-bridging" },
      { label: "C", x: 0.0, y: 0.5, z: 0.9, role: "O-equatorial" },
      { label: "C", x: 0.5, y: 0.0, z: 0.9, role: "O-equatorial" },
    ],
    stoichiometryRatio: [3, 2, 7],
    coordination: [9, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      const hasAE = elements.some(e => ["Sr", "Ca", "Ba", "La", "Nd", "Pr"].includes(e));
      const hasTM = elements.some(e => ["Ru", "Mn", "Ni", "Co", "Fe", "Ti", "Ir"].includes(e));
      return hasO && hasAE && hasTM;
    },
  },

  // Bi-2212 cuprate: I4/mmm, A2B2CB'2O8 (Bi2Sr2CaCu2O8+δ type)
  // 5-element layered cuprate. Body-centered tetragonal primitive cell (15 atoms).
  // A = Bi/Tl (heavy post-TM), B = Sr/Ba (alkaline earth), C = Ca (spacer),
  // D = Cu (TM in CuO2 planes), E = O (oxygen)
  // Wyckoff positions from literature: Bi 4e, Sr 4e, Ca 2a, Cu 4e, O at 8g+4e sites
  {
    name: "Bi2212-A2B2CB2O8",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 8.1,
    sites: [
      // A-sites: Bi/Tl (2 in primitive cell, from 4e)
      { label: "A", x: 0.0, y: 0.0, z: 0.199, role: "BiO-layer" },
      { label: "A", x: 0.0, y: 0.0, z: 0.801, role: "BiO-layer" },
      // B-sites: Sr/Ba (2 in primitive cell, from 4e)
      { label: "B", x: 0.0, y: 0.0, z: 0.110, role: "SrO-layer" },
      { label: "B", x: 0.0, y: 0.0, z: 0.890, role: "SrO-layer" },
      // C-site: Ca (1 in primitive cell, from 2a)
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "Ca-spacer" },
      // D-sites: Cu (2 in primitive cell, from 4e)
      { label: "D", x: 0.0, y: 0.0, z: 0.054, role: "CuO2-plane" },
      { label: "D", x: 0.0, y: 0.0, z: 0.946, role: "CuO2-plane" },
      // E-sites: O (8 in primitive cell)
      { label: "E", x: 0.5, y: 0.0, z: 0.054, role: "O-planar" },
      { label: "E", x: 0.0, y: 0.5, z: 0.054, role: "O-planar" },
      { label: "E", x: 0.0, y: 0.0, z: 0.149, role: "O-apical" },
      { label: "E", x: 0.0, y: 0.0, z: 0.851, role: "O-apical" },
      { label: "E", x: 0.5, y: 0.0, z: 0.946, role: "O-planar" },
      { label: "E", x: 0.0, y: 0.5, z: 0.946, role: "O-planar" },
      { label: "E", x: 0.0, y: 0.0, z: 0.250, role: "O-BiO" },
      { label: "E", x: 0.0, y: 0.0, z: 0.750, role: "O-BiO" },
    ],
    stoichiometryRatio: [2, 2, 1, 2, 8],
    coordination: [6, 9, 8, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 5) return false;
      const hasO = elements.includes("O");
      const hasCu = elements.includes("Cu");
      const hasHeavyPost = elements.some(e => ["Bi", "Tl", "Hg", "Pb"].includes(e));
      const hasAE = elements.some(e => ["Sr", "Ba"].includes(e));
      const hasSpacer = elements.includes("Ca") || elements.some(e => isRareEarth(e) || e === "Y");
      return hasO && hasCu && hasHeavyPost && hasAE && hasSpacer;
    },
  },

  // Inverse Heusler: F-43m (216), XA2B ordering
  // Differs from L21 in that the majority-element occupies TWO distinct
  // Wyckoff sites (4c + 4d) instead of one (8c), lowering symmetry from Fm-3m to F-43m.
  // Primitive cell (FCC → 1/4 conventional): 4 atoms.
  // For: Mn2CoAl, Ti2CoSi, Mn2CuAl, Cr2CoGa, etc.
  {
    name: "InverseHeusler-XA2B",
    spaceGroup: "F-43m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "X-site-4a" },
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "A-site-4c" },
      { label: "B", x: 0.75, y: 0.75, z: 0.75, role: "A-site-4d" },
      { label: "C", x: 0.5, y: 0.5, z: 0.5, role: "B-site-4b" },
    ],
    stoichiometryRatio: [1, 2, 1],
    coordination: [4, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      // Inverse Heusler: the majority element is typically Mn, Ti, Cr, or V
      const hasMajorTM = elements.some(e => ["Mn", "Ti", "Cr", "V", "Sc", "Hf", "Zr"].includes(e));
      const hasMinorTM = elements.some(e => ["Co", "Ni", "Fe", "Cu", "Pd", "Pt", "Rh", "Ir"].includes(e));
      const hasSp = elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Sb", "Bi"].includes(e));
      return hasMajorTM && hasMinorTM && hasSp;
    },
  },
  // MAX phase M3AX2 (312): P63/mmc (194), hexagonal
  // For: Ti3SiC2, Ti3AlC2, Ti3GeC2, Zr3Al2C — 312 MAX phases
  // 3 M layers around each A layer, 2 X in octahedral interstices
  // Conventional cell: 2 formula units = 12 atoms; primitive same for hexagonal.
  // Wyckoff: M at 2a + 4f, A at 2b, X at 4f
  {
    name: "MAX312-M3AX2",
    spaceGroup: "P6_3/mmc",
    latticeType: "hexagonal",
    cOverA: 5.5,
    sites: [
      // M at 2a: (0, 0, 0) — 1 in primitive per formula unit
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "TM-inner" },
      // M at 4f: (1/3, 2/3, z) with z ≈ 0.135 — 2 in primitive per f.u.
      { label: "M", x: 0.3333, y: 0.6667, z: 0.135, role: "TM-outer" },
      { label: "M", x: 0.3333, y: 0.6667, z: 0.865, role: "TM-outer" },
      // A at 2b: (0, 0, 1/4) — 1 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.25, role: "A-layer" },
      // X at 4f: (1/3, 2/3, z) with z ≈ 0.070 — 2 in primitive
      { label: "X", x: 0.3333, y: 0.6667, z: 0.070, role: "X-interstitial" },
      { label: "X", x: 0.3333, y: 0.6667, z: 0.930, role: "X-interstitial" },
    ],
    stoichiometryRatio: [3, 1, 2],
    coordination: [6, 12, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasTM = elements.some(e => ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"].includes(e));
      const hasA = elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Pb", "P", "As", "S"].includes(e));
      const hasX = elements.some(e => ["C", "N"].includes(e));
      return hasTM && hasA && hasX;
    },
  },

  // MAX phase M4AX3 (413): P63/mmc (194), hexagonal
  // For: Ti4AlN3, Ta4AlC3, Nb4AlC3, Ti4SiC3 — 413 MAX phases
  // 4 M layers around each A layer, 3 X in octahedral interstices
  // Wyckoff: M at 4f(z≈0.05) + 4e(z≈0.155), A at 2c, X at 2a + 4f(z≈0.103)
  // Primitive cell: 8 atoms per formula unit
  {
    name: "MAX413-M4AX3",
    spaceGroup: "P6_3/mmc",
    latticeType: "hexagonal",
    cOverA: 7.4,
    sites: [
      // M at 4f: inner TM layer (z ≈ 0.054)
      { label: "M", x: 0.3333, y: 0.6667, z: 0.054, role: "TM-inner" },
      { label: "M", x: 0.3333, y: 0.6667, z: 0.946, role: "TM-inner" },
      // M at 4e: outer TM layer (z ≈ 0.155)
      { label: "M", x: 0.0, y: 0.0, z: 0.155, role: "TM-outer" },
      { label: "M", x: 0.0, y: 0.0, z: 0.845, role: "TM-outer" },
      // A at 2c: A-element layer
      { label: "A", x: 0.3333, y: 0.6667, z: 0.25, role: "A-layer" },
      // X at 2a: center interstitial
      { label: "X", x: 0.0, y: 0.0, z: 0.0, role: "X-center" },
      // X at 4f: outer interstitial (z ≈ 0.103)
      { label: "X", x: 0.3333, y: 0.6667, z: 0.103, role: "X-outer" },
      { label: "X", x: 0.3333, y: 0.6667, z: 0.897, role: "X-outer" },
    ],
    stoichiometryRatio: [4, 1, 3],
    coordination: [6, 12, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasTM = elements.some(e => ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"].includes(e));
      const hasA = elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Pb", "P", "As", "S"].includes(e));
      const hasX = elements.some(e => ["C", "N"].includes(e));
      return hasTM && hasA && hasX;
    },
  },
  // Brownmillerite: Ibm2 (46) / Pnma — A2B2O5 (oxygen-deficient perovskite)
  // For: Ca2Fe2O5, Sr2Fe2O5, Ca2Al2O5, Sr2MnFeO5 — solid oxide fuel cells
  // Structure: alternating octahedral BO6 and tetrahedral BO4 layers
  // Simplified orthorhombic → tetragonal mapping, primitive cell: 9 atoms
  // Wyckoff: A at 8d, B at 4a + 4b, O at 8d + 4c + 8d
  {
    name: "Brownmillerite-A2B2O5",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 0.27,
    sites: [
      // A-sites (Ca/Sr/Ba): 2 in reduced formula
      { label: "A", x: 0.027, y: 0.25, z: 0.509, role: "A-site-1" },
      { label: "A", x: 0.522, y: 0.25, z: 0.039, role: "A-site-2" },
      // B-sites (Fe/Al/Mn): octahedral + tetrahedral
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-octahedral" },
      { label: "B", x: 0.928, y: 0.25, z: 0.929, role: "B-tetrahedral" },
      // O-sites: 5 per formula unit
      { label: "C", x: 0.250, y: 0.007, z: 0.231, role: "O-equatorial" },
      { label: "C", x: 0.028, y: 0.25, z: 0.744, role: "O-apical-1" },
      { label: "C", x: 0.595, y: 0.25, z: 0.875, role: "O-apical-2" },
      { label: "C", x: 0.860, y: 0.25, z: 0.070, role: "O-bridging-1" },
      { label: "C", x: 0.371, y: 0.25, z: 0.419, role: "O-bridging-2" },
    ],
    stoichiometryRatio: [2, 2, 5],
    coordination: [8, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      const hasAE = elements.some(e => ["Ca", "Sr", "Ba", "La", "Y"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Al", "Mn", "Co", "Cr", "Ga", "In"].includes(e));
      return hasO && hasAE && hasTM;
    },
  },

  // Ruddlesden-Popper n=3: I4/mmm (139), A4B3O10
  // Triple-perovskite-block intergrown with rock-salt layer.
  // For: Sr4Ru3O10, La4Ni3O10, Ca4Mn3O10, Sr4V3O10
  // Primitive cell (BCC → half conventional): 4A + 3B + 10O = 17 atoms
  // Based on La4Ni3O10 literature (Zhang et al., Nature 2024 high-Tc)
  // z-coordinates scaled for c/a ≈ 7.2 (c ≈ 28 Å, a ≈ 3.85 Å)
  {
    name: "RP-n3-A4B3O10",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 7.2,
    sites: [
      // A-sites: 4 in primitive cell
      // A1 at 2b: rock-salt boundary layer
      { label: "A", x: 0.0, y: 0.0, z: 0.5, role: "A-rock-salt" },
      // A2 at 4e: between outer and inner B layers
      { label: "A", x: 0.0, y: 0.0, z: 0.321, role: "A-outer-perovskite" },
      { label: "A", x: 0.0, y: 0.0, z: 0.679, role: "A-outer-perovskite" },
      // A3 at 4e: between two inner B layers (at center of triple block)
      { label: "A", x: 0.0, y: 0.0, z: 0.178, role: "A-inner-perovskite" },
      // B-sites: 3 in primitive cell
      // B1 at 4e: outer octahedral layers
      { label: "B", x: 0.0, y: 0.0, z: 0.071, role: "B-outer-oct" },
      { label: "B", x: 0.0, y: 0.0, z: 0.929, role: "B-outer-oct" },
      // B2 at 2a: inner octahedral layer (center of triple block)
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-inner-oct" },
      // O-sites: 10 in primitive cell
      // O-equatorial around outer B (4 sites from 8g)
      { label: "C", x: 0.0, y: 0.5, z: 0.071, role: "O-eq-outer" },
      { label: "C", x: 0.5, y: 0.0, z: 0.071, role: "O-eq-outer" },
      { label: "C", x: 0.0, y: 0.5, z: 0.929, role: "O-eq-outer" },
      { label: "C", x: 0.5, y: 0.0, z: 0.929, role: "O-eq-outer" },
      // O-equatorial around inner B (2 sites from 4c)
      { label: "C", x: 0.0, y: 0.5, z: 0.0, role: "O-eq-inner" },
      { label: "C", x: 0.5, y: 0.0, z: 0.0, role: "O-eq-inner" },
      // O-apical between outer B and rock-salt (2 from 4e)
      { label: "C", x: 0.0, y: 0.0, z: 0.393, role: "O-apical-outer" },
      { label: "C", x: 0.0, y: 0.0, z: 0.607, role: "O-apical-outer" },
      // O-apical bridging outer-inner B layers (2 from 4e)
      { label: "C", x: 0.0, y: 0.0, z: 0.107, role: "O-apical-bridge" },
      { label: "C", x: 0.0, y: 0.0, z: 0.893, role: "O-apical-bridge" },
    ],
    stoichiometryRatio: [4, 3, 10],
    coordination: [9, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      const hasAE = elements.some(e => ["Sr", "Ca", "Ba", "La", "Nd", "Pr", "Y"].includes(e));
      const hasTM = elements.some(e => ["Ru", "Mn", "Ni", "Co", "Fe", "Ti", "Ir", "V"].includes(e));
      return hasO && hasAE && hasTM;
    },
  },

  // ── End of additional families ─────────────────────────────────────────
  {
    name: "Heusler-L21",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "main-metal" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "main-metal" },
      { label: "A", x: 0.25, y: 0.75, z: 0.75, role: "main-metal" },
      { label: "A", x: 0.75, y: 0.25, z: 0.25, role: "main-metal" },
      { label: "A", x: 0.75, y: 0.75, z: 0.25, role: "main-metal" },
      { label: "A", x: 0.25, y: 0.25, z: 0.75, role: "main-metal" },
      { label: "A", x: 0.75, y: 0.25, z: 0.75, role: "main-metal" },
      { label: "A", x: 0.25, y: 0.75, z: 0.25, role: "main-metal" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "secondary-metal" },
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "secondary-metal" },
      { label: "B", x: 0.0, y: 0.5, z: 0.5, role: "secondary-metal" },
      { label: "B", x: 0.5, y: 0.0, z: 0.5, role: "secondary-metal" },
      { label: "C", x: 0.5, y: 0.5, z: 0.5, role: "sp-element" },
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "sp-element" },
      { label: "C", x: 0.0, y: 0.5, z: 0.0, role: "sp-element" },
      { label: "C", x: 0.5, y: 0.0, z: 0.0, role: "sp-element" },
    ],
    stoichiometryRatio: [2, 1, 1],
    coordination: [4, 8, 8],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasTM = elements.some(e => ["Cu", "Ni", "Co", "Fe", "Mn", "Pd", "Pt", "Rh", "Ir"].includes(e));
      const hasSecondTM = elements.filter(e => ["Ti", "V", "Mn", "Zr", "Nb", "Hf", "Ta", "Sc", "Y"].includes(e)).length >= 1;
      const hasSp = elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Sb", "Bi"].includes(e));
      return hasTM && (hasSecondTM || hasSp);
    },
  },
  {
    name: "Skutterudite",
    spaceGroup: "Im-3",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "metal" },
      { label: "M", x: 0.5, y: 0.5, z: 0.0, role: "metal" },
      { label: "M", x: 0.0, y: 0.5, z: 0.5, role: "metal" },
      { label: "M", x: 0.5, y: 0.0, z: 0.5, role: "metal" },
      { label: "X", x: 0.0, y: 0.34, z: 0.16, role: "pnicogen" },
      { label: "X", x: 0.34, y: 0.16, z: 0.0, role: "pnicogen" },
      { label: "X", x: 0.16, y: 0.0, z: 0.34, role: "pnicogen" },
    ],
    stoichiometryRatio: [4, 12],
    coordination: [6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Co", "Rh", "Ir", "Fe", "Ru", "Os", "Ni"].includes(e));
      const hasPn = elements.some(e => ["Sb", "As", "P", "Bi"].includes(e));
      return hasTM && hasPn;
    },
  },
  {
    name: "Chevrel",
    spaceGroup: "R-3",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "guest" },
      { label: "M", x: 0.15, y: 0.15, z: 0.15, role: "cluster-metal" },
      { label: "M", x: 0.85, y: 0.85, z: 0.85, role: "cluster-metal" },
      { label: "M", x: 0.85, y: 0.15, z: 0.15, role: "cluster-metal" },
      { label: "M", x: 0.15, y: 0.85, z: 0.15, role: "cluster-metal" },
      { label: "M", x: 0.15, y: 0.15, z: 0.85, role: "cluster-metal" },
      { label: "M", x: 0.85, y: 0.85, z: 0.15, role: "cluster-metal" },
      { label: "X", x: 0.22, y: 0.0, z: 0.28, role: "chalcogen" },
      { label: "X", x: 0.0, y: 0.28, z: 0.22, role: "chalcogen" },
      { label: "X", x: 0.28, y: 0.22, z: 0.0, role: "chalcogen" },
      { label: "X", x: 0.78, y: 0.0, z: 0.72, role: "chalcogen" },
      { label: "X", x: 0.0, y: 0.72, z: 0.78, role: "chalcogen" },
      { label: "X", x: 0.72, y: 0.78, z: 0.0, role: "chalcogen" },
      { label: "X", x: 0.28, y: 0.0, z: 0.72, role: "chalcogen" },
      { label: "X", x: 0.0, y: 0.72, z: 0.28, role: "chalcogen" },
    ],
    stoichiometryRatio: [1, 6, 8],
    coordination: [12, 5, 3],
    chemistryRules: (elements) => {
      if (elements.length < 2 || elements.length > 3) return false;
      const hasMo = elements.includes("Mo") || elements.includes("Re") || elements.includes("W");
      const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
      return hasMo && hasChalcogen;
    },
  },
  {
    name: "Infinite-layer",
    spaceGroup: "P4/mmm",
    latticeType: "tetragonal",
    cOverA: 1.7,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "spacer" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "TM-square-planar" },
      { label: "O", x: 0.5, y: 0.0, z: 0.5, role: "in-plane-O" },
      { label: "O", x: 0.0, y: 0.5, z: 0.5, role: "in-plane-O" },
    ],
    stoichiometryRatio: [1, 1, 2],
    coordination: [8, 4, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasSpacer = elements.some(e => ["Sr", "Ca", "Ba", "La", "Nd", "Pr"].includes(e));
      const hasTM = elements.some(e => ["Cu", "Ni", "Co"].includes(e));
      const hasO = elements.includes("O");
      return hasSpacer && hasTM && hasO;
    },
  },
  {
    name: "Pyrite",
    spaceGroup: "Pa-3",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "metal" },
      { label: "M", x: 0.5, y: 0.5, z: 0.0, role: "metal" },
      { label: "M", x: 0.0, y: 0.5, z: 0.5, role: "metal" },
      { label: "M", x: 0.5, y: 0.0, z: 0.5, role: "metal" },
      { label: "X", x: 0.38, y: 0.38, z: 0.38, role: "dimer-1" },
      { label: "X", x: 0.62, y: 0.62, z: 0.62, role: "dimer-1" },
      { label: "X", x: 0.88, y: 0.12, z: 0.62, role: "dimer-2" },
      { label: "X", x: 0.12, y: 0.88, z: 0.38, role: "dimer-2" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => isTransitionMetal(e));
      const hasChalcogen = elements.some(e => ["S", "Se", "Te", "As", "Sb", "P"].includes(e));
      return hasTM && hasChalcogen;
    },
  },
  {
    name: "PuCoGa5-115",
    spaceGroup: "P4/mmm",
    latticeType: "tetragonal",
    cOverA: 1.6,
    sites: [
      { label: "R", x: 0.0, y: 0.0, z: 0.0, role: "rare-earth" },
      { label: "T", x: 0.5, y: 0.5, z: 0.0, role: "TM" },
      { label: "X", x: 0.5, y: 0.0, z: 0.31, role: "p-block-1" },
      { label: "X", x: 0.0, y: 0.5, z: 0.31, role: "p-block-1" },
      { label: "X", x: 0.5, y: 0.0, z: 0.69, role: "p-block-2" },
      { label: "X", x: 0.0, y: 0.5, z: 0.69, role: "p-block-2" },
      { label: "X", x: 0.0, y: 0.0, z: 0.5, role: "p-block-3" },
    ],
    stoichiometryRatio: [1, 1, 5],
    coordination: [16, 9, 5],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasRE = elements.some(e => isRareEarth(e) || ["U", "Pu", "Np"].includes(e));
      const hasTM = elements.some(e => ["Co", "Rh", "Ir", "Ni", "Fe"].includes(e));
      const hasPBlock = elements.some(e => ["Ga", "In", "Al", "Si", "Ge", "Sn"].includes(e));
      return hasRE && hasTM && hasPBlock;
    },
  },
  {
    name: "1111-Type",
    spaceGroup: "P4/nmm",
    latticeType: "tetragonal",
    cOverA: 2.2,
    sites: [
      { label: "R", x: 0.0, y: 0.0, z: 0.14, role: "spacer" },
      { label: "T", x: 0.5, y: 0.0, z: 0.5, role: "TM-layer" },
      { label: "X", x: 0.5, y: 0.5, z: 0.66, role: "pnictide" },
      { label: "O", x: 0.0, y: 0.0, z: 0.66, role: "oxide" },
    ],
    stoichiometryRatio: [1, 1, 1, 1],
    coordination: [8, 4, 4, 4],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 4) return false;
      const hasSpacer = elements.some(e => ["La", "Ce", "Pr", "Nd", "Sm", "Gd", "Ba", "Sr", "Ca"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Co", "Ni", "Mn"].includes(e));
      const hasPnictide = elements.some(e => ["As", "P", "Sb"].includes(e));
      const hasO = elements.includes("O") || elements.includes("F");
      return hasSpacer && hasTM && hasPnictide && hasO;
    },
  },
  {
    name: "K2NiF4-214",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 3.3,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.36, role: "spacer" },
      { label: "A", x: 0.0, y: 0.0, z: 0.64, role: "spacer" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "TM-square-planar" },
      { label: "X", x: 0.5, y: 0.0, z: 0.0, role: "in-plane-O" },
      { label: "X", x: 0.0, y: 0.5, z: 0.0, role: "in-plane-O" },
      { label: "X", x: 0.0, y: 0.0, z: 0.17, role: "apical-O" },
      { label: "X", x: 0.0, y: 0.0, z: 0.83, role: "apical-O" },
    ],
    stoichiometryRatio: [2, 1, 4],
    coordination: [9, 6, 2],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 3) return false;
      const hasSpacer = elements.some(e => ["La", "Sr", "Ba", "Ca", "Nd", "Pr", "K"].includes(e));
      const hasTM = elements.some(e => ["Cu", "Ni", "Co", "Mn", "Fe", "Ru"].includes(e));
      const hasAnion = elements.some(e => ["O", "F", "S"].includes(e));
      return hasSpacer && hasTM && hasAnion;
    },
  },
  {
    name: "FeSe-11",
    spaceGroup: "P4/nmm",
    latticeType: "tetragonal",
    cOverA: 1.46,
    sites: [
      { label: "A", x: 0.75, y: 0.25, z: 0.0, role: "TM" },
      { label: "A", x: 0.25, y: 0.75, z: 0.0, role: "TM" },
      { label: "B", x: 0.25, y: 0.25, z: 0.27, role: "chalcogen" },
      { label: "B", x: 0.75, y: 0.75, z: 0.73, role: "chalcogen" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [4, 4],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Fe", "Co", "Ni", "Mn"].includes(e));
      const hasChalcogen = elements.some(e => ["Se", "Te", "S"].includes(e));
      return hasTM && hasChalcogen;
    },
  },
  {
    name: "Laves-C15",
    spaceGroup: "Fd-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "large-atom" },
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "large-atom" },
      { label: "B", x: 0.625, y: 0.625, z: 0.625, role: "small-atom" },
      { label: "B", x: 0.625, y: 0.375, z: 0.375, role: "small-atom" },
      { label: "B", x: 0.375, y: 0.625, z: 0.375, role: "small-atom" },
      { label: "B", x: 0.375, y: 0.375, z: 0.625, role: "small-atom" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [16, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => isTransitionMetal(e));
      return hasTM;
    },
  },
  {
    name: "Kagome-variant",
    spaceGroup: "P6/mmm",
    latticeType: "hexagonal",
    cOverA: 2.7,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "spacer" },
      { label: "B", x: 0.5, y: 0.0, z: 0.25, role: "kagome" },
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "kagome" },
      { label: "B", x: 0.5, y: 0.5, z: 0.25, role: "kagome" },
      { label: "C", x: 0.333, y: 0.167, z: 0.5, role: "honeycomb" },
      { label: "C", x: 0.167, y: 0.333, z: 0.5, role: "honeycomb" },
    ],
    stoichiometryRatio: [1, 6, 6],
    coordination: [12, 4, 3],
    chemistryRules: (elements) => {
      if (elements.length < 2 || elements.length > 3) return false;
      const hasTM = elements.some(e => isTransitionMetal(e));
      return hasTM;
    },
  },
  {
    name: "T-prime",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 3.0,
    sites: [
      { label: "R", x: 0.0, y: 0.0, z: 0.35, role: "rare-earth" },
      { label: "R", x: 0.0, y: 0.0, z: 0.65, role: "rare-earth" },
      { label: "T", x: 0.5, y: 0.5, z: 0.0, role: "TM" },
      { label: "O", x: 0.0, y: 0.5, z: 0.0, role: "apical-O" },
      { label: "O", x: 0.0, y: 0.0, z: 0.17, role: "planar-O" },
      { label: "O", x: 0.0, y: 0.0, z: 0.83, role: "planar-O" },
    ],
    stoichiometryRatio: [2, 1, 4],
    coordination: [8, 4, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasRE = elements.some(e => isRareEarth(e));
      const hasTM = elements.some(e => ["Cu", "Ni", "Co"].includes(e));
      const hasO = elements.includes("O");
      return hasRE && hasTM && hasO;
    },
  },
  {
    name: "BiS2-type",
    spaceGroup: "P4/nmm",
    latticeType: "tetragonal",
    cOverA: 2.8,
    sites: [
      { label: "R", x: 0.25, y: 0.25, z: 0.10, role: "spacer-layer" },
      { label: "T", x: 0.75, y: 0.25, z: 0.50, role: "conducting-layer" },
      { label: "X", x: 0.25, y: 0.25, z: 0.63, role: "chalcogen-top" },
      { label: "X", x: 0.75, y: 0.75, z: 0.37, role: "chalcogen-bot" },
    ],
    stoichiometryRatio: [1, 1, 2],
    coordination: [8, 4, 3],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 3) return false;
      const hasSpacer = elements.some(e => ["La", "Ce", "Pr", "Nd", "Sr", "Ba", "Bi"].includes(e));
      const hasTM = elements.some(e => ["Bi", "Pb", "Sn", "Sb", "Ni", "Cu"].includes(e));
      const hasChalcogen = elements.some(e => ["S", "Se", "Te", "O"].includes(e));
      return hasSpacer && hasTM && hasChalcogen;
    },
  },
  {
    name: "CaBe2Ge2",
    spaceGroup: "P4/nmm",
    latticeType: "tetragonal",
    cOverA: 2.4,
    sites: [
      { label: "A", x: 0.25, y: 0.25, z: 0.0, role: "spacer" },
      { label: "A", x: 0.75, y: 0.75, z: 0.5, role: "spacer" },
      { label: "B", x: 0.75, y: 0.25, z: 0.14, role: "TM-layer-1" },
      { label: "B", x: 0.25, y: 0.75, z: 0.64, role: "TM-layer-2" },
      { label: "C", x: 0.25, y: 0.25, z: 0.25, role: "sp-up" },
      { label: "C", x: 0.75, y: 0.75, z: 0.75, role: "sp-down" },
    ],
    stoichiometryRatio: [1, 2, 2],
    coordination: [8, 4, 4],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 3) return false;
      const hasSpacer = elements.some(e => CATIONS_LARGE.has(e));
      const hasTM = elements.some(e => isTransitionMetal(e));
      const hasSp = elements.some(e => ["Si", "Ge", "Sn", "Sb", "As", "P", "Ga", "Al"].includes(e));
      return hasSpacer && hasTM && hasSp;
    },
  },
  {
    name: "Borocarbide-1221",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 3.5,
    sites: [
      { label: "R", x: 0.0, y: 0.0, z: 0.0, role: "rare-earth" },
      { label: "T", x: 0.0, y: 0.5, z: 0.25, role: "TM-1" },
      { label: "T", x: 0.5, y: 0.0, z: 0.25, role: "TM-2" },
      { label: "B", x: 0.0, y: 0.0, z: 0.35, role: "boron-1" },
      { label: "B", x: 0.0, y: 0.0, z: 0.65, role: "boron-2" },
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "carbon" },
    ],
    stoichiometryRatio: [1, 2, 2, 1],
    coordination: [8, 4, 3, 4],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 4) return false;
      const hasRE = elements.some(e => ["Y", "Lu", "Tm", "Er", "Ho", "Dy", "Sc", "La"].includes(e));
      const hasTM = elements.some(e => ["Ni", "Pd", "Pt", "Co", "Rh"].includes(e));
      const hasB = elements.includes("B");
      const hasC = elements.includes("C");
      return hasRE && hasTM && hasB && hasC;
    },
  },
  {
    name: "Half-Heusler-XYZ",
    spaceGroup: "F-43m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "X", x: 0.0, y: 0.0, z: 0.0, role: "electropositive" },
      { label: "Y", x: 0.25, y: 0.25, z: 0.25, role: "TM" },
      { label: "Z", x: 0.5, y: 0.5, z: 0.5, role: "main-group" },
    ],
    stoichiometryRatio: [1, 1, 1],
    coordination: [4, 4, 4],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 3) return false;
      const hasRE = elements.some(e => ["Y", "La", "Lu", "Sc", "Gd", "Nd", "Ce"].includes(e));
      const hasTM = elements.some(e => ["Pt", "Pd", "Ni", "Au", "Ir", "Rh"].includes(e));
      const hasSp = elements.some(e => ["Bi", "Sb", "Sn", "Pb", "In", "Te"].includes(e));
      return hasRE && hasTM && hasSp;
    },
  },
  {
    name: "Kagome-AV3Sb5",
    spaceGroup: "P6/mmm",
    latticeType: "hexagonal",
    cOverA: 1.7,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "alkali-spacer" },
      { label: "V", x: 0.5, y: 0.0, z: 0.5, role: "kagome-1" },
      { label: "V", x: 0.0, y: 0.5, z: 0.5, role: "kagome-2" },
      { label: "V", x: 0.5, y: 0.5, z: 0.5, role: "kagome-3" },
      { label: "Sb", x: 0.333, y: 0.667, z: 0.0, role: "pnictogen-1" },
      { label: "Sb", x: 0.667, y: 0.333, z: 0.0, role: "pnictogen-2" },
      { label: "Sb", x: 0.0, y: 0.0, z: 0.5, role: "pnictogen-hub" },
      { label: "Sb", x: 0.333, y: 0.667, z: 0.5, role: "pnictogen-3" },
      { label: "Sb", x: 0.667, y: 0.333, z: 0.5, role: "pnictogen-4" },
    ],
    stoichiometryRatio: [1, 3, 5],
    coordination: [12, 6, 6],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 3) return false;
      const hasAlkali = elements.some(e => ["K", "Rb", "Cs", "Na"].includes(e));
      const hasTM = elements.some(e => ["V", "Ti", "Nb", "Ta", "Cr"].includes(e));
      const hasPn = elements.some(e => ["Sb", "Sn", "Bi", "As", "Ge"].includes(e));
      return hasAlkali && hasTM && hasPn;
    },
  },
  {
    name: "FilledSkutterudite-RT4X12",
    spaceGroup: "Im-3",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "R", x: 0.0, y: 0.0, z: 0.0, role: "filler-cage" },
      { label: "T", x: 0.25, y: 0.25, z: 0.25, role: "TM-1" },
      { label: "T", x: 0.75, y: 0.25, z: 0.25, role: "TM-2" },
      { label: "T", x: 0.25, y: 0.75, z: 0.25, role: "TM-3" },
      { label: "T", x: 0.25, y: 0.25, z: 0.75, role: "TM-4" },
      { label: "X", x: 0.0, y: 0.35, z: 0.15, role: "pnictogen-1" },
      { label: "X", x: 0.15, y: 0.0, z: 0.35, role: "pnictogen-2" },
      { label: "X", x: 0.35, y: 0.15, z: 0.0, role: "pnictogen-3" },
      { label: "X", x: 0.0, y: 0.65, z: 0.85, role: "pnictogen-4" },
      { label: "X", x: 0.85, y: 0.0, z: 0.65, role: "pnictogen-5" },
      { label: "X", x: 0.65, y: 0.85, z: 0.0, role: "pnictogen-6" },
      { label: "X", x: 0.0, y: 0.65, z: 0.15, role: "pnictogen-7" },
      { label: "X", x: 0.15, y: 0.0, z: 0.65, role: "pnictogen-8" },
      { label: "X", x: 0.65, y: 0.15, z: 0.0, role: "pnictogen-9" },
      { label: "X", x: 0.0, y: 0.35, z: 0.85, role: "pnictogen-10" },
      { label: "X", x: 0.85, y: 0.0, z: 0.35, role: "pnictogen-11" },
      { label: "X", x: 0.35, y: 0.85, z: 0.0, role: "pnictogen-12" },
    ],
    stoichiometryRatio: [1, 4, 12],
    coordination: [12, 6, 3],
    chemistryRules: (elements: string[]) => {
      if (elements.length !== 3) return false;
      const hasFiller = elements.some(e => ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Yb", "Ba", "Sr", "Ca"].includes(e));
      const hasTM = elements.some(e => ["Os", "Ru", "Fe", "Co", "Rh", "Ir"].includes(e));
      const hasPn = elements.some(e => ["Sb", "As", "P", "Bi"].includes(e));
      return hasFiller && hasTM && hasPn;
    },
  },
  {
    name: "A15-W3O",
    spaceGroup: "Pm-3n",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "body-center" },
      { label: "B", x: 0.25, y: 0.0, z: 0.5, role: "chain" },
      { label: "B", x: 0.75, y: 0.0, z: 0.5, role: "chain" },
      { label: "B", x: 0.5, y: 0.25, z: 0.0, role: "chain" },
      { label: "B", x: 0.5, y: 0.75, z: 0.0, role: "chain" },
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "chain" },
      { label: "B", x: 0.0, y: 0.5, z: 0.75, role: "chain" },
    ],
    stoichiometryRatio: [1, 3],
    coordination: [12, 14],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Nb", "V", "Ti", "Zr", "Mo", "Ta"].includes(e));
    },
  },
  {
    name: "Fluorite-C1",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation" },
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "anion" },
      { label: "B", x: 0.75, y: 0.75, z: 0.75, role: "anion" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [8, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasLarge = elements.some(e => ["Zr", "Hf", "Ce", "Th", "U", "Pb", "Ca"].includes(e));
      const hasAnion = elements.some(e => ["O", "F", "Cl", "H"].includes(e));
      return hasLarge && hasAnion;
    },
  },
  {
    name: "Cu3Au-L1_2",
    spaceGroup: "Pm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner" },
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "face-center" },
      { label: "B", x: 0.5, y: 0.0, z: 0.5, role: "face-center" },
      { label: "B", x: 0.0, y: 0.5, z: 0.5, role: "face-center" },
    ],
    stoichiometryRatio: [1, 3],
    coordination: [12, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Au", "Pt", "Pd", "Ni", "Al", "Sn", "In"].includes(e));
    },
  },
  {
    name: "ZincBlende-B3",
    spaceGroup: "F-43m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "fcc-site" },
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "tetrahedral-site" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Zn", "Cd", "Hg", "Al", "Ga", "In", "Si", "Ge"].includes(e));
    },
  },
  {
    name: "C14-Laves",
    spaceGroup: "P6_3/mmc",
    latticeType: "hexagonal",
    cOverA: 1.63,
    sites: [
      { label: "A", x: 0.333, y: 0.667, z: 0.063, role: "large-atom" },
      { label: "B", x: 0.5, y: 0.0, z: 0.0, role: "small-atom-1" },
      { label: "B", x: 0.833, y: 0.667, z: 0.25, role: "small-atom-2" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [16, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => isTransitionMetal(e) || isRareEarth(e) || ["Mg"].includes(e));
    },
  },
  {
    name: "NiAs-B8_1",
    spaceGroup: "P6_3/mmc",
    latticeType: "hexagonal",
    cOverA: 1.39,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal-octahedral" },
      { label: "B", x: 0.333, y: 0.667, z: 0.25, role: "pnictogen-trigonal" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Ni", "Fe", "Co", "Mn", "Cr", "Ti", "V"].includes(e));
      const hasPn = elements.some(e => ["As", "Sb", "Bi", "Sn", "Te"].includes(e));
      return hasTM && hasPn;
    },
  },
  {
    name: "Wurtzite-B4",
    spaceGroup: "P6_3mc",
    latticeType: "hexagonal",
    cOverA: 1.6,
    sites: [
      { label: "A", x: 0.333, y: 0.667, z: 0.0, role: "cation" },
      { label: "B", x: 0.333, y: 0.667, z: 0.375, role: "anion" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Zn", "Cd", "Al", "Ga", "In", "N", "O"].includes(e));
    },
  },
  {
    name: "CdI2-C6",
    spaceGroup: "P-3m1",
    latticeType: "hexagonal",
    cOverA: 1.5,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal" },
      { label: "B", x: 0.333, y: 0.667, z: 0.25, role: "anion-top" },
      { label: "B", x: 0.667, y: 0.333, z: 0.75, role: "anion-bot" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Ti", "Zr", "V", "Fe", "Co", "Ni", "Mg"].includes(e)) && elements.some(e => ["Cl", "Br", "I", "S", "Se", "Te"].includes(e));
    },
  },
  {
    name: "Rutile-C4",
    spaceGroup: "P4_2/mnm",
    latticeType: "tetragonal",
    cOverA: 0.64,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "cation" },
      { label: "B", x: 0.3, y: 0.3, z: 0.0, role: "anion" },
      { label: "B", x: 0.7, y: 0.7, z: 0.0, role: "anion" },
      { label: "B", x: 0.8, y: 0.2, z: 0.5, role: "anion" },
      { label: "B", x: 0.2, y: 0.8, z: 0.5, role: "anion" },
    ],
    stoichiometryRatio: [2, 4],
    coordination: [6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Ti", "Sn", "Pb", "Mn", "Cr", "Ru", "Ir", "Ge"].includes(e)) && elements.includes("O");
    },
  },
  {
    name: "CuAl2-C16",
    spaceGroup: "I4/mcm",
    latticeType: "tetragonal",
    cOverA: 0.8,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.25, role: "minor-metal" },
      { label: "B", x: 0.16, y: 0.66, z: 0.0, role: "major-metal" },
    ],
    stoichiometryRatio: [4, 8],
    coordination: [8, 11],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => isTransitionMetal(e)) && elements.some(e => ["Al", "Ga", "In", "Sn", "Si"].includes(e));
    },
  },
  {
    name: "ZrBe2-type",
    spaceGroup: "P6/mmm",
    latticeType: "hexagonal",
    cOverA: 0.85,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "spacer" },
      { label: "B", x: 0.333, y: 0.667, z: 0.5, role: "honeycomb" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [12, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Zr", "Hf", "Ti", "Sc", "Y", "La"].includes(e)) && elements.includes("Be");
    },
  },
  {
    name: "MoC-type",
    spaceGroup: "P-6m2",
    latticeType: "hexagonal",
    cOverA: 0.95,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal" },
      { label: "B", x: 0.333, y: 0.667, z: 0.5, role: "interstitial" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Mo", "W", "Nb", "Ta"].includes(e)) && elements.some(e => ["C", "N"].includes(e));
    },
  },
  {
    name: "Th3P4-D7_3",
    spaceGroup: "I-43d",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.062, y: 0.062, z: 0.062, role: "cation" },
      { label: "B", x: 0.25, y: 0.0, z: 0.0, role: "anion" },
    ],
    stoichiometryRatio: [12, 16],
    coordination: [8, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => isRareEarth(e) || ["Th", "U"].includes(e)) && elements.some(e => ["P", "As", "Sb", "Bi", "S", "Se", "Te"].includes(e));
    },
  },
  {
    name: "AuCu-L1_0",
    spaceGroup: "P4/mmm",
    latticeType: "tetragonal",
    cOverA: 0.9,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "body-center" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [12, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Pt", "Pd", "Au", "Cu", "Fe", "Co", "Ni", "Mn", "Ti", "Al"].includes(e));
    },
  },
  {
    name: "CoSn-B35",
    spaceGroup: "P6/mmm",
    latticeType: "hexagonal",
    cOverA: 0.8,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "pnictogen-1a" },
      { label: "A", x: 0.333, y: 0.667, z: 0.5, role: "pnictogen-2d" },
      { label: "A", x: 0.667, y: 0.333, z: 0.5, role: "pnictogen-2d" },
      { label: "B", x: 0.5, y: 0.0, z: 0.0, role: "kagome-metal-3f" },
      { label: "B", x: 0.0, y: 0.5, z: 0.0, role: "kagome-metal-3f" },
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "kagome-metal-3f" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [12, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => ["Co", "Fe", "Ni", "Mn"].includes(e)) && elements.some(e => ["Sn", "Ge", "Sb"].includes(e));
    },
  },
  {
    name: "MgCu2-C15",
    spaceGroup: "Fd-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.125, y: 0.125, z: 0.125, role: "large-atom" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "small-atom-network" },
    ],
    stoichiometryRatio: [8, 16],
    coordination: [16, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.some(e => isRareEarth(e) || ["Mg", "Ca", "Y", "Zr", "Hf", "Ti"].includes(e)) && elements.some(e => isTransitionMetal(e) || ["Al"].includes(e));
    },
  },
  {
    name: "AIB2-variant",
    spaceGroup: "P-6m2",
    latticeType: "hexagonal",
    cOverA: 1.1,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal-1" },
      { label: "B", x: 0.333, y: 0.667, z: 0.5, role: "metalloid-1" },
      { label: "C", x: 0.667, y: 0.333, z: 0.5, role: "metalloid-2" },
    ],
    stoichiometryRatio: [1, 1, 1],
    coordination: [12, 6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      return elements.some(e => isRareEarth(e) || isTransitionMetal(e)) && elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Sb", "Bi"].includes(e));
    },
  },

  // ── Elemental structures (BCC / FCC / HCP) ───────────────────────────────
  // Critical for modelling the 30+ elemental superconductors in the training set.

  {
    name: "BCC-W",
    spaceGroup: "Im-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "bcc-corner" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "bcc-body" },
    ],
    stoichiometryRatio: [1],
    coordination: [8],
    chemistryRules: (elements) => {
      if (elements.length !== 1) return false;
      return ["W", "Mo", "Cr", "Fe", "Nb", "Ta", "V", "Ba", "K", "Na", "Li", "Rb", "Cs", "Eu", "Yb"].includes(elements[0]);
    },
  },
  {
    name: "FCC-Cu",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "fcc-corner" },
      { label: "A", x: 0.5, y: 0.5, z: 0.0, role: "fcc-face-1" },
      { label: "A", x: 0.5, y: 0.0, z: 0.5, role: "fcc-face-2" },
      { label: "A", x: 0.0, y: 0.5, z: 0.5, role: "fcc-face-3" },
    ],
    stoichiometryRatio: [1],
    coordination: [12],
    chemistryRules: (elements) => {
      if (elements.length !== 1) return false;
      return ["Cu", "Ni", "Pd", "Pt", "Au", "Ag", "Al", "Ca", "Sr", "Pb", "Ce", "La", "Pr", "Nd",
              "Rh", "Ir", "Th", "Yb", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Lu"].includes(elements[0]);
    },
  },
  {
    name: "HCP-elemental",
    spaceGroup: "P6_3/mmc",
    latticeType: "hexagonal",
    cOverA: 1.633,  // ideal HCP
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "hcp-site-1" },
      { label: "A", x: 0.333, y: 0.667, z: 0.5, role: "hcp-site-2" },
    ],
    stoichiometryRatio: [1],
    coordination: [12],
    chemistryRules: (elements) => {
      if (elements.length !== 1) return false;
      return ["Ti", "Zr", "Hf", "Mg", "Co", "Os", "Ru", "Re", "Tc", "Y", "Sc", "Tl", "Be",
              "Cd", "Zn", "Lu", "Tm", "Er", "Ho", "Dy", "Tb", "Gd", "Sm", "Nd", "Pr"].includes(elements[0]);
    },
  },

  // ── Orthorhombic prototypes (SG 16-74) — previously missing entirely ──────

  {
    // FeB-type: many binary intermetallics (NbB, TaB, VB, CrB, MoB, WB, etc.)
    // Space group 62 (Pnma). Commonly seen for TM borides and carbides.
    name: "FeB-Pnma",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",  // approximate as tetragonal for site generation
    cOverA: 1.15,
    sites: [
      { label: "A", x: 0.18, y: 0.25, z: 0.12, role: "TM-chain" },
      { label: "A", x: 0.32, y: 0.75, z: 0.62, role: "TM-chain" },
      { label: "A", x: 0.68, y: 0.25, z: 0.88, role: "TM-chain" },
      { label: "A", x: 0.82, y: 0.75, z: 0.38, role: "TM-chain" },
      { label: "B", x: 0.03, y: 0.25, z: 0.60, role: "pnictogen" },
      { label: "B", x: 0.47, y: 0.75, z: 0.10, role: "pnictogen" },
      { label: "B", x: 0.53, y: 0.25, z: 0.40, role: "pnictogen" },
      { label: "B", x: 0.97, y: 0.75, z: 0.90, role: "pnictogen" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [7, 9],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Nb", "Ta", "V", "Cr", "Mo", "W", "Fe", "Mn", "Re", "Ti", "Zr", "Hf"].includes(e));
      const hasSp = elements.some(e => ["B", "C", "N", "P", "Si", "Ge", "Al"].includes(e));
      return hasTM && hasSp;
    },
  },
  {
    // CrB-type (Cmcm, SG 63): many rare-earth diborides, CrB, MoB high-Tc phases
    name: "CrB-Cmcm",
    spaceGroup: "Cmcm",
    latticeType: "tetragonal",
    cOverA: 2.5,
    sites: [
      { label: "A", x: 0.0, y: 0.15, z: 0.25, role: "TM-zigzag" },
      { label: "A", x: 0.0, y: 0.85, z: 0.75, role: "TM-zigzag" },
      { label: "B", x: 0.0, y: 0.44, z: 0.25, role: "sp-chain" },
      { label: "B", x: 0.0, y: 0.56, z: 0.75, role: "sp-chain" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [7, 5],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => isTransitionMetal(e) || isRareEarth(e));
      const hasSp = elements.some(e => ["B", "C", "N", "Si", "P", "Ge", "Al"].includes(e));
      return hasTM && hasSp;
    },
  },
  {
    // GdFeO3-type (Pbnm/Pnma, SG 62): distorted perovskite — manganites, nickelates
    // Many oxide superconductors (RNiO3, RMnO3) adopt this structure under pressure
    name: "GdFeO3-Pbnm",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 1.41,
    sites: [
      { label: "A", x: 0.06, y: 0.25, z: 0.98, role: "A-site-large" },
      { label: "A", x: 0.44, y: 0.75, z: 0.48, role: "A-site-large" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "B-site-TM" },
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "B-site-TM" },
      { label: "O", x: 0.1, y: 0.25, z: 0.1, role: "apical-O" },
      { label: "O", x: 0.4, y: 0.75, z: 0.6, role: "apical-O" },
      { label: "O", x: 0.7, y: 0.04, z: 0.3, role: "equatorial-O" },
      { label: "O", x: 0.8, y: 0.46, z: 0.2, role: "equatorial-O" },
      { label: "O", x: 0.3, y: 0.96, z: 0.7, role: "equatorial-O" },
      { label: "O", x: 0.2, y: 0.54, z: 0.8, role: "equatorial-O" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [8, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasRE = elements.some(e => isRareEarth(e) || ["Bi", "Pb", "Ba", "Sr", "Ca"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Ni", "Mn", "Co", "Cr", "Ru", "Cu"].includes(e));
      return hasRE && hasTM && elements.includes("O");
    },
  },
  {
    // Cu2Sb / PbFCl-type (P4/nmm, SG 129): parent structure for iron pnictide 122-type
    // Also parent of many SC chalcogenides: Cu2SbSe2, Cu2SbS2
    name: "Cu2Sb-P4nmm",
    spaceGroup: "P4/nmm",
    latticeType: "tetragonal",
    cOverA: 1.85,
    sites: [
      { label: "A", x: 0.25, y: 0.25, z: 0.0, role: "Cu-square-net" },
      { label: "A", x: 0.75, y: 0.75, z: 0.0, role: "Cu-square-net" },
      { label: "B", x: 0.25, y: 0.75, z: 0.30, role: "Cu-tetrahedral" },
      { label: "B", x: 0.75, y: 0.25, z: 0.70, role: "Cu-tetrahedral" },
      { label: "C", x: 0.75, y: 0.25, z: 0.27, role: "Sb-layer" },
      { label: "C", x: 0.25, y: 0.75, z: 0.73, role: "Sb-layer" },
    ],
    stoichiometryRatio: [2, 1],
    coordination: [4, 8],
    chemistryRules: (elements) => {
      if (elements.length < 2 || elements.length > 3) return false;
      const hasTM = elements.some(e => ["Cu", "Ag", "Ni", "Fe", "Co", "Mn"].includes(e));
      const hasSp = elements.some(e => ["Sb", "As", "Bi", "P", "Sn", "Ge", "Se", "Te", "S"].includes(e));
      return hasTM && hasSp;
    },
  },

  // ── Trigonal/Rhombohedral (SG 143-167) — expanded coverage ──────────────

  {
    // Bi2Te3-type (R-3m, SG 166): topological insulators, thermoelectrics, SC under pressure
    // Quintuple-layer structure: Te-Bi-Te-Bi-Te stacking with van der Waals gap
    name: "Bi2Te3-R3m",
    spaceGroup: "R-3m",
    latticeType: "hexagonal",
    cOverA: 6.85,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Te-vdW" },
      { label: "B", x: 0.0, y: 0.0, z: 0.40, role: "Bi-inner" },
      { label: "B", x: 0.0, y: 0.0, z: 0.60, role: "Bi-inner" },
      { label: "A", x: 0.0, y: 0.0, z: 0.21, role: "Te-inner" },
      { label: "A", x: 0.0, y: 0.0, z: 0.79, role: "Te-inner" },
    ],
    stoichiometryRatio: [3, 2],
    coordination: [6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasPnictogen = elements.some(e => ["Bi", "Sb", "As"].includes(e));
      const hasChalcogen = elements.some(e => ["Te", "Se", "S"].includes(e));
      return hasPnictogen && hasChalcogen;
    },
  },
  {
    // MnP-type (Pnma, SG 62): many binary TM phosphides/arsenides — MnAs, FeP, CrAs
    // Different Pnma prototype from FeB; MnP has zigzag chains vs FeB has linear
    name: "MnP-Pnma",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 1.6,
    sites: [
      { label: "A", x: 0.0, y: 0.2, z: 0.06, role: "TM-helical" },
      { label: "A", x: 0.5, y: 0.3, z: 0.44, role: "TM-helical" },
      { label: "A", x: 0.0, y: 0.8, z: 0.94, role: "TM-helical" },
      { label: "A", x: 0.5, y: 0.7, z: 0.56, role: "TM-helical" },
      { label: "B", x: 0.2, y: 0.0, z: 0.39, role: "pnictogen" },
      { label: "B", x: 0.3, y: 0.5, z: 0.11, role: "pnictogen" },
      { label: "B", x: 0.8, y: 0.0, z: 0.61, role: "pnictogen" },
      { label: "B", x: 0.7, y: 0.5, z: 0.89, role: "pnictogen" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [6, 7],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Mn", "Fe", "Cr", "Co", "Ni", "Ru", "Rh"].includes(e));
      const hasPn = elements.some(e => ["As", "P", "Sb", "S", "Se", "Te"].includes(e));
      return hasTM && hasPn;
    },
  },

  // ── Monoclinic (SG 3-15) — previously missing entirely ───────────────────

  {
    // C2/m (SG 12): very common for layered SC and complex oxides.
    // Includes: FeSe polymorphs, β-FeSe, many ternary oxychalcogenides, REFeAsO-related
    name: "C2m-monoclinic",
    spaceGroup: "C2/m",
    latticeType: "tetragonal",  // approximate as quasi-tetragonal for site generation
    cOverA: 1.8,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal-1" },
      { label: "A", x: 0.5, y: 0.5, z: 0.0, role: "metal-2" },
      { label: "B", x: 0.25, y: 0.25, z: 0.5, role: "anion-1" },
      { label: "B", x: 0.75, y: 0.75, z: 0.5, role: "anion-2" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [6, 6],
    chemistryRules: (elements) => {
      if (elements.length < 2 || elements.length > 4) return false;
      return elements.some(e => isTransitionMetal(e) || isRareEarth(e));
    },
  },

  // ── High-symmetry cubic — superconducting pyrochlore / spinel ────────────

  {
    // Spinel AB2X4 (Fd-3m, SG 227): superconducting spinels CuRh2Se4, CuV2S4, LiTi2O4
    // Distinct from Laves-C15 (also Fd-3m) by stoichiometry and site occupancy
    name: "Spinel-AB2X4",
    spaceGroup: "Fd-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.125, y: 0.125, z: 0.125, role: "tetrahedral-A" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "octahedral-B-1" },
      { label: "B", x: 0.5, y: 0.75, z: 0.75, role: "octahedral-B-2" },
      { label: "X", x: 0.25, y: 0.25, z: 0.25, role: "anion-1" },
      { label: "X", x: 0.25, y: 0.75, z: 0.75, role: "anion-2" },
      { label: "X", x: 0.75, y: 0.25, z: 0.75, role: "anion-3" },
      { label: "X", x: 0.75, y: 0.75, z: 0.25, role: "anion-4" },
    ],
    stoichiometryRatio: [1, 2, 4],
    coordination: [4, 6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasTM = elements.some(e => isTransitionMetal(e));
      const hasAnion = elements.some(e => ["O", "S", "Se", "Te", "N"].includes(e));
      const hasAsite = elements.some(e => ["Cu", "Zn", "Mg", "Fe", "Co", "Ni", "Li", "Mn", "Cd"].includes(e));
      return hasTM && hasAnion && hasAsite;
    },
  },
  {
    // Pyrochlore A2B2O7 (Fd-3m, SG 227): KOs2O6 (Tc≈9.6 K), Cd2Re2O7, superconducting pyrochlores
    name: "Pyrochlore-A2B2O7",
    spaceGroup: "Fd-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "A-16d" },
      { label: "A", x: 0.25, y: 0.25, z: 0.75, role: "A-16d" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-16c" },
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "B-16c" },
      { label: "O", x: 0.31, y: 0.125, z: 0.125, role: "O-48f" },
      { label: "O", x: 0.125, y: 0.31, z: 0.125, role: "O-48f" },
      { label: "O", x: 0.125, y: 0.125, z: 0.31, role: "O-48f" },
      { label: "O", x: 0.69, y: 0.875, z: 0.875, role: "O-48f" },
      { label: "O", x: 0.875, y: 0.69, z: 0.875, role: "O-48f" },
      { label: "O", x: 0.875, y: 0.875, z: 0.69, role: "O-48f" },
      { label: "O", x: 0.375, y: 0.375, z: 0.375, role: "O-8b" },
    ],
    stoichiometryRatio: [2, 2, 7],
    coordination: [8, 6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasA = elements.some(e => ["K", "Na", "Cs", "Rb", "Ca", "Cd", "Bi", "Pb", "Tl", "Y", "La"].includes(e));
      const hasB = elements.some(e => ["Os", "Re", "Ir", "Ru", "Mo", "W", "Nb", "Ta", "Ti", "Zr"].includes(e));
      return hasA && hasB && elements.includes("O");
    },
  },
  {
    // CsCl-B2 (Pm-3m, SG 221): CsCl, TlBr, AuCd ordered alloys, many binary SC phases
    // Distinguished from Perovskite by 2-atom basis rather than 5
    name: "CsCl-B2",
    spaceGroup: "Pm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "body-center" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [8, 8],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // Distinct from Perovskite — binary only, at least one non-anion
      const noAnion = !elements.some(e => ["O", "F"].includes(e));
      const hasMetal = elements.some(e => isTransitionMetal(e) || ["Cs", "Tl", "Au", "Al", "Ag", "Cd", "In", "Cu"].includes(e));
      return noAnion && hasMetal;
    },
  },

  // ── MAX-phase (P6_3/mmc, SG 194): Ti2AlC, Ti3SiC2, Nb2AlC — layered ternaries ──

  {
    // MAX-phase Mn+1AXn: M = early TM, A = A-group, X = C or N
    // Superconducting examples: Mo2GaC (Tc~4 K), Nb2SnC (Tc~7.8 K)
    name: "MAX-phase-M2AX",
    spaceGroup: "P6_3/mmc",
    latticeType: "hexagonal",
    cOverA: 4.0,
    sites: [
      { label: "M", x: 0.333, y: 0.667, z: 0.085, role: "TM-layer-1" },
      { label: "M", x: 0.667, y: 0.333, z: 0.915, role: "TM-layer-2" },
      { label: "M", x: 0.333, y: 0.667, z: 0.585, role: "TM-layer-3" },
      { label: "M", x: 0.667, y: 0.333, z: 0.415, role: "TM-layer-4" },
      { label: "A", x: 0.333, y: 0.667, z: 0.25, role: "A-layer" },
      { label: "A", x: 0.667, y: 0.333, z: 0.75, role: "A-layer" },
      { label: "X", x: 0.0, y: 0.0, z: 0.0, role: "X-interstitial-1" },
      { label: "X", x: 0.0, y: 0.0, z: 0.5, role: "X-interstitial-2" },
    ],
    stoichiometryRatio: [2, 1, 1],
    coordination: [6, 9, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasTM = elements.some(e => ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"].includes(e));
      const hasA = elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Pb", "P", "As", "S"].includes(e));
      const hasX = elements.some(e => ["C", "N"].includes(e));
      return hasTM && hasA && hasX;
    },
  },
];

function parseFormulaCounts(formula: string): Record<string, number> {
  return parseFormulaCountsCanonical(formula);
}

function gcd(a: number, b: number): number {
  a = Math.abs(Math.round(a));
  b = Math.abs(Math.round(b));
  while (b) { [a, b] = [b, a % b]; }
  return a || 1;
}

function getReducedRatio(vals: number[]): number[] {
  const rounded = vals.map(v => Math.round(v));
  const g = rounded.reduce((a, b) => gcd(a, b));
  return rounded.map(r => r / g);
}

function sortElementsBySite(elements: string[], counts: Record<string, number>, template: PrototypeTemplate): Record<string, string> | null {
  const nSites = new Set(template.sites.map(s => s.label)).size;
  if (elements.length !== nSites) return null;

  const siteCounts: Record<string, number> = {};
  for (const s of template.sites) {
    siteCounts[s.label] = (siteCounts[s.label] || 0) + 1;
  }
  const siteLabels = Object.keys(siteCounts);
  const siteRatios = getReducedRatio(siteLabels.map(l => siteCounts[l]));
  const elemRatios = getReducedRatio(elements.map(e => counts[e]));

  const sortedSR = [...siteRatios].sort((a, b) => b - a);
  const sortedER = [...elemRatios].sort((a, b) => b - a);

  if (sortedSR.length !== sortedER.length) return null;
  for (let i = 0; i < sortedSR.length; i++) {
    if (sortedSR[i] !== sortedER[i]) return null;
  }

  const siteOrder = siteLabels.map((l, i) => ({ label: l, ratio: siteRatios[i] }))
    .sort((a, b) => b.ratio - a.ratio);
  const elemOrder = elements.map((e, i) => ({ el: e, ratio: elemRatios[i] }))
    .sort((a, b) => b.ratio - a.ratio || a.el.localeCompare(b.el));

  const siteByRatio: Record<number, string[]> = {};
  for (const s of siteOrder) {
    if (!siteByRatio[s.ratio]) siteByRatio[s.ratio] = [];
    siteByRatio[s.ratio].push(s.label);
  }
  const elemByRatio: Record<number, string[]> = {};
  const elemReduced = elements.map((e, i) => ({
    el: e,
    reduced: elemRatios[i]
  }));
  for (const er of elemReduced) {
    if (!elemByRatio[er.reduced]) elemByRatio[er.reduced] = [];
    elemByRatio[er.reduced].push(er.el);
  }

  const siteMap: Record<string, string> = {};
  const usedElements = new Set<string>();
  for (const ratio of Object.keys(siteByRatio).map(Number).sort((a, b) => b - a)) {
    const sLabels = siteByRatio[ratio];
    const eLabels = elemByRatio[ratio];
    if (!eLabels || sLabels.length !== eLabels.length) return null;

    if (sLabels.length > 1) {
      const roleMap: Record<string, string> = {};
      for (const l of sLabels) {
        const site = template.sites.find(s => s.label === l);
        roleMap[l] = site?.role || "";
      }
      eLabels.sort((a, b) => {
        const aIsAnion = ANIONS.has(a);
        const bIsAnion = ANIONS.has(b);
        if (aIsAnion !== bIsAnion) return aIsAnion ? 1 : -1;
        const aIsLarge = CATIONS_LARGE.has(a);
        const bIsLarge = CATIONS_LARGE.has(b);
        if (aIsLarge !== bIsLarge) return aIsLarge ? -1 : 1;
        const aIsTM = CATIONS_SMALL_TM.has(a);
        const bIsTM = CATIONS_SMALL_TM.has(b);
        if (aIsTM !== bIsTM) return aIsTM ? 1 : -1;
        const rA = IONIC_RADII[a] || 0.8;
        const rB = IONIC_RADII[b] || 0.8;
        return rB - rA;
      });
      const anionRoles = ["chalcogen", "anion", "oxide", "pnictide", "halide", "apical-O", "planar-O", "in-plane-O"];
      const spacerRoles = ["spacer", "large-atom", "guest", "rare-earth"];
      sLabels.sort((a, b) => {
        const roleA = roleMap[a] || "";
        const roleB = roleMap[b] || "";
        const aIsAnionRole = anionRoles.some(r => roleA.includes(r));
        const bIsAnionRole = anionRoles.some(r => roleB.includes(r));
        if (aIsAnionRole !== bIsAnionRole) return aIsAnionRole ? 1 : -1;
        const aIsSpacer = spacerRoles.some(r => roleA.includes(r));
        const bIsSpacer = spacerRoles.some(r => roleB.includes(r));
        if (aIsSpacer !== bIsSpacer) return aIsSpacer ? -1 : 1;
        return 0;
      });
    }

    for (let i = 0; i < sLabels.length; i++) {
      if (usedElements.has(eLabels[i])) return null;
      siteMap[sLabels[i]] = eLabels[i];
      usedElements.add(eLabels[i]);
    }
  }

  return siteMap;
}

export function selectPrototype(formula: string): { template: PrototypeTemplate; siteMap: Record<string, string> } | null {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  for (const template of PROTOTYPE_TEMPLATES) {
    if (!template.chemistryRules(elements)) continue;
    const siteMap = sortElementsBySite(elements, counts, template);
    if (siteMap) return { template, siteMap };
  }

  return null;
}

function getAvgRadius(el: string): number {
  const data = getElementData(el);
  return (data?.atomicRadius ?? 150) / 100;
}

export function fillPrototype(formula: string): FilledPrototype | null {
  const result = selectPrototype(formula);
  if (!result) return null;

  const { template, siteMap } = result;
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  const { a, c } = estimateLatticeConstant(elements, counts, template);

  const atoms: { element: string; x: number; y: number; z: number }[] = [];
  const cos60 = 0.5;
  const sin60 = Math.sqrt(3) / 2;
  for (const site of template.sites) {
    const element = siteMap[site.label];
    if (!element) continue;

    let x: number, y: number, z: number;
    if (template.latticeType === "hexagonal") {
      x = a * (site.x + site.y * cos60);
      y = a * site.y * sin60;
      z = c * site.z;
    } else {
      x = a * site.x;
      y = a * site.y;
      z = (template.latticeType === "tetragonal" ? c : a) * site.z;
    }
    atoms.push({ element, x, y, z });
  }

  return {
    templateName: template.name,
    siteMap,
    atoms,
    latticeParam: a,
  };
}

export function computeBondValenceSum(formula: string): { bvs: Record<string, number>; valid: boolean; deviation: number } {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  const cations: string[] = [];
  const anions: string[] = [];
  for (const el of elements) {
    if (ANIONS.has(el)) anions.push(el);
    else cations.push(el);
  }

  if (cations.length === 0 || anions.length === 0) {
    return { bvs: {}, valid: true, deviation: 0 };
  }

  const bvs: Record<string, number> = {};
  let totalDeviation = 0;
  let nChecked = 0;

  const totalCationCount = cations.reduce((s, e) => s + (counts[e] || 1), 0);
  const totalAnionCount = anions.reduce((s, e) => s + (counts[e] || 1), 0);

  for (const cat of cations) {
    const oxStates = OXIDATION_STATES[cat] || [2];
    const expectedValence = Math.abs(oxStates[0]);
    const rCat = IONIC_RADII[cat] || 0.8;
    const nCat = counts[cat] || 1;

    let sumBV = 0;
    for (const an of anions) {
      const rAn = IONIC_RADII[an] || 1.4;
      const R0 = rCat + rAn;
      const avgCoord = Math.min(12, Math.max(2, Math.round(6 * totalAnionCount / totalCationCount)));
      const estimatedBondDist = R0 * (1.0 + 0.02 * Math.max(0, avgCoord - 6));
      const bondOrder = Math.exp((R0 - estimatedBondDist) / 0.37);
      const nAnions = counts[an] || 1;
      const coordContrib = (nAnions / nCat) * bondOrder;
      sumBV += coordContrib;
    }

    bvs[cat] = sumBV;
    const dev = Math.abs(sumBV - expectedValence) / Math.max(1, expectedValence);
    totalDeviation += dev;
    nChecked++;
  }

  const avgDev = nChecked > 0 ? totalDeviation / nChecked : 0;
  return { bvs, valid: avgDev < 0.5, deviation: avgDev };
}

export function checkIonicRadiusCompatibility(formula: string): { compatible: boolean; toleranceFactor: number | null; radiusRatio: number } {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  const cations = elements.filter(e => !ANIONS.has(e));
  const anions = elements.filter(e => ANIONS.has(e));

  if (cations.length === 0 || anions.length === 0) {
    return { compatible: true, toleranceFactor: null, radiusRatio: 1.0 };
  }

  const avgCatRadius = cations.reduce((s, e) => s + (IONIC_RADII[e] || 0.8) * (counts[e] || 1), 0) /
    cations.reduce((s, e) => s + (counts[e] || 1), 0);
  const avgAnRadius = anions.reduce((s, e) => s + (IONIC_RADII[e] || 1.4) * (counts[e] || 1), 0) /
    anions.reduce((s, e) => s + (counts[e] || 1), 0);

  const radiusRatio = avgCatRadius / Math.max(0.1, avgAnRadius);

  let toleranceFactor: number | null = null;
  if (cations.length >= 2 && anions.length >= 1) {
    const sorted = cations.sort((a, b) => (IONIC_RADII[b] || 0.8) - (IONIC_RADII[a] || 0.8));
    const rA = IONIC_RADII[sorted[0]] || 1.2;
    const rB = IONIC_RADII[sorted[1]] || 0.6;
    const rX = avgAnRadius;
    toleranceFactor = (rA + rX) / (Math.sqrt(2) * (rB + rX));
  }

  let compatible = true;
  if (radiusRatio < 0.15 || radiusRatio > 1.5) compatible = false;
  if (toleranceFactor !== null && (toleranceFactor < 0.7 || toleranceFactor > 1.1)) compatible = false;

  return { compatible, toleranceFactor, radiusRatio };
}

export function checkCoordinationMismatch(formula: string): { mismatch: number; plausible: boolean; details: string[] } {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const details: string[] = [];

  const cations = elements.filter(e => !ANIONS.has(e));
  const anionsList = elements.filter(e => ANIONS.has(e));

  if (cations.length === 0 || anionsList.length === 0) {
    return { mismatch: 0, plausible: true, details: ["All-metallic or all-nonmetal composition"] };
  }

  const totalCations = cations.reduce((s, e) => s + (counts[e] || 0), 0);
  const totalAnions = anionsList.reduce((s, e) => s + (counts[e] || 0), 0);

  const anionCationRatio = totalAnions / Math.max(1, totalCations);

  let expectedCoordNum = 6;
  const avgCatRadius = cations.reduce((s, e) => s + (IONIC_RADII[e] || 0.8), 0) / cations.length;
  const avgAnRadius = anionsList.reduce((s, e) => s + (IONIC_RADII[e] || 1.4), 0) / anionsList.length;
  const rRatio = avgCatRadius / Math.max(0.1, avgAnRadius);

  if (rRatio > 0.73) expectedCoordNum = 8;
  else if (rRatio > 0.41) expectedCoordNum = 6;
  else if (rRatio > 0.22) expectedCoordNum = 4;
  else expectedCoordNum = 3;

  const impliedCoord = anionCationRatio * expectedCoordNum;
  const mismatch = Math.abs(impliedCoord - expectedCoordNum) / expectedCoordNum;

  if (anionCationRatio > 6) {
    details.push(`Very high anion:cation ratio (${anionCationRatio.toFixed(1)})`);
  }
  const hFraction = (counts["H"] || 0) / Math.max(1, Object.values(counts).reduce((s, v) => s + v, 0));
  if (anionCationRatio < 0.5 && hFraction < 0.5) {
    details.push(`Very low anion:cation ratio (${anionCationRatio.toFixed(1)})`);
  }
  if (rRatio < 0.15) {
    details.push(`Cation too small for any coordination (r_ratio=${rRatio.toFixed(2)})`);
  }

  return { mismatch, plausible: mismatch < 0.6 && details.length === 0, details };
}

export interface PrototypeEnumResult {
  formula: string;
  prototype: string;
  spaceGroup: string;
  crystalSystem: string;
  latticeParam: number;
  cOverA: number;
  siteMap: Record<string, string>;
  compatibilityScore: number;
}

export function enumeratePrototypesForFormula(formula: string): PrototypeEnumResult[] {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length === 0) return [];

  const results: PrototypeEnumResult[] = [];

  for (const template of PROTOTYPE_TEMPLATES) {
    if (!template.chemistryRules(elements)) continue;
    const siteMap = sortElementsBySite(elements, counts, template);
    if (!siteMap) continue;

    const { a, c } = estimateLatticeConstant(elements, counts, template);
    let compatScore = 0.5;

    const useIonic = isIonicCompound(elements, template.name);
    let minDist = Infinity;
    const atoms: { x: number; y: number; z: number }[] = [];
    const cos60 = 0.5;
    const sin60 = Math.sqrt(3) / 2;
    for (const site of template.sites) {
      let x: number, y: number, z: number;
      if (template.latticeType === "hexagonal") {
        x = a * (site.x + site.y * cos60);
        y = a * site.y * sin60;
        z = c * site.z;
      } else {
        x = a * site.x;
        y = a * site.y;
        z = (template.latticeType === "tetragonal" ? c : a) * site.z;
      }
      atoms.push({ x, y, z });
    }
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const dx = atoms[i].x - atoms[j].x;
        const dy = atoms[i].y - atoms[j].y;
        const dz = atoms[i].z - atoms[j].z;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d > 0.01 && d < minDist) minDist = d;
      }
    }
    if (minDist > 1.0 && minDist < 10.0) compatScore += 0.2;
    else if (minDist <= 0.5) compatScore -= 0.3;

    const knownPacking = PACKING_FACTORS[template.name];
    if (knownPacking) compatScore += 0.1;

    const totalAtoms = elements.reduce((s, e) => s + Math.round(counts[e] || 1), 0);
    const vol = template.latticeType === "cubic" ? a * a * a
      : template.latticeType === "hexagonal" ? a * a * c * Math.sqrt(3) / 2
      : a * a * c;
    const volPerAtom = vol / Math.max(1, totalAtoms);
    if (volPerAtom >= 8 && volPerAtom <= 40) compatScore += 0.15;
    else if (volPerAtom < 5 || volPerAtom > 60) compatScore -= 0.2;

    compatScore = Math.max(0, Math.min(1, compatScore));

    results.push({
      formula,
      prototype: template.name,
      spaceGroup: template.spaceGroup,
      crystalSystem: template.latticeType,
      latticeParam: Number(a.toFixed(3)),
      cOverA: template.cOverA,
      siteMap,
      compatibilityScore: Number(compatScore.toFixed(3)),
    });
  }

  results.sort((a, b) => b.compatibilityScore - a.compatibilityScore);
  return results;
}

export function estimatePhononStability(formula: string): PreFilterResult {
  const reasons: string[] = [];
  let score = 1.0;

  const bvResult = computeBondValenceSum(formula);
  if (!bvResult.valid) {
    score -= 0.3;
    reasons.push(`bond valence deviation=${bvResult.deviation.toFixed(2)}`);
  }
  if (bvResult.deviation > 0.8) {
    score -= 0.2;
    reasons.push("extreme bond valence mismatch");
  }

  const radResult = checkIonicRadiusCompatibility(formula);
  if (!radResult.compatible) {
    score -= 0.3;
    if (radResult.toleranceFactor !== null) {
      reasons.push(`tolerance factor=${radResult.toleranceFactor.toFixed(3)} out of range`);
    } else {
      reasons.push(`radius ratio=${radResult.radiusRatio.toFixed(3)} incompatible`);
    }
  }

  const coordResult = checkCoordinationMismatch(formula);
  if (!coordResult.plausible) {
    score -= 0.2;
    reasons.push(...coordResult.details);
  }
  if (coordResult.mismatch > 0.8) {
    score -= 0.15;
    reasons.push(`coordination mismatch=${coordResult.mismatch.toFixed(2)}`);
  }

  score = Math.max(0, Math.min(1, score));

  return {
    stable: score >= 0.4,
    score,
    reasons,
  };
}
