import { getElementData, isTransitionMetal, isRareEarth } from "./elemental-data";

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

const IONIC_RADII: Record<string, number> = {
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
      { label: "X", x: 0.333, y: 0.667, z: -0.121, role: "chalcogen-bot" },
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
  {
    name: "Clathrate-H32",
    spaceGroup: "Im-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "metal-center" },
      { label: "H", x: 0.18, y: 0.18, z: 0.18, role: "cage" },
      { label: "H", x: 0.31, y: 0.31, z: 0.0, role: "cage" },
      { label: "H", x: 0.0, y: 0.31, z: 0.31, role: "cage" },
      { label: "H", x: 0.31, y: 0.0, z: 0.31, role: "cage" },
    ],
    stoichiometryRatio: [1, 10],
    coordination: [32, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      return elements.includes("H") && elements.some(e => isRareEarth(e) || ["Ca", "Sr", "Ba", "Y", "Sc", "Th"].includes(e));
    },
  },
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
    latticeType: "hexagonal",
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
    name: "YBCO-123",
    spaceGroup: "Pmmm",
    latticeType: "orthorhombic",
    cOverA: 3.0,
    sites: [
      { label: "A1", x: 0.5, y: 0.5, z: 0.18, role: "spacer" },
      { label: "A2", x: 0.5, y: 0.5, z: 0.82, role: "spacer" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "chain-TM" },
      { label: "B", x: 0.0, y: 0.0, z: 0.36, role: "plane-TM" },
      { label: "B", x: 0.0, y: 0.0, z: 0.64, role: "plane-TM" },
      { label: "X", x: 0.0, y: 0.5, z: 0.0, role: "chain-O" },
      { label: "X", x: 0.5, y: 0.0, z: 0.38, role: "plane-O" },
      { label: "X", x: 0.0, y: 0.5, z: 0.38, role: "plane-O" },
      { label: "X", x: 0.5, y: 0.0, z: 0.62, role: "plane-O" },
      { label: "X", x: 0.0, y: 0.5, z: 0.62, role: "plane-O" },
      { label: "X", x: 0.0, y: 0.0, z: 0.16, role: "apical-O" },
      { label: "X", x: 0.0, y: 0.0, z: 0.84, role: "apical-O" },
    ],
    stoichiometryRatio: [1, 2, 3, 7],
    coordination: [10, 5, 4, 2],
    chemistryRules: (elements: string[]) => {
      if (elements.length < 3 || elements.length > 4) return false;
      const hasSpacer = elements.some(e => ["Y", "La", "Nd", "Sm", "Gd", "Eu", "Ho", "Er", "Dy", "Yb"].includes(e));
      const hasAlkalineEarth = elements.some(e => ["Ba", "Sr", "Ca"].includes(e));
      const hasTM = elements.some(e => ["Cu", "Co", "Ni", "Fe"].includes(e));
      const hasO = elements.includes("O");
      return (hasSpacer || hasAlkalineEarth) && hasTM && hasO;
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
];

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula
    .replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
    .replace(/\s+/g, "")
    .replace(/-/g, "");
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    if (num > 0) counts[el] = (counts[el] || 0) + num;
  }
  return counts;
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
  const elemOrder = elements.map(e => ({ el: e, ratio: Math.round(counts[e]) }))
    .sort((a, b) => {
      const aR = getReducedRatio([a.ratio])[0];
      const bR = getReducedRatio([b.ratio])[0];
      return bR - aR || a.el.localeCompare(b.el);
    });

  const siteByRatio: Record<number, string[]> = {};
  for (const s of siteOrder) {
    if (!siteByRatio[s.ratio]) siteByRatio[s.ratio] = [];
    siteByRatio[s.ratio].push(s.label);
  }
  const elemByRatio: Record<number, string[]> = {};
  const elemReduced = elements.map(e => ({
    el: e,
    reduced: getReducedRatio([Math.round(counts[e])])[0]
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

  for (const cat of cations) {
    const oxStates = OXIDATION_STATES[cat] || [2];
    const expectedValence = oxStates[0];
    const rCat = IONIC_RADII[cat] || 0.8;

    let sumBV = 0;
    for (const an of anions) {
      const rAn = IONIC_RADII[an] || 1.4;
      const d0 = rCat + rAn;
      const bondOrder = Math.exp((d0 - (rCat + rAn) * 0.95) / 0.37);
      const nAnions = counts[an] || 1;
      const nCations = counts[cat] || 1;
      const coordContrib = (nAnions / nCations) * bondOrder;
      sumBV += coordContrib;
    }

    bvs[cat] = sumBV;
    const dev = Math.abs(sumBV - Math.abs(expectedValence)) / Math.max(1, Math.abs(expectedValence));
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
  if (anionCationRatio < 0.5) {
    details.push(`Very low anion:cation ratio (${anionCationRatio.toFixed(1)})`);
  }
  if (rRatio < 0.15) {
    details.push(`Cation too small for any coordination (r_ratio=${rRatio.toFixed(2)})`);
  }

  return { mismatch, plausible: mismatch < 0.6 && details.length === 0, details };
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
