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
  "Corundum-A2O3": 0.65,
  "ThMn12-RFe12": 0.68,
  "Ilmenite-ABO3": 0.62,
  "Marcasite-AB2": 0.60,
  "Tl2201-A2B2CO6": 0.55,
  "PostPerovskite-ABO3": 0.65,
  "Aurivillius-Bi2BO6": 0.55,
  "Stannite-A2BCS4": 0.52,
  "Scheelite-ABO4": 0.62,
  "Cementite-A3B": 0.68,
  "Zintl-AB2C2": 0.58,
  "Antifluorite-A2B": 0.68,
  "Cuprite-A2B": 0.52,
  "HexPerovskite-2H": 0.68,
  "Cuprate2223-A2B2C2D3O10": 0.55,
  "HexLayered-AB": 0.34,
  "Anatase-AB2": 0.58,
  "LayeredChalc-AMX2": 0.55,
  "D019-A3B": 0.74,
  "Sesquicarbide-A3C2": 0.65,
  "Dodecaboride-AB12": 0.62,
  "Hg1201-AB2CO4": 0.58,
  "C11b-Disilicide-AB2": 0.68,
  "Tetraboride-AB4": 0.60,
  "B20-FeSi": 0.68,
  "Matlockite-ABX": 0.58,
  "CaCu5-AB5": 0.74,
  "DO3-A3B": 0.68,
  "LiNbO3-R3c": 0.65,
  "D88-A5B3": 0.74,
  "GeS-Pnma": 0.58,
  "A7-Rhombohedral": 0.34,
  "A-RE2O3": 0.62,
  "A5-betaSn": 0.53,
  "OxynitridePerovskite-ABOX": 0.74,
  "Stibnite-A2B3": 0.55,
  "HalidePerovskite-ABX3": 0.74,
  "Hg1212-AB2CD2O6": 0.55,
  "Hg1223-AB2C2D3O8": 0.55,
  "BaAl4-AB4": 0.68,
  "QuaternaryHeusler-ABCD": 0.68,
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
      // A15 compounds are two metals/semimetals (Nb3Sn, V3Si), never TM + light interstitial
      const hasLightInterstitial = elements.some(e => ["C", "N", "B", "H"].includes(e));
      return hasTM && !hasLightInterstitial;
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

  // Corundum: R-3c (167), A2O3 — hexagonal setting
  // For: Al2O3, Fe2O3, Cr2O3, V2O3, Ti2O3 — extremely common binary oxide
  // Rhombohedral primitive cell: 2 A + 3 O = 5 atoms (from hex cell 12A + 18O → rhombo 1/6)
  // Actually for R-3c hex cell: 12 M + 18 O = 30 atoms, rhombo primitive = 10 atoms (4M + 6O)
  // Using rhombohedral primitive: 4 metal + 6 oxygen = 10 atoms, ratio [2,3]
  {
    name: "Corundum-A2O3",
    spaceGroup: "R-3c",
    latticeType: "hexagonal",
    cOverA: 2.73,
    sites: [
      // M at 12c Wyckoff (4 in rhomb primitive): (0,0,z) with z≈0.352
      { label: "A", x: 0.0, y: 0.0, z: 0.352, role: "cation" },
      { label: "A", x: 0.0, y: 0.0, z: 0.648, role: "cation" },
      { label: "A", x: 0.0, y: 0.0, z: 0.852, role: "cation" },
      { label: "A", x: 0.0, y: 0.0, z: 0.148, role: "cation" },
      // O at 18e Wyckoff (6 in rhomb primitive): (x,0,1/4) with x≈0.306
      { label: "B", x: 0.306, y: 0.0, z: 0.25, role: "anion" },
      { label: "B", x: 0.0, y: 0.306, z: 0.25, role: "anion" },
      { label: "B", x: 0.694, y: 0.694, z: 0.25, role: "anion" },
      { label: "B", x: 0.694, y: 0.0, z: 0.75, role: "anion" },
      { label: "B", x: 0.0, y: 0.694, z: 0.75, role: "anion" },
      { label: "B", x: 0.306, y: 0.306, z: 0.75, role: "anion" },
    ],
    stoichiometryRatio: [2, 3],
    coordination: [6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasO = elements.includes("O");
      const hasMetal = elements.some(e => ["Al", "Fe", "Cr", "V", "Ti", "Mn", "Ga", "In", "Rh"].includes(e));
      return hasO && hasMetal;
    },
  },
  // ThMn12-type: I4/mmm (139), RFe12 — rare-earth permanent magnet structure
  // For: NdFe12, SmFe12, YFe12, CeFe12 and nitrides NdFe12N
  // Body-centered tetragonal. Primitive cell (half conventional):
  // 1 R + 12 Fe = 13 atoms (conventional: 2R + 24Fe = 26)
  // Wyckoff: R at 2a, Fe at 8f + 8i + 8j
  // Primitive: R(1) + Fe-8f(4) + Fe-8i(4) + Fe-8j(4) = 13
  {
    name: "ThMn12-RFe12",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 0.57,
    sites: [
      // R at 2a: (0, 0, 0)
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "rare-earth" },
      // Fe at 8f: (1/4, 1/4, 1/4) — 4 in primitive
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "Fe-8f" },
      { label: "B", x: 0.75, y: 0.25, z: 0.25, role: "Fe-8f" },
      { label: "B", x: 0.25, y: 0.75, z: 0.25, role: "Fe-8f" },
      { label: "B", x: 0.75, y: 0.75, z: 0.25, role: "Fe-8f" },
      // Fe at 8i: (x, 0, 0) with x≈0.36 — 4 in primitive
      { label: "B", x: 0.36, y: 0.0, z: 0.0, role: "Fe-8i" },
      { label: "B", x: 0.64, y: 0.0, z: 0.0, role: "Fe-8i" },
      { label: "B", x: 0.0, y: 0.36, z: 0.0, role: "Fe-8i" },
      { label: "B", x: 0.0, y: 0.64, z: 0.0, role: "Fe-8i" },
      // Fe at 8j: (x, 1/2, 0) with x≈0.28 — 4 in primitive
      { label: "B", x: 0.28, y: 0.5, z: 0.0, role: "Fe-8j" },
      { label: "B", x: 0.72, y: 0.5, z: 0.0, role: "Fe-8j" },
      { label: "B", x: 0.5, y: 0.28, z: 0.0, role: "Fe-8j" },
      { label: "B", x: 0.5, y: 0.72, z: 0.0, role: "Fe-8j" },
    ],
    stoichiometryRatio: [1, 12],
    coordination: [20, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasRE = elements.some(e => isRareEarth(e) || ["Y", "Sc", "Th", "Zr"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Co", "Ni", "Mn", "V", "Cr"].includes(e));
      return hasRE && hasTM;
    },
  },

  // Ilmenite: R-3 (148), ABO3 — ordered corundum derivative
  // Distinct from perovskite: rhombohedral with alternating A/B cation layers
  // For: FeTiO3, MnTiO3, NiTiO3, CoTiO3, MgTiO3 — photocatalysts, geoscience
  // Rhombohedral primitive cell: 2A + 2B + 6O = 10 atoms, ratio [1,1,3]
  // A at 4c (0,0,zA≈0.356), B at 4c (0,0,zB≈0.146), O at 18f-derived
  {
    name: "Ilmenite-ABO3",
    spaceGroup: "R-3",
    latticeType: "hexagonal",
    cOverA: 2.77,
    sites: [
      // A cations (Fe/Mn/Co/Ni): 2 in primitive from 4c
      { label: "A", x: 0.0, y: 0.0, z: 0.356, role: "A-layer" },
      { label: "A", x: 0.0, y: 0.0, z: 0.644, role: "A-layer" },
      // B cations (Ti): 2 in primitive from 4c
      { label: "B", x: 0.0, y: 0.0, z: 0.146, role: "B-layer" },
      { label: "B", x: 0.0, y: 0.0, z: 0.854, role: "B-layer" },
      // O anions: 6 in primitive from 18f-derived
      { label: "C", x: 0.317, y: 0.020, z: 0.245, role: "O-1" },
      { label: "C", x: 0.980, y: 0.297, z: 0.245, role: "O-2" },
      { label: "C", x: 0.703, y: 0.683, z: 0.245, role: "O-3" },
      { label: "C", x: 0.683, y: 0.980, z: 0.755, role: "O-4" },
      { label: "C", x: 0.020, y: 0.703, z: 0.755, role: "O-5" },
      { label: "C", x: 0.297, y: 0.317, z: 0.755, role: "O-6" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [6, 6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      // A-site: divalent TM (Fe2+, Mn2+, Co2+, Ni2+, Mg2+)
      const hasA = elements.some(e => ["Fe", "Mn", "Co", "Ni", "Mg", "Zn", "Cd"].includes(e));
      // B-site: tetravalent (Ti4+, Sn4+, Ge4+, Hf4+)
      const hasB = elements.some(e => ["Ti", "Sn", "Ge", "Hf", "Zr", "Nb"].includes(e));
      return hasO && hasA && hasB;
    },
  },
  // Marcasite: Pnnm (58), AB2 — orthorhombic FeS2 polymorph
  // Distinct from pyrite (Pa-3): edge-sharing octahedra instead of corner-sharing
  // For: FeS2-marcasite, FeSb2, CrSb2, OsP2, NiAs2 — thermoelectric/catalytic
  // Primitive cell: 2A + 4B = 6 atoms, ratio [1,2]
  // A at 2a (0,0,0), B at 4g (x,y,0) with x≈0.20, y≈0.38
  {
    name: "Marcasite-AB2",
    spaceGroup: "Pnnm",
    latticeType: "tetragonal",
    cOverA: 0.55,
    sites: [
      // M at 2a: (0,0,0) — 2 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "metal" },
      // X at 4g: (x,y,0) with x≈0.200, y≈0.378 — 4 in primitive
      { label: "B", x: 0.200, y: 0.378, z: 0.0, role: "anion-1" },
      { label: "B", x: 0.800, y: 0.622, z: 0.0, role: "anion-2" },
      { label: "B", x: 0.300, y: 0.878, z: 0.5, role: "anion-3" },
      { label: "B", x: 0.700, y: 0.122, z: 0.5, role: "anion-4" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // Marcasite-type specifically: TM + diantimonides/diarsenides/diphosphides
      // These adopt marcasite (Pnnm) rather than pyrite (Pa-3).
      // FeS2 → pyrite (handled by Pyrite template); FeSb2 → marcasite
      const hasTM = elements.some(e => ["Fe", "Co", "Ni", "Cr", "Os", "Ru", "Mn", "Cu", "Ir", "Rh"].includes(e));
      const hasHeavyPn = elements.some(e => ["Sb", "Bi"].includes(e));
      return hasTM && hasHeavyPn;
    },
  },

  // Tl-2201 single-layer cuprate: I4/mmm (139), A2B2CuO6
  // Single CuO2 layer variant — Tl2Ba2CuO6+δ (Tc ≈ 90 K)
  // Distinct from Bi-2212 (5 elements, 2 CuO2 layers) and YBCO-123 (4 elem, chains)
  // For: Tl2Ba2CuO6, Bi2Sr2CuO6 (Bi-2201), Hg2Ba2CuO6
  // Primitive cell (BCC → half conventional): 2A + 2B + 1Cu + 6O = 11 atoms
  {
    name: "Tl2201-A2B2CO6",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 5.95,
    sites: [
      // A-sites: Tl/Bi/Hg (2 in primitive, from 4e)
      { label: "A", x: 0.0, y: 0.0, z: 0.210, role: "AO-layer" },
      { label: "A", x: 0.0, y: 0.0, z: 0.790, role: "AO-layer" },
      // B-sites: Ba/Sr (2 in primitive, from 4e)
      { label: "B", x: 0.0, y: 0.0, z: 0.116, role: "BO-layer" },
      { label: "B", x: 0.0, y: 0.0, z: 0.884, role: "BO-layer" },
      // Cu at 2a: CuO2 plane
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "CuO2-plane" },
      // O-sites: 6 in primitive
      { label: "D", x: 0.5, y: 0.0, z: 0.0, role: "O-planar" },
      { label: "D", x: 0.0, y: 0.5, z: 0.0, role: "O-planar" },
      { label: "D", x: 0.0, y: 0.0, z: 0.163, role: "O-apical" },
      { label: "D", x: 0.0, y: 0.0, z: 0.837, role: "O-apical" },
      { label: "D", x: 0.0, y: 0.0, z: 0.279, role: "O-AO-layer" },
      { label: "D", x: 0.0, y: 0.0, z: 0.721, role: "O-AO-layer" },
    ],
    stoichiometryRatio: [2, 2, 1, 6],
    coordination: [6, 9, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasCu = elements.includes("Cu");
      const hasHeavy = elements.some(e => ["Tl", "Bi", "Hg", "Pb"].includes(e));
      const hasAE = elements.some(e => ["Ba", "Sr"].includes(e));
      return hasO && hasCu && hasHeavy && hasAE;
    },
  },
  // Post-perovskite: Cmcm (63), ABO3 — high-pressure MgSiO3 phase
  // CaIrO3-type structure. Important for deep Earth mineralogy and
  // high-pressure superconductor candidates.
  // For: MgSiO3 (>120 GPa), CaIrO3, CaRuO3, CaPtO3, NaMgF3
  // Primitive cell: 4 atoms per formula unit → 4A + 4B + 12O = 20 atoms
  // Actually Cmcm with Z=4: simplified to 1 f.u. in reduced description
  // Wyckoff: A at 4c (0,y,1/4), B at 4a (0,0,0), O1 at 4c, O2 at 8f
  // Primitive = half conventional for C-centering: 2A + 2B + 6O = 10 atoms
  {
    name: "PostPerovskite-ABO3",
    spaceGroup: "Cmcm",
    latticeType: "tetragonal",
    cOverA: 2.42,
    sites: [
      // A at 4c: (0, y, 1/4) with y ≈ 0.25 — 2 in primitive
      { label: "A", x: 0.0, y: 0.25, z: 0.25, role: "A-site" },
      { label: "A", x: 0.0, y: 0.75, z: 0.75, role: "A-site" },
      // B at 4a: (0, 0, 0) — 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-site" },
      { label: "B", x: 0.0, y: 0.5, z: 0.5, role: "B-site" },
      // O1 at 4c: (0, y, 1/4) with y ≈ 0.93 — 2 in primitive
      { label: "C", x: 0.0, y: 0.93, z: 0.25, role: "O-apical" },
      { label: "C", x: 0.0, y: 0.07, z: 0.75, role: "O-apical" },
      // O2 at 8f: (0, y, z) with y ≈ 0.63, z ≈ 0.07 — 4 in primitive
      { label: "C", x: 0.0, y: 0.63, z: 0.07, role: "O-equatorial" },
      { label: "C", x: 0.0, y: 0.37, z: 0.93, role: "O-equatorial" },
      { label: "C", x: 0.0, y: 0.13, z: 0.57, role: "O-equatorial" },
      { label: "C", x: 0.0, y: 0.87, z: 0.43, role: "O-equatorial" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [8, 6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O") || elements.includes("F");
      // A-site: large cation (Mg, Ca, Na for fluoride analog)
      const hasA = elements.some(e => ["Ca", "Mg", "Na", "Sr", "Ba"].includes(e));
      // B-site: small high-valence cation (Ir, Ru, Pt, Si, Ge, Sn)
      const hasB = elements.some(e => ["Ir", "Ru", "Pt", "Rh", "Os", "Si", "Ge", "Sn"].includes(e));
      return hasO && hasA && hasB;
    },
  },

  // Aurivillius n=1: I4/mmm-like, Bi2BO6 — layered bismuth oxide
  // Bi2O2 fluorite-like layers alternating with perovskite-like BO4 blocks.
  // For: Bi2WO6, Bi2MoO6, Bi2CrO6 — photocatalysis, ferroelectrics
  // Primitive cell: 2 Bi + 1 B + 6 O = 9 atoms, ratio [2,1,6]
  // Wyckoff in pseudo-I4/mmm: Bi at 4e, B at 2a, O at multiple sites
  {
    name: "Aurivillius-Bi2BO6",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 2.7,
    sites: [
      // Bi at 4e: Bi2O2 layer
      { label: "A", x: 0.0, y: 0.0, z: 0.330, role: "Bi2O2-layer" },
      { label: "A", x: 0.0, y: 0.0, z: 0.670, role: "Bi2O2-layer" },
      // B at 2a: perovskite B-site
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "perovskite-B" },
      // O-sites: 6 in primitive
      { label: "C", x: 0.5, y: 0.0, z: 0.0, role: "O-equatorial" },
      { label: "C", x: 0.0, y: 0.5, z: 0.0, role: "O-equatorial" },
      { label: "C", x: 0.0, y: 0.0, z: 0.145, role: "O-apical" },
      { label: "C", x: 0.0, y: 0.0, z: 0.855, role: "O-apical" },
      { label: "C", x: 0.0, y: 0.0, z: 0.415, role: "O-BiO-layer" },
      { label: "C", x: 0.0, y: 0.0, z: 0.585, role: "O-BiO-layer" },
    ],
    stoichiometryRatio: [2, 1, 6],
    coordination: [8, 6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasBi = elements.includes("Bi");
      const hasO = elements.includes("O");
      // B-site: W, Mo, Cr, Ti, V, Nb, Ta (high-valence TM for charge balance with Bi3+)
      const hasB = elements.some(e => ["W", "Mo", "Cr", "Ti", "V", "Nb", "Ta"].includes(e));
      return hasBi && hasO && hasB;
    },
  },
  // Stannite: I-42m (121), A2BCS4 — quaternary chalcogenide photovoltaic
  // For: Cu2ZnSnS4 (CZTS), Cu2ZnSnSe4, Cu2ZnGeSe4, Cu2CdSnS4
  // Primitive cell (body-centered tetragonal → half conventional):
  // 2 A + 1 B + 1 C + 4 S = 8 atoms, ratio [2,1,1,4]
  // Wyckoff: A at 4d, B at 2a, C at 2b, S at 8i
  {
    name: "Stannite-A2BCS4",
    spaceGroup: "I-42m",
    latticeType: "tetragonal",
    cOverA: 1.97,
    sites: [
      // A (Cu) at 4d: (0, 1/2, 1/4) → 2 in primitive
      { label: "A", x: 0.0, y: 0.5, z: 0.25, role: "Cu-site" },
      { label: "A", x: 0.5, y: 0.0, z: 0.25, role: "Cu-site" },
      // B (Zn) at 2a: (0, 0, 0) → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "Zn-site" },
      // C (Sn) at 2b: (0, 0, 1/2) → 1 in primitive
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "Sn-site" },
      // S at 8i: (x, x, z) with x≈0.245, z≈0.128 → 4 in primitive
      { label: "D", x: 0.245, y: 0.245, z: 0.128, role: "S-1" },
      { label: "D", x: 0.755, y: 0.755, z: 0.128, role: "S-2" },
      { label: "D", x: 0.755, y: 0.245, z: 0.872, role: "S-3" },
      { label: "D", x: 0.245, y: 0.755, z: 0.872, role: "S-4" },
    ],
    stoichiometryRatio: [2, 1, 1, 4],
    coordination: [4, 4, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasCu = elements.some(e => ["Cu", "Ag"].includes(e));
      const hasZn = elements.some(e => ["Zn", "Cd", "Fe", "Mn", "Co", "Ni"].includes(e));
      const hasSn = elements.some(e => ["Sn", "Ge", "Si"].includes(e));
      const hasCh = elements.some(e => ["S", "Se", "Te"].includes(e));
      return hasCu && hasZn && hasSn && hasCh;
    },
  },

  // Scheelite: I41/a (88), ABO4 — scintillator/phosphor host
  // For: CaWO4, SrWO4, BaWO4, PbWO4, CaMoO4, SrMoO4, YVO4
  // Body-centered tetragonal. Primitive cell (half conventional):
  // 2A + 2B + 8O = 12 atoms, ratio [1,1,4]
  // Wyckoff: A at 4b, B at 4a, O at 16f
  {
    name: "Scheelite-ABO4",
    spaceGroup: "I41/a",
    latticeType: "tetragonal",
    cOverA: 2.17,
    sites: [
      // A (Ca/Sr/Ba/Pb/RE) at 4b → 2 in primitive
      { label: "A", x: 0.0, y: 0.25, z: 0.625, role: "A-site" },
      { label: "A", x: 0.0, y: 0.75, z: 0.125, role: "A-site" },
      // B (W/Mo/V) at 4a → 2 in primitive
      { label: "B", x: 0.0, y: 0.25, z: 0.125, role: "B-site" },
      { label: "B", x: 0.0, y: 0.75, z: 0.625, role: "B-site" },
      // O at 16f → 8 in primitive
      { label: "C", x: 0.241, y: 0.150, z: 0.081, role: "O-1" },
      { label: "C", x: 0.759, y: 0.350, z: 0.081, role: "O-2" },
      { label: "C", x: 0.150, y: 0.741, z: 0.169, role: "O-3" },
      { label: "C", x: 0.350, y: 0.259, z: 0.169, role: "O-4" },
      { label: "C", x: 0.759, y: 0.850, z: 0.919, role: "O-5" },
      { label: "C", x: 0.241, y: 0.650, z: 0.919, role: "O-6" },
      { label: "C", x: 0.850, y: 0.259, z: 0.831, role: "O-7" },
      { label: "C", x: 0.650, y: 0.741, z: 0.831, role: "O-8" },
    ],
    stoichiometryRatio: [1, 1, 4],
    coordination: [8, 4, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      const hasA = elements.some(e => ["Ca", "Sr", "Ba", "Pb", "Bi", "Y", "La", "Ce", "Nd", "Gd", "Eu"].includes(e));
      const hasB = elements.some(e => ["W", "Mo", "V", "Cr", "Re"].includes(e));
      return hasO && hasA && hasB;
    },
  },
  // Cementite: Pnma (62), A3B — Fe3C-type intermetallic carbide
  // For: Fe3C, Mn3C, Cr3C, Co3C, Ni3C — steels, hard coatings, catalysts
  // Orthorhombic primitive cell = conventional. Using 1 formula unit:
  // 3 A + 1 B = 4 atoms. Reduced from full cell (Z=4, 16 atoms).
  // Wyckoff: Fe1 at 4c (x,1/4,z), Fe2 at 8d (x,y,z), C at 4c (x,1/4,z)
  // Simplified template with representative 4-atom f.u.
  {
    name: "Cementite-A3B",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 0.59,
    sites: [
      // Fe1 at 4c-like
      { label: "A", x: 0.036, y: 0.25, z: 0.838, role: "metal-4c" },
      // Fe2 at 8d-like (2 per f.u.)
      { label: "A", x: 0.186, y: 0.065, z: 0.334, role: "metal-8d-1" },
      { label: "A", x: 0.186, y: 0.435, z: 0.334, role: "metal-8d-2" },
      // C at 4c-like
      { label: "B", x: 0.890, y: 0.25, z: 0.443, role: "carbon" },
    ],
    stoichiometryRatio: [3, 1],
    coordination: [6, 9],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Fe", "Mn", "Cr", "Co", "Ni", "W", "Mo", "V"].includes(e));
      const hasLight = elements.some(e => ["C", "N", "B"].includes(e));
      return hasTM && hasLight;
    },
  },

  // Zintl CaAl2Si2-type: P-3m1 (164), AB2C2 — thermoelectric/topological
  // For: CaAl2Si2, Mg3Sb2 (as MgMg2Sb2), CaMg2Bi2, YbMg2Bi2, EuMg2Bi2
  // Trigonal layered structure. A at 1a, B at 2d, C at 2d.
  // Primitive cell: 1A + 2B + 2C = 5 atoms, ratio [1,2,2]
  {
    name: "Zintl-AB2C2",
    spaceGroup: "P-3m1",
    latticeType: "hexagonal",
    cOverA: 1.6,
    sites: [
      // A at 1a: (0, 0, 0) — large cation (Ca, Yb, Eu, Mg)
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation" },
      // B at 2d: (1/3, 2/3, z) with z ≈ 0.63 — tetrahedral B (Mg, Al, Zn)
      { label: "B", x: 0.3333, y: 0.6667, z: 0.63, role: "tetra-B" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.37, role: "tetra-B" },
      // C at 2d: (1/3, 2/3, z) with z ≈ 0.24 — anion (Si, Sb, Bi, As, Sn)
      { label: "C", x: 0.3333, y: 0.6667, z: 0.24, role: "anion" },
      { label: "C", x: 0.6667, y: 0.3333, z: 0.76, role: "anion" },
    ],
    stoichiometryRatio: [1, 2, 2],
    coordination: [6, 4, 5],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      // A-site: alkaline earth, Yb, Eu, or similar large cation
      const hasA = elements.some(e => ["Ca", "Sr", "Ba", "Mg", "Yb", "Eu", "Sm", "La"].includes(e));
      // B-site: small electropositive (Mg, Al, Zn, Cd, Mn)
      const hasB = elements.some(e => ["Mg", "Al", "Zn", "Cd", "Mn", "Cu"].includes(e));
      // C-site: pnictogen or group-14 (Si, Ge, Sn, Sb, Bi, As, P)
      const hasC = elements.some(e => ["Si", "Ge", "Sn", "Sb", "Bi", "As", "P", "Te", "Se"].includes(e));
      // Need at least 2 of 3 distinct (A and B can overlap, e.g., Mg3Sb2 = MgMg2Sb2)
      return hasA && hasB && hasC;
    },
  },
  // Antifluorite: Fm-3m (225), A2B — inverse of fluorite
  // For: Li2O, Na2O, K2O, Li2S, Li2Se, Na2S, K2S — solid electrolytes, nuclear
  // Primitive cell (FCC → 1/4 conventional): 2A + 1B = 3 atoms, ratio [2,1]
  // Wyckoff: A at 8c (1/4,1/4,1/4), B at 4a (0,0,0)
  {
    name: "Antifluorite-A2B",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A (Li/Na/K) at 8c → 2 in primitive
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "cation" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "cation" },
      // B (O/S/Se/Te) at 4a → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "anion" },
    ],
    stoichiometryRatio: [2, 1],
    coordination: [4, 8],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // Antifluorite: alkali + chalcogenide/oxide (A2X). Cu2O is cuprite, not antifluorite.
      const hasAlkali = elements.some(e => ["Li", "Na", "K", "Rb", "Cs"].includes(e));
      const hasAnion = elements.some(e => ["O", "S", "Se", "Te", "F"].includes(e));
      return hasAlkali && hasAnion;
    },
  },

  // Cuprite: Pn-3m (224), A2B — Cu2O-type
  // Distinct from antifluorite: linear O-Cu-O coordination, not tetrahedral.
  // For: Cu2O, Ag2O — photovoltaics, catalysis, p-type TCO
  // Primitive cell: 2 Cu + 1 O = 3 atoms (from conventional 4Cu + 2O = 6, /2)
  // Cu at 4b: (1/4, 1/4, 1/4), O at 2a: (0, 0, 0)
  {
    name: "Cuprite-A2B",
    spaceGroup: "Pn-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // Cu at 4b → 2 in primitive
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "linear-coord" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "linear-coord" },
      // O at 2a → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "tetrahedral" },
    ],
    stoichiometryRatio: [2, 1],
    coordination: [2, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasCuAg = elements.some(e => ["Cu", "Ag"].includes(e));
      const hasO = elements.includes("O");
      return hasCuAg && hasO;
    },
  },
  // Hexagonal perovskite 2H: P63/mmc (194), ABO3 — face-sharing octahedra
  // Distinct from cubic perovskite (corner-sharing). For: BaNiO3, BaCoO3,
  // BaMnO3-2H, CsNiF3, BaRuO3 — quantum magnets, catalysts
  // Primitive cell: 2A + 2B + 6O = 10 atoms, ratio [1,1,3]
  // Wyckoff: A at 2d, B at 2a, O at 6h
  {
    name: "HexPerovskite-2H",
    spaceGroup: "P63/mmc",
    latticeType: "hexagonal",
    cOverA: 2.45,
    sites: [
      // A (Ba/Cs) at 2d: (1/3, 2/3, 3/4)
      { label: "A", x: 0.3333, y: 0.6667, z: 0.75, role: "A-site" },
      { label: "A", x: 0.6667, y: 0.3333, z: 0.25, role: "A-site" },
      // B (Ni/Co/Mn/Ru) at 2a: (0, 0, 0)
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-face-share" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "B-face-share" },
      // O at 6h: (x, 2x, 1/4) with x ≈ 0.515
      { label: "C", x: 0.515, y: 0.030, z: 0.25, role: "O-1" },
      { label: "C", x: 0.970, y: 0.485, z: 0.25, role: "O-2" },
      { label: "C", x: 0.485, y: 0.515, z: 0.25, role: "O-3" },
      { label: "C", x: 0.485, y: 0.970, z: 0.75, role: "O-4" },
      { label: "C", x: 0.030, y: 0.515, z: 0.75, role: "O-5" },
      { label: "C", x: 0.515, y: 0.485, z: 0.75, role: "O-6" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [12, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasAnion = elements.includes("O") || elements.includes("F");
      // A-site: Ba, Cs specifically (large cations that stabilize hexagonal stacking)
      const hasLargeA = elements.some(e => ["Ba", "Cs"].includes(e));
      // B-site: TM that prefers face-sharing (Ni, Co, Mn, Ru, Ti, V, Cr)
      const hasTM = elements.some(e => ["Ni", "Co", "Mn", "Ru", "Ti", "V", "Cr", "Fe", "Ir"].includes(e));
      return hasAnion && hasLargeA && hasTM;
    },
  },

  // 2223 triple-layer cuprate: I4/mmm, A2B2C2D3O10
  // Triple CuO2 layer — highest Tc in Tl/Bi cuprate families (~125 K)
  // For: Tl2Ba2Ca2Cu3O10, Bi2Sr2Ca2Cu3O10 (Bi-2223)
  // Primitive cell (BCC → half conventional): 2A + 2B + 2C + 3D + 10O = 19 atoms
  {
    name: "Cuprate2223-A2B2C2D3O10",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 9.5,
    sites: [
      // A (Tl/Bi) at 4e: double AO layer
      { label: "A", x: 0.0, y: 0.0, z: 0.220, role: "AO-layer" },
      { label: "A", x: 0.0, y: 0.0, z: 0.780, role: "AO-layer" },
      // B (Ba/Sr) at 4e: BO layer
      { label: "B", x: 0.0, y: 0.0, z: 0.140, role: "BO-layer" },
      { label: "B", x: 0.0, y: 0.0, z: 0.860, role: "BO-layer" },
      // C (Ca) at 4e: spacer between CuO2 planes
      { label: "C", x: 0.0, y: 0.0, z: 0.045, role: "Ca-spacer" },
      { label: "C", x: 0.0, y: 0.0, z: 0.955, role: "Ca-spacer" },
      // D (Cu) at 2a + 4e: 3 CuO2 planes (1 inner + 2 outer)
      { label: "D", x: 0.0, y: 0.0, z: 0.0, role: "Cu-inner-plane" },
      { label: "D", x: 0.0, y: 0.0, z: 0.090, role: "Cu-outer-plane" },
      { label: "D", x: 0.0, y: 0.0, z: 0.910, role: "Cu-outer-plane" },
      // O: 10 sites in primitive (equatorial + apical + AO-layer)
      { label: "E", x: 0.5, y: 0.0, z: 0.0, role: "O-inner-eq" },
      { label: "E", x: 0.0, y: 0.5, z: 0.0, role: "O-inner-eq" },
      { label: "E", x: 0.5, y: 0.0, z: 0.090, role: "O-outer-eq" },
      { label: "E", x: 0.0, y: 0.5, z: 0.090, role: "O-outer-eq" },
      { label: "E", x: 0.5, y: 0.0, z: 0.910, role: "O-outer-eq" },
      { label: "E", x: 0.0, y: 0.5, z: 0.910, role: "O-outer-eq" },
      { label: "E", x: 0.0, y: 0.0, z: 0.170, role: "O-apical" },
      { label: "E", x: 0.0, y: 0.0, z: 0.830, role: "O-apical" },
      { label: "E", x: 0.0, y: 0.0, z: 0.280, role: "O-AO-layer" },
      { label: "E", x: 0.0, y: 0.0, z: 0.720, role: "O-AO-layer" },
    ],
    stoichiometryRatio: [2, 2, 2, 3, 10],
    coordination: [6, 9, 8, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 5) return false;
      const hasO = elements.includes("O");
      const hasCu = elements.includes("Cu");
      const hasHeavy = elements.some(e => ["Bi", "Tl", "Hg", "Pb"].includes(e));
      const hasAE = elements.some(e => ["Ba", "Sr"].includes(e));
      const hasSpacer = elements.includes("Ca") || elements.some(e => isRareEarth(e) || e === "Y");
      return hasO && hasCu && hasHeavy && hasAE && hasSpacer;
    },
  },
  // Hexagonal layered AB: P63/mmc (194), hBN-type
  // For: hBN, AlN (wurtzite is different!), SiC-2H, graphite-like layered
  // AB stacking with sp2 layers. Primitive cell: 2A + 2B = 4 atoms, ratio [1,1]
  // A at 2b: (0, 0, 1/4), B at 2c: (1/3, 2/3, 1/4)
  {
    name: "HexLayered-AB",
    spaceGroup: "P63/mmc",
    latticeType: "hexagonal",
    cOverA: 2.66,
    sites: [
      // A (B in hBN) at 2b
      { label: "A", x: 0.0, y: 0.0, z: 0.25, role: "layer-A-1" },
      { label: "A", x: 0.0, y: 0.0, z: 0.75, role: "layer-A-2" },
      // B (N in hBN) at 2c
      { label: "B", x: 0.3333, y: 0.6667, z: 0.25, role: "layer-B-1" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.75, role: "layer-B-2" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [3, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // hBN-type: light p-block + p-block or N combinations
      const hasB = elements.includes("B");
      const hasN = elements.includes("N");
      // Also covers: AlN-2H, GaN-2H (wurtzite handled separately), SiC-2H
      return hasB && hasN;
    },
  },

  // Anatase: I41/amd (141), AB2 — TiO2 photocatalyst polymorph
  // Distinct from rutile (P42/mnm): edge-sharing vs corner-sharing octahedra
  // For: TiO2-anatase, SnO2-anatase (metastable), VO2 (at high T)
  // Primitive cell (BCC → half conventional): 2A + 4B = 6 atoms, ratio [1,2]
  // Ti at 4a (0,0,0), O at 8e (0,0,z) with z ≈ 0.208
  {
    name: "Anatase-AB2",
    spaceGroup: "I41/amd",
    latticeType: "tetragonal",
    cOverA: 2.51,
    sites: [
      // Ti at 4a → 2 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation" },
      { label: "A", x: 0.0, y: 0.5, z: 0.25, role: "cation" },
      // O at 8e → 4 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.208, role: "anion" },
      { label: "B", x: 0.0, y: 0.0, z: 0.792, role: "anion" },
      { label: "B", x: 0.0, y: 0.5, z: 0.458, role: "anion" },
      { label: "B", x: 0.0, y: 0.5, z: 0.042, role: "anion" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // Anatase-specific: Ti/V/Sn + O (photocatalyst oxides)
      // Narrow to avoid shadowing Rutile, Fluorite, etc.
      const hasTi = elements.some(e => ["Ti", "V", "Sn"].includes(e));
      const hasO = elements.includes("O");
      return hasTi && hasO;
    },
  },
  // Layered chalcogenide: R-3m (166), AMX2 — NaCrS2-type
  // For: NaCrS2, LiTiS2, NaCoO2-analogs with S/Se/Te, KCrSe2, AgCrS2
  // Distinct from delafossite (same SG but oxide-specific with noble metals)
  // Primitive cell (rhombohedral): 1A + 1M + 2X = 4 atoms, ratio [1,1,2]
  {
    name: "LayeredChalc-AMX2",
    spaceGroup: "R-3m",
    latticeType: "hexagonal",
    cOverA: 5.8,
    sites: [
      // A (alkali/Ag) at 3a: (0, 0, 0)
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "intercalant" },
      // M (TM) at 3b: (0, 0, 1/2)
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "TM-oct" },
      // X (S/Se/Te) at 6c: (0, 0, z) with z ≈ 0.26
      { label: "C", x: 0.0, y: 0.0, z: 0.26, role: "chalcogen" },
      { label: "C", x: 0.0, y: 0.0, z: 0.74, role: "chalcogen" },
    ],
    stoichiometryRatio: [1, 1, 2],
    coordination: [6, 6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      // A-site: alkali or Ag/Cu
      const hasA = elements.some(e => ["Li", "Na", "K", "Rb", "Cs", "Ag", "Cu", "Tl"].includes(e));
      // M-site: transition metal
      const hasTM = elements.some(e => ["Cr", "Ti", "V", "Mn", "Fe", "Co", "Ni", "Nb", "Mo", "W", "Ta"].includes(e));
      // X-site: chalcogenide (NOT oxide — that's delafossite/layered oxide)
      const hasCh = elements.some(e => ["S", "Se", "Te"].includes(e));
      return hasA && hasTM && hasCh;
    },
  },

  // D019 ordered HCP: P63/mmc (194), A3B — Ni3Sn-type intermetallic
  // For: Ni3Al, Ti3Al, Ni3Sn, Co3Ti, Fe3Al, Pt3Sn — structural alloys, catalysts
  // Hexagonal DO19 superstructure of HCP.
  // Primitive cell: 6A + 2B = 8 atoms, ratio [3,1]
  // A at 6h: (x, 2x, 1/4) with x ≈ 0.833; B at 2c: (1/3, 2/3, 1/4)
  {
    name: "D019-A3B",
    spaceGroup: "P63/mmc",
    latticeType: "hexagonal",
    cOverA: 0.8,
    sites: [
      // A at 6h → 6 atoms (x ≈ 5/6)
      { label: "A", x: 0.833, y: 0.666, z: 0.25, role: "majority-1" },
      { label: "A", x: 0.334, y: 0.167, z: 0.25, role: "majority-2" },
      { label: "A", x: 0.167, y: 0.833, z: 0.25, role: "majority-3" },
      { label: "A", x: 0.167, y: 0.334, z: 0.75, role: "majority-4" },
      { label: "A", x: 0.666, y: 0.833, z: 0.75, role: "majority-5" },
      { label: "A", x: 0.833, y: 0.167, z: 0.75, role: "majority-6" },
      // B at 2c
      { label: "B", x: 0.3333, y: 0.6667, z: 0.25, role: "minority-1" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.75, role: "minority-2" },
    ],
    stoichiometryRatio: [3, 1],
    coordination: [12, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // D019: two metals, majority is late-TM or Al, minority is p-block or early-TM
      const hasTM = elements.some(e => ["Ni", "Co", "Fe", "Pt", "Pd", "Ir", "Rh"].includes(e));
      const hasPartner = elements.some(e => ["Al", "Ti", "Sn", "V", "Ga", "In", "Si", "Ge", "Mn"].includes(e));
      return hasTM && hasPartner;
    },
  },
  // Sesquicarbide: Pnma (62), A3C2 — Cr3C2-type hard coating
  // For: Cr3C2, Mn3C2, Fe3C2, V3C2 — wear-resistant coatings, cutting tools
  // Distinct from cementite A3C (ratio [3,1]) — this is [3,2]
  // Primitive cell = conventional: 1 formula unit = 3A + 2B = 5 atoms
  // Simplified representative positions from Cr3C2 literature
  {
    name: "Sesquicarbide-A3C2",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 0.36,
    sites: [
      // Cr1 at 4c: (x, 1/4, z)
      { label: "A", x: 0.100, y: 0.25, z: 0.770, role: "metal-4c-1" },
      // Cr2 at 4c: (x, 1/4, z)
      { label: "A", x: 0.190, y: 0.25, z: 0.440, role: "metal-4c-2" },
      // Cr3 at 4c: (x, 1/4, z)
      { label: "A", x: 0.392, y: 0.25, z: 0.100, role: "metal-4c-3" },
      // C1 at 4c: (x, 1/4, z)
      { label: "B", x: 0.050, y: 0.25, z: 0.100, role: "carbon-1" },
      // C2 at 4c: (x, 1/4, z)
      { label: "B", x: 0.280, y: 0.25, z: 0.885, role: "carbon-2" },
    ],
    stoichiometryRatio: [3, 2],
    coordination: [6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Cr", "Mn", "Fe", "V", "W", "Mo", "Ta", "Nb"].includes(e));
      const hasLight = elements.some(e => ["C", "N", "B"].includes(e));
      return hasTM && hasLight;
    },
  },

  // Dodecaboride: Fm-3m (225), AB12 — UB12/YB12-type boride cage
  // B12 cubo-octahedral clusters enclosing metal atoms.
  // For: YB12, ZrB12, ScB12, LuB12 — some superconducting (ZrB12 Tc≈6K)
  // Primitive cell (FCC → 1/4 conventional): 1 M + 12 B = 13 atoms, ratio [1,12]
  // M at 4a (0,0,0), B at 48i (1/2, x, x) with x ≈ 0.178
  {
    name: "Dodecaboride-AB12",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // M at 4a → 1 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cage-center" },
      // B at 48i → 12 in primitive (cubo-octahedral cage)
      { label: "B", x: 0.5, y: 0.178, z: 0.178, role: "B-cage" },
      { label: "B", x: 0.5, y: 0.822, z: 0.178, role: "B-cage" },
      { label: "B", x: 0.5, y: 0.178, z: 0.822, role: "B-cage" },
      { label: "B", x: 0.5, y: 0.822, z: 0.822, role: "B-cage" },
      { label: "B", x: 0.178, y: 0.5, z: 0.178, role: "B-cage" },
      { label: "B", x: 0.822, y: 0.5, z: 0.178, role: "B-cage" },
      { label: "B", x: 0.178, y: 0.5, z: 0.822, role: "B-cage" },
      { label: "B", x: 0.822, y: 0.5, z: 0.822, role: "B-cage" },
      { label: "B", x: 0.178, y: 0.178, z: 0.5, role: "B-cage" },
      { label: "B", x: 0.822, y: 0.178, z: 0.5, role: "B-cage" },
      { label: "B", x: 0.178, y: 0.822, z: 0.5, role: "B-cage" },
      { label: "B", x: 0.822, y: 0.822, z: 0.5, role: "B-cage" },
    ],
    stoichiometryRatio: [1, 12],
    coordination: [24, 5],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasB = elements.includes("B");
      const hasMetal = elements.some(e =>
        isRareEarth(e) || ["Y", "Sc", "Zr", "Hf", "Th", "U"].includes(e)
      );
      return hasB && hasMetal;
    },
  },
  // Hg-1201 cuprate: P4/mmm (123), AB2CO4 — single-layer Hg cuprate
  // HgBa2CuO4+δ — highest Tc for single-CuO2-layer cuprate (~97 K)
  // Distinct from Tl-2201 (I4/mmm, ratio [2,2,1,6]) — Hg-1201 has no
  // double Hg-layer and uses simple tetragonal P4/mmm.
  // Primitive cell: 1A + 2B + 1C + 4O = 8 atoms, ratio [1,2,1,4]
  {
    name: "Hg1201-AB2CO4",
    spaceGroup: "P4/mmm",
    latticeType: "tetragonal",
    cOverA: 2.48,
    sites: [
      // Hg at 1a: (0, 0, 0)
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Hg-layer" },
      // Ba at 2h: (1/2, 1/2, z) with z ≈ 0.29
      { label: "B", x: 0.5, y: 0.5, z: 0.29, role: "Ba-site" },
      { label: "B", x: 0.5, y: 0.5, z: 0.71, role: "Ba-site" },
      // Cu at 1b: (0, 0, 1/2)
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "CuO2-plane" },
      // O at 2g (0, 1/2, 1/2) — CuO2 in-plane
      { label: "D", x: 0.0, y: 0.5, z: 0.5, role: "O-planar" },
      { label: "D", x: 0.5, y: 0.0, z: 0.5, role: "O-planar" },
      // O at 2e (1/2, 1/2, 0) or 1c+1d — apical O
      { label: "D", x: 0.0, y: 0.0, z: 0.16, role: "O-apical" },
      { label: "D", x: 0.0, y: 0.0, z: 0.84, role: "O-apical" },
    ],
    stoichiometryRatio: [1, 2, 1, 4],
    coordination: [2, 10, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasCu = elements.includes("Cu");
      // A-site: Hg specifically (Tl-1201 uses different structure/ratio)
      const hasHg = elements.includes("Hg");
      const hasAE = elements.some(e => ["Ba", "Sr"].includes(e));
      return hasO && hasCu && hasHg && hasAE;
    },
  },

  // C11b Disilicide: I4/mmm (139), AB2 — MoSi2-type
  // Refractory disilicides/digermanides for high-temperature structural use.
  // For: MoSi2, WSi2, CrSi2, TiSi2, NbSi2, TaSi2, VSi2
  // Primitive cell (BCC → half conventional): 1A + 2B = 3 atoms, ratio [1,2]
  // A at 2a (0,0,0), B at 4e (0,0,z) with z ≈ 0.335
  {
    name: "C11b-Disilicide-AB2",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 2.45,
    sites: [
      // A (Mo/W/Cr/Ti) at 2a → 1 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "refractory-TM" },
      // B (Si/Ge) at 4e → 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.335, role: "Si-layer" },
      { label: "B", x: 0.0, y: 0.0, z: 0.665, role: "Si-layer" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [10, 5],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // Refractory TM + group-14 semimetal
      const hasTM = elements.some(e => ["Mo", "W", "Cr", "Ti", "V", "Nb", "Ta", "Zr", "Hf", "Re"].includes(e));
      const hasSi = elements.some(e => ["Si", "Ge"].includes(e));
      return hasTM && hasSi;
    },
  },
  // Tetraboride: P4/mbm (127), AB4 — YB4/ThB4-type boride
  // For: YB4, LaB4, CeB4, ThB4, SmB4, NdB4, GdB4 — heavy-fermion, SC
  // Primitive cell (P-type = conventional): per f.u. 1A + 4B = 5 atoms, ratio [1,4]
  // A at 2a, B at 4g (x, x+1/2, 0) with x ≈ 0.18 + 4h (x, x+1/2, 1/2) with x ≈ 0.04
  // Simplified to 1 formula unit representation
  {
    name: "Tetraboride-AB4",
    spaceGroup: "P4/mbm",
    latticeType: "tetragonal",
    cOverA: 0.57,
    sites: [
      // M at 2a-like
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal" },
      // B at 4g-like (x, x+1/2, 0)
      { label: "B", x: 0.18, y: 0.68, z: 0.0, role: "B-ring-1" },
      { label: "B", x: 0.32, y: 0.18, z: 0.0, role: "B-ring-2" },
      // B at 4h-like (x, x+1/2, 1/2)
      { label: "B", x: 0.04, y: 0.54, z: 0.5, role: "B-chain-1" },
      { label: "B", x: 0.46, y: 0.04, z: 0.5, role: "B-chain-2" },
    ],
    stoichiometryRatio: [1, 4],
    coordination: [18, 5],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasB = elements.includes("B");
      const hasMetal = elements.some(e =>
        isRareEarth(e) || ["Y", "Sc", "Th", "U", "Zr", "Hf", "Ca", "Sr"].includes(e)
      );
      return hasB && hasMetal;
    },
  },

  // B20 FeSi-type: P213 (198), AB — chiral cubic topological semimetal
  // For: FeSi, CoSi, MnSi, RhSi, CrSi, CoGe — skyrmion hosts, Weyl semimetals
  // P213 is the only chiral cubic SG in our library — unique for topology.
  // Primitive cell (simple cubic): 4A + 4B = 8 atoms, ratio [1,1]
  // A at 4a: (x, x, x) with x ≈ 0.137; B at 4a: (x, x, x) with x ≈ 0.845
  {
    name: "B20-FeSi",
    spaceGroup: "P213",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // Fe at 4a: (x, x, x) and symmetry-equivalents, x ≈ 0.137
      { label: "A", x: 0.137, y: 0.137, z: 0.137, role: "TM-1" },
      { label: "A", x: 0.637, y: 0.363, z: 0.863, role: "TM-2" },
      { label: "A", x: 0.363, y: 0.863, z: 0.637, role: "TM-3" },
      { label: "A", x: 0.863, y: 0.637, z: 0.363, role: "TM-4" },
      // Si at 4a: (x, x, x) and symmetry-equivalents, x ≈ 0.845
      { label: "B", x: 0.845, y: 0.845, z: 0.845, role: "Si-1" },
      { label: "B", x: 0.345, y: 0.655, z: 0.155, role: "Si-2" },
      { label: "B", x: 0.655, y: 0.155, z: 0.345, role: "Si-3" },
      { label: "B", x: 0.155, y: 0.345, z: 0.655, role: "Si-4" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [7, 7],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // B20: TM + group-14 semimetal (Si, Ge, Sn)
      const hasTM = elements.some(e => ["Fe", "Co", "Mn", "Cr", "Rh", "Ir", "Ru", "Os", "Ni", "Pt", "Pd"].includes(e));
      const hasSemimetal = elements.some(e => ["Si", "Ge", "Sn"].includes(e));
      return hasTM && hasSemimetal;
    },
  },
  // Matlockite PbFCl-type: P4/nmm (129), ABX — layered oxyhalide/chalcohalide
  // For: LaOF, CeOF, BiSCl, BiSeBr, NdOBr, PrOF — scintillators, optical
  // Layered structure: RE-O and halide layers alternate.
  // Primitive cell: 2A + 2B + 2X = 6 atoms, ratio [1,1,1]
  // A at 2c (1/4, 1/4, z≈0.18), B at 2a (3/4, 1/4, 0), X at 2c (1/4, 1/4, z≈0.62)
  {
    name: "Matlockite-ABX",
    spaceGroup: "P4/nmm",
    latticeType: "tetragonal",
    cOverA: 1.82,
    sites: [
      // A (La/Ce/Bi/Pb) at 2c
      { label: "A", x: 0.25, y: 0.25, z: 0.18, role: "cation-layer" },
      { label: "A", x: 0.75, y: 0.75, z: 0.82, role: "cation-layer" },
      // B (O/S/Se) at 2a — smaller anion in cation layer
      { label: "B", x: 0.75, y: 0.25, z: 0.0, role: "inner-anion" },
      { label: "B", x: 0.25, y: 0.75, z: 0.0, role: "inner-anion" },
      // X (F/Cl/Br/I) at 2c — halide in interlayer
      { label: "C", x: 0.25, y: 0.25, z: 0.62, role: "halide-layer" },
      { label: "C", x: 0.75, y: 0.75, z: 0.38, role: "halide-layer" },
    ],
    stoichiometryRatio: [1, 1, 1],
    coordination: [8, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      // A-site: rare earth, Bi, Pb, or alkaline earth
      const hasCation = elements.some(e => isRareEarth(e) || ["Bi", "Pb", "Y", "Sb"].includes(e));
      // B-site: O or S/Se/Te (chalcogen in the cation plane)
      const hasSmallAnion = elements.some(e => ["O", "S", "Se", "Te"].includes(e));
      // X-site: halide (F, Cl, Br, I)
      const hasHalide = elements.some(e => ["F", "Cl", "Br", "I"].includes(e));
      return hasCation && hasSmallAnion && hasHalide;
    },
  },

  // CaCu5-type: P6/mmm (191), AB5 — permanent magnet / H-storage
  // For: SmCo5, LaNi5, CeCo5, NdCo5, YCo5, PrCo5 — rare-earth magnets
  // Also: LaNi5 (hydrogen storage alloy), CaCu5 (prototype)
  // Primitive cell (P-type = conventional per f.u.): 1A + 5B = 6 atoms, ratio [1,5]
  // A at 1a (0,0,0), B at 2c (1/3,2/3,0) + 3g (1/2,0,1/2)
  {
    name: "CaCu5-AB5",
    spaceGroup: "P6/mmm",
    latticeType: "hexagonal",
    cOverA: 0.81,
    sites: [
      // A (Ca/Sm/La/Nd/Ce/Y) at 1a
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "RE-site" },
      // B at 2c: (1/3, 2/3, 0)
      { label: "B", x: 0.3333, y: 0.6667, z: 0.0, role: "TM-2c-1" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.0, role: "TM-2c-2" },
      // B at 3g: (1/2, 0, 1/2) and equivalents
      { label: "B", x: 0.5, y: 0.0, z: 0.5, role: "TM-3g-1" },
      { label: "B", x: 0.0, y: 0.5, z: 0.5, role: "TM-3g-2" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "TM-3g-3" },
    ],
    stoichiometryRatio: [1, 5],
    coordination: [18, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasRE = elements.some(e => isRareEarth(e) || ["Y", "Sc", "Ca", "Sr", "Ba", "Th", "Zr", "Hf", "Mg"].includes(e));
      const hasTM = elements.some(e => ["Co", "Ni", "Fe", "Cu", "Mn", "Pt", "Pd", "Ir", "Rh"].includes(e));
      return hasRE && hasTM;
    },
  },
  // DO3 / BiF3-type: Fm-3m (225), A3B — alkali/sp-metal topological
  // For: Na3Bi (Dirac semimetal), Li3Bi, K3Bi, Na3Sb, Li3Sb
  // Distinct from D019 (hexagonal, TM+partner) and Cementite (TM+light)
  // Primitive cell (FCC → 1/4 conventional): 3A + 1B = 4 atoms, ratio [3,1]
  // A at 8c (1/4,1/4,1/4) + 4b (1/2,1/2,1/2), B at 4a (0,0,0)
  {
    name: "DO3-A3B",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A (Li/Na/K) at 8c → 2 in primitive
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "alkali-8c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "alkali-8c" },
      // A at 4b → 1 in primitive
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "alkali-4b" },
      // B (Bi/Sb/As/Sn/Pb) at 4a → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "heavy-pnictogen" },
    ],
    stoichiometryRatio: [3, 1],
    coordination: [8, 12],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // DO3: alkali/alkaline-earth + heavy p-block (topological materials)
      const hasLight = elements.some(e => ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca"].includes(e));
      const hasHeavy = elements.some(e => ["Bi", "Sb", "Sn", "Pb", "As", "Te", "In"].includes(e));
      return hasLight && hasHeavy;
    },
  },

  // LiNbO3-type: R3c (161), ABO3 — ferroelectric/electro-optic
  // Rhombohedral distortion of perovskite with small A-site cation.
  // For: LiNbO3, LiTaO3, LiIO3, KNbO3-R3c — electro-optic, piezoelectric
  // CRITICAL: These fail the cubic Perovskite template (Li/K too small for
  // CATIONS_LARGE) and fail Ilmenite (needs divalent+tetravalent pair).
  // Primitive cell: 2 f.u. = 2A + 2B + 6O = 10 atoms, ratio [1,1,3]
  // A at 6a (0,0,zA≈0.278), B at 6a (0,0,zB≈0.0), O at 18b (x,y,z)
  {
    name: "LiNbO3-R3c",
    spaceGroup: "R3c",
    latticeType: "hexagonal",
    cOverA: 2.69,
    sites: [
      // A (Li) at 6a → 2 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.278, role: "A-site" },
      { label: "A", x: 0.0, y: 0.0, z: 0.778, role: "A-site" },
      // B (Nb/Ta) at 6a → 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-site" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "B-site" },
      // O at 18b → 6 in primitive
      { label: "C", x: 0.047, y: 0.343, z: 0.063, role: "O-1" },
      { label: "C", x: 0.657, y: 0.953, z: 0.063, role: "O-2" },
      { label: "C", x: 0.343, y: 0.047, z: 0.563, role: "O-3" },
      { label: "C", x: 0.953, y: 0.657, z: 0.563, role: "O-4" },
      { label: "C", x: 0.657, y: 0.704, z: 0.396, role: "O-5" },
      { label: "C", x: 0.296, y: 0.953, z: 0.896, role: "O-6" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [6, 6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      // A-site: small cations that don't fit in CATIONS_LARGE (Li, Na, K, Ag, Cu)
      const hasSmallA = elements.some(e => ["Li", "Na", "K", "Ag", "Cu"].includes(e));
      // B-site: high-valence TM (Nb, Ta, V, W, Mo, Ti)
      const hasB = elements.some(e => ["Nb", "Ta", "V", "W", "Mo", "Ti"].includes(e));
      return hasO && hasSmallA && hasB;
    },
  },
  // D88 Mn5Si3-type: P63/mcm (193), A5B3 — refractory silicide/germanide
  // For: Mn5Si3, Ti5Si3, Cr5Si3, V5Si3, Nb5Si3, W5Si3 — high-temp structural
  // Ratio [5,3] previously uncovered. Hexagonal with TM chains.
  // Primitive = conventional (P-type). Per f.u.: 5A + 3B = 8 atoms.
  // But Z=2 in conventional, so per f.u. in primitive: 5+3=8 from Z=2→ half = no,
  // P63/mcm primitive IS conventional. 2 f.u.: 10A + 6B = 16 atoms.
  // Use 1 f.u. representation: 5A + 3B = 8 atoms.
  {
    name: "D88-A5B3",
    spaceGroup: "P63/mcm",
    latticeType: "hexagonal",
    cOverA: 0.66,
    sites: [
      // A (Mn/Ti/Cr) — 5 per f.u.
      // 4d: (1/3, 2/3, 0) → 2 per f.u.
      { label: "A", x: 0.3333, y: 0.6667, z: 0.0, role: "TM-4d-1" },
      { label: "A", x: 0.6667, y: 0.3333, z: 0.0, role: "TM-4d-2" },
      // 6g: (x, 0, 1/4) with x ≈ 0.236 → 3 per f.u.
      { label: "A", x: 0.236, y: 0.0, z: 0.25, role: "TM-6g-1" },
      { label: "A", x: 0.0, y: 0.236, z: 0.25, role: "TM-6g-2" },
      { label: "A", x: 0.764, y: 0.764, z: 0.25, role: "TM-6g-3" },
      // B (Si/Ge/Sn/Ga) — 3 per f.u.
      // 6g: (x, 0, 1/4) with x ≈ 0.599
      { label: "B", x: 0.599, y: 0.0, z: 0.25, role: "Si-6g-1" },
      { label: "B", x: 0.0, y: 0.599, z: 0.25, role: "Si-6g-2" },
      { label: "B", x: 0.401, y: 0.401, z: 0.25, role: "Si-6g-3" },
    ],
    stoichiometryRatio: [5, 3],
    coordination: [12, 9],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasTM = elements.some(e => ["Mn", "Ti", "Cr", "V", "Nb", "Ta", "W", "Mo", "Fe", "Zr", "Hf"].includes(e));
      const hasSemimetal = elements.some(e => ["Si", "Ge", "Sn", "Ga", "Al", "In", "P", "As", "B"].includes(e));
      return hasTM && hasSemimetal;
    },
  },

  // GeS/SnSe-type: Pnma (62), AB — orthorhombic IV-VI thermoelectric
  // For: SnSe, SnS, GeSe, GeS, PbS (ortho), InSe — record-ZT thermoelectrics
  // CRITICAL: SnSe had ZERO template match — Sn is not TM, so MnP/FeB/NiAs fail.
  // Pnma with 2 formula units: 2A + 2B = 4 sites, ratio [1,1]
  // Sn at 4c (x, 1/4, z), Se at 4c (x, 1/4, z)
  {
    name: "GeS-Pnma",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 0.92,
    sites: [
      // A (Sn/Ge/Pb/In) at 4c
      { label: "A", x: 0.12, y: 0.25, z: 0.10, role: "cation" },
      { label: "A", x: 0.62, y: 0.75, z: 0.40, role: "cation" },
      // B (Se/S/Te) at 4c
      { label: "B", x: 0.85, y: 0.25, z: 0.52, role: "anion" },
      { label: "B", x: 0.35, y: 0.75, z: 0.98, role: "anion" },
    ],
    stoichiometryRatio: [1, 1],
    coordination: [3, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // Group-14/15 main-group + chalcogen (no TM required)
      const hasCation = elements.some(e => ["Sn", "Ge", "Pb", "In", "Tl", "Sb", "Bi", "As"].includes(e));
      const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
      return hasCation && hasChalcogen;
    },
  },
  // A7 Rhombohedral elemental: R-3m (166) — Bi/Sb/As structure
  // For: Bi, Sb, As — topological semimetals, pnictogen elements
  // Distinct from FCC/BCC/HCP: buckled bilayer with rhombohedral stacking.
  // Primitive cell: 2 atoms at (x, x, x) and (-x, -x, -x) with x ≈ 0.234
  // Using hexagonal setting: 2 atoms, ratio [1]
  {
    name: "A7-Rhombohedral",
    spaceGroup: "R-3m",
    latticeType: "hexagonal",
    cOverA: 2.61,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.234, role: "pnictogen-up" },
      { label: "A", x: 0.0, y: 0.0, z: 0.766, role: "pnictogen-down" },
    ],
    stoichiometryRatio: [1],
    coordination: [3],
    chemistryRules: (elements) => {
      if (elements.length !== 1) return false;
      return ["Bi", "Sb", "As", "P"].includes(elements[0]);
    },
  },

  // A-type RE sesquioxide: P-3m1 (164), A2O3 — hexagonal La2O3-type
  // For: La2O3, Ce2O3, Pr2O3, Nd2O3, Pm2O3, Sm2O3, Eu2O3
  // Distinct from corundum (R-3c): hexagonal layered vs rhombohedral.
  // CRITICAL: Corundum excludes RE elements — ALL RE sesquioxides had no match.
  // Primitive cell: 2A + 3O = 5 atoms, ratio [2,3]
  // A at 2d (1/3, 2/3, z≈0.245), O at 1a (0,0,0) + 2d (1/3, 2/3, z≈0.645)
  {
    name: "A-RE2O3",
    spaceGroup: "P-3m1",
    latticeType: "hexagonal",
    cOverA: 1.56,
    sites: [
      // RE at 2d
      { label: "A", x: 0.3333, y: 0.6667, z: 0.245, role: "RE-layer" },
      { label: "A", x: 0.6667, y: 0.3333, z: 0.755, role: "RE-layer" },
      // O at 1a
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "O-central" },
      // O at 2d
      { label: "B", x: 0.3333, y: 0.6667, z: 0.645, role: "O-layer" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.355, role: "O-layer" },
    ],
    stoichiometryRatio: [2, 3],
    coordination: [7, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasO = elements.includes("O");
      // A-type sesquioxide: RE or actinide + O
      const hasRE = elements.some(e => isRareEarth(e) || ["Y", "Sc", "Bi", "Pu", "Am"].includes(e));
      return hasO && hasRE;
    },
  },
  // A5 β-Sn (white tin): I41/amd (141), elemental tetragonal
  // For: Sn (Tc=3.7K superconductor), In (Tc=3.4K), Pa — tetragonal elements
  // 5th elemental structure type after BCC, FCC, HCP, Diamond, A7.
  // Primitive cell (BCC → half conventional): 2 atoms, ratio [1]
  // Sn at 4a (0, 0, 0) in conventional → (0, 0, 0) + (0, 1/2, 1/4) in primitive
  {
    name: "A5-betaSn",
    spaceGroup: "I41/amd",
    latticeType: "tetragonal",
    cOverA: 0.546,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "tetragonal-1" },
      { label: "A", x: 0.0, y: 0.5, z: 0.25, role: "tetragonal-2" },
    ],
    stoichiometryRatio: [1],
    coordination: [4],
    chemistryRules: (elements) => {
      if (elements.length !== 1) return false;
      // β-Sn type: tetragonal elements (between diamond and metallic)
      return ["Sn", "In", "Ga", "Pa"].includes(elements[0]);
    },
  },

  // Oxynitride/Oxyfluoride perovskite: Pm-3m-derived, ABO2X (4 elements)
  // For: SrTaO2N, BaTaO2N, LaTiO2N, KNbO2F — visible-light photocatalysts
  // CRITICAL: Cubic Perovskite template only accepts 3 elements.
  // These 4-element ABO2X perovskites had ZERO match.
  // Positions identical to cubic perovskite but with mixed anion site.
  // Primitive: 1A + 1B + 2O + 1X = 5 atoms, ratio [1,1,2,1]
  {
    name: "OxynitridePerovskite-ABOX",
    spaceGroup: "Pm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A (Sr/Ba/La/Ca) at 1a
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site" },
      // B (Ta/Nb/Ti/V/W) at 1b
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "B-site" },
      // O at 3c (face centers) — 2 of 3 are O
      { label: "C", x: 0.5, y: 0.5, z: 0.0, role: "O-equatorial" },
      { label: "C", x: 0.5, y: 0.0, z: 0.5, role: "O-equatorial" },
      // X (N/F) at 3c — 1 of 3 is N or F
      { label: "D", x: 0.0, y: 0.5, z: 0.5, role: "X-anion" },
    ],
    stoichiometryRatio: [1, 1, 2, 1],
    coordination: [12, 6, 2, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasX = elements.some(e => ["N", "F", "Cl"].includes(e));
      const hasA = elements.some(e => CATIONS_LARGE.has(e));
      const hasB = elements.some(e => ["Ta", "Nb", "Ti", "V", "W", "Mo", "Zr", "Hf", "Mn", "Fe"].includes(e));
      return hasO && hasX && hasA && hasB;
    },
  },
  // Stibnite: Pnma (62), A2B3 — orthorhombic pnictide/chalcogenide
  // For: Bi2S3, Sb2S3, Bi2Se3-ortho, Sb2Se3 — solar absorbers, thermoelectrics
  // Distinct from Bi2Te3-R3m (rhombohedral quintuple-layer topology).
  // Bi2S3 adopts stibnite structure, NOT rhombohedral.
  // Primitive: 2A + 3B = 5 atoms per f.u., ratio [2,3]
  {
    name: "Stibnite-A2B3",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 0.29,
    sites: [
      // A (Bi/Sb) at 4c-like: 2 per f.u.
      { label: "A", x: 0.517, y: 0.25, z: 0.174, role: "cation-1" },
      { label: "A", x: 0.660, y: 0.25, z: 0.532, role: "cation-2" },
      // B (S/Se/Te) at 4c-like: 3 per f.u.
      { label: "B", x: 0.375, y: 0.25, z: 0.056, role: "anion-1" },
      { label: "B", x: 0.722, y: 0.25, z: 0.809, role: "anion-2" },
      { label: "B", x: 0.459, y: 0.25, z: 0.379, role: "anion-3" },
    ],
    stoichiometryRatio: [2, 3],
    coordination: [7, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // Stibnite: heavy pnictogen/post-TM + chalcogen
      const hasCation = elements.some(e => ["Bi", "Sb", "As", "In", "Tl"].includes(e));
      const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
      return hasCation && hasChalcogen;
    },
  },

  // Halide Perovskite: Pm-3m (221), ABX3 — solar cell materials
  // For: CsPbI3, CsPbBr3, CsPbCl3, CsSnI3, CsSnBr3, CsGeBr3
  // CRITICAL: Oxide Perovskite template excludes Pb/Sn/Ge at B-site
  // (not in CATIONS_SMALL_TM) and halides at X-site.
  // Same cubic positions as oxide perovskite but distinct chemistry.
  // Primitive: 1A + 1B + 3X = 5 atoms, ratio [1,1,3]
  {
    name: "HalidePerovskite-ABX3",
    spaceGroup: "Pm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A (Cs/Rb/K) at 1a
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site" },
      // B (Pb/Sn/Ge) at 1b
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "B-site" },
      // X (I/Br/Cl/F) at 3c
      { label: "C", x: 0.5, y: 0.5, z: 0.0, role: "halide" },
      { label: "C", x: 0.5, y: 0.0, z: 0.5, role: "halide" },
      { label: "C", x: 0.0, y: 0.5, z: 0.5, role: "halide" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [12, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      // A-site: large monovalent cation
      const hasA = elements.some(e => ["Cs", "Rb", "K", "Na", "Tl"].includes(e));
      // B-site: Pb, Sn, Ge, Bi (heavy p-block metals)
      const hasB = elements.some(e => ["Pb", "Sn", "Ge", "Bi", "Sb", "In"].includes(e));
      // X-site: halide
      const hasX = elements.some(e => ["I", "Br", "Cl", "F"].includes(e));
      return hasA && hasB && hasX;
    },
  },

  // Hg-1212 cuprate: P4/mmm (123), AB2CD2O6 — double-CuO2-layer
  // HgBa2CaCu2O6+δ — Tc ≈ 127K, highest CONFIRMED ambient-pressure Tc.
  // Simple tetragonal like Hg-1201 but with Ca spacer + double CuO2 planes.
  // Primitive: 1Hg + 2Ba + 1Ca + 2Cu + 6O = 12 atoms, ratio [1,2,1,2,6]
  {
    name: "Hg1212-AB2CD2O6",
    spaceGroup: "P4/mmm",
    latticeType: "tetragonal",
    cOverA: 3.16,
    sites: [
      // Hg at 1a (0,0,0)
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Hg-layer" },
      // Ba at 2h (1/2, 1/2, z≈0.22)
      { label: "B", x: 0.5, y: 0.5, z: 0.22, role: "Ba-site" },
      { label: "B", x: 0.5, y: 0.5, z: 0.78, role: "Ba-site" },
      // Ca at 1b (0, 0, 1/2) — spacer between CuO2 planes
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "Ca-spacer" },
      // Cu at 2g (0, 0, z≈0.38) — double CuO2 plane
      { label: "D", x: 0.0, y: 0.0, z: 0.38, role: "CuO2-plane" },
      { label: "D", x: 0.0, y: 0.0, z: 0.62, role: "CuO2-plane" },
      // O: 6 total
      { label: "E", x: 0.5, y: 0.0, z: 0.38, role: "O-planar" },
      { label: "E", x: 0.0, y: 0.5, z: 0.38, role: "O-planar" },
      { label: "E", x: 0.5, y: 0.0, z: 0.62, role: "O-planar" },
      { label: "E", x: 0.0, y: 0.5, z: 0.62, role: "O-planar" },
      { label: "E", x: 0.0, y: 0.0, z: 0.15, role: "O-apical" },
      { label: "E", x: 0.0, y: 0.0, z: 0.85, role: "O-apical" },
    ],
    stoichiometryRatio: [1, 2, 1, 2, 6],
    coordination: [2, 10, 8, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 5) return false;
      const hasO = elements.includes("O");
      const hasCu = elements.includes("Cu");
      const hasHg = elements.includes("Hg");
      const hasAE = elements.some(e => ["Ba", "Sr"].includes(e));
      const hasSpacer = elements.includes("Ca") || elements.some(e => isRareEarth(e) || e === "Y");
      return hasO && hasCu && hasHg && hasAE && hasSpacer;
    },
  },
  // Hg-1223 cuprate: P4/mmm (123), AB2C2D3O8 — triple-CuO2-layer
  // HgBa2Ca2Cu3O8+δ — Tc ≈ 134K, THE HIGHEST Tc at ambient pressure.
  // World record holder. Triple CuO2 planes with Hg-O charge reservoir.
  // Primitive: 1Hg + 2Ba + 2Ca + 3Cu + 8O = 16 atoms, ratio [1,2,2,3,8]
  {
    name: "Hg1223-AB2C2D3O8",
    spaceGroup: "P4/mmm",
    latticeType: "tetragonal",
    cOverA: 3.95,
    sites: [
      // Hg at 1a
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Hg-layer" },
      // Ba at 2h
      { label: "B", x: 0.5, y: 0.5, z: 0.175, role: "Ba-site" },
      { label: "B", x: 0.5, y: 0.5, z: 0.825, role: "Ba-site" },
      // Ca at 2g — double spacer
      { label: "C", x: 0.0, y: 0.0, z: 0.39, role: "Ca-spacer" },
      { label: "C", x: 0.0, y: 0.0, z: 0.61, role: "Ca-spacer" },
      // Cu: 3 total (1 inner + 2 outer)
      { label: "D", x: 0.0, y: 0.0, z: 0.5, role: "Cu-inner" },
      { label: "D", x: 0.0, y: 0.0, z: 0.30, role: "Cu-outer" },
      { label: "D", x: 0.0, y: 0.0, z: 0.70, role: "Cu-outer" },
      // O: 8 total
      { label: "E", x: 0.5, y: 0.0, z: 0.5, role: "O-inner-eq" },
      { label: "E", x: 0.0, y: 0.5, z: 0.5, role: "O-inner-eq" },
      { label: "E", x: 0.5, y: 0.0, z: 0.30, role: "O-outer-eq" },
      { label: "E", x: 0.0, y: 0.5, z: 0.30, role: "O-outer-eq" },
      { label: "E", x: 0.5, y: 0.0, z: 0.70, role: "O-outer-eq" },
      { label: "E", x: 0.0, y: 0.5, z: 0.70, role: "O-outer-eq" },
      { label: "E", x: 0.0, y: 0.0, z: 0.12, role: "O-apical" },
      { label: "E", x: 0.0, y: 0.0, z: 0.88, role: "O-apical" },
    ],
    stoichiometryRatio: [1, 2, 2, 3, 8],
    coordination: [2, 10, 8, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 5) return false;
      const hasO = elements.includes("O");
      const hasCu = elements.includes("Cu");
      const hasHg = elements.includes("Hg");
      const hasAE = elements.some(e => ["Ba", "Sr"].includes(e));
      const hasSpacer = elements.includes("Ca") || elements.some(e => isRareEarth(e) || e === "Y");
      return hasO && hasCu && hasHg && hasAE && hasSpacer;
    },
  },

  // BaAl4-type: I4/mmm (139), AB4 — heavy-fermion / topological intermetallic
  // For: CeAl4, BaAl4, SrGa4, EuIn4, CeRh2Si2-related aluminides
  // Distinct from Tetraboride (P4/mbm, B only) — this handles Al/Ga/In/Si.
  // Primitive cell (BCC → half conv.): 1A + 4B = 5 atoms, ratio [1,4]
  // A at 2a, B at 4d (0,1/2,1/4) + 4e (0,0,z≈0.38)
  {
    name: "BaAl4-AB4",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 2.77,
    sites: [
      // A (Ba/Sr/Ca/Ce/Eu) at 2a → 1 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site" },
      // B at 4d → 2 in primitive
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "B-basal" },
      { label: "B", x: 0.5, y: 0.0, z: 0.25, role: "B-basal" },
      // B at 4e → 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.38, role: "B-apical" },
      { label: "B", x: 0.0, y: 0.0, z: 0.62, role: "B-apical" },
    ],
    stoichiometryRatio: [1, 4],
    coordination: [16, 9],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasA = elements.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Y", "Sc", "Th", "Eu", "Yb"].includes(e));
      const hasB = elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Cu", "Ag", "Au", "Zn", "Cd"].includes(e));
      return hasA && hasB;
    },
  },
  // Quaternary Heusler: F-43m (216), ABCD — equiatomic spintronic
  // For: CoFeMnSi, CoFeMnGe, NiFeMnGa, LiMgPdSn — spin-gapless, half-metals
  // 4 distinct sublattices in FCC framework. Growing family for spintronics.
  // Primitive cell (FCC → 1/4 conv.): 1A + 1B + 1C + 1D = 4 atoms, ratio [1,1,1,1]
  // A at 4a, B at 4b, C at 4c, D at 4d
  {
    name: "QuaternaryHeusler-ABCD",
    spaceGroup: "F-43m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "site-4a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "site-4b" },
      { label: "C", x: 0.25, y: 0.25, z: 0.25, role: "site-4c" },
      { label: "D", x: 0.75, y: 0.75, z: 0.75, role: "site-4d" },
    ],
    stoichiometryRatio: [1, 1, 1, 1],
    coordination: [4, 4, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      // Quaternary Heusler: 4 different metals, typically 2-3 TMs + sp-element
      const tmCount = elements.filter(e => isTransitionMetal(e)).length;
      const hasSp = elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Sb", "Bi"].includes(e));
      // Need at least 2 TMs and at least 1 sp-block element
      return tmCount >= 2 && hasSp;
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
      const hasRE = elements.some(e => ["Y", "La", "Lu", "Sc", "Gd", "Nd", "Ce", "Ti", "Zr", "Hf", "Nb", "Ta", "V"].includes(e));
      const hasTM = elements.some(e => ["Pt", "Pd", "Ni", "Au", "Ir", "Rh", "Co", "Fe", "Mn", "Cu"].includes(e));
      const hasSp = elements.some(e => ["Bi", "Sb", "Sn", "Pb", "In", "Te", "Ge", "Si", "Ga", "As", "Se"].includes(e));
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
  const formulaAtomCount = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  // Collect all matching templates, then prefer the one whose atom count
  // best matches the formula (exact > smallest-multiple > fewest-atoms)
  const matches: { template: PrototypeTemplate; siteMap: Record<string, string>; nAtoms: number }[] = [];
  for (const template of PROTOTYPE_TEMPLATES) {
    if (!template.chemistryRules(elements)) continue;
    const siteMap = sortElementsBySite(elements, counts, template);
    if (siteMap) matches.push({ template, siteMap, nAtoms: template.sites.length });
  }

  if (matches.length === 0) return null;

  // Sort: exact atom count first, then clean multiples, then fewest atoms
  matches.sort((a, b) => {
    const aExact = a.nAtoms === formulaAtomCount ? 0 : 1;
    const bExact = b.nAtoms === formulaAtomCount ? 0 : 1;
    if (aExact !== bExact) return aExact - bExact;
    const aMult = a.nAtoms % formulaAtomCount === 0 ? 0 : 1;
    const bMult = b.nAtoms % formulaAtomCount === 0 ? 0 : 1;
    if (aMult !== bMult) return aMult - bMult;
    return a.nAtoms - b.nAtoms;
  });

  return { template: matches[0].template, siteMap: matches[0].siteMap };
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
