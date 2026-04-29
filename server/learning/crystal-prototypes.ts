import { getElementData, isTransitionMetal, isRareEarth } from "./elemental-data";
import { parseFormulaCounts as parseFormulaCountsCanonical } from "./utils";

/** Supported crystal systems. Orthorhombic/monoclinic/triclinic added for
 *  lower-symmetry structures (wolframite, baddeleyite, etc.). */
export type LatticeType = "cubic" | "hexagonal" | "tetragonal" | "orthorhombic" | "monoclinic" | "triclinic";

export interface PrototypeTemplate {
  name: string;
  spaceGroup: string;
  latticeType: LatticeType;
  cOverA: number;
  /** b/a ratio — defaults to 1.0 for cubic/hex/tet. Required for ortho/mono/tri. */
  bOverA?: number;
  /** Monoclinic beta angle in degrees (default 90). */
  beta?: number;
  /** Triclinic angles in degrees (default 90 each). */
  alpha?: number;
  gamma?: number;
  sites: {
    label: string;
    x: number; y: number; z: number;
    role: string;
    /** Wyckoff label (e.g., "4a", "32f", "12d"). Populated for cage templates. */
    wyckoff?: string;
    /** Orbit fill priority for partial occupation (lower = fill first). */
    orbitPriority?: number;
  }[];
  stoichiometryRatio: number[];
  coordination: number[];
  chemistryRules: (elements: string[]) => boolean;
  /** If this template represents a cage structure, tag the cage type. */
  cageType?: "clathrate" | "sodalite" | "hex-clathrate" | "bcc-hydride";
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
  "Tl1212-AB2CD2O7": 0.55,
  "MAB-phase": 0.68,
  "Tl1223-AB2C2D3O9": 0.55,
  "Tetradymite-A2B2C": 0.52,
  "Aurivillius-n2-AB2C2O9": 0.55,
  // ── Monoclinic prototypes ──
  "Wolframite-ABO4": 0.62,
  "Baddeleyite-AO2": 0.62,
  "VO2-monoclinic": 0.65,
  "Monazite-ABO4": 0.60,
  // ── Ratio-gap fills ──
  "VacancyPerovskite-A2BX6": 0.68,
  "Columbite-AB2O6": 0.62,
  "NormalSpinel-AB3O4": 0.74,
  "Antibixbyite-A3N2": 0.65,
  "CoSb3-AB3C3": 0.60,
  "InverseSpinelTernary-A2B3": 0.74,
  "OrderedPerovskite-ABCO3": 0.74,
  "QuaternaryOlivine-ABCO4": 0.60,
  "OrderedSpinel-AB2CO4": 0.68,
  "QuaternaryChalc-ABCX2": 0.55,
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

/** Estimated lattice parameters. For cubic/hex/tet, b=a and all angles=90
 *  (except hex gamma=120). For ortho/mono/tri, b and angles may differ. */
export interface LatticeParams {
  a: number;
  b: number;
  c: number;
  alpha: number;   // degrees
  beta: number;    // degrees
  gamma: number;   // degrees
}

export function estimateLatticeConstant(
  elements: string[],
  counts: Record<string, number>,
  template: PrototypeTemplate
): LatticeParams {
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

  const bOverA = template.bOverA ?? 1.0;
  const alphaD = template.alpha ?? 90;
  const betaD = template.beta ?? 90;
  const gammaD = template.gamma ?? 90;

  let a: number;
  let b: number;
  let c: number;

  if (template.latticeType === "cubic") {
    a = Math.pow(cellVolume, 1.0 / 3.0);
    b = a;
    c = a;
  } else if (template.latticeType === "hexagonal") {
    const hexFactor = (Math.sqrt(3) / 2) * template.cOverA;
    a = Math.pow(cellVolume / hexFactor, 1.0 / 3.0);
    b = a;
    c = a * template.cOverA;
  } else if (template.latticeType === "monoclinic" || template.latticeType === "triclinic") {
    // V = a * b * c * sin(beta) for monoclinic; general formula for triclinic
    const alphaR = alphaD * Math.PI / 180;
    const betaR = betaD * Math.PI / 180;
    const gammaR = gammaD * Math.PI / 180;
    const cosA = Math.cos(alphaR), cosB = Math.cos(betaR), cosG = Math.cos(gammaR);
    const angleFactor = Math.sqrt(1 - cosA * cosA - cosB * cosB - cosG * cosG + 2 * cosA * cosB * cosG);
    // V = a * b * c * angleFactor = a * (bOverA*a) * (cOverA*a) * angleFactor
    const volumeFactor = bOverA * template.cOverA * angleFactor;
    a = Math.pow(cellVolume / volumeFactor, 1.0 / 3.0);
    b = a * bOverA;
    c = a * template.cOverA;
  } else if (template.latticeType === "orthorhombic") {
    // V = a * b * c = a * (bOverA*a) * (cOverA*a) = a^3 * bOverA * cOverA
    const volumeFactor = bOverA * template.cOverA;
    a = Math.pow(cellVolume / volumeFactor, 1.0 / 3.0);
    b = a * bOverA;
    c = a * template.cOverA;
  } else {
    // tetragonal: V = a^2 * c = a^3 * cOverA
    a = Math.pow(cellVolume / template.cOverA, 1.0 / 3.0);
    b = a;
    c = a * template.cOverA;
  }

  return {
    a, b, c,
    alpha: alphaD,
    beta: betaD,
    gamma: template.latticeType === "hexagonal" ? 120 : gammaD,
  };
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner", wyckoff: "2a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "body-center", wyckoff: "2a" },
      { label: "B", x: 0.25, y: 0.0, z: 0.5, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.75, y: 0.0, z: 0.5, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.5, y: 0.25, z: 0.0, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.5, y: 0.75, z: 0.0, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.0, y: 0.5, z: 0.75, role: "chain", wyckoff: "6c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal-layer", wyckoff: "1a" },
      { label: "B", x: 0.333, y: 0.667, z: 0.5, role: "honeycomb", wyckoff: "2d" },
      { label: "B", x: 0.667, y: 0.333, z: 0.5, role: "honeycomb", wyckoff: "2d" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "spacer", wyckoff: "2a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "spacer", wyckoff: "2a" },
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "TM-layer", wyckoff: "4d" },
      { label: "B", x: 0.5, y: 0.0, z: 0.25, role: "TM-layer", wyckoff: "4d" },
      { label: "B", x: 0.0, y: 0.5, z: 0.75, role: "TM-layer", wyckoff: "4d" },
      { label: "B", x: 0.5, y: 0.0, z: 0.75, role: "TM-layer", wyckoff: "4d" },
      { label: "C", x: 0.0, y: 0.0, z: 0.35, role: "pnictogen", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.65, role: "pnictogen", wyckoff: "4e" },
      { label: "C", x: 0.5, y: 0.5, z: 0.85, role: "pnictogen", wyckoff: "4e" },
      { label: "C", x: 0.5, y: 0.5, z: 0.15, role: "pnictogen", wyckoff: "4e" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation", wyckoff: "4a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.0, role: "cation", wyckoff: "4a" },
      { label: "A", x: 0.5, y: 0.0, z: 0.5, role: "cation", wyckoff: "4a" },
      { label: "A", x: 0.0, y: 0.5, z: 0.5, role: "cation", wyckoff: "4a" },
      { label: "B", x: 0.5, y: 0.0, z: 0.0, role: "anion", wyckoff: "4b" },
      { label: "B", x: 0.0, y: 0.5, z: 0.0, role: "anion", wyckoff: "4b" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "anion", wyckoff: "4b" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "anion", wyckoff: "4b" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site", wyckoff: "1a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "B-site", wyckoff: "1b" },
      { label: "C", x: 0.5, y: 0.5, z: 0.0, role: "anion", wyckoff: "3c" },
      { label: "C", x: 0.5, y: 0.0, z: 0.5, role: "anion", wyckoff: "3c" },
      { label: "C", x: 0.0, y: 0.5, z: 0.5, role: "anion", wyckoff: "3c" },
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
      { label: "M", x: 0.333, y: 0.667, z: 0.25, role: "metal", wyckoff: "2c" },
      { label: "X", x: 0.333, y: 0.667, z: 0.621, role: "chalcogen-top", wyckoff: "4f" },
      { label: "X", x: 0.333, y: 0.667, z: 0.879, role: "chalcogen-bot", wyckoff: "4f" },
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
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "body-center-anion", wyckoff: "1b" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "corner-cation", wyckoff: "1a" },
      { label: "X", x: 0.5, y: 0.5, z: 0.0, role: "face-center", wyckoff: "3c" },
      { label: "X", x: 0.5, y: 0.0, z: 0.5, role: "face-center", wyckoff: "3c" },
      { label: "X", x: 0.0, y: 0.5, z: 0.5, role: "face-center", wyckoff: "3c" },
    ],
    stoichiometryRatio: [1, 1, 3],
    coordination: [6, 12, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasAnion = elements.some(e => ["C", "N", "B", "O"].includes(e));
      // Expanded to include alkali metals: Li3OCl, Na3OBr are key solid electrolytes
      const hasMetal = elements.some(e => isTransitionMetal(e) || CATIONS_LARGE.has(e) || ["Li", "Na", "K", "Mg"].includes(e));
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
    cageType: "bcc-hydride",
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "metal-center", wyckoff: "2a", orbitPriority: 0 },
      { label: "H", x: 0.0, y: 0.5, z: 0.5, role: "octahedral", wyckoff: "6b", orbitPriority: 1 },
      { label: "H", x: 0.5, y: 0.0, z: 0.5, role: "octahedral", wyckoff: "6b", orbitPriority: 1 },
      { label: "H", x: 0.5, y: 0.5, z: 0.0, role: "octahedral", wyckoff: "6b", orbitPriority: 1 },
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
    cageType: "sodalite",
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "cage-center", wyckoff: "2a", orbitPriority: 0 },
      { label: "H", x: 0.50, y: 0.75, z: 0.25, role: "sodalite-cage", wyckoff: "12d", orbitPriority: 1 },
      { label: "H", x: 0.50, y: 0.25, z: 0.75, role: "sodalite-cage", wyckoff: "12d", orbitPriority: 1 },
      { label: "H", x: 0.75, y: 0.50, z: 0.25, role: "sodalite-cage", wyckoff: "12d", orbitPriority: 1 },
      { label: "H", x: 0.25, y: 0.50, z: 0.75, role: "sodalite-cage", wyckoff: "12d", orbitPriority: 1 },
      { label: "H", x: 0.75, y: 0.25, z: 0.50, role: "sodalite-cage", wyckoff: "12d", orbitPriority: 1 },
      { label: "H", x: 0.25, y: 0.75, z: 0.50, role: "sodalite-cage", wyckoff: "12d", orbitPriority: 1 },
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
    cageType: "hex-clathrate",
    sites: [
      { label: "M", x: 0.3333, y: 0.6667, z: 0.25, role: "cage-center", wyckoff: "2d", orbitPriority: 0 },
      { label: "H", x: 0.155, y: 0.310, z: 0.25, role: "hex-cage-6h", wyckoff: "6h", orbitPriority: 1 },
      { label: "H", x: 0.690, y: 0.845, z: 0.25, role: "hex-cage-6h", wyckoff: "6h", orbitPriority: 1 },
      { label: "H", x: 0.845, y: 0.155, z: 0.25, role: "hex-cage-6h", wyckoff: "6h", orbitPriority: 1 },
      { label: "H", x: 0.0, y: 0.0, z: 0.25, role: "hex-cage-2b", wyckoff: "2b", orbitPriority: 2 },
      { label: "H", x: 0.520, y: 0.040, z: 0.08, role: "hex-cage-12k", wyckoff: "12k", orbitPriority: 3 },
      { label: "H", x: 0.960, y: 0.480, z: 0.08, role: "hex-cage-12k", wyckoff: "12k", orbitPriority: 3 },
      { label: "H", x: 0.480, y: 0.520, z: 0.08, role: "hex-cage-12k", wyckoff: "12k", orbitPriority: 3 },
      { label: "H", x: 0.040, y: 0.520, z: 0.42, role: "hex-cage-12k", wyckoff: "12k", orbitPriority: 3 },
      { label: "H", x: 0.520, y: 0.480, z: 0.42, role: "hex-cage-12k", wyckoff: "12k", orbitPriority: 3 },
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
    cageType: "clathrate",
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "cage-center", wyckoff: "4a", orbitPriority: 0 },
      { label: "H", x: 0.375, y: 0.375, z: 0.375, role: "32f-cage", wyckoff: "32f", orbitPriority: 1 },
      { label: "H", x: 0.375, y: 0.375, z: 0.875, role: "32f-cage", wyckoff: "32f", orbitPriority: 1 },
      { label: "H", x: 0.375, y: 0.875, z: 0.375, role: "32f-cage", wyckoff: "32f", orbitPriority: 1 },
      { label: "H", x: 0.875, y: 0.375, z: 0.375, role: "32f-cage", wyckoff: "32f", orbitPriority: 1 },
      { label: "H", x: 0.625, y: 0.625, z: 0.625, role: "32f-cage", wyckoff: "32f", orbitPriority: 1 },
      { label: "H", x: 0.625, y: 0.625, z: 0.125, role: "32f-cage", wyckoff: "32f", orbitPriority: 1 },
      { label: "H", x: 0.625, y: 0.125, z: 0.625, role: "32f-cage", wyckoff: "32f", orbitPriority: 1 },
      { label: "H", x: 0.125, y: 0.625, z: 0.625, role: "32f-cage", wyckoff: "32f", orbitPriority: 1 },
      { label: "H", x: 0.25, y: 0.25, z: 0.25, role: "8c-interstitial", wyckoff: "8c", orbitPriority: 2 },
      { label: "H", x: 0.75, y: 0.75, z: 0.75, role: "8c-interstitial", wyckoff: "8c", orbitPriority: 2 },
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
    cageType: "clathrate",
    sites: [
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "cage-center", wyckoff: "4a", orbitPriority: 0 },
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "interstitial", wyckoff: "8c", orbitPriority: 1 },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "interstitial", wyckoff: "8c", orbitPriority: 1 },
      { label: "H", x: 0.375, y: 0.375, z: 0.375, role: "32f-cage", wyckoff: "32f", orbitPriority: 2 },
      { label: "H", x: 0.375, y: 0.375, z: 0.875, role: "32f-cage", wyckoff: "32f", orbitPriority: 2 },
      { label: "H", x: 0.375, y: 0.875, z: 0.375, role: "32f-cage", wyckoff: "32f", orbitPriority: 2 },
      { label: "H", x: 0.875, y: 0.375, z: 0.375, role: "32f-cage", wyckoff: "32f", orbitPriority: 2 },
      { label: "H", x: 0.625, y: 0.625, z: 0.625, role: "32f-cage", wyckoff: "32f", orbitPriority: 2 },
      { label: "H", x: 0.625, y: 0.625, z: 0.125, role: "32f-cage", wyckoff: "32f", orbitPriority: 2 },
      { label: "H", x: 0.625, y: 0.125, z: 0.625, role: "32f-cage", wyckoff: "32f", orbitPriority: 2 },
      { label: "H", x: 0.125, y: 0.625, z: 0.625, role: "32f-cage", wyckoff: "32f", orbitPriority: 2 },
      { label: "H", x: 0.125, y: 0.125, z: 0.125, role: "extra", wyckoff: "32f", orbitPriority: 3 },
      { label: "H", x: 0.875, y: 0.875, z: 0.875, role: "extra", wyckoff: "32f", orbitPriority: 3 },
      { label: "H", x: 0.125, y: 0.375, z: 0.125, role: "extra", wyckoff: "32f", orbitPriority: 3 },
      { label: "H", x: 0.375, y: 0.125, z: 0.125, role: "extra", wyckoff: "32f", orbitPriority: 3 },
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
    cageType: "hex-clathrate",
    sites: [
      { label: "M", x: 0.3333, y: 0.6667, z: 0.25, role: "cage-center", wyckoff: "2d", orbitPriority: 0 },
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "interstice", wyckoff: "2a", orbitPriority: 1 },
      { label: "A", x: 0.0, y: 0.0, z: 0.5, role: "interstice", wyckoff: "2a", orbitPriority: 1 },
      { label: "H", x: 0.155, y: 0.310, z: 0.25, role: "hex-cage", wyckoff: "6h", orbitPriority: 2 },
      { label: "H", x: 0.690, y: 0.845, z: 0.25, role: "hex-cage", wyckoff: "6h", orbitPriority: 2 },
      { label: "H", x: 0.845, y: 0.155, z: 0.25, role: "hex-cage", wyckoff: "6h", orbitPriority: 2 },
      { label: "H", x: 0.0, y: 0.0, z: 0.25, role: "hex-cage", wyckoff: "2b", orbitPriority: 3 },
      { label: "H", x: 0.520, y: 0.040, z: 0.08, role: "hex-cage", wyckoff: "12k", orbitPriority: 4 },
      { label: "H", x: 0.960, y: 0.480, z: 0.08, role: "hex-cage", wyckoff: "12k", orbitPriority: 4 },
      { label: "H", x: 0.480, y: 0.520, z: 0.08, role: "hex-cage", wyckoff: "12k", orbitPriority: 4 },
      { label: "H", x: 0.040, y: 0.520, z: 0.42, role: "hex-cage", wyckoff: "12k", orbitPriority: 4 },
      { label: "H", x: 0.520, y: 0.480, z: 0.42, role: "hex-cage", wyckoff: "12k", orbitPriority: 4 },
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
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "rare-earth", wyckoff: "1h" },
      { label: "B", x: 0.5, y: 0.5, z: 0.185, role: "Ba-site", wyckoff: "2t" },
      { label: "B", x: 0.5, y: 0.5, z: 0.815, role: "Ba-site", wyckoff: "2t" },
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "Cu-chain", wyckoff: "1a" },
      { label: "C", x: 0.0, y: 0.0, z: 0.356, role: "Cu-plane", wyckoff: "2q" },
      { label: "C", x: 0.0, y: 0.0, z: 0.644, role: "Cu-plane", wyckoff: "2q" },
      { label: "D", x: 0.0, y: 0.5, z: 0.0, role: "O-chain", wyckoff: "1e" },
      { label: "D", x: 0.5, y: 0.0, z: 0.378, role: "O-plane", wyckoff: "2s" },
      { label: "D", x: 0.5, y: 0.0, z: 0.622, role: "O-plane", wyckoff: "2s" },
      { label: "D", x: 0.0, y: 0.5, z: 0.378, role: "O-plane", wyckoff: "2r" },
      { label: "D", x: 0.0, y: 0.5, z: 0.622, role: "O-plane", wyckoff: "2r" },
      { label: "D", x: 0.0, y: 0.0, z: 0.159, role: "O-apical", wyckoff: "2q" },
      { label: "D", x: 0.0, y: 0.0, z: 0.841, role: "O-apical", wyckoff: "2q" },
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
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "cage-center", wyckoff: "1a" },
      { label: "B", x: 0.2, y: 0.5, z: 0.5, role: "B-octahedron", wyckoff: "6f" },
      { label: "B", x: 0.8, y: 0.5, z: 0.5, role: "B-octahedron", wyckoff: "6f" },
      { label: "B", x: 0.5, y: 0.2, z: 0.5, role: "B-octahedron", wyckoff: "6f" },
      { label: "B", x: 0.5, y: 0.8, z: 0.5, role: "B-octahedron", wyckoff: "6f" },
      { label: "B", x: 0.5, y: 0.5, z: 0.2, role: "B-octahedron", wyckoff: "6f" },
      { label: "B", x: 0.5, y: 0.5, z: 0.8, role: "B-octahedron", wyckoff: "6f" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "site-1", wyckoff: "8a" },
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "site-2", wyckoff: "8a" },
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
      { label: "A", x: 0.25, y: 0.25, z: 0.345, role: "alkali", wyckoff: "2c" },
      { label: "B", x: 0.75, y: 0.25, z: 0.0, role: "Fe-site", wyckoff: "2a" },
      { label: "C", x: 0.25, y: 0.25, z: 0.737, role: "pnictogen", wyckoff: "2c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "linear-coord", wyckoff: "3a" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "octahedral", wyckoff: "3b" },
      { label: "C", x: 0.0, y: 0.0, z: 0.11, role: "oxygen", wyckoff: "6c" },
      { label: "C", x: 0.0, y: 0.0, z: 0.89, role: "oxygen", wyckoff: "6c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.5, role: "alkali-layer", wyckoff: "3b" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "TM-layer", wyckoff: "3a" },
      { label: "C", x: 0.0, y: 0.0, z: 0.26, role: "O-layer", wyckoff: "6c" },
      { label: "C", x: 0.0, y: 0.0, z: 0.74, role: "O-layer", wyckoff: "6c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation-1", wyckoff: "4a" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "cation-2", wyckoff: "4b" },
      { label: "C", x: 0.25, y: 0.125, z: 0.625, role: "anion", wyckoff: "8d" },
      { label: "C", x: 0.75, y: 0.125, z: 0.875, role: "anion", wyckoff: "8d" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "octahedral-M1", wyckoff: "4a" },
      { label: "A", x: 0.28, y: 0.25, z: 0.97, role: "octahedral-M2", wyckoff: "4c" },
      { label: "B", x: 0.09, y: 0.25, z: 0.42, role: "tetrahedral", wyckoff: "4c" },
      { label: "C", x: 0.10, y: 0.25, z: 0.74, role: "O1", wyckoff: "4c" },
      { label: "C", x: 0.45, y: 0.25, z: 0.22, role: "O2", wyckoff: "4c" },
      { label: "C", x: 0.16, y: 0.04, z: 0.28, role: "O3", wyckoff: "8d" },
      { label: "C", x: 0.16, y: 0.46, z: 0.28, role: "O3", wyckoff: "8d" },
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
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "A-site", wyckoff: "8c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "A-site", wyckoff: "8c" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-site", wyckoff: "4a" },
      { label: "C", x: 0.5, y: 0.5, z: 0.5, role: "B'-site", wyckoff: "4b" },
      { label: "D", x: 0.25, y: 0.0, z: 0.0, role: "O-site", wyckoff: "24e" },
      { label: "D", x: 0.75, y: 0.0, z: 0.0, role: "O-site", wyckoff: "24e" },
      { label: "D", x: 0.0, y: 0.25, z: 0.0, role: "O-site", wyckoff: "24e" },
      { label: "D", x: 0.0, y: 0.75, z: 0.0, role: "O-site", wyckoff: "24e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.25, role: "O-site", wyckoff: "24e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.75, role: "O-site", wyckoff: "24e" },
    ],
    stoichiometryRatio: [2, 1, 1, 6],
    coordination: [12, 6, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      // Expanded: Pb/Bi for piezoelectric double perovskites (PZT, PMN, PFN)
      const hasAE = elements.some(e => ["Sr", "Ba", "Ca", "La", "Nd", "Pr", "Y", "Pb", "Bi", "K", "Na"].includes(e));
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
      { label: "A", x: 0.0, y: 0.0, z: 0.5, role: "A-rock-salt", wyckoff: "2b" },
      { label: "A", x: 0.0, y: 0.0, z: 0.318, role: "A-perovskite", wyckoff: "4e" },
      { label: "A", x: 0.0, y: 0.0, z: 0.682, role: "A-perovskite", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.1, role: "B-site", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.9, role: "B-site", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.5, z: 0.1, role: "O-equatorial", wyckoff: "8g" },
      { label: "C", x: 0.5, y: 0.0, z: 0.1, role: "O-equatorial", wyckoff: "8g" },
      { label: "C", x: 0.0, y: 0.0, z: 0.2, role: "O-apical", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.8, role: "O-apical", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "O-bridging", wyckoff: "2a" },
      { label: "C", x: 0.0, y: 0.5, z: 0.9, role: "O-equatorial", wyckoff: "8g" },
      { label: "C", x: 0.5, y: 0.0, z: 0.9, role: "O-equatorial", wyckoff: "8g" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.199, role: "BiO-layer", wyckoff: "4e" },
      { label: "A", x: 0.0, y: 0.0, z: 0.801, role: "BiO-layer", wyckoff: "4e" },
      // B-sites: Sr/Ba (2 in primitive cell, from 4e)
      { label: "B", x: 0.0, y: 0.0, z: 0.110, role: "SrO-layer", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.890, role: "SrO-layer", wyckoff: "4e" },
      // C-site: Ca (1 in primitive cell, from 2a)
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "Ca-spacer", wyckoff: "2a" },
      // D-sites: Cu (2 in primitive cell, from 4e)
      { label: "D", x: 0.0, y: 0.0, z: 0.054, role: "CuO2-plane", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.946, role: "CuO2-plane", wyckoff: "4e" },
      // E-sites: O (8 in primitive cell)
      { label: "E", x: 0.5, y: 0.0, z: 0.054, role: "O-planar", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.5, z: 0.054, role: "O-planar", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.0, z: 0.149, role: "O-apical", wyckoff: "4e" },
      { label: "E", x: 0.0, y: 0.0, z: 0.851, role: "O-apical", wyckoff: "4e" },
      { label: "E", x: 0.5, y: 0.0, z: 0.946, role: "O-planar", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.5, z: 0.946, role: "O-planar", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.0, z: 0.250, role: "O-BiO", wyckoff: "4e" },
      { label: "E", x: 0.0, y: 0.0, z: 0.750, role: "O-BiO", wyckoff: "4e" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "X-site-4a", wyckoff: "4a" },
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "A-site-4c", wyckoff: "4c" },
      { label: "B", x: 0.75, y: 0.75, z: 0.75, role: "A-site-4d", wyckoff: "4d" },
      { label: "C", x: 0.5, y: 0.5, z: 0.5, role: "B-site-4b", wyckoff: "4b" },
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
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "TM-inner", wyckoff: "2a" },
      // M at 4f: (1/3, 2/3, z) with z ≈ 0.135 — 2 in primitive per f.u.
      { label: "M", x: 0.3333, y: 0.6667, z: 0.135, role: "TM-outer", wyckoff: "4f" },
      { label: "M", x: 0.3333, y: 0.6667, z: 0.865, role: "TM-outer", wyckoff: "4f" },
      // A at 2b: (0, 0, 1/4) — 1 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.25, role: "A-layer", wyckoff: "2b" },
      // X at 4f: (1/3, 2/3, z) with z ≈ 0.070 — 2 in primitive
      { label: "X", x: 0.3333, y: 0.6667, z: 0.070, role: "X-interstitial", wyckoff: "4f" },
      { label: "X", x: 0.3333, y: 0.6667, z: 0.930, role: "X-interstitial", wyckoff: "4f" },
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
      { label: "M", x: 0.3333, y: 0.6667, z: 0.054, role: "TM-inner", wyckoff: "4f" },
      { label: "M", x: 0.3333, y: 0.6667, z: 0.946, role: "TM-inner", wyckoff: "4f" },
      // M at 4e: outer TM layer (z ≈ 0.155)
      { label: "M", x: 0.0, y: 0.0, z: 0.155, role: "TM-outer", wyckoff: "4e" },
      { label: "M", x: 0.0, y: 0.0, z: 0.845, role: "TM-outer", wyckoff: "4e" },
      // A at 2c: A-element layer
      { label: "A", x: 0.3333, y: 0.6667, z: 0.25, role: "A-layer", wyckoff: "2c" },
      // X at 2a: center interstitial
      { label: "X", x: 0.0, y: 0.0, z: 0.0, role: "X-center", wyckoff: "2a" },
      // X at 4f: outer interstitial (z ≈ 0.103)
      { label: "X", x: 0.3333, y: 0.6667, z: 0.103, role: "X-outer", wyckoff: "4f" },
      { label: "X", x: 0.3333, y: 0.6667, z: 0.897, role: "X-outer", wyckoff: "4f" },
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
      { label: "A", x: 0.027, y: 0.25, z: 0.509, role: "A-site-1", wyckoff: "4c" },
      { label: "A", x: 0.522, y: 0.25, z: 0.039, role: "A-site-2", wyckoff: "4c" },
      // B-sites (Fe/Al/Mn): octahedral + tetrahedral
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-octahedral", wyckoff: "4a" },
      { label: "B", x: 0.928, y: 0.25, z: 0.929, role: "B-tetrahedral", wyckoff: "4c" },
      // O-sites: 5 per formula unit
      { label: "C", x: 0.250, y: 0.007, z: 0.231, role: "O-equatorial", wyckoff: "8d" },
      { label: "C", x: 0.028, y: 0.25, z: 0.744, role: "O-apical-1", wyckoff: "4c" },
      { label: "C", x: 0.595, y: 0.25, z: 0.875, role: "O-apical-2", wyckoff: "4c" },
      { label: "C", x: 0.860, y: 0.25, z: 0.070, role: "O-bridging-1", wyckoff: "4c" },
      { label: "C", x: 0.371, y: 0.25, z: 0.419, role: "O-bridging-2", wyckoff: "4c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.5, role: "A-rock-salt", wyckoff: "2b" },
      // A2 at 4e: between outer and inner B layers
      { label: "A", x: 0.0, y: 0.0, z: 0.321, role: "A-outer-perovskite", wyckoff: "4e" },
      { label: "A", x: 0.0, y: 0.0, z: 0.679, role: "A-outer-perovskite", wyckoff: "4e" },
      // A3 at 4e: between two inner B layers (at center of triple block)
      { label: "A", x: 0.0, y: 0.0, z: 0.178, role: "A-inner-perovskite", wyckoff: "4e" },
      // B-sites: 3 in primitive cell
      // B1 at 4e: outer octahedral layers
      { label: "B", x: 0.0, y: 0.0, z: 0.071, role: "B-outer-oct", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.929, role: "B-outer-oct", wyckoff: "4e" },
      // B2 at 2a: inner octahedral layer (center of triple block)
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-inner-oct", wyckoff: "2a" },
      // O-sites: 10 in primitive cell
      // O-equatorial around outer B (4 sites from 8g)
      { label: "C", x: 0.0, y: 0.5, z: 0.071, role: "O-eq-outer", wyckoff: "8g" },
      { label: "C", x: 0.5, y: 0.0, z: 0.071, role: "O-eq-outer", wyckoff: "8g" },
      { label: "C", x: 0.0, y: 0.5, z: 0.929, role: "O-eq-outer", wyckoff: "8g" },
      { label: "C", x: 0.5, y: 0.0, z: 0.929, role: "O-eq-outer", wyckoff: "8g" },
      // O-equatorial around inner B (2 sites from 4c)
      { label: "C", x: 0.0, y: 0.5, z: 0.0, role: "O-eq-inner", wyckoff: "4c" },
      { label: "C", x: 0.5, y: 0.0, z: 0.0, role: "O-eq-inner", wyckoff: "4c" },
      // O-apical between outer B and rock-salt (2 from 4e)
      { label: "C", x: 0.0, y: 0.0, z: 0.393, role: "O-apical-outer", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.607, role: "O-apical-outer", wyckoff: "4e" },
      // O-apical bridging outer-inner B layers (2 from 4e)
      { label: "C", x: 0.0, y: 0.0, z: 0.107, role: "O-apical-bridge", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.893, role: "O-apical-bridge", wyckoff: "4e" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.352, role: "cation", wyckoff: "12c" },
      { label: "A", x: 0.0, y: 0.0, z: 0.648, role: "cation", wyckoff: "12c" },
      { label: "A", x: 0.0, y: 0.0, z: 0.852, role: "cation", wyckoff: "12c" },
      { label: "A", x: 0.0, y: 0.0, z: 0.148, role: "cation", wyckoff: "12c" },
      // O at 18e Wyckoff (6 in rhomb primitive): (x,0,1/4) with x≈0.306
      { label: "B", x: 0.306, y: 0.0, z: 0.25, role: "anion", wyckoff: "18e" },
      { label: "B", x: 0.0, y: 0.306, z: 0.25, role: "anion", wyckoff: "18e" },
      { label: "B", x: 0.694, y: 0.694, z: 0.25, role: "anion", wyckoff: "18e" },
      { label: "B", x: 0.694, y: 0.0, z: 0.75, role: "anion", wyckoff: "18e" },
      { label: "B", x: 0.0, y: 0.694, z: 0.75, role: "anion", wyckoff: "18e" },
      { label: "B", x: 0.306, y: 0.306, z: 0.75, role: "anion", wyckoff: "18e" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "rare-earth", wyckoff: "2a" },
      // Fe at 8f: (1/4, 1/4, 1/4) — 4 in primitive
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "Fe-8f", wyckoff: "8f" },
      { label: "B", x: 0.75, y: 0.25, z: 0.25, role: "Fe-8f", wyckoff: "8f" },
      { label: "B", x: 0.25, y: 0.75, z: 0.25, role: "Fe-8f", wyckoff: "8f" },
      { label: "B", x: 0.75, y: 0.75, z: 0.25, role: "Fe-8f", wyckoff: "8f" },
      // Fe at 8i: (x, 0, 0) with x≈0.36 — 4 in primitive
      { label: "B", x: 0.36, y: 0.0, z: 0.0, role: "Fe-8i", wyckoff: "8i" },
      { label: "B", x: 0.64, y: 0.0, z: 0.0, role: "Fe-8i", wyckoff: "8i" },
      { label: "B", x: 0.0, y: 0.36, z: 0.0, role: "Fe-8i", wyckoff: "8i" },
      { label: "B", x: 0.0, y: 0.64, z: 0.0, role: "Fe-8i", wyckoff: "8i" },
      // Fe at 8j: (x, 1/2, 0) with x≈0.28 — 4 in primitive
      { label: "B", x: 0.28, y: 0.5, z: 0.0, role: "Fe-8j", wyckoff: "8j" },
      { label: "B", x: 0.72, y: 0.5, z: 0.0, role: "Fe-8j", wyckoff: "8j" },
      { label: "B", x: 0.5, y: 0.28, z: 0.0, role: "Fe-8j", wyckoff: "8j" },
      { label: "B", x: 0.5, y: 0.72, z: 0.0, role: "Fe-8j", wyckoff: "8j" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.356, role: "A-layer", wyckoff: "6c" },
      { label: "A", x: 0.0, y: 0.0, z: 0.644, role: "A-layer", wyckoff: "6c" },
      // B cations (Ti): 2 in primitive from 4c
      { label: "B", x: 0.0, y: 0.0, z: 0.146, role: "B-layer", wyckoff: "6c" },
      { label: "B", x: 0.0, y: 0.0, z: 0.854, role: "B-layer", wyckoff: "6c" },
      // O anions: 6 in primitive from 18f-derived
      { label: "C", x: 0.317, y: 0.020, z: 0.245, role: "O-1", wyckoff: "18f" },
      { label: "C", x: 0.980, y: 0.297, z: 0.245, role: "O-2", wyckoff: "18f" },
      { label: "C", x: 0.703, y: 0.683, z: 0.245, role: "O-3", wyckoff: "18f" },
      { label: "C", x: 0.683, y: 0.980, z: 0.755, role: "O-4", wyckoff: "18f" },
      { label: "C", x: 0.020, y: 0.703, z: 0.755, role: "O-5", wyckoff: "18f" },
      { label: "C", x: 0.297, y: 0.317, z: 0.755, role: "O-6", wyckoff: "18f" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal", wyckoff: "2a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "metal", wyckoff: "2a" },
      // X at 4g: (x,y,0) with x≈0.200, y≈0.378 — 4 in primitive
      { label: "B", x: 0.200, y: 0.378, z: 0.0, role: "anion-1", wyckoff: "4g" },
      { label: "B", x: 0.800, y: 0.622, z: 0.0, role: "anion-2", wyckoff: "4g" },
      { label: "B", x: 0.300, y: 0.878, z: 0.5, role: "anion-3", wyckoff: "4g" },
      { label: "B", x: 0.700, y: 0.122, z: 0.5, role: "anion-4", wyckoff: "4g" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.210, role: "AO-layer", wyckoff: "4e" },
      { label: "A", x: 0.0, y: 0.0, z: 0.790, role: "AO-layer", wyckoff: "4e" },
      // B-sites: Ba/Sr (2 in primitive, from 4e)
      { label: "B", x: 0.0, y: 0.0, z: 0.116, role: "BO-layer", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.884, role: "BO-layer", wyckoff: "4e" },
      // Cu at 2a: CuO2 plane
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "CuO2-plane", wyckoff: "2a" },
      // O-sites: 6 in primitive
      { label: "D", x: 0.5, y: 0.0, z: 0.0, role: "O-planar", wyckoff: "4c" },
      { label: "D", x: 0.0, y: 0.5, z: 0.0, role: "O-planar", wyckoff: "4c" },
      { label: "D", x: 0.0, y: 0.0, z: 0.163, role: "O-apical", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.837, role: "O-apical", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.279, role: "O-AO-layer", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.721, role: "O-AO-layer", wyckoff: "4e" },
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
      { label: "A", x: 0.0, y: 0.25, z: 0.25, role: "A-site", wyckoff: "4c" },
      { label: "A", x: 0.0, y: 0.75, z: 0.75, role: "A-site", wyckoff: "4c" },
      // B at 4a: (0, 0, 0) — 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-site", wyckoff: "4a" },
      { label: "B", x: 0.0, y: 0.5, z: 0.5, role: "B-site", wyckoff: "4a" },
      // O1 at 4c: (0, y, 1/4) with y ≈ 0.93 — 2 in primitive
      { label: "C", x: 0.0, y: 0.93, z: 0.25, role: "O-apical", wyckoff: "4c" },
      { label: "C", x: 0.0, y: 0.07, z: 0.75, role: "O-apical", wyckoff: "4c" },
      // O2 at 8f: (0, y, z) with y ≈ 0.63, z ≈ 0.07 — 4 in primitive
      { label: "C", x: 0.0, y: 0.63, z: 0.07, role: "O-equatorial", wyckoff: "8f" },
      { label: "C", x: 0.0, y: 0.37, z: 0.93, role: "O-equatorial", wyckoff: "8f" },
      { label: "C", x: 0.0, y: 0.13, z: 0.57, role: "O-equatorial", wyckoff: "8f" },
      { label: "C", x: 0.0, y: 0.87, z: 0.43, role: "O-equatorial", wyckoff: "8f" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.330, role: "Bi2O2-layer", wyckoff: "4e" },
      { label: "A", x: 0.0, y: 0.0, z: 0.670, role: "Bi2O2-layer", wyckoff: "4e" },
      // B at 2a: perovskite B-site
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "perovskite-B", wyckoff: "2a" },
      // O-sites: 6 in primitive
      { label: "C", x: 0.5, y: 0.0, z: 0.0, role: "O-equatorial", wyckoff: "4c" },
      { label: "C", x: 0.0, y: 0.5, z: 0.0, role: "O-equatorial", wyckoff: "4c" },
      { label: "C", x: 0.0, y: 0.0, z: 0.145, role: "O-apical", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.855, role: "O-apical", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.415, role: "O-BiO-layer", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.585, role: "O-BiO-layer", wyckoff: "4e" },
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
      { label: "A", x: 0.0, y: 0.5, z: 0.25, role: "Cu-site", wyckoff: "4d" },
      { label: "A", x: 0.5, y: 0.0, z: 0.25, role: "Cu-site", wyckoff: "4d" },
      // B (Zn) at 2a: (0, 0, 0) → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "Zn-site", wyckoff: "2a" },
      // C (Sn) at 2b: (0, 0, 1/2) → 1 in primitive
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "Sn-site", wyckoff: "2b" },
      // S at 8i: (x, x, z) with x≈0.245, z≈0.128 → 4 in primitive
      { label: "D", x: 0.245, y: 0.245, z: 0.128, role: "S-1", wyckoff: "8i" },
      { label: "D", x: 0.755, y: 0.755, z: 0.128, role: "S-2", wyckoff: "8i" },
      { label: "D", x: 0.755, y: 0.245, z: 0.872, role: "S-3", wyckoff: "8i" },
      { label: "D", x: 0.245, y: 0.755, z: 0.872, role: "S-4", wyckoff: "8i" },
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
      { label: "A", x: 0.0, y: 0.25, z: 0.625, role: "A-site", wyckoff: "4b" },
      { label: "A", x: 0.0, y: 0.75, z: 0.125, role: "A-site", wyckoff: "4b" },
      // B (W/Mo/V) at 4a → 2 in primitive
      { label: "B", x: 0.0, y: 0.25, z: 0.125, role: "B-site", wyckoff: "4a" },
      { label: "B", x: 0.0, y: 0.75, z: 0.625, role: "B-site", wyckoff: "4a" },
      // O at 16f → 8 in primitive
      { label: "C", x: 0.241, y: 0.150, z: 0.081, role: "O-1", wyckoff: "16f" },
      { label: "C", x: 0.759, y: 0.350, z: 0.081, role: "O-2", wyckoff: "16f" },
      { label: "C", x: 0.150, y: 0.741, z: 0.169, role: "O-3", wyckoff: "16f" },
      { label: "C", x: 0.350, y: 0.259, z: 0.169, role: "O-4", wyckoff: "16f" },
      { label: "C", x: 0.759, y: 0.850, z: 0.919, role: "O-5", wyckoff: "16f" },
      { label: "C", x: 0.241, y: 0.650, z: 0.919, role: "O-6", wyckoff: "16f" },
      { label: "C", x: 0.850, y: 0.259, z: 0.831, role: "O-7", wyckoff: "16f" },
      { label: "C", x: 0.650, y: 0.741, z: 0.831, role: "O-8", wyckoff: "16f" },
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
      { label: "A", x: 0.036, y: 0.25, z: 0.838, role: "metal-4c", wyckoff: "4c" },
      // Fe2 at 8d-like (2 per f.u.)
      { label: "A", x: 0.186, y: 0.065, z: 0.334, role: "metal-8d-1", wyckoff: "8d" },
      { label: "A", x: 0.186, y: 0.435, z: 0.334, role: "metal-8d-2", wyckoff: "8d" },
      // C at 4c-like
      { label: "B", x: 0.890, y: 0.25, z: 0.443, role: "carbon", wyckoff: "4c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation", wyckoff: "1a" },
      // B at 2d: (1/3, 2/3, z) with z ≈ 0.63 — tetrahedral B (Mg, Al, Zn)
      { label: "B", x: 0.3333, y: 0.6667, z: 0.63, role: "tetra-B", wyckoff: "2d" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.37, role: "tetra-B", wyckoff: "2d" },
      // C at 2d: (1/3, 2/3, z) with z ≈ 0.24 — anion (Si, Sb, Bi, As, Sn)
      { label: "C", x: 0.3333, y: 0.6667, z: 0.24, role: "anion", wyckoff: "2d" },
      { label: "C", x: 0.6667, y: 0.3333, z: 0.76, role: "anion", wyckoff: "2d" },
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
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "cation", wyckoff: "8c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "cation", wyckoff: "8c" },
      // B (O/S/Se/Te) at 4a → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "anion", wyckoff: "4a" },
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
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "linear-coord", wyckoff: "4b" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "linear-coord", wyckoff: "4b" },
      // O at 2a → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "tetrahedral", wyckoff: "2a" },
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
      { label: "A", x: 0.3333, y: 0.6667, z: 0.75, role: "A-site", wyckoff: "2d" },
      { label: "A", x: 0.6667, y: 0.3333, z: 0.25, role: "A-site", wyckoff: "2d" },
      // B (Ni/Co/Mn/Ru) at 2a: (0, 0, 0)
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-face-share", wyckoff: "2a" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "B-face-share", wyckoff: "2a" },
      // O at 6h: (x, 2x, 1/4) with x ≈ 0.515
      { label: "C", x: 0.515, y: 0.030, z: 0.25, role: "O-1", wyckoff: "6h" },
      { label: "C", x: 0.970, y: 0.485, z: 0.25, role: "O-2", wyckoff: "6h" },
      { label: "C", x: 0.485, y: 0.515, z: 0.25, role: "O-3", wyckoff: "6h" },
      { label: "C", x: 0.485, y: 0.970, z: 0.75, role: "O-4", wyckoff: "6h" },
      { label: "C", x: 0.030, y: 0.515, z: 0.75, role: "O-5", wyckoff: "6h" },
      { label: "C", x: 0.515, y: 0.485, z: 0.75, role: "O-6", wyckoff: "6h" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.220, role: "AO-layer", wyckoff: "4e" },
      { label: "A", x: 0.0, y: 0.0, z: 0.780, role: "AO-layer", wyckoff: "4e" },
      // B (Ba/Sr) at 4e: BO layer
      { label: "B", x: 0.0, y: 0.0, z: 0.140, role: "BO-layer", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.860, role: "BO-layer", wyckoff: "4e" },
      // C (Ca) at 4e: spacer between CuO2 planes
      { label: "C", x: 0.0, y: 0.0, z: 0.045, role: "Ca-spacer", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.955, role: "Ca-spacer", wyckoff: "4e" },
      // D (Cu) at 2a + 4e: 3 CuO2 planes (1 inner + 2 outer)
      { label: "D", x: 0.0, y: 0.0, z: 0.0, role: "Cu-inner-plane", wyckoff: "2a" },
      { label: "D", x: 0.0, y: 0.0, z: 0.090, role: "Cu-outer-plane", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.910, role: "Cu-outer-plane", wyckoff: "4e" },
      // O: 10 sites in primitive (equatorial + apical + AO-layer)
      { label: "E", x: 0.5, y: 0.0, z: 0.0, role: "O-inner-eq", wyckoff: "4c" },
      { label: "E", x: 0.0, y: 0.5, z: 0.0, role: "O-inner-eq", wyckoff: "4c" },
      { label: "E", x: 0.5, y: 0.0, z: 0.090, role: "O-outer-eq", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.5, z: 0.090, role: "O-outer-eq", wyckoff: "8g" },
      { label: "E", x: 0.5, y: 0.0, z: 0.910, role: "O-outer-eq", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.5, z: 0.910, role: "O-outer-eq", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.0, z: 0.170, role: "O-apical", wyckoff: "4e" },
      { label: "E", x: 0.0, y: 0.0, z: 0.830, role: "O-apical", wyckoff: "4e" },
      { label: "E", x: 0.0, y: 0.0, z: 0.280, role: "O-AO-layer", wyckoff: "4e" },
      { label: "E", x: 0.0, y: 0.0, z: 0.720, role: "O-AO-layer", wyckoff: "4e" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.25, role: "layer-A-1", wyckoff: "2b" },
      { label: "A", x: 0.0, y: 0.0, z: 0.75, role: "layer-A-2", wyckoff: "2b" },
      // B (N in hBN) at 2c
      { label: "B", x: 0.3333, y: 0.6667, z: 0.25, role: "layer-B-1", wyckoff: "2c" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.75, role: "layer-B-2", wyckoff: "2c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation", wyckoff: "4a" },
      { label: "A", x: 0.0, y: 0.5, z: 0.25, role: "cation", wyckoff: "4a" },
      // O at 8e → 4 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.208, role: "anion", wyckoff: "8e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.792, role: "anion", wyckoff: "8e" },
      { label: "B", x: 0.0, y: 0.5, z: 0.458, role: "anion", wyckoff: "8e" },
      { label: "B", x: 0.0, y: 0.5, z: 0.042, role: "anion", wyckoff: "8e" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "intercalant", wyckoff: "3a" },
      // M (TM) at 3b: (0, 0, 1/2)
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "TM-oct", wyckoff: "3b" },
      // X (S/Se/Te) at 6c: (0, 0, z) with z ≈ 0.26
      { label: "C", x: 0.0, y: 0.0, z: 0.26, role: "chalcogen", wyckoff: "6c" },
      { label: "C", x: 0.0, y: 0.0, z: 0.74, role: "chalcogen", wyckoff: "6c" },
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
      { label: "A", x: 0.833, y: 0.666, z: 0.25, role: "majority-1", wyckoff: "6h" },
      { label: "A", x: 0.334, y: 0.167, z: 0.25, role: "majority-2", wyckoff: "6h" },
      { label: "A", x: 0.167, y: 0.833, z: 0.25, role: "majority-3", wyckoff: "6h" },
      { label: "A", x: 0.167, y: 0.334, z: 0.75, role: "majority-4", wyckoff: "6h" },
      { label: "A", x: 0.666, y: 0.833, z: 0.75, role: "majority-5", wyckoff: "6h" },
      { label: "A", x: 0.833, y: 0.167, z: 0.75, role: "majority-6", wyckoff: "6h" },
      // B at 2c
      { label: "B", x: 0.3333, y: 0.6667, z: 0.25, role: "minority-1", wyckoff: "2c" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.75, role: "minority-2", wyckoff: "2c" },
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
      { label: "A", x: 0.100, y: 0.25, z: 0.770, role: "metal-4c-1", wyckoff: "4c" },
      // Cr2 at 4c: (x, 1/4, z)
      { label: "A", x: 0.190, y: 0.25, z: 0.440, role: "metal-4c-2", wyckoff: "4c" },
      // Cr3 at 4c: (x, 1/4, z)
      { label: "A", x: 0.392, y: 0.25, z: 0.100, role: "metal-4c-3", wyckoff: "4c" },
      // C1 at 4c: (x, 1/4, z)
      { label: "B", x: 0.050, y: 0.25, z: 0.100, role: "carbon-1", wyckoff: "4c" },
      // C2 at 4c: (x, 1/4, z)
      { label: "B", x: 0.280, y: 0.25, z: 0.885, role: "carbon-2", wyckoff: "4c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cage-center", wyckoff: "4a" },
      // B at 48i → 12 in primitive (cubo-octahedral cage)
      { label: "B", x: 0.5, y: 0.178, z: 0.178, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.5, y: 0.822, z: 0.178, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.5, y: 0.178, z: 0.822, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.5, y: 0.822, z: 0.822, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.178, y: 0.5, z: 0.178, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.822, y: 0.5, z: 0.178, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.178, y: 0.5, z: 0.822, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.822, y: 0.5, z: 0.822, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.178, y: 0.178, z: 0.5, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.822, y: 0.178, z: 0.5, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.178, y: 0.822, z: 0.5, role: "B-cage", wyckoff: "48i" },
      { label: "B", x: 0.822, y: 0.822, z: 0.5, role: "B-cage", wyckoff: "48i" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Hg-layer", wyckoff: "1a" },
      // Ba at 2h: (1/2, 1/2, z) with z ≈ 0.29
      { label: "B", x: 0.5, y: 0.5, z: 0.29, role: "Ba-site", wyckoff: "2h" },
      { label: "B", x: 0.5, y: 0.5, z: 0.71, role: "Ba-site", wyckoff: "2h" },
      // Cu at 1b: (0, 0, 1/2)
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "CuO2-plane", wyckoff: "1b" },
      // O at 2g (0, 1/2, 1/2) — CuO2 in-plane
      { label: "D", x: 0.0, y: 0.5, z: 0.5, role: "O-planar", wyckoff: "2g" },
      { label: "D", x: 0.5, y: 0.0, z: 0.5, role: "O-planar", wyckoff: "2g" },
      // O at 2e (1/2, 1/2, 0) or 1c+1d — apical O
      { label: "D", x: 0.0, y: 0.0, z: 0.16, role: "O-apical", wyckoff: "2g" },
      { label: "D", x: 0.0, y: 0.0, z: 0.84, role: "O-apical", wyckoff: "2g" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "refractory-TM", wyckoff: "2a" },
      // B (Si/Ge) at 4e → 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.335, role: "Si-layer", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.665, role: "Si-layer", wyckoff: "4e" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal", wyckoff: "2a" },
      // B at 4g-like (x, x+1/2, 0)
      { label: "B", x: 0.18, y: 0.68, z: 0.0, role: "B-ring-1", wyckoff: "4g" },
      { label: "B", x: 0.32, y: 0.18, z: 0.0, role: "B-ring-2", wyckoff: "4g" },
      // B at 4h-like (x, x+1/2, 1/2)
      { label: "B", x: 0.04, y: 0.54, z: 0.5, role: "B-chain-1", wyckoff: "4h" },
      { label: "B", x: 0.46, y: 0.04, z: 0.5, role: "B-chain-2", wyckoff: "4h" },
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
      { label: "A", x: 0.137, y: 0.137, z: 0.137, role: "TM-1", wyckoff: "4a" },
      { label: "A", x: 0.637, y: 0.363, z: 0.863, role: "TM-2", wyckoff: "4a" },
      { label: "A", x: 0.363, y: 0.863, z: 0.637, role: "TM-3", wyckoff: "4a" },
      { label: "A", x: 0.863, y: 0.637, z: 0.363, role: "TM-4", wyckoff: "4a" },
      // Si at 4a: (x, x, x) and symmetry-equivalents, x ≈ 0.845
      { label: "B", x: 0.845, y: 0.845, z: 0.845, role: "Si-1", wyckoff: "4a" },
      { label: "B", x: 0.345, y: 0.655, z: 0.155, role: "Si-2", wyckoff: "4a" },
      { label: "B", x: 0.655, y: 0.155, z: 0.345, role: "Si-3", wyckoff: "4a" },
      { label: "B", x: 0.155, y: 0.345, z: 0.655, role: "Si-4", wyckoff: "4a" },
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
      { label: "A", x: 0.25, y: 0.25, z: 0.18, role: "cation-layer", wyckoff: "2c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.82, role: "cation-layer", wyckoff: "2c" },
      // B (O/S/Se) at 2a — smaller anion in cation layer
      { label: "B", x: 0.75, y: 0.25, z: 0.0, role: "inner-anion", wyckoff: "2a" },
      { label: "B", x: 0.25, y: 0.75, z: 0.0, role: "inner-anion", wyckoff: "2a" },
      // X (F/Cl/Br/I) at 2c — halide in interlayer
      { label: "C", x: 0.25, y: 0.25, z: 0.62, role: "halide-layer", wyckoff: "2c" },
      { label: "C", x: 0.75, y: 0.75, z: 0.38, role: "halide-layer", wyckoff: "2c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "RE-site", wyckoff: "1a" },
      // B at 2c: (1/3, 2/3, 0)
      { label: "B", x: 0.3333, y: 0.6667, z: 0.0, role: "TM-2c-1", wyckoff: "2c" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.0, role: "TM-2c-2", wyckoff: "2c" },
      // B at 3g: (1/2, 0, 1/2) and equivalents
      { label: "B", x: 0.5, y: 0.0, z: 0.5, role: "TM-3g-1", wyckoff: "3g" },
      { label: "B", x: 0.0, y: 0.5, z: 0.5, role: "TM-3g-2", wyckoff: "3g" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "TM-3g-3", wyckoff: "3g" },
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
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "alkali-8c", wyckoff: "8c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "alkali-8c", wyckoff: "8c" },
      // A at 4b → 1 in primitive
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "alkali-4b", wyckoff: "4b" },
      // B (Bi/Sb/As/Sn/Pb) at 4a → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "heavy-pnictogen", wyckoff: "4a" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.278, role: "A-site", wyckoff: "6a" },
      { label: "A", x: 0.0, y: 0.0, z: 0.778, role: "A-site", wyckoff: "6a" },
      // B (Nb/Ta) at 6a → 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-site", wyckoff: "6a" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "B-site", wyckoff: "6a" },
      // O at 18b → 6 in primitive
      { label: "C", x: 0.047, y: 0.343, z: 0.063, role: "O-1", wyckoff: "18b" },
      { label: "C", x: 0.657, y: 0.953, z: 0.063, role: "O-2", wyckoff: "18b" },
      { label: "C", x: 0.343, y: 0.047, z: 0.563, role: "O-3", wyckoff: "18b" },
      { label: "C", x: 0.953, y: 0.657, z: 0.563, role: "O-4", wyckoff: "18b" },
      { label: "C", x: 0.657, y: 0.704, z: 0.396, role: "O-5", wyckoff: "18b" },
      { label: "C", x: 0.296, y: 0.953, z: 0.896, role: "O-6", wyckoff: "18b" },
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
      { label: "A", x: 0.3333, y: 0.6667, z: 0.0, role: "TM-4d-1", wyckoff: "4d" },
      { label: "A", x: 0.6667, y: 0.3333, z: 0.0, role: "TM-4d-2", wyckoff: "4d" },
      // 6g: (x, 0, 1/4) with x ≈ 0.236 → 3 per f.u.
      { label: "A", x: 0.236, y: 0.0, z: 0.25, role: "TM-6g-1", wyckoff: "6g" },
      { label: "A", x: 0.0, y: 0.236, z: 0.25, role: "TM-6g-2", wyckoff: "6g" },
      { label: "A", x: 0.764, y: 0.764, z: 0.25, role: "TM-6g-3", wyckoff: "6g" },
      // B (Si/Ge/Sn/Ga) — 3 per f.u.
      // 6g: (x, 0, 1/4) with x ≈ 0.599
      { label: "B", x: 0.599, y: 0.0, z: 0.25, role: "Si-6g-1", wyckoff: "6g" },
      { label: "B", x: 0.0, y: 0.599, z: 0.25, role: "Si-6g-2", wyckoff: "6g" },
      { label: "B", x: 0.401, y: 0.401, z: 0.25, role: "Si-6g-3", wyckoff: "6g" },
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
      { label: "A", x: 0.12, y: 0.25, z: 0.10, role: "cation", wyckoff: "4c" },
      { label: "A", x: 0.62, y: 0.75, z: 0.40, role: "cation", wyckoff: "4c" },
      // B (Se/S/Te) at 4c
      { label: "B", x: 0.85, y: 0.25, z: 0.52, role: "anion", wyckoff: "4c" },
      { label: "B", x: 0.35, y: 0.75, z: 0.98, role: "anion", wyckoff: "4c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.234, role: "pnictogen-up", wyckoff: "6c" },
      { label: "A", x: 0.0, y: 0.0, z: 0.766, role: "pnictogen-down", wyckoff: "6c" },
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
      { label: "A", x: 0.3333, y: 0.6667, z: 0.245, role: "RE-layer", wyckoff: "2d" },
      { label: "A", x: 0.6667, y: 0.3333, z: 0.755, role: "RE-layer", wyckoff: "2d" },
      // O at 1a
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "O-central", wyckoff: "1a" },
      // O at 2d
      { label: "B", x: 0.3333, y: 0.6667, z: 0.645, role: "O-layer", wyckoff: "2d" },
      { label: "B", x: 0.6667, y: 0.3333, z: 0.355, role: "O-layer", wyckoff: "2d" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "tetragonal-1", wyckoff: "4a" },
      { label: "A", x: 0.0, y: 0.5, z: 0.25, role: "tetragonal-2", wyckoff: "4a" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site", wyckoff: "1a" },
      // B (Ta/Nb/Ti/V/W) at 1b
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "B-site", wyckoff: "1b" },
      // O at 3c (face centers) — 2 of 3 are O
      { label: "C", x: 0.5, y: 0.5, z: 0.0, role: "O-equatorial", wyckoff: "3c" },
      { label: "C", x: 0.5, y: 0.0, z: 0.5, role: "O-equatorial", wyckoff: "3c" },
      // X (N/F) at 3c — 1 of 3 is N or F
      { label: "D", x: 0.0, y: 0.5, z: 0.5, role: "X-anion", wyckoff: "3c" },
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
      { label: "A", x: 0.517, y: 0.25, z: 0.174, role: "cation-1", wyckoff: "4c" },
      { label: "A", x: 0.660, y: 0.25, z: 0.532, role: "cation-2", wyckoff: "4c" },
      // B (S/Se/Te) at 4c-like: 3 per f.u.
      { label: "B", x: 0.375, y: 0.25, z: 0.056, role: "anion-1", wyckoff: "4c" },
      { label: "B", x: 0.722, y: 0.25, z: 0.809, role: "anion-2", wyckoff: "4c" },
      { label: "B", x: 0.459, y: 0.25, z: 0.379, role: "anion-3", wyckoff: "4c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site", wyckoff: "1a" },
      // B (Pb/Sn/Ge) at 1b
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "B-site", wyckoff: "1b" },
      // X (I/Br/Cl/F) at 3c
      { label: "C", x: 0.5, y: 0.5, z: 0.0, role: "halide", wyckoff: "3c" },
      { label: "C", x: 0.5, y: 0.0, z: 0.5, role: "halide", wyckoff: "3c" },
      { label: "C", x: 0.0, y: 0.5, z: 0.5, role: "halide", wyckoff: "3c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Hg-layer", wyckoff: "1a" },
      // Ba at 2h (1/2, 1/2, z≈0.22)
      { label: "B", x: 0.5, y: 0.5, z: 0.22, role: "Ba-site", wyckoff: "2h" },
      { label: "B", x: 0.5, y: 0.5, z: 0.78, role: "Ba-site", wyckoff: "2h" },
      // Ca at 1b (0, 0, 1/2) — spacer between CuO2 planes
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "Ca-spacer", wyckoff: "1b" },
      // Cu at 2g (0, 0, z≈0.38) — double CuO2 plane
      { label: "D", x: 0.0, y: 0.0, z: 0.38, role: "CuO2-plane", wyckoff: "2g" },
      { label: "D", x: 0.0, y: 0.0, z: 0.62, role: "CuO2-plane", wyckoff: "2g" },
      // O: 6 total
      { label: "E", x: 0.5, y: 0.0, z: 0.38, role: "O-planar", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.5, z: 0.38, role: "O-planar", wyckoff: "4i" },
      { label: "E", x: 0.5, y: 0.0, z: 0.62, role: "O-planar", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.5, z: 0.62, role: "O-planar", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.0, z: 0.15, role: "O-apical", wyckoff: "2g" },
      { label: "E", x: 0.0, y: 0.0, z: 0.85, role: "O-apical", wyckoff: "2g" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Hg-layer", wyckoff: "1a" },
      // Ba at 2h
      { label: "B", x: 0.5, y: 0.5, z: 0.175, role: "Ba-site", wyckoff: "2h" },
      { label: "B", x: 0.5, y: 0.5, z: 0.825, role: "Ba-site", wyckoff: "2h" },
      // Ca at 2g — double spacer
      { label: "C", x: 0.0, y: 0.0, z: 0.39, role: "Ca-spacer", wyckoff: "2g" },
      { label: "C", x: 0.0, y: 0.0, z: 0.61, role: "Ca-spacer", wyckoff: "2g" },
      // Cu: 3 total (1 inner + 2 outer)
      { label: "D", x: 0.0, y: 0.0, z: 0.5, role: "Cu-inner", wyckoff: "1b" },
      { label: "D", x: 0.0, y: 0.0, z: 0.30, role: "Cu-outer", wyckoff: "2g" },
      { label: "D", x: 0.0, y: 0.0, z: 0.70, role: "Cu-outer", wyckoff: "2g" },
      // O: 8 total
      { label: "E", x: 0.5, y: 0.0, z: 0.5, role: "O-inner-eq", wyckoff: "2f" },
      { label: "E", x: 0.0, y: 0.5, z: 0.5, role: "O-inner-eq", wyckoff: "2f" },
      { label: "E", x: 0.5, y: 0.0, z: 0.30, role: "O-outer-eq", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.5, z: 0.30, role: "O-outer-eq", wyckoff: "4i" },
      { label: "E", x: 0.5, y: 0.0, z: 0.70, role: "O-outer-eq", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.5, z: 0.70, role: "O-outer-eq", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.0, z: 0.12, role: "O-apical", wyckoff: "2g" },
      { label: "E", x: 0.0, y: 0.0, z: 0.88, role: "O-apical", wyckoff: "2g" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site", wyckoff: "2a" },
      // B at 4d → 2 in primitive
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "B-basal", wyckoff: "4d" },
      { label: "B", x: 0.5, y: 0.0, z: 0.25, role: "B-basal", wyckoff: "4d" },
      // B at 4e → 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.38, role: "B-apical", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.62, role: "B-apical", wyckoff: "4e" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "site-4a", wyckoff: "4a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "site-4b", wyckoff: "4b" },
      { label: "C", x: 0.25, y: 0.25, z: 0.25, role: "site-4c", wyckoff: "4c" },
      { label: "D", x: 0.75, y: 0.75, z: 0.75, role: "site-4d", wyckoff: "4d" },
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

  // Tl-1212 cuprate: I4/mmm (139), AB2CD2O7 — single-Tl double-CuO2
  // TlBa2CaCu2O7 — Tc ≈ 108K. Single Tl-O layer + double CuO2 planes.
  // Distinct from Tl-2201 (single CuO2) and 2223 (triple, double Tl-O).
  // Body-centered → primitive half: 1Tl + 2Ba + 1Ca + 2Cu + 7O = 13
  // Ratio [1,2,1,2,7]
  {
    name: "Tl1212-AB2CD2O7",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 3.32,
    sites: [
      // Tl at 2b (0,0,1/2) → 1 in primitive
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Tl-layer", wyckoff: "2b" },
      // Ba at 4e → 2 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.15, role: "Ba-site", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.85, role: "Ba-site", wyckoff: "4e" },
      // Ca at 2a → 1 in primitive
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "Ca-spacer", wyckoff: "2a" },
      // Cu at 4e → 2 in primitive
      { label: "D", x: 0.0, y: 0.0, z: 0.37, role: "CuO2-plane", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.63, role: "CuO2-plane", wyckoff: "4e" },
      // O: 7 in primitive
      { label: "E", x: 0.5, y: 0.0, z: 0.37, role: "O-planar", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.5, z: 0.37, role: "O-planar", wyckoff: "8g" },
      { label: "E", x: 0.5, y: 0.0, z: 0.63, role: "O-planar", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.5, z: 0.63, role: "O-planar", wyckoff: "8g" },
      { label: "E", x: 0.0, y: 0.0, z: 0.22, role: "O-apical", wyckoff: "4e" },
      { label: "E", x: 0.0, y: 0.0, z: 0.78, role: "O-apical", wyckoff: "4e" },
      { label: "E", x: 0.5, y: 0.5, z: 0.0, role: "O-Tl-layer", wyckoff: "4c" },
    ],
    stoichiometryRatio: [1, 2, 1, 2, 7],
    coordination: [6, 10, 8, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 5) return false;
      const hasO = elements.includes("O");
      const hasCu = elements.includes("Cu");
      const hasTl = elements.includes("Tl");
      const hasAE = elements.some(e => ["Ba", "Sr"].includes(e));
      const hasSpacer = elements.includes("Ca") || elements.some(e => isRareEarth(e) || e === "Y");
      return hasO && hasCu && hasTl && hasAE && hasSpacer;
    },
  },

  // MAB phase: Cmcm (63), MAB — atomically layered ternary boride
  // For: MoAlB, WAlB, CrAlB, Mn2AlB2, Fe2AlB2 — emerging hard/conductive
  // Distinct from MAX phases (which have C/N, not B).
  // Orthorhombic → tetragonal approximation. 1 f.u. = 3 atoms, ratio [1,1,1]
  // M at 4c (0, y≈0.11, 1/4), A at 4c (0, y≈0.42, 1/4), B at 4c (0, y≈0.70, 1/4)
  {
    name: "MAB-phase",
    spaceGroup: "Cmcm",
    latticeType: "tetragonal",
    cOverA: 0.46,
    sites: [
      // M (Mo/W/Cr/Fe/Mn) at 4c
      { label: "A", x: 0.0, y: 0.107, z: 0.25, role: "TM-layer", wyckoff: "4c" },
      { label: "A", x: 0.0, y: 0.893, z: 0.75, role: "TM-layer", wyckoff: "4c" },
      // A-element (Al/Ga/In) at 4c
      { label: "B", x: 0.0, y: 0.416, z: 0.25, role: "A-layer", wyckoff: "4c" },
      { label: "B", x: 0.0, y: 0.584, z: 0.75, role: "A-layer", wyckoff: "4c" },
      // B (boron) at 4c
      { label: "C", x: 0.0, y: 0.703, z: 0.25, role: "B-layer", wyckoff: "4c" },
      { label: "C", x: 0.0, y: 0.297, z: 0.75, role: "B-layer", wyckoff: "4c" },
    ],
    stoichiometryRatio: [1, 1, 1],
    coordination: [6, 6, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      // MAB: TM + A-group element + boron (specifically B, not C/N like MAX)
      const hasB = elements.includes("B");
      const hasTM = elements.some(e => ["Mo", "W", "Cr", "Fe", "Mn", "V", "Nb", "Ta", "Ti"].includes(e));
      const hasA = elements.some(e => ["Al", "Ga", "In", "Si", "Ge"].includes(e));
      return hasB && hasTM && hasA;
    },
  },

  // Tl-1223 cuprate: P4/mmm (123), AB2C2D3O9 — single-Tl triple-CuO2
  // TlBa2Ca2Cu3O9 — Tc ≈ 133K. Completes the Tl-family:
  //   Tl-2201 (1 CuO2, 2 TlO), Tl-1212 (2 CuO2, 1 TlO),
  //   Tl-1223 (3 CuO2, 1 TlO), Tl-2223 (3 CuO2, 2 TlO via generic 2223)
  // Primitive: 1Tl + 2Ba + 2Ca + 3Cu + 9O = 17 atoms, ratio [1,2,2,3,9]
  {
    name: "Tl1223-AB2C2D3O9",
    spaceGroup: "P4/mmm",
    latticeType: "tetragonal",
    cOverA: 4.43,
    sites: [
      // Tl at 1a
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Tl-layer", wyckoff: "1a" },
      // Ba at 2h
      { label: "B", x: 0.5, y: 0.5, z: 0.14, role: "Ba-site", wyckoff: "2h" },
      { label: "B", x: 0.5, y: 0.5, z: 0.86, role: "Ba-site", wyckoff: "2h" },
      // Ca at 2g — double spacer
      { label: "C", x: 0.0, y: 0.0, z: 0.33, role: "Ca-spacer", wyckoff: "2g" },
      { label: "C", x: 0.0, y: 0.0, z: 0.67, role: "Ca-spacer", wyckoff: "2g" },
      // Cu: 3 (1 inner + 2 outer)
      { label: "D", x: 0.0, y: 0.0, z: 0.5, role: "Cu-inner", wyckoff: "1b" },
      { label: "D", x: 0.0, y: 0.0, z: 0.24, role: "Cu-outer", wyckoff: "2g" },
      { label: "D", x: 0.0, y: 0.0, z: 0.76, role: "Cu-outer", wyckoff: "2g" },
      // O: 9 total
      { label: "E", x: 0.5, y: 0.0, z: 0.5, role: "O-inner-eq", wyckoff: "2f" },
      { label: "E", x: 0.0, y: 0.5, z: 0.5, role: "O-inner-eq", wyckoff: "2f" },
      { label: "E", x: 0.5, y: 0.0, z: 0.24, role: "O-outer-eq", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.5, z: 0.24, role: "O-outer-eq", wyckoff: "4i" },
      { label: "E", x: 0.5, y: 0.0, z: 0.76, role: "O-outer-eq", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.5, z: 0.76, role: "O-outer-eq", wyckoff: "4i" },
      { label: "E", x: 0.0, y: 0.0, z: 0.18, role: "O-apical", wyckoff: "2g" },
      { label: "E", x: 0.0, y: 0.0, z: 0.82, role: "O-apical", wyckoff: "2g" },
      { label: "E", x: 0.5, y: 0.5, z: 0.0, role: "O-Tl-layer", wyckoff: "2e" },
    ],
    stoichiometryRatio: [1, 2, 2, 3, 9],
    coordination: [6, 10, 8, 5, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 5) return false;
      const hasO = elements.includes("O");
      const hasCu = elements.includes("Cu");
      const hasTl = elements.includes("Tl");
      const hasAE = elements.some(e => ["Ba", "Sr"].includes(e));
      const hasSpacer = elements.includes("Ca") || elements.some(e => isRareEarth(e) || e === "Y");
      return hasO && hasCu && hasTl && hasAE && hasSpacer;
    },
  },

  // Tetradymite: R-3m (166), A2B2C — ordered quintuple-layer topological insulator
  // For: Bi2Te2Se, Bi2Te2S, Bi2Se2Te, Sb2Te2Se — topological insulators
  // Same quintuple-layer as Bi2Te3 but with 3 elements (ordered chalcogen sites).
  // CRITICAL: Bi2Te2Se (most-studied TI) had ZERO match — 3-element [2,2,1]
  // doesn't fit 2-element Bi2Te3 template or any ternary [1,2,2] templates.
  // Primitive: 2A + 2B + 1C = 5 atoms, ratio [2,2,1]
  {
    name: "Tetradymite-A2B2C",
    spaceGroup: "R-3m",
    latticeType: "hexagonal",
    cOverA: 6.96,
    sites: [
      // A (Bi/Sb) at 6c: outer layer of quintuple
      { label: "A", x: 0.0, y: 0.0, z: 0.400, role: "pnictogen-outer", wyckoff: "6c" },
      { label: "A", x: 0.0, y: 0.0, z: 0.600, role: "pnictogen-outer", wyckoff: "6c" },
      // B (Te) at 6c: inner chalcogen layer
      { label: "B", x: 0.0, y: 0.0, z: 0.212, role: "chalcogen-inner", wyckoff: "6c" },
      { label: "B", x: 0.0, y: 0.0, z: 0.788, role: "chalcogen-inner", wyckoff: "6c" },
      // C (Se/S) at 3a: central layer of quintuple
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "chalcogen-center", wyckoff: "3a" },
    ],
    stoichiometryRatio: [2, 2, 1],
    coordination: [6, 3, 6],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      // Tetradymite: heavy pnictogen + 2 different chalcogens
      const hasPn = elements.some(e => ["Bi", "Sb"].includes(e));
      const chalcogens = elements.filter(e => ["S", "Se", "Te"].includes(e));
      return hasPn && chalcogens.length >= 2;
    },
  },

  // Aurivillius n=2: I4/mmm-like, AB2C2O9 — FeRAM ferroelectric
  // Bi2O2 layers + double perovskite block. SrBi2Ta2O9 (SBT) — THE FeRAM material.
  // For: SrBi2Ta2O9, SrBi2Nb2O9, BaBi2Ta2O9, CaBi2Nb2O9
  // Primitive: 1A + 2Bi + 2B + 9O = 14 atoms, ratio [1,2,2,9]
  {
    name: "Aurivillius-n2-AB2C2O9",
    spaceGroup: "I4/mmm",
    latticeType: "tetragonal",
    cOverA: 4.55,
    sites: [
      // A (Sr/Ba/Ca) at 2a: center of perovskite block
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-perovskite", wyckoff: "2a" },
      // Bi at 4e: Bi2O2 layer
      { label: "B", x: 0.0, y: 0.0, z: 0.28, role: "Bi2O2-layer", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.72, role: "Bi2O2-layer", wyckoff: "4e" },
      // C (Ta/Nb) at 4e: B-site of double perovskite
      { label: "C", x: 0.0, y: 0.0, z: 0.10, role: "B-perovskite", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.90, role: "B-perovskite", wyckoff: "4e" },
      // O: 9 per primitive
      { label: "D", x: 0.5, y: 0.0, z: 0.10, role: "O-eq-1", wyckoff: "8g" },
      { label: "D", x: 0.0, y: 0.5, z: 0.10, role: "O-eq-2", wyckoff: "8g" },
      { label: "D", x: 0.5, y: 0.0, z: 0.90, role: "O-eq-3", wyckoff: "8g" },
      { label: "D", x: 0.0, y: 0.5, z: 0.90, role: "O-eq-4", wyckoff: "8g" },
      { label: "D", x: 0.0, y: 0.0, z: 0.19, role: "O-apical", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.81, role: "O-apical", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.5, role: "O-bridging", wyckoff: "2b" },
      { label: "D", x: 0.0, y: 0.0, z: 0.36, role: "O-BiO", wyckoff: "4e" },
      { label: "D", x: 0.0, y: 0.0, z: 0.64, role: "O-BiO", wyckoff: "4e" },
    ],
    stoichiometryRatio: [1, 2, 2, 9],
    coordination: [12, 8, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasBi = elements.includes("Bi");
      const hasAE = elements.some(e => ["Sr", "Ba", "Ca", "Pb"].includes(e));
      const hasB = elements.some(e => ["Ta", "Nb", "Ti", "W", "Mo", "V"].includes(e));
      return hasO && hasBi && hasAE && hasB;
    },
  },

  // ══════════════════════════════════════════════════════════════════════
  // ── Monoclinic prototype templates (Phase 2) ─────────────────────────
  // ══════════════════════════════════════════════════════════════════════

  // Wolframite: P2/c (13), ABO4 — monoclinic tungstate/molybdate
  // For: MnWO4, FeWO4, NiWO4, CoWO4, MnMoO4 — multiferroics, photocatalysts
  // Distinct from Scheelite (I41/a, tetragonal): wolframite forms when
  // A-site cation is smaller (Mn, Fe, Co, Ni vs Ca, Sr, Ba in scheelite).
  // Primitive cell: 2 formula units → 2A + 2B + 8O = 12 atoms, ratio [1,1,4]
  // beta ≈ 91° (nearly orthorhombic but genuinely monoclinic)
  {
    name: "Wolframite-ABO4",
    spaceGroup: "P2/c",
    latticeType: "monoclinic",
    cOverA: 1.033,
    bOverA: 1.192,
    beta: 91.1,
    sites: [
      // A (Mn/Fe/Co/Ni) at 2f: (1/2, y, 1/4)
      { label: "A", x: 0.5, y: 0.685, z: 0.25, role: "A-site", wyckoff: "2f" },
      { label: "A", x: 0.5, y: 0.315, z: 0.75, role: "A-site", wyckoff: "2f" },
      // B (W/Mo) at 2e: (0, y, 1/4)
      { label: "B", x: 0.0, y: 0.180, z: 0.25, role: "B-site", wyckoff: "2e" },
      { label: "B", x: 0.0, y: 0.820, z: 0.75, role: "B-site", wyckoff: "2e" },
      // O at 4g: general position (x, y, z) — 4 per f.u., 8 total
      { label: "C", x: 0.222, y: 0.106, z: 0.075, role: "O-1", wyckoff: "4g" },
      { label: "C", x: 0.778, y: 0.894, z: 0.925, role: "O-1", wyckoff: "4g" },
      { label: "C", x: 0.778, y: 0.106, z: 0.425, role: "O-1", wyckoff: "4g" },
      { label: "C", x: 0.222, y: 0.894, z: 0.575, role: "O-1", wyckoff: "4g" },
      { label: "C", x: 0.259, y: 0.379, z: 0.395, role: "O-2", wyckoff: "4g" },
      { label: "C", x: 0.741, y: 0.621, z: 0.605, role: "O-2", wyckoff: "4g" },
      { label: "C", x: 0.741, y: 0.379, z: 0.105, role: "O-2", wyckoff: "4g" },
      { label: "C", x: 0.259, y: 0.621, z: 0.895, role: "O-2", wyckoff: "4g" },
    ],
    stoichiometryRatio: [1, 1, 4],
    coordination: [6, 6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      // A-site: smaller divalent TM (not in Scheelite's A-site list)
      const hasA = elements.some(e => ["Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mg"].includes(e));
      // B-site: W or Mo (same as Scheelite)
      const hasB = elements.some(e => ["W", "Mo"].includes(e));
      return hasO && hasA && hasB;
    },
  },

  // Baddeleyite: P21/c (14), AO2 — monoclinic dioxide
  // For: ZrO2, HfO2, CeO2(mono), PuO2 — structural ceramics, gate dielectrics
  // Distinct from Rutile (tetragonal) and Fluorite (cubic): 7-fold A coordination.
  // Primitive cell: 4 formula units → 4A + 8O = 12 atoms, ratio [1,2]
  // beta ≈ 99.2°
  {
    name: "Baddeleyite-AO2",
    spaceGroup: "P21/c",
    latticeType: "monoclinic",
    cOverA: 1.031,
    bOverA: 1.012,
    beta: 99.2,
    sites: [
      // A (Zr/Hf) at 4e: (x, y, z) — 4 in primitive
      { label: "A", x: 0.276, y: 0.040, z: 0.208, role: "cation", wyckoff: "4e" },
      { label: "A", x: 0.724, y: 0.960, z: 0.792, role: "cation", wyckoff: "4e" },
      { label: "A", x: 0.724, y: 0.540, z: 0.292, role: "cation", wyckoff: "4e" },
      { label: "A", x: 0.276, y: 0.460, z: 0.708, role: "cation", wyckoff: "4e" },
      // O1 at 4e — 4 atoms
      { label: "B", x: 0.070, y: 0.342, z: 0.341, role: "O-type1", wyckoff: "4e" },
      { label: "B", x: 0.930, y: 0.658, z: 0.659, role: "O-type1", wyckoff: "4e" },
      { label: "B", x: 0.930, y: 0.842, z: 0.159, role: "O-type1", wyckoff: "4e" },
      { label: "B", x: 0.070, y: 0.158, z: 0.841, role: "O-type1", wyckoff: "4e" },
      // O2 at 4e — 4 atoms
      { label: "B", x: 0.442, y: 0.758, z: 0.479, role: "O-type2", wyckoff: "4e" },
      { label: "B", x: 0.558, y: 0.242, z: 0.521, role: "O-type2", wyckoff: "4e" },
      { label: "B", x: 0.558, y: 0.258, z: 0.021, role: "O-type2", wyckoff: "4e" },
      { label: "B", x: 0.442, y: 0.742, z: 0.979, role: "O-type2", wyckoff: "4e" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [7, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasO = elements.includes("O");
      // Baddeleyite: Zr, Hf, Ce, Pu, U + O (monoclinic at ambient pressure)
      const hasA = elements.some(e => ["Zr", "Hf", "Ce", "Pu", "U", "Th"].includes(e));
      return hasO && hasA;
    },
  },

  // VO2 monoclinic: P21/c (14), M1 phase — Mott insulator
  // For: VO2 — metal-insulator transition at 68°C, smart windows, neuromorphic
  // Distinct from Rutile (high-T metallic phase): V-V dimerization + tilting.
  // Primitive cell: 4 V + 8 O = 12 atoms, ratio [1,2]
  // beta ≈ 122.6° (strongly monoclinic)
  {
    name: "VO2-monoclinic",
    spaceGroup: "P21/c",
    latticeType: "monoclinic",
    cOverA: 1.006,
    bOverA: 0.793,
    beta: 122.6,
    sites: [
      // V at 4e: dimerized pairs along a-axis
      { label: "A", x: 0.242, y: 0.975, z: 0.025, role: "V-dimer", wyckoff: "4e" },
      { label: "A", x: 0.758, y: 0.025, z: 0.975, role: "V-dimer", wyckoff: "4e" },
      { label: "A", x: 0.758, y: 0.525, z: 0.475, role: "V-dimer", wyckoff: "4e" },
      { label: "A", x: 0.242, y: 0.475, z: 0.525, role: "V-dimer", wyckoff: "4e" },
      // O1 at 4e
      { label: "B", x: 0.100, y: 0.210, z: 0.200, role: "O-1", wyckoff: "4e" },
      { label: "B", x: 0.900, y: 0.790, z: 0.800, role: "O-1", wyckoff: "4e" },
      { label: "B", x: 0.900, y: 0.710, z: 0.300, role: "O-1", wyckoff: "4e" },
      { label: "B", x: 0.100, y: 0.290, z: 0.700, role: "O-1", wyckoff: "4e" },
      // O2 at 4e
      { label: "B", x: 0.390, y: 0.690, z: 0.290, role: "O-2", wyckoff: "4e" },
      { label: "B", x: 0.610, y: 0.310, z: 0.710, role: "O-2", wyckoff: "4e" },
      { label: "B", x: 0.610, y: 0.810, z: 0.210, role: "O-2", wyckoff: "4e" },
      { label: "B", x: 0.390, y: 0.190, z: 0.790, role: "O-2", wyckoff: "4e" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      const hasO = elements.includes("O");
      // VO2 specifically: V + O (other TM dioxides use rutile or fluorite)
      const hasV = elements.includes("V");
      return hasO && hasV;
    },
  },

  // Monazite: P21/n (14), ABO4 — rare-earth phosphate/arsenate
  // For: LaPO4, CePO4, NdPO4, LaAsO4 — nuclear waste storage, phosphors, catalysts
  // Distinct from Scheelite (needs Ca/Sr/Ba) and Wolframite (needs W/Mo).
  // Monazite forms with larger RE + P/As.
  // Primitive cell: 4 formula units → 4A + 4B + 16O = 24 atoms, ratio [1,1,4]
  // beta ≈ 103.2°. Simplified to 1 f.u.: 1A + 1B + 4O = 6 atoms.
  {
    name: "Monazite-ABO4",
    spaceGroup: "P21/n",
    latticeType: "monoclinic",
    cOverA: 0.941,
    bOverA: 1.024,
    beta: 103.2,
    sites: [
      // A (La/Ce/Nd) at 4e — 1 per f.u.
      { label: "A", x: 0.282, y: 0.158, z: 0.100, role: "RE-site", wyckoff: "4e" },
      // B (P/As) at 4e — 1 per f.u.
      { label: "B", x: 0.305, y: 0.163, z: 0.612, role: "P-site", wyckoff: "4e" },
      // O at 4e — 4 per f.u.
      { label: "C", x: 0.250, y: 0.007, z: 0.447, role: "O-1", wyckoff: "4e" },
      { label: "C", x: 0.381, y: 0.338, z: 0.497, role: "O-2", wyckoff: "4e" },
      { label: "C", x: 0.474, y: 0.105, z: 0.811, role: "O-3", wyckoff: "4e" },
      { label: "C", x: 0.127, y: 0.211, z: 0.710, role: "O-4", wyckoff: "4e" },
    ],
    stoichiometryRatio: [1, 1, 4],
    coordination: [9, 4, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      // A-site: rare-earth (La, Ce, Nd, Pr, Sm, Gd, Y, Bi)
      const hasRE = elements.some(e => isRareEarth(e) || ["Y", "Bi", "Th", "U"].includes(e));
      // B-site: P or As (phosphate/arsenate, not W/Mo like wolframite)
      const hasP = elements.some(e => ["P", "As"].includes(e));
      return hasO && hasRE && hasP;
    },
  },

  // ══════════════════════════════════════════════════════════════════════
  // ── Ratio-gap fill templates ─────────────────────────────────────────
  // ══════════════════════════════════════════════════════════════════════

  // ── TERNARY [1,1,6]: Vacancy-ordered perovskite A2BX6 ──
  // Fm-3m (225). Cs2SnI6, K2PtCl6, Rb2SnBr6 — lead-free solar absorbers
  // B-site vacancy-ordered double perovskite. A at 8c, B at 4a, X at 24e.
  // Primitive (FCC 1/4): 2A + 1B + 6X = 9 atoms, ratio [2,1,6] BUT
  // stoichiometry sorted = [6,2,1]. Using [1,1,6] reduced form for A2BX6.
  // Actually A₂BX₆ has reduced = [2,1,6] which sorted matches.
  {
    name: "VacancyPerovskite-A2BX6",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A (Cs/K/Rb) at 8c → 2 in primitive
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "A-site", wyckoff: "8c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "A-site", wyckoff: "8c" },
      // B (Sn/Pt/Ti/Te) at 4a → 1 in primitive
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-site", wyckoff: "4a" },
      // X (I/Cl/Br) at 24e → 6 in primitive
      { label: "C", x: 0.24, y: 0.0, z: 0.0, role: "halide", wyckoff: "24e" },
      { label: "C", x: 0.76, y: 0.0, z: 0.0, role: "halide", wyckoff: "24e" },
      { label: "C", x: 0.0, y: 0.24, z: 0.0, role: "halide", wyckoff: "24e" },
      { label: "C", x: 0.0, y: 0.76, z: 0.0, role: "halide", wyckoff: "24e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.24, role: "halide", wyckoff: "24e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.76, role: "halide", wyckoff: "24e" },
    ],
    stoichiometryRatio: [2, 1, 6],
    coordination: [12, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasA = elements.some(e => ["Cs", "Rb", "K", "Na", "Tl"].includes(e));
      const hasB = elements.some(e => ["Sn", "Te", "Pt", "Pd", "Ti", "Zr", "Hf", "Se", "W", "Re"].includes(e));
      const hasX = elements.some(e => ["I", "Br", "Cl", "F"].includes(e));
      return hasA && hasB && hasX;
    },
  },

  // ── TERNARY [1,2,6]: Columbite AB2O6 ──
  // Pbcn (60) → orthorhombic. FeTa2O6, MnNb2O6, FeNb2O6 — capacitor dielectrics
  // A at 4c, B at 8d, O at 8d×3. Per f.u.: 1A + 2B + 6O = 9 atoms.
  {
    name: "Columbite-AB2O6",
    spaceGroup: "Pbcn",
    latticeType: "orthorhombic",
    cOverA: 0.36,
    bOverA: 1.16,
    sites: [
      // A (Fe/Mn/Co/Ni) — 1 per f.u.
      { label: "A", x: 0.0, y: 0.16, z: 0.25, role: "A-oct", wyckoff: "4c" },
      // B (Ta/Nb) — 2 per f.u.
      { label: "B", x: 0.16, y: 0.33, z: 0.75, role: "B-oct-1", wyckoff: "8d" },
      { label: "B", x: 0.34, y: 0.33, z: 0.25, role: "B-oct-2", wyckoff: "8d" },
      // O — 6 per f.u.
      { label: "C", x: 0.10, y: 0.10, z: 0.07, role: "O-1", wyckoff: "8d" },
      { label: "C", x: 0.40, y: 0.10, z: 0.43, role: "O-2", wyckoff: "8d" },
      { label: "C", x: 0.25, y: 0.38, z: 0.08, role: "O-3", wyckoff: "8d" },
      { label: "C", x: 0.10, y: 0.42, z: 0.43, role: "O-4", wyckoff: "8d" },
      { label: "C", x: 0.40, y: 0.42, z: 0.07, role: "O-5", wyckoff: "8d" },
      { label: "C", x: 0.25, y: 0.12, z: 0.92, role: "O-6", wyckoff: "8d" },
    ],
    stoichiometryRatio: [1, 2, 6],
    coordination: [6, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      const hasA = elements.some(e => ["Fe", "Mn", "Co", "Ni", "Mg", "Zn", "Cu"].includes(e));
      const hasB = elements.some(e => ["Ta", "Nb", "Sb", "Ti"].includes(e));
      return hasO && hasA && hasB;
    },
  },

  // ── TERNARY [1,3,4]: Normal spinel AB3O4 (distinct cation ratio) ──
  // Fd-3m (227). CoCr2O4, FeCr2O4, NiCr2O4 — but written as ACr₂O₄
  // where A:Cr:O = 1:2:4. This is ALREADY covered by Spinel-AB2X4 [1,2,4].
  // However, some spinels like Fe₃O₄ = Fe[Fe₂]O₄ are written as A₃O₄ → [3,4].
  // For [1,3,4] specifically: no common 3-element compound has this exact ratio.
  // Instead, add the [3,4] binary magnetite variant.
  // NOTE: Re-mapped as MagnetiteFe3O4 binary [3,4].
  {
    name: "NormalSpinel-AB3O4",
    spaceGroup: "Fd-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A at 8a (tetrahedral) → 1 in primitive (from conv/8)
      { label: "A", x: 0.125, y: 0.125, z: 0.125, role: "tetra-A", wyckoff: "8a" },
      // B at 16d (octahedral) → 2 in primitive
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "octa-B1", wyckoff: "16d" },
      { label: "B", x: 0.5, y: 0.25, z: 0.25, role: "octa-B2", wyckoff: "16d" },
      // B at 8a (additional) → 1 in primitive (for A₁B₃ = 1+3 metals)
      { label: "B", x: 0.875, y: 0.875, z: 0.875, role: "tetra-B", wyckoff: "8a" },
      // O at 32e → 4 in primitive
      { label: "C", x: 0.263, y: 0.263, z: 0.263, role: "O-1", wyckoff: "32e" },
      { label: "C", x: 0.263, y: 0.237, z: 0.737, role: "O-2", wyckoff: "32e" },
      { label: "C", x: 0.737, y: 0.263, z: 0.737, role: "O-3", wyckoff: "32e" },
      { label: "C", x: 0.737, y: 0.737, z: 0.263, role: "O-4", wyckoff: "32e" },
    ],
    stoichiometryRatio: [1, 3, 4],
    coordination: [4, 6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      const tmCount = elements.filter(e => isTransitionMetal(e)).length;
      return hasO && tmCount >= 2;
    },
  },

  // ── TERNARY [3,1,4] / binary [3,2]: Anti-bixbyite A3N2 ──
  // Ia-3 (206). Mg3N2, Ca3N2, Be3N2, Zn3N2 — III-V nitride semiconductors
  // Actually Mg3N2 is Ia-3 with 40 atoms conventional. Too large.
  // Use simplified 5-atom representation: 3A + 2N = 5, ratio [3,2].
  // Note: [3,2] is already covered by Sesquicarbide. Let's use the ternary
  // ratio [3,1,4] for a genuine ternary: Li3AlN2 (anti-fluorite nitride).
  {
    name: "Antibixbyite-A3N2",
    spaceGroup: "Ia-3",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // Simplified: A₃BN₄ with A at 24d, B at 8a, N at 32e-like
      // Using Li₃AlN₂ positions → 3 Li + 1 Al + 2 N per formula
      // Primitive (BCC 1/2): reduced representation
      { label: "A", x: 0.25, y: 0.0, z: 0.25, role: "Li-1", wyckoff: "24d" },
      { label: "A", x: 0.0, y: 0.25, z: 0.25, role: "Li-2", wyckoff: "24d" },
      { label: "A", x: 0.25, y: 0.25, z: 0.0, role: "Li-3", wyckoff: "24d" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "Al-site", wyckoff: "8a" },
      { label: "C", x: 0.38, y: 0.38, z: 0.38, role: "N-1", wyckoff: "32e" },
      { label: "C", x: 0.62, y: 0.62, z: 0.62, role: "N-2", wyckoff: "32e" },
      { label: "C", x: 0.12, y: 0.12, z: 0.12, role: "N-3", wyckoff: "32e" },
      { label: "C", x: 0.88, y: 0.88, z: 0.88, role: "N-4", wyckoff: "32e" },
    ],
    stoichiometryRatio: [3, 1, 4],
    coordination: [4, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasN = elements.includes("N");
      const hasAlkali = elements.some(e => ["Li", "Na", "K", "Mg", "Ca"].includes(e));
      const hasTMorSP = elements.some(e => ["Al", "Ga", "In", "Si", "B", "Ti", "Fe"].includes(e));
      return hasN && hasAlkali && hasTMorSP;
    },
  },

  // ── TERNARY [1,3,3]: Filled skutterudite variant A1B3C3 ──
  // Im-3 (204). Reinterpretation of filled skutterudite RT4X12 for
  // compositions like LaRu3As3 (reduced from LaRu₃As₃).
  // Actually this is ratio [1,3,3] which is distinct from [1,4,12].
  {
    name: "CoSb3-AB3C3",
    spaceGroup: "Im-3",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A (La/Ce/filler) at cage center
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "filler", wyckoff: "2a" },
      // B (Ru/Co/TM) at octahedral
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "TM-1", wyckoff: "8c" },
      { label: "B", x: 0.75, y: 0.25, z: 0.25, role: "TM-2", wyckoff: "8c" },
      { label: "B", x: 0.25, y: 0.75, z: 0.25, role: "TM-3", wyckoff: "8c" },
      // C (As/Sb/P) pnictogen ring
      { label: "C", x: 0.0, y: 0.34, z: 0.16, role: "Pn-1", wyckoff: "24g" },
      { label: "C", x: 0.34, y: 0.16, z: 0.0, role: "Pn-2", wyckoff: "24g" },
      { label: "C", x: 0.16, y: 0.0, z: 0.34, role: "Pn-3", wyckoff: "24g" },
    ],
    stoichiometryRatio: [1, 3, 3],
    coordination: [12, 6, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasRE = elements.some(e => isRareEarth(e) || ["Y", "Sc", "Ca", "Ba", "Sr", "Th"].includes(e));
      const hasTM = elements.some(e => ["Co", "Rh", "Ir", "Ru", "Fe", "Os", "Ni"].includes(e));
      const hasPn = elements.some(e => ["Sb", "As", "P", "Bi", "Se", "Te"].includes(e));
      return hasRE && hasTM && hasPn;
    },
  },

  // ── TERNARY [1,2,3] / [2,3] variant: Inverse spinel ternary ──
  // Fd-3m (227). For compositions like MgAl2O4 written differently or
  // Al2MgO4 with ratio [1,2,4] already covered. But [1,2,3] covers
  // e.g. ABe2B3 intermetallics or A(B2C3) chalcogenides like In2S3.
  // In2S3 (beta): I41/amd, ratio [2,3]. Already covered by [2,3] Corundum.
  // Genuine [1,2,3]: e.g., CaAl2O4 (grossite) — 1:2:4 not 1:2:3.
  // Let's do Li2TiO3 which is [2,1,3] = sorted [3,2,1].
  {
    name: "InverseSpinelTernary-A2B3",
    spaceGroup: "C2/c",
    latticeType: "monoclinic",
    cOverA: 0.96,
    bOverA: 1.68,
    beta: 100.1,
    sites: [
      // Li2TiO3: A=Li, B=Ti, C=O; ratio [2,1,3]
      { label: "A", x: 0.0, y: 0.08, z: 0.25, role: "Li-1", wyckoff: "4e" },
      { label: "A", x: 0.0, y: 0.42, z: 0.25, role: "Li-2", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.75, z: 0.25, role: "Ti-site", wyckoff: "4e" },
      { label: "C", x: 0.14, y: 0.25, z: 0.43, role: "O-1", wyckoff: "8f" },
      { label: "C", x: 0.24, y: 0.58, z: 0.38, role: "O-2", wyckoff: "8f" },
      { label: "C", x: 0.24, y: 0.92, z: 0.38, role: "O-3", wyckoff: "8f" },
    ],
    stoichiometryRatio: [2, 1, 3],
    coordination: [4, 6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasO = elements.includes("O");
      const hasAlkali = elements.some(e => ["Li", "Na", "K", "Mg", "Ca", "Sr"].includes(e));
      const hasTM = elements.some(e => isTransitionMetal(e) || ["Ti", "Zr", "Sn", "Mn", "Fe"].includes(e));
      return hasO && hasAlkali && hasTM;
    },
  },

  // ── QUATERNARY [1,1,1,3]: Ordered perovskite ABCO3 ──
  // Pm-3m or Pnma. For NaLaCoO3, KLaMnO3 — A-site ordered perovskites
  // where two distinct A-site cations alternate layers.
  // 1A + 1A' + 1B + 3O = 6 atoms, ratio [1,1,1,3]
  {
    name: "OrderedPerovskite-ABCO3",
    spaceGroup: "Pm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A (Na/K/Li) at corner
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "A-site", wyckoff: "1a" },
      // A' (La/Sr/Ba) at body-center-like
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "A'-site", wyckoff: "3c" },
      // B (Co/Mn/Fe) at center
      { label: "C", x: 0.5, y: 0.5, z: 0.5, role: "B-site", wyckoff: "1b" },
      // O at face centers
      { label: "D", x: 0.5, y: 0.0, z: 0.5, role: "O-1", wyckoff: "3c" },
      { label: "D", x: 0.0, y: 0.5, z: 0.5, role: "O-2", wyckoff: "3c" },
      { label: "D", x: 0.5, y: 0.5, z: 0.0, role: "O-3", wyckoff: "3c" },
    ],
    stoichiometryRatio: [1, 1, 1, 3],
    coordination: [12, 12, 6, 2],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasSmallA = elements.some(e => ["Li", "Na", "K", "Ag", "Cu"].includes(e));
      const hasLargeA = elements.some(e => CATIONS_LARGE.has(e));
      const hasTM = elements.some(e => isTransitionMetal(e));
      return hasO && hasSmallA && hasLargeA && hasTM;
    },
  },

  // ── QUATERNARY [1,1,1,4]: Quaternary olivine ABCO4 (LiFePO4 as 4-elem) ──
  // Pnma (62). LiFePO4, LiMnPO4, LiCoPO4, NaFePO4 — battery cathodes
  // When formula parsed as 4 elements: Li+Fe+P+O = [1,1,1,4]
  {
    name: "QuaternaryOlivine-ABCO4",
    spaceGroup: "Pnma",
    latticeType: "tetragonal",
    cOverA: 0.47,
    bOverA: 1.94,
    sites: [
      // A (Li/Na) at M1 octahedral
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "alkali-M1", wyckoff: "4a" },
      // B (Fe/Mn/Co) at M2 octahedral
      { label: "B", x: 0.28, y: 0.25, z: 0.97, role: "TM-M2", wyckoff: "4c" },
      // C (P/As) at tetrahedral
      { label: "C", x: 0.09, y: 0.25, z: 0.42, role: "P-tetra", wyckoff: "4c" },
      // O — 4 per f.u.
      { label: "D", x: 0.10, y: 0.25, z: 0.74, role: "O-1", wyckoff: "4c" },
      { label: "D", x: 0.45, y: 0.25, z: 0.22, role: "O-2", wyckoff: "4c" },
      { label: "D", x: 0.16, y: 0.04, z: 0.28, role: "O-3", wyckoff: "8d" },
      { label: "D", x: 0.16, y: 0.46, z: 0.28, role: "O-4", wyckoff: "8d" },
    ],
    stoichiometryRatio: [1, 1, 1, 4],
    coordination: [6, 6, 4, 3],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasAlkali = elements.some(e => ["Li", "Na", "K"].includes(e));
      const hasTM = elements.some(e => ["Fe", "Mn", "Co", "Ni", "V", "Cr", "Ti"].includes(e));
      const hasP = elements.some(e => ["P", "As", "Si", "Ge", "S"].includes(e));
      return hasO && hasAlkali && hasTM && hasP;
    },
  },

  // ── QUATERNARY [1,1,2,4]: Ordered spinel AB2CO4 (LiMn2O4-type) ──
  // Fd-3m (227). LiMn2O4, LiCo2O4, LiFe2O4 — Li-ion battery cathodes
  // 4-element: Li+Mn+Mn+O or Li(1) + Mn(2) + O(4) but with distinct ordering.
  // When another element like Ni partially substitutes: LiNiMnO4 = [1,1,1,4].
  // For LiMn2O4 as 3-element [1,2,4] → already covered by Spinel-AB2X4.
  // This covers 4-element ordered spinels like LiNi0.5Mn1.5O4.
  {
    name: "OrderedSpinel-AB2CO4",
    spaceGroup: "Fd-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // A (Li/Na) at 8a tetrahedral → 1 in primitive
      { label: "A", x: 0.125, y: 0.125, z: 0.125, role: "tetra-A", wyckoff: "8a" },
      // B (Mn/Co/Ni) at 16d octahedral → 2 in primitive
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "octa-B1", wyckoff: "16d" },
      { label: "B", x: 0.5, y: 0.25, z: 0.25, role: "octa-B2", wyckoff: "16d" },
      // C (different TM at 16d or 8a) → 1 in primitive
      { label: "C", x: 0.875, y: 0.875, z: 0.875, role: "octa-C", wyckoff: "8a" },
      // O at 32e → 4 in primitive
      { label: "D", x: 0.263, y: 0.263, z: 0.263, role: "O-1", wyckoff: "32e" },
      { label: "D", x: 0.263, y: 0.237, z: 0.737, role: "O-2", wyckoff: "32e" },
      { label: "D", x: 0.737, y: 0.263, z: 0.737, role: "O-3", wyckoff: "32e" },
      { label: "D", x: 0.737, y: 0.737, z: 0.263, role: "O-4", wyckoff: "32e" },
    ],
    stoichiometryRatio: [1, 2, 1, 4],
    coordination: [4, 6, 6, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasO = elements.includes("O");
      const hasAlkali = elements.some(e => ["Li", "Na", "K", "Mg", "Ca"].includes(e));
      const tmCount = elements.filter(e => isTransitionMetal(e)).length;
      return hasO && hasAlkali && tmCount >= 2;
    },
  },

  // ── QUATERNARY [1,1,1,2]: Quaternary chalcogenide ABCX2 ──
  // P-1 or P4/nmm. CuLaOS, AgBiSe2, CuBiSSe — thermoelectrics, TI
  // Covers 4-element chalcogenides with 1:1:1:2 stoichiometry.
  {
    name: "QuaternaryChalc-ABCX2",
    spaceGroup: "P4/nmm",
    latticeType: "tetragonal",
    cOverA: 2.2,
    sites: [
      // A (Cu/Ag) at 2a
      { label: "A", x: 0.75, y: 0.25, z: 0.0, role: "d-metal", wyckoff: "2a" },
      // B (La/Bi/Sb) at 2c
      { label: "B", x: 0.25, y: 0.25, z: 0.14, role: "heavy-cation", wyckoff: "2c" },
      // C (O/S/Se — minority anion) at 2a
      { label: "C", x: 0.75, y: 0.25, z: 0.5, role: "anion-1", wyckoff: "2a" },
      // X (S/Se/Te — majority anion) at 2c
      { label: "D", x: 0.25, y: 0.25, z: 0.65, role: "anion-2", wyckoff: "2c" },
      { label: "D", x: 0.75, y: 0.75, z: 0.35, role: "anion-2", wyckoff: "2c" },
    ],
    stoichiometryRatio: [1, 1, 1, 2],
    coordination: [4, 8, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 4) return false;
      const hasCuAg = elements.some(e => ["Cu", "Ag"].includes(e));
      const hasHeavy = elements.some(e => isRareEarth(e) || ["Bi", "Sb", "Pb", "In", "Y"].includes(e));
      const anionCount = elements.filter(e => ["O", "S", "Se", "Te", "F", "Cl"].includes(e)).length;
      return hasCuAg && hasHeavy && anionCount >= 1;
    },
  },

  // ── End of additional families ─────────────────────────────────────────
  {
    name: "Heusler-L21",
    spaceGroup: "Fm-3m",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "main-metal", wyckoff: "8c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.75, role: "main-metal", wyckoff: "8c" },
      { label: "A", x: 0.25, y: 0.75, z: 0.75, role: "main-metal", wyckoff: "8c" },
      { label: "A", x: 0.75, y: 0.25, z: 0.25, role: "main-metal", wyckoff: "8c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.25, role: "main-metal", wyckoff: "8c" },
      { label: "A", x: 0.25, y: 0.25, z: 0.75, role: "main-metal", wyckoff: "8c" },
      { label: "A", x: 0.75, y: 0.25, z: 0.75, role: "main-metal", wyckoff: "8c" },
      { label: "A", x: 0.25, y: 0.75, z: 0.25, role: "main-metal", wyckoff: "8c" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "secondary-metal", wyckoff: "4a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "secondary-metal", wyckoff: "4a" },
      { label: "B", x: 0.0, y: 0.5, z: 0.5, role: "secondary-metal", wyckoff: "4a" },
      { label: "B", x: 0.5, y: 0.0, z: 0.5, role: "secondary-metal", wyckoff: "4a" },
      { label: "C", x: 0.5, y: 0.5, z: 0.5, role: "sp-element", wyckoff: "4b" },
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "sp-element", wyckoff: "4b" },
      { label: "C", x: 0.0, y: 0.5, z: 0.0, role: "sp-element", wyckoff: "4b" },
      { label: "C", x: 0.5, y: 0.0, z: 0.0, role: "sp-element", wyckoff: "4b" },
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
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "metal", wyckoff: "8c" },
      { label: "M", x: 0.5, y: 0.5, z: 0.0, role: "metal", wyckoff: "8c" },
      { label: "M", x: 0.0, y: 0.5, z: 0.5, role: "metal", wyckoff: "8c" },
      { label: "M", x: 0.5, y: 0.0, z: 0.5, role: "metal", wyckoff: "8c" },
      { label: "X", x: 0.0, y: 0.34, z: 0.16, role: "pnicogen", wyckoff: "24g" },
      { label: "X", x: 0.34, y: 0.16, z: 0.0, role: "pnicogen", wyckoff: "24g" },
      { label: "X", x: 0.16, y: 0.0, z: 0.34, role: "pnicogen", wyckoff: "24g" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "guest", wyckoff: "1a" },
      { label: "M", x: 0.15, y: 0.15, z: 0.15, role: "cluster-metal", wyckoff: "6c" },
      { label: "M", x: 0.85, y: 0.85, z: 0.85, role: "cluster-metal", wyckoff: "6c" },
      { label: "M", x: 0.85, y: 0.15, z: 0.15, role: "cluster-metal", wyckoff: "6c" },
      { label: "M", x: 0.15, y: 0.85, z: 0.15, role: "cluster-metal", wyckoff: "6c" },
      { label: "M", x: 0.15, y: 0.15, z: 0.85, role: "cluster-metal", wyckoff: "6c" },
      { label: "M", x: 0.85, y: 0.85, z: 0.15, role: "cluster-metal", wyckoff: "6c" },
      { label: "X", x: 0.22, y: 0.0, z: 0.28, role: "chalcogen", wyckoff: "6c" },
      { label: "X", x: 0.0, y: 0.28, z: 0.22, role: "chalcogen", wyckoff: "6c" },
      { label: "X", x: 0.28, y: 0.22, z: 0.0, role: "chalcogen", wyckoff: "6c" },
      { label: "X", x: 0.78, y: 0.0, z: 0.72, role: "chalcogen", wyckoff: "6c" },
      { label: "X", x: 0.0, y: 0.72, z: 0.78, role: "chalcogen", wyckoff: "6c" },
      { label: "X", x: 0.72, y: 0.78, z: 0.0, role: "chalcogen", wyckoff: "6c" },
      { label: "X", x: 0.28, y: 0.0, z: 0.72, role: "chalcogen", wyckoff: "6c" },
      { label: "X", x: 0.0, y: 0.72, z: 0.28, role: "chalcogen", wyckoff: "6c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "spacer", wyckoff: "1a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "TM-square-planar", wyckoff: "1b" },
      { label: "O", x: 0.5, y: 0.0, z: 0.5, role: "in-plane-O", wyckoff: "2f" },
      { label: "O", x: 0.0, y: 0.5, z: 0.5, role: "in-plane-O", wyckoff: "2f" },
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
      { label: "M", x: 0.0, y: 0.0, z: 0.0, role: "metal", wyckoff: "4a" },
      { label: "M", x: 0.5, y: 0.5, z: 0.0, role: "metal", wyckoff: "4a" },
      { label: "M", x: 0.0, y: 0.5, z: 0.5, role: "metal", wyckoff: "4a" },
      { label: "M", x: 0.5, y: 0.0, z: 0.5, role: "metal", wyckoff: "4a" },
      { label: "X", x: 0.38, y: 0.38, z: 0.38, role: "dimer-1", wyckoff: "8c" },
      { label: "X", x: 0.62, y: 0.62, z: 0.62, role: "dimer-1", wyckoff: "8c" },
      { label: "X", x: 0.88, y: 0.12, z: 0.62, role: "dimer-2", wyckoff: "8c" },
      { label: "X", x: 0.12, y: 0.88, z: 0.38, role: "dimer-2", wyckoff: "8c" },
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
      { label: "R", x: 0.0, y: 0.0, z: 0.0, role: "rare-earth", wyckoff: "1a" },
      { label: "T", x: 0.5, y: 0.5, z: 0.0, role: "TM", wyckoff: "1b" },
      { label: "X", x: 0.5, y: 0.0, z: 0.31, role: "p-block-1", wyckoff: "4i" },
      { label: "X", x: 0.0, y: 0.5, z: 0.31, role: "p-block-1", wyckoff: "4i" },
      { label: "X", x: 0.5, y: 0.0, z: 0.69, role: "p-block-2", wyckoff: "4i" },
      { label: "X", x: 0.0, y: 0.5, z: 0.69, role: "p-block-2", wyckoff: "4i" },
      { label: "X", x: 0.0, y: 0.0, z: 0.5, role: "p-block-3", wyckoff: "1c" },
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
      { label: "R", x: 0.0, y: 0.0, z: 0.14, role: "spacer", wyckoff: "2c" },
      { label: "T", x: 0.5, y: 0.0, z: 0.5, role: "TM-layer", wyckoff: "2a" },
      { label: "X", x: 0.5, y: 0.5, z: 0.66, role: "pnictide", wyckoff: "2c" },
      { label: "O", x: 0.0, y: 0.0, z: 0.66, role: "oxide", wyckoff: "2c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.36, role: "spacer", wyckoff: "4e" },
      { label: "A", x: 0.0, y: 0.0, z: 0.64, role: "spacer", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "TM-square-planar", wyckoff: "2a" },
      { label: "X", x: 0.5, y: 0.0, z: 0.0, role: "in-plane-O", wyckoff: "4c" },
      { label: "X", x: 0.0, y: 0.5, z: 0.0, role: "in-plane-O", wyckoff: "4c" },
      { label: "X", x: 0.0, y: 0.0, z: 0.17, role: "apical-O", wyckoff: "4e" },
      { label: "X", x: 0.0, y: 0.0, z: 0.83, role: "apical-O", wyckoff: "4e" },
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
      { label: "A", x: 0.75, y: 0.25, z: 0.0, role: "TM", wyckoff: "2a" },
      { label: "A", x: 0.25, y: 0.75, z: 0.0, role: "TM", wyckoff: "2a" },
      { label: "B", x: 0.25, y: 0.25, z: 0.27, role: "chalcogen", wyckoff: "2c" },
      { label: "B", x: 0.75, y: 0.75, z: 0.73, role: "chalcogen", wyckoff: "2c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "large-atom", wyckoff: "8a" },
      { label: "A", x: 0.25, y: 0.25, z: 0.25, role: "large-atom", wyckoff: "8a" },
      { label: "B", x: 0.625, y: 0.625, z: 0.625, role: "small-atom", wyckoff: "16d" },
      { label: "B", x: 0.625, y: 0.375, z: 0.375, role: "small-atom", wyckoff: "16d" },
      { label: "B", x: 0.375, y: 0.625, z: 0.375, role: "small-atom", wyckoff: "16d" },
      { label: "B", x: 0.375, y: 0.375, z: 0.625, role: "small-atom", wyckoff: "16d" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "spacer", wyckoff: "1a" },
      { label: "B", x: 0.5, y: 0.0, z: 0.25, role: "kagome", wyckoff: "3f" },
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "kagome", wyckoff: "3f" },
      { label: "B", x: 0.5, y: 0.5, z: 0.25, role: "kagome", wyckoff: "3f" },
      { label: "C", x: 0.333, y: 0.167, z: 0.5, role: "honeycomb", wyckoff: "6m" },
      { label: "C", x: 0.167, y: 0.333, z: 0.5, role: "honeycomb", wyckoff: "6m" },
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
      { label: "R", x: 0.0, y: 0.0, z: 0.35, role: "rare-earth", wyckoff: "4e" },
      { label: "R", x: 0.0, y: 0.0, z: 0.65, role: "rare-earth", wyckoff: "4e" },
      { label: "T", x: 0.5, y: 0.5, z: 0.0, role: "TM", wyckoff: "2a" },
      { label: "O", x: 0.0, y: 0.5, z: 0.0, role: "apical-O", wyckoff: "4c" },
      { label: "O", x: 0.0, y: 0.0, z: 0.17, role: "planar-O", wyckoff: "4e" },
      { label: "O", x: 0.0, y: 0.0, z: 0.83, role: "planar-O", wyckoff: "4e" },
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
      { label: "R", x: 0.25, y: 0.25, z: 0.10, role: "spacer-layer", wyckoff: "2c" },
      { label: "T", x: 0.75, y: 0.25, z: 0.50, role: "conducting-layer", wyckoff: "2a" },
      { label: "X", x: 0.25, y: 0.25, z: 0.63, role: "chalcogen-top", wyckoff: "2c" },
      { label: "X", x: 0.75, y: 0.75, z: 0.37, role: "chalcogen-bot", wyckoff: "2c" },
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
      { label: "A", x: 0.25, y: 0.25, z: 0.0, role: "spacer", wyckoff: "2c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.5, role: "spacer", wyckoff: "2c" },
      { label: "B", x: 0.75, y: 0.25, z: 0.14, role: "TM-layer-1", wyckoff: "2a" },
      { label: "B", x: 0.25, y: 0.75, z: 0.64, role: "TM-layer-2", wyckoff: "2a" },
      { label: "C", x: 0.25, y: 0.25, z: 0.25, role: "sp-up", wyckoff: "2c" },
      { label: "C", x: 0.75, y: 0.75, z: 0.75, role: "sp-down", wyckoff: "2c" },
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
      { label: "R", x: 0.0, y: 0.0, z: 0.0, role: "rare-earth", wyckoff: "2a" },
      { label: "T", x: 0.0, y: 0.5, z: 0.25, role: "TM-1", wyckoff: "4d" },
      { label: "T", x: 0.5, y: 0.0, z: 0.25, role: "TM-2", wyckoff: "4d" },
      { label: "B", x: 0.0, y: 0.0, z: 0.35, role: "boron-1", wyckoff: "4e" },
      { label: "B", x: 0.0, y: 0.0, z: 0.65, role: "boron-2", wyckoff: "4e" },
      { label: "C", x: 0.0, y: 0.0, z: 0.5, role: "carbon", wyckoff: "2b" },
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
      { label: "X", x: 0.0, y: 0.0, z: 0.0, role: "electropositive", wyckoff: "4a" },
      { label: "Y", x: 0.25, y: 0.25, z: 0.25, role: "TM", wyckoff: "4c" },
      { label: "Z", x: 0.5, y: 0.5, z: 0.5, role: "main-group", wyckoff: "4b" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "alkali-spacer", wyckoff: "1a" },
      { label: "V", x: 0.5, y: 0.0, z: 0.5, role: "kagome-1", wyckoff: "3g" },
      { label: "V", x: 0.0, y: 0.5, z: 0.5, role: "kagome-2", wyckoff: "3g" },
      { label: "V", x: 0.5, y: 0.5, z: 0.5, role: "kagome-3", wyckoff: "3g" },
      { label: "Sb", x: 0.333, y: 0.667, z: 0.0, role: "pnictogen-1", wyckoff: "2d" },
      { label: "Sb", x: 0.667, y: 0.333, z: 0.0, role: "pnictogen-2", wyckoff: "2d" },
      { label: "Sb", x: 0.0, y: 0.0, z: 0.5, role: "pnictogen-hub", wyckoff: "1b" },
      { label: "Sb", x: 0.333, y: 0.667, z: 0.5, role: "pnictogen-3", wyckoff: "2d" },
      { label: "Sb", x: 0.667, y: 0.333, z: 0.5, role: "pnictogen-4", wyckoff: "2d" },
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
      { label: "R", x: 0.0, y: 0.0, z: 0.0, role: "filler-cage", wyckoff: "2a" },
      { label: "T", x: 0.25, y: 0.25, z: 0.25, role: "TM-1", wyckoff: "8c" },
      { label: "T", x: 0.75, y: 0.25, z: 0.25, role: "TM-2", wyckoff: "8c" },
      { label: "T", x: 0.25, y: 0.75, z: 0.25, role: "TM-3", wyckoff: "8c" },
      { label: "T", x: 0.25, y: 0.25, z: 0.75, role: "TM-4", wyckoff: "8c" },
      { label: "X", x: 0.0, y: 0.35, z: 0.15, role: "pnictogen-1", wyckoff: "24g" },
      { label: "X", x: 0.15, y: 0.0, z: 0.35, role: "pnictogen-2", wyckoff: "24g" },
      { label: "X", x: 0.35, y: 0.15, z: 0.0, role: "pnictogen-3", wyckoff: "24g" },
      { label: "X", x: 0.0, y: 0.65, z: 0.85, role: "pnictogen-4", wyckoff: "24g" },
      { label: "X", x: 0.85, y: 0.0, z: 0.65, role: "pnictogen-5", wyckoff: "24g" },
      { label: "X", x: 0.65, y: 0.85, z: 0.0, role: "pnictogen-6", wyckoff: "24g" },
      { label: "X", x: 0.0, y: 0.65, z: 0.15, role: "pnictogen-7", wyckoff: "24g" },
      { label: "X", x: 0.15, y: 0.0, z: 0.65, role: "pnictogen-8", wyckoff: "24g" },
      { label: "X", x: 0.65, y: 0.15, z: 0.0, role: "pnictogen-9", wyckoff: "24g" },
      { label: "X", x: 0.0, y: 0.35, z: 0.85, role: "pnictogen-10", wyckoff: "24g" },
      { label: "X", x: 0.85, y: 0.0, z: 0.35, role: "pnictogen-11", wyckoff: "24g" },
      { label: "X", x: 0.35, y: 0.85, z: 0.0, role: "pnictogen-12", wyckoff: "24g" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner", wyckoff: "2a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "body-center", wyckoff: "2a" },
      { label: "B", x: 0.25, y: 0.0, z: 0.5, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.75, y: 0.0, z: 0.5, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.5, y: 0.25, z: 0.0, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.5, y: 0.75, z: 0.0, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "chain", wyckoff: "6c" },
      { label: "B", x: 0.0, y: 0.5, z: 0.75, role: "chain", wyckoff: "6c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation", wyckoff: "4a" },
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "anion", wyckoff: "8c" },
      { label: "B", x: 0.75, y: 0.75, z: 0.75, role: "anion", wyckoff: "8c" },
    ],
    stoichiometryRatio: [1, 2],
    coordination: [8, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 2) return false;
      // Expanded: Ti/Ba/Sr/La for fluorite-type hydrides (TiH2, BaH2) and oxides
      const hasLarge = elements.some(e => ["Zr", "Hf", "Ce", "Th", "U", "Pb", "Ca", "Ti", "Ba", "Sr", "La", "Pr", "Nd"].includes(e));
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner", wyckoff: "1a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "face-center", wyckoff: "3c" },
      { label: "B", x: 0.5, y: 0.0, z: 0.5, role: "face-center", wyckoff: "3c" },
      { label: "B", x: 0.0, y: 0.5, z: 0.5, role: "face-center", wyckoff: "3c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "fcc-site", wyckoff: "4a" },
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "tetrahedral-site", wyckoff: "4c" },
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
      { label: "A", x: 0.333, y: 0.667, z: 0.063, role: "large-atom", wyckoff: "4f" },
      { label: "B", x: 0.5, y: 0.0, z: 0.0, role: "small-atom-1", wyckoff: "2a" },
      { label: "B", x: 0.833, y: 0.667, z: 0.25, role: "small-atom-2", wyckoff: "6h" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal-octahedral", wyckoff: "2a" },
      { label: "B", x: 0.333, y: 0.667, z: 0.25, role: "pnictogen-trigonal", wyckoff: "2c" },
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
      { label: "A", x: 0.333, y: 0.667, z: 0.0, role: "cation", wyckoff: "2b" },
      { label: "B", x: 0.333, y: 0.667, z: 0.375, role: "anion", wyckoff: "2b" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal", wyckoff: "1a" },
      { label: "B", x: 0.333, y: 0.667, z: 0.25, role: "anion-top", wyckoff: "2d" },
      { label: "B", x: 0.667, y: 0.333, z: 0.75, role: "anion-bot", wyckoff: "2d" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation", wyckoff: "2a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "cation", wyckoff: "2a" },
      { label: "B", x: 0.3, y: 0.3, z: 0.0, role: "anion", wyckoff: "4f" },
      { label: "B", x: 0.7, y: 0.7, z: 0.0, role: "anion", wyckoff: "4f" },
      { label: "B", x: 0.8, y: 0.2, z: 0.5, role: "anion", wyckoff: "4f" },
      { label: "B", x: 0.2, y: 0.8, z: 0.5, role: "anion", wyckoff: "4f" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.25, role: "minor-metal", wyckoff: "4a" },
      { label: "B", x: 0.16, y: 0.66, z: 0.0, role: "major-metal", wyckoff: "8h" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "spacer", wyckoff: "1a" },
      { label: "B", x: 0.333, y: 0.667, z: 0.5, role: "honeycomb", wyckoff: "2d" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal", wyckoff: "1a" },
      { label: "B", x: 0.333, y: 0.667, z: 0.5, role: "interstitial", wyckoff: "1d" },
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
      { label: "A", x: 0.062, y: 0.062, z: 0.062, role: "cation", wyckoff: "12a" },
      { label: "B", x: 0.25, y: 0.0, z: 0.0, role: "anion", wyckoff: "16c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner", wyckoff: "1a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "body-center", wyckoff: "1d" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "pnictogen-1a", wyckoff: "1a" },
      { label: "A", x: 0.333, y: 0.667, z: 0.5, role: "pnictogen-2d", wyckoff: "2d" },
      { label: "A", x: 0.667, y: 0.333, z: 0.5, role: "pnictogen-2d", wyckoff: "2d" },
      { label: "B", x: 0.5, y: 0.0, z: 0.0, role: "kagome-metal-3f", wyckoff: "3f" },
      { label: "B", x: 0.0, y: 0.5, z: 0.0, role: "kagome-metal-3f", wyckoff: "3f" },
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "kagome-metal-3f", wyckoff: "3f" },
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
      { label: "A", x: 0.125, y: 0.125, z: 0.125, role: "large-atom", wyckoff: "8a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "small-atom-network", wyckoff: "16d" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal-1", wyckoff: "1a" },
      { label: "B", x: 0.333, y: 0.667, z: 0.5, role: "metalloid-1", wyckoff: "1d" },
      { label: "C", x: 0.667, y: 0.333, z: 0.5, role: "metalloid-2", wyckoff: "1f" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "bcc-corner", wyckoff: "2a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "bcc-body", wyckoff: "2a" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "fcc-corner", wyckoff: "4a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.0, role: "fcc-face-1", wyckoff: "4a" },
      { label: "A", x: 0.5, y: 0.0, z: 0.5, role: "fcc-face-2", wyckoff: "4a" },
      { label: "A", x: 0.0, y: 0.5, z: 0.5, role: "fcc-face-3", wyckoff: "4a" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "hcp-site-1", wyckoff: "2a" },
      { label: "A", x: 0.333, y: 0.667, z: 0.5, role: "hcp-site-2", wyckoff: "2d" },
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
      { label: "A", x: 0.18, y: 0.25, z: 0.12, role: "TM-chain", wyckoff: "4c" },
      { label: "A", x: 0.32, y: 0.75, z: 0.62, role: "TM-chain", wyckoff: "4c" },
      { label: "A", x: 0.68, y: 0.25, z: 0.88, role: "TM-chain", wyckoff: "4c" },
      { label: "A", x: 0.82, y: 0.75, z: 0.38, role: "TM-chain", wyckoff: "4c" },
      { label: "B", x: 0.03, y: 0.25, z: 0.60, role: "pnictogen", wyckoff: "4c" },
      { label: "B", x: 0.47, y: 0.75, z: 0.10, role: "pnictogen", wyckoff: "4c" },
      { label: "B", x: 0.53, y: 0.25, z: 0.40, role: "pnictogen", wyckoff: "4c" },
      { label: "B", x: 0.97, y: 0.75, z: 0.90, role: "pnictogen", wyckoff: "4c" },
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
      { label: "A", x: 0.0, y: 0.15, z: 0.25, role: "TM-zigzag", wyckoff: "4c" },
      { label: "A", x: 0.0, y: 0.85, z: 0.75, role: "TM-zigzag", wyckoff: "4c" },
      { label: "B", x: 0.0, y: 0.44, z: 0.25, role: "sp-chain", wyckoff: "4c" },
      { label: "B", x: 0.0, y: 0.56, z: 0.75, role: "sp-chain", wyckoff: "4c" },
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
      { label: "A", x: 0.06, y: 0.25, z: 0.98, role: "A-site-large", wyckoff: "4c" },
      { label: "A", x: 0.44, y: 0.75, z: 0.48, role: "A-site-large", wyckoff: "4c" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "B-site-TM", wyckoff: "4b" },
      { label: "B", x: 0.5, y: 0.5, z: 0.0, role: "B-site-TM", wyckoff: "4b" },
      { label: "O", x: 0.1, y: 0.25, z: 0.1, role: "apical-O", wyckoff: "4c" },
      { label: "O", x: 0.4, y: 0.75, z: 0.6, role: "apical-O", wyckoff: "4c" },
      { label: "O", x: 0.7, y: 0.04, z: 0.3, role: "equatorial-O", wyckoff: "8d" },
      { label: "O", x: 0.8, y: 0.46, z: 0.2, role: "equatorial-O", wyckoff: "8d" },
      { label: "O", x: 0.3, y: 0.96, z: 0.7, role: "equatorial-O", wyckoff: "8d" },
      { label: "O", x: 0.2, y: 0.54, z: 0.8, role: "equatorial-O", wyckoff: "8d" },
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
      { label: "A", x: 0.25, y: 0.25, z: 0.0, role: "Cu-square-net", wyckoff: "2c" },
      { label: "A", x: 0.75, y: 0.75, z: 0.0, role: "Cu-square-net", wyckoff: "2c" },
      { label: "B", x: 0.25, y: 0.75, z: 0.30, role: "Cu-tetrahedral", wyckoff: "2a" },
      { label: "B", x: 0.75, y: 0.25, z: 0.70, role: "Cu-tetrahedral", wyckoff: "2a" },
      { label: "C", x: 0.75, y: 0.25, z: 0.27, role: "Sb-layer", wyckoff: "2c" },
      { label: "C", x: 0.25, y: 0.75, z: 0.73, role: "Sb-layer", wyckoff: "2c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "Te-vdW", wyckoff: "3a" },
      { label: "B", x: 0.0, y: 0.0, z: 0.40, role: "Bi-inner", wyckoff: "6c" },
      { label: "B", x: 0.0, y: 0.0, z: 0.60, role: "Bi-inner", wyckoff: "6c" },
      { label: "A", x: 0.0, y: 0.0, z: 0.21, role: "Te-inner", wyckoff: "6c" },
      { label: "A", x: 0.0, y: 0.0, z: 0.79, role: "Te-inner", wyckoff: "6c" },
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
      { label: "A", x: 0.0, y: 0.2, z: 0.06, role: "TM-helical", wyckoff: "4c" },
      { label: "A", x: 0.5, y: 0.3, z: 0.44, role: "TM-helical", wyckoff: "4c" },
      { label: "A", x: 0.0, y: 0.8, z: 0.94, role: "TM-helical", wyckoff: "4c" },
      { label: "A", x: 0.5, y: 0.7, z: 0.56, role: "TM-helical", wyckoff: "4c" },
      { label: "B", x: 0.2, y: 0.0, z: 0.39, role: "pnictogen", wyckoff: "4c" },
      { label: "B", x: 0.3, y: 0.5, z: 0.11, role: "pnictogen", wyckoff: "4c" },
      { label: "B", x: 0.8, y: 0.0, z: 0.61, role: "pnictogen", wyckoff: "4c" },
      { label: "B", x: 0.7, y: 0.5, z: 0.89, role: "pnictogen", wyckoff: "4c" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "metal-1", wyckoff: "2a" },
      { label: "A", x: 0.5, y: 0.5, z: 0.0, role: "metal-2", wyckoff: "2a" },
      { label: "B", x: 0.25, y: 0.25, z: 0.5, role: "anion-1", wyckoff: "4i" },
      { label: "B", x: 0.75, y: 0.75, z: 0.5, role: "anion-2", wyckoff: "4i" },
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
      { label: "A", x: 0.125, y: 0.125, z: 0.125, role: "tetrahedral-A", wyckoff: "8a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "octahedral-B-1", wyckoff: "16d" },
      { label: "B", x: 0.5, y: 0.75, z: 0.75, role: "octahedral-B-2", wyckoff: "16d" },
      { label: "X", x: 0.25, y: 0.25, z: 0.25, role: "anion-1", wyckoff: "32e" },
      { label: "X", x: 0.25, y: 0.75, z: 0.75, role: "anion-2", wyckoff: "32e" },
      { label: "X", x: 0.75, y: 0.25, z: 0.75, role: "anion-3", wyckoff: "32e" },
      { label: "X", x: 0.75, y: 0.75, z: 0.25, role: "anion-4", wyckoff: "32e" },
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
      { label: "A", x: 0.5, y: 0.5, z: 0.5, role: "A-16d", wyckoff: "16d" },
      { label: "A", x: 0.25, y: 0.25, z: 0.75, role: "A-16d", wyckoff: "16d" },
      { label: "B", x: 0.0, y: 0.0, z: 0.0, role: "B-16c", wyckoff: "16c" },
      { label: "B", x: 0.25, y: 0.25, z: 0.25, role: "B-16c", wyckoff: "16c" },
      { label: "O", x: 0.31, y: 0.125, z: 0.125, role: "O-48f", wyckoff: "48f" },
      { label: "O", x: 0.125, y: 0.31, z: 0.125, role: "O-48f", wyckoff: "48f" },
      { label: "O", x: 0.125, y: 0.125, z: 0.31, role: "O-48f", wyckoff: "48f" },
      { label: "O", x: 0.69, y: 0.875, z: 0.875, role: "O-48f", wyckoff: "48f" },
      { label: "O", x: 0.875, y: 0.69, z: 0.875, role: "O-48f", wyckoff: "48f" },
      { label: "O", x: 0.875, y: 0.875, z: 0.69, role: "O-48f", wyckoff: "48f" },
      { label: "O", x: 0.375, y: 0.375, z: 0.375, role: "O-8b", wyckoff: "8b" },
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
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "corner", wyckoff: "1a" },
      { label: "B", x: 0.5, y: 0.5, z: 0.5, role: "body-center", wyckoff: "1b" },
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
      { label: "M", x: 0.333, y: 0.667, z: 0.085, role: "TM-layer-1", wyckoff: "4f" },
      { label: "M", x: 0.667, y: 0.333, z: 0.915, role: "TM-layer-2", wyckoff: "4f" },
      { label: "M", x: 0.333, y: 0.667, z: 0.585, role: "TM-layer-3", wyckoff: "4f" },
      { label: "M", x: 0.667, y: 0.333, z: 0.415, role: "TM-layer-4", wyckoff: "4f" },
      { label: "A", x: 0.333, y: 0.667, z: 0.25, role: "A-layer", wyckoff: "2d" },
      { label: "A", x: 0.667, y: 0.333, z: 0.75, role: "A-layer", wyckoff: "2d" },
      { label: "X", x: 0.0, y: 0.0, z: 0.0, role: "X-interstitial-1", wyckoff: "2a" },
      { label: "X", x: 0.0, y: 0.0, z: 0.5, role: "X-interstitial-2", wyckoff: "2a" },
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

  // ── A3B3C (Pm-3n, SG 223): Nb3Re3Sn-type — intermetallic 3:3:1 ──
  // Cr3Si-derived ternary: two TMs share the chain sites + one sp-element
  // at the corner/body-center. Examples: Nb3Os3B, Mo3Re3C, V3Ir3Sn
  {
    name: "A3B3C-Intermetallic",
    spaceGroup: "Pm-3n",
    latticeType: "cubic",
    cOverA: 1.0,
    sites: [
      // C at 2a Wyckoff (corner + body center)
      { label: "C", x: 0.0, y: 0.0, z: 0.0, role: "corner", wyckoff: "2a" },
      // A at 6c chain sites (face-center edges)
      { label: "A", x: 0.25, y: 0.0, z: 0.5, role: "chain-A", wyckoff: "6c" },
      { label: "A", x: 0.75, y: 0.0, z: 0.5, role: "chain-A", wyckoff: "6c" },
      { label: "A", x: 0.5, y: 0.25, z: 0.0, role: "chain-A", wyckoff: "6c" },
      // B at remaining 6c positions
      { label: "B", x: 0.0, y: 0.5, z: 0.25, role: "chain-B", wyckoff: "6c" },
      { label: "B", x: 0.0, y: 0.5, z: 0.75, role: "chain-B", wyckoff: "6c" },
      { label: "B", x: 0.5, y: 0.75, z: 0.0, role: "chain-B", wyckoff: "6c" },
    ],
    stoichiometryRatio: [3, 3, 1],
    coordination: [12, 12, 14],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      // Two transition metals + one metalloid/post-TM, or three TMs
      const tmCount = elements.filter(e => isTransitionMetal(e)).length;
      const hasMetalloid = elements.some(e => ["Al", "Ga", "In", "Si", "Ge", "Sn", "Sb", "Bi", "B", "C", "N", "Pb"].includes(e));
      return tmCount >= 2 && (tmCount === 3 || hasMetalloid);
    },
  },

  // ── IV-IV-VI₂ Chalcogenide (I-42d, SG 122): GeSnTe₂-type ──
  // Chalcopyrite-derived but with group-IV cations instead of I-III.
  // Common thermoelectrics: GeSeTe, SnGeSe₂, PbSnTe₂
  {
    name: "Chalcopyrite-IV-IV-VI2",
    spaceGroup: "I-42d",
    latticeType: "tetragonal",
    cOverA: 1.97,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation-1", wyckoff: "4a" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "cation-2", wyckoff: "4b" },
      { label: "C", x: 0.25, y: 0.125, z: 0.625, role: "anion", wyckoff: "8d" },
      { label: "C", x: 0.75, y: 0.125, z: 0.875, role: "anion", wyckoff: "8d" },
    ],
    stoichiometryRatio: [1, 1, 2],
    coordination: [4, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
      const hasGroupIV = elements.some(e => ["Si", "Ge", "Sn", "Pb"].includes(e));
      // Need chalcogen + at least one group-IV; the third can be TM, another IV, or metalloid
      const groupIVcount = elements.filter(e => ["Si", "Ge", "Sn", "Pb"].includes(e)).length;
      const hasTMorMet = elements.some(e =>
        isTransitionMetal(e) || ["Bi", "Sb", "In", "Ga", "Al", "Tl", "Cu", "Ag"].includes(e)
      );
      return hasChalcogen && hasGroupIV && (groupIVcount >= 2 || hasTMorMet);
    },
  },

  // ── Broadened Chalcopyrite (I-42d): for TM-based ABX₂ not caught by original ──
  // Original requires Cu/Ag; this covers TM-TM-chalcogenide and TM-metalloid-chalcogenide
  {
    name: "Chalcopyrite-TM-ABX2",
    spaceGroup: "I-42d",
    latticeType: "tetragonal",
    cOverA: 1.97,
    sites: [
      { label: "A", x: 0.0, y: 0.0, z: 0.0, role: "cation-1", wyckoff: "4a" },
      { label: "B", x: 0.0, y: 0.0, z: 0.5, role: "cation-2", wyckoff: "4b" },
      { label: "C", x: 0.25, y: 0.125, z: 0.625, role: "anion", wyckoff: "8d" },
      { label: "C", x: 0.75, y: 0.125, z: 0.875, role: "anion", wyckoff: "8d" },
    ],
    stoichiometryRatio: [1, 1, 2],
    coordination: [4, 4, 4],
    chemistryRules: (elements) => {
      if (elements.length !== 3) return false;
      const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
      const hasTM = elements.some(e => isTransitionMetal(e));
      const hasCation2 = elements.some(e =>
        isTransitionMetal(e) || isRareEarth(e) ||
        ["Al", "Ga", "In", "Si", "Ge", "Sn", "Sb", "Bi", "Pb", "Mg", "Ca", "Sr", "Ba"].includes(e)
      );
      return hasChalcogen && hasTM && hasCation2;
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

/** Build Cartesian lattice vectors from lattice parameters (a, b, c, alpha, beta, gamma). */
function buildLatticeVectorsFromParams(p: LatticeParams): [number, number, number][] {
  const { a, b, c, alpha, beta, gamma } = p;
  const ar = alpha * Math.PI / 180;
  const br = beta * Math.PI / 180;
  const gr = gamma * Math.PI / 180;

  const cosA = Math.cos(ar), cosB = Math.cos(br), cosG = Math.cos(gr);
  const sinG = Math.sin(gr);

  // Standard crystallographic convention:
  // v1 along x, v2 in xy-plane, v3 general
  const v1: [number, number, number] = [a, 0, 0];
  const v2: [number, number, number] = [b * cosG, b * sinG, 0];
  const cx = c * cosB;
  const cy = sinG > 1e-10 ? c * (cosA - cosB * cosG) / sinG : 0;
  const cz = Math.sqrt(Math.max(0, c * c - cx * cx - cy * cy));
  const v3: [number, number, number] = [cx, cy, cz];

  return [v1, v2, v3];
}

export function fillPrototype(formula: string): FilledPrototype | null {
  const result = selectPrototype(formula);
  if (!result) return null;

  const { template, siteMap } = result;
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  const params = estimateLatticeConstant(elements, counts, template);
  const vecs = buildLatticeVectorsFromParams(params);

  const atoms: { element: string; x: number; y: number; z: number }[] = [];
  for (const site of template.sites) {
    const element = siteMap[site.label];
    if (!element) continue;

    // General fractional → Cartesian via lattice vectors
    const x = site.x * vecs[0][0] + site.y * vecs[1][0] + site.z * vecs[2][0];
    const y = site.x * vecs[0][1] + site.y * vecs[1][1] + site.z * vecs[2][1];
    const z = site.x * vecs[0][2] + site.y * vecs[1][2] + site.z * vecs[2][2];
    atoms.push({ element, x, y, z });
  }

  return {
    templateName: template.name,
    siteMap,
    atoms,
    latticeParam: params.a,
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

    const latticeParams = estimateLatticeConstant(elements, counts, template);
    const { a, c } = latticeParams;
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
