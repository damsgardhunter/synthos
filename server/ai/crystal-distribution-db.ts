import { getElementData, isTransitionMetal, isRareEarth } from "../learning/elemental-data";

export interface CrystallographicEntry {
  formula: string;
  spaceGroupNumber: number;
  spaceGroupSymbol: string;
  crystalSystem: string;
  a: number;
  b: number;
  c: number;
  alpha: number;
  beta: number;
  gamma: number;
  wyckoffSites: WyckoffSite[];
  atomicVolume: number;
  packingFraction: number;
  tc?: number;
  formationEnergy?: number;
  source: string;
}

export interface WyckoffSite {
  letter: string;
  multiplicity: number;
  x: number;
  y: number;
  z: number;
  element: string;
  occupancy: number;
}

export interface CrystalSystemDistribution {
  system: string;
  spaceGroups: { sg: number; symbol: string; weight: number; count: number }[];
  latticeParams: {
    aMean: number; aStd: number;
    bMean: number; bStd: number;
    cMean: number; cStd: number;
    cOverAMean: number; cOverAStd: number;
  };
  volumePerAtom: { mean: number; std: number };
  packingFraction: { mean: number; std: number };
  commonWyckoff: { letter: string; multiplicity: number; typicalX: number; typicalY: number; typicalZ: number; spread: number }[];
}

export interface ElementSitePreference {
  element: string;
  preferredWyckoff: string[];
  coordPreference: number[];
  siteFractions: Record<string, number>;
}

const CRYSTAL_SYSTEM_DISTRIBUTIONS: CrystalSystemDistribution[] = [
  {
    system: "cubic",
    spaceGroups: [
      { sg: 225, symbol: "Fm-3m", weight: 0.22, count: 48000 },
      { sg: 229, symbol: "Im-3m", weight: 0.15, count: 32000 },
      { sg: 221, symbol: "Pm-3m", weight: 0.18, count: 38000 },
      { sg: 227, symbol: "Fd-3m", weight: 0.12, count: 25000 },
      { sg: 216, symbol: "F-43m", weight: 0.10, count: 21000 },
      { sg: 223, symbol: "Pm-3n", weight: 0.08, count: 17000 },
      { sg: 215, symbol: "P-43m", weight: 0.05, count: 10000 },
      { sg: 226, symbol: "Fm-3c", weight: 0.04, count: 8500 },
      { sg: 230, symbol: "Ia-3d", weight: 0.03, count: 6300 },
      { sg: 220, symbol: "I-43d", weight: 0.03, count: 6200 },
    ],
    latticeParams: { aMean: 5.2, aStd: 1.5, bMean: 5.2, bStd: 1.5, cMean: 5.2, cStd: 1.5, cOverAMean: 1.0, cOverAStd: 0.0 },
    volumePerAtom: { mean: 18.5, std: 6.2 },
    packingFraction: { mean: 0.68, std: 0.08 },
    commonWyckoff: [
      { letter: "a", multiplicity: 4, typicalX: 0, typicalY: 0, typicalZ: 0, spread: 0 },
      { letter: "b", multiplicity: 4, typicalX: 0.5, typicalY: 0.5, typicalZ: 0.5, spread: 0 },
      { letter: "c", multiplicity: 8, typicalX: 0.25, typicalY: 0.25, typicalZ: 0.25, spread: 0.02 },
      { letter: "d", multiplicity: 24, typicalX: 0, typicalY: 0.25, typicalZ: 0.25, spread: 0.03 },
      { letter: "e", multiplicity: 24, typicalX: 0.25, typicalY: 0, typicalZ: 0, spread: 0.04 },
    ],
  },
  {
    system: "hexagonal",
    spaceGroups: [
      { sg: 194, symbol: "P6₃/mmc", weight: 0.30, count: 28000 },
      { sg: 191, symbol: "P6/mmm", weight: 0.15, count: 14000 },
      { sg: 186, symbol: "P6₃mc", weight: 0.12, count: 11200 },
      { sg: 193, symbol: "P6₃/mcm", weight: 0.10, count: 9300 },
      { sg: 189, symbol: "P-62m", weight: 0.08, count: 7500 },
      { sg: 187, symbol: "P-6m2", weight: 0.07, count: 6500 },
      { sg: 176, symbol: "P6₃/m", weight: 0.06, count: 5600 },
      { sg: 192, symbol: "P6/mcc", weight: 0.05, count: 4700 },
      { sg: 185, symbol: "P6₃cm", weight: 0.04, count: 3700 },
      { sg: 190, symbol: "P-62c", weight: 0.03, count: 2800 },
    ],
    latticeParams: { aMean: 4.5, aStd: 1.8, bMean: 4.5, bStd: 1.8, cMean: 7.2, cStd: 3.5, cOverAMean: 1.6, cOverAStd: 0.5 },
    volumePerAtom: { mean: 20.1, std: 7.5 },
    packingFraction: { mean: 0.64, std: 0.10 },
    commonWyckoff: [
      { letter: "a", multiplicity: 2, typicalX: 0, typicalY: 0, typicalZ: 0, spread: 0 },
      { letter: "b", multiplicity: 2, typicalX: 0, typicalY: 0, typicalZ: 0.25, spread: 0.02 },
      { letter: "c", multiplicity: 2, typicalX: 1/3, typicalY: 2/3, typicalZ: 0.25, spread: 0.03 },
      { letter: "d", multiplicity: 4, typicalX: 1/3, typicalY: 2/3, typicalZ: 0, spread: 0.02 },
      { letter: "f", multiplicity: 6, typicalX: 0.5, typicalY: 0, typicalZ: 0, spread: 0.03 },
      { letter: "g", multiplicity: 6, typicalX: 0.5, typicalY: 0, typicalZ: 0.5, spread: 0.04 },
    ],
  },
  {
    system: "tetragonal",
    spaceGroups: [
      { sg: 139, symbol: "I4/mmm", weight: 0.25, count: 22000 },
      { sg: 127, symbol: "P4/mbm", weight: 0.10, count: 8800 },
      { sg: 129, symbol: "P4/nmm", weight: 0.12, count: 10500 },
      { sg: 140, symbol: "I4/mcm", weight: 0.08, count: 7000 },
      { sg: 123, symbol: "P4/mmm", weight: 0.15, count: 13200 },
      { sg: 136, symbol: "P4₂/mnm", weight: 0.08, count: 7000 },
      { sg: 141, symbol: "I4₁/amd", weight: 0.07, count: 6200 },
      { sg: 137, symbol: "P4₂/nmc", weight: 0.05, count: 4400 },
      { sg: 142, symbol: "I4₁/acd", weight: 0.05, count: 4400 },
      { sg: 130, symbol: "P4/ncc", weight: 0.05, count: 4400 },
    ],
    latticeParams: { aMean: 4.2, aStd: 1.2, bMean: 4.2, bStd: 1.2, cMean: 9.8, cStd: 4.5, cOverAMean: 2.3, cOverAStd: 1.0 },
    volumePerAtom: { mean: 19.2, std: 6.8 },
    packingFraction: { mean: 0.62, std: 0.11 },
    commonWyckoff: [
      { letter: "a", multiplicity: 2, typicalX: 0, typicalY: 0, typicalZ: 0, spread: 0 },
      { letter: "b", multiplicity: 2, typicalX: 0, typicalY: 0, typicalZ: 0.5, spread: 0 },
      { letter: "c", multiplicity: 4, typicalX: 0, typicalY: 0.5, typicalZ: 0, spread: 0.02 },
      { letter: "d", multiplicity: 4, typicalX: 0, typicalY: 0.5, typicalZ: 0.25, spread: 0.03 },
      { letter: "e", multiplicity: 4, typicalX: 0, typicalY: 0, typicalZ: 0.35, spread: 0.05 },
      { letter: "f", multiplicity: 8, typicalX: 0.25, typicalY: 0.25, typicalZ: 0.25, spread: 0.05 },
    ],
  },
  {
    system: "orthorhombic",
    spaceGroups: [
      { sg: 62, symbol: "Pnma", weight: 0.22, count: 35000 },
      { sg: 63, symbol: "Cmcm", weight: 0.12, count: 19000 },
      { sg: 71, symbol: "Immm", weight: 0.10, count: 16000 },
      { sg: 69, symbol: "Fmmm", weight: 0.08, count: 12800 },
      { sg: 64, symbol: "Cmce", weight: 0.07, count: 11200 },
      { sg: 65, symbol: "Cmmm", weight: 0.07, count: 11200 },
      { sg: 55, symbol: "Pbam", weight: 0.06, count: 9600 },
      { sg: 47, symbol: "Pmmm", weight: 0.06, count: 9600 },
      { sg: 66, symbol: "Cccm", weight: 0.05, count: 8000 },
      { sg: 59, symbol: "Pmmn", weight: 0.05, count: 8000 },
    ],
    latticeParams: { aMean: 5.5, aStd: 2.5, bMean: 6.2, bStd: 3.0, cMean: 8.5, cStd: 4.0, cOverAMean: 1.55, cOverAStd: 0.8 },
    volumePerAtom: { mean: 21.0, std: 8.0 },
    packingFraction: { mean: 0.60, std: 0.12 },
    commonWyckoff: [
      { letter: "a", multiplicity: 4, typicalX: 0, typicalY: 0, typicalZ: 0, spread: 0 },
      { letter: "b", multiplicity: 4, typicalX: 0.5, typicalY: 0, typicalZ: 0, spread: 0.02 },
      { letter: "c", multiplicity: 4, typicalX: 0.25, typicalY: 0.25, typicalZ: 0, spread: 0.03 },
      { letter: "d", multiplicity: 8, typicalX: 0.15, typicalY: 0.25, typicalZ: 0.06, spread: 0.06 },
    ],
  },
  {
    system: "trigonal",
    spaceGroups: [
      { sg: 166, symbol: "R-3m", weight: 0.35, count: 18000 },
      { sg: 167, symbol: "R-3c", weight: 0.15, count: 7700 },
      { sg: 164, symbol: "P-3m1", weight: 0.12, count: 6200 },
      { sg: 148, symbol: "R-3", weight: 0.10, count: 5100 },
      { sg: 160, symbol: "R3m", weight: 0.08, count: 4100 },
      { sg: 163, symbol: "P-31c", weight: 0.05, count: 2600 },
      { sg: 165, symbol: "P-3c1", weight: 0.05, count: 2600 },
      { sg: 161, symbol: "R3c", weight: 0.05, count: 2600 },
      { sg: 162, symbol: "P-31m", weight: 0.03, count: 1500 },
      { sg: 147, symbol: "P-3", weight: 0.02, count: 1000 },
    ],
    latticeParams: { aMean: 4.8, aStd: 1.5, bMean: 4.8, bStd: 1.5, cMean: 12.5, cStd: 5.0, cOverAMean: 2.6, cOverAStd: 1.2 },
    volumePerAtom: { mean: 19.5, std: 7.0 },
    packingFraction: { mean: 0.63, std: 0.10 },
    commonWyckoff: [
      { letter: "a", multiplicity: 3, typicalX: 0, typicalY: 0, typicalZ: 0, spread: 0 },
      { letter: "b", multiplicity: 3, typicalX: 0, typicalY: 0, typicalZ: 0.5, spread: 0 },
      { letter: "c", multiplicity: 6, typicalX: 0, typicalY: 0, typicalZ: 0.23, spread: 0.04 },
      { letter: "d", multiplicity: 6, typicalX: 0.5, typicalY: 0, typicalZ: 0.5, spread: 0.03 },
    ],
  },
  {
    system: "monoclinic",
    spaceGroups: [
      { sg: 14, symbol: "P2₁/c", weight: 0.30, count: 42000 },
      { sg: 12, symbol: "C2/m", weight: 0.20, count: 28000 },
      { sg: 15, symbol: "C2/c", weight: 0.15, count: 21000 },
      { sg: 11, symbol: "P2₁/m", weight: 0.10, count: 14000 },
      { sg: 13, symbol: "P2/c", weight: 0.08, count: 11200 },
      { sg: 10, symbol: "P2/m", weight: 0.05, count: 7000 },
      { sg: 4, symbol: "P2₁", weight: 0.04, count: 5600 },
      { sg: 5, symbol: "C2", weight: 0.03, count: 4200 },
      { sg: 9, symbol: "Cc", weight: 0.03, count: 4200 },
      { sg: 8, symbol: "Cm", weight: 0.02, count: 2800 },
    ],
    latticeParams: { aMean: 5.8, aStd: 2.5, bMean: 6.5, bStd: 3.0, cMean: 8.0, cStd: 4.0, cOverAMean: 1.38, cOverAStd: 0.7 },
    volumePerAtom: { mean: 22.0, std: 9.0 },
    packingFraction: { mean: 0.58, std: 0.12 },
    commonWyckoff: [
      { letter: "a", multiplicity: 2, typicalX: 0, typicalY: 0, typicalZ: 0, spread: 0 },
      { letter: "b", multiplicity: 2, typicalX: 0.5, typicalY: 0.5, typicalZ: 0, spread: 0.02 },
      { letter: "e", multiplicity: 4, typicalX: 0.23, typicalY: 0.25, typicalZ: 0.12, spread: 0.08 },
      { letter: "f", multiplicity: 4, typicalX: 0.15, typicalY: 0.35, typicalZ: 0.3, spread: 0.10 },
    ],
  },
];

const SC_SYSTEM_WEIGHTS: Record<string, number> = {
  cubic: 0.30,
  tetragonal: 0.25,
  hexagonal: 0.15,
  orthorhombic: 0.12,
  trigonal: 0.10,
  monoclinic: 0.08,
};

const ELEMENT_SITE_PREFERENCES: ElementSitePreference[] = [
  { element: "La", preferredWyckoff: ["a", "b"], coordPreference: [12, 9, 8], siteFractions: { a: 0.6, b: 0.3, c: 0.1 } },
  { element: "Y", preferredWyckoff: ["a", "b"], coordPreference: [12, 9, 8], siteFractions: { a: 0.55, b: 0.35, c: 0.1 } },
  { element: "Ba", preferredWyckoff: ["a"], coordPreference: [12, 10], siteFractions: { a: 0.7, b: 0.2, c: 0.1 } },
  { element: "Sr", preferredWyckoff: ["a"], coordPreference: [12, 10], siteFractions: { a: 0.65, b: 0.25, c: 0.1 } },
  { element: "Ca", preferredWyckoff: ["a", "b"], coordPreference: [8, 6], siteFractions: { a: 0.5, b: 0.35, c: 0.15 } },
  { element: "Nb", preferredWyckoff: ["a", "c"], coordPreference: [6, 8], siteFractions: { a: 0.4, c: 0.4, d: 0.2 } },
  { element: "V", preferredWyckoff: ["a", "c"], coordPreference: [6, 4], siteFractions: { a: 0.4, c: 0.35, d: 0.25 } },
  { element: "Ti", preferredWyckoff: ["a", "b"], coordPreference: [6, 8], siteFractions: { a: 0.5, b: 0.3, c: 0.2 } },
  { element: "Ta", preferredWyckoff: ["a", "c"], coordPreference: [6, 8], siteFractions: { a: 0.45, c: 0.35, d: 0.2 } },
  { element: "Mo", preferredWyckoff: ["a", "c"], coordPreference: [6, 8], siteFractions: { a: 0.4, c: 0.4, d: 0.2 } },
  { element: "Fe", preferredWyckoff: ["c", "d"], coordPreference: [4, 6], siteFractions: { c: 0.45, d: 0.35, e: 0.2 } },
  { element: "Cu", preferredWyckoff: ["c", "d"], coordPreference: [4, 6, 2], siteFractions: { c: 0.4, d: 0.35, e: 0.25 } },
  { element: "H", preferredWyckoff: ["d", "e", "f"], coordPreference: [4, 6, 3], siteFractions: { d: 0.3, e: 0.35, f: 0.35 } },
  { element: "B", preferredWyckoff: ["c", "d"], coordPreference: [6, 4, 3], siteFractions: { c: 0.35, d: 0.4, e: 0.25 } },
  { element: "C", preferredWyckoff: ["c", "d"], coordPreference: [4, 6], siteFractions: { c: 0.4, d: 0.4, e: 0.2 } },
  { element: "N", preferredWyckoff: ["c", "d"], coordPreference: [4, 3], siteFractions: { c: 0.4, d: 0.35, e: 0.25 } },
  { element: "O", preferredWyckoff: ["d", "e"], coordPreference: [4, 6, 2], siteFractions: { d: 0.4, e: 0.35, f: 0.25 } },
  { element: "S", preferredWyckoff: ["d", "e"], coordPreference: [4, 6], siteFractions: { d: 0.45, e: 0.35, f: 0.2 } },
  { element: "As", preferredWyckoff: ["d", "e"], coordPreference: [4, 3], siteFractions: { d: 0.4, e: 0.35, f: 0.25 } },
  { element: "Se", preferredWyckoff: ["d", "e"], coordPreference: [4, 6], siteFractions: { d: 0.45, e: 0.35, f: 0.2 } },
  { element: "Al", preferredWyckoff: ["b", "c"], coordPreference: [6, 4, 8], siteFractions: { b: 0.35, c: 0.4, d: 0.25 } },
  { element: "Ge", preferredWyckoff: ["c", "d"], coordPreference: [4, 6], siteFractions: { c: 0.4, d: 0.35, e: 0.25 } },
  { element: "Sn", preferredWyckoff: ["b", "c"], coordPreference: [6, 4], siteFractions: { b: 0.4, c: 0.35, d: 0.25 } },
  { element: "Mg", preferredWyckoff: ["a", "b"], coordPreference: [6, 12], siteFractions: { a: 0.5, b: 0.35, c: 0.15 } },
  { element: "P", preferredWyckoff: ["c", "d"], coordPreference: [4, 3], siteFractions: { c: 0.35, d: 0.4, e: 0.25 } },
  { element: "Si", preferredWyckoff: ["c", "d"], coordPreference: [4, 6], siteFractions: { c: 0.4, d: 0.35, e: 0.25 } },
  { element: "Pb", preferredWyckoff: ["a", "b"], coordPreference: [12, 6, 8], siteFractions: { a: 0.5, b: 0.3, c: 0.2 } },
  { element: "Bi", preferredWyckoff: ["a", "b"], coordPreference: [6, 8], siteFractions: { a: 0.45, b: 0.35, c: 0.2 } },
  { element: "Te", preferredWyckoff: ["d", "e"], coordPreference: [6, 4], siteFractions: { d: 0.4, e: 0.35, f: 0.25 } },
];

const sitePreferenceMap = new Map<string, ElementSitePreference>();
for (const pref of ELEMENT_SITE_PREFERENCES) {
  sitePreferenceMap.set(pref.element, pref);
}

function gaussRandom(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
}

function sampleGaussian(mean: number, std: number): number {
  return mean + gaussRandom() * std;
}

export function sampleCrystalSystem(scBias: boolean = true): CrystalSystemDistribution {
  const weights = CRYSTAL_SYSTEM_DISTRIBUTIONS.map(d => {
    const base = SC_SYSTEM_WEIGHTS[d.system] ?? 0.1;
    return scBias ? base : 1 / CRYSTAL_SYSTEM_DISTRIBUTIONS.length;
  });
  const total = weights.reduce((s, w) => s + w, 0);
  let r = Math.random() * total;
  for (let i = 0; i < CRYSTAL_SYSTEM_DISTRIBUTIONS.length; i++) {
    r -= weights[i];
    if (r <= 0) return CRYSTAL_SYSTEM_DISTRIBUTIONS[i];
  }
  return CRYSTAL_SYSTEM_DISTRIBUTIONS[0];
}

export function sampleSpaceGroup(dist: CrystalSystemDistribution): { sg: number; symbol: string } {
  const total = dist.spaceGroups.reduce((s, g) => s + g.weight, 0);
  let r = Math.random() * total;
  for (const g of dist.spaceGroups) {
    r -= g.weight;
    if (r <= 0) return { sg: g.sg, symbol: g.symbol };
  }
  return { sg: dist.spaceGroups[0].sg, symbol: dist.spaceGroups[0].symbol };
}

export function sampleLatticeParams(dist: CrystalSystemDistribution): { a: number; b: number; c: number; alpha: number; beta: number; gamma: number } {
  const lp = dist.latticeParams;
  const a = Math.max(2.5, sampleGaussian(lp.aMean, lp.aStd));
  let b: number, c: number;

  if (dist.system === "cubic") {
    b = a;
    c = a;
  } else if (dist.system === "tetragonal" || dist.system === "hexagonal" || dist.system === "trigonal") {
    b = a;
    c = Math.max(2.5, a * Math.max(0.5, sampleGaussian(lp.cOverAMean, lp.cOverAStd)));
  } else {
    b = Math.max(2.5, sampleGaussian(lp.bMean, lp.bStd));
    c = Math.max(2.5, sampleGaussian(lp.cMean, lp.cStd));
  }

  let alpha = 90, beta = 90, gamma = 90;
  if (dist.system === "hexagonal" || dist.system === "trigonal") {
    gamma = 120;
  } else if (dist.system === "monoclinic") {
    beta = 90 + sampleGaussian(10, 5);
    beta = Math.max(91, Math.min(120, beta));
  }

  return {
    a: Math.round(a * 100) / 100,
    b: Math.round(b * 100) / 100,
    c: Math.round(c * 100) / 100,
    alpha, beta: Math.round(beta * 10) / 10, gamma,
  };
}

export function sampleWyckoffPositions(
  dist: CrystalSystemDistribution,
  composition: Record<string, number>,
): WyckoffSite[] {
  const sites: WyckoffSite[] = [];
  const elements = Object.entries(composition);
  const availableWyckoff = [...dist.commonWyckoff];
  let wyckoffIdx = 0;

  for (const [el, count] of elements) {
    const pref = sitePreferenceMap.get(el);
    let bestWyckoff = availableWyckoff[wyckoffIdx % availableWyckoff.length];

    if (pref) {
      const preferred = availableWyckoff.find(w => pref.preferredWyckoff.includes(w.letter));
      if (preferred) bestWyckoff = preferred;
    }

    for (let i = 0; i < count; i++) {
      const spreadFactor = pref ? bestWyckoff.spread : bestWyckoff.spread * 1.5;
      sites.push({
        letter: bestWyckoff.letter,
        multiplicity: bestWyckoff.multiplicity,
        x: bestWyckoff.typicalX + gaussRandom() * spreadFactor,
        y: bestWyckoff.typicalY + gaussRandom() * spreadFactor,
        z: bestWyckoff.typicalZ + gaussRandom() * spreadFactor,
        element: el,
        occupancy: 1.0,
      });
    }

    wyckoffIdx++;
  }

  for (const site of sites) {
    site.x = Math.max(0, Math.min(1, site.x));
    site.y = Math.max(0, Math.min(1, site.y));
    site.z = Math.max(0, Math.min(1, site.z));
  }

  return sites;
}

export function getDistributionForSystem(system: string): CrystalSystemDistribution | undefined {
  return CRYSTAL_SYSTEM_DISTRIBUTIONS.find(d => d.system === system);
}

export function getElementSitePreference(el: string): ElementSitePreference | undefined {
  return sitePreferenceMap.get(el);
}

export function getAllDistributions(): CrystalSystemDistribution[] {
  return CRYSTAL_SYSTEM_DISTRIBUTIONS;
}

export function getDistributionStats() {
  return {
    totalSystems: CRYSTAL_SYSTEM_DISTRIBUTIONS.length,
    totalSpaceGroups: CRYSTAL_SYSTEM_DISTRIBUTIONS.reduce((s, d) => s + d.spaceGroups.length, 0),
    totalElementPreferences: ELEMENT_SITE_PREFERENCES.length,
    systemWeights: SC_SYSTEM_WEIGHTS,
    estimatedStructures: CRYSTAL_SYSTEM_DISTRIBUTIONS.reduce(
      (s, d) => s + d.spaceGroups.reduce((sg, g) => sg + g.count, 0), 0
    ),
  };
}