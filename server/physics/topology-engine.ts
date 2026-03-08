import type { ElectronicStructure, TightBindingTopology } from "../learning/physics-engine";

export interface TopologicalAnalysis {
  topologicalScore: number;
  z2Score: number;
  chernScore: number;
  mirrorSymmetryIndicator: number;
  socStrength: number;
  bandInversionProbability: number;
  diracNodeProbability: number;
  flatBandIndicator: number;
  majoranaFeasibility: number;
  topologicalClass: string;
  indicators: string[];
  details: {
    socContribution: number;
    bandInversionContribution: number;
    symmetryContribution: number;
    flatBandContribution: number;
  };
}

const HEAVY_ELEMENT_SOC: Record<string, number> = {
  Bi: 1.25, Pb: 0.91, Tl: 0.79, Hg: 0.75, Po: 1.0,
  Te: 0.47, Sb: 0.42, Sn: 0.33, In: 0.29,
  W: 0.72, Re: 0.70, Os: 0.68, Ir: 0.65, Pt: 0.62, Au: 0.58,
  Ta: 0.55, Hf: 0.50, Lu: 0.48, La: 0.30, Ce: 0.32, Pr: 0.33,
  Nd: 0.34, Sm: 0.36, Gd: 0.38, Dy: 0.40, Er: 0.42, Yb: 0.44,
  U: 0.45, Th: 0.25, Pu: 0.50, Np: 0.42, Am: 0.48,
  Ba: 0.22, Cs: 0.20, Rb: 0.08, Sr: 0.10,
  Mo: 0.15, Nb: 0.12, Zr: 0.10, Y: 0.09,
  Se: 0.18, As: 0.14, Ge: 0.11, Ga: 0.09,
  Cu: 0.04, Ni: 0.04, Co: 0.03, Fe: 0.03, Mn: 0.02,
  Cr: 0.02, V: 0.02, Ti: 0.02, Sc: 0.01,
  Si: 0.01, Al: 0.01, Mg: 0.005, Na: 0.003,
  B: 0.001, C: 0.001, N: 0.001, O: 0.001, F: 0.001,
  H: 0.0001, Li: 0.001, Be: 0.001, K: 0.01, Ca: 0.02,
  S: 0.05, P: 0.03, Cl: 0.02, Br: 0.12, I: 0.30,
  Ag: 0.15, Pd: 0.12, Rh: 0.10, Ru: 0.09,
  Zn: 0.05,
};

const TOPO_MATERIAL_PATTERNS: { elements: string[]; minCount: number; bonus: number; label: string }[] = [
  { elements: ["Bi", "Se"], minCount: 2, bonus: 0.35, label: "Bi2Se3-class TI" },
  { elements: ["Bi", "Te"], minCount: 2, bonus: 0.35, label: "Bi2Te3-class TI" },
  { elements: ["Sb", "Te"], minCount: 2, bonus: 0.30, label: "Sb2Te3-class TI" },
  { elements: ["Bi", "Sb"], minCount: 2, bonus: 0.25, label: "BiSb alloy TI" },
  { elements: ["Sn", "Te"], minCount: 2, bonus: 0.25, label: "SnTe-class TCI" },
  { elements: ["Pb", "Te"], minCount: 2, bonus: 0.20, label: "PbTe-class" },
  { elements: ["Pb", "Sn", "Se"], minCount: 3, bonus: 0.30, label: "PbSnSe TCI" },
  { elements: ["Hf", "Te"], minCount: 2, bonus: 0.20, label: "HfTe-type Weyl" },
  { elements: ["W", "Te"], minCount: 2, bonus: 0.25, label: "WTe2-type Weyl" },
  { elements: ["Ta", "As"], minCount: 2, bonus: 0.25, label: "TaAs-type Weyl" },
  { elements: ["Nb", "As"], minCount: 2, bonus: 0.20, label: "NbAs-type Weyl" },
  { elements: ["Cu", "Bi", "Se"], minCount: 3, bonus: 0.40, label: "CuxBi2Se3 TSC" },
  { elements: ["Sr", "Ru", "O"], minCount: 3, bonus: 0.30, label: "Sr2RuO4-type TSC" },
  { elements: ["Fe", "Te", "Se"], minCount: 3, bonus: 0.30, label: "FeTeSe TSC candidate" },
  { elements: ["Ir"], minCount: 1, bonus: 0.15, label: "Iridate SOC system" },
  { elements: ["Os"], minCount: 1, bonus: 0.12, label: "Osmate SOC system" },
  { elements: ["U", "Te"], minCount: 2, bonus: 0.35, label: "UTe2-type triplet TSC" },
];

const LAYERED_INDICATORS = ["P6/mmm", "P4/mmm", "I4/mmm", "R-3m", "Pnma", "P-3m1", "P6_3/mmc", "C2/m", "P2_1/c", "Cmcm", "Fmmm", "P-1"];

function parseFormulaElements(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const cnt = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + cnt;
  }
  return counts;
}

const ATOMIC_NUMBERS: Record<string, number> = {
  H: 1, He: 2, Li: 3, Be: 4, B: 5, C: 6, N: 7, O: 8, F: 9, Ne: 10,
  Na: 11, Mg: 12, Al: 13, Si: 14, P: 15, S: 16, Cl: 17, Ar: 18,
  K: 19, Ca: 20, Sc: 21, Ti: 22, V: 23, Cr: 24, Mn: 25, Fe: 26, Co: 27, Ni: 28, Cu: 29, Zn: 30,
  Ga: 31, Ge: 32, As: 33, Se: 34, Br: 35, Kr: 36,
  Rb: 37, Sr: 38, Y: 39, Zr: 40, Nb: 41, Mo: 42, Tc: 43, Ru: 44, Rh: 45, Pd: 46, Ag: 47, Cd: 48,
  In: 49, Sn: 50, Sb: 51, Te: 52, I: 53, Xe: 54,
  Cs: 55, Ba: 56,
  La: 57, Ce: 58, Pr: 59, Nd: 60, Pm: 61, Sm: 62, Eu: 63, Gd: 64, Tb: 65, Dy: 66, Ho: 67, Er: 68, Tm: 69, Yb: 70, Lu: 71,
  Hf: 72, Ta: 73, W: 74, Re: 75, Os: 76, Ir: 77, Pt: 78, Au: 79, Hg: 80,
  Tl: 81, Pb: 82, Bi: 83, Po: 84, At: 85, Rn: 86,
  Fr: 87, Ra: 88,
  Ac: 89, Th: 90, Pa: 91, U: 92, Np: 93, Pu: 94, Am: 95, Cm: 96, Bk: 97, Cf: 98, Es: 99, Fm: 100, Md: 101, No: 102, Lr: 103,
  Rf: 104, Db: 105, Sg: 106, Bh: 107, Hs: 108, Mt: 109, Ds: 110, Rg: 111, Cn: 112, Nh: 113, Fl: 114, Mc: 115, Lv: 116, Ts: 117, Og: 118,
};

function computeSOCStrength(elements: Record<string, number>): number {
  const totalAtoms = Object.values(elements).reduce((a, b) => a + b, 0);
  if (totalAtoms === 0) return 0;

  const Z_REF = 83;
  let weightedSOC = 0;
  let maxSOC = 0;
  for (const [el, count] of Object.entries(elements)) {
    const Z = ATOMIC_NUMBERS[el] ?? 0;
    const zFourScaled = Math.pow(Z / Z_REF, 4);
    const tableFactor = HEAVY_ELEMENT_SOC[el] ?? zFourScaled;
    const soc = Math.max(zFourScaled, tableFactor);
    weightedSOC += soc * (count / totalAtoms);
    maxSOC = Math.max(maxSOC, soc);
  }

  return Math.min(1.0, weightedSOC * 1.5 + maxSOC * 0.3);
}

function estimateZ2Invariant(
  socStrength: number,
  bandInversionProb: number,
  orbitalFractions: { s: number; p: number; d: number; f: number },
  isLayered: boolean
): number {
  let z2 = 0;

  if (socStrength > 0.3 && bandInversionProb > 0.3) {
    z2 = Math.min(1.0, socStrength * bandInversionProb * 2);
  }

  const pdMixing = orbitalFractions.p * orbitalFractions.d;
  if (pdMixing > 0.05) {
    z2 += pdMixing * 0.5;
  }

  const spInversion = orbitalFractions.s * orbitalFractions.p;
  if (spInversion > 0.1 && socStrength > 0.2) {
    z2 += spInversion * 0.3;
  }

  if (isLayered && socStrength > 0.15) {
    z2 *= 1.2;
  }

  return Math.min(1.0, Math.max(0, z2));
}

function estimateChernIndicator(
  socStrength: number,
  hasDiracCrossing: boolean,
  hasBandInversion: boolean,
  magneticElements: boolean
): number {
  let chern = 0;

  if (hasBandInversion && socStrength > 0.2) {
    chern += 0.4;
  }

  if (hasDiracCrossing) {
    chern += 0.3;
  }

  if (magneticElements && socStrength > 0.3) {
    chern += 0.2;
  }

  chern *= Math.min(1.0, socStrength * 2);

  return Math.min(1.0, Math.max(0, chern));
}

function estimateMirrorSymmetryIndicator(
  crystalSystem: string | undefined,
  socStrength: number,
  z2: number
): number {
  let mirror = 0;

  const highSymSystems = ["cubic", "tetragonal", "hexagonal"];
  if (crystalSystem && highSymSystems.includes(crystalSystem.toLowerCase())) {
    mirror += 0.3;
  }

  if (z2 > 0.3 && socStrength > 0.2) {
    mirror += z2 * 0.4;
  }

  mirror *= Math.min(1.0, socStrength * 1.5);

  return Math.min(1.0, Math.max(0, mirror));
}

function estimateBandInversionProbability(
  socStrength: number,
  orbitalFractions: { s: number; p: number; d: number; f: number },
  topo?: TightBindingTopology
): number {
  if (topo?.hasBandInversion) {
    return Math.min(1.0, 0.6 + socStrength * 0.3);
  }

  let prob = 0;

  const pdMixing = orbitalFractions.p * orbitalFractions.d * 8;
  prob += Math.min(0.4, pdMixing * 0.3);

  if (socStrength > 0.3) {
    prob += (socStrength - 0.3) * 0.8;
  }

  const fContrib = orbitalFractions.f;
  if (fContrib > 0.1 && socStrength > 0.2) {
    prob += fContrib * 0.3;
  }

  return Math.min(1.0, Math.max(0, prob));
}

function estimateDiracNodeProbability(
  socStrength: number,
  bandInversionProb: number,
  topo?: TightBindingTopology
): number {
  if (topo?.hasDiracCrossing) {
    return Math.min(1.0, 0.5 + topo.diracCrossingCount * 0.15 + socStrength * 0.2);
  }

  let prob = 0;

  if (bandInversionProb > 0.3 && socStrength > 0.15) {
    prob += bandInversionProb * 0.5;
  }

  if (socStrength > 0.4) {
    prob += (socStrength - 0.4) * 0.4;
  }

  return Math.min(1.0, Math.max(0, prob));
}

function estimateFlatBandIndicator(
  electronic: ElectronicStructure,
  topo?: TightBindingTopology
): number {
  let indicator = electronic.flatBandIndicator;

  if (topo?.hasFlatBand) {
    indicator = Math.max(indicator, 0.5 + topo.flatBandCount * 0.1);
  }

  if (electronic.vanHoveProximity > 0.5) {
    indicator = Math.max(indicator, electronic.vanHoveProximity * 0.6);
  }

  return Math.min(1.0, indicator);
}

function estimateMajoranaFeasibility(
  z2: number,
  socStrength: number,
  bandInversionProb: number,
  diracProb: number,
  isLayered: boolean
): number {
  let feasibility = 0;

  if (z2 > 0.4 && socStrength > 0.3) {
    feasibility += z2 * 0.4;
  }

  if (bandInversionProb > 0.4) {
    feasibility += bandInversionProb * 0.25;
  }

  if (diracProb > 0.3) {
    feasibility += diracProb * 0.2;
  }

  if (isLayered) {
    feasibility *= 1.15;
  }

  feasibility *= Math.min(1.0, socStrength * 2);

  return Math.min(1.0, Math.max(0, feasibility));
}

function estimateMajoranaWithMetallicity(
  z2: number,
  socStrength: number,
  bandInversionProb: number,
  diracProb: number,
  isLayered: boolean,
  metallicity: number
): number {
  const baseFeasibility = estimateMajoranaFeasibility(z2, socStrength, bandInversionProb, diracProb, isLayered);
  const metallicWeight = Math.min(1.0, metallicity * 1.5);
  return baseFeasibility * (0.3 + 0.7 * metallicWeight);
}

function classifyTopologicalState(
  z2: number,
  chern: number,
  mirror: number,
  majorana: number,
  socStrength: number
): string {
  if (majorana > 0.5 && z2 > 0.5) return "topological-superconductor";
  if (z2 > 0.6 && socStrength > 0.4) return "strong-topological-insulator";
  if (chern > 0.5 && socStrength > 0.3) return "Chern-insulator";
  if (mirror > 0.5 && z2 > 0.3) return "topological-crystalline-insulator";
  if (z2 > 0.3 || chern > 0.3) return "weak-topological";
  if (socStrength > 0.2) return "SOC-enhanced";
  return "trivial";
}

export function analyzeTopology(
  formula: string,
  electronic: ElectronicStructure,
  spaceGroup?: string,
  crystalSystem?: string
): TopologicalAnalysis {
  const elements = parseFormulaElements(formula);
  const elementNames = Object.keys(elements);

  const socStrength = computeSOCStrength(elements);

  const topo = electronic.tightBindingTopology;

  const isLayered = spaceGroup ? LAYERED_INDICATORS.some(sg =>
    spaceGroup.includes(sg) || sg.includes(spaceGroup)
  ) : (crystalSystem === "hexagonal" || crystalSystem === "tetragonal" || crystalSystem === "monoclinic");

  const bandInversionProbability = estimateBandInversionProbability(
    socStrength, electronic.orbitalFractions, topo
  );

  const diracNodeProbability = estimateDiracNodeProbability(
    socStrength, bandInversionProbability, topo
  );

  const magneticElements = elementNames.some(el =>
    ["Fe", "Co", "Ni", "Mn", "Cr", "V"].includes(el)
  );

  const z2Score = estimateZ2Invariant(
    socStrength, bandInversionProbability, electronic.orbitalFractions, isLayered
  );

  const chernScore = estimateChernIndicator(
    socStrength,
    topo?.hasDiracCrossing ?? false,
    topo?.hasBandInversion ?? false,
    magneticElements
  );

  const mirrorSymmetryIndicator = estimateMirrorSymmetryIndicator(
    crystalSystem, socStrength, z2Score
  );

  const flatBandIndicator = estimateFlatBandIndicator(electronic, topo);

  const majoranaFeasibility = estimateMajoranaWithMetallicity(
    z2Score, socStrength, bandInversionProbability, diracNodeProbability, isLayered,
    electronic.metallicity ?? 0.5
  );

  let patternBonus = 0;
  const indicators: string[] = [];

  for (const pattern of TOPO_MATERIAL_PATTERNS) {
    const matchCount = pattern.elements.filter(el => elementNames.includes(el)).length;
    if (matchCount >= pattern.minCount) {
      patternBonus = Math.max(patternBonus, pattern.bonus);
      indicators.push(pattern.label);
    }
  }

  if (socStrength > 0.5) indicators.push("strong SOC");
  if (bandInversionProbability > 0.4) indicators.push("band inversion likely");
  if (diracNodeProbability > 0.3) indicators.push("Dirac nodes probable");
  if (flatBandIndicator > 0.5) indicators.push("flat band present");
  if (isLayered) indicators.push("layered structure");
  if (magneticElements) indicators.push("magnetic elements");
  if (z2Score > 0.5) indicators.push("Z2 nontrivial");
  if (chernScore > 0.3) indicators.push("nonzero Chern indicator");
  if (majoranaFeasibility > 0.3) indicators.push("Majorana-hosting potential");

  const socContribution = socStrength;
  const bandInversionContribution = bandInversionProbability;
  const symmetryContribution = (z2Score * 0.4 + chernScore * 0.3 + mirrorSymmetryIndicator * 0.3);
  const flatBandContribution = flatBandIndicator;

  let topologicalScore =
    0.30 * socContribution +
    0.25 * bandInversionContribution +
    0.25 * symmetryContribution +
    0.20 * flatBandContribution;

  topologicalScore = Math.min(1.0, topologicalScore + patternBonus * 0.3);

  const topologicalClass = classifyTopologicalState(
    z2Score, chernScore, mirrorSymmetryIndicator, majoranaFeasibility, socStrength
  );

  return {
    topologicalScore: Math.round(topologicalScore * 1000) / 1000,
    z2Score: Math.round(z2Score * 1000) / 1000,
    chernScore: Math.round(chernScore * 1000) / 1000,
    mirrorSymmetryIndicator: Math.round(mirrorSymmetryIndicator * 1000) / 1000,
    socStrength: Math.round(socStrength * 1000) / 1000,
    bandInversionProbability: Math.round(bandInversionProbability * 1000) / 1000,
    diracNodeProbability: Math.round(diracNodeProbability * 1000) / 1000,
    flatBandIndicator: Math.round(flatBandIndicator * 1000) / 1000,
    majoranaFeasibility: Math.round(majoranaFeasibility * 1000) / 1000,
    topologicalClass,
    indicators,
    details: {
      socContribution: Math.round(socContribution * 1000) / 1000,
      bandInversionContribution: Math.round(bandInversionContribution * 1000) / 1000,
      symmetryContribution: Math.round(symmetryContribution * 1000) / 1000,
      flatBandContribution: Math.round(flatBandContribution * 1000) / 1000,
    },
  };
}

let totalAnalyzed = 0;
let totalTopological = 0;
let totalTSC = 0;
let classBreakdown: Record<string, number> = {};

export function trackTopologyResult(analysis: TopologicalAnalysis) {
  totalAnalyzed++;
  if (analysis.topologicalScore > 0.3) totalTopological++;
  if (analysis.topologicalClass === "topological-superconductor") totalTSC++;
  classBreakdown[analysis.topologicalClass] = (classBreakdown[analysis.topologicalClass] || 0) + 1;
}

export function getTopologyStats() {
  return {
    totalAnalyzed,
    totalTopological,
    totalTSC,
    topologicalRate: totalAnalyzed > 0 ? Math.round(totalTopological / totalAnalyzed * 1000) / 1000 : 0,
    tscRate: totalAnalyzed > 0 ? Math.round(totalTSC / totalAnalyzed * 1000) / 1000 : 0,
    classBreakdown,
  };
}
