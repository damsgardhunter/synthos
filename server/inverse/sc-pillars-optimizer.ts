import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  computeDynamicSpinSusceptibility,
  evaluateCompetingPhases,
  classifyHydrogenBonding,
  computeDimensionalityScore,
  parseFormulaElements,
} from "../learning/physics-engine";
import {
  ELEMENTAL_DATA,
  getElementData,
  isTransitionMetal,
  isRareEarth,
} from "../learning/elemental-data";
import { computeFullTightBinding } from "../learning/tight-binding";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";
import { checkPhysicsConstraints } from "./physics-constraint-engine";

export interface SCPillarTargets {
  minLambda: number;
  minOmegaLogK: number;
  minDOS: number;
  minNesting: number;
  minFlatBand: number;
  minPairingGlue: number;
  minInstability: number;
  minHydrogenCage: number;
  preferredMotifs: ("layered" | "cage" | "kagome" | "A15" | "perovskite")[];
}

export const DEFAULT_PILLAR_TARGETS: SCPillarTargets = {
  minLambda: 1.5,
  minOmegaLogK: 700,
  minDOS: 2.0,
  minNesting: 0.5,
  minFlatBand: 0.5,
  minPairingGlue: 0.5,
  minInstability: 0.4,
  minHydrogenCage: 0.5,
  preferredMotifs: ["cage", "layered", "kagome"],
};

export interface PairingGlueBreakdown {
  electronPhononContribution: number;
  spinFluctuationContribution: number;
  chargeFluctuationContribution: number;
  excitonicContribution: number;
  compositePairingGlue: number;
  dominantMechanism: string;
}

export interface InstabilityBreakdown {
  vanHoveProximity: number;
  nestingContribution: number;
  dosContribution: number;
  spinSusceptibility: number;
  mottProximity: number;
  cdwSusceptibility: number;
  sdwSusceptibility: number;
  structuralInstability: number;
  compositeInstability: number;
  dominantInstability: string;
}

export interface HydrogenCageMetrics {
  hydrogenNetworkDimensionality: number;
  hydrogenCageScore: number;
  hhBondDistribution: number;
  cageSymmetry: number;
  hCoordination: number;
  bondingType: string;
  compositeHydrogenScore: number;
  isHydride: boolean;
}

export interface FermiSurfaceGeometry {
  cylindricalScore: number;
  kzVariance: number;
  fsDimensionality: number;
  nestingStrength: number;
  electronHolePocketOverlap: number;
  nestingVectorQ: string;
  vanHoveDistance: number;
  vanHoveSaddleCount: number;
  vanHoveNearFermi: boolean;
  compositeFSScore: number;
  dominantGeometry: string;
}

export interface PillarEvaluation {
  formula: string;
  lambda: number;
  omegaLogK: number;
  dos: number;
  nestingScore: number;
  flatBandScore: number;
  motifMatch: string;
  motifScore: number;
  pairingGlue: PairingGlueBreakdown;
  instability: InstabilityBreakdown;
  hydrogenCage: HydrogenCageMetrics;
  fermiSurface: FermiSurfaceGeometry;
  pillarScores: {
    coupling: number;
    phonon: number;
    dos: number;
    nesting: number;
    structure: number;
    pairingGlue: number;
    instability: number;
    hydrogenCage: number;
  };
  compositeFitness: number;
  satisfiedPillars: number;
  weakestPillar: string;
  tcPredicted: number;
  metallicity: number;
  muStar: number;
  physicsValid: boolean;
}

export interface MutationLineage {
  parentFormula: string;
  targetedPillar: string;
  fitnessImprovement: number;
  generation: number;
}

export interface PillarGuidedCandidate {
  formula: string;
  evaluation: PillarEvaluation;
  designRationale: string;
  lineage?: MutationLineage;
}

export interface PillarOptimizerStats {
  totalEvaluated: number;
  totalGenerated: number;
  avgCompositeFitness: number;
  bestFitness: number;
  bestFormula: string;
  pillarSatisfactionRates: Record<string, number>;
  familySatisfactionRates: Record<string, Record<string, number>>;
  elementAffinityScores: Record<string, number>;
  elementSurpriseFactors: Record<string, number>;
  topCandidates: { formula: string; fitness: number; tc: number; pillars: number }[];
  pillarWeights: Record<string, number>;
}

const LIGHT_ATOMS = ["H", "B", "C", "N"];
const HIGH_COUPLING_TM = ["Nb", "V", "Ta", "Mo", "Ti", "Zr", "Hf", "W"];
const CAGE_FORMERS = ["La", "Y", "Ca", "Sr", "Ba", "Sc", "Ce", "Th"];
const LAYER_FORMERS = ["Cu", "Fe", "Ni", "Co", "Mn"];
const KAGOME_ELEMENTS = ["V", "Mn", "Fe", "Co", "Nb"];
const PNICTOGEN_ELEMENTS = ["As", "P", "Sb", "Bi"];
const CHALCOGEN_ELEMENTS = ["S", "Se", "Te"];

const SODALITE_CAGE_ELEMENTS = ["La", "Y", "Ce", "Th", "Ac", "Ca", "Sr", "Ba"];
const CLATHRATE_CAGE_ELEMENTS = ["La", "Y", "Ca", "Ba", "Sr", "Sc", "Ce"];

function parseCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    const el = m[1];
    const num = m[2] ? parseFloat(m[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function countsToFormula(counts: Record<string, number>): string {
  const sorted = Object.entries(counts)
    .filter(([, n]) => n > 0)
    .sort(([a], [b]) => {
      const enA = ELEMENTAL_DATA[a]?.paulingElectronegativity ?? 2.0;
      const enB = ELEMENTAL_DATA[b]?.paulingElectronegativity ?? 2.0;
      if (enA !== enB) return enA - enB;
      return a.localeCompare(b);
    });
  return sorted.map(([el, n]) => {
    if (Number.isInteger(n)) return n === 1 ? el : `${el}${n}`;
    if (n < 1) return `${el}${parseFloat(n.toFixed(2))}`;
    const rounded = Math.round(n);
    if (Math.abs(n - rounded) < 0.01) return rounded === 1 ? el : `${el}${rounded}`;
    return `${el}${parseFloat(n.toFixed(1))}`;
  }).join("");
}

const CAGE_FORMING_METALS = new Set([
  "La", "Y", "Sc", "Ca", "Sr", "Ba", "Ce", "Pr", "Nd", "Th", "Ac",
  "Lu", "Gd", "Eu", "Sm", "Yb", "Dy", "Ho", "Er", "Tm",
]);

function getHRatio(counts: Record<string, number>): number {
  const hCount = counts["H"] || 0;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  return hCount / Math.max(1, totalAtoms - hCount);
}

function detectMotif(formula: string, elements: string[], counts: Record<string, number>): { match: string; score: number } {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const hCount = counts["H"] || 0;
  const nonHElements = elements.filter(e => e !== "H");
  const hRatio = getHRatio(counts);

  if (hCount > 0 && hRatio >= 2) {
    const hasCageFormer = nonHElements.some(e => CAGE_FORMING_METALS.has(e));
    if (hRatio >= 6 && hasCageFormer) return { match: "cage-clathrate", score: 0.95 };
    if (hRatio >= 4 && hasCageFormer) return { match: "cage-clathrate", score: 0.90 };
    if (hRatio >= 4) return { match: "cage-clathrate", score: 0.80 };
    if (nonHElements.length === 1 && hRatio < 4) return { match: "covalent-hydride", score: 0.85 };
    return { match: "layered-hydride", score: 0.85 };
  }

  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  if (hasCu && hasO && elements.length >= 3) return { match: "layered-cuprate", score: 0.90 };

  const hasFeOrCo = elements.includes("Fe") || elements.includes("Co");
  const hasPnictogen = elements.some(e => PNICTOGEN_ELEMENTS.includes(e));
  const hasChalcogen = elements.some(e => CHALCOGEN_ELEMENTS.includes(e));
  if (hasFeOrCo && (hasPnictogen || hasChalcogen)) return { match: "layered-pnictide", score: 0.85 };

  const hasKagomeEl = elements.some(e => KAGOME_ELEMENTS.includes(e));
  const hasSb = elements.includes("Sb");
  if (hasKagomeEl && hasSb && elements.length >= 3) return { match: "kagome", score: 0.80 };

  if (elements.length === 2) {
    const tmEls = elements.filter(e => isTransitionMetal(e) && !CAGE_FORMING_METALS.has(e));
    const nonTmEls = elements.filter(e => !isTransitionMetal(e));
    if (tmEls.length === 1 && nonTmEls.length === 1) {
      const tmCount = counts[tmEls[0]] || 0;
      const nonTmCount = counts[nonTmEls[0]] || 0;
      if (tmCount === 3 && nonTmCount === 1) return { match: "A15", score: 0.88 };
    }
  }

  const hasRE = elements.some(e => isRareEarth(e));
  if (hasRE && hasO) return { match: "perovskite", score: 0.70 };

  const hasTM = elements.some(e => isTransitionMetal(e));
  const hasLight = elements.some(e => LIGHT_ATOMS.includes(e));
  if (hasTM && hasLight) return { match: "metal-light-bond", score: 0.60 };

  return { match: "generic", score: 0.30 };
}

function scorePillar(value: number, target: number, softness: number = 0.5): number {
  if (value >= target) return Math.min(1.0, 0.85 + 0.15 * Math.min(value / target, 2.0));
  const ratio = value / Math.max(target, 0.001);
  return Math.pow(ratio, 1.0 + softness);
}

function computePairingGlue(
  formula: string,
  lambda: number,
  electronic: ReturnType<typeof computeElectronicStructure>,
): PairingGlueBreakdown {
  const elements = parseFormulaElements(formula);
  const spin = computeDynamicSpinSusceptibility(formula, electronic);

  const phononContribution = lambda <= 2.0
    ? Math.min(1.0, lambda / 2.0)
    : Math.min(1.0, Math.sqrt(lambda) / Math.sqrt(2.0));

  let spinContribution = 0;
  if (spin.stonerEnhancement > 2.0) {
    spinContribution = Math.min(1.0, (spin.stonerEnhancement - 1) / 10);
  }
  if (spin.isNearQCP) {
    spinContribution = Math.max(spinContribution, 0.7);
  }
  const hasMagnetic = elements.some(e => ["Fe", "Co", "Ni", "Mn", "Cr", "Cu"].includes(e));
  const hasPnictogenOrChalcogen = elements.some(e =>
    PNICTOGEN_ELEMENTS.includes(e) || CHALCOGEN_ELEMENTS.includes(e) || e === "O"
  );
  if (hasMagnetic && hasPnictogenOrChalcogen) {
    spinContribution = Math.max(spinContribution, 0.4 + electronic.correlationStrength * 0.4);
  }

  let chargeContribution = 0;
  const nestingVal = electronic.nestingScore ?? 0;
  if (nestingVal > 0.5 && electronic.densityOfStatesAtFermi > 2.0) {
    chargeContribution = Math.min(1.0, nestingVal * 0.5 + electronic.densityOfStatesAtFermi * 0.05);
  }

  let excitonicContribution = 0;
  if (electronic.correlationStrength > 0.5 && electronic.metallicity > 0.05) {
    const mottScore = electronic.mottProximityScore ?? 0;
    if (mottScore > 0.3) {
      const metalFactor = electronic.metallicity < 0.3
        ? 0.5 + 0.5 * (electronic.metallicity / 0.3)
        : 1.0;
      excitonicContribution = Math.min(0.8, (mottScore * 0.6 + electronic.correlationStrength * 0.2) * metalFactor);
    }
  }

  const compositePairingGlue =
    0.50 * phononContribution +
    0.25 * spinContribution +
    0.10 * chargeContribution +
    0.15 * excitonicContribution;

  let dominantMechanism = "phonon";
  const rawContributions = [
    { name: "phonon", val: phononContribution },
    { name: "spin-fluctuation", val: spinContribution },
    { name: "charge-fluctuation", val: chargeContribution },
    { name: "excitonic", val: excitonicContribution },
  ];
  rawContributions.sort((a, b) => b.val - a.val);
  if (rawContributions[0].val > 0) dominantMechanism = rawContributions[0].name;

  return {
    electronPhononContribution: Number(phononContribution.toFixed(4)),
    spinFluctuationContribution: Number(spinContribution.toFixed(4)),
    chargeFluctuationContribution: Number(chargeContribution.toFixed(4)),
    excitonicContribution: Number(excitonicContribution.toFixed(4)),
    compositePairingGlue: Number(compositePairingGlue.toFixed(4)),
    dominantMechanism,
  };
}

function computeInstabilityProximity(
  formula: string,
  electronic: ReturnType<typeof computeElectronicStructure>,
  lambda: number,
  pressureGPa?: number,
): InstabilityBreakdown {
  const elements = parseFormulaElements(formula);
  const spin = computeDynamicSpinSusceptibility(formula, electronic);
  const competing = evaluateCompetingPhases(formula, electronic);

  const vanHove = Math.min(1.0, electronic.vanHoveProximity ?? 0);
  const nestingVal = Math.min(1.0, electronic.nestingScore ?? 0);
  const dosVal = Math.min(1.0, electronic.densityOfStatesAtFermi / 5.0);
  const spinSusc = Math.min(1.0, (spin.stonerEnhancement - 1) / 15);

  const mottProx = Math.min(1.0, electronic.mottProximityScore ?? 0);

  let cdwSusc = 0;
  const hasDichalcogenide = elements.some(e => CHALCOGEN_ELEMENTS.includes(e)) &&
    elements.some(e => ["Nb", "Ta", "Ti", "V", "Mo", "W"].includes(e));
  if (hasDichalcogenide && nestingVal > 0.4) {
    cdwSusc = Math.min(1.0, nestingVal * 0.6 + dosVal * 0.4);
  }
  if (electronic.flatBandIndicator > 0.5 && nestingVal > 0.3) {
    cdwSusc = Math.max(cdwSusc, electronic.flatBandIndicator * 0.5);
  }

  let sdwSusc = 0;
  const hasMagnetic = elements.some(e => ["Fe", "Cr", "Mn", "Co", "V"].includes(e));
  if (hasMagnetic && nestingVal > 0.5) {
    sdwSusc = Math.min(1.0, spinSusc * 0.5 + nestingVal * 0.3 + electronic.correlationStrength * 0.2);
  }
  for (const phase of competing) {
    if (phase.type === "magnetism" && !phase.suppressesSC) {
      sdwSusc = Math.max(sdwSusc, phase.strength * 0.7);
    }
  }

  let structInstab = 0;
  const hasH = elements.includes("H");
  if (hasH) {
    const counts = parseCounts(formula);
    const hRatio = getHRatio(counts);
    const pressureSupport = Math.min(1.0, (pressureGPa ?? 0) / 150);
    if (hRatio >= 4) {
      structInstab = Math.max(0, 0.8 - pressureSupport);
    }
    if (lambda > 3.5 && hRatio < 4) {
      structInstab = Math.max(structInstab, Math.min(1.0, 0.4 + (lambda - 3.5) * 0.3));
    } else if (lambda > 3.5 && hRatio >= 4) {
      const pressureMitigation = pressureSupport * 0.4;
      structInstab = Math.max(structInstab, Math.min(0.5, (lambda - 3.5) * 0.15 - pressureMitigation));
    }
    if (lambda > 2.0) structInstab = Math.max(structInstab, Math.min(1.0, (lambda - 2.0) * 0.2));
  }
  if (electronic.flatBandIndicator > 0.6) {
    structInstab = Math.max(structInstab, electronic.flatBandIndicator * 0.4);
  }

  const compositeInstability =
    0.20 * vanHove +
    0.15 * nestingVal +
    0.10 * dosVal +
    0.15 * Math.max(spinSusc, 0) +
    0.15 * mottProx +
    0.10 * cdwSusc +
    0.10 * sdwSusc +
    0.05 * structInstab;

  const instabilities = [
    { name: "van-Hove", val: vanHove },
    { name: "Mott", val: mottProx },
    { name: "CDW", val: cdwSusc },
    { name: "SDW", val: sdwSusc },
    { name: "structural", val: structInstab },
    { name: "nesting", val: nestingVal },
  ];
  instabilities.sort((a, b) => b.val - a.val);
  const dominantInstability = instabilities[0].val > 0.1 ? instabilities[0].name : "none";

  return {
    vanHoveProximity: Number(vanHove.toFixed(4)),
    nestingContribution: Number(nestingVal.toFixed(4)),
    dosContribution: Number(dosVal.toFixed(4)),
    spinSusceptibility: Number(Math.max(0, spinSusc).toFixed(4)),
    mottProximity: Number(mottProx.toFixed(4)),
    cdwSusceptibility: Number(cdwSusc.toFixed(4)),
    sdwSusceptibility: Number(sdwSusc.toFixed(4)),
    structuralInstability: Number(structInstab.toFixed(4)),
    compositeInstability: Number(compositeInstability.toFixed(4)),
    dominantInstability,
  };
}

function computeHydrogenCageMetrics(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
): HydrogenCageMetrics {
  const hCount = counts["H"] || 0;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const nonHAtoms = totalAtoms - hCount;

  if (hCount === 0) {
    return {
      hydrogenNetworkDimensionality: 0,
      hydrogenCageScore: 0,
      hhBondDistribution: 0,
      cageSymmetry: 0,
      hCoordination: 0,
      bondingType: "none",
      compositeHydrogenScore: 0,
      isHydride: false,
    };
  }

  const hRatio = getHRatio(counts);
  const bondingType = classifyHydrogenBonding(formula, hRatio >= 6 ? 150 : hRatio >= 4 ? 100 : 50);

  const networkDim = 1.0 + 2.0 / (1.0 + Math.exp(-1.5 * (hRatio - 3)));

  let cageScore = 0;
  const hasSodaliteFormer = elements.some(e => SODALITE_CAGE_ELEMENTS.includes(e));
  const hasClathFormer = elements.some(e => CLATHRATE_CAGE_ELEMENTS.includes(e));

  if (hRatio >= 6 && hasSodaliteFormer) {
    cageScore = 0.95;
    if (hRatio >= 8 && hRatio <= 12) cageScore = 1.0;
  } else if (hRatio >= 4 && hasClathFormer) {
    cageScore = 0.80;
    if (hRatio >= 6) cageScore = 0.90;
  } else if (hRatio >= 4) {
    cageScore = 0.60;
  } else if (hRatio >= 2) {
    cageScore = 0.35;
  } else {
    cageScore = 0.15;
  }

  let hhBondDist = 0;
  if (hRatio >= 6) {
    hhBondDist = 0.90;
    if (nonHAtoms === 1) hhBondDist = 0.95;
  } else if (hRatio >= 4) {
    hhBondDist = 0.70;
  } else if (hRatio >= 2) {
    hhBondDist = 0.50;
  } else {
    hhBondDist = 0.20;
  }

  let cageSymmetry = 0;
  if (cageScore >= 0.8) {
    const metalCount = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) ||
      SODALITE_CAGE_ELEMENTS.includes(e)).length;
    const cageFormingCount = elements.filter(e =>
      SODALITE_CAGE_ELEMENTS.includes(e) || CLATHRATE_CAGE_ELEMENTS.includes(e) || CAGE_FORMING_METALS.has(e)).length;
    const isDoubleCage = cageFormingCount >= 2;
    if (metalCount === 1 && nonHAtoms <= 2) {
      cageSymmetry = 0.95;
    } else if (isDoubleCage && metalCount <= 3) {
      cageSymmetry = 0.90;
    } else if (metalCount <= 2) {
      cageSymmetry = 0.75;
    } else {
      cageSymmetry = 0.50;
    }
  } else if (cageScore >= 0.5) {
    cageSymmetry = 0.40;
  }

  let hCoordination: number;
  if (hRatio >= 8 && hasSodaliteFormer) {
    hCoordination = 12;
  } else if (hRatio >= 6 && hasSodaliteFormer) {
    hCoordination = 9;
  } else if (hRatio >= 6 && hasClathFormer) {
    hCoordination = 8;
  } else if (hRatio >= 4) {
    hCoordination = 6;
  } else if (hRatio >= 2) {
    hCoordination = 4;
  } else if (hRatio >= 1) {
    hCoordination = 3;
  } else {
    hCoordination = 2;
  }

  const compositeHydrogenScore =
    0.25 * (networkDim / 3.0) +
    0.35 * cageScore +
    0.20 * hhBondDist +
    0.20 * cageSymmetry;

  return {
    hydrogenNetworkDimensionality: Number(networkDim.toFixed(2)),
    hydrogenCageScore: Number(cageScore.toFixed(4)),
    hhBondDistribution: Number(hhBondDist.toFixed(4)),
    cageSymmetry: Number(cageSymmetry.toFixed(4)),
    hCoordination,
    bondingType,
    compositeHydrogenScore: Number(compositeHydrogenScore.toFixed(4)),
    isHydride: hCount > 0,
  };
}

function computeFermiSurfaceGeometry(
  formula: string,
  electronic: ReturnType<typeof computeElectronicStructure>,
  elements: string[],
): FermiSurfaceGeometry {
  const dimScore = computeDimensionalityScore(formula, null);

  const fsTopology = electronic.fermiSurfaceTopology;
  const is2D = fsTopology.includes("2D") || fsTopology.includes("cylindrical");
  const hasNestingFS = fsTopology.includes("nesting");
  const hasMultiBand = fsTopology.includes("multi-band") || fsTopology.includes("multi-sheet");
  const hasPockets = fsTopology.includes("pocket") || fsTopology.includes("hole");

  let cylindricalScore = dimScore;
  if (is2D) cylindricalScore = Math.max(cylindricalScore, 0.85);

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.some(e => ["Fe", "Co", "Ni"].includes(e)) &&
    elements.some(e => PNICTOGEN_ELEMENTS.includes(e) || CHALCOGEN_ELEMENTS.includes(e));
  const isDichalcogenide = elements.some(e => ["Nb", "Ta", "Mo", "W"].includes(e)) &&
    elements.some(e => CHALCOGEN_ELEMENTS.includes(e));
  const isNickelate = elements.includes("Ni") && elements.includes("O") && elements.length >= 3;

  if (isCuprate) cylindricalScore = Math.max(cylindricalScore, 0.95);
  else if (isPnictide) cylindricalScore = Math.max(cylindricalScore, 0.90);
  else if (isDichalcogenide) cylindricalScore = Math.max(cylindricalScore, 0.85);
  else if (isNickelate) cylindricalScore = Math.max(cylindricalScore, 0.88);

  const kzVariance = Math.max(0, 1.0 - cylindricalScore);
  const fsDimensionality = 2.0 + 1.0 / (1.0 + Math.exp(10 * (cylindricalScore - 0.5)));

  let nestingStrength = electronic.nestingScore ?? 0;
  let electronHolePocketOverlap = 0;
  let nestingVectorQ = "none";

  if (isPnictide) {
    nestingStrength = 0.6 * nestingStrength + 0.4 * 0.80;
    electronHolePocketOverlap = Math.min(1.0, nestingStrength * 0.9);
    nestingVectorQ = "(pi,pi)";
  } else if (isCuprate) {
    nestingStrength = 0.55 * nestingStrength + 0.45 * 0.85;
    electronHolePocketOverlap = Math.min(1.0, nestingStrength * 0.85);
    nestingVectorQ = "(pi,pi)";
  } else if (isNickelate) {
    nestingStrength = 0.6 * nestingStrength + 0.4 * 0.75;
    electronHolePocketOverlap = Math.min(1.0, nestingStrength * 0.80);
    nestingVectorQ = "(pi,pi)";
  } else if (isDichalcogenide) {
    nestingStrength = 0.65 * nestingStrength + 0.35 * 0.65;
    electronHolePocketOverlap = Math.min(1.0, nestingStrength * 0.7);
    nestingVectorQ = "(2/3pi,0)";
  }

  if (hasMultiBand && hasPockets) {
    electronHolePocketOverlap = Math.max(electronHolePocketOverlap, 0.5);
    if (nestingVectorQ === "none") nestingVectorQ = "(pi,0)";
  }
  if (hasNestingFS) {
    nestingStrength = Math.max(nestingStrength, 0.6);
    electronHolePocketOverlap = Math.max(electronHolePocketOverlap, 0.4);
  }

  const correlation = electronic.correlationStrength ?? 0;
  if (correlation > 0.5 && nestingStrength > 0.4) {
    nestingStrength = Math.min(1.0, nestingStrength + correlation * 0.15);
  }

  let vanHoveDistance = 1.0;
  let vanHoveSaddleCount = 0;
  let vanHoveNearFermi = false;

  const vanHoveProx = electronic.vanHoveProximity ?? 0;
  if (vanHoveProx > 0) {
    vanHoveDistance = Math.exp(-3.0 * vanHoveProx);
  }

  try {
    const tb = computeFullTightBinding(formula, null);
    if (tb.bands.tbConfidence > 0.3 && tb.topology.vanHoveSingularities.length > 0) {
      vanHoveSaddleCount = tb.topology.vanHoveSingularities.length;

      let minDist = Infinity;
      for (const vhs of tb.topology.vanHoveSingularities) {
        const dist = Math.abs(vhs.energy - tb.bands.fermiEnergy);
        if (dist < minDist) minDist = dist;
      }
      const tbDistance = Math.min(1.0, minDist);
      vanHoveDistance = Math.min(vanHoveDistance, tbDistance);
      vanHoveNearFermi = vanHoveDistance < 0.05;

      if (vanHoveNearFermi) {
        vanHoveDistance = Math.min(vanHoveDistance, 0.01 + vanHoveSaddleCount * 0.002);
      }
    }
  } catch {}

  if (isCuprate) {
    vanHoveNearFermi = true;
    vanHoveDistance = Math.min(vanHoveDistance, 0.02);
    vanHoveSaddleCount = Math.max(vanHoveSaddleCount, 2);
  }

  const cylindricalContrib = cylindricalScore;
  const nestingContrib = nestingStrength;
  const vhContrib = Math.min(1.0, 1.0 / (1.0 + vanHoveDistance * 20));

  const compositeFSScore =
    0.35 * cylindricalContrib +
    0.35 * nestingContrib +
    0.30 * vhContrib;

  let dominantGeometry = "spherical";
  const geoScores = [
    { name: "cylindrical-2D", val: cylindricalContrib },
    { name: "nested-pockets", val: nestingContrib },
    { name: "van-Hove-saddle", val: vhContrib },
  ];
  geoScores.sort((a, b) => b.val - a.val);
  if (geoScores[0].val > 0.3) dominantGeometry = geoScores[0].name;

  return {
    cylindricalScore: Number(cylindricalScore.toFixed(4)),
    kzVariance: Number(kzVariance.toFixed(4)),
    fsDimensionality: Number(fsDimensionality.toFixed(2)),
    nestingStrength: Number(nestingStrength.toFixed(4)),
    electronHolePocketOverlap: Number(electronHolePocketOverlap.toFixed(4)),
    nestingVectorQ,
    vanHoveDistance: Number(vanHoveDistance.toFixed(4)),
    vanHoveSaddleCount,
    vanHoveNearFermi,
    compositeFSScore: Number(compositeFSScore.toFixed(4)),
    dominantGeometry,
  };
}

export class PillarOptimizerContext {
  pillarWeights = {
    coupling: 0.18,
    phonon: 0.12,
    dos: 0.12,
    nesting: 0.10,
    structure: 0.10,
    pairingGlue: 0.18,
    instability: 0.10,
    hydrogenCage: 0.10,
  };
  totalEvaluated = 0;
  totalGenerated = 0;
  fitnessSum = 0;
  bestFitness = 0;
  bestFormula = "";
  bestTc = 0;
  pillarSatisfied: Record<string, number> = {
    coupling: 0, phonon: 0, dos: 0, nesting: 0, structure: 0,
    pairingGlue: 0, instability: 0, hydrogenCage: 0,
  };
  elementAffinity: Record<string, { totalFitness: number; count: number }> = {};
  elementAffinityHistory: Record<string, number[]> = {};
  topCandidates: { formula: string; fitness: number; tc: number; pillars: number }[] = [];
  familySatisfied: Record<string, Record<string, number>> = {};
  familyCounts: Record<string, number> = {};

  reset() {
    this.totalEvaluated = 0;
    this.totalGenerated = 0;
    this.fitnessSum = 0;
    this.bestFitness = 0;
    this.bestFormula = "";
    this.bestTc = 0;
    this.pillarSatisfied = {
      coupling: 0, phonon: 0, dos: 0, nesting: 0, structure: 0,
      pairingGlue: 0, instability: 0, hydrogenCage: 0,
    };
    this.elementAffinity = {};
    this.elementAffinityHistory = {};
    this.topCandidates = [];
    this.familySatisfied = {};
    this.familyCounts = {};
  }
}

const defaultCtx = new PillarOptimizerContext();

export async function evaluatePillars(
  formula: string,
  targets: SCPillarTargets = DEFAULT_PILLAR_TARGETS,
  options?: { maxPressureGPa?: number; ctx?: PillarOptimizerContext },
): Promise<PillarEvaluation> {
  const ctx = options?.ctx ?? defaultCtx;
  ctx.totalEvaluated++;

  const elements = parseFormulaElements(formula);
  const counts = parseCounts(formula);

  const electronic = computeElectronicStructure(formula, null);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

  const lambda = coupling.lambda;
  const omegaLogK = coupling.omegaLog * 1.4388;
  const dos = electronic.densityOfStatesAtFermi;
  const nestingScore = electronic.nestingScore ?? 0;
  const flatBandScore = electronic.flatBandIndicator ?? 0;
  const muStar = coupling.muStar;
  const metallicity = electronic.metallicity;

  const motif = detectMotif(formula, elements, counts);
  const pairingGlue = computePairingGlue(formula, lambda, electronic);
  const instability = computeInstabilityProximity(formula, electronic, lambda, options?.maxPressureGPa);
  const hydrogenCage = computeHydrogenCageMetrics(formula, elements, counts);

  if (instability.cdwSusceptibility > 0.3) {
    const cdwSuppression = Math.exp(-2.0 * (instability.cdwSusceptibility - 0.3));
    pairingGlue.compositePairingGlue *= Math.max(0.25, cdwSuppression);
  }
  const fermiSurface = computeFermiSurfaceGeometry(formula, electronic, elements);

  const motifBonus = (targets.preferredMotifs ?? []).some(pm => motif.match.includes(pm)) ? 0.2 : 0;

  const isHydride = hydrogenCage.isHydride;
  const hydrogenCageWeight = isHydride ? ctx.pillarWeights.hydrogenCage : 0;

  const activeWeights = { ...ctx.pillarWeights };
  if (!isHydride) {
    activeWeights.hydrogenCage = 0;
    const freed = ctx.pillarWeights.hydrogenCage;
    const motifMatch = motif.match;
    if (motifMatch === "layered-cuprate" || motifMatch === "layered-pnictide") {
      activeWeights.pairingGlue += freed * 0.35;
      activeWeights.nesting += freed * 0.30;
      activeWeights.dos += freed * 0.15;
      activeWeights.instability += freed * 0.10;
      activeWeights.coupling += freed * 0.10;
    } else if (motifMatch === "kagome" || motifMatch === "flat-band") {
      activeWeights.dos += freed * 0.30;
      activeWeights.nesting += freed * 0.25;
      activeWeights.pairingGlue += freed * 0.25;
      activeWeights.instability += freed * 0.10;
      activeWeights.coupling += freed * 0.10;
    } else if (motifMatch === "A15") {
      activeWeights.coupling += freed * 0.35;
      activeWeights.phonon += freed * 0.30;
      activeWeights.dos += freed * 0.20;
      activeWeights.pairingGlue += freed * 0.15;
    } else if (motifMatch === "perovskite") {
      activeWeights.pairingGlue += freed * 0.30;
      activeWeights.instability += freed * 0.25;
      activeWeights.nesting += freed * 0.20;
      activeWeights.structure += freed * 0.15;
      activeWeights.coupling += freed * 0.10;
    } else {
      const redistrib = freed / 7;
      activeWeights.coupling += redistrib;
      activeWeights.phonon += redistrib;
      activeWeights.dos += redistrib;
      activeWeights.nesting += redistrib;
      activeWeights.structure += redistrib;
      activeWeights.pairingGlue += redistrib;
      activeWeights.instability += redistrib;
    }
  }

  const dosGate = Math.min(1.0, dos / 2.0);
  const fsNestingBoost = fermiSurface.nestingStrength > nestingScore
    ? (fermiSurface.nestingStrength - nestingScore) * 0.4 * dosGate
    : 0;
  const enhancedNesting = Math.min(1.0, nestingScore + fsNestingBoost);

  const vhBoost = fermiSurface.vanHoveNearFermi ? 0.15 : 0;
  const cylindricalBoost = fermiSurface.cylindricalScore > 0.7 ? 0.10 : 0;

  const pillarScores = {
    coupling: scorePillar(lambda, targets.minLambda, 0.6),
    phonon: scorePillar(omegaLogK, targets.minOmegaLogK, 0.4),
    dos: Math.min(1.0, scorePillar(dos, targets.minDOS, 0.5) + vhBoost),
    nesting: scorePillar(enhancedNesting, targets.minNesting, 0.3),
    structure: Math.min(1.0, motif.score + motifBonus + scorePillar(flatBandScore, targets.minFlatBand, 0.3) * 0.3 + cylindricalBoost),
    pairingGlue: Math.min(1.0, scorePillar(pairingGlue.compositePairingGlue, targets.minPairingGlue, 0.5) + fermiSurface.compositeFSScore * 0.1),
    instability: scorePillar(instability.compositeInstability, targets.minInstability, 0.4),
    hydrogenCage: isHydride ? scorePillar(hydrogenCage.compositeHydrogenScore, targets.minHydrogenCage, 0.5) : 0,
  };

  let compositeFitness =
    activeWeights.coupling * pillarScores.coupling +
    activeWeights.phonon * pillarScores.phonon +
    activeWeights.dos * pillarScores.dos +
    activeWeights.nesting * pillarScores.nesting +
    activeWeights.structure * pillarScores.structure +
    activeWeights.pairingGlue * pillarScores.pairingGlue +
    activeWeights.instability * pillarScores.instability +
    activeWeights.hydrogenCage * pillarScores.hydrogenCage;

  if (metallicity < 0.3) {
    compositeFitness *= 0.5;
  }

  const activePillarScores = isHydride ? pillarScores : {
    coupling: pillarScores.coupling,
    phonon: pillarScores.phonon,
    dos: pillarScores.dos,
    nesting: pillarScores.nesting,
    structure: pillarScores.structure,
    pairingGlue: pillarScores.pairingGlue,
    instability: pillarScores.instability,
  };

  const totalPillars = isHydride ? 8 : 7;
  const satisfiedPillars =
    (lambda >= targets.minLambda ? 1 : 0) +
    (omegaLogK >= targets.minOmegaLogK ? 1 : 0) +
    (dos >= targets.minDOS ? 1 : 0) +
    (nestingScore >= targets.minNesting ? 1 : 0) +
    (flatBandScore >= targets.minFlatBand || motif.score >= 0.7 ? 1 : 0) +
    (pairingGlue.compositePairingGlue >= targets.minPairingGlue ? 1 : 0) +
    (instability.compositeInstability >= targets.minInstability ? 1 : 0) +
    (isHydride && hydrogenCage.compositeHydrogenScore >= targets.minHydrogenCage ? 1 : 0);

  const activeEntries = (Object.entries(activePillarScores) as [string, number][])
    .filter(([key]) => activeWeights[key as keyof typeof activeWeights] > 0);
  const weakestPillar = activeEntries.length > 0
    ? activeEntries.reduce((a, b) => a[1] < b[1] ? a : b)[0]
    : "coupling";

  let tcPredicted = 0;
  try {
    const features = await extractFeatures(formula);
    const gb = await gbPredict(features);
    tcPredicted = gb.tcPredicted;
  } catch {}

  const constraint = checkPhysicsConstraints(formula, { maxPressureGPa: options?.maxPressureGPa });

  if (!constraint.isValid) {
    if (constraint.totalPenalty > 2.0) {
      compositeFitness *= 0.05;
    } else if (constraint.totalPenalty > 1.0) {
      compositeFitness *= 0.15;
    } else {
      compositeFitness *= 0.4;
    }
  }

  if (tcPredicted > 0 && tcPredicted < 20) {
    const tcScaling = Math.pow(tcPredicted / 20, 0.8);
    compositeFitness *= Math.max(0.4, tcScaling);
  }

  ctx.fitnessSum += compositeFitness;

  const pillarSatisfiedFlags: Record<string, boolean> = {
    coupling: lambda >= targets.minLambda,
    phonon: omegaLogK >= targets.minOmegaLogK,
    dos: dos >= targets.minDOS,
    nesting: nestingScore >= targets.minNesting,
    structure: flatBandScore >= targets.minFlatBand || motif.score >= 0.7,
    pairingGlue: pairingGlue.compositePairingGlue >= targets.minPairingGlue,
    instability: instability.compositeInstability >= targets.minInstability,
    hydrogenCage: isHydride && hydrogenCage.compositeHydrogenScore >= targets.minHydrogenCage,
  };
  for (const [key, satisfied] of Object.entries(pillarSatisfiedFlags)) {
    if (satisfied) {
      ctx.pillarSatisfied[key] = (ctx.pillarSatisfied[key] || 0) + 1;
    }
  }

  const motifFamily = motif.match.includes("cuprate") ? "cuprate"
    : motif.match.includes("pnictide") ? "pnictide"
    : motif.match.includes("kagome") ? "kagome"
    : motif.match === "A15" ? "A15"
    : motif.match.includes("perovskite") ? "perovskite"
    : isHydride ? "hydride"
    : "other";
  if (!ctx.familySatisfied[motifFamily]) {
    ctx.familySatisfied[motifFamily] = {};
  }
  ctx.familyCounts[motifFamily] = (ctx.familyCounts[motifFamily] || 0) + 1;
  for (const [key, satisfied] of Object.entries(pillarSatisfiedFlags)) {
    if (satisfied) {
      ctx.familySatisfied[motifFamily][key] = (ctx.familySatisfied[motifFamily][key] || 0) + 1;
    }
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  for (const el of elements) {
    if (!ctx.elementAffinity[el]) ctx.elementAffinity[el] = { totalFitness: 0, count: 0 };
    const atomFraction = (counts[el] || 1) / Math.max(1, totalAtoms);
    ctx.elementAffinity[el].totalFitness += compositeFitness * atomFraction;
    ctx.elementAffinity[el].count++;
    if (!ctx.elementAffinityHistory[el]) ctx.elementAffinityHistory[el] = [];
    ctx.elementAffinityHistory[el].push(compositeFitness * atomFraction);
    if (ctx.elementAffinityHistory[el].length > 100) {
      ctx.elementAffinityHistory[el] = ctx.elementAffinityHistory[el].slice(-100);
    }
  }

  if (compositeFitness > ctx.bestFitness) {
    ctx.bestFitness = compositeFitness;
    ctx.bestFormula = formula;
    ctx.bestTc = tcPredicted;
  }

  if (compositeFitness > 0.5) {
    ctx.topCandidates.push({ formula, fitness: compositeFitness, tc: tcPredicted, pillars: satisfiedPillars });
    ctx.topCandidates.sort((a, b) => b.fitness - a.fitness);
    if (ctx.topCandidates.length > 20) ctx.topCandidates = ctx.topCandidates.slice(0, 20);
  }

  return {
    formula,
    lambda,
    omegaLogK,
    dos,
    nestingScore,
    flatBandScore,
    motifMatch: motif.match,
    motifScore: motif.score,
    pairingGlue,
    instability,
    hydrogenCage,
    fermiSurface,
    pillarScores,
    compositeFitness,
    satisfiedPillars,
    weakestPillar,
    tcPredicted,
    metallicity,
    muStar,
    physicsValid: constraint.isValid,
  };
}

interface DesignTemplate {
  name: string;
  targetMotif: string;
  baseElements: string[][];
  lightAtom: string;
  stoichiometryRange: [number, number];
  lightAtomRange: [number, number];
  pressureAffinity: "high" | "ambient" | "any";
}

const DESIGN_TEMPLATES: DesignTemplate[] = [
  {
    name: "clathrate-hydride",
    targetMotif: "cage",
    baseElements: [
      ["La", "H"], ["Y", "H"], ["Ca", "H"], ["Sr", "H"], ["Ba", "H"],
      ["Sc", "H"], ["Ce", "H"], ["Th", "H"], ["Zr", "H"], ["Ti", "H"],
    ],
    lightAtom: "H",
    stoichiometryRange: [1, 2],
    lightAtomRange: [6, 10],
    pressureAffinity: "high",
  },
  {
    name: "metal-boride",
    targetMotif: "layered",
    baseElements: [
      ["Nb", "B"], ["Mo", "B"], ["Ti", "B"], ["V", "B"], ["Zr", "B"],
      ["Ta", "B"], ["W", "B"], ["Hf", "B"], ["Mg", "B"], ["Al", "B"],
    ],
    lightAtom: "B",
    stoichiometryRange: [1, 3],
    lightAtomRange: [2, 6],
    pressureAffinity: "any",
  },
  {
    name: "metal-carbide",
    targetMotif: "A15",
    baseElements: [
      ["Nb", "C"], ["Mo", "C"], ["Ti", "C"], ["V", "C"], ["Zr", "C"],
      ["Ta", "C"], ["W", "C"], ["Hf", "C"],
    ],
    lightAtom: "C",
    stoichiometryRange: [1, 3],
    lightAtomRange: [1, 4],
    pressureAffinity: "any",
  },
  {
    name: "metal-nitride",
    targetMotif: "layered",
    baseElements: [
      ["Nb", "N"], ["Ti", "N"], ["V", "N"], ["Zr", "N"], ["Ta", "N"],
      ["Mo", "N"], ["Hf", "N"], ["W", "N"],
    ],
    lightAtom: "N",
    stoichiometryRange: [1, 3],
    lightAtomRange: [1, 4],
    pressureAffinity: "any",
  },
  {
    name: "ternary-hydride",
    targetMotif: "cage",
    baseElements: [
      ["La", "B", "H"], ["Y", "C", "H"], ["Ca", "B", "H"], ["Sr", "N", "H"],
      ["Ba", "C", "H"], ["La", "N", "H"], ["Y", "B", "H"], ["Sc", "C", "H"],
    ],
    lightAtom: "H",
    stoichiometryRange: [1, 2],
    lightAtomRange: [4, 8],
    pressureAffinity: "high",
  },
  {
    name: "high-dos-intermetallic",
    targetMotif: "A15",
    baseElements: [
      ["Nb", "Ge"], ["Nb", "Sn"], ["V", "Si"], ["V", "Ga"],
      ["Nb", "Al"], ["Mo", "Ge"], ["Ta", "Si"], ["Nb", "Ga"],
    ],
    lightAtom: "",
    stoichiometryRange: [3, 3],
    lightAtomRange: [1, 1],
    pressureAffinity: "ambient",
  },
  {
    name: "layered-pnictide",
    targetMotif: "layered",
    baseElements: [
      ["Ba", "Fe", "As"], ["Sr", "Fe", "As"], ["La", "Fe", "P"],
      ["Ca", "Fe", "As"], ["Ba", "Co", "As"], ["Sr", "Ni", "P"],
    ],
    lightAtom: "",
    stoichiometryRange: [1, 2],
    lightAtomRange: [1, 2],
    pressureAffinity: "ambient",
  },
  {
    name: "cuprate-layered",
    targetMotif: "layered",
    baseElements: [
      ["Y", "Ba", "Cu", "O"], ["La", "Sr", "Cu", "O"], ["Bi", "Sr", "Cu", "O"],
      ["La", "Ba", "Cu", "O"], ["Nd", "Ce", "Cu", "O"],
    ],
    lightAtom: "O",
    stoichiometryRange: [1, 3],
    lightAtomRange: [4, 7],
    pressureAffinity: "ambient",
  },
  {
    name: "sodalite-superhydride",
    targetMotif: "cage",
    baseElements: [
      ["La", "H"], ["Y", "H"], ["Ce", "H"], ["Th", "H"],
      ["Ca", "H"], ["Ba", "H"], ["Sc", "H"],
    ],
    lightAtom: "H",
    stoichiometryRange: [1, 1],
    lightAtomRange: [8, 12],
    pressureAffinity: "high",
  },
  {
    name: "spin-fluctuation-pnictide",
    targetMotif: "layered",
    baseElements: [
      ["Ba", "Fe", "As"], ["Sr", "Fe", "As"], ["Ca", "Fe", "As"],
      ["Ba", "Fe", "P"], ["Sr", "Co", "As"], ["La", "Fe", "As", "O"],
    ],
    lightAtom: "",
    stoichiometryRange: [1, 2],
    lightAtomRange: [2, 2],
    pressureAffinity: "ambient",
  },
  {
    name: "mott-proximate",
    targetMotif: "layered",
    baseElements: [
      ["La", "Sr", "Cu", "O"], ["La", "Ba", "Cu", "O"],
      ["Ca", "V", "O"], ["Sr", "V", "O"], ["La", "Ni", "O"],
    ],
    lightAtom: "O",
    stoichiometryRange: [1, 2],
    lightAtomRange: [3, 7],
    pressureAffinity: "ambient",
  },
  {
    name: "dichalcogenide-nested",
    targetMotif: "layered",
    baseElements: [
      ["Nb", "Se"], ["Nb", "S"], ["Ta", "Se"], ["Ta", "S"],
      ["Mo", "Se"], ["Mo", "S"], ["W", "Se"], ["W", "Te"],
    ],
    lightAtom: "",
    stoichiometryRange: [1, 2],
    lightAtomRange: [2, 3],
    pressureAffinity: "ambient",
  },
  {
    name: "nickelate-layered",
    targetMotif: "layered",
    baseElements: [
      ["La", "Ni", "O"], ["Nd", "Ni", "O"], ["Pr", "Ni", "O"],
      ["La", "Sr", "Ni", "O"], ["Nd", "Sr", "Ni", "O"],
    ],
    lightAtom: "O",
    stoichiometryRange: [1, 2],
    lightAtomRange: [2, 4],
    pressureAffinity: "ambient",
  },
];

const STOICH_MULTIPLIERS_LIGHT = [1, 1.5, 2, 3];
const STOICH_MULTIPLIERS_METAL = [0.5, 1];
let stoichIndex = 0;

const TYPICAL_VALENCE: Record<string, number> = {
  Y: 3, La: 3, Nd: 3, Pr: 3, Ce: 3, Sm: 3, Gd: 3,
  Ba: 2, Sr: 2, Ca: 2, Bi: 3,
  Cu: 2, Ni: 2, V: 3, Fe: 3, Co: 2,
};

function valenceBalancedOxygen(counts: Record<string, number>, lightAtom: string): number | null {
  if (lightAtom !== "O") return null;
  let totalPositiveCharge = 0;
  for (const [el, n] of Object.entries(counts)) {
    if (el === "O") continue;
    const v = TYPICAL_VALENCE[el];
    if (v === undefined) return null;
    totalPositiveCharge += v * n;
  }
  if (totalPositiveCharge <= 0) return null;
  return Math.max(1, Math.round(totalPositiveCharge / 2));
}

function generateFromTemplate(template: DesignTemplate, count: number, globalSeen?: Set<string>): string[] {
  if (count <= 0) return [];
  const formulas: string[] = [];
  const seen = globalSeen ?? new Set<string>();
  const isA15 = template.targetMotif === "A15" && template.baseElements[0]?.length === 2;

  for (const baseEls of template.baseElements) {
    if (formulas.length >= count) break;

    if (isA15) {
      const counts: Record<string, number> = { [baseEls[0]]: 3, [baseEls[1]]: 1 };
      const f = countsToFormula(counts);
      if (!seen.has(f) && f.length > 1) {
        seen.add(f);
        formulas.push(f);
      }
      continue;
    }

    for (let metalStoich = template.stoichiometryRange[0]; metalStoich <= template.stoichiometryRange[1]; metalStoich++) {
      for (let lightCount = template.lightAtomRange[0]; lightCount <= template.lightAtomRange[1]; lightCount++) {
        if (formulas.length >= count) break;

        const counts: Record<string, number> = {};
        for (let i = 0; i < baseEls.length; i++) {
          const el = baseEls[i];
          if (el === template.lightAtom) {
            counts[el] = lightCount;
          } else if (i === 0) {
            counts[el] = metalStoich;
          } else if (LIGHT_ATOMS.includes(el) || PNICTOGEN_ELEMENTS.includes(el) || CHALCOGEN_ELEMENTS.includes(el)) {
            const mult = STOICH_MULTIPLIERS_LIGHT[stoichIndex % STOICH_MULTIPLIERS_LIGHT.length];
            stoichIndex++;
            counts[el] = Math.max(1, Math.round(metalStoich * mult));
          } else {
            const mult = STOICH_MULTIPLIERS_METAL[stoichIndex % STOICH_MULTIPLIERS_METAL.length];
            stoichIndex++;
            counts[el] = Math.max(1, Math.round(metalStoich * mult));
          }
        }

        if (template.lightAtom === "O") {
          const balancedO = valenceBalancedOxygen(counts, "O");
          if (balancedO !== null) {
            const [lo, hi] = template.lightAtomRange;
            counts["O"] = Math.max(lo, Math.min(hi, balancedO));
          }
        }

        const f = countsToFormula(counts);
        if (!seen.has(f) && f.length > 1) {
          seen.add(f);
          formulas.push(f);
        }
      }
    }
  }

  return formulas;
}

export async function runPillarGuidedGeneration(
  targets: SCPillarTargets = DEFAULT_PILLAR_TARGETS,
  candidatesPerTemplate: number = 8,
  pressureGPa?: number,
): Promise<PillarGuidedCandidate[]> {
  const results: PillarGuidedCandidate[] = [];
  const allFormulas: string[] = [];
  const globalSeen = new Set<string>();

  const isHighPressure = pressureGPa !== undefined && pressureGPa > 50;
  const isAmbient = pressureGPa === undefined || pressureGPa <= 5;
  const isMidPressure = !isHighPressure && !isAmbient;

  const prioritized: DesignTemplate[] = [];
  const secondary: DesignTemplate[] = [];
  for (const template of DESIGN_TEMPLATES) {
    if (template.pressureAffinity === "any") {
      prioritized.push(template);
    } else if (isHighPressure && template.pressureAffinity === "high") {
      prioritized.push(template);
    } else if (isAmbient && template.pressureAffinity === "ambient") {
      prioritized.push(template);
    } else if (isMidPressure) {
      secondary.push(template);
    } else {
      secondary.push(template);
    }
  }

  const prioritizedAllocation = candidatesPerTemplate <= 0 ? 0 :
    isMidPressure ? candidatesPerTemplate : Math.max(1, Math.round(candidatesPerTemplate * 1.5));
  const secondaryAllocation = candidatesPerTemplate <= 0 ? 0 :
    isMidPressure ? candidatesPerTemplate : Math.max(1, Math.round(candidatesPerTemplate * 0.5));

  for (const template of prioritized) {
    const formulas = generateFromTemplate(template, prioritizedAllocation, globalSeen);
    allFormulas.push(...formulas);
  }
  for (const template of secondary) {
    const formulas = generateFromTemplate(template, secondaryAllocation, globalSeen);
    allFormulas.push(...formulas);
  }

  defaultCtx.totalGenerated += allFormulas.length;

  for (const formula of allFormulas) {
    try {
      const evaluation = await evaluatePillars(formula, targets, { maxPressureGPa: pressureGPa });
      if (!evaluation.physicsValid) continue;

      const strengths: string[] = [];
      if (evaluation.pillarScores.coupling >= 0.7) strengths.push(`strong e-ph coupling (lambda=${evaluation.lambda.toFixed(2)})`);
      if (evaluation.pillarScores.phonon >= 0.7) strengths.push(`high phonons (omega=${evaluation.omegaLogK.toFixed(0)}K)`);
      if (evaluation.pillarScores.dos >= 0.7) strengths.push(`high DOS (${evaluation.dos.toFixed(2)})`);
      if (evaluation.pillarScores.nesting >= 0.7) strengths.push(`good nesting (${evaluation.nestingScore.toFixed(2)})`);
      if (evaluation.pillarScores.structure >= 0.7) strengths.push(`favorable ${evaluation.motifMatch}`);
      if (evaluation.pillarScores.pairingGlue >= 0.7) strengths.push(`strong ${evaluation.pairingGlue.dominantMechanism} glue (${evaluation.pairingGlue.compositePairingGlue.toFixed(2)})`);
      if (evaluation.pillarScores.instability >= 0.7) strengths.push(`near ${evaluation.instability.dominantInstability} instability (${evaluation.instability.compositeInstability.toFixed(2)})`);
      if (evaluation.pillarScores.hydrogenCage >= 0.7) strengths.push(`H-cage ${evaluation.hydrogenCage.bondingType} (${evaluation.hydrogenCage.compositeHydrogenScore.toFixed(2)})`);

      const rationale = strengths.length > 0
        ? `${evaluation.satisfiedPillars}/${evaluation.hydrogenCage.isHydride ? 8 : 7} pillars met: ${strengths.join("; ")}. Weakest: ${evaluation.weakestPillar}`
        : `Low pillar satisfaction (${evaluation.satisfiedPillars}/${evaluation.hydrogenCage.isHydride ? 8 : 7}). Weakest: ${evaluation.weakestPillar}`;

      results.push({
        formula,
        evaluation,
        designRationale: rationale,
      });
    } catch {}
  }

  results.sort((a, b) => b.evaluation.compositeFitness - a.evaluation.compositeFitness);
  return results;
}

let rewardBaseline = 30;
let rewardCount = 0;

export function updatePillarWeightsFromReward(tcReward: number, evaluation: PillarEvaluation, ctx: PillarOptimizerContext = defaultCtx): void {
  rewardCount++;
  rewardBaseline = rewardBaseline * 0.99 + tcReward * 0.01;

  const lr = 0.003;
  const tcDelta = tcReward - rewardBaseline;
  const normalizedReward = Math.min(1.0, Math.max(-1.0, tcDelta / 100));

  const pillarEntries = Object.entries(evaluation.pillarScores) as [string, number][];
  for (const [pillar, score] of pillarEntries) {
    const key = pillar as keyof typeof ctx.pillarWeights;
    if (score >= 0.7 && normalizedReward > 0.1) {
      ctx.pillarWeights[key] = Math.min(0.5, ctx.pillarWeights[key] + lr);
    } else if (score < 0.3 && normalizedReward < -0.1) {
      ctx.pillarWeights[key] = Math.max(0.05, ctx.pillarWeights[key] - lr * 0.5);
    }
  }

  const totalWeight = Object.values(ctx.pillarWeights).reduce((s, w) => s + w, 0);
  for (const key of Object.keys(ctx.pillarWeights) as (keyof typeof ctx.pillarWeights)[]) {
    ctx.pillarWeights[key] /= totalWeight;
  }

  if (rewardCount > 0 && rewardCount % 10 === 0) {
    regularizePillarWeights(ctx);
  }
}

function regularizePillarWeights(ctx: PillarOptimizerContext = defaultCtx): void {
  const weights = Object.values(ctx.pillarWeights);
  const nPillars = weights.length;
  const uniform = 1.0 / nPillars;
  const mean = weights.reduce((s, w) => s + w, 0) / nPillars;
  const variance = weights.reduce((s, w) => s + (w - mean) ** 2, 0) / nPillars;
  const decayStrength = Math.min(0.5, 0.10 + 2.0 * variance);
  for (const key of Object.keys(ctx.pillarWeights) as (keyof typeof ctx.pillarWeights)[]) {
    ctx.pillarWeights[key] = ctx.pillarWeights[key] * (1 - decayStrength) + uniform * decayStrength;
  }
}

const pillarDFTFeedback: { pillar: string; correct: number; total: number }[] = [];

const MECHANISM_TO_PILLARS: Record<string, string[]> = {
  "phonon": ["coupling", "phonon"],
  "spin-fluctuation": ["nesting", "pairingGlue"],
  "charge-fluctuation": ["dos", "pairingGlue"],
  "excitonic": ["dos", "pairingGlue", "instability"],
};

export async function incorporateDFTFeedbackIntoPillars(
  formula: string,
  predictedTc: number,
  actualTc: number,
  actualStable: boolean,
  pressureGPa?: number,
  ctx: PillarOptimizerContext = defaultCtx,
): Promise<void> {
  let evaluation: PillarEvaluation | null = null;
  try {
    evaluation = await evaluatePillars(formula, undefined, { maxPressureGPa: pressureGPa, ctx });
  } catch {
    return;
  }

  const predictionError = predictedTc - actualTc;
  const overestimated = predictionError > 20;
  const underestimated = predictionError < -20;
  const accurate = Math.abs(predictionError) <= 20;

  const lr = 0.005;
  const mechanism = evaluation.pairingGlue.dominantMechanism;
  const mechanismAlignedPillars = new Set(MECHANISM_TO_PILLARS[mechanism] ?? []);
  const pillarEntries = Object.entries(evaluation.pillarScores) as [string, number][];

  for (const [pillar, score] of pillarEntries) {
    const key = pillar as keyof typeof ctx.pillarWeights;
    if (ctx.pillarWeights[key] === undefined) continue;

    let existing = pillarDFTFeedback.find(f => f.pillar === pillar);
    if (!existing) {
      existing = { pillar, correct: 0, total: 0 };
      pillarDFTFeedback.push(existing);
    }
    existing.total++;

    const isAligned = mechanismAlignedPillars.has(pillar);
    const isStructural = pillar === "structure" || pillar === "instability" || pillar === "hydrogenCage";

    if (overestimated && score >= 0.6) {
      ctx.pillarWeights[key] = Math.max(0.04, ctx.pillarWeights[key] - lr);
    } else if (accurate && score >= 0.6) {
      if (isAligned) {
        ctx.pillarWeights[key] = Math.min(0.5, ctx.pillarWeights[key] + lr * 0.5);
        existing.correct++;
      } else if (isStructural) {
        ctx.pillarWeights[key] = Math.min(0.5, ctx.pillarWeights[key] + lr * 0.2);
        existing.correct++;
      }
    } else if (underestimated && score < 0.4) {
      if (isAligned) {
        ctx.pillarWeights[key] = Math.min(0.5, ctx.pillarWeights[key] + lr * 0.3);
        existing.correct++;
      }
    }

    if (!actualStable && score >= 0.5) {
      const instabilityPenaltyMult = (pillar === "instability") ? 1.0 : 0.3;
      ctx.pillarWeights[key] = Math.max(0.04, ctx.pillarWeights[key] - lr * instabilityPenaltyMult);
    }
  }

  const totalWeight = Object.values(ctx.pillarWeights).reduce((s, w) => s + w, 0);
  for (const key of Object.keys(ctx.pillarWeights) as (keyof typeof ctx.pillarWeights)[]) {
    ctx.pillarWeights[key] /= totalWeight;
  }
}

export function getPillarDFTFeedbackStats(): { pillar: string; accuracy: number; total: number }[] {
  return pillarDFTFeedback
    .filter(f => f.total >= 3)
    .map(f => ({ pillar: f.pillar, accuracy: f.correct / f.total, total: f.total }))
    .sort((a, b) => b.total - a.total);
}

let mutationGeneration = 0;

export async function runPillarCycle(
  existingFormulas: string[],
  targetTc: number = 200,
  pressureGPa?: number,
): Promise<{ formulas: string[]; evaluations: PillarEvaluation[]; bestFormula: string; bestFitness: number; bestTc: number }> {
  mutationGeneration++;
  const t = Math.max(0, Math.min(1, (targetTc - 20) / 280));
  const targets: SCPillarTargets = {
    ...DEFAULT_PILLAR_TARGETS,
    minLambda: 1.0 + 1.0 * t,
    minOmegaLogK: 400 + 400 * t,
    minDOS: 2.0 + 1.0 * t,
    minPairingGlue: 0.3 + 0.3 * t,
    minInstability: 0.25 + 0.25 * t,
    minHydrogenCage: 0.4 + 0.2 * t,
  };

  const guided = await runPillarGuidedGeneration(targets, 6, pressureGPa);

  const reEvalExisting: PillarEvaluation[] = [];
  for (const f of existingFormulas.slice(0, 10)) {
    try {
      reEvalExisting.push(await evaluatePillars(f, targets, { maxPressureGPa: pressureGPa }));
    } catch {}
  }

  for (const ex of reEvalExisting) {
    if (ex.compositeFitness > 0.6) {
      try {
        const mutated = mutateTowardWeakPillar(ex, targets, pressureGPa);
        if (mutated) {
          const mutEval = await evaluatePillars(mutated, targets, { maxPressureGPa: pressureGPa });
          if (mutEval.compositeFitness > ex.compositeFitness) {
            const fitnessImprovement = mutEval.compositeFitness - ex.compositeFitness;
            guided.push({
              formula: mutated,
              evaluation: mutEval,
              designRationale: `Mutation of ${ex.formula}: improved ${ex.weakestPillar} (+${(fitnessImprovement * 100).toFixed(1)}%)`,
              lineage: {
                parentFormula: ex.formula,
                targetedPillar: ex.weakestPillar,
                fitnessImprovement,
                generation: mutationGeneration,
              },
            });
          }
        }
      } catch {}
    }
  }

  guided.sort((a, b) => b.evaluation.compositeFitness - a.evaluation.compositeFitness);

  const passingFormulas = guided
    .filter(g => g.evaluation.compositeFitness > 0.4 && g.evaluation.physicsValid)
    .map(g => g.formula);

  const evals = guided.map(g => g.evaluation);

  let cycleBest = "";
  let cycleBestFitness = 0;
  let cycleBestTc = 0;
  for (const g of guided) {
    if (g.evaluation.compositeFitness > cycleBestFitness) {
      cycleBestFitness = g.evaluation.compositeFitness;
      cycleBest = g.formula;
      cycleBestTc = g.evaluation.tcPredicted;
    }
  }

  return {
    formulas: passingFormulas,
    evaluations: evals,
    bestFormula: cycleBest,
    bestFitness: cycleBestFitness,
    bestTc: cycleBestTc,
  };
}

let mutationSeed = 0;
function pickFrom<T>(arr: T[]): T {
  mutationSeed++;
  return arr[mutationSeed % arr.length];
}

const RARE_EARTH_CAGE = ["La", "Y", "Ce", "Sc", "Ca", "Sr", "Ba", "Th"];
const MAGNETIC_3D = ["Fe", "Co", "Mn", "Ni", "Cr"];
const VAN_HOVE_TM = ["V", "Nb", "Ti", "Mo", "Ta"];

const MUTATION_OXIDATION: Record<string, number> = {
  H: 1, Li: 1, Na: 1, K: 1,
  Be: 2, Mg: 2, Ca: 2, Sr: 2, Ba: 2,
  Sc: 3, Y: 3, La: 3, Ce: 3, Th: 4,
  Ti: 4, Zr: 4, Hf: 4,
  V: 5, Nb: 5, Ta: 5,
  Cr: 3, Mo: 6, W: 6,
  Mn: 2, Fe: 3, Co: 2, Ni: 2, Cu: 2,
  B: 3, Al: 3, Ga: 3, In: 3,
  C: -4, Si: -4, Ge: -4, Sn: 4,
  N: -3, P: -3, As: -3, Sb: -3,
  O: -2, S: -2, Se: -2, Te: -2,
  F: -1, Cl: -1, Br: -1,
};

const ISOVALENT_GROUPS: string[][] = [
  ["La", "Y", "Ce", "Nd", "Pr", "Gd", "Sc"],
  ["Ca", "Sr", "Ba"],
  ["Ti", "Zr", "Hf"],
  ["V", "Nb", "Ta"],
  ["Cr", "Mo", "W"],
  ["Fe", "Co", "Ni"],
  ["Cu", "Ag"],
  ["As", "P", "Sb"],
  ["S", "Se", "Te"],
  ["B", "Al", "Ga"],
  ["Si", "Ge", "Sn"],
];

function quickChargeCheck(counts: Record<string, number>): boolean {
  let totalCharge = 0;
  let unknownCount = 0;
  for (const [el, n] of Object.entries(counts)) {
    const ox = MUTATION_OXIDATION[el];
    if (ox === undefined) {
      unknownCount++;
      continue;
    }
    totalCharge += ox * n;
  }
  if (unknownCount > 0) return true;
  return Math.abs(totalCharge) <= 4;
}

function findIsovalentSubstitute(el: string, exclude: string[]): string | null {
  for (const group of ISOVALENT_GROUPS) {
    if (group.includes(el)) {
      const candidates = group.filter(e => e !== el && !exclude.includes(e));
      if (candidates.length > 0) return pickFrom(candidates);
    }
  }
  return null;
}

function applyMutation(
  evaluation: PillarEvaluation,
  mutated: Record<string, number>,
  elements: string[],
): boolean {
  switch (evaluation.weakestPillar) {
    case "coupling": {
      const hasLight = elements.some(e => LIGHT_ATOMS.includes(e));
      if (!hasLight) {
        const lightEl = pickFrom(["H", "B", "C", "N"]);
        mutated[lightEl] = pickFrom([2, 3, 4]);
      } else {
        const lightEls = elements.filter(e => LIGHT_ATOMS.includes(e));
        const el = pickFrom(lightEls);
        mutated[el] = Math.min(10, mutated[el] + pickFrom([1, 2, 3]));
      }
      return true;
    }
    case "phonon": {
      if (!elements.includes("H")) {
        mutated["H"] = pickFrom([4, 6, 8]);
      } else {
        mutated["H"] = Math.min(12, (mutated["H"] || 0) + pickFrom([1, 2, 3]));
      }
      return true;
    }
    case "dos": {
      const hasTM = elements.some(e => HIGH_COUPLING_TM.includes(e));
      if (!hasTM) {
        const tm = pickFrom(HIGH_COUPLING_TM);
        mutated[tm] = pickFrom([1, 2]);
      } else {
        const tmEls = elements.filter(e => HIGH_COUPLING_TM.includes(e));
        const el = pickFrom(tmEls);
        mutated[el] = Math.min(4, mutated[el] + 1);
      }
      return true;
    }
    case "nesting": {
      const hasPnictogen = elements.some(e => PNICTOGEN_ELEMENTS.includes(e));
      if (!hasPnictogen && !elements.includes("O")) {
        mutated[pickFrom(PNICTOGEN_ELEMENTS)] = 2;
      }
      const hasLayerFormer = elements.some(e => LAYER_FORMERS.includes(e));
      if (!hasLayerFormer) {
        mutated[pickFrom(LAYER_FORMERS)] = pickFrom([1, 2]);
      }
      return true;
    }
    case "structure": {
      const hasCageFormer = elements.some(e => CAGE_FORMERS.includes(e));
      if (!hasCageFormer) {
        const existingMetal = elements.find(e => isTransitionMetal(e) || isRareEarth(e));
        if (existingMetal) {
          const sub = findIsovalentSubstitute(existingMetal, elements);
          if (sub && CAGE_FORMERS.includes(sub)) {
            mutated[sub] = mutated[existingMetal];
            delete mutated[existingMetal];
          } else {
            mutated[pickFrom(CAGE_FORMERS)] = 1;
          }
        } else {
          mutated[pickFrom(CAGE_FORMERS)] = 1;
        }
      }
      if (!elements.includes("H") && !elements.includes("B")) {
        const existingLight = elements.find(e => LIGHT_ATOMS.includes(e));
        if (existingLight) {
          const sub = findIsovalentSubstitute(existingLight, elements);
          if (sub) {
            mutated[sub] = mutated[existingLight];
            delete mutated[existingLight];
          } else {
            mutated[pickFrom(["H", "B"])] = pickFrom([4, 6]);
          }
        } else {
          mutated[pickFrom(["H", "B"])] = pickFrom([4, 6]);
        }
      }
      return true;
    }
    case "pairingGlue": {
      if (evaluation.pairingGlue.spinFluctuationContribution < 0.3) {
        const hasMagnetic = elements.some(e => MAGNETIC_3D.includes(e));
        if (!hasMagnetic) {
          mutated[pickFrom(MAGNETIC_3D)] = 2;
          if (!elements.some(e => PNICTOGEN_ELEMENTS.includes(e))) {
            mutated[pickFrom(["As", "P"])] = 2;
          }
        }
      } else {
        const hasLight = elements.some(e => LIGHT_ATOMS.includes(e));
        if (!hasLight) {
          mutated[pickFrom(["H", "B"])] = pickFrom([4, 6]);
        }
      }
      return true;
    }
    case "instability": {
      if (evaluation.instability.mottProximity < 0.3) {
        if (!elements.includes("Cu") && !elements.includes("O")) {
          mutated["Cu"] = pickFrom([1, 2]);
          mutated["O"] = pickFrom([3, 4]);
          const hasRE = elements.some(e => isRareEarth(e));
          if (!hasRE) mutated[pickFrom(RARE_EARTH_CAGE)] = 1;
        }
      }
      if (evaluation.instability.vanHoveProximity < 0.3) {
        const hasTM = elements.some(e => isTransitionMetal(e));
        if (!hasTM) {
          mutated[pickFrom(VAN_HOVE_TM)] = pickFrom([1, 2]);
        }
      }
      return true;
    }
    case "hydrogenCage": {
      if (!evaluation.hydrogenCage.isHydride) return false;
      if (!elements.includes("H")) {
        mutated["H"] = pickFrom([6, 8, 10]);
      } else {
        mutated["H"] = Math.min(12, (mutated["H"] || 0) + pickFrom([2, 3]));
      }
      const hasCage = elements.some(e => SODALITE_CAGE_ELEMENTS.includes(e));
      if (!hasCage) {
        mutated[pickFrom(RARE_EARTH_CAGE)] = 1;
      }
      return true;
    }
    default:
      return false;
  }
}

function mutateTowardWeakPillar(
  evaluation: PillarEvaluation,
  targets: SCPillarTargets,
  pressureGPa?: number,
): string | null {
  const maxAttempts = 3;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const counts = parseCounts(evaluation.formula);
    const elements = Object.keys(counts);
    const mutated = { ...counts };

    const applied = applyMutation(evaluation, mutated, elements);
    if (!applied) return null;

    if (!quickChargeCheck(mutated)) continue;

    const result = countsToFormula(mutated);
    if (result === evaluation.formula) continue;

    const constraint = checkPhysicsConstraints(result, { maxPressureGPa: pressureGPa });
    if (!constraint.isValid && !constraint.repairedFormula) continue;

    return constraint.isValid ? result : constraint.repairedFormula;
  }
  return null;
}

export function getPillarOptimizerStats(ctx: PillarOptimizerContext = defaultCtx): PillarOptimizerStats {
  const affinityScores: Record<string, number> = {};
  const surpriseFactors: Record<string, number> = {};

  const allAverages: number[] = [];
  for (const [, data] of Object.entries(ctx.elementAffinity)) {
    if (data.count >= 3) {
      allAverages.push(data.totalFitness / data.count);
    }
  }
  const globalMean = allAverages.length > 0
    ? allAverages.reduce((s, v) => s + v, 0) / allAverages.length
    : 0;

  for (const [el, data] of Object.entries(ctx.elementAffinity)) {
    if (data.count >= 3) {
      const currentAvg = data.totalFitness / data.count;
      affinityScores[el] = Math.round(currentAvg * 1000) / 1000;

      const history = ctx.elementAffinityHistory[el] || [];
      if (history.length >= 5) {
        const midpoint = Math.floor(history.length / 2);
        const recentSlice = history.slice(midpoint);
        const recentMean = recentSlice.reduce((s, v) => s + v, 0) / recentSlice.length;
        const historicalMean = history.slice(0, midpoint).reduce((s, v) => s + v, 0) / midpoint;
        const baseline = globalMean > 0 ? globalMean : 1;
        surpriseFactors[el] = Math.round(((recentMean - historicalMean) / baseline) * 1000) / 1000;
      }
    }
  }

  const sorted = Object.entries(affinityScores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);
  const topAffinity: Record<string, number> = {};
  for (const [el, score] of sorted) topAffinity[el] = score;

  const topSurprise: Record<string, number> = {};
  const sortedSurprise = Object.entries(surpriseFactors)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 20);
  for (const [el, sf] of sortedSurprise) topSurprise[el] = sf;

  const satisfactionRates: Record<string, number> = {};
  for (const [key, count] of Object.entries(ctx.pillarSatisfied)) {
    satisfactionRates[key] = ctx.totalEvaluated > 0 ? Math.round((count / ctx.totalEvaluated) * 1000) / 1000 : 0;
  }

  const familySatisfactionRates: Record<string, Record<string, number>> = {};
  for (const [family, pillarCounts] of Object.entries(ctx.familySatisfied)) {
    const familyTotal = ctx.familyCounts[family] || 1;
    familySatisfactionRates[family] = {};
    for (const [pillar, count] of Object.entries(pillarCounts)) {
      familySatisfactionRates[family][pillar] = Math.round((count / familyTotal) * 1000) / 1000;
    }
  }

  return {
    totalEvaluated: ctx.totalEvaluated,
    totalGenerated: ctx.totalGenerated,
    avgCompositeFitness: ctx.totalEvaluated > 0 ? Math.round((ctx.fitnessSum / ctx.totalEvaluated) * 1000) / 1000 : 0,
    bestFitness: Math.round(ctx.bestFitness * 1000) / 1000,
    bestFormula: ctx.bestFormula,
    pillarSatisfactionRates: satisfactionRates,
    familySatisfactionRates,
    elementAffinityScores: topAffinity,
    elementSurpriseFactors: topSurprise,
    topCandidates: ctx.topCandidates.slice(0, 10),
    pillarWeights: { ...ctx.pillarWeights },
  };
}
