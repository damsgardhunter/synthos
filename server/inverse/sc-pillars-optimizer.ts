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

export interface PillarGuidedCandidate {
  formula: string;
  evaluation: PillarEvaluation;
  designRationale: string;
}

export interface PillarOptimizerStats {
  totalEvaluated: number;
  totalGenerated: number;
  avgCompositeFitness: number;
  bestFitness: number;
  bestFormula: string;
  pillarSatisfactionRates: Record<string, number>;
  elementAffinityScores: Record<string, number>;
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
  const metals: string[] = [];
  const nonmetals: string[] = [];
  for (const el of Object.keys(counts)) {
    if (counts[el] <= 0) continue;
    const ed = ELEMENTAL_DATA[el];
    const en = ed?.paulingElectronegativity ?? 2.0;
    if (en <= 2.0 || isTransitionMetal(el) || isRareEarth(el)) metals.push(el);
    else nonmetals.push(el);
  }
  metals.sort((a, b) => a.localeCompare(b));
  nonmetals.sort((a, b) => a.localeCompare(b));
  return [...metals, ...nonmetals].map(el => {
    const n = counts[el];
    return n === 1 ? el : `${el}${Math.round(n)}`;
  }).join("");
}

function detectMotif(formula: string, elements: string[], counts: Record<string, number>): { match: string; score: number } {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const hCount = counts["H"] || 0;
  const hRatio = hCount / Math.max(1, totalAtoms - hCount);

  if (hRatio >= 4) return { match: "cage-clathrate", score: 0.95 };
  if (hRatio >= 2) return { match: "layered-hydride", score: 0.85 };

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
    const hasTM = elements.some(e => isTransitionMetal(e));
    const hasLightAtom = elements.some(e => LIGHT_ATOMS.includes(e));
    if (hasTM && hasLightAtom) {
      const tmEl = elements.find(e => isTransitionMetal(e))!;
      const tmCount = counts[tmEl] || 0;
      if (tmCount === 3) return { match: "A15", score: 0.88 };
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

  const phononContribution = Math.min(1.0, lambda / 2.0);

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
  if (electronic.correlationStrength > 0.5 && electronic.metallicity > 0.3) {
    const mottScore = electronic.mottProximityScore ?? 0;
    if (mottScore > 0.3) {
      excitonicContribution = Math.min(0.8, mottScore * 0.6 + electronic.correlationStrength * 0.2);
    }
  }

  const compositePairingGlue =
    0.50 * phononContribution +
    0.25 * spinContribution +
    0.10 * chargeContribution +
    0.15 * excitonicContribution;

  let dominantMechanism = "phonon";
  const contributions = [
    { name: "phonon", val: phononContribution * 0.50 },
    { name: "spin-fluctuation", val: spinContribution * 0.25 },
    { name: "charge-fluctuation", val: chargeContribution * 0.10 },
    { name: "excitonic", val: excitonicContribution * 0.15 },
  ];
  contributions.sort((a, b) => b.val - a.val);
  if (contributions[0].val > 0) dominantMechanism = contributions[0].name;

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
    const hRatio = (counts["H"] || 0) / Math.max(1, Object.values(counts).reduce((s, n) => s + n, 0) - (counts["H"] || 0));
    if (hRatio >= 4) structInstab = Math.min(1.0, 0.5 + hRatio * 0.05);
    if (lambda > 2.0) structInstab = Math.max(structInstab, Math.min(1.0, lambda * 0.3));
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

  const hRatio = hCount / Math.max(1, nonHAtoms);
  const bondingType = classifyHydrogenBonding(formula, hRatio >= 6 ? 150 : hRatio >= 4 ? 100 : 50);

  let networkDim = 0;
  if (hRatio >= 6) networkDim = 3.0;
  else if (hRatio >= 4) networkDim = 2.5;
  else if (hRatio >= 2) networkDim = 2.0;
  else if (hRatio >= 1) networkDim = 1.5;
  else networkDim = 1.0;

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
    if (metalCount === 1 && nonHAtoms <= 2) {
      cageSymmetry = 0.95;
    } else if (metalCount <= 2) {
      cageSymmetry = 0.75;
    } else {
      cageSymmetry = 0.50;
    }
  } else if (cageScore >= 0.5) {
    cageSymmetry = 0.40;
  }

  const hCoordination = Math.min(12, Math.round(hRatio * 2));

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
  const fsDimensionality = cylindricalScore >= 0.8 ? 2.0 :
    cylindricalScore >= 0.5 ? 2.0 + (0.8 - cylindricalScore) / 0.3 :
    3.0;

  let nestingStrength = electronic.nestingScore ?? 0;
  let electronHolePocketOverlap = 0;
  let nestingVectorQ = "none";

  if (isPnictide) {
    nestingStrength = Math.max(nestingStrength, 0.80);
    electronHolePocketOverlap = Math.min(1.0, nestingStrength * 0.9);
    nestingVectorQ = "(pi,pi)";
  } else if (isCuprate) {
    nestingStrength = Math.max(nestingStrength, 0.85);
    electronHolePocketOverlap = Math.min(1.0, nestingStrength * 0.85);
    nestingVectorQ = "(pi,pi)";
  } else if (isNickelate) {
    nestingStrength = Math.max(nestingStrength, 0.75);
    electronHolePocketOverlap = Math.min(1.0, nestingStrength * 0.80);
    nestingVectorQ = "(pi,pi)";
  } else if (isDichalcogenide) {
    nestingStrength = Math.max(nestingStrength, 0.65);
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
    vanHoveDistance = Math.max(0, (1.0 - vanHoveProx) * 0.5);
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
      vanHoveDistance = Number(Math.min(1.0, minDist).toFixed(4));
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

let pillarWeights = {
  coupling: 0.18,
  phonon: 0.12,
  dos: 0.12,
  nesting: 0.10,
  structure: 0.10,
  pairingGlue: 0.18,
  instability: 0.10,
  hydrogenCage: 0.10,
};

let totalEvaluated = 0;
let totalGenerated = 0;
let fitnessSum = 0;
let bestFitness = 0;
let bestFormula = "";
let bestTc = 0;
let pillarSatisfied: Record<string, number> = {
  coupling: 0, phonon: 0, dos: 0, nesting: 0, structure: 0,
  pairingGlue: 0, instability: 0, hydrogenCage: 0,
};
let elementAffinity: Record<string, { totalFitness: number; count: number }> = {};
let topCandidates: { formula: string; fitness: number; tc: number; pillars: number }[] = [];

export function evaluatePillars(
  formula: string,
  targets: SCPillarTargets = DEFAULT_PILLAR_TARGETS
): PillarEvaluation {
  totalEvaluated++;

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
  const instability = computeInstabilityProximity(formula, electronic, lambda);
  const hydrogenCage = computeHydrogenCageMetrics(formula, elements, counts);
  const fermiSurface = computeFermiSurfaceGeometry(formula, electronic, elements);

  const motifBonus = (targets.preferredMotifs ?? []).some(pm => motif.match.includes(pm)) ? 0.2 : 0;

  const isHydride = hydrogenCage.isHydride;
  const hydrogenCageWeight = isHydride ? pillarWeights.hydrogenCage : 0;

  const activeWeights = { ...pillarWeights };
  if (!isHydride) {
    activeWeights.hydrogenCage = 0;
    const redistrib = pillarWeights.hydrogenCage / 7;
    activeWeights.coupling += redistrib;
    activeWeights.phonon += redistrib;
    activeWeights.dos += redistrib;
    activeWeights.nesting += redistrib;
    activeWeights.structure += redistrib;
    activeWeights.pairingGlue += redistrib;
    activeWeights.instability += redistrib;
  }

  const fsNestingBoost = fermiSurface.nestingStrength > nestingScore
    ? (fermiSurface.nestingStrength - nestingScore) * 0.4
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

  const weakestPillar = (Object.entries(activePillarScores) as [string, number][])
    .reduce((a, b) => a[1] < b[1] ? a : b)[0];

  let tcPredicted = 0;
  try {
    const features = extractFeatures(formula);
    const gb = gbPredict(features);
    tcPredicted = gb.tcPredicted;
  } catch {}

  const constraint = checkPhysicsConstraints(formula);

  if (!constraint.isValid && constraint.totalPenalty > 1.0) {
    compositeFitness *= 0.5;
  }

  if (tcPredicted > 0 && tcPredicted < 50) {
    const tcScaling = Math.pow(tcPredicted / 50, 0.6);
    compositeFitness *= Math.max(0.3, tcScaling);
  }

  fitnessSum += compositeFitness;

  for (const [key, score] of Object.entries(pillarScores)) {
    if (score >= 0.7) {
      pillarSatisfied[key] = (pillarSatisfied[key] || 0) + 1;
    }
  }

  for (const el of elements) {
    if (!elementAffinity[el]) elementAffinity[el] = { totalFitness: 0, count: 0 };
    elementAffinity[el].totalFitness += compositeFitness;
    elementAffinity[el].count++;
  }

  if (compositeFitness > bestFitness) {
    bestFitness = compositeFitness;
    bestFormula = formula;
    bestTc = tcPredicted;
  }

  if (compositeFitness > 0.5) {
    topCandidates.push({ formula, fitness: compositeFitness, tc: tcPredicted, pillars: satisfiedPillars });
    topCandidates.sort((a, b) => b.fitness - a.fitness);
    if (topCandidates.length > 20) topCandidates = topCandidates.slice(0, 20);
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
  },
  {
    name: "high-dos-intermetallic",
    targetMotif: "A15",
    baseElements: [
      ["Nb", "Ge"], ["Nb", "Sn"], ["V", "Si"], ["V", "Ga"],
      ["Nb", "Al"], ["Mo", "Ge"], ["Ta", "Si"], ["Nb", "Ga"],
    ],
    lightAtom: "",
    stoichiometryRange: [1, 3],
    lightAtomRange: [1, 1],
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
  },
];

function generateFromTemplate(template: DesignTemplate, count: number): string[] {
  const formulas: string[] = [];
  const seen = new Set<string>();

  for (const baseEls of template.baseElements) {
    if (formulas.length >= count) break;

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
            counts[el] = Math.max(1, Math.round(metalStoich * (1 + Math.random())));
          } else {
            counts[el] = Math.max(1, Math.round(metalStoich * (0.5 + Math.random() * 0.5)));
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

export function runPillarGuidedGeneration(
  targets: SCPillarTargets = DEFAULT_PILLAR_TARGETS,
  candidatesPerTemplate: number = 8,
): PillarGuidedCandidate[] {
  const results: PillarGuidedCandidate[] = [];
  const allFormulas: string[] = [];

  for (const template of DESIGN_TEMPLATES) {
    const formulas = generateFromTemplate(template, candidatesPerTemplate);
    allFormulas.push(...formulas);
  }

  totalGenerated += allFormulas.length;

  for (const formula of allFormulas) {
    try {
      const evaluation = evaluatePillars(formula, targets);
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

export function updatePillarWeightsFromReward(tcReward: number, evaluation: PillarEvaluation): void {
  rewardCount++;
  rewardBaseline = rewardBaseline * 0.99 + tcReward * 0.01;

  const lr = 0.003;
  const tcDelta = tcReward - rewardBaseline;
  const normalizedReward = Math.min(1.0, Math.max(-1.0, tcDelta / 100));

  const pillarEntries = Object.entries(evaluation.pillarScores) as [string, number][];
  for (const [pillar, score] of pillarEntries) {
    const key = pillar as keyof typeof pillarWeights;
    if (score >= 0.7 && normalizedReward > 0.1) {
      pillarWeights[key] = Math.min(0.5, pillarWeights[key] + lr);
    } else if (score < 0.3 && normalizedReward < -0.1) {
      pillarWeights[key] = Math.max(0.05, pillarWeights[key] - lr * 0.5);
    }
  }

  const totalWeight = Object.values(pillarWeights).reduce((s, w) => s + w, 0);
  for (const key of Object.keys(pillarWeights) as (keyof typeof pillarWeights)[]) {
    pillarWeights[key] /= totalWeight;
  }
}

const pillarDFTFeedback: { pillar: string; correct: number; total: number }[] = [];

export function incorporateDFTFeedbackIntoPillars(
  formula: string,
  predictedTc: number,
  actualTc: number,
  actualStable: boolean
): void {
  let evaluation: PillarEvaluation | null = null;
  try {
    evaluation = evaluatePillars(formula);
  } catch {
    return;
  }

  const predictionError = predictedTc - actualTc;
  const overestimated = predictionError > 20;
  const underestimated = predictionError < -20;
  const accurate = Math.abs(predictionError) <= 20;

  const lr = 0.005;
  const pillarEntries = Object.entries(evaluation.pillarScores) as [string, number][];

  for (const [pillar, score] of pillarEntries) {
    const key = pillar as keyof typeof pillarWeights;
    if (pillarWeights[key] === undefined) continue;

    let existing = pillarDFTFeedback.find(f => f.pillar === pillar);
    if (!existing) {
      existing = { pillar, correct: 0, total: 0 };
      pillarDFTFeedback.push(existing);
    }
    existing.total++;

    if (overestimated && score >= 0.6) {
      pillarWeights[key] = Math.max(0.04, pillarWeights[key] - lr);
    } else if (accurate && score >= 0.6) {
      pillarWeights[key] = Math.min(0.5, pillarWeights[key] + lr * 0.5);
      existing.correct++;
    } else if (underestimated && score < 0.4) {
      pillarWeights[key] = Math.min(0.5, pillarWeights[key] + lr * 0.3);
      existing.correct++;
    }

    if (!actualStable && score >= 0.5) {
      pillarWeights[key] = Math.max(0.04, pillarWeights[key] - lr * 0.3);
    }
  }

  const totalWeight = Object.values(pillarWeights).reduce((s, w) => s + w, 0);
  for (const key of Object.keys(pillarWeights) as (keyof typeof pillarWeights)[]) {
    pillarWeights[key] /= totalWeight;
  }
}

export function getPillarDFTFeedbackStats(): { pillar: string; accuracy: number; total: number }[] {
  return pillarDFTFeedback
    .filter(f => f.total >= 3)
    .map(f => ({ pillar: f.pillar, accuracy: f.correct / f.total, total: f.total }))
    .sort((a, b) => b.total - a.total);
}

export function runPillarCycle(
  existingFormulas: string[],
  targetTc: number = 200,
): { formulas: string[]; evaluations: PillarEvaluation[]; bestFormula: string; bestFitness: number; bestTc: number } {
  const targets: SCPillarTargets = {
    ...DEFAULT_PILLAR_TARGETS,
    minLambda: targetTc > 150 ? 2.0 : targetTc > 50 ? 1.5 : 1.0,
    minOmegaLogK: targetTc > 150 ? 800 : targetTc > 50 ? 600 : 400,
    minDOS: targetTc > 150 ? 3.0 : 2.0,
    minPairingGlue: targetTc > 150 ? 0.6 : targetTc > 50 ? 0.45 : 0.3,
    minInstability: targetTc > 150 ? 0.5 : targetTc > 50 ? 0.35 : 0.25,
    minHydrogenCage: targetTc > 150 ? 0.6 : 0.4,
  };

  const guided = runPillarGuidedGeneration(targets, 6);

  const reEvalExisting: PillarEvaluation[] = [];
  for (const f of existingFormulas.slice(0, 10)) {
    try {
      reEvalExisting.push(evaluatePillars(f, targets));
    } catch {}
  }

  for (const ex of reEvalExisting) {
    if (ex.compositeFitness > 0.6) {
      try {
        const mutated = mutateTowardWeakPillar(ex, targets);
        if (mutated) {
          const mutEval = evaluatePillars(mutated, targets);
          if (mutEval.compositeFitness > ex.compositeFitness) {
            guided.push({
              formula: mutated,
              evaluation: mutEval,
              designRationale: `Mutation of ${ex.formula}: improved ${ex.weakestPillar}`,
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

function mutateTowardWeakPillar(
  evaluation: PillarEvaluation,
  targets: SCPillarTargets,
): string | null {
  const counts = parseCounts(evaluation.formula);
  const elements = Object.keys(counts);
  const mutated = { ...counts };

  switch (evaluation.weakestPillar) {
    case "coupling": {
      const hasLight = elements.some(e => LIGHT_ATOMS.includes(e));
      if (!hasLight) {
        const lightEl = ["H", "B", "C"][Math.floor(Math.random() * 3)];
        mutated[lightEl] = 2 + Math.floor(Math.random() * 4);
      } else {
        for (const el of elements) {
          if (LIGHT_ATOMS.includes(el)) {
            mutated[el] = Math.min(10, mutated[el] + 2);
            break;
          }
        }
      }
      break;
    }
    case "phonon": {
      if (!elements.includes("H")) {
        mutated["H"] = 4 + Math.floor(Math.random() * 4);
      } else {
        mutated["H"] = Math.min(12, (mutated["H"] || 0) + 2);
      }
      break;
    }
    case "dos": {
      const hasTM = elements.some(e => HIGH_COUPLING_TM.includes(e));
      if (!hasTM) {
        const tm = HIGH_COUPLING_TM[Math.floor(Math.random() * HIGH_COUPLING_TM.length)];
        mutated[tm] = 1 + Math.floor(Math.random() * 2);
      } else {
        for (const el of elements) {
          if (HIGH_COUPLING_TM.includes(el)) {
            mutated[el] = Math.min(4, mutated[el] + 1);
            break;
          }
        }
      }
      break;
    }
    case "nesting": {
      const hasPnictogen = elements.some(e => PNICTOGEN_ELEMENTS.includes(e));
      if (!hasPnictogen && !elements.includes("O")) {
        const pn = PNICTOGEN_ELEMENTS[Math.floor(Math.random() * PNICTOGEN_ELEMENTS.length)];
        mutated[pn] = 2;
      }
      const hasLayerFormer = elements.some(e => LAYER_FORMERS.includes(e));
      if (!hasLayerFormer) {
        const lf = LAYER_FORMERS[Math.floor(Math.random() * LAYER_FORMERS.length)];
        mutated[lf] = 2;
      }
      break;
    }
    case "structure": {
      const hasCageFormer = elements.some(e => CAGE_FORMERS.includes(e));
      if (!hasCageFormer) {
        const cf = CAGE_FORMERS[Math.floor(Math.random() * CAGE_FORMERS.length)];
        mutated[cf] = 1;
      }
      if (!elements.includes("H") && !elements.includes("B")) {
        mutated["H"] = 6;
      }
      break;
    }
    case "pairingGlue": {
      if (evaluation.pairingGlue.spinFluctuationContribution < 0.3) {
        const hasMagnetic = elements.some(e => ["Fe", "Co", "Mn"].includes(e));
        if (!hasMagnetic) {
          mutated["Fe"] = 2;
          if (!elements.some(e => PNICTOGEN_ELEMENTS.includes(e))) {
            mutated["As"] = 2;
          }
        }
      } else {
        const hasLight = elements.some(e => LIGHT_ATOMS.includes(e));
        if (!hasLight) {
          mutated["H"] = 4 + Math.floor(Math.random() * 4);
        }
      }
      break;
    }
    case "instability": {
      if (evaluation.instability.mottProximity < 0.3) {
        if (!elements.includes("Cu") && !elements.includes("O")) {
          mutated["Cu"] = 1;
          mutated["O"] = 4;
          const hasRE = elements.some(e => isRareEarth(e));
          if (!hasRE) mutated["La"] = 1;
        }
      }
      if (evaluation.instability.vanHoveProximity < 0.3) {
        const hasTM = elements.some(e => isTransitionMetal(e));
        if (!hasTM) {
          mutated["V"] = 2;
        }
      }
      break;
    }
    case "hydrogenCage": {
      if (!elements.includes("H")) {
        mutated["H"] = 8;
      } else {
        mutated["H"] = Math.min(12, (mutated["H"] || 0) + 3);
      }
      const hasCage = elements.some(e => SODALITE_CAGE_ELEMENTS.includes(e));
      if (!hasCage) {
        mutated["La"] = 1;
      }
      break;
    }
  }

  const result = countsToFormula(mutated);
  if (result === evaluation.formula) return null;

  const constraint = checkPhysicsConstraints(result);
  if (!constraint.isValid && !constraint.repairedFormula) return null;

  return constraint.isValid ? result : constraint.repairedFormula;
}

export function getPillarOptimizerStats(): PillarOptimizerStats {
  const affinityScores: Record<string, number> = {};
  for (const [el, data] of Object.entries(elementAffinity)) {
    if (data.count >= 3) {
      affinityScores[el] = Math.round((data.totalFitness / data.count) * 1000) / 1000;
    }
  }

  const sorted = Object.entries(affinityScores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);
  const topAffinity: Record<string, number> = {};
  for (const [el, score] of sorted) topAffinity[el] = score;

  const satisfactionRates: Record<string, number> = {};
  for (const [key, count] of Object.entries(pillarSatisfied)) {
    satisfactionRates[key] = totalEvaluated > 0 ? Math.round((count / totalEvaluated) * 1000) / 1000 : 0;
  }

  return {
    totalEvaluated,
    totalGenerated,
    avgCompositeFitness: totalEvaluated > 0 ? Math.round((fitnessSum / totalEvaluated) * 1000) / 1000 : 0,
    bestFitness: Math.round(bestFitness * 1000) / 1000,
    bestFormula,
    pillarSatisfactionRates: satisfactionRates,
    elementAffinityScores: topAffinity,
    topCandidates: topCandidates.slice(0, 10),
    pillarWeights: { ...pillarWeights },
  };
}
