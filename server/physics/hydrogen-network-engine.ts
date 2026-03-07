import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  classifyHydrogenBonding,
  parseFormulaElements,
  type HydrogenBondingType,
} from "../learning/physics-engine";
import {
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "../learning/elemental-data";

export interface HHDistanceDistribution {
  estimatedMeanHH: number;
  estimatedMinHH: number;
  estimatedMaxHH: number;
  shortBondFraction: number;
  metallicBondFraction: number;
  distributionType: string;
}

export interface CageTopology {
  cageType: string;
  cageSymmetry: string;
  verticesPerCage: number;
  facesPerCage: number;
  cageVolume: number;
  cageCompleteness: number;
  sodaliteCharacter: number;
  clathrateCharacter: number;
}

export interface HydrogenNetworkAnalysis {
  formula: string;
  isHydride: boolean;
  hydrogenCount: number;
  hydrogenFraction: number;
  hydrogenToMetalRatio: number;
  bondingType: HydrogenBondingType;

  hydrogenNetworkDim: number;
  hydrogenCageScore: number;
  Hcoordination: number;
  hydrogenConnectivity: number;
  hydrogenPhononCouplingScore: number;

  hhDistribution: HHDistanceDistribution;
  cageTopology: CageTopology;
  hydrogenDensity: number;
  networkPercolation: number;
  phononContribution: {
    hydrogenPhononFreq: number;
    hydrogenPhononLambda: number;
    anharmonicCorrection: number;
  };

  compositeSCScore: number;
  networkClass: string;
  insights: string[];
}

const SODALITE_CAGE_ELEMENTS = ["La", "Y", "Ce", "Th", "Ac", "Ca", "Sr", "Ba"];
const CLATHRATE_CAGE_ELEMENTS = ["La", "Y", "Ca", "Ba", "Sr", "Sc", "Ce"];

const CAGE_TEMPLATES: Record<string, { vertices: number; faces: number; symmetry: string; baseVolume: number }> = {
  "H24-sodalite": { vertices: 24, faces: 14, symmetry: "Oh", baseVolume: 58.0 },
  "H32-clathrate-I": { vertices: 32, faces: 18, symmetry: "Pm-3n", baseVolume: 72.0 },
  "H29-clathrate-II": { vertices: 29, faces: 16, symmetry: "Fd-3m", baseVolume: 65.0 },
  "H20-dodecahedron": { vertices: 20, faces: 12, symmetry: "Ih", baseVolume: 45.0 },
  "H16-truncated-tetrahedron": { vertices: 16, faces: 8, symmetry: "Td", baseVolume: 35.0 },
  "H12-icosahedron": { vertices: 12, faces: 20, symmetry: "Ih", baseVolume: 25.0 },
  "H8-cube": { vertices: 8, faces: 6, symmetry: "Oh", baseVolume: 15.0 },
  "H6-octahedron": { vertices: 6, faces: 8, symmetry: "Oh", baseVolume: 10.0 },
};

function parseFormulaCounts(formula: string): Record<string, number> {
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

function estimateHHDistances(
  hCount: number,
  metalElements: string[],
  metalAtomCount: number,
  hRatio: number,
  bondingType: HydrogenBondingType,
): HHDistanceDistribution {
  let meanHH = 2.0;
  let minHH = 1.6;
  let maxHH = 3.0;

  if (bondingType === "metallic-network" || bondingType === "cage-clathrate") {
    const hasHeavyMetal = metalElements.some(e => {
      const d = getElementData(e);
      return d && d.atomicMass > 88;
    });

    if (hRatio >= 10) {
      meanHH = 1.05;
      minHH = 0.85;
      maxHH = 1.30;
    } else if (hRatio >= 6) {
      meanHH = 1.15;
      minHH = 0.95;
      maxHH = 1.45;
    } else if (hRatio >= 4) {
      meanHH = 1.30;
      minHH = 1.05;
      maxHH = 1.65;
    } else {
      meanHH = 1.55;
      minHH = 1.20;
      maxHH = 2.00;
    }

    if (hasHeavyMetal) {
      meanHH *= 0.95;
      minHH *= 0.93;
    }
  } else if (bondingType === "interstitial") {
    meanHH = 2.10;
    minHH = 1.80;
    maxHH = 2.80;
  } else if (bondingType === "covalent-molecular") {
    meanHH = 0.74;
    minHH = 0.70;
    maxHH = 1.00;
  }

  const shortBondFraction = minHH < 1.2 ? Math.min(1.0, (1.2 - minHH) / 0.5) : 0;
  const metallicBondFraction = meanHH < 1.5 && bondingType !== "covalent-molecular"
    ? Math.min(1.0, (1.5 - meanHH) / 0.6)
    : 0;

  let distributionType = "sparse";
  if (metallicBondFraction > 0.7) distributionType = "metallic-condensed";
  else if (metallicBondFraction > 0.3) distributionType = "intermediate";
  else if (shortBondFraction > 0.5) distributionType = "molecular";

  return {
    estimatedMeanHH: Number(meanHH.toFixed(3)),
    estimatedMinHH: Number(minHH.toFixed(3)),
    estimatedMaxHH: Number(maxHH.toFixed(3)),
    shortBondFraction: Number(shortBondFraction.toFixed(4)),
    metallicBondFraction: Number(metallicBondFraction.toFixed(4)),
    distributionType,
  };
}

function classifyCageTopology(
  hCount: number,
  hRatio: number,
  metalElements: string[],
  bondingType: HydrogenBondingType,
): CageTopology {
  const hasSodaliteFormer = metalElements.some(e => SODALITE_CAGE_ELEMENTS.includes(e));
  const hasClathFormer = metalElements.some(e => CLATHRATE_CAGE_ELEMENTS.includes(e));

  if (hCount === 0 || hRatio < 2 || (bondingType !== "cage-clathrate" && bondingType !== "metallic-network")) {
    return {
      cageType: "none",
      cageSymmetry: "N/A",
      verticesPerCage: 0,
      facesPerCage: 0,
      cageVolume: 0,
      cageCompleteness: 0,
      sodaliteCharacter: 0,
      clathrateCharacter: 0,
    };
  }

  let bestTemplate = "H8-cube";
  let sodaliteChar = 0;
  let clathrateChar = 0;

  if (hRatio >= 10 && hasSodaliteFormer) {
    bestTemplate = "H32-clathrate-I";
    sodaliteChar = 0.3;
    clathrateChar = 0.95;
  } else if (hRatio >= 8 && hasSodaliteFormer) {
    bestTemplate = "H24-sodalite";
    sodaliteChar = 0.95;
    clathrateChar = 0.4;
  } else if (hRatio >= 6 && hasClathFormer) {
    bestTemplate = "H20-dodecahedron";
    sodaliteChar = 0.6;
    clathrateChar = 0.7;
  } else if (hRatio >= 6) {
    bestTemplate = "H16-truncated-tetrahedron";
    sodaliteChar = 0.3;
    clathrateChar = 0.5;
  } else if (hRatio >= 4) {
    bestTemplate = "H12-icosahedron";
    sodaliteChar = 0.2;
    clathrateChar = 0.4;
  } else if (hRatio >= 3) {
    bestTemplate = "H8-cube";
    sodaliteChar = 0.1;
    clathrateChar = 0.2;
  } else {
    bestTemplate = "H6-octahedron";
    sodaliteChar = 0.05;
    clathrateChar = 0.1;
  }

  const template = CAGE_TEMPLATES[bestTemplate];
  const completeness = Math.min(1.0, hCount / template.vertices);

  const metalRadiiSum = metalElements.reduce((s, e) => {
    const d = getElementData(e);
    return s + (d ? d.atomicRadius : 150);
  }, 0);
  const avgMetalRadius = metalElements.length > 0 ? metalRadiiSum / metalElements.length : 150;
  const volumeScale = Math.pow(avgMetalRadius / 150, 3);

  return {
    cageType: bestTemplate,
    cageSymmetry: template.symmetry,
    verticesPerCage: template.vertices,
    facesPerCage: template.faces,
    cageVolume: Number((template.baseVolume * volumeScale).toFixed(2)),
    cageCompleteness: Number(completeness.toFixed(4)),
    sodaliteCharacter: Number(sodaliteChar.toFixed(4)),
    clathrateCharacter: Number(clathrateChar.toFixed(4)),
  };
}

function computeNetworkDimensionality(
  hRatio: number,
  bondingType: HydrogenBondingType,
  hhDist: HHDistanceDistribution,
): number {
  if (bondingType === "none") return 0;
  if (bondingType === "covalent-molecular") return 0.5;
  if (bondingType === "interstitial") return 1.0 + Math.min(1.0, hRatio * 0.3);

  let dim = 1.0;

  if (hhDist.metallicBondFraction > 0.7) {
    dim = 3.0;
  } else if (hhDist.metallicBondFraction > 0.3) {
    dim = 2.0 + hhDist.metallicBondFraction;
  } else if (hRatio >= 6) {
    dim = 2.5 + Math.min(0.5, (hRatio - 6) * 0.1);
  } else if (hRatio >= 4) {
    dim = 2.0 + (hRatio - 4) * 0.25;
  } else if (hRatio >= 2) {
    dim = 1.5 + (hRatio - 2) * 0.25;
  }

  return Number(Math.min(3.0, dim).toFixed(2));
}

function computeCoordinationNumber(
  hRatio: number,
  bondingType: HydrogenBondingType,
  cageTopology: CageTopology,
): number {
  if (bondingType === "none") return 0;
  if (bondingType === "covalent-molecular") return 1;
  if (bondingType === "interstitial") return Math.min(6, Math.round(2 + hRatio * 0.5));

  let coord = 2;

  if (cageTopology.cageType.includes("sodalite") || cageTopology.cageType.includes("clathrate")) {
    const vertices = cageTopology.verticesPerCage;
    if (vertices >= 24) coord = 4;
    else if (vertices >= 16) coord = 3;
    else coord = 3;

    if (cageTopology.cageCompleteness > 0.8) coord += 1;
  } else if (hRatio >= 6) {
    coord = 4;
    if (hRatio >= 10) coord = 5;
  } else if (hRatio >= 4) {
    coord = 3;
  }

  return Math.min(6, coord);
}

function computeHydrogenConnectivity(
  networkDim: number,
  coordination: number,
  hhDist: HHDistanceDistribution,
  cageTopology: CageTopology,
): number {
  const dimContrib = networkDim / 3.0;
  const coordContrib = coordination / 6.0;
  const bondContrib = hhDist.metallicBondFraction;
  const cageContrib = Math.max(cageTopology.sodaliteCharacter, cageTopology.clathrateCharacter);

  const connectivity = 0.30 * dimContrib + 0.25 * coordContrib + 0.25 * bondContrib + 0.20 * cageContrib;
  return Number(Math.min(1.0, connectivity).toFixed(4));
}

function computeHydrogenPhononCouplingScore(
  formula: string,
  hRatio: number,
  hhDist: HHDistanceDistribution,
  bondingType: HydrogenBondingType,
): number {
  if (bondingType === "none") return 0;
  if (bondingType === "covalent-molecular") return 0.05;

  try {
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

    const lambda = coupling.lambda;
    const maxPhonon = phonon.maxPhononFrequency;

    let hPhononFraction = 0;
    if (hRatio >= 6 && maxPhonon > 1000) {
      hPhononFraction = 0.7 + Math.min(0.25, (hRatio - 6) * 0.05);
    } else if (hRatio >= 4 && maxPhonon > 800) {
      hPhononFraction = 0.5 + Math.min(0.2, (hRatio - 4) * 0.1);
    } else if (hRatio >= 2) {
      hPhononFraction = 0.3 + Math.min(0.2, (hRatio - 2) * 0.05);
    } else {
      hPhononFraction = 0.1 + hRatio * 0.1;
    }

    if (hhDist.metallicBondFraction > 0.5) {
      hPhononFraction = Math.min(1.0, hPhononFraction * 1.2);
    }

    const hLambda = lambda * hPhononFraction;
    const score = Math.min(1.0, hLambda / 2.0);

    const anharmonicBonus = phonon.anharmonicityIndex > 0.3
      ? Math.min(0.15, phonon.anharmonicityIndex * 0.2)
      : 0;

    return Number(Math.min(1.0, score + anharmonicBonus).toFixed(4));
  } catch {
    let baseScore = 0.1;
    if (bondingType === "metallic-network") baseScore = 0.6;
    else if (bondingType === "cage-clathrate") baseScore = 0.5;
    else if (bondingType === "interstitial") baseScore = 0.2;
    return Number(Math.min(1.0, baseScore + hRatio * 0.03).toFixed(4));
  }
}

function computeHydrogenDensity(
  hCount: number,
  metalElements: string[],
  counts: Record<string, number>,
): number {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms === 0) return 0;

  let totalVolume = 0;
  for (const el of Object.keys(counts)) {
    const d = getElementData(el);
    const radius = d ? d.atomicRadius / 100 : 1.5;
    totalVolume += (4 / 3) * Math.PI * Math.pow(radius, 3) * counts[el];
  }

  if (totalVolume === 0) return 0;
  return Number((hCount / totalVolume).toFixed(4));
}

function computeCompositeSCScore(
  networkDim: number,
  cageScore: number,
  coordination: number,
  connectivity: number,
  phononCoupling: number,
  hhDist: HHDistanceDistribution,
): number {
  const dimContrib = networkDim / 3.0;
  const cageContrib = cageScore;
  const coordContrib = coordination / 6.0;
  const connContrib = connectivity;
  const phononContrib = phononCoupling;
  const bondContrib = hhDist.metallicBondFraction;

  const composite =
    0.15 * dimContrib +
    0.20 * cageContrib +
    0.10 * coordContrib +
    0.15 * connContrib +
    0.25 * phononContrib +
    0.15 * bondContrib;

  return Number(Math.min(1.0, composite).toFixed(4));
}

function classifyNetworkClass(
  bondingType: HydrogenBondingType,
  networkDim: number,
  cageTopology: CageTopology,
  hRatio: number,
): string {
  if (bondingType === "none") return "non-hydride";
  if (bondingType === "covalent-molecular") return "molecular-hydrogen";

  if (cageTopology.sodaliteCharacter > 0.7) return "sodalite-cage";
  if (cageTopology.clathrateCharacter > 0.7) return "clathrate-cage";
  if (cageTopology.clathrateCharacter > 0.3 && cageTopology.sodaliteCharacter > 0.3) return "mixed-cage";

  if (networkDim >= 2.5 && hRatio >= 6) return "3D-metallic-network";
  if (networkDim >= 2.0) return "2D-layered-network";
  if (networkDim >= 1.5) return "1D-chain-network";

  if (bondingType === "interstitial") return "interstitial-hydride";
  return "disordered-hydrogen";
}

function generateInsights(analysis: Omit<HydrogenNetworkAnalysis, "insights">): string[] {
  const insights: string[] = [];

  if (!analysis.isHydride) {
    insights.push("Non-hydride compound - hydrogen network analysis not applicable");
    return insights;
  }

  if (analysis.hydrogenToMetalRatio >= 10) {
    insights.push(`Extreme hydrogen ratio (H:M = ${analysis.hydrogenToMetalRatio.toFixed(1)}) suggests dense metallic hydrogen sublattice`);
  } else if (analysis.hydrogenToMetalRatio >= 6) {
    insights.push(`High hydrogen ratio (H:M = ${analysis.hydrogenToMetalRatio.toFixed(1)}) consistent with clathrate-like cage structures`);
  }

  if (analysis.hhDistribution.metallicBondFraction > 0.7) {
    insights.push(`Metallic H-H bonds dominate (${(analysis.hhDistribution.metallicBondFraction * 100).toFixed(0)}%) - strong phonon-mediated pairing expected`);
  }

  if (analysis.hydrogenNetworkDim >= 2.5) {
    insights.push(`3D hydrogen network (dim=${analysis.hydrogenNetworkDim}) provides high DOS and strong electron-phonon coupling`);
  }

  if (analysis.cageTopology.sodaliteCharacter > 0.5) {
    insights.push(`Sodalite-like cage topology (score=${analysis.cageTopology.sodaliteCharacter.toFixed(2)}) - optimal for high-Tc superconductivity`);
  }
  if (analysis.cageTopology.clathrateCharacter > 0.5) {
    insights.push(`Clathrate-like cage topology (score=${analysis.cageTopology.clathrateCharacter.toFixed(2)}) - favorable for phonon-mediated pairing`);
  }

  if (analysis.hydrogenPhononCouplingScore > 0.6) {
    insights.push(`Strong hydrogen phonon coupling (score=${analysis.hydrogenPhononCouplingScore.toFixed(3)}) - dominant contribution to lambda`);
  }

  if (analysis.Hcoordination >= 4) {
    insights.push(`High H coordination number (${analysis.Hcoordination}) indicates well-connected hydrogen sublattice`);
  }

  if (analysis.hydrogenDensity > 0.5) {
    insights.push(`High hydrogen density (${analysis.hydrogenDensity.toFixed(3)}) enhances phonon frequencies and coupling`);
  }

  if (analysis.networkPercolation > 0.8) {
    insights.push("Hydrogen network fully percolated - continuous metallic H sublattice");
  }

  return insights;
}

export function analyzeHydrogenNetwork(formula: string): HydrogenNetworkAnalysis {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const hCount = counts["H"] || 0;

  const nonmetals = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const metalElements = elements.filter(e => !nonmetals.includes(e));
  const metalAtomCount = metalElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;
  const hFraction = hCount / totalAtoms;

  const isHydride = hCount > 0;

  const estimatedPressure = hRatio >= 6 ? 150 : hRatio >= 4 ? 100 : hRatio >= 2 ? 50 : 0;
  const bondingType = classifyHydrogenBonding(formula, estimatedPressure);

  const hhDist = estimateHHDistances(hCount, metalElements, metalAtomCount, hRatio, bondingType);

  const cageTopology = classifyCageTopology(hCount, hRatio, metalElements, bondingType);

  const networkDim = computeNetworkDimensionality(hRatio, bondingType, hhDist);

  const cageScore = Math.max(cageTopology.sodaliteCharacter, cageTopology.clathrateCharacter);

  const coordination = computeCoordinationNumber(hRatio, bondingType, cageTopology);

  const connectivity = computeHydrogenConnectivity(networkDim, coordination, hhDist, cageTopology);

  const phononCoupling = computeHydrogenPhononCouplingScore(formula, hRatio, hhDist, bondingType);

  const hydrogenDensity = computeHydrogenDensity(hCount, metalElements, counts);

  let networkPercolation = 0;
  if (bondingType === "metallic-network") {
    networkPercolation = Math.min(1.0, 0.7 + hRatio * 0.03);
  } else if (bondingType === "cage-clathrate") {
    networkPercolation = Math.min(1.0, 0.5 + cageTopology.cageCompleteness * 0.4);
  } else if (bondingType === "interstitial") {
    networkPercolation = Math.min(0.6, hRatio * 0.15);
  }

  const compositeSCScore = computeCompositeSCScore(
    networkDim, cageScore, coordination, connectivity, phononCoupling, hhDist
  );

  const networkClass = classifyNetworkClass(bondingType, networkDim, cageTopology, hRatio);

  let hydrogenPhononFreq = 0;
  let hydrogenPhononLambda = 0;
  let anharmonicCorrection = 1.0;
  try {
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

    hydrogenPhononFreq = phonon.maxPhononFrequency * (hRatio >= 4 ? 0.8 : 0.5);
    hydrogenPhononLambda = coupling.lambda * (hRatio >= 6 ? 0.7 : hRatio >= 4 ? 0.5 : 0.3);
    anharmonicCorrection = coupling.anharmonicCorrectionFactor;
  } catch {}

  const partial: Omit<HydrogenNetworkAnalysis, "insights"> = {
    formula,
    isHydride,
    hydrogenCount: hCount,
    hydrogenFraction: Number(hFraction.toFixed(4)),
    hydrogenToMetalRatio: Number(hRatio.toFixed(2)),
    bondingType,
    hydrogenNetworkDim: networkDim,
    hydrogenCageScore: Number(cageScore.toFixed(4)),
    Hcoordination: coordination,
    hydrogenConnectivity: Number(connectivity.toFixed(4)),
    hydrogenPhononCouplingScore: phononCoupling,
    hhDistribution: hhDist,
    cageTopology,
    hydrogenDensity,
    networkPercolation: Number(networkPercolation.toFixed(4)),
    phononContribution: {
      hydrogenPhononFreq: Number(hydrogenPhononFreq.toFixed(2)),
      hydrogenPhononLambda: Number(hydrogenPhononLambda.toFixed(4)),
      anharmonicCorrection: Number(anharmonicCorrection.toFixed(4)),
    },
    compositeSCScore,
    networkClass,
  };

  const insights = generateInsights(partial);

  return { ...partial, insights };
}

export function extractHydrogenNetworkFeatures(formula: string): Record<string, number> {
  const analysis = analyzeHydrogenNetwork(formula);
  return {
    hydrogenNetworkDim: analysis.hydrogenNetworkDim,
    hydrogenCageScore: analysis.hydrogenCageScore,
    Hcoordination: analysis.Hcoordination,
    hydrogenConnectivity: analysis.hydrogenConnectivity,
    hydrogenPhononCouplingScore: analysis.hydrogenPhononCouplingScore,
  };
}

let totalAnalyzed = 0;
let totalHydrides = 0;
let networkClassCounts: Record<string, number> = {};
let avgCompositeSCScore = 0;

export function trackHydrogenNetworkResult(analysis: HydrogenNetworkAnalysis): void {
  totalAnalyzed++;
  if (analysis.isHydride) totalHydrides++;
  networkClassCounts[analysis.networkClass] = (networkClassCounts[analysis.networkClass] || 0) + 1;
  avgCompositeSCScore = (avgCompositeSCScore * (totalAnalyzed - 1) + analysis.compositeSCScore) / totalAnalyzed;
}

export function getHydrogenNetworkStats(): {
  totalAnalyzed: number;
  totalHydrides: number;
  networkClassDistribution: Record<string, number>;
  avgCompositeSCScore: number;
} {
  return {
    totalAnalyzed,
    totalHydrides,
    networkClassDistribution: { ...networkClassCounts },
    avgCompositeSCScore: Number(avgCompositeSCScore.toFixed(4)),
  };
}
