import type { MLFeatureVector } from "./ml-predictor";
import { getElementData } from "./elemental-data";

function parseFormulaElements(formula: string): string[] {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  return total > 0 ? total : 1;
}

interface FamilyFilterResult {
  pass: boolean;
  score: number;
  reasons: string[];
}

function applyMAXPhaseFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const M_ELEMENTS = ["Ti", "V", "Cr", "Nb", "Mo", "Zr", "Hf", "Ta", "W", "Sc"];
  const A_ELEMENTS = ["Al", "Si", "Ga", "Ge", "Sn", "In", "Tl", "Pb", "S", "As", "P"];

  const metalLayers = elements.filter(e => M_ELEMENTS.includes(e)).reduce((s, e) => s + (counts[e] || 0), 0);
  const aLayers = elements.filter(e => A_ELEMENTS.includes(e)).reduce((s, e) => s + (counts[e] || 0), 0);

  if (aLayers > 0) {
    const layerRatio = metalLayers / aLayers;
    if (layerRatio >= 2) {
      score += 0.25;
      reasons.push(`Layer ratio M/A = ${layerRatio.toFixed(1)} >= 2 (good for MAX phase)`);
    } else {
      pass = false;
      reasons.push(`Layer ratio M/A = ${layerRatio.toFixed(1)} < 2 (insufficient metal layers)`);
    }
  } else {
    pass = false;
    reasons.push("No A-element layers detected");
  }

  if (features.dosAtEF > 1.5) {
    score += 0.25;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} > 1.5 (sufficient states at Fermi level)`);
  } else {
    pass = false;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} <= 1.5 (insufficient density of states)`);
  }

  const maxPhononTHz = features.debyeTemperature > 0 ? features.debyeTemperature * 0.0208 : (features.logPhononFreq > 0 ? features.logPhononFreq / 33.356 : 0);
  if (maxPhononTHz > 5) {
    score += 0.25;
    reasons.push(`Max phonon ~ ${maxPhononTHz.toFixed(1)} THz > 5 THz (adequate phonon spectrum)`);
  } else {
    pass = false;
    reasons.push(`Max phonon ~ ${maxPhononTHz.toFixed(1)} THz <= 5 THz (weak phonon coupling expected)`);
  }

  const anisotropy = features.dimensionalityScore > 0.7 ? features.dimensionalityScore * 5 : 1.0;
  if (anisotropy > 3) {
    score += 0.15;
    reasons.push(`Anisotropy score ${anisotropy.toFixed(1)} > 3 (bonus for layered anisotropy)`);
  } else {
    reasons.push(`Anisotropy score ${anisotropy.toFixed(1)} <= 3 (no anisotropy bonus)`);
  }

  if (features.metallicity < 0.4) {
    pass = false;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} too low for MAX phase`);
  }

  score = Math.min(1.0, score);
  return { pass, score, reasons };
}

function applyBorideFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const bCount = counts["B"] || 0;
  const totalAtoms = getTotalAtoms(counts);
  const nonBElements = elements.filter(e => e !== "B");
  const metalCount = nonBElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const bPerMetal = metalCount > 0 ? bCount / metalCount : 0;
  const bCoordination = bPerMetal >= 2 ? 3 + Math.min(3, bPerMetal - 2) : bPerMetal * 1.5;

  if (bCoordination >= 3) {
    score += 0.25;
    reasons.push(`B-B coordination ~ ${bCoordination.toFixed(1)} >= 3 (B/M=${bPerMetal.toFixed(0)}, boron network formed)`);
  } else {
    pass = false;
    reasons.push(`B-B coordination ~ ${bCoordination.toFixed(1)} < 3 (B/M=${bPerMetal.toFixed(0)}, insufficient boron network)`);
  }

  const sigmaBandWeight = features.electronDensityEstimate * 100;
  if (sigmaBandWeight > 30) {
    score += 0.25;
    reasons.push(`Sigma band weight ~ ${sigmaBandWeight.toFixed(0)}% > 30% (strong covalent bonding)`);
  } else {
    pass = false;
    reasons.push(`Sigma band weight ~ ${sigmaBandWeight.toFixed(0)}% <= 30% (weak sigma bands)`);
  }

  const phononAvgTHz = features.logPhononFreq > 0 ? features.logPhononFreq / 33.356 : 0;
  if (phononAvgTHz > 20) {
    score += 0.25;
    reasons.push(`Phonon avg ~ ${phononAvgTHz.toFixed(1)} THz > 20 THz (high-frequency boron modes)`);
  } else {
    pass = false;
    reasons.push(`Phonon avg ~ ${phononAvgTHz.toFixed(1)} THz <= 20 THz (insufficient phonon frequencies)`);
  }

  const boronEN = 2.04;
  const metalENs = nonBElements.map(e => getElementData(e)?.paulingElectronegativity ?? 2.0);
  const allMetalsBelowBoron = metalENs.length > 0 && metalENs.every(en => en < boronEN);
  if (allMetalsBelowBoron) {
    score += 0.15;
    reasons.push("Metal electronegativity < boron (2.04): favorable charge transfer");
  } else {
    reasons.push("Some metals have electronegativity >= boron (2.04): less favorable charge transfer");
  }

  if (features.metallicity < 0.3) {
    pass = false;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} too low for metallic boride`);
  }

  score = Math.min(1.0, score);
  return { pass, score, reasons };
}

function applyHydrideFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const hCount = counts["H"] || 0;
  const metalElements = elements.filter(e => e !== "H");
  const metalCount = metalElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const hCoordination = metalCount > 0 ? hCount / metalCount : 0;

  if (hCoordination >= 4) {
    score += 0.25;
    reasons.push(`H coordination (H/M ratio) = ${hCoordination.toFixed(1)} >= 4 (cage/clathrate structure detected)`);
  } else {
    pass = false;
    reasons.push(`H coordination (H/M ratio) = ${hCoordination.toFixed(1)} < 4 (no cage structure)`);
  }

  const estimatedPressure = hCoordination >= 6 ? 150 : hCoordination >= 4 ? 100 : 50;
  if (estimatedPressure >= 50 && estimatedPressure <= 200) {
    score += 0.20;
    reasons.push(`Estimated pressure stability ~ ${estimatedPressure} GPa (within 50-200 GPa range)`);
  } else {
    pass = false;
    reasons.push(`Estimated pressure stability ~ ${estimatedPressure} GPa (outside 50-200 GPa range)`);
  }

  if (features.electronPhononLambda >= 1.5) {
    score += 0.30;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} >= 1.5 (strong electron-phonon coupling)`);
  } else if (features.electronPhononLambda >= 1.0) {
    score += 0.15;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} >= 1.0 (moderate coupling, borderline)`);
  } else {
    pass = false;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} < 1.0 (insufficient electron-phonon coupling for superhydride)`);
  }

  const muStar = 0.10;
  const lambda = features.electronPhononLambda;
  if (lambda > 0 && features.logPhononFreq > 0) {
    const denom = lambda - muStar * (1 + 0.62 * lambda);
    if (denom > 0) {
      const exponent = -1.04 * (1 + lambda) / denom;
      const tcMcMillan = (features.logPhononFreq / 1.2) * Math.exp(exponent);
      score += 0.10;
      reasons.push(`McMillan Tc estimate (mu*=0.10) ~ ${tcMcMillan.toFixed(1)} K`);
    }
  }

  if (features.metallicity < 0.5) {
    pass = false;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} too low for metallic hydride`);
  }

  score = Math.min(1.0, score);
  return { pass, score, reasons };
}

function applyNitrideFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const alkaliElements = ["Li", "Na", "K", "Rb", "Cs"];
  const alkaliValences: Record<string, number> = { Li: 1, Na: 1, K: 1, Rb: 1, Cs: 1 };
  let extraElectrons = 0;
  for (const a of alkaliElements) {
    if (counts[a]) {
      extraElectrons += counts[a] * (alkaliValences[a] || 1);
    }
  }

  if (extraElectrons > 0.1) {
    score += 0.30;
    reasons.push(`Electron doping: x*valence = ${extraElectrons.toFixed(2)} > 0.1 (sufficient carrier concentration)`);
  } else {
    pass = false;
    reasons.push(`Electron doping: x*valence = ${extraElectrons.toFixed(2)} <= 0.1 (insufficient carriers)`);
  }

  const layerSpacingIncrease = extraElectrons > 0 ? Math.min(25, extraElectrons * 8) : 0;
  if (layerSpacingIncrease > 5) {
    score += 0.25;
    reasons.push(`Layer spacing increase ~ ${layerSpacingIncrease.toFixed(0)}% > 5% (intercalation expanding layers)`);
  } else {
    pass = false;
    reasons.push(`Layer spacing increase ~ ${layerSpacingIncrease.toFixed(0)}% <= 5% (insufficient intercalation)`);
  }

  const is2DFermiSurface = features.dimensionalityScore >= 0.7 ||
    features.fermiSurfaceType.includes("2D") ||
    features.layeredStructure;
  if (is2DFermiSurface) {
    score += 0.25;
    reasons.push(`2D Fermi surface detected (dimensionality=${features.dimensionalityScore.toFixed(2)}, topology=${features.fermiSurfaceType})`);
  } else {
    pass = false;
    reasons.push(`No 2D Fermi surface detected (dimensionality=${features.dimensionalityScore.toFixed(2)})`);
  }

  if (features.metallicity < 0.3) {
    pass = false;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} too low for intercalated nitride`);
  } else {
    score += 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} adequate`);
  }

  score = Math.min(1.0, score);
  return { pass, score, reasons };
}

export function applyFamilyFilter(
  formula: string,
  family: string,
  features: MLFeatureVector
): FamilyFilterResult {
  switch (family) {
    case "MAX-phase":
      return applyMAXPhaseFilter(formula, features);
    case "Boride":
      return applyBorideFilter(formula, features);
    case "Hydride":
      return applyHydrideFilter(formula, features);
    case "Nitride":
    case "Intercalated-nitride":
      return applyNitrideFilter(formula, features);
    default:
      return { pass: false, score: 0, reasons: [`Unknown family: ${family}`] };
  }
}

export interface DiscoveryScoreInput {
  predictedTc: number;
  formula: string;
  hullDistance?: number | null;
  synthesisScore?: number | null;
  prototype?: string | null;
  existingFormulas?: string[];
}

export function computeDiscoveryScore(candidate: DiscoveryScoreInput): {
  discoveryScore: number;
  normalizedTc: number;
  noveltyScore: number;
  stabilityScore: number;
  synthesisFeasibility: number;
} {
  const normalizedTc = Math.min(1.0, Math.max(0, (candidate.predictedTc || 0) / 300));

  const hullDist = candidate.hullDistance ?? 0.05;
  const stabilityScore = Math.min(1.0, Math.max(0, 1.0 - hullDist / 0.1));

  const synthesisFeasibility = Math.min(1.0, Math.max(0, candidate.synthesisScore ?? 0.5));

  let noveltyScore = 0.5;

  const elements = parseFormulaElements(candidate.formula);
  const counts = parseFormulaCounts(candidate.formula);
  const totalAtoms = getTotalAtoms(counts);

  if (elements.length >= 4) {
    noveltyScore += 0.15;
  } else if (elements.length >= 3) {
    noveltyScore += 0.05;
  }

  const rareElements = ["Sc", "Y", "Hf", "Ta", "Re", "Os", "Ir", "Ru", "Rh", "Pd"];
  const hasRare = elements.some(e => rareElements.includes(e));
  if (hasRare) {
    noveltyScore += 0.1;
  }

  if (candidate.prototype) {
    const exploredPrototypes = ["Perovskite", "A15", "ThCr2Si2"];
    const isExplored = exploredPrototypes.some(p =>
      candidate.prototype!.toLowerCase().includes(p.toLowerCase())
    );
    if (!isExplored) {
      noveltyScore += 0.15;
    }
  }

  if (candidate.existingFormulas && candidate.existingFormulas.length > 0) {
    let minDistance = 1.0;
    for (const existing of candidate.existingFormulas) {
      const existingElements = parseFormulaElements(existing);
      const allElements = Array.from(new Set(elements.concat(existingElements)));
      const commonElements = elements.filter(e => existingElements.includes(e));
      const jaccard = allElements.length > 0 ? commonElements.length / allElements.length : 0;
      const distance = 1.0 - jaccard;
      if (distance < minDistance) {
        minDistance = distance;
      }
    }
    noveltyScore += minDistance * 0.1;
  }

  noveltyScore = Math.min(1.0, Math.max(0, noveltyScore));

  const discoveryScore = 0.4 * normalizedTc + 0.3 * noveltyScore + 0.2 * stabilityScore + 0.1 * synthesisFeasibility;

  return {
    discoveryScore: Math.round(discoveryScore * 1000) / 1000,
    normalizedTc,
    noveltyScore: Math.round(noveltyScore * 1000) / 1000,
    stabilityScore: Math.round(stabilityScore * 1000) / 1000,
    synthesisFeasibility: Math.round(synthesisFeasibility * 1000) / 1000,
  };
}

export function rankCandidate(
  formula: string,
  family: string,
  features: MLFeatureVector,
  gbScore: { tcPredicted: number; score: number }
): number {
  const tcNormalized = Math.min(1.0, gbScore.tcPredicted / 300);

  const stabilityRaw = features.metallicity * 0.4 +
    (features.formationEnergy !== null ? Math.max(0, 1 - Math.abs(features.formationEnergy) / 2) : 0.5) * 0.3 +
    (features.stability !== null ? features.stability : 0.5) * 0.3;
  const stability = Math.min(1.0, Math.max(0, stabilityRaw));

  const lambdaNormalized = Math.min(1.0, features.electronPhononLambda / 3.0);

  let synthesisFeasibility = 0.5;
  if (family === "Hydride") {
    synthesisFeasibility = features.hydrogenRatio >= 6 ? 0.3 : 0.5;
  } else if (family === "MAX-phase") {
    synthesisFeasibility = 0.7;
  } else if (family === "Boride") {
    synthesisFeasibility = 0.6;
  } else if (family === "Nitride") {
    synthesisFeasibility = 0.55;
  }

  const composite = 0.35 * tcNormalized +
    0.25 * stability +
    0.20 * lambdaNormalized +
    0.20 * synthesisFeasibility;

  return Math.round(composite * 1000) / 1000;
}
