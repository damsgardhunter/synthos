import type { MLFeatureVector } from "./ml-predictor";
import { getElementData } from "./elemental-data";

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
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

  const M_ELEMENTS = ["Ti", "V", "Cr", "Nb", "Mo", "Zr", "Hf", "Ta", "W", "Sc", "Y"];
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

  if (features.metallicity < 0.3) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for MAX phase (heavy score penalty)`);
  } else if (features.metallicity < 0.4) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for MAX phase (score penalized)`);
  }

  score = Math.max(0, Math.min(1.0, score));
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

  const is2DBoronSheet = features.dimensionalityScore >= 0.6 || features.layeredStructure;
  if (is2DBoronSheet) {
    score += 0.15;
    reasons.push(`2D boron sheet character detected (dimensionality=${features.dimensionalityScore.toFixed(2)}, layered=${features.layeredStructure})`);
  } else if (bCoordination >= 3) {
    score -= 0.10;
    reasons.push(`Low anisotropy (dimensionality=${features.dimensionalityScore.toFixed(2)}): boron network may be 3D cluster, not superconducting sheet`);
  }

  if (features.metallicity < 0.2) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for metallic boride (heavy score penalty)`);
  } else if (features.metallicity < 0.3) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for metallic boride (score penalized)`);
  }

  score = Math.max(0, Math.min(1.0, score));
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

  const totalAtoms = getTotalAtoms(counts);
  let sumAtomicVol = 0;
  let sumCubeR = 0;
  let atomsWithData = 0;
  for (const [el, n] of Object.entries(counts)) {
    const d = getElementData(el);
    if (d) {
      const rA = (d.atomicRadius || 150) / 100;
      sumAtomicVol += n * (4 / 3) * Math.PI * Math.pow(rA, 3);
      sumCubeR += n * Math.pow(rA, 3);
      atomsWithData += n;
    }
  }
  const volPerAtom = atomsWithData > 0 ? sumAtomicVol / atomsWithData : 14.0;
  const avgCubeR = atomsWithData > 0 ? sumCubeR / atomsWithData : 3.375;
  const cellVolEstimate = totalAtoms * Math.pow(2 * Math.pow(avgCubeR, 1 / 3), 3);
  const packingFraction = cellVolEstimate > 0 ? sumAtomicVol / cellVolEstimate : 0.52;

  const volumeFactor = volPerAtom > 12 ? 1.4 : volPerAtom > 8 ? 1.15 : volPerAtom < 4 ? 0.7 : 1.0;
  const packingAdjust = packingFraction > 0.68 ? 0.8 : packingFraction < 0.5 ? 1.3 : 1.0;

  const basePressure = hCoordination >= 6 ? 150 : hCoordination >= 4 ? 100 : 50;
  const estimatedPressure = Math.round(basePressure * volumeFactor * packingAdjust);

  if (estimatedPressure >= 30 && estimatedPressure <= 250) {
    score += 0.20;
    reasons.push(`Estimated pressure stability ~ ${estimatedPressure} GPa (vol/atom=${volPerAtom.toFixed(1)} A^3, APF=${packingFraction.toFixed(2)})`);
  } else {
    pass = false;
    reasons.push(`Estimated pressure stability ~ ${estimatedPressure} GPa (outside 30-250 GPa range, vol/atom=${volPerAtom.toFixed(1)} A^3)`);
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
    if (denom >= 0.1) {
      const lambdaBar = 2.46 * (1 + 3.8 * muStar);
      const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);
      const exponent = -1.04 * (1 + lambda) / denom;
      const omegaLogK = features.logPhononFreq * 1.4388;
      const tcMcMillan = Math.min((omegaLogK / 1.2) * f1 * Math.exp(exponent), 500);
      score += 0.10;
      reasons.push(`McMillan Tc estimate (mu*=0.10) ~ ${tcMcMillan.toFixed(1)} K`);
    } else if (denom > 0) {
      reasons.push(`McMillan denom=${denom.toFixed(3)} too small for reliable Tc (near instability)`);
    }
  }

  if (features.metallicity < 0.35) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for metallic hydride (heavy score penalty)`);
  } else if (features.metallicity < 0.5) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for metallic hydride (score penalized)`);
  }

  score = Math.max(0, Math.min(1.0, score));
  return { pass, score, reasons };
}

function applyNitrideFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const donorValences: Record<string, number> = {
    Li: 1, Na: 1, K: 1, Rb: 1, Cs: 1,
    Ca: 2, Sr: 2, Ba: 2, Mg: 2,
    Eu: 2, Yb: 2,
    La: 3, Ce: 3, Pr: 3, Nd: 3, Sm: 3, Gd: 3, Y: 3,
  };
  const donorElements = Object.keys(donorValences);
  const nCount = counts["N"] || 0;
  let rawDonorElectrons = 0;
  for (const a of donorElements) {
    if (counts[a]) {
      rawDonorElectrons += counts[a] * donorValences[a];
    }
  }
  const normBase = nCount > 0 ? nCount : getTotalAtoms(counts);
  const extraElectrons = normBase > 0 ? rawDonorElectrons / normBase : 0;

  if (extraElectrons > 0.1) {
    score += 0.30;
    reasons.push(`Electron doping: ${rawDonorElectrons.toFixed(2)} e / ${normBase.toFixed(1)} N = ${extraElectrons.toFixed(2)} e/N > 0.1 (sufficient carrier concentration)`);
  } else {
    pass = false;
    reasons.push(`Electron doping: ${rawDonorElectrons.toFixed(2)} e / ${normBase.toFixed(1)} N = ${extraElectrons.toFixed(2)} e/N <= 0.1 (insufficient carriers)`);
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

  if (features.metallicity < 0.2) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for intercalated nitride (heavy score penalty)`);
  } else if (features.metallicity < 0.3) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for intercalated nitride (score penalized)`);
  } else {
    score += 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} adequate`);
  }

  score = Math.max(0, Math.min(1.0, score));
  return { pass, score, reasons };
}

function applyKagomeFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const KAGOME_METALS = ["V", "Ti", "Cr", "Mn", "Fe", "Co", "Ni"];
  const SUBLATTICE_ELEMENTS = ["Sb", "Bi", "Sn", "Ge", "As", "P", "S", "Se", "Te"];

  const kagomeMetalCount = elements
    .filter(e => KAGOME_METALS.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const sublatticeCount = elements
    .filter(e => SUBLATTICE_ELEMENTS.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);

  if (kagomeMetalCount >= 3) {
    score += 0.25;
    reasons.push(`Kagome metal count = ${kagomeMetalCount} >= 3 (frustrated lattice sites present)`);
  } else {
    pass = false;
    reasons.push(`Kagome metal count = ${kagomeMetalCount} < 3 (insufficient frustrated lattice sites)`);
  }

  if (sublatticeCount >= 4) {
    score += 0.20;
    reasons.push(`Sublattice element count = ${sublatticeCount} >= 4 (adequate pnictogen/chalcogen/metalloid sublattice)`);
  } else {
    reasons.push(`Sublattice element count = ${sublatticeCount} < 4 (sparse sublattice)`);
  }

  if (features.dosAtEF > 2.0) {
    score += 0.25;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} > 2.0 (van Hove singularity proximity)`);
  } else if (features.dosAtEF > 1.0) {
    score += 0.10;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} > 1.0 (moderate states at Fermi level)`);
  } else {
    pass = false;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} <= 1.0 (insufficient DOS for Kagome superconductivity)`);
  }

  const totalAtoms = getTotalAtoms(counts);
  let totalValence = 0;
  for (const [el, n] of Object.entries(counts)) {
    const d = getElementData(el);
    if (d) {
      totalValence += (d.valenceElectrons ?? 0) * n;
    }
  }
  const vecPerSite = totalAtoms > 0 ? totalValence / totalAtoms : 0;
  const kagomeFilling5_12 = 5 / 12;
  const kagomeFilling3_4 = 3 / 4;
  const fracFilling = vecPerSite > 0 ? (vecPerSite % 1) : 0;
  const dist5_12 = Math.abs(fracFilling - kagomeFilling5_12);
  const dist3_4 = Math.abs(fracFilling - kagomeFilling3_4);
  const minFillingDist = Math.min(dist5_12, dist3_4);
  if (minFillingDist < 0.05) {
    score += 0.15;
    const matchType = dist5_12 < dist3_4 ? "5/12" : "3/4";
    reasons.push(`VEC filling ${fracFilling.toFixed(3)} near Kagome ${matchType} VHS (dist=${minFillingDist.toFixed(3)}): symmetry-match bonus`);
  } else if (minFillingDist < 0.10) {
    score += 0.05;
    reasons.push(`VEC filling ${fracFilling.toFixed(3)} moderately near Kagome VHS (dist=${minFillingDist.toFixed(3)})`);
  }

  if (features.bandFlatness > 0.5) {
    score += 0.10;
    reasons.push(`Flat band indicator = ${features.bandFlatness.toFixed(2)} > 0.5 (Kagome flat band character)`);
  }

  const is2DLike = features.dimensionalityScore >= 0.6 || features.layeredStructure;
  if (is2DLike) {
    score += 0.15;
    reasons.push(`2D character detected (dimensionality=${features.dimensionalityScore.toFixed(2)}, layered=${features.layeredStructure})`);
  } else {
    reasons.push(`Limited 2D character (dimensionality=${features.dimensionalityScore.toFixed(2)})`);
  }

  if (features.electronPhononLambda >= 0.5) {
    score += 0.15;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} >= 0.5 (adequate e-ph coupling for Kagome)`);
  } else {
    pass = false;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} < 0.5 (weak electron-phonon coupling)`);
  }

  if (features.metallicity < 0.1 && features.mottProximityScore > 0.5) {
    score -= 0.05;
    reasons.push(`Near-Mott insulator (metallicity=${features.metallicity.toFixed(2)}, Mott proximity=${features.mottProximityScore.toFixed(2)}): may superconduct upon doping`);
  } else if (features.metallicity < 0.3) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for Kagome metal (heavy score penalty)`);
  } else if (features.metallicity < 0.4) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for Kagome metal (score penalized)`);
  }

  score = Math.max(0, Math.min(1.0, score));
  return { pass, score, reasons };
}

function applyMixedMechanismFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const MAGNETIC_TM = ["Fe", "Ni", "Co", "Cu", "Mn", "Cr"];
  const magneticCount = elements
    .filter(e => MAGNETIC_TM.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);

  const ALL_TM = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"];
  const cationCount = elements
    .filter(e => ALL_TM.includes(e) || (getElementData(e)?.paulingElectronegativity ?? 3) < 2.0)
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const magneticSiteFraction = cationCount > 0 ? magneticCount / cationCount : 0;

  if (magneticCount >= 1 && magneticSiteFraction <= 0.25) {
    score += 0.25;
    reasons.push(`Magnetic cation-site fraction = ${(magneticSiteFraction * 100).toFixed(0)}% <= 25% (near QCP, spin fluctuations active)`);
  } else if (magneticCount >= 1 && magneticSiteFraction <= 0.50) {
    score += 0.10;
    reasons.push(`Magnetic cation-site fraction = ${(magneticSiteFraction * 100).toFixed(0)}% in 25-50% (moderate magnetism, reduced bonus)`);
  } else if (magneticCount >= 1) {
    score -= 0.10;
    reasons.push(`Magnetic cation-site fraction = ${(magneticSiteFraction * 100).toFixed(0)}% > 50% (possible static magnetism risk, penalized)`);
  } else {
    pass = false;
    reasons.push(`No magnetic transition metals (Fe/Ni/Co/Cu) detected`);
  }

  if (features.electronPhononLambda >= 0.3) {
    score += 0.25;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} >= 0.3 (phonon channel active)`);
  } else {
    pass = false;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} < 0.3 (phonon channel too weak)`);
  }

  const hasMagneticProximity = magneticCount >= 1 && features.dosAtEF > 1.0;
  if (hasMagneticProximity) {
    score += 0.25;
    reasons.push(`Magnetic fluctuation proximity: DOS(EF)=${features.dosAtEF.toFixed(2)} > 1.0 with magnetic TM (spin channel active)`);
  } else {
    pass = false;
    reasons.push(`No magnetic fluctuation proximity: DOS(EF)=${features.dosAtEF.toFixed(2)}, magnetic TM count=${magneticCount}`);
  }

  const is2DLike = features.dimensionalityScore >= 0.6 || features.layeredStructure;
  if (is2DLike) {
    score += 0.15;
    reasons.push(`Layered/2D character detected (dimensionality=${features.dimensionalityScore.toFixed(2)}, layered=${features.layeredStructure})`);
  } else {
    reasons.push(`Limited 2D character (dimensionality=${features.dimensionalityScore.toFixed(2)})`);
  }

  if (features.metallicity < 0.1 && features.mottProximityScore > 0.5) {
    score -= 0.05;
    reasons.push(`Near-Mott insulator (metallicity=${features.metallicity.toFixed(2)}, Mott proximity=${features.mottProximityScore.toFixed(2)}): may superconduct upon doping`);
  } else if (features.metallicity < 0.2) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for mixed-mechanism superconductor (heavy score penalty)`);
  } else if (features.metallicity < 0.3) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for mixed-mechanism superconductor (score penalized)`);
  } else {
    score += 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} adequate`);
  }

  score = Math.max(0, Math.min(1.0, score));
  return { pass, score, reasons };
}

function applyLayeredChalcogenideFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const CHALCOGENS = ["Se", "S", "Te"];
  const LAYER_METALS = ["Nb", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "V", "Re"];

  const chalcogenCount = elements
    .filter(e => CHALCOGENS.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const metalCount = elements
    .filter(e => LAYER_METALS.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);

  if (chalcogenCount >= 2 && metalCount >= 1) {
    const ratio = chalcogenCount / metalCount;
    score += 0.25;
    reasons.push(`MX2-type stoichiometry: chalcogen/metal ratio = ${ratio.toFixed(1)} (layered structure likely)`);
  } else {
    pass = false;
    reasons.push(`Insufficient MX2 stoichiometry: chalcogen=${chalcogenCount}, metal=${metalCount}`);
  }

  const is2DLike = features.dimensionalityScore >= 0.6 || features.layeredStructure;
  const isPerfectly2D = features.dimensionalityScore >= 0.95 && features.layeredStructure;
  if (is2DLike) {
    score += 0.25;
    reasons.push(`2D/layered character detected (dimensionality=${features.dimensionalityScore.toFixed(2)}, layered=${features.layeredStructure})`);
    if (isPerfectly2D) {
      score += 0.05;
      reasons.push(`Perfectly layered (2H-polytype regime): strong CDW/SC competition expected — Tc may be suppressed by CDW ordering`);
    }
  } else {
    reasons.push(`Limited 2D character (dimensionality=${features.dimensionalityScore.toFixed(2)})`);
  }

  if (features.dosAtEF > 1.0) {
    score += 0.20;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} > 1.0 (adequate states at Fermi level for CDW/SC competition)`);
  } else {
    pass = false;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} <= 1.0 (insufficient density of states)`);
  }

  if (features.nestingScore > 0.7 && isPerfectly2D) {
    reasons.push(`High nesting (${features.nestingScore.toFixed(2)}) in perfectly layered system: CDW instability likely competes with SC — observed Tc may be lower than predicted`);
  }

  if (features.electronPhononLambda >= 0.4) {
    score += 0.20;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} >= 0.4 (adequate electron-phonon coupling for layered chalcogenide)`);
  } else {
    pass = false;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} < 0.4 (weak electron-phonon coupling)`);
  }

  if (features.metallicity < 0.1 && features.mottProximityScore > 0.5) {
    score -= 0.05;
    reasons.push(`Near-Mott insulator (metallicity=${features.metallicity.toFixed(2)}, Mott proximity=${features.mottProximityScore.toFixed(2)}): may superconduct upon doping`);
  } else if (features.metallicity < 0.2) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for metallic chalcogenide (heavy score penalty)`);
  } else if (features.metallicity < 0.3) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for metallic chalcogenide (score penalized)`);
  } else {
    score += 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} adequate`);
  }

  score = Math.max(0, Math.min(1.0, score));
  return { pass, score, reasons };
}

function applyLayeredPnictideFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const PNICTIDE_TM = ["Fe", "Co", "Ni", "Mn", "Ru"];
  const PNICTOGENS = ["As", "P", "Sb"];
  const SPACERS = ["La", "Ce", "Pr", "Nd", "Sm", "Gd", "Ba", "Sr", "Ca", "Y"];

  const tmCount = elements
    .filter(e => PNICTIDE_TM.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const pnCount = elements
    .filter(e => PNICTOGENS.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const spacerCount = elements
    .filter(e => SPACERS.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);

  if (tmCount >= 1 && pnCount >= 1) {
    score += 0.25;
    reasons.push(`Iron-pnictide structure: TM=${tmCount}, pnictogen=${pnCount} (FeAs/FeP-type layers present)`);
  } else {
    pass = false;
    reasons.push(`Missing iron-pnictide structure: TM=${tmCount}, pnictogen=${pnCount}`);
  }

  if (spacerCount >= 1) {
    const activeLayerCount = tmCount + pnCount;
    const layerRatio = activeLayerCount / spacerCount;
    if (layerRatio >= 1.0 && layerRatio <= 4.0) {
      score += 0.15;
      reasons.push(`Spacer layer present with valid ratio (TM+Pn)/Spacer = ${layerRatio.toFixed(1)} in [1.0, 4.0] (1111/122-type)`);
    } else if (layerRatio > 4.0) {
      score += 0.05;
      reasons.push(`Spacer present but (TM+Pn)/Spacer = ${layerRatio.toFixed(1)} > 4.0 (spacer-deficient, reduced bonus)`);
    } else {
      score -= 0.10;
      reasons.push(`Spacer-dominated: (TM+Pn)/Spacer = ${layerRatio.toFixed(1)} < 1.0 (likely spacer with TM impurities, not layered pnictide)`);
    }
  } else {
    reasons.push(`No spacer layer detected`);
  }

  if (features.dosAtEF > 1.5) {
    score += 0.25;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} > 1.5 (high density of states for nesting-driven SC)`);
  } else if (features.dosAtEF > 0.8) {
    score += 0.10;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} > 0.8 (moderate density of states)`);
  } else {
    pass = false;
    reasons.push(`DOS(EF) = ${features.dosAtEF.toFixed(2)} <= 0.8 (insufficient DOS for pnictide SC)`);
  }

  if (features.fermiSurfaceNestingScore > 0.7) {
    score += 0.15;
    reasons.push(`FS nesting score = ${features.fermiSurfaceNestingScore.toFixed(2)} > 0.7 (strong (pi,pi) nesting — definitive pnictide pairing signature)`);
  } else if (features.fermiSurfaceNestingScore > 0.4) {
    score += 0.05;
    reasons.push(`FS nesting score = ${features.fermiSurfaceNestingScore.toFixed(2)} > 0.4 (moderate nesting)`);
  }

  const is2DLike = features.dimensionalityScore >= 0.6 || features.layeredStructure;
  if (is2DLike) {
    score += 0.20;
    reasons.push(`Layered/2D character detected (dimensionality=${features.dimensionalityScore.toFixed(2)}, layered=${features.layeredStructure})`);
  } else {
    pass = false;
    reasons.push(`No layered character detected (dimensionality=${features.dimensionalityScore.toFixed(2)})`);
  }

  if (features.metallicity < 0.1 && features.mottProximityScore > 0.5) {
    score -= 0.05;
    reasons.push(`Near-Mott insulator (metallicity=${features.metallicity.toFixed(2)}, Mott proximity=${features.mottProximityScore.toFixed(2)}): parent may superconduct upon doping`);
  } else if (features.metallicity < 0.2) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for pnictide superconductor (heavy score penalty)`);
  } else if (features.metallicity < 0.3) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for pnictide superconductor (score penalized)`);
  } else {
    score += 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} adequate`);
  }

  score = Math.max(0, Math.min(1.0, score));
  return { pass, score, reasons };
}

function applyIntercalatedLayeredFilter(formula: string, features: MLFeatureVector): FamilyFilterResult {
  const counts = parseFormulaCounts(formula);
  const elements = parseFormulaElements(formula);
  const reasons: string[] = [];
  let score = 0;
  let pass = true;

  const INTERCALANTS = ["Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba", "Mg", "Eu", "Yb"];
  const ANION_ELEMENTS = ["O", "S", "Se", "Te", "N", "C", "F", "Cl", "Br", "I"];

  const intercalantCount = elements
    .filter(e => INTERCALANTS.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hostElements = elements.filter(e => !INTERCALANTS.includes(e) && !ANION_ELEMENTS.includes(e));
  const hostMetalCount = hostElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const anionCount = elements
    .filter(e => ANION_ELEMENTS.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);

  if (intercalantCount > 0) {
    score += 0.25;
    reasons.push(`Intercalant present (count=${intercalantCount}): electron doping into host layers`);
  } else {
    pass = false;
    reasons.push(`No intercalant species detected`);
  }

  const hasHostLattice = anionCount >= 1 || (hostMetalCount >= 1 && (features.layeredStructure || features.dimensionalityScore >= 0.5));
  if (hasHostLattice) {
    if (anionCount >= 1 && hostMetalCount >= 1) {
      score += 0.20;
      reasons.push(`Host lattice: ${hostElements.join("/")} with anions (count=${anionCount}), metals (count=${hostMetalCount})`);
    } else if (anionCount >= 1) {
      score += 0.15;
      reasons.push(`Host anions present (count=${anionCount}), host metals not in standard list but layered character may indicate valid host`);
    } else {
      score += 0.10;
      reasons.push(`Host metals ${hostElements.join("/")} detected with layered character (dimensionality=${features.dimensionalityScore.toFixed(2)})`);
    }
  } else {
    pass = false;
    reasons.push(`No host lattice detected (anions=${anionCount}, host metals=${hostMetalCount}, layered=${features.layeredStructure})`);
  }

  const is2DFermiSurface = features.dimensionalityScore >= 0.6 ||
    features.fermiSurfaceType.includes("2D") ||
    features.layeredStructure;
  if (is2DFermiSurface) {
    score += 0.25;
    reasons.push(`2D Fermi surface / layered character detected (dimensionality=${features.dimensionalityScore.toFixed(2)})`);
  } else {
    pass = false;
    reasons.push(`No 2D Fermi surface detected (dimensionality=${features.dimensionalityScore.toFixed(2)})`);
  }

  const lambdaThreshold = (is2DFermiSurface && features.dosAtEF > 2.0) ? 0.15 :
    (is2DFermiSurface && features.dosAtEF > 1.5) ? 0.20 : 0.30;

  if (features.electronPhononLambda >= 0.3) {
    score += 0.15;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} >= 0.3 (adequate e-ph coupling for intercalated system)`);
  } else if (features.electronPhononLambda >= lambdaThreshold) {
    score += 0.10;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} below 0.3 but >= ${lambdaThreshold.toFixed(2)} (2D electronic enhancement with DOS(EF)=${features.dosAtEF.toFixed(2)} may compensate weak phonon coupling)`);
  } else {
    pass = false;
    reasons.push(`Lambda = ${features.electronPhononLambda.toFixed(2)} < ${lambdaThreshold.toFixed(2)} (insufficient coupling even with electronic enhancement)`);
  }

  if (features.metallicity < 0.1 && features.mottProximityScore > 0.5) {
    score -= 0.05;
    reasons.push(`Near-Mott insulator (metallicity=${features.metallicity.toFixed(2)}, Mott proximity=${features.mottProximityScore.toFixed(2)}): may superconduct upon doping/intercalation`);
  } else if (features.metallicity < 0.2) {
    score -= 0.20;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} very low for intercalated superconductor (heavy score penalty)`);
  } else if (features.metallicity < 0.3) {
    score -= 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} marginal for intercalated superconductor (score penalized)`);
  } else {
    score += 0.10;
    reasons.push(`Metallicity ${features.metallicity.toFixed(2)} adequate`);
  }

  score = Math.max(0, Math.min(1.0, score));
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
    case "Kagome":
      return applyKagomeFilter(formula, features);
    case "Mixed-mechanism":
      return applyMixedMechanismFilter(formula, features);
    case "Layered-chalcogenide":
      return applyLayeredChalcogenideFilter(formula, features);
    case "Layered-pnictide":
      return applyLayeredPnictideFilter(formula, features);
    case "Intercalated-layered":
      return applyIntercalatedLayeredFilter(formula, features);
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
  topologicalScore?: number | null;
  uncertaintyEstimate?: number | null;
}

const ALL_ELEMENTS_LIST = [
  "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
  "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "In", "Sn", "Sb", "Te",
  "I", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Yb", "Hf", "Ta", "W", "Re", "Os",
  "Ir", "Pt", "Au", "Tl", "Pb", "Bi",
];
const ELEMENT_TO_BIT: Record<string, number> = {};
ALL_ELEMENTS_LIST.forEach((el, i) => { ELEMENT_TO_BIT[el] = i; });

function elementSetToBitmask(elements: string[]): [number, number, number] {
  let lo = 0, mid = 0, hi = 0;
  for (const el of elements) {
    const bit = ELEMENT_TO_BIT[el];
    if (bit === undefined) continue;
    if (bit < 26) lo |= (1 << bit);
    else if (bit < 52) mid |= (1 << (bit - 26));
    else hi |= (1 << (bit - 52));
  }
  return [lo, mid, hi];
}

function bitmaskPopcount(n: number): number {
  n = n - ((n >> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  return (((n + (n >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

function bitmaskJaccard(a: [number, number, number], b: [number, number, number]): number {
  const intersect = bitmaskPopcount(a[0] & b[0]) + bitmaskPopcount(a[1] & b[1]) + bitmaskPopcount(a[2] & b[2]);
  const union = bitmaskPopcount(a[0] | b[0]) + bitmaskPopcount(a[1] | b[1]) + bitmaskPopcount(a[2] | b[2]);
  return union > 0 ? intersect / union : 0;
}

function bitmaskDiffElements(a: [number, number, number], b: [number, number, number]): string[] {
  const diff: string[] = [];
  const d0 = a[0] & ~b[0], d1 = a[1] & ~b[1], d2 = a[2] & ~b[2];
  for (let i = 0; i < 26; i++) if (d0 & (1 << i)) diff.push(ALL_ELEMENTS_LIST[i]);
  for (let i = 0; i < 26; i++) if (d1 & (1 << i)) diff.push(ALL_ELEMENTS_LIST[i + 26]);
  for (let i = 0; i < 24; i++) if (d2 & (1 << i)) diff.push(ALL_ELEMENTS_LIST[i + 52]);
  return diff;
}

let existingFormulasBitmaskCache: Map<string, [number, number, number]> | null = null;

function getExistingBitmask(formula: string): [number, number, number] {
  if (!existingFormulasBitmaskCache) existingFormulasBitmaskCache = new Map();
  let cached = existingFormulasBitmaskCache.get(formula);
  if (!cached) {
    cached = elementSetToBitmask(parseFormulaElements(formula));
    existingFormulasBitmaskCache.set(formula, cached);
    if (existingFormulasBitmaskCache.size > 5000) {
      const firstKey = existingFormulasBitmaskCache.keys().next().value;
      if (firstKey) existingFormulasBitmaskCache.delete(firstKey);
    }
  }
  return cached;
}

export function computeDiscoveryScore(candidate: DiscoveryScoreInput): {
  discoveryScore: number;
  normalizedTc: number;
  noveltyScore: number;
  stabilityScore: number;
  synthesisFeasibility: number;
  topologyContribution: number;
  uncertaintyBonus: number;
} {
  const tc = candidate.predictedTc || 0;
  const normalizedTc = tc <= 300
    ? Math.min(1.0, Math.max(0, tc / 300)) * 0.8
    : Math.min(1.0, 0.8 + 0.2 * Math.min(1.0, (tc - 300) / 200));

  const hullDistKnown = candidate.hullDistance != null;
  const hullDist = candidate.hullDistance ?? 0.10;
  const stabilityScore = Math.min(1.0, Math.max(0, hullDistKnown ? (1.0 - hullDist / 0.15) : (0.3 - hullDist / 0.5)));

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

  let commonFraction = 0;
  for (const el of elements) {
    const availability = COMMON_ELEMENTS[el] ?? 0.15;
    commonFraction += availability * ((counts[el] || 1) / totalAtoms);
  }
  if (commonFraction > 0.85) {
    noveltyScore -= 0.10;
  } else if (commonFraction > 0.70) {
    noveltyScore -= 0.05;
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
    const candidateBits = elementSetToBitmask(elements);
    let maxWeightedDistance = 0;

    for (const existing of candidate.existingFormulas) {
      const existingBits = getExistingBitmask(existing);
      const jaccard = bitmaskJaccard(candidateBits, existingBits);
      const baseDistance = 1.0 - jaccard;

      const uniqueToCandidate = bitmaskDiffElements(candidateBits, existingBits);
      const uniqueToExisting = bitmaskDiffElements(existingBits, candidateBits);

      let chemicalDistance = 0;
      if (uniqueToCandidate.length > 0 && uniqueToExisting.length > 0) {
        let totalDiff = 0;
        let pairCount = 0;
        for (const nc of uniqueToCandidate) {
          const dC = getElementData(nc);
          for (const ne of uniqueToExisting) {
            const dE = getElementData(ne);
            if (dC && dE) {
              const enDiff = Math.abs((dC.paulingElectronegativity ?? 2) - (dE.paulingElectronegativity ?? 2));
              const rC = (dC.atomicRadius ?? 150) / 100;
              const rE = (dE.atomicRadius ?? 150) / 100;
              const rDiff = Math.abs(rC - rE) / Math.max(rC, rE);
              totalDiff += enDiff * 0.6 + rDiff * 0.4;
              pairCount++;
            }
          }
        }
        chemicalDistance = pairCount > 0 ? Math.min(1.0, totalDiff / pairCount) : 0;
      }

      const weightedDist = baseDistance * Math.min(1.5, 0.5 + chemicalDistance);
      if (weightedDist > maxWeightedDistance) {
        maxWeightedDistance = weightedDist;
      }
    }
    noveltyScore += maxWeightedDistance * 0.15;
  }

  noveltyScore = Math.min(1.0, Math.max(0, noveltyScore));

  const topologyContribution = Math.min(1.0, Math.max(0, candidate.topologicalScore ?? 0));

  const rawUncertainty = candidate.uncertaintyEstimate ?? 0.5;
  const uncertaintyBonus = Math.min(1.0, Math.max(0, rawUncertainty * 1.2));

  const discoveryScore =
    0.55 * normalizedTc +
    0.15 * stabilityScore +
    0.10 * noveltyScore +
    0.10 * synthesisFeasibility +
    0.05 * topologyContribution +
    0.05 * uncertaintyBonus;

  return {
    discoveryScore: Math.round(discoveryScore * 1000) / 1000,
    normalizedTc,
    noveltyScore: Math.round(noveltyScore * 1000) / 1000,
    stabilityScore: Math.round(stabilityScore * 1000) / 1000,
    synthesisFeasibility: Math.round(synthesisFeasibility * 1000) / 1000,
    topologyContribution: Math.round(topologyContribution * 1000) / 1000,
    uncertaintyBonus: Math.round(uncertaintyBonus * 1000) / 1000,
  };
}

const COMMON_ELEMENTS: Record<string, number> = {
  Fe: 1.0, Al: 1.0, Si: 1.0, O: 1.0, C: 1.0, N: 0.95, Ti: 0.95,
  Cu: 0.9, Ni: 0.9, Zn: 0.85, Mg: 0.85, Ca: 0.85, Na: 0.85, K: 0.85,
  Mn: 0.8, Cr: 0.8, Co: 0.75, V: 0.75, B: 0.75, S: 0.8, P: 0.8,
  Mo: 0.7, W: 0.7, Nb: 0.65, Sn: 0.7, Zr: 0.65, Ba: 0.65, Sr: 0.6,
  Y: 0.55, La: 0.55, Ce: 0.55, Bi: 0.55, Se: 0.6, Te: 0.5,
  Ga: 0.5, Ge: 0.5, Sb: 0.5, In: 0.45, Hf: 0.4, Ta: 0.4,
  Pd: 0.35, Pt: 0.3, Ru: 0.3, Rh: 0.25, Os: 0.2, Ir: 0.2, Re: 0.25,
  Sc: 0.35, Tl: 0.3, H: 0.9, Li: 0.6, Rb: 0.35, Cs: 0.3,
  Nd: 0.45, Sm: 0.4, Gd: 0.4, Pr: 0.4, Eu: 0.35, Yb: 0.35,
  As: 0.5, Pb: 0.6,
};

const FAMILY_SYNTHESIS_DEFAULTS: Record<string, {
  baseScore: number;
  typicalTempC: number;
  pressureGPa: number;
  atmosphereComplexity: number;
}> = {
  "MAX-phase": { baseScore: 0.70, typicalTempC: 1400, pressureGPa: 0, atmosphereComplexity: 0.1 },
  "Boride": { baseScore: 0.55, typicalTempC: 1800, pressureGPa: 0, atmosphereComplexity: 0.15 },
  "Hydride": { baseScore: 0.25, typicalTempC: 1500, pressureGPa: 150, atmosphereComplexity: 0.4 },
  "Nitride": { baseScore: 0.55, typicalTempC: 1200, pressureGPa: 0, atmosphereComplexity: 0.2 },
  "Intercalated-nitride": { baseScore: 0.50, typicalTempC: 1000, pressureGPa: 0, atmosphereComplexity: 0.25 },
  "Kagome": { baseScore: 0.60, typicalTempC: 1100, pressureGPa: 0, atmosphereComplexity: 0.1 },
  "Layered-chalcogenide": { baseScore: 0.65, typicalTempC: 900, pressureGPa: 0, atmosphereComplexity: 0.15 },
  "Layered-pnictide": { baseScore: 0.55, typicalTempC: 1100, pressureGPa: 0, atmosphereComplexity: 0.2 },
  "Intercalated-layered": { baseScore: 0.50, typicalTempC: 800, pressureGPa: 0, atmosphereComplexity: 0.2 },
  "Mixed-mechanism": { baseScore: 0.55, typicalTempC: 1200, pressureGPa: 0, atmosphereComplexity: 0.15 },
};

export interface SynthesisScoreBreakdown {
  total: number;
  precursorAvailability: number;
  temperatureFactor: number;
  phaseCompetitionPenalty: number;
  familyBase: number;
  pressurePenalty: number;
}

export function computeSynthesisScore(
  formula: string,
  family: string,
  features: MLFeatureVector,
  hullDistance?: number | null,
  competingPhaseCount?: number | null
): SynthesisScoreBreakdown {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const defaults = FAMILY_SYNTHESIS_DEFAULTS[family] ?? {
    baseScore: 0.5, typicalTempC: 1200, pressureGPa: 0, atmosphereComplexity: 0.15
  };

  let precursorAvailability = 0;
  for (const el of elements) {
    const availability = COMMON_ELEMENTS[el] ?? 0.15;
    const fraction = (counts[el] || 1) / totalAtoms;
    precursorAvailability += availability * fraction;
  }
  precursorAvailability = Math.min(1.0, Math.max(0, precursorAvailability));

  const meltingPoints: number[] = [];
  for (const el of elements) {
    const data = getElementData(el);
    if (data?.meltingPoint) meltingPoints.push(data.meltingPoint);
  }
  const avgMeltingPoint = meltingPoints.length > 0
    ? meltingPoints.reduce((a, b) => a + b, 0) / meltingPoints.length
    : 1500;

  const requiredTempK = defaults.typicalTempC + 273;
  let temperatureFactor = 1.0;
  if (requiredTempK > avgMeltingPoint * 0.9) {
    temperatureFactor = 0.4;
  } else if (requiredTempK > avgMeltingPoint * 0.7) {
    temperatureFactor = 0.7;
  } else if (requiredTempK > 2000) {
    temperatureFactor = 0.6;
  } else if (requiredTempK > 1500) {
    temperatureFactor = 0.8;
  }

  let phaseCompetitionPenalty = 0;
  const hull = hullDistance ?? 0.05;
  if (hull > 0.2) {
    phaseCompetitionPenalty = 0.3;
  } else if (hull > 0.1) {
    phaseCompetitionPenalty = 0.2;
  } else if (hull > 0.05) {
    phaseCompetitionPenalty = 0.1;
  } else if (hull > 0.01) {
    phaseCompetitionPenalty = 0.05;
  }

  const phases = competingPhaseCount ?? 0;
  if (phases > 5) {
    phaseCompetitionPenalty += 0.15;
  } else if (phases > 3) {
    phaseCompetitionPenalty += 0.1;
  } else if (phases > 1) {
    phaseCompetitionPenalty += 0.05;
  }
  phaseCompetitionPenalty = Math.min(0.5, phaseCompetitionPenalty);

  let pressurePenalty = 0;
  if (defaults.pressureGPa > 100) {
    pressurePenalty = 0.35;
  } else if (defaults.pressureGPa > 50) {
    pressurePenalty = 0.2;
  } else if (defaults.pressureGPa > 10) {
    pressurePenalty = 0.1;
  }

  if (family === "Hydride") {
    const hCount = counts["H"] || 0;
    const metalAtoms = elements.filter(e => e !== "H").reduce((s, e) => s + (counts[e] || 0), 0);
    const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;
    if (hRatio >= 10) {
      pressurePenalty = Math.min(0.5, pressurePenalty + 0.15);
    } else if (hRatio >= 6) {
      pressurePenalty = Math.min(0.45, pressurePenalty + 0.1);
    }
  }

  const total = Math.min(1.0, Math.max(0,
    defaults.baseScore * 0.3 +
    precursorAvailability * 0.25 +
    temperatureFactor * 0.2 -
    phaseCompetitionPenalty -
    pressurePenalty +
    (1 - defaults.atmosphereComplexity) * 0.1
  ));

  return {
    total: Math.round(total * 1000) / 1000,
    precursorAvailability: Math.round(precursorAvailability * 1000) / 1000,
    temperatureFactor: Math.round(temperatureFactor * 1000) / 1000,
    phaseCompetitionPenalty: Math.round(phaseCompetitionPenalty * 1000) / 1000,
    familyBase: Math.round(defaults.baseScore * 1000) / 1000,
    pressurePenalty: Math.round(pressurePenalty * 1000) / 1000,
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

  const synthesisBreakdown = computeSynthesisScore(formula, family, features);
  const synthesisFeasibility = synthesisBreakdown.total;

  const composite = 0.35 * tcNormalized +
    0.20 * stability +
    0.15 * lambdaNormalized +
    0.10 * synthesisFeasibility +
    0.10 * Math.min(1.0, (features as any)?.topology?.topologicalScore ?? 0) +
    0.10 * Math.min(1.0, Math.max(0, ((features as any)?.uncertaintyEstimate ?? 0.5) * 1.2));

  return Math.round(composite * 1000) / 1000;
}
