import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  computeDynamicSpinSusceptibility,
  evaluateCompetingPhases,
  parseFormulaElements,
  estimateBandwidthW,
  type ElectronicStructure,
  type PhononSpectrum,
  type ElectronPhononCoupling,
} from "../learning/physics-engine";
import {
  getElementData,
  getHubbardU,
  getStonerParameter,
  isTransitionMetal,
  isRareEarth,
} from "../learning/elemental-data";

export type QCPType =
  | "Mott"
  | "SDW"
  | "CDW"
  | "nematic"
  | "structural"
  | "orbital-selective"
  | "none";

export interface QCPChannelScores {
  mott: number;
  sdw: number;
  cdw: number;
  nematic: number;
  structural: number;
  orbitalSelective: number;
}

export interface DomeProfile {
  controlParameter: number;
  distanceFromQCP: number;
  domeAmplitude: number;
  isOnSCside: boolean;
  tcEnhancementFactor: number;
}

export interface QuantumCriticalAnalysis {
  formula: string;
  qcpType: QCPType;
  secondaryQCPType: QCPType;
  quantumCriticalScore: number;
  channelScores: QCPChannelScores;
  dome: DomeProfile;
  stonerEnhancement: number;
  mottProximity: number;
  nestingScore: number;
  vanHoveProximity: number;
  cdwOrderParameter: number;
  sdwOrderParameter: number;
  correlationStrength: number;
  fluctuationSpectrum: {
    spinFluctuations: number;
    chargeFluctuations: number;
    orbitalFluctuations: number;
    latticeFluctuations: number;
  };
  pairingBoostFromQCP: number;
  summary: string;
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    counts[m[1]] = (counts[m[1]] || 0) + (m[2] ? parseFloat(m[2]) : 1);
  }
  return counts;
}

function computeMottChannel(
  elements: string[],
  counts: Record<string, number>,
  electronic: ElectronicStructure,
): number {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  let maxUoverW = 0;

  for (const el of elements) {
    const U = getHubbardU(el);
    if (U === null) continue;
    const W = Math.max(0.5, estimateBandwidthW(el));
    const frac = (counts[el] || 1) / totalAtoms;
    const ratio = (U / W) * Math.sqrt(frac);
    if (ratio > maxUoverW) maxUoverW = ratio;
  }

  const mottProx = electronic.mottProximityScore ?? 0;
  const correlation = electronic.correlationStrength ?? 0;

  let mottScore = 0;
  if (maxUoverW > 0.3) {
    const distFromTransition = Math.abs(maxUoverW - 1.0);
    mottScore = Math.exp(-distFromTransition * distFromTransition * 4.0);
  }

  mottScore = Math.max(mottScore, mottProx * 0.8);

  if (correlation > 0.5) {
    mottScore = Math.min(1.0, mottScore + (correlation - 0.5) * 0.3);
  }

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3 &&
    elements.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Bi", "Tl", "Hg"].includes(e));
  const isNickelate = elements.includes("Ni") && elements.includes("O") && elements.length >= 3;

  if (isCuprate) mottScore = Math.max(mottScore, 0.90);
  if (isNickelate) mottScore = Math.max(mottScore, 0.75);

  return Math.min(1.0, mottScore);
}

function computeSDWChannel(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  electronic: ElectronicStructure,
): number {
  const spin = computeDynamicSpinSusceptibility(formula, electronic);
  const nestingScore = electronic.nestingScore ?? 0;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  let maxStoner = 0;
  for (const el of elements) {
    const I = getStonerParameter(el);
    if (I === null) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    maxStoner = Math.max(maxStoner, I * frac);
  }

  let sdwScore = 0;

  const stonerProduct = spin.stonerEnhancement;
  if (stonerProduct > 1.5) {
    const distFromCritical = Math.abs(stonerProduct - 5.0) / 5.0;
    sdwScore = Math.exp(-distFromCritical * distFromCritical * 2.0) * 0.6;
  }

  if (nestingScore > 0.4) {
    sdwScore += nestingScore * 0.3;
  }

  if (spin.isNearQCP) {
    sdwScore = Math.max(sdwScore, 0.7);
  }

  const isPnictide = elements.some(e => ["Fe", "Co"].includes(e)) &&
    elements.some(e => ["As", "P", "Se", "S"].includes(e));
  const hasCr = elements.includes("Cr");
  const hasMn = elements.includes("Mn");

  if (isPnictide) sdwScore = Math.max(sdwScore, 0.85);
  if (hasCr && nestingScore > 0.3) sdwScore = Math.max(sdwScore, 0.60);
  if (hasMn && nestingScore > 0.3) sdwScore = Math.max(sdwScore, 0.50);

  return Math.min(1.0, sdwScore);
}

function computeCDWChannel(
  elements: string[],
  electronic: ElectronicStructure,
  phonon: PhononSpectrum,
): number {
  const nestingScore = electronic.nestingScore ?? 0;
  const vanHoveProx = electronic.vanHoveProximity ?? 0;
  const dosEF = electronic.densityOfStatesAtFermi;

  let cdwScore = 0;

  const nestingDriven = Math.min(1.0, nestingScore * dosEF * 0.3);
  cdwScore += nestingDriven * 0.4;

  if (vanHoveProx > 0.3) {
    cdwScore += (vanHoveProx - 0.3) * 0.4;
  }

  if (phonon.softModePresent) {
    cdwScore += phonon.softModeScore * 0.3;
  }

  const isNbSe2 = elements.includes("Nb") && elements.includes("Se");
  const isTaSe2 = elements.includes("Ta") && elements.includes("Se");
  const isTMD = elements.some(e => ["Nb", "Ta", "Ti", "Mo", "W"].includes(e)) &&
    elements.some(e => ["S", "Se", "Te"].includes(e));

  if (isNbSe2) cdwScore = Math.max(cdwScore, 0.85);
  if (isTaSe2) cdwScore = Math.max(cdwScore, 0.80);
  if (isTMD && nestingScore > 0.4) cdwScore = Math.max(cdwScore, 0.65);

  return Math.min(1.0, cdwScore);
}

function computeNematicChannel(
  elements: string[],
  electronic: ElectronicStructure,
): number {
  const dFrac = electronic.orbitalFractions?.d ?? 0;
  const correlation = electronic.correlationStrength ?? 0;
  const nestingScore = electronic.nestingScore ?? 0;

  let nematicScore = 0;

  if (dFrac > 0.3 && correlation > 0.3) {
    nematicScore = (dFrac - 0.3) * correlation * 1.5;
  }

  if (electronic.bandFlatness > 0.4 && nestingScore > 0.3) {
    nematicScore += electronic.bandFlatness * 0.2;
  }

  const isPnictide = elements.some(e => ["Fe", "Co"].includes(e)) &&
    elements.some(e => ["As", "P", "Se", "S"].includes(e));
  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;

  if (isPnictide) nematicScore = Math.max(nematicScore, 0.70);
  if (isCuprate) nematicScore = Math.max(nematicScore, 0.50);

  return Math.min(1.0, nematicScore);
}

function computeStructuralChannel(
  elements: string[],
  counts: Record<string, number>,
  phonon: PhononSpectrum,
  coupling: ElectronPhononCoupling,
): number {
  let structScore = 0;

  if (phonon.softModePresent) {
    structScore += phonon.softModeScore * 0.5;
  }

  if (phonon.anharmonicityIndex > 0.3) {
    structScore += phonon.anharmonicityIndex * 0.3;
  }

  if (coupling.lambda > 2.0) {
    structScore += Math.min(0.3, (coupling.lambda - 2.0) * 0.15);
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const hCount = counts["H"] || 0;
  const hRatio = hCount / Math.max(1, totalAtoms - hCount);
  if (hRatio >= 6) {
    structScore = Math.max(structScore, 0.50 + hRatio * 0.03);
  }

  const hasSTO = elements.includes("Sr") && elements.includes("Ti") && elements.includes("O");
  if (hasSTO) structScore = Math.max(structScore, 0.60);

  return Math.min(1.0, structScore);
}

function computeOrbitalSelectiveChannel(
  elements: string[],
  counts: Record<string, number>,
  electronic: ElectronicStructure,
): number {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const dFrac = electronic.orbitalFractions?.d ?? 0;
  const fFrac = electronic.orbitalFractions?.f ?? 0;
  const correlation = electronic.correlationStrength ?? 0;

  let osmpScore = 0;

  let tmCount = 0;
  let maxUoverW = 0;
  let minUoverW = Infinity;
  for (const el of elements) {
    if (!isTransitionMetal(el)) continue;
    tmCount++;
    const U = getHubbardU(el);
    if (U === null) continue;
    const W = Math.max(0.5, estimateBandwidthW(el));
    const frac = (counts[el] || 1) / totalAtoms;
    const ratio = (U / W) * Math.sqrt(frac);
    if (ratio > maxUoverW) maxUoverW = ratio;
    if (ratio < minUoverW) minUoverW = ratio;
  }

  if (tmCount >= 1 && maxUoverW > 0.3 && minUoverW < Infinity) {
    const spread = maxUoverW - minUoverW;
    if (spread > 0.2) {
      osmpScore = Math.min(1.0, spread * 1.5);
    }
  }

  if (dFrac > 0.3 && correlation > 0.4) {
    osmpScore = Math.max(osmpScore, dFrac * correlation * 0.8);
  }

  const isPnictide = elements.some(e => ["Fe"].includes(e)) &&
    elements.some(e => ["Se", "S", "Te"].includes(e)) && !elements.includes("As");
  const isRuthenate = elements.includes("Ru") && elements.includes("O");

  if (isPnictide) osmpScore = Math.max(osmpScore, 0.65);
  if (isRuthenate) osmpScore = Math.max(osmpScore, 0.70);

  const hasHeavyFermion = elements.some(e => isRareEarth(e)) && fFrac > 0.1;
  if (hasHeavyFermion) osmpScore = Math.max(osmpScore, 0.55);

  return Math.min(1.0, osmpScore);
}

function computeDomeProfile(
  qcpType: QCPType,
  channelScores: QCPChannelScores,
  electronic: ElectronicStructure,
): DomeProfile {
  const primaryScore = channelScores[
    qcpType === "Mott" ? "mott" :
    qcpType === "SDW" ? "sdw" :
    qcpType === "CDW" ? "cdw" :
    qcpType === "nematic" ? "nematic" :
    qcpType === "structural" ? "structural" :
    qcpType === "orbital-selective" ? "orbitalSelective" :
    "mott"
  ] as number;

  const controlParameter = primaryScore;

  const optimalPoint = 0.75;
  const distanceFromQCP = Math.abs(controlParameter - optimalPoint);

  const sigma = 0.3;
  const domeAmplitude = Math.exp(-distanceFromQCP * distanceFromQCP / (2 * sigma * sigma));

  const isOnSCside = controlParameter >= 0.3 && controlParameter <= 0.95;

  let tcEnhancementFactor = 1.0;
  if (isOnSCside) {
    tcEnhancementFactor = 1.0 + domeAmplitude * 1.5;

    if (controlParameter > 0.9) {
      tcEnhancementFactor *= 0.7;
    }
  }

  return {
    controlParameter: Number(controlParameter.toFixed(4)),
    distanceFromQCP: Number(distanceFromQCP.toFixed(4)),
    domeAmplitude: Number(domeAmplitude.toFixed(4)),
    isOnSCside,
    tcEnhancementFactor: Number(tcEnhancementFactor.toFixed(4)),
  };
}

function classifyQCP(channelScores: QCPChannelScores): { primary: QCPType; secondary: QCPType } {
  const entries: { type: QCPType; score: number }[] = [
    { type: "Mott", score: channelScores.mott },
    { type: "SDW", score: channelScores.sdw },
    { type: "CDW", score: channelScores.cdw },
    { type: "nematic", score: channelScores.nematic },
    { type: "structural", score: channelScores.structural },
    { type: "orbital-selective", score: channelScores.orbitalSelective },
  ];

  entries.sort((a, b) => b.score - a.score);

  const primary = entries[0].score > 0.2 ? entries[0].type : "none";
  const secondary = entries[1].score > 0.15 ? entries[1].type : "none";

  return { primary, secondary };
}

function computeFluctuationSpectrum(
  formula: string,
  electronic: ElectronicStructure,
  phonon: PhononSpectrum,
  channelScores: QCPChannelScores,
): { spinFluctuations: number; chargeFluctuations: number; orbitalFluctuations: number; latticeFluctuations: number } {
  const spin = computeDynamicSpinSusceptibility(formula, electronic);

  const spinFluctuations = Math.min(1.0,
    (channelScores.sdw * 0.5) +
    (spin.isNearQCP ? 0.3 : 0) +
    (spin.stonerEnhancement > 3 ? 0.2 : 0)
  );

  const chargeFluctuations = Math.min(1.0,
    (channelScores.cdw * 0.5) +
    (electronic.nestingScore ?? 0) * 0.2 +
    (electronic.vanHoveProximity ?? 0) * 0.2
  );

  const orbitalFluctuations = Math.min(1.0,
    (channelScores.orbitalSelective * 0.5) +
    (channelScores.nematic * 0.3) +
    ((electronic.orbitalFractions?.d ?? 0) > 0.4 ? 0.15 : 0)
  );

  const latticeFluctuations = Math.min(1.0,
    (channelScores.structural * 0.5) +
    (phonon.softModePresent ? phonon.softModeScore * 0.3 : 0) +
    (phonon.anharmonicityIndex * 0.2)
  );

  return {
    spinFluctuations: Number(spinFluctuations.toFixed(4)),
    chargeFluctuations: Number(chargeFluctuations.toFixed(4)),
    orbitalFluctuations: Number(orbitalFluctuations.toFixed(4)),
    latticeFluctuations: Number(latticeFluctuations.toFixed(4)),
  };
}

function generateSummary(
  qcpType: QCPType,
  secondaryType: QCPType,
  qcScore: number,
  dome: DomeProfile,
): string {
  if (qcpType === "none") {
    return "No significant quantum critical behavior detected. Material is far from any quantum phase transition.";
  }

  const intensity = qcScore > 0.7 ? "strong" : qcScore > 0.4 ? "moderate" : "weak";
  const domeStatus = dome.isOnSCside
    ? `within the SC dome (enhancement factor ${dome.tcEnhancementFactor.toFixed(2)}x)`
    : "outside the optimal SC dome region";

  let msg = `${intensity.charAt(0).toUpperCase() + intensity.slice(1)} ${qcpType} quantum criticality detected (score: ${qcScore.toFixed(2)}). `;
  msg += `Material is ${domeStatus}. `;

  if (secondaryType !== "none") {
    msg += `Secondary ${secondaryType} instability also present. `;
  }

  if (qcpType === "Mott") {
    msg += "Proximity to Mott insulator transition enhances spin-fluctuation pairing.";
  } else if (qcpType === "SDW") {
    msg += "Near spin-density-wave instability provides magnetic fluctuation glue for pairing.";
  } else if (qcpType === "CDW") {
    msg += "Charge-density-wave fluctuations near QCP can coexist with and enhance superconductivity.";
  } else if (qcpType === "nematic") {
    msg += "Nematic fluctuations break rotational symmetry and can enhance anisotropic pairing.";
  } else if (qcpType === "structural") {
    msg += "Structural instability and soft phonon modes provide strong lattice-mediated pairing.";
  } else if (qcpType === "orbital-selective") {
    msg += "Orbital-selective correlations create differentiated quasiparticles favorable for unconventional pairing.";
  }

  return msg;
}

export interface PhysicsDataInput {
  electronic?: ElectronicStructure;
  phonon?: PhononSpectrum;
  coupling?: ElectronPhononCoupling;
}

export function detectQuantumCriticality(
  formula: string,
  physicsData?: PhysicsDataInput,
): QuantumCriticalAnalysis {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);

  const electronic = physicsData?.electronic ?? computeElectronicStructure(formula, null);
  const phonon = physicsData?.phonon ?? computePhononSpectrum(formula, electronic);
  const coupling = physicsData?.coupling ?? computeElectronPhononCoupling(electronic, phonon, formula, 0);

  const mottScore = computeMottChannel(elements, counts, electronic);
  const sdwScore = computeSDWChannel(formula, elements, counts, electronic);
  const cdwScore = computeCDWChannel(elements, electronic, phonon);
  const nematicScore = computeNematicChannel(elements, electronic);
  const structuralScore = computeStructuralChannel(elements, counts, phonon, coupling);
  const orbitalSelectiveScore = computeOrbitalSelectiveChannel(elements, counts, electronic);

  const channelScores: QCPChannelScores = {
    mott: Number(mottScore.toFixed(4)),
    sdw: Number(sdwScore.toFixed(4)),
    cdw: Number(cdwScore.toFixed(4)),
    nematic: Number(nematicScore.toFixed(4)),
    structural: Number(structuralScore.toFixed(4)),
    orbitalSelective: Number(orbitalSelectiveScore.toFixed(4)),
  };

  const { primary: qcpType, secondary: secondaryQCPType } = classifyQCP(channelScores);

  const allScores = [mottScore, sdwScore, cdwScore, nematicScore, structuralScore, orbitalSelectiveScore];
  const maxScore = Math.max(...allScores);
  const secondMax = allScores.sort((a, b) => b - a)[1] ?? 0;
  const quantumCriticalScore = Number(Math.min(1.0, maxScore * 0.7 + secondMax * 0.3).toFixed(4));

  const dome = computeDomeProfile(qcpType, channelScores, electronic);

  const spin = computeDynamicSpinSusceptibility(formula, electronic);
  const stonerEnhancement = spin.stonerEnhancement;
  const mottProximity = electronic.mottProximityScore ?? 0;
  const nestingScore = electronic.nestingScore ?? 0;
  const vanHoveProximity = electronic.vanHoveProximity ?? 0;
  const correlationStrength = electronic.correlationStrength ?? 0;

  let cdwOrderParameter = 0;
  if (nestingScore > 0.3 && electronic.densityOfStatesAtFermi > 1.0) {
    cdwOrderParameter = Math.min(1.0, nestingScore * electronic.densityOfStatesAtFermi * 0.2);
  }
  const sdwOrderParameter = Math.min(1.0,
    (stonerEnhancement > 2 ? (stonerEnhancement - 1) / 10 : 0) +
    nestingScore * 0.2
  );

  const fluctuationSpectrum = computeFluctuationSpectrum(formula, electronic, phonon, channelScores);

  let pairingBoostFromQCP = 0;
  if (dome.isOnSCside && quantumCriticalScore > 0.3) {
    pairingBoostFromQCP = dome.domeAmplitude * quantumCriticalScore;

    const totalFluctuations = fluctuationSpectrum.spinFluctuations +
      fluctuationSpectrum.chargeFluctuations +
      fluctuationSpectrum.orbitalFluctuations +
      fluctuationSpectrum.latticeFluctuations;
    pairingBoostFromQCP *= (1 + totalFluctuations * 0.1);
  }
  pairingBoostFromQCP = Number(Math.min(1.0, pairingBoostFromQCP).toFixed(4));

  const summary = generateSummary(qcpType, secondaryQCPType, quantumCriticalScore, dome);

  return {
    formula,
    qcpType,
    secondaryQCPType,
    quantumCriticalScore,
    channelScores,
    dome,
    stonerEnhancement,
    mottProximity: Number(mottProximity.toFixed(4)),
    nestingScore: Number(nestingScore.toFixed(4)),
    vanHoveProximity: Number(vanHoveProximity.toFixed(4)),
    cdwOrderParameter: Number(cdwOrderParameter.toFixed(4)),
    sdwOrderParameter: Number(sdwOrderParameter.toFixed(4)),
    correlationStrength: Number(correlationStrength.toFixed(4)),
    fluctuationSpectrum,
    pairingBoostFromQCP,
    summary,
  };
}
