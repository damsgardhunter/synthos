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
import { computeFermiSurface, type LindhardSusceptibility } from "./fermi-surface-engine";

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

function expandParentheses(formula: string): string {
  let result = formula.replace(/\[/g, "(").replace(/\]/g, ")");
  const parenRegex = /\(([^()]+)\)(\d*\.?\d*)/;
  let iterations = 0;
  while (result.includes("(") && iterations < 20) {
    const prev = result;
    result = result.replace(parenRegex, (_, group: string, mult: string) => {
      const m = mult ? parseFloat(mult) : 1;
      if (isNaN(m) || m <= 0) return group;
      if (m === 1) return group;
      return group.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_x: string, el: string, num: string) => {
        const n = num ? parseFloat(num) : 1;
        const newN = (isNaN(n) || n <= 0 ? 1 : n) * m;
        return newN === 1 ? el : `${el}${newN}`;
      });
    });
    if (result === prev) break;
    iterations++;
  }
  return result.replace(/[()]/g, "");
}

function parseFormulaCounts(formula: string): Record<string, number> {
  let cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  cleaned = cleaned.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]/g, c => String("⁰¹²³⁴⁵⁶⁷⁸⁹".indexOf(c)));
  cleaned = expandParentheses(cleaned);
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    const val = m[2] ? parseFloat(m[2]) : 1;
    counts[m[1]] = (counts[m[1]] || 0) + (isNaN(val) || val <= 0 ? 1 : val);
  }
  return counts;
}

function guardedStonerEnhancement(rawStoner: number): number {
  const STONER_DENOM_FLOOR = 0.05;
  const STONER_MAX = 20;
  if (!Number.isFinite(rawStoner) || rawStoner < 0) return 1.0;
  return Math.min(STONER_MAX, rawStoner);
}

function estimateStructuralTransitionPressure(
  elements: string[],
  counts: Record<string, number>,
): number {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const hCount = counts["H"] || 0;
  const hRatio = hCount / totalAtoms;

  if (hRatio < 0.3) return -1;

  let avgBulk = 0;
  let metalFrac = 0;
  for (const el of elements) {
    if (el === "H") continue;
    const data = getElementData(el);
    const frac = (counts[el] || 1) / totalAtoms;
    avgBulk += (data?.bulkModulus ?? 50) * frac;
    metalFrac += frac;
  }
  avgBulk /= Math.max(metalFrac, 0.01);

  const baseTransitionP = avgBulk * 0.8 + hRatio * 100;

  if (hRatio > 0.8) return baseTransitionP * 1.2;
  if (hRatio > 0.6) return baseTransitionP;
  return baseTransitionP * 0.7;
}

function estimateFilling(
  elements: string[],
  counts: Record<string, number>,
  targetElement: string,
): { filling: number; integerDistance: number } {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const data = getElementData(targetElement);
  const valence = data?.valenceElectrons ?? 2;

  let chargeTransfer = 0;
  for (const el of elements) {
    if (el === targetElement) continue;
    const elData = getElementData(el);
    const elEN = elData?.paulingElectronegativity ?? 1.5;
    const targetEN = data?.paulingElectronegativity ?? 1.5;
    const enDiff = elEN - targetEN;
    const frac = (counts[el] || 1) / totalAtoms;
    chargeTransfer += enDiff * 0.3 * frac;
  }

  const effectiveElectrons = valence + chargeTransfer;

  const dOrbitals = 10;
  const fOrbitals = 14;
  const orbitalCapacity = valence > 8 ? fOrbitals : (valence > 2 || isTransitionMetal(targetElement) || isRareEarth(targetElement)) ? dOrbitals : 2;

  const filling = Math.max(0, Math.min(1, effectiveElectrons / orbitalCapacity));

  const nearestInteger = Math.round(filling * orbitalCapacity);
  const integerDistance = Math.abs(filling * orbitalCapacity - nearestInteger) / orbitalCapacity;

  return { filling, integerDistance };
}

function computeWeightedUoverW(
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number,
): number {
  let maxUoverW = 0;
  let weightedSum = 0;
  let weightTotal = 0;
  let correlatedCount = 0;

  for (const el of elements) {
    const U = getHubbardU(el);
    if (U === null || U < 2.0) continue;
    const W = Math.max(0.5, estimateBandwidthW(el));
    const frac = (counts[el] || 1) / totalAtoms;
    const ratio = (U / W) * Math.sqrt(frac);

    if (ratio > maxUoverW) maxUoverW = ratio;
    weightedSum += ratio * frac;
    weightTotal += frac;
    correlatedCount++;
  }

  if (correlatedCount <= 1) return maxUoverW;

  const weightedAvg = weightTotal > 0 ? weightedSum / weightTotal : 0;
  return 0.6 * maxUoverW + 0.4 * weightedAvg;
}

function computeMottChannel(
  elements: string[],
  counts: Record<string, number>,
  electronic: ElectronicStructure,
): number {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const maxUoverW = computeWeightedUoverW(elements, counts, totalAtoms);

  let bestIntegerDistance = 1.0;
  let bestFillingPenalty = 1.0;
  let bestDopingBoost = 0;

  for (const el of elements) {
    const U = getHubbardU(el);
    if (U === null) continue;
    const { integerDistance } = estimateFilling(elements, counts, el);

    if (integerDistance < bestIntegerDistance) {
      bestIntegerDistance = integerDistance;
    }
  }

  if (bestIntegerDistance > 0.15) {
    bestFillingPenalty = Math.max(0.2, 1.0 - (bestIntegerDistance - 0.15) * 4.0);
  }

  if (bestIntegerDistance >= 0.05 && bestIntegerDistance <= 0.35) {
    bestDopingBoost = 1.0 - Math.abs(bestIntegerDistance - 0.18) / 0.18;
    bestDopingBoost = Math.max(0, Math.min(1.0, bestDopingBoost));
  }

  const mottProx = electronic.mottProximityScore ?? 0;
  const correlation = electronic.correlationStrength ?? 0;

  let mottScore = 0;

  if (maxUoverW > 1.5) {
    mottScore = 0.95;
  } else if (maxUoverW > 0.8) {
    const distFromTransition = Math.abs(maxUoverW - 1.0);
    mottScore = Math.exp(-distFromTransition * distFromTransition * 8.0);
  } else if (maxUoverW > 0.3) {
    mottScore = maxUoverW * 0.3;
  }

  if (bestIntegerDistance < 0.05) {
    mottScore *= 1.0;
  } else {
    mottScore *= bestFillingPenalty;
    if (bestDopingBoost > 0 && maxUoverW > 0.6) {
      mottScore = Math.min(1.0, mottScore + bestDopingBoost * 0.25);
    }
  }

  mottScore = Math.max(mottScore, mottProx * 0.8 * bestFillingPenalty);

  if (correlation > 0.5) {
    mottScore = Math.min(1.0, mottScore + (correlation - 0.5) * 0.3 * bestFillingPenalty);
  }

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3 &&
    elements.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Bi", "Tl", "Hg"].includes(e));
  const isNickelate = elements.includes("Ni") && elements.includes("O") && elements.length >= 3;

  if (isCuprate) mottScore = Math.max(mottScore, 0.90 * bestFillingPenalty);
  if (isNickelate) mottScore = Math.max(mottScore, 0.75 * bestFillingPenalty);

  return Math.min(1.0, mottScore);
}

function computeSDWChannel(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  electronic: ElectronicStructure,
  lindhard?: LindhardSusceptibility,
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

  if (lindhard && lindhard.sdwSusceptibility > 0) {
    const chi0Contribution = Math.min(0.5, lindhard.sdwSusceptibility * 0.1);
    sdwScore += chi0Contribution;

    if (lindhard.divergenceProximity > 0.5) {
      sdwScore += (lindhard.divergenceProximity - 0.5) * 0.4;
    }
  } else if (nestingScore > 0.4) {
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
  lindhard?: LindhardSusceptibility,
): number {
  const nestingScore = electronic.nestingScore ?? 0;
  const vanHoveProx = electronic.vanHoveProximity ?? 0;
  const dosEF = electronic.densityOfStatesAtFermi;

  let cdwScore = 0;

  if (lindhard && lindhard.cdwSusceptibility > 0) {
    const chi0Contribution = Math.min(0.5, lindhard.cdwSusceptibility * 0.08);
    cdwScore += chi0Contribution * 0.5;

    if (lindhard.divergenceProximity > 0.3 && phonon.softModePresent) {
      cdwScore += lindhard.divergenceProximity * phonon.softModeScore * 0.4;
    }
  } else {
    const nestingDriven = Math.min(1.0, nestingScore * dosEF * 0.3);
    cdwScore += nestingDriven * 0.4;
  }

  if (vanHoveProx > 0.3) {
    cdwScore += (vanHoveProx - 0.3) * 0.4;
  }

  if (phonon.softModePresent) {
    const lindhardBoost = lindhard ? (1 + lindhard.divergenceProximity * 0.5) : 1.0;
    cdwScore += phonon.softModeScore * 0.3 * lindhardBoost;
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
  pressureGpa?: number,
  structuralTransitionPressure?: number,
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

  if (pressureGpa !== undefined && pressureGpa > 0 &&
      structuralTransitionPressure !== undefined && structuralTransitionPressure > 0) {
    const pRatio = pressureGpa / structuralTransitionPressure;
    const pDistance = Math.abs(pRatio - 1.0);

    if (pDistance < 0.10) {
      const proximityBoost = (0.10 - pDistance) / 0.10;
      tcEnhancementFactor *= (1.0 + proximityBoost * 2.0);
    } else if (pDistance < 0.20) {
      const proximityBoost = (0.20 - pDistance) / 0.20;
      tcEnhancementFactor *= (1.0 + proximityBoost * 0.8);
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
  pressureGpa?: number;
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
  const pressureGpa = physicsData?.pressureGpa;

  let lindhard: LindhardSusceptibility | undefined;
  try {
    const fs = computeFermiSurface(formula);
    lindhard = fs.lindhardSusceptibility;
  } catch {
    lindhard = undefined;
  }

  const mottScore = computeMottChannel(elements, counts, electronic);
  const sdwScore = computeSDWChannel(formula, elements, counts, electronic, lindhard);
  const cdwScore = computeCDWChannel(elements, electronic, phonon, lindhard);
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

  const structTransP = estimateStructuralTransitionPressure(elements, counts);
  const dome = computeDomeProfile(
    qcpType, channelScores, electronic,
    pressureGpa, structTransP > 0 ? structTransP : undefined,
  );

  const spin = computeDynamicSpinSusceptibility(formula, electronic);
  const stonerEnhancement = guardedStonerEnhancement(spin.stonerEnhancement);
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
