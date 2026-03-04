import OpenAI from "openai";
import type { EventEmitter } from "./engine";
import type { SuperconductorCandidate } from "@shared/schema";
import {
  ELEMENTAL_DATA,
  getElementData,
  getCompositionWeightedProperty,
  getAverageMass,
  getLightestMass,
  isTransitionMetal,
  isRareEarth,
  isActinide,
  hasDOrFElectrons,
  getHubbardU,
  getStonerParameter,
  getMcMillanHopfieldEta,
  getDebyeTemperature,
} from "./elemental-data";
import type { MPSummaryData, MPElasticityData } from "./materials-project-client";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export interface ElectronicStructure {
  bandStructureType: string;
  fermiSurfaceTopology: string;
  densityOfStatesAtFermi: number;
  correlationStrength: number;
  metallicity: number;
  orbitalCharacter: string;
  nestingFeatures: string;
}

export interface PhononSpectrum {
  maxPhononFrequency: number;
  logAverageFrequency: number;
  hasImaginaryModes: boolean;
  anharmonicityIndex: number;
  softModePresent: boolean;
  debyeTemperature: number;
}

export interface ElectronPhononCoupling {
  lambda: number;
  omegaLog: number;
  muStar: number;
  isStrongCoupling: boolean;
  dominantPhononBranch: string;
}

export interface EliashbergResult {
  predictedTc: number;
  gapRatio: number;
  isotropicGap: boolean;
  strongCouplingCorrection: number;
  confidenceBand: [number, number];
}

export interface CompetingPhase {
  phaseName: string;
  type: string;
  transitionTemp: number | null;
  strength: number;
  suppressesSC: boolean;
}

export interface CriticalFieldResult {
  upperCriticalField: number;
  coherenceLength: number;
  londonPenetrationDepth: number;
  anisotropyRatio: number;
  criticalCurrentDensity: number;
  typeIorII: string;
}

function parseFormulaElements(formula: string): string[] {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? [...new Set(matches)] : [];
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

const LAMBDA_CONVERSION = 562000;

function estimateBandwidthW(el: string): number {
  const data = getElementData(el);
  if (!data) return 6.0;

  const period = data.atomicNumber <= 36 ? 4 : data.atomicNumber <= 54 ? 5 : 6;
  if (isTransitionMetal(el)) {
    if (period === 4) return 5.5;
    if (period === 5) return 8.0;
    return 10.0;
  }
  if (isRareEarth(el)) return 1.5;
  if (isActinide(el)) return 2.0;

  if (data.sommerfeldGamma && data.sommerfeldGamma > 0) {
    const nEf = data.sommerfeldGamma / 2.359;
    if (nEf > 0.1 && data.valenceElectrons > 0) {
      const W = data.valenceElectrons / nEf;
      return Math.max(2.0, Math.min(15.0, W));
    }
  }

  return 8.0;
}

function invertMcMillanLambda(tc: number, thetaD: number, muStar: number): number {
  if (tc <= 0 || thetaD <= 0) return 0;
  const omegaLogK = thetaD * 0.695 * 1.44 * 0.65;
  let lambdaLow = 0.05;
  let lambdaHigh = 4.0;
  for (let i = 0; i < 50; i++) {
    const lambdaMid = (lambdaLow + lambdaHigh) / 2;
    const denom = lambdaMid - muStar * (1 + 0.62 * lambdaMid);
    if (denom <= 0) { lambdaLow = lambdaMid; continue; }
    const exponent = -1.04 * (1 + lambdaMid) / denom;
    const tcCalc = (omegaLogK / 1.2) * Math.exp(exponent);
    if (tcCalc < tc) lambdaLow = lambdaMid;
    else lambdaHigh = lambdaMid;
  }
  return (lambdaLow + lambdaHigh) / 2;
}

function getElementalLambda(el: string): number | null {
  const data = getElementData(el);
  if (!data || !data.debyeTemperature) return null;
  if (data.elementalTc && data.elementalTc > 0) {
    return invertMcMillanLambda(data.elementalTc, data.debyeTemperature, 0.10);
  }
  return null;
}

function estimateHubbardUoverW(elements: string[], counts: Record<string, number>): number {
  let maxUoverW = 0;
  const totalAtoms = getTotalAtoms(counts);

  for (const el of elements) {
    const U = getHubbardU(el);
    if (U === null) continue;

    const data = getElementData(el);
    if (!data) continue;

    const elFraction = (counts[el] || 1) / totalAtoms;

    let W = estimateBandwidthW(el);

    if (elements.includes("O")) W *= 0.7;
    if (elements.length > 3) W *= 0.85;

    const ratio = (U / W) * Math.sqrt(elFraction);
    if (ratio > maxUoverW) maxUoverW = ratio;
  }

  return maxUoverW;
}

function estimateDOSatFermi(elements: string[], counts: Record<string, number>): number {
  const gammaAvg = getCompositionWeightedProperty(counts, "sommerfeldGamma");

  if (gammaAvg !== null && gammaAvg > 0) {
    const nEf = gammaAvg / 2.359;
    return Math.max(0.1, Math.min(10, nEf));
  }

  let dos = 0.5;
  const totalAtoms = getTotalAtoms(counts);
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    if (isTransitionMetal(el)) {
      dos += frac * (1.5 + data.valenceElectrons * 0.15);
    } else if (isRareEarth(el)) {
      dos += frac * 3.0;
    } else {
      dos += frac * 0.3;
    }
  }

  return Math.max(0.1, Math.min(10, dos));
}

function estimateMetallicity(elements: string[], counts: Record<string, number>, mpData?: MPSummaryData | null): number {
  if (mpData) {
    if (mpData.isMetallic) return Math.max(0.7, 1.0 - mpData.bandGap * 0.1);
    if (mpData.bandGap > 3.0) return 0.05;
    if (mpData.bandGap > 1.0) return 0.15;
    if (mpData.bandGap > 0.1) return 0.35;
    return 0.6;
  }

  const totalAtoms = getTotalAtoms(counts);
  const hasH = elements.includes("H");
  const hasO = elements.includes("O");
  const hCount = counts["H"] || 0;
  const oCount = counts["O"] || 0;

  let metalFrac = 0;
  let nonmetalFrac = 0;
  const nonmetals = ["H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "I", "Xe"];
  const halogens = ["F", "Cl", "Br", "I"];

  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    if (nonmetals.includes(el)) {
      nonmetalFrac += frac;
    } else {
      metalFrac += frac;
    }
  }

  const halogenFrac = halogens.reduce((s, h) => s + ((counts[h] || 0) / totalAtoms), 0);
  const enSpread = (() => {
    const enValues = elements.map(el => getElementData(el)?.paulingElectronegativity ?? 1.8);
    if (enValues.length < 2) return 0;
    return Math.max(...enValues) - Math.min(...enValues);
  })();

  let metallicity: number;

  if (hasH && hCount / totalAtoms > 0.5) {
    const metalElements = elements.filter(e => !nonmetals.includes(e));
    if (metalElements.length > 0) {
      const metalPresence = metalElements.reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
      const hToMetal = hCount / Math.max(1, metalElements.reduce((s, e) => s + (counts[e] || 0), 0));
      if (hToMetal >= 6) {
        metallicity = 0.80 + metalPresence * 0.1;
      } else {
        metallicity = 0.5 + metalPresence * 0.4;
      }
    } else {
      metallicity = 0.2;
    }
  } else if (metalFrac < 0.15 && nonmetalFrac > 0.7) {
    metallicity = 0.1;
  } else if (halogenFrac > 0.3) {
    metallicity = 0.12;
  } else if (enSpread > 2.5) {
    metallicity = 0.15;
  } else if (metalFrac > 0.8 && enSpread < 0.5) {
    metallicity = 0.92;
  } else if (metalFrac > 0.5) {
    metallicity = 0.6 + metalFrac * 0.3;
  } else {
    metallicity = 0.3 + metalFrac * 0.4;
  }

  if (hasO && oCount / totalAtoms > 0.5) metallicity *= 0.6;
  if (elements.some(e => isTransitionMetal(e)) && !hasO && elements.length <= 3) {
    metallicity = Math.max(metallicity, 0.85);
  }

  return Math.max(0.05, Math.min(1.0, metallicity));
}

export function computeElectronicStructure(
  formula: string,
  spacegroup?: string | null,
  mpData?: MPSummaryData | null
): ElectronicStructure {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const uOverW = estimateHubbardUoverW(elements, counts);
  let correlationStrength: number;
  if (uOverW > 1.5) correlationStrength = Math.min(1.0, 0.85 + (uOverW - 1.5) * 0.1);
  else if (uOverW > 1.0) correlationStrength = 0.65 + (uOverW - 1.0) * 0.4;
  else if (uOverW > 0.5) correlationStrength = 0.35 + (uOverW - 0.5) * 0.6;
  else correlationStrength = uOverW * 0.7;

  if (elements.includes("Cu") && elements.includes("O")) {
    correlationStrength = Math.max(correlationStrength, 0.78);
  }
  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("Se"))) {
    correlationStrength = Math.max(correlationStrength, 0.55);
  }

  const metallicity = estimateMetallicity(elements, counts, mpData);
  const densityOfStatesAtFermi = estimateDOSatFermi(elements, counts);

  const hasTM = elements.some(e => isTransitionMetal(e));
  const hasRE = elements.some(e => isRareEarth(e));
  const hasH = elements.includes("H");
  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

  const vec = (() => {
    let totalVE = 0;
    for (const el of elements) {
      const data = getElementData(el);
      if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
    }
    return totalVE / totalAtoms;
  })();

  let fermiSurfaceTopology = "simple spherical";
  if (elements.includes("Cu") && elements.includes("O")) {
    fermiSurfaceTopology = "quasi-2D cylindrical with nesting features at (pi,pi)";
  } else if (hasH && hRatio >= 6) {
    fermiSurfaceTopology = "nested multi-sheet with strong e-ph coupling pockets";
  } else if (hasTM && elements.length >= 3) {
    if (vec > 4 && vec < 7) {
      fermiSurfaceTopology = "multi-band with electron and hole pockets (partial nesting)";
    } else {
      fermiSurfaceTopology = "multi-band with electron and hole pockets";
    }
  } else if (hasTM) {
    fermiSurfaceTopology = "complex multi-sheet d-band dominated";
  } else if (vec > 1 && vec < 3) {
    fermiSurfaceTopology = "nearly free electron (spherical)";
  }

  let orbitalCharacter = "sp-hybridized";
  if (elements.includes("Cu") && elements.includes("O")) {
    orbitalCharacter = "Cu-3d(x2-y2) / O-2p hybridized";
  } else if (elements.includes("Fe") && (elements.includes("As") || elements.includes("P") || elements.includes("Se"))) {
    orbitalCharacter = "Fe-3d multi-orbital (t2g/eg)";
  } else if (hasTM) {
    const tmElements = elements.filter(e => isTransitionMetal(e));
    const mainTM = tmElements[0] || "d";
    orbitalCharacter = `${mainTM}-d band dominated with p-hybridization`;
  } else if (hasH && hRatio >= 4) {
    orbitalCharacter = "H-1s sigma-bonding network";
  } else if (hasRE) {
    orbitalCharacter = "f-electron hybridized with conduction band";
  }

  let bandStructureType = "metallic";
  if (metallicity < 0.2) bandStructureType = "insulating";
  else if (metallicity < 0.4) bandStructureType = "semiconductor/semimetal";
  else if (correlationStrength > 0.65) bandStructureType = "strongly correlated metal";

  const nestingStrength = correlationStrength > 0.5 || (vec > 4 && vec < 7 && hasTM);
  const nestingFeatures = nestingStrength
    ? "Significant Fermi surface nesting promoting spin/charge instabilities"
    : "Weak nesting, conventional metallic behavior";

  return {
    bandStructureType,
    fermiSurfaceTopology,
    densityOfStatesAtFermi: Number(densityOfStatesAtFermi.toFixed(3)),
    correlationStrength: Number(correlationStrength.toFixed(4)),
    metallicity: Number(metallicity.toFixed(4)),
    orbitalCharacter,
    nestingFeatures,
  };
}

export function computePhononSpectrum(
  formula: string,
  electronicStructure: ElectronicStructure,
  mpElasticity?: MPElasticityData | null,
  mpSummary?: MPSummaryData | null
): PhononSpectrum {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const hasH = elements.includes("H");
  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;
  const isHydrogenRich = hRatio >= 6;

  const thetaDAvg = getCompositionWeightedProperty(counts, "debyeTemperature");
  const avgMass = getAverageMass(counts);
  let debyeTemperature: number;
  if (thetaDAvg !== null && thetaDAvg > 0) {
    debyeTemperature = thetaDAvg;
    if (isHydrogenRich) {
      const hFraction = hCount / totalAtoms;
      debyeTemperature = thetaDAvg * (1 - hFraction) + 2000 * hFraction;
    }
  } else {
    debyeTemperature = 300 * Math.sqrt(30 / Math.max(avgMass, 1));
    if (isHydrogenRich) debyeTemperature = Math.max(debyeTemperature, 1500);
  }
  debyeTemperature = Math.max(50, Math.round(debyeTemperature));

  const THETA_D_TO_CM1 = 0.695;
  const omegaD_cm1 = debyeTemperature * THETA_D_TO_CM1;

  let maxPhononFreq: number;
  if (hasH && isHydrogenRich) {
    maxPhononFreq = Math.max(omegaD_cm1, 3000 + hRatio * 50);
  } else if (hasH) {
    maxPhononFreq = Math.max(omegaD_cm1, 1200 + hRatio * 150);
  } else if (elements.length === 1) {
    maxPhononFreq = omegaD_cm1 * 1.2;
  } else {
    const lightestMass = getLightestMass(elements);
    const massRatio = Math.sqrt(avgMass / Math.max(lightestMass, 1));
    maxPhononFreq = omegaD_cm1 * Math.min(2.5, massRatio * 1.1);
  }
  maxPhononFreq = Math.max(50, Math.min(5000, Math.round(maxPhononFreq)));

  let logAvgFreqRatio: number;
  if (isHydrogenRich) {
    logAvgFreqRatio = 0.30 + 0.05 * Math.min(hRatio / 10, 1);
  } else if (hasH) {
    logAvgFreqRatio = 0.25 + hRatio * 0.02;
  } else if (elements.length === 1) {
    logAvgFreqRatio = 0.65;
  } else {
    logAvgFreqRatio = 0.45;
  }
  const logAvgFreq = Math.max(20, Math.round(maxPhononFreq * logAvgFreqRatio));

  let hasImaginaryModes = false;
  if (mpSummary && mpSummary.energyAboveHull > 0.1) {
    hasImaginaryModes = true;
  } else if (!mpSummary) {
    const enSpread = (() => {
      const enValues = elements.map(el => getElementData(el)?.paulingElectronegativity ?? 1.8);
      if (enValues.length < 2) return 0;
      return Math.max(...enValues) - Math.min(...enValues);
    })();
    if (enSpread > 2.8 && elements.length >= 4) hasImaginaryModes = true;
    if (electronicStructure.correlationStrength > 0.85 && elements.length >= 4) hasImaginaryModes = true;
  }

  const grunAvg = getCompositionWeightedProperty(counts, "gruneisenParameter");
  let anharmonicityIndex: number;
  if (grunAvg !== null && grunAvg > 0) {
    anharmonicityIndex = (grunAvg - 1.0) * 0.4;
    if (hasH) {
      const hFrac = hCount / totalAtoms;
      anharmonicityIndex += hFrac * 0.3;
    }
  } else {
    anharmonicityIndex = 0.15;
    if (isHydrogenRich) anharmonicityIndex = 0.45;
    else if (hasH) anharmonicityIndex = 0.2 + hRatio * 0.03;
  }
  anharmonicityIndex = Math.max(0.0, Math.min(1.0, anharmonicityIndex));

  let softModePresent = false;
  if (electronicStructure.correlationStrength > 0.6 && electronicStructure.densityOfStatesAtFermi > 3.0) {
    softModePresent = true;
  }
  if (elements.some(e => ["Ba", "Sr", "Ca", "Pb"].includes(e)) && elements.includes("O") && elements.length >= 3) {
    softModePresent = true;
  }

  return {
    maxPhononFrequency: Math.round(maxPhononFreq),
    logAverageFrequency: logAvgFreq,
    hasImaginaryModes,
    anharmonicityIndex: Number(anharmonicityIndex.toFixed(3)),
    softModePresent,
    debyeTemperature,
  };
}

export function computeElectronPhononCoupling(
  electronicStructure: ElectronicStructure,
  phononSpectrum: PhononSpectrum,
  formula?: string
): ElectronPhononCoupling {
  const omega_log = phononSpectrum.logAverageFrequency;
  const metal = electronicStructure.metallicity;
  const corr = electronicStructure.correlationStrength;
  const N_EF = electronicStructure.densityOfStatesAtFermi;

  let lambda: number;

  if (formula) {
    const elements = parseFormulaElements(formula);
    const counts = parseFormulaCounts(formula);
    const totalAtoms = getTotalAtoms(counts);

    const hCount = counts["H"] || 0;
    const metalAtomCount = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
      .reduce((s, e) => s + (counts[e] || 0), 0);
    const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;

    let lambdaSum = 0;
    let totalWeight = 0;

    for (const el of elements) {
      const data = getElementData(el);
      if (!data) continue;

      const frac = (counts[el] || 1) / totalAtoms;

      const knownLambda = getElementalLambda(el);
      if (knownLambda !== null && knownLambda > 0) {
        lambdaSum += knownLambda * frac;
        totalWeight += frac;
      } else {
        const eta = getMcMillanHopfieldEta(el);
        if (eta !== null && eta > 0) {
          const M = data.atomicMass;
          const thetaD = data.debyeTemperature || phononSpectrum.debyeTemperature;
          const lambdaEl = (eta * LAMBDA_CONVERSION) / (M * thetaD * thetaD);
          lambdaSum += lambdaEl * frac;
          totalWeight += frac;
        }
      }
    }

    if (totalWeight > 0) {
      lambda = lambdaSum;
      if (totalWeight < 0.5) {
        lambda = Math.max(lambda, N_EF * 0.1 * (1 + phononSpectrum.anharmonicityIndex * 0.5));
      } else if (totalWeight < 0.99) {
        const missingFrac = 1 - totalWeight;
        const inferredLambda = lambda / totalWeight;
        lambda = lambda + inferredLambda * missingFrac * 0.5;
      }

      if (hCount > 0 && hRatio >= 4) {
        const H_theta = 2000;
        const H_eta = 3.0 + hRatio * 0.5;
        const H_mass = 1.008;
        const lambda_H = (H_eta * LAMBDA_CONVERSION) / (H_mass * H_theta * H_theta) * (hCount / totalAtoms);
        lambda += lambda_H;
      }

      const lightEl = elements.filter(e => {
        const d = getElementData(e);
        return d && d.atomicMass < 15 && e !== "H";
      });
      if (lightEl.length > 0) {
        const lightFrac = lightEl.reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
        if (lightFrac > 0.3) {
          const lightBoost = 1.0 + lightFrac * 2.5 * (phononSpectrum.debyeTemperature / 500);
          lambda *= lightBoost;
        }
      }
    } else {
      lambda = N_EF * 0.15 * (1 + phononSpectrum.anharmonicityIndex * 0.5);
    }

    if (electronicStructure.fermiSurfaceTopology.includes("nesting")) lambda *= 1.15;

    if (hRatio >= 6) {
      // no correlation suppression or metallicity penalty for superhydrides
    } else {
      if (corr > 0.7) lambda *= (1.0 - (corr - 0.7) * 0.5);
      if (metal < 0.4) lambda *= metal;
    }

    if (hCount > 0 && hRatio < 4 && hRatio > 0) {
      lambda *= 0.3 + 0.175 * hRatio;
    }
  } else {
    lambda = N_EF * 0.15 * (1 + phononSpectrum.anharmonicityIndex * 0.5);
    if (electronicStructure.fermiSurfaceTopology.includes("nesting")) lambda *= 1.15;
    if (corr > 0.7) lambda *= 0.6;
    if (metal < 0.4) lambda *= metal;
  }

  lambda = Math.max(0.05, Math.min(3.5, lambda));

  const avgEN = formula ? (getCompositionWeightedProperty(parseFormulaCounts(formula), "paulingElectronegativity") || 1.8) : 1.8;
  const mu_bare = 0.1 + avgEN * 0.02;
  const thetaD = phononSpectrum.debyeTemperature;
  const E_F_eV = 5.0 + N_EF * 0.5;
  const omega_D_eV = thetaD * 8.617e-5;
  const logRatio = Math.log(Math.max(E_F_eV / Math.max(omega_D_eV, 0.001), 1.1));
  const muStar = mu_bare / (1 + mu_bare * logRatio);
  const muStarClamped = Math.max(0.08, Math.min(0.20, muStar));

  const isStrongCoupling = lambda > 1.5;

  let dominantPhononBranch = "acoustic";
  if (phononSpectrum.maxPhononFrequency > 2000) dominantPhononBranch = "high-frequency optical (H vibrations)";
  else if (phononSpectrum.softModePresent) dominantPhononBranch = "soft optical mode";
  else if (lambda > 1.0) dominantPhononBranch = "low-energy optical";

  return {
    lambda: Number(lambda.toFixed(3)),
    omegaLog: omega_log,
    muStar: Number(muStarClamped.toFixed(4)),
    isStrongCoupling,
    dominantPhononBranch,
  };
}

export function predictTcEliashberg(coupling: ElectronPhononCoupling): EliashbergResult {
  const { lambda, omegaLog, muStar } = coupling;

  const omegaLogK = omegaLog * 1.44;

  let tc: number;
  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denominator) < 1e-6 || denominator <= 0) {
    tc = 0;
  } else if (lambda < 1.5) {
    const f1 = Math.pow(1 + (lambda / 2.46 / (1 + 3.8 * muStar)), 1/3);
    const exponent = -1.04 * (1 + lambda) / denominator;
    tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
  } else {
    const f1 = Math.sqrt(1 + (lambda / 2.46));
    const exponent = -1.04 * (1 + lambda) / denominator;
    tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
  }

  tc = Number.isFinite(tc) ? Math.max(0, tc) : 0;

  const gapRatio = lambda > 1.5 ? 2 * 1.764 * (1 + 12.5 * (lambda / (lambda + 5)) * (lambda / (lambda + 5))) : 2 * 1.764;
  const isotropicGap = lambda < 1.0;
  const strongCouplingCorrection = lambda > 1.5 ? 1 + 5.3 * (lambda / (lambda + 6)) * (lambda / (lambda + 6)) : 1.0;

  const uncertainty = tc * 0.15;
  const confidenceBand: [number, number] = [
    Math.max(0, Math.round(tc - uncertainty)),
    Math.round(tc + uncertainty),
  ];

  return {
    predictedTc: Math.round(tc * 10) / 10,
    gapRatio: Number(gapRatio.toFixed(3)),
    isotropicGap,
    strongCouplingCorrection: Number(strongCouplingCorrection.toFixed(3)),
    confidenceBand,
  };
}

export function evaluateCompetingPhases(
  formula: string,
  electronicStructure: ElectronicStructure,
  mpSummary?: MPSummaryData | null
): CompetingPhase[] {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const phases: CompetingPhase[] = [];

  const magneticElements3d = ["Cr", "Mn", "Fe", "Co", "Ni"];
  const hasMagnetic3d = elements.filter(e => magneticElements3d.includes(e));

  for (const magEl of hasMagnetic3d) {
    const stonerI = getStonerParameter(magEl);
    const N_EF = electronicStructure.densityOfStatesAtFermi;

    if (stonerI !== null) {
      const stonerProduct = stonerI * N_EF;
      const isFM = stonerProduct > 1.0;

      const data = getElementData(magEl);
      const S_eff = data ? Math.min(data.valenceElectrons * 0.3, 2.5) : 1.5;
      const z_coord = 8;
      const J_exchange = (stonerI || 0.5) * 0.1;
      const T_mag = Math.round(2 * z_coord * J_exchange * S_eff * (S_eff + 1) / (3 * 8.617e-5) * 0.01);

      const cuprateType = elements.includes("Cu") && elements.includes("O");
      const ironPnictide = magEl === "Fe" && (elements.includes("As") || elements.includes("P") || elements.includes("Se"));

      if (isFM && !cuprateType && !ironPnictide) {
        const strength = Math.min(1.0, stonerProduct * 0.5);
        phases.push({
          phaseName: `Ferromagnetic order (${magEl} sublattice)`,
          type: "magnetism",
          transitionTemp: Math.min(1500, Math.max(10, T_mag * 3)),
          strength,
          suppressesSC: strength > 0.5,
        });
      } else {
        const afmStrength = Math.min(1.0, electronicStructure.correlationStrength * 0.8);
        let T_neel = Math.round(T_mag * 1.5);
        if (ironPnictide) T_neel = Math.min(T_neel, 200);
        if (cuprateType) T_neel = Math.min(T_neel, 400);

        phases.push({
          phaseName: `Antiferromagnetic order (${magEl} sublattice)`,
          type: "magnetism",
          transitionTemp: Math.max(5, T_neel),
          strength: afmStrength,
          suppressesSC: afmStrength > 0.7,
        });
      }
    }
  }

  if (mpSummary && mpSummary.magneticOrdering !== "NM" && hasMagnetic3d.length === 0) {
    const ordering = mpSummary.magneticOrdering;
    phases.push({
      phaseName: `Magnetic order (${ordering} from DFT)`,
      type: "magnetism",
      transitionTemp: null,
      strength: Math.min(1.0, mpSummary.totalMagnetization * 0.3),
      suppressesSC: mpSummary.totalMagnetization > 2.0,
    });
  }

  const vec = (() => {
    let totalVE = 0;
    const totalAtoms = getTotalAtoms(counts);
    for (const el of elements) {
      const data = getElementData(el);
      if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
    }
    return totalVE / totalAtoms;
  })();

  const hasTM = elements.some(e => isTransitionMetal(e));
  if (hasTM && vec > 4 && vec < 7 && electronicStructure.fermiSurfaceTopology.includes("nesting")) {
    const cdwStrength = 0.3 + (6 - Math.abs(vec - 5.5)) * 0.1;
    const T_cdw = Math.round(50 + cdwStrength * 200);
    phases.push({
      phaseName: "Charge density wave",
      type: "CDW",
      transitionTemp: T_cdw,
      strength: Math.min(1.0, cdwStrength),
      suppressesSC: cdwStrength > 0.6,
    });
  }

  const uOverW = estimateHubbardUoverW(elements, counts);
  if (uOverW > 1.0) {
    const mottStrength = Math.min(1.0, uOverW * 0.5);
    phases.push({
      phaseName: "Mott insulating phase",
      type: "Mott",
      transitionTemp: null,
      strength: mottStrength,
      suppressesSC: uOverW > 1.5,
    });
  }

  if (elements.includes("Cu") && elements.includes("O")) {
    const cuCount = counts["Cu"] || 1;
    const T_pseudo = Math.round(200 + cuCount * 30);
    phases.push({
      phaseName: "Pseudogap phase",
      type: "pseudogap",
      transitionTemp: Math.min(400, T_pseudo),
      strength: 0.5,
      suppressesSC: false,
    });
    const T_sdw = Math.round(80 + electronicStructure.correlationStrength * 150);
    phases.push({
      phaseName: "Spin-density wave",
      type: "SDW",
      transitionTemp: Math.min(350, T_sdw),
      strength: 0.6,
      suppressesSC: true,
    });
  }

  if (elements.includes("O") && elements.length >= 3) {
    const aElements = elements.filter(e => {
      const d = getElementData(e);
      return d && d.atomicRadius > 120 && !isTransitionMetal(e);
    });
    const bElements = elements.filter(e => isTransitionMetal(e));

    if (aElements.length > 0 && bElements.length > 0) {
      const rA = Math.max(...aElements.map(e => getElementData(e)!.atomicRadius));
      const rB = Math.max(...bElements.map(e => getElementData(e)!.atomicRadius));
      const rO = 140;
      const toleranceFactor = (rA + rO) / (Math.sqrt(2) * (rB + rO));

      if (toleranceFactor < 0.8 || toleranceFactor > 1.05) {
        const distortion = Math.abs(toleranceFactor - 0.95);
        phases.push({
          phaseName: `Structural distortion (t=${toleranceFactor.toFixed(2)})`,
          type: "structural",
          transitionTemp: Math.round(100 + distortion * 800),
          strength: Math.min(0.8, distortion * 2),
          suppressesSC: distortion > 0.3,
        });
      }
    }
  }

  return phases;
}

export function computeCriticalFields(
  tc: number,
  coupling: ElectronPhononCoupling,
  dimensionality: string
): CriticalFieldResult {
  if (tc <= 0) {
    return {
      upperCriticalField: 0,
      coherenceLength: 0,
      londonPenetrationDepth: 0,
      anisotropyRatio: 1,
      criticalCurrentDensity: 0,
      typeIorII: "N/A",
    };
  }

  const lambda = Math.max(coupling.lambda, 0.01);
  const vF = 2e5;
  const kB = 1.381e-23;
  const hbar = 1.055e-34;
  const delta0 = 1.764 * kB * tc * (1 + lambda * 0.3);
  const xiRaw = (hbar * vF) / (Math.PI * delta0);
  const xiNm = xiRaw * 1e9;
  const coherenceLength = Math.max(1.0, Math.min(500, Number.isFinite(xiNm) ? xiNm : 100));

  const PHI0 = 2.07e-15;
  const xiM = coherenceLength * 1e-9;
  const Hc2Tesla = PHI0 / (2 * Math.PI * xiM * xiM);
  const hc2Raw = Math.max(0, Number.isFinite(Hc2Tesla) ? Hc2Tesla : 0);
  const upperCriticalField = Math.min(hc2Raw, 300);

  const lambdaL = 50 + 200 * lambda * (1 + coupling.muStar);
  const londonPenetrationDepth = Math.max(30, Math.min(2000, lambdaL));

  let anisotropyRatio = 1.0;
  if (dimensionality === "2D" || dimensionality === "quasi-2D") anisotropyRatio = 8.0 + lambda * 2;
  else if (dimensionality === "layered") anisotropyRatio = 4.0 + lambda;

  const Jc = tc * 1e4 * lambda / (1 + anisotropyRatio * 0.1);
  const criticalCurrentDensity = Math.round(Jc);

  const kappa = coherenceLength > 0 ? londonPenetrationDepth / coherenceLength : 1;
  const typeIorII = kappa > 0.707 ? "Type-II" : "Type-I";

  return {
    upperCriticalField: Number(upperCriticalField.toFixed(2)),
    coherenceLength: Number(coherenceLength.toFixed(1)),
    londonPenetrationDepth: Number(londonPenetrationDepth.toFixed(1)),
    anisotropyRatio: Number(anisotropyRatio.toFixed(2)),
    criticalCurrentDensity,
    typeIorII,
  };
}

export function assessCorrelationStrength(formula: string): {
  ratio: number;
  regime: string;
  treatmentRequired: string;
} {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);

  const uOverW = estimateHubbardUoverW(elements, counts);

  let ratio = uOverW;

  if (elements.includes("Cu") && elements.includes("O")) {
    ratio = Math.max(ratio, 0.85);
  }
  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("Se"))) {
    ratio = Math.max(ratio, 0.55);
  }

  if (ratio === 0) {
    if (elements.includes("H")) ratio = 0.1;
    else {
      const avgEN = getCompositionWeightedProperty(counts, "paulingElectronegativity") || 1.8;
      ratio = avgEN > 2.5 ? 0.25 : 0.15;
    }
  }

  ratio = Math.max(0, Math.min(1.0, ratio));

  let regime = "weakly correlated";
  let treatmentRequired = "DFT + DFPT + Migdal-Eliashberg";

  if (ratio > 0.7) {
    regime = "strongly correlated";
    treatmentRequired = "DMFT + beyond-DFT (GW/QMC) + unconventional pairing analysis";
  } else if (ratio > 0.4) {
    regime = "moderately correlated";
    treatmentRequired = "DFT+U or hybrid functionals + extended Eliashberg";
  }

  return {
    ratio: Number(ratio.toFixed(3)),
    regime,
    treatmentRequired,
  };
}

export async function runFullPhysicsAnalysis(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{
  electronicStructure: ElectronicStructure;
  phononSpectrum: PhononSpectrum;
  coupling: ElectronPhononCoupling;
  eliashberg: EliashbergResult;
  competingPhases: CompetingPhase[];
  criticalFields: CriticalFieldResult;
  correlation: ReturnType<typeof assessCorrelationStrength>;
  dimensionality: string;
  uncertaintyEstimate: number;
}> {
  const formula = candidate.formula;

  emit("log", {
    phase: "phase-10",
    event: "Physics analysis started",
    detail: `Computing electronic structure, phonons, e-ph coupling for ${formula}`,
    dataSource: "Physics Engine",
  });

  let mpSummary: MPSummaryData | null = null;
  let mpElasticity: MPElasticityData | null = null;
  try {
    const mpClient = await import("./materials-project-client");
    if (mpClient.isApiAvailable()) {
      const mpData = await mpClient.fetchAllData(formula);
      mpSummary = mpData.summary;
      mpElasticity = mpData.elasticity;
    }
  } catch {}

  const correlation = assessCorrelationStrength(formula);
  const electronicStructure = computeElectronicStructure(formula, candidate.crystalStructure, mpSummary);

  const phononSpectrum = computePhononSpectrum(formula, electronicStructure, mpElasticity, mpSummary);
  const coupling = computeElectronPhononCoupling(electronicStructure, phononSpectrum, formula);

  let eliashberg: EliashbergResult;
  eliashberg = predictTcEliashberg(coupling);

  const competingPhases = evaluateCompetingPhases(formula, electronicStructure, mpSummary);
  const hasMottPhase = competingPhases.some(p => p.type === "Mott");
  const isMottInsulator = hasMottPhase && correlation.ratio > 0.7;
  const isNonMetallic = electronicStructure.metallicity < 0.4;

  if (isNonMetallic) {
    const metalFactor = Math.max(0.02, electronicStructure.metallicity);
    eliashberg.predictedTc = eliashberg.predictedTc * metalFactor;
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2)];
  } else if (isMottInsulator) {
    eliashberg.predictedTc = eliashberg.predictedTc * 0.05;
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 3)];
  } else if (correlation.ratio > 0.7) {
    eliashberg.predictedTc = eliashberg.predictedTc * 0.3;
    eliashberg.confidenceBand = [
      Math.round(eliashberg.predictedTc * 0.3),
      Math.round(eliashberg.predictedTc * 2.5),
    ];
  } else if (correlation.ratio > 0.5) {
    eliashberg.predictedTc = eliashberg.predictedTc * 0.7;
    eliashberg.confidenceBand = [
      Math.round(eliashberg.predictedTc * 0.6),
      Math.round(eliashberg.predictedTc * 1.5),
    ];
  }

  let dimensionality = candidate.dimensionality || "3D";
  if (!candidate.dimensionality) {
    if (electronicStructure.fermiSurfaceTopology.includes("2D")) dimensionality = "quasi-2D";
    else if (electronicStructure.orbitalCharacter.includes("hybridized") && formula.includes("O")) dimensionality = "layered";
    else dimensionality = "3D";
  }

  const criticalFields = computeCriticalFields(eliashberg.predictedTc, coupling, dimensionality);

  const suppressingPhases = competingPhases.filter(p => p.suppressesSC);
  let uncertaintyEstimate = 0.3;
  if (correlation.ratio > 0.7) uncertaintyEstimate += 0.2;
  if (phononSpectrum.hasImaginaryModes) uncertaintyEstimate += 0.15;
  if (suppressingPhases.length > 0) uncertaintyEstimate += 0.1;
  if (phononSpectrum.anharmonicityIndex > 0.5) uncertaintyEstimate += 0.1;
  if (!mpSummary) uncertaintyEstimate += 0.05;
  uncertaintyEstimate = Math.min(0.95, uncertaintyEstimate);

  emit("log", {
    phase: "phase-10",
    event: "Physics analysis complete",
    detail: `${formula}: Tc=${eliashberg.predictedTc}K (${eliashberg.confidenceBand[0]}-${eliashberg.confidenceBand[1]}K), lambda=${coupling.lambda}, Hc2=${criticalFields.upperCriticalField}T, ${correlation.regime}, ${competingPhases.length} competing phases${mpSummary ? " [MP data]" : ""}`,
    dataSource: "Physics Engine",
  });

  return {
    electronicStructure,
    phononSpectrum,
    coupling,
    eliashberg,
    competingPhases,
    criticalFields,
    correlation,
    dimensionality,
    uncertaintyEstimate,
  };
}
