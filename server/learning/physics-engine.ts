import OpenAI from "openai";
import type { EventEmitter } from "./engine";
import type { SuperconductorCandidate } from "@shared/schema";
import type { DFTResolvedFeatures } from "./dft-feature-resolver";

export interface PhysicsConstraintMode {
  allowBeyondEmpirical: boolean;
  empiricalPenaltyStrength: number;
}

const defaultConstraintMode: PhysicsConstraintMode = {
  allowBeyondEmpirical: true,
  empiricalPenaltyStrength: 2.5,
};

let activeConstraintMode: PhysicsConstraintMode = { ...defaultConstraintMode };

export function setConstraintMode(mode: Partial<PhysicsConstraintMode>): void {
  activeConstraintMode = { ...activeConstraintMode, ...mode };
}

export function getConstraintMode(): PhysicsConstraintMode {
  return { ...activeConstraintMode };
}

function softCeiling(tc: number, threshold: number, penaltyStrength: number): number {
  if (tc <= threshold) return tc;
  const excess = tc - threshold;
  const dampened = threshold + excess / (1 + penaltyStrength * excess / threshold);
  return Math.round(dampened * 10) / 10;
}

function computePhysicsDerivedBonus(formula: string, lambda: number): number {
  const els = parseFormulaElements(formula);
  const cts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(cts);

  const isCuprate = els.includes("Cu") && els.includes("O") && els.length >= 3
    && els.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Bi", "Tl", "Hg"].includes(e));
  const isHEA = detectHighEntropyAlloy(formula);

  let bonus = 0;
  if (isCuprate && lambda > 0.5) {
    const cuFrac = (cts["Cu"] || 0) / totalAtoms;
    const oFrac = (cts["O"] || 0) / totalAtoms;
    bonus = Math.round(cuFrac * oFrac * 200 * Math.min(lambda, 2.0));
  }
  if (isHEA && lambda > 1.0) {
    const metalEls = els.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e) || HEA_EXTRA_METALS.includes(e));
    const entropyFactor = Math.log(Math.max(2, metalEls.length)) / Math.log(6);
    const heaBonus = Math.round(entropyFactor * lambda * 8);
    bonus = Math.max(bonus, heaBonus);
  }
  return bonus;
}

export function applyAmbientTcCap(tc: number, lambda: number, pressureGpa: number, metallicity: number, formula?: string): number {
  if (tc <= 0) return tc;
  const mode = activeConstraintMode;

  if (metallicity <= 0) return 0;

  let pressureThresholdLow = 10;
  let materialBonus = 0;

  if (formula) {
    const cts = parseFormulaCounts(formula);
    const els = parseFormulaElements(formula);
    const hCount = cts["H"] || 0;
    const metalEls = els.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e) || HEA_EXTRA_METALS.includes(e));
    const metalAtomCount = metalEls.reduce((s, e) => s + (cts[e] || 0), 0);
    const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;
    if (hRatio >= 6) pressureThresholdLow = 5;
    materialBonus = computePhysicsDerivedBonus(formula, lambda);
  }

  const isAmbient = pressureGpa < pressureThresholdLow;
  const isHighPressure = pressureGpa >= 50;
  const pressureFactor = isHighPressure ? 1.0 : isAmbient ? 0.0 : (pressureGpa - pressureThresholdLow) / (50 - pressureThresholdLow);

  if (!mode.allowBeyondEmpirical) {
    let tcCap: number;
    if (metallicity < 0.3) {
      tcCap = 20;
    } else if (metallicity < 0.5) {
      tcCap = 80;
    } else if (lambda < 0.3) {
      tcCap = 50;
    } else if (lambda < 0.5) {
      tcCap = 80;
    } else if (lambda < 1.0) {
      tcCap = Math.round(80 + (150 - 80) * pressureFactor);
    } else if (lambda < 1.5) {
      tcCap = Math.round(120 + (250 - 120) * pressureFactor);
    } else if (lambda < 2.5) {
      tcCap = Math.round(160 + (350 - 160) * pressureFactor);
    } else {
      tcCap = Math.round(200 + (350 - 200) * pressureFactor);
    }
    if (pressureGpa < 10) {
      tcCap += materialBonus;
      tcCap = Math.min(tcCap, 250);
    }
    return Math.min(tc, tcCap);
  }

  let baseExpectation: number;
  if (metallicity < 0.3) {
    baseExpectation = 20 + metallicity * 100;
  } else if (metallicity < 0.5) {
    baseExpectation = 50 + (metallicity - 0.3) * 200;
  } else if (lambda < 0.3) {
    baseExpectation = 30 + lambda * 100;
  } else if (lambda < 0.5) {
    baseExpectation = 50 + (lambda - 0.3) * 200;
  } else if (lambda < 1.0) {
    const ambientBase = 60;
    const hpBase = 150;
    baseExpectation = ambientBase + (hpBase - ambientBase) * pressureFactor;
  } else if (lambda < 1.5) {
    const ambientBase = 100;
    const hpBase = 250;
    baseExpectation = ambientBase + (hpBase - ambientBase) * pressureFactor;
  } else if (lambda < 2.5) {
    const ambientBase = 150;
    const hpBase = 400;
    baseExpectation = ambientBase + (hpBase - ambientBase) * pressureFactor;
  } else {
    const ambientBase = 200;
    const hpBase = 500;
    baseExpectation = ambientBase + (hpBase - ambientBase) * pressureFactor;
  }

  baseExpectation += materialBonus;
  baseExpectation = Math.round(baseExpectation);

  const penaltyStr = mode.empiricalPenaltyStrength;
  const result = softCeiling(tc, baseExpectation, penaltyStr);

  return Math.round(result);
}

const HEA_EXTRA_METALS = ["Al", "Mg", "Ca", "Sr", "Ba", "Li", "Na", "K", "Ti", "Zn", "Ga", "Ge", "Sn"];

function detectHighEntropyAlloy(formula: string): boolean {
  const els = parseFormulaElements(formula);
  const cts = parseFormulaCounts(formula);
  const metalEls = els.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e) ||
    HEA_EXTRA_METALS.includes(e));
  if (metalEls.length < 4) return false;
  const metalCounts = metalEls.map(e => cts[e] || 1);
  const totalMetal = metalCounts.reduce((s, n) => s + n, 0);
  const maxFrac = Math.max(...metalCounts) / totalMetal;
  return maxFrac <= 0.4;
}
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

export type HydrogenBondingType = "metallic-network" | "cage-clathrate" | "covalent-molecular" | "interstitial" | "ambiguous" | "none";

export function classifyHydrogenBonding(formula: string, pressureGpa: number = 0): HydrogenBondingType {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const hCount = counts["H"] || 0;
  if (hCount === 0) return "none";

  const nonmetals = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const metalElements = elements.filter(e => !nonmetals.includes(e));
  const metalAtomCount = metalElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const metalFrac = metalAtomCount / totalAtoms;
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;

  const cCount = counts["C"] || 0;
  const nCount = counts["N"] || 0;
  const oCount = counts["O"] || 0;
  const bCount = counts["B"] || 0;
  const cFrac = cCount / totalAtoms;
  const nonHNonMetFrac = elements.filter(e => nonmetals.includes(e) && e !== "H")
    .reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;

  const isOrganic = cFrac > 0.15 && hCount >= cCount;
  if (isOrganic) return "covalent-molecular";

  if ((cCount + nCount + oCount + bCount) / totalAtoms > 0.15 && hRatio < 6) {
    return "covalent-molecular";
  }

  if (hRatio >= 6 && metalFrac > 0.05 && pressureGpa >= 100 && nonHNonMetFrac < 0.1) {
    return "metallic-network";
  }

  if (hRatio >= 4 && metalFrac > 0.1 && pressureGpa >= 50) {
    return "cage-clathrate";
  }

  if (hRatio < 3 && metalFrac > 0.3) {
    return "interstitial";
  }

  if (pressureGpa < 50 && hRatio >= 4) {
    return "covalent-molecular";
  }

  if (hRatio >= 4 && pressureGpa >= 50) {
    return "cage-clathrate";
  }

  return "ambiguous";
}

type MaterialClass = "conventional-metal" | "cuprate" | "iron-pnictide" | "hydride-low-p" | "hydride-high-p" | "superhydride" | "light-element" | "heavy-fermion" | "other";

function classifyMaterialForLambda(formula: string, pressureGpa: number = 0): MaterialClass {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

  if (elements.includes("Cu") && elements.includes("O") && elements.length >= 3 &&
      elements.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Bi", "Tl", "Hg"].includes(e))) {
    return "cuprate";
  }
  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"))) {
    return "iron-pnictide";
  }
  if (elements.some(e => isRareEarth(e) || isActinide(e)) &&
      elements.some(e => isTransitionMetal(e)) && elements.length >= 3 &&
      !elements.includes("H")) {
    const reOrAct = elements.filter(e => isRareEarth(e) || isActinide(e));
    if (reOrAct.length > 0) return "heavy-fermion";
  }
  if (hCount > 0 && hRatio >= 6 && pressureGpa >= 100) return "superhydride";
  if (hCount > 0 && hRatio >= 4 && pressureGpa >= 50) return "hydride-high-p";
  if (hCount > 0 && hRatio >= 2) return "hydride-low-p";

  const lightEls = elements.filter(e => {
    const d = getElementData(e);
    return d && d.atomicMass < 15 && e !== "H";
  });
  const lightFrac = lightEls.reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
  if (lightFrac > 0.3) return "light-element";

  return "conventional-metal";
}

function getLambdaCapForClass(matClass: MaterialClass): number {
  switch (matClass) {
    case "conventional-metal": return 2.0;
    case "cuprate": return 1.5;
    case "iron-pnictide": return 1.8;
    case "heavy-fermion": return 1.0;
    case "hydride-low-p": return 2.5;
    case "hydride-high-p": return 3.0;
    case "superhydride": return 3.5;
    case "light-element": return 2.5;
    case "other": return 2.5;
  }
}

function getOmegaLogRangeForClass(matClass: MaterialClass): [number, number] {
  switch (matClass) {
    case "conventional-metal": return [50, 400];
    case "cuprate": return [100, 600];
    case "iron-pnictide": return [100, 500];
    case "heavy-fermion": return [20, 200];
    case "hydride-low-p": return [200, 800];
    case "hydride-high-p": return [500, 1500];
    case "superhydride": return [500, 1500];
    case "light-element": return [200, 1000];
    case "other": return [100, 600];
  }
}

const ELEMENT_BANDWIDTH: Record<string, number> = {
  Li: 3.5, Be: 6.0, Na: 3.0, Mg: 6.0, Al: 11.0, K: 2.5, Ca: 3.0,
  Sc: 4.0, Ti: 4.5, V: 5.0, Cr: 5.5, Mn: 4.0, Fe: 4.5, Co: 4.5,
  Ni: 4.5, Cu: 4.0, Zn: 7.0, Ga: 8.0, Sr: 3.5, Y: 3.5, Zr: 6.0,
  Nb: 5.5, Mo: 7.5, Tc: 7.0, Ru: 7.0, Rh: 6.5, Pd: 5.0, Ag: 4.0,
  In: 7.5, Sn: 8.0, Ba: 3.0, La: 3.0, Hf: 5.5, Ta: 6.5, W: 9.0,
  Re: 8.0, Os: 8.5, Ir: 7.5, Pt: 6.5, Au: 5.5, Tl: 6.0, Pb: 7.0,
  Bi: 6.5, Th: 3.5, U: 2.5,
};

function estimateBandwidthW(el: string): number {
  const known = ELEMENT_BANDWIDTH[el];
  if (known !== undefined) return known;

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

  const totalAtoms = getTotalAtoms(counts);
  let totalVE = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
  }
  const vec = totalVE / totalAtoms;

  let wAvg = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    wAvg += estimateBandwidthW(el) * frac;
  }
  wAvg = Math.max(1.0, wAvg);

  let dos = vec / (2 * wAvg);

  for (const el of elements) {
    const I = getStonerParameter(el);
    if (I !== null && I > 0) {
      const frac = (counts[el] || 1) / totalAtoms;
      const stonerProduct = I * dos;
      if (stonerProduct < 1.0 && stonerProduct > 0) {
        dos = dos / (1 - stonerProduct * frac);
      }
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
  const nonmetals = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const halogens = ["F", "Cl", "Br", "I"];
  const hasH = elements.includes("H");
  const hasB = elements.includes("B");
  const hCount = counts["H"] || 0;
  const bCount = counts["B"] || 0;
  const bFrac = bCount / totalAtoms;
  const bhFrac = (bCount + hCount) / totalAtoms;

  const metalElements = elements.filter(e => !nonmetals.includes(e));
  const metalAtomCount = metalElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const metalFrac = metalAtomCount / totalAtoms;
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;

  const isBoraneCage = hasB && hasH && bCount >= 4 && hCount >= 4
    && Math.abs(bCount - hCount) <= Math.max(bCount, hCount) * 0.5
    && bhFrac > 0.7;

  const isMetallicBoride = hasB && !hasH && metalFrac >= 0.2
    && metalElements.length > 0;

  if (isBoraneCage) {
    if (metalAtomCount === 0) return 0.05;
    return Math.max(0.05, Math.min(0.25, metalFrac * 0.8));
  }

  const nonHNonMetalFrac = elements.filter(e => nonmetals.includes(e) && e !== "H")
    .reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
  const cCount = counts["C"] || 0;
  const cFrac = cCount / totalAtoms;
  const isOrganic = cFrac > 0.15 && hasH && hCount >= cCount;
  const isPureSuperhydride = hasH && hRatio >= 6 && nonHNonMetalFrac < 0.1 && !isOrganic && metalFrac >= 0.05;

  let cationEN = 0, cationWeight = 0;
  let anionEN = 0, anionWeight = 0;

  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    const en = data.paulingElectronegativity;

    if (nonmetals.includes(el)) {
      anionEN += en * frac;
      anionWeight += frac;
    } else {
      cationEN += en * frac;
      cationWeight += frac;
    }
  }

  if (elements.length === 1) {
    const data = getElementData(elements[0]);
    if (data) {
      if (nonmetals.includes(elements[0])) return 0.1;
      if (isTransitionMetal(elements[0]) || isRareEarth(elements[0]) || isActinide(elements[0])) return 0.92;
      if (data.paulingElectronegativity < 1.8) return 0.90;
      return 0.5;
    }
  }

  if (anionWeight > 0 && cationWeight > 0) {
    const avgCation = cationEN / cationWeight;
    const avgAnion = anionEN / anionWeight;
    const deltaEN = avgAnion - avgCation;

    const k = 3.0;
    const threshold = 1.4;
    let metallicity = 1.0 / (1.0 + Math.exp(k * (deltaEN - threshold)));

    const halogenFrac = halogens.reduce((s, h) => s + ((counts[h] || 0) / totalAtoms), 0);
    if (halogenFrac > 0.2) metallicity *= 0.3;

    const oCount = counts["O"] || 0;
    const oFrac = oCount / totalAtoms;
    if (oFrac > 0.5) metallicity *= 0.5;

    if (metalElements.some(e => isTransitionMetal(e)) && oFrac < 0.3 && elements.length <= 3 && bFrac < 0.3 && cFrac < 0.2 && metalFrac >= 0.25) {
      metallicity = Math.max(metallicity, 0.85);
    }

    if (isMetallicBoride) {
      metallicity = Math.max(metallicity, 0.80);
    }

    if (hasB && bFrac > 0.4 && !isMetallicBoride) {
      metallicity *= Math.max(0.1, 1.0 - bFrac);
    }

    if (isOrganic) {
      metallicity = Math.min(metallicity, 0.35);
    }

    if (metalFrac < 0.15 && elements.length > 2) {
      metallicity *= Math.max(0.2, metalFrac * 5);
    }

    if (isPureSuperhydride) {
      const metalPresence = metalAtomCount / totalAtoms;
      metallicity = Math.max(metallicity, 0.65 + metalPresence * 0.2);
    } else if (hasH && hRatio >= 2 && !isOrganic) {
      if (bFrac < 0.3 && metalFrac >= 0.15) {
        const metalPresence = metalAtomCount / totalAtoms;
        metallicity = Math.max(metallicity, 0.5 + metalPresence * 0.4);
      }
    }

    return Math.max(0.05, Math.min(1.0, metallicity));
  }

  if (metalFrac > 0.8) return 0.92;
  if (metalFrac > 0.5) return 0.6 + metalFrac * 0.3;
  return 0.3 + metalFrac * 0.4;
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

  let metallicity = estimateMetallicity(elements, counts, mpData);
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

  const isHEA = detectHighEntropyAlloy(formula);
  if (isHEA) {
    metallicity = Math.min(1.0, metallicity + 0.08);
    if (bandStructureType === "semiconductor/semimetal" && metallicity >= 0.4) {
      bandStructureType = "metallic";
    }
    fermiSurfaceTopology = "multi-band with entropy-stabilized metallic bonding";
    orbitalCharacter = `multi-principal d-orbital cocktail (${elements.filter(e => isTransitionMetal(e)).join("/")})`; 
  }

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
  const hBondType = hasH ? classifyHydrogenBonding(formula, hRatio >= 4 ? 150 : 0) : "none";
  const isHydrogenRich = hBondType === "metallic-network" || hBondType === "cage-clathrate";

  const thetaDAvg = getCompositionWeightedProperty(counts, "debyeTemperature");
  const avgMass = getAverageMass(counts);
  const isHEAPhonon = detectHighEntropyAlloy(formula);
  let debyeTemperature: number;
  if (isHEAPhonon && !isHydrogenRich) {
    let logSum = 0;
    let totalFrac = 0;
    for (const el of elements) {
      const data = getElementData(el);
      if (data && data.debyeTemperature && data.debyeTemperature > 0) {
        const frac = (counts[el] || 1) / totalAtoms;
        logSum += frac * Math.log(data.debyeTemperature);
        totalFrac += frac;
      }
    }
    debyeTemperature = totalFrac > 0 ? Math.exp(logSum / totalFrac) : 300;
  } else if (thetaDAvg !== null && thetaDAvg > 0) {
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
  formula?: string,
  pressureGpa: number = 0
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

      const hBondType = formula ? classifyHydrogenBonding(formula, pressureGpa) : "none";

      if (hCount > 0 && hRatio >= 4 && metal > 0.4) {
        if (hBondType === "metallic-network" || hBondType === "cage-clathrate") {
          const H_theta = hBondType === "metallic-network" ? 2000 : 1500;
          const H_eta = Math.min(3.0 + hRatio * 0.3, 5.0);
          const H_mass = 1.008;
          const pressureScale = pressureGpa >= 100 ? 1.0 : pressureGpa / 100;
          const boostFraction = hBondType === "metallic-network" ? 1.0 : 0.7;
          const lambda_H = (H_eta * LAMBDA_CONVERSION) / (H_mass * H_theta * H_theta) * (hCount / totalAtoms) * pressureScale * boostFraction;
          lambda += lambda_H;
        } else if (hBondType === "interstitial") {
          const H_theta = 1200;
          const H_eta = 1.5;
          const H_mass = 1.008;
          const lambda_H = (H_eta * LAMBDA_CONVERSION) / (H_mass * H_theta * H_theta) * (hCount / totalAtoms) * 0.3;
          lambda += lambda_H;
        }
      }

      if (metal > 0.4) {
        const lightEl = elements.filter(e => {
          const d = getElementData(e);
          return d && d.atomicMass < 15 && e !== "H";
        });
        if (lightEl.length > 0) {
          const lightFrac = lightEl.reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
          if (lightFrac > 0.3) {
            const lightBoost = 1.0 + lightFrac * 1.2 * (phononSpectrum.debyeTemperature / 500);
            lambda *= lightBoost;
          }
        }
      }
    } else {
      lambda = N_EF * 0.15 * (1 + phononSpectrum.anharmonicityIndex * 0.5);
    }

    if (electronicStructure.fermiSurfaceTopology.includes("nesting")) lambda *= 1.15;

    const nonHNonMetFrac = elements.filter(e => {
      const nm = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"];
      return nm.includes(e) && e !== "H";
    }).reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
    const cFracLambda = (counts["C"] || 0) / totalAtoms;
    const isOrganicLambda = cFracLambda > 0.15 && hCount > 0 && hCount >= (counts["C"] || 0);
    const isPureSuperhydrideLambda = hRatio >= 6 && nonHNonMetFrac < 0.1 && !isOrganicLambda;

    const isHEALambda = formula ? detectHighEntropyAlloy(formula) : false;

    if (isPureSuperhydrideLambda) {
    } else if (isHEALambda && metal > 0.4) {
      const massDisorderBoost = 1.0 + elements.length * 0.03;
      lambda *= massDisorderBoost;
      if (corr > 0.7) lambda *= (1.0 - (corr - 0.7) * 0.3);
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

  lambda = Math.max(0.05, lambda);

  const matClass = formula ? classifyMaterialForLambda(formula, pressureGpa) : "other";
  const classLambdaCap = getLambdaCapForClass(matClass);

  if (activeConstraintMode.allowBeyondEmpirical) {
    if (lambda > classLambdaCap) {
      lambda = classLambdaCap + (lambda - classLambdaCap) * 0.15;
    }
    lambda = Math.min(classLambdaCap * 1.2, lambda);
  } else {
    lambda = Math.min(classLambdaCap, lambda);
  }

  if (lambda > 2.0 && matClass !== "superhydride" && matClass !== "hydride-high-p") {
    const instabilityDamp = 1.0 - (lambda - 2.0) * 0.15;
    lambda *= Math.max(0.5, instabilityDamp);
  }

  const omegaLogRange = getOmegaLogRangeForClass(matClass);
  let clampedOmegaLog = omega_log;
  const omegaLogK = omega_log * 1.44;
  if (omegaLogK < omegaLogRange[0]) clampedOmegaLog = omegaLogRange[0] / 1.44;
  if (omegaLogK > omegaLogRange[1]) clampedOmegaLog = omegaLogRange[1] / 1.44;

  const formulaCounts = formula ? parseFormulaCounts(formula) : {};
  const formulaElements = formula ? parseFormulaElements(formula) : [];
  const avgEN = formula ? (getCompositionWeightedProperty(formulaCounts, "paulingElectronegativity") || 1.8) : 1.8;
  let mu_bare = 0.1 + avgEN * 0.02;

  if (formulaElements.length >= 2) {
    const enValues = formulaElements.map(el => getElementData(el)?.paulingElectronegativity ?? 1.8);
    const enSpread = Math.max(...enValues) - Math.min(...enValues);
    if (enSpread > 1.5) mu_bare += 0.02 * (enSpread - 1.5);
  }
  if (formulaElements.some(e => isTransitionMetal(e) && hasDOrFElectrons(e))) {
    mu_bare += 0.02;
  }

  const thetaD = phononSpectrum.debyeTemperature;
  const E_F_eV = 5.0 + N_EF * 0.5;
  const omega_D_eV = thetaD * 8.617e-5;
  const logRatio = Math.log(Math.max(E_F_eV / Math.max(omega_D_eV, 0.001), 1.1));
  const muStar = mu_bare / (1 + mu_bare * logRatio);
  const muStarClamped = Math.max(0.10, Math.min(0.20, muStar));

  const isStrongCoupling = lambda > 1.5;

  let dominantPhononBranch = "acoustic";
  if (phononSpectrum.maxPhononFrequency > 2000) dominantPhononBranch = "high-frequency optical (H vibrations)";
  else if (phononSpectrum.softModePresent) dominantPhononBranch = "soft optical mode";
  else if (lambda > 1.0) dominantPhononBranch = "low-energy optical";

  return {
    lambda: Number(lambda.toFixed(3)),
    omegaLog: Math.round(clampedOmegaLog),
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

  let uncertaintyFrac = 0.15;
  if (tc > 200 && lambda < 2.0) {
    uncertaintyFrac = 0.5;
  } else if (tc > 150 && lambda < 1.5) {
    uncertaintyFrac = 0.4;
  }

  const uncertainty = tc * uncertaintyFrac;
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

export interface PairingMechanismResult {
  mechanism: string;
  tcEstimate: number;
  confidence: number;
  description: string;
}

export interface UnifiedPairingResult {
  dominant: PairingMechanismResult;
  all: PairingMechanismResult[];
  enhancedTc: number;
  uncertaintyFromMechanism: number;
}

function estimateSpinFluctuationTc(
  formula: string,
  electronic: ElectronicStructure,
  competingPhases: CompetingPhase[]
): PairingMechanismResult {
  const elements = parseFormulaElements(formula);
  const N_EF = electronic.densityOfStatesAtFermi;
  const corr = electronic.correlationStrength;

  let stonerMax = 0;
  for (const el of elements) {
    const I = getStonerParameter(el);
    if (I !== null) stonerMax = Math.max(stonerMax, I * N_EF);
  }

  const nearQCP = stonerMax > 0.7 && stonerMax < 1.2;
  const hasAFM = competingPhases.some(p => p.phaseName.includes("Antiferromagnetic"));

  if (!nearQCP && !hasAFM && corr < 0.5) {
    return { mechanism: "spin-fluctuation", tcEstimate: 0, confidence: 0.1, description: "No magnetic proximity" };
  }

  const T_sf = 200 * (1 - Math.abs(stonerMax - 1.0));
  const V_sf = corr * 0.8;
  const exponent = V_sf * N_EF;

  let tc_sf = 0;
  if (exponent > 0.1) {
    tc_sf = T_sf * Math.exp(-1 / exponent);
  }

  if (nearQCP) tc_sf *= 2.0;
  if (hasAFM) tc_sf *= 1.5;

  const isCuprate = elements.includes("Cu") && elements.includes("O");
  const isIronBased = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  if (isCuprate) tc_sf = Math.max(tc_sf, 50 + corr * 100);
  if (isIronBased) tc_sf = Math.max(tc_sf, 20 + corr * 60);

  tc_sf = Math.max(0, Math.round(tc_sf));
  const confidence = nearQCP ? 0.5 : (hasAFM ? 0.4 : 0.2);

  return {
    mechanism: "spin-fluctuation",
    tcEstimate: tc_sf,
    confidence,
    description: nearQCP ? "Near magnetic quantum critical point" : (hasAFM ? "AFM-proximity-mediated pairing" : "Weak spin-fluctuation contribution"),
  };
}

function estimateExcitonicPairingTc(
  formula: string,
  electronic: ElectronicStructure
): PairingMechanismResult {
  const corr = electronic.correlationStrength;
  const metal = electronic.metallicity;

  const isMixedDim = electronic.fermiSurfaceTopology.includes("2D") && electronic.orbitalCharacter.includes("hybridized");
  const nearMIT = metal > 0.3 && metal < 0.6;

  if (!isMixedDim && !nearMIT) {
    return { mechanism: "excitonic", tcEstimate: 0, confidence: 0.05, description: "No excitonic conditions" };
  }

  let tc_exc = 0;
  if (isMixedDim && nearMIT) {
    const excStrength = (0.6 - metal) * corr * 2;
    tc_exc = Math.round(50 * excStrength);
  } else if (nearMIT) {
    tc_exc = Math.round(20 * (0.6 - metal) * 3);
  }

  return {
    mechanism: "excitonic",
    tcEstimate: Math.max(0, tc_exc),
    confidence: 0.2,
    description: isMixedDim ? "Mixed-dimensional excitonic coupling" : "Near metal-insulator boundary",
  };
}

function estimatePlasmonicPairingTc(
  formula: string,
  electronic: ElectronicStructure
): PairingMechanismResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const N_EF = electronic.densityOfStatesAtFermi;

  const totalAtoms = getTotalAtoms(counts);
  let totalVE = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
  }
  const vec = totalVE / totalAtoms;
  const carrierDensity = vec * electronic.metallicity;

  const lowCarrier = carrierDensity < 3 && N_EF > 2;
  if (!lowCarrier) {
    return { mechanism: "plasmonic", tcEstimate: 0, confidence: 0.05, description: "No plasmonic conditions" };
  }

  const plasmaStrength = N_EF / Math.max(carrierDensity, 0.5);
  const tc_pl = Math.round(Math.min(100, plasmaStrength * 10));

  return {
    mechanism: "plasmonic",
    tcEstimate: Math.max(0, tc_pl),
    confidence: 0.15,
    description: "Low carrier density with high DOS — plasmon-mediated",
  };
}

function estimateFlatBandTc(
  formula: string,
  electronic: ElectronicStructure,
  coupling: ElectronPhononCoupling
): PairingMechanismResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const N_EF = electronic.densityOfStatesAtFermi;

  let wAvg = 0;
  const totalAtoms = getTotalAtoms(counts);
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    wAvg += estimateBandwidthW(el) * frac;
  }

  const isFlatBand = wAvg < 1.5 && N_EF > 3.0;

  if (!isFlatBand) {
    return { mechanism: "flat-band", tcEstimate: 0, confidence: 0.05, description: "No flat-band conditions" };
  }

  const lambda = coupling.lambda;
  const tc_fb = Math.round(wAvg * 11604 * Math.sqrt(Math.max(0.1, lambda)) * 0.01);

  const isKagome = elements.length >= 2 && electronic.fermiSurfaceTopology.includes("nesting");

  return {
    mechanism: "flat-band",
    tcEstimate: Math.max(0, Math.min(400, tc_fb * (isKagome ? 1.5 : 1.0))),
    confidence: isKagome ? 0.35 : 0.25,
    description: isKagome ? "Kagome-type flat band with geometric frustration" : `Narrow bandwidth (W=${wAvg.toFixed(1)}eV) with high DOS`,
  };
}

export function runUnifiedPairingAnalysis(
  formula: string,
  electronic: ElectronicStructure,
  coupling: ElectronPhononCoupling,
  eliashberg: EliashbergResult,
  competingPhases: CompetingPhase[]
): UnifiedPairingResult {
  const bcs: PairingMechanismResult = {
    mechanism: "phonon-mediated BCS",
    tcEstimate: eliashberg.predictedTc,
    confidence: coupling.lambda > 0.3 ? 0.7 : 0.3,
    description: `BCS: lambda=${coupling.lambda.toFixed(2)}, omega_log=${coupling.omegaLog}cm-1`,
  };

  const spinFluc = estimateSpinFluctuationTc(formula, electronic, competingPhases);
  const excitonic = estimateExcitonicPairingTc(formula, electronic);
  const plasmonic = estimatePlasmonicPairingTc(formula, electronic);
  const flatBand = estimateFlatBandTc(formula, electronic, coupling);

  const all = [bcs, spinFluc, excitonic, plasmonic, flatBand];

  const weightedAll = all.map(m => ({
    ...m,
    effectiveTc: m.tcEstimate * m.confidence,
  }));

  weightedAll.sort((a, b) => b.effectiveTc - a.effectiveTc);
  const dominant = weightedAll[0];

  const activeCount = all.filter(m => m.tcEstimate > 0).length;
  let enhancedTc = dominant.tcEstimate;
  const secondary = weightedAll[1];
  if (secondary && secondary.tcEstimate > 0 && secondary.mechanism !== dominant.mechanism) {
    enhancedTc = Math.round(enhancedTc + secondary.tcEstimate * 0.15);
  }

  const uncertaintyFromMechanism = activeCount > 2 ? 0.4 : (dominant.confidence > 0.5 ? 0.2 : 0.35);

  return {
    dominant: { mechanism: dominant.mechanism, tcEstimate: dominant.tcEstimate, confidence: dominant.confidence, description: dominant.description },
    all,
    enhancedTc,
    uncertaintyFromMechanism,
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

export interface InstabilityProximity {
  magneticQCP: number;
  structuralBoundary: number;
  metalInsulatorTransition: number;
  cdwInstability: number;
  softPhononCollapse: number;
  overallProximity: number;
  nearestBoundary: string;
}

export function computeInstabilityProximity(
  formula: string,
  electronic: ElectronicStructure,
  phononSpectrum: PhononSpectrum,
  competingPhases: CompetingPhase[]
): InstabilityProximity {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  let magneticQCP = 0;
  for (const el of elements) {
    const I = getStonerParameter(el);
    if (I !== null) {
      const stonerProduct = I * electronic.densityOfStatesAtFermi;
      if (stonerProduct > 0.7 && stonerProduct < 1.2) {
        const proximity = 1.0 - Math.abs(stonerProduct - 1.0) * 5;
        magneticQCP = Math.max(magneticQCP, Math.max(0, proximity));
      }
    }
  }
  const hasAFM = competingPhases.some(p => p.type === "magnetism" && p.phaseName.includes("Antiferromagnetic"));
  if (hasAFM) magneticQCP = Math.max(magneticQCP, 0.5);

  let structuralBoundary = 0;
  const structPhases = competingPhases.filter(p => p.type === "structural");
  for (const sp of structPhases) {
    const match = sp.phaseName.match(/t=([\d.]+)/);
    if (match) {
      const tf = parseFloat(match[1]);
      if (tf >= 0.82 && tf <= 0.88) structuralBoundary = Math.max(structuralBoundary, 0.8);
      else if (tf >= 1.02 && tf <= 1.08) structuralBoundary = Math.max(structuralBoundary, 0.7);
      else if (tf < 0.82 || tf > 1.08) structuralBoundary = Math.max(structuralBoundary, 0.3);
    }
  }

  const metallicity = electronic.metallicity;
  let metalInsulatorTransition = 0;
  if (metallicity > 0.3 && metallicity < 0.5) {
    metalInsulatorTransition = 1.0 - Math.abs(metallicity - 0.4) * 10;
    metalInsulatorTransition = Math.max(0, metalInsulatorTransition);
  }

  let cdwInstability = 0;
  const cdwPhases = competingPhases.filter(p => p.type === "CDW");
  for (const cdw of cdwPhases) {
    cdwInstability = Math.max(cdwInstability, cdw.strength);
  }
  if (electronic.fermiSurfaceTopology.includes("nesting")) {
    cdwInstability = Math.max(cdwInstability, 0.4);
  }

  let softPhononCollapse = 0;
  if (phononSpectrum.softModePresent) {
    softPhononCollapse = 0.6;
  }
  if (phononSpectrum.logAverageFrequency < 100) {
    softPhononCollapse = Math.max(softPhononCollapse, 0.8);
  }
  if (phononSpectrum.anharmonicityIndex > 0.6) {
    softPhononCollapse = Math.max(softPhononCollapse, 0.5);
  }

  const scores = [
    { name: "Magnetic QCP", val: magneticQCP },
    { name: "Structural boundary", val: structuralBoundary },
    { name: "Metal-insulator transition", val: metalInsulatorTransition },
    { name: "CDW instability", val: cdwInstability },
    { name: "Soft phonon collapse", val: softPhononCollapse },
  ];

  const overallProximity = Math.max(...scores.map(s => s.val));
  const nearest = scores.reduce((a, b) => b.val > a.val ? b : a);

  return {
    magneticQCP,
    structuralBoundary,
    metalInsulatorTransition,
    cdwInstability,
    softPhononCollapse,
    overallProximity,
    nearestBoundary: nearest.name,
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
  pairingAnalysis: UnifiedPairingResult;
  instabilityProximity: InstabilityProximity;
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
  let dftData: DFTResolvedFeatures | null = null;
  try {
    const mpClient = await import("./materials-project-client");
    if (mpClient.isApiAvailable()) {
      const mpData = await mpClient.fetchAllData(formula);
      mpSummary = mpData.summary;
      mpElasticity = mpData.elasticity;
    }
  } catch {}

  try {
    const dftResolver = await import("./dft-feature-resolver");
    dftData = await dftResolver.resolveDFTFeatures(formula);
    if (dftData.dftCoverage > 0) {
      const desc = dftResolver.describeDFTSources(dftData);
      emit("log", {
        phase: "phase-10",
        event: "DFT data resolved",
        detail: `${formula}: ${desc} (coverage=${(dftData.dftCoverage * 100).toFixed(0)}%)`,
        dataSource: "DFT Resolver",
      });
    }
  } catch {}

  const correlation = assessCorrelationStrength(formula);
  const electronicStructure = computeElectronicStructure(formula, candidate.crystalStructure, mpSummary);

  if (dftData) {
    if (dftData.dosAtFermi.source !== "analytical" && dftData.dosAtFermi.value != null && dftData.dosAtFermi.value > 0) {
      electronicStructure.densityOfStatesAtFermi = Number(dftData.dosAtFermi.value.toFixed(3));
    }
    if (dftData.isMetallic.source !== "analytical") {
      electronicStructure.metallicity = dftData.isMetallic.value
        ? Math.max(electronicStructure.metallicity, 0.85)
        : Math.min(electronicStructure.metallicity, 0.15);
    }
  }

  const phononSpectrum = computePhononSpectrum(formula, electronicStructure, mpElasticity, mpSummary);

  if (dftData) {
    if (dftData.debyeTemp.source !== "analytical" && dftData.debyeTemp.value > 0) {
      const analytical = phononSpectrum.debyeTemperature;
      phononSpectrum.debyeTemperature = dftData.debyeTemp.value;
      if (Math.abs(analytical - dftData.debyeTemp.value) > 20) {
        emit("log", {
          phase: "phase-10",
          event: "DFT override",
          detail: `Using DFT-computed Debye temp ${dftData.debyeTemp.value}K for ${formula} (vs analytical ${analytical}K)`,
          dataSource: "DFT Resolver",
        });
      }
    }
    if (dftData.phononFreqMax.value != null && dftData.phononFreqMax.source !== "analytical") {
      phononSpectrum.maxPhononFrequency = dftData.phononFreqMax.value;
    }
  }

  const candidatePressure = candidate.pressureGpa ?? 0;
  const coupling = computeElectronPhononCoupling(electronicStructure, phononSpectrum, formula, candidatePressure);

  let eliashberg: EliashbergResult;
  eliashberg = predictTcEliashberg(coupling);

  const competingPhases = evaluateCompetingPhases(formula, electronicStructure, mpSummary);
  const hasMottPhase = competingPhases.some(p => p.type === "Mott");
  const isMottInsulator = hasMottPhase && correlation.ratio > 0.7;
  const isNonMetallic = electronicStructure.metallicity < 0.4;

  const formulaEls = parseFormulaElements(formula);
  const hasDWaveSymmetry = formulaEls.includes("Cu") && formulaEls.includes("O") && formulaEls.length >= 3;
  const hasLayeredStructure = electronicStructure.fermiSurfaceTopology.includes("2D") ||
    (electronicStructure.orbitalCharacter.includes("hybridized") && formula.includes("O"));
  const isHEAPhysics = detectHighEntropyAlloy(formula);

  if (isNonMetallic) {
    const metalFactor = Math.max(0.02, electronicStructure.metallicity);
    eliashberg.predictedTc = eliashberg.predictedTc * metalFactor;
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2)];
  } else if (isMottInsulator) {
    eliashberg.predictedTc = eliashberg.predictedTc * 0.05;
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 3)];
  } else if (correlation.ratio > 0.7) {
    const corrFactor = hasDWaveSymmetry ? 0.5 : hasLayeredStructure ? 0.4 : 0.3;
    eliashberg.predictedTc = eliashberg.predictedTc * corrFactor;
    eliashberg.confidenceBand = [
      Math.round(eliashberg.predictedTc * 0.3),
      Math.round(eliashberg.predictedTc * (hasDWaveSymmetry ? 3.0 : 2.5)),
    ];
  } else if (correlation.ratio > 0.5) {
    const corrFactor = hasLayeredStructure ? 0.85 : hasDWaveSymmetry ? 0.8 : 0.7;
    eliashberg.predictedTc = eliashberg.predictedTc * corrFactor;
    eliashberg.confidenceBand = [
      Math.round(eliashberg.predictedTc * 0.6),
      Math.round(eliashberg.predictedTc * 1.5),
    ];
  }

  let dimensionality = candidate.dimensionality || "3D";
  if (!candidate.dimensionality) {
    if (isHEAPhysics) dimensionality = "3D-HEA";
    else if (electronicStructure.fermiSurfaceTopology.includes("2D")) dimensionality = "quasi-2D";
    else if (electronicStructure.orbitalCharacter.includes("hybridized") && formula.includes("O")) dimensionality = "layered";
    else dimensionality = "3D";
  }

  const pairingAnalysis = runUnifiedPairingAnalysis(formula, electronicStructure, coupling, eliashberg, competingPhases);

  if (pairingAnalysis.dominant.mechanism !== "phonon-mediated BCS" && pairingAnalysis.dominant.tcEstimate > eliashberg.predictedTc) {
    const unconvTc = pairingAnalysis.enhancedTc;
    const blendWeight = pairingAnalysis.dominant.confidence;
    eliashberg.predictedTc = Math.round(
      eliashberg.predictedTc * (1 - blendWeight) + unconvTc * blendWeight
    );
    eliashberg.confidenceBand = [
      Math.round(eliashberg.predictedTc * 0.5),
      Math.round(eliashberg.predictedTc * 2.0),
    ];
  }

  const criticalFields = computeCriticalFields(eliashberg.predictedTc, coupling, dimensionality);

  const suppressingPhases = competingPhases.filter(p => p.suppressesSC);
  let uncertaintyEstimate = 0.3;
  if (correlation.ratio > 0.7) uncertaintyEstimate += 0.2;
  if (phononSpectrum.hasImaginaryModes) uncertaintyEstimate += 0.15;
  if (suppressingPhases.length > 0) uncertaintyEstimate += 0.1;
  if (phononSpectrum.anharmonicityIndex > 0.5) uncertaintyEstimate += 0.1;
  if (!mpSummary) uncertaintyEstimate += 0.05;
  if (dftData && dftData.dftCoverage > 0.3) uncertaintyEstimate -= dftData.dftCoverage * 0.15;
  uncertaintyEstimate = Math.max(uncertaintyEstimate, pairingAnalysis.uncertaintyFromMechanism);
  uncertaintyEstimate = Math.max(0.05, Math.min(0.95, uncertaintyEstimate));

  const instabilityProximity = computeInstabilityProximity(formula, electronicStructure, phononSpectrum, competingPhases);

  emit("log", {
    phase: "phase-10",
    event: "Instability proximity computed",
    detail: `${formula}: nearest=${instabilityProximity.nearestBoundary} (${instabilityProximity.overallProximity.toFixed(2)}), QCP=${instabilityProximity.magneticQCP.toFixed(2)}, CDW=${instabilityProximity.cdwInstability.toFixed(2)}, MIT=${instabilityProximity.metalInsulatorTransition.toFixed(2)}`,
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
    pairingAnalysis,
    instabilityProximity,
  };
}
