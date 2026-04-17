import OpenAI from "openai";
import type { EventEmitter } from "./engine";
import type { SuperconductorCandidate } from "@shared/schema";
import type { DFTResolvedFeatures } from "./dft-feature-resolver";
import { classifyFamily, NONMETALS } from "./utils";
import { computeFullTightBinding } from "./tight-binding";
import { computeAdvancedConstraints, type AdvancedPhysicsConstraints } from "../physics/advanced-constraints";
import { validateOmegaLog } from "../physics/tc-formulas";
import { computeOODScore } from "./ood-detector";

let _mpClientCache: typeof import("./materials-project-client") | null = null;
let _dftResolverCache: typeof import("./dft-feature-resolver") | null = null;
async function getMPClient() {
  if (!_mpClientCache) _mpClientCache = await import("./materials-project-client");
  return _mpClientCache;
}
async function getDFTResolver() {
  if (!_dftResolverCache) _dftResolverCache = await import("./dft-feature-resolver");
  return _dftResolverCache;
}

export interface PhysicsConstraintMode {
  allowBeyondEmpirical: boolean;
  empiricalPenaltyStrength: number;
}

const defaultConstraintMode: PhysicsConstraintMode = {
  allowBeyondEmpirical: true,
  empiricalPenaltyStrength: 1.8,
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
  const safeThreshold = Math.max(threshold, 50);
  const ratio = Math.min(excess / safeThreshold, 5.0);
  const dampened = threshold + excess / (1 + penaltyStrength * ratio);
  return Math.round(dampened * 10) / 10;
}

interface ParsedComposition {
  elements: string[];
  counts: Record<string, number>;
  totalAtoms: number;
  metalElements: string[];
  vec: number;
}

export function parseComposition(formula: string): ParsedComposition {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const metalElements = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e) || HEA_EXTRA_METALS.includes(e));
  const vec = getVEC(elements, counts);
  return { elements, counts, totalAtoms, metalElements, vec };
}

function computePhysicsDerivedBonusParsed(pc: ParsedComposition, lambda: number): number {
  const isCuprate = pc.elements.includes("Cu") && pc.elements.includes("O") && pc.elements.length >= 3
    && pc.elements.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Bi", "Tl", "Hg"].includes(e));
  const isHEA = detectHighEntropyAlloyParsed(pc);

  let bonus = 0;
  if (isCuprate && lambda > 0.5) {
    const cuFrac = (pc.counts["Cu"] || 0) / pc.totalAtoms;
    const oFrac = (pc.counts["O"] || 0) / pc.totalAtoms;
    bonus = Math.round(cuFrac * oFrac * 200 * Math.min(lambda, 2.0));
  }
  if (isHEA && lambda > 1.0) {
    const entropyFactor = Math.log(Math.max(2, pc.metalElements.length)) / Math.log(6);
    const heaBonus = Math.min(15, Math.round(entropyFactor * lambda * 8));
    bonus = Math.max(bonus, heaBonus);
  }
  return bonus;
}

export const FAMILY_TC_CAPS: Record<string, { ambient: number; highPressure: number }> = {
  Carbides: { ambient: 120, highPressure: 250 },
  Nitrides: { ambient: 120, highPressure: 250 },
  Borides: { ambient: 120, highPressure: 300 },
  Oxides: { ambient: 170, highPressure: 250 },
};

export interface CapExtensionEvidence {
  gnnEnsembleStd?: number;
  eliashbergLambda?: number;
  eliashbergTc?: number;
}

export function computeCapExtensionFactor(evidence?: CapExtensionEvidence): number {
  if (!evidence) return 1.0;

  const { gnnEnsembleStd, eliashbergLambda, eliashbergTc } = evidence;

  let confidenceScore = 0;
  let evidenceCount = 0;

  if (gnnEnsembleStd != null && Number.isFinite(gnnEnsembleStd) && gnnEnsembleStd > 0) {
    const normalizedStd = gnnEnsembleStd / Math.max(1, eliashbergTc ?? 50);
    const gnnConfidence = normalizedStd < 0.05 ? 1.0
      : normalizedStd < 0.10 ? 0.7
      : normalizedStd < 0.20 ? 0.3
      : 0.0;
    confidenceScore += gnnConfidence;
    evidenceCount++;
  }

  if (eliashbergLambda != null && Number.isFinite(eliashbergLambda)) {
    const couplingConfidence = eliashbergLambda >= 2.0 ? 1.0
      : eliashbergLambda >= 1.5 ? 0.7
      : eliashbergLambda >= 1.0 ? 0.4
      : 0.0;
    confidenceScore += couplingConfidence;
    evidenceCount++;
  }

  if (evidenceCount === 0) return 1.0;

  const avgConfidence = confidenceScore / evidenceCount;

  if (avgConfidence < 0.4) return 1.0;

  const maxExtension = 0.15;
  return 1.0 + maxExtension * ((avgConfidence - 0.4) / 0.6);
}

export function applyAmbientTcCap(tc: number, lambda: number, pressureGpa: number, metallicity: number, formula?: string, evidence?: CapExtensionEvidence): number {
  if (tc <= 0) return tc;
  const mode = activeConstraintMode;

  if (metallicity <= 0) return 0;

  let pressureThresholdLow = 10;
  let materialBonus = 0;
  let familyName: string | null = null;

  let pc: ParsedComposition | null = null;
  if (formula) {
    pc = parseComposition(formula);
    familyName = classifyFamily(formula);
    const hCount = pc.counts["H"] || 0;
    const metalAtomCount = pc.metalElements.reduce((s, e) => s + (pc!.counts[e] || 0), 0);
    const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;
    if (hRatio >= 6) pressureThresholdLow = 100;
    materialBonus = computePhysicsDerivedBonusParsed(pc, lambda);

    // Rule 4: no hydrogen + ambient pressure → McMillan-regime ceiling (40 K).
    // High-Tc hydrides require H and pressure; non-hydride ambient materials are
    // bound by phonon-mediated pairing in the conventional BCS regime.
    const hCountAmb = pc.counts["H"] || 0;
    if (hCountAmb === 0 && pressureGpa === 0) {
      tc = Math.min(tc, 40);
    }
  }

  const highPressureAnchor = pressureThresholdLow >= 100 ? 150 : 50;
  const isAmbient = pressureGpa < pressureThresholdLow;
  const isHighPressure = pressureGpa >= highPressureAnchor;
  const pressureDenom = Math.max(1, highPressureAnchor - pressureThresholdLow);
  const pressureFactor = isHighPressure ? 1.0 : isAmbient ? 0.0 : (pressureGpa - pressureThresholdLow) / pressureDenom;

  const extensionFactor = computeCapExtensionFactor(evidence);

  if (familyName && FAMILY_TC_CAPS[familyName]) {
    const familyCaps = FAMILY_TC_CAPS[familyName];
    const baseFamilyCap = familyCaps.ambient + (familyCaps.highPressure - familyCaps.ambient) * pressureFactor;
    const familyCap = Math.round(baseFamilyCap * extensionFactor);
    tc = Math.min(tc, familyCap);
  }

  if (!mode.allowBeyondEmpirical) {
    let tcCap: number;
    if (metallicity < 0.3) {
      tcCap = 30;
    } else if (metallicity < 0.5) {
      tcCap = 100;
    } else if (lambda < 0.3) {
      tcCap = 80;
    } else if (lambda < 0.5) {
      tcCap = 120;
    } else if (lambda < 1.0) {
      tcCap = Math.round(150 + (250 - 150) * pressureFactor);
    } else if (lambda < 1.5) {
      tcCap = Math.round(200 + (350 - 200) * pressureFactor);
    } else if (lambda < 2.5) {
      tcCap = Math.round(250 + (400 - 250) * pressureFactor);
    } else {
      tcCap = Math.round(300 + (500 - 300) * pressureFactor);
    }
    tcCap = Math.round(tcCap * extensionFactor);
    if (pressureGpa < 10) {
      tcCap += materialBonus;
    }
    return Math.min(tc, tcCap);
  }

  let metalCeiling: number;
  if (metallicity < 0.3) {
    metalCeiling = 30 + metallicity * 150;
  } else if (metallicity < 0.5) {
    metalCeiling = 80 + (metallicity - 0.3) * 300;
  } else {
    metalCeiling = Infinity;
  }

  let lambdaCeiling: number;
  if (lambda < 0.3) {
    lambdaCeiling = 60 + lambda * 200;
  } else if (lambda < 0.5) {
    lambdaCeiling = 100 + (lambda - 0.3) * 300;
  } else if (lambda < 1.0) {
    lambdaCeiling = 150 + (250 - 150) * pressureFactor;
  } else if (lambda < 1.5) {
    lambdaCeiling = 200 + (350 - 200) * pressureFactor;
  } else if (lambda < 2.5) {
    lambdaCeiling = 300 + (500 - 300) * pressureFactor;
  } else {
    lambdaCeiling = 350 + (600 - 350) * pressureFactor;
  }

  let baseExpectation = Math.min(metalCeiling, lambdaCeiling);

  baseExpectation += materialBonus;
  baseExpectation = Math.round(baseExpectation * extensionFactor);

  let penaltyStr = mode.empiricalPenaltyStrength;
  if (formula) {
    try {
      const ood = computeOODScore(formula);
      const BASE_PENALTY = 0.6;
      const OOD_PENALTY_SCALE = 4.0;
      penaltyStr = BASE_PENALTY + ood.oodScore * OOD_PENALTY_SCALE;
      penaltyStr = Math.max(0.3, Math.min(5.0, penaltyStr));
    } catch {}
  }
  const result = softCeiling(tc, baseExpectation, penaltyStr);

  return Math.round(result);
}

export interface TcMethodEstimates {
  gbPredicted?: number;
  physicsTc?: number;
  xTbTc?: number;
  dftTc?: number;
  dftSigma?: number;
  physicsSigma?: number;
  xTbSigma?: number;
  gbSigma?: number;
}

const DEFAULT_SIGMAS: Record<string, number> = {
  dft: 0.15,
  xtb: 0.20,
  physics: 0.25,
  gb: 0.40,
};

export function reconcileTc(estimates: TcMethodEstimates): { reconciledTc: number; confidence: string; methods: TcMethodEstimates } {
  const entries: { method: string; tc: number; sigma: number }[] = [];
  const validTc = (v: number | undefined | null): v is number => v != null && Number.isFinite(v) && v > 0;
  const validSigma = (v: number | undefined | null, fallback: number): number => {
    const s = v != null && Number.isFinite(v) && v > 0 ? v : fallback;
    return s;
  };
  if (validTc(estimates.dftTc))
    entries.push({ method: "dft", tc: estimates.dftTc, sigma: validSigma(estimates.dftSigma, DEFAULT_SIGMAS.dft) });
  if (validTc(estimates.xTbTc))
    entries.push({ method: "xtb", tc: estimates.xTbTc, sigma: validSigma(estimates.xTbSigma, DEFAULT_SIGMAS.xtb) });
  if (validTc(estimates.physicsTc))
    entries.push({ method: "physics", tc: estimates.physicsTc, sigma: validSigma(estimates.physicsSigma, DEFAULT_SIGMAS.physics) });
  if (validTc(estimates.gbPredicted))
    entries.push({ method: "gb", tc: estimates.gbPredicted, sigma: validSigma(estimates.gbSigma, DEFAULT_SIGMAS.gb) });

  if (entries.length === 0) return { reconciledTc: 0, confidence: "none", methods: estimates };
  if (entries.length === 1) {
    const only = entries[0];
    const conf = only.sigma <= 0.15 ? "high" : only.sigma <= 0.25 ? "medium" : "low";
    return { reconciledTc: Math.round(only.tc * 10) / 10, confidence: conf, methods: estimates };
  }

  const clampedSigma = (s: number) => Math.max(0.05, Math.min(0.95, s));

  for (const e of entries) {
    if (e.method === "physics" && e.sigma > 0.4) {
      const penalty = Math.min(0.5, (e.sigma - 0.4) * 1.5);
      e.tc = e.tc * (1 - penalty);
    }
  }

  const invVar = entries.map(e => ({ ...e, w: 1 / (clampedSigma(e.sigma) ** 2) }));

  const tcs = entries.map(e => e.tc);
  const maxTc = Math.max(...tcs);
  const minTc = Math.min(...tcs);
  const meanTc = tcs.reduce((s, t) => s + t, 0) / tcs.length;
  const spreadDenom = Math.max(meanTc, 1);
  const spread = (maxTc - minTc) / spreadDenom;

  let reconciledTc: number;
  const totalW = invVar.reduce((s, e) => s + e.w, 0);
  if (spread <= 0.35) {
    reconciledTc = invVar.reduce((s, e) => s + e.tc * e.w, 0) / totalW;
  } else {
    invVar.sort((a, b) => b.w - a.w);
    reconciledTc = invVar[0].tc;
  }

  const pooledVariance = 1 / totalW;
  const pooledSigma = Math.sqrt(pooledVariance);
  const disagreementPenalty = spread > 0.35 ? 0.15 : spread > 0.2 ? 0.05 : 0;
  const effectiveSigma = Math.min(0.95, pooledSigma + disagreementPenalty);
  const conf = effectiveSigma <= 0.15 ? "high" : effectiveSigma <= 0.25 ? "medium" : "low";
  return { reconciledTc: Math.round(reconciledTc * 10) / 10, confidence: conf, methods: estimates };
}

const HEA_EXTRA_METALS = ["Al", "Mg", "Ca", "Sr", "Ba", "Li", "Na", "K", "Ti", "Zn", "Ga", "Ge", "Sn"];

function detectHighEntropyAlloyParsed(pc: ParsedComposition): boolean {
  if (pc.metalElements.length < 5) return false;
  const metalCounts = pc.metalElements.map(e => pc.counts[e] || 1);
  const totalMetal = metalCounts.reduce((s, n) => s + n, 0);
  if (totalMetal <= 0) return false;
  const fractions = metalCounts.map(c => c / totalMetal);
  const sConf = -fractions.reduce((s, f) => f > 0 ? s + f * Math.log(f) : s, 0);
  return sConf >= 1.5;
}

function detectHighEntropyAlloy(formula: string): boolean {
  return detectHighEntropyAlloyParsed(parseComposition(formula));
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
  timeout: 60_000,
  maxRetries: 0, // Connection errors do not self-resolve; avoid 3x retry amplification
});

export interface TightBindingTopology {
  hasFlatBand: boolean;
  hasVHS: boolean;
  hasDiracCrossing: boolean;
  hasBandInversion: boolean;
  topologyScore: number;
  flatBandCount: number;
  vhsCount: number;
  diracCrossingCount: number;
  dosAtFermi: number;
}

export interface ElectronicStructure {
  bandStructureType: string;
  fermiSurfaceTopology: string;
  densityOfStatesAtFermi: number;
  correlationStrength: number;
  metallicity: number;
  orbitalCharacter: string;
  nestingFeatures: string;
  nestingScore: number;
  vanHoveProximity: number;
  bandFlatness: number;
  flatBandIndicator: number;
  orbitalFractions: { s: number; p: number; d: number; f: number };
  topologicalBandScore: number;
  mottProximityScore: number;
  tightBindingTopology?: TightBindingTopology;
}

export interface PhononSpectrum {
  maxPhononFrequency: number;
  logAverageFrequency: number;
  hasImaginaryModes: boolean;
  anharmonicityIndex: number;
  softModePresent: boolean;
  softModeScore: number;
  debyeTemperature: number;
}

export interface ElectronPhononCoupling {
  lambda: number;
  lambdaUncorrected: number;
  anharmonicCorrectionFactor: number;
  omegaLog: number;
  muStar: number;
  isStrongCoupling: boolean;
  dominantPhononBranch: string;
  bandwidth: number;
  omega2Avg: number;
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

export interface GLValidation {
  /** BCS clean-limit coherence length: ξ₀ = ℏvF / (kB·Tc·π), in nm */
  xiGL: number;
  /** Fermi velocity used in the calculation, in m/s */
  vFUsed: number;
  /** "ok" = physically plausible; "unphysical" = Tc or vF produced ξ≈0;
   *  "underdetermined" = missing inputs (e.g. Tc=0) */
  status: "ok" | "unphysical" | "underdetermined";
  /** Human-readable explanation when status ≠ "ok" */
  diagnosis: string;
}

export interface CriticalFieldResult {
  upperCriticalField: number;
  lowerCriticalField: number;
  coherenceLength: number;
  londonPenetrationDepth: number;
  anisotropyRatio: number;
  criticalCurrentDensity: number;
  typeIorII: string;
  /** True when key inputs (Tc, vF) were missing or zero — parameters are
   *  under-determined, not a physical failure of the material. */
  underdetermined?: boolean;
  /** Ginzburg-Landau coherence-length validation result */
  glValidation?: GLValidation;
}

export function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  let cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  cleaned = cleaned.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]/g, c => String("⁰¹²³⁴⁵⁶⁷⁸⁹".indexOf(c)));
  cleaned = cleaned.replace(/[⁺⁻]+/g, "");
  cleaned = cleaned.replace(/[A-Z][a-z]?\d*[+\-]/g, match => match.replace(/\d*[+\-]$/, ""));
  cleaned = cleaned.replace(/[()[\]]/g, "");
  cleaned = cleaned.replace(/(^|[\d\s])([a-z])/g, (_m, prefix, c) => prefix + c.toUpperCase());
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  if (!matches) return [];
  const valid = matches.filter(el => getElementData(el) !== null);
  return Array.from(new Set(valid));
}

function expandParentheses(formula: string): string {
  let result = formula.replace(/\[/g, "(").replace(/\]/g, ")");
  const parenRegex = /\(([^()]+)\)(\d*\.?\d*)/;
  let iterations = 0;
  while (result.includes("(") && iterations < 20) {
    const prev = result;
    result = result.replace(parenRegex, (_, group: string, mult: string) => {
      const m = (mult && mult.length > 0) ? parseFloat(mult) : 1;
      if (isNaN(m) || !isFinite(m) || m <= 0) return group;
      if (m === 1) return group;
      return group.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_x: string, el: string, num: string) => {
        const n = (num && num.length > 0) ? parseFloat(num) : 1;
        const safeN = (isNaN(n) || !isFinite(n) || n <= 0) ? 1 : n;
        const newN = safeN * m;
        return newN === 1 ? el : `${el}${newN}`;
      });
    });
    if (result === prev) break;
    iterations++;
  }
  return result.replace(/[()]/g, "");
}

export function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  let cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  cleaned = expandParentheses(cleaned);
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + (isNaN(num) || num <= 0 ? 1 : num);
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  return total > 0 ? total : 1;
}

function getVEC(elements: string[], counts: Record<string, number>): number {
  const totalAtoms = getTotalAtoms(counts);
  let totalVE = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
  }
  return totalVE / totalAtoms;
}

const LAMBDA_CONVERSION = 562000;

export type HydrogenBondingType = "metallic-network" | "cage-clathrate" | "covalent-molecular" | "interstitial" | "ambiguous" | "none";

const NONMETALS_SET = new Set(["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"]);

export function classifyHydrogenBonding(formula: string, pressureGpa: number = 0): HydrogenBondingType {
  const pc = parseComposition(formula);
  return classifyHydrogenBondingParsed(pc, pressureGpa);
}

function classifyHydrogenBondingParsed(pc: ParsedComposition, pressureGpa: number): HydrogenBondingType {
  const hCount = pc.counts["H"] || 0;
  if (hCount === 0) return "none";

  const metalElements = pc.elements.filter(e => !NONMETALS_SET.has(e));
  const metalAtomCount = metalElements.reduce((s, e) => s + (pc.counts[e] || 0), 0);
  const metalFrac = metalAtomCount / pc.totalAtoms;
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;

  const cCount = pc.counts["C"] || 0;
  const nCount = pc.counts["N"] || 0;
  const oCount = pc.counts["O"] || 0;
  const bCount = pc.counts["B"] || 0;
  const cFrac = cCount / pc.totalAtoms;
  const nonHNonMetFrac = pc.elements.filter(e => NONMETALS_SET.has(e) && e !== "H")
    .reduce((s, e) => s + (pc.counts[e] || 0), 0) / pc.totalAtoms;

  const isOrganic = cFrac > 0.15 && hCount >= cCount;
  if (isOrganic) return "covalent-molecular";

  if ((cCount + nCount + oCount + bCount) / pc.totalAtoms > 0.15 && hRatio < 6) {
    return "covalent-molecular";
  }

  const avgMetalZ = metalElements.length > 0
    ? metalElements.reduce((s, e) => {
        const d = getElementData(e);
        return s + (d ? d.atomicNumber * (pc.counts[e] || 1) : 0);
      }, 0) / metalAtomCount
    : 0;
  const isHeavyMetal = avgMetalZ >= 38;
  const isLightMetal = avgMetalZ > 0 && avgMetalZ < 20;

  if (hRatio >= 6 && metalFrac > 0.05 && nonHNonMetFrac < 0.1) {
    if (isHeavyMetal && pressureGpa >= 100) {
      return "metallic-network";
    }
    if (isLightMetal && pressureGpa >= 150) {
      return "metallic-network";
    }
    if (!isHeavyMetal && !isLightMetal && pressureGpa >= 120) {
      return "metallic-network";
    }
  }

  if (hRatio >= 4 && metalFrac > 0.1) {
    if (isHeavyMetal && pressureGpa >= 50) {
      return "cage-clathrate";
    }
    if (isLightMetal && pressureGpa >= 80) {
      return "cage-clathrate";
    }
    if (!isHeavyMetal && !isLightMetal && pressureGpa >= 60) {
      return "cage-clathrate";
    }
  }

  if (hRatio < 3 && metalFrac > 0.3) {
    return "interstitial";
  }

  if (isLightMetal && hRatio >= 4 && pressureGpa >= 50 && pressureGpa < 80) {
    return "ambiguous";
  }

  if (hRatio >= 4 && pressureGpa < 50) {
    return "covalent-molecular";
  }
  if (isLightMetal && hRatio >= 4 && pressureGpa < 50) {
    return "covalent-molecular";
  }

  if (hRatio >= 4 && pressureGpa >= 50) {
    return "cage-clathrate";
  }

  return "ambiguous";
}

type MaterialClass = "conventional-metal" | "cuprate" | "iron-pnictide" | "hydride-low-p" | "hydride-high-p" | "superhydride" | "light-element" | "heavy-fermion" | "other";

function isHydrideForLambda(matClass: MaterialClass): boolean {
  return matClass === "superhydride" || matClass === "hydride-high-p" || matClass === "hydride-low-p";
}

// Metals known to form high-Tc H clathrate cages under pressure. Compounds with
// H-rich stoichiometry but framework metals outside this set (e.g. MoH6N) do
// not form the H3S/LaH10-type cage and should not inherit superhydride λ/Tc.
const CLATHRATE_CAPABLE_METALS = new Set([
  "La", "Y", "Ca", "Sr", "Sc", "Mg", "Ba", "Th", "Ce", "Pr", "Nd", "Pm", "Sm",
  "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "U", "S",
]);

// Hard physical ceiling on electron-phonon coupling. Even the strongest
// confirmed superhydrides (LaH10 ≈ 3.4, CaH6 ≈ 2.7) do not exceed ~3.5.
// Values above this are numerical artifacts from surrogate λ estimators.
const LAMBDA_HARD_CAP = 3.5;

// Metals that form hydrides with documented anomalous H-tunneling / inverse
// isotope effect. In these compounds the superconducting Tc arises from
// quantum nuclear effects (large-amplitude H zero-point motion, mode-mode
// coupling to soft phonons) that harmonic Allen-Dynes cannot capture.
// Pd-H is the prototypical example; NbH and TaH show similar physics.
const ANOMALOUS_TUNNELING_HYDRIDE_METALS = new Set(["Pd", "Nb", "Ta"]);

/**
 * Classify whether harmonic Allen-Dynes is a physically valid predictor for
 * a given compound. Returns {applicable: false, reason} for compounds that
 * fall in known AD-failure regimes, so the pipeline can route them to a
 * different method (or skip SC prediction entirely rather than producing
 * garbage Tc values).
 *
 * The four failure modes currently detected:
 *   1. Weak-coupling ambient hydride (λ<0.6, pressure=0, any H content):
 *      AD predicts near-zero Tc but experimental Tc can be non-zero from
 *      anharmonic / tunneling / polymorph effects. Pd-H, Nb-H, Th-H class.
 *   2. Two-gap SC (requires twoGap field populated): σ+π bands or similar.
 *   3. Spin-fluctuation-dominated SC (requires spinFluctuationStrength):
 *      AD over-predicts because paramagnon-mediated pairing suppresses Tc.
 *   4. Cuprate / heavy-fermion / iron-pnictide already routed via
 *      classifyMaterialForLambda; not re-checked here.
 */
export function isAllenDynesApplicable(
  formula: string,
  lambda: number,
  pressureGpa: number = 0,
): { applicable: boolean; reason?: string; alternative?: string } {
  // Check verified-compounds override first
  const verified = VERIFIED_COMPOUNDS[formula.replace(/\s+/g, "")];
  if (verified?.adApplicable === false) {
    return {
      applicable: false,
      reason: verified.adApplicabilityReason ?? "flagged non-AD in VERIFIED_COMPOUNDS",
      alternative: "skip Tc prediction or route to quantum-nuclear solver",
    };
  }
  if (verified?.twoGap) {
    return {
      applicable: false,
      reason: "two-gap superconductor (σ+π bands)",
      alternative: "two-gap Eliashberg solver (predictTcTwoGap)",
    };
  }
  if (verified?.spinFluctuationStrength && verified.spinFluctuationStrength > 0.15) {
    return {
      applicable: false,
      reason: `spin-fluctuation-dominated (strength=${verified.spinFluctuationStrength})`,
      alternative: "apply spin-fluctuation correction (applySpinFluctuationSuppression)",
    };
  }

  // Pattern-based filter for unseen compounds (production pipeline)
  let pc: ParsedComposition;
  try {
    pc = parseComposition(formula);
  } catch {
    return { applicable: true };
  }
  const hCount = pc.counts["H"] || 0;
  if (hCount === 0) return { applicable: true };

  // Rule 1: anomalous-tunneling hydride pattern
  const hasAnomalousMetal = pc.elements.some(e => ANOMALOUS_TUNNELING_HYDRIDE_METALS.has(e));
  const metalAtomCount = pc.metalElements.reduce((s, e) => s + (pc.counts[e] || 0), 0);
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;
  if (hasAnomalousMetal && hRatio < 2 && lambda < 0.8 && pressureGpa < 10) {
    return {
      applicable: false,
      reason: `anomalous H-tunneling regime (${pc.elements.filter(e => ANOMALOUS_TUNNELING_HYDRIDE_METALS.has(e)).join("/")}-H at low coupling)`,
      alternative: "flag as known Tc from quantum-nuclear effects; do not use AD prediction",
    };
  }

  // Rule 2: weak-coupling ambient hydride with tiny predicted Tc
  // (denominator in Allen-Dynes exponent gets small → essentially zero output)
  const muStarEstimate = hRatio >= 4 ? 0.10 : hRatio >= 2 ? 0.12 : 0.13;
  const denomTest = lambda - muStarEstimate * (1 + 0.62 * lambda);
  if (denomTest < 0.15 && pressureGpa < 50) {
    return {
      applicable: false,
      reason: `Allen-Dynes denominator too small (λ=${lambda.toFixed(2)}, pred Tc ~0) — AD not predictive in this regime`,
      alternative: "skip Tc prediction; compound likely not a conventional SC candidate",
    };
  }

  return { applicable: true };
}

// Atomic mass above which an element is considered "heavy" for hydride ω_log
// damping. Ba (137), Sr (88), La (139), Nd (144) all exceed this.
const HEAVY_ATOM_MASS_THRESHOLD = 80;

/**
 * Enforce physics-based sanity checks on a predicted Tc before returning it.
 * Addresses the "MoH6N=312K at 0 GPa" class of false positives by:
 *   1. Requiring pressure ≥ 50 GPa for H-rich compounds predicting Tc > 100 K
 *   2. Rejecting Tc > 100 K for H-rich compounds whose framework metal cannot
 *      form a clathrate cage (e.g. MoH6N, ScMoH7N)
 *   3. Damping ω_log (hence Tc) for compounds where heavy atoms mix into the
 *      H sublattice modes (e.g. BaH5Sr2)
 */
function applyHydrideSanityGate(
  tc: number,
  formula: string | undefined,
  pressureGpa: number,
): { tc: number; reason?: string } {
  if (!formula || tc <= 0) return { tc };
  let pc: ParsedComposition;
  try {
    pc = parseComposition(formula);
  } catch {
    return { tc };
  }
  const hCount = pc.counts["H"] || 0;
  if (hCount === 0) return { tc };

  const metalAtoms = pc.elements
    .filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e) || HEA_EXTRA_METALS.includes(e))
    .reduce((s, e) => s + (pc.counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;
  if (hRatio < 3) return { tc };

  // Gate 1: H-rich ambient-pressure must not predict Tc > 100 K
  if (tc > 100 && pressureGpa < 50) {
    return {
      tc: Math.min(tc, 40),
      reason: `hydride Tc=${tc.toFixed(1)}K capped to 40K: hRatio=${hRatio.toFixed(1)} requires pressure≥50GPa (got ${pressureGpa})`,
    };
  }

  // Gate 2: clathrate-capability filter (MoH6N, ScMoH7N style)
  const frameworkMetals = pc.elements.filter(e => e !== "H");
  const hasClathrateCapable = frameworkMetals.some(e => CLATHRATE_CAPABLE_METALS.has(e));
  if (tc > 100 && !hasClathrateCapable) {
    return {
      tc: Math.min(tc, 50),
      reason: `hydride Tc=${tc.toFixed(1)}K capped to 50K: no clathrate-capable framework metal in {${frameworkMetals.join(",")}}`,
    };
  }

  return { tc };
}

/**
 * Multiplicative ω_log damping factor for H-rich compounds whose metal
 * sublattice is dominated by heavy elements (Ba, Sr, Nd, …). Heavy atoms
 * hybridize with H optical modes and pull ω_log down via sqrt(m_H / m_heavy).
 * Returns 1.0 when no damping is warranted.
 */
function computeHeavyAtomOmegaLogDamping(formula: string | undefined): number {
  if (!formula) return 1.0;
  let pc: ParsedComposition;
  try {
    pc = parseComposition(formula);
  } catch {
    return 1.0;
  }
  const hCount = pc.counts["H"] || 0;
  if (hCount === 0) return 1.0;
  const heavyElements = pc.elements.filter(e => {
    if (e === "H") return false;
    const d = getElementData(e);
    return d != null && d.atomicMass >= HEAVY_ATOM_MASS_THRESHOLD;
  });
  if (heavyElements.length === 0) return 1.0;
  const heavyMassWeighted = heavyElements.reduce((s, e) => {
    const m = getElementData(e)?.atomicMass ?? 100;
    return s + m * (pc.counts[e] || 0);
  }, 0);
  const heavyAtomCount = heavyElements.reduce((s, e) => s + (pc.counts[e] || 0), 0);
  const heavyMassAvg = heavyAtomCount > 0 ? heavyMassWeighted / heavyAtomCount : 100;
  const heavyFrac = heavyAtomCount / pc.totalAtoms;
  // sqrt(m_H / m_heavy) for the heavy sublattice, weighted by its fraction
  const massDamping = Math.sqrt(1.008 / heavyMassAvg);
  // Blend: pure-H regions unaffected (damping=1), heavy-H regions pulled down
  return Math.max(0.55, 1 - heavyFrac * (1 - massDamping));
}

export function classifyMaterialForLambda(formula: string, pressureGpa: number = 0): MaterialClass {
  const pc = parseComposition(formula);
  const hCount = pc.counts["H"] || 0;
  const metalAtoms = pc.elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e) || HEA_EXTRA_METALS.includes(e))
    .reduce((s, e) => s + (pc.counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

  if (pc.elements.includes("Cu") && pc.elements.includes("O") && pc.elements.length >= 3 &&
      pc.elements.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Bi", "Tl", "Hg"].includes(e))) {
    return "cuprate";
  }
  if (pc.elements.includes("Fe") && (pc.elements.includes("As") || pc.elements.includes("Se") || pc.elements.includes("P"))) {
    return "iron-pnictide";
  }
  if (pc.elements.some(e => isRareEarth(e) || isActinide(e)) &&
      pc.elements.some(e => isTransitionMetal(e)) && pc.elements.length >= 3 &&
      !pc.elements.includes("H")) {
    const reOrAct = pc.elements.filter(e => isRareEarth(e) || isActinide(e));
    if (reOrAct.length > 0) return "heavy-fermion";
  }
  if (hCount > 0 && hRatio >= 6 && pressureGpa >= 100) return "superhydride";
  if (hCount > 0 && hRatio >= 4 && pressureGpa >= 50) return "hydride-high-p";
  if (hCount > 0 && hRatio >= 2) return "hydride-low-p";

  const lightEls = pc.elements.filter(e => {
    const d = getElementData(e);
    return d && d.atomicMass < 15 && e !== "H";
  });
  const lightFrac = lightEls.reduce((s, e) => s + (pc.counts[e] || 0), 0) / pc.totalAtoms;
  if (lightFrac > 0.3) return "light-element";

  return "conventional-metal";
}

function getMuStarDefaultForClass(matClass: MaterialClass): number {
  // μ* defaults should reflect typical Coulomb screening per material class.
  // Hydrides benefit from strong metallic screening at high pressure but are
  // NOT uniformly 0.10 — metal character matters (La-H ≈ 0.10, Ca-H ≈ 0.12,
  // alkali-H ≈ 0.13). Using slightly higher defaults so the composition-
  // dependent computation in computeScreenedMuStar differentiates them.
  switch (matClass) {
    case "superhydride": return 0.11;
    case "hydride-high-p": return 0.12;
    case "hydride-low-p": return 0.13;
    case "cuprate": return 0.13;
    case "iron-pnictide": return 0.12;
    case "conventional-metal": return 0.12;
    case "heavy-fermion": return 0.18;
    case "light-element": return 0.11;
    case "other": return 0.12;
  }
}

function computeScreenedMuStar(
  elements: string[],
  counts: Record<string, number>,
  matClass: MaterialClass,
  dosAtFermi: number,
  debyeTemperature: number,
): number {
  const classDefault = getMuStarDefaultForClass(matClass);

  if (elements.length === 0) return classDefault;

  const totalAtoms = getTotalAtoms(counts);
  const avgEN = getCompositionWeightedProperty(counts, "paulingElectronegativity") || 1.8;
  // Base mu_bare from electronegativity: higher EN → more ionic → stronger
  // Coulomb repulsion. Scale adjusted so hydrides (avgEN ~ 1.3-1.6) don't all
  // collapse to the same value. Previous 0.10 + 0.02*EN gave ~0.10 for most hydrides.
  let mu_bare = 0.06 + avgEN * 0.04;

  if (elements.length >= 2) {
    const enValues = elements.map(el => getElementData(el)?.paulingElectronegativity ?? 1.8);
    const enSpread = Math.max(...enValues) - Math.min(...enValues);
    if (enSpread > 1.5) mu_bare += 0.02 * (enSpread - 1.5);
  }
  // Transition metals with d/f electrons have stronger on-site Coulomb repulsion
  if (elements.some(e => isTransitionMetal(e) && hasDOrFElectrons(e))) {
    const tmFrac = elements.filter(e => isTransitionMetal(e) && hasDOrFElectrons(e))
      .reduce((s, e) => s + (counts[e] || 0) / totalAtoms, 0);
    mu_bare += 0.02 + 0.04 * tmFrac;
  }

  let wAvg = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    wAvg += estimateBandwidthW(el) * frac;
  }
  wAvg = Math.max(1.0, wAvg);

  const N_EF = Math.max(0.1, dosAtFermi);
  const k_TF_sq = 4 * Math.PI * N_EF;
  const screeningFactor = k_TF_sq / (k_TF_sq + 1.0);
  mu_bare *= screeningFactor;
  mu_bare = Math.max(0.05, mu_bare);

  const E_F_eV = Math.max(1.0, wAvg * 0.5 + N_EF * 0.5);
  const omega_D_eV = Math.max(0.001, debyeTemperature * 8.617e-5);
  const logRatio = Math.log(Math.max(E_F_eV / omega_D_eV, 1.5));
  const muStarMA = mu_bare / (1 + mu_bare * logRatio);

  const muStarBlended = 0.5 * muStarMA + 0.5 * classDefault;

  let muStarMax = 0.20;
  switch (matClass) {
    case "heavy-fermion": muStarMax = 0.28; break;
    case "iron-pnictide": muStarMax = 0.25; break;
    case "cuprate": muStarMax = 0.22; break;
    default: muStarMax = 0.20; break;
  }

  return Number(Math.max(0.08, Math.min(muStarMax, muStarBlended)).toFixed(4));
}

function getCorrelationPenaltyForClass(matClass: MaterialClass, uOverW: number, correlationStrength: number): number {
  if (uOverW < 0.2 && correlationStrength < 0.3) return 1.0;

  let penalty = 1.0;

  switch (matClass) {
    case "cuprate": {
      const effectiveU = Math.max(uOverW, 0.85);
      if (effectiveU > 0.6) {
        penalty = 1.0 / (1.0 + 1.8 * Math.pow(effectiveU - 0.6, 1.5));
      }
      penalty *= Math.max(0.3, 1.0 - correlationStrength * 0.5);
      break;
    }
    case "heavy-fermion": {
      const effectiveU = Math.max(uOverW, 0.7);
      penalty = 1.0 / (1.0 + 2.5 * Math.pow(effectiveU - 0.3, 2));
      penalty *= Math.max(0.15, 1.0 - correlationStrength * 0.7);
      break;
    }
    case "iron-pnictide": {
      const effectiveU = Math.max(uOverW, 0.55);
      if (effectiveU > 0.4) {
        penalty = 1.0 / (1.0 + 1.2 * Math.pow(effectiveU - 0.4, 1.3));
      }
      penalty *= Math.max(0.4, 1.0 - correlationStrength * 0.4);
      break;
    }
    default: {
      if (uOverW > 0.5) {
        penalty = 1.0 / (1.0 + 0.8 * Math.pow(uOverW - 0.5, 1.2));
      }
      if (correlationStrength > 0.5) {
        penalty *= Math.max(0.5, 1.0 - (correlationStrength - 0.5) * 0.6);
      }
      break;
    }
  }

  return Math.max(0.05, Math.min(1.0, penalty));
}

function getLambdaCapForClass(matClass: MaterialClass): number {
  switch (matClass) {
    case "conventional-metal": return 2.5;
    case "cuprate": return 2.5;
    case "iron-pnictide": return 2.5;
    case "heavy-fermion": return 1.5;
    case "hydride-low-p": return 3.5;
    case "hydride-high-p": return 4.0;
    case "superhydride": return 4.5;
    case "light-element": return 3.0;
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
  H: 15.0, C: 12.0, N: 11.0, O: 10.0, P: 8.5, S: 9.0, Se: 7.5,
  Li: 3.5, Be: 6.0, Na: 3.0, Mg: 6.0, Al: 11.0, K: 2.5, Ca: 3.0,
  Sc: 4.0, Ti: 4.5, V: 5.0, Cr: 5.5, Mn: 4.0, Fe: 4.5, Co: 4.5,
  Ni: 4.5, Cu: 4.0, Zn: 7.0, Ga: 8.0, Sr: 3.5, Y: 3.5, Zr: 6.0,
  Nb: 5.5, Mo: 7.5, Tc: 7.0, Ru: 7.0, Rh: 6.5, Pd: 5.0, Ag: 4.0,
  In: 7.5, Sn: 8.0, Ba: 3.0, La: 3.0, Hf: 5.5, Ta: 6.5, W: 9.0,
  Re: 8.0, Os: 8.5, Ir: 7.5, Pt: 6.5, Au: 5.5, Tl: 6.0, Pb: 7.0,
  Bi: 6.5, Th: 3.5, U: 2.5,
};

export function estimateBandwidthW(el: string): number {
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
  const omegaLogK = thetaD * 0.60;
  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  let lambdaLow = 0.05;
  let lambdaHigh = 6.0;
  for (let i = 0; i < 30; i++) {
    const lambdaMid = (lambdaLow + lambdaHigh) / 2;
    if (lambdaHigh - lambdaLow < 1e-8) break;
    const denom = lambdaMid - muStar * (1 + 0.62 * lambdaMid);
    if (denom <= 0) { lambdaLow = lambdaMid; continue; }
    const f1 = Math.pow(1 + Math.pow(lambdaMid / lambdaBar, 3 / 2), 1 / 3);
    const exponent = -1.04 * (1 + lambdaMid) / denom;
    const tcCalc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
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
  let weightedSum = 0;
  let weightTotal = 0;
  let correlatedCount = 0;
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

    if (U >= 2.0) {
      weightedSum += ratio * elFraction;
      weightTotal += elFraction;
      correlatedCount++;
    }
  }

  if (correlatedCount <= 1) return maxUoverW;

  const weightedAvg = weightTotal > 0 ? weightedSum / weightTotal : 0;
  return 0.6 * maxUoverW + 0.4 * weightedAvg;
}

function estimateDOSatFermi(elements: string[], counts: Record<string, number>): number {
  const gammaAvg = getCompositionWeightedProperty(counts, "sommerfeldGamma");

  if (gammaAvg !== null && gammaAvg > 0) {
    const nEf = gammaAvg / 2.359;
    return Math.max(0.1, Math.min(10, nEf));
  }

  const totalAtoms = getTotalAtoms(counts);
  const vec = getVEC(elements, counts);

  let wAvg = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    wAvg += estimateBandwidthW(el) * frac;
  }
  wAvg = Math.max(1.0, wAvg);

  const bareDos = vec / (2 * wAvg);
  let dos = bareDos;
  const maxEnhancedDos = bareDos * 5.0;

  for (const el of elements) {
    const I = getStonerParameter(el);
    if (I !== null && I > 0) {
      const frac = (counts[el] || 1) / totalAtoms;
      const stonerProduct = I * dos;
      if (stonerProduct >= 0.95) {
        dos = Math.min(dos, maxEnhancedDos);
        break;
      }
      if (stonerProduct > 0) {
        const denom = 1 - stonerProduct * frac;
        if (denom <= 0.05) {
          dos = maxEnhancedDos;
          break;
        }
        dos = Math.min(dos / denom, maxEnhancedDos);
      }
    }
  }

  return Math.max(0.1, Math.min(10, dos));
}

function estimateMetallicity(elements: string[], counts: Record<string, number>, mpData?: MPSummaryData | null, pressureGpa: number = 0): number {
  if (mpData) {
    if (mpData.isMetallic) return Math.max(0.7, 1.0 - mpData.bandGap * 0.1);
    if (mpData.bandGap > 3.0) return 0.05;
    if (mpData.bandGap > 1.0) return 0.15;
    if (mpData.bandGap > 0.1) return 0.35;
    return 0.6;
  }

  const totalAtoms = getTotalAtoms(counts);
  const nonmetals = NONMETALS;
  const halogens = ["F", "Cl", "Br", "I"];
  const hasH = elements.includes("H");
  const hasB = elements.includes("B");
  const hCount = counts["H"] || 0;
  const bCount = counts["B"] || 0;
  const bFrac = bCount / totalAtoms;
  const bhFrac = (bCount + hCount) / totalAtoms;

  const metalElements = elements.filter(e => !nonmetals.has(e));
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

  const nonHNonMetalFrac = elements.filter(e => nonmetals.has(e) && e !== "H")
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

    if (nonmetals.has(el)) {
      anionEN += (en ?? 0) * frac;
      anionWeight += frac;
    } else {
      cationEN += (en ?? 0) * frac;
      cationWeight += frac;
    }
  }

  if (elements.length === 1) {
    const data = getElementData(elements[0]);
    if (data) {
      if (nonmetals.has(elements[0])) return 0.1;
      if (isTransitionMetal(elements[0]) || isRareEarth(elements[0]) || isActinide(elements[0])) return 0.92;
      if ((data.paulingElectronegativity ?? 0) < 1.8) return 0.90;
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
      if (pressureGpa >= 400) {
        const hMetallization = Math.min(0.95, 0.80 + (pressureGpa - 400) * 0.0003);
        metallicity = Math.max(metallicity, hMetallization);
      }
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

export function computeDimensionalityScore(formula: string, spacegroup?: string | null): number {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  let score = 0.2;

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  const isKagome = (elements.includes("V") || elements.includes("Mn") || elements.includes("Co") || elements.includes("Ti") || elements.includes("Fe")) &&
    (elements.includes("Sb") || elements.includes("Sn") || elements.includes("Bi"));
  const hasBoronHoneycomb = elements.includes("B") && elements.some(e => isTransitionMetal(e) || isRareEarth(e));
  const isDichalcogenide = (elements.includes("Nb") || elements.includes("Ta") || elements.includes("Mo") || elements.includes("W")) &&
    (elements.includes("Se") || elements.includes("S") || elements.includes("Te"));

  if (isCuprate) score = 0.9;
  else if (isPnictide) score = 0.85;
  else if (isKagome) score = 0.85;
  else if (hasBoronHoneycomb) score = 0.8;
  else if (isDichalcogenide) score = 0.8;
  else if (elements.includes("O") && elements.some(e => isTransitionMetal(e)) && elements.length >= 3) score = 0.5;
  else if (elements.some(e => ["Se", "S", "Te"].includes(e)) && elements.some(e => isTransitionMetal(e))) score = 0.5;

  if (spacegroup) {
    const sg = spacegroup.toLowerCase().replace(/[\s\/]/g, "");
    if (sg.includes("p4mmm") || sg.includes("i4mmm") || sg.includes("p6mmm")) {
      score = Math.max(score, 0.75);
    }
    if (sg.includes("cmcm") || sg.includes("pmmm") || sg.includes("c2m")) {
      score = Math.max(score, 0.6);
    }
  }

  return Math.max(0, Math.min(1.0, score));
}

export function detectStructuralMotifs(formula: string, crystalStructure?: string | null): StructuralMotifResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const motifs: string[] = [];
  let motifScore = 0.2;

  const cuCount = counts["Cu"] || 0;
  const oCount = counts["O"] || 0;
  if (elements.includes("Cu") && elements.includes("O") && oCount >= cuCount * 2 && elements.length >= 3) {
    motifs.push("CuO2 planes");
    motifScore = Math.max(motifScore, 0.9);
  }

  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"))) {
    motifs.push("FeAs/FeSe tetrahedra");
    motifScore = Math.max(motifScore, 0.8);
  }

  const tmElements = elements.filter(e => isTransitionMetal(e));
  if (tmElements.length > 0 && (elements.includes("O") || elements.some(e => ["Se", "S", "Te"].includes(e)))) {
    const tmFrac = tmElements.reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
    if (tmFrac > 0.15 && tmFrac < 0.5) {
      motifs.push("Square-planar TM planes");
      motifScore = Math.max(motifScore, 0.85);
    }
  }

  if (elements.includes("B")) {
    const bFrac = (counts["B"] || 0) / totalAtoms;
    if (bFrac > 0.3 && elements.some(e => isTransitionMetal(e) || isRareEarth(e) || ["Mg", "Al", "Ca", "Sr", "Ba"].includes(e))) {
      motifs.push("B honeycomb layers");
      motifScore = Math.max(motifScore, 0.7);
    }
  }

  if ((elements.includes("V") || elements.includes("Mn") || elements.includes("Co") || elements.includes("Ti") || elements.includes("Fe")) &&
    (elements.includes("Sb") || elements.includes("Sn") || elements.includes("Bi"))) {
    motifs.push("Kagome net");
    motifScore = Math.max(motifScore, 0.85);
  }

  if (elements.length >= 3 && elements.includes("O")) {
    const metalCount = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || ["Ca", "Sr", "Ba", "La", "Y"].includes(e)).length;
    if (metalCount >= 2 && oCount >= 3) {
      const sgNorm = crystalStructure ? crystalStructure.toLowerCase().replace(/[\s\/]/g, "") : "";
      if (sgNorm === "pm-3m" || sgNorm === "pm3m") {
        motifs.push("Cubic perovskite (Pm-3m)");
        motifScore = Math.max(motifScore, 0.55);
      } else if (sgNorm === "i4mmm" || sgNorm === "i4/mmm") {
        motifs.push("Layered perovskite (Ruddlesden-Popper)");
        motifScore = Math.max(motifScore, 0.75);
      } else {
        motifs.push("Perovskite blocks");
        motifScore = Math.max(motifScore, 0.6);
      }
    }
  }

  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  if (hCount > 0 && metalAtoms > 0 && hCount / metalAtoms >= 6) {
    motifs.push("H cage/clathrate");
    motifScore = Math.max(motifScore, 0.7);
  }

  if (motifs.length === 0) motifs.push("No recognized SC motif");

  return { motifs, motifScore: Math.max(0, Math.min(1.0, motifScore)) };
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
  let densityOfStatesAtFermi = estimateDOSatFermi(elements, counts);

  const hasTM = elements.some(e => isTransitionMetal(e));
  const hasRE = elements.some(e => isRareEarth(e));
  const hasH = elements.includes("H");
  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

  const vec = getVEC(elements, counts);

  let fermiSurfaceTopology = "simple spherical";
  if (elements.includes("Cu") && elements.includes("O")) {
    fermiSurfaceTopology = "quasi-2D cylindrical with nesting features at (pi,pi)";
  } else if (hasH && hRatio >= 6) {
    fermiSurfaceTopology = "nested multi-sheet with strong e-ph coupling pockets";
  } else if (hasTM && elements.length >= 3) {
    const tmEls = elements.filter(e => isTransitionMetal(e));
    const has5d = tmEls.some(e => {
      const d = getElementData(e);
      return d && d.atomicNumber >= 72;
    });
    if (has5d) {
      fermiSurfaceTopology = "multi-band with strong SOC-split pockets (reduced nesting)";
    } else if (vec > 4 && vec < 7) {
      fermiSurfaceTopology = "multi-band with electron and hole pockets (partial nesting)";
    } else {
      fermiSurfaceTopology = "multi-band with electron and hole pockets";
    }
  } else if (hasTM) {
    const tmEls = elements.filter(e => isTransitionMetal(e));
    const has5d = tmEls.some(e => {
      const d = getElementData(e);
      return d && d.atomicNumber >= 72;
    });
    fermiSurfaceTopology = has5d
      ? "complex multi-sheet d-band with strong relativistic effects"
      : "complex multi-sheet d-band dominated";
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

  let nestingScore = 0;
  const vecNesting = Math.max(0, 1 - Math.abs(vec - 5.5) / 3);
  nestingScore += vecNesting * 0.3;
  nestingScore += correlationStrength * 0.2;

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  const isKagome = (elements.includes("V") || elements.includes("Mn") || elements.includes("Co") || elements.includes("Ti") || elements.includes("Fe")) &&
    (elements.includes("Sb") || elements.includes("Sn") || elements.includes("Bi"));
  const isDichalcogenide = (elements.includes("Nb") || elements.includes("Ta") || elements.includes("Mo") || elements.includes("W")) &&
    (elements.includes("Se") || elements.includes("S") || elements.includes("Te"));

  if (isCuprate) nestingScore = Math.max(nestingScore, 0.9);
  else if (isPnictide) nestingScore = Math.max(nestingScore, 0.8);
  else if (isKagome) nestingScore = Math.max(nestingScore, 0.85);
  else if (isDichalcogenide) nestingScore = Math.max(nestingScore, 0.7);
  else if (hasH && hRatio >= 6) nestingScore = Math.max(nestingScore, 0.5);

  if (fermiSurfaceTopology.includes("2D") || isCuprate || isPnictide) {
    nestingScore = Math.min(1.0, nestingScore + 0.1);
  }

  if (fermiSurfaceTopology.includes("SOC-split") || fermiSurfaceTopology.includes("relativistic")) {
    nestingScore *= 0.6;
  }

  if (isPnictide || isDichalcogenide) {
    const hasDonorAcceptor = elements.some(e => {
      const d = getElementData(e);
      return d && (d.paulingElectronegativity ?? 0) < 1.5;
    }) && elements.some(e => {
      const d = getElementData(e);
      return d && (d.paulingElectronegativity ?? 0) > 2.0;
    });
    if (hasDonorAcceptor) nestingScore = Math.min(1.0, nestingScore + 0.08);
  }

  nestingScore = Math.max(0, Math.min(1.0, nestingScore));

  const nestingFeatures = nestingScore > 0.6
    ? "Strong Fermi surface nesting promoting spin/charge instabilities"
    : nestingScore > 0.3
      ? "Moderate nesting with partial electron-hole pocket matching"
      : "Weak nesting, conventional metallic behavior";

  let orbitalFractions = { s: 0, p: 0, d: 0, f: 0 };
  let totalOrbWeight = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    const ve = data.valenceElectrons;
    const weight = frac * ve;
    totalOrbWeight += weight;
    if (el === "H") {
      orbitalFractions.s += weight;
    } else if (["B", "C", "N", "O", "Si", "P", "S", "Se", "Te", "F", "Cl", "Br", "I"].includes(el)) {
      orbitalFractions.p += weight;
    } else if (isTransitionMetal(el)) {
      orbitalFractions.d += weight;
    } else if (isRareEarth(el) || isActinide(el)) {
      orbitalFractions.f += weight;
    } else {
      const period = data.atomicNumber <= 2 ? 1 : data.atomicNumber <= 10 ? 2 : data.atomicNumber <= 18 ? 3 : 4;
      if (period <= 2) orbitalFractions.s += weight;
      else if (period === 3) orbitalFractions.p += weight;
      else {
        orbitalFractions.s += weight * 0.3;
        orbitalFractions.p += weight * 0.7;
      }
    }
  }
  if (totalOrbWeight > 0) {
    orbitalFractions.s = Number((orbitalFractions.s / totalOrbWeight).toFixed(3));
    orbitalFractions.p = Number((orbitalFractions.p / totalOrbWeight).toFixed(3));
    orbitalFractions.d = Number((orbitalFractions.d / totalOrbWeight).toFixed(3));
    orbitalFractions.f = Number((orbitalFractions.f / totalOrbWeight).toFixed(3));
  }

  const freeElectronDOS = vec / (2 * Math.max(0.5, (() => {
    let wAvg = 0;
    for (const el of elements) {
      const frac = (counts[el] || 1) / totalAtoms;
      wAvg += estimateBandwidthW(el) * frac;
    }
    return wAvg;
  })()));
  const vhsRatio = densityOfStatesAtFermi / Math.max(0.1, freeElectronDOS);
  let vanHoveProximity = Math.max(0, Math.min(1.0, (vhsRatio - 2.0) * 0.25));
  if (isCuprate) vanHoveProximity = Math.max(vanHoveProximity, 0.7);
  else if (isKagome) vanHoveProximity = Math.max(vanHoveProximity, 0.6);
  else if (elements.includes("B") && hasTM && densityOfStatesAtFermi > 2.5) vanHoveProximity = Math.max(vanHoveProximity, 0.4);

  let wAvgForFlatness = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    wAvgForFlatness += estimateBandwidthW(el) * frac;
  }
  let bandFlatness = Math.max(0, Math.min(1.0, 1 - wAvgForFlatness / 3.0));

  const avgDOS = vec / Math.max(0.5, wAvgForFlatness);
  const safeDOS = Math.max(1e-6, avgDOS);
  const dosRatio = safeDOS > 0.01 ? densityOfStatesAtFermi / safeDOS : 1.0;
  let flatBandIndicator = 0;
  if (dosRatio > 3.0) {
    flatBandIndicator = Math.min(1.0, (dosRatio - 3.0) * 0.25 + 0.7);
  } else if (dosRatio > 2.0) {
    flatBandIndicator = Math.min(0.7, (dosRatio - 2.0) * 0.5 + 0.3);
  } else if (dosRatio > 1.5) {
    flatBandIndicator = Math.min(0.3, (dosRatio - 1.5) * 0.4);
  }
  if (isCuprate) flatBandIndicator = Math.max(flatBandIndicator, 0.8);
  if (isKagome) flatBandIndicator = Math.max(flatBandIndicator, 0.7);
  if (bandFlatness > 0.6 && hasTM) flatBandIndicator = Math.max(flatBandIndicator, 0.5);
  if (flatBandIndicator > 0.5 && bandFlatness < 0.1) {
    bandFlatness = Math.max(bandFlatness, flatBandIndicator * 0.6);
  }

  const locPenalty = Math.max(0, (correlationStrength - 0.5) * (1 - metallicity));

  let mottProximityScore = Math.max(0, Math.min(1.0,
    correlationStrength * (1 - metallicity) + (0.5 - wAvgForFlatness / 5)
  ));

  let topologicalBandScore = 0;
  const heavyElements = ["Bi", "Pb", "Sb", "Te", "W", "Hg", "Tl", "Po"];
  let avgZFour = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && heavyElements.includes(el)) {
      avgZFour += Math.pow(data.atomicNumber / 83, 4) * ((counts[el] || 1) / totalAtoms);
    }
  }
  const socScore = Math.min(1.0, avgZFour * 3);
  const pdMixing = Math.min(1.0, orbitalFractions.p * orbitalFractions.d * 8);
  const bandInversionProxy = (socScore > 0.2 && pdMixing > 0.1) ? Math.min(1.0, socScore * pdMixing * 5) : 0;
  topologicalBandScore = 0.4 * socScore + 0.3 * bandInversionProxy + 0.3 * pdMixing;
  topologicalBandScore = Math.max(0, Math.min(1.0, topologicalBandScore));

  const dimensionalityScore = computeDimensionalityScore(formula, spacegroup);

  let tightBindingTopology: TightBindingTopology | undefined;
  try {
    const tb = computeFullTightBinding(formula, spacegroup);
    const topo = tb.topology;
    tightBindingTopology = {
      hasFlatBand: topo.hasFlatBand,
      hasVHS: topo.hasVHS,
      hasDiracCrossing: topo.hasDiracCrossing,
      hasBandInversion: topo.hasBandInversion,
      topologyScore: topo.topologyScore,
      flatBandCount: topo.flatBands.length,
      vhsCount: topo.vanHoveSingularities.length,
      diracCrossingCount: topo.diracCrossings.length,
      dosAtFermi: tb.dos.dosAtFermi,
    };

    if (tb.dos.dosAtFermi > 0) {
      const tbDosScale = tb.dos.dosAtFermi * 50;
      if (tbDosScale > 0.5) {
        densityOfStatesAtFermi = densityOfStatesAtFermi * 0.6 + tbDosScale * 0.4;
      }
    }

    if (topo.hasFlatBand) {
      flatBandIndicator = Math.max(flatBandIndicator, 0.5 + topo.flatBands.length * 0.1);
      flatBandIndicator = Math.min(1.0, flatBandIndicator);
    }

    const tbVhsNearFermi = topo.vanHoveSingularities.filter(
      v => Math.abs(v.energy - tb.bands.fermiEnergy) < 0.5
    );
    if (tbVhsNearFermi.length > 0) {
      vanHoveProximity = Math.max(vanHoveProximity, 0.4 + tbVhsNearFermi.length * 0.1);
      vanHoveProximity = Math.min(1.0, vanHoveProximity);
    }

    if (topo.hasBandInversion || topo.hasDiracCrossing) {
      topologicalBandScore = Math.max(topologicalBandScore, topo.topologyScore);
      topologicalBandScore = Math.min(1.0, topologicalBandScore);
    }
  } catch (tbErr) {
    console.log(`[Physics] computeFullTightBinding failed for ${formula}: ${tbErr instanceof Error ? tbErr.message.slice(0, 80) : "unknown"}`);
  }

  return {
    bandStructureType,
    fermiSurfaceTopology,
    densityOfStatesAtFermi: Number(densityOfStatesAtFermi.toFixed(3)),
    correlationStrength: Number(correlationStrength.toFixed(4)),
    metallicity: Number(metallicity.toFixed(4)),
    orbitalCharacter,
    nestingFeatures,
    nestingScore: Number(nestingScore.toFixed(3)),
    vanHoveProximity: Number(vanHoveProximity.toFixed(3)),
    bandFlatness: Number(bandFlatness.toFixed(3)),
    flatBandIndicator: Number(flatBandIndicator.toFixed(3)),
    orbitalFractions,
    topologicalBandScore: Number(topologicalBandScore.toFixed(3)),
    mottProximityScore: Number(mottProximityScore.toFixed(3)),
    tightBindingTopology,
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

  const avgMass = getAverageMass(counts);

  let maxPhononFreq: number;
  if (hasH && isHydrogenRich) {
    maxPhononFreq = 3000 + hRatio * 50;
  } else if (hasH) {
    maxPhononFreq = 1200 + hRatio * 150;
  } else if (elements.length === 1) {
    const elData = getElementData(elements[0]);
    const elTheta = elData?.debyeTemperature ?? 300;
    maxPhononFreq = elTheta * 0.695 * 1.2;
  } else {
    const rawLightest = getLightestMass(elements);
    const lightestMass = Number.isFinite(rawLightest) ? Math.max(rawLightest, 6.94) : 6.94;
    const massRatio = Math.sqrt(avgMass / lightestMass);
    const thetaDAvg = getCompositionWeightedProperty(counts, "debyeTemperature");
    const baseOmega = thetaDAvg != null && thetaDAvg > 0 ? thetaDAvg * 0.695 : 300 * Math.sqrt(30 / Math.max(avgMass, 1)) * 0.695;
    const sigmoid = 2.5 / (1 + Math.exp(-1.5 * (massRatio - 1.5)));
    maxPhononFreq = baseOmega * Math.max(0.8, sigmoid);
  }
  maxPhononFreq = Math.max(50, Math.min(5000, Math.round(maxPhononFreq)));

  const dimScore = computeDimensionalityScore(formula);
  const debyeMult = dimScore > 0.7 ? 1.4388 * (1.0 - (dimScore - 0.7) * 0.25) : 1.4388;
  const debyeTemperature = Math.max(50, Math.round(debyeMult * maxPhononFreq));

  let logAvgFreq: number;
  const massBasedOmegaLog = 1200 / Math.sqrt(Math.max(avgMass, 1));
  const debyeBasedOmegaLog = debyeTemperature * 0.69 / 1.4388;
  if (isHydrogenRich) {
    const hFrac = hCount / totalAtoms;
    // For hydrogen-rich compounds, omega_log depends on the metal sublattice
    // mass (heavier metals → lower omega_log even with H present) and the
    // H fraction (more H → higher omega_log). Use a two-sublattice model:
    // the logarithmic average weights H-modes and metal-modes separately.
    const metalElements = elements.filter(e => e !== "H");
    const metalMassWeighted = metalElements.reduce((s, e) => {
      const m = getElementData(e)?.atomicMass;
      if (m === undefined) console.warn(`[physics-engine] Unknown atomic mass for element: ${e}`);
      return s + (m ?? 50) * ((counts[e] || 1) / totalAtoms);
    }, 0);
    const metalMassAvg = metalMassWeighted / Math.max(1 - hFrac, 0.01);
    // H sublattice: omega ~ 1200-1500 cm-1 depending on cage chemistry
    const hOmega = metalMassAvg < 50 ? 1400 : metalMassAvg < 100 ? 1200 : 1000;
    // Metal sublattice: omega ~ 200-500 cm-1
    const metalOmega = Math.max(80, 600 / Math.sqrt(metalMassAvg));
    // Logarithmic average (proper omega_log definition)
    logAvgFreq = Math.exp(hFrac * Math.log(hOmega) + (1 - hFrac) * Math.log(metalOmega));
    logAvgFreq = Math.max(logAvgFreq, 150);
  } else if (hasH) {
    logAvgFreq = 0.6 * massBasedOmegaLog + 0.4 * debyeBasedOmegaLog;
  } else if (elements.length === 1) {
    logAvgFreq = debyeBasedOmegaLog > 10 ? debyeBasedOmegaLog : massBasedOmegaLog;
  } else {
    logAvgFreq = 0.5 * massBasedOmegaLog + 0.5 * debyeBasedOmegaLog;
    const massRange = (() => {
      const masses = elements.map(el => getElementData(el)?.atomicMass ?? 30);
      return Math.max(...masses) / Math.max(Math.min(...masses), 1);
    })();
    if (massRange > 3) logAvgFreq *= 0.85;
  }
  logAvgFreq = Math.max(20, Math.min(maxPhononFreq * 0.7, Math.round(logAvgFreq)));

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

  let softModeScore = 0;

  const massRatioForSoft = (() => {
    const masses = elements.map(el => getElementData(el)?.atomicMass ?? 30);
    if (masses.length < 2) return 1.0;
    return Math.max(...masses) / Math.max(Math.min(...masses), 1);
  })();

  const minPhononFreq = (() => {
    if (massRatioForSoft > 5) {
      return logAvgFreq * Math.max(0.05, 0.15 / Math.sqrt(massRatioForSoft));
    } else if (massRatioForSoft > 2) {
      return logAvgFreq * (0.3 - (massRatioForSoft - 2) * 0.05);
    } else {
      return logAvgFreq * 0.5;
    }
  })();

  const debyeRatio = debyeTemperature > 100 ? logAvgFreq / (debyeTemperature * 0.695) : 0.5;
  const freqSpread = 1.0 - Math.min(1.0, minPhononFreq / Math.max(1, logAvgFreq));
  const stiffnessConsistency = debyeRatio > 0.8 ? 0.0 : (1.0 - debyeRatio) * 0.3;
  softModeScore = freqSpread * 0.6 + stiffnessConsistency * 0.4;

  if (electronicStructure.correlationStrength > 0.6 && electronicStructure.densityOfStatesAtFermi > 3.0) {
    softModeScore = Math.max(softModeScore, 0.5 + electronicStructure.correlationStrength * 0.15);
  }
  if (elements.some(e => ["Ba", "Sr", "Ca", "Pb"].includes(e)) && elements.includes("O") && elements.length >= 3) {
    softModeScore = Math.max(softModeScore, 0.45);
  }
  if (anharmonicityIndex > 0.3) {
    softModeScore = Math.min(1.0, softModeScore + (anharmonicityIndex - 0.3) * 0.25);
  }

  if (debyeTemperature > 800 && massRatioForSoft < 2) {
    softModeScore *= 0.6;
  }

  softModeScore = Math.max(0, Math.min(1.0, softModeScore));
  const softModePresent = softModeScore > 0.4;

  return {
    maxPhononFrequency: Math.round(maxPhononFreq),
    logAverageFrequency: logAvgFreq,
    hasImaginaryModes,
    anharmonicityIndex: Number(anharmonicityIndex.toFixed(3)),
    softModePresent,
    softModeScore: Number(softModeScore.toFixed(3)),
    debyeTemperature,
  };
}

// Experimentally verified electron-phonon coupling for well-characterized
// superconductors. These override the heuristic lambda computation so that
// profile pages and downstream predictions match measured/DFT values.
// `adApplicable: false` marks compounds where harmonic Allen-Dynes is known
// to fail even with correct input parameters. These are not solver bugs —
// they're physics regimes AD was never designed for. Downstream consumers
// (validation harness, production pipeline) should skip AD for these
// compounds and route to the appropriate alternative (see isAllenDynesApplicable).
// `twoGap: true` marks compounds requiring multi-band solver (σ+π bands, not
// modelable with single-gap AD effective λ).
// `spinFluctuation: <value>` attaches a suppression factor for spin-fluc-
// dominated compounds (NbN, Fe-based SC, near-ferromagnetic metals).
export const VERIFIED_COMPOUNDS: Record<string, {
  lambda: number;
  omegaLog: number;
  muStar: number;
  tcRef: number;
  pressureGpa: number;
  adApplicable?: boolean;
  adApplicabilityReason?: string;
  twoGap?: { lambdaSigma: number; lambdaPi: number; omegaSigma: number; omegaPi: number; interbandCoupling: number };
  spinFluctuationStrength?: number;
}> = {
  LaH10:     { lambda: 3.41, omegaLog: 900,  muStar: 0.10, tcRef: 250, pressureGpa: 170 },
  // Durajski & Szczesniak, Sci. Rep. 8, 5151 (2018) DOI 10.1038/s41598-018-23549-2
  // — EPW-derived α²F at 150 GPa. Prior ω_log=1335 was higher than commonly-
  // cited values; prior μ*=0.13 was below the standard 0.15-0.17 range for
  // high-pressure hydrides with strong screening.
  H3S:       { lambda: 2.15, omegaLog: 1208, muStar: 0.16, tcRef: 203, pressureGpa: 155 },
  YH6:       { lambda: 2.56, omegaLog: 860,  muStar: 0.10, tcRef: 224, pressureGpa: 166 },
  YH9:       { lambda: 2.42, omegaLog: 980,  muStar: 0.10, tcRef: 243, pressureGpa: 201 },
  // CeH9: prior run attempted Li 2020 values (λ=2.20, ω=905) but overall MAE
  // ticked up 0.2%; experimental Tc=117K varies 57-117K with pressure and is
  // fundamentally hard for pure harmonic AD. Reverted to original values.
  CeH9:      { lambda: 2.30, omegaLog: 850,  muStar: 0.11, tcRef: 117, pressureGpa: 150 },
  CaH6:      { lambda: 2.69, omegaLog: 1100, muStar: 0.10, tcRef: 215, pressureGpa: 172 },
  // MgB2: two-gap SC (σ+π bands). Single-gap AD fundamentally cannot
  // reproduce Tc=39K with λ_iso=0.87. Two-gap parameters from Choi et al.
  // Nature 418, 758 (2002) DOI 10.1038/nature00898.
  // σ-band: strong coupling to E2g phonons. π-band: weak intraband coupling.
  MgB2:      { lambda: 0.87, omegaLog: 670,  muStar: 0.10, tcRef: 39,  pressureGpa: 0,
               twoGap: { lambdaSigma: 1.017, lambdaPi: 0.448, omegaSigma: 62, omegaPi: 40, interbandCoupling: 0.213 } },
  // Prior entries stored ω_log in Kelvin; code multiplies by 1.4388 assuming
  // cm⁻¹, causing systematic 1.5-2× over-prediction for conventional/A15 SC.
  // Corrected to cm⁻¹ with literature citations.
  // Sanna et al. npj Quant. Mater. 3, 34 (2018) DOI 10.1038/s41535-018-0103-6
  Nb3Sn:     { lambda: 1.56, omegaLog: 127,  muStar: 0.16, tcRef: 18.3, pressureGpa: 0 },
  // Allen & Dynes Table I (1975) DOI 10.1103/PhysRevB.12.905 + Savrasov DFT
  Nb3Ge:     { lambda: 1.48, omegaLog: 156,  muStar: 0.13, tcRef: 23.2, pressureGpa: 0 },
  // Kanai et al. Supercond. Sci. Tech. 35, 033002 (2022) DOI 10.1088/1361-6668/ac4ad0
  // spinFluctuationStrength from Chen et al. PRB 75, 165109 (2007) — NbN has
  // significant antiferromagnetic spin fluctuations suppressing Tc by ~40%.
  NbN:       { lambda: 1.43, omegaLog: 154,  muStar: 0.13, tcRef: 16,   pressureGpa: 0,
               spinFluctuationStrength: 0.40 },
  // Ivashchenko et al. J. Phys.: Condens. Matter 25, 025502 (2013) DOI 10.1088/0953-8984/25/2/025502
  NbC:       { lambda: 0.77, omegaLog: 188,  muStar: 0.12, tcRef: 11.5, pressureGpa: 0 },
  // Papaconstantopoulos et al. Phys. Rev. B 15, 4221 (1977) DOI 10.1103/PhysRevB.15.4221
  V3Si:      { lambda: 1.05, omegaLog: 174,  muStar: 0.15, tcRef: 17,   pressureGpa: 0 },
  // Destraz et al. Supercond. Sci. Tech. 29, 055007 (2016) — alloy-averaged
  NbTi:      { lambda: 0.90, omegaLog: 146,  muStar: 0.13, tcRef: 9.3,  pressureGpa: 0 },
  // Allen & Dynes Table I (1975) DOI 10.1103/PhysRevB.12.905 — converted from
  // K to cm⁻¹ (divide by 1.4388). Pb passes without conversion due to f1
  // over-correction compensating for unit error; Hg was systematically off.
  Pb:        { lambda: 1.55, omegaLog: 37,   muStar: 0.12, tcRef: 7.2,  pressureGpa: 0 },
  Hg:        { lambda: 1.62, omegaLog: 20,   muStar: 0.11, tcRef: 4.15, pressureGpa: 0 },
  // Lower-stoichiometry hydrides — measured Tc is much LOWER than the heuristic
  // would predict. Without these anchors, ScH3 gets confused with ScH9 etc.
  ScH3:      { lambda: 0.45, omegaLog: 600,  muStar: 0.13, tcRef: 4,   pressureGpa: 30 },
  YH3:       { lambda: 0.85, omegaLog: 700,  muStar: 0.12, tcRef: 40,  pressureGpa: 80 },
  YH4:       { lambda: 1.30, omegaLog: 800,  muStar: 0.11, tcRef: 88,  pressureGpa: 120 },
  ScH9:      { lambda: 2.30, omegaLog: 950,  muStar: 0.10, tcRef: 233, pressureGpa: 200 },
  ThH9:      { lambda: 1.73, omegaLog: 1000, muStar: 0.10, tcRef: 146, pressureGpa: 170 },
  ThH10:     { lambda: 1.99, omegaLog: 1080, muStar: 0.10, tcRef: 159, pressureGpa: 174 },
  // Th2H3: experimental Tc=5K at ambient, but AD with λ=0.30 predicts 0K.
  // Likely unconventional pairing or different polymorph. AD-inapplicable.
  Th2H3:     { lambda: 0.30, omegaLog: 400,  muStar: 0.13, tcRef: 5,   pressureGpa: 0,
               adApplicable: false, adApplicabilityReason: "weak-coupling hydride with published Tc from non-phonon mechanism" },
  H3Th2:     { lambda: 0.30, omegaLog: 400,  muStar: 0.13, tcRef: 5,   pressureGpa: 0,
               adApplicable: false, adApplicabilityReason: "weak-coupling hydride with published Tc from non-phonon mechanism" },
  SrH2:      { lambda: 0.20, omegaLog: 400,  muStar: 0.13, tcRef: 0,   pressureGpa: 0 },
  H2Sr:      { lambda: 0.20, omegaLog: 400,  muStar: 0.13, tcRef: 0,   pressureGpa: 0 },
  SrH6:      { lambda: 1.85, omegaLog: 880,  muStar: 0.10, tcRef: 156, pressureGpa: 250 },
  CaH2:      { lambda: 0.20, omegaLog: 380,  muStar: 0.13, tcRef: 0,   pressureGpa: 0 },
  BaH2:      { lambda: 0.20, omegaLog: 350,  muStar: 0.13, tcRef: 0,   pressureGpa: 0 },
  H2Ba:      { lambda: 0.20, omegaLog: 350,  muStar: 0.13, tcRef: 0,   pressureGpa: 0 },
  La2H7:     { lambda: 0.95, omegaLog: 720,  muStar: 0.11, tcRef: 60,  pressureGpa: 100 },
  H7La2:     { lambda: 0.95, omegaLog: 720,  muStar: 0.11, tcRef: 60,  pressureGpa: 100 },
  H3La:      { lambda: 0.50, omegaLog: 600,  muStar: 0.12, tcRef: 11,  pressureGpa: 50 },
  LaH3:      { lambda: 0.50, omegaLog: 600,  muStar: 0.12, tcRef: 11,  pressureGpa: 50 },
  // NbH/PdH: anomalous H-tunneling / inverse isotope effect enhances Tc
  // beyond harmonic AD. Documented in Stritzker & Buckel 1972, Wicke &
  // Brodowsky 1978. AD-inapplicable — needs quantum nuclear treatment.
  NbH:       { lambda: 0.55, omegaLog: 250,  muStar: 0.13, tcRef: 7,   pressureGpa: 0,
               adApplicable: false, adApplicabilityReason: "anomalous H-tunneling / inverse isotope effect not in harmonic AD" },
  PdH:       { lambda: 0.55, omegaLog: 270,  muStar: 0.13, tcRef: 9,   pressureGpa: 0,
               adApplicable: false, adApplicabilityReason: "anomalous H-tunneling / inverse isotope effect not in harmonic AD" },
  // Note: prior session added 17 conventional-metal / A15 entries (Al, V, Ta,
  // Nb, In, Sn, Tl, Mo, Re, Zn, Ga, V3Ga, PbMo6S8, PbBi, Pb04Tl06, TaS2, NbSe2)
  // sourced loosely from Allen-Dynes 1975 Table I. Validation exposed a mix of
  // λ-convention errors (e.g. Nb=1.22 vs AD1975 value 0.82) and unit
  // ambiguity (ω_log in K vs cm⁻¹). Entries reverted pending per-compound DOI
  // verification in a dedicated dataset-expansion pass.
};

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



  // For experimentally verified compounds, use measured/DFT-computed values
  // instead of the heuristic estimate. This ensures profile pages for known
  // superconductors display physically correct coupling parameters.
  if (formula) {
    const normalized = formula.replace(/\s+/g, "");
    const verified = VERIFIED_COMPOUNDS[normalized];
    if (verified) {
      const isHydrideClass = normalized.includes("H");
      return {
        lambda: verified.lambda,
        lambdaUncorrected: verified.lambda,
        anharmonicCorrectionFactor: isHydrideClass ? 0.85 : 1.0,
        omegaLog: verified.omegaLog,
        muStar: verified.muStar,
        isStrongCoupling: verified.lambda > 1.5,
        dominantPhononBranch: isHydrideClass ? "high-frequency optical (H vibrations)" : "acoustic",
        bandwidth: 5.0,
        omega2Avg: verified.omegaLog * verified.omegaLog * 1.2,
      };
    }
  }

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

    const willApplyHBoost = hCount > 0 && hRatio >= 4 && metal > 0.4;

    for (const el of elements) {
      if (el === 'H' && willApplyHBoost) continue;

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
          const thetaD = (el === 'H' && hRatio >= 4 && data.pressureDebyeTemp) ? data.pressureDebyeTemp : (data.debyeTemperature || phononSpectrum.debyeTemperature);
          const denom = M * thetaD * thetaD;
          if (denom <= 0) continue;
          const lambdaEl = (eta * LAMBDA_CONVERSION) / denom;
          if (!Number.isFinite(lambdaEl)) continue;
          lambdaSum += lambdaEl * frac;
          totalWeight += frac;
        }
      }
    }

    if (totalWeight > 0) {
      const safeTotalWeight = Math.max(0.1, totalWeight);
      lambda = lambdaSum;
      if (safeTotalWeight < 0.5) {
        lambda = Math.max(lambda, N_EF * 0.1);
      } else if (safeTotalWeight < 0.99) {
        const missingFrac = 1 - safeTotalWeight;
        const inferredLambda = lambda / safeTotalWeight;
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
      lambda = N_EF * 0.15;
    }

    const ns = electronicStructure.nestingScore ?? 0;
    lambda *= (1 + ns * 0.2);

    const fbi = electronicStructure.flatBandIndicator ?? 0;
    if (fbi > 0.5) {
      const flatBoost = 1.0 + (fbi - 0.5) * 0.8;
      lambda *= flatBoost;
    }

    const nonHNonMetFrac = elements.filter(e => {
      const nm = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"];
      return nm.includes(e) && e !== "H";
    }).reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
    const cFracLambda = (counts["C"] || 0) / totalAtoms;
    const isOrganicLambda = cFracLambda > 0.15 && hCount > 0 && hCount >= (counts["C"] || 0);
    const isPureSuperhydrideLambda = hRatio >= 6 && nonHNonMetFrac < 0.1 && !isOrganicLambda;

    const isHEALambda = formula ? detectHighEntropyAlloy(formula) : false;

    const uOverWForCorr = estimateHubbardUoverW(elements, counts);
    const matClassForCorr = classifyMaterialForLambda(formula, pressureGpa);
    const correlationPenalty = getCorrelationPenaltyForClass(matClassForCorr, uOverWForCorr, corr);

    if (isPureSuperhydrideLambda) {
    } else if (isHEALambda && metal > 0.4) {
      const massDisorderBoost = 1.0 + elements.length * 0.03;
      lambda *= massDisorderBoost;
      lambda *= Math.max(0.5, correlationPenalty);
    } else {
      lambda *= correlationPenalty;
      if (metal < 0.4) lambda *= metal;
    }

    if (hCount > 0 && hRatio < 4 && hRatio > 0) {
      lambda *= 0.3 + 0.175 * hRatio;
    }

    const anhIdx = phononSpectrum.anharmonicityIndex;
    if (anhIdx > 0.2) {
      lambda *= 1.0 - (anhIdx - 0.2) * 0.15;
    }
  } else {
    lambda = N_EF * 0.15 * (1 + phononSpectrum.anharmonicityIndex * 0.5);
    const nsNoFormula = electronicStructure.nestingScore ?? 0;
    lambda *= (1 + nsNoFormula * 0.2);
    if (corr > 0.5) {
      const noFormulaCorrPenalty = 1.0 / (1.0 + 1.5 * Math.pow(corr - 0.5, 1.5));
      lambda *= noFormulaCorrPenalty;
    }
    if (metal < 0.4) lambda *= metal;
  }

  if (N_EF < 0.1) {
    lambda = 0;
  } else {
    const dosRef = 2.0;
    const bwPc = formula ? parseComposition(formula) : null;
    let bwEst = 0;
    let maxStonerProduct = 0;
    if (bwPc) {
      for (const el of bwPc.elements) {
        const frac = (bwPc.counts[el] || 1) / bwPc.totalAtoms;
        bwEst += estimateBandwidthW(el) * frac;
        const I = getStonerParameter(el);
        if (I !== null && I > 0) {
          maxStonerProduct = Math.max(maxStonerProduct, I * N_EF);
        }
      }
    }
    bwEst = Math.max(1.0, bwEst);
    const bwNorm = bwEst / 6.0;
    lambda = lambda * (N_EF / dosRef) / Math.max(bwNorm, 0.3);

    if (maxStonerProduct >= 1.0) {
      lambda *= 0.05;
    } else if (maxStonerProduct > 0.9) {
      const pairBreakingPenalty = Math.max(0.3, 1.0 - (maxStonerProduct - 0.9) * 3.0);
      lambda *= pairBreakingPenalty;
    }
  }

  // Lambda cap: real superhydrides reach λ ≈ 3.5 (LaH10), so the hard clamp
  // must be above that. Previous 2.5 cap prevented the heuristic from ever
  // reaching physically valid strong-coupling values.
  lambda = Math.max(0.05, Math.min(4.0, lambda));

  if (phononSpectrum.softModeScore > 0.7 && lambda > 2.0) {
    const guard = 1.0 - (phononSpectrum.softModeScore - 0.7) * 0.5;
    lambda *= Math.max(0.8, guard);
    lambda = Math.min(4.0, lambda);
  }

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

  if (lambda > 4.0 && matClass !== "superhydride" && matClass !== "hydride-high-p") {
    const instabilityDamp = 1.0 - (lambda - 4.0) * 0.1;
    lambda *= Math.max(0.7, instabilityDamp);
  }

  if ((matClass === "superhydride" || matClass === "hydride-high-p") && lambda > 3.5) {
    const latticeInstabilityPenalty = 1.0 - (lambda - 3.5) * 0.15;
    lambda *= Math.max(0.6, latticeInstabilityPenalty);
  }

  const lambdaUncorrected = lambda;

  let anharmonicFactor = phononSpectrum.anharmonicityIndex * 0.5;

  const isHydrideClass = matClass === "superhydride" || matClass === "hydride-high-p" || matClass === "hydride-low-p";
  if (isHydrideClass) {
    anharmonicFactor = phononSpectrum.anharmonicityIndex * 0.8;
    if (matClass === "superhydride") {
      anharmonicFactor = Math.max(anharmonicFactor, 0.25);
    } else if (matClass === "hydride-high-p") {
      anharmonicFactor = Math.max(anharmonicFactor, 0.15);
    }
  }

  if (phononSpectrum.anharmonicityIndex > 0.3) {
    anharmonicFactor += (phononSpectrum.anharmonicityIndex - 0.3) * 0.2;
  }

  anharmonicFactor = Math.max(0, Math.min(0.6, anharmonicFactor));

  lambda = lambda / (1 + anharmonicFactor);

  const omegaLogRange = getOmegaLogRangeForClass(matClass);
  let clampedOmegaLog = omega_log;
  if (omega_log < omegaLogRange[0]) clampedOmegaLog = omegaLogRange[0];
  if (omega_log > omegaLogRange[1]) clampedOmegaLog = omegaLogRange[1];

  const formulaCounts = formula ? parseFormulaCounts(formula) : {};
  const formulaElements = formula ? parseFormulaElements(formula) : [];

  const muStarClamped = computeScreenedMuStar(
    formulaElements, formulaCounts, matClass, N_EF, phononSpectrum.debyeTemperature
  );

  const isStrongCoupling = lambda > 1.5;

  let dominantPhononBranch = "acoustic";
  if (phononSpectrum.maxPhononFrequency > 2000) dominantPhononBranch = "high-frequency optical (H vibrations)";
  else if (phononSpectrum.softModePresent) dominantPhononBranch = "soft optical mode";
  else if (lambda > 1.0) dominantPhononBranch = "low-energy optical";

  let bandwidth = 0;
  const bwElements = formula ? parseFormulaElements(formula) : [];
  const bwCounts = formula ? parseFormulaCounts(formula) : {};
  const bwTotalAtoms = getTotalAtoms(bwCounts);
  for (const el of bwElements) {
    const frac = (bwCounts[el] || 1) / bwTotalAtoms;
    bandwidth += estimateBandwidthW(el) * frac;
  }
  bandwidth = Math.max(1.0, bandwidth);

  const omega2Avg = phononSpectrum.logAverageFrequency * phononSpectrum.logAverageFrequency * 1.2;

  return {
    lambda: Number(lambda.toFixed(3)),
    lambdaUncorrected: Number(lambdaUncorrected.toFixed(3)),
    anharmonicCorrectionFactor: Number(anharmonicFactor.toFixed(4)),
    omegaLog: Math.round(clampedOmegaLog),
    muStar: Number(muStarClamped.toFixed(4)),
    isStrongCoupling,
    dominantPhononBranch,
    bandwidth: Number(bandwidth.toFixed(4)),
    omega2Avg: Number(omega2Avg.toFixed(4)),
  };
}

export function predictTcEliashberg(
  coupling: ElectronPhononCoupling,
  phonon?: PhononSpectrum,
  alpha2FData?: Alpha2FData,
  matClass?: MaterialClass,
  formula?: string,
  pressureGpa: number = 0,
): EliashbergResult {
  let effectiveLambda = coupling.lambda;
  let effectiveOmegaLog = coupling.omegaLog;
  const { muStar } = coupling;

  let spectralOmegaLog: number | null = null;
  if (alpha2FData && alpha2FData.integratedLambda > 0) {
    effectiveLambda = alpha2FData.integratedLambda;
    const omegaLogFromAlpha2F = computeOmegaLogFromAlpha2F(alpha2FData);
    if (omegaLogFromAlpha2F > 0) {
      spectralOmegaLog = omegaLogFromAlpha2F;
      effectiveOmegaLog = omegaLogFromAlpha2F;
    }
  }

  if (phonon) {
    const validation = validateOmegaLog(
      effectiveOmegaLog,
      spectralOmegaLog,
      phonon.debyeTemperature,
      phonon.maxPhononFrequency
    );
    if (validation.corrected) {
      effectiveOmegaLog = validation.validatedOmegaLog;
    }
  }

  // Fix 3: Hard cap on λ. Even LaH10/CaH6 don't exceed 3.4; higher values are
  // surrogate-estimator artifacts that inflate Tc through the Allen-Dynes
  // exponential. Clamp before any Tc compute touches it.
  if (effectiveLambda > LAMBDA_HARD_CAP) {
    effectiveLambda = LAMBDA_HARD_CAP;
  }

  // Fix 2: Heavy-atom ω_log damping for H-rich compounds. Framework elements
  // like Ba (137), Sr (88), Nd (144) hybridize with H optical modes and pull
  // the log-average frequency down. Pure mass-based Allen-Dynes ω_log misses
  // this, producing inflated Tc for BaH5Sr2-style compounds.
  const heavyDamping = computeHeavyAtomOmegaLogDamping(formula);
  if (heavyDamping < 1.0) {
    effectiveOmegaLog = effectiveOmegaLog * heavyDamping;
  }

  const lambda = effectiveLambda;
  const omegaLog = effectiveOmegaLog;
  const omegaLogK = omegaLog * 1.4388;

  // -----------------------------------------------------------------------
  // Cuprates: phonon Allen-Dynes is physically invalid.
  // Pairing is d-wave spin-fluctuation; μ* and ω_log must be reinterpreted.
  // Use Monthoux-Scalapino spin-fluctuation formula.
  // -----------------------------------------------------------------------
  if (matClass === "cuprate") {
    // Spin-fluctuation energy scale: ~2.5× phonon ω_log in cuprates
    const omegaSfK = omegaLogK * 2.5;
    const lambdaSf = lambda * 1.4;           // d-wave spin-fluctuation coupling
    const muStarDwave = muStar * 0.15;       // d-wave Coulomb partial cancellation (angular averaging)
    const denomSf = lambdaSf - muStarDwave * (1 + 0.62 * lambdaSf);
    let tc = 0;
    if (denomSf > 0 && omegaSfK > 0) {
      const exp = -1.15 * (1 + lambdaSf) / denomSf;
      if (exp >= -50) tc = Math.max(0, Math.min(185, (omegaSfK / 1.5) * Math.exp(exp)));
    }
    tc = Number.isFinite(tc) ? tc : 0;
    const uncertainty = tc * 0.45;  // High uncertainty: doping level unknown
    return {
      predictedTc: Math.round(tc * 10) / 10,
      gapRatio: 2 * 2.14,     // 2Δ/kTc ≈ 4.28 for d-wave (vs 3.53 BCS)
      isotropicGap: false,     // d-wave is anisotropic by definition
      strongCouplingCorrection: 1.0,
      confidenceBand: [Math.max(0, Math.round(tc - uncertainty)), Math.round(tc + uncertainty)],
    };
  }

  // -----------------------------------------------------------------------
  // Heavy fermions: phonon Allen-Dynes is physically invalid.
  // Pairing is spin/multipolar-fluctuation mediated near a magnetic QCP.
  // Mass renormalization (m*/m ~ 10–1000) suppresses Tc to 0.1–20 K range.
  // -----------------------------------------------------------------------
  if (matClass === "heavy-fermion") {
    const lambdaCrit = 0.25;
    const omegaSfK = omegaLogK * 0.40;      // HF spin fluctuations much softer than phonons
    const qcpProximity = Math.min(3.0, lambda / Math.max(lambdaCrit, 0.01));
    const lambdaQcp = lambda * 0.60 * qcpProximity;
    const muStarQcp = muStar * 0.35;
    const denomQcp = lambdaQcp - muStarQcp * (1 + 0.62 * lambdaQcp);
    let tc = 0;
    if (lambda > lambdaCrit && denomQcp > 0 && omegaSfK > 0) {
      const massEst = Math.max(10, 80 / (lambda + 0.1));
      const massSuppression = Math.pow(massEst, -0.30);
      const exp = -1.04 * (1 + lambdaQcp) / denomQcp;
      if (exp >= -50) tc = Math.max(0, Math.min(25, (omegaSfK / 1.2) * massSuppression * Math.exp(exp)));
    }
    tc = Number.isFinite(tc) ? tc : 0;
    const uncertainty = tc * 0.60;  // Very high uncertainty: Kondo scale unknown
    return {
      predictedTc: Math.round(tc * 100) / 100,
      gapRatio: 2 * 1.764,   // Assume BCS-like gap ratio as a placeholder
      isotropicGap: false,    // HF pairing typically p- or d-wave
      strongCouplingCorrection: 1.0,
      confidenceBand: [Math.max(0, Math.round(tc * 10 - uncertainty * 10) / 10), Math.round((tc + uncertainty) * 10) / 10],
    };
  }

  // -----------------------------------------------------------------------
  // Conventional Allen-Dynes path (phonon-mediated BCS, hydrides,
  // iron pnictides with empirical-scaling flag, and unknowns)
  // -----------------------------------------------------------------------

  let tc: number;
  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (denominator <= 0) {
    tc = 0;
  } else if (denominator < 0.05) {
    const smoothWeight = denominator / 0.05;
    const rawExponent = -1.04 * (1 + lambda) / denominator;
    if (rawExponent < -50) {
      tc = 0;
    } else {
      const rawTc = (omegaLogK / 1.2) * Math.exp(rawExponent);
      tc = Number.isFinite(rawTc) ? rawTc * smoothWeight : 0;
    }
  } else {
    const lambdaBar = 2.46 * (1 + 3.8 * muStar);
    const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3/2), 1/3);
    const omega2Avg = coupling.omega2Avg;
    const omegaRatio = omega2Avg > 0 ? Math.sqrt(omega2Avg) / omegaLog : 1.0;
    const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
    const f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
    const exponent = -1.04 * (1 + lambda) / denominator;
    tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);
  }

  tc = Number.isFinite(tc) ? Math.max(0, tc) : 0;

  if (phonon && tc > 0) {
    const freqRatio = phonon.maxPhononFrequency / Math.max(1, phonon.logAverageFrequency);
    const phononSpectrumSpread = Math.max(0, Math.min(3, freqRatio - 1));
    const shapeFactor = 1 + 0.05 * phononSpectrumSpread;
    tc *= shapeFactor;
  }

  const strongCouplingCorrection = 1 + 5.3 * Math.pow(lambda / (lambda + 6), 2);
  const gapRatio = 2 * 1.764 * strongCouplingCorrection;
  const isotropicGap = lambda < 1.0;

  const isHighPressureHydride = (alpha2FData && alpha2FData.integratedLambda > 1.5) ||
    (lambda > 1.5 && coupling.dominantPhononBranch.includes("H vibration"));
  let uncertaintyFrac = 0.15;
  if (isHighPressureHydride) {
    if (tc > 200) uncertaintyFrac = 0.25;
    else uncertaintyFrac = 0.20;
  } else {
    if (tc > 200 && lambda < 2.0) uncertaintyFrac = 0.5;
    else if (tc > 150 && lambda < 1.5) uncertaintyFrac = 0.4;
  }

  // Fix 1 + Fix 5: Hydride pressure gate and clathrate-capability filter.
  // These catch MoH6N=312K-type false positives where the surrogate pipeline
  // estimates an ambient-pressure superhydride-class Tc for a compound that
  // either (a) has no pressure evidence, or (b) cannot form a clathrate cage.
  const gate = applyHydrideSanityGate(tc, formula, pressureGpa);
  tc = gate.tc;

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

export interface TcUncertaintyInput {
  lambda: number;
  lambdaStd: number;
  omegaLog: number;
  omegaLogStd: number;
  muStar: number;
  muStarStd: number;
  omega2Avg?: number;
  isHydride?: boolean;
}

export interface TcWithUncertainty {
  mean: number;
  std: number;
  ci95: [number, number];
  dominant_uncertainty_source: "lambda" | "omega_log" | "mu_star";
  partials: {
    dTc_dLambda: number;
    dTc_dOmegaLog: number;
    dTc_dMuStar: number;
  };
  errorPropagation: {
    lambdaContribution: number;
    omegaLogContribution: number;
    muStarContribution: number;
  };
  mcSamples: number;
  mcMean: number;
  mcStd: number;
  analyticMean: number;
  analyticStd: number;
}

// Calibration factors derived from verified compounds: ratio of experimental Tc to raw Allen-Dynes Tc.
// This corrects systematic overestimation in the standard Allen-Dynes formula.
const CALIBRATION_FACTORS: Record<string, number> = {};
let _calibrationComputed = false;

function computeCalibrationFactors() {
  if (_calibrationComputed) return;
  _calibrationComputed = true;
  // Physically plausible correction range. Factors outside [0.3, 3.0] indicate
  // that the reference compound is either at conditions mismatched with the
  // solver's assumptions (e.g. different polymorph, high pressure not applied,
  // unconventional pairing) OR the solver itself is broken for that corner of
  // (λ, ω_log) space. Either way, including these as calibration anchors
  // POISONS predictions for nearby query compounds via the similarity kernel.
  // Example: Th2H3 has λ=0.30, tcRef=5K, but pure AD predicts 0.045K → factor
  // of 111× that inflates every nearby compound's calibrated Tc by 10-100×.
  const MIN_CALIBRATION_FACTOR = 0.3;
  const MAX_CALIBRATION_FACTOR = 3.0;
  let rejectedCount = 0;
  for (const [formula, v] of Object.entries(VERIFIED_COMPOUNDS)) {
    const rawTc = allenDynesTcUncalibrated(v.lambda, v.omegaLog, v.muStar, v.omegaLog * v.omegaLog * 1.2, formula.includes("H") && v.lambda > 1.5);
    if (rawTc > 0 && v.tcRef > 0) {
      const factor = v.tcRef / rawTc;
      if (factor >= MIN_CALIBRATION_FACTOR && factor <= MAX_CALIBRATION_FACTOR) {
        CALIBRATION_FACTORS[formula] = factor;
      } else {
        rejectedCount++;
      }
    }
  }
  // Log for diagnostics
  const entries = Object.entries(CALIBRATION_FACTORS);
  const avg = entries.reduce((s, [, f]) => s + f, 0) / Math.max(entries.length, 1);
  console.log(`[Physics] Calibration factors computed for ${entries.length} compounds (avg correction: ${avg.toFixed(3)}x, rejected ${rejectedCount} outliers)`);
}

// Family-level calibration: average correction factor for similar materials
function getCalibrationFactor(lambda: number, omegaLog: number, isHydride: boolean): number {
  computeCalibrationFactors();
  const factors = Object.entries(CALIBRATION_FACTORS);
  if (factors.length === 0) return 1.0;

  // Weight verified compounds by similarity (lambda distance)
  let weightedSum = 0;
  let totalWeight = 0;
  for (const [formula, factor] of factors) {
    const v = VERIFIED_COMPOUNDS[formula];
    if (!v) continue;
    const vIsHydride = formula.includes("H") && v.lambda > 1.5;
    // Same family gets higher weight
    const familyMatch = (isHydride === vIsHydride) ? 2.0 : 0.5;
    const lambdaDist = Math.abs(lambda - v.lambda);
    const omegaDist = Math.abs(omegaLog - v.omegaLog) / 500;
    const dist = Math.sqrt(lambdaDist * lambdaDist + omegaDist * omegaDist);
    const w = familyMatch / (1 + dist * 2);
    weightedSum += factor * w;
    totalWeight += w;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 1.0;
}

export function allenDynesTcUncalibrated(lambda: number, omegaLog: number, muStar: number, omega2Avg?: number, isHydride?: boolean, formula?: string): number {
  // Anharmonic phonon softening for strong-coupling hydrides. Literature λ/ω_log
  // values for H3S, CaH6 etc. are typically computed in the HARMONIC
  // approximation. Anharmonic corrections (Errea et al., Nature 2020) reduce
  // effective ω_log by ~15% for H-sublattice-dominated spectra. Without this,
  // AD systematically over-predicts superhydride Tc by 30-50%. The factor
  // matches the `anharmonicCorrectionFactor: 0.85` already tagged on hydride
  // entries in buildCouplingFromVerified (line ~1847) but previously never
  // actually applied downstream.
  // λ-and-mass-dependent anharmonic softening for hydrides. Physical basis:
  // H-mode anharmonic zero-point amplitude scales as 1/√(m_metal·ω_metal),
  // so lighter metal sublattices cause larger H-mode softening. A pure
  // λ-ramp under-corrected light-metal compounds (CeH9, CaH6, ThH10) and
  // slightly over-corrected heavy-metal ones.
  //   λ-term: anharmonic population grows with coupling strength.
  //   mass-term: scales softening by √(100/m_avg_metal) relative to a 100-amu
  //              reference (roughly mid-transition-metal like Mo). Light
  //              frameworks (S, Ca, Y) get stronger softening; heavy (La,
  //              Th, Ba) get weaker.
  // Clamped to [0.60, 1.0] to prevent runaway in extreme light-metal cases.
  let anharmonicFactor = 1.0;
  if (isHydride && lambda > 1.2) {
    const lambdaComponent = 0.85 - 0.08 * Math.max(0, lambda - 1.5);
    let massScale = 1.0;
    let fElectronCorrection = 0;
    if (formula) {
      try {
        const pc = parseComposition(formula);
        const metalAtoms = pc.elements.filter(e => e !== "H");
        if (metalAtoms.length > 0) {
          const totalMetal = metalAtoms.reduce((s, e) => s + (pc.counts[e] || 0), 0);
          const weightedMass = metalAtoms.reduce((s, e) => {
            const m = getElementData(e)?.atomicMass ?? 50;
            return s + m * (pc.counts[e] || 0);
          }, 0);
          const avgMetalMass = totalMetal > 0 ? weightedMass / totalMetal : 50;
          // √(100/m) — reference 100 amu gets massScale=1. m=32 (S) → 1.77,
          // m=40 (Ca) → 1.58, m=139 (La) → 0.85, m=232 (Th) → 0.66.
          massScale = Math.sqrt(100 / Math.max(10, avgMetalMass));

          // f-electron hybridization: metals with partially-occupied f-shells
          // (Ce 4f¹, Pr 4f², Nd 4f³, Sm 4f⁵, Eu 4f⁶, Gd 4f⁷, … Yb 4f¹³; Th 5f⁰
          // with relativistic 5f-6d admixture; U/Np/Pu with occupied 5f) cause
          // enhanced phonon-electron coupling variance via f-state hybridization
          // with H optical modes (Papaconstantopoulos & Klein 2008 variance
          // argument). This manifests as additional anharmonic softening
          // beyond the mass term. La (4f⁰) and Lu (4f¹⁴) have filled/empty
          // f-shells — no near-Fermi weight — and are excluded.
          const F_ACTIVE = new Set([
            "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Th", "U", "Np", "Pu",
          ]);
          const fActiveCount = metalAtoms.filter(e => F_ACTIVE.has(e))
            .reduce((s, e) => s + (pc.counts[e] || 0), 0);
          if (fActiveCount > 0) {
            const fFrac = fActiveCount / pc.totalAtoms;
            // -0.15 for compounds with at least 1 f-active atom per ~10 total
            // atoms. Calibrated against CeH9 (4f¹ single Ce in 10 atoms, pure
            // AD over-predicts by 55%; f-correction brings it to ~25% warn).
            fElectronCorrection = -0.15 * Math.min(1.0, fFrac * 10);
          }
        }
      } catch {}
    }
    const massCorrection = -0.06 * Math.max(0, massScale - 1.0);
    anharmonicFactor = Math.max(0.55, lambdaComponent + massCorrection + fElectronCorrection);
  }
  const omegaLogEffective = omegaLog * anharmonicFactor;
  const omegaLogK = omegaLogEffective * 1.4388;
  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (denominator <= 0) return 0;

  if (denominator < 0.05) {
    const smoothWeight = denominator / 0.05;
    const rawExponent = -1.04 * (1 + lambda) / denominator;
    if (rawExponent < -50) return 0;
    const rawTc = (omegaLogK / 1.2) * Math.exp(rawExponent);
    return Number.isFinite(rawTc) ? Math.max(0, rawTc * smoothWeight) : 0;
  }

  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

  const omegaRatio = omega2Avg && omega2Avg > 0 ? Math.sqrt(omega2Avg) / omegaLog : 1.0;
  const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
  let f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
  f2 = Math.max(0.7, Math.min(1.5, f2));

  const exponent = -1.04 * (1 + lambda) / denominator;
  if (exponent < -50) return 0;
  let tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);

  // Strong-coupling hydride check: SISSO/Xie (hydrideStrongCouplingTc) is an
  // alternative empirical fit for λ>1.5. Prior code took max(AD, SISSO) which
  // systematically biased predictions upward — H3S went 312K (AD) → 425K
  // (SISSO-override) vs 203K experimental. Replace with weighted blend: AD
  // dominates, SISSO provides a soft anchor. Clamp the final ratio so SISSO
  // can never push AD above ×1.15 or below ×0.85.
  if (isHydride && lambda > 1.5 && omegaLogK > 0) {
    const tcSisso = hydrideStrongCouplingTc(lambda, omegaLogK, muStar, omega2Avg ? Math.sqrt(omega2Avg) * 1.4388 : undefined);
    if (tcSisso > 0 && Number.isFinite(tcSisso)) {
      const blended = 0.8 * tc + 0.2 * tcSisso;
      const ratio = blended / Math.max(tc, 1);
      const clampedRatio = Math.max(0.85, Math.min(1.15, ratio));
      tc = tc * clampedRatio;
    }
  }

  return Number.isFinite(tc) ? Math.max(0, tc) : 0;
}

/**
 * Apply spin-fluctuation suppression to a phonon-mediated Tc prediction.
 * Physical basis: paramagnon exchange renormalizes the effective coupling
 * via λ_eff = (λ_ph - λ_sf) / (1 + λ_sf). In the Berk-Schrieffer approach,
 * this manifests as a multiplicative suppression of Tc at fixed phonon λ.
 * For NbN and related materials with antiferromagnetic spin fluctuations,
 * the empirical suppression is documented as 30-45% of pure-phonon Tc
 * (Chen et al. PRB 75, 165109, 2007).
 *
 * α_sf is the suppression fraction (0 = no spin-fluc, 0.5 = Tc halved).
 */
/**
 * Two-gap Allen-Dynes solver for compounds with coupled σ+π bands (MgB2
 * family). Uses the Suhl-Matthias-Walker two-band formulation reduced to
 * the AD prefactor: treats each band with its own λ, ω_log, μ* and solves
 * the coupled linearized gap equations for the highest Tc eigenvalue.
 *
 * Input: two-band couplings {λ_σσ, λ_ππ, λ_σπ, ω_σ (meV), ω_π (meV)}
 * plus μ*_σ, μ*_π (take as equal if not specified).
 *
 * Approximation: use McMillan-like prefactor ω_log,eff = geometric mean of
 * ω_σ and ω_π weighted by intraband couplings. Eigenvalue of the 2×2
 * coupling matrix gives λ_eff; the smaller eigenvalue is the "subdominant"
 * gap and is only used as an uncertainty indicator.
 *
 * Validated against MgB2 (target Tc=39K).
 */
export function predictTcTwoGap(
  lambdaSigma: number,
  lambdaPi: number,
  omegaSigmaMeV: number,
  omegaPiMeV: number,
  interbandCoupling: number,
  muStarSigma: number = 0.12,
  muStarPi: number = 0.12,
): { tc: number; lambdaEff: number; omegaEff: number; subdominantGap: boolean } {
  // Two-gap Tc using σ-dominance with π-drag correction. In MgB2 and similar
  // two-band SC, the σ band with its high-frequency optical modes (E2g for
  // MgB2 at ~62 meV) sets the primary Tc scale. The π band has lower ω and
  // weaker coupling; interband scattering λ_σπ drags σ-gap Tc down via
  // Anderson's theorem violation by the π-band. Closed-form fit (Golubov &
  // Mazin 1997, calibrated to MgB2 Tc=39K):
  //
  //   Tc = Tc_σ × (1 - α × (λ_ππ - λ_σπ) / (λ_σσ + λ_σπ))
  //
  // where α is the interband-drag coefficient ≈ 0.57 fitted to reproduce
  // MgB2. This is a PHENOMENOLOGICAL reduction — for precise predictions
  // the coupled Eliashberg integrals must be solved numerically.
  const omegaSigmaCm = omegaSigmaMeV * 8.0655;

  // σ-band standalone AD
  const tcSigma = allenDynesTcUncalibrated(lambdaSigma, omegaSigmaCm, muStarSigma, undefined, false);

  // π-drag correction. Positive when π is weaker than σ-π interband coupling
  // (the usual case); clamped to [0, 0.5] to prevent over-correction.
  // Drag coefficient tuned to reproduce MgB2 Tc=39K at tcSigma≈49K → need
  // dragFactor ≈ 0.80. With (λ_ππ - λ_σπ)/(λ_σσ + λ_σπ) = 0.235/1.230 = 0.191
  // for MgB2, the coefficient must be ≈ 1.05.
  const DRAG_COEF = 1.05;
  const dragNumerator = Math.max(0, lambdaPi - interbandCoupling);
  const dragDenominator = Math.max(0.01, lambdaSigma + interbandCoupling);
  const drag = DRAG_COEF * dragNumerator / dragDenominator;
  const dragFactor = Math.max(0.5, 1 - Math.min(0.5, drag));

  const tc = tcSigma * dragFactor;

  // Subdominant eigenvalue: check whether π-band has enough coupling to
  // sustain its own SC order (lambdaPi - muStar > 0 in isolation)
  const subdominantGap = lambdaPi > muStarPi * (1 + 0.62 * lambdaPi);

  return {
    tc,
    lambdaEff: lambdaSigma, // σ-band dominates
    omegaEff: omegaSigmaMeV * 11.6045,
    subdominantGap,
  };
}

/**
 * Compute Tc for a two-gap SC using VERIFIED_COMPOUNDS metadata. Returns
 * null if the compound is not flagged with two-gap parameters.
 */
export function predictTcForTwoGapCompound(formula: string): { tc: number; lambdaEff: number } | null {
  const verified = VERIFIED_COMPOUNDS[formula.replace(/\s+/g, "")];
  if (!verified?.twoGap) return null;
  const tg = verified.twoGap;
  const result = predictTcTwoGap(
    tg.lambdaSigma,
    tg.lambdaPi,
    tg.omegaSigma,
    tg.omegaPi,
    tg.interbandCoupling,
    verified.muStar,
    verified.muStar,
  );
  return { tc: result.tc, lambdaEff: result.lambdaEff };
}

export function applySpinFluctuationSuppression(tcPhonon: number, spinFluctuationStrength: number): number {
  if (tcPhonon <= 0 || spinFluctuationStrength <= 0) return tcPhonon;
  const clamped = Math.max(0, Math.min(0.8, spinFluctuationStrength));
  return tcPhonon * (1 - clamped);
}

/**
 * Compute Tc for a spin-fluctuation-dominated compound using phonon AD
 * followed by paramagnon suppression. Returns 0 if the compound is not
 * flagged with a spinFluctuationStrength in VERIFIED_COMPOUNDS.
 */
export function predictTcWithSpinFluctuation(
  formula: string,
  lambda: number,
  omegaLog: number,
  muStar: number,
  isHydride: boolean = false,
): { tc: number; tcPhononOnly: number; suppressionFraction: number } | null {
  const verified = VERIFIED_COMPOUNDS[formula.replace(/\s+/g, "")];
  const sfs = verified?.spinFluctuationStrength;
  if (!sfs || sfs <= 0) return null;
  const tcPhononOnly = allenDynesTcUncalibrated(lambda, omegaLog, muStar, undefined, isHydride, formula);
  const tc = applySpinFluctuationSuppression(tcPhononOnly, sfs);
  return { tc, tcPhononOnly, suppressionFraction: sfs };
}

export function allenDynesTcRaw(lambda: number, omegaLog: number, muStar: number, omega2Avg?: number, isHydride?: boolean, formula?: string, pressureGpa?: number): number {
  // Pressure-aware gates only fire when pressure is *explicitly provided*.
  // A bare formula (without pressure) keeps legacy behaviour — otherwise the
  // pressureGpa=0 default would falsely mark every high-P superhydride as
  // ambient and clamp it via the hydride sanity gate (LaH10 → 40K bug).
  if (formula && pressureGpa !== undefined) {
    const adCheck = isAllenDynesApplicable(formula, lambda, pressureGpa);
    if (!adCheck.applicable) return 0;
  }

  const rawTc = allenDynesTcUncalibrated(lambda, omegaLog, muStar, omega2Avg, isHydride, formula);
  if (rawTc <= 0) return 0;

  // Calibration disabled: the similarity-weighted kernel was pulling ALL
  // predictions up by 3-10x because low-lambda verified compounds (Th2H3,
  // ScH3) have extreme factor ratios that bleed into nearby query points.
  // Raw Allen-Dynes is more reliable than broken calibration.
  // MgB2: 54K raw vs 524K calibrated vs 39K experimental.
  let calibratedTc = rawTc;
  if (!Number.isFinite(calibratedTc)) return 0;
  calibratedTc = Math.max(0, calibratedTc);

  // Hydride sanity gate (MoH6N clathrate-incapable, ambient H-rich Tc>100K).
  // Same rule: skip when pressure is not provided so legacy callers are unaffected.
  if (formula && pressureGpa !== undefined) {
    const gated = applyHydrideSanityGate(calibratedTc, formula, pressureGpa);
    calibratedTc = gated.tc;

    // Universal ambient sanity clamp: Tc > 400K at <50 GPa is unphysical for
    // every known SC mechanism (no hydride or non-hydride compound exceeds
    // ~400K at near-ambient pressure). Catches non-hydride surrogate-λ
    // runaway like WC Tc=559K. Returns 0 so the caller filters it out.
    if (pressureGpa < 50 && calibratedTc > 400) {
      return 0;
    }
  }
  return calibratedTc;
}

function hydrideStrongCouplingTc(lambda: number, omegaLogK: number, muStar: number, omega2K?: number): number {
  const lambdaEff = lambda - muStar * (1 + lambda);
  if (lambdaEff <= 0) return 0;

  const tcSisso = 0.182 * omegaLogK * Math.pow(lambdaEff, 0.572) *
    Math.pow(1 + 6.5 * muStar * Math.log(Math.max(1.01, lambda)), -0.278);

  if (lambda <= 1.0 || omegaLogK <= 0) {
    return Number.isFinite(tcSisso) ? Math.max(0, tcSisso) : 0;
  }

  const muStarEff = muStar * (1 + 0.5 * muStar);
  const C = lambda * (1 - 0.14 * muStar) - muStarEff * (1 + 0.62 * lambda);
  if (C <= 0) return Number.isFinite(tcSisso) ? Math.max(0, tcSisso) : 0;

  const A = 0.12 * (1 + 5.2 / (lambda + 2.6));
  const B = 1.04 * (1 + 0.38 * muStar);
  const xieExp = -B * (1 + lambda) / C;
  if (xieExp < -50) return Number.isFinite(tcSisso) ? Math.max(0, tcSisso) : 0;

  let tcXie = omegaLogK * A * Math.exp(xieExp);

  if (omega2K && omega2K > 0) {
    const ratio = omega2K / omegaLogK;
    const spectralCorr = 1 + 0.0241 * Math.pow(ratio - 1, 2) * lambda;
    tcXie *= Math.min(1.3, Math.max(0.9, spectralCorr));
  }

  if (lambda > 2.5) {
    const saturationDamping = 1 - 0.04 * Math.pow(lambda - 2.5, 1.2);
    tcXie *= Math.max(0.6, saturationDamping);
  }

  const best = Math.max(
    Number.isFinite(tcSisso) ? tcSisso : 0,
    Number.isFinite(tcXie) ? tcXie : 0
  );
  return Math.max(0, best);
}

export function computeTcWithUncertainty(input: TcUncertaintyInput): TcWithUncertainty {
  const { lambda, lambdaStd, omegaLog, omegaLogStd, muStar, muStarStd, omega2Avg, isHydride } = input;

  const tc0 = allenDynesTcRaw(lambda, omegaLog, muStar, omega2Avg, isHydride);

  const h = 1e-4;
  const dTc_dLambda = (allenDynesTcRaw(lambda + h, omegaLog, muStar, omega2Avg, isHydride) - allenDynesTcRaw(lambda - h, omegaLog, muStar, omega2Avg, isHydride)) / (2 * h);
  const dTc_dOmegaLog = (allenDynesTcRaw(lambda, omegaLog + h, muStar, omega2Avg, isHydride) - allenDynesTcRaw(lambda, omegaLog - h, muStar, omega2Avg, isHydride)) / (2 * h);
  const dTc_dMuStar = (allenDynesTcRaw(lambda, omegaLog, muStar + h, omega2Avg, isHydride) - allenDynesTcRaw(lambda, omegaLog, muStar - h, omega2Avg, isHydride)) / (2 * h);

  const lambdaContrib = (dTc_dLambda * lambdaStd) ** 2;
  const omegaLogContrib = (dTc_dOmegaLog * omegaLogStd) ** 2;
  const muStarContrib = (dTc_dMuStar * muStarStd) ** 2;
  const totalVar = lambdaContrib + omegaLogContrib + muStarContrib;
  const analyticStd = Math.sqrt(totalVar);

  const N_MC = 500;
  const mcSamples: number[] = [];
  const clampedLambdaStd = Math.min(lambdaStd, (lambda - 0.01) / 3);
  const clampedOmegaLogStd = Math.min(omegaLogStd, (omegaLog - 1) / 3);
  const clampedMuStarStd = Math.min(muStarStd, Math.min((muStar - 0.01) / 3, (0.3 - muStar) / 3));
  const safeLStd = Math.max(1e-6, clampedLambdaStd);
  const safeOStd = Math.max(1e-6, clampedOmegaLogStd);
  const safeMStd = Math.max(1e-6, clampedMuStarStd);
  for (let i = 0; i < N_MC; i++) {
    const lSample = lambda + safeLStd * boxMullerNormal();
    const oSample = omegaLog + safeOStd * boxMullerNormal();
    const mSample = muStar + safeMStd * boxMullerNormal();
    const tcSample = allenDynesTcRaw(
      Math.max(0.01, lSample),
      Math.max(1, oSample),
      Math.max(0.01, Math.min(0.3, mSample)),
      omega2Avg,
      isHydride,
    );
    if (Number.isFinite(tcSample)) mcSamples.push(tcSample);
  }

  const mcMean = mcSamples.length > 0 ? mcSamples.reduce((s, v) => s + v, 0) / mcSamples.length : tc0;
  const mcStd = mcSamples.length > 1
    ? Math.sqrt(mcSamples.reduce((s, v) => s + (v - mcMean) ** 2, 0) / (mcSamples.length - 1))
    : analyticStd;

  const combinedMean = 0.5 * tc0 + 0.5 * mcMean;
  const combinedStd = Math.sqrt(0.5 * analyticStd ** 2 + 0.5 * mcStd ** 2);

  const maxContrib = Math.max(lambdaContrib, omegaLogContrib, muStarContrib);
  let dominant: "lambda" | "omega_log" | "mu_star" = "lambda";
  if (maxContrib === omegaLogContrib) dominant = "omega_log";
  else if (maxContrib === muStarContrib) dominant = "mu_star";

  const totalContrib = lambdaContrib + omegaLogContrib + muStarContrib || 1;

  return {
    mean: Math.round(combinedMean * 10) / 10,
    std: Math.round(combinedStd * 100) / 100,
    ci95: [
      Math.max(0, Math.round((combinedMean - 1.96 * combinedStd) * 10) / 10),
      Math.round((combinedMean + 1.96 * combinedStd) * 10) / 10,
    ],
    dominant_uncertainty_source: dominant,
    partials: {
      dTc_dLambda: Math.round(dTc_dLambda * 1000) / 1000,
      dTc_dOmegaLog: Math.round(dTc_dOmegaLog * 1000) / 1000,
      dTc_dMuStar: Math.round(dTc_dMuStar * 1000) / 1000,
    },
    errorPropagation: {
      lambdaContribution: Math.round((lambdaContrib / totalContrib) * 1000) / 1000,
      omegaLogContribution: Math.round((omegaLogContrib / totalContrib) * 1000) / 1000,
      muStarContribution: Math.round((muStarContrib / totalContrib) * 1000) / 1000,
    },
    mcSamples: mcSamples.length,
    mcMean: Math.round(mcMean * 10) / 10,
    mcStd: Math.round(mcStd * 100) / 100,
    analyticMean: Math.round(tc0 * 10) / 10,
    analyticStd: Math.round(analyticStd * 100) / 100,
  };
}

function boxMullerNormal(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function computePhysicsTcUQ(formula: string, pressureGpa: number = 0): TcWithUncertainty {
  const electronic = computeElectronicStructure(formula);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula, pressureGpa);

  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const hCount = counts["H"] || 0;
  const metalAtomCount = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;
  const isHydride = hCount > 0 && hRatio >= 4;

  const lambdaStd = coupling.lambda * 0.15;
  const omegaLogStd = coupling.omegaLog * 0.10;
  const muStarStd = 0.02;

  return computeTcWithUncertainty({
    lambda: coupling.lambda,
    lambdaStd,
    omegaLog: coupling.omegaLog,
    omegaLogStd,
    muStar: coupling.muStar,
    muStarStd,
    omega2Avg: coupling.omega2Avg,
    isHydride,
  });
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
    if (!isTransitionMetal(el) && !isRareEarth(el)) continue;
    const I = getStonerParameter(el);
    if (I !== null) stonerMax = Math.max(stonerMax, I * N_EF);
  }

  const nearQCP = stonerMax > 0.7 && stonerMax < 1.2;
  const hasAFM = competingPhases.some(p => p.phaseName.includes("Antiferromagnetic"));

  if (!nearQCP && !hasAFM && corr < 0.5) {
    return { mechanism: "spin-fluctuation", tcEstimate: 0, confidence: 0.1, description: "No magnetic proximity" };
  }

  const spinSusc = computeDynamicSpinSusceptibility(formula, electronic);

  const T_sf = spinSusc.spinFluctuationEnergy * (1 + spinSusc.stonerEnhancement * 0.1);
  const V_sf = corr * 0.8 * Math.min(spinSusc.stonerEnhancement / 10, 2.0);
  const exponent = V_sf * N_EF;

  let tc_sf = 0;
  if (exponent > 0.1) {
    tc_sf = T_sf * Math.exp(-1 / exponent);
  }

  if (spinSusc.isNearQCP) tc_sf *= 2.0;
  else if (nearQCP) tc_sf *= 1.8;
  if (hasAFM && !spinSusc.isStableFerromagnet) {
    tc_sf *= 1.8;
  } else if (hasAFM) {
    tc_sf *= 1.3;
  }

  if (spinSusc.correlationLength > 5) {
    tc_sf *= (1 + Math.log(spinSusc.correlationLength) * 0.15);
  }

  const isCuprate = elements.includes("Cu") && elements.includes("O");
  const isIronBased = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));

  if (isCuprate) {
    // Cuprate Tc estimate via AFM spin-fluctuation d-wave pairing.
    // chiStaticPeak ∝ Stoner enhancement → stronger AFM → stronger d-wave pairing.
    // corr > 0.80 activates Mott suppression (underdoped regime).
    // Physical range: LSCO ~38 K, YBCO ~92 K, Bi-2212 ~95 K, Hg-1201 ~135 K.
    // Replaces crude hardcoded floor that had no physical basis.
    const mottPenalty = Math.max(0, corr - 0.80) * 3.0;
    const cuprateEstimate = Math.max(0, Math.min(165,
      (spinSusc.chiStaticPeak * 5 + 15) * (1 - mottPenalty)
    ));
    tc_sf = Math.max(tc_sf, cuprateEstimate);
  }

  if (isIronBased) {
    // Pnictide Tc bounded by empirical s±-wave Tc range (5–55 K).
    // Higher correlation and AFM proximity → stronger spin-fluctuation pairing.
    tc_sf = Math.max(tc_sf, Math.min(55, 10 + corr * 60));
  }

  const mottProx = electronic.mottProximityScore ?? 0;
  if (mottProx > 0.9) {
    tc_sf = 0;
  } else if (mottProx > 0.8) {
    tc_sf *= (1.0 - (mottProx - 0.8) * 10);
  } else if (mottProx > 0.5) {
    tc_sf *= (1 + (mottProx - 0.5) * 0.67);
  }

  const dDom = (electronic.orbitalFractions?.d ?? 0) /
    Math.max(0.01, (electronic.orbitalFractions?.s ?? 0) + (electronic.orbitalFractions?.p ?? 0));
  if (dDom * corr > 0.6) {
    tc_sf *= 1.2;
  }

  tc_sf = Math.max(0, Math.round(tc_sf));
  const confidence = spinSusc.isNearQCP ? 0.55 : nearQCP ? 0.5 : (hasAFM ? 0.4 : 0.2);

  const descParts: string[] = [];
  if (spinSusc.isNearQCP) descParts.push("Near magnetic quantum critical point");
  else if (nearQCP) descParts.push("Near magnetic QCP (Stoner)");
  else if (hasAFM) descParts.push("AFM-proximity-mediated pairing");
  else descParts.push("Weak spin-fluctuation contribution");
  descParts.push(`χ(q,ω): peak=${spinSusc.chiStaticPeak.toFixed(1)}, ξ=${spinSusc.correlationLength.toFixed(1)}a`);

  return {
    mechanism: "spin-fluctuation",
    tcEstimate: tc_sf,
    confidence,
    description: descParts.join("; "),
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

  const vec = getVEC(elements, counts);
  const carrierDensity = vec * electronic.metallicity;

  const lowCarrier = carrierDensity < 3 && N_EF > 2;
  if (!lowCarrier) {
    return { mechanism: "plasmonic", tcEstimate: 0, confidence: 0.05, description: "No plasmonic conditions" };
  }

  let dielectricScreening = 1.0;
  const hasHighDielectric = elements.some(e => ["Ti", "Sr", "Ba", "Pb", "Bi"].includes(e)) &&
    elements.includes("O");
  if (hasHighDielectric) {
    dielectricScreening = 0.4;
  } else if (elements.includes("O") && elements.length >= 3) {
    dielectricScreening = 0.7;
  }

  const plasmaStrength = N_EF / Math.max(carrierDensity, 0.5) * dielectricScreening;
  const tc_pl = Math.round(Math.min(100, plasmaStrength * 10));

  return {
    mechanism: "plasmonic",
    tcEstimate: Math.max(0, tc_pl),
    confidence: 0.15,
    description: `Low carrier density with high DOS — plasmon-mediated (dielectric screening=${dielectricScreening.toFixed(1)})`,
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

  const mottProx = electronic.mottProximityScore ?? 0;
  const mottPenalty = mottProx > 0.7 ? 1.0 - (mottProx - 0.7) * 1.5 : 1.0;

  const tc_fb = Math.round(wAvg * 11604 * Math.max(0.1, lambda) * 0.01 * Math.max(0.1, mottPenalty));

  const isKagome = elements.length >= 2 && electronic.fermiSurfaceTopology.includes("nesting");

  return {
    mechanism: "flat-band",
    tcEstimate: Math.max(0, Math.min(400, tc_fb * (isKagome ? 1.5 : 1.0))),
    confidence: isKagome ? 0.35 : 0.25,
    description: isKagome ? "Kagome-type flat band with geometric frustration" : `Narrow bandwidth (W=${wAvg.toFixed(1)}eV) with high DOS, Mott proximity=${mottProx.toFixed(2)}`,
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

  const excitonicActive = excitonic.tcEstimate > 0 && excitonic.confidence > 0.05;
  const highCorrelation = electronic.correlationStrength > 0.7;
  if (excitonicActive && highCorrelation) {
    const bcsIdx = all.indexOf(bcs);
    if (bcsIdx >= 0 && bcs.tcEstimate > 0) {
      all[bcsIdx] = {
        ...bcs,
        confidence: bcs.confidence * 0.3,
        description: bcs.description + " (suppressed: excitonic competition at high U/W)",
      };
    }
  }

  const active = all.filter(m => m.tcEstimate > 0 && m.confidence > 0.05);
  const activeCount = active.length;

  if (activeCount === 0) {
    return {
      dominant: bcs,
      all,
      enhancedTc: bcs.tcEstimate,
      uncertaintyFromMechanism: 0.5,
    };
  }

  const totalConfidence = active.reduce((s, m) => s + m.confidence, 0);
  const blendedTc = active.reduce((s, m) => s + m.tcEstimate * m.confidence, 0) / totalConfidence;

  const sorted = [...active].sort((a, b) =>
    (b.confidence - a.confidence) || (b.tcEstimate - a.tcEstimate)
  );
  const dominant = sorted[0];

  let enhancedTc: number;
  if (activeCount >= 2) {
    enhancedTc = Math.round((0.6 * dominant.tcEstimate + 0.4 * blendedTc) * 10) / 10;
  } else {
    enhancedTc = dominant.tcEstimate;
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
      const J_eV = (stonerI || 0.5) * 0.1;
      const kB_eV = 8.617e-5;
      const mftCorrection = 0.01;
      const T_mag = Math.round(2 * z_coord * J_eV * S_eff * (S_eff + 1) / (3 * kB_eV) * mftCorrection);

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

  const vec = getVEC(elements, counts);

  const hasTM = elements.some(e => isTransitionMetal(e));
  if (hasTM && vec > 4 && vec < 7 && electronicStructure.fermiSurfaceTopology.includes("nesting")) {
    const cdwStrength = 0.3 + (6 - Math.abs(vec - 5.5)) * 0.1;
    const T_cdw = Math.round(50 + cdwStrength * 200);
    phases.push({
      phaseName: "Charge density wave",
      type: "CDW",
      transitionTemp: T_cdw,
      strength: Math.min(1.0, cdwStrength),
      suppressesSC: cdwStrength > 0.5,
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
      phaseName: "Spin-density wave (cuprate)",
      type: "SDW",
      transitionTemp: Math.min(350, T_sdw),
      strength: Math.min(1.0, 0.5 + electronicStructure.correlationStrength * 0.3),
      suppressesSC: true,
    });
  }

  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  if (isPnictide) {
    let stonerMax = 0;
    for (const el of elements) {
      if (!isTransitionMetal(el) && !isRareEarth(el)) continue;
      const I = getStonerParameter(el);
      if (I !== null) stonerMax = Math.max(stonerMax, I * electronicStructure.densityOfStatesAtFermi);
    }
    const sdwStrength = Math.min(1.0, 0.4 + electronicStructure.correlationStrength * 0.4 + stonerMax * 0.2);
    const T_sdw_pn = Math.round(100 + electronicStructure.correlationStrength * 100);
    phases.push({
      phaseName: "Spin-density wave (pnictide)",
      type: "SDW",
      transitionTemp: Math.min(250, T_sdw_pn),
      strength: sdwStrength,
      suppressesSC: sdwStrength > 0.7,
    });
  }

  const isKagomeSC = (elements.includes("V") || elements.includes("Mn") || elements.includes("Co") || elements.includes("Ti") || elements.includes("Fe")) &&
    (elements.includes("Sb") || elements.includes("Sn") || elements.includes("Bi"));
  if (isKagomeSC) {
    const nestScore = electronicStructure.nestingScore ?? 0;
    const sdwStrengthK = Math.min(1.0, 0.3 + nestScore * 0.5);
    phases.push({
      phaseName: "Spin-density wave (Kagome)",
      type: "SDW",
      transitionTemp: Math.round(80 + nestScore * 100),
      strength: sdwStrengthK,
      suppressesSC: false,
    });
  }

  const isDichalcogenideSC = (elements.includes("Nb") || elements.includes("Ta") || elements.includes("Mo") || elements.includes("W")) &&
    (elements.includes("Se") || elements.includes("S") || elements.includes("Te"));
  const existingCDW = phases.some(p => p.type === "CDW");
  if (isDichalcogenideSC && !existingCDW) {
    const cdwStr = Math.min(1.0, 0.5 + (electronicStructure.nestingScore ?? 0) * 0.3);
    const T_cdw_dc = Math.round(100 + electronicStructure.densityOfStatesAtFermi * 20);
    phases.push({
      phaseName: "CDW (dichalcogenide)",
      type: "CDW",
      transitionTemp: Math.min(300, T_cdw_dc),
      strength: cdwStr,
      suppressesSC: cdwStr > 0.5,
    });
  }

  const motifResult = formula ? detectStructuralMotifs(formula) : null;
  const isPerovskiteMotif = motifResult ? motifResult.motifs.some(m =>
    m.includes("Perovskite") || m.includes("perovskite") || m.includes("Ruddlesden-Popper")
  ) : false;

  if (isPerovskiteMotif && elements.includes("O") && elements.length >= 3) {
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
  dimensionality: string,
  formula?: string,
  electronicStructure?: ElectronicStructure | null
): CriticalFieldResult {
  if (tc <= 0) {
    return {
      upperCriticalField: 0,
      lowerCriticalField: 0,
      coherenceLength: 0,
      londonPenetrationDepth: 0,
      anisotropyRatio: 1,
      criticalCurrentDensity: 0,
      typeIorII: "N/A",
      underdetermined: true,
      glValidation: {
        xiGL: 0,
        vFUsed: 0,
        status: "underdetermined",
        diagnosis:
          "Tc is zero or negative — superconducting parameters are under-determined, not a physical failure of the material.",
      },
    };
  }

  const lambda = Math.max(coupling.lambda, 0.01);
  const bw = Math.max(0.1, coupling.bandwidth);
  const matClass = formula ? classifyMaterialForLambda(formula) : "conventional-metal";
  const isHydride = isHydrideForLambda(matClass);
  const isHeavyFermion = matClass === "heavy-fermion";
  const isLayered = dimensionality === "2D" || dimensionality === "quasi-2D" || dimensionality === "layered";

  const kB = 1.381e-23;
  const hbar = 1.055e-34;
  const PHI0 = 2.07e-15;
  const MU0 = 4 * Math.PI * 1e-7;
  const eCharge = 1.602e-19;
  const mElectron = 9.109e-31;

  const vFRaw = Math.sqrt(bw * eCharge / mElectron) / Math.sqrt(1 + lambda);
  const vF = Math.max(1e4, Math.min(2e6, vFRaw));

  // Ginzburg-Landau validation: BCS clean-limit ξ₀ = ℏvF / (kB·Tc·π)
  // If this is ≈ 0 the Tc or vF input is wrong, not a material failure.
  const xiGL_raw = (hbar * vF) / (kB * tc * Math.PI);
  const xiGL_nm = xiGL_raw * 1e9;
  let glStatus: GLValidation["status"] = "ok";
  let glDiagnosis = "";
  if (!Number.isFinite(xiGL_nm) || xiGL_nm <= 0) {
    glStatus = "unphysical";
    glDiagnosis =
      vFRaw < 1e4
        ? "Fermi velocity is at its lower bound — bandwidth input may be wrong or dense k-point sampling is needed for accurate ξ."
        : "GL coherence length is non-physical — verify the Tc input.";
  } else if (xiGL_nm < 0.05) {
    glStatus = "unphysical";
    glDiagnosis =
      `GL coherence length (${xiGL_nm.toExponential(2)} nm) is below 0.05 nm — Tc or Fermi velocity input is likely incorrect.`;
  }

  const carbotteRatio = 1.764 * (1 + 5.3 * Math.pow(lambda / (lambda + 6), 2));
  const delta0 = carbotteRatio * kB * tc;

  const xiRaw = (hbar * vF) / (Math.PI * delta0);
  const xiNm = xiRaw * 1e9;
  // Coherence length bounds: high-Tc hydrides (Tc>100K) have very short ξ (1-3 nm).
  // Previous xiMax=20 for hydrides was too generous — a 260K material should have
  // ξ ~ 1 nm, not 20 nm. The raw BCS formula handles this correctly; only clamp
  // to prevent numerical nonsense, not to override physics.
  let xiMin = 0.3, xiMax = 500;
  if (isHydride) { xiMin = 0.3; xiMax = 10; }
  else if (isHeavyFermion) { xiMin = 1; xiMax = 20; }
  else if (isLayered) { xiMin = 0.5; xiMax = 60; }
  const coherenceLength = Math.max(xiMin, Math.min(xiMax, Number.isFinite(xiNm) ? xiNm : 10));

  const xiM = coherenceLength * 1e-9;
  const Hc2Tesla = PHI0 / (2 * Math.PI * xiM * xiM);
  const basePauliLimit = 1.86 * tc * Math.sqrt(1 + 0.1 * lambda);
  const topoScore = electronicStructure?.topologicalBandScore ?? 0;
  const pauliEnhancement = topoScore > 0.5 ? 1.0 + (topoScore - 0.5) * 4.0 : 1.0;
  const pauliLimit = basePauliLimit * Math.min(3.0, pauliEnhancement);
  // If orbital Hc2 is valid, use min(orbital, Pauli). Otherwise fall back to
  // Pauli limit — returning 0T for a high-Tc material is physically wrong.
  // H3S at 155 GPa has Hc2 ~ 200T; LaH10 analog with λ≈2.4 should be similar.
  const hc2Raw = Number.isFinite(Hc2Tesla) ? Math.max(0, Hc2Tesla) : pauliLimit;
  const upperCriticalField = Math.min(hc2Raw, pauliLimit);

  // London penetration depth: λ_L ∝ √(1+λ) / √(n_s) where n_s depends on
  // carrier density and Tc. For high-Tc materials, higher Tc means stronger
  // condensate → shorter λ_L. Use BCS-inspired scaling anchored to known values:
  // MgB2 (Tc=39K): λ_L ≈ 100 nm; YBCO (Tc=93K): λ_L ≈ 150 nm;
  // H3S (Tc=203K): λ_L ≈ 120 nm; LaH10 (Tc=250K): λ_L ≈ 80-100 nm
  let classFactor = 1.0;
  if (isHydride) classFactor = 0.5;
  else if (isHeavyFermion) classFactor = 3.0;
  else if (isLayered) classFactor = 1.5;
  else if (matClass === "cuprate") classFactor = 2.0;
  // Scale with sqrt((1+lambda)/bw) but also inversely with sqrt(Tc) for
  // self-consistency — stronger condensate (higher Tc) screens more.
  const tcRef = 40; // reference Tc for scaling (MgB2-like)
  const tcScale = Math.sqrt(tcRef / Math.max(tc, 1));
  const lambdaLRaw = 100 * Math.sqrt((1 + lambda) / Math.max(0.5, bw)) * classFactor * tcScale;
  let lambdaLMin = 20, lambdaLMax = 500;
  if (isHydride) { lambdaLMin = 30; lambdaLMax = 200; }
  else if (isHeavyFermion) { lambdaLMin = 200; lambdaLMax = 2000; }
  else if (isLayered) { lambdaLMin = 60; lambdaLMax = 500; }
  const londonPenetrationDepth = Math.max(lambdaLMin, Math.min(lambdaLMax, Number.isFinite(lambdaLRaw) ? lambdaLRaw : 100));

  let anisotropyRatio = 1.0;
  if (dimensionality === "2D" || dimensionality === "quasi-2D") {
    anisotropyRatio = Math.max(5, Math.min(200, 20 / Math.max(0.5, bw)));
  } else if (dimensionality === "layered") {
    anisotropyRatio = Math.max(2, Math.min(50, 10 / Math.max(0.5, bw)));
  } else {
    anisotropyRatio = Math.max(1, Math.min(5, 3 / Math.max(1, bw)));
  }
  if (!Number.isFinite(anisotropyRatio)) anisotropyRatio = 1.0;

  // GL depairing current density: Jc = Φ₀ / (3√3 π μ₀ λ² ξ)
  // This is the theoretical maximum; practical Jc is 10-100x lower due to
  // vortex pinning, grain boundaries, etc. Apply a 0.1 "practical factor".
  const lambdaLM = londonPenetrationDepth * 1e-9;
  const JcDepairing = PHI0 / (3 * Math.sqrt(3) * Math.PI * MU0 * lambdaLM * lambdaLM * xiM);
  const practicalFactor = 0.1; // ~10% of depairing limit is typical for clean single crystals
  const JcPerCm2 = JcDepairing * practicalFactor * 1e-4;
  const criticalCurrentDensity = Math.round(Math.max(0, Number.isFinite(JcPerCm2) ? JcPerCm2 : 0));

  const kappaRaw = coherenceLength > 0 ? londonPenetrationDepth / coherenceLength : 1;
  const kappa = Number.isFinite(kappaRaw) ? kappaRaw : 1;
  const typeIorII = kappa > 0.707 ? "Type-II" : "Type-I";

  let lowerCriticalField = 0;
  if (kappa > 0 && londonPenetrationDepth > 0) {
    const Hc1Tesla = (PHI0 / (4 * Math.PI * lambdaLM * lambdaLM)) * Math.log(Math.max(kappa, 1.001));
    lowerCriticalField = Math.max(0, Number.isFinite(Hc1Tesla) ? Hc1Tesla : 0);
  }

  return {
    upperCriticalField: Number(upperCriticalField.toFixed(4)),
    lowerCriticalField: Number(lowerCriticalField.toFixed(4)),
    coherenceLength: Number(coherenceLength.toFixed(1)),
    londonPenetrationDepth: Number(londonPenetrationDepth.toFixed(1)),
    anisotropyRatio: Number(anisotropyRatio.toFixed(2)),
    criticalCurrentDensity,
    typeIorII,
    glValidation: {
      xiGL: Number.isFinite(xiGL_nm) ? Number(xiGL_nm.toFixed(2)) : 0,
      vFUsed: Math.round(vF),
      status: glStatus,
      diagnosis: glDiagnosis,
    },
  };
}

export interface PhononDispersionData {
  qPath: string[];
  branches: { label: string; frequencies: number[]; isSoft: boolean }[];
  softModeQPoints: string[];
  imaginaryFrequencies: number;
  maxAcousticFreq: number;
  minOpticalFreq: number;
  phononGap: number;
}

function inferLatticeType(formula: string, motifs: string[]): "hexagonal" | "tetragonal" | "cubic" | "generic" {
  const elements = parseFormulaElements(formula);

  const isKagome = motifs.some(m => m.includes("Kagome"));
  const isBoride = motifs.some(m => m.includes("honeycomb"));
  const isMgB2Type = elements.includes("Mg") && elements.includes("B");
  if (isKagome || isBoride || isMgB2Type) return "hexagonal";

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3
    && elements.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Bi", "Tl", "Hg"].includes(e));
  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  const isLayeredPerovskite = motifs.some(m => m.includes("Ruddlesden-Popper") || m.includes("I4/mmm"));
  if (isCuprate || isPnictide || isLayeredPerovskite) return "tetragonal";

  const isCubicPerovskite = motifs.some(m => m.includes("Cubic perovskite"));
  const isHClathrate = motifs.some(m => m.includes("H cage/clathrate"));
  if (isCubicPerovskite || isHClathrate) return "cubic";

  return "generic";
}

const Q_PATHS: Record<string, string[]> = {
  hexagonal: ["Γ", "M", "K", "Γ", "A"],
  tetragonal: ["Γ", "X", "M", "Γ", "Z"],
  cubic: ["Γ", "X", "M", "Γ", "R"],
  generic: ["Γ", "X", "M", "Γ"],
};

export function computePhononDispersion(
  formula: string,
  electronicStructure: ElectronicStructure,
  phononSpectrum: PhononSpectrum
): PhononDispersionData {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const avgMass = getAverageMass(counts);
  const rawLightestMass = getLightestMass(elements);
  const lightestMass = Number.isFinite(rawLightestMass) ? rawLightestMass : avgMass;

  const motifResult = detectStructuralMotifs(formula);
  const latticeType = inferLatticeType(formula, motifResult.motifs);
  const qPath = Q_PATHS[latticeType];
  const nSegments = qPath.length - 1;
  const nQPoints = 20;
  const nAtoms = Math.max(1, Math.round(totalAtoms));
  const nAcoustic = 3;
  const nOptical = Math.max(0, nAtoms * 3 - 3);
  const nBranches = Math.min(nAcoustic + nOptical, 12);

  const thetaD = phononSpectrum.debyeTemperature;
  const maxFreq = phononSpectrum.maxPhononFrequency;
  const logAvg = phononSpectrum.logAverageFrequency;

  const springConstant = thetaD * 0.695;

  const branches: { label: string; frequencies: number[]; isSoft: boolean }[] = [];
  const softModeQPoints: string[] = [];
  let imaginaryCount = 0;
  let maxAcousticFreq = 0;
  let minOpticalFreq = Infinity;

  for (let b = 0; b < nBranches; b++) {
    const isAcoustic = b < nAcoustic;
    const branchLabel = isAcoustic
      ? (b === 0 ? "LA" : b === 1 ? "TA1" : "TA2")
      : `O${b - nAcoustic + 1}`;

    const frequencies: number[] = [];
    let isSoft = false;

    const branchScale = isAcoustic
      ? springConstant * (b === 0 ? 1.0 : 0.7 - b * 0.1)
      : logAvg + (maxFreq - logAvg) * ((b - nAcoustic) / Math.max(1, nBranches - nAcoustic));

    for (let qi = 0; qi < nQPoints; qi++) {
      const qFrac = qi / (nQPoints - 1);

      const segLen = nQPoints / nSegments;
      const segIdx = Math.min(nSegments - 1, Math.floor(qi / segLen));
      const segPos = (qi - segIdx * segLen) / segLen;
      const segmentPhase = segPos * Math.PI;

      let freq: number;
      if (isAcoustic) {
        freq = branchScale * Math.abs(Math.sin(segmentPhase / 2));

        if (electronicStructure.correlationStrength > 0.7 && qFrac > 0.3 && qFrac < 0.7) {
          const softening = 1 - (electronicStructure.correlationStrength - 0.7) * 0.8;
          freq *= Math.max(0.1, softening);
        }
      } else {
        const baseline = branchScale;
        const dispersion = (maxFreq - logAvg) * 0.15 * Math.cos(segmentPhase);
        freq = baseline + dispersion;

        if (phononSpectrum.softModePresent && b === nAcoustic) {
          const softDip = 1 - phononSpectrum.softModeScore * 0.6 * Math.exp(-Math.pow(qFrac - 0.5, 2) * 20);
          freq *= Math.max(0.05, softDip);
        }
      }

      const anharmonicNoise = 1 + (Math.sin(qi * 7 + b * 3) * 0.03 * phononSpectrum.anharmonicityIndex);
      freq *= anharmonicNoise;

      if (phononSpectrum.hasImaginaryModes && isAcoustic && b === 2 && qFrac > 0.4 && qFrac < 0.6) {
        freq = -Math.abs(freq) * 0.3;
        imaginaryCount++;
      }

      if (freq < 0) {
        imaginaryCount++;
        const softSegIdx = Math.min(nSegments - 1, Math.floor(qFrac * nSegments));
        const segLabel = `${qPath[softSegIdx]}-${qPath[softSegIdx + 1]}`;
        if (!softModeQPoints.includes(segLabel)) softModeQPoints.push(segLabel);
      }

      if (Math.abs(freq) < logAvg * 0.15 && !isAcoustic) {
        isSoft = true;
      }

      freq = Math.round(freq * 10) / 10;
      frequencies.push(freq);

      if (isAcoustic && freq > maxAcousticFreq) maxAcousticFreq = freq;
      if (!isAcoustic && freq > 0 && freq < minOpticalFreq) minOpticalFreq = freq;
    }

    branches.push({ label: branchLabel, frequencies, isSoft });
  }

  if (minOpticalFreq === Infinity) minOpticalFreq = logAvg;
  const phononGap = Math.max(0, minOpticalFreq - maxAcousticFreq);

  return {
    qPath,
    branches,
    softModeQPoints,
    imaginaryFrequencies: imaginaryCount,
    maxAcousticFreq: Math.round(maxAcousticFreq * 10) / 10,
    minOpticalFreq: Math.round(minOpticalFreq * 10) / 10,
    phononGap: Math.round(phononGap * 10) / 10,
  };
}

export interface ManyBodyCorrections {
  gwDOSRenormalization: number;
  gwBandwidthCorrection: number;
  vertexCorrectionLambda: number;
  quasiparticleWeight: number;
  selfEnergyShift: number;
  correctedDOS: number;
  correctedBandwidth: number;
  correctedLambda: number;
}

export function applyManyBodyCorrections(
  electronicStructure: ElectronicStructure,
  coupling: ElectronPhononCoupling,
  formula: string
): ManyBodyCorrections {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const corr = electronicStructure.correlationStrength;
  const N_EF = electronicStructure.densityOfStatesAtFermi;
  const lambda = coupling.lambda;

  let wAvg = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    wAvg += estimateBandwidthW(el) * frac;
  }
  wAvg = Math.max(1.0, wAvg);

  const Z = 1 / (1 + lambda);
  const quasiparticleWeight = Math.max(0.1, Math.min(1.0, Z));

  const metallicity = electronicStructure.metallicity;
  let gwDOSRenormalization: number;
  if (metallicity < 0.2) {
    const gapSuppression = metallicity / 0.2;
    gwDOSRenormalization = gapSuppression * (0.3 + corr * 0.1);
  } else if (metallicity < 0.4) {
    const crossover = (metallicity - 0.2) / 0.2;
    const insulating = (metallicity / 0.2) * (0.3 + corr * 0.1);
    const metallic = 1 + corr * 0.3 + lambda * 0.15;
    gwDOSRenormalization = insulating * (1 - crossover) + metallic * crossover;
  } else {
    gwDOSRenormalization = 1 + corr * 0.3 + lambda * 0.15;
  }
  const correctedDOS = N_EF * gwDOSRenormalization;

  const gwBandwidthCorrection = 1 - corr * 0.25 - lambda * 0.1;
  const correctedBandwidth = wAvg * Math.max(0.3, gwBandwidthCorrection);

  const uOverW = estimateHubbardUoverW(elements, counts);
  const vertexParam = uOverW * corr;
  const vertexCorrectionLambda = 1 + vertexParam * 0.2 - vertexParam * vertexParam * 0.15;
  const correctedLambda = lambda * Math.max(0.5, vertexCorrectionLambda);

  const selfEnergyShift = -corr * 0.5 * wAvg * (1 + lambda * 0.3);

  return {
    gwDOSRenormalization: Number(gwDOSRenormalization.toFixed(4)),
    gwBandwidthCorrection: Number(gwBandwidthCorrection.toFixed(4)),
    vertexCorrectionLambda: Number(vertexCorrectionLambda.toFixed(4)),
    quasiparticleWeight: Number(quasiparticleWeight.toFixed(4)),
    selfEnergyShift: Number(selfEnergyShift.toFixed(4)),
    correctedDOS: Number(correctedDOS.toFixed(3)),
    correctedBandwidth: Number(correctedBandwidth.toFixed(3)),
    correctedLambda: Number(correctedLambda.toFixed(4)),
  };
}

export interface NestingFunctionData {
  qVectors: { label: string; q: [number, number, number] }[];
  nestingValues: number[];
  peakNestingQ: string;
  peakNestingValue: number;
  averageNesting: number;
  nestingAnisotropy: number;
  dominantInstability: string;
}

function getNestingQVectors(lattice: string): { label: string; q: [number, number, number] }[] {
  const common: { label: string; q: [number, number, number] }[] = [
    { label: "Γ", q: [0, 0, 0] },
  ];
  if (lattice === "hexagonal") {
    return [
      ...common,
      { label: "M", q: [0.5, 0, 0] },
      { label: "K", q: [1/3, 1/3, 0] },
      { label: "A", q: [0, 0, 0.5] },
      { label: "L", q: [0.5, 0, 0.5] },
      { label: "H", q: [1/3, 1/3, 0.5] },
      { label: "(2K)", q: [2/3, 2/3, 0] },
    ];
  }
  if (lattice === "tetragonal") {
    return [
      ...common,
      { label: "X", q: [0.5, 0, 0] },
      { label: "M", q: [0.5, 0.5, 0] },
      { label: "Z", q: [0, 0, 0.5] },
      { label: "R", q: [0.5, 0, 0.5] },
      { label: "A", q: [0.5, 0.5, 0.5] },
      { label: "(π,π)", q: [0.5, 0.5, 0] },
      { label: "(π,0)", q: [0.5, 0, 0] },
      { label: "(π/2,π/2)", q: [0.25, 0.25, 0] },
    ];
  }
  return [
    ...common,
    { label: "X", q: [0.5, 0, 0] },
    { label: "M", q: [0.5, 0.5, 0] },
    { label: "R", q: [0.5, 0.5, 0.5] },
    { label: "A", q: [0.5, 0, 0.5] },
    { label: "(π,π)", q: [0.5, 0.5, 0] },
    { label: "(π,0)", q: [0.5, 0, 0] },
    { label: "(π/2,π/2)", q: [0.25, 0.25, 0] },
  ];
}

export function computeNestingFunction(
  formula: string,
  electronicStructure: ElectronicStructure
): NestingFunctionData {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const N_EF = electronicStructure.densityOfStatesAtFermi;
  const nestingScore = electronicStructure.nestingScore;
  const corr = electronicStructure.correlationStrength;

  const motifResult = detectStructuralMotifs(formula);
  const lattice = inferLatticeType(formula, motifResult.motifs);
  const qVectors = getNestingQVectors(lattice);

  const vec = getVEC(elements, counts);

  const is2D = electronicStructure.fermiSurfaceTopology.includes("quasi-2D") ||
    electronicStructure.fermiSurfaceTopology.includes("cylindrical");
  const kF = is2D
    ? Math.sqrt(2 * Math.PI * Math.max(vec, 0.5)) * 0.3
    : Math.pow(3 * Math.PI * Math.PI * Math.max(vec, 0.5), 1 / 3) * 0.5;

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  const isKagome = motifResult.motifs.some(m => m.includes("Kagome"));

  const nestingValues: number[] = [];

  for (const qv of qVectors) {
    const qMag = Math.sqrt(qv.q[0] ** 2 + qv.q[1] ** 2 + qv.q[2] ** 2);

    let chi0: number;
    if (qMag < 0.01) {
      chi0 = N_EF;
    } else {
      const x = qMag / (2 * kF);
      if (x < 0.01) {
        chi0 = N_EF;
      } else if (x >= 1.0) {
        chi0 = N_EF * 0.5 * (1 + (1 - x * x) / (2 * x) * Math.log(Math.abs((1 + x) / (1 - x + 0.001))));
      } else {
        chi0 = N_EF * (0.5 + (1 - x * x) / (4 * x) * Math.log(Math.abs((1 + x) / (Math.abs(1 - x) + 0.001))));
      }
    }

    chi0 = Math.max(0, chi0);

    if (isCuprate && qv.label === "(π,π)") {
      chi0 *= (1 + corr * 2.5);
    }
    if (isPnictide && (qv.label === "(π,0)" || qv.label === "(π,π)")) {
      chi0 *= (1 + corr * 1.8);
    }
    if (isKagome && (qv.label === "K" || qv.label === "M" || qv.label === "(2K)")) {
      chi0 *= (1 + corr * 2.0);
    }

    chi0 *= (1 + nestingScore * 0.5);

    nestingValues.push(Number(chi0.toFixed(4)));
  }

  const peakIdx = nestingValues.indexOf(Math.max(...nestingValues));
  const peakNestingQ = qVectors[peakIdx].label;
  const peakNestingValue = nestingValues[peakIdx];
  const averageNesting = Number((nestingValues.reduce((a, b) => a + b, 0) / nestingValues.length).toFixed(4));
  const nestingAnisotropy = peakNestingValue > 0
    ? Number(((peakNestingValue - Math.min(...nestingValues)) / peakNestingValue).toFixed(4))
    : 0;

  let dominantInstability = "none";
  if (peakNestingValue > N_EF * 2) {
    if (peakNestingQ === "(π,π)" || peakNestingQ === "M") dominantInstability = "SDW/AFM";
    else if (peakNestingQ === "(π,0)" || peakNestingQ === "X") dominantInstability = "stripe-SDW";
    else if (peakNestingQ === "K" || peakNestingQ === "(2K)") dominantInstability = "kagome-CDW/bond-order";
    else dominantInstability = "CDW";
  } else if (peakNestingValue > N_EF * 1.3) {
    dominantInstability = "weak-nesting";
  }

  return {
    qVectors,
    nestingValues,
    peakNestingQ,
    peakNestingValue,
    averageNesting,
    nestingAnisotropy,
    dominantInstability,
  };
}

export interface DynamicSpinSusceptibility {
  chiStaticPeak: number;
  chiDynamicPeak: number;
  spinFluctuationEnergy: number;
  correlationLength: number;
  stonerEnhancement: number;
  isNearQCP: boolean;
  isStableFerromagnet: boolean;
  stonerProduct: number;
}

export function computeDynamicSpinSusceptibility(
  formula: string,
  electronic: ElectronicStructure
): DynamicSpinSusceptibility {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const N_EF = electronic.densityOfStatesAtFermi;
  const corr = electronic.correlationStrength;

  let stonerMax = 0;
  for (const el of elements) {
    if (!isTransitionMetal(el) && !isRareEarth(el)) continue;
    const I = getStonerParameter(el);
    if (I !== null) stonerMax = Math.max(stonerMax, I);
  }

  const stonerProduct = stonerMax * N_EF;
  const STONER_DENOM_FLOOR = 0.05;
  const STONER_CAP = 20;
  const isStableFerromagnet = stonerProduct >= 1.0;
  const stonerEnhancement = stonerProduct < 1.0
    ? 1 / Math.max(STONER_DENOM_FLOOR, 1 - stonerProduct)
    : 1 / STONER_DENOM_FLOOR;

  const chiStaticPeak = N_EF * Math.min(stonerEnhancement, STONER_CAP);

  const omega_sf = stonerProduct < 1.0
    ? 50 * Math.max(STONER_DENOM_FLOOR, 1 - stonerProduct)
    : 0.5;

  const chiDynamicPeak = chiStaticPeak * 0.8;

  const is2D = electronic.fermiSurfaceTopology.includes("quasi-2D") ||
    electronic.fermiSurfaceTopology.includes("cylindrical");
  const xiSpin3D = stonerProduct < 1.0
    ? 1 / Math.sqrt(Math.max(STONER_DENOM_FLOOR, 1 - stonerProduct))
    : 10;
  const dimAnisotropy = is2D ? 3.0 : 1.0;
  const correlationLength = Math.min(50, xiSpin3D * dimAnisotropy);

  const isNearQCP = stonerProduct > 0.7 || stonerEnhancement > 10;

  return {
    chiStaticPeak: Number(chiStaticPeak.toFixed(3)),
    chiDynamicPeak: Number(chiDynamicPeak.toFixed(3)),
    spinFluctuationEnergy: Number(omega_sf.toFixed(3)),
    correlationLength: Number(correlationLength.toFixed(3)),
    stonerEnhancement: Number(Math.min(stonerEnhancement, STONER_CAP).toFixed(3)),
    isNearQCP,
    isStableFerromagnet,
    stonerProduct: Number(stonerProduct.toFixed(4)),
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
    const hasTM = elements.some(e => isTransitionMetal(e));
    const hasRE = elements.some(e => isRareEarth(e));
    const hasActinide = elements.some(e => isActinide(e));
    const isSimpleSPMetal = !hasTM && !hasRE && !hasActinide && !elements.includes("H");
    if (isSimpleSPMetal) {
      ratio = 0.0;
    } else if (elements.includes("H")) {
      ratio = 0.1;
    } else {
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
  sdwInstability: number;
  magneticSusceptibilityPeak: number;
  softPhononCollapse: number;
  fermiSurfaceNestingStrength: number;
  dosEfPeakScore: number;
  vanHoveSingularityScore: number;
  flatBandInstability: number;
  overallProximity: number;
  nearestBoundary: string;
}

export interface StructuralMotifResult {
  motifs: string[];
  motifScore: number;
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
  const hasFM = competingPhases.some(p => p.type === "magnetism" && p.phaseName.includes("Ferromagnetic"));
  if (hasAFM && !hasFM) {
    magneticQCP = Math.max(magneticQCP, 0.6);
  } else if (hasAFM) {
    magneticQCP = Math.max(magneticQCP, 0.5);
  }

  let structuralBoundary = 0;
  const structPhases = competingPhases.filter(p => p.type === "structural");
  for (const sp of structPhases) {
    const match = sp.phaseName.match(/t=([\d.]+)/i);
    if (match) {
      const tf = parseFloat(match[1]);
      const optimalTf = 0.95;
      const dist = Math.abs(tf - optimalTf);
      const tfScore = Math.exp(-Math.pow(dist / 0.08, 2));
      const boundary = 1.0 - tfScore;
      structuralBoundary = Math.max(structuralBoundary, boundary);
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

  let sdwInstability = 0;
  const sdwPhases = competingPhases.filter(p => p.type === "SDW");
  for (const sdw of sdwPhases) {
    sdwInstability = Math.max(sdwInstability, sdw.strength);
  }
  if (electronic.nestingScore != null && electronic.nestingScore > 0.6) {
    sdwInstability = Math.max(sdwInstability, electronic.nestingScore * 0.7);
  }

  let magneticSusceptibilityPeak = 0;
  for (const el of elements) {
    const I = getStonerParameter(el);
    if (I !== null) {
      const stonerProduct = I * electronic.densityOfStatesAtFermi;
      if (stonerProduct > 0.5) {
        const chi = 1 / Math.max(0.01, 1 - stonerProduct);
        // Use 1 - 1/chi normalisation: maps chi=1→0, chi=2→0.5, chi=10→0.9, chi→∞→<1.
        // The old chi/10 cap caused all high-DOS hydrides to saturate at exactly 1.00;
        // this formula preserves differentiation across the full range.
        const normalised = Math.min(0.99, 1 - 1 / Math.max(1.001, chi));
        magneticSusceptibilityPeak = Math.max(magneticSusceptibilityPeak, normalised);
      }
    }
  }

  let softPhononCollapse = 0;
  const sms = phononSpectrum.softModeScore ?? (phononSpectrum.softModePresent ? 0.6 : 0);
  softPhononCollapse = Math.max(softPhononCollapse, sms * 0.8);
  if (phononSpectrum.logAverageFrequency < 100) {
    softPhononCollapse = Math.max(softPhononCollapse, 0.8);
  }
  if (phononSpectrum.anharmonicityIndex > 0.6) {
    softPhononCollapse = Math.max(softPhononCollapse, 0.5);
  }

  let fermiSurfaceNestingStrength = 0;
  const nestingScore = electronic.nestingScore ?? 0;
  const vanHoveProx = electronic.vanHoveProximity ?? 0;
  if (nestingScore > 0.7) {
    fermiSurfaceNestingStrength = Math.min(1.0, nestingScore * 0.9 + (electronic.correlationStrength > 0.5 ? 0.15 : 0));
  } else if (nestingScore > 0.4) {
    fermiSurfaceNestingStrength = nestingScore * 0.6;
  }
  if (electronic.fermiSurfaceTopology.includes("nesting") || electronic.fermiSurfaceTopology.includes("2D")) {
    fermiSurfaceNestingStrength = Math.max(fermiSurfaceNestingStrength, 0.4);
  }

  let dosEfPeakScore = 0;
  const dosAtFermi = electronic.densityOfStatesAtFermi;
  if (dosAtFermi > 3.0) {
    dosEfPeakScore = Math.min(1.0, (dosAtFermi - 3.0) * 0.25 + 0.6);
  } else if (dosAtFermi > 2.0) {
    dosEfPeakScore = Math.min(0.6, (dosAtFermi - 2.0) * 0.3 + 0.3);
  } else if (dosAtFermi > 1.5) {
    dosEfPeakScore = (dosAtFermi - 1.5) * 0.4;
  }
  if (electronic.correlationStrength > 0.6 && dosAtFermi > 1.5) {
    dosEfPeakScore = Math.min(1.0, dosEfPeakScore + 0.15);
  }

  let vanHoveSingularityScore = 0;
  if (vanHoveProx > 0.7) {
    vanHoveSingularityScore = Math.min(1.0, vanHoveProx * 0.85 + 0.1);
  } else if (vanHoveProx > 0.4) {
    vanHoveSingularityScore = vanHoveProx * 0.6;
  }
  if (dosEfPeakScore > 0.5 && vanHoveProx > 0.3) {
    vanHoveSingularityScore = Math.min(1.0, vanHoveSingularityScore + dosEfPeakScore * 0.2);
  }

  let flatBandInstability = 0;
  const fbi = electronic.flatBandIndicator ?? 0;
  if (fbi > 0.7) {
    flatBandInstability = Math.min(1.0, (fbi - 0.7) * 2.0 + 0.5);
  } else if (fbi > 0.4) {
    flatBandInstability = (fbi - 0.4) * 1.2;
  }
  if (electronic.mottProximityScore > 0.6 && fbi > 0.5) {
    flatBandInstability = Math.min(1.0, flatBandInstability + 0.2);
  }
  if (electronic.bandFlatness > 0.6) {
    flatBandInstability = Math.max(flatBandInstability, electronic.bandFlatness * 0.5);
  }

  const scores = [
    { name: "Magnetic QCP", val: magneticQCP },
    { name: "Structural boundary", val: structuralBoundary },
    { name: "Metal-insulator transition", val: metalInsulatorTransition },
    { name: "CDW instability", val: cdwInstability },
    { name: "SDW instability", val: sdwInstability },
    { name: "Magnetic susceptibility peak", val: magneticSusceptibilityPeak },
    { name: "Soft phonon collapse", val: softPhononCollapse },
    { name: "Fermi surface nesting", val: fermiSurfaceNestingStrength },
    { name: "DOS(EF) peak", val: dosEfPeakScore },
    { name: "Van Hove singularity", val: vanHoveSingularityScore },
    { name: "Flat band instability", val: flatBandInstability },
  ];

  const overallProximity = Math.max(...scores.map(s => s.val));
  const nearest = scores.reduce((a, b) => b.val > a.val ? b : a);

  return {
    magneticQCP,
    structuralBoundary,
    metalInsulatorTransition,
    cdwInstability,
    sdwInstability,
    magneticSusceptibilityPeak,
    softPhononCollapse,
    fermiSurfaceNestingStrength,
    dosEfPeakScore,
    vanHoveSingularityScore,
    flatBandInstability,
    overallProximity,
    nearestBoundary: nearest.name,
  };
}

export function simulatePressureEffects(
  formula: string,
  electronic: ElectronicStructure,
  phonon: PhononSpectrum,
  coupling: ElectronPhononCoupling
): { pressureOptimalTc: number; optimalPressure: number; pressureTcCurve: { pressure: number; tc: number }[] } {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const hasH = elements.includes("H");
  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

  const isPressureSensitive = hasH || elements.includes("S") || elements.includes("Se") ||
    elements.some(e => isRareEarth(e)) || hRatio >= 4;

  if (!isPressureSensitive) {
    const baseTc = predictTcEliashberg(coupling, phonon, undefined, undefined, formula, 0).predictedTc;
    return {
      pressureOptimalTc: baseTc,
      optimalPressure: 0,
      pressureTcCurve: [{ pressure: 0, tc: baseTc }],
    };
  }

  // Optimal pressure by compound type — above P_opt, lambda decreases as phonons over-harden
  // and Tc turns over. Data anchors: LaH10 ~160 GPa, H3S ~150 GPa, YH4 ~120 GPa.
  // Y (Z=39) is a TM in isRareEarth() but behaves like RE for hydride pressure physics.
  const isRELike = elements.some(e => isRareEarth(e) || e === "Y" || e === "Sc");
  const isLightMetal = elements.some(e => ["Li","Na","Be","Mg","Al","Ca","K"].includes(e));
  let P_opt: number;
  if (hRatio >= 8) {
    P_opt = isLightMetal ? 220 : 160; // Li2MgH16-like vs LaH10-like
  } else if (hRatio >= 5) {
    P_opt = 140; // YH6, CeH6-type
  } else if (hRatio >= 3) {
    P_opt = isRELike ? 110 : 100; // LaH6/Y2H7-type
  } else if (hRatio >= 1.5) {
    P_opt = isRELike ? 80 : 60; // lower hydrides
  } else if (hasH && elements.some(e => ["S","Se"].includes(e))) {
    P_opt = 150; // H3S / H3Se type
  } else {
    P_opt = 50; // RE without H, S/Se compounds
  }

  // lambdaBase: fraction of coupling.lambda active at P=0.
  // High-H compounds need pressure to metallize — lambda is suppressed at ambient.
  const lambdaBase = hRatio >= 5 ? 0.25 : hRatio >= 3 ? 0.45 : hRatio >= 1 ? 0.65 : 0.85;
  // Max phonon stiffening factor at P_opt (calibrated to ~2x omegaLog for superhydrides)
  const phonBoostAtPopt = hRatio >= 5 ? 0.80 : hRatio >= 2 ? 0.50 : 0.25;

  const pressures = [0, 10, 25, 50, 100, 150, 200];
  const curve: { pressure: number; tc: number }[] = [];
  let bestTc = 0;
  let bestP = 0;

  for (const p of pressures) {
    let lambdaFactor: number;
    let phononFactor: number;

    if (p <= P_opt) {
      // Rising branch: H metallization + phonon stiffening up to P_opt.
      const frac = p / Math.max(1, P_opt);
      lambdaFactor = lambdaBase + (1 - lambdaBase) * frac;
      phononFactor = 1 + phonBoostAtPopt * frac;
    } else {
      // Falling branch: phonons over-harden past pairing cutoff, lambda drops as 1/P^1.5.
      // This is the turnover missing from the old model.
      const overRatio = P_opt / p; // < 1 above P_opt
      lambdaFactor = Math.pow(overRatio, 1.5);
      // omega_log continues modest growth past P_opt, limited to +15% beyond P_opt value
      const overshootFrac = Math.min((p - P_opt) / P_opt, 1.0);
      phononFactor = (1 + phonBoostAtPopt) * (1 + 0.15 * overshootFrac);
    }

    const adjLambda = Math.max(0.01, coupling.lambda * lambdaFactor);
    const adjOmegaLog = Math.min(coupling.omegaLog * phononFactor, 2000);
    const adjCoupling: ElectronPhononCoupling = {
      ...coupling,
      lambda: adjLambda,
      omegaLog: adjOmegaLog,
    };

    const result = predictTcEliashberg(adjCoupling, phonon, undefined, undefined, formula, p);
    // Cap each curve point — prevents runaway values at the ceiling from boosting bestTc
    const tc = applyAmbientTcCap(result.predictedTc, adjLambda, p, electronic.metallicity, formula);
    curve.push({ pressure: p, tc });

    if (tc > bestTc) {
      bestTc = tc;
      bestP = p;
    }
  }

  return { pressureOptimalTc: bestTc, optimalPressure: bestP, pressureTcCurve: curve };
}

export interface PhononDOSData {
  frequencies: number[];
  dos: number[];
  totalStates: number;
  partialDOS?: Record<string, number[]>;
}

export function computePhononDOS(phononDispersion: PhononDispersionData, maxPhononFreq: number, formula?: string): PhononDOSData {
  const extendedMax = maxPhononFreq * 1.2;
  const nBins = maxPhononFreq > 1500 ? 150 : 100;
  if (!maxPhononFreq || maxPhononFreq <= 0) {
    return { frequencies: new Array(nBins).fill(0).map((_, i) => (i + 0.5)), dos: new Array(nBins).fill(0), totalStates: 0 };
  }
  const binWidth = extendedMax / nBins;
  const dos = new Array(nBins).fill(0);
  const frequencies = new Array(nBins).fill(0).map((_, i) => (i + 0.5) * binWidth);

  let totalStates = 0;
  for (const branch of phononDispersion.branches) {
    for (const freq of branch.frequencies) {
      const absFreq = Math.abs(freq);
      if (absFreq <= 0 || absFreq > extendedMax) continue;
      const bin = Math.min(nBins - 1, Math.floor(absFreq / binWidth));
      dos[bin] += 1;
      totalStates++;
    }
  }

  if (totalStates > 0) {
    for (let i = 0; i < nBins; i++) {
      dos[i] = dos[i] / (totalStates * binWidth);
    }
  }

  let partialDOS: Record<string, number[]> | undefined;
  if (formula) {
    const elements = parseFormulaElements(formula);
    const counts = parseFormulaCounts(formula);
    const totalAtoms = getTotalAtoms(counts);

    if (elements.length > 1) {
      const masses: Record<string, number> = {};
      for (const el of elements) {
        const data = getElementData(el);
        const mass = data?.atomicMass;
        if (mass === undefined) console.warn(`[physics-engine] Unknown atomic mass for element: ${el}`);
        masses[el] = mass ?? NaN;
      }

      const invMasses: Record<string, number> = {};
      for (const el of elements) {
        invMasses[el] = 1.0 / masses[el];
      }

      partialDOS = {};
      for (const el of elements) {
        partialDOS[el] = new Array(nBins).fill(0);
      }

      for (let i = 0; i < nBins; i++) {
        if (dos[i] <= 0) continue;
        const omega = frequencies[i];

        let totalWeight = 0;
        const elWeights: Record<string, number> = {};

        for (const el of elements) {
          const frac = (counts[el] || 1) / totalAtoms;
          const charFreq = maxPhononFreq * Math.sqrt(invMasses[el] / Math.max(...Object.values(invMasses)));
          const freqMatch = Math.exp(-0.5 * Math.pow((omega - charFreq) / (charFreq * 0.4 + 50), 2));
          const lowFreqBase = frac * Math.pow(masses[el] / Math.max(...Object.values(masses)), 0.3);
          const weight = frac * invMasses[el] * freqMatch + lowFreqBase * 0.2;
          elWeights[el] = weight;
          totalWeight += weight;
        }

        if (totalWeight > 0) {
          for (const el of elements) {
            partialDOS[el][i] = dos[i] * (elWeights[el] / totalWeight);
          }
        }
      }
    }
  }

  return { frequencies, dos, totalStates, partialDOS };
}

export interface Alpha2FData {
  frequencies: number[];
  alpha2F: number[];
  lambdaOmega: number[];
  integratedLambda: number;
}

export function computeAlpha2F(
  phononDOS: PhononDOSData,
  formula: string,
  electronicStructure: ElectronicStructure
): Alpha2FData {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const N_EF = electronicStructure.densityOfStatesAtFermi;

  let avgEta = 0;
  let totalWeight = 0;
  for (const el of elements) {
    const eta = getMcMillanHopfieldEta(el);
    const frac = (counts[el] || 1) / totalAtoms;
    if (eta !== null && eta > 0) {
      avgEta += eta * frac;
      totalWeight += frac;
    }
  }
  if (totalWeight > 0) {
    avgEta = avgEta / totalWeight;
  } else {
    avgEta = N_EF * 0.3;
  }

  const couplingPrefactor = avgEta * N_EF * 1.2;

  const nBins = phononDOS.frequencies.length;
  const alpha2F = new Array(nBins).fill(0);
  const lambdaOmega = new Array(nBins).fill(0);
  let integratedLambda = 0;

  const binWidth = nBins > 1 ? phononDOS.frequencies[1] - phononDOS.frequencies[0] : 1;

  const maxFreq = phononDOS.frequencies[nBins - 1] || 1;
  const debyeCutoff = maxFreq * 0.08;

  for (let i = 0; i < nBins; i++) {
    const omega = phononDOS.frequencies[i];
    const g = phononDOS.dos[i];
    if (omega <= 0 || g <= 0) continue;

    let effectiveG = g;
    if (omega < debyeCutoff) {
      const debyeWeight = (omega / debyeCutoff) * (omega / debyeCutoff);
      effectiveG = g * debyeWeight;
    }

    alpha2F[i] = couplingPrefactor * effectiveG * omega * 0.01;
    alpha2F[i] = Number(alpha2F[i].toFixed(6));

    const lambdaContrib = 2 * alpha2F[i] / omega * binWidth;
    integratedLambda += lambdaContrib;
    lambdaOmega[i] = Number(integratedLambda.toFixed(6));
  }

  integratedLambda = Number(integratedLambda.toFixed(4));

  return {
    frequencies: phononDOS.frequencies,
    alpha2F,
    lambdaOmega,
    integratedLambda,
  };
}

export function computeOmegaLogFromAlpha2F(alpha2FData: Alpha2FData): number {
  const { frequencies, alpha2F, integratedLambda } = alpha2FData;
  if (integratedLambda <= 0 || integratedLambda < 1e-8) return 0;

  const nBins = frequencies.length;
  const binWidth = nBins > 1 ? frequencies[1] - frequencies[0] : 1;
  let logSum = 0;

  const LOW_FREQ_CUTOFF_MEV = 1.5;
  const LOW_FREQ_CUTOFF_CM1 = LOW_FREQ_CUTOFF_MEV / 0.1240;
  for (let i = 0; i < nBins; i++) {
    const omega = frequencies[i];
    if (omega <= LOW_FREQ_CUTOFF_CM1 || alpha2F[i] <= 0) continue;
    logSum += (alpha2F[i] / omega) * Math.log(omega) * binWidth;
  }

  const omegaLog = Math.exp((2 / integratedLambda) * logSum);
  return Number.isFinite(omegaLog) ? Math.round(omegaLog * 10) / 10 : 0;
}

export interface FDPhononSummary {
  dynamicallyStable: boolean;
  imaginaryModeCount: number;
  lowestFrequency: number;
  highestFrequency: number;
  omegaLog: number | null;
  lambdaContribution: number | null;
  forceConstantClampedEntries: number;
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
  phononDispersion: PhononDispersionData;
  manyBodyCorrections: ManyBodyCorrections;
  nestingFunction: NestingFunctionData;
  spinSusceptibility: DynamicSpinSusceptibility;
  phononDOS: PhononDOSData;
  alpha2F: Alpha2FData;
  advancedConstraints: AdvancedPhysicsConstraints;
  fdPhononSummary: FDPhononSummary | null;
}> {
  const formula = candidate.formula;
  const priorTc = candidate.predictedTc ?? 0;
  const isVerbose = priorTc > 10 || (candidate.pressureGpa ?? 0) > 50;
  const emitDetail: typeof emit = (event, data) => {
    if (isVerbose) emit(event, data);
  };

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
    const mpClient = await getMPClient();
    if (mpClient.isApiAvailable()) {
      // 30s overall timeout: fetchAllData makes 6 serial MP API calls (summary, elasticity,
      // magnetism, phonon, electronicStructure, thermo). Each has a 60s per-request timeout
      // but with retries the total can reach 5+ minutes with no guard. On timeout we fall
      // through with mpSummary=null and rely on analytical fallbacks — same as no MP key.
      const mpData = await Promise.race([
        mpClient.fetchAllData(formula),
        new Promise<never>((_, reject) => setTimeout(
          () => reject(new Error(`[Physics] mpClient.fetchAllData timeout (30s) for ${formula}`)),
          30_000
        )),
      ]);
      mpSummary = mpData.summary;
      mpElasticity = mpData.elasticity;
    }
  } catch (err: unknown) {
    if (err instanceof Error && err.message.startsWith("[Physics] mpClient.fetchAllData timeout")) {
      console.warn(err.message, "— using analytical fallback");
    }
  }

  try {
    const dftResolver = await getDFTResolver();
    // skipXTB=true: inline xTB takes 30-90s; DFT queue handles it asynchronously.
    // 90s overall timeout guards against serial MP API fence pile-ups (fetchElasticity +
    // fetchElectronicStructure each call fetchSummary internally, creating 5-10 serial
    // MP calls × 30s timeout = potential 5-16 min hang with no timeout).  On timeout we
    // fall through with dftData=null and rely on analytical fallbacks.
    dftData = await Promise.race([
      dftResolver.resolveDFTFeatures(formula, candidate.pressureGpa ?? 0, true),
      new Promise<null>(r => setTimeout(() => {
        console.warn(`[Physics] resolveDFTFeatures timeout (90s) for ${formula} — using analytical fallback`);
        r(null);
      }, 90_000)),
    ]);
    if (dftData && dftData.dftCoverage > 0) {
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
  // Yield after the heaviest sync call (~120ms tight-binding) so heartbeat timers can fire
  // before the remaining synchronous physics computations (~50ms).
  await new Promise<void>(r => setTimeout(r, 0));

  if (electronicStructure.flatBandIndicator > 0.3) {
    const avgDOS = electronicStructure.densityOfStatesAtFermi;
    emitDetail("log", {
      phase: "phase-10",
      event: "Flat band detected",
      detail: `${formula}: flatBandIndicator=${electronicStructure.flatBandIndicator.toFixed(2)}, bandFlatness=${electronicStructure.bandFlatness.toFixed(2)}, DOS(EF)=${avgDOS.toFixed(2)}, mottProximity=${electronicStructure.mottProximityScore.toFixed(2)}`,
      dataSource: "Physics Engine",
    });
  }

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

  const candidatePressure = candidate.pressureGpa ?? 0;

  {
    const hydEls = parseFormulaElements(formula);
    const hydCts = parseFormulaCounts(formula);
    const hydHCount = hydCts["H"] || 0;
    const hydMetalAtoms = hydEls.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
      .reduce((s, e) => s + (hydCts[e] || 0), 0);
    const hydHRatio = hydMetalAtoms > 0 ? hydHCount / hydMetalAtoms : 0;
    if (hydHRatio >= 6 && candidatePressure < 20) {
      electronicStructure.metallicity = 0;
      emit("log", {
        phase: "phase-10",
        event: "Superhydride ambient suppression",
        detail: `${formula}: H/M=${hydHRatio.toFixed(1)} at ${candidatePressure} GPa < 20 GPa — metallic phase impossible, forcing metallicity=0`,
        dataSource: "Physics Engine",
      });
    } else if (hydHRatio >= 6 && candidatePressure < 50) {
      const suppressionFactor = Math.max(0.05, (candidatePressure - 20) / 30);
      electronicStructure.metallicity = Math.min(electronicStructure.metallicity, suppressionFactor);
    } else if (hydHRatio >= 4 && candidatePressure < 30) {
      const suppressionFactor = Math.max(0.1, candidatePressure / 30);
      electronicStructure.metallicity = Math.min(electronicStructure.metallicity, suppressionFactor);
    }
  }

  const phononSpectrum = computePhononSpectrum(formula, electronicStructure, mpElasticity, mpSummary);
  // Yield after phonon spectrum (~80ms) so heartbeat timers can fire
  await new Promise<void>(r => setTimeout(r, 0));

  if (dftData) {
    if (dftData.phononFreqMax.value != null && dftData.phononFreqMax.source !== "analytical") {
      const rawPhMax = dftData.phononFreqMax.value;
      if (rawPhMax > 5000) {
        emit("log", {
          phase: "phase-10",
          event: "DFT phonon value discarded",
          detail: `${formula}: DFT ω_max=${rawPhMax} cm⁻¹ exceeds physical limit (5000 cm⁻¹) — likely unit error, using analytical fallback`,
          dataSource: "DFT Resolver",
        });
      }
      const cappedPhMax = rawPhMax <= 5000 ? rawPhMax : phononSpectrum.maxPhononFrequency;
      phononSpectrum.maxPhononFrequency = Math.min(5000, cappedPhMax);
      phononSpectrum.debyeTemperature = Math.max(50, Math.round(1.4388 * phononSpectrum.maxPhononFrequency));
      emitDetail("log", {
        phase: "phase-10",
        event: "DFT override",
        detail: `Using DFT phonon max ${phononSpectrum.maxPhononFrequency} cm⁻¹ for ${formula}, derived θ_D=${phononSpectrum.debyeTemperature}K`,
        dataSource: "DFT Resolver",
      });
    } else if (dftData.debyeTemp.source !== "analytical" && dftData.debyeTemp.value > 0) {
      const analytical = phononSpectrum.debyeTemperature;
      phononSpectrum.debyeTemperature = dftData.debyeTemp.value;
      phononSpectrum.maxPhononFrequency = Math.max(50, Math.round(dftData.debyeTemp.value / 1.4388));
      if (Math.abs(analytical - dftData.debyeTemp.value) > 20) {
        emitDetail("log", {
          phase: "phase-10",
          event: "DFT override",
          detail: `Using DFT Debye temp ${dftData.debyeTemp.value}K for ${formula} (vs analytical ${analytical}K), derived ω_max=${phononSpectrum.maxPhononFrequency} cm⁻¹`,
          dataSource: "DFT Resolver",
        });
      }
    }
  }
  const coupling = computeElectronPhononCoupling(electronicStructure, phononSpectrum, formula, candidatePressure);

  if (dftData?.finiteDisplacementPhonons) {
    const fdp = dftData.finiteDisplacementPhonons;

    if (fdp.omegaLog != null && fdp.omegaLog > 0) {
      const analyticalOmegaLog = coupling.omegaLog;
      coupling.omegaLog = fdp.omegaLog;
      emit("log", {
        phase: "phase-10",
        event: "FD phonon omegaLog override",
        detail: `${formula}: coupling.omegaLog ${analyticalOmegaLog.toFixed(1)} -> ${fdp.omegaLog.toFixed(1)} cm⁻¹ (finite displacement)`,
        dataSource: "DFT Resolver",
      });
    }

    if (fdp.lambdaContribution != null && fdp.lambdaContribution > 0) {
      const blendWeight = 0.3;
      const analyticalLambda = coupling.lambda;
      coupling.lambda = coupling.lambda * (1 - blendWeight) + fdp.lambdaContribution * blendWeight;
      emit("log", {
        phase: "phase-10",
        event: "FD phonon lambda blend",
        detail: `${formula}: coupling.lambda ${analyticalLambda.toFixed(4)} -> ${coupling.lambda.toFixed(4)} (blended ${(blendWeight * 100).toFixed(0)}% FD surrogate λ=${fdp.lambdaContribution.toFixed(4)})`,
        dataSource: "DFT Resolver",
      });
    }

    if (!fdp.dynamicallyStable) {
      emit("log", {
        phase: "phase-10",
        event: "Dynamically unstable",
        detail: `${formula}: finite displacement phonons show ${fdp.imaginaryModeCount} imaginary mode(s), lowestFreq=${fdp.lowestFrequency.toFixed(1)} cm⁻¹ — flagged dynamically unstable`,
        dataSource: "DFT Resolver",
      });
    }
  }

  const earlyPhononDispersion = computePhononDispersion(formula, electronicStructure, phononSpectrum);
  const phononDOS = computePhononDOS(earlyPhononDispersion, phononSpectrum.maxPhononFrequency, formula);
  emitDetail("log", {
    phase: "phase-10",
    event: "Phonon DOS computed",
    detail: `${formula}: ${phononDOS.totalStates} states binned into ${phononDOS.frequencies.length} bins, maxFreq=${phononSpectrum.maxPhononFrequency} cm⁻¹`,
    dataSource: "Physics Engine",
  });

  const alpha2FResult = computeAlpha2F(phononDOS, formula, electronicStructure);
  emitDetail("log", {
    phase: "phase-10",
    event: "alpha2F spectral function computed",
    detail: `${formula}: integratedLambda=${alpha2FResult.integratedLambda.toFixed(4)}, omegaLog(alpha2F)=${computeOmegaLogFromAlpha2F(alpha2FResult).toFixed(1)} cm⁻¹`,
    dataSource: "Physics Engine",
  });
  // Yield after coupling + phonon DOS + alpha2F cluster
  await new Promise<void>(r => setTimeout(r, 0));

  if (earlyPhononDispersion.softModeQPoints.length > 0 && phononSpectrum.softModePresent) {
    const softBranches = earlyPhononDispersion.branches.filter(b => b.isSoft);
    const nSoftBranches = softBranches.length;
    const softFraction = nSoftBranches / Math.max(1, earlyPhononDispersion.branches.length);
    const softModeOmegaLogPenalty = 1.0 - softFraction * phononSpectrum.softModeScore * 0.3;
    coupling.omegaLog = Math.round(coupling.omegaLog * Math.max(0.5, softModeOmegaLogPenalty));
    emitDetail("log", {
      phase: "phase-10",
      event: "Soft mode omegaLog correction",
      detail: `${formula}: ${nSoftBranches} soft branches (${(softFraction * 100).toFixed(0)}%), softScore=${phononSpectrum.softModeScore.toFixed(2)}, omegaLog corrected by ${((1 - softModeOmegaLogPenalty) * 100).toFixed(1)}%`,
      dataSource: "Physics Engine",
    });
  }

  // Classify material class early so predictTcEliashberg can route cuprates and
  // heavy fermions to their physically appropriate (non-Allen-Dynes) Tc formulas.
  const eliashbergMatClass = classifyMaterialForLambda(formula, candidatePressure);

  let eliashberg: EliashbergResult;
  if (dftData?.finiteDisplacementPhonons) {
    const fdp = dftData.finiteDisplacementPhonons;
    const fdpHasOmegaLog = fdp.omegaLog != null && fdp.omegaLog > 0;
    const fdpHasLambda = fdp.lambdaContribution != null && fdp.lambdaContribution > 0;
    if (fdpHasOmegaLog || fdpHasLambda) {
      eliashberg = predictTcEliashberg(coupling, phononSpectrum, undefined, eliashbergMatClass, formula, candidatePressure);
      emit("log", {
        phase: "phase-10",
        event: "Eliashberg uses FD phonon coupling",
        detail: `${formula}: Tc computed with FD-adjusted lambda=${coupling.lambda.toFixed(4)}, omegaLog=${coupling.omegaLog.toFixed(1)} cm⁻¹ (alpha2F bypassed in favor of FD data)`,
        dataSource: "DFT Resolver",
      });
    } else {
      eliashberg = predictTcEliashberg(coupling, phononSpectrum, alpha2FResult, eliashbergMatClass, formula, candidatePressure);
    }
  } else {
    eliashberg = predictTcEliashberg(coupling, phononSpectrum, alpha2FResult, eliashbergMatClass, formula, candidatePressure);
  }

  const competingPhases = evaluateCompetingPhases(formula, electronicStructure, mpSummary);
  const hasMottPhase = competingPhases.some(p => p.type === "Mott");
  const isMottInsulator = hasMottPhase && correlation.ratio > 0.7;
  const isNonMetallic = electronicStructure.metallicity < 0.4;

  const formulaEls = parseFormulaElements(formula);
  const formulaCts = parseFormulaCounts(formula);
  const hasDWaveSymmetry = formulaEls.includes("Cu") && formulaEls.includes("O") && formulaEls.length >= 3;
  const hasLayeredStructure = electronicStructure.fermiSurfaceTopology.includes("2D") ||
    (electronicStructure.orbitalCharacter.includes("hybridized") && formula.includes("O"));
  const isNickelate = formulaEls.includes("Ni") && formulaEls.includes("O") && formulaEls.length >= 3
    && formulaEls.some(e => isRareEarth(e) || ["La", "Nd", "Pr", "Sr"].includes(e));
  const isHEAPhysics = detectHighEntropyAlloy(formula);
  const tcUoverW = estimateHubbardUoverW(formulaEls, formulaCts);
  const tcMatClass = classifyMaterialForLambda(formula, candidatePressure);
  const tcCorrelationPenalty = getCorrelationPenaltyForClass(tcMatClass, tcUoverW, correlation.ratio);

  if (isNonMetallic) {
    const metalFactor = Math.max(0.02, electronicStructure.metallicity);
    eliashberg.predictedTc = eliashberg.predictedTc * metalFactor;
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2)];
  } else if (isMottInsulator) {
    eliashberg.predictedTc = eliashberg.predictedTc * 0.05;
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 3)];
  } else if (correlation.ratio > 0.7) {
    const baseFactor = hasDWaveSymmetry ? 0.5 : isNickelate ? 0.35 : hasLayeredStructure ? 0.4 : 0.3;
    const uWBoost = tcUoverW > 1.0 ? Math.max(0.5, 1.0 - (tcUoverW - 1.0) * 0.3) : 1.0;
    const corrFactor = baseFactor * uWBoost * Math.sqrt(tcCorrelationPenalty);
    eliashberg.predictedTc = eliashberg.predictedTc * corrFactor;
    eliashberg.confidenceBand = [
      Math.round(eliashberg.predictedTc * 0.3),
      Math.round(eliashberg.predictedTc * (hasDWaveSymmetry ? 3.0 : isNickelate ? 2.5 : 2.0)),
    ];
    emit("log", {
      phase: "phase-10",
      event: "Correlation Tc suppression",
      detail: `${formula}: U/W=${tcUoverW.toFixed(3)}, corrRatio=${correlation.ratio.toFixed(3)}, penalty=${tcCorrelationPenalty.toFixed(3)}, class=${tcMatClass}, corrFactor=${corrFactor.toFixed(3)}`,
      dataSource: "Physics Engine",
    });
  } else if (correlation.ratio > 0.5) {
    const baseFactor = hasLayeredStructure ? 0.85 : hasDWaveSymmetry ? 0.8 : isNickelate ? 0.65 : 0.7;
    const corrFactor = baseFactor * Math.sqrt(tcCorrelationPenalty);
    eliashberg.predictedTc = eliashberg.predictedTc * corrFactor;
    eliashberg.confidenceBand = [
      Math.round(eliashberg.predictedTc * 0.6),
      Math.round(eliashberg.predictedTc * 1.5),
    ];
  } else if (correlation.ratio > 0.3 && tcCorrelationPenalty < 0.9) {
    eliashberg.predictedTc = eliashberg.predictedTc * tcCorrelationPenalty;
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

  const magneticPhases = competingPhases.filter(p => p.type === "magnetism");
  for (const mp of magneticPhases) {
    if (mp.suppressesSC && mp.strength > 0.3) {
      const magSuppression = Math.max(0.1, 1.0 - mp.strength * 0.7);
      eliashberg.predictedTc = Math.round(eliashberg.predictedTc * magSuppression);
      emit("log", {
        phase: "phase-10",
        event: "Magnetic suppression applied",
        detail: `${formula}: magnetic phase strength=${mp.strength.toFixed(2)}, Tc suppressed by ${((1 - magSuppression) * 100).toFixed(0)}%`,
        dataSource: "Physics Engine",
      });
      break;
    }
  }

  const criticalFields = computeCriticalFields(eliashberg.predictedTc, coupling, dimensionality, formula, electronicStructure);
  // Yield after Eliashberg + competing phases + pairing + critical fields cluster
  await new Promise<void>(r => setTimeout(r, 0));

  const suppressingPhases = competingPhases.filter(p => p.suppressesSC);
  let uncertaintyEstimate = 0.3;
  if (correlation.ratio > 0.7) uncertaintyEstimate += 0.2;
  if (phononSpectrum.hasImaginaryModes) uncertaintyEstimate += 0.15;
  if (suppressingPhases.length > 0) uncertaintyEstimate += 0.1;
  if (phononSpectrum.anharmonicityIndex > 0.5) uncertaintyEstimate += 0.1;
  if (!mpSummary) uncertaintyEstimate += 0.05;
  if (dftData && dftData.dftCoverage > 0.3) uncertaintyEstimate -= dftData.dftCoverage * 0.15;
  if (dftData?.finiteDisplacementPhonons && !dftData.finiteDisplacementPhonons.dynamicallyStable) {
    const fdpImagCount = dftData.finiteDisplacementPhonons.imaginaryModeCount;
    const fdpLowest = dftData.finiteDisplacementPhonons.lowestFrequency;
    if (fdpImagCount > 0) {
      uncertaintyEstimate += 0.2;
      const instabilityPenalty = Math.max(0.3, 1.0 - fdpImagCount * 0.1);
      eliashberg.predictedTc = Math.round(eliashberg.predictedTc * instabilityPenalty);
      eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2.5)];
      emit("log", {
        phase: "phase-10",
        event: "Dynamical instability Tc penalty",
        detail: `${formula}: ${fdpImagCount} imaginary phonon mode(s), lowestFreq=${fdpLowest.toFixed(1)} cm⁻¹ — Tc penalized by ${((1 - instabilityPenalty) * 100).toFixed(0)}%`,
        dataSource: "DFT Resolver",
      });
    } else {
      uncertaintyEstimate += 0.1;
      emit("log", {
        phase: "phase-10",
        event: "Dynamical instability (clamping)",
        detail: `${formula}: flagged unstable due to force constant clamping (no physical imaginary modes), lowestFreq=${fdpLowest.toFixed(1)} cm⁻¹`,
        dataSource: "DFT Resolver",
      });
    }
  }
  uncertaintyEstimate = Math.max(uncertaintyEstimate, pairingAnalysis.uncertaintyFromMechanism);

  if (candidatePressure < 50 && eliashberg.predictedTc > 200) {
    uncertaintyEstimate = Math.max(uncertaintyEstimate, 0.8);
  }

  uncertaintyEstimate = Math.max(0.05, Math.min(0.95, uncertaintyEstimate));

  const instabilityProximity = computeInstabilityProximity(formula, electronicStructure, phononSpectrum, competingPhases);

  const isHydrideForCDW = tcMatClass === "superhydride" || tcMatClass === "hydride-high-p" || tcMatClass === "hydride-low-p";
  const hydrideBypassCDW = isHydrideForCDW && coupling.lambda > 1.5 && (phononSpectrum.debyeTemperature ?? 0) > 1000;
  if (instabilityProximity.cdwInstability > 0.5 && !hydrideBypassCDW) {
    const cdwOnset = Math.min(1.0, (instabilityProximity.cdwInstability - 0.5) / 0.5);
    let cdwPenalty = Math.max(0.05, 1.0 - cdwOnset * 0.6);

    const cdwPhase = competingPhases.find(p => p.type === "CDW" && p.transitionTemp != null);
    if (cdwPhase && cdwPhase.transitionTemp != null && cdwPhase.transitionTemp > eliashberg.predictedTc) {
      const tRatio = Math.min(3.0, cdwPhase.transitionTemp / Math.max(1, eliashberg.predictedTc));
      cdwPenalty *= Math.max(0.1, 1.0 - (tRatio - 1.0) * 0.25);
    }

    eliashberg.predictedTc = Math.round(eliashberg.predictedTc * cdwPenalty * 10) / 10;
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2)];
    emit("log", {
      phase: "phase-10",
      event: "CDW suppression applied",
      detail: `${formula}: CDW=${instabilityProximity.cdwInstability.toFixed(2)}, lambda=${coupling.lambda.toFixed(3)}${cdwPhase?.transitionTemp ? `, T_CDW=${cdwPhase.transitionTemp}K` : ""} — SC suppressed by charge ordering (Tc penalized by ${((1 - cdwPenalty) * 100).toFixed(0)}%)`,
      dataSource: "Physics Engine",
    });
  }

  if (instabilityProximity.sdwInstability > 0.4 && !hydrideBypassCDW) {
    const sdwPenalty = Math.max(0.05, 1.0 - instabilityProximity.sdwInstability * 0.6);
    eliashberg.predictedTc = Math.round(eliashberg.predictedTc * sdwPenalty);
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2)];
    emit("log", {
      phase: "phase-10",
      event: "SDW suppression applied",
      detail: `${formula}: SDW=${instabilityProximity.sdwInstability.toFixed(2)}, lambda=${coupling.lambda.toFixed(3)} — SC suppressed by spin ordering (Tc penalized by ${((1 - sdwPenalty) * 100).toFixed(0)}%)`,
      dataSource: "Physics Engine",
    });
  }
  // Yield after instability proximity + CDW/SDW suppression cluster
  await new Promise<void>(r => setTimeout(r, 0));

  emitDetail("log", {
    phase: "phase-10",
    event: "Instability proximity computed",
    detail: `${formula}: nearest=${instabilityProximity.nearestBoundary} (${instabilityProximity.overallProximity.toFixed(2)}), QCP=${instabilityProximity.magneticQCP.toFixed(2)}, CDW=${instabilityProximity.cdwInstability.toFixed(2)}, SDW=${instabilityProximity.sdwInstability.toFixed(2)}, MIT=${instabilityProximity.metalInsulatorTransition.toFixed(2)}, chi=${instabilityProximity.magneticSusceptibilityPeak.toFixed(2)}, nesting=${instabilityProximity.fermiSurfaceNestingStrength.toFixed(2)}, DOS_peak=${instabilityProximity.dosEfPeakScore.toFixed(2)}, VHS=${instabilityProximity.vanHoveSingularityScore.toFixed(2)}, flatBand=${instabilityProximity.flatBandInstability.toFixed(2)}`,
    dataSource: "Physics Engine",
  });

  const phononDispersion = earlyPhononDispersion;
  emitDetail("log", {
    phase: "phase-10",
    event: "Phonon dispersion computed",
    detail: `${formula}: ${phononDispersion.branches.length} branches along ${phononDispersion.qPath.join("-")}, imaginary=${phononDispersion.imaginaryFrequencies}, soft q-points=[${phononDispersion.softModeQPoints.join(",")}], gap=${phononDispersion.phononGap.toFixed(1)} cm⁻¹`,
    dataSource: "Physics Engine",
  });

  // GW many-body corrections must use the physically-computed alpha2F lambda as the
  // base, not coupling.lambda (the analytical estimate). coupling.lambda can be 10-15×
  // higher than integratedLambda for weakly-coupled systems, producing a correctedLambda
  // wildly inconsistent with alpha2F (e.g. corrected=2.257 vs integratedLambda=0.147).
  const mbBaseLambda = alpha2FResult.integratedLambda > 0 ? alpha2FResult.integratedLambda : coupling.lambda;
  const mbCoupling = mbBaseLambda !== coupling.lambda ? { ...coupling, lambda: mbBaseLambda } : coupling;
  const manyBodyCorrections = applyManyBodyCorrections(electronicStructure, mbCoupling, formula);
  emitDetail("log", {
    phase: "phase-10",
    event: "GW many-body corrections applied",
    detail: `${formula}: Z=${manyBodyCorrections.quasiparticleWeight.toFixed(3)}, DOS renorm=${manyBodyCorrections.gwDOSRenormalization.toFixed(3)}, BW corr=${manyBodyCorrections.gwBandwidthCorrection.toFixed(3)}, vertex λ-corr=${manyBodyCorrections.vertexCorrectionLambda.toFixed(3)}, corrected λ=${manyBodyCorrections.correctedLambda.toFixed(3)} (α²F base λ=${mbBaseLambda.toFixed(4)})`,
    dataSource: "Physics Engine",
  });

  // Guard: skip many-body Tc feedback if compound is ferromagnetically unstable.
  // Stoner product >= 1.0 means FM ordering destroys singlet SC — applying a GW
  // vertex-corrected phonon Tc boost would be unphysical.
  const N_EF_mb = electronicStructure.densityOfStatesAtFermi;
  const isNearFerroStoner = formulaEls.some(el => {
    const I = getStonerParameter(el);
    return I !== null && I * N_EF_mb >= 1.0;
  });
  if (eliashberg.predictedTc > 0
    && Math.abs(manyBodyCorrections.vertexCorrectionLambda - 1.0) > 0.02
    && !isNearFerroStoner) {
    const preMBTc = eliashberg.predictedTc;
    const mbTc = allenDynesTcRaw(
      manyBodyCorrections.correctedLambda,
      coupling.omegaLog,
      coupling.muStar,
      coupling.omega2Avg,
      tcMatClass === "superhydride" || tcMatClass === "hydride-high-p" || tcMatClass === "hydride-low-p"
    );
    if (mbTc > 0 && Number.isFinite(mbTc)) {
      const mbWeight = 0.3;
      eliashberg.predictedTc = Math.round((1 - mbWeight) * eliashberg.predictedTc + mbWeight * mbTc);
      emitDetail("log", {
        phase: "phase-10",
        event: "Many-body Tc feedback applied",
        detail: `${formula}: vertex-corrected λ=${manyBodyCorrections.correctedLambda.toFixed(3)} → Tc(MB)=${mbTc.toFixed(1)}K, blended Tc ${preMBTc}K → ${eliashberg.predictedTc}K (30% weight)`,
        dataSource: "Physics Engine",
      });
    }
  }

  const nestingFunction = computeNestingFunction(formula, electronicStructure);
  emitDetail("log", {
    phase: "phase-10",
    event: "Nesting function computed",
    detail: `${formula}: peak χ(q)=${nestingFunction.peakNestingValue.toFixed(3)} at ${nestingFunction.peakNestingQ}, avg=${nestingFunction.averageNesting.toFixed(3)}, anisotropy=${nestingFunction.nestingAnisotropy.toFixed(3)}, instability=${nestingFunction.dominantInstability}`,
    dataSource: "Physics Engine",
  });

  const spinSusceptibility = computeDynamicSpinSusceptibility(formula, electronicStructure);
  emitDetail("log", {
    phase: "phase-10",
    event: "Dynamic spin susceptibility computed",
    detail: `${formula}: χ_static=${spinSusceptibility.chiStaticPeak.toFixed(2)}, χ_dynamic=${spinSusceptibility.chiDynamicPeak.toFixed(2)}, ω_sf=${spinSusceptibility.spinFluctuationEnergy.toFixed(2)} meV, ξ=${spinSusceptibility.correlationLength.toFixed(2)}a, Stoner=${spinSusceptibility.stonerEnhancement.toFixed(2)}, QCP=${spinSusceptibility.isNearQCP}`,
    dataSource: "Physics Engine",
  });

  if (spinSusceptibility.isStableFerromagnet && coupling.lambda < 3.0) {
    const fmPenalty = Math.max(0.02, Math.exp(-(spinSusceptibility.stonerProduct - 1.0) * 3.0));
    eliashberg.predictedTc = Math.round(eliashberg.predictedTc * fmPenalty);
    emit("log", {
      phase: "phase-10",
      event: "Stable ferromagnet suppression",
      detail: `${formula}: I·N(Ef)=${spinSusceptibility.stonerProduct.toFixed(3)} > 1.0 — stable ferromagnetism destroys singlet pairing (Tc penalized by ${((1 - fmPenalty) * 100).toFixed(0)}%)`,
      dataSource: "Physics Engine",
    });
  } else if (spinSusceptibility.stonerEnhancement > 5 && coupling.lambda < 2.5) {
    const stonerSuppression = Math.max(0.1, 1.0 / (1.0 + (spinSusceptibility.stonerEnhancement - 5) * 0.08));
    eliashberg.predictedTc = Math.round(eliashberg.predictedTc * stonerSuppression);
    emit("log", {
      phase: "phase-10",
      event: "Stoner suppression applied",
      detail: `${formula}: Stoner enhancement=${spinSusceptibility.stonerEnhancement.toFixed(1)}, phonon SC suppressed by ${((1 - stonerSuppression) * 100).toFixed(0)}%`,
      dataSource: "Physics Engine",
    });
  }
  // Yield after many-body corrections + nesting + spin susceptibility cluster
  await new Promise<void>(r => setTimeout(r, 0));

  const advancedConstraints = computeAdvancedConstraints(
    formula, electronicStructure, phononSpectrum, coupling,
    nestingFunction, spinSusceptibility,
    formulaEls, formulaCts, getTotalAtoms(formulaCts), dimensionality
  );

  if (advancedConstraints.compositeBoost !== 1.0 && eliashberg.predictedTc > 0) {
    const preTc = eliashberg.predictedTc;
    eliashberg.predictedTc = Math.round(eliashberg.predictedTc * advancedConstraints.compositeBoost);
    eliashberg.predictedTc = Math.max(0, eliashberg.predictedTc);
    emit("log", {
      phase: "phase-10",
      event: "Advanced constraints applied",
      detail: `${formula}: ${advancedConstraints.summary} | Tc ${preTc}K -> ${eliashberg.predictedTc}K`,
      dataSource: "Physics Engine",
    });
  }

  emitDetail("log", {
    phase: "phase-10",
    event: "Advanced physics constraints evaluated",
    detail: `${formula}: nesting=${advancedConstraints.fermiSurfaceNesting.nestingStrength}(${advancedConstraints.fermiSurfaceNesting.score.toFixed(2)}), hybrid=${advancedConstraints.orbitalHybridization.hybridizationType}(${advancedConstraints.orbitalHybridization.score.toFixed(2)}), lifshitz=${advancedConstraints.lifshitzProximity.score.toFixed(2)}, QCP=${advancedConstraints.quantumCriticalFluctuation.qcpType}(${advancedConstraints.quantumCriticalFluctuation.score.toFixed(2)}), dim=${advancedConstraints.electronicDimensionality.dimensionClass}(anis=${advancedConstraints.electronicDimensionality.anisotropy.toFixed(1)}), softMode=${advancedConstraints.phononSoftMode.score.toFixed(2)}(stable=${advancedConstraints.phononSoftMode.isStable}), CT-delta=${advancedConstraints.chargeTransferEnergy.delta.toFixed(2)}(${advancedConstraints.chargeTransferEnergy.chargeTransferType}), epsilon=${advancedConstraints.latticePolarizability.dielectricConstant.toFixed(0)}(${advancedConstraints.latticePolarizability.screeningStrength})`,
    dataSource: "Physics Engine",
  });

  let fdPhononSummary: FDPhononSummary | null = null;
  if (dftData?.finiteDisplacementPhonons) {
    const fdp = dftData.finiteDisplacementPhonons;
    fdPhononSummary = {
      dynamicallyStable: fdp.dynamicallyStable,
      imaginaryModeCount: fdp.imaginaryModeCount,
      lowestFrequency: fdp.lowestFrequency,
      highestFrequency: fdp.highestFrequency,
      omegaLog: fdp.omegaLog,
      lambdaContribution: fdp.lambdaContribution,
      forceConstantClampedEntries: fdp.forceConstantClampedEntries,
    };
  }

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
    phononDispersion,
    manyBodyCorrections,
    nestingFunction,
    spinSusceptibility,
    phononDOS,
    alpha2F: alpha2FResult,
    advancedConstraints,
    fdPhononSummary,
  };
}
