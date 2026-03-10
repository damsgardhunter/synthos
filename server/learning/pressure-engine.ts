import type { EventEmitter } from "./engine";
import {
  ELEMENTAL_DATA,
  getElementData,
  getCompositionWeightedProperty,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "./elemental-data";
import {
  parseFormulaElements,
  classifyHydrogenBonding,
} from "./physics-engine";
import { predictEquilibriumLatticeConstant, isVolumeDNNTrained } from "../crystal/volume-predictor-dnn";

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

function estimateDefaultBulkModulus(el: string): number {
  if (el === "H") return 1;
  if (["He", "Ne", "Ar", "Kr", "Xe"].includes(el)) return 2;
  if (["Li", "Na", "K", "Rb", "Cs"].includes(el)) return 12;
  if (["Be", "Mg", "Ca", "Sr", "Ba"].includes(el)) return 25;
  if (["B", "C", "Si", "Ge"].includes(el)) return 40;
  if (["N", "O", "F", "Cl", "Br", "I"].includes(el)) return 10;
  if (["P", "S", "Se", "Te"].includes(el)) return 15;
  const POST_TM_BULK: Record<string, number> = { Al: 76, Ga: 56, In: 42, Sn: 58, Pb: 46, Bi: 31, Tl: 43 };
  if (POST_TM_BULK[el]) return POST_TM_BULK[el];
  if (isTransitionMetal(el)) return 120;
  if (isRareEarth(el)) return 35;
  if (isActinide(el)) return 50;
  return 50;
}

export interface BirchMurnaghanResult {
  compressedVolume: number;
  compressedLattice: { a: number; b: number; c: number };
  bulkModulus: number;
  pressureDerivative: number;
}

export interface HydrideFormationResult {
  stableHydrides: { formula: string; Hf: number; pressure: number }[];
  hydrogenCapacity: number;
  decompositionPressure: number;
}

export interface HighPressureStabilityResult {
  isStable: boolean;
  compressedVolume: number;
  phononStable: boolean;
  enthalpyMargin: number;
  decompositionProducts: string[];
}

export interface PressureTcPoint {
  pressure: number;
  Tc: number;
  lambda: number;
  stable: boolean;
  hc2: number;
}

export function relaxStructureAtPressure(
  formula: string,
  pressure: number,
  latticeParams?: { a: number; b: number; c: number }
): BirchMurnaghanResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  let B0 = 0;
  let totalWeight = 0;
  const hRatio = (counts["H"] || 0) / totalAtoms;
  const isHydrideComp = elements.includes("H") && hRatio > 0.3;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    if (isHydrideComp && el === "H") {
      if (pressure >= 100) {
        const hBulk = 10 + pressure * 0.3;
        const frac = (counts[el] || 1) / totalAtoms;
        B0 += hBulk * frac;
        totalWeight += frac;
      }
      continue;
    }
    const frac = (counts[el] || 1) / totalAtoms;
    const elBulk = data.bulkModulus ?? estimateDefaultBulkModulus(el);
    B0 += elBulk * frac;
    totalWeight += frac;
  }
  if (totalWeight > 0) B0 /= totalWeight;
  else B0 = 50;
  B0 = Math.max(10, B0);

  const B0p = 4.0 + (elements.length - 1) * 0.3;

  let V0_a = 0;
  if (isVolumeDNNTrained()) {
    V0_a = predictEquilibriumLatticeConstant(formula, pressure);
  }
  if (V0_a <= 0) {
    for (const el of elements) {
      const data = getElementData(el);
      if (data && data.latticeConstant) {
        V0_a += data.latticeConstant * (counts[el] || 1);
      } else if (data) {
        V0_a += data.atomicRadius * 2.5 * (counts[el] || 1);
      }
    }
    V0_a = Math.max(3.0, V0_a / totalAtoms);
  }

  const a0 = latticeParams?.a ?? V0_a;
  const b0 = latticeParams?.b ?? V0_a * 1.0;
  const c0 = latticeParams?.c ?? V0_a * 1.0;
  const V0 = a0 * b0 * c0;

  let eta = 1.0;
  if (pressure > 0 && B0 > 0) {
    const pOverB = pressure / B0;
    const inner = 1 + B0p * pOverB;
    if (inner > 0) {
      eta = Math.pow(inner, -1 / B0p);
    } else {
      eta = 0.5;
    }
    eta = Math.max(0.5, Math.min(1.0, eta));
  }

  const compressedVolume = V0 * eta;
  const cubicEta = Math.pow(eta, 1 / 3);

  return {
    compressedVolume: Math.round(compressedVolume * 1000) / 1000,
    compressedLattice: {
      a: Math.round(a0 * cubicEta * 1000) / 1000,
      b: Math.round(b0 * cubicEta * 1000) / 1000,
      c: Math.round(c0 * cubicEta * 1000) / 1000,
    },
    bulkModulus: Math.round(B0 * 10) / 10,
    pressureDerivative: Math.round(B0p * 100) / 100,
  };
}

const KNOWN_HYDRIDE_FORMERS: Record<string, { maxH: number; Hf0: number; optimalP: number }> = {
  La: { maxH: 10, Hf0: -0.3, optimalP: 170 },
  Y: { maxH: 9, Hf0: -0.25, optimalP: 200 },
  Ca: { maxH: 6, Hf0: -0.15, optimalP: 200 },
  Th: { maxH: 10, Hf0: -0.28, optimalP: 175 },
  Ce: { maxH: 9, Hf0: -0.22, optimalP: 150 },
  Ac: { maxH: 10, Hf0: -0.35, optimalP: 150 },
  Mg: { maxH: 6, Hf0: -0.10, optimalP: 250 },
  Ba: { maxH: 6, Hf0: -0.18, optimalP: 200 },
  Sr: { maxH: 6, Hf0: -0.14, optimalP: 200 },
  Sc: { maxH: 6, Hf0: -0.20, optimalP: 250 },
  Ti: { maxH: 4, Hf0: -0.12, optimalP: 250 },
  Zr: { maxH: 4, Hf0: -0.15, optimalP: 200 },
  Li: { maxH: 8, Hf0: -0.08, optimalP: 300 },
  Na: { maxH: 8, Hf0: -0.06, optimalP: 300 },
  K: { maxH: 6, Hf0: -0.05, optimalP: 300 },
  Pd: { maxH: 2, Hf0: -0.05, optimalP: 50 },
  Fe: { maxH: 6, Hf0: -0.10, optimalP: 200 },
};

export function predictHydrideFormation(
  metalElements: string[],
  pressure: number
): HydrideFormationResult {
  const stableHydrides: { formula: string; Hf: number; pressure: number }[] = [];
  let totalHCapacity = 0;
  let minDecompP = Infinity;

  const H2_DISSOCIATION = 4.52;

  for (const metal of metalElements) {
    const known = KNOWN_HYDRIDE_FORMERS[metal];
    const data = getElementData(metal);
    if (!data) continue;

    let maxH: number;
    let Hf0: number;
    let optP: number;

    if (known) {
      maxH = known.maxH;
      Hf0 = known.Hf0;
      optP = known.optimalP;
    } else {
      const en = data.paulingElectronegativity ?? 1.5;
      maxH = Math.max(2, Math.round(8 - en * 3));
      Hf0 = -(2.5 - en) * 0.15;
      optP = 200 + en * 50;
    }

    const pressureStabilization = pressure >= optP * 0.5
      ? -0.3 * Math.min(1, pressure / optP)
      : -0.1 * pressure / optP;

    const Hf = Hf0 + pressureStabilization - H2_DISSOCIATION * 0.01 * (1 - pressure / 300);

    const hFrac = Math.min(1, Math.max(0.3, pressure / optP));
    const discreteSteps = [2, 3, 4, 6, 8, 10, 12];
    const targetH = maxH * hFrac;
    let effectiveH = discreteSteps[0];
    for (const step of discreteSteps) {
      if (step <= maxH && Math.abs(step - targetH) < Math.abs(effectiveH - targetH)) {
        effectiveH = step;
      }
    }

    if (effectiveH >= 2 && Hf < 0) {
      const hydrideFormula = `${metal}H${effectiveH}`;
      stableHydrides.push({
        formula: hydrideFormula,
        Hf: Math.round(Hf * 1000) / 1000,
        pressure: Math.round(pressure),
      });

      const molarMass = data.atomicMass + effectiveH * 1.008;
      const hMassFrac = (effectiveH * 1.008) / molarMass;
      totalHCapacity += hMassFrac;

      const decompP = Math.max(0, optP * 0.3 * (-Hf0 / 0.3));
      if (decompP < minDecompP) minDecompP = decompP;
    }
  }

  if (metalElements.length > 1 && stableHydrides.length >= 2 && pressure >= 100) {
    const combinedFormula = metalElements.join("") + `H${Math.round(metalElements.length * 4)}`;
    const avgHf = stableHydrides.reduce((s, h) => s + h.Hf, 0) / stableHydrides.length;
    const synergy = -0.05 * (metalElements.length - 1);
    stableHydrides.push({
      formula: combinedFormula,
      Hf: Math.round((avgHf + synergy) * 1000) / 1000,
      pressure: Math.round(pressure),
    });
  }

  return {
    stableHydrides,
    hydrogenCapacity: Math.round(totalHCapacity * 10000) / 10000,
    decompositionPressure: minDecompP === Infinity ? 0 : Math.round(minDecompP),
  };
}

export function assessHighPressureStability(
  formula: string,
  targetPressure: number
): HighPressureStabilityResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const bm = relaxStructureAtPressure(formula, targetPressure);

  const hCount = counts["H"] || 0;
  const hRatio = (totalAtoms - hCount) > 0 ? hCount / (totalAtoms - hCount) : 0;

  let phononStable = true;
  if (targetPressure > 300) {
    let fHash = 0;
    for (let ci = 0; ci < formula.length; ci++) fHash = ((fHash << 5) - fHash + formula.charCodeAt(ci)) | 0;
    const pseudoRand = ((fHash * 2654435761) >>> 0) / 4294967296;
    phononStable = pseudoRand > 0.3 && bm.bulkModulus > 30;
  } else if (targetPressure > 200) {
    phononStable = bm.bulkModulus > 50;
  }

  if (hRatio >= 6 && targetPressure >= 100) {
    phononStable = targetPressure <= 350;
  }

  const hBondType = classifyHydrogenBonding(formula, targetPressure);
  if (hBondType === "metallic-network" || hBondType === "cage-clathrate") {
    phononStable = phononStable && targetPressure <= 400;
  }

  let enthalpyMargin = 0;
  const avgEN = elements.reduce((s, el) => {
    const d = getElementData(el);
    return s + (d?.paulingElectronegativity ?? 1.8) * (counts[el] || 1);
  }, 0) / totalAtoms;

  const enSpread = (() => {
    const enValues = elements.map(el => getElementData(el)?.paulingElectronegativity ?? 1.8);
    if (enValues.length < 2) return 0;
    return Math.max(...enValues) - Math.min(...enValues);
  })();

  enthalpyMargin = -0.1 * (1 + targetPressure / 200) + enSpread * 0.05;
  if (hRatio >= 6) enthalpyMargin -= 0.2 * Math.min(1, targetPressure / 200);

  const decompositionProducts: string[] = [];
  if (elements.length >= 2) {
    for (const el of elements) {
      if (el === "H" && hCount > 0) continue;
      decompositionProducts.push(el);
    }
    if (hCount > 0) decompositionProducts.push("H2");
  }

  const isStable = phononStable && enthalpyMargin < 0;

  return {
    isStable,
    compressedVolume: bm.compressedVolume,
    phononStable,
    enthalpyMargin: Math.round(enthalpyMargin * 1000) / 1000,
    decompositionProducts,
  };
}

export function scanPressureTcCurve(
  formula: string,
  pressureRange: { min: number; max: number; steps: number } = { min: 0, max: 300, steps: 31 }
): PressureTcPoint[] {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const hCount = counts["H"] || 0;
  const nonmetals = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const metalElements = elements.filter(e => !nonmetals.includes(e));
  const metalAtomCount = metalElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;

  const isHydride = hCount > 0 && hRatio >= 2;
  const isSuperhydride = hRatio >= 6;
  const hasNi = elements.includes("Ni");
  const hasO = elements.includes("O");
  const isNickelate = hasNi && hasO && elements.some(e => ["La", "Nd", "Pr", "Sr", "Ba", "Ca"].includes(e));

  let B0 = 0;
  let weightSum = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    B0 += (data.bulkModulus ?? estimateDefaultBulkModulus(el)) * frac;
    weightSum += frac;
  }
  if (weightSum > 0) B0 /= weightSum;
  B0 = Math.max(10, B0);

  let lambda0 = 0;
  let lambdaWeight = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    const eta = data.mcMillanHopfieldEta ?? 0;
    if (eta > 0 && data.debyeTemperature && data.debyeTemperature > 0) {
      const lambdaEl = (eta * 562000) / (data.atomicMass * data.debyeTemperature * data.debyeTemperature);
      lambda0 += lambdaEl * frac;
      lambdaWeight += frac;
    }
  }
  if (lambdaWeight > 0) {
    lambda0 /= lambdaWeight;
  } else {
    lambda0 = 0.5;
  }

  if (isHydride) {
    lambda0 = Math.max(lambda0, 0.8 + hRatio * 0.1);
  }

  let omegaLog0 = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    const td = data.debyeTemperature ?? 300;
    omegaLog0 += td * 0.695 * 0.45 * frac;
  }
  if (isHydride) {
    const hFrac = hCount / totalAtoms;
    omegaLog0 = omegaLog0 * (1 - hFrac) + 800 * hFrac;
  }
  omegaLog0 = Math.max(50, omegaLog0);

  const muStar = isHydride ? 0.12 : 0.13;

  const { min: pMin, max: pMax, steps } = pressureRange;
  const stepSize = (pMax - pMin) / Math.max(1, steps - 1);
  const result: PressureTcPoint[] = [];

  for (let i = 0; i < steps; i++) {
    const P = pMin + i * stepSize;

    const volumeRatio = Math.pow(1 + 4.0 * P / Math.max(B0, 1), -1 / 4);
    const hardeningFactor = 1 / Math.max(0.5, Math.pow(volumeRatio, 2));

    let lambdaP = lambda0;
    if (isHydride && P >= 50) {
      const pressureBoost = isSuperhydride
        ? 1.0 + Math.min(2.0, (P / 150) * 1.5)
        : 1.0 + Math.min(1.2, (P / 200) * 0.8);
      lambdaP = lambda0 * pressureBoost;

      if (P > 250 && isSuperhydride) {
        const overPressure = (P - 250) / 100;
        lambdaP *= Math.max(0.7, 1 - overPressure * 0.3);
      }
    } else if (isNickelate) {
      const optP = 20;
      const sigmaP = 15;
      const domeBoost = 1.0 + 1.5 * Math.exp(-Math.pow(P - optP, 2) / (2 * sigmaP * sigmaP));
      lambdaP = lambda0 * domeBoost;
    } else {
      const dosEnhancement = 1 + P / 500 * 0.3;
      lambdaP = lambda0 * dosEnhancement / hardeningFactor;
    }
    lambdaP = Math.max(0.05, Math.min(4.0, lambdaP));

    let omegaLogP = omegaLog0 * Math.pow(hardeningFactor, 0.5);
    if (isHydride && P >= 50) {
      omegaLogP *= 1 + Math.min(0.5, P / 400);
    }
    omegaLogP = Math.max(30, Math.min(2000, omegaLogP));

    const omegaLogK = omegaLogP * 1.44;
    const denom = lambdaP - muStar * (1 + 0.62 * lambdaP);
    let Tc = 0;
    if (denom > 0.001) {
      const f1 = lambdaP < 1.5
        ? Math.pow(1 + (lambdaP / 2.46 / (1 + 3.8 * muStar)), 1 / 3)
        : Math.pow(1 + Math.pow(lambdaP / (2.46 * (1 + 3.8 * muStar)), 3/2), 1/3);
      const exponent = -1.04 * (1 + lambdaP) / denom;
      if (exponent > 50) { Tc = 0; } else {
        Tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
        if (!Number.isFinite(Tc) || Tc < 0) Tc = 0;
      }
      if (denom < 0.05) {
        Tc *= Math.pow(denom / 0.05, 0.5);
      }
    }

    const stability = assessHighPressureStability(formula, P);

    const hc2 = Tc > 0 ? 1.84 * Tc * Math.sqrt(1 + lambdaP) : 0;

    result.push({
      pressure: Math.round(P),
      Tc: Math.round(Tc * 10) / 10,
      lambda: Math.round(lambdaP * 1000) / 1000,
      stable: stability.isStable,
      hc2: Math.round(hc2 * 10) / 10,
    });
  }

  return result;
}

export function runPressureAnalysis(
  emit: EventEmitter,
  formula: string,
  pressureRange?: { min: number; max: number; steps: number }
): {
  pressureTcCurve: PressureTcPoint[];
  optimalPressure: number;
  maxTc: number;
  hydrideFormation: HydrideFormationResult | null;
} {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const hCount = counts["H"] || 0;
  const nonmetals = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const metalElements = elements.filter(e => !nonmetals.includes(e));
  const metalAtomCount = metalElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;
  const isHydride = hCount > 0 && hRatio >= 2;

  const curve = scanPressureTcCurve(formula, pressureRange);

  let maxTc = 0;
  let optimalPressure = 0;
  for (const pt of curve) {
    if (pt.Tc > maxTc && pt.stable) {
      maxTc = pt.Tc;
      optimalPressure = pt.pressure;
    }
  }
  if (maxTc === 0) {
    for (const pt of curve) {
      if (pt.Tc > maxTc) {
        maxTc = pt.Tc;
        optimalPressure = pt.pressure;
      }
    }
  }

  let hydrideFormation: HydrideFormationResult | null = null;
  if (isHydride || metalElements.length > 0) {
    hydrideFormation = predictHydrideFormation(metalElements, optimalPressure > 0 ? optimalPressure : 150);
  }

  emit("log", {
    phase: "phase-10",
    event: "Pressure-Tc curve computed",
    detail: `${formula}: optimal pressure=${optimalPressure} GPa, max Tc=${maxTc}K, ${curve.length} points, ${curve.filter(p => p.stable).length} stable${hydrideFormation ? `, ${hydrideFormation.stableHydrides.length} hydride phases` : ""}`,
    dataSource: "Pressure Engine",
  });

  return {
    pressureTcCurve: curve,
    optimalPressure,
    maxTc,
    hydrideFormation,
  };
}
