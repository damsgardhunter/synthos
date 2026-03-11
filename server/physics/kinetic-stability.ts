import { ELEMENTAL_DATA, getElementData } from "../learning/elemental-data";
import { computeMiedemaFormationEnergy, assessMetastability } from "../learning/phase-diagram-engine";

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
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
  if (iterations >= 20) {
    console.warn(`[kinetic-stability] expandParentheses hit 20-iteration limit for formula: "${formula}" — result may be incomplete`);
  }
  if (/[()]/.test(result)) {
    console.warn(`[kinetic-stability] expandParentheses has dangling parentheses in: "${result}" from input: "${formula}"`);
  }
  return result.replace(/[()]/g, "");
}

function parseFormulaCounts(formula: string): Record<string, number> {
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

export interface GrainBoundaryAnalysis {
  averageGBEnergy: number;
  gbDecompositionFactor: number;
  grainSizeEffect: string;
}

export interface DiffusionBarrierAnalysis {
  elementBarriers: { element: string; barrier: number; diffusionRate300K: number }[];
  rateControllingElement: string;
  effectiveBarrier: number;
  effectiveDiffusionRate300K: number;
}

export interface NucleationBarrierAnalysis {
  nucleationBarrier: number;
  criticalNucleusRadius: number;
  nucleationRate300K: number;
  competingPhases: string[];
}

export interface PressureStabilizationAnalysis {
  estimatedBulkModulus: number;
  compressibility: number;
  criticalDecompressionRate: number;
  pressureRetentionWindow: string;
  ambientStabilizable: boolean;
  minStabilizationPressure: number;
}

export interface StabilizationStrategy {
  strategy: string;
  mechanism: string;
  expectedLifetimeImprovement: string;
  difficulty: "easy" | "moderate" | "hard" | "very_hard";
  applicability: number;
}

export interface KineticStabilityResult {
  formula: string;
  kineticScore: number;
  metastableLifetime300K: number;
  lifetimeString: string;
  confidenceLow: number;
  confidenceHigh: number;
  grainBoundary: GrainBoundaryAnalysis;
  diffusionBarriers: DiffusionBarrierAnalysis;
  nucleationBarrier: NucleationBarrierAnalysis;
  pressureStabilization: PressureStabilizationAnalysis;
  stabilizationStrategies: StabilizationStrategy[];
  overallVerdict: string;
}

const kB = 8.617e-5;

function computeGrainBoundaryEnergy(formula: string): GrainBoundaryAnalysis {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  let avgSurfaceEnergy = 0;
  let avgBulkModulus = 0;
  let avgShearModulus = 0;
  let totalWeight = 0;
  const isHydride = elements.includes("H") && ((counts["H"] || 0) / totalAtoms) > 0.3;

  for (const el of elements) {
    const data = getElementData(el);
    const frac = (counts[el] || 1) / totalAtoms;
    const mp = data?.meltingPoint ?? 1000;
    const bulk = data?.bulkModulus ?? 50;
    const shear = bulk * 0.4;
    const surfE = mp * 0.0012 + bulk * 0.005;
    avgSurfaceEnergy += surfE * frac;
    avgBulkModulus += bulk * frac;
    avgShearModulus += shear * frac;
    totalWeight += frac;
  }
  avgSurfaceEnergy /= Math.max(totalWeight, 0.01);
  avgBulkModulus /= Math.max(totalWeight, 0.01);
  avgShearModulus /= Math.max(totalWeight, 0.01);

  const nElements = elements.length;
  const complexityFactor = 1 + 0.15 * Math.log(Math.max(nElements, 1));

  const shearContribution = avgShearModulus * 0.008;
  const averageGBEnergy = (avgSurfaceEnergy * 0.35 + shearContribution) * complexityFactor;

  const formE = computeMiedemaFormationEnergy(formula);

  let gbDecompositionFactor: number;
  if (formE > 0) {
    gbDecompositionFactor = Math.min(1.0, averageGBEnergy / Math.max(formE * 5, 0.1));
  } else {
    gbDecompositionFactor = Math.min(0.3, averageGBEnergy / (Math.abs(formE) * 10 + 1));
  }

  const radii: number[] = [];
  for (const el of elements) {
    const data = getElementData(el);
    if (data?.atomicRadius) radii.push(data.atomicRadius);
  }
  if (radii.length >= 2) {
    const minR = Math.min(...radii);
    const maxR = Math.max(...radii);
    const radiusMismatch = (maxR - minR) / Math.max(maxR, 1);
    if (radiusMismatch > 0.15) {
      const strainPenalty = Math.min(0.4, (radiusMismatch - 0.15) * 2.0);
      gbDecompositionFactor = Math.min(1.0, gbDecompositionFactor + strainPenalty);
    }
  }

  if (isHydride) {
    const shearStressFactor = Math.min(0.3, avgShearModulus * 0.003);
    gbDecompositionFactor = Math.min(1.0, gbDecompositionFactor + shearStressFactor);
  }

  let grainSizeEffect: string;
  if (averageGBEnergy < 0.3) {
    grainSizeEffect = "Low GB energy — grain boundaries unlikely to accelerate decomposition";
  } else if (averageGBEnergy < 0.8) {
    grainSizeEffect = isHydride
      ? "Moderate GB energy — H diffusion along grain boundaries may initiate decomposition"
      : "Moderate GB energy — nanostructuring may accelerate decomposition at grain boundaries";
  } else {
    grainSizeEffect = isHydride
      ? "High GB energy — grain boundaries are primary H desorption pathways and decomposition nucleation sites"
      : "High GB energy — grain boundaries are likely decomposition nucleation sites";
  }

  return {
    averageGBEnergy: Math.round(averageGBEnergy * 1000) / 1000,
    gbDecompositionFactor: Math.round(gbDecompositionFactor * 1000) / 1000,
    grainSizeEffect,
  };
}

function computeDiffusionBarriers(formula: string): DiffusionBarrierAnalysis {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const attemptFreq = 1e13;
  const isHydride = elements.includes("H") && ((counts["H"] || 0) / totalAtoms) > 0.3;
  const elementBarriers: { element: string; barrier: number; diffusionRate300K: number }[] = [];

  for (const el of elements) {
    const data = getElementData(el);
    const mass = data?.atomicMass ?? 50;
    const radius = data?.atomicRadius ?? 130;
    const mp = data?.meltingPoint ?? 1000;
    const bulk = data?.bulkModulus ?? 50;

    let barrierBase = mp * kB * 18;
    const massCorrection = 0.02 * Math.sqrt(mass / 50);
    const sizeCorrection = 0.01 * (radius / 130);
    const stiffnessCorrection = 0.005 * Math.sqrt(bulk / 100);

    if (el === "H") {
      const tunnelingReduction = 0.3;
      barrierBase *= (1 - tunnelingReduction);
    }

    const barrier = barrierBase + massCorrection + sizeCorrection + stiffnessCorrection;

    let effectiveAttemptFreq = attemptFreq;
    if (el === "H") {
      effectiveAttemptFreq *= 3;
    }
    const rate = effectiveAttemptFreq * Math.exp(-barrier / (kB * 300));

    elementBarriers.push({
      element: el,
      barrier: Math.round(barrier * 10000) / 10000,
      diffusionRate300K: rate,
    });
  }

  elementBarriers.sort((a, b) => a.barrier - b.barrier);

  const fastestDiffuser = elementBarriers[0];

  const minBarrier = fastestDiffuser.barrier;

  let effectiveBarrier: number;
  if (isHydride && fastestDiffuser.element === "H") {
    const hFrac = (counts["H"] || 0) / totalAtoms;
    effectiveBarrier = minBarrier * (0.7 + 0.3 * (1 - hFrac));
  } else {
    effectiveBarrier = minBarrier;
    const nElements = elements.length;
    if (nElements > 2) {
      const latticeTrappingBonus = 0.03 * (nElements - 2);
      effectiveBarrier += latticeTrappingBonus;
    }
  }

  const effectiveRate = attemptFreq * Math.exp(-effectiveBarrier / (kB * 300));

  return {
    elementBarriers,
    rateControllingElement: fastestDiffuser.element,
    effectiveBarrier: Math.round(effectiveBarrier * 10000) / 10000,
    effectiveDiffusionRate300K: effectiveRate,
  };
}

function computeNucleationBarrier(formula: string, eAboveHull: number): NucleationBarrierAnalysis {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  let avgBulkModulus = 0;
  let avgMp = 0;
  let totalFrac = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const frac = (counts[el] || 1) / totalAtoms;
    avgBulkModulus += (data?.bulkModulus ?? 50) * frac;
    avgMp += (data?.meltingPoint ?? 1000) * frac;
    totalFrac += frac;
  }
  avgBulkModulus /= Math.max(totalFrac, 0.01);
  avgMp /= Math.max(totalFrac, 0.01);

  const surfaceEnergy = avgMp * 0.0008 + avgBulkModulus * 0.003;
  const deltaGv = Math.max(eAboveHull, 0.001);

  const criticalRadius = 2 * surfaceEnergy / (deltaGv * 10 + 0.01);
  const nucleationBarrier = (16 * Math.PI * Math.pow(surfaceEnergy, 3)) /
    (3 * Math.pow(deltaGv * 10 + 0.01, 2));

  const effectiveBarrier = Math.min(5.0, Math.max(0.05, nucleationBarrier * 0.1));

  const attemptFreq = 1e13;
  const nucleationRate = attemptFreq * Math.exp(-effectiveBarrier / (kB * 300));

  const competingPhases: string[] = [];
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const binary = `${elements[i]}${elements[j]}`;
      const binaryE = computeMiedemaFormationEnergy(binary);
      if (binaryE < -0.1) {
        competingPhases.push(binary);
      }
    }
  }
  if (competingPhases.length === 0) {
    competingPhases.push(...elements);
  }

  return {
    nucleationBarrier: Math.round(effectiveBarrier * 10000) / 10000,
    criticalNucleusRadius: Math.round(criticalRadius * 100) / 100,
    nucleationRate300K: nucleationRate,
    competingPhases: competingPhases.slice(0, 5),
  };
}

function computePressureStabilization(formula: string, eAboveHull: number): PressureStabilizationAnalysis {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  let avgBulkModulus = 0;
  let avgMass = 0;
  let avgMp = 0;
  let totalFrac = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const frac = (counts[el] || 1) / totalAtoms;
    avgBulkModulus += (data?.bulkModulus ?? 50) * frac;
    avgMass += (data?.atomicMass ?? 50) * frac;
    avgMp += (data?.meltingPoint ?? 1000) * frac;
    totalFrac += frac;
  }
  avgBulkModulus /= Math.max(totalFrac, 0.01);
  avgMass /= Math.max(totalFrac, 0.01);
  avgMp /= Math.max(totalFrac, 0.01);

  const compressibility = 1 / Math.max(avgBulkModulus, 1);

  const hasH = elements.includes("H");
  const hFraction = hasH ? (counts["H"] || 0) / totalAtoms : 0;

  const minPressureGPa = eAboveHull > 0
    ? eAboveHull * avgBulkModulus * 0.5 + (hasH ? hFraction * 50 : 0)
    : 0;

  const diffusionBarrierEv = avgMp * kB * 18 * (hasH ? 0.7 : 1.0);
  const kineticTrappingEnergy = diffusionBarrierEv - eAboveHull;

  const decompressionPrefactor = avgBulkModulus * 0.001;
  const trappingFactor = Math.max(0, kineticTrappingEnergy) / Math.max(eAboveHull, 0.01);
  const hMobilityPenalty = hasH ? Math.exp(-hFraction * 2.5) : 1.0;

  const criticalDecompRate = decompressionPrefactor
    * Math.pow(trappingFactor, 1.5)
    * hMobilityPenalty
    * (1 + 0.1 * elements.length);

  const MIN_VIABLE_DECOMP_RATE = 0.5;

  let ambientStabilizable: boolean;
  if (minPressureGPa > 50) {
    ambientStabilizable = false;
  } else if (minPressureGPa > 10) {
    ambientStabilizable = criticalDecompRate >= MIN_VIABLE_DECOMP_RATE
      && kineticTrappingEnergy > 0.3
      && eAboveHull < 0.15;
  } else if (minPressureGPa > 1) {
    ambientStabilizable = criticalDecompRate >= MIN_VIABLE_DECOMP_RATE * 0.5
      && eAboveHull < 0.2;
  } else {
    ambientStabilizable = eAboveHull < 0.05 || (eAboveHull < 0.15 && avgBulkModulus > 150);
  }

  if (criticalDecompRate < MIN_VIABLE_DECOMP_RATE && minPressureGPa > 5) {
    ambientStabilizable = false;
  }

  let pressureRetention: string;
  if (minPressureGPa < 1) {
    pressureRetention = "Ambient-stable — no pressure required";
  } else if (minPressureGPa < 10) {
    const rateNote = criticalDecompRate < MIN_VIABLE_DECOMP_RATE
      ? "; kinetic trapping insufficient for ambient recovery"
      : "; controlled decompression may preserve metastable phase";
    pressureRetention = `Low pressure needed (~${Math.round(minPressureGPa)} GPa) — diamond anvil cell synthesis possible${rateNote}`;
  } else if (minPressureGPa < 100) {
    const rateNote = criticalDecompRate < MIN_VIABLE_DECOMP_RATE
      ? "; rapid H diffusion prevents ambient phase retention"
      : "; slow decompression required to trap metastable structure";
    pressureRetention = `Moderate pressure (~${Math.round(minPressureGPa)} GPa) — requires high-pressure synthesis${rateNote}`;
  } else {
    pressureRetention = `Very high pressure (~${Math.round(minPressureGPa)} GPa) — extreme conditions required; ambient recovery highly unlikely (critDecompRate=${criticalDecompRate.toFixed(2)} GPa/s)`;
  }

  return {
    estimatedBulkModulus: Math.round(avgBulkModulus * 10) / 10,
    compressibility: Math.round(compressibility * 100000) / 100000,
    criticalDecompressionRate: Math.round(criticalDecompRate * 1000) / 1000,
    pressureRetentionWindow: pressureRetention,
    ambientStabilizable,
    minStabilizationPressure: Math.round(minPressureGPa * 10) / 10,
  };
}

function identifyStabilizationStrategies(
  formula: string,
  eAboveHull: number,
  gb: GrainBoundaryAnalysis,
  diffusion: DiffusionBarrierAnalysis,
  nucleation: NucleationBarrierAnalysis,
  pressure: PressureStabilizationAnalysis,
): StabilizationStrategy[] {
  const strategies: StabilizationStrategy[] = [];
  const elements = parseFormulaElements(formula);
  const hasH = elements.includes("H");
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const hFraction = hasH ? (counts["H"] || 0) / totalAtoms : 0;

  if (eAboveHull < 0.2 && gb.gbDecompositionFactor > 0.3) {
    strategies.push({
      strategy: "Nanostructuring",
      mechanism: "Reduce grain size to limit GB-mediated decomposition nucleation sites; nano-confinement raises effective barrier",
      expectedLifetimeImprovement: "10x–100x through grain boundary engineering",
      difficulty: "moderate",
      applicability: Math.min(1.0, gb.gbDecompositionFactor * 1.5),
    });
  }

  if (diffusion.effectiveBarrier < 1.0) {
    const slowDiffusers = ["W", "Mo", "Ta", "Nb", "Re", "Os", "Ir", "Hf", "Zr"];
    const candidates = slowDiffusers.filter(d => !elements.includes(d));
    if (candidates.length > 0) {
      strategies.push({
        strategy: `Alloying with slow diffusers (${candidates.slice(0, 3).join(", ")})`,
        mechanism: "Substitute low-barrier elements with high-melting-point, heavy refractory elements to increase diffusion barriers",
        expectedLifetimeImprovement: "100x–10000x by raising rate-controlling diffusion barrier",
        difficulty: "moderate",
        applicability: Math.min(1.0, (1.0 - diffusion.effectiveBarrier) * 2),
      });
    }
  }

  if (eAboveHull > 0.01 && pressure.estimatedBulkModulus > 50) {
    strategies.push({
      strategy: "Epitaxial strain stabilization",
      mechanism: "Grow thin film on lattice-matched substrate to impose biaxial strain that raises decomposition barrier",
      expectedLifetimeImprovement: "10x–1000x depending on strain coherence",
      difficulty: eAboveHull < 0.1 ? "moderate" : "hard",
      applicability: Math.min(1.0, pressure.estimatedBulkModulus / 200),
    });
  }

  if (eAboveHull > 0.05) {
    strategies.push({
      strategy: "Encapsulation / passivation",
      mechanism: "Coat material with inert barrier (BN, Al2O3, diamond-like carbon) to prevent atmospheric decomposition and diffusion",
      expectedLifetimeImprovement: "100x–10^6x by blocking external nucleation and oxidation",
      difficulty: "moderate",
      applicability: Math.min(1.0, eAboveHull * 5),
    });
  }

  if (hasH && hFraction > 0.3 && pressure.minStabilizationPressure > 5) {
    strategies.push({
      strategy: "Chemical pre-compression (cage structure)",
      mechanism: "Use heavy-atom sublattice to provide internal chemical pressure that stabilizes hydrogen cages at lower external pressure",
      expectedLifetimeImprovement: "Potentially ambient-stable if chemical pressure exceeds ~50% of required external pressure",
      difficulty: "hard",
      applicability: Math.min(1.0, hFraction * 1.5),
    });
  }

  if (nucleation.nucleationBarrier < 0.5 && elements.length >= 3) {
    strategies.push({
      strategy: "Kinetic quenching (rapid solidification)",
      mechanism: "Use melt spinning, splat cooling, or pulsed laser deposition to freeze metastable phase before nucleation of competing phases",
      expectedLifetimeImprovement: "Indefinite if cooling rate exceeds critical nucleation rate",
      difficulty: "moderate",
      applicability: Math.min(1.0, (0.5 - nucleation.nucleationBarrier) * 4),
    });
  }

  if (eAboveHull < 0.15 && elements.length <= 4) {
    strategies.push({
      strategy: "Point defect engineering",
      mechanism: "Introduce controlled vacancies or anti-site defects to pin phase boundaries and suppress long-range ordering of decomposition products",
      expectedLifetimeImprovement: "5x–50x through defect-mediated kinetic arrest",
      difficulty: "hard",
      applicability: Math.min(1.0, (0.15 - eAboveHull) * 10),
    });
  }

  strategies.sort((a, b) => b.applicability - a.applicability);
  return strategies.slice(0, 5);
}

function lifetimeToString(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds > 1e30) {
    return "effectively infinite (>10^10 years)";
  }
  if (seconds > 3.15e16) {
    return `${(seconds / 3.15e7).toExponential(1)} years`;
  }
  if (seconds > 3.15e7) {
    return `${Math.round(seconds / 3.15e7)} years`;
  }
  if (seconds > 86400) {
    return `${Math.round(seconds / 86400)} days`;
  }
  if (seconds > 3600) {
    return `${Math.round(seconds / 3600)} hours`;
  }
  if (seconds > 60) {
    return `${Math.round(seconds / 60)} minutes`;
  }
  return `${seconds.toFixed(1)} seconds`;
}

export function predictKineticStability(formula: string, eAboveHull: number): KineticStabilityResult {
  const gb = computeGrainBoundaryEnergy(formula);
  const diffusion = computeDiffusionBarriers(formula);
  const nucleation = computeNucleationBarrier(formula, eAboveHull);
  const pressure = computePressureStabilization(formula, eAboveHull);

  const metastability = assessMetastability(formula, eAboveHull);

  const effectiveBarrier = Math.min(
    diffusion.effectiveBarrier,
    nucleation.nucleationBarrier,
  );

  const gbPenalty = gb.gbDecompositionFactor * 0.15;
  const correctedBarrier = Math.max(0.01, effectiveBarrier - gbPenalty);

  const attemptFreq = 1e13;
  const rate300K = attemptFreq * Math.exp(-correctedBarrier / (kB * 300));
  const lifetime300K = 1 / Math.max(rate300K, 1e-100);

  const uncertaintyFactor = 3.0;
  const confidenceLow = lifetime300K / uncertaintyFactor;
  const confidenceHigh = lifetime300K * uncertaintyFactor;

  const strategies = identifyStabilizationStrategies(
    formula, eAboveHull, gb, diffusion, nucleation, pressure,
  );

  let kineticScore = 0;
  if (lifetime300K > 1e15) kineticScore = 1.0;
  else if (lifetime300K > 3.15e7) kineticScore = 0.85;
  else if (lifetime300K > 86400) kineticScore = 0.6;
  else if (lifetime300K > 3600) kineticScore = 0.4;
  else if (lifetime300K > 60) kineticScore = 0.2;
  else kineticScore = 0.05;

  if (pressure.ambientStabilizable) kineticScore = Math.min(1.0, kineticScore + 0.1);
  if (strategies.length > 3) kineticScore = Math.min(1.0, kineticScore + 0.05);

  kineticScore = Math.round(kineticScore * 1000) / 1000;

  let overallVerdict: string;
  if (eAboveHull <= 0.005) {
    overallVerdict = "Thermodynamically stable — kinetic stability not limiting";
  } else if (kineticScore >= 0.8) {
    overallVerdict = `Kinetically trapped — estimated lifetime ${lifetimeToString(lifetime300K)} at 300K`;
  } else if (kineticScore >= 0.5) {
    overallVerdict = `Moderately metastable — lifetime ${lifetimeToString(lifetime300K)}, stabilization strategies available`;
  } else if (kineticScore >= 0.2) {
    overallVerdict = `Short-lived metastable — lifetime ${lifetimeToString(lifetime300K)}, requires active stabilization`;
  } else {
    overallVerdict = `Kinetically unstable — rapid decomposition expected (${lifetimeToString(lifetime300K)})`;
  }

  return {
    formula,
    kineticScore,
    metastableLifetime300K: lifetime300K,
    lifetimeString: lifetimeToString(lifetime300K),
    confidenceLow,
    confidenceHigh,
    grainBoundary: gb,
    diffusionBarriers: diffusion,
    nucleationBarrier: nucleation,
    pressureStabilization: pressure,
    stabilizationStrategies: strategies,
    overallVerdict,
  };
}

export function formatKineticStabilityNote(result: KineticStabilityResult): string {
  const stratNames = result.stabilizationStrategies.slice(0, 3).map(s => s.strategy.split("(")[0].trim());
  const stratStr = stratNames.length > 0 ? `, strategies=[${stratNames.join("; ")}]` : "";
  return `[KineticStability: score=${result.kineticScore}, lifetime=${result.lifetimeString}, gbEnergy=${result.grainBoundary.averageGBEnergy}, diffBarrier=${result.diffusionBarriers.effectiveBarrier}, nucleationBarrier=${result.nucleationBarrier.nucleationBarrier}, pressure=${result.pressureStabilization.minStabilizationPressure}GPa${stratStr}]`;
}
