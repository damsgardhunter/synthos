import { SynthesisVector } from "../physics/synthesis-simulator";

const kB = 8.617e-5; // Boltzmann constant in eV/K

export interface GrainStructure {
  grainSize: number;
  grainOrientation: "random" | "textured" | "epitaxial" | "single-crystal";
  boundaryDensity: number;
  phaseHomogeneity: number;
}

export interface CrystalGrowthResult {
  formula: string;
  materialClass: string;
  nucleationProbability: number;
  growthRate: number;
  grainStructure: GrainStructure;
  qualityScore: number;
  criticalCurrentImpact: { jcEstimate: number; grainBoundaryLimiting: boolean; recommendation: string };
  notes: string[];
}

const growthStats = {
  totalSimulations: 0,
  totalGrainSizeSum: 0,
  qualityDistribution: { poor: 0, fair: 0, good: 0, excellent: 0 } as Record<string, number>,
  bestQualityScore: 0,
  bestFormula: "",
};

export function computeNucleationProbability(deltaG: number, temperature: number): number {
  if (temperature <= 0) return 0;
  const exponent = -deltaG / (kB * temperature);
  if (exponent > 20) return 1.0;
  if (exponent < -50) return 0;
  return Math.exp(exponent);
}

function getActivationEnergy(materialClass: string): number {
  const mc = materialClass.toLowerCase();
  if (mc.includes("hydride")) return 0.1;
  if (mc.includes("cuprate")) return 0.6;
  if (mc.includes("pnictide") || mc.includes("iron")) return 0.4;
  if (mc.includes("boride")) return 1.5;
  if (mc.includes("carbide")) return 1.4;
  if (mc.includes("nitride")) return 1.2;
  if (mc.includes("oxide")) return 0.7;
  return 0.5;
}

export function estimateGrowthRate(diffusionRate: number, supersaturation: number, temperature: number, materialClass: string = ""): number {
  if (temperature <= 0 || diffusionRate <= 0) return 0;
  const Ea = getActivationEnergy(materialClass);
  const thermalFactor = Math.exp(-Ea / (kB * temperature));
  const rate = diffusionRate * supersaturation * thermalFactor * 1e3;
  return Math.max(0.001, Math.min(1e4, rate));
}

export function predictGrainStructure(
  formula: string,
  synthesisTemp: number,
  coolingRate: number,
  pressure: number,
  synthesisMethod?: string,
  materialClass?: string
): GrainStructure {
  let grainSize = 100;

  if (coolingRate < 1) {
    grainSize = 5000 + (1 - coolingRate) * 5000;
  } else if (coolingRate < 10) {
    grainSize = 500 + (10 - coolingRate) * 500;
  } else if (coolingRate < 100) {
    grainSize = 50 + (100 - coolingRate) * 5;
  } else if (coolingRate < 1000) {
    grainSize = 5 + (1000 - coolingRate) * 0.05;
  } else {
    grainSize = Math.max(1, 5 - (coolingRate - 1000) * 0.001);
  }

  if (synthesisTemp > 1200) {
    grainSize *= 1 + (synthesisTemp - 1200) / 2000;
  }

  if (pressure > 10) {
    grainSize *= Math.max(0.3, 1 - pressure / 500);
  }

  const method = synthesisMethod ?? "";
  const SINGLE_CRYSTAL_METHODS = new Set(["bridgman", "czochralski", "flux-growth"]);
  const TEXTURED_METHODS = new Set(["melt-textured", "vapor-deposition", "sputtering"]);

  let orientation: GrainStructure["grainOrientation"] = "random";
  if (SINGLE_CRYSTAL_METHODS.has(method)) {
    orientation = "single-crystal";
    grainSize = Math.max(grainSize, 10000);
  } else if (TEXTURED_METHODS.has(method)) {
    orientation = coolingRate < 10 ? "epitaxial" : "textured";
    grainSize *= 1.5;
  } else if (method === "arc-melting") {
    orientation = coolingRate < 50 ? "textured" : "random";
  } else {
    if (coolingRate < 1 && synthesisTemp > 800) {
      orientation = "epitaxial";
    } else if (coolingRate < 10 && synthesisTemp > 600) {
      orientation = "textured";
    }
  }

  const boundaryDensity = grainSize > 0 ? 1e7 / grainSize : 1e14;

  const mc = (materialClass ?? "").toLowerCase();
  const isHydride = mc.includes("hydride") || /H\d/.test(formula) || formula.includes("H3") || formula.includes("H10");

  let phaseHomogeneity = 0.85;
  if (coolingRate < 50) {
    phaseHomogeneity += 0.1 * (1 - coolingRate / 50);
  }
  if (synthesisTemp > 1000 && synthesisTemp < 1800) {
    phaseHomogeneity += 0.03;
  }
  if (pressure > 5 && pressure < 100) {
    phaseHomogeneity += 0.02;
  }

  if (isHydride) {
    const hMatch = formula.match(/H(\d+)/);
    const hCount = hMatch ? parseInt(hMatch[1]) : 1;
    const minPressureForStability = hCount > 6 ? 100 : hCount > 3 ? 50 : 10;
    if (pressure < minPressureForStability) {
      const pressureDeficit = (minPressureForStability - pressure) / minPressureForStability;
      phaseHomogeneity *= Math.max(0.4, 1 - pressureDeficit * 0.5);
    }
    if (pressure < 5 && hCount > 3) {
      phaseHomogeneity *= 0.5;
    }
  }

  phaseHomogeneity = Math.min(0.99, phaseHomogeneity);

  grainSize = Math.max(0.5, grainSize);

  return {
    grainSize,
    grainOrientation: orientation,
    boundaryDensity,
    phaseHomogeneity,
  };
}

export function assessCriticalCurrentImpact(
  grainSize: number,
  boundaryDensity: number,
  materialClass: string = ""
): { jcEstimate: number; grainBoundaryLimiting: boolean; recommendation: string } {
  let jcEstimate = 1e6;
  const mc = materialClass.toLowerCase();
  const isWeakLinkLimited = mc.includes("cuprate") || mc.includes("oxide");
  const isPinningDominated = mc.includes("pnictide") || mc.includes("iron") ||
    mc.includes("boride") || mc.includes("borocarbide") ||
    mc.includes("intermetallic") || mc.includes("a15") || mc.includes("nb3sn");

  if (isWeakLinkLimited) {
    if (grainSize > 1000) {
      jcEstimate = 1e7;
    } else if (grainSize > 100) {
      jcEstimate = 1e6 * (grainSize / 100);
    } else if (grainSize > 10) {
      jcEstimate = 1e5 * (grainSize / 10);
    } else {
      jcEstimate = 1e4 * grainSize;
    }

    if (boundaryDensity > 1e12) {
      jcEstimate *= Math.max(0.01, 1 - Math.log10(boundaryDensity / 1e12) * 0.2);
    }
  } else if (isPinningDominated) {
    const optimalGrainSize = 50;
    const sizeRatio = grainSize / optimalGrainSize;

    if (sizeRatio < 0.2) {
      jcEstimate = 1e5 * (sizeRatio / 0.2);
    } else if (sizeRatio <= 2.0) {
      jcEstimate = 1e7 * Math.exp(-0.5 * Math.pow(Math.log(sizeRatio), 2));
    } else {
      jcEstimate = 1e6 / sizeRatio;
    }

    if (boundaryDensity > 1e10 && boundaryDensity < 1e13) {
      const pinningBoost = 1 + Math.log10(boundaryDensity / 1e10) * 0.3;
      jcEstimate *= pinningBoost;
    } else if (boundaryDensity >= 1e13) {
      jcEstimate *= Math.max(0.5, 1 - Math.log10(boundaryDensity / 1e13) * 0.15);
    }
  } else {
    const midGrain = 200;
    if (grainSize > midGrain) {
      jcEstimate = 5e6 * Math.sqrt(midGrain / grainSize);
    } else if (grainSize > 10) {
      jcEstimate = 5e6 * Math.sqrt(grainSize / midGrain);
    } else {
      jcEstimate = 1e5 * (grainSize / 10);
    }

    if (boundaryDensity > 1e12) {
      jcEstimate *= Math.max(0.1, 1 - Math.log10(boundaryDensity / 1e12) * 0.1);
    }
  }

  const grainBoundaryLimiting = isWeakLinkLimited
    ? boundaryDensity > 1e10
    : boundaryDensity > 1e13;

  let recommendation: string;
  if (jcEstimate > 1e6) {
    recommendation = isWeakLinkLimited
      ? "Excellent Jc; large grain / single-crystal structure minimizes weak-link losses"
      : isPinningDominated
        ? "Excellent Jc; optimal grain boundary pinning density"
        : "Excellent Jc expected; suitable for high-field applications";
  } else if (jcEstimate > 1e5) {
    recommendation = isWeakLinkLimited
      ? "Good Jc; consider melt-texturing or seeded growth for larger grains"
      : isPinningDominated
        ? "Good Jc; fine-tune grain size toward 30-80 nm for optimal pinning"
        : "Good Jc; adequate for most applications";
  } else if (jcEstimate > 1e4) {
    recommendation = isWeakLinkLimited
      ? "Moderate Jc; grain boundaries are limiting — texture or epitaxial growth recommended"
      : isPinningDominated
        ? "Moderate Jc; grain size may be too large — increase cooling rate or add artificial pinning centers"
        : "Moderate Jc; grain boundary engineering recommended";
  } else {
    recommendation = isWeakLinkLimited
      ? "Low Jc; grain boundaries severely limiting — melt-texturing essential"
      : isPinningDominated
        ? "Low Jc; grain structure suboptimal — consider mechanical alloying or nano-powder sintering"
        : "Low Jc; significant optimization needed";
  }

  return { jcEstimate, grainBoundaryLimiting, recommendation };
}

function estimateDiffusionRate(formula: string, materialClass: string): number {
  const mc = materialClass.toLowerCase();
  if (mc.includes("hydride")) return 1e-4;
  if (mc.includes("cuprate")) return 5e-6;
  if (mc.includes("pnictide") || mc.includes("iron")) return 2e-5;
  if (mc.includes("boride")) return 1e-5;
  if (mc.includes("carbide")) return 8e-6;
  if (mc.includes("nitride")) return 1e-5;
  return 3e-5;
}

function estimateSupersaturation(synthesisTemp: number, pressure: number): number {
  let ss = 0.1;
  if (synthesisTemp > 1000) ss += (synthesisTemp - 1000) / 5000;
  if (pressure > 1) ss += Math.log10(pressure) * 0.05;
  return Math.min(1.0, ss);
}

function estimateDeltaG(formula: string, materialClass: string, temperature: number): number {
  const mc = materialClass.toLowerCase();
  let baseEnergy = 0.3;
  if (mc.includes("hydride")) baseEnergy = 0.5;
  else if (mc.includes("cuprate")) baseEnergy = 0.25;
  else if (mc.includes("pnictide")) baseEnergy = 0.35;
  else if (mc.includes("boride")) baseEnergy = 0.4;

  const thermalReduction = Math.min(0.2, temperature * 1e-4);
  return Math.max(0.05, baseEnergy - thermalReduction);
}

export function simulateCrystalGrowth(
  formula: string,
  materialClass: string,
  synthesisVector: Partial<SynthesisVector>,
  targetApplication: "research" | "wire" | "thin-film" | "bulk" = "research"
): CrystalGrowthResult {
  growthStats.totalSimulations++;

  const sv: SynthesisVector = {
    temperature: synthesisVector.temperature ?? 1000,
    pressure: synthesisVector.pressure ?? 1,
    coolingRate: synthesisVector.coolingRate ?? 100,
    annealTime: synthesisVector.annealTime ?? 8,
    currentDensity: synthesisVector.currentDensity ?? 0,
    magneticField: synthesisVector.magneticField ?? 0,
    thermalCycles: synthesisVector.thermalCycles ?? 1,
    strain: synthesisVector.strain ?? 0,
    oxygenPressure: synthesisVector.oxygenPressure ?? 0,
    synthesisMethod: synthesisVector.synthesisMethod,
  };

  const notes: string[] = [];

  const deltaG = estimateDeltaG(formula, materialClass, sv.temperature);
  const nucleationProbability = computeNucleationProbability(deltaG, sv.temperature);

  const diffusionRate = estimateDiffusionRate(formula, materialClass);
  const supersaturation = estimateSupersaturation(sv.temperature, sv.pressure);
  const growthRate = estimateGrowthRate(diffusionRate, supersaturation, sv.temperature, materialClass);

  const grainStructure = predictGrainStructure(formula, sv.temperature, sv.coolingRate, sv.pressure, sv.synthesisMethod, materialClass);

  if (sv.annealTime > 4) {
    const annealBenefit = Math.min(2.0, sv.annealTime / 12);
    grainStructure.grainSize *= (1 + annealBenefit * 0.3);
    grainStructure.phaseHomogeneity = Math.min(0.99, grainStructure.phaseHomogeneity + annealBenefit * 0.02);
    grainStructure.boundaryDensity *= Math.max(0.3, 1 - annealBenefit * 0.2);
    notes.push(`Annealing for ${sv.annealTime.toFixed(1)}h improved grain structure`);
  }

  if (sv.thermalCycles > 1) {
    grainStructure.phaseHomogeneity = Math.min(0.99, grainStructure.phaseHomogeneity + sv.thermalCycles * 0.005);
    notes.push(`${sv.thermalCycles} thermal cycles improved phase homogeneity`);
  }

  if (Math.abs(sv.strain) > 2) {
    grainStructure.boundaryDensity *= (1 + Math.abs(sv.strain) * 0.1);
    notes.push(`Applied strain of ${sv.strain.toFixed(1)}% increased boundary density`);
  }

  if (growthRate > 500) {
    grainStructure.phaseHomogeneity *= 0.9;
    notes.push("Very fast growth rate (>500) reduces phase homogeneity due to anti-site defects");
  } else if (growthRate > 200) {
    grainStructure.phaseHomogeneity *= 0.95;
    notes.push("Fast growth rate may introduce point defects and stacking faults");
  } else if (growthRate > 100) {
    notes.push("Fast growth rate may introduce minor defects");
  }

  const criticalCurrentImpact = assessCriticalCurrentImpact(grainStructure.grainSize, grainStructure.boundaryDensity, materialClass);

  const orientationScore = grainStructure.grainOrientation === "single-crystal" ? 1 :
    grainStructure.grainOrientation === "epitaxial" ? 0.8 :
    grainStructure.grainOrientation === "textured" ? 0.5 : 0.2;

  const grainSizeScore = Math.min(1, grainStructure.grainSize / 1000);
  const invertedGrainSizeScore = Math.min(1, 100 / Math.max(1, grainStructure.grainSize));

  let qualityScore = 0;
  if (targetApplication === "wire") {
    qualityScore += grainStructure.phaseHomogeneity * 0.25;
    qualityScore += invertedGrainSizeScore * 0.30;
    qualityScore += (grainStructure.grainOrientation === "textured" ? 1 :
      grainStructure.grainOrientation === "epitaxial" ? 0.8 :
      grainStructure.grainOrientation === "random" ? 0.4 :
      grainStructure.grainOrientation === "single-crystal" ? 0.3 : 0.2) * 0.25;
    qualityScore += nucleationProbability * 0.20;
  } else if (targetApplication === "thin-film") {
    qualityScore += grainStructure.phaseHomogeneity * 0.35;
    qualityScore += (grainStructure.grainOrientation === "epitaxial" ? 1 :
      grainStructure.grainOrientation === "single-crystal" ? 0.9 :
      grainStructure.grainOrientation === "textured" ? 0.6 : 0.2) * 0.35;
    qualityScore += grainSizeScore * 0.15;
    qualityScore += nucleationProbability * 0.15;
  } else if (targetApplication === "bulk") {
    qualityScore += grainStructure.phaseHomogeneity * 0.35;
    qualityScore += grainSizeScore * 0.20;
    qualityScore += orientationScore * 0.20;
    qualityScore += nucleationProbability * 0.15;
    qualityScore += Math.min(1, criticalCurrentImpact.jcEstimate / 1e7) * 0.10;
  } else {
    qualityScore += grainStructure.phaseHomogeneity * 0.30;
    qualityScore += grainSizeScore * 0.25;
    qualityScore += orientationScore * 0.25;
    qualityScore += nucleationProbability * 0.20;
  }
  qualityScore = Math.min(1, qualityScore);

  if (qualityScore >= 0.8) growthStats.qualityDistribution.excellent++;
  else if (qualityScore >= 0.6) growthStats.qualityDistribution.good++;
  else if (qualityScore >= 0.4) growthStats.qualityDistribution.fair++;
  else growthStats.qualityDistribution.poor++;

  growthStats.totalGrainSizeSum += grainStructure.grainSize;

  if (qualityScore > growthStats.bestQualityScore) {
    growthStats.bestQualityScore = qualityScore;
    growthStats.bestFormula = formula;
  }

  if (nucleationProbability > 0.9) notes.push("High nucleation probability indicates easy crystal formation");
  if (growthRate < 1) notes.push("Slow growth rate favors high crystal quality");

  return {
    formula,
    materialClass,
    nucleationProbability,
    growthRate,
    grainStructure,
    qualityScore,
    criticalCurrentImpact,
    notes,
  };
}

export function getCrystalGrowthStats() {
  const totalSims = Math.max(1, growthStats.totalSimulations);
  return {
    totalSimulations: growthStats.totalSimulations,
    avgGrainSize: growthStats.totalGrainSizeSum / totalSims,
    qualityDistribution: { ...growthStats.qualityDistribution },
    bestQualityScore: growthStats.bestQualityScore,
    bestFormula: growthStats.bestFormula,
  };
}
