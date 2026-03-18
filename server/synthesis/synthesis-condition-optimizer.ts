import {
  SynthesisConditionSet,
  ALL_NUMERIC_VARIABLES,
  GAS_ENVIRONMENTS,
  THERMAL_CYCLING_RANGES,
  getApplicableVariables,
  getParameterSpace,
} from "./synthesis-variables";

export interface MaterialContext {
  formula: string;
  materialClass: string;
  predictedTc: number;
  lambda: number;
  pressure: number;
  isHydride: boolean;
  isCuprate: boolean;
  isLayered: boolean;
  meltingPointEstimate: number;
  stabilityClass: string;
  energyAboveHull: number;
}

export interface OptimizationResult {
  formula: string;
  conditions: SynthesisConditionSet;
  alternativeConditions: SynthesisConditionSet[];
  keyParameters: { parameter: string; value: number | string; reason: string }[];
  overallFeasibility: number;
  synthesisComplexity: "straightforward" | "moderate" | "challenging" | "extreme";
  estimatedDuration: string;
  criticalSteps: string[];
}

interface OptimizerStats {
  totalOptimized: number;
  avgFeasibility: number;
  complexityBreakdown: Record<string, number>;
  methodBreakdown: Record<string, number>;
  categoryUsage: Record<string, number>;
  topConditions: { formula: string; method: string; feasibility: number; tc: number }[];
  parameterRangesExplored: Record<string, { min: number; max: number; count: number }>;
}

const optimizationHistory: OptimizationResult[] = [];
const methodCounts: Record<string, number> = {};
const complexityCounts: Record<string, number> = {};
const categoryUsageCounts: Record<string, number> = {};
const paramRanges: Record<string, { min: number; max: number; count: number }> = {};
let totalFeasibility = 0;

function classifyMaterialForSynthesis(ctx: MaterialContext): string {
  const f = ctx.formula.toLowerCase();
  const mc = ctx.materialClass.toLowerCase();

  if (ctx.isHydride || mc.includes("hydride") || mc.includes("superhydride")) {
    return ctx.pressure > 50 ? "high-pressure-hydride" : "low-pressure-hydride";
  }
  if (ctx.isCuprate || mc.includes("cuprate")) return "cuprate";
  if (mc.includes("pnictide") || mc.includes("iron")) return "pnictide";
  if (mc.includes("heavy-fermion") || mc.includes("heavy")) return "heavy-fermion";
  if (mc.includes("intermetallic") || mc.includes("alloy")) return "intermetallic";
  if (mc.includes("carbide")) return "carbide";
  if (mc.includes("nitride")) return "nitride";
  if (mc.includes("boride")) return "boride";
  if (mc.includes("oxide")) return "oxide";
  if (mc.includes("chalcogenide") || mc.includes("sulfide")) return "chalcogenide";
  if (ctx.isLayered) return "layered";
  return "conventional";
}

function selectOptimalTemperature(synthClass: string, meltingPoint: number, stability: string): number {
  if (!meltingPoint || meltingPoint <= 0) meltingPoint = 1500; // conservative fallback for unknown elements

  const sinterFraction = stability === "thermodynamically-stable" ? 0.65
    : stability === "metastable-accessible" ? 0.55
    : 0.45;

  let baseTemp = meltingPoint * sinterFraction;

  switch (synthClass) {
    case "high-pressure-hydride": return Math.min(baseTemp, 500);
    case "cuprate": return Math.min(Math.max(baseTemp, 850), 1100);
    case "pnictide": return Math.min(Math.max(baseTemp, 700), 1000);
    case "carbide":
    case "boride": return Math.min(Math.max(baseTemp, 1200), 2000);
    case "nitride": return Math.min(Math.max(baseTemp, 800), 1500);
    case "intermetallic": return Math.min(Math.max(baseTemp, 900), 1800);
    default: return Math.min(Math.max(baseTemp, 600), 1500);
  }
}

function selectOptimalPressure(synthClass: string, ctx: MaterialContext): number {
  switch (synthClass) {
    case "high-pressure-hydride": return Math.max(ctx.pressure, 100);
    case "low-pressure-hydride": return Math.max(ctx.pressure, 5);
    case "cuprate": return 0;
    case "carbide":
    case "boride":
    case "nitride": return ctx.energyAboveHull > 0.1 ? 5 : 0;
    default: return ctx.energyAboveHull > 0.2 ? 2 : 0;
  }
}

function selectCoolingRate(synthClass: string, stability: string): number {
  if (stability === "likely-unstable") return 5000;
  if (stability === "metastable-difficult") return 1000;
  if (stability === "metastable-accessible") return 100;

  switch (synthClass) {
    case "high-pressure-hydride": return 50;
    case "cuprate": return 1;
    case "pnictide": return 10;
    case "intermetallic": return 100;
    case "carbide":
    case "boride": return 5;
    default: return 10;
  }
}

function selectGasEnvironment(synthClass: string): string {
  switch (synthClass) {
    case "cuprate": return "oxygen";
    case "high-pressure-hydride":
    case "low-pressure-hydride": return "hydrogen";
    case "nitride": return "nitrogen";
    case "oxide": return "oxygen";
    default: return "argon";
  }
}

function selectMethod(synthClass: string, stability: string): string {
  if (stability === "likely-unstable" || stability === "metastable-difficult") {
    return synthClass === "high-pressure-hydride" ? "Diamond Anvil Cell"
      : "Magnetron Sputtering (Thin Film)";
  }

  switch (synthClass) {
    case "high-pressure-hydride": return "Diamond Anvil Cell";
    case "low-pressure-hydride": return "High-Pressure Gas Loading";
    case "cuprate": return "Sol-Gel with Controlled Calcination";
    case "pnictide": return "Solid-State Reaction";
    case "intermetallic": return "Arc Melting";
    case "carbide":
    case "boride": return "Hot Isostatic Pressing";
    case "nitride": return "Reactive Sputtering";
    case "heavy-fermion": return "Flux Growth";
    case "layered": return "Chemical Vapor Deposition";
    default: return "Solid-State Reaction";
  }
}

function selectAnnealParams(synthClass: string, synthTemp: number): { temp: number; time: number } {
  switch (synthClass) {
    case "cuprate": return { temp: Math.min(synthTemp * 0.7, 900), time: 24 };
    case "pnictide": return { temp: Math.min(synthTemp * 0.65, 800), time: 10 };
    case "carbide":
    case "boride": return { temp: Math.min(synthTemp * 0.6, 1200), time: 5 };
    case "high-pressure-hydride": return { temp: 300, time: 1 };
    default: return { temp: Math.min(synthTemp * 0.6, 700), time: 5 };
  }
}

function selectInterfaceParams(synthClass: string, isLayered: boolean): { thickness: number; spacing: number } {
  if (!isLayered && synthClass !== "cuprate" && synthClass !== "pnictide") {
    return { thickness: 0, spacing: 0 };
  }
  switch (synthClass) {
    case "cuprate": return { thickness: 2, spacing: 6 };
    case "pnictide": return { thickness: 5, spacing: 5 };
    default: return { thickness: 10, spacing: 4 };
  }
}

function computeFeasibility(conditions: SynthesisConditionSet, ctx: MaterialContext, synthClass: string): number {
  let score = 0.5;

  if (ctx.stabilityClass === "thermodynamically-stable") score += 0.25;
  else if (ctx.stabilityClass === "metastable-accessible") score += 0.10;
  else if (ctx.stabilityClass === "metastable-difficult") score -= 0.05;
  else score -= 0.15;

  if (conditions.synthesisPressure > 100) score -= 0.15;
  else if (conditions.synthesisPressure > 20) score -= 0.08;

  if (conditions.coolingRate > 5000) score -= 0.10;
  if (conditions.synthesisTemperature > 2000) score -= 0.08;

  if (ctx.lambda > 1.0) score += 0.05;
  if (ctx.predictedTc > 50) score += 0.05;

  if (synthClass === "high-pressure-hydride" && conditions.synthesisPressure < 50) score -= 0.2;
  if (synthClass === "cuprate" && conditions.gasEnvironment !== "oxygen") score -= 0.1;

  return Math.max(0.05, Math.min(0.98, score));
}

function determineComplexity(conditions: SynthesisConditionSet): "straightforward" | "moderate" | "challenging" | "extreme" {
  let score = 0;
  if (conditions.synthesisPressure > 100) score += 3;
  else if (conditions.synthesisPressure > 10) score += 2;
  else if (conditions.synthesisPressure > 1) score += 1;

  if (conditions.synthesisTemperature > 2000) score += 2;
  else if (conditions.synthesisTemperature > 1500) score += 1;

  if (conditions.coolingRate > 5000) score += 2;
  else if (conditions.coolingRate > 100) score += 1;

  if (conditions.magneticField > 5) score += 1;
  if (conditions.currentDensity > 50) score += 1;
  if (conditions.thermalCycles > 10) score += 1;
  if (conditions.layerThickness > 0 && conditions.layerThickness < 5) score += 2;

  if (score <= 2) return "straightforward";
  if (score <= 5) return "moderate";
  if (score <= 8) return "challenging";
  return "extreme";
}

function estimateDuration(conditions: SynthesisConditionSet, synthClass: string): string {
  let hours = 0;
  hours += conditions.synthesisTemperature > 1500 ? 8 : 4;
  hours += conditions.annealTime;
  hours += conditions.thermalCycles * 2;
  if (conditions.synthesisPressure > 50) hours += 12;
  if (synthClass === "cuprate") hours += 24;
  if (synthClass === "high-pressure-hydride") hours += 48;

  if (hours < 8) return `${Math.round(hours)} hours`;
  if (hours < 48) return `${Math.round(hours / 24 * 10) / 10} days`;
  return `${Math.round(hours / 24)} days`;
}

function generateCriticalSteps(conditions: SynthesisConditionSet, synthClass: string): string[] {
  const steps: string[] = [];

  if (conditions.synthesisPressure > 50) {
    steps.push(`Apply ${conditions.synthesisPressure} GPa using diamond anvil cell with laser heating`);
  } else if (conditions.synthesisPressure > 1) {
    steps.push(`Compress to ${conditions.synthesisPressure} GPa using multi-anvil press`);
  }

  steps.push(`Heat to ${Math.round(conditions.synthesisTemperature)}K in ${conditions.gasEnvironment} atmosphere`);

  if (conditions.annealTime > 0) {
    steps.push(`Anneal at ${Math.round(conditions.annealTemperature)}K for ${conditions.annealTime} hours`);
  }

  if (conditions.coolingRate > 1000) {
    steps.push(`Rapid quench at ${conditions.coolingRate} K/s to trap metastable phase`);
  } else if (conditions.coolingRate < 1) {
    steps.push(`Slow cool at ${conditions.coolingRate} K/s for equilibrium crystal growth`);
  } else {
    steps.push(`Cool at ${conditions.coolingRate} K/s`);
  }

  if (conditions.magneticField > 0) {
    steps.push(`Apply ${conditions.magneticField}T magnetic field during growth for grain alignment`);
  }

  if (conditions.thermalCycles > 0) {
    steps.push(`Perform ${conditions.thermalCycles} thermal cycles (${conditions.cycleTemperatureRange})`);
  }

  if (conditions.dopantConcentration > 0) {
    steps.push(`Introduce dopant at x=${conditions.dopantConcentration} concentration`);
  }

  if (conditions.layerThickness > 0) {
    steps.push(`Deposit ${conditions.layerThickness}nm layers with ${conditions.interlayerSpacing}A interlayer spacing`);
  }

  return steps;
}

function trackParameterUsage(conditions: SynthesisConditionSet) {
  const params: Record<string, number> = {
    synthesisTemperature: conditions.synthesisTemperature,
    annealTemperature: conditions.annealTemperature,
    coolingRate: conditions.coolingRate,
    synthesisPressure: conditions.synthesisPressure,
    magneticField: conditions.magneticField,
    grainSize: conditions.grainSize,
    dopantConcentration: conditions.dopantConcentration,
  };

  for (const [key, val] of Object.entries(params)) {
    if (!paramRanges[key]) {
      paramRanges[key] = { min: val, max: val, count: 1 };
    } else {
      paramRanges[key].min = Math.min(paramRanges[key].min, val);
      paramRanges[key].max = Math.max(paramRanges[key].max, val);
      paramRanges[key].count++;
    }
  }
}

export function optimizeSynthesisConditions(ctx: MaterialContext): OptimizationResult {
  const synthClass = classifyMaterialForSynthesis(ctx);
  const synthTemp = selectOptimalTemperature(synthClass, ctx.meltingPointEstimate, ctx.stabilityClass);
  const pressure = selectOptimalPressure(synthClass, ctx);
  const coolingRate = selectCoolingRate(synthClass, ctx.stabilityClass);
  const gas = selectGasEnvironment(synthClass);
  const method = selectMethod(synthClass, ctx.stabilityClass);
  const anneal = selectAnnealParams(synthClass, synthTemp);
  const iface = selectInterfaceParams(synthClass, ctx.isLayered);

  const optimalDoping = synthClass === "cuprate" ? 0.1
    : synthClass === "pnictide" ? 0.05
    : 0.02;

  const thermalCycles = synthClass === "cuprate" ? 5
    : ctx.stabilityClass === "metastable-accessible" ? 2
    : 0;

  const grainSize = synthClass === "high-pressure-hydride" ? 50
    : synthClass === "cuprate" ? 500
    : 100;

  const conditions: SynthesisConditionSet = {
    synthesisTemperature: Math.round(synthTemp),
    annealTemperature: Math.round(anneal.temp),
    annealTime: anneal.time,
    coolingRate,
    synthesisPressure: pressure,
    pressureRampRate: pressure > 50 ? 1 : pressure > 1 ? 2 : 0.1,
    currentDensity: synthClass === "intermetallic" ? 50 : 0,
    electricField: synthClass === "cuprate" ? 100 : 0,
    magneticField: synthClass === "cuprate" || synthClass === "pnictide" ? 2 : 0,
    mechanicalStress: ctx.isLayered ? 0.5 : 0,
    latticeStrain: ctx.isLayered ? 1 : 0,
    thermalCycles,
    cycleTemperatureRange: thermalCycles > 0 ? "300-800 K" : "N/A",
    oxygenPartialPressure: synthClass === "cuprate" || synthClass === "oxide" ? 1 : 0,
    hydrogenPressure: ctx.isHydride ? Math.max(pressure, 1) : 0,
    gasEnvironment: gas,
    grainSize,
    defectDensity: 1e16,
    dopantConcentration: optimalDoping,
    vacancyFraction: synthClass === "cuprate" ? 0.02 : 0.01,
    layerThickness: iface.thickness,
    interlayerSpacing: iface.spacing,
    feasibilityScore: 0,
    method,
    notes: "",
  };

  conditions.feasibilityScore = computeFeasibility(conditions, ctx, synthClass);
  const complexity = determineComplexity(conditions);
  const duration = estimateDuration(conditions, synthClass);
  const criticalSteps = generateCriticalSteps(conditions, synthClass);

  const keyParameters: { parameter: string; value: number | string; reason: string }[] = [];
  keyParameters.push({ parameter: "synthesisTemperature", value: conditions.synthesisTemperature, reason: `Optimal sintering at ${Math.round(conditions.synthesisTemperature / ctx.meltingPointEstimate * 100)}% of melting point` });
  if (conditions.synthesisPressure > 0) {
    keyParameters.push({ parameter: "synthesisPressure", value: conditions.synthesisPressure, reason: synthClass.includes("hydride") ? "Required for hydrogen cage stabilization" : "Stabilizes metastable phase" });
  }
  keyParameters.push({ parameter: "coolingRate", value: conditions.coolingRate, reason: coolingRate > 100 ? "Rapid quench to preserve metastable structure" : "Controlled cooling for ordered phase" });
  if (conditions.dopantConcentration > 0) {
    keyParameters.push({ parameter: "dopantConcentration", value: conditions.dopantConcentration, reason: "Optimal carrier density for max Tc" });
  }

  const alt1 = { ...conditions };
  alt1.coolingRate = coolingRate * 10;
  alt1.feasibilityScore = computeFeasibility(alt1, ctx, synthClass) * 0.9;
  alt1.notes = "Faster quench variant";

  const alt2 = { ...conditions };
  alt2.annealTime = anneal.time * 2;
  alt2.annealTemperature = Math.round(anneal.temp * 0.9);
  alt2.feasibilityScore = computeFeasibility(alt2, ctx, synthClass) * 0.95;
  alt2.notes = "Extended anneal variant";

  const result: OptimizationResult = {
    formula: ctx.formula,
    conditions,
    alternativeConditions: [alt1, alt2],
    keyParameters,
    overallFeasibility: conditions.feasibilityScore,
    synthesisComplexity: complexity,
    estimatedDuration: duration,
    criticalSteps,
  };

  optimizationHistory.push(result);
  if (optimizationHistory.length > 200) optimizationHistory.shift();

  methodCounts[method] = (methodCounts[method] || 0) + 1;
  complexityCounts[complexity] = (complexityCounts[complexity] || 0) + 1;
  totalFeasibility += conditions.feasibilityScore;

  const applicable = getApplicableVariables(ctx.materialClass);
  for (const v of applicable) {
    categoryUsageCounts[v.category] = (categoryUsageCounts[v.category] || 0) + 1;
  }

  trackParameterUsage(conditions);

  return result;
}

export function getSynthesisOptimizerStats(): OptimizerStats {
  const count = optimizationHistory.length;
  const topConditions = optimizationHistory
    .sort((a, b) => b.overallFeasibility - a.overallFeasibility)
    .slice(0, 10)
    .map(r => ({
      formula: r.formula,
      method: r.conditions.method,
      feasibility: r.overallFeasibility,
      tc: 0,
    }));

  return {
    totalOptimized: count,
    avgFeasibility: count > 0 ? totalFeasibility / count : 0,
    complexityBreakdown: { ...complexityCounts },
    methodBreakdown: { ...methodCounts },
    categoryUsage: { ...categoryUsageCounts },
    topConditions,
    parameterRangesExplored: { ...paramRanges },
  };
}

export function getOptimizationHistory(): OptimizationResult[] {
  return optimizationHistory.slice(-50);
}
