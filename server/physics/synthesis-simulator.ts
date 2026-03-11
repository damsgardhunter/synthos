export interface SynthesisVector {
  temperature: number;
  pressure: number;
  coolingRate: number;
  annealTime: number;
  currentDensity: number;
  magneticField: number;
  thermalCycles: number;
  strain: number;
  oxygenPressure: number;
}

export interface MaterialVector {
  elements: string[];
  stoichiometry: Record<string, number>;
  structure: string;
}

export interface CombinedSearchVector {
  material: MaterialVector;
  synthesis: SynthesisVector;
}

export interface SynthesisEffects {
  lambdaModifier: number;
  omegaLogModifier: number;
  defectDensity: number;
  latticeStrain: number;
  grainSize: number;
  metastableAllowed: boolean;
  bondCompression: number;
  bandShift: number;
  phasePurity: number;
  effectiveTcMultiplier: number;
}

export interface SynthesisPath {
  steps: SynthesisStep[];
  totalDuration: number;
  overallComplexity: number;
  feasibilityScore: number;
}

export interface SynthesisStep {
  order: number;
  method: string;
  temperature: number;
  pressure: number;
  coolingRate: number;
  annealTemp: number;
  duration: number;
  atmosphere: string;
  notes: string;
}

export interface FeasibilityResult {
  labFeasible: boolean;
  industrialFeasible: boolean;
  feasibilityScore: number;
  constraintViolations: string[];
  classification: "practical" | "experimental" | "unrealistic";
}

export const SYNTHESIS_CONSTRAINTS = {
  temperature: { min: 200, max: 2500 },
  pressure: { min: 0, max: 300 },
  coolingRate: { min: 0.1, max: 10000 },
  annealTime: { min: 0.1, max: 168 },
  currentDensity: { min: 0, max: 200 },
  magneticField: { min: 0, max: 45 },
  thermalCycles: { min: 0, max: 100 },
  strain: { min: -5, max: 10 },
  oxygenPressure: { min: 0, max: 200 },
} as const;

const METASTABLE_QUENCH_THRESHOLD = 1000;

const SYNTHESIS_DEFAULTS: Record<string, Partial<SynthesisVector>> = {
  hydride: { temperature: 1000, pressure: 100, coolingRate: 50, annealTime: 8, strain: 0.3 },
  cuprate: { temperature: 950, pressure: 0.1, coolingRate: 2, annealTime: 24, oxygenPressure: 0.21 },
  "iron-pnictide": { temperature: 1100, pressure: 2, coolingRate: 10, annealTime: 16 },
  intermetallic: { temperature: 1200, pressure: 0.1, coolingRate: 5, annealTime: 12 },
  boride: { temperature: 1400, pressure: 5, coolingRate: 20, annealTime: 10 },
  carbide: { temperature: 1500, pressure: 3, coolingRate: 15, annealTime: 12 },
  nitride: { temperature: 1300, pressure: 1, coolingRate: 10, annealTime: 12 },
  default: { temperature: 1000, pressure: 1, coolingRate: 10, annealTime: 12 },
};

const simulatorStats = {
  totalSimulations: 0,
  totalMutations: 0,
  totalPathsOptimized: 0,
  avgTcImprovement: 0,
  totalTcImprovement: 0,
  bestTcImprovement: 0,
  bestFormula: "",
  constraintViolations: 0,
  feasibilityBreakdown: { practical: 0, experimental: 0, unrealistic: 0 } as Record<string, number>,
  modeBreakdown: { materialDiscovery: 0, synthesisOptimization: 0 } as Record<string, number>,
};

export function defaultSynthesisVector(materialClass: string = "default"): SynthesisVector {
  const base: SynthesisVector = {
    temperature: 1000,
    pressure: 1,
    coolingRate: 100,
    annealTime: 8,
    currentDensity: 0,
    magneticField: 0,
    thermalCycles: 1,
    strain: 0,
    oxygenPressure: 0,
  };
  const classKey = Object.keys(SYNTHESIS_DEFAULTS).find(k =>
    materialClass.toLowerCase().includes(k)
  ) || "default";
  return { ...base, ...SYNTHESIS_DEFAULTS[classKey] };
}

export function randomSynthesisVector(materialClass: string = "default"): SynthesisVector {
  const C = SYNTHESIS_CONSTRAINTS;
  const isHydride = materialClass.toLowerCase().includes("hydride");
  return {
    temperature: randRange(C.temperature.min, C.temperature.max),
    pressure: randRange(C.pressure.min, C.pressure.max * 0.5),
    coolingRate: randRange(C.coolingRate.min, C.coolingRate.max * 0.3),
    annealTime: randRange(C.annealTime.min, C.annealTime.max * 0.5),
    currentDensity: randRange(0, C.currentDensity.max * 0.3),
    magneticField: randRange(0, C.magneticField.max * 0.2),
    thermalCycles: Math.floor(randRange(0, 10)),
    strain: randRange(C.strain.min * 0.3, C.strain.max * 0.5),
    oxygenPressure: isHydride ? randRange(0, 0.1) : randRange(0, C.oxygenPressure.max),
  };
}

export function clampSynthesisVector(v: SynthesisVector): SynthesisVector {
  const C = SYNTHESIS_CONSTRAINTS;
  return {
    temperature: clamp(v.temperature, C.temperature.min, C.temperature.max),
    pressure: clamp(v.pressure, C.pressure.min, C.pressure.max),
    coolingRate: clamp(v.coolingRate, C.coolingRate.min, C.coolingRate.max),
    annealTime: clamp(v.annealTime, C.annealTime.min, C.annealTime.max),
    currentDensity: clamp(v.currentDensity, C.currentDensity.min, C.currentDensity.max),
    magneticField: clamp(v.magneticField, C.magneticField.min, C.magneticField.max),
    thermalCycles: clamp(Math.round(v.thermalCycles), C.thermalCycles.min, C.thermalCycles.max),
    strain: clamp(v.strain, C.strain.min, C.strain.max),
    oxygenPressure: clamp(v.oxygenPressure, C.oxygenPressure.min, C.oxygenPressure.max),
  };
}

export function simulateSynthesisEffects(
  formula: string,
  materialClass: string,
  sv: SynthesisVector
): SynthesisEffects {
  simulatorStats.totalSimulations++;

  let lambdaMod = 1.0;
  let omegaMod = 1.0;
  let defectDensity = 1e14;
  let latticeStrain = 0;
  let grainSize = 50;
  let metastable = false;
  let bondComp = 0;
  let bandShift = 0;
  let phasePurity = 0.95;

  if (sv.coolingRate > 500) {
    metastable = true;
    defectDensity *= (1 + sv.coolingRate / 2000);
    grainSize = Math.max(0.1, grainSize * Math.exp(-sv.coolingRate / 3000));
    lambdaMod *= 1.0 + Math.min(0.15, Math.sqrt(sv.coolingRate / 20000));
  } else if (sv.coolingRate < 10) {
    phasePurity = Math.min(0.99, phasePurity + 0.03);
    grainSize *= (1 + (10 - sv.coolingRate) / 5);
    defectDensity *= 0.5;
  }

  if (sv.pressure > 50) {
    bondComp = Math.min(0.2, (sv.pressure - 50) / 500);
    lambdaMod *= 1.0 + Math.sqrt(bondComp * 0.5);
    omegaMod *= 1.0 + bondComp * 0.8;
    if (sv.pressure > 100) {
      lambdaMod *= 1.0 + Math.min(0.3, Math.sqrt((sv.pressure - 100) / 300));
    }
  }

  if (Math.abs(sv.strain) > 2) {
    bandShift = sv.strain * 0.05;
    const strainMag = Math.abs(sv.strain);
    if (strainMag < 5) {
      lambdaMod *= 1.0 + Math.sqrt(strainMag) * 0.02;
    } else {
      lambdaMod *= 1.0 - Math.sqrt(strainMag - 5) * 0.03;
      phasePurity *= Math.max(0.7, 1 - (strainMag - 5) * 0.05);
    }
    latticeStrain = sv.strain;
  }

  if (sv.temperature > 1500) {
    defectDensity *= Math.exp(Math.sqrt((sv.temperature - 1500) / 500));
    if (sv.temperature > 2000) {
      phasePurity *= Math.max(0.6, 1 - Math.sqrt((sv.temperature - 2000) / 2000));
    }
  }

  if (sv.annealTime > 0.5) {
    const annealFactor = Math.min(1, sv.annealTime / 24);
    defectDensity *= Math.max(0.1, 1 - annealFactor * 0.5);
    grainSize *= (1 + annealFactor * 0.5);
    phasePurity = Math.min(0.99, phasePurity + annealFactor * 0.03);
  }

  if (sv.thermalCycles > 1) {
    const cycleBenefit = Math.min(0.1, sv.thermalCycles * 0.005);
    lambdaMod *= 1.0 + cycleBenefit;
    defectDensity *= Math.max(0.5, 1 - sv.thermalCycles * 0.01);
  }

  if (sv.magneticField > 0) {
    const magFactor = Math.min(0.05, sv.magneticField / 200);
    if (materialClass.toLowerCase().includes("pnictide")) {
      lambdaMod *= 1.0 + magFactor;
    }
  }

  const mcLower = materialClass.toLowerCase();
  if (sv.oxygenPressure > 0.1 && mcLower.includes("hydride")) {
    const o2Penalty = Math.min(sv.oxygenPressure, 10) * 0.2;
    phasePurity *= Math.max(0.3, 1 - o2Penalty);
    lambdaMod *= 0.85;
  }

  if (sv.oxygenPressure > 0 && mcLower.includes("cuprate")) {
    const optimalO2 = sv.oxygenPressure > 10 ? 50 : 0.21;
    const o2Diff = Math.abs(sv.oxygenPressure - optimalO2) / Math.max(optimalO2, 1);
    phasePurity *= Math.max(0.7, 1 - o2Diff * 0.3);
    if (sv.oxygenPressure >= 10 && sv.oxygenPressure <= 150) {
      lambdaMod *= 1.0 + Math.min(0.12, Math.log(sv.oxygenPressure / 10) * 0.03);
      phasePurity = Math.min(0.99, phasePurity + Math.min(0.04, (sv.oxygenPressure - 10) / 1000));
    } else if (sv.oxygenPressure > 0.1) {
      lambdaMod *= 1.0 + 0.05;
    }
  }

  if (metastable && sv.coolingRate < METASTABLE_QUENCH_THRESHOLD) {
    const quenchDeficit = (METASTABLE_QUENCH_THRESHOLD - sv.coolingRate) / METASTABLE_QUENCH_THRESHOLD;
    phasePurity *= Math.max(0.3, 1 - quenchDeficit * 0.6);
    lambdaMod *= Math.max(0.5, 1 - quenchDeficit * 0.3);
  }

  if (sv.currentDensity > 0) {
    const currentEffect = Math.min(0.03, sv.currentDensity / 5000);
    defectDensity *= (1 + sv.currentDensity / 500);
    lambdaMod *= 1.0 + currentEffect;
  }

  let purityPenalty: number;
  if (phasePurity >= 0.8) {
    purityPenalty = phasePurity * phasePurity;
  } else {
    const impurityFraction = 1 - phasePurity;
    purityPenalty = Math.max(0.05, Math.exp(-4.0 * impurityFraction));
  }
  const effectiveMult = lambdaMod * purityPenalty * Math.min(1, omegaMod);

  return {
    lambdaModifier: lambdaMod,
    omegaLogModifier: omegaMod,
    defectDensity,
    latticeStrain,
    grainSize,
    metastableAllowed: metastable,
    bondCompression: bondComp,
    bandShift,
    phasePurity,
    effectiveTcMultiplier: effectiveMult,
  };
}

export function checkSynthesisFeasibility(sv: SynthesisVector, materialClass: string = ""): FeasibilityResult {
  const violations: string[] = [];
  let score = 0;
  const C = SYNTHESIS_CONSTRAINTS;

  if (sv.temperature > C.temperature.max) violations.push(`Temperature ${sv.temperature}K exceeds ${C.temperature.max}K`);
  if (sv.pressure > C.pressure.max) violations.push(`Pressure ${sv.pressure} GPa exceeds ${C.pressure.max} GPa`);
  if (sv.coolingRate > C.coolingRate.max) violations.push(`Cooling rate ${sv.coolingRate} K/s exceeds ${C.coolingRate.max} K/s`);
  if (sv.currentDensity > C.currentDensity.max) violations.push(`Current density ${sv.currentDensity} A/cm2 exceeds ${C.currentDensity.max} A/cm2`);
  if (sv.oxygenPressure > C.oxygenPressure.max) violations.push(`Oxygen pressure ${sv.oxygenPressure} bar exceeds ${C.oxygenPressure.max} bar`);

  const isMetastableCandidate = sv.pressure > 50 || sv.coolingRate > 500 ||
    materialClass.toLowerCase().includes("hydride");
  if (isMetastableCandidate && sv.coolingRate < METASTABLE_QUENCH_THRESHOLD) {
    violations.push(`Metastable phase requires quench rate >= ${METASTABLE_QUENCH_THRESHOLD} K/s, got ${sv.coolingRate.toFixed(1)} K/s — material will relax to non-superconducting phase`);
  }

  score += sv.pressure / 100;
  score += sv.coolingRate / 1000;
  score += sv.thermalCycles / 50;
  score += sv.temperature > 1500 ? (sv.temperature - 1500) / 500 : 0;
  score += sv.magneticField > 10 ? (sv.magneticField - 10) / 20 : 0;
  score += sv.currentDensity > 50 ? (sv.currentDensity - 50) / 100 : 0;

  if (sv.temperature > 1000 && sv.pressure > 50) {
    const tNorm = (sv.temperature - 1000) / 1500;
    const pNorm = (sv.pressure - 50) / 250;
    score += Math.exp(tNorm * pNorm * 3) - 1;
  }

  let labFeasible = violations.length === 0 && score < 8;
  let industrialFeasible = violations.length === 0 && score < 3;

  if (sv.pressure > 100) { industrialFeasible = false; }
  if (sv.pressure > 200) { labFeasible = false; }

  let classification: "practical" | "experimental" | "unrealistic";
  if (score < 2) classification = "practical";
  else if (score < 5) classification = "experimental";
  else classification = "unrealistic";

  simulatorStats.feasibilityBreakdown[classification] = (simulatorStats.feasibilityBreakdown[classification] || 0) + 1;

  return { labFeasible, industrialFeasible, feasibilityScore: score, constraintViolations: violations, classification };
}

export function computeSynthesisComplexity(sv: SynthesisVector): number {
  return sv.pressure / 100 + sv.coolingRate / 1000 + sv.thermalCycles / 50 +
    (sv.temperature > 1500 ? 0.5 : 0) + (sv.magneticField > 5 ? 0.3 : 0) +
    (sv.currentDensity > 50 ? 0.2 : 0) + (Math.abs(sv.strain) > 3 ? 0.4 : 0);
}

export function computeSynthesisCost(
  sv: SynthesisVector,
  predictedTc: number,
  stability: number = 0.5
): number {
  const w1 = 1.0;
  const w2 = 15.0;
  const w3 = 0.3;
  const w4 = 20.0;

  const complexity = computeSynthesisComplexity(sv);
  const pressurePenalty = sv.pressure > 50 ? sv.pressure / 100 : 0;
  const instability = Math.max(0, 1 - stability);

  return w1 * predictedTc - w2 * complexity - w3 * pressurePenalty - w4 * instability;
}

export function mutateSynthesisVector(sv: SynthesisVector): SynthesisVector {
  simulatorStats.totalMutations++;
  const mutated = { ...sv };
  const fields: (keyof SynthesisVector)[] = [
    "temperature", "pressure", "coolingRate", "annealTime",
    "currentDensity", "magneticField", "thermalCycles", "strain", "oxygenPressure",
  ];
  const numMutations = 1 + Math.floor(Math.random() * 3);

  for (let i = 0; i < numMutations; i++) {
    const field = fields[Math.floor(Math.random() * fields.length)];
    switch (field) {
      case "temperature":
        mutated.temperature += randRange(-200, 200);
        break;
      case "pressure":
        mutated.pressure *= randRange(0.7, 1.4);
        break;
      case "coolingRate":
        mutated.coolingRate *= randRange(0.5, 2.0);
        break;
      case "annealTime":
        mutated.annealTime += randRange(-4, 4);
        break;
      case "currentDensity":
        mutated.currentDensity += randRange(-20, 20);
        break;
      case "magneticField":
        mutated.magneticField += randRange(-2, 2);
        break;
      case "thermalCycles":
        mutated.thermalCycles += Math.floor(randRange(-3, 3));
        break;
      case "strain":
        mutated.strain += randRange(-1, 1);
        break;
      case "oxygenPressure":
        mutated.oxygenPressure += randRange(-0.05, 0.05);
        break;
    }
  }

  return clampSynthesisVector(mutated);
}

export function crossoverSynthesisVectors(a: SynthesisVector, b: SynthesisVector): SynthesisVector {
  const blend = (x: number, y: number) => {
    const alpha = Math.random();
    return alpha * x + (1 - alpha) * y;
  };

  return clampSynthesisVector({
    temperature: blend(a.temperature, b.temperature),
    pressure: blend(a.pressure, b.pressure),
    coolingRate: blend(a.coolingRate, b.coolingRate),
    annealTime: blend(a.annealTime, b.annealTime),
    currentDensity: blend(a.currentDensity, b.currentDensity),
    magneticField: blend(a.magneticField, b.magneticField),
    thermalCycles: Math.round(blend(a.thermalCycles, b.thermalCycles)),
    strain: blend(a.strain, b.strain),
    oxygenPressure: blend(a.oxygenPressure, b.oxygenPressure),
  });
}

const STEP_METHODS = [
  "heat-treatment", "high-pressure", "quench", "anneal",
  "ball-milling", "cvd", "sputtering", "arc-melting",
];

export function optimizeSynthesisPath(
  formula: string,
  materialClass: string,
  targetTc: number = 100
): SynthesisPath {
  simulatorStats.totalPathsOptimized++;
  const mc = materialClass.toLowerCase();

  const steps: SynthesisStep[] = [];

  if (mc.includes("hydride")) {
    steps.push({
      order: 1, method: "ball-milling",
      temperature: 300, pressure: 0, coolingRate: 0, annealTemp: 0,
      duration: 2, atmosphere: "argon",
      notes: "Precursor mixing and mechanical alloying",
    });
    steps.push({
      order: 2, method: "high-pressure",
      temperature: 1000 + Math.min(500, targetTc * 2), pressure: Math.min(200, 50 + targetTc / 3),
      coolingRate: 0, annealTemp: 0,
      duration: 4, atmosphere: "hydrogen",
      notes: "Diamond anvil cell or large volume press synthesis",
    });
    steps.push({
      order: 3, method: "quench",
      temperature: 0, pressure: 0, coolingRate: 1000 + Math.min(5000, targetTc * 10),
      annealTemp: 0, duration: 0.01, atmosphere: "argon",
      notes: "Rapid quench to preserve high-pressure phase",
    });
  } else if (mc.includes("cuprate")) {
    steps.push({
      order: 1, method: "heat-treatment",
      temperature: 850, pressure: 0, coolingRate: 0, annealTemp: 0,
      duration: 12, atmosphere: "oxygen",
      notes: "Calcination of oxide precursors",
    });
    steps.push({
      order: 2, method: "heat-treatment",
      temperature: 950, pressure: 0, coolingRate: 0, annealTemp: 0,
      duration: 24, atmosphere: "oxygen",
      notes: "Sintering at reaction temperature",
    });
    steps.push({
      order: 3, method: "anneal",
      temperature: 0, pressure: 0, coolingRate: 1,
      annealTemp: 500, duration: 48, atmosphere: "oxygen",
      notes: "Slow oxygen anneal for optimal doping",
    });
  } else if (mc.includes("pnictide") || mc.includes("iron")) {
    steps.push({
      order: 1, method: "arc-melting",
      temperature: 1500, pressure: 0, coolingRate: 0, annealTemp: 0,
      duration: 0.5, atmosphere: "argon",
      notes: "Arc-melt stoichiometric precursors",
    });
    steps.push({
      order: 2, method: "anneal",
      temperature: 0, pressure: 0, coolingRate: 0,
      annealTemp: 800, duration: 72, atmosphere: "vacuum",
      notes: "Homogenization anneal in sealed quartz tube",
    });
    steps.push({
      order: 3, method: "quench",
      temperature: 0, pressure: 0, coolingRate: 200,
      annealTemp: 0, duration: 0.01, atmosphere: "argon",
      notes: "Quench to room temperature",
    });
  } else {
    steps.push({
      order: 1, method: "heat-treatment",
      temperature: 1000 + Math.min(500, targetTc * 3),
      pressure: Math.min(10, targetTc / 20),
      coolingRate: 0, annealTemp: 0,
      duration: 8, atmosphere: "argon",
      notes: "Primary solid-state reaction",
    });
    steps.push({
      order: 2, method: "anneal",
      temperature: 0, pressure: 0, coolingRate: 0,
      annealTemp: 600 + Math.min(400, targetTc), duration: 24, atmosphere: "vacuum",
      notes: "Post-synthesis anneal for phase ordering",
    });
    steps.push({
      order: 3, method: "quench",
      temperature: 0, pressure: 0, coolingRate: 100,
      annealTemp: 0, duration: 0.01, atmosphere: "argon",
      notes: "Controlled cooling",
    });
  }

  const totalDuration = steps.reduce((s, st) => s + st.duration, 0);
  const maxP = Math.max(...steps.map(s => s.pressure));
  const maxCR = Math.max(...steps.map(s => s.coolingRate));
  const overallComplexity = maxP / 100 + maxCR / 1000 + steps.length / 5;

  const feasibility = checkSynthesisFeasibility({
    temperature: Math.max(...steps.map(s => Math.max(s.temperature, s.annealTemp))),
    pressure: maxP,
    coolingRate: maxCR,
    annealTime: steps.filter(s => s.method === "anneal").reduce((s, st) => s + st.duration, 0),
    currentDensity: 0,
    magneticField: 0,
    thermalCycles: 1,
    strain: 0,
    oxygenPressure: steps.some(s => s.atmosphere === "oxygen") ? 0.21 : 0,
  }, materialClass);

  return {
    steps,
    totalDuration,
    overallComplexity,
    feasibilityScore: feasibility.feasibilityScore,
  };
}

export function mutateSynthesisPath(path: SynthesisPath): SynthesisPath {
  const newSteps = path.steps.map(s => ({ ...s }));

  const mutType = Math.random();
  if (mutType < 0.4 && newSteps.length > 0) {
    const idx = Math.floor(Math.random() * newSteps.length);
    newSteps[idx].temperature += randRange(-100, 100);
    newSteps[idx].pressure *= randRange(0.8, 1.2);
    newSteps[idx].coolingRate *= randRange(0.7, 1.5);
    newSteps[idx].duration *= randRange(0.5, 2);
    newSteps[idx].temperature = Math.min(2500, Math.max(200, newSteps[idx].temperature));
    newSteps[idx].pressure = Math.min(300, Math.max(0, newSteps[idx].pressure));
    newSteps[idx].coolingRate = Math.min(10000, Math.max(0.1, newSteps[idx].coolingRate));
    newSteps[idx].duration = Math.min(168, Math.max(0.01, newSteps[idx].duration));
  } else if (mutType < 0.6 && newSteps.length > 2) {
    const i = Math.floor(Math.random() * (newSteps.length - 1));
    [newSteps[i], newSteps[i + 1]] = [newSteps[i + 1], newSteps[i]];
    newSteps.forEach((s, idx) => { s.order = idx + 1; });
  } else if (mutType < 0.8 && newSteps.length < 5) {
    const method = STEP_METHODS[Math.floor(Math.random() * STEP_METHODS.length)];
    newSteps.push({
      order: newSteps.length + 1,
      method,
      temperature: randRange(300, 1500),
      pressure: randRange(0, 20),
      coolingRate: randRange(1, 500),
      annealTemp: randRange(300, 800),
      duration: randRange(0.5, 12),
      atmosphere: Math.random() > 0.5 ? "argon" : "vacuum",
      notes: `Additional ${method} step`,
    });
  } else if (newSteps.length > 2) {
    const idx = Math.floor(Math.random() * newSteps.length);
    newSteps.splice(idx, 1);
    newSteps.forEach((s, i) => { s.order = i + 1; });
  }

  const totalDuration = newSteps.reduce((s, st) => s + st.duration, 0);
  const maxP = Math.max(0, ...newSteps.map(s => s.pressure));
  const maxCR = Math.max(0.1, ...newSteps.map(s => s.coolingRate));
  const overallComplexity = maxP / 100 + maxCR / 1000 + newSteps.length / 5;

  return {
    steps: newSteps,
    totalDuration,
    overallComplexity,
    feasibilityScore: path.feasibilityScore,
  };
}

export function computeSynthesisGradients(
  formula: string,
  materialClass: string,
  baseSv: SynthesisVector,
  evaluateTc: (sv: SynthesisVector) => number
): Record<keyof SynthesisVector, number> {
  const baseTc = evaluateTc(baseSv);
  const gradients: Record<string, number> = {};

  const perturbations: Record<keyof SynthesisVector, number> = {
    temperature: 50,
    pressure: 5,
    coolingRate: 50,
    annealTime: 2,
    currentDensity: 10,
    magneticField: 1,
    thermalCycles: 1,
    strain: 0.5,
    oxygenPressure: 0.05,
  };

  for (const [key, delta] of Object.entries(perturbations)) {
    const svPlus = clampSynthesisVector({ ...baseSv, [key]: (baseSv as any)[key] + delta });
    const svMinus = clampSynthesisVector({ ...baseSv, [key]: (baseSv as any)[key] - delta });
    const tcPlus = evaluateTc(svPlus);
    const tcMinus = evaluateTc(svMinus);
    gradients[key] = (tcPlus - tcMinus) / (2 * delta);
  }

  return gradients as Record<keyof SynthesisVector, number>;
}

export function optimizeSynthesisForFixedMaterial(
  formula: string,
  materialClass: string,
  evaluateTc: (sv: SynthesisVector) => number,
  maxIterations: number = 30,
  populationSize: number = 12
): { bestVector: SynthesisVector; bestTc: number; bestCost: number; iterations: number } {
  simulatorStats.modeBreakdown.synthesisOptimization = (simulatorStats.modeBreakdown.synthesisOptimization || 0) + 1;

  let population: { sv: SynthesisVector; tc: number; cost: number }[] = [];

  const defaultSv = defaultSynthesisVector(materialClass);
  for (let i = 0; i < populationSize; i++) {
    const sv = i === 0 ? defaultSv : i < 3 ? mutateSynthesisVector(defaultSv) : randomSynthesisVector(materialClass);
    const effects = simulateSynthesisEffects(formula, materialClass, sv);
    const tc = evaluateTc(sv);
    const cost = computeSynthesisCost(sv, tc);
    population.push({ sv, tc, cost });
  }

  let bestTc = 0;
  let bestCost = -Infinity;
  let bestSv = defaultSv;

  for (let iter = 0; iter < maxIterations; iter++) {
    population.sort((a, b) => b.cost - a.cost);

    if (population[0].cost > bestCost) {
      bestCost = population[0].cost;
      bestTc = population[0].tc;
      bestSv = { ...population[0].sv };
    }

    const elite = population.slice(0, Math.ceil(populationSize / 3));
    const newPop: typeof population = [...elite];

    while (newPop.length < populationSize) {
      const parent1 = elite[Math.floor(Math.random() * elite.length)];
      const parent2 = elite[Math.floor(Math.random() * elite.length)];
      let child: SynthesisVector;

      if (Math.random() < 0.3) {
        child = crossoverSynthesisVectors(parent1.sv, parent2.sv);
      } else {
        child = mutateSynthesisVector(parent1.sv);
      }

      const tc = evaluateTc(child);
      const cost = computeSynthesisCost(child, tc);
      newPop.push({ sv: child, tc, cost });
    }

    population = newPop;
  }

  const baseTc = evaluateTc(defaultSv);
  const improvement = bestTc - baseTc;
  if (improvement > 0) {
    simulatorStats.totalTcImprovement += improvement;
    if (improvement > simulatorStats.bestTcImprovement) {
      simulatorStats.bestTcImprovement = improvement;
      simulatorStats.bestFormula = formula;
    }
  }

  return { bestVector: bestSv, bestTc, bestCost, iterations: maxIterations };
}

export function getSimulatorStats() {
  const totalSims = Math.max(1, simulatorStats.totalSimulations);
  return {
    totalSimulations: simulatorStats.totalSimulations,
    totalMutations: simulatorStats.totalMutations,
    totalPathsOptimized: simulatorStats.totalPathsOptimized,
    avgTcImprovement: simulatorStats.totalPathsOptimized > 0
      ? simulatorStats.totalTcImprovement / simulatorStats.totalPathsOptimized
      : 0,
    bestTcImprovement: simulatorStats.bestTcImprovement,
    bestFormula: simulatorStats.bestFormula,
    constraintViolations: simulatorStats.constraintViolations,
    feasibilityBreakdown: { ...simulatorStats.feasibilityBreakdown },
    modeBreakdown: { ...simulatorStats.modeBreakdown },
  };
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function randRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}
