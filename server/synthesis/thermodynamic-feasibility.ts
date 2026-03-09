import { ELEMENTAL_DATA, getMeltingPoint, getElementData } from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";

export interface ReactionFeasibilityResult {
  deltaE: number;
  gibbsFreeEnergy: number;
  gibbsFreeEnergyAtSynthesisTemp: number;
  kineticBarrier: number;
  arrheniusRate: number;
  metastableQuenchFeasibility: number;
  overallFeasibility: number;
  synthesisTemperature: number;
  pressureRequirement: number;
  thermodynamicDriving: "favorable" | "marginal" | "unfavorable";
  kineticAccessibility: "easy" | "moderate" | "difficult" | "very-difficult";
  notes: string[];
}

export interface SynthesisTemperatureEstimate {
  temperature: number;
  method: string;
  confidence: number;
  basis: string;
}

export interface PressureAssessment {
  pressureGpa: number;
  isAmbient: boolean;
  isModerate: boolean;
  isHighPressure: boolean;
  isUltraHigh: boolean;
  method: string;
  equipment: string;
  notes: string;
}

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

function parseFormulaElements(formula: string): string[] {
  return Object.keys(parseFormulaCounts(formula));
}

function computeMiedemaFormationEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return 0;

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }

  let deltaH = 0;
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const dA = ELEMENTAL_DATA[elements[i]];
      const dB = ELEMENTAL_DATA[elements[j]];
      if (!dA || !dB) continue;

      const phiA = dA.miedemaPhiStar;
      const phiB = dB.miedemaPhiStar;
      const nwsA = dA.miedemaNws13;
      const nwsB = dB.miedemaNws13;
      const vA = dA.miedemaV23;
      const vB = dB.miedemaV23;

      if (phiA == null || phiB == null || nwsA == null || nwsB == null || vA == null || vB == null) continue;

      const deltaPhi = phiA - phiB;
      const deltaNws = nwsA - nwsB;
      const nwsAvg = (nwsA + nwsB) / 2;
      const fAB = 2 * fractions[elements[i]] * fractions[elements[j]];
      const vAvg = (vA * fractions[elements[i]] + vB * fractions[elements[j]]) / (fractions[elements[i]] + fractions[elements[j]]);
      const interfaceEnergy = -14.1 * deltaPhi * deltaPhi + 9.4 * deltaNws * deltaNws;
      deltaH += fAB * vAvg * interfaceEnergy / (nwsAvg * nwsAvg);
    }
  }

  return deltaH / totalAtoms;
}

function computeConfigurationalEntropy(elements: string[], counts: Record<string, number>): number {
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  if (totalAtoms <= 0) return 0;

  let entropy = 0;
  for (const el of elements) {
    const frac = counts[el] / totalAtoms;
    if (frac > 0 && frac < 1) {
      entropy -= frac * Math.log(frac);
    }
  }
  return 8.314e-3 * entropy;
}

function computeGibbsFreeEnergy(deltaH: number, temperature: number, entropy: number): number {
  return deltaH - temperature * entropy * 1e-3;
}

function computeTammannTemperature(elements: string[]): number {
  let maxMp = 0;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp !== null && mp > maxMp) maxMp = mp;
  }
  return maxMp > 0 ? 0.57 * maxMp : 1200;
}

function computeKineticBarrier(elements: string[], temperature: number): number {
  const tammTemp = computeTammannTemperature(elements);
  const barrier = Math.max(0.1, (tammTemp - temperature) / tammTemp);
  return Math.max(0, Math.min(3.0, barrier * 2.0));
}

function computeArrheniusRate(barrierEv: number, temperature: number): number {
  const kB = 8.617e-5;
  if (temperature <= 0) return 0;
  const exponent = -barrierEv / (kB * temperature);
  return Math.exp(Math.max(-50, Math.min(50, exponent)));
}

function computeMetastableQuenchFeasibility(formula: string, pressureGpa: number): number {
  const elements = parseFormulaElements(formula);
  const hasH = elements.includes("H");
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hFrac = (counts["H"] || 0) / totalAtoms;

  let feasibility = 0.7;

  if (hasH && hFrac > 0.5) {
    feasibility -= 0.3;
    if (pressureGpa > 100) feasibility += 0.15;
  }

  if (pressureGpa > 50) {
    feasibility -= 0.1;
    if (pressureGpa > 200) feasibility -= 0.2;
  }

  const dH = computeMiedemaFormationEnergy(formula);
  if (dH > 0.5) feasibility -= 0.2;
  if (dH < -0.5) feasibility += 0.1;

  const nElements = elements.length;
  if (nElements >= 4) feasibility -= 0.05;
  if (nElements >= 5) feasibility -= 0.1;

  return Math.max(0, Math.min(1, feasibility));
}

export function computeSynthesisTemperature(formula: string, family?: string): SynthesisTemperatureEstimate {
  const elements = parseFormulaElements(formula);
  const resolvedFamily = family || classifyFamily(formula);
  const counts = parseFormulaCounts(formula);

  let maxMp = 0;
  let avgMp = 0;
  let mpCount = 0;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp !== null) {
      if (mp > maxMp) maxMp = mp;
      avgMp += mp;
      mpCount++;
    }
  }
  avgMp = mpCount > 0 ? avgMp / mpCount : 1200;

  const tammTemp = maxMp > 0 ? 0.57 * maxMp : 1200;

  const hasH = elements.includes("H");
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hFrac = (counts["H"] || 0) / totalAtoms;

  let temperature: number;
  let method: string;
  let basis: string;
  let confidence: number;

  switch (resolvedFamily) {
    case "cuprate":
      temperature = Math.round(Math.max(850, Math.min(1100, tammTemp * 0.8)));
      method = "solid-state";
      basis = "Cuprate synthesis typically requires 850-1100 C for phase formation";
      confidence = 0.8;
      break;

    case "iron-based":
      temperature = Math.round(Math.max(800, Math.min(1200, tammTemp * 0.75)));
      method = "solid-state or flux";
      basis = "Iron-based superconductors synthesized at 800-1200 C";
      confidence = 0.75;
      break;

    case "hydride":
      if (hFrac > 0.6) {
        temperature = Math.round(Math.min(2500, 300 + 5 * 100));
        method = "high-pressure DAC with laser heating";
        basis = "Hydrogen-rich phases require high-pressure laser heating";
        confidence = 0.5;
      } else {
        temperature = Math.round(Math.max(500, Math.min(1000, tammTemp * 0.6)));
        method = "high-pressure";
        basis = "Metal hydrides formed at moderate temperatures under H2 pressure";
        confidence = 0.6;
      }
      break;

    case "boride":
    case "borocarbide":
      temperature = Math.round(Math.max(1200, Math.min(2000, tammTemp * 0.85)));
      method = "arc-melting or solid-state";
      basis = "Borides require high temperatures due to refractory nature";
      confidence = 0.7;
      break;

    case "A15":
    case "intermetallic":
      temperature = Math.round(Math.max(900, Math.min(1800, tammTemp * 0.7)));
      method = "arc-melting with annealing";
      basis = "Intermetallics formed via arc-melting and long annealing";
      confidence = 0.75;
      break;

    case "heavy-fermion":
      temperature = Math.round(Math.max(800, Math.min(1500, tammTemp * 0.65)));
      method = "flux growth or arc-melting";
      basis = "Heavy fermion compounds require careful single-crystal growth";
      confidence = 0.65;
      break;

    case "chalcogenide":
      temperature = Math.round(Math.max(600, Math.min(1000, tammTemp * 0.6)));
      method = "sealed tube or CVD";
      basis = "Chalcogenides typically synthesized at 600-1000 C in sealed tubes";
      confidence = 0.75;
      break;

    case "oxide":
      temperature = Math.round(Math.max(800, Math.min(1400, tammTemp * 0.75)));
      method = "solid-state";
      basis = "Oxide synthesis via solid-state reaction";
      confidence = 0.7;
      break;

    case "nitride":
      temperature = Math.round(Math.max(800, Math.min(1600, tammTemp * 0.7)));
      method = "ammonolysis or reactive sputtering";
      basis = "Nitrides synthesized under nitrogen/ammonia atmosphere";
      confidence = 0.6;
      break;

    case "carbide":
      temperature = Math.round(Math.max(1200, Math.min(2200, tammTemp * 0.8)));
      method = "arc-melting or solid-state";
      basis = "Carbides require high synthesis temperatures";
      confidence = 0.7;
      break;

    default:
      temperature = Math.round(Math.max(700, Math.min(1500, tammTemp * 0.7)));
      method = "solid-state";
      basis = "Generic estimate based on Tammann temperature";
      confidence = 0.5;
      break;
  }

  return {
    temperature,
    method,
    confidence,
    basis,
  };
}

export function assessPressureRequirement(formula: string, family?: string): PressureAssessment {
  const elements = parseFormulaElements(formula);
  const resolvedFamily = family || classifyFamily(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hasH = elements.includes("H");
  const hFrac = (counts["H"] || 0) / totalAtoms;

  if (resolvedFamily === "hydride" && hFrac > 0.5) {
    const nHeavy = elements.filter(e => e !== "H").length;
    let pressureGpa: number;

    if (hFrac > 0.8) {
      pressureGpa = 200;
    } else if (hFrac > 0.6) {
      pressureGpa = 150;
    } else {
      pressureGpa = 50;
    }

    if (nHeavy === 1) {
      const heavyEl = elements.find(e => e !== "H")!;
      const data = getElementData(heavyEl);
      if (data && data.bulkModulus && data.bulkModulus > 150) {
        pressureGpa = Math.max(pressureGpa, 100);
      }
    }

    return {
      pressureGpa,
      isAmbient: false,
      isModerate: pressureGpa <= 20,
      isHighPressure: pressureGpa > 20 && pressureGpa <= 100,
      isUltraHigh: pressureGpa > 100,
      method: pressureGpa > 100 ? "diamond anvil cell" : "multi-anvil press",
      equipment: pressureGpa > 100 ? "DAC with laser heating" : "multi-anvil press",
      notes: `Hydrogen-rich compound requires ${pressureGpa} GPa for stabilization`,
    };
  }

  if (resolvedFamily === "hydride" && hasH) {
    return {
      pressureGpa: 5,
      isAmbient: false,
      isModerate: true,
      isHighPressure: false,
      isUltraHigh: false,
      method: "gas-loading or multi-anvil",
      equipment: "High-pressure H2 gas system or multi-anvil press",
      notes: "Metal hydride with moderate hydrogen content; moderate pressure needed",
    };
  }

  if (resolvedFamily === "cuprate" || resolvedFamily === "iron-based" || resolvedFamily === "oxide") {
    return {
      pressureGpa: 0,
      isAmbient: true,
      isModerate: false,
      isHighPressure: false,
      isUltraHigh: false,
      method: "ambient pressure",
      equipment: "Standard furnace",
      notes: "Ambient pressure synthesis sufficient",
    };
  }

  if (resolvedFamily === "A15" || resolvedFamily === "intermetallic" || resolvedFamily === "boride") {
    return {
      pressureGpa: 0,
      isAmbient: true,
      isModerate: false,
      isHighPressure: false,
      isUltraHigh: false,
      method: "ambient pressure",
      equipment: "Arc furnace or tube furnace",
      notes: "Can be synthesized at ambient pressure",
    };
  }

  const miedemaE = computeMiedemaFormationEnergy(formula);
  if (miedemaE > 0.3) {
    return {
      pressureGpa: 10,
      isAmbient: false,
      isModerate: true,
      isHighPressure: false,
      isUltraHigh: false,
      method: "moderate pressure may help stabilize",
      equipment: "Multi-anvil press or hot isostatic press",
      notes: "Positive formation energy suggests pressure may stabilize the phase",
    };
  }

  return {
    pressureGpa: 0,
    isAmbient: true,
    isModerate: false,
    isHighPressure: false,
    isUltraHigh: false,
    method: "ambient pressure",
    equipment: "Standard furnace",
    notes: "No special pressure requirements identified",
  };
}

export function computeReactionFeasibility(
  formula: string,
  formationEnergyPerAtom: number | null,
  precursors: string[]
): ReactionFeasibilityResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const family = classifyFamily(formula);
  const notes: string[] = [];

  let deltaE: number;
  if (formationEnergyPerAtom !== null && isFinite(formationEnergyPerAtom)) {
    deltaE = formationEnergyPerAtom;
    notes.push("Using DFT formation energy from database");
  } else {
    deltaE = computeMiedemaFormationEnergy(formula);
    notes.push("Using Miedema model estimate (no DFT data available)");
  }

  const tempEstimate = computeSynthesisTemperature(formula, family);
  const synthesisTemperature = tempEstimate.temperature;

  const pressureAssessment = assessPressureRequirement(formula, family);
  const pressureRequirement = pressureAssessment.pressureGpa;

  const entropy = computeConfigurationalEntropy(elements, counts);
  const gibbsRT = computeGibbsFreeEnergy(deltaE, 300, entropy);
  const gibbsSynthTemp = computeGibbsFreeEnergy(deltaE, synthesisTemperature, entropy);

  const kineticBarrier = computeKineticBarrier(elements, synthesisTemperature);
  const arrheniusRate = computeArrheniusRate(kineticBarrier, synthesisTemperature);
  const metastableQuench = computeMetastableQuenchFeasibility(formula, pressureRequirement);

  let thermoDriving: "favorable" | "marginal" | "unfavorable";
  if (gibbsSynthTemp < -0.1) {
    thermoDriving = "favorable";
    notes.push(`Gibbs free energy is negative (${gibbsSynthTemp.toFixed(3)} eV/atom) at synthesis temperature — thermodynamically favorable`);
  } else if (gibbsSynthTemp < 0.2) {
    thermoDriving = "marginal";
    notes.push(`Gibbs free energy is near zero (${gibbsSynthTemp.toFixed(3)} eV/atom) — marginal thermodynamic driving force`);
  } else {
    thermoDriving = "unfavorable";
    notes.push(`Gibbs free energy is positive (${gibbsSynthTemp.toFixed(3)} eV/atom) — thermodynamically unfavorable, may need pressure or quenching`);
  }

  let kineticAccess: "easy" | "moderate" | "difficult" | "very-difficult";
  if (kineticBarrier < 0.5) {
    kineticAccess = "easy";
    notes.push("Low kinetic barrier — reaction should proceed readily at synthesis temperature");
  } else if (kineticBarrier < 1.0) {
    kineticAccess = "moderate";
    notes.push("Moderate kinetic barrier — extended reaction time or higher temperature may be needed");
  } else if (kineticBarrier < 2.0) {
    kineticAccess = "difficult";
    notes.push("High kinetic barrier — requires elevated temperatures or mechanical activation");
  } else {
    kineticAccess = "very-difficult";
    notes.push("Very high kinetic barrier — synthesis will be challenging");
  }

  if (pressureRequirement > 100) {
    notes.push(`Ultra-high pressure (${pressureRequirement} GPa) required — DAC synthesis only`);
  } else if (pressureRequirement > 20) {
    notes.push(`High pressure (${pressureRequirement} GPa) required — multi-anvil or DAC`);
  } else if (pressureRequirement > 0) {
    notes.push(`Moderate pressure (${pressureRequirement} GPa) beneficial`);
  }

  if (precursors.length > 0) {
    notes.push(`Precursors: ${precursors.join(", ")}`);
  }

  const gibbsFactor = gibbsSynthTemp < 0 ? 1.0 : Math.max(0, 1.0 - gibbsSynthTemp * 0.5);
  const barrierFactor = Math.max(0, 1.0 - kineticBarrier * 0.3);
  const rateFactor = Math.min(1.0, arrheniusRate * 10);

  let pressurePenalty = 1.0;
  if (pressureRequirement > 200) pressurePenalty = 0.3;
  else if (pressureRequirement > 100) pressurePenalty = 0.5;
  else if (pressureRequirement > 50) pressurePenalty = 0.7;
  else if (pressureRequirement > 10) pressurePenalty = 0.85;

  const overallFeasibility = Math.max(0, Math.min(1,
    (gibbsFactor * 0.30 + barrierFactor * 0.20 + rateFactor * 0.15 + metastableQuench * 0.15) * pressurePenalty + 0.2 * pressurePenalty
  ));

  return {
    deltaE: Number(deltaE.toFixed(4)),
    gibbsFreeEnergy: Number(gibbsRT.toFixed(4)),
    gibbsFreeEnergyAtSynthesisTemp: Number(gibbsSynthTemp.toFixed(4)),
    kineticBarrier: Number(kineticBarrier.toFixed(4)),
    arrheniusRate: Number(arrheniusRate.toFixed(6)),
    metastableQuenchFeasibility: Number(metastableQuench.toFixed(4)),
    overallFeasibility: Number(overallFeasibility.toFixed(4)),
    synthesisTemperature,
    pressureRequirement,
    thermodynamicDriving: thermoDriving,
    kineticAccessibility: kineticAccess,
    notes,
  };
}
