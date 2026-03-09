import { getElementData, isTransitionMetal, isRareEarth, isActinide } from "./elemental-data";
import { classifyFamily } from "./utils";
import { computeSurrogateFitness, getCurrentFitnessWeights, getExplorationWeight } from "./surrogate-fitness";
import type { TopologicalAnalysis } from "../physics/topology-engine";
import type { FermiSurfaceResult } from "../physics/fermi-surface-engine";
import type { PairingProfile } from "../physics/pairing-mechanisms";
import type { DefectStructure } from "../physics/defect-engine";
import type { PressureTcPoint, HydrideFormationResult } from "./pressure-engine";
import type { SynthesisPath, SynthesisStep, SynthesisVector } from "../physics/synthesis-simulator";
import {
  defaultSynthesisVector,
  clampSynthesisVector,
  mutateSynthesisVector,
  crossoverSynthesisVectors,
  checkSynthesisFeasibility,
  optimizeSynthesisPath,
  simulateSynthesisEffects,
} from "../physics/synthesis-simulator";

export interface MultiEngineInsights {
  formula: string;
  materialClass: string;
  predictedTc: number;

  physics?: {
    lambda: number;
    omegaLog: number;
    dosAtFermi: number;
    metallicity: number;
    stabilityScore: number;
    correlationStrength: number;
  };

  topology?: {
    topologicalScore: number;
    z2Score: number;
    socStrength: number;
    topologicalClass: string;
    majoranaFeasibility: number;
    bandInversionProbability: number;
  };

  fermi?: {
    nestingScore: number;
    pocketCount: number;
    electronHoleBalance: number;
    cylindricalCharacter: number;
    multiBandScore: number;
    sigmaBandPresence: number;
  };

  pairing?: {
    dominantMechanism: string;
    pairingSymmetry: string;
    compositePairingStrength: number;
    phononStrength: number;
    spinStrength: number;
    orbitalStrength: number;
  };

  pressure?: {
    optimalPressure: number;
    maxTc: number;
    pressureTcCurve: PressureTcPoint[];
    hydrideFormation: HydrideFormationResult | null;
  };

  defect?: {
    variants: DefectStructure[];
    bestDopant: string;
    bestTcBoost: number;
    optimalDefectType: string;
  };
}

export interface SynthesisGenome {
  precursorStrategy: number;
  reactionSequenceType: number;
  temperatureProfile: number;
  pressureProfile: number;
  coolingStrategy: number;
  atmosphereChoice: number;
  annealingIntensity: number;
  dopingLevel: number;
  thermalCycleCount: number;
  strainEngineering: number;
  postProcessingType: number;
  topologicalProtection: number;
}

export interface NovelSynthesisRoute {
  id: string;
  formula: string;
  genome: SynthesisGenome;
  steps: SynthesisStep[];
  precursors: string[];
  totalDuration: number;
  feasibilityScore: number;
  noveltyScore: number;
  fitnessScore: number;
  engineContributions: string[];
  rationale: string[];
  synthesisVector: SynthesisVector;
  classification: "practical" | "experimental" | "unrealistic";
}

export interface SynthesisDiscoveryResult {
  formula: string;
  materialClass: string;
  bestRoute: NovelSynthesisRoute | null;
  allRoutes: NovelSynthesisRoute[];
  evolutionGenerations: number;
  convergenceScore: number;
  engineInputSummary: string[];
}

const ATMOSPHERE_OPTIONS = ["argon", "nitrogen", "vacuum", "hydrogen", "oxygen", "ammonia", "forming-gas"];
const PRECURSOR_STRATEGIES = ["elemental", "binary-oxide", "binary-halide", "organometallic", "hydride-precursor", "carbonate"];
const POST_PROCESSING = ["none", "ion-implantation", "thin-film-deposition", "surface-passivation", "epitaxial-growth", "mechanical-exfoliation"];
const REACTION_SEQUENCES = ["solid-state", "sol-gel", "arc-melting", "ball-milling", "high-pressure", "CVD", "MBE", "hydrothermal", "flux-growth"];

const discoveryStats = {
  totalDiscoveries: 0,
  totalRoutes: 0,
  avgFitness: 0,
  bestFitness: 0,
  bestFormula: "",
  bestKnownTc: 0,
  engineUsage: {} as Record<string, number>,
};

interface CompositionFeedback {
  goodMotifs: Map<string, { score: number; count: number }>;
  badMotifs: Map<string, { penalty: number; count: number }>;
  formulaOutcomes: Map<string, { tc: number; stable: boolean; formationEnergy: number }>;
}

const compositionFeedback: CompositionFeedback = {
  goodMotifs: new Map(),
  badMotifs: new Map(),
  formulaOutcomes: new Map(),
};

const adaptiveMutationState = {
  currentRate: 0.20,
  baseRate: 0.20,
  minRate: 0.08,
  maxRate: 0.50,
  generationsWithoutImprovement: 0,
  lastBestFitness: 0,
  stagnationThreshold: 30,
  totalAdaptations: 0,
};

function extractMotifs(formula: string): string[] {
  const elements = parseFormulaElements(formula);
  const motifs: string[] = [];
  for (const el of elements) motifs.push(el);
  const sorted = [...elements].sort();
  for (let i = 0; i < sorted.length; i++) {
    for (let j = i + 1; j < sorted.length; j++) {
      motifs.push(`${sorted[i]}-${sorted[j]}`);
    }
  }
  const family = classifyFamily(formula);
  if (family) motifs.push(`family:${family}`);
  return motifs;
}

export function recordDFTFeedbackForGA(
  formula: string,
  result: { tc: number; stable: boolean; formationEnergy: number }
): void {
  compositionFeedback.formulaOutcomes.set(formula, result);
  const motifs = extractMotifs(formula);

  const isGood = result.tc > 20 && result.stable && result.formationEnergy < 0.5;
  const isBad = !result.stable || result.formationEnergy > 1.0 || result.tc < 1;

  for (const motif of motifs) {
    if (isGood) {
      const existing = compositionFeedback.goodMotifs.get(motif) || { score: 0, count: 0 };
      const reward = Math.min(1, result.tc / 200) * (result.stable ? 1.0 : 0.3);
      existing.score = (existing.score * existing.count + reward) / (existing.count + 1);
      existing.count++;
      compositionFeedback.goodMotifs.set(motif, existing);
    }
    if (isBad) {
      const existing = compositionFeedback.badMotifs.get(motif) || { penalty: 0, count: 0 };
      let penalty = 0;
      if (result.formationEnergy > 1.0) penalty += 0.3;
      if (result.formationEnergy > 2.0) penalty += 0.2;
      if (!result.stable) penalty += 0.3;
      if (result.tc < 1) penalty += 0.2;
      existing.penalty = (existing.penalty * existing.count + penalty) / (existing.count + 1);
      existing.count++;
      compositionFeedback.badMotifs.set(motif, existing);
    }
  }
}

function getCompositionBias(formula: string): number {
  const motifs = extractMotifs(formula);
  let bias = 0;
  let motifCount = 0;

  for (const motif of motifs) {
    const good = compositionFeedback.goodMotifs.get(motif);
    if (good && good.count >= 2) {
      bias += good.score * Math.min(1, good.count / 5);
      motifCount++;
    }
    const bad = compositionFeedback.badMotifs.get(motif);
    if (bad && bad.count >= 2) {
      bias -= bad.penalty * Math.min(1, bad.count / 5);
      motifCount++;
    }
  }

  if (motifCount === 0) return 0;
  return Math.max(-0.3, Math.min(0.3, bias / motifCount));
}

function adaptMutationRate(currentBestFitness: number): number {
  if (currentBestFitness > adaptiveMutationState.lastBestFitness + 0.005) {
    adaptiveMutationState.generationsWithoutImprovement = 0;
    adaptiveMutationState.lastBestFitness = currentBestFitness;
    adaptiveMutationState.currentRate = Math.max(
      adaptiveMutationState.minRate,
      adaptiveMutationState.currentRate * 0.8
    );
  } else {
    adaptiveMutationState.generationsWithoutImprovement++;
    if (adaptiveMutationState.generationsWithoutImprovement >= adaptiveMutationState.stagnationThreshold) {
      adaptiveMutationState.currentRate = Math.min(
        adaptiveMutationState.maxRate,
        adaptiveMutationState.currentRate * 1.5
      );
      adaptiveMutationState.generationsWithoutImprovement = 0;
      adaptiveMutationState.totalAdaptations++;
    }
  }
  return adaptiveMutationState.currentRate;
}

function selectParentsByFitness(
  population: NovelSynthesisRoute[],
  eliteRatio: number = 0.2,
  randomRatio: number = 0.08,
): { parent1: NovelSynthesisRoute; parent2: NovelSynthesisRoute } {
  const sorted = [...population].sort((a, b) => b.fitnessScore - a.fitnessScore);
  const eliteCount = Math.max(2, Math.ceil(sorted.length * eliteRatio));
  const randomCount = Math.max(1, Math.ceil(sorted.length * randomRatio));
  const pool: NovelSynthesisRoute[] = sorted.slice(0, eliteCount);
  const nonElite = sorted.slice(eliteCount);
  for (let i = 0; i < randomCount && nonElite.length > 0; i++) {
    const idx = Math.floor(Math.random() * nonElite.length);
    pool.push(nonElite.splice(idx, 1)[0]);
  }

  const totalFitness = pool.reduce((s, r) => s + Math.max(0.001, r.fitnessScore), 0);
  const pick = (): NovelSynthesisRoute => {
    let r = Math.random() * totalFitness;
    for (const individual of pool) {
      r -= Math.max(0.001, individual.fitnessScore);
      if (r <= 0) return individual;
    }
    return pool[pool.length - 1];
  };

  const p1 = pick();
  let p2 = pick();
  let attempts = 0;
  while (p2 === p1 && attempts < 5 && pool.length > 1) {
    p2 = pick();
    attempts++;
  }
  return { parent1: p1, parent2: p2 };
}

export function getGAEvolutionStats() {
  return {
    mutationRate: Math.round(adaptiveMutationState.currentRate * 1000) / 1000,
    generationsWithoutImprovement: adaptiveMutationState.generationsWithoutImprovement,
    totalAdaptations: adaptiveMutationState.totalAdaptations,
    stagnationThreshold: adaptiveMutationState.stagnationThreshold,
    goodMotifCount: compositionFeedback.goodMotifs.size,
    badMotifCount: compositionFeedback.badMotifs.size,
    formulaOutcomeCount: compositionFeedback.formulaOutcomes.size,
    topGoodMotifs: [...compositionFeedback.goodMotifs.entries()]
      .sort((a, b) => b[1].score * b[1].count - a[1].score * a[1].count)
      .slice(0, 10)
      .map(([motif, data]) => ({ motif, score: Math.round(data.score * 1000) / 1000, count: data.count })),
    topBadMotifs: [...compositionFeedback.badMotifs.entries()]
      .sort((a, b) => b[1].penalty * b[1].count - a[1].penalty * a[1].count)
      .slice(0, 10)
      .map(([motif, data]) => ({ motif, penalty: Math.round(data.penalty * 1000) / 1000, count: data.count })),
  };
}

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
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

function randRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function generateId(): string {
  return `synth-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function randomGenome(): SynthesisGenome {
  return {
    precursorStrategy: Math.random(),
    reactionSequenceType: Math.random(),
    temperatureProfile: Math.random(),
    pressureProfile: Math.random(),
    coolingStrategy: Math.random(),
    atmosphereChoice: Math.random(),
    annealingIntensity: Math.random(),
    dopingLevel: Math.random(),
    thermalCycleCount: Math.random(),
    strainEngineering: Math.random(),
    postProcessingType: Math.random(),
    topologicalProtection: Math.random(),
  };
}

function mutateGenome(g: SynthesisGenome, mutationRate: number = 0.2): SynthesisGenome {
  const result = { ...g };
  const keys = Object.keys(result) as (keyof SynthesisGenome)[];
  const numMutations = Math.max(1, Math.floor(keys.length * mutationRate));

  for (let i = 0; i < numMutations; i++) {
    const key = keys[Math.floor(Math.random() * keys.length)];
    result[key] = Math.max(0, Math.min(1, result[key] + randRange(-0.3, 0.3)));
  }
  return result;
}

function crossoverGenomes(a: SynthesisGenome, b: SynthesisGenome): SynthesisGenome {
  const result = { ...a };
  const keys = Object.keys(result) as (keyof SynthesisGenome)[];
  for (const key of keys) {
    if (Math.random() < 0.5) {
      result[key] = b[key];
    }
  }
  return result;
}

function genomeToSynthesisVector(genome: SynthesisGenome, insights: MultiEngineInsights): SynthesisVector {
  const mc = insights.materialClass.toLowerCase();
  const base = defaultSynthesisVector(mc);

  let tempScale = 800 + genome.temperatureProfile * 1700;
  let pressScale = genome.pressureProfile * 300;
  let coolRate = 1 + genome.coolingStrategy * 5000;
  let annealTime = 0.5 + genome.annealingIntensity * 100;
  let thermalCycles = Math.round(genome.thermalCycleCount * 20);
  let strain = (genome.strainEngineering - 0.5) * 10;
  let oxygenP = 0;

  if (insights.physics) {
    if (insights.physics.lambda > 1.5) {
      pressScale = Math.max(pressScale, 50 + insights.physics.lambda * 30);
    }
    if (insights.physics.stabilityScore < 0.3) {
      coolRate = Math.max(coolRate, 500);
    }
  }

  if (insights.topology) {
    if (insights.topology.topologicalScore > 0.5) {
      coolRate = Math.min(coolRate, 100);
      annealTime = Math.max(annealTime, 24);
    }
    if (insights.topology.socStrength > 0.3) {
      strain = Math.max(strain, 1.0);
    }
  }

  if (insights.fermi) {
    if (insights.fermi.nestingScore > 0.5) {
      const optimalAnneal = 400 + insights.fermi.nestingScore * 600;
      tempScale = Math.min(tempScale, optimalAnneal + 500);
      annealTime = Math.max(annealTime, 12);
    }
  }

  if (insights.pressure) {
    if (insights.pressure.optimalPressure > 0) {
      pressScale = Math.max(pressScale, insights.pressure.optimalPressure * 0.8);
    }
  }

  if (insights.pairing) {
    if (insights.pairing.dominantMechanism === "phonon") {
      coolRate = Math.min(coolRate, 200);
    } else if (insights.pairing.dominantMechanism === "spin") {
      oxygenP = insights.pairing.pairingSymmetry.includes("d-wave") ? 0.21 : 0;
    }
    if (mc.includes("cuprate") || insights.pairing.pairingSymmetry.includes("d-wave")) {
      oxygenP = Math.max(oxygenP, 0.15);
    }
  }

  if (insights.defect) {
    if (insights.defect.bestTcBoost > 0) {
      strain += insights.defect.bestTcBoost * 0.5;
    }
  }

  return clampSynthesisVector({
    temperature: tempScale,
    pressure: pressScale,
    coolingRate: coolRate,
    annealTime: annealTime,
    currentDensity: genome.dopingLevel * 100,
    magneticField: 0,
    thermalCycles,
    strain,
    oxygenPressure: oxygenP,
  });
}

function selectAtmosphere(genome: SynthesisGenome, insights: MultiEngineInsights): string {
  const mc = insights.materialClass.toLowerCase();

  if (mc.includes("cuprate")) return "oxygen";
  if (mc.includes("hydride")) return "hydrogen";
  if (mc.includes("nitride")) return "nitrogen";

  if (insights.pairing) {
    if (insights.pairing.dominantMechanism === "spin" && insights.pairing.pairingSymmetry.includes("d-wave")) {
      return "oxygen";
    }
  }

  const idx = Math.floor(genome.atmosphereChoice * ATMOSPHERE_OPTIONS.length);
  return ATMOSPHERE_OPTIONS[Math.min(idx, ATMOSPHERE_OPTIONS.length - 1)];
}

function selectPrecursors(genome: SynthesisGenome, insights: MultiEngineInsights): string[] {
  const elements = parseFormulaElements(insights.formula);
  const mc = insights.materialClass.toLowerCase();
  const precursors: string[] = [];

  const stratIdx = Math.floor(genome.precursorStrategy * PRECURSOR_STRATEGIES.length);
  const strategy = PRECURSOR_STRATEGIES[Math.min(stratIdx, PRECURSOR_STRATEGIES.length - 1)];

  switch (strategy) {
    case "elemental":
      precursors.push(...elements.map(el => `${el} (elemental)`));
      break;
    case "binary-oxide":
      for (const el of elements) {
        if (["O", "F", "Cl", "H"].includes(el)) continue;
        precursors.push(`${el}2O3`);
      }
      if (elements.includes("O")) precursors.push("O2 gas");
      break;
    case "binary-halide":
      for (const el of elements) {
        if (["F", "Cl", "Br", "H", "O"].includes(el)) continue;
        precursors.push(`${el}Cl2`);
      }
      break;
    case "hydride-precursor":
      for (const el of elements) {
        if (el === "H") continue;
        const data = getElementData(el);
        if (data && (isTransitionMetal(el) || isRareEarth(el))) {
          precursors.push(`${el}H2`);
        } else {
          precursors.push(`${el} (elemental)`);
        }
      }
      if (!elements.includes("H")) precursors.push("H2 gas");
      break;
    case "carbonate":
      for (const el of elements) {
        if (["O", "C", "H", "N"].includes(el)) continue;
        precursors.push(`${el}CO3`);
      }
      break;
    default:
      precursors.push(...elements.map(el => `${el} (elemental)`));
  }

  if (precursors.length === 0) {
    precursors.push(...elements.map(el => `${el} (elemental)`));
  }

  return precursors;
}

function genomeToSteps(genome: SynthesisGenome, insights: MultiEngineInsights): SynthesisStep[] {
  const sv = genomeToSynthesisVector(genome, insights);
  const atmosphere = selectAtmosphere(genome, insights);
  const mc = insights.materialClass.toLowerCase();
  const steps: SynthesisStep[] = [];

  const seqIdx = Math.floor(genome.reactionSequenceType * REACTION_SEQUENCES.length);
  const reactionType = REACTION_SEQUENCES[Math.min(seqIdx, REACTION_SEQUENCES.length - 1)];

  if (reactionType === "ball-milling" || genome.precursorStrategy < 0.3) {
    steps.push({
      order: 1,
      method: "ball-milling",
      temperature: 300,
      pressure: 0,
      coolingRate: 0,
      annealTemp: 0,
      duration: 2 + genome.annealingIntensity * 8,
      atmosphere: "argon",
      notes: "Precursor mixing and mechanical activation",
    });
  }

  if (sv.pressure > 50 || reactionType === "high-pressure") {
    steps.push({
      order: steps.length + 1,
      method: "high-pressure",
      temperature: sv.temperature,
      pressure: sv.pressure,
      coolingRate: 0,
      annealTemp: 0,
      duration: 2 + genome.annealingIntensity * 6,
      atmosphere: atmosphere,
      notes: `High-pressure synthesis at ${sv.pressure.toFixed(0)} GPa`,
    });
  } else {
    steps.push({
      order: steps.length + 1,
      method: reactionType,
      temperature: sv.temperature,
      pressure: Math.max(0, sv.pressure),
      coolingRate: 0,
      annealTemp: 0,
      duration: 4 + genome.annealingIntensity * 20,
      atmosphere: atmosphere,
      notes: `Primary ${reactionType} synthesis`,
    });
  }

  if (genome.annealingIntensity > 0.3) {
    const annealTemp = sv.temperature * 0.6 + genome.annealingIntensity * sv.temperature * 0.2;
    steps.push({
      order: steps.length + 1,
      method: "anneal",
      temperature: 0,
      pressure: 0,
      coolingRate: 0,
      annealTemp: Math.min(annealTemp, sv.temperature * 0.8),
      duration: sv.annealTime,
      atmosphere: atmosphere,
      notes: `Annealing for phase ordering and defect healing`,
    });
  }

  if (insights.topology && insights.topology.topologicalScore > 0.4) {
    steps.push({
      order: steps.length + 1,
      method: "anneal",
      temperature: 0,
      pressure: 0,
      coolingRate: Math.min(sv.coolingRate, 50),
      annealTemp: 300 + insights.topology.topologicalScore * 200,
      duration: 12 + genome.topologicalProtection * 36,
      atmosphere: "vacuum",
      notes: "Slow cool for topological surface state preservation",
    });
  }

  if (sv.coolingRate > 100) {
    steps.push({
      order: steps.length + 1,
      method: "quench",
      temperature: 0,
      pressure: 0,
      coolingRate: sv.coolingRate,
      annealTemp: 0,
      duration: Math.max(0.001, 10 / sv.coolingRate),
      atmosphere: "argon",
      notes: `Rapid quench at ${sv.coolingRate.toFixed(0)} K/s`,
    });
  } else {
    steps.push({
      order: steps.length + 1,
      method: "quench",
      temperature: 0,
      pressure: 0,
      coolingRate: sv.coolingRate,
      annealTemp: 0,
      duration: Math.max(0.01, 100 / Math.max(1, sv.coolingRate)),
      atmosphere: "argon",
      notes: "Controlled cooling",
    });
  }

  if (sv.thermalCycles > 1) {
    steps.push({
      order: steps.length + 1,
      method: "heat-treatment",
      temperature: sv.temperature * 0.7,
      pressure: 0,
      coolingRate: sv.coolingRate * 0.5,
      annealTemp: 0,
      duration: sv.thermalCycles * 2,
      atmosphere: atmosphere,
      notes: `${sv.thermalCycles} thermal cycles for grain refinement`,
    });
  }

  if (insights.defect && insights.defect.bestTcBoost > 0.05 && genome.dopingLevel > 0.3) {
    steps.push({
      order: steps.length + 1,
      method: "heat-treatment",
      temperature: 400 + genome.dopingLevel * 400,
      pressure: 0,
      coolingRate: 10,
      annealTemp: 0,
      duration: 4 + genome.dopingLevel * 20,
      atmosphere: atmosphere,
      notes: `Doping-optimized anneal (target: ${insights.defect.bestDopant} incorporation)`,
    });
  }

  const ppIdx = Math.floor(genome.postProcessingType * POST_PROCESSING.length);
  const postProc = POST_PROCESSING[Math.min(ppIdx, POST_PROCESSING.length - 1)];
  if (postProc !== "none" && genome.postProcessingType > 0.5) {
    steps.push({
      order: steps.length + 1,
      method: postProc,
      temperature: 300,
      pressure: 0,
      coolingRate: 0,
      annealTemp: 0,
      duration: 1,
      atmosphere: "vacuum",
      notes: `Post-processing: ${postProc}`,
    });
  }

  return steps;
}

function computeNoveltyScore(genome: SynthesisGenome, insights: MultiEngineInsights): number {
  let novelty = 0;

  const mc = insights.materialClass.toLowerCase();
  if (mc.includes("hydride") && genome.pressureProfile < 0.2) {
    novelty += 0.2;
  }
  if (!mc.includes("cuprate") && genome.atmosphereChoice > 0.7) {
    novelty += 0.1;
  }

  if (insights.topology && insights.topology.topologicalScore > 0.4 && genome.topologicalProtection > 0.5) {
    novelty += 0.15;
  }

  if (insights.fermi && insights.fermi.nestingScore > 0.5 && genome.annealingIntensity > 0.6) {
    novelty += 0.1;
  }

  if (insights.pairing && insights.pairing.dominantMechanism !== "phonon" && genome.strainEngineering > 0.5) {
    novelty += 0.15;
  }

  if (insights.defect && insights.defect.bestTcBoost > 0.1 && genome.dopingLevel > 0.4) {
    novelty += 0.1;
  }

  if (genome.precursorStrategy > 0.6 && genome.reactionSequenceType > 0.5) {
    novelty += 0.1;
  }

  if (genome.postProcessingType > 0.5) {
    novelty += 0.1;
  }

  return Math.min(1.0, novelty);
}

function computeFitnessScore(
  route: { steps: SynthesisStep[]; feasibilityScore: number; noveltyScore: number },
  insights: MultiEngineInsights,
  bestKnownTc: number = 0
): number {
  let surrogateFitness = 0;
  let usedSurrogate = false;
  let surrogateExplorationBonus = 0;
  try {
    const sf = computeSurrogateFitness(insights.formula);
    surrogateFitness = sf.fitness;
    surrogateExplorationBonus = sf.uncertaintyBreakdown.explorationBonus;
    usedSurrogate = true;
  } catch {}

  const weights = getCurrentFitnessWeights();
  const surrogateWeight = usedSurrogate ? 0.40 : 0.0;
  const remainingWeight = 1.0 - surrogateWeight;

  const tcWeight = 0.30 * remainingWeight;
  const feasWeight = 0.25 * remainingWeight;
  const novelWeight = 0.20 * remainingWeight;
  const engineWeight = 0.25 * remainingWeight;

  const adaptiveDenom = Math.max(100, bestKnownTc * 0.5);
  const tcNorm = Math.min(1.0, insights.predictedTc / adaptiveDenom);
  const feasNorm = route.feasibilityScore;
  const novelNorm = route.noveltyScore;

  let engineScore = 0;
  let engineCount = 0;

  if (insights.physics) {
    engineScore += Math.min(1.0, insights.physics.lambda / 2.0) * 0.3;
    engineScore += insights.physics.stabilityScore * 0.2;
    engineCount++;
  }
  if (insights.topology) {
    engineScore += insights.topology.topologicalScore * 0.15;
    engineCount++;
  }
  if (insights.fermi) {
    engineScore += insights.fermi.multiBandScore * 0.1;
    engineScore += insights.fermi.nestingScore * 0.1;
    engineCount++;
  }
  if (insights.pairing) {
    engineScore += insights.pairing.compositePairingStrength * 0.15;
    engineCount++;
  }
  if (insights.pressure) {
    const pTcNorm = Math.min(1.0, insights.pressure.maxTc / 200);
    engineScore += pTcNorm * 0.1;
    engineCount++;
  }
  if (insights.defect) {
    engineScore += Math.min(1.0, insights.defect.bestTcBoost * 5) * 0.1;
    engineCount++;
  }

  const engineNorm = engineCount > 0 ? Math.min(1.0, engineScore / Math.max(1, Math.sqrt(engineCount))) : 0.5;

  const baseFitness = surrogateWeight * surrogateFitness +
    tcWeight * tcNorm + feasWeight * feasNorm + novelWeight * novelNorm + engineWeight * engineNorm;

  const compositionBias = getCompositionBias(insights.formula);

  return Math.max(0, Math.min(1.0, baseFitness + surrogateExplorationBonus + compositionBias));
}

let gaAdaptationCalls = 0;
export function getSynthesisGAAdaptationStats() {
  const weights = getCurrentFitnessWeights();
  const expWeight = getExplorationWeight();
  return {
    currentSurrogateWeights: weights,
    gaAdaptationCalls,
    surrogateIntegrated: true,
    explorationWeight: expWeight,
    explorationDriven: expWeight > 0.05,
  };
}

function buildEngineContributions(insights: MultiEngineInsights): string[] {
  const contributions: string[] = [];

  if (insights.physics) {
    contributions.push(`Physics: lambda=${insights.physics.lambda.toFixed(2)}, stability=${insights.physics.stabilityScore.toFixed(2)}`);
    discoveryStats.engineUsage["physics"] = (discoveryStats.engineUsage["physics"] || 0) + 1;
  }
  if (insights.topology) {
    contributions.push(`Topology: score=${insights.topology.topologicalScore.toFixed(2)}, class=${insights.topology.topologicalClass}`);
    discoveryStats.engineUsage["topology"] = (discoveryStats.engineUsage["topology"] || 0) + 1;
  }
  if (insights.fermi) {
    contributions.push(`Fermi: nesting=${insights.fermi.nestingScore.toFixed(2)}, pockets=${insights.fermi.pocketCount}`);
    discoveryStats.engineUsage["fermi"] = (discoveryStats.engineUsage["fermi"] || 0) + 1;
  }
  if (insights.pairing) {
    contributions.push(`Pairing: mechanism=${insights.pairing.dominantMechanism}, symmetry=${insights.pairing.pairingSymmetry}`);
    discoveryStats.engineUsage["pairing"] = (discoveryStats.engineUsage["pairing"] || 0) + 1;
  }
  if (insights.pressure) {
    contributions.push(`Pressure: optimal=${insights.pressure.optimalPressure} GPa, maxTc=${insights.pressure.maxTc.toFixed(1)}K`);
    discoveryStats.engineUsage["pressure"] = (discoveryStats.engineUsage["pressure"] || 0) + 1;
  }
  if (insights.defect) {
    contributions.push(`Defect: dopant=${insights.defect.bestDopant}, boost=${insights.defect.bestTcBoost.toFixed(3)}`);
    discoveryStats.engineUsage["defect"] = (discoveryStats.engineUsage["defect"] || 0) + 1;
  }

  return contributions;
}

function buildRationale(genome: SynthesisGenome, insights: MultiEngineInsights): string[] {
  const rationale: string[] = [];
  const mc = insights.materialClass.toLowerCase();

  if (insights.physics && insights.physics.lambda > 1.0) {
    rationale.push(`Strong electron-phonon coupling (lambda=${insights.physics.lambda.toFixed(2)}) suggests phonon-mediated pairing; optimized cooling preserves lattice dynamics`);
  }

  if (insights.topology && insights.topology.topologicalScore > 0.4) {
    rationale.push(`Topological character (${insights.topology.topologicalClass}) requires careful surface preservation via slow cooling and vacuum post-anneal`);
  }

  if (insights.fermi && insights.fermi.nestingScore > 0.5) {
    rationale.push(`High Fermi surface nesting (${insights.fermi.nestingScore.toFixed(2)}) drives instability-mediated pairing; anneal temperature tuned near nesting-driven transition`);
  }

  if (insights.pairing) {
    if (insights.pairing.dominantMechanism === "spin") {
      rationale.push(`Spin-fluctuation mediated pairing with ${insights.pairing.pairingSymmetry} symmetry; oxygen atmosphere supports exchange coupling`);
    } else if (insights.pairing.dominantMechanism === "phonon") {
      rationale.push(`Conventional BCS pairing; slow cooling maximizes phase purity for optimal phonon spectrum`);
    }
  }

  if (insights.pressure && insights.pressure.optimalPressure > 50) {
    rationale.push(`High-pressure synthesis at ${insights.pressure.optimalPressure} GPa followed by decompression route to retain metastable structure`);
  }

  if (insights.defect && insights.defect.bestTcBoost > 0.05) {
    rationale.push(`${insights.defect.bestDopant} doping enhances Tc by ${(insights.defect.bestTcBoost * 100).toFixed(1)}%; dedicated doping anneal step included`);
  }

  if (rationale.length === 0) {
    rationale.push(`Standard ${mc} synthesis route with evolutionary optimization of parameters`);
  }

  return rationale;
}

function genomeToRoute(genome: SynthesisGenome, insights: MultiEngineInsights, bestKnownTc: number = 0): NovelSynthesisRoute {
  const steps = genomeToSteps(genome, insights);
  const sv = genomeToSynthesisVector(genome, insights);
  const precursors = selectPrecursors(genome, insights);
  const totalDuration = steps.reduce((s, step) => s + step.duration, 0);

  const feasResult = checkSynthesisFeasibility(sv);
  const feasibilityScore = Math.max(0, 1 - feasResult.feasibilityScore / 10);
  const noveltyScore = computeNoveltyScore(genome, insights);

  const routeObj = { steps, feasibilityScore, noveltyScore };
  const fitnessScore = computeFitnessScore(routeObj, insights, bestKnownTc);

  const engineContributions = buildEngineContributions(insights);
  const rationale = buildRationale(genome, insights);

  return {
    id: generateId(),
    formula: insights.formula,
    genome,
    steps,
    precursors,
    totalDuration: Math.round(totalDuration * 100) / 100,
    feasibilityScore: Math.round(feasibilityScore * 1000) / 1000,
    noveltyScore: Math.round(noveltyScore * 1000) / 1000,
    fitnessScore: Math.round(fitnessScore * 1000) / 1000,
    engineContributions,
    rationale,
    synthesisVector: sv,
    classification: feasResult.classification,
  };
}

export function discoverNovelSynthesisPaths(
  insights: MultiEngineInsights,
  populationSize: number = 20,
  generations: number = 15,
  eliteRatio: number = 0.3,
): SynthesisDiscoveryResult {
  discoveryStats.totalDiscoveries++;

  if (insights.predictedTc > discoveryStats.bestKnownTc) {
    discoveryStats.bestKnownTc = insights.predictedTc;
  }
  const currentBestKnownTc = discoveryStats.bestKnownTc;

  let population: NovelSynthesisRoute[] = [];

  const basePath = optimizeSynthesisPath(insights.formula, insights.materialClass, insights.predictedTc);
  const baseGenome = randomGenome();
  population.push(genomeToRoute(baseGenome, insights, currentBestKnownTc));

  for (let i = 1; i < populationSize; i++) {
    const genome = randomGenome();
    population.push(genomeToRoute(genome, insights, currentBestKnownTc));
  }

  let bestEverFitness = 0;
  let bestEverRoute: NovelSynthesisRoute | null = null;

  for (let gen = 0; gen < generations; gen++) {
    population.sort((a, b) => b.fitnessScore - a.fitnessScore);

    if (population[0].fitnessScore > bestEverFitness) {
      bestEverFitness = population[0].fitnessScore;
      bestEverRoute = population[0];
    }

    const mutRate = adaptMutationRate(population[0].fitnessScore);

    const preserveCount = Math.max(1, Math.ceil(populationSize * 0.05));
    const newPop: NovelSynthesisRoute[] = population.slice(0, preserveCount);

    while (newPop.length < populationSize) {
      const { parent1, parent2 } = selectParentsByFitness(population, 0.20, 0.08);

      let childGenome: SynthesisGenome;
      const r = Math.random();
      if (r < 0.4) {
        childGenome = crossoverGenomes(parent1.genome, parent2.genome);
      } else if (r < 0.4 + 0.45) {
        childGenome = mutateGenome(parent1.genome, mutRate);
      } else {
        childGenome = randomGenome();
      }

      newPop.push(genomeToRoute(childGenome, insights, currentBestKnownTc));
    }

    population = newPop;
  }

  population.sort((a, b) => b.fitnessScore - a.fitnessScore);

  if (population[0].fitnessScore > bestEverFitness) {
    bestEverRoute = population[0];
    bestEverFitness = population[0].fitnessScore;
  }

  const topRoutes = population
    .slice(0, 5)
    .filter(r => r.classification !== "unrealistic" || r.fitnessScore > 0.5);

  if (topRoutes.length === 0 && population.length > 0) {
    topRoutes.push(population[0]);
  }

  const convergenceScore = population.length >= 2
    ? 1 - Math.abs(population[0].fitnessScore - population[Math.min(4, population.length - 1)].fitnessScore) / Math.max(0.01, population[0].fitnessScore)
    : 0;

  const engineInputSummary: string[] = [];
  if (insights.physics) engineInputSummary.push("physics-engine");
  if (insights.topology) engineInputSummary.push("topology-engine");
  if (insights.fermi) engineInputSummary.push("fermi-surface-engine");
  if (insights.pairing) engineInputSummary.push("pairing-mechanism-engine");
  if (insights.pressure) engineInputSummary.push("pressure-engine");
  if (insights.defect) engineInputSummary.push("defect-engine");

  discoveryStats.totalRoutes += topRoutes.length;
  if (bestEverFitness > discoveryStats.bestFitness) {
    discoveryStats.bestFitness = bestEverFitness;
    discoveryStats.bestFormula = insights.formula;
  }

  return {
    formula: insights.formula,
    materialClass: insights.materialClass,
    bestRoute: bestEverRoute,
    allRoutes: topRoutes,
    evolutionGenerations: generations,
    convergenceScore: Math.round(Math.max(0, Math.min(1, convergenceScore)) * 1000) / 1000,
    engineInputSummary,
  };
}

export function getSynthesisDiscoveryStats() {
  return {
    totalDiscoveries: discoveryStats.totalDiscoveries,
    totalRoutes: discoveryStats.totalRoutes,
    avgFitness: discoveryStats.totalDiscoveries > 0
      ? Math.round((discoveryStats.bestFitness / discoveryStats.totalDiscoveries) * 1000) / 1000
      : 0,
    bestFitness: Math.round(discoveryStats.bestFitness * 1000) / 1000,
    bestFormula: discoveryStats.bestFormula,
    engineUsage: { ...discoveryStats.engineUsage },
  };
}
