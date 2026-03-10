import { getElementData, isTransitionMetal, isRareEarth, isActinide } from "./elemental-data";
import { classifyFamily } from "./utils";
import { computeSurrogateFitness, getCurrentFitnessWeights, getExplorationWeight } from "./surrogate-fitness";
import type { TopologicalAnalysis } from "../physics/topology-engine";
import type { FermiSurfaceResult } from "../physics/fermi-surface-engine";
import type { PairingProfile } from "../physics/pairing-mechanisms";
import type { DefectStructure } from "../physics/defect-engine";
import type { PressureTcPoint, HydrideFormationResult } from "./pressure-engine";
import { assessHighPressureStability, scanPressureTcCurve } from "./pressure-engine";
import { samplePressureFromClusters } from "./pressure-screening";
import { evaluateSynthesisGate, HARD_GATE_THRESHOLD } from "../synthesis/synthesis-gate";
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
  pressureGene: number;
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
  decodedPressureGpa: number;
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

const STRUCTURAL_MOTIFS = [
  "hydride-cage", "layered-boride", "A15-chain", "perovskite-3D",
  "clathrate-cage", "CuO2-plane", "FeAs-layer", "kagome-flat",
  "hexagonal-layer", "NaCl-rocksalt", "ThCr2Si2-pnictide",
  "MgB2-sigma", "Ruddlesden-Popper", "nickelate-IL", "H-channel",
  "anti-perovskite", "Chevrel-phase", "Heusler-L21", "pyrochlore",
  "skutterudite", "BiS2-layer", "infinite-layer", "Laves-MgZn2",
] as const;

type StructuralMotif = typeof STRUCTURAL_MOTIFS[number];

interface MotifRewardEntry {
  weight: number;
  successes: number;
  failures: number;
  totalReward: number;
  avgTc: number;
  totalTc: number;
}

const structuralMotifRewards = new Map<string, MotifRewardEntry>();

function initMotifRewards(): void {
  if (structuralMotifRewards.size > 0) return;
  for (const motif of STRUCTURAL_MOTIFS) {
    structuralMotifRewards.set(motif, {
      weight: 1.0,
      successes: 0,
      failures: 0,
      totalReward: 0,
      avgTc: 0,
      totalTc: 0,
    });
  }
}

function classifyStructuralMotif(formula: string, materialClass: string): StructuralMotif[] {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const mc = materialClass.toLowerCase();
  const motifs: StructuralMotif[] = [];

  const hasH = elements.includes("H");
  const hCount = counts["H"] || 0;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const hFraction = hCount / totalAtoms;

  if (hasH && hFraction > 0.5) motifs.push("hydride-cage");
  if (hasH && hFraction > 0.3 && hFraction <= 0.5) motifs.push("H-channel");

  if (elements.includes("B") && elements.some(e => isTransitionMetal(e) || isRareEarth(e))) {
    motifs.push("layered-boride");
  }

  if (mc.includes("a15") || (elements.length === 2 && elements.every(e => isTransitionMetal(e)))) {
    motifs.push("A15-chain");
  }

  if (mc.includes("perovskite") || (elements.length === 3 && elements.includes("O") && (counts["O"] || 0) === 3)) {
    motifs.push("perovskite-3D");
  }

  if (mc.includes("clathrate") || (hasH && hCount >= 6)) {
    motifs.push("clathrate-cage");
  }

  if (mc.includes("cuprate") || (elements.includes("Cu") && elements.includes("O"))) {
    motifs.push("CuO2-plane");
  }

  if (mc.includes("pnictide") || mc.includes("iron") || (elements.includes("Fe") && (elements.includes("As") || elements.includes("P") || elements.includes("Se")))) {
    motifs.push("FeAs-layer");
  }

  if (mc.includes("kagome") || elements.includes("V") && elements.includes("Si")) {
    motifs.push("kagome-flat");
  }

  if (mc.includes("hexagonal") || elements.includes("B") && !elements.some(e => isTransitionMetal(e))) {
    motifs.push("hexagonal-layer");
  }

  if (mc.includes("nickelate") || (elements.includes("Ni") && elements.includes("O"))) {
    motifs.push("nickelate-IL");
  }

  if (mc.includes("heusler") || (elements.length >= 3 && elements.filter(e => isTransitionMetal(e)).length >= 2)) {
    motifs.push("Heusler-L21");
  }

  if (elements.includes("Mg") && elements.includes("B")) {
    motifs.push("MgB2-sigma");
  }

  if (mc.includes("thcr2si2") || (elements.length >= 3 && (elements.includes("As") || elements.includes("P")) && elements.some(e => isRareEarth(e)))) {
    motifs.push("ThCr2Si2-pnictide");
  }

  if (mc.includes("chevrel") || (elements.includes("Mo") && (elements.includes("S") || elements.includes("Se")))) {
    motifs.push("Chevrel-phase");
  }

  if (mc.includes("laves") || (elements.length === 2 && elements.every(e => isTransitionMetal(e)) && Math.abs((counts[elements[0]] || 1) / (counts[elements[1]] || 1) - 2) < 0.5)) {
    motifs.push("Laves-MgZn2");
  }

  if (motifs.length === 0) {
    if (elements.length <= 2 && elements.includes("O")) motifs.push("NaCl-rocksalt");
    else motifs.push("hexagonal-layer");
  }

  return motifs;
}

export function rewardStructuralMotif(
  formula: string,
  materialClass: string,
  result: { tc: number; stable: boolean; formationEnergy: number }
): void {
  initMotifRewards();
  const motifs = classifyStructuralMotif(formula, materialClass);
  const isSuccess = result.tc > 20 && result.stable && result.formationEnergy < 0.5;

  for (const motif of motifs) {
    const entry = structuralMotifRewards.get(motif);
    if (!entry) continue;

    if (isSuccess) {
      const reward = Math.min(1.0, result.tc / 200);
      entry.successes++;
      entry.totalReward += reward;
      entry.weight = Math.min(3.0, entry.weight + reward * 0.15);
    } else {
      entry.failures++;
      entry.weight = Math.max(0.2, entry.weight - 0.05);
    }
    entry.totalTc += result.tc;
    entry.avgTc = entry.totalTc / (entry.successes + entry.failures);
  }
}

export function getMotifWeightedMutationBias(formula: string, materialClass: string): Record<string, number> {
  initMotifRewards();
  const motifs = classifyStructuralMotif(formula, materialClass);
  const bias: Record<string, number> = {};

  for (const motif of motifs) {
    const entry = structuralMotifRewards.get(motif);
    if (!entry) continue;
    bias[motif] = entry.weight;
  }

  return bias;
}

function getMotifMutationProbability(formula: string, materialClass: string): number {
  initMotifRewards();
  const motifs = classifyStructuralMotif(formula, materialClass);
  if (motifs.length === 0) return 1.0;

  let totalWeight = 0;
  let count = 0;
  for (const motif of motifs) {
    const entry = structuralMotifRewards.get(motif);
    if (entry) {
      totalWeight += entry.weight;
      count++;
    }
  }

  const avgWeight = count > 0 ? totalWeight / count : 1.0;
  return Math.max(0.3, Math.min(2.0, avgWeight));
}

export function getStructuralMotifStats(): {
  motifs: { name: string; weight: number; successes: number; failures: number; avgTc: number; successRate: number }[];
  totalMotifs: number;
  activeMotifs: number;
} {
  initMotifRewards();
  const motifs: { name: string; weight: number; successes: number; failures: number; avgTc: number; successRate: number }[] = [];

  structuralMotifRewards.forEach((entry, name) => {
    const total = entry.successes + entry.failures;
    motifs.push({
      name,
      weight: Math.round(entry.weight * 1000) / 1000,
      successes: entry.successes,
      failures: entry.failures,
      avgTc: Math.round(entry.avgTc * 10) / 10,
      successRate: total > 0 ? Math.round((entry.successes / total) * 1000) / 1000 : 0,
    });
  });

  motifs.sort((a, b) => b.weight - a.weight);

  return {
    motifs,
    totalMotifs: STRUCTURAL_MOTIFS.length,
    activeMotifs: motifs.filter(m => m.successes + m.failures > 0).length,
  };
}

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
  result: { tc: number; stable: boolean; formationEnergy: number },
  materialClass?: string
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

  const mc = materialClass || classifyFamily(formula) || "other";
  rewardStructuralMotif(formula, mc, result);
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
    eliteArchive: getEliteArchive(),
    eliteArchiveSize: ELITE_ARCHIVE_SIZE,
    structuralMotifs: getStructuralMotifStats(),
  };
}

const ELITE_ARCHIVE_SIZE = 5;
const eliteArchive: NovelSynthesisRoute[] = [];

function updateEliteArchive(route: NovelSynthesisRoute): void {
  const existingIdx = eliteArchive.findIndex(r => r.formula === route.formula && r.id === route.id);
  if (existingIdx >= 0) {
    if (route.fitnessScore > eliteArchive[existingIdx].fitnessScore) {
      eliteArchive[existingIdx] = route;
    }
    return;
  }
  if (eliteArchive.length < ELITE_ARCHIVE_SIZE) {
    eliteArchive.push(route);
    eliteArchive.sort((a, b) => b.fitnessScore - a.fitnessScore);
    return;
  }
  const worstArchive = eliteArchive[eliteArchive.length - 1];
  if (route.fitnessScore > worstArchive.fitnessScore) {
    eliteArchive[eliteArchive.length - 1] = route;
    eliteArchive.sort((a, b) => b.fitnessScore - a.fitnessScore);
  }
}

export function getEliteArchive(): { formula: string; fitness: number; classification: string }[] {
  return eliteArchive.map(r => ({
    formula: r.formula,
    fitness: Math.round(r.fitnessScore * 1000) / 1000,
    classification: r.classification,
  }));
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

export function decodePressureGene(gene: number): number {
  return Math.max(0, Math.min(350, gene * 350));
}

function randomGenome(): SynthesisGenome {
  let pressureGene = Math.random();
  if (Math.random() < 0.3) {
    try {
      const clusterPressures = samplePressureFromClusters(1);
      if (clusterPressures.length > 0) {
        pressureGene = Math.min(1, Math.max(0, clusterPressures[0] / 350));
      }
    } catch {}
  }

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
    pressureGene,
  };
}

function mutateGenome(g: SynthesisGenome, mutationRate: number = 0.2): SynthesisGenome {
  const result = { ...g };
  const keys = Object.keys(result) as (keyof SynthesisGenome)[];
  const numMutations = Math.max(1, Math.floor(keys.length * mutationRate));

  for (let i = 0; i < numMutations; i++) {
    const key = keys[Math.floor(Math.random() * keys.length)];
    if (key === "pressureGene") {
      const step = randRange(-0.03, 0.03);
      result[key] = Math.max(0, Math.min(1, result[key] + step));
    } else {
      result[key] = Math.max(0, Math.min(1, result[key] + randRange(-0.3, 0.3)));
    }
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

  const genomePressureGpa = decodePressureGene(genome.pressureGene);
  let tempScale = 800 + genome.temperatureProfile * 1700;
  let pressScale = Math.max(genome.pressureProfile * 300, genomePressureGpa);
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
  const genomePressureGpa = decodePressureGene(genome.pressureGene);
  sv.pressure = Math.max(sv.pressure, genomePressureGpa);
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
  bestKnownTc: number = 0,
  genome?: SynthesisGenome
): number {
  const genomePressureGpa = genome ? decodePressureGene(genome.pressureGene) : 0;

  let pressureTc = insights.predictedTc;
  if (genome && genomePressureGpa > 0) {
    try {
      const curve = scanPressureTcCurve(insights.formula, {
        min: Math.max(0, genomePressureGpa - 10),
        max: genomePressureGpa + 10,
        steps: 3,
      });
      const atPressure = curve.find(pt => Math.abs(pt.pressure - genomePressureGpa) <= 10);
      if (atPressure && atPressure.Tc > 0) {
        pressureTc = atPressure.Tc;
      }
    } catch {}
  }

  const adaptiveDenom = Math.max(100, bestKnownTc * 0.5);
  const tcNorm = Math.min(1.0, pressureTc / adaptiveDenom);

  let stabilityAtPressure = 0.5;
  if (genome && genomePressureGpa > 0) {
    try {
      const stability = assessHighPressureStability(insights.formula, genomePressureGpa);
      if (stability.isStable && stability.enthalpyMargin < 0) {
        stabilityAtPressure = Math.min(1.0, 0.7 + Math.abs(stability.enthalpyMargin) * 0.3);
      } else if (stability.isStable) {
        stabilityAtPressure = 0.6;
      } else {
        stabilityAtPressure = Math.max(0.1, 0.4 - stability.enthalpyMargin * 0.2);
      }
    } catch {}
  } else {
    if (insights.physics) {
      stabilityAtPressure = insights.physics.stabilityScore;
    }
  }

  let engineBonus = 0;
  if (insights.physics) {
    engineBonus += Math.min(1.0, insights.physics.lambda / 2.0) * 0.05;
  }
  if (insights.topology) {
    engineBonus += insights.topology.topologicalScore * 0.03;
  }
  if (insights.pairing) {
    engineBonus += insights.pairing.compositePairingStrength * 0.02;
  }

  const feasNorm = route.feasibilityScore;
  const novelNorm = route.noveltyScore;

  const pressurePenalty = genomePressureGpa / 500;

  const rawFitness =
    0.4 * tcNorm +
    0.3 * stabilityAtPressure +
    0.2 * feasNorm +
    0.1 * novelNorm;

  let surrogateExplorationBonus = 0;
  try {
    const sf = computeSurrogateFitness(insights.formula);
    surrogateExplorationBonus = sf.uncertaintyBreakdown.explorationBonus * 0.15;
  } catch {}

  const compositionBias = getCompositionBias(insights.formula);

  return Math.max(0, Math.min(1.0, rawFitness - pressurePenalty + engineBonus + surrogateExplorationBonus + compositionBias));
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
  let feasibilityScore = Math.max(0, 1 - feasResult.feasibilityScore / 10);

  try {
    const gateResult = evaluateSynthesisGate(insights.formula);
    feasibilityScore = feasibilityScore * 0.4 + gateResult.compositeScore * 0.6;
    if (gateResult.chemicalDistance.toxicElements.length > 0) {
      feasibilityScore *= (1 - gateResult.chemicalDistance.toxicityPenalty * 0.3);
    }
    if (gateResult.chemicalDistance.isOnePot) {
      feasibilityScore = Math.min(1, feasibilityScore + 0.05);
    }
  } catch {}

  feasibilityScore = Math.max(0, Math.min(1, feasibilityScore));
  const noveltyScore = computeNoveltyScore(genome, insights);

  const routeObj = { steps, feasibilityScore, noveltyScore };
  const fitnessScore = computeFitnessScore(routeObj, insights, bestKnownTc, genome);

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
    decodedPressureGpa: Math.round(decodePressureGene(genome.pressureGene) * 10) / 10,
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

  for (const elite of eliteArchive) {
    if (population.length < Math.min(ELITE_ARCHIVE_SIZE, Math.floor(populationSize * 0.25))) {
      const mutated = mutateGenome(elite.genome, 0.1);
      population.push(genomeToRoute(mutated, insights, currentBestKnownTc));
    }
  }

  const basePath = optimizeSynthesisPath(insights.formula, insights.materialClass, insights.predictedTc);
  const baseGenome = randomGenome();
  population.push(genomeToRoute(baseGenome, insights, currentBestKnownTc));

  while (population.length < populationSize) {
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

    const motifBoost = getMotifMutationProbability(insights.formula, insights.materialClass);
    const adjustedMutRate = Math.max(
      adaptiveMutationState.minRate,
      Math.min(adaptiveMutationState.maxRate, mutRate * motifBoost)
    );

    while (newPop.length < populationSize) {
      const { parent1, parent2 } = selectParentsByFitness(population, 0.20, 0.08);

      let childGenome: SynthesisGenome;
      const crossoverProb = motifBoost > 1.2 ? 0.50 : 0.40;
      const mutationProb = motifBoost > 1.2 ? 0.40 : 0.45;
      const r = Math.random();
      if (r < crossoverProb) {
        childGenome = crossoverGenomes(parent1.genome, parent2.genome);
      } else if (r < crossoverProb + mutationProb) {
        childGenome = mutateGenome(parent1.genome, adjustedMutRate);
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

  for (const candidate of population.slice(0, 5)) {
    updateEliteArchive(candidate);
  }

  const topRoutes = population
    .slice(0, 5)
    .filter(r => r.classification !== "unrealistic" || r.fitnessScore > 0.5)
    .filter(r => r.feasibilityScore >= HARD_GATE_THRESHOLD);

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
