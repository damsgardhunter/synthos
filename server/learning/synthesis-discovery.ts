import { getElementData, isTransitionMetal, isRareEarth, isActinide } from "./elemental-data";
import { classifyFamily, parseFormulaCounts, normalizeFormula } from "./utils";
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
  "generic-metallic",
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

const CUPRATE_SPACERS = new Set(["Y","La","Ba","Sr","Ca","Nd","Sm","Gd","Pr","Eu","Bi","Tl","Hg","Pb"]);
const HEXAGONAL_SPACE_GROUPS = new Set(["P6/mmm","P63/mmc","P6mm","P-6m2","P6/m","P63/mcm","P-62m"]);

export function classifyStructuralMotif(formula: string, materialClass: string): StructuralMotif[] {
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

  const cuCount = counts["Cu"] || 0;
  const oCount = counts["O"] || 0;
  if (mc.includes("cuprate")) {
    motifs.push("CuO2-plane");
  } else if (elements.includes("Cu") && elements.includes("O") && oCount >= cuCount * 2 && elements.some(e => CUPRATE_SPACERS.has(e))) {
    motifs.push("CuO2-plane");
  }

  if (mc.includes("pnictide") || mc.includes("iron") || (elements.includes("Fe") && (elements.includes("As") || elements.includes("P") || elements.includes("Se")))) {
    motifs.push("FeAs-layer");
  }

  if (mc.includes("kagome") || (mc.includes("hexagonal") && elements.includes("V"))) {
    motifs.push("kagome-flat");
  }

  if (mc.includes("hexagonal") || (elements.includes("B") && !elements.some(e => isTransitionMetal(e)))) {
    motifs.push("hexagonal-layer");
  }

  const niCount = counts["Ni"] || 0;
  const oCountNi = counts["O"] || 0;
  if (mc.includes("nickelate") || mc.includes("infinite-layer") || mc.includes("infinite layer")) {
    if (elements.includes("Ni") && elements.includes("O") && oCountNi <= niCount * 2 && elements.some(e => isRareEarth(e))) {
      motifs.push("nickelate-IL");
    } else {
      motifs.push("nickelate-IL");
    }
  } else if (elements.includes("Ni") && elements.includes("O") && elements.some(e => isRareEarth(e))) {
    const nonNiO = elements.filter(e => e !== "Ni" && e !== "O");
    const reCount = nonNiO.filter(e => isRareEarth(e)).reduce((s, e) => s + (counts[e] || 0), 0);
    const ratioONi = niCount > 0 ? oCountNi / niCount : 0;
    if (ratioONi <= 2.5 && reCount > 0 && reCount <= niCount * 1.5) {
      motifs.push("nickelate-IL");
    } else if (oCountNi > niCount * 2.5) {
      motifs.push("Ruddlesden-Popper");
    }
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
    motifs.push("generic-metallic");
  }

  return motifs;
}

const VERIFICATION_STAGE_MULTIPLIERS: Record<string, number> = {
  dft: 5.0,
  "dft-external": 5.0,
  external: 4.0,
  xtb: 1.0,
  surrogate: 0.3,
};

const FAMILY_SUCCESS_TC_THRESHOLDS: Record<string, number> = {
  Hydride: 100,
  Cuprate: 50,
  Nickelate: 30,
  Pnictide: 20,
  Bismuthate: 10,
  Boride: 10,
  Chalcogenide: 5,
  Intermetallic: 5,
  HeavyFermion: 0.5,
  Organic: 2,
  Elemental: 2,
};

function getFamilySuccessThreshold(formula: string, materialClass: string): number {
  const family = classifyFamily(formula) || materialClass;
  return FAMILY_SUCCESS_TC_THRESHOLDS[family] ?? 20;
}

export function rewardStructuralMotif(
  formula: string,
  materialClass: string,
  result: { tc: number; stable: boolean; formationEnergy: number },
  verificationStage: string = "surrogate"
): void {
  initMotifRewards();
  const motifs = classifyStructuralMotif(formula, materialClass);
  const successThreshold = getFamilySuccessThreshold(formula, materialClass);
  const isSuccess = result.tc > successThreshold && result.stable && result.formationEnergy < 0.5;
  const stageMult = VERIFICATION_STAGE_MULTIPLIERS[verificationStage] ?? 1.0;

  for (const motif of motifs) {
    const entry = structuralMotifRewards.get(motif);
    if (!entry) continue;

    if (isSuccess) {
      const baseReward = Math.min(1.0, result.tc / 200);
      const reward = baseReward * stageMult;
      entry.successes++;
      entry.totalReward += reward;
      entry.weight = Math.min(3.0, entry.weight + reward * 0.15);
    } else {
      entry.failures++;
      entry.weight = Math.max(0.2, entry.weight - 0.05 * stageMult);
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

function extractMotifs(formula: string, preParsedCounts?: Record<string, number>): string[] {
  const counts = preParsedCounts ?? parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const motifs: string[] = elements.slice();
  const sorted = elements.slice().sort();
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
  materialClass?: string,
  verificationStage: string = "surrogate"
): void {
  compositionFeedback.formulaOutcomes.set(formula, result);
  const counts = parseFormulaCounts(formula);
  const motifs = extractMotifs(formula, counts);

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
  rewardStructuralMotif(formula, mc, result, verificationStage);
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

const STAGNATION_BOOST_FACTOR = 1.2;
const STAGNATION_MAX_RATE = 0.35;

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
        STAGNATION_MAX_RATE,
        adaptiveMutationState.currentRate * STAGNATION_BOOST_FACTOR
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

  const weights = pool.map(r => Math.max(0.001, r.fitnessScore));
  const totalFitness = weights.reduce((s, w) => s + w, 0);
  const pick = (): NovelSynthesisRoute => {
    let r = Math.random() * totalFitness;
    for (let i = 0; i < pool.length; i++) {
      r -= weights[i];
      if (r <= 0) return pool[i];
    }
    return pool[0];
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

function insertSorted(archive: NovelSynthesisRoute[], route: NovelSynthesisRoute): void {
  let insertAt = archive.length;
  for (let i = 0; i < archive.length; i++) {
    if (route.fitnessScore > archive[i].fitnessScore) {
      insertAt = i;
      break;
    }
  }
  archive.splice(insertAt, 0, route);
}

function updateEliteArchive(route: NovelSynthesisRoute): void {
  const normFormula = normalizeFormula(route.formula);
  const existingIdx = eliteArchive.findIndex(r => normalizeFormula(r.formula) === normFormula);
  if (existingIdx >= 0) {
    if (route.fitnessScore > eliteArchive[existingIdx].fitnessScore) {
      eliteArchive.splice(existingIdx, 1);
      insertSorted(eliteArchive, route);
    }
    return;
  }
  if (eliteArchive.length < ELITE_ARCHIVE_SIZE) {
    insertSorted(eliteArchive, route);
    return;
  }
  const worstArchive = eliteArchive[eliteArchive.length - 1];
  if (route.fitnessScore > worstArchive.fitnessScore) {
    eliteArchive.pop();
    insertSorted(eliteArchive, route);
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
  return Object.keys(parseFormulaCounts(formula));
}

function randRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function generateId(): string {
  return `synth-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

const MAX_PRESSURE_GPA = 350;

export function decodePressureGene(gene: number): number {
  return Math.max(0, Math.min(MAX_PRESSURE_GPA, gene * MAX_PRESSURE_GPA));
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

const GENOME_KEYS: readonly (keyof SynthesisGenome)[] = [
  "precursorStrategy", "reactionSequenceType", "temperatureProfile",
  "pressureProfile", "coolingStrategy", "atmosphereChoice",
  "annealingIntensity", "dopingLevel", "thermalCycleCount",
  "strainEngineering", "postProcessingType", "topologicalProtection",
  "pressureGene",
] as const;
const GENOME_KEY_COUNT = GENOME_KEYS.length;

function mutateGenome(g: SynthesisGenome, mutationRate: number = 0.2): SynthesisGenome {
  const result = { ...g };
  const numMutations = Math.max(1, Math.floor(GENOME_KEY_COUNT * mutationRate));

  for (let i = 0; i < numMutations; i++) {
    const key = GENOME_KEYS[Math.floor(Math.random() * GENOME_KEY_COUNT)];
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
  for (let i = 0; i < GENOME_KEY_COUNT; i++) {
    const key = GENOME_KEYS[i];
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
  let pressScale = Math.max(genome.pressureProfile * MAX_PRESSURE_GPA, genomePressureGpa);
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
      coolRate = Math.min(coolRate, 10);
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

const HYDROGEN_SAFE_ATMOSPHERES = ["hydrogen", "argon", "forming-gas", "vacuum"];
const OXIDIZING_ATMOSPHERES = new Set(["oxygen", "ammonia"]);

function selectAtmosphere(genome: SynthesisGenome, insights: MultiEngineInsights): string {
  const mc = insights.materialClass.toLowerCase();
  const counts = parseFormulaCounts(insights.formula);
  const totalAtoms = Object.values(counts).reduce((s, v) => s + v, 0);
  const hFraction = (counts["H"] || 0) / Math.max(1, totalAtoms);
  const isHydrogenRich = hFraction > 0.3 || mc.includes("hydride");

  if (mc.includes("cuprate")) return "oxygen";
  if (isHydrogenRich) return "hydrogen";
  if (mc.includes("nitride")) return "nitrogen";

  if (insights.pairing) {
    if (insights.pairing.dominantMechanism === "spin" && insights.pairing.pairingSymmetry.includes("d-wave")) {
      return "oxygen";
    }
  }

  const idx = Math.floor(genome.atmosphereChoice * ATMOSPHERE_OPTIONS.length);
  let choice = ATMOSPHERE_OPTIONS[Math.min(idx, ATMOSPHERE_OPTIONS.length - 1)];

  if (isHydrogenRich && OXIDIZING_ATMOSPHERES.has(choice)) {
    const safeIdx = Math.floor(genome.atmosphereChoice * HYDROGEN_SAFE_ATMOSPHERES.length);
    choice = HYDROGEN_SAFE_ATMOSPHERES[Math.min(safeIdx, HYDROGEN_SAFE_ATMOSPHERES.length - 1)];
  }

  return choice;
}

function stableOxideFormula(el: string): string {
  const states = VALENCE_OXIDATION_STATES[el];
  const posStates = states ? states.filter(s => s > 0) : [3];
  const ox = posStates.length > 0 ? posStates[0] : 3;

  const oCount = ox;
  const metalCount = 2;
  const g = gcd(metalCount, oCount);
  const m = metalCount / g;
  const o = oCount / g;
  return `${m > 1 ? el + m : el}O${o > 1 ? o : ""}`;
}

function stableHalideFormula(el: string): string {
  const states = VALENCE_OXIDATION_STATES[el];
  const posStates = states ? states.filter(s => s > 0) : [2];
  const ox = posStates.length > 0 ? posStates[0] : 2;
  return `${el}Cl${ox > 1 ? ox : ""}`;
}

function stableHydrideFormula(el: string): string {
  const states = VALENCE_OXIDATION_STATES[el];
  const posStates = states ? states.filter(s => s > 0) : [2];
  const ox = posStates.length > 0 ? posStates[0] : 2;
  return `${el}H${ox > 1 ? ox : ""}`;
}

function stableCarbonateFormula(el: string): string {
  const states = VALENCE_OXIDATION_STATES[el];
  const posStates = states ? states.filter(s => s > 0) : [2];
  const ox = posStates.length > 0 ? posStates[0] : 2;
  if (ox === 1) return `${el}2CO3`;
  if (ox === 2) return `${el}CO3`;
  if (ox === 3) return `${el}2(CO3)3`;
  return `${el}(CO3)${Math.floor(ox / 2)}`;
}

function gcd(a: number, b: number): number {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
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
        precursors.push(stableOxideFormula(el));
      }
      if (elements.includes("O")) precursors.push("O2 gas");
      break;
    case "binary-halide":
      for (const el of elements) {
        if (["F", "Cl", "Br", "H", "O"].includes(el)) continue;
        precursors.push(stableHalideFormula(el));
      }
      break;
    case "hydride-precursor":
      for (const el of elements) {
        if (el === "H") continue;
        const data = getElementData(el);
        if (data && (isTransitionMetal(el) || isRareEarth(el))) {
          precursors.push(stableHydrideFormula(el));
        } else {
          precursors.push(`${el} (elemental)`);
        }
      }
      if (!elements.includes("H")) precursors.push("H2 gas");
      break;
    case "carbonate":
      for (const el of elements) {
        if (["O", "C", "H", "N"].includes(el)) continue;
        precursors.push(stableCarbonateFormula(el));
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
    const clampedAnnealTemp = Math.min(annealTemp, sv.temperature * 0.8);
    steps.push({
      order: steps.length + 1,
      method: "anneal",
      temperature: clampedAnnealTemp,
      pressure: 0,
      coolingRate: 0,
      annealTemp: clampedAnnealTemp,
      duration: sv.annealTime,
      atmosphere: atmosphere,
      notes: `Annealing at ${clampedAnnealTemp.toFixed(0)}K for phase ordering and defect healing`,
    });
  }

  if (insights.topology && insights.topology.topologicalScore > 0.4) {
    const topoAnnealTemp = 300 + insights.topology.topologicalScore * 200;
    steps.push({
      order: steps.length + 1,
      method: "anneal",
      temperature: topoAnnealTemp,
      pressure: 0,
      coolingRate: Math.min(sv.coolingRate, 10),
      annealTemp: topoAnnealTemp,
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

interface PressureTcInterpolator {
  getTcAtPressure(pressureGpa: number): number | null;
}

function buildPressureTcInterpolator(formula: string): PressureTcInterpolator {
  let curvePoints: { pressure: number; Tc: number }[] = [];
  try {
    curvePoints = scanPressureTcCurve(formula, {
      min: 0,
      max: MAX_PRESSURE_GPA,
      steps: 36,
    });
    curvePoints.sort((a, b) => a.pressure - b.pressure);
  } catch {}

  return {
    getTcAtPressure(pressureGpa: number): number | null {
      if (curvePoints.length === 0) return null;
      let closest = curvePoints[0];
      let bestDist = Math.abs(closest.pressure - pressureGpa);
      for (let i = 1; i < curvePoints.length; i++) {
        const dist = Math.abs(curvePoints[i].pressure - pressureGpa);
        if (dist < bestDist) {
          bestDist = dist;
          closest = curvePoints[i];
        }
      }
      if (bestDist > 15) return null;

      let lo = -1, hi = -1;
      for (let i = 0; i < curvePoints.length - 1; i++) {
        if (curvePoints[i].pressure <= pressureGpa && curvePoints[i + 1].pressure >= pressureGpa) {
          lo = i; hi = i + 1; break;
        }
      }
      if (lo >= 0 && hi >= 0) {
        const pLo = curvePoints[lo], pHi = curvePoints[hi];
        const span = pHi.pressure - pLo.pressure;
        if (span < 0.01) return pLo.Tc;
        const t = (pressureGpa - pLo.pressure) / span;
        return pLo.Tc + t * (pHi.Tc - pLo.Tc);
      }
      return closest.Tc > 0 ? closest.Tc : null;
    },
  };
}

function computeFitnessScore(
  route: { steps: SynthesisStep[]; feasibilityScore: number; noveltyScore: number },
  insights: MultiEngineInsights,
  bestKnownTc: number = 0,
  genome?: SynthesisGenome,
  pressureInterpolator?: PressureTcInterpolator
): number {
  const genomePressureGpa = genome ? decodePressureGene(genome.pressureGene) : 0;

  let pressureTc = insights.predictedTc;
  if (genome && genomePressureGpa > 0 && pressureInterpolator) {
    const interpolated = pressureInterpolator.getTcAtPressure(genomePressureGpa);
    if (interpolated !== null && interpolated > 0) {
      pressureTc = interpolated;
    }
  }

  const adaptiveDenom = Math.max(20, bestKnownTc * 0.8);
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

  const pressurePenalty = genomePressureGpa / MAX_PRESSURE_GPA;

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

const VALENCE_OXIDATION_STATES: Record<string, number[]> = {
  H: [1, -1], Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
  Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
  B: [3], Al: [3], Ga: [3], In: [3], Tl: [1, 3],
  C: [-4, 4], Si: [4, -4], Ge: [4], Sn: [2, 4], Pb: [2, 4],
  N: [-3, 3, 5], P: [-3, 3, 5], As: [-3, 3, 5], Sb: [-3, 3, 5], Bi: [3, 5],
  O: [-2], S: [-2, 4, 6], Se: [-2, 4, 6], Te: [-2, 4, 6],
  F: [-1], Cl: [-1], Br: [-1], I: [-1],
  Ti: [2, 3, 4], V: [2, 3, 4, 5], Cr: [2, 3, 6], Mn: [2, 3, 4, 7],
  Fe: [2, 3], Co: [2, 3], Ni: [2, 3], Cu: [1, 2, 3], Zn: [2],
  Zr: [4], Nb: [3, 5], Mo: [4, 6], Ru: [3, 4], Rh: [3], Pd: [2, 4],
  Ag: [1], Hf: [4], Ta: [5], W: [4, 6], Re: [4, 7],
  Os: [4, 8], Ir: [3, 4], Pt: [2, 4], Au: [1, 3],
  Sc: [3], Y: [3], La: [3], Ce: [3, 4], Pr: [3], Nd: [3], Sm: [3],
  Eu: [2, 3], Gd: [3], Lu: [3], Th: [4], U: [3, 4, 5, 6],
};

function computeValenceSumError(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return 0;

  let bestError = Infinity;
  const statesPerEl = elements.map(el => VALENCE_OXIDATION_STATES[el] ?? [2, 3, -2]);

  function search(idx: number, runningSum: number): void {
    if (idx === elements.length) {
      bestError = Math.min(bestError, Math.abs(runningSum));
      return;
    }
    if (bestError === 0) return;
    for (const ox of statesPerEl[idx]) {
      search(idx + 1, runningSum + ox * counts[elements[idx]]);
    }
  }
  search(0, 0);
  return bestError === Infinity ? 2.0 : bestError;
}

function genomeToRoute(genome: SynthesisGenome, insights: MultiEngineInsights, bestKnownTc: number = 0, pressureInterpolator?: PressureTcInterpolator): NovelSynthesisRoute {
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

  const vse = computeValenceSumError(insights.formula);
  if (vse > 1.0) {
    feasibilityScore *= Math.max(0.1, 1 - (vse - 1.0) * 0.3);
  }

  feasibilityScore = Math.max(0, Math.min(1, feasibilityScore));
  const noveltyScore = computeNoveltyScore(genome, insights);

  const routeObj = { steps, feasibilityScore, noveltyScore };
  const fitnessScore = computeFitnessScore(routeObj, insights, bestKnownTc, genome, pressureInterpolator);

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

  const pressureInterp = buildPressureTcInterpolator(insights.formula);

  let population: NovelSynthesisRoute[] = [];

  for (const elite of eliteArchive) {
    if (population.length < Math.min(ELITE_ARCHIVE_SIZE, Math.floor(populationSize * 0.25))) {
      const mutated = mutateGenome(elite.genome, 0.1);
      population.push(genomeToRoute(mutated, insights, currentBestKnownTc, pressureInterp));
    }
  }

  const basePath = optimizeSynthesisPath(insights.formula, insights.materialClass, insights.predictedTc);
  const baseGenome = randomGenome();
  population.push(genomeToRoute(baseGenome, insights, currentBestKnownTc, pressureInterp));

  while (population.length < populationSize) {
    const genome = randomGenome();
    population.push(genomeToRoute(genome, insights, currentBestKnownTc, pressureInterp));
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

      newPop.push(genomeToRoute(childGenome, insights, currentBestKnownTc, pressureInterp));
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
