import {
  generateCandidates,
  isGenerativeReady,
  type GeneratedCandidate,
} from "./generative-crystal-engine";
import { getTrainingData } from "./crystal-structure-dataset";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";
import { scoreFormulaNovelty, addKnownFingerprint, computeStructureFingerprint } from "./structure-novelty-detector";
import { predictStabilityScreen } from "./stability-predictor";
import { sampleWeightedPrototype, getMotifGenerationWeights } from "./structure-reward-system";

export interface CrystalStructure {
  formula: string;
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  crystalSystem: string;
  spacegroup: number | null;
  spacegroupSymbol: string | null;
  predictedPrototype: string | null;
  noveltyScore: number;
  generationMethod: string;
  confidence: number;
  valid: boolean;
  predictedTc: number | null;
  mutationHistory: string[];
}

interface HybridGeneratorOptions {
  mutationRate?: number;
  mlWeight?: number;
  targetPressure?: number;
  targetSystem?: string;
}

interface MutationStats {
  applied: number;
  accepted: number;
}

const mutationStats: Record<string, MutationStats> = {
  latticeDistortion: { applied: 0, accepted: 0 },
  atomSwap: { applied: 0, accepted: 0 },
  pressureCompression: { applied: 0, accepted: 0 },
  vacancyInsertion: { applied: 0, accepted: 0 },
  crossover: { applied: 0, accepted: 0 },
};

let totalMLProposals = 0;
let totalMutationProposals = 0;
let bestTcFromML = 0;
let bestTcFromMutation = 0;
let totalGenerated = 0;

function parseFormulaCounts(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function countsToFormula(counts: Record<string, number>): string {
  const elements = Object.keys(counts).filter(el => counts[el] > 0).sort();
  let result = "";
  for (const el of elements) {
    const c = Math.round(counts[el]);
    if (c <= 0) continue;
    result += c === 1 ? el : `${el}${c}`;
  }
  return result || "Unknown";
}

export function mutateLatticeDistortion(
  structure: CrystalStructure,
  intensity: number = 0.1
): CrystalStructure {
  mutationStats.latticeDistortion.applied++;
  const clampedIntensity = Math.max(0.05, Math.min(0.15, intensity));

  const distortParam = (val: number) => {
    const factor = 1 + (Math.random() * 2 - 1) * clampedIntensity;
    return Math.max(1.5, val * factor);
  };

  const distortAngle = (val: number) => {
    const delta = (Math.random() * 2 - 1) * 10 * clampedIntensity;
    return Math.max(30, Math.min(150, val + delta));
  };

  const newLattice = {
    a: Math.round(distortParam(structure.lattice.a) * 100) / 100,
    b: Math.round(distortParam(structure.lattice.b) * 100) / 100,
    c: Math.round(distortParam(structure.lattice.c) * 100) / 100,
    alpha: Math.round(distortAngle(structure.lattice.alpha) * 10) / 10,
    beta: Math.round(distortAngle(structure.lattice.beta) * 10) / 10,
    gamma: Math.round(distortAngle(structure.lattice.gamma) * 10) / 10,
  };

  return {
    ...structure,
    lattice: newLattice,
    generationMethod: `${structure.generationMethod}+latticeDistortion`,
    mutationHistory: [...structure.mutationHistory, `latticeDistortion(${clampedIntensity.toFixed(2)})`],
    predictedTc: null,
  };
}

export function mutateAtomSwap(structure: CrystalStructure): CrystalStructure {
  mutationStats.atomSwap.applied++;
  const counts = parseFormulaCounts(structure.formula);
  const elements = Object.keys(counts);

  if (elements.length < 2) {
    return { ...structure, mutationHistory: [...structure.mutationHistory, "atomSwap(skipped-single-element)"] };
  }

  const idx1 = Math.floor(Math.random() * elements.length);
  let idx2 = Math.floor(Math.random() * elements.length);
  while (idx2 === idx1 && elements.length > 1) {
    idx2 = Math.floor(Math.random() * elements.length);
  }

  const el1 = elements[idx1];
  const el2 = elements[idx2];
  const newCounts = { ...counts };
  const temp = newCounts[el1];
  newCounts[el1] = newCounts[el2];
  newCounts[el2] = temp;

  const newFormula = countsToFormula(newCounts);

  return {
    ...structure,
    formula: newFormula,
    generationMethod: `${structure.generationMethod}+atomSwap`,
    mutationHistory: [...structure.mutationHistory, `atomSwap(${el1}<->${el2})`],
    predictedTc: null,
  };
}

export function mutatePressureCompression(
  structure: CrystalStructure,
  targetPressure: number = 10
): CrystalStructure {
  mutationStats.pressureCompression.applied++;
  const clampedPressure = Math.max(0, Math.min(300, targetPressure));

  const B0 = 100;
  const B0_prime = 4.0;
  const eta = Math.pow(1 + (B0_prime / B0) * clampedPressure, -1 / (3 * B0_prime));
  const compressionFactor = Math.max(0.7, Math.min(1.0, eta));

  const newLattice = {
    a: Math.round(structure.lattice.a * compressionFactor * 100) / 100,
    b: Math.round(structure.lattice.b * compressionFactor * 100) / 100,
    c: Math.round(structure.lattice.c * compressionFactor * 100) / 100,
    alpha: structure.lattice.alpha,
    beta: structure.lattice.beta,
    gamma: structure.lattice.gamma,
  };

  if (newLattice.a < 1.5) newLattice.a = 1.5;
  if (newLattice.b < 1.5) newLattice.b = 1.5;
  if (newLattice.c < 1.5) newLattice.c = 1.5;

  return {
    ...structure,
    lattice: newLattice,
    generationMethod: `${structure.generationMethod}+pressureCompression`,
    mutationHistory: [...structure.mutationHistory, `pressureCompression(${clampedPressure}GPa,eta=${compressionFactor.toFixed(3)})`],
    predictedTc: null,
  };
}

export function mutateVacancyInsertion(structure: CrystalStructure): CrystalStructure {
  mutationStats.vacancyInsertion.applied++;
  const counts = parseFormulaCounts(structure.formula);
  const elements = Object.keys(counts);

  const removableElements = elements.filter(el => counts[el] > 1);
  if (removableElements.length === 0) {
    return { ...structure, mutationHistory: [...structure.mutationHistory, "vacancyInsertion(skipped-no-removable)"] };
  }

  const targetEl = removableElements[Math.floor(Math.random() * removableElements.length)];
  const newCounts = { ...counts };
  newCounts[targetEl] = Math.max(1, newCounts[targetEl] - 1);

  const totalOld = Object.values(counts).reduce((s, n) => s + n, 0);
  const totalNew = Object.values(newCounts).reduce((s, n) => s + n, 0);
  const scaleFactor = Math.pow(totalNew / totalOld, 1 / 3);

  const newLattice = {
    a: Math.round(structure.lattice.a * scaleFactor * 100) / 100,
    b: Math.round(structure.lattice.b * scaleFactor * 100) / 100,
    c: Math.round(structure.lattice.c * scaleFactor * 100) / 100,
    alpha: structure.lattice.alpha,
    beta: structure.lattice.beta,
    gamma: structure.lattice.gamma,
  };

  const newFormula = countsToFormula(newCounts);

  return {
    ...structure,
    formula: newFormula,
    lattice: newLattice,
    generationMethod: `${structure.generationMethod}+vacancyInsertion`,
    mutationHistory: [...structure.mutationHistory, `vacancyInsertion(removed ${targetEl})`],
    predictedTc: null,
  };
}

export function crossoverStructures(
  parent1: CrystalStructure,
  parent2: CrystalStructure
): CrystalStructure {
  mutationStats.crossover.applied++;

  const useLatticeFrom = Math.random() < 0.5 ? parent1 : parent2;
  const useCompositionFrom = useLatticeFrom === parent1 ? parent2 : parent1;

  const alpha = 0.3 + Math.random() * 0.4;
  const blendedLattice = {
    a: Math.round((parent1.lattice.a * alpha + parent2.lattice.a * (1 - alpha)) * 100) / 100,
    b: Math.round((parent1.lattice.b * alpha + parent2.lattice.b * (1 - alpha)) * 100) / 100,
    c: Math.round((parent1.lattice.c * alpha + parent2.lattice.c * (1 - alpha)) * 100) / 100,
    alpha: Math.round((parent1.lattice.alpha * alpha + parent2.lattice.alpha * (1 - alpha)) * 10) / 10,
    beta: Math.round((parent1.lattice.beta * alpha + parent2.lattice.beta * (1 - alpha)) * 10) / 10,
    gamma: Math.round((parent1.lattice.gamma * alpha + parent2.lattice.gamma * (1 - alpha)) * 10) / 10,
  };

  const useBlended = Math.random() < 0.5;

  return {
    formula: useCompositionFrom.formula,
    lattice: useBlended ? blendedLattice : useLatticeFrom.lattice,
    crystalSystem: useLatticeFrom.crystalSystem,
    spacegroup: useLatticeFrom.spacegroup,
    spacegroupSymbol: useLatticeFrom.spacegroupSymbol,
    predictedPrototype: useLatticeFrom.predictedPrototype,
    noveltyScore: (parent1.noveltyScore + parent2.noveltyScore) / 2,
    generationMethod: `crossover(${parent1.generationMethod},${parent2.generationMethod})`,
    confidence: Math.min(parent1.confidence, parent2.confidence) * 0.9,
    valid: true,
    predictedTc: null,
    mutationHistory: [`crossover(${parent1.formula}x${parent2.formula},alpha=${alpha.toFixed(2)}${useBlended ? ",blended" : ",direct"})`],
  };
}

function hasValidLattice(lattice: any): boolean {
  if (!lattice) return false;
  const { a, b, c, alpha, beta, gamma } = lattice;
  return (
    typeof a === "number" && !isNaN(a) && a > 0 &&
    typeof b === "number" && !isNaN(b) && b > 0 &&
    typeof c === "number" && !isNaN(c) && c > 0 &&
    typeof alpha === "number" && !isNaN(alpha) && alpha > 0 &&
    typeof beta === "number" && !isNaN(beta) && beta > 0 &&
    typeof gamma === "number" && !isNaN(gamma) && gamma > 0
  );
}

const DEFAULT_LATTICE_BY_SYSTEM: Record<string, { a: number; b: number; c: number; alpha: number; beta: number; gamma: number }> = {
  cubic: { a: 5.0, b: 5.0, c: 5.0, alpha: 90, beta: 90, gamma: 90 },
  tetragonal: { a: 4.5, b: 4.5, c: 6.0, alpha: 90, beta: 90, gamma: 90 },
  hexagonal: { a: 3.5, b: 3.5, c: 5.5, alpha: 90, beta: 90, gamma: 120 },
  orthorhombic: { a: 4.0, b: 5.0, c: 6.0, alpha: 90, beta: 90, gamma: 90 },
  rhombohedral: { a: 5.0, b: 5.0, c: 5.0, alpha: 80, beta: 80, gamma: 80 },
  monoclinic: { a: 5.0, b: 5.5, c: 6.0, alpha: 90, beta: 100, gamma: 90 },
  triclinic: { a: 5.0, b: 5.5, c: 6.0, alpha: 85, beta: 95, gamma: 80 },
};

function candidateToStructure(candidate: GeneratedCandidate): CrystalStructure {
  let lattice = candidate.lattice;
  if (!hasValidLattice(lattice)) {
    const sys = (candidate.crystalSystem || "cubic").toLowerCase();
    lattice = { ...(DEFAULT_LATTICE_BY_SYSTEM[sys] || DEFAULT_LATTICE_BY_SYSTEM.cubic) };
    const jitter = () => 1 + (Math.random() - 0.5) * 0.1;
    lattice.a = Math.round(lattice.a * jitter() * 100) / 100;
    lattice.b = Math.round(lattice.b * jitter() * 100) / 100;
    lattice.c = Math.round(lattice.c * jitter() * 100) / 100;
  }
  return {
    formula: candidate.formula,
    lattice: { ...lattice },
    crystalSystem: candidate.crystalSystem,
    spacegroup: candidate.spacegroup,
    spacegroupSymbol: candidate.spacegroupSymbol,
    predictedPrototype: candidate.predictedPrototype,
    noveltyScore: candidate.noveltyScore,
    generationMethod: candidate.generationMethod,
    confidence: candidate.confidence,
    valid: candidate.valid,
    predictedTc: null,
    mutationHistory: [],
  };
}

async function scoreTcSurrogate(structure: CrystalStructure): Promise<number> {
  try {
    const features = await extractFeatures(structure.formula);
    const result = await gbPredict(features, structure.formula);
    return Math.max(0, result.tcPredicted);
  } catch {
    return 0;
  }
}

function checkGeometricValidity(lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number }, formula: string): boolean {
  if (lattice.a < 1.5 || lattice.b < 1.5 || lattice.c < 1.5) return false;
  if (lattice.a > 20 || lattice.b > 20 || lattice.c > 40) return false;
  if (lattice.alpha < 30 || lattice.alpha > 150) return false;
  if (lattice.beta < 30 || lattice.beta > 150) return false;
  if (lattice.gamma < 30 || lattice.gamma > 150) return false;

  const vol = lattice.a * lattice.b * lattice.c *
    Math.sqrt(Math.max(0,
      1 - Math.cos(lattice.alpha * Math.PI / 180) ** 2
      - Math.cos(lattice.beta * Math.PI / 180) ** 2
      - Math.cos(lattice.gamma * Math.PI / 180) ** 2
      + 2 * Math.cos(lattice.alpha * Math.PI / 180) * Math.cos(lattice.beta * Math.PI / 180) * Math.cos(lattice.gamma * Math.PI / 180)));

  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms <= 0) return false;

  const volPerAtom = vol / totalAtoms;
  if (volPerAtom < 5 || volPerAtom > 80) return false;

  return true;
}

async function checkFailureDB(formula: string, lattice?: any, system?: string): Promise<boolean> {
  try {
    const failureDB = await import("./structure-failure-db");
    if (failureDB.shouldAvoidStructure) {
      return failureDB.shouldAvoidStructure(formula, lattice, system);
    }
  } catch {
  }
  return false;
}

function applyRandomMutations(
  structure: CrystalStructure,
  count: number,
  targetPressure?: number
): CrystalStructure {
  let mutated = structure;
  const mutationFns = [
    () => mutateLatticeDistortion(mutated, 0.05 + Math.random() * 0.10),
    () => mutateAtomSwap(mutated),
    () => mutatePressureCompression(mutated, targetPressure ?? (Math.random() * 50)),
    () => mutateVacancyInsertion(mutated),
  ];

  for (let i = 0; i < count; i++) {
    const fnIdx = Math.floor(Math.random() * mutationFns.length);
    mutated = mutationFns[fnIdx]();
  }

  return mutated;
}

function generateWithTimeout(mlCount: number, targetSystem?: string): GeneratedCandidate[] {
  const TIMEOUT_MS = 5000;
  const startTime = Date.now();

  if (!isGenerativeReady()) {
    console.log("[HybridGenerator] Generative engine not ready, skipping ML proposals");
    return [];
  }

  try {
    const candidates = generateCandidates(mlCount, "hybrid", { targetSystem });
    const elapsed = Date.now() - startTime;
    if (elapsed > TIMEOUT_MS) {
      console.warn(`[HybridGenerator] WARNING: generateCandidates took ${elapsed}ms (>${TIMEOUT_MS}ms timeout)`);
    }
    return candidates;
  } catch (err) {
    console.warn(`[HybridGenerator] generateCandidates failed: ${err}`);
    return [];
  }
}

function generateSeedBasedMutations(count: number, targetPressure?: number, targetSystem?: string): CrystalStructure[] {
  const seedData = getTrainingData();
  if (seedData.length === 0) return [];

  const results: CrystalStructure[] = [];
  const weights = getMotifGenerationWeights();
  let filtered = targetSystem
    ? seedData.filter(s => s.crystalSystem.toLowerCase() === targetSystem.toLowerCase())
    : seedData;

  if (weights.size > 0 && Math.random() < 0.5) {
    const motif = sampleWeightedPrototype();
    if (motif) {
      const motifFiltered = seedData.filter(s =>
        s.prototype.toLowerCase() === motif.prototype &&
        s.crystalSystem.toLowerCase() === motif.crystalSystem
      );
      if (motifFiltered.length > 0) {
        filtered = motifFiltered;
      }
    }
  }

  const pool = filtered.length > 0 ? filtered : seedData;

  for (let i = 0; i < count * 3 && results.length < count; i++) {
    const seed = pool[Math.floor(Math.random() * pool.length)];
    const structure: CrystalStructure = {
      formula: seed.formula,
      lattice: { ...seed.lattice },
      crystalSystem: seed.crystalSystem,
      spacegroup: seed.spacegroup,
      spacegroupSymbol: seed.spacegroupSymbol,
      predictedPrototype: seed.prototype,
      noveltyScore: 0.5,
      generationMethod: "seed-mutation",
      confidence: 0.6,
      valid: true,
      predictedTc: null,
      mutationHistory: [],
    };

    const numMutations = 1 + Math.floor(Math.random() * 3);
    const mutated = applyRandomMutations(structure, numMutations, targetPressure);

    if (checkGeometricValidity(mutated.lattice, mutated.formula)) {
      results.push(mutated);
    }
  }

  return results;
}

export async function generateHybridCandidates(
  count: number = 10,
  options?: HybridGeneratorOptions
): Promise<CrystalStructure[]> {
  const clampedCount = Math.min(Math.max(1, count), 100);
  const mutationRate = options?.mutationRate ?? 0.7;
  const mlWeight = options?.mlWeight ?? 0.6;
  const targetPressure = options?.targetPressure;
  const targetSystem = options?.targetSystem;

  const mlCount = Math.max(1, Math.ceil(clampedCount * mlWeight * 2));
  const mlCandidates = generateWithTimeout(mlCount, targetSystem);
  totalMLProposals += mlCandidates.length;

  let mlStructures = mlCandidates
    .filter(c => c.formula && c.formula.length > 0)
    .map(c => candidateToStructure(c))
    .filter(s => hasValidLattice(s.lattice));

  if (mlStructures.length === 0) {
    console.log("[HybridGenerator] ML proposals empty, falling back to seed-based mutations");
    const seedMutations = generateSeedBasedMutations(clampedCount, targetPressure, targetSystem);
    mlStructures = seedMutations;
  }

  const allCandidates: CrystalStructure[] = [];

  for (const structure of mlStructures) {
    if (Math.random() < mutationRate) {
      const numMutations = 1 + Math.floor(Math.random() * 3);
      const mutated = applyRandomMutations(structure, numMutations, targetPressure);
      totalMutationProposals++;

      if (checkGeometricValidity(mutated.lattice, mutated.formula)) {
        const shouldAvoid = await checkFailureDB(mutated.formula, mutated.lattice, mutated.crystalSystem);
        if (!shouldAvoid) {
          const stability = await predictStabilityScreen(mutated.formula);
          if (!stability.isLikelyStable) continue;
          const tc = await scoreTcSurrogate(mutated);
          mutated.predictedTc = tc;
          mutated.valid = true;
          allCandidates.push(mutated);

          if (tc > bestTcFromMutation) bestTcFromMutation = tc;
          for (const key of Object.keys(mutationStats)) {
            if (mutated.mutationHistory.some(h => h.startsWith(key))) {
              mutationStats[key].accepted++;
            }
          }
        }
      }
    } else {
      const stability = await predictStabilityScreen(structure.formula);
      if (!stability.isLikelyStable) continue;
      const tc = await scoreTcSurrogate(structure);
      structure.predictedTc = tc;
      allCandidates.push(structure);
      if (tc > bestTcFromML) bestTcFromML = tc;
    }
  }

  if (mlStructures.length >= 2) {
    const crossoverCount = Math.max(1, Math.floor(clampedCount * 0.2));
    for (let i = 0; i < crossoverCount && allCandidates.length < clampedCount * 2; i++) {
      const p1 = mlStructures[Math.floor(Math.random() * mlStructures.length)];
      const p2 = mlStructures[Math.floor(Math.random() * mlStructures.length)];
      if (p1.formula === p2.formula) continue;
      if (!hasValidLattice(p1.lattice) || !hasValidLattice(p2.lattice)) continue;

      const child = crossoverStructures(p1, p2);
      if (hasValidLattice(child.lattice) && checkGeometricValidity(child.lattice, child.formula)) {
        const shouldAvoid = await checkFailureDB(child.formula, child.lattice, child.crystalSystem);
        if (!shouldAvoid) {
          const tc = await scoreTcSurrogate(child);
          child.predictedTc = tc;
          allCandidates.push(child);
          mutationStats.crossover.accepted++;
          if (tc > bestTcFromMutation) bestTcFromMutation = tc;
        }
      }
    }
  }

  for (const candidate of allCandidates) {
    try {
      const noveltyResult = scoreFormulaNovelty(candidate.formula);
      candidate.noveltyScore = noveltyResult.noveltyScore;
    } catch {
      candidate.noveltyScore = 0.5;
    }
  }

  allCandidates.sort((a, b) => {
    const tcA = a.predictedTc ?? 0;
    const tcB = b.predictedTc ?? 0;
    const noveltyA = a.noveltyScore;
    const noveltyB = b.noveltyScore;
    return (tcB + noveltyB * 20) - (tcA + noveltyA * 20);
  });

  const seen = new Set<string>();
  const deduped: CrystalStructure[] = [];
  for (const c of allCandidates) {
    if (!seen.has(c.formula)) {
      seen.add(c.formula);
      deduped.push(c);
    }
    if (deduped.length >= clampedCount) break;
  }

  for (const c of deduped) {
    try {
      const fp = computeStructureFingerprint(c.formula);
      addKnownFingerprint(c.formula, fp);
    } catch {}
  }

  totalGenerated += deduped.length;

  return deduped;
}

export function getHybridGeneratorStats(): {
  totalGenerated: number;
  totalMLProposals: number;
  totalMutationProposals: number;
  bestTcFromML: number;
  bestTcFromMutation: number;
  mutationAcceptanceRates: Record<string, { applied: number; accepted: number; rate: number }>;
  mlVsMutationContribution: { ml: number; mutation: number };
} {
  const mutationAcceptanceRates: Record<string, { applied: number; accepted: number; rate: number }> = {};
  for (const [key, stats] of Object.entries(mutationStats)) {
    mutationAcceptanceRates[key] = {
      applied: stats.applied,
      accepted: stats.accepted,
      rate: stats.applied > 0 ? Math.round((stats.accepted / stats.applied) * 10000) / 10000 : 0,
    };
  }

  const totalContrib = totalMLProposals + totalMutationProposals;

  return {
    totalGenerated,
    totalMLProposals,
    totalMutationProposals,
    bestTcFromML: Math.round(bestTcFromML * 10) / 10,
    bestTcFromMutation: Math.round(bestTcFromMutation * 10) / 10,
    mutationAcceptanceRates,
    mlVsMutationContribution: {
      ml: totalContrib > 0 ? Math.round((totalMLProposals / totalContrib) * 10000) / 10000 : 0,
      mutation: totalContrib > 0 ? Math.round((totalMutationProposals / totalContrib) * 10000) / 10000 : 0,
    },
  };
}
