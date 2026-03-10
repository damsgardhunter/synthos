import {
  generateNovelCrystal, interpolateCrystals, getCrystalVAEStats,
  type GeneratedCrystal,
} from "./crystal-vae";
import {
  sampleStructures, getDiffusionModelStats,
  type GeneratedCrystalStructure,
} from "./crystal-diffusion-model";
import { getTrainingData } from "./crystal-structure-dataset";
import { computeCompositionFeatures } from "../learning/composition-features";
import {
  generatePrototypeFreeStructure, getLatticeGeneratorStats,
  type GeneratedStructure,
} from "./lattice-generator";

export type GenerationStrategy = "vae" | "diffusion" | "interpolation" | "hybrid" | "lattice_free";

export interface GeneratedCandidate {
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
  validationDetails: {
    chemicallyValid: boolean;
    geometricallyValid: boolean;
    stabilityPrescreen: boolean;
  };
}

interface GenerativeEngineStats {
  totalGenerated: number;
  totalAccepted: number;
  acceptanceRate: number;
  generationsByMethod: Record<string, number>;
  acceptanceByMethod: Record<string, number>;
  bestCandidates: GeneratedCandidate[];
  vaeStats: ReturnType<typeof getCrystalVAEStats> | null;
  diffusionStats: ReturnType<typeof getDiffusionModelStats> | null;
}

let totalGenerated = 0;
let totalAccepted = 0;
const generationsByMethod: Record<string, number> = {};
const acceptanceByMethod: Record<string, number> = {};
const bestCandidates: GeneratedCandidate[] = [];
const MAX_BEST = 20;

const VALID_ELEMENTS = new Set([
  "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
  "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
  "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
  "Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","U",
]);

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

function checkChemicalValidity(formula: string): boolean {
  if (!formula || formula.length < 1 || formula === "Unknown") return false;

  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  if (elements.length < 1 || elements.length > 6) return false;

  for (const el of elements) {
    if (!VALID_ELEMENTS.has(el)) return false;
    if (counts[el] <= 0 || !Number.isFinite(counts[el])) return false;
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms < 1 || totalAtoms > 30) return false;

  return true;
}

function checkGeometricValidity(lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number }, formula: string): boolean {
  if (lattice.a < 1.5 || lattice.b < 1.5 || lattice.c < 1.5) return false;
  if (lattice.a > 20 || lattice.b > 20 || lattice.c > 40) return false;

  if (lattice.alpha < 30 || lattice.alpha > 150) return false;
  if (lattice.beta < 30 || lattice.beta > 150) return false;
  if (lattice.gamma < 30 || lattice.gamma > 150) return false;

  const vol = lattice.a * lattice.b * lattice.c *
    Math.sqrt(1 - Math.cos(lattice.alpha * Math.PI / 180) ** 2
      - Math.cos(lattice.beta * Math.PI / 180) ** 2
      - Math.cos(lattice.gamma * Math.PI / 180) ** 2
      + 2 * Math.cos(lattice.alpha * Math.PI / 180) * Math.cos(lattice.beta * Math.PI / 180) * Math.cos(lattice.gamma * Math.PI / 180));

  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms <= 0) return false;

  const volPerAtom = vol / totalAtoms;
  if (volPerAtom < 5 || volPerAtom > 80) return false;

  return true;
}

function checkStabilityPrescreen(formula: string): boolean {
  try {
    const comp = computeCompositionFeatures(formula);
    const fE = (comp as any).formationEnergyEstimate;
    if (fE !== undefined && fE > 0.5) return false;
    return true;
  } catch {
    return true;
  }
}

function validateCandidate(formula: string, lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number }): {
  valid: boolean;
  chemicallyValid: boolean;
  geometricallyValid: boolean;
  stabilityPrescreen: boolean;
} {
  const chemicallyValid = checkChemicalValidity(formula);
  const geometricallyValid = chemicallyValid ? checkGeometricValidity(lattice, formula) : false;
  const stabilityPrescreen = chemicallyValid ? checkStabilityPrescreen(formula) : false;

  return {
    valid: chemicallyValid && geometricallyValid && stabilityPrescreen,
    chemicallyValid,
    geometricallyValid,
    stabilityPrescreen,
  };
}

function vaeToCandidate(crystal: GeneratedCrystal): GeneratedCandidate {
  const validation = validateCandidate(crystal.formula, crystal.lattice);
  return {
    formula: crystal.formula,
    lattice: crystal.lattice,
    crystalSystem: crystal.crystalSystem,
    spacegroup: crystal.spacegroup,
    spacegroupSymbol: crystal.spacegroupSymbol,
    predictedPrototype: crystal.prototype,
    noveltyScore: crystal.noveltyScore,
    generationMethod: crystal.generationMethod,
    confidence: Math.round((1 - crystal.noveltyScore * 0.3) * 100) / 100,
    valid: validation.valid,
    validationDetails: {
      chemicallyValid: validation.chemicallyValid,
      geometricallyValid: validation.geometricallyValid,
      stabilityPrescreen: validation.stabilityPrescreen,
    },
  };
}

function diffusionToCandidate(structure: GeneratedCrystalStructure): GeneratedCandidate {
  const validation = validateCandidate(structure.formula, structure.lattice);
  return {
    formula: structure.formula,
    lattice: structure.lattice,
    crystalSystem: structure.crystalSystem,
    spacegroup: null,
    spacegroupSymbol: null,
    predictedPrototype: null,
    noveltyScore: structure.noveltyScore,
    generationMethod: structure.generationMethod,
    confidence: structure.confidence,
    valid: validation.valid,
    validationDetails: {
      chemicallyValid: validation.chemicallyValid,
      geometricallyValid: validation.geometricallyValid,
      stabilityPrescreen: validation.stabilityPrescreen,
    },
  };
}

function trackCandidate(candidate: GeneratedCandidate, method: string): void {
  totalGenerated++;
  generationsByMethod[method] = (generationsByMethod[method] || 0) + 1;

  if (candidate.valid) {
    totalAccepted++;
    acceptanceByMethod[method] = (acceptanceByMethod[method] || 0) + 1;

    bestCandidates.push(candidate);
    bestCandidates.sort((a, b) => b.noveltyScore - a.noveltyScore);
    if (bestCandidates.length > MAX_BEST) {
      bestCandidates.length = MAX_BEST;
    }
  }
}

function generateVAE(count: number, targetSystem?: string): GeneratedCandidate[] {
  const results: GeneratedCandidate[] = [];
  for (let i = 0; i < count * 2 && results.length < count; i++) {
    const crystal = generateNovelCrystal(targetSystem);
    if (crystal) {
      const candidate = vaeToCandidate(crystal);
      trackCandidate(candidate, "vae");
      if (candidate.valid) results.push(candidate);
    }
  }
  return results;
}

function generateDiffusion(count: number, targetSystem?: string, elements?: string[]): GeneratedCandidate[] {
  const conditions = (targetSystem || elements) ? { crystalSystem: targetSystem, elements } : undefined;
  const samples = sampleStructures(count * 2, conditions);
  const results: GeneratedCandidate[] = [];
  for (const sample of samples) {
    const candidate = diffusionToCandidate(sample);
    trackCandidate(candidate, "diffusion");
    if (candidate.valid && results.length < count) results.push(candidate);
  }
  return results;
}

function generateInterpolation(count: number): GeneratedCandidate[] {
  const results: GeneratedCandidate[] = [];
  const trainingData = getTrainingData();
  if (trainingData.length < 2) return results;

  for (let i = 0; i < count * 3 && results.length < count; i++) {
    const idx1 = Math.floor(Math.random() * trainingData.length);
    let idx2 = Math.floor(Math.random() * trainingData.length);
    while (idx2 === idx1 && trainingData.length > 1) {
      idx2 = Math.floor(Math.random() * trainingData.length);
    }
    const alpha = 0.2 + Math.random() * 0.6;
    const crystal = interpolateCrystals(trainingData[idx1].formula, trainingData[idx2].formula, alpha);
    if (crystal) {
      const candidate = vaeToCandidate(crystal);
      trackCandidate(candidate, "interpolation");
      if (candidate.valid) results.push(candidate);
    }
  }
  return results;
}

function generateRandomLatent(count: number, targetSystem?: string): GeneratedCandidate[] {
  return generateVAE(count, targetSystem);
}

function latticeFreeToCandiate(structure: GeneratedStructure): GeneratedCandidate {
  const validation = validateCandidate(structure.formula, structure.lattice);
  return {
    formula: structure.formula,
    lattice: structure.lattice,
    crystalSystem: structure.bravaisType,
    spacegroup: null,
    spacegroupSymbol: null,
    predictedPrototype: null,
    noveltyScore: structure.noveltyScore,
    generationMethod: structure.generationMethod,
    confidence: structure.confidence,
    valid: validation.valid,
    validationDetails: {
      chemicallyValid: validation.chemicallyValid,
      geometricallyValid: validation.geometricallyValid,
      stabilityPrescreen: validation.stabilityPrescreen,
    },
  };
}

function generateLatticeFree(count: number, elements?: string[]): GeneratedCandidate[] {
  const results: GeneratedCandidate[] = [];
  const formulaPool = buildFormulaPool(elements);

  for (let i = 0; i < count * 3 && results.length < count; i++) {
    const formula = formulaPool[i % formulaPool.length];
    const structure = generatePrototypeFreeStructure(formula);
    if (structure) {
      const candidate = latticeFreeToCandiate(structure);
      trackCandidate(candidate, "lattice_free");
      if (candidate.valid) results.push(candidate);
    }
  }
  return results;
}

function buildFormulaPool(elements?: string[]): string[] {
  const pool: string[] = [];

  if (elements && elements.length >= 2) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const stoichs = [[1, 1], [1, 2], [2, 1], [1, 3], [3, 1], [2, 3]];
        for (const [a, b] of stoichs) {
          pool.push(`${elements[i]}${a > 1 ? a : ""}${elements[j]}${b > 1 ? b : ""}`);
        }
      }
    }
  }

  const sc_elements = ["Nb", "Ti", "V", "Zr", "Mo", "Ta", "W", "La", "Y", "Ca", "Sr", "Ba"];
  const anions = ["H", "N", "B", "C", "O", "S", "Se", "P", "As"];
  for (let i = 0; i < 30; i++) {
    const m1 = sc_elements[Math.floor(Math.random() * sc_elements.length)];
    const m2 = sc_elements[Math.floor(Math.random() * sc_elements.length)];
    const an = anions[Math.floor(Math.random() * anions.length)];
    if (m1 === m2) continue;
    const n1 = Math.floor(Math.random() * 3) + 1;
    const n2 = Math.floor(Math.random() * 3) + 1;
    const n3 = Math.floor(Math.random() * 6) + 1;
    pool.push(`${m1}${n1 > 1 ? n1 : ""}${m2}${n2 > 1 ? n2 : ""}${an}${n3 > 1 ? n3 : ""}`);
  }

  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }

  return pool;
}

export function generateCandidates(
  count: number = 10,
  strategy: GenerationStrategy = "hybrid",
  options?: { targetSystem?: string; elements?: string[] }
): GeneratedCandidate[] {
  const clampedCount = Math.min(Math.max(1, count), 100);
  const targetSystem = options?.targetSystem;
  const elements = options?.elements;

  switch (strategy) {
    case "vae":
      return generateVAE(clampedCount, targetSystem);

    case "diffusion":
      return generateDiffusion(clampedCount, targetSystem, elements);

    case "interpolation":
      return generateInterpolation(clampedCount);

    case "lattice_free":
      return generateLatticeFree(clampedCount, elements);

    case "hybrid": {
      const diffusionCount = Math.max(1, Math.round(clampedCount * 0.30));
      const vaeCount = Math.max(1, Math.round(clampedCount * 0.25));
      const lattFreeCount = Math.max(1, Math.round(clampedCount * 0.20));
      const interpCount = Math.max(1, Math.round(clampedCount * 0.15));
      const randomCount = Math.max(1, clampedCount - diffusionCount - vaeCount - lattFreeCount - interpCount);

      const results: GeneratedCandidate[] = [];
      results.push(...generateDiffusion(diffusionCount, targetSystem, elements));
      results.push(...generateVAE(vaeCount, targetSystem));
      results.push(...generateLatticeFree(lattFreeCount, elements));
      results.push(...generateInterpolation(interpCount));
      results.push(...generateRandomLatent(randomCount, targetSystem));

      return results.slice(0, clampedCount);
    }

    default:
      return generateVAE(clampedCount, targetSystem);
  }
}

export function isGenerativeReady(): boolean {
  try {
    const vaeStats = getCrystalVAEStats();
    if (vaeStats && vaeStats.trained) return true;
  } catch {}
  try {
    const diffStats = getDiffusionModelStats();
    if (diffStats && diffStats.trained) return true;
  } catch {}
  return false;
}

export function getGenerativeEngineStats(): GenerativeEngineStats {
  let vaeStats: ReturnType<typeof getCrystalVAEStats> | null = null;
  let diffusionStats: ReturnType<typeof getDiffusionModelStats> | null = null;
  let latticeStats: ReturnType<typeof getLatticeGeneratorStats> | null = null;

  try { vaeStats = getCrystalVAEStats(); } catch {}
  try { diffusionStats = getDiffusionModelStats(); } catch {}
  try { latticeStats = getLatticeGeneratorStats(); } catch {}

  return {
    totalGenerated,
    totalAccepted,
    acceptanceRate: totalGenerated > 0 ? Math.round((totalAccepted / totalGenerated) * 10000) / 10000 : 0,
    generationsByMethod: { ...generationsByMethod },
    acceptanceByMethod: { ...acceptanceByMethod },
    bestCandidates: bestCandidates.slice(0, 10),
    vaeStats,
    diffusionStats,
    latticeStats,
  };
}
