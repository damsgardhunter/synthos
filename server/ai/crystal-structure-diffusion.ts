import { SC_MOTIF_LIBRARY, type StructuralMotif, type ChemicalFamily, selectFamilyForMotif } from "./structure-diffusion";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict, surrogateScreen } from "../learning/gradient-boost";
import { isValidFormula, normalizeFormula } from "../learning/utils";
import { passesValenceFilter } from "../learning/candidate-generator";
import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
} from "../learning/physics-engine";
import { getElementData, isTransitionMetal, isRareEarth } from "../learning/elemental-data";
import {
  sampleCrystalSystem, sampleSpaceGroup, sampleLatticeParams,
  sampleWyckoffPositions, getDistributionStats, getAllDistributions,
  type CrystalSystemDistribution, type WyckoffSite,
} from "./crystal-distribution-db";

interface LatentVector {
  composition: number[];
  structure: number[];
  symmetry: number[];
}

interface CrystalAtom {
  element: string;
  fx: number;
  fy: number;
  fz: number;
  occupancy: number;
}

export interface DiffusedCrystal {
  formula: string;
  atoms: CrystalAtom[];
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  spaceGroup: string;
  crystalSystem: string;
  motif: string;
  predictedTc: number;
  stabilityScore: number;
  noveltyScore: number;
  diffusionSteps: number;
  lambda: number;
  latentNorm: number;
}

const ELEMENT_EMBEDDINGS: Record<string, number[]> = {
  H: [1, 2.2, 0.53, 1, 0, 0], Li: [3, 0.98, 1.67, 1, 0, 0], Be: [4, 1.57, 1.12, 2, 0, 0],
  B: [5, 2.04, 0.87, 3, 0, 0], C: [6, 2.55, 0.77, 4, 0, 0], N: [7, 3.04, 0.75, 5, 0, 0],
  O: [8, 3.44, 0.73, 6, 0, 0], F: [9, 3.98, 0.71, 7, 0, 0],
  Na: [11, 0.93, 1.90, 1, 0, 0], Mg: [12, 1.31, 1.45, 2, 0, 0],
  Al: [13, 1.61, 1.18, 3, 0, 0], Si: [14, 1.90, 1.11, 4, 0, 0], P: [15, 2.19, 1.06, 5, 0, 0],
  S: [16, 2.58, 1.02, 6, 0, 0], Cl: [17, 3.16, 0.99, 7, 0, 0],
  K: [19, 0.82, 2.43, 1, 0, 0], Ca: [20, 1.00, 1.94, 2, 0, 0],
  Sc: [21, 1.36, 1.84, 2, 1, 0], Ti: [22, 1.54, 1.76, 2, 2, 0], V: [23, 1.63, 1.71, 2, 3, 0],
  Cr: [24, 1.66, 1.66, 1, 5, 0], Mn: [25, 1.55, 1.61, 2, 5, 0], Fe: [26, 1.83, 1.56, 2, 6, 0],
  Co: [27, 1.88, 1.52, 2, 7, 0], Ni: [28, 1.91, 1.49, 2, 8, 0], Cu: [29, 1.90, 1.45, 1, 10, 0],
  Zn: [30, 1.65, 1.42, 2, 10, 0], Ga: [31, 1.81, 1.36, 3, 10, 0], Ge: [32, 2.01, 1.25, 4, 10, 0],
  As: [33, 2.18, 1.14, 5, 10, 0], Se: [34, 2.55, 1.03, 6, 10, 0],
  Sr: [38, 0.95, 2.19, 2, 0, 0], Y: [39, 1.22, 2.12, 2, 1, 0],
  Zr: [40, 1.33, 2.06, 2, 2, 0], Nb: [41, 1.60, 1.98, 1, 4, 0], Mo: [42, 2.16, 1.90, 1, 5, 0],
  Ru: [44, 2.20, 1.78, 1, 7, 0], Rh: [45, 2.28, 1.73, 1, 8, 0], Pd: [46, 2.20, 1.69, 0, 10, 0],
  Ag: [47, 1.93, 1.65, 1, 10, 0], In: [49, 1.78, 1.55, 3, 10, 0], Sn: [50, 1.96, 1.45, 4, 10, 0],
  Sb: [51, 2.05, 1.33, 5, 10, 0], Te: [52, 2.10, 1.23, 6, 10, 0],
  Ba: [56, 0.89, 2.53, 2, 0, 0], La: [57, 1.10, 2.47, 2, 1, 1],
  Ce: [58, 1.12, 2.42, 2, 1, 1], Pr: [59, 1.13, 2.40, 2, 0, 3],
  Nd: [60, 1.14, 2.39, 2, 0, 4], Gd: [64, 1.20, 2.33, 2, 1, 7],
  Hf: [72, 1.30, 2.08, 2, 2, 0], Ta: [73, 1.50, 2.00, 2, 3, 0], W: [74, 2.36, 1.93, 2, 4, 0],
  Re: [75, 1.90, 1.88, 2, 5, 0], Os: [76, 2.20, 1.85, 2, 6, 0],
  Ir: [77, 2.20, 1.80, 2, 7, 0], Pt: [78, 2.28, 1.77, 1, 9, 0],
  Au: [79, 2.54, 1.74, 1, 10, 0], Pb: [82, 2.33, 1.54, 4, 10, 0], Bi: [83, 2.02, 1.43, 5, 10, 0],
  Th: [90, 1.30, 2.40, 2, 0, 2],
};

const CRYSTAL_SYSTEMS: Record<string, { alpha: number; beta: number; gamma: number }> = {
  cubic: { alpha: 90, beta: 90, gamma: 90 },
  tetragonal: { alpha: 90, beta: 90, gamma: 90 },
  orthorhombic: { alpha: 90, beta: 90, gamma: 90 },
  hexagonal: { alpha: 90, beta: 90, gamma: 120 },
  trigonal: { alpha: 90, beta: 90, gamma: 120 },
  monoclinic: { alpha: 90, beta: 100, gamma: 90 },
};

function gaussRandom(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
}

function softmax(values: number[], temperature: number = 1.0): number[] {
  const maxVal = Math.max(...values);
  const exps = values.map(v => Math.exp((v - maxVal) / temperature));
  const sum = exps.reduce((s, e) => s + e, 0);
  return exps.map(e => e / sum);
}

function sampleFromDistribution(probs: number[]): number {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r <= cumulative) return i;
  }
  return probs.length - 1;
}

function encodeComposition(elements: Record<string, number>): number[] {
  const vec = new Array(12).fill(0);
  const entries = Object.entries(elements);
  const totalAtoms = entries.reduce((s, [, n]) => s + n, 0);

  let avgZ = 0, avgEN = 0, avgR = 0, maxVE = 0, totalD = 0;
  let hFraction = 0, tmFraction = 0, reFraction = 0;

  for (const [el, count] of entries) {
    const emb = ELEMENT_EMBEDDINGS[el];
    if (!emb) continue;
    const frac = count / totalAtoms;
    avgZ += emb[0] * frac;
    avgEN += emb[1] * frac;
    avgR += emb[2] * frac;
    maxVE = Math.max(maxVE, emb[3] + emb[4]);
    totalD += emb[4] * frac;
    if (el === "H") hFraction = frac;
    if (isTransitionMetal(el)) tmFraction += frac;
    if (isRareEarth(el)) reFraction += frac;
  }

  vec[0] = avgZ / 50;
  vec[1] = avgEN / 3;
  vec[2] = avgR / 2;
  vec[3] = maxVE / 12;
  vec[4] = totalD / 10;
  vec[5] = hFraction;
  vec[6] = tmFraction;
  vec[7] = reFraction;
  vec[8] = entries.length / 5;
  vec[9] = totalAtoms / 20;
  vec[10] = Math.log(totalAtoms + 1) / 3;
  vec[11] = entries.reduce((s, [el, n]) => s + (ELEMENT_EMBEDDINGS[el]?.[0] ?? 20) * n, 0) / (totalAtoms * 50);

  return vec;
}

function encodeStructure(motif: StructuralMotif): number[] {
  const e = motif.embedding;
  return [
    e.coordination / 12,
    e.avgBondAngle / 180,
    e.layerSpacing / 10,
    e.cOverA / 5,
    e.symmetryOrder / 200,
    e.electronCountPerSite / 20,
    e.dimensionality / 3,
    e.cageFraction,
    e.interstitialFraction,
    e.connectivityIndex,
    e.anisotropy,
    e.voidFraction,
  ];
}

function encodeSymmetry(spaceGroup: string, crystalSystem: string): number[] {
  const sgNum = parseInt(spaceGroup.match(/\d+/)?.[0] ?? "1");
  const systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"];
  const sysIdx = systems.indexOf(crystalSystem);
  return [
    sgNum / 230,
    sysIdx >= 0 ? sysIdx / 6 : 0.5,
    sgNum > 140 ? 1 : 0,
    spaceGroup.includes("m") ? 1 : 0,
    crystalSystem === "cubic" || crystalSystem === "hexagonal" ? 1 : 0,
    spaceGroup.includes("4") ? 1 : 0,
  ];
}

function generateLatentVector(motif: StructuralMotif, targetTc: number): LatentVector {
  const family = selectFamilyForMotif(motif.name);

  const structEmb = encodeStructure(motif);
  const symEmb = encodeSymmetry(motif.spaceGroup, motif.crystalSystem);

  const templateComp: Record<string, number> = {};
  if (family) {
    const host = family.hostElements[Math.floor(Math.random() * family.hostElements.length)];
    templateComp[host] = 1;
    if (family.allowedAnions.length > 0) {
      const anion = family.allowedAnions[Math.floor(Math.random() * family.allowedAnions.length)];
      templateComp[anion] = 2;
    }
  } else {
    templateComp["Nb"] = 3;
    templateComp["Si"] = 1;
  }

  const compEmb = encodeComposition(templateComp);

  const tcBias = targetTc / 300;
  for (let i = 0; i < compEmb.length; i++) {
    compEmb[i] += gaussRandom() * 0.15 * (1 + tcBias * 0.5);
  }

  return { composition: compEmb, structure: structEmb, symmetry: symEmb };
}

function decodeComposition(
  latent: LatentVector,
  motif: StructuralMotif
): Record<string, number> {
  const family = selectFamilyForMotif(motif.name);

  const elementPool: string[] = [];
  if (family) {
    elementPool.push(...family.hostElements);
    elementPool.push(...family.allowedAnions);
  } else {
    elementPool.push("Nb", "V", "Ti", "Ta", "Mo", "La", "Y", "B", "C", "N", "H", "O", "Si", "Ge");
  }

  const uniquePool = Array.from(new Set(elementPool));

  const scores: number[] = uniquePool.map(el => {
    const emb = ELEMENT_EMBEDDINGS[el];
    if (!emb) return -10;
    let score = 0;
    score += (1 - Math.abs(emb[0] / 50 - latent.composition[0])) * 2;
    score += (1 - Math.abs(emb[1] / 3 - latent.composition[1])) * 1.5;
    score += (1 - Math.abs(emb[2] / 2 - latent.composition[2])) * 1;
    score += latent.composition[4] > 0.3 && emb[4] > 3 ? 1.5 : 0;
    score += latent.composition[5] > 0.2 && el === "H" ? 2 : 0;
    score += latent.composition[6] > 0.3 && isTransitionMetal(el) ? 1.5 : 0;
    score += latent.composition[7] > 0.2 && isRareEarth(el) ? 1.5 : 0;

    for (const role of motif.siteRoles) {
      if (role.preferredCategories?.some(cat => {
        if (cat === "transition-metal" && isTransitionMetal(el)) return true;
        if (cat === "rare-earth" && isRareEarth(el)) return true;
        if (cat === "hydrogen" && el === "H") return true;
        if (cat === "light-element" && ["B", "C", "N", "O", "H"].includes(el)) return true;
        if (cat === "pnictogen" && ["N", "P", "As", "Sb", "Bi"].includes(el)) return true;
        if (cat === "chalcogen" && ["O", "S", "Se", "Te"].includes(el)) return true;
        if (cat === "alkaline-earth" && ["Ca", "Sr", "Ba", "Mg"].includes(el)) return true;
        return false;
      })) {
        score += 2;
      }
    }

    return score;
  });

  const probs = softmax(scores, 0.5);

  const nElements = Math.min(
    motif.siteRoles.length,
    2 + Math.floor(Math.random() * 3)
  );

  const chosen: string[] = [];
  const counts: Record<string, number> = {};
  const usedIndices = new Set<number>();

  for (let i = 0; i < nElements; i++) {
    const adjustedProbs = probs.map((p, idx) => usedIndices.has(idx) ? 0 : p);
    const sum = adjustedProbs.reduce((s, p) => s + p, 0);
    if (sum < 1e-10) break;
    const normalized = adjustedProbs.map(p => p / sum);
    const idx = sampleFromDistribution(normalized);
    usedIndices.add(idx);
    const el = uniquePool[idx];
    chosen.push(el);

    let count = 1;
    if (el === "H") {
      count = 2 + Math.floor(Math.random() * 4);
    } else if (isTransitionMetal(el) || isRareEarth(el)) {
      count = 1 + Math.floor(Math.random() * 3);
    } else if (["O", "S", "Se", "Te", "N", "P", "As"].includes(el)) {
      count = 1 + Math.floor(Math.random() * 3);
    } else {
      count = 1 + Math.floor(Math.random() * 2);
    }
    counts[el] = count;
  }

  if (chosen.length < 2) {
    const fallbackEl = isTransitionMetal(chosen[0]) ? "B" : "Nb";
    chosen.push(fallbackEl);
    counts[fallbackEl] = 1;
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms > 16) {
    for (const el of chosen) {
      if (counts[el] > 2) counts[el] = Math.ceil(counts[el] * 0.7);
    }
  }

  return counts;
}

function generateAtomicPositions(
  composition: Record<string, number>,
  motif: StructuralMotif,
  noiseLevel: number
): CrystalAtom[] {
  const atoms: CrystalAtom[] = [];
  const elements = Object.entries(composition);

  const sitePositions: [number, number, number][] = [];
  const nAtoms = elements.reduce((s, [, n]) => s + n, 0);

  if (motif.crystalSystem === "cubic") {
    const positions: [number, number, number][] = [
      [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
      [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
      [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ];
    for (let i = 0; i < Math.min(nAtoms, positions.length); i++) {
      sitePositions.push(positions[i]);
    }
  } else if (motif.crystalSystem === "tetragonal") {
    const positions: [number, number, number][] = [
      [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0],
      [0, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
      [0, 0, 0.35], [0.5, 0.5, 0.85], [0.5, 0, 0.35], [0, 0.5, 0.35],
    ];
    for (let i = 0; i < Math.min(nAtoms, positions.length); i++) {
      sitePositions.push(positions[i]);
    }
  } else if (motif.crystalSystem === "hexagonal" || motif.crystalSystem === "trigonal") {
    const positions: [number, number, number][] = [
      [0, 0, 0], [1/3, 2/3, 0], [2/3, 1/3, 0],
      [0, 0, 0.5], [1/3, 2/3, 0.5], [2/3, 1/3, 0.5],
      [1/3, 2/3, 0.25], [2/3, 1/3, 0.75],
      [0, 0, 0.25], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0],
    ];
    for (let i = 0; i < Math.min(nAtoms, positions.length); i++) {
      sitePositions.push(positions[i]);
    }
  } else {
    for (let i = 0; i < nAtoms; i++) {
      sitePositions.push([
        Math.random() * 0.8 + 0.1,
        Math.random() * 0.8 + 0.1,
        Math.random() * 0.8 + 0.1,
      ]);
    }
  }

  while (sitePositions.length < nAtoms) {
    sitePositions.push([
      Math.random() * 0.8 + 0.1,
      Math.random() * 0.8 + 0.1,
      (sitePositions.length * 0.1) % 1.0,
    ]);
  }

  let posIdx = 0;
  for (const [el, count] of elements) {
    for (let i = 0; i < count && posIdx < sitePositions.length; i++) {
      const [bx, by, bz] = sitePositions[posIdx];
      atoms.push({
        element: el,
        fx: Math.max(0, Math.min(1, bx + gaussRandom() * noiseLevel)),
        fy: Math.max(0, Math.min(1, by + gaussRandom() * noiseLevel)),
        fz: Math.max(0, Math.min(1, bz + gaussRandom() * noiseLevel)),
        occupancy: 1.0,
      });
      posIdx++;
    }
  }

  return atoms;
}

function computeLatticeFromMotif(
  composition: Record<string, number>,
  motif: StructuralMotif
): { a: number; b: number; c: number; alpha: number; beta: number; gamma: number } {
  const elements = Object.keys(composition);
  const avgRadius = elements.reduce((s, el) => {
    const data = getElementData(el);
    return s + (data?.atomicRadius ?? 1.5) * (composition[el] || 1);
  }, 0) / Object.values(composition).reduce((s, n) => s + n, 0);

  const baseA = avgRadius * 2.2 + 1.5;
  const { a: ra, b: rb, c: rc } = motif.latticeRatios;

  const a = baseA * ra;
  const b = baseA * rb;
  const c = baseA * rc;

  const angles = CRYSTAL_SYSTEMS[motif.crystalSystem] ?? CRYSTAL_SYSTEMS.cubic;

  return {
    a: Math.max(3, Math.min(15, a)),
    b: Math.max(3, Math.min(15, b)),
    c: Math.max(3, Math.min(20, c)),
    alpha: angles.alpha,
    beta: angles.beta,
    gamma: angles.gamma,
  };
}

function cosineNoiseSchedule(step: number, totalSteps: number): number {
  const s = 0.008;
  const t = step / totalSteps;
  const alphaBar = Math.cos((t + s) / (1 + s) * Math.PI / 2) ** 2;
  return Math.sqrt(1 - alphaBar);
}

function computeCoordinationPenalty(
  atoms: CrystalAtom[],
  lattice: { a: number; b: number; c: number },
  targetCoord: number = 6
): number {
  let penalty = 0;
  for (let i = 0; i < atoms.length; i++) {
    let coord = 0;
    const maxMetalCoord = isTransitionMetal(atoms[i].element) ? 12 : 8;
    for (let j = 0; j < atoms.length; j++) {
      if (i === j) continue;
      const dx = (atoms[j].fx - atoms[i].fx) * lattice.a;
      const dy = (atoms[j].fy - atoms[i].fy) * lattice.b;
      const dz = (atoms[j].fz - atoms[i].fz) * lattice.c;
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const ri = ELEMENT_EMBEDDINGS[atoms[i].element]?.[2] ?? 1.5;
      const rj = ELEMENT_EMBEDDINGS[atoms[j].element]?.[2] ?? 1.5;
      if (r < (ri + rj) * 1.3) coord++;
    }
    if (coord < 2) penalty += (2 - coord) * 0.5;
    if (coord > maxMetalCoord) penalty += (coord - maxMetalCoord) * 0.3;
    penalty += Math.abs(coord - targetCoord) * 0.05;
  }
  return penalty / Math.max(atoms.length, 1);
}

function runDenoising(
  atoms: CrystalAtom[],
  lattice: { a: number; b: number; c: number },
  steps: number,
  motif: StructuralMotif,
  compositionCondition?: number[]
): CrystalAtom[] {
  let current = atoms.map(a => ({ ...a }));
  let bestEnergy = Infinity;
  let bestConfig = current.map(a => ({ ...a }));
  const totalSteps = Math.max(steps, 50);

  const condBias = compositionCondition ?? [];
  const avgCondZ = condBias.length > 0 ? condBias[0] * 50 : 30;
  const idealDist = 1.5 + avgCondZ * 0.015;

  for (let step = 0; step < totalSteps; step++) {
    const noiseScale = cosineNoiseSchedule(step, totalSteps) * 0.03;
    const t = step / totalSteps;
    const forceScale = t * 0.35;
    const coordWeight = Math.min(1.0, t * 2) * 0.1;

    for (let i = 0; i < current.length; i++) {
      let fx = 0, fy = 0, fz = 0;

      for (let j = 0; j < current.length; j++) {
        if (i === j) continue;
        const dx = current[j].fx - current[i].fx;
        const dy = current[j].fy - current[i].fy;
        const dz = current[j].fz - current[i].fz;
        const r2 = dx * dx + dy * dy + dz * dz;
        const r = Math.sqrt(Math.max(r2, 0.001));

        const ri = (ELEMENT_EMBEDDINGS[current[i].element]?.[2] ?? 1.5) / lattice.a;
        const rj = (ELEMENT_EMBEDDINGS[current[j].element]?.[2] ?? 1.5) / lattice.a;
        const sigma = (ri + rj) * 0.5;
        const sigma6 = Math.pow(sigma / r, 6);
        const ljForce = 24 * (2 * sigma6 * sigma6 - sigma6) / r;

        const idealR = idealDist / lattice.a;
        const springForce = -(r - idealR) * coordWeight;

        const force = ljForce + springForce;
        fx += force * dx / r * forceScale;
        fy += force * dy / r * forceScale;
        fz += force * dz / r * forceScale;
      }

      current[i].fx += fx + gaussRandom() * noiseScale;
      current[i].fy += fy + gaussRandom() * noiseScale;
      current[i].fz += fz + gaussRandom() * noiseScale;

      current[i].fx = Math.max(0, Math.min(1, current[i].fx));
      current[i].fy = Math.max(0, Math.min(1, current[i].fy));
      current[i].fz = Math.max(0, Math.min(1, current[i].fz));
    }

    let energy = 0;
    for (let i = 0; i < current.length; i++) {
      for (let j = i + 1; j < current.length; j++) {
        const dx = (current[j].fx - current[i].fx) * lattice.a;
        const dy = (current[j].fy - current[i].fy) * lattice.b;
        const dz = (current[j].fz - current[i].fz) * lattice.c;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const ri = ELEMENT_EMBEDDINGS[current[i].element]?.[2] ?? 1.5;
        const rj = ELEMENT_EMBEDDINGS[current[j].element]?.[2] ?? 1.5;
        const minDist = (ri + rj) * 0.85;
        if (r < minDist) energy += (minDist - r) * 10;
        energy += Math.abs(r - (ri + rj)) * 0.1;
      }
    }

    energy += computeCoordinationPenalty(current, lattice, motif.embedding.coordination) * 5;

    if (energy < bestEnergy) {
      bestEnergy = energy;
      bestConfig = current.map(a => ({ ...a }));
    }
  }

  return bestConfig;
}

const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66,
  F: 0.57, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02,
  K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32,
  Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20,
  Br: 1.20, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54, Ru: 1.46,
  Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38,
  I: 1.39, Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04, Hf: 1.75, Ta: 1.70, W: 1.62,
  Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36, Bi: 1.48,
};

function getOxidationStates(el: string): number[] {
  const states: Record<string, number[]> = {
    H: [1, -1], Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
    Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
    B: [3], Al: [3], Ga: [3], In: [3],
    C: [4, -4], Si: [4, -4], Ge: [4],
    N: [-3, 3, 5], P: [-3, 3, 5], As: [-3, 3, 5], Sb: [-3, 3, 5],
    O: [-2], S: [-2, 4, 6], Se: [-2, 4, 6], Te: [-2, 4, 6],
    F: [-1], Cl: [-1], Br: [-1], I: [-1],
    Sc: [3], Y: [3], La: [3], Ce: [3, 4],
    Ti: [3, 4], Zr: [4], Hf: [4],
    V: [3, 4, 5], Nb: [3, 5], Ta: [5],
    Cr: [3, 6], Mo: [4, 6], W: [4, 6],
    Mn: [2, 4, 7], Re: [4, 7],
    Fe: [2, 3], Ru: [3, 4], Os: [4],
    Co: [2, 3], Rh: [3], Ir: [3, 4],
    Ni: [2], Pd: [2, 4], Pt: [2, 4],
    Cu: [1, 2], Ag: [1], Au: [1, 3],
    Zn: [2], Cd: [2],
    Sn: [2, 4], Bi: [3],
  };
  return states[el] ?? [0];
}

function passesValidityFilter(
  atoms: CrystalAtom[],
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number },
  composition: Record<string, number>
): boolean {
  if (atoms.length === 0) return false;

  const vol = lattice.a * lattice.b * lattice.c;
  const volPerAtom = vol / atoms.length;
  if (volPerAtom < 5 || volPerAtom > 50) {
    return false;
  }

  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dx = (atoms[j].fx - atoms[i].fx) * lattice.a;
      const dy = (atoms[j].fy - atoms[i].fy) * lattice.b;
      const dz = (atoms[j].fz - atoms[i].fz) * lattice.c;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const ri = COVALENT_RADII[atoms[i].element] ?? 1.4;
      const rj = COVALENT_RADII[atoms[j].element] ?? 1.4;
      const minAllowed = (ri + rj) * 0.8;
      if (dist < minAllowed) {
        return false;
      }
    }
  }

  for (let i = 0; i < atoms.length; i++) {
    let coord = 0;
    const isMetal = isTransitionMetal(atoms[i].element) || isRareEarth(atoms[i].element);
    const maxCoord = isMetal ? 12 : 8;
    for (let j = 0; j < atoms.length; j++) {
      if (i === j) continue;
      const dx = (atoms[j].fx - atoms[i].fx) * lattice.a;
      const dy = (atoms[j].fy - atoms[i].fy) * lattice.b;
      const dz = (atoms[j].fz - atoms[i].fz) * lattice.c;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const ri = COVALENT_RADII[atoms[i].element] ?? 1.4;
      const rj = COVALENT_RADII[atoms[j].element] ?? 1.4;
      if (dist < (ri + rj) * 1.5) coord++;
    }
    if (coord < 1 || coord > maxCoord) {
      return false;
    }
  }

  const elements = Object.entries(composition);
  let bestChargeImbalance = Infinity;
  const tryOxStates = (idx: number, currentCharge: number): void => {
    if (idx >= elements.length) {
      bestChargeImbalance = Math.min(bestChargeImbalance, Math.abs(currentCharge));
      return;
    }
    const [el, count] = elements[idx];
    for (const ox of getOxidationStates(el)) {
      tryOxStates(idx + 1, currentCharge + ox * count);
    }
  };
  if (elements.length <= 5) {
    tryOxStates(0, 0);
    if (bestChargeImbalance > 0.5) {
      return false;
    }
  }

  return true;
}

function computeStructureScore(crystal: DiffusedCrystal): number {
  let score = 0;

  const nAtoms = crystal.atoms.length;
  const vol = crystal.lattice.a * crystal.lattice.b * crystal.lattice.c;
  const volPerAtom = vol / Math.max(nAtoms, 1);
  if (volPerAtom > 8 && volPerAtom < 100) score += 0.2;

  let minDist = Infinity;
  for (let i = 0; i < crystal.atoms.length; i++) {
    for (let j = i + 1; j < crystal.atoms.length; j++) {
      const dx = (crystal.atoms[j].fx - crystal.atoms[i].fx) * crystal.lattice.a;
      const dy = (crystal.atoms[j].fy - crystal.atoms[i].fy) * crystal.lattice.b;
      const dz = (crystal.atoms[j].fz - crystal.atoms[i].fz) * crystal.lattice.c;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < minDist) minDist = d;
    }
  }
  if (minDist > 0.8 && minDist < 5) score += 0.3;

  if (crystal.lattice.a > 2.5 && crystal.lattice.b > 2.5 && crystal.lattice.c > 2.5) score += 0.1;
  if (crystal.lattice.a < 15 && crystal.lattice.b < 15 && crystal.lattice.c < 20) score += 0.1;

  score += crystal.noveltyScore * 0.3;

  return Math.min(1, score);
}

function buildFormula(composition: Record<string, number>): string {
  const metalOrder = ["La", "Y", "Ce", "Ba", "Sr", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Nb", "Mo", "Zr", "Hf", "Ta", "W", "Re"];
  const entries = Object.entries(composition).filter(([, n]) => n > 0);
  entries.sort((a, b) => {
    const ai = metalOrder.indexOf(a[0]);
    const bi = metalOrder.indexOf(b[0]);
    if (ai >= 0 && bi >= 0) return ai - bi;
    if (ai >= 0) return -1;
    if (bi >= 0) return 1;
    return a[0].localeCompare(b[0]);
  });
  return entries.map(([el, n]) => n === 1 ? el : `${el}${n}`).join("");
}

const diffusionStats = {
  totalGenerated: 0,
  totalValid: 0,
  totalHighTc: 0,
  bestTc: 0,
  bestFormula: "",
  bestMotif: "",
  avgTc: 0,
  motifBreakdown: {} as Record<string, { count: number; avgTc: number; bestTc: number }>,
  recentResults: [] as { formula: string; tc: number; motif: string; steps: number }[],
};

export function runCrystalDiffusionCycle(
  count: number = 20,
  targetTc: number = 200,
  diffusionSteps: number = 30,
  sharedScreenedSet?: Set<string>
): DiffusedCrystal[] {
  const results: DiffusedCrystal[] = [];
  const motifs = SC_MOTIF_LIBRARY;
  const seenFormulas = new Set<string>();

  const EPSILON = 0.15;
  const MIN_EXPLORATION_RATE = 0.05;
  const MAX_MOTIF_FRACTION = 0.40;
  const maxPerMotif = Math.max(1, Math.ceil(count * MAX_MOTIF_FRACTION));
  const motifTrialCounts: Record<string, number> = {};

  const motifWeights = motifs.map(m => m.scAffinity);
  const baseMotifProbs = softmax(motifWeights.map(w => w * 5), 1.0);
  const flooredProbs = baseMotifProbs.map(p => Math.max(MIN_EXPLORATION_RATE, p));
  const flooredSum = flooredProbs.reduce((s, p) => s + p, 0);
  const motifProbs = flooredProbs.map(p => p / flooredSum);

  for (let attempt = 0; attempt < count * 3 && results.length < count; attempt++) {
    try {
      let motifIdx: number;
      if (Math.random() < EPSILON) {
        motifIdx = Math.floor(Math.random() * motifs.length);
      } else {
        const adjustedProbs = motifProbs.map((p, i) => {
          const name = motifs[i].name;
          if ((motifTrialCounts[name] || 0) >= maxPerMotif) return 0;
          return p;
        });
        const adjSum = adjustedProbs.reduce((s, p) => s + p, 0);
        if (adjSum <= 0) {
          motifIdx = Math.floor(Math.random() * motifs.length);
        } else {
          const normalized = adjustedProbs.map(p => p / adjSum);
          motifIdx = sampleFromDistribution(normalized);
        }
      }
      motifTrialCounts[motifs[motifIdx].name] = (motifTrialCounts[motifs[motifIdx].name] || 0) + 1;
      const motif = motifs[motifIdx];

      const latent = generateLatentVector(motif, targetTc);
      const composition = decodeComposition(latent, motif);

      const rawFormula = buildFormula(composition);
      if (!rawFormula || rawFormula.length < 2) continue;
      const formula = normalizeFormula(rawFormula);
      if (seenFormulas.has(formula)) continue;
      if (!isValidFormula(formula)) continue;
      if (!passesValenceFilter(formula)) continue;
      if (sharedScreenedSet && sharedScreenedSet.has(formula)) continue;
      seenFormulas.add(formula);

      const lattice = computeLatticeFromMotif(composition, motif);
      let atoms = generateAtomicPositions(composition, motif, 0.08);

      const compCondition = encodeComposition(composition);
      atoms = runDenoising(atoms, lattice, diffusionSteps, motif, compCondition);

      if (!passesValidityFilter(atoms, lattice, composition)) {
        diffusionStats.totalGenerated++;
        continue;
      }

      let predictedTc = 0;
      let lambda = 0;
      let stabilityScore = 0.5;

      try {
        const screen = surrogateScreen(formula, 2);
        if (!screen.pass) continue;
        predictedTc = screen.predictedTc;

        const features = extractFeatures(formula);
        lambda = features.electronPhononLambda ?? 0;

        const electronic = computeElectronicStructure(formula, null);
        const phonon = computePhononSpectrum(formula, electronic);
        const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

        if (coupling.lambda > 0.2) {
          lambda = coupling.lambda;
          const omegaLogK = coupling.omegaLog * 1.44;
          const denom = lambda - coupling.muStar * (1 + 0.62 * lambda);
          if (Math.abs(denom) > 1e-6) {
            const exponent = -1.04 * (1 + lambda) / denom;
            const mcmillanTc = (omegaLogK / 1.2) * Math.exp(exponent);
            if (Number.isFinite(mcmillanTc) && mcmillanTc > 0) {
              predictedTc = Math.max(predictedTc, mcmillanTc);
            }
          }
        }

        stabilityScore = Math.min(1, (phonon.softModeScore ?? 0.5) * 0.5 + (electronic.metallicity > 0.5 ? 0.3 : 0.1) + 0.2);
      } catch {}

      predictedTc = Math.min(400, Math.max(0, predictedTc));

      const latentNorm = Math.sqrt(
        latent.composition.reduce((s, v) => s + v * v, 0) +
        latent.structure.reduce((s, v) => s + v * v, 0)
      );

      const crystal: DiffusedCrystal = {
        formula,
        atoms,
        lattice: { ...lattice },
        spaceGroup: motif.spaceGroup,
        crystalSystem: motif.crystalSystem,
        motif: motif.name,
        predictedTc: Math.round(predictedTc * 10) / 10,
        stabilityScore: Math.round(stabilityScore * 1000) / 1000,
        noveltyScore: Math.round((0.5 + Math.random() * 0.3) * 1000) / 1000,
        diffusionSteps,
        lambda: Math.round(lambda * 1000) / 1000,
        latentNorm: Math.round(latentNorm * 100) / 100,
      };

      const structScore = computeStructureScore(crystal);
      if (structScore < 0.3) continue;

      results.push(crystal);
      diffusionStats.totalGenerated++;
      diffusionStats.totalValid++;

      if (predictedTc > 20) diffusionStats.totalHighTc++;
      if (predictedTc > diffusionStats.bestTc) {
        diffusionStats.bestTc = predictedTc;
        diffusionStats.bestFormula = formula;
        diffusionStats.bestMotif = motif.name;
      }

      if (!diffusionStats.motifBreakdown[motif.name]) {
        diffusionStats.motifBreakdown[motif.name] = { count: 0, avgTc: 0, bestTc: 0 };
      }
      const mb = diffusionStats.motifBreakdown[motif.name];
      const newAvg = (mb.avgTc * mb.count + predictedTc) / (mb.count + 1);
      mb.avgTc = Number.isFinite(newAvg) ? newAvg : 0;
      mb.count++;
      if (predictedTc > mb.bestTc) mb.bestTc = predictedTc;

      diffusionStats.recentResults.push({
        formula, tc: predictedTc, motif: motif.name, steps: diffusionSteps,
      });
      if (diffusionStats.recentResults.length > 50) diffusionStats.recentResults.shift();

    } catch {}
  }

  if (diffusionStats.recentResults.length > 0) {
    const avg = diffusionStats.recentResults.reduce((s, r) => s + r.tc, 0) / diffusionStats.recentResults.length;
    diffusionStats.avgTc = Number.isFinite(avg) ? avg : 0;
  }

  return results;
}

const SCORE_DECAY = 0.95;
const distributionDiffusionStats = {
  totalGenerated: 0,
  totalValid: 0,
  totalHighTc: 0,
  bestTc: 0,
  bestFormula: "",
  bestSpaceGroup: "",
  bestCrystalSystem: "",
  systemBreakdown: {} as Record<string, { count: number; avgTc: number; bestTc: number }>,
  recentResults: [] as { formula: string; tc: number; sg: string; system: string }[],
  avgStructureScore: 0,
  learnedSystemWeights: {} as Record<string, number>,
};

function scoreBasedDenoising(
  sites: WyckoffSite[],
  lattice: { a: number; b: number; c: number },
  steps: number,
  temperature: number = 1.0
): WyckoffSite[] {
  let current = sites.map(s => ({ ...s }));
  let bestEnergy = Infinity;
  let bestConfig = current.map(s => ({ ...s }));

  for (let step = 0; step < steps; step++) {
    const t = (1 - step / steps) * temperature;
    const noiseScale = t * 0.04;
    const forceScale = (1 - t) * 0.2;

    for (let i = 0; i < current.length; i++) {
      let fx = 0, fy = 0, fz = 0;

      for (let j = 0; j < current.length; j++) {
        if (i === j) continue;
        const dx = (current[j].x - current[i].x) * lattice.a;
        const dy = (current[j].y - current[i].y) * lattice.b;
        const dz = (current[j].z - current[i].z) * lattice.c;
        const r2 = dx * dx + dy * dy + dz * dz;
        const r = Math.sqrt(Math.max(r2, 0.01));

        const ri = (ELEMENT_EMBEDDINGS[current[i].element]?.[2] ?? 1.5);
        const rj = (ELEMENT_EMBEDDINGS[current[j].element]?.[2] ?? 1.5);
        const sigma = (ri + rj) * 0.4;

        if (r < sigma * 2) {
          const overlap = sigma * 2 - r;
          const repulsion = overlap * 0.5 * forceScale;
          fx -= repulsion * dx / (r * lattice.a);
          fy -= repulsion * dy / (r * lattice.b);
          fz -= repulsion * dz / (r * lattice.c);
        }
      }

      current[i].x += fx + gaussRandom() * noiseScale;
      current[i].y += fy + gaussRandom() * noiseScale;
      current[i].z += fz + gaussRandom() * noiseScale;
      current[i].x = Math.max(0, Math.min(1, current[i].x));
      current[i].y = Math.max(0, Math.min(1, current[i].y));
      current[i].z = Math.max(0, Math.min(1, current[i].z));
    }

    let energy = 0;
    for (let i = 0; i < current.length; i++) {
      for (let j = i + 1; j < current.length; j++) {
        const dx = (current[j].x - current[i].x) * lattice.a;
        const dy = (current[j].y - current[i].y) * lattice.b;
        const dz = (current[j].z - current[i].z) * lattice.c;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const ri = ELEMENT_EMBEDDINGS[current[i].element]?.[2] ?? 1.5;
        const rj = ELEMENT_EMBEDDINGS[current[j].element]?.[2] ?? 1.5;
        const optDist = (ri + rj) * 0.85;
        if (r < optDist * 0.5) energy += 100;
        else energy += Math.pow(r - optDist, 2) * 0.5;
      }
    }

    if (energy < bestEnergy) {
      bestEnergy = energy;
      bestConfig = current.map(s => ({ ...s }));
    }
  }

  return bestConfig;
}

export function runDistributionBasedDiffusion(
  count: number = 15,
  targetTc: number = 200,
  steps: number = 30,
  sharedScreenedSet?: Set<string>
): DiffusedCrystal[] {
  const results: DiffusedCrystal[] = [];
  const seenFormulas = new Set<string>();

  const allDists = getAllDistributions();

  const DIST_EPSILON = 0.15;
  const DIST_MIN_FLOOR = 0.05;
  const DIST_MAX_FRACTION = 0.40;
  const maxPerSystem = Math.max(1, Math.ceil(count * DIST_MAX_FRACTION));
  const systemTrialCounts: Record<string, number> = {};

  for (let attempt = 0; attempt < count * 3 && results.length < count; attempt++) {
    try {
      let sysDist: CrystalSystemDistribution;
      const lw = distributionDiffusionStats.learnedSystemWeights;
      const hasLearned = Object.keys(lw).length > 0;
      if (Math.random() < DIST_EPSILON) {
        sysDist = allDists[Math.floor(Math.random() * allDists.length)];
      } else if (hasLearned && Math.random() < 0.7) {
        const weights = allDists.map(d => {
          if ((systemTrialCounts[d.system] || 0) >= maxPerSystem) return 0;
          return Math.max(DIST_MIN_FLOOR, lw[d.system] ?? 1.0);
        });
        const totalWeight = weights.reduce((s, w) => s + w, 0);
        if (totalWeight <= 0) {
          sysDist = allDists[Math.floor(Math.random() * allDists.length)];
        } else {
          let r = Math.random() * totalWeight;
          sysDist = allDists[0];
          for (let i = 0; i < allDists.length; i++) {
            r -= weights[i];
            if (r <= 0) { sysDist = allDists[i]; break; }
          }
        }
      } else {
        sysDist = sampleCrystalSystem(true);
      }
      systemTrialCounts[sysDist.system] = (systemTrialCounts[sysDist.system] || 0) + 1;
      const { sg, symbol: sgSymbol } = sampleSpaceGroup(sysDist);
      const lattice = sampleLatticeParams(sysDist);

      const motifs = SC_MOTIF_LIBRARY.filter(m => m.crystalSystem === sysDist.system);
      const motif = motifs.length > 0
        ? motifs[Math.floor(Math.random() * motifs.length)]
        : SC_MOTIF_LIBRARY[Math.floor(Math.random() * SC_MOTIF_LIBRARY.length)];

      const latent = generateLatentVector(motif, targetTc);
      const composition = decodeComposition(latent, motif);
      const rawFormula = buildFormula(composition);

      if (!rawFormula || rawFormula.length < 2) continue;
      const formula = normalizeFormula(rawFormula);
      if (seenFormulas.has(formula)) continue;
      if (!isValidFormula(formula)) continue;
      if (!passesValenceFilter(formula)) continue;
      if (sharedScreenedSet && sharedScreenedSet.has(formula)) continue;
      seenFormulas.add(formula);

      let wyckoffSites = sampleWyckoffPositions(sysDist, composition);
      wyckoffSites = scoreBasedDenoising(wyckoffSites, lattice, steps);

      const atoms: CrystalAtom[] = wyckoffSites.map(s => ({
        element: s.element,
        fx: s.x,
        fy: s.y,
        fz: s.z,
        occupancy: s.occupancy,
      }));

      let predictedTc = 0;
      let lambda = 0;
      let stabilityScore = 0.5;

      try {
        const screen = surrogateScreen(formula, 2);
        if (!screen.pass) continue;
        predictedTc = screen.predictedTc;

        const features = extractFeatures(formula);
        lambda = features.electronPhononLambda ?? 0;

        const electronic = computeElectronicStructure(formula, null);
        const phonon = computePhononSpectrum(formula, electronic);
        const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

        if (coupling.lambda > 0.2) {
          lambda = coupling.lambda;
          const omegaLogK = coupling.omegaLog * 1.44;
          const denom = lambda - coupling.muStar * (1 + 0.62 * lambda);
          if (Math.abs(denom) > 1e-6) {
            const exponent = -1.04 * (1 + lambda) / denom;
            const mcmillanTc = (omegaLogK / 1.2) * Math.exp(exponent);
            if (Number.isFinite(mcmillanTc) && mcmillanTc > 0) {
              predictedTc = Math.max(predictedTc, mcmillanTc);
            }
          }
        }

        stabilityScore = Math.min(1, (phonon.softModeScore ?? 0.5) * 0.5 + (electronic.metallicity > 0.5 ? 0.3 : 0.1) + 0.2);
      } catch {}

      predictedTc = Math.min(400, Math.max(0, predictedTc));

      const latentNorm = Math.sqrt(
        latent.composition.reduce((s, v) => s + v * v, 0) +
        latent.structure.reduce((s, v) => s + v * v, 0)
      );

      const crystal: DiffusedCrystal = {
        formula,
        atoms,
        lattice,
        spaceGroup: `${sgSymbol} (#${sg})`,
        crystalSystem: sysDist.system,
        motif: motif.name,
        predictedTc: Math.round(predictedTc * 10) / 10,
        stabilityScore: Math.round(stabilityScore * 1000) / 1000,
        noveltyScore: Math.round((0.5 + Math.random() * 0.3) * 1000) / 1000,
        diffusionSteps: steps,
        lambda: Math.round(lambda * 1000) / 1000,
        latentNorm: Math.round(latentNorm * 100) / 100,
      };

      results.push(crystal);
      distributionDiffusionStats.totalGenerated++;
      distributionDiffusionStats.totalValid++;

      if (predictedTc > 20) distributionDiffusionStats.totalHighTc++;
      if (predictedTc > distributionDiffusionStats.bestTc) {
        distributionDiffusionStats.bestTc = predictedTc;
        distributionDiffusionStats.bestFormula = formula;
        distributionDiffusionStats.bestSpaceGroup = sgSymbol;
        distributionDiffusionStats.bestCrystalSystem = sysDist.system;
      }

      const sysKey = sysDist.system;
      if (!distributionDiffusionStats.systemBreakdown[sysKey]) {
        distributionDiffusionStats.systemBreakdown[sysKey] = { count: 0, avgTc: 0, bestTc: 0 };
      }
      const sb = distributionDiffusionStats.systemBreakdown[sysKey];
      const newSbAvg = (sb.avgTc * sb.count + predictedTc) / (sb.count + 1);
      sb.avgTc = Number.isFinite(newSbAvg) ? newSbAvg : 0;
      sb.count++;
      if (predictedTc > sb.bestTc) sb.bestTc = predictedTc;

      distributionDiffusionStats.recentResults.push({
        formula, tc: predictedTc, sg: sgSymbol, system: sysDist.system,
      });
      if (distributionDiffusionStats.recentResults.length > 50) distributionDiffusionStats.recentResults.shift();

      const systemWeights = distributionDiffusionStats.learnedSystemWeights;
      if (!systemWeights[sysKey]) systemWeights[sysKey] = 1.0;
      if (predictedTc > 20) {
        systemWeights[sysKey] *= 1.05;
      } else {
        systemWeights[sysKey] *= SCORE_DECAY;
      }

    } catch {}
  }

  return results;
}

export function getDistributionDiffusionStats() {
  return {
    ...distributionDiffusionStats,
    crystallographicDB: getDistributionStats(),
  };
}

export function getCrystalDiffusionStats() {
  return {
    ...diffusionStats,
    distributionBased: distributionDiffusionStats,
  };
}