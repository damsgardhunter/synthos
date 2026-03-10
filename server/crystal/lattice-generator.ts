import { computeCompositionFeatures } from "../learning/composition-features";

export type BravaisLattice = "cubic" | "tetragonal" | "orthorhombic" | "hexagonal" | "monoclinic" | "trigonal" | "triclinic";

export interface LatticeParams {
  a: number;
  b: number;
  c: number;
  alpha: number;
  beta: number;
  gamma: number;
}

export interface AtomPosition {
  element: string;
  fx: number;
  fy: number;
  fz: number;
}

export interface GeneratedStructure {
  formula: string;
  lattice: LatticeParams;
  atoms: AtomPosition[];
  bravaisType: BravaisLattice;
  volumePerAtom: number;
  generationMethod: string;
  noveltyScore: number;
  confidence: number;
}

const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57,
  Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02,
  K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39,
  Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20,
  As: 1.19, Se: 1.20, Br: 1.20, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75,
  Nb: 1.64, Mo: 1.54, Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44,
  In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39, Cs: 2.44, Ba: 2.15,
  La: 2.07, Ce: 2.04, Pr: 2.03, Nd: 2.01, Sm: 1.98, Gd: 1.96,
  Tb: 1.94, Dy: 1.92, Ho: 1.92, Er: 1.89, Tm: 1.90, Yb: 1.87, Lu: 1.87,
  Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36,
  Au: 1.36, Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48, Th: 2.06, U: 1.96,
};

const ATOMIC_VOLUMES: Record<string, number> = {
  H: 5.08, He: 4.39, Li: 21.3, Be: 8.11, B: 7.24, C: 5.31, N: 22.6, O: 11.2, F: 11.2,
  Na: 23.7, Mg: 23.2, Al: 16.6, Si: 12.1, P: 17.0, S: 15.5, Cl: 22.7,
  K: 73.5, Ca: 43.6, Sc: 25.0, Ti: 17.6, V: 13.8, Cr: 12.0, Mn: 12.2,
  Fe: 11.8, Co: 11.1, Ni: 10.9, Cu: 11.8, Zn: 15.2, Ga: 19.6, Ge: 22.6,
  As: 21.5, Se: 25.2, Br: 23.5, Rb: 87.2, Sr: 56.0, Y: 33.1, Zr: 23.3,
  Nb: 18.0, Mo: 15.6, Ru: 13.6, Rh: 13.7, Pd: 14.7, Ag: 17.1, Cd: 21.7,
  In: 26.2, Sn: 27.3, Sb: 30.3, Te: 33.8, I: 42.7, Cs: 110.0, Ba: 63.4,
  La: 37.5, Ce: 34.0, Pr: 35.0, Nd: 34.2, Sm: 33.2, Gd: 33.0,
  Hf: 22.3, Ta: 18.0, W: 15.8, Re: 14.7, Os: 14.0, Ir: 14.1, Pt: 15.1,
  Au: 17.0, Hg: 23.4, Tl: 28.6, Pb: 30.3, Bi: 35.4, Th: 32.9, U: 20.8,
};

const BRAVAIS_WEIGHTS: [BravaisLattice, number][] = [
  ["cubic", 0.25],
  ["tetragonal", 0.20],
  ["hexagonal", 0.15],
  ["orthorhombic", 0.20],
  ["monoclinic", 0.10],
  ["trigonal", 0.07],
  ["triclinic", 0.03],
];

function weightedChoice<T>(items: [T, number][]): T {
  const total = items.reduce((s, [, w]) => s + w, 0);
  let r = Math.random() * total;
  for (const [item, weight] of items) {
    r -= weight;
    if (r <= 0) return item;
  }
  return items[items.length - 1][0];
}

function rand(lo: number, hi: number): number {
  return lo + Math.random() * (hi - lo);
}

function estimateTargetVolume(elements: string[], counts: number[]): number {
  let totalVol = 0;
  for (let i = 0; i < elements.length; i++) {
    const vol = ATOMIC_VOLUMES[elements[i]] ?? 15.0;
    totalVol += vol * counts[i];
  }
  const packingFactor = 1.2 + Math.random() * 0.3;
  return totalVol * packingFactor;
}

function generateBravaisLattice(type: BravaisLattice, targetVolume: number): LatticeParams {
  const cbrt = Math.cbrt(targetVolume);

  switch (type) {
    case "cubic": {
      const a = cbrt * rand(0.9, 1.1);
      return { a, b: a, c: a, alpha: 90, beta: 90, gamma: 90 };
    }
    case "tetragonal": {
      const cOverA = rand(0.8, 2.5);
      const a = Math.cbrt(targetVolume / cOverA) * rand(0.95, 1.05);
      const c = a * cOverA;
      return { a, b: a, c, alpha: 90, beta: 90, gamma: 90 };
    }
    case "hexagonal": {
      const cOverA = rand(1.0, 3.0);
      const hexFactor = Math.sqrt(3) / 2;
      const a = Math.cbrt(targetVolume / (cOverA * hexFactor)) * rand(0.95, 1.05);
      const c = a * cOverA;
      return { a, b: a, c, alpha: 90, beta: 90, gamma: 120 };
    }
    case "trigonal": {
      const a = cbrt * rand(0.9, 1.1);
      const alpha = rand(70, 110);
      return { a, b: a, c: a, alpha, beta: alpha, gamma: alpha };
    }
    case "orthorhombic": {
      const ra = rand(0.7, 1.3);
      const rb = rand(0.7, 1.3);
      const rc = rand(0.7, 1.3);
      const scale = Math.cbrt(targetVolume / (ra * rb * rc));
      return {
        a: ra * scale, b: rb * scale, c: rc * scale,
        alpha: 90, beta: 90, gamma: 90,
      };
    }
    case "monoclinic": {
      const ra = rand(0.7, 1.3);
      const rb = rand(0.7, 1.3);
      const rc = rand(0.7, 1.3);
      const beta = rand(90, 120);
      const sinBeta = Math.sin(beta * Math.PI / 180);
      const scale = Math.cbrt(targetVolume / (ra * rb * rc * sinBeta));
      return {
        a: ra * scale, b: rb * scale, c: rc * scale,
        alpha: 90, beta, gamma: 90,
      };
    }
    case "triclinic": {
      const ra = rand(0.7, 1.3);
      const rb = rand(0.7, 1.3);
      const rc = rand(0.7, 1.3);
      const alpha = rand(70, 110);
      const beta = rand(70, 110);
      const gamma = rand(70, 110);
      const cosA = Math.cos(alpha * Math.PI / 180);
      const cosB = Math.cos(beta * Math.PI / 180);
      const cosG = Math.cos(gamma * Math.PI / 180);
      const volFactor = Math.sqrt(1 - cosA * cosA - cosB * cosB - cosG * cosG + 2 * cosA * cosB * cosG);
      const scale = Math.cbrt(targetVolume / (ra * rb * rc * volFactor));
      return {
        a: ra * scale, b: rb * scale, c: rc * scale,
        alpha, beta, gamma,
      };
    }
  }
}

function computeCellVolume(lp: LatticeParams): number {
  const cosA = Math.cos(lp.alpha * Math.PI / 180);
  const cosB = Math.cos(lp.beta * Math.PI / 180);
  const cosG = Math.cos(lp.gamma * Math.PI / 180);
  return lp.a * lp.b * lp.c * Math.sqrt(1 - cosA * cosA - cosB * cosB - cosG * cosG + 2 * cosA * cosB * cosG);
}

function fractionalToCartesian(fx: number, fy: number, fz: number, lp: LatticeParams): [number, number, number] {
  const { a, b, c, alpha, beta, gamma } = lp;
  const cosA = Math.cos(alpha * Math.PI / 180);
  const cosB = Math.cos(beta * Math.PI / 180);
  const cosG = Math.cos(gamma * Math.PI / 180);
  const sinG = Math.sin(gamma * Math.PI / 180);

  const ax = a;
  const bx = b * cosG;
  const by = b * sinG;
  const cx = c * cosB;
  const cy = c * (cosA - cosB * cosG) / sinG;
  const cz = c * Math.sqrt(1 - cosB * cosB - ((cosA - cosB * cosG) / sinG) ** 2);

  return [
    fx * ax + fy * bx + fz * cx,
    fy * by + fz * cy,
    fz * cz,
  ];
}

function cartesianDistance(
  a: AtomPosition, b: AtomPosition, lp: LatticeParams
): number {
  const [ax, ay, az] = fractionalToCartesian(a.fx, a.fy, a.fz, lp);
  const [bx, by, bz] = fractionalToCartesian(b.fx, b.fy, b.fz, lp);

  let minDist = Infinity;
  for (let di = -1; di <= 1; di++) {
    for (let dj = -1; dj <= 1; dj++) {
      for (let dk = -1; dk <= 1; dk++) {
        const [px, py, pz] = fractionalToCartesian(
          b.fx + di, b.fy + dj, b.fz + dk, lp
        );
        const d = Math.sqrt((ax - px) ** 2 + (ay - py) ** 2 + (az - pz) ** 2);
        if (d < minDist) minDist = d;
      }
    }
  }
  return minDist;
}

function minAllowedDistance(elA: string, elB: string): number {
  const rA = COVALENT_RADII[elA] ?? 1.4;
  const rB = COVALENT_RADII[elB] ?? 1.4;
  return 0.75 * (rA + rB);
}

function isPositionValid(
  newAtom: AtomPosition,
  placed: AtomPosition[],
  lp: LatticeParams,
): boolean {
  for (const existing of placed) {
    const dist = cartesianDistance(newAtom, existing, lp);
    const dMin = minAllowedDistance(newAtom.element, existing.element);
    if (dist < dMin) return false;
  }
  return true;
}

function placeAtoms(
  elements: string[],
  counts: number[],
  lp: LatticeParams,
  maxRetries: number = 200,
): AtomPosition[] | null {
  const placed: AtomPosition[] = [];
  const atomList: string[] = [];
  for (let i = 0; i < elements.length; i++) {
    for (let j = 0; j < counts[i]; j++) {
      atomList.push(elements[i]);
    }
  }

  for (let k = 0; k < atomList.length; k++) {
    atomList.sort(() => Math.random() - 0.5);
  }

  for (const element of atomList) {
    let success = false;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      const candidate: AtomPosition = {
        element,
        fx: Math.random(),
        fy: Math.random(),
        fz: Math.random(),
      };
      if (isPositionValid(candidate, placed, lp)) {
        placed.push(candidate);
        success = true;
        break;
      }
    }
    if (!success) return null;
  }
  return placed;
}

function placeAtomsWithSymmetry(
  elements: string[],
  counts: number[],
  lp: LatticeParams,
  bravais: BravaisLattice,
): AtomPosition[] | null {
  const placed: AtomPosition[] = [];
  const atomList: string[] = [];
  for (let i = 0; i < elements.length; i++) {
    for (let j = 0; j < counts[i]; j++) {
      atomList.push(elements[i]);
    }
  }

  const highSymSites = generateHighSymmetrySites(bravais);

  let siteIdx = 0;
  for (const element of atomList) {
    let success = false;

    if (siteIdx < highSymSites.length && Math.random() < 0.6) {
      const site = highSymSites[siteIdx];
      const candidate: AtomPosition = {
        element,
        fx: site[0] + rand(-0.02, 0.02),
        fy: site[1] + rand(-0.02, 0.02),
        fz: site[2] + rand(-0.02, 0.02),
      };
      candidate.fx = ((candidate.fx % 1) + 1) % 1;
      candidate.fy = ((candidate.fy % 1) + 1) % 1;
      candidate.fz = ((candidate.fz % 1) + 1) % 1;
      if (isPositionValid(candidate, placed, lp)) {
        placed.push(candidate);
        siteIdx++;
        success = true;
      }
    }

    if (!success) {
      for (let attempt = 0; attempt < 200; attempt++) {
        const candidate: AtomPosition = {
          element,
          fx: Math.random(),
          fy: Math.random(),
          fz: Math.random(),
        };
        if (isPositionValid(candidate, placed, lp)) {
          placed.push(candidate);
          success = true;
          break;
        }
      }
    }
    if (!success) return null;
  }
  return placed;
}

function generateHighSymmetrySites(bravais: BravaisLattice): [number, number, number][] {
  const sites: [number, number, number][] = [];

  const origins: [number, number, number][] = [
    [0, 0, 0],
    [0.5, 0.5, 0.5],
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5],
  ];

  switch (bravais) {
    case "cubic":
      sites.push(...origins);
      sites.push([0.25, 0.25, 0.25], [0.75, 0.75, 0.75]);
      break;
    case "tetragonal":
      sites.push(
        [0, 0, 0], [0.5, 0.5, 0.5],
        [0.5, 0, 0.25], [0, 0.5, 0.25],
        [0.5, 0.5, 0], [0, 0, 0.5],
      );
      break;
    case "hexagonal":
      sites.push(
        [0, 0, 0], [1/3, 2/3, 0.5], [2/3, 1/3, 0.5],
        [1/3, 2/3, 0], [2/3, 1/3, 0],
        [0, 0, 0.5], [0.5, 0, 0], [0, 0.5, 0],
      );
      break;
    case "orthorhombic":
      sites.push(...origins);
      break;
    case "trigonal":
      sites.push(
        [0, 0, 0], [1/3, 2/3, 2/3], [2/3, 1/3, 1/3],
        [0, 0, 0.5], [0.5, 0.5, 0.5],
      );
      break;
    case "monoclinic":
      sites.push(
        [0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
      );
      break;
    case "triclinic":
      sites.push([0, 0, 0], [0.5, 0.5, 0.5]);
      break;
  }

  for (let i = sites.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [sites[i], sites[j]] = [sites[j], sites[i]];
  }

  return sites;
}

function parseFormula(formula: string): { elements: string[]; counts: number[] } {
  const elements: string[] = [];
  const counts: number[] = [];
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    if (!match[1]) continue;
    elements.push(match[1]);
    counts.push(match[2] ? parseFloat(match[2]) : 1);
  }
  return { elements, counts };
}

function clampLattice(lp: LatticeParams): LatticeParams {
  return {
    a: Math.max(2.5, Math.min(50, lp.a)),
    b: Math.max(2.5, Math.min(50, lp.b)),
    c: Math.max(2.5, Math.min(50, lp.c)),
    alpha: Math.max(40, Math.min(140, lp.alpha)),
    beta: Math.max(40, Math.min(140, lp.beta)),
    gamma: Math.max(40, Math.min(140, lp.gamma)),
  };
}

let stats = {
  totalAttempts: 0,
  successful: 0,
  failedPlacement: 0,
  failedVolume: 0,
  byBravais: {} as Record<string, { attempts: number; success: number }>,
};

export function generatePrototypeFreeStructure(
  formula: string,
  preferredBravais?: BravaisLattice,
): GeneratedStructure | null {
  stats.totalAttempts++;

  const { elements, counts } = parseFormula(formula);
  if (elements.length === 0) return null;

  const totalAtoms = counts.reduce((s, n) => s + n, 0);
  if (totalAtoms < 1 || totalAtoms > 30) return null;

  const bravais = preferredBravais ?? weightedChoice(BRAVAIS_WEIGHTS);
  if (!stats.byBravais[bravais]) stats.byBravais[bravais] = { attempts: 0, success: 0 };
  stats.byBravais[bravais].attempts++;

  const targetVolume = estimateTargetVolume(elements, counts);

  for (let latticeAttempt = 0; latticeAttempt < 5; latticeAttempt++) {
    const rawLattice = generateBravaisLattice(bravais, targetVolume);
    const lattice = clampLattice(rawLattice);

    const vol = computeCellVolume(lattice);
    const volPerAtom = vol / totalAtoms;

    const hasH = elements.includes("H");
    const minVPA = hasH ? 2.5 : 5.0;
    if (volPerAtom < minVPA || volPerAtom > 80) {
      stats.failedVolume++;
      continue;
    }

    const useSymmetry = Math.random() < 0.7;
    const atoms = useSymmetry
      ? placeAtomsWithSymmetry(elements, counts, lattice, bravais)
      : placeAtoms(elements, counts, lattice);

    if (!atoms) {
      stats.failedPlacement++;
      continue;
    }

    stats.successful++;
    stats.byBravais[bravais].success++;

    return {
      formula,
      lattice,
      atoms,
      bravaisType: bravais,
      volumePerAtom: volPerAtom,
      generationMethod: "prototype_free_lattice",
      noveltyScore: 0.5 + Math.random() * 0.4,
      confidence: 0.3 + Math.random() * 0.3,
    };
  }

  return null;
}

export function generatePrototypeFreeStructures(
  formulas: string[],
  count: number = 10,
  preferredBravais?: BravaisLattice,
): GeneratedStructure[] {
  const results: GeneratedStructure[] = [];
  const maxAttempts = count * 3;

  for (let i = 0; i < maxAttempts && results.length < count; i++) {
    const formula = formulas[i % formulas.length];
    const structure = generatePrototypeFreeStructure(formula, preferredBravais);
    if (structure) results.push(structure);
  }

  return results;
}

export function getLatticeGeneratorStats() {
  return {
    ...stats,
    successRate: stats.totalAttempts > 0
      ? Math.round((stats.successful / stats.totalAttempts) * 10000) / 10000
      : 0,
    evoPopulationSize: evoPopulation.length,
    evoGenerations: evoGeneration,
    evoTotalOffspring: evoStats.totalOffspring,
    evoMutations: evoStats.mutations,
    evoCrossovers: evoStats.crossovers,
    evoImprovements: evoStats.improvements,
  };
}

export function resetLatticeGeneratorStats(): void {
  stats = {
    totalAttempts: 0,
    successful: 0,
    failedPlacement: 0,
    failedVolume: 0,
    byBravais: {},
  };
}

interface EvoIndividual {
  structure: GeneratedStructure;
  fitness: number;
  generation: number;
}

const MAX_POPULATION = 50;
const TOURNAMENT_SIZE = 3;
let evoPopulation: EvoIndividual[] = [];
let evoGeneration = 0;
let evoStats = { totalOffspring: 0, mutations: 0, crossovers: 0, improvements: 0 };

function computeStructureFitness(s: GeneratedStructure): number {
  let fitness = 0.3;

  const vpa = s.volumePerAtom;
  if (vpa >= 8 && vpa <= 25) fitness += 0.2;
  else if (vpa >= 5 && vpa <= 40) fitness += 0.1;

  if (s.bravaisType === "cubic" || s.bravaisType === "hexagonal" || s.bravaisType === "tetragonal") {
    fitness += 0.1;
  }

  const uniqueElements = new Set(s.atoms.map(a => a.element)).size;
  if (uniqueElements >= 2 && uniqueElements <= 4) fitness += 0.1;

  let minDist = Infinity;
  for (let i = 0; i < s.atoms.length; i++) {
    for (let j = i + 1; j < s.atoms.length; j++) {
      const d = cartesianDistance(s.atoms[i], s.atoms[j], s.lattice);
      if (d < minDist) minDist = d;
    }
  }
  const avgCovRad = s.atoms.reduce((sum, a) => sum + (COVALENT_RADII[a.element] ?? 1.4), 0) / s.atoms.length;
  const idealMinDist = 1.6 * avgCovRad;
  if (minDist > 0 && minDist < 20) {
    const distRatio = minDist / idealMinDist;
    if (distRatio >= 0.8 && distRatio <= 1.5) fitness += 0.2;
    else if (distRatio >= 0.6 && distRatio <= 2.0) fitness += 0.1;
  }

  fitness += s.noveltyScore * 0.1;

  return Math.min(1.0, fitness);
}

function mutateLattice(lp: LatticeParams, intensity: number = 0.1): LatticeParams {
  const perturb = (v: number, scale: number) => v * (1 + (Math.random() * 2 - 1) * scale);
  return clampLattice({
    a: perturb(lp.a, intensity),
    b: perturb(lp.b, intensity),
    c: perturb(lp.c, intensity),
    alpha: lp.alpha + (Math.random() * 2 - 1) * intensity * 15,
    beta: lp.beta + (Math.random() * 2 - 1) * intensity * 15,
    gamma: lp.gamma + (Math.random() * 2 - 1) * intensity * 15,
  });
}

function mutatePositions(atoms: AtomPosition[], intensity: number = 0.05): AtomPosition[] {
  return atoms.map(a => ({
    element: a.element,
    fx: ((a.fx + (Math.random() * 2 - 1) * intensity) % 1 + 1) % 1,
    fy: ((a.fy + (Math.random() * 2 - 1) * intensity) % 1 + 1) % 1,
    fz: ((a.fz + (Math.random() * 2 - 1) * intensity) % 1 + 1) % 1,
  }));
}

function mutateStructure(parent: GeneratedStructure): GeneratedStructure | null {
  const mutType = Math.random();
  let newLattice: LatticeParams;
  let newAtoms: AtomPosition[];

  if (mutType < 0.4) {
    newLattice = mutateLattice(parent.lattice, 0.08 + Math.random() * 0.12);
    newAtoms = parent.atoms.map(a => ({ ...a }));
  } else if (mutType < 0.7) {
    newLattice = { ...parent.lattice };
    newAtoms = mutatePositions(parent.atoms, 0.03 + Math.random() * 0.07);
  } else if (mutType < 0.9) {
    newLattice = mutateLattice(parent.lattice, 0.05);
    newAtoms = mutatePositions(parent.atoms, 0.04);
  } else {
    if (newAtoms = parent.atoms.map(a => ({ ...a })), newAtoms.length >= 2) {
      const i = Math.floor(Math.random() * newAtoms.length);
      let j = Math.floor(Math.random() * newAtoms.length);
      while (j === i && newAtoms.length > 1) j = Math.floor(Math.random() * newAtoms.length);
      if (newAtoms[i].element !== newAtoms[j].element) {
        [newAtoms[i].element, newAtoms[j].element] = [newAtoms[j].element, newAtoms[i].element];
      }
    }
    newLattice = { ...parent.lattice };
  }

  for (let i = 0; i < newAtoms.length; i++) {
    for (let j = i + 1; j < newAtoms.length; j++) {
      const d = cartesianDistance(newAtoms[i], newAtoms[j], newLattice);
      const dMin = minAllowedDistance(newAtoms[i].element, newAtoms[j].element);
      if (d < dMin) return null;
    }
  }

  const vol = computeCellVolume(newLattice);
  const vpa = vol / newAtoms.length;
  const hasH = newAtoms.some(a => a.element === "H");
  if (vpa < (hasH ? 2.5 : 5.0) || vpa > 80) return null;

  return {
    ...parent,
    lattice: newLattice,
    atoms: newAtoms,
    volumePerAtom: vpa,
    generationMethod: "evo_mutation",
    noveltyScore: parent.noveltyScore * 0.9 + Math.random() * 0.1,
  };
}

function crossoverStructures(parentA: GeneratedStructure, parentB: GeneratedStructure): GeneratedStructure | null {
  if (parentA.formula !== parentB.formula) return null;
  if (parentA.atoms.length !== parentB.atoms.length) return null;

  const alpha = 0.3 + Math.random() * 0.4;
  const newLattice = clampLattice({
    a: parentA.lattice.a * alpha + parentB.lattice.a * (1 - alpha),
    b: parentA.lattice.b * alpha + parentB.lattice.b * (1 - alpha),
    c: parentA.lattice.c * alpha + parentB.lattice.c * (1 - alpha),
    alpha: parentA.lattice.alpha * alpha + parentB.lattice.alpha * (1 - alpha),
    beta: parentA.lattice.beta * alpha + parentB.lattice.beta * (1 - alpha),
    gamma: parentA.lattice.gamma * alpha + parentB.lattice.gamma * (1 - alpha),
  });

  const newAtoms: AtomPosition[] = parentA.atoms.map((a, i) => {
    const b = parentB.atoms[i];
    return {
      element: a.element,
      fx: ((a.fx * alpha + b.fx * (1 - alpha)) % 1 + 1) % 1,
      fy: ((a.fy * alpha + b.fy * (1 - alpha)) % 1 + 1) % 1,
      fz: ((a.fz * alpha + b.fz * (1 - alpha)) % 1 + 1) % 1,
    };
  });

  for (let i = 0; i < newAtoms.length; i++) {
    for (let j = i + 1; j < newAtoms.length; j++) {
      const d = cartesianDistance(newAtoms[i], newAtoms[j], newLattice);
      const dMin = minAllowedDistance(newAtoms[i].element, newAtoms[j].element);
      if (d < dMin) return null;
    }
  }

  const vol = computeCellVolume(newLattice);
  const vpa = vol / newAtoms.length;
  if (vpa < 2.5 || vpa > 80) return null;

  return {
    formula: parentA.formula,
    lattice: newLattice,
    atoms: newAtoms,
    bravaisType: parentA.bravaisType,
    volumePerAtom: vpa,
    generationMethod: "evo_crossover",
    noveltyScore: (parentA.noveltyScore + parentB.noveltyScore) / 2,
    confidence: (parentA.confidence + parentB.confidence) / 2,
  };
}

function tournamentSelect(): EvoIndividual {
  let best: EvoIndividual | null = null;
  for (let i = 0; i < TOURNAMENT_SIZE; i++) {
    const idx = Math.floor(Math.random() * evoPopulation.length);
    if (!best || evoPopulation[idx].fitness > best.fitness) {
      best = evoPopulation[idx];
    }
  }
  return best!;
}

export function seedEvoPopulation(structures: GeneratedStructure[]): void {
  for (const s of structures) {
    const fitness = computeStructureFitness(s);
    evoPopulation.push({ structure: s, fitness, generation: evoGeneration });
  }
  evoPopulation.sort((a, b) => b.fitness - a.fitness);
  if (evoPopulation.length > MAX_POPULATION) {
    evoPopulation.length = MAX_POPULATION;
  }
}

export function addToEvoPopulation(structure: GeneratedStructure, externalFitness?: number): void {
  const fitness = externalFitness ?? computeStructureFitness(structure);
  evoPopulation.push({ structure, fitness, generation: evoGeneration });
  evoPopulation.sort((a, b) => b.fitness - a.fitness);
  if (evoPopulation.length > MAX_POPULATION) {
    evoPopulation.length = MAX_POPULATION;
  }
}

export function runEvolutionaryGeneration(count: number = 10): GeneratedStructure[] {
  if (evoPopulation.length < 3) {
    return [];
  }

  evoGeneration++;
  const offspring: GeneratedStructure[] = [];
  const maxAttempts = count * 5;

  for (let attempt = 0; attempt < maxAttempts && offspring.length < count; attempt++) {
    let child: GeneratedStructure | null = null;

    if (Math.random() < 0.6 || evoPopulation.length < 4) {
      const parent = tournamentSelect();
      child = mutateStructure(parent.structure);
      if (child) evoStats.mutations++;
    } else {
      const parentA = tournamentSelect();
      let parentB = tournamentSelect();
      let tries = 0;
      while (parentB === parentA && tries < 5) {
        parentB = tournamentSelect();
        tries++;
      }
      child = crossoverStructures(parentA.structure, parentB.structure);
      if (child) evoStats.crossovers++;
    }

    if (child) {
      evoStats.totalOffspring++;
      const fitness = computeStructureFitness(child);
      offspring.push(child);

      if (evoPopulation.length < MAX_POPULATION || fitness > evoPopulation[evoPopulation.length - 1].fitness) {
        evoPopulation.push({ structure: child, fitness, generation: evoGeneration });
        evoPopulation.sort((a, b) => b.fitness - a.fitness);
        if (evoPopulation.length > MAX_POPULATION) {
          evoPopulation.length = MAX_POPULATION;
        }
        evoStats.improvements++;
      }
    }
  }

  return offspring;
}

export function getEvoPopulationSummary(): {
  size: number;
  generation: number;
  bestFitness: number;
  avgFitness: number;
  formulaDiversity: number;
  bravaisDist: Record<string, number>;
} {
  if (evoPopulation.length === 0) {
    return { size: 0, generation: evoGeneration, bestFitness: 0, avgFitness: 0, formulaDiversity: 0, bravaisDist: {} };
  }
  const formulas = new Set(evoPopulation.map(i => i.structure.formula));
  const bravaisDist: Record<string, number> = {};
  let totalFitness = 0;
  for (const ind of evoPopulation) {
    totalFitness += ind.fitness;
    bravaisDist[ind.structure.bravaisType] = (bravaisDist[ind.structure.bravaisType] || 0) + 1;
  }
  return {
    size: evoPopulation.length,
    generation: evoGeneration,
    bestFitness: evoPopulation[0].fitness,
    avgFitness: totalFitness / evoPopulation.length,
    formulaDiversity: formulas.size,
    bravaisDist,
  };
}
