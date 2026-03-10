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
  return 0.7 * (rA + rB);
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
