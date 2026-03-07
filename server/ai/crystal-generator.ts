interface AtomPosition {
  symbol: string;
  x: number;
  y: number;
  z: number;
}

interface LatticeParams {
  a: number;
  b: number;
  c: number;
  alpha: number;
  beta: number;
  gamma: number;
}

interface WyckoffSite {
  multiplicity: number;
  letter: string;
  positions: [number, number, number][];
}

export interface GeneratedCrystal {
  atoms: AtomPosition[];
  lattice: LatticeParams;
  spaceGroup: string;
  crystalSystem: string;
  formula: string;
  prototypeMatch: string | null;
  prototypeMatchScore: number;
  noveltyScore: number;
  densityGcm3: number;
  minBondLength: number;
  coordNumbers: number[];
  source: "diffusion";
}

const ATOMIC_RADII: Record<string, number> = {
  H: 0.53, He: 0.31, Li: 1.67, Be: 1.12, B: 0.87, C: 0.67, N: 0.56, O: 0.48, F: 0.42,
  Na: 1.90, Mg: 1.45, Al: 1.18, Si: 1.11, P: 0.98, S: 0.88, Cl: 0.79,
  K: 2.43, Ca: 1.94, Sc: 1.84, Ti: 1.76, V: 1.71, Cr: 1.66, Mn: 1.61, Fe: 1.56,
  Co: 1.52, Ni: 1.49, Cu: 1.45, Zn: 1.42, Ga: 1.36, Ge: 1.25, As: 1.14, Se: 1.03, Br: 0.94,
  Rb: 2.65, Sr: 2.19, Y: 2.12, Zr: 2.06, Nb: 1.98, Mo: 1.90, Ru: 1.78, Rh: 1.73, Pd: 1.69, Ag: 1.65,
  In: 1.56, Sn: 1.45, Sb: 1.33, Te: 1.23,
  Cs: 2.98, Ba: 2.53, La: 2.74, Ce: 2.70, Hf: 2.08, Ta: 2.00, W: 1.93, Re: 1.88, Os: 1.85,
  Ir: 1.80, Pt: 1.77, Au: 1.74, Tl: 1.56, Pb: 1.54, Bi: 1.43,
  Pr: 2.67, Nd: 2.64, Sm: 2.59, Gd: 2.54, Dy: 2.51, Er: 2.45, Yb: 2.40, Lu: 2.38,
};

const ATOMIC_MASSES: Record<string, number> = {
  H: 1.008, Li: 6.941, Be: 9.012, B: 10.81, C: 12.01, N: 14.01, O: 16.00, F: 19.00,
  Na: 22.99, Mg: 24.31, Al: 26.98, Si: 28.09, P: 30.97, S: 32.07, Cl: 35.45,
  K: 39.10, Ca: 40.08, Sc: 44.96, Ti: 47.87, V: 50.94, Cr: 52.00, Mn: 54.94, Fe: 55.85,
  Co: 58.93, Ni: 58.69, Cu: 63.55, Zn: 65.38, Ga: 69.72, Ge: 72.63, As: 74.92, Se: 78.97, Br: 79.90,
  Rb: 85.47, Sr: 87.62, Y: 88.91, Zr: 91.22, Nb: 92.91, Mo: 95.95, Ru: 101.1, Rh: 102.9, Pd: 106.4, Ag: 107.9,
  In: 114.8, Sn: 118.7, Sb: 121.8, Te: 127.6,
  Cs: 132.9, Ba: 137.3, La: 138.9, Ce: 140.1, Hf: 178.5, Ta: 180.9, W: 183.8, Re: 186.2, Os: 190.2,
  Ir: 192.2, Pt: 195.1, Au: 197.0, Tl: 204.4, Pb: 207.2, Bi: 209.0,
  Pr: 140.9, Nd: 144.2, Sm: 150.4, Gd: 157.3, Dy: 162.5, Er: 167.3, Yb: 173.0, Lu: 175.0,
};

const SPACE_GROUPS: {
  name: string;
  number: number;
  system: string;
  wyckoff: WyckoffSite[];
}[] = [
  {
    name: "Pm-3m", number: 221, system: "cubic",
    wyckoff: [
      { multiplicity: 1, letter: "a", positions: [[0, 0, 0]] },
      { multiplicity: 1, letter: "b", positions: [[0.5, 0.5, 0.5]] },
      { multiplicity: 3, letter: "c", positions: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]] },
      { multiplicity: 3, letter: "d", positions: [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]] },
      { multiplicity: 6, letter: "e", positions: [[0.25, 0, 0.5], [0.75, 0, 0.5], [0.5, 0.25, 0], [0.5, 0.75, 0], [0, 0.5, 0.25], [0, 0.5, 0.75]] },
    ],
  },
  {
    name: "Fm-3m", number: 225, system: "cubic",
    wyckoff: [
      { multiplicity: 4, letter: "a", positions: [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]] },
      { multiplicity: 4, letter: "b", positions: [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]] },
      { multiplicity: 8, letter: "c", positions: [[0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75], [0.75, 0.75, 0.75], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.75, 0.25, 0.25]] },
    ],
  },
  {
    name: "Im-3m", number: 229, system: "cubic",
    wyckoff: [
      { multiplicity: 2, letter: "a", positions: [[0, 0, 0], [0.5, 0.5, 0.5]] },
      { multiplicity: 6, letter: "b", positions: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]] },
      { multiplicity: 8, letter: "c", positions: [[0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75], [0.75, 0.75, 0.75], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.75, 0.25, 0.25]] },
    ],
  },
  {
    name: "P6/mmm", number: 191, system: "hexagonal",
    wyckoff: [
      { multiplicity: 1, letter: "a", positions: [[0, 0, 0]] },
      { multiplicity: 2, letter: "c", positions: [[1/3, 2/3, 0], [2/3, 1/3, 0]] },
      { multiplicity: 2, letter: "d", positions: [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5]] },
      { multiplicity: 3, letter: "f", positions: [[0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]] },
      { multiplicity: 3, letter: "g", positions: [[0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]] },
    ],
  },
  {
    name: "P4/mmm", number: 123, system: "tetragonal",
    wyckoff: [
      { multiplicity: 1, letter: "a", positions: [[0, 0, 0]] },
      { multiplicity: 1, letter: "b", positions: [[0, 0, 0.5]] },
      { multiplicity: 1, letter: "c", positions: [[0.5, 0.5, 0]] },
      { multiplicity: 1, letter: "d", positions: [[0.5, 0.5, 0.5]] },
      { multiplicity: 2, letter: "e", positions: [[0, 0.5, 0], [0.5, 0, 0]] },
      { multiplicity: 2, letter: "f", positions: [[0, 0.5, 0.5], [0.5, 0, 0.5]] },
      { multiplicity: 4, letter: "i", positions: [[0, 0.5, 0.25], [0.5, 0, 0.25], [0, 0.5, 0.75], [0.5, 0, 0.75]] },
    ],
  },
  {
    name: "I4/mmm", number: 139, system: "tetragonal",
    wyckoff: [
      { multiplicity: 2, letter: "a", positions: [[0, 0, 0], [0.5, 0.5, 0.5]] },
      { multiplicity: 2, letter: "b", positions: [[0, 0, 0.5], [0.5, 0.5, 0]] },
      { multiplicity: 4, letter: "d", positions: [[0, 0.5, 0.25], [0.5, 0, 0.25], [0.5, 0, 0.75], [0, 0.5, 0.75]] },
      { multiplicity: 4, letter: "e", positions: [[0, 0, 0.33], [0, 0, 0.67], [0.5, 0.5, 0.83], [0.5, 0.5, 0.17]] },
    ],
  },
  {
    name: "R-3m", number: 166, system: "trigonal",
    wyckoff: [
      { multiplicity: 3, letter: "a", positions: [[0, 0, 0], [1/3, 2/3, 2/3], [2/3, 1/3, 1/3]] },
      { multiplicity: 3, letter: "b", positions: [[0, 0, 0.5], [1/3, 2/3, 1/6], [2/3, 1/3, 5/6]] },
      { multiplicity: 6, letter: "c", positions: [[0, 0, 0.25], [0, 0, 0.75], [1/3, 2/3, 0.917], [1/3, 2/3, 0.417], [2/3, 1/3, 0.583], [2/3, 1/3, 0.083]] },
    ],
  },
  {
    name: "Pnma", number: 62, system: "orthorhombic",
    wyckoff: [
      { multiplicity: 4, letter: "a", positions: [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0], [0.5, 0.5, 0.5]] },
      { multiplicity: 4, letter: "c", positions: [[0.25, 0.25, 0], [0.75, 0.75, 0], [0.75, 0.25, 0.5], [0.25, 0.75, 0.5]] },
    ],
  },
];

const SC_ELEMENT_POOL = {
  highCoupling: ["Nb", "V", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "Re"],
  lightPhonon: ["B", "C", "N", "O", "H", "Si"],
  chargeReservoir: ["La", "Y", "Ca", "Sr", "Ba", "K", "Rb", "Cs"],
  pBlockMetal: ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Ge", "Sb", "Te"],
  magnetic: ["Fe", "Co", "Ni", "Cu", "Mn", "Cr"],
  chalcogen: ["S", "Se", "Te", "O"],
  pnictogen: ["N", "P", "As", "Sb", "Bi"],
};

function parseFormulaCounts(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + count;
  }
  return counts;
}

function buildFormula(elements: Record<string, number>): string {
  const sorted = Object.entries(elements).sort((a, b) => {
    const order = ["La", "Y", "Ca", "Sr", "Ba", "Cs", "Rb", "K", "Na", "Li",
      "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
      "Nb", "Mo", "Zr", "Hf", "Ta", "W", "Re",
      "Al", "Ga", "In", "Sn", "Pb", "Bi",
      "B", "C", "N", "O", "Si", "P", "S", "Se", "Te", "H", "F"];
    const ia = order.indexOf(a[0]);
    const ib = order.indexOf(b[0]);
    return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
  });
  return sorted.map(([el, cnt]) => cnt === 1 ? el : `${el}${cnt}`).join("");
}

function gaussRandom(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function estimateLatticeParam(elements: string[]): number {
  const radii = elements.map(el => ATOMIC_RADII[el] || 1.5);
  const avgR = radii.reduce((a, b) => a + b, 0) / radii.length;
  return avgR * 2.6 + 0.8 + gaussRandom() * 0.3;
}

function minimumImageDist(
  dx: number, dy: number, dz: number,
  la: number, lb: number, lc: number
): [number, number, number] {
  if (dx > la / 2) dx -= la; else if (dx < -la / 2) dx += la;
  if (dy > lb / 2) dy -= lb; else if (dy < -lb / 2) dy += lb;
  if (dz > lc / 2) dz -= lc; else if (dz < -lc / 2) dz += lc;
  return [dx, dy, dz];
}

function geometricPenalty(atoms: AtomPosition[], lattice: LatticeParams): number {
  let penalty = 0;
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const [dx, dy, dz] = minimumImageDist(
        atoms[i].x - atoms[j].x, atoms[i].y - atoms[j].y, atoms[i].z - atoms[j].z,
        lattice.a, lattice.b, lattice.c
      );
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const ri = ATOMIC_RADII[atoms[i].symbol] || 1.5;
      const rj = ATOMIC_RADII[atoms[j].symbol] || 1.5;
      const minDist = (ri + rj) * 0.6;
      if (r < minDist) {
        penalty += (minDist - r) * 10;
      }
      const idealDist = (ri + rj) * 0.95;
      penalty += Math.abs(r - idealDist) * 0.1;
    }
  }
  return penalty;
}

function bondLengthScore(atoms: AtomPosition[]): number {
  let score = 0;
  for (let i = 0; i < atoms.length; i++) {
    let nearestDist = Infinity;
    for (let j = 0; j < atoms.length; j++) {
      if (i === j) continue;
      const dx = atoms[i].x - atoms[j].x;
      const dy = atoms[i].y - atoms[j].y;
      const dz = atoms[i].z - atoms[j].z;
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (r < nearestDist) nearestDist = r;
    }
    const ri = ATOMIC_RADII[atoms[i].symbol] || 1.5;
    const idealNN = ri * 1.8;
    score += Math.abs(nearestDist - idealNN);
  }
  return score;
}

function symmetryScore(atoms: AtomPosition[], lattice: LatticeParams): number {
  let symPenalty = 0;
  const cx = lattice.a / 2;
  const cy = lattice.b / 2;
  const cz = lattice.c / 2;
  for (const atom of atoms) {
    const invX = 2 * cx - atom.x;
    const invY = 2 * cy - atom.y;
    const invZ = 2 * cz - atom.z;
    let minDist = Infinity;
    for (const other of atoms) {
      if (other.symbol !== atom.symbol) continue;
      const dx = invX - other.x;
      const dy = invY - other.y;
      const dz = invZ - other.z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < minDist) minDist = d;
    }
    symPenalty += Math.min(minDist, 2.0) * 0.05;
  }
  return symPenalty;
}

function denoiseStep(
  atoms: AtomPosition[],
  lattice: LatticeParams,
  stepSize: number,
  temperature: number
): AtomPosition[] {
  const newAtoms: AtomPosition[] = atoms.map(a => ({ ...a }));

  for (let i = 0; i < newAtoms.length; i++) {
    let fx = 0, fy = 0, fz = 0;

    for (let j = 0; j < newAtoms.length; j++) {
      if (i === j) continue;
      const [dx, dy, dz] = minimumImageDist(
        newAtoms[i].x - newAtoms[j].x,
        newAtoms[i].y - newAtoms[j].y,
        newAtoms[i].z - newAtoms[j].z,
        lattice.a, lattice.b, lattice.c
      );
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01;
      const ri = ATOMIC_RADII[newAtoms[i].symbol] || 1.5;
      const rj = ATOMIC_RADII[newAtoms[j].symbol] || 1.5;
      const sigma = (ri + rj) * 0.9;

      const ljForce = 24 * (2 * Math.pow(sigma / r, 13) - Math.pow(sigma / r, 7)) / r;
      fx += ljForce * dx / r;
      fy += ljForce * dy / r;
      fz += ljForce * dz / r;
    }

    const wallMargin = 0.5;
    if (newAtoms[i].x < wallMargin) fx += 2 * (wallMargin - newAtoms[i].x);
    if (newAtoms[i].x > lattice.a - wallMargin) fx -= 2 * (newAtoms[i].x - (lattice.a - wallMargin));
    if (newAtoms[i].y < wallMargin) fy += 2 * (wallMargin - newAtoms[i].y);
    if (newAtoms[i].y > lattice.b - wallMargin) fy -= 2 * (newAtoms[i].y - (lattice.b - wallMargin));
    if (newAtoms[i].z < wallMargin) fz += 2 * (wallMargin - newAtoms[i].z);
    if (newAtoms[i].z > lattice.c - wallMargin) fz -= 2 * (newAtoms[i].z - (lattice.c - wallMargin));

    const noise = temperature;
    newAtoms[i].x += stepSize * fx + noise * gaussRandom() * 0.1;
    newAtoms[i].y += stepSize * fy + noise * gaussRandom() * 0.1;
    newAtoms[i].z += stepSize * fz + noise * gaussRandom() * 0.1;

    newAtoms[i].x = Math.max(0.1, Math.min(lattice.a - 0.1, newAtoms[i].x));
    newAtoms[i].y = Math.max(0.1, Math.min(lattice.b - 0.1, newAtoms[i].y));
    newAtoms[i].z = Math.max(0.1, Math.min(lattice.c - 0.1, newAtoms[i].z));
  }

  return newAtoms;
}

function generateFromWyckoff(
  elements: Record<string, number>,
  sgIdx: number
): { atoms: AtomPosition[]; lattice: LatticeParams } | null {
  const sg = SPACE_GROUPS[sgIdx];
  const elList = Object.entries(elements);
  const totalAtoms = elList.reduce((s, [, cnt]) => s + Math.round(cnt), 0);

  if (totalAtoms < 2 || totalAtoms > 20) return null;

  const allElements = elList.flatMap(([el, cnt]) => Array(Math.round(cnt)).fill(el));

  const a = estimateLatticeParam(allElements);
  const lattice: LatticeParams = {
    a,
    b: sg.system === "cubic" ? a : a * (0.9 + Math.random() * 0.3),
    c: sg.system === "cubic" ? a : a * (0.8 + Math.random() * 0.8),
    alpha: sg.system === "cubic" || sg.system === "tetragonal" || sg.system === "orthorhombic" ? 90 : 90 + gaussRandom() * 5,
    beta: sg.system === "cubic" || sg.system === "tetragonal" || sg.system === "hexagonal" ? 90 : 90 + gaussRandom() * 5,
    gamma: sg.system === "hexagonal" || sg.system === "trigonal" ? 120 : 90,
  };

  const atoms: AtomPosition[] = [];
  let elIdx = 0;

  const usableSites = sg.wyckoff.filter(site => {
    const remaining = totalAtoms - elIdx;
    if (remaining <= 0) return false;
    return site.multiplicity <= remaining;
  });

  for (const site of (usableSites.length > 0 ? usableSites : sg.wyckoff)) {
    if (elIdx >= allElements.length) break;

    if (allElements.length - elIdx >= site.multiplicity) {
      const siteElement = allElements[elIdx];
      for (let i = 0; i < site.positions.length && elIdx < allElements.length; i++) {
        const [fx, fy, fz] = site.positions[i];
        atoms.push({
          symbol: siteElement,
          x: fx * lattice.a + gaussRandom() * 0.05,
          y: fy * lattice.b + gaussRandom() * 0.05,
          z: fz * lattice.c + gaussRandom() * 0.05,
        });
        elIdx++;
      }
    } else {
      for (let i = 0; i < site.positions.length && elIdx < allElements.length; i++) {
        const [fx, fy, fz] = site.positions[i];
        atoms.push({
          symbol: allElements[elIdx],
          x: fx * lattice.a + gaussRandom() * 0.05,
          y: fy * lattice.b + gaussRandom() * 0.05,
          z: fz * lattice.c + gaussRandom() * 0.05,
        });
        elIdx++;
      }
    }
  }

  while (elIdx < allElements.length) {
    atoms.push({
      symbol: allElements[elIdx],
      x: Math.random() * lattice.a,
      y: Math.random() * lattice.b,
      z: Math.random() * lattice.c,
    });
    elIdx++;
  }

  return { atoms, lattice };
}

function runDiffusion(
  atoms: AtomPosition[],
  lattice: LatticeParams,
  nSteps: number = 50
): AtomPosition[] {
  let current = atoms.map(a => ({ ...a }));
  let bestAtoms = current;
  let bestScore = Infinity;

  for (let step = 0; step < nSteps; step++) {
    const t = 1 - step / nSteps;
    const temperature = t * 0.5;
    const stepSize = 0.01 * (1 + t * 2);

    current = denoiseStep(current, lattice, stepSize, temperature);

    const score = geometricPenalty(current, lattice) + bondLengthScore(current) + symmetryScore(current, lattice);
    if (score < bestScore) {
      bestScore = score;
      bestAtoms = current.map(a => ({ ...a }));
    }
  }

  return bestAtoms;
}

function computeCoordinationNumbers(atoms: AtomPosition[], cutoff: number = 3.5): number[] {
  return atoms.map((atom, i) => {
    let count = 0;
    for (let j = 0; j < atoms.length; j++) {
      if (i === j) continue;
      const dx = atom.x - atoms[j].x;
      const dy = atom.y - atoms[j].y;
      const dz = atom.z - atoms[j].z;
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (r < cutoff) count++;
    }
    return count;
  });
}

function computeMinBondLength(atoms: AtomPosition[]): number {
  let minDist = Infinity;
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dx = atoms[i].x - atoms[j].x;
      const dy = atoms[i].y - atoms[j].y;
      const dz = atoms[i].z - atoms[j].z;
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (r < minDist) minDist = r;
    }
  }
  return minDist;
}

function computeDensity(atoms: AtomPosition[], lattice: LatticeParams): number {
  const totalMass = atoms.reduce((s, a) => s + (ATOMIC_MASSES[a.symbol] || 50), 0);
  const volume = lattice.a * lattice.b * lattice.c * 1e-24;
  return (totalMass / 6.022e23) / volume;
}

function matchPrototype(atoms: AtomPosition[], lattice: LatticeParams): { name: string; score: number } {
  const n = atoms.length;
  const coords = computeCoordinationNumbers(atoms);
  const avgCoord = coords.reduce((a, b) => a + b, 0) / coords.length;
  const ratio = lattice.c / lattice.a;

  const prototypes: { name: string; coordRange: [number, number]; ratioRange: [number, number]; atomRange: [number, number] }[] = [
    { name: "Perovskite", coordRange: [4, 8], ratioRange: [0.9, 1.1], atomRange: [4, 6] },
    { name: "A15", coordRange: [8, 14], ratioRange: [0.95, 1.05], atomRange: [6, 10] },
    { name: "NaCl", coordRange: [4, 7], ratioRange: [0.95, 1.05], atomRange: [2, 8] },
    { name: "AlB2", coordRange: [3, 8], ratioRange: [0.5, 0.9], atomRange: [3, 6] },
    { name: "ThCr2Si2", coordRange: [4, 10], ratioRange: [1.5, 3.5], atomRange: [4, 10] },
    { name: "Fluorite", coordRange: [4, 8], ratioRange: [0.9, 1.1], atomRange: [3, 6] },
    { name: "Heusler", coordRange: [6, 14], ratioRange: [0.95, 1.05], atomRange: [4, 16] },
    { name: "Layered", coordRange: [3, 6], ratioRange: [2.0, 6.0], atomRange: [2, 12] },
    { name: "Kagome", coordRange: [4, 6], ratioRange: [0.5, 1.0], atomRange: [3, 12] },
    { name: "Clathrate", coordRange: [8, 16], ratioRange: [0.95, 1.05], atomRange: [8, 20] },
  ];

  let bestMatch = { name: "novel", score: 0 };
  for (const proto of prototypes) {
    let score = 0;
    if (avgCoord >= proto.coordRange[0] && avgCoord <= proto.coordRange[1]) score += 0.4;
    if (ratio >= proto.ratioRange[0] && ratio <= proto.ratioRange[1]) score += 0.3;
    if (n >= proto.atomRange[0] && n <= proto.atomRange[1]) score += 0.3;
    if (score > bestMatch.score) {
      bestMatch = { name: proto.name, score };
    }
  }

  return bestMatch;
}

function sampleComposition(targetElements?: string[]): Record<string, number> {
  const strategies = [
    () => sampleHydride(targetElements),
    () => sampleTernary(targetElements),
    () => sampleBinaryTM(targetElements),
    () => sampleQuaternary(targetElements),
    () => sampleExotic(targetElements),
    () => sampleCageCompound(targetElements),
    () => sampleLayered(targetElements),
    () => sampleBorocarbide(targetElements),
  ];

  const strategy = strategies[Math.floor(Math.random() * strategies.length)];
  return strategy();
}

function pick(arr: string[]): string {
  return arr[Math.floor(Math.random() * arr.length)];
}

function sampleHydride(targetElements?: string[]): Record<string, number> {
  const metal = targetElements?.[0] || pick(["La", "Y", "Ca", "Sr", "Ba", "Sc", "Ce", "Th", "Pr", "Nd"]);
  const hCount = 6 + Math.floor(Math.random() * 8);
  return { [metal]: 1, H: hCount };
}

function sampleTernary(targetElements?: string[]): Record<string, number> {
  const tm = targetElements?.[0] || pick(SC_ELEMENT_POOL.highCoupling);
  const light = targetElements?.[1] || pick(SC_ELEMENT_POOL.lightPhonon);
  const reservoir = pick(SC_ELEMENT_POOL.chargeReservoir);
  const counts = [
    { [reservoir]: 1, [tm]: 2, [light]: 2 },
    { [reservoir]: 1, [tm]: 1, [light]: 3 },
    { [reservoir]: 2, [tm]: 1, [light]: 1 },
    { [reservoir]: 1, [tm]: 3, [light]: 1 },
  ];
  return counts[Math.floor(Math.random() * counts.length)];
}

function sampleBinaryTM(targetElements?: string[]): Record<string, number> {
  const tm = targetElements?.[0] || pick(SC_ELEMENT_POOL.highCoupling);
  const light = targetElements?.[1] || pick(SC_ELEMENT_POOL.lightPhonon);
  const a = 1 + Math.floor(Math.random() * 3);
  const b = 1 + Math.floor(Math.random() * 4);
  return { [tm]: a, [light]: b };
}

function sampleQuaternary(targetElements?: string[]): Record<string, number> {
  const tm = targetElements?.[0] || pick(SC_ELEMENT_POOL.highCoupling);
  const light1 = pick(SC_ELEMENT_POOL.lightPhonon);
  const light2 = pick(SC_ELEMENT_POOL.lightPhonon.filter(e => e !== light1));
  const reservoir = pick(SC_ELEMENT_POOL.chargeReservoir);
  return { [reservoir]: 1, [tm]: 2, [light1]: 1, [light2]: 1 };
}

function sampleExotic(targetElements?: string[]): Record<string, number> {
  const groups = [SC_ELEMENT_POOL.highCoupling, SC_ELEMENT_POOL.pBlockMetal, SC_ELEMENT_POOL.magnetic, SC_ELEMENT_POOL.chalcogen];
  const g1 = groups[Math.floor(Math.random() * groups.length)];
  const g2 = groups[Math.floor(Math.random() * groups.length)];
  const el1 = targetElements?.[0] || pick(g1);
  const el2 = pick(g2.filter(e => e !== el1));
  if (!el2) return { [el1]: 3, [pick(SC_ELEMENT_POOL.lightPhonon)]: 1 };
  return { [el1]: 2, [el2]: 1, [pick(SC_ELEMENT_POOL.lightPhonon)]: 1 };
}

function sampleCageCompound(targetElements?: string[]): Record<string, number> {
  const cage = targetElements?.[0] || pick(["B", "Si", "Ge", "C"]);
  const guest = pick(SC_ELEMENT_POOL.chargeReservoir);
  return { [guest]: 2, [cage]: 6 };
}

function sampleLayered(targetElements?: string[]): Record<string, number> {
  const tm = targetElements?.[0] || pick(SC_ELEMENT_POOL.magnetic.concat(SC_ELEMENT_POOL.highCoupling));
  const chalcogen = pick(SC_ELEMENT_POOL.chalcogen);
  const spacer = pick(SC_ELEMENT_POOL.chargeReservoir);
  return { [spacer]: 1, [tm]: 1, [chalcogen]: 2 };
}

function sampleBorocarbide(targetElements?: string[]): Record<string, number> {
  const tm = targetElements?.[0] || pick(["Ni", "Co", "Pd", "Pt"]);
  const re = pick(["Y", "La", "Lu", "Sc"]);
  return { [re]: 1, [tm]: 2, B: 2, C: 1 };
}

function isPhysicallyValid(atoms: AtomPosition[], lattice: LatticeParams): boolean {
  const minBond = computeMinBondLength(atoms);
  if (minBond < 0.5) return false;

  const density = computeDensity(atoms, lattice);
  if (density < 0.5 || density > 25) return false;

  if (lattice.a < 2 || lattice.a > 20) return false;
  if (lattice.b < 2 || lattice.b > 20) return false;
  if (lattice.c < 2 || lattice.c > 30) return false;

  return true;
}

function computeNoveltyScore(formula: string, protoMatch: string): number {
  let novelty = 0.5;

  if (protoMatch === "novel") {
    novelty += 0.3;
  }

  const parsed = parseFormulaCounts(formula);
  const elements = Object.keys(parsed);

  const unusualPairs = [
    ["La", "H"], ["Y", "H"], ["Nb", "Ge"], ["Fe", "Se"],
    ["Ba", "Cu"], ["Sr", "Ru"], ["Bi", "S"], ["Ta", "N"],
  ];
  for (const [a, b] of unusualPairs) {
    if (elements.includes(a) && elements.includes(b)) {
      novelty += 0.1;
      break;
    }
  }

  if (elements.length >= 4) novelty += 0.1;

  const totalAtoms = Object.values(parsed).reduce((a, b) => a + b, 0);
  const maxRatio = Math.max(...Object.values(parsed)) / totalAtoms;
  if (maxRatio > 0.7) novelty -= 0.1;

  return Math.max(0, Math.min(1, novelty));
}

const globalFormulaHistory = new Set<string>();
const HISTORY_MAX = 2000;

function trackFormula(formula: string): boolean {
  if (globalFormulaHistory.has(formula)) return false;
  globalFormulaHistory.add(formula);
  if (globalFormulaHistory.size > HISTORY_MAX) {
    const iter = globalFormulaHistory.values();
    for (let i = 0; i < 500; i++) {
      globalFormulaHistory.delete(iter.next().value);
    }
  }
  return true;
}

export function generateCrystals(
  count: number = 30,
  targetElements?: string[],
  targetTcHint?: number
): GeneratedCrystal[] {
  const results: GeneratedCrystal[] = [];
  const seenFormulas = new Set<string>();
  const maxAttempts = count * 4;

  for (let attempt = 0; attempt < maxAttempts && results.length < count; attempt++) {
    try {
      const composition = sampleComposition(targetElements);
      const formula = buildFormula(composition);

      if (seenFormulas.has(formula)) continue;
      seenFormulas.add(formula);
      if (!trackFormula(formula)) continue;

      const sgIdx = Math.floor(Math.random() * SPACE_GROUPS.length);
      const generated = generateFromWyckoff(composition, sgIdx);
      if (!generated) continue;

      let { atoms, lattice } = generated;

      atoms = runDiffusion(atoms, lattice, 40);

      if (!isPhysicallyValid(atoms, lattice)) continue;

      const protoResult = matchPrototype(atoms, lattice);
      const coordNumbers = computeCoordinationNumbers(atoms);
      const minBond = computeMinBondLength(atoms);
      const density = computeDensity(atoms, lattice);
      const novelty = computeNoveltyScore(formula, protoResult.name);

      results.push({
        atoms,
        lattice,
        spaceGroup: SPACE_GROUPS[sgIdx].name,
        crystalSystem: SPACE_GROUPS[sgIdx].system,
        formula,
        prototypeMatch: protoResult.score > 0.5 ? protoResult.name : null,
        prototypeMatchScore: protoResult.score,
        noveltyScore: novelty,
        densityGcm3: Math.round(density * 100) / 100,
        minBondLength: Math.round(minBond * 1000) / 1000,
        coordNumbers,
        source: "diffusion",
      });
    } catch {
      continue;
    }
  }

  return results;
}

export function generateCrystalsForPipeline(
  count: number = 30,
  targetElements?: string[]
): { formula: string; structure: GeneratedCrystal }[] {
  const crystals = generateCrystals(count, targetElements);
  return crystals.map(c => ({
    formula: c.formula,
    structure: c,
  }));
}

let totalGenerated = 0;
let totalValid = 0;
let totalNovel = 0;

export function runDiffusionGenerationCycle(count: number = 30, targetElements?: string[]): {
  formulas: string[];
  structures: GeneratedCrystal[];
  stats: {
    attempted: number;
    valid: number;
    novel: number;
    avgNovelty: number;
    avgDensity: number;
    protoBreakdown: Record<string, number>;
  };
} {
  const crystals = generateCrystals(count, targetElements);
  totalGenerated += count * 4;
  totalValid += crystals.length;

  const novel = crystals.filter(c => c.prototypeMatch === null);
  totalNovel += novel.length;

  const protoBreakdown: Record<string, number> = {};
  for (const c of crystals) {
    const key = c.prototypeMatch || "novel";
    protoBreakdown[key] = (protoBreakdown[key] || 0) + 1;
  }

  const avgNovelty = crystals.length > 0
    ? crystals.reduce((s, c) => s + c.noveltyScore, 0) / crystals.length
    : 0;
  const avgDensity = crystals.length > 0
    ? crystals.reduce((s, c) => s + c.densityGcm3, 0) / crystals.length
    : 0;

  return {
    formulas: crystals.map(c => c.formula),
    structures: crystals,
    stats: {
      attempted: count * 4,
      valid: crystals.length,
      novel: novel.length,
      avgNovelty: Math.round(avgNovelty * 1000) / 1000,
      avgDensity: Math.round(avgDensity * 100) / 100,
      protoBreakdown,
    },
  };
}

export function getDiffusionStats() {
  return {
    totalGenerated,
    totalValid,
    totalNovel,
    novelRate: totalValid > 0 ? Math.round(totalNovel / totalValid * 1000) / 1000 : 0,
    validRate: totalGenerated > 0 ? Math.round(totalValid / totalGenerated * 1000) / 1000 : 0,
  };
}
