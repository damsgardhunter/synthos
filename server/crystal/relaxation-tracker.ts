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
  x: number;
  y: number;
  z: number;
}

export interface RelaxationEntry {
  formula: string;
  beforeLattice: LatticeParams;
  afterLattice: LatticeParams;
  beforePositions?: AtomPosition[];
  afterPositions?: AtomPosition[];
  energyBefore?: number;
  energyAfter?: number;
  pressureBefore?: number;
  pressureAfter?: number;
  relaxationStrain: number;
  volumeChange: number;
  forcesConverged: boolean;
  relaxedAt: number;
  tier: string;
  crystalSystem?: string;
  prototype?: string;
}

const MAX_ENTRIES = 5000;
const relaxationDB: RelaxationEntry[] = [];

function latticeVolume(l: LatticeParams): number {
  const toRad = Math.PI / 180;
  const cosA = Math.cos(l.alpha * toRad);
  const cosB = Math.cos(l.beta * toRad);
  const cosG = Math.cos(l.gamma * toRad);
  const factor = Math.sqrt(1 - cosA * cosA - cosB * cosB - cosG * cosG + 2 * cosA * cosB * cosG);
  return l.a * l.b * l.c * factor;
}

export function computeRelaxationStrain(before: LatticeParams, after: LatticeParams): number {
  const strainA = (after.a - before.a) / before.a;
  const strainB = (after.b - before.b) / before.b;
  const strainC = (after.c - before.c) / before.c;
  const magnitude = Math.sqrt(strainA * strainA + strainB * strainB + strainC * strainC);
  return magnitude;
}

export function recordRelaxation(entry: Omit<RelaxationEntry, "relaxationStrain" | "volumeChange">): RelaxationEntry {
  const strain = computeRelaxationStrain(entry.beforeLattice, entry.afterLattice);
  const volBefore = latticeVolume(entry.beforeLattice);
  const volAfter = latticeVolume(entry.afterLattice);
  const volumeChange = volBefore > 0 ? ((volAfter - volBefore) / volBefore) * 100 : 0;

  const full: RelaxationEntry = {
    ...entry,
    relaxationStrain: strain,
    volumeChange,
  };

  if (relaxationDB.length >= MAX_ENTRIES) {
    relaxationDB.splice(0, Math.floor(MAX_ENTRIES * 0.1));
  }
  relaxationDB.push(full);
  return full;
}

export function getRelaxationPatterns(): {
  byCrystalSystem: Record<string, { count: number; avgStrain: number; avgVolumeChange: number }>;
  byPrototype: Record<string, { count: number; avgStrain: number; avgVolumeChange: number }>;
  largestRelaxations: Array<{ formula: string; strain: number; volumeChange: number }>;
  elementEffects: Record<string, { count: number; avgStrain: number }>;
} {
  const byCrystalSystem: Record<string, { count: number; strainSum: number; volSum: number }> = {};
  const byPrototype: Record<string, { count: number; strainSum: number; volSum: number }> = {};
  const elementCounts: Record<string, { count: number; strainSum: number }> = {};

  for (const e of relaxationDB) {
    const cs = e.crystalSystem || "unknown";
    if (!byCrystalSystem[cs]) byCrystalSystem[cs] = { count: 0, strainSum: 0, volSum: 0 };
    byCrystalSystem[cs].count++;
    byCrystalSystem[cs].strainSum += e.relaxationStrain;
    byCrystalSystem[cs].volSum += e.volumeChange;

    const proto = e.prototype || "unknown";
    if (!byPrototype[proto]) byPrototype[proto] = { count: 0, strainSum: 0, volSum: 0 };
    byPrototype[proto].count++;
    byPrototype[proto].strainSum += e.relaxationStrain;
    byPrototype[proto].volSum += e.volumeChange;

    const elements = extractElements(e.formula);
    for (const el of elements) {
      if (!elementCounts[el]) elementCounts[el] = { count: 0, strainSum: 0 };
      elementCounts[el].count++;
      elementCounts[el].strainSum += e.relaxationStrain;
    }
  }

  const formatGroup = (g: Record<string, { count: number; strainSum: number; volSum: number }>) => {
    const result: Record<string, { count: number; avgStrain: number; avgVolumeChange: number }> = {};
    for (const [k, v] of Object.entries(g)) {
      result[k] = {
        count: v.count,
        avgStrain: v.count > 0 ? v.strainSum / v.count : 0,
        avgVolumeChange: v.count > 0 ? v.volSum / v.count : 0,
      };
    }
    return result;
  };

  const elementEffects: Record<string, { count: number; avgStrain: number }> = {};
  for (const [el, v] of Object.entries(elementCounts)) {
    elementEffects[el] = { count: v.count, avgStrain: v.count > 0 ? v.strainSum / v.count : 0 };
  }

  const sorted = [...relaxationDB].sort((a, b) => b.relaxationStrain - a.relaxationStrain);
  const largestRelaxations = sorted.slice(0, 10).map(e => ({
    formula: e.formula,
    strain: e.relaxationStrain,
    volumeChange: e.volumeChange,
  }));

  return {
    byCrystalSystem: formatGroup(byCrystalSystem),
    byPrototype: formatGroup(byPrototype),
    largestRelaxations,
    elementEffects,
  };
}

export function predictRelaxationMagnitude(formula: string, lattice: LatticeParams): {
  predictedStrain: number;
  predictedVolumeChange: number;
  confidence: number;
  basedOnCount: number;
} {
  const elements = extractElements(formula);
  const relevant = relaxationDB.filter(e => {
    const eElements = extractElements(e.formula);
    return elements.some(el => eElements.includes(el));
  });

  if (relevant.length === 0) {
    return { predictedStrain: 0.05, predictedVolumeChange: 2.0, confidence: 0.1, basedOnCount: 0 };
  }

  let strainSum = 0;
  let volSum = 0;
  let weightSum = 0;

  for (const entry of relevant) {
    const eElements = extractElements(entry.formula);
    const overlap = elements.filter(el => eElements.includes(el)).length;
    const weight = overlap / Math.max(elements.length, eElements.length);

    const latticeSim = 1 / (1 + Math.abs(entry.beforeLattice.a - lattice.a) / lattice.a);

    const w = weight * latticeSim;
    strainSum += entry.relaxationStrain * w;
    volSum += entry.volumeChange * w;
    weightSum += w;
  }

  const predictedStrain = weightSum > 0 ? strainSum / weightSum : 0.05;
  const predictedVolumeChange = weightSum > 0 ? volSum / weightSum : 2.0;
  const confidence = Math.min(1, relevant.length / 20);

  return { predictedStrain, predictedVolumeChange, confidence, basedOnCount: relevant.length };
}

export function getRelaxationStats(): {
  totalEntries: number;
  maxEntries: number;
  avgStrain: number;
  avgVolumeChange: number;
  byCrystalSystem: Record<string, number>;
  byTier: Record<string, number>;
  convergenceRate: number;
} {
  const n = relaxationDB.length || 1;
  let strainSum = 0;
  let volSum = 0;
  let convergedCount = 0;
  const byCrystalSystem: Record<string, number> = {};
  const byTier: Record<string, number> = {};

  for (const e of relaxationDB) {
    strainSum += e.relaxationStrain;
    volSum += e.volumeChange;
    if (e.forcesConverged) convergedCount++;

    const cs = e.crystalSystem || "unknown";
    byCrystalSystem[cs] = (byCrystalSystem[cs] || 0) + 1;

    byTier[e.tier] = (byTier[e.tier] || 0) + 1;
  }

  return {
    totalEntries: relaxationDB.length,
    maxEntries: MAX_ENTRIES,
    avgStrain: strainSum / n,
    avgVolumeChange: volSum / n,
    byCrystalSystem,
    byTier,
    convergenceRate: relaxationDB.length > 0 ? convergedCount / relaxationDB.length : 0,
  };
}

export function getRelaxationEntry(formula: string): RelaxationEntry[] {
  return relaxationDB.filter(e => e.formula === formula);
}

function extractElements(formula: string): string[] {
  const matches = formula.match(/[A-Z][a-z]?/g);
  if (!matches) return [];
  const seen: Record<string, boolean> = {};
  const result: string[] = [];
  for (const m of matches) {
    if (!seen[m]) {
      seen[m] = true;
      result.push(m);
    }
  }
  return result;
}
