import { computeCompositionFeatures, compositionFeatureVector } from "../learning/composition-features";

export type FailureReason =
  | "unstable_phonons"
  | "structure_collapse"
  | "high_formation_energy"
  | "non_metallic"
  | "scf_divergence"
  | "geometry_rejected";

export type FailureSource = "dft" | "xtb" | "pipeline" | "phonon_surrogate" | "learning_loop";

export interface StructureFailureEntry {
  formula: string;
  lattice?: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  atomicPositions?: { element: string; x: number; y: number; z: number }[];
  crystalSystem?: string;
  spacegroup?: number;
  failureReason: FailureReason;
  failedAt: number;
  source: FailureSource;
  formationEnergy?: number;
  imaginaryModeCount?: number;
  lowestPhononFreq?: number;
  bandGap?: number;
  stage?: number;
  details?: string;
}

interface FailurePattern {
  elementPairs: { pair: string; failureCount: number; failureRate: number }[];
  crystalSystems: { system: string; failureCount: number; failureRate: number }[];
  latticeRanges: { param: string; min: number; max: number; failureCount: number }[];
  compositionFeatures: { feature: string; avgValue: number; correlation: number }[];
}

const MAX_ENTRIES = 2000;
const entries: StructureFailureEntry[] = [];
const formulaReasonIndex = new Map<string, Set<FailureReason>>();

function parseElements(formula: string): string[] {
  const elems: string[] = [];
  const regex = /([A-Z][a-z]?)/g;
  let m;
  while ((m = regex.exec(formula)) !== null) {
    if (!elems.includes(m[1])) elems.push(m[1]);
  }
  return elems;
}

function getElementPairs(elements: string[]): string[] {
  const pairs: string[] = [];
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const sorted = [elements[i], elements[j]].sort();
      pairs.push(`${sorted[0]}-${sorted[1]}`);
    }
  }
  return pairs;
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

export function recordStructureFailure(entry: StructureFailureEntry): void {
  const key = entry.formula;
  const existing = formulaReasonIndex.get(key);
  if (existing && existing.has(entry.failureReason)) {
    const idx = entries.findIndex(
      e => e.formula === entry.formula && e.failureReason === entry.failureReason
    );
    if (idx !== -1) {
      entries[idx] = { ...entry };
      return;
    }
  }

  if (entries.length >= MAX_ENTRIES) {
    const removed = entries.shift()!;
    const removedReasons = formulaReasonIndex.get(removed.formula);
    if (removedReasons) {
      removedReasons.delete(removed.failureReason);
      if (removedReasons.size === 0) formulaReasonIndex.delete(removed.formula);
    }
  }

  entries.push({ ...entry });
  if (!formulaReasonIndex.has(key)) {
    formulaReasonIndex.set(key, new Set());
  }
  formulaReasonIndex.get(key)!.add(entry.failureReason);
}

export function getFailurePatterns(): FailurePattern {
  const pairCounts = new Map<string, number>();
  const systemCounts = new Map<string, number>();
  const latticeBuckets = {
    a: { values: [] as number[], count: 0 },
    b: { values: [] as number[], count: 0 },
    c: { values: [] as number[], count: 0 },
  };

  for (const entry of entries) {
    const elements = parseElements(entry.formula);
    const pairs = getElementPairs(elements);
    for (const pair of pairs) {
      pairCounts.set(pair, (pairCounts.get(pair) || 0) + 1);
    }

    if (entry.crystalSystem) {
      systemCounts.set(entry.crystalSystem, (systemCounts.get(entry.crystalSystem) || 0) + 1);
    }

    if (entry.lattice) {
      latticeBuckets.a.values.push(entry.lattice.a);
      latticeBuckets.a.count++;
      latticeBuckets.b.values.push(entry.lattice.b);
      latticeBuckets.b.count++;
      latticeBuckets.c.values.push(entry.lattice.c);
      latticeBuckets.c.count++;
    }
  }

  const totalEntries = entries.length || 1;

  const elementPairs = Array.from(pairCounts.entries())
    .map(([pair, count]) => ({ pair, failureCount: count, failureRate: count / totalEntries }))
    .sort((a, b) => b.failureCount - a.failureCount)
    .slice(0, 20);

  const crystalSystems = Array.from(systemCounts.entries())
    .map(([system, count]) => ({ system, failureCount: count, failureRate: count / totalEntries }))
    .sort((a, b) => b.failureCount - a.failureCount);

  const latticeRanges: { param: string; min: number; max: number; failureCount: number }[] = [];
  for (const [param, bucket] of Object.entries(latticeBuckets)) {
    if (bucket.values.length > 0) {
      latticeRanges.push({
        param,
        min: Math.min(...bucket.values),
        max: Math.max(...bucket.values),
        failureCount: bucket.count,
      });
    }
  }

  const featureVectors: number[][] = [];
  for (const entry of entries.slice(-200)) {
    try {
      const cf = computeCompositionFeatures(entry.formula);
      featureVectors.push(compositionFeatureVector(cf));
    } catch {}
  }

  const compositionFeatures: { feature: string; avgValue: number; correlation: number }[] = [];
  if (featureVectors.length > 2) {
    const dim = featureVectors[0].length;
    const featureNames = [
      "enMean", "enStd", "enRange", "radiusMean", "massMean",
      "vecMean", "debyeMean", "bulkModMean", "dElectronFrac",
    ];
    const indices = [0, 1, 4, 6, 11, 16, 24, 27, 43];

    for (let fi = 0; fi < indices.length && fi < featureNames.length; fi++) {
      const idx = indices[fi];
      if (idx >= dim) continue;
      const vals = featureVectors.map(v => v[idx]);
      const avg = vals.reduce((s, v) => s + v, 0) / vals.length;
      compositionFeatures.push({
        feature: featureNames[fi],
        avgValue: Number(avg.toFixed(4)),
        correlation: 1.0,
      });
    }
  }

  return { elementPairs, crystalSystems, latticeRanges, compositionFeatures };
}

export function getFailureFeatureVector(formula: string): number[] {
  const elements = parseElements(formula);
  const pairs = getElementPairs(elements);

  let sameFormulaFailures = 0;
  let sameElementFailures = 0;
  let samePairFailures = 0;
  let totalFailures = entries.length || 1;

  const failedFormulas = new Set<string>();
  const failedElements = new Set<string>();
  const failedPairs = new Set<string>();

  for (const entry of entries) {
    if (entry.formula === formula) sameFormulaFailures++;
    failedFormulas.add(entry.formula);

    const entryElements = parseElements(entry.formula);
    for (const el of entryElements) failedElements.add(el);
    const entryPairs = getElementPairs(entryElements);
    for (const p of entryPairs) failedPairs.add(p);
  }

  let elementOverlap = 0;
  for (const el of elements) {
    if (failedElements.has(el)) elementOverlap++;
  }

  let pairOverlap = 0;
  for (const p of pairs) {
    if (failedPairs.has(p)) pairOverlap++;
  }

  const sameFormulaRate = sameFormulaFailures / totalFailures;
  const elementOverlapRate = elements.length > 0 ? elementOverlap / elements.length : 0;
  const pairOverlapRate = pairs.length > 0 ? pairOverlap / pairs.length : 0;

  let minCompDist = 1.0;
  try {
    const cf = computeCompositionFeatures(formula);
    const targetVec = compositionFeatureVector(cf);

    for (const entry of entries.slice(-500)) {
      try {
        const entryCf = computeCompositionFeatures(entry.formula);
        const entryVec = compositionFeatureVector(entryCf);
        const sim = cosineSimilarity(targetVec, entryVec);
        const dist = 1 - sim;
        if (dist < minCompDist) minCompDist = dist;
      } catch {}
    }
  } catch {}

  const reasonCounts: Record<string, number> = {};
  for (const entry of entries) {
    const entryElements = parseElements(entry.formula);
    const hasOverlap = elements.some(el => entryElements.includes(el));
    if (hasOverlap) {
      reasonCounts[entry.failureReason] = (reasonCounts[entry.failureReason] || 0) + 1;
    }
  }

  const unstablePhononRate = (reasonCounts["unstable_phonons"] || 0) / totalFailures;
  const collapseRate = (reasonCounts["structure_collapse"] || 0) / totalFailures;
  const highEnergyRate = (reasonCounts["high_formation_energy"] || 0) / totalFailures;
  const nonMetallicRate = (reasonCounts["non_metallic"] || 0) / totalFailures;
  const scfDivRate = (reasonCounts["scf_divergence"] || 0) / totalFailures;
  const geoRejectRate = (reasonCounts["geometry_rejected"] || 0) / totalFailures;

  return [
    sameFormulaRate,
    elementOverlapRate,
    pairOverlapRate,
    minCompDist,
    unstablePhononRate,
    collapseRate,
    highEnergyRate,
    nonMetallicRate,
    scfDivRate,
    geoRejectRate,
  ];
}

export function shouldAvoidStructure(
  formula: string,
  lattice?: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number },
  system?: string,
): boolean {
  const reasons = formulaReasonIndex.get(formula);
  if (reasons && reasons.size >= 3) return true;

  try {
    const cf = computeCompositionFeatures(formula);
    const targetVec = compositionFeatureVector(cf);

    for (const entry of entries.slice(-500)) {
      try {
        const entryCf = computeCompositionFeatures(entry.formula);
        const entryVec = compositionFeatureVector(entryCf);
        const sim = cosineSimilarity(targetVec, entryVec);
        if (sim > 0.9) {
          if (lattice && entry.lattice) {
            const latticeSim =
              1 - (Math.abs(lattice.a - entry.lattice.a) / Math.max(lattice.a, 0.1) +
                   Math.abs(lattice.b - entry.lattice.b) / Math.max(lattice.b, 0.1) +
                   Math.abs(lattice.c - entry.lattice.c) / Math.max(lattice.c, 0.1)) / 3;
            if (latticeSim > 0.9) return true;
          } else if (system && entry.crystalSystem === system) {
            return true;
          } else if (!lattice && !system) {
            return true;
          }
        }
      } catch {}
    }
  } catch {}

  return false;
}

export function getFailureDBStats(): {
  totalEntries: number;
  maxEntries: number;
  byReason: Record<string, number>;
  bySource: Record<string, number>;
  topFailingElements: { element: string; count: number }[];
  topFailingPrototypes: { system: string; count: number }[];
  recentFailures: { formula: string; reason: string; source: string; failedAt: number }[];
} {
  const byReason: Record<string, number> = {};
  const bySource: Record<string, number> = {};
  const elementCounts = new Map<string, number>();
  const systemCounts = new Map<string, number>();

  for (const entry of entries) {
    byReason[entry.failureReason] = (byReason[entry.failureReason] || 0) + 1;
    bySource[entry.source] = (bySource[entry.source] || 0) + 1;

    const elements = parseElements(entry.formula);
    for (const el of elements) {
      elementCounts.set(el, (elementCounts.get(el) || 0) + 1);
    }

    if (entry.crystalSystem) {
      systemCounts.set(entry.crystalSystem, (systemCounts.get(entry.crystalSystem) || 0) + 1);
    }
  }

  const topFailingElements = Array.from(elementCounts.entries())
    .map(([element, count]) => ({ element, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 15);

  const topFailingPrototypes = Array.from(systemCounts.entries())
    .map(([system, count]) => ({ system, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  const recentFailures = entries.slice(-10).reverse().map(e => ({
    formula: e.formula,
    reason: e.failureReason,
    source: e.source,
    failedAt: e.failedAt,
  }));

  return {
    totalEntries: entries.length,
    maxEntries: MAX_ENTRIES,
    byReason,
    bySource,
    topFailingElements,
    topFailingPrototypes,
    recentFailures,
  };
}

export function getFailureEntries(): StructureFailureEntry[] {
  return [...entries];
}
