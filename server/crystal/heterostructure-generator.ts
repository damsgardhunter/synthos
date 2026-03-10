import { getElementData } from "../learning/elemental-data";

export interface SubstrateEntry {
  formula: string;
  latticeA: number;
  latticeC: number;
  crystalSystem: string;
  spaceGroup: string;
  orientation: string;
  dielectricConstant: number;
  category: "oxide-perovskite" | "oxide-corundum" | "semiconductor" | "fluoride" | "nitride" | "other";
}

export interface BilayerStructure {
  layer1: string;
  layer2: string;
  orientation: string;
  latticeMismatch: number;
  mismatchQuality: "ideal" | "workable" | "unlikely";
  strain: number;
  strainType: "compressive" | "tensile" | "matched";
  interlayerDistance: number;
  supercellMatch: { nA: number; nB: number; effectiveMismatch: number } | null;
}

export interface HeterostructureAtom {
  element: string;
  x: number;
  y: number;
  z: number;
  layer: number;
}

export interface StackedStructure {
  atoms: HeterostructureAtom[];
  totalAtoms: number;
  numLayers: number;
  latticeA: number;
  latticeB: number;
  latticeC: number;
  layerThicknesses: number[];
}

export interface HeterostructureResult {
  bilayer: BilayerStructure;
  structure: StackedStructure;
  interfaceScore: number;
  chargeTransferPotential: number;
  phononMismatchFactor: number;
  scEnhancementEstimate: number;
  generatedAt: number;
}

export interface HeterostructureGeneratorStats {
  totalGenerated: number;
  idealMismatchCount: number;
  workableMismatchCount: number;
  unlikelyMismatchCount: number;
  avgMismatch: number;
  avgInterfaceScore: number;
  substrateUsage: Record<string, number>;
  topCandidates: Array<{
    layer1: string;
    layer2: string;
    mismatch: number;
    mismatchQuality: string;
    interfaceScore: number;
    scEnhancement: number;
  }>;
  recentGenerations: Array<{
    layer1: string;
    layer2: string;
    orientation: string;
    mismatch: number;
    quality: string;
    atoms: number;
  }>;
}

const SUBSTRATE_DATABASE: SubstrateEntry[] = [
  { formula: "SrTiO3", latticeA: 3.905, latticeC: 3.905, crystalSystem: "cubic", spaceGroup: "Pm-3m", orientation: "(001)", dielectricConstant: 300, category: "oxide-perovskite" },
  { formula: "LaAlO3", latticeA: 3.789, latticeC: 3.789, crystalSystem: "cubic", spaceGroup: "Pm-3m", orientation: "(001)", dielectricConstant: 25, category: "oxide-perovskite" },
  { formula: "MgO", latticeA: 4.212, latticeC: 4.212, crystalSystem: "cubic", spaceGroup: "Fm-3m", orientation: "(001)", dielectricConstant: 9.7, category: "oxide-corundum" },
  { formula: "Al2O3", latticeA: 4.758, latticeC: 12.991, crystalSystem: "hexagonal", spaceGroup: "R-3c", orientation: "(0001)", dielectricConstant: 9.4, category: "oxide-corundum" },
  { formula: "BaTiO3", latticeA: 3.992, latticeC: 4.036, crystalSystem: "tetragonal", spaceGroup: "P4mm", orientation: "(001)", dielectricConstant: 1700, category: "oxide-perovskite" },
  { formula: "KTaO3", latticeA: 3.989, latticeC: 3.989, crystalSystem: "cubic", spaceGroup: "Pm-3m", orientation: "(001)", dielectricConstant: 4500, category: "oxide-perovskite" },
  { formula: "NdGaO3", latticeA: 3.864, latticeC: 3.864, crystalSystem: "orthorhombic", spaceGroup: "Pbnm", orientation: "(001)", dielectricConstant: 22, category: "oxide-perovskite" },
  { formula: "DyScO3", latticeA: 3.944, latticeC: 3.944, crystalSystem: "orthorhombic", spaceGroup: "Pbnm", orientation: "(110)", dielectricConstant: 21, category: "oxide-perovskite" },
  { formula: "TiO2", latticeA: 4.594, latticeC: 2.959, crystalSystem: "tetragonal", spaceGroup: "P42/mnm", orientation: "(001)", dielectricConstant: 86, category: "oxide-corundum" },
  { formula: "SrVO3", latticeA: 3.842, latticeC: 3.842, crystalSystem: "cubic", spaceGroup: "Pm-3m", orientation: "(001)", dielectricConstant: 18, category: "oxide-perovskite" },
  { formula: "Si", latticeA: 5.431, latticeC: 5.431, crystalSystem: "cubic", spaceGroup: "Fd-3m", orientation: "(001)", dielectricConstant: 11.7, category: "semiconductor" },
  { formula: "GaAs", latticeA: 5.653, latticeC: 5.653, crystalSystem: "cubic", spaceGroup: "F-43m", orientation: "(001)", dielectricConstant: 12.9, category: "semiconductor" },
  { formula: "SrLaAlO4", latticeA: 3.756, latticeC: 12.636, crystalSystem: "tetragonal", spaceGroup: "I4/mmm", orientation: "(001)", dielectricConstant: 24, category: "oxide-perovskite" },
  { formula: "LiF", latticeA: 4.027, latticeC: 4.027, crystalSystem: "cubic", spaceGroup: "Fm-3m", orientation: "(001)", dielectricConstant: 9.0, category: "fluoride" },
  { formula: "AlN", latticeA: 3.111, latticeC: 4.978, crystalSystem: "hexagonal", spaceGroup: "P63mc", orientation: "(0001)", dielectricConstant: 8.5, category: "nitride" },
  { formula: "GaN", latticeA: 3.189, latticeC: 5.185, crystalSystem: "hexagonal", spaceGroup: "P63mc", orientation: "(0001)", dielectricConstant: 8.9, category: "nitride" },
  { formula: "LSAT", latticeA: 3.868, latticeC: 3.868, crystalSystem: "cubic", spaceGroup: "Pm-3m", orientation: "(001)", dielectricConstant: 22, category: "oxide-perovskite" },
  { formula: "YAlO3", latticeA: 3.680, latticeC: 3.680, crystalSystem: "orthorhombic", spaceGroup: "Pbnm", orientation: "(001)", dielectricConstant: 16, category: "oxide-perovskite" },
];

const KNOWN_FILM_LATTICE: Record<string, { a: number; c: number; system: string }> = {
  "FeSe": { a: 3.765, c: 5.518, system: "tetragonal" },
  "FeTe": { a: 3.822, c: 6.272, system: "tetragonal" },
  "NbSe2": { a: 3.445, c: 12.547, system: "hexagonal" },
  "TaS2": { a: 3.315, c: 12.097, system: "hexagonal" },
  "MoS2": { a: 3.160, c: 12.295, system: "hexagonal" },
  "WTe2": { a: 3.477, c: 14.018, system: "hexagonal" },
  "NbN": { a: 4.394, c: 4.394, system: "cubic" },
  "NbC": { a: 4.470, c: 4.470, system: "cubic" },
  "MgB2": { a: 3.086, c: 3.524, system: "hexagonal" },
  "YBa2Cu3O7": { a: 3.820, c: 11.680, system: "orthorhombic" },
  "La2CuO4": { a: 3.787, c: 13.226, system: "tetragonal" },
  "Bi2Sr2CaCu2O8": { a: 3.814, c: 30.89, system: "tetragonal" },
  "NdNiO2": { a: 3.921, c: 3.281, system: "tetragonal" },
  "PrNiO2": { a: 3.920, c: 3.280, system: "tetragonal" },
  "LaNiO3": { a: 3.838, c: 3.838, system: "cubic" },
  "SrRuO3": { a: 3.930, c: 3.930, system: "cubic" },
  "Nb3Sn": { a: 5.289, c: 5.289, system: "cubic" },
  "Nb3Ge": { a: 5.166, c: 5.166, system: "cubic" },
  "V3Si": { a: 4.722, c: 4.722, system: "cubic" },
  "LaH10": { a: 5.100, c: 5.100, system: "cubic" },
  "H3S": { a: 3.089, c: 3.089, system: "cubic" },
  "CaH6": { a: 3.520, c: 3.520, system: "cubic" },
  "BaFe2As2": { a: 3.963, c: 13.017, system: "tetragonal" },
  "FeSe0.5Te0.5": { a: 3.800, c: 5.900, system: "tetragonal" },
  "LaFeAsO": { a: 4.035, c: 8.741, system: "tetragonal" },
  "BiS2": { a: 3.940, c: 13.500, system: "tetragonal" },
};

function parseFormula(formula: string): Record<string, number> {
  const cleaned = formula
    .replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
    .replace(/\s+/g, "");
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    if (num > 0) counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function estimateFilmLattice(formula: string): { a: number; c: number; system: string } {
  if (KNOWN_FILM_LATTICE[formula]) return KNOWN_FILM_LATTICE[formula];

  const counts = parseFormula(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  let sumRadius = 0;
  let count = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const radius = (data?.atomicRadius ?? 150) / 100;
    sumRadius += radius * (counts[el] / totalAtoms);
    count++;
  }

  const hasO = elements.includes("O");
  const hasTM = elements.some(el => {
    const data = getElementData(el);
    return data && data.atomicNumber >= 21 && data.atomicNumber <= 30;
  });

  let a: number;
  let system = "cubic";

  if (totalAtoms <= 3) {
    a = sumRadius * 2.8;
  } else if (hasO && hasTM && totalAtoms >= 5) {
    a = 3.8 + (sumRadius - 1.3) * 0.5;
    system = "tetragonal";
  } else if (elements.includes("H") && Object.values(counts).some(n => n >= 6)) {
    a = 3.0 + sumRadius * 0.8;
  } else {
    a = sumRadius * 3.0;
  }

  a = Math.max(2.5, Math.min(8.0, a));

  const cOverA = system === "tetragonal" ? 2.5 + Math.random() * 1.5 :
                 system === "hexagonal" ? 1.6 + Math.random() * 0.4 : 1.0;

  return { a, c: a * cOverA, system };
}

export function computeLatticeMismatch(aFilm: number, aSubstrate: number): number {
  return Math.abs(aFilm - aSubstrate) / aFilm;
}

export function classifyMismatch(mismatch: number): "ideal" | "workable" | "unlikely" {
  if (mismatch < 0.03) return "ideal";
  if (mismatch < 0.07) return "workable";
  return "unlikely";
}

function findSupercellMatch(aFilm: number, aSubstrate: number, maxN: number = 5): { nA: number; nB: number; effectiveMismatch: number } | null {
  let bestMatch: { nA: number; nB: number; effectiveMismatch: number } | null = null;
  let bestMismatch = 1.0;

  for (let nA = 1; nA <= maxN; nA++) {
    for (let nB = 1; nB <= maxN; nB++) {
      if (nA === 1 && nB === 1) continue;
      const superA = aFilm * nA;
      const superB = aSubstrate * nB;
      const mismatch = Math.abs(superA - superB) / superA;
      if (mismatch < bestMismatch) {
        bestMismatch = mismatch;
        bestMatch = { nA, nB, effectiveMismatch: Math.round(mismatch * 100000) / 100000 };
      }
    }
  }

  if (bestMatch && bestMatch.effectiveMismatch < 0.07) return bestMatch;
  return null;
}

function estimateInterlayerDistance(filmFormula: string, substrateFormula: string): number {
  const filmCounts = parseFormula(filmFormula);
  const subCounts = parseFormula(substrateFormula);

  const filmElements = Object.keys(filmCounts);
  const subElements = Object.keys(subCounts);

  const hasVdW = filmElements.some(el =>
    el === "Se" || el === "Te" || el === "S"
  );

  if (hasVdW) return 3.0 + Math.random() * 0.4;

  const filmEN = filmElements.reduce((s, el) => {
    const data = getElementData(el);
    return s + (data?.paulingElectronegativity ?? 1.5) * (filmCounts[el] / Object.values(filmCounts).reduce((a, b) => a + b, 0));
  }, 0);

  const subEN = subElements.reduce((s, el) => {
    const data = getElementData(el);
    return s + (data?.paulingElectronegativity ?? 1.5) * (subCounts[el] / Object.values(subCounts).reduce((a, b) => a + b, 0));
  }, 0);

  const enDiff = Math.abs(filmEN - subEN);
  return 2.0 + enDiff * 0.8 + Math.random() * 0.3;
}

export function buildBilayer(filmFormula: string, substrateFormula: string, substrate?: SubstrateEntry): BilayerStructure {
  const sub = substrate || SUBSTRATE_DATABASE.find(s => s.formula === substrateFormula) || null;
  const filmLattice = estimateFilmLattice(filmFormula);
  const subLatticeA = sub ? sub.latticeA : estimateFilmLattice(substrateFormula).a;

  const mismatch = computeLatticeMismatch(filmLattice.a, subLatticeA);
  const quality = classifyMismatch(mismatch);
  const strainMag = (subLatticeA - filmLattice.a) / filmLattice.a;
  const strainType = Math.abs(strainMag) < 0.005 ? "matched" :
                     strainMag > 0 ? "tensile" : "compressive";

  const supercellMatch = quality === "unlikely" ? findSupercellMatch(filmLattice.a, subLatticeA) : null;
  const interlayerDistance = estimateInterlayerDistance(filmFormula, substrateFormula);
  const orientation = sub?.orientation || "(001)";

  return {
    layer1: filmFormula,
    layer2: substrateFormula,
    orientation,
    latticeMismatch: Math.round(mismatch * 100000) / 100000,
    mismatchQuality: quality,
    strain: Math.round(strainMag * 100000) / 100000,
    strainType,
    interlayerDistance: Math.round(interlayerDistance * 1000) / 1000,
    supercellMatch,
  };
}

function buildLayerAtoms(formula: string, latticeA: number, zOffset: number, layerIndex: number, nLayers: number): HeterostructureAtom[] {
  const counts = parseFormula(formula);
  const elements = Object.keys(counts);
  const atoms: HeterostructureAtom[] = [];

  const totalAtoms = Math.round(Object.values(counts).reduce((s, n) => s + n, 0));
  const layerThickness = estimateFilmLattice(formula).c / (totalAtoms > 5 ? 2 : 1);

  let currentZ = zOffset;
  for (const el of elements) {
    const n = Math.round(counts[el]);
    for (let i = 0; i < n; i++) {
      for (let ly = 0; ly < nLayers; ly++) {
        const fracX = (i % 2 === 0 ? 0.0 : 0.5) + (ly % 2 === 0 ? 0.0 : 0.25);
        const fracY = (i % 2 === 0 ? 0.0 : 0.5);
        const fracZ = currentZ + ly * layerThickness / nLayers;

        atoms.push({
          element: el,
          x: fracX * latticeA,
          y: fracY * latticeA,
          z: fracZ,
          layer: layerIndex,
        });
      }
    }
    currentZ += layerThickness / elements.length;
  }

  return atoms;
}

export function buildStackedStructure(bilayer: BilayerStructure, numRepeats: number = 2): StackedStructure {
  const filmLattice = estimateFilmLattice(bilayer.layer1);
  const subLattice = estimateFilmLattice(bilayer.layer2);

  const effectiveA = filmLattice.a * (1 + bilayer.strain);

  const filmThickness = filmLattice.c;
  const subThickness = subLattice.c;
  const gap = bilayer.interlayerDistance;

  const allAtoms: HeterostructureAtom[] = [];
  const layerThicknesses: number[] = [];
  let zCursor = 0;

  for (let rep = 0; rep < numRepeats; rep++) {
    const filmAtoms = buildLayerAtoms(bilayer.layer1, effectiveA, zCursor, rep * 2, 2);
    allAtoms.push(...filmAtoms);
    layerThicknesses.push(filmThickness);
    zCursor += filmThickness + gap;

    const subAtoms = buildLayerAtoms(bilayer.layer2, effectiveA, zCursor, rep * 2 + 1, 2);
    allAtoms.push(...subAtoms);
    layerThicknesses.push(subThickness);
    zCursor += subThickness + gap;
  }

  return {
    atoms: allAtoms,
    totalAtoms: allAtoms.length,
    numLayers: numRepeats * 2,
    latticeA: Math.round(effectiveA * 1000) / 1000,
    latticeB: Math.round(effectiveA * 1000) / 1000,
    latticeC: Math.round(zCursor * 1000) / 1000,
    layerThicknesses,
  };
}

function computeChargeTransferPotential(filmFormula: string, subFormula: string): number {
  const filmCounts = parseFormula(filmFormula);
  const subCounts = parseFormula(subFormula);

  const avgEN = (counts: Record<string, number>) => {
    const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    return Object.entries(counts).reduce((s, [el, n]) => {
      const data = getElementData(el);
      return s + (data?.paulingElectronegativity ?? 1.5) * (n / total);
    }, 0);
  };

  const enDiff = Math.abs(avgEN(filmCounts) - avgEN(subCounts));
  return Math.min(1.0, enDiff * 0.5);
}

function computePhononMismatchFactor(filmFormula: string, subFormula: string): number {
  const avgMass = (formula: string) => {
    const counts = parseFormula(formula);
    const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    return Object.entries(counts).reduce((s, [el, n]) => {
      const data = getElementData(el);
      return s + (data?.atomicMass ?? 50) * (n / total);
    }, 0);
  };

  const massRatio = avgMass(filmFormula) / avgMass(subFormula);
  const mismatch = Math.abs(1 - massRatio);
  return Math.min(1.0, mismatch * 0.8);
}

function estimateScEnhancement(bilayer: BilayerStructure, chargeTransfer: number, phononMismatch: number): number {
  let score = 0;

  if (bilayer.mismatchQuality === "ideal") score += 0.3;
  else if (bilayer.mismatchQuality === "workable") score += 0.15;

  score += chargeTransfer * 0.25;

  if (phononMismatch > 0.2 && phononMismatch < 0.6) score += 0.2;

  const absStrain = Math.abs(bilayer.strain);
  if (absStrain > 0.005 && absStrain < 0.04) score += 0.15;

  if (bilayer.interlayerDistance < 3.5) score += 0.1;

  return Math.min(1.0, Math.round(score * 10000) / 10000);
}

const generationHistory: HeterostructureResult[] = [];
const MAX_HISTORY = 500;

export function generateHeterostructure(filmFormula: string, substrateFormula: string): HeterostructureResult {
  const bilayer = buildBilayer(filmFormula, substrateFormula);
  const structure = buildStackedStructure(bilayer, 2);
  const chargeTransferPotential = computeChargeTransferPotential(filmFormula, substrateFormula);
  const phononMismatchFactor = computePhononMismatchFactor(filmFormula, substrateFormula);
  const scEnhancementEstimate = estimateScEnhancement(bilayer, chargeTransferPotential, phononMismatchFactor);

  const interfaceScore =
    (bilayer.mismatchQuality === "ideal" ? 0.4 : bilayer.mismatchQuality === "workable" ? 0.2 : 0.05) +
    chargeTransferPotential * 0.3 +
    phononMismatchFactor * 0.15 +
    scEnhancementEstimate * 0.15;

  const result: HeterostructureResult = {
    bilayer,
    structure,
    interfaceScore: Math.round(Math.min(1.0, interfaceScore) * 10000) / 10000,
    chargeTransferPotential,
    phononMismatchFactor,
    scEnhancementEstimate,
    generatedAt: Date.now(),
  };

  generationHistory.push(result);
  if (generationHistory.length > MAX_HISTORY) {
    generationHistory.splice(0, Math.floor(MAX_HISTORY * 0.1));
  }

  return result;
}

export function generateBilayerCandidates(
  filmCandidates: string[],
  maxPerFilm: number = 5
): HeterostructureResult[] {
  const results: HeterostructureResult[] = [];

  for (const film of filmCandidates) {
    const scored: Array<{ sub: SubstrateEntry; mismatch: number }> = [];

    for (const sub of SUBSTRATE_DATABASE) {
      if (sub.formula === film) continue;
      const filmLattice = estimateFilmLattice(film);
      const mismatch = computeLatticeMismatch(filmLattice.a, sub.latticeA);
      scored.push({ sub, mismatch });
    }

    scored.sort((a, b) => a.mismatch - b.mismatch);
    const topSubs = scored.slice(0, maxPerFilm);

    for (const { sub } of topSubs) {
      const result = generateHeterostructure(film, sub.formula);
      results.push(result);
    }
  }

  results.sort((a, b) => b.interfaceScore - a.interfaceScore);
  return results;
}

export function getSubstrateDatabase(): SubstrateEntry[] {
  return [...SUBSTRATE_DATABASE];
}

export function findBestSubstrates(filmFormula: string, topN: number = 5): Array<{
  substrate: SubstrateEntry;
  mismatch: number;
  quality: string;
  supercellMatch: { nA: number; nB: number; effectiveMismatch: number } | null;
}> {
  const filmLattice = estimateFilmLattice(filmFormula);
  const results: Array<{
    substrate: SubstrateEntry;
    mismatch: number;
    quality: string;
    supercellMatch: { nA: number; nB: number; effectiveMismatch: number } | null;
    effectiveMismatch: number;
  }> = [];

  for (const sub of SUBSTRATE_DATABASE) {
    const mismatch = computeLatticeMismatch(filmLattice.a, sub.latticeA);
    const quality = classifyMismatch(mismatch);
    const supercellMatch = quality === "unlikely" ? findSupercellMatch(filmLattice.a, sub.latticeA) : null;
    const effectiveMismatch = supercellMatch ? supercellMatch.effectiveMismatch : mismatch;
    results.push({ substrate: sub, mismatch: Math.round(mismatch * 100000) / 100000, quality, supercellMatch, effectiveMismatch });
  }

  results.sort((a, b) => a.effectiveMismatch - b.effectiveMismatch);
  return results.slice(0, topN).map(({ effectiveMismatch, ...rest }) => rest);
}

export function getHeterostructureStats(): HeterostructureGeneratorStats {
  const n = generationHistory.length;
  if (n === 0) {
    return {
      totalGenerated: 0,
      idealMismatchCount: 0,
      workableMismatchCount: 0,
      unlikelyMismatchCount: 0,
      avgMismatch: 0,
      avgInterfaceScore: 0,
      substrateUsage: {},
      topCandidates: [],
      recentGenerations: [],
    };
  }

  let idealCount = 0, workableCount = 0, unlikelyCount = 0;
  let mismatchSum = 0, scoreSum = 0;
  const subUsage: Record<string, number> = {};

  for (const h of generationHistory) {
    if (h.bilayer.mismatchQuality === "ideal") idealCount++;
    else if (h.bilayer.mismatchQuality === "workable") workableCount++;
    else unlikelyCount++;
    mismatchSum += h.bilayer.latticeMismatch;
    scoreSum += h.interfaceScore;
    subUsage[h.bilayer.layer2] = (subUsage[h.bilayer.layer2] || 0) + 1;
  }

  const sorted = [...generationHistory].sort((a, b) => b.interfaceScore - a.interfaceScore);
  const topCandidates = sorted.slice(0, 10).map(h => ({
    layer1: h.bilayer.layer1,
    layer2: h.bilayer.layer2,
    mismatch: h.bilayer.latticeMismatch,
    mismatchQuality: h.bilayer.mismatchQuality,
    interfaceScore: h.interfaceScore,
    scEnhancement: h.scEnhancementEstimate,
  }));

  const recentGenerations = generationHistory.slice(-10).reverse().map(h => ({
    layer1: h.bilayer.layer1,
    layer2: h.bilayer.layer2,
    orientation: h.bilayer.orientation,
    mismatch: h.bilayer.latticeMismatch,
    quality: h.bilayer.mismatchQuality,
    atoms: h.structure.totalAtoms,
  }));

  return {
    totalGenerated: n,
    idealMismatchCount: idealCount,
    workableMismatchCount: workableCount,
    unlikelyMismatchCount: unlikelyCount,
    avgMismatch: Math.round((mismatchSum / n) * 100000) / 100000,
    avgInterfaceScore: Math.round((scoreSum / n) * 10000) / 10000,
    substrateUsage: subUsage,
    topCandidates,
    recentGenerations,
  };
}
