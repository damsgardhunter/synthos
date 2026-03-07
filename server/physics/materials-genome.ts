import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  computeDimensionalityScore,
  detectStructuralMotifs,
  parseFormulaElements,
  classifyHydrogenBonding,
  estimateBandwidthW,
  assessCorrelationStrength,
  type ElectronicStructure,
  type PhononSpectrum,
  type ElectronPhononCoupling,
} from "../learning/physics-engine";
import {
  getElementData,
  getCompositionWeightedProperty,
  isTransitionMetal,
  isRareEarth,
  isActinide,
  getHubbardU,
  getStonerParameter,
} from "../learning/elemental-data";
import { computePairingFeatureVector } from "./pairing-mechanisms";
import { classifyFamily } from "../learning/utils";

export interface MaterialGenome {
  formula: string;
  vector: number[];
  segments: {
    structure: number[];
    orbitals: number[];
    phonons: number[];
    coupling: number[];
    topology: number[];
    dimensionality: number[];
    composition: number[];
    pairing: number[];
  };
  metadata: {
    family: string;
    metallicity: number;
    lambda: number;
    dimensionalityScore: number;
    dominantOrbital: string;
    encodedAt: number;
  };
}

export interface GenomeSimilarityResult {
  formula: string;
  similarity: number;
  distance: number;
  sharedSegments: string[];
}

export interface GenomeInverseCandidate {
  formula: string;
  genomeDistance: number;
  predictedProperties: {
    estimatedLambda: number;
    estimatedMetallicity: number;
    estimatedDimensionality: number;
  };
}

const GENOME_DIM = 256;
const STRUCTURE_DIM = 40;
const ORBITAL_DIM = 36;
const PHONON_DIM = 32;
const COUPLING_DIM = 32;
const TOPOLOGY_DIM = 28;
const DIMENSIONALITY_DIM = 24;
const COMPOSITION_DIM = 40;
const PAIRING_DIM = 24;

const genomeCache = new Map<string, MaterialGenome>();
const MAX_CACHE = 500;

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    counts[m[1]] = (counts[m[1]] || 0) + (m[2] ? parseFloat(m[2]) : 1);
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  return total > 0 ? total : 1;
}

function normalize(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  if (norm < 1e-10) return vec;
  return vec.map(v => v / norm);
}

function padOrTruncate(vec: number[], dim: number): number[] {
  if (vec.length >= dim) return vec.slice(0, dim);
  return [...vec, ...new Array(dim - vec.length).fill(0)];
}

function hashEncode(value: number, bins: number): number[] {
  const result = new Array(bins).fill(0);
  const idx = Math.min(bins - 1, Math.max(0, Math.floor(value * bins)));
  result[idx] = 1.0;
  if (idx > 0) result[idx - 1] = 0.3;
  if (idx < bins - 1) result[idx + 1] = 0.3;
  return result;
}

function fourierEncode(value: number, nFreqs: number): number[] {
  const result: number[] = [];
  for (let i = 1; i <= nFreqs; i++) {
    result.push(Math.sin(value * i * Math.PI));
    result.push(Math.cos(value * i * Math.PI));
  }
  return result;
}

function encodeStructureSegment(
  formula: string,
  electronic: ElectronicStructure,
  counts: Record<string, number>,
  elements: string[],
): number[] {
  const totalAtoms = getTotalAtoms(counts);
  const motifs = detectStructuralMotifs(formula);

  const numElements = Math.min(elements.length / 10, 1.0);
  const atomCountNorm = Math.min(totalAtoms / 20, 1.0);

  const metalFrac = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
  const hFrac = (counts["H"] || 0) / totalAtoms;
  const oFrac = (counts["O"] || 0) / totalAtoms;

  const avgEN = getCompositionWeightedProperty(counts, "paulingElectronegativity") ?? 1.5;
  const avgRadius = getCompositionWeightedProperty(counts, "atomicRadius") ?? 130;
  const avgMass = getCompositionWeightedProperty(counts, "atomicMass") ?? 50;

  const enValues = elements.map(e => getElementData(e)?.paulingElectronegativity ?? 1.5);
  const enSpread = enValues.length > 1 ? (Math.max(...enValues) - Math.min(...enValues)) / 3.0 : 0;

  const massValues = elements.map(e => getElementData(e)?.atomicMass ?? 50);
  const massSpread = massValues.length > 1 ?
    (Math.max(...massValues) - Math.min(...massValues)) / 200 : 0;

  const vec = [
    numElements,
    atomCountNorm,
    metalFrac,
    hFrac,
    oFrac,
    avgEN / 4.0,
    enSpread,
    avgRadius / 250,
    avgMass / 250,
    massSpread,
    motifs.motifScore,
    motifs.motifs.some(m => m.toLowerCase().includes("perovskite")) ? 1 : 0,
    motifs.motifs.some(m => m.toLowerCase().includes("layer")) ? 1 : 0,
    motifs.motifs.some(m => m.toLowerCase().includes("cage") || m.toLowerCase().includes("clathrate")) ? 1 : 0,
    motifs.motifs.some(m => m.toLowerCase().includes("chain")) ? 1 : 0,
    ...hashEncode(metalFrac, 5),
    ...fourierEncode(avgEN / 4.0, 5),
    ...fourierEncode(enSpread, 5),
  ];

  return padOrTruncate(vec, STRUCTURE_DIM);
}

function encodeOrbitalSegment(
  electronic: ElectronicStructure,
  elements: string[],
  counts: Record<string, number>,
): number[] {
  const totalAtoms = getTotalAtoms(counts);
  const of = electronic.orbitalFractions;

  let avgValence = 0;
  for (const el of elements) {
    const d = getElementData(el);
    if (d) avgValence += d.valenceElectrons * (counts[el] || 1);
  }
  avgValence /= totalAtoms;

  const hasDElectrons = elements.some(e => {
    const d = getElementData(e);
    return d && d.valenceElectrons >= 3 && isTransitionMetal(e);
  });
  const hasFElectrons = elements.some(e => isRareEarth(e) || isActinide(e));

  let maxHubbardU = 0;
  let maxStonerI = 0;
  for (const el of elements) {
    const u = getHubbardU(el);
    if (u !== null && u > maxHubbardU) maxHubbardU = u;
    const s = getStonerParameter(el);
    if (s !== null && s > maxStonerI) maxStonerI = s;
  }

  let avgBandwidth = 0;
  for (const el of elements) {
    avgBandwidth += estimateBandwidthW(el) * ((counts[el] || 1) / totalAtoms);
  }

  const vec = [
    of.s,
    of.p,
    of.d,
    of.f,
    avgValence / 8,
    hasDElectrons ? 1 : 0,
    hasFElectrons ? 1 : 0,
    maxHubbardU / 10,
    maxStonerI,
    avgBandwidth / 15,
    electronic.correlationStrength,
    electronic.mottProximityScore,
    electronic.topologicalBandScore,
    ...fourierEncode(of.d, 4),
    ...fourierEncode(of.f, 3),
    ...hashEncode(electronic.correlationStrength, 5),
  ];

  return padOrTruncate(vec, ORBITAL_DIM);
}

function encodePhononSegment(
  phonon: PhononSpectrum,
  formula: string,
): number[] {
  const maxFreqNorm = Math.min(phonon.maxPhononFrequency / 2000, 1.0);
  const logFreqNorm = Math.min(phonon.logAverageFrequency / 1500, 1.0);
  const debyeNorm = Math.min(phonon.debyeTemperature / 2000, 1.0);

  const vec = [
    maxFreqNorm,
    logFreqNorm,
    debyeNorm,
    phonon.anharmonicityIndex,
    phonon.softModePresent ? 1 : 0,
    phonon.softModeScore,
    phonon.hasImaginaryModes ? 1 : 0,
    maxFreqNorm - logFreqNorm,
    ...fourierEncode(maxFreqNorm, 4),
    ...fourierEncode(debyeNorm, 4),
    ...hashEncode(phonon.anharmonicityIndex, 5),
  ];

  return padOrTruncate(vec, PHONON_DIM);
}

function encodeCouplingSegment(
  coupling: ElectronPhononCoupling,
  electronic: ElectronicStructure,
): number[] {
  const lambdaNorm = Math.min(coupling.lambda / 3.5, 1.0);
  const omegaLogNorm = Math.min(coupling.omegaLog / 1500, 1.0);
  const isStrong = coupling.isStrongCoupling ? 1 : 0;

  const vec = [
    lambdaNorm,
    omegaLogNorm,
    coupling.muStar,
    isStrong,
    coupling.anharmonicCorrectionFactor,
    electronic.densityOfStatesAtFermi / 10,
    electronic.metallicity,
    electronic.nestingScore,
    electronic.vanHoveProximity,
    electronic.bandFlatness,
    ...fourierEncode(lambdaNorm, 4),
    ...hashEncode(electronic.metallicity, 5),
    ...hashEncode(lambdaNorm, 5),
  ];

  return padOrTruncate(vec, COUPLING_DIM);
}

function encodeTopologySegment(
  electronic: ElectronicStructure,
  elements: string[],
  counts: Record<string, number>,
): number[] {
  const topo = electronic.tightBindingTopology;
  const totalAtoms = getTotalAtoms(counts);

  let maxZ = 0;
  for (const el of elements) {
    const d = getElementData(el);
    if (d && d.atomicNumber > maxZ) maxZ = d.atomicNumber;
  }
  const socProxy = Math.min(1.0, Math.pow(maxZ / 83, 4));

  const vec = [
    topo?.hasFlatBand ? 1 : 0,
    topo?.hasVHS ? 1 : 0,
    topo?.hasDiracCrossing ? 1 : 0,
    topo?.hasBandInversion ? 1 : 0,
    (topo?.topologyScore ?? 0) / 10,
    (topo?.flatBandCount ?? 0) / 5,
    (topo?.vhsCount ?? 0) / 5,
    (topo?.dosAtFermi ?? 0) / 10,
    socProxy,
    electronic.topologicalBandScore,
    electronic.flatBandIndicator,
    ...fourierEncode(socProxy, 3),
    ...hashEncode(electronic.topologicalBandScore, 5),
  ];

  return padOrTruncate(vec, TOPOLOGY_DIM);
}

function encodeDimensionalitySegment(
  formula: string,
  electronic: ElectronicStructure,
): number[] {
  const dimScore = computeDimensionalityScore(formula);
  const hBonding = classifyHydrogenBonding(formula);
  const family = classifyFamily(formula);

  const familyMap: Record<string, number> = {
    "Cuprates": 0.1, "Iron-based": 0.2, "Hydrides": 0.3,
    "Borides": 0.4, "Carbides": 0.5, "Nitrides": 0.6,
    "Oxides": 0.7, "Heavy-fermion": 0.8, "Conventional": 0.9,
  };
  const familyCode = familyMap[family] ?? 0.5;

  const hBondingMap: Record<string, number> = {
    "metallic-network": 0.9, "cage-clathrate": 0.7,
    "covalent-molecular": 0.3, "interstitial": 0.5,
    "ambiguous": 0.4, "none": 0.0,
  };
  const hBondCode = hBondingMap[hBonding] ?? 0.0;

  const fsTopology = electronic.fermiSurfaceTopology;
  const is2D = fsTopology.includes("2D") ? 1 : 0;
  const isMulti = fsTopology.includes("multi") ? 1 : 0;
  const isCylindrical = fsTopology.includes("cylindrical") ? 1 : 0;

  const vec = [
    dimScore,
    familyCode,
    hBondCode,
    is2D,
    isMulti,
    isCylindrical,
    ...fourierEncode(dimScore, 3),
    ...hashEncode(familyCode, 5),
    ...hashEncode(dimScore, 5),
  ];

  return padOrTruncate(vec, DIMENSIONALITY_DIM);
}

function getPeriodFromZ(z: number): number {
  if (z <= 2) return 1;
  if (z <= 10) return 2;
  if (z <= 18) return 3;
  if (z <= 36) return 4;
  if (z <= 54) return 5;
  if (z <= 86) return 6;
  return 7;
}

function encodeCompositionSegment(
  elements: string[],
  counts: Record<string, number>,
): number[] {
  const totalAtoms = getTotalAtoms(counts);

  const periodicPeriods = new Array(7).fill(0);
  const tmFrac = elements.filter(e => isTransitionMetal(e))
    .reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
  const reFrac = elements.filter(e => isRareEarth(e))
    .reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
  const actFrac = elements.filter(e => isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;

  for (const el of elements) {
    const d = getElementData(el);
    if (!d) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    const period = getPeriodFromZ(d.atomicNumber);
    if (period >= 1 && period <= 7) periodicPeriods[period - 1] += frac;
  }

  const avgBulk = getCompositionWeightedProperty(counts, "bulkModulus") ?? 50;
  const avgDebye = getCompositionWeightedProperty(counts, "debyeTemperature") ?? 300;
  const avgIE = getCompositionWeightedProperty(counts, "firstIonizationEnergy") ?? 7;

  const vec = [
    ...periodicPeriods,
    tmFrac,
    reFrac,
    actFrac,
    avgBulk / 500,
    avgDebye / 2000,
    avgIE / 25,
    elements.length / 10,
    ...hashEncode(elements.length / 10, 5),
    ...hashEncode(tmFrac, 5),
    ...fourierEncode(avgBulk / 500, 4),
    ...fourierEncode(avgDebye / 2000, 3),
  ];

  return padOrTruncate(vec, COMPOSITION_DIM);
}

function encodePairingSegment(formula: string): number[] {
  try {
    const pf = computePairingFeatureVector(formula);
    const vec = [
      pf.phononPairingStrength,
      pf.spinPairingStrength,
      pf.orbitalPairingStrength,
      pf.excitonPairingStrength,
      pf.dominantPairingType / 5,
      pf.compositePairing,
      ...fourierEncode(pf.compositePairing, 3),
      ...hashEncode(pf.dominantPairingType / 5, 5),
      ...hashEncode(pf.phononPairingStrength, 3),
    ];
    return padOrTruncate(vec, PAIRING_DIM);
  } catch {
    return new Array(PAIRING_DIM).fill(0);
  }
}

export function encodeGenome(formula: string): MaterialGenome {
  const cached = genomeCache.get(formula);
  if (cached) return cached;

  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);

  const electronic = computeElectronicStructure(formula, null);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

  const structureVec = encodeStructureSegment(formula, electronic, counts, elements);
  const orbitalVec = encodeOrbitalSegment(electronic, elements, counts);
  const phononVec = encodePhononSegment(phonon, formula);
  const couplingVec = encodeCouplingSegment(coupling, electronic);
  const topologyVec = encodeTopologySegment(electronic, elements, counts);
  const dimensionalityVec = encodeDimensionalitySegment(formula, electronic);
  const compositionVec = encodeCompositionSegment(elements, counts);
  const pairingVec = encodePairingSegment(formula);

  const rawVector = [
    ...structureVec,
    ...orbitalVec,
    ...phononVec,
    ...couplingVec,
    ...topologyVec,
    ...dimensionalityVec,
    ...compositionVec,
    ...pairingVec,
  ];

  const vector = padOrTruncate(rawVector, GENOME_DIM).map(v =>
    Number((Math.max(-1, Math.min(1, v))).toFixed(6))
  );

  const of = electronic.orbitalFractions;
  let dominantOrbital = "s";
  if (of.d >= of.s && of.d >= of.p && of.d >= of.f) dominantOrbital = "d";
  else if (of.p >= of.s && of.p >= of.f) dominantOrbital = "p";
  else if (of.f >= of.s) dominantOrbital = "f";

  const genome: MaterialGenome = {
    formula,
    vector,
    segments: {
      structure: structureVec,
      orbitals: orbitalVec,
      phonons: phononVec,
      coupling: couplingVec,
      topology: topologyVec,
      dimensionality: dimensionalityVec,
      composition: compositionVec,
      pairing: pairingVec,
    },
    metadata: {
      family: classifyFamily(formula),
      metallicity: electronic.metallicity,
      lambda: coupling.lambda,
      dimensionalityScore: computeDimensionalityScore(formula),
      dominantOrbital,
      encodedAt: Date.now(),
    },
  };

  if (genomeCache.size >= MAX_CACHE) {
    const oldest = genomeCache.keys().next().value;
    if (oldest) genomeCache.delete(oldest);
  }
  genomeCache.set(formula, genome);

  return genome;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < len; i++) {
    dot += (a[i] ?? 0) * (b[i] ?? 0);
    normA += (a[i] ?? 0) * (a[i] ?? 0);
    normB += (b[i] ?? 0) * (b[i] ?? 0);
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 1e-10 ? dot / denom : 0;
}

export function euclideanDistance(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function segmentSimilarity(
  segA: Record<string, number[]>,
  segB: Record<string, number[]>,
): string[] {
  const shared: string[] = [];
  const threshold = 0.7;
  for (const key of Object.keys(segA)) {
    if (segB[key]) {
      const sim = cosineSimilarity(segA[key], segB[key]);
      if (sim > threshold) shared.push(key);
    }
  }
  return shared;
}

export function findSimilar(
  targetFormula: string,
  candidateFormulas: string[],
  topK: number = 10,
): GenomeSimilarityResult[] {
  const targetGenome = encodeGenome(targetFormula);
  const results: GenomeSimilarityResult[] = [];

  for (const formula of candidateFormulas) {
    if (formula === targetFormula) continue;
    try {
      const genome = encodeGenome(formula);
      const similarity = cosineSimilarity(targetGenome.vector, genome.vector);
      const distance = euclideanDistance(targetGenome.vector, genome.vector);
      const sharedSegments = segmentSimilarity(
        targetGenome.segments as any,
        genome.segments as any,
      );

      results.push({ formula, similarity, distance, sharedSegments });
    } catch {}
  }

  results.sort((a, b) => b.similarity - a.similarity);
  return results.slice(0, topK);
}

export function genomeDiversity(formulas: string[]): number {
  if (formulas.length < 2) return 0;

  const genomes = formulas.map(f => {
    try { return encodeGenome(f); }
    catch { return null; }
  }).filter(Boolean) as MaterialGenome[];

  if (genomes.length < 2) return 0;

  let totalDistance = 0;
  let pairs = 0;
  for (let i = 0; i < genomes.length; i++) {
    for (let j = i + 1; j < genomes.length; j++) {
      totalDistance += euclideanDistance(genomes[i].vector, genomes[j].vector);
      pairs++;
    }
  }

  return pairs > 0 ? totalDistance / pairs : 0;
}

export function genomeGuidedInverseDesign(
  targetFormula: string,
  candidatePool: string[],
  topK: number = 5,
): GenomeInverseCandidate[] {
  const targetGenome = encodeGenome(targetFormula);
  const results: GenomeInverseCandidate[] = [];

  for (const formula of candidatePool) {
    if (formula === targetFormula) continue;
    try {
      const genome = encodeGenome(formula);
      const distance = euclideanDistance(targetGenome.vector, genome.vector);

      results.push({
        formula,
        genomeDistance: Number(distance.toFixed(4)),
        predictedProperties: {
          estimatedLambda: genome.metadata.lambda,
          estimatedMetallicity: genome.metadata.metallicity,
          estimatedDimensionality: genome.metadata.dimensionalityScore,
        },
      });
    } catch {}
  }

  results.sort((a, b) => a.genomeDistance - b.genomeDistance);
  return results.slice(0, topK);
}

export function interpolateGenomes(
  formulaA: string,
  formulaB: string,
  alpha: number = 0.5,
): number[] {
  const genA = encodeGenome(formulaA);
  const genB = encodeGenome(formulaB);
  const result: number[] = [];
  for (let i = 0; i < GENOME_DIM; i++) {
    result.push(
      Number(((1 - alpha) * (genA.vector[i] ?? 0) + alpha * (genB.vector[i] ?? 0)).toFixed(6))
    );
  }
  return result;
}

export function getGenomeCacheStats() {
  return {
    cachedGenomes: genomeCache.size,
    maxCache: MAX_CACHE,
    genomeDimension: GENOME_DIM,
    segments: {
      structure: STRUCTURE_DIM,
      orbitals: ORBITAL_DIM,
      phonons: PHONON_DIM,
      coupling: COUPLING_DIM,
      topology: TOPOLOGY_DIM,
      dimensionality: DIMENSIONALITY_DIM,
      composition: COMPOSITION_DIM,
      pairing: PAIRING_DIM,
    },
  };
}
