import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { ELEMENTAL_DATA, getElementData } from "./elemental-data";

const STABILITY_TOLERANCE_MEV = 25;
const STABILITY_TOLERANCE = STABILITY_TOLERANCE_MEV / 1000;

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  return total > 0 ? total : 1;
}

const MIEDEMA_NONMETALS = new Set(["H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "I", "Xe", "Te", "As"]);

interface ParsedFormula {
  counts: Record<string, number>;
  elements: string[];
  totalAtoms: number;
  fractions: Record<string, number>;
}

function parseOnce(formula: string): ParsedFormula {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }
  return { counts, elements, totalAtoms, fractions };
}

function computeMiedemaFromParsed(parsed: ParsedFormula): number {
  const { elements, fractions } = parsed;
  if (elements.length < 2) return 0;

  const P = 14.1;
  const Q_P = 9.4;

  let deltaH = 0;

  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const elA = elements[i];
      const elB = elements[j];
      const dA = ELEMENTAL_DATA[elA];
      const dB = ELEMENTAL_DATA[elB];

      if (!dA || !dB) continue;

      const phiA = dA.miedemaPhiStar;
      const phiB = dB.miedemaPhiStar;
      const nwsA = dA.miedemaNws13;
      const nwsB = dB.miedemaNws13;
      const vA = dA.miedemaV23;
      const vB = dB.miedemaV23;

      if (phiA == null || phiB == null || nwsA == null || nwsB == null || vA == null || vB == null) continue;

      const safeNwsA = Math.max(1e-6, nwsA);
      const safeNwsB = Math.max(1e-6, nwsB);
      const deltaPhi = phiA - phiB;
      const deltaNws = safeNwsA - safeNwsB;
      const nwsAvgInv = 2 / (1 / safeNwsA + 1 / safeNwsB);

      const fAB = 2 * fractions[elA] * fractions[elB];

      const isDorFBlock = (z: number) =>
        (z >= 21 && z <= 30) || (z >= 39 && z <= 48) || (z >= 72 && z <= 80) ||
        (z >= 57 && z <= 71) || (z >= 89 && z <= 103);
      const isTransitionA = isDorFBlock(dA.atomicNumber);
      const isTransitionB = isDorFBlock(dB.atomicNumber);

      let Q = Q_P;
      if (isTransitionA && isTransitionB) {
        Q = Q_P;
      } else if (!isTransitionA && !isTransitionB) {
        Q = Q_P * 0.73;
      }

      const R_P = (isTransitionA !== isTransitionB) ? 0.5 : 0;

      const vAvg = (vA * fractions[elA] + vB * fractions[elB]) / (fractions[elA] + fractions[elB]);
      let interfaceEnergy = (-P * deltaPhi * deltaPhi + Q * deltaNws * deltaNws - R_P) / nwsAvgInv;

      const aIsNonmetal = MIEDEMA_NONMETALS.has(elA);
      const bIsNonmetal = MIEDEMA_NONMETALS.has(elB);
      if (aIsNonmetal !== bIsNonmetal) {
        interfaceEnergy -= (0.73 * deltaPhi * deltaPhi) / nwsAvgInv;
      }

      const contribution = fAB * vAvg * interfaceEnergy;

      deltaH += contribution;
    }
  }

  return Math.max(-5.0, Math.min(2.0, deltaH));
}

export function computeMiedemaFormationEnergy(formula: string): number {
  return computeMiedemaFromParsed(parseOnce(formula));
}

const OXIDE_ANIONS = new Set(["O", "F", "Cl", "Br", "I"]);
const CHALCOGENIDE_ANIONS = new Set(["S", "Se", "Te"]);
const PNICTIDE_ANIONS = new Set(["N", "P", "As", "Sb"]);

function classifyCompoundTypeParsed(parsed: ParsedFormula): "intermetallic" | "oxide" | "chalcogenide" | "pnictide" | "mixed" {
  const { elements, fractions } = parsed;
  const hasOxide = elements.some(el => OXIDE_ANIONS.has(el) && fractions[el] > 0.1);
  const hasChalc = elements.some(el => CHALCOGENIDE_ANIONS.has(el) && fractions[el] > 0.1);
  const hasPnic = elements.some(el => PNICTIDE_ANIONS.has(el) && fractions[el] > 0.1);
  if (hasOxide && (hasChalc || hasPnic)) return "mixed";
  if (hasOxide) return "oxide";
  if (hasChalc) return "chalcogenide";
  if (hasPnic) return "pnictide";
  return "intermetallic";
}

function computeIonicFromParsed(parsed: ParsedFormula): number {
  const { elements, counts, totalAtoms } = parsed;

  const OXIDE_ENERGIES: Record<string, number> = {
    "Li": -3.0, "Na": -2.1, "K": -1.8, "Rb": -1.7, "Cs": -1.6,
    "Mg": -3.0, "Ca": -3.2, "Sr": -3.0, "Ba": -2.8,
    "Al": -2.8, "Ga": -1.8, "In": -1.5,
    "Ti": -4.8, "Zr": -5.5, "Hf": -5.6,
    "V": -2.6, "Nb": -3.9, "Ta": -4.1,
    "Cr": -1.8, "Mo": -1.2, "W": -1.4,
    "Mn": -1.9, "Fe": -1.3, "Co": -1.0, "Ni": -1.0,
    "Cu": -0.6, "Zn": -1.7, "Y": -4.7, "La": -4.6,
    "Ce": -4.5, "Sc": -4.7, "Bi": -0.8,
  };

  let ionicEnergy = 0;
  let ionicWeight = 0;

  const anionElements = elements.filter(
    e => OXIDE_ANIONS.has(e) || CHALCOGENIDE_ANIONS.has(e) || PNICTIDE_ANIONS.has(e)
  );
  const anionFrac = anionElements.reduce((s, e) => s + counts[e] / totalAtoms, 0);

  for (const el of elements) {
    if (OXIDE_ANIONS.has(el) || CHALCOGENIDE_ANIONS.has(el) || PNICTIDE_ANIONS.has(el)) continue;
    const cationData = ELEMENTAL_DATA[el];
    if (!cationData) continue;

    const frac = counts[el] / totalAtoms;
    const basePairEnergy = OXIDE_ENERGIES[el] ?? -1.5;
    const cationEN = cationData.paulingElectronegativity ?? 1.5;

    let weightedPairEnergy = 0;
    let anionWeightSum = 0;

    for (const anEl of anionElements) {
      const anionData = ELEMENTAL_DATA[anEl];
      const anionEN = anionData?.paulingElectronegativity ?? 3.0;
      const deltaChi = Math.abs(anionEN - cationEN);
      const ionicityFactor = 0.5 + 0.5 * deltaChi / 2.5;
      const anionW = counts[anEl] / totalAtoms;
      weightedPairEnergy += anionW * basePairEnergy * ionicityFactor;
      anionWeightSum += anionW;
    }

    const effectivePairEnergy = anionWeightSum > 0 ? weightedPairEnergy / anionWeightSum : basePairEnergy;

    ionicEnergy += frac * effectivePairEnergy * anionFrac * 2;
    ionicWeight += frac;
  }

  if (ionicWeight < 0.01) return 0;
  return Math.max(-10.0, Math.min(2.0, ionicEnergy / ionicWeight));
}

const HYDRIDE_FORMATION_ENERGIES: Record<string, number> = {
  "Li": -0.94, "Na": -0.56, "K": -0.57, "Rb": -0.52, "Cs": -0.54,
  "Mg": -0.37, "Ca": -0.96, "Sr": -0.93, "Ba": -0.89,
  "Ti": -0.70, "Zr": -0.91, "Hf": -0.72,
  "V": -0.36, "Nb": -0.21, "Ta": -0.20,
  "Y": -1.13, "La": -1.09, "Sc": -0.96,
  "Pd": -0.10, "Ce": -1.06,
  "Fe": 0.15, "Co": 0.20, "Ni": 0.08, "Cu": 0.25,
  "Al": -0.04, "Ga": 0.12,
};

function computeHydrideFromParsed(parsed: ParsedFormula): number {
  const { counts, elements, totalAtoms } = parsed;

  const hCount = counts["H"] ?? 0;
  const hFrac = hCount / totalAtoms;
  const metalElements = elements.filter(el => el !== "H");

  if (metalElements.length === 0) return 0;

  let weightedEnergy = 0;
  let metalWeight = 0;

  for (const el of metalElements) {
    const frac = counts[el] / totalAtoms;
    const hydrideE = HYDRIDE_FORMATION_ENERGIES[el] ?? -0.3;
    weightedEnergy += frac * hydrideE;
    metalWeight += frac;
  }

  if (metalWeight < 0.01) return 0;

  const baseEnergy = weightedEnergy / metalWeight;
  const stoichFactor = Math.min(hFrac / (1 - hFrac + 0.01), 3.0);
  const energy = baseEnergy * stoichFactor;

  if (metalElements.length > 1) {
    const metalMiedema = computeMiedemaFormationEnergy(
      metalElements.map(el => el + (counts[el] ?? 1)).join("")
    );
    return Math.max(-5.0, Math.min(2.0, energy + 0.3 * metalMiedema));
  }

  return Math.max(-5.0, Math.min(2.0, energy));
}

function isHydrideParsed(parsed: ParsedFormula): boolean {
  const hFrac = (parsed.counts["H"] ?? 0) / parsed.totalAtoms;
  return parsed.elements.includes("H") && hFrac > 0.05;
}

export function estimateFormationEnergy(formula: string): number {
  const parsed = parseOnce(formula);
  const compType = classifyCompoundTypeParsed(parsed);
  if (compType === "intermetallic") {
    if (isHydrideParsed(parsed)) {
      return computeHydrideFromParsed(parsed);
    }
    return computeMiedemaFromParsed(parsed);
  }
  const ionicE = computeIonicFromParsed(parsed);
  if (compType === "mixed") {
    const { elements, fractions } = parsed;

    let ionicAnionFrac = 0;
    let metallicFrac = 0;
    for (const el of elements) {
      const f = fractions[el];
      if (OXIDE_ANIONS.has(el) || CHALCOGENIDE_ANIONS.has(el) || PNICTIDE_ANIONS.has(el)) {
        ionicAnionFrac += f;
      } else {
        metallicFrac += f;
      }
    }
    const ionicW = Math.max(0.1, Math.min(0.9, ionicAnionFrac / (ionicAnionFrac + metallicFrac + 0.01)));
    const miedemaE = computeMiedemaFromParsed(parsed);
    return ionicW * ionicE + (1 - ionicW) * miedemaE;
  }
  return ionicE;
}

export interface HullVertex {
  composition: string;
  energy: number;
  fractions?: Record<string, number>;
}

export interface ConvexHullResult {
  energyAboveHull: number;
  hullVertices: HullVertex[];
  decompositionProducts: string[];
  isOnHull: boolean;
}

export interface MetastabilityAssessment {
  isMetastable: boolean;
  kineticBarrier: number;
  estimatedLifetime: string;
  decompositionPathway: string[];
}

export interface PhaseDiagramEntry {
  formula: string;
  fractions: Record<string, number>;
  formationEnergy: number;
  isStable: boolean;
}

export interface PhaseBoundary {
  from: string;
  to: string;
  energy: number;
  reactionEnthalpy: number;
}

export interface PhaseDiagramResult {
  elements: string[];
  stablePhases: PhaseDiagramEntry[];
  unstablePhases: PhaseDiagramEntry[];
  hullVertices: HullVertex[];
  phaseBoundaries: PhaseBoundary[];
}

function getCompositionFractions(formula: string, elementSet: string[]): Record<string, number> {
  const counts = parseFormulaCounts(formula);
  const total = getTotalAtoms(counts);
  const fractions: Record<string, number> = {};
  for (const el of elementSet) {
    fractions[el] = (counts[el] || 0) / total;
  }
  return fractions;
}

function binaryCompositionX(counts: Record<string, number>, total: number, elA: string, elB: string): number {
  const fracA = (counts[elA] || 0) / total;
  const fracB = (counts[elB] || 0) / total;
  const sum = fracA + fracB;
  return sum > 1e-12 ? fracA / sum : 0.5;
}

function lowerConvexHull2D(points: { x: number; y: number; label: string }[]): { x: number; y: number; label: string }[] {
  if (points.length <= 2) return [...points];

  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);

  const hull: typeof sorted = [];
  for (const p of sorted) {
    while (hull.length >= 2) {
      const a = hull[hull.length - 2];
      const b = hull[hull.length - 1];
      const cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
      if (cross <= 0) {
        hull.pop();
      } else {
        break;
      }
    }
    hull.push(p);
  }

  return hull;
}

function interpolateHullEnergy(x: number, hull: { x: number; y: number }[]): number {
  if (hull.length === 0) return 0;
  if (hull.length === 1) return hull[0].y;

  if (x <= hull[0].x) return hull[0].y;
  if (x >= hull[hull.length - 1].x) return hull[hull.length - 1].y;

  for (let i = 0; i < hull.length - 1; i++) {
    if (x >= hull[i].x && x <= hull[i + 1].x) {
      const dx = hull[i + 1].x - hull[i].x;
      if (dx < 1e-12) return hull[i].y;
      const t = (x - hull[i].x) / dx;
      return hull[i].y * (1 - t) + hull[i + 1].y * t;
    }
  }

  return 0;
}

function computeBinarySliceHull(
  targetFormula: string,
  targetFormE: number,
  elA: string,
  elB: string,
  formationEnergies: { formula: string; energy: number }[]
): { eAboveHull: number; hull: { x: number; y: number; label: string }[]; targetX: number } {
  const targetCounts = parseFormulaCounts(targetFormula);
  const targetTotal = getTotalAtoms(targetCounts);
  const targetX = binaryCompositionX(targetCounts, targetTotal, elA, elB);

  const refPoints: { x: number; y: number; label: string }[] = [
    { x: 0, y: 0, label: elB },
    { x: 1, y: 0, label: elA },
  ];

  for (const entry of formationEnergies) {
    if (entry.formula === targetFormula) continue;
    const entryCounts = parseFormulaCounts(entry.formula);
    const entryTotal = getTotalAtoms(entryCounts);
    const entryElements = Object.keys(entryCounts);
    const relevant = entryElements.some(e => e === elA || e === elB);
    if (!relevant) continue;
    const x = binaryCompositionX(entryCounts, entryTotal, elA, elB);
    refPoints.push({ x, y: entry.energy, label: entry.formula });
  }

  const hull = lowerConvexHull2D(refPoints);
  const hullEnergy = interpolateHullEnergy(targetX, hull);
  const eAboveHull = Math.max(0, targetFormE - hullEnergy);

  return { eAboveHull, hull, targetX };
}

export function computeConvexHull(
  formula: string,
  elements: string[],
  formationEnergies: { formula: string; energy: number }[]
): ConvexHullResult {
  if (elements.length < 2) {
    return { energyAboveHull: 0, hullVertices: [], decompositionProducts: [], isOnHull: true };
  }

  const targetFormE = estimateFormationEnergy(formula);

  if (elements.length === 2) {
    const result = computeBinarySliceHull(formula, targetFormE, elements[0], elements[1], formationEnergies);
    const isOnHull = result.eAboveHull < STABILITY_TOLERANCE;

    const hullVertices: HullVertex[] = result.hull.map(p => ({
      composition: p.label,
      energy: p.y,
    }));

    const decompositionProducts: string[] = [];
    if (!isOnHull) {
      for (let i = 0; i < result.hull.length - 1; i++) {
        if (result.targetX >= result.hull[i].x && result.targetX <= result.hull[i + 1].x) {
          if (result.hull[i].label !== formula) decompositionProducts.push(result.hull[i].label);
          if (result.hull[i + 1].label !== formula) decompositionProducts.push(result.hull[i + 1].label);
          break;
        }
      }
      if (decompositionProducts.length === 0 && result.hull.length > 0) {
        decompositionProducts.push(result.hull[0].label);
      }
    }

    return { energyAboveHull: result.eAboveHull, hullVertices, decompositionProducts, isOnHull };
  }

  let worstEAboveHull = 0;
  let worstSliceHull: { x: number; y: number; label: string }[] = [];
  let worstTargetX = 0;
  let worstPair: [string, string] = [elements[0], elements[1]];

  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const slice = computeBinarySliceHull(
        formula, targetFormE, elements[i], elements[j], formationEnergies
      );
      if (slice.eAboveHull > worstEAboveHull) {
        worstEAboveHull = slice.eAboveHull;
        worstSliceHull = slice.hull;
        worstTargetX = slice.targetX;
        worstPair = [elements[i], elements[j]];
      }
    }
  }

  const isOnHull = worstEAboveHull < STABILITY_TOLERANCE;

  const hullVertices: HullVertex[] = worstSliceHull.map(p => ({
    composition: p.label,
    energy: p.y,
  }));

  const decompositionProducts: string[] = [];
  if (!isOnHull) {
    for (let i = 0; i < worstSliceHull.length - 1; i++) {
      if (worstTargetX >= worstSliceHull[i].x && worstTargetX <= worstSliceHull[i + 1].x) {
        if (worstSliceHull[i].label !== formula) decompositionProducts.push(worstSliceHull[i].label);
        if (worstSliceHull[i + 1].label !== formula) decompositionProducts.push(worstSliceHull[i + 1].label);
        break;
      }
    }
    if (decompositionProducts.length === 0 && worstSliceHull.length > 0) {
      decompositionProducts.push(worstSliceHull[0].label);
    }
  }

  return { energyAboveHull: worstEAboveHull, hullVertices, decompositionProducts, isOnHull };
}

const PSEUDO_BINARY_ELEMENT_LIMIT = 4;

interface CachedCompetingPhases {
  materialEntries: { formula: string; energy: number; elements: string[] }[];
  candidateEntries: { formula: string; energy: number; elements: string[] }[];
  builtAt: number;
}

let competingPhasesCache: CachedCompetingPhases | null = null;
const COMPETING_PHASES_CACHE_TTL_MS = 60_000;

async function getOrBuildCompetingPhasesCache(): Promise<CachedCompetingPhases> {
  const now = Date.now();
  if (competingPhasesCache && (now - competingPhasesCache.builtAt) < COMPETING_PHASES_CACHE_TTL_MS) {
    return competingPhasesCache;
  }

  const allMaterials = await storage.getMaterials(500, 0);
  const materialEntries = allMaterials.map(mat => ({
    formula: mat.formula,
    energy: mat.formationEnergy ?? estimateFormationEnergy(mat.formula),
    elements: parseFormulaElements(mat.formula),
  }));

  const candidates = await storage.getSuperconductorCandidates(200);
  const seenFormulas = new Set(materialEntries.map(m => m.formula));
  const candidateEntries = candidates
    .filter(c => !seenFormulas.has(c.formula))
    .map(cand => ({
      formula: cand.formula,
      energy: estimateFormationEnergy(cand.formula),
      elements: parseFormulaElements(cand.formula),
    }));

  competingPhasesCache = { materialEntries, candidateEntries, builtAt: now };
  return competingPhasesCache;
}

function selectPseudoBinaryAxes(
  formula: string,
  allElements: string[]
): string[] {
  if (allElements.length <= PSEUDO_BINARY_ELEMENT_LIMIT) return allElements;

  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const sorted = allElements
    .map(el => ({ el, frac: (counts[el] ?? 0) / totalAtoms }))
    .sort((a, b) => b.frac - a.frac);

  const majorElements = sorted.slice(0, 2).map(s => s.el);

  const remainingPool = sorted.slice(2);
  const groupA: string[] = [];
  const groupB: string[] = [];

  for (const { el } of remainingPool) {
    const data = ELEMENTAL_DATA[el];
    const isAnion = OXIDE_ANIONS.has(el) || CHALCOGENIDE_ANIONS.has(el) || PNICTIDE_ANIONS.has(el);
    if (isAnion) {
      groupA.push(el);
    } else {
      groupB.push(el);
    }
  }

  const pseudoElements = [...majorElements];
  if (groupA.length > 0) pseudoElements.push(groupA[0]);
  if (groupB.length > 0 && pseudoElements.length < PSEUDO_BINARY_ELEMENT_LIMIT) {
    pseudoElements.push(groupB[0]);
  }
  while (pseudoElements.length < Math.min(PSEUDO_BINARY_ELEMENT_LIMIT, allElements.length)) {
    const next = remainingPool.find(s => !pseudoElements.includes(s.el));
    if (!next) break;
    pseudoElements.push(next.el);
  }

  return pseudoElements;
}

export async function getCompetingPhases(
  formula: string
): Promise<ConvexHullResult> {
  const allElements = parseFormulaElements(formula);
  const hullElements = selectPseudoBinaryAxes(formula, allElements);
  const hullSet = new Set(hullElements);

  const cache = await getOrBuildCompetingPhasesCache();
  const competingFormulas: { formula: string; energy: number }[] = [];

  for (const entry of cache.materialEntries) {
    if (entry.formula !== formula && entry.elements.every(el => hullSet.has(el))) {
      competingFormulas.push({ formula: entry.formula, energy: entry.energy });
    }
  }

  for (const entry of cache.candidateEntries) {
    if (entry.formula !== formula && entry.elements.every(el => hullSet.has(el))) {
      competingFormulas.push({ formula: entry.formula, energy: entry.energy });
    }
  }

  return computeConvexHull(formula, hullElements, competingFormulas);
}

function surrogateKineticBarrier(
  eAboveHull: number,
  minMeltingPoint: number,
  avgMeltingPoint: number,
  numElements: number,
  avgMass: number
): number {
  if (eAboveHull <= 0) return 1.5;

  const minMeltFactor = 0.35 * Math.log(Math.max(minMeltingPoint, 200) / 200);
  const avgMeltFactor = 0.10 * Math.log(Math.max(avgMeltingPoint, 300) / 300);

  const configEntropy = numElements > 1 ? 0.05 * Math.log(numElements) : 0;

  const massFactor = 0.02 * Math.log(Math.max(avgMass, 10) / 10);

  const barrier = eAboveHull * 5.5 + minMeltFactor + avgMeltFactor + configEntropy + massFactor;

  return Math.max(0.05, Math.min(3.0, barrier));
}

export function assessMetastability(
  formula: string,
  eAboveHull: number,
  hullDecompositionProducts?: string[]
): MetastabilityAssessment {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);

  let avgMeltingPoint = 0;
  let minMeltingPoint = Infinity;
  let avgMass = 0;
  let count = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const n = counts[el] || 1;
    if (data) {
      const mp = data.meltingPoint ?? 1000;
      avgMeltingPoint += mp * n;
      if (mp < minMeltingPoint) minMeltingPoint = mp;
      avgMass += (data.atomicMass ?? 50) * n;
      count += n;
    }
  }
  avgMeltingPoint = count > 0 ? avgMeltingPoint / count : 1000;
  if (!isFinite(minMeltingPoint)) minMeltingPoint = 500;
  avgMass = count > 0 ? avgMass / count : 50;

  const kB = 8.617e-5;
  const attemptFrequency = 1e13;
  const kineticBarrier = surrogateKineticBarrier(eAboveHull, minMeltingPoint, avgMeltingPoint, elements.length, avgMass);

  const roomTempRate = attemptFrequency * Math.exp(-kineticBarrier / (kB * 300));
  let estimatedLifetime: string;

  if (roomTempRate < 1e-30) {
    estimatedLifetime = "effectively infinite (>10^10 years)";
  } else {
    const lifetimeSeconds = 1 / Math.max(roomTempRate, 1e-100);
    if (lifetimeSeconds > 3.15e16) {
      estimatedLifetime = `${(lifetimeSeconds / 3.15e7).toExponential(1)} years`;
    } else if (lifetimeSeconds > 3.15e7) {
      estimatedLifetime = `${Math.round(lifetimeSeconds / 3.15e7)} years`;
    } else if (lifetimeSeconds > 86400) {
      estimatedLifetime = `${Math.round(lifetimeSeconds / 86400)} days`;
    } else if (lifetimeSeconds > 3600) {
      estimatedLifetime = `${Math.round(lifetimeSeconds / 3600)} hours`;
    } else if (lifetimeSeconds > 60) {
      estimatedLifetime = `${Math.round(lifetimeSeconds / 60)} minutes`;
    } else {
      estimatedLifetime = `${lifetimeSeconds.toFixed(1)} seconds`;
    }
  }

  const isMetastable = eAboveHull > STABILITY_TOLERANCE && eAboveHull <= 0.2 && kineticBarrier > 0.5;

  let decompositionPathway: string[];
  if (hullDecompositionProducts && hullDecompositionProducts.length > 0) {
    decompositionPathway = [...hullDecompositionProducts];
  } else if (eAboveHull > 0) {
    decompositionPathway = elements.slice();
  } else {
    decompositionPathway = [];
  }

  return {
    isMetastable,
    kineticBarrier: Math.round(kineticBarrier * 1000) / 1000,
    estimatedLifetime,
    decompositionPathway,
  };
}

export function computePhaseDiagram(
  elementSet: string[],
  knownPhases: { formula: string; formationEnergy: number }[]
): PhaseDiagramResult {
  const allEntries: PhaseDiagramEntry[] = [];

  for (const el of elementSet) {
    allEntries.push({
      formula: el,
      fractions: { [el]: 1.0 },
      formationEnergy: 0,
      isStable: true,
    });
  }

  for (const phase of knownPhases) {
    const fracs = getCompositionFractions(phase.formula, elementSet);
    allEntries.push({
      formula: phase.formula,
      fractions: fracs,
      formationEnergy: phase.formationEnergy,
      isStable: false,
    });
  }

  if (elementSet.length === 2) {
    const hullPoints: { x: number; y: number; label: string }[] = [];
    for (const entry of allEntries) {
      const counts = parseFormulaCounts(entry.formula);
      const total = getTotalAtoms(counts);
      const x = binaryCompositionX(counts, total, elementSet[0], elementSet[1]);
      hullPoints.push({ x, y: entry.formationEnergy, label: entry.formula });
    }

    const hull = lowerConvexHull2D(hullPoints);
    const hullLabels = new Set(hull.map(h => h.label));

    for (const entry of allEntries) {
      entry.isStable = hullLabels.has(entry.formula);
    }

    const hullVertices: HullVertex[] = hull.map(p => ({ composition: p.label, energy: p.y }));
    const phaseBoundaries: PhaseBoundary[] = [];
    for (let i = 0; i < hull.length - 1; i++) {
      const dx = hull[i + 1].x - hull[i].x;
      const midEnergy = (hull[i].y + hull[i + 1].y) / 2;
      const reactionEnthalpy = dx > 1e-12 ? (hull[i + 1].y - hull[i].y) / dx : 0;
      phaseBoundaries.push({ from: hull[i].label, to: hull[i + 1].label, energy: midEnergy, reactionEnthalpy });
    }

    return {
      elements: elementSet,
      stablePhases: allEntries.filter(e => e.isStable),
      unstablePhases: allEntries.filter(e => !e.isStable),
      hullVertices,
      phaseBoundaries,
    };
  }

  const stableOnAllSlices = new Set(allEntries.map(e => e.formula));
  const allHullVertices: HullVertex[] = [];
  const allPhaseBoundaries: PhaseBoundary[] = [];

  for (let i = 0; i < elementSet.length; i++) {
    for (let j = i + 1; j < elementSet.length; j++) {
      const elA = elementSet[i];
      const elB = elementSet[j];

      const slicePoints: { x: number; y: number; label: string }[] = [];
      for (const entry of allEntries) {
        const counts = parseFormulaCounts(entry.formula);
        const total = getTotalAtoms(counts);
        const x = binaryCompositionX(counts, total, elA, elB);
        slicePoints.push({ x, y: entry.formationEnergy, label: entry.formula });
      }

      const hull = lowerConvexHull2D(slicePoints);
      const hullLabels = new Set(hull.map(h => h.label));

      for (const entry of allEntries) {
        if (!hullLabels.has(entry.formula)) {
          stableOnAllSlices.delete(entry.formula);
        }
      }

      for (const p of hull) {
        if (!allHullVertices.some(v => v.composition === p.label)) {
          allHullVertices.push({ composition: p.label, energy: p.y });
        }
      }
      for (let k = 0; k < hull.length - 1; k++) {
        const dx = hull[k + 1].x - hull[k].x;
        const midEnergy = (hull[k].y + hull[k + 1].y) / 2;
        const reactionEnthalpy = dx > 1e-12 ? (hull[k + 1].y - hull[k].y) / dx : 0;
        allPhaseBoundaries.push({ from: hull[k].label, to: hull[k + 1].label, energy: midEnergy, reactionEnthalpy });
      }
    }
  }

  for (const entry of allEntries) {
    entry.isStable = stableOnAllSlices.has(entry.formula);
  }

  return {
    elements: elementSet,
    stablePhases: allEntries.filter(e => e.isStable),
    unstablePhases: allEntries.filter(e => !e.isStable),
    hullVertices: allHullVertices,
    phaseBoundaries: allPhaseBoundaries,
  };
}

export interface StabilityGateResult {
  pass: boolean;
  verdict: "stable" | "near-hull" | "metastable" | "unstable" | "entropy-stabilized";
  reason: string;
  hullDistance: number;
  formationEnergy: number;
  kineticBarrier?: number;
  configurationalEntropy?: number;
}

function computeConfigurationalEntropy(formula: string): { deltaSMix: number; numElements: number; fractions: number[] } {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  if (totalAtoms === 0 || elements.length < 2) {
    return { deltaSMix: 0, numElements: elements.length, fractions: [] };
  }
  const fractions = elements.map(el => counts[el] / totalAtoms);
  const R = 8.314;
  const deltaSMix = -R * fractions.reduce((sum, xi) => sum + (xi > 0 ? xi * Math.log(xi) : 0), 0);
  return { deltaSMix, numElements: elements.length, fractions };
}

function isEntropyStabilized(formula: string, formationEnergy: number): { qualifies: boolean; deltaSMix: number; reason: string } {
  const R = 8.314;
  const { deltaSMix, numElements, fractions } = computeConfigurationalEntropy(formula);

  if (numElements < 4) {
    return { qualifies: false, deltaSMix, reason: "fewer than 4 elements" };
  }

  if (deltaSMix <= 1.5 * R) {
    return { qualifies: false, deltaSMix, reason: `configurational entropy ${(deltaSMix / R).toFixed(2)}R <= 1.5R` };
  }

  if (formationEnergy > 0.1) {
    return { qualifies: false, deltaSMix, reason: `formation energy ${formationEnergy.toFixed(4)} eV/atom > 0.1 threshold` };
  }

  const synthesisTempK = 1000;
  const tDeltaS_eV = (deltaSMix * synthesisTempK) / 96485;
  return {
    qualifies: true,
    deltaSMix,
    reason: `high-entropy alloy (${numElements} elements, ΔS_mix=${(deltaSMix / R).toFixed(2)}R, T·ΔS@1000K=${tDeltaS_eV.toFixed(3)} eV/atom overcomes ΔH=${formationEnergy.toFixed(4)} eV/atom)`,
  };
}

function pressureHullMultiplier(pressureGpa: number): number {
  if (pressureGpa <= 0) return 1.0;
  if (pressureGpa >= 200) return 3.0;
  return 1.0 + (pressureGpa / 100);
}

export async function passesStabilityGate(formula: string, pressureGpa: number = 0): Promise<StabilityGateResult> {
  const compoundType = classifyCompoundType(formula);
  const miedemaApplicable = compoundType === "intermetallic" || compoundType === "pnictide";
  const formationEnergy = miedemaApplicable ? computeMiedemaFormationEnergy(formula) : 0;
  let hullDistance: number;
  let decompositionProducts: string[] = [];

  try {
    const hullResult = await getCompetingPhases(formula);
    hullDistance = hullResult.energyAboveHull;
    decompositionProducts = hullResult.decompositionProducts;
  } catch {
    hullDistance = miedemaApplicable ? Math.max(0, formationEnergy * 0.3) : 0.05;
  }

  const pMult = pressureHullMultiplier(pressureGpa);

  if (hullDistance > 0.20 * pMult) {
    const entropyCheck = isEntropyStabilized(formula, formationEnergy);
    if (entropyCheck.qualifies) {
      return {
        pass: true,
        verdict: "entropy-stabilized",
        reason: entropyCheck.reason,
        hullDistance,
        formationEnergy,
        configurationalEntropy: entropyCheck.deltaSMix,
      };
    }
    return {
      pass: false,
      verdict: "unstable",
      reason: `hull distance ${hullDistance.toFixed(4)} eV/atom > ${(0.20 * pMult).toFixed(3)} threshold${pressureGpa > 0 ? ` @${pressureGpa}GPa` : ""}` +
        (decompositionProducts.length > 0 ? `, decomposes to ${decompositionProducts.join("+")}` : ""),
      hullDistance,
      formationEnergy,
    };
  }

  if (hullDistance > 0.15 * pMult) {
    const metastabilityCheck = assessMetastability(formula, hullDistance);
    if (metastabilityCheck.kineticBarrier > 0.2) {
      return {
        pass: true,
        verdict: "metastable" as any,
        reason: `exploratory-metastable (distance=${hullDistance.toFixed(4)} eV/atom, barrier=${metastabilityCheck.kineticBarrier.toFixed(3)} eV, lifetime=${metastabilityCheck.estimatedLifetime})`,
        hullDistance,
        formationEnergy,
        kineticBarrier: metastabilityCheck.kineticBarrier,
      };
    }
    const entropyCheck = isEntropyStabilized(formula, formationEnergy);
    if (entropyCheck.qualifies) {
      return {
        pass: true,
        verdict: "entropy-stabilized",
        reason: entropyCheck.reason,
        hullDistance,
        formationEnergy,
        configurationalEntropy: entropyCheck.deltaSMix,
      };
    }
    return {
      pass: false,
      verdict: "unstable",
      reason: `hull distance ${hullDistance.toFixed(4)} eV/atom > ${(0.15 * pMult).toFixed(3)}${pressureGpa > 0 ? ` @${pressureGpa}GPa` : ""}, kinetic barrier ${metastabilityCheck.kineticBarrier.toFixed(3)} eV <= 0.2 eV` +
        (decompositionProducts.length > 0 ? `, decomposes to ${decompositionProducts.join("+")}` : ""),
      hullDistance,
      formationEnergy,
    };
  }

  if (hullDistance > 0.1 * pMult) {
    const metastabilityCheck = assessMetastability(formula, hullDistance);
    if (metastabilityCheck.kineticBarrier > 0.3) {
      return {
        pass: true,
        verdict: "metastable",
        reason: `metastable-tier3 (distance=${hullDistance.toFixed(4)} eV/atom, barrier=${metastabilityCheck.kineticBarrier.toFixed(3)} eV, lifetime=${metastabilityCheck.estimatedLifetime})`,
        hullDistance,
        formationEnergy,
        kineticBarrier: metastabilityCheck.kineticBarrier,
      };
    }
    const entropyCheck2 = isEntropyStabilized(formula, formationEnergy);
    if (entropyCheck2.qualifies) {
      return {
        pass: true,
        verdict: "entropy-stabilized",
        reason: entropyCheck2.reason,
        hullDistance,
        formationEnergy,
        configurationalEntropy: entropyCheck2.deltaSMix,
      };
    }
    return {
      pass: false,
      verdict: "unstable",
      reason: `hull distance ${hullDistance.toFixed(4)} eV/atom > ${(0.1 * pMult).toFixed(3)}${pressureGpa > 0 ? ` @${pressureGpa}GPa` : ""}, kinetic barrier ${metastabilityCheck.kineticBarrier.toFixed(3)} eV <= 0.3 eV` +
        (decompositionProducts.length > 0 ? `, decomposes to ${decompositionProducts.join("+")}` : ""),
      hullDistance,
      formationEnergy,
      kineticBarrier: metastabilityCheck.kineticBarrier,
    };
  }

  if (hullDistance <= STABILITY_TOLERANCE) {
    return {
      pass: true,
      verdict: "stable",
      reason: `on convex hull (distance=${hullDistance.toFixed(4)} eV/atom)`,
      hullDistance,
      formationEnergy,
    };
  }

  if (hullDistance <= 0.05) {
    return {
      pass: true,
      verdict: "near-hull",
      reason: `near hull (distance=${hullDistance.toFixed(4)} eV/atom)`,
      hullDistance,
      formationEnergy,
    };
  }

  const metastability = assessMetastability(formula, hullDistance);
  if (metastability.kineticBarrier > 0.5) {
    return {
      pass: true,
      verdict: "metastable",
      reason: `metastable (distance=${hullDistance.toFixed(4)} eV/atom, barrier=${metastability.kineticBarrier.toFixed(3)} eV, lifetime=${metastability.estimatedLifetime})`,
      hullDistance,
      formationEnergy,
      kineticBarrier: metastability.kineticBarrier,
    };
  }

  const entropyCheck3 = isEntropyStabilized(formula, formationEnergy);
  if (entropyCheck3.qualifies) {
    return {
      pass: true,
      verdict: "entropy-stabilized",
      reason: entropyCheck3.reason,
      hullDistance,
      formationEnergy,
      configurationalEntropy: entropyCheck3.deltaSMix,
    };
  }

  return {
    pass: false,
    verdict: "unstable",
    reason: `hull distance ${hullDistance.toFixed(4)} eV/atom in metastable range but kinetic barrier ${metastability.kineticBarrier.toFixed(3)} eV <= 0.5 eV`,
    hullDistance,
    formationEnergy,
    kineticBarrier: metastability.kineticBarrier,
  };
}

const convexHullCache = new Map<string, { result: ConvexHullResult; timestamp: number }>();
const HULL_CACHE_TTL_MS = 30 * 60 * 1000;
const MAX_HULL_CACHE_SIZE = 200;

function getChemicalSystem(formula: string): string {
  const elements = formula.match(/[A-Z][a-z]*/g) ?? [];
  return [...new Set(elements)].sort().join("-");
}

export function invalidateHullCache(formula: string): void {
  convexHullCache.delete(formula);
}

export async function runConvexHullAnalysis(
  emit: EventEmitter,
  formula: string
): Promise<ConvexHullResult> {
  const cached = convexHullCache.get(formula);
  if (cached && (Date.now() - cached.timestamp) < HULL_CACHE_TTL_MS) {
    emit("log", {
      phase: "phase-11",
      event: "Convex hull cache hit",
      detail: `${formula}: using cached result (age=${((Date.now() - cached.timestamp) / 1000).toFixed(0)}s)`,
      dataSource: "Phase Diagram Engine",
    });
    return cached.result;
  }

  const hullResult = await getCompetingPhases(formula);
  const metastability = assessMetastability(formula, hullResult.energyAboveHull, hullResult.decompositionProducts);

  const decompStr = hullResult.decompositionProducts.length > 0
    ? hullResult.decompositionProducts.join(" + ")
    : "none";

  emit("log", {
    phase: "phase-11",
    event: "Convex hull computed",
    detail: `${formula}: eAboveHull=${hullResult.energyAboveHull.toFixed(4)} eV/atom, onHull=${hullResult.isOnHull}, decomposition=${decompStr}, metastable=${metastability.isMetastable}, barrier=${metastability.kineticBarrier.toFixed(3)} eV, lifetime=${metastability.estimatedLifetime}`,
    dataSource: "Phase Diagram Engine",
  });

  if (convexHullCache.size >= MAX_HULL_CACHE_SIZE) {
    let oldestKey = "";
    let oldestTime = Infinity;
    for (const [k, v] of convexHullCache) {
      if (v.timestamp < oldestTime) { oldestTime = v.timestamp; oldestKey = k; }
    }
    if (oldestKey) convexHullCache.delete(oldestKey);
  }
  convexHullCache.set(formula, { result: hullResult, timestamp: Date.now() });

  return hullResult;
}
