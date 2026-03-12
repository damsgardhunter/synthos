import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { ELEMENTAL_DATA, getElementData } from "./elemental-data";

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

export function computeMiedemaFormationEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return 0;

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }

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

      const deltaPhi = phiA - phiB;
      const deltaNws = nwsA - nwsB;
      const nwsAvgInv = 2 / (1 / nwsA + 1 / nwsB);

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

const OXIDE_ANIONS = new Set(["O", "F", "Cl", "Br", "I"]);
const CHALCOGENIDE_ANIONS = new Set(["S", "Se", "Te"]);
const PNICTIDE_ANIONS = new Set(["N", "P", "As", "Sb"]);

function classifyCompoundType(formula: string): "intermetallic" | "oxide" | "chalcogenide" | "pnictide" | "mixed" {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hasOxide = elements.some(el => OXIDE_ANIONS.has(el) && (counts[el] / totalAtoms) > 0.1);
  const hasChalc = elements.some(el => CHALCOGENIDE_ANIONS.has(el) && (counts[el] / totalAtoms) > 0.1);
  const hasPnic = elements.some(el => PNICTIDE_ANIONS.has(el) && (counts[el] / totalAtoms) > 0.1);
  if (hasOxide && (hasChalc || hasPnic)) return "mixed";
  if (hasOxide) return "oxide";
  if (hasChalc) return "chalcogenide";
  if (hasPnic) return "pnictide";
  return "intermetallic";
}

function computeIonicFormationEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);

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

function computeHydrideFormationEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);

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

function isHydrideIntermetallic(formula: string): boolean {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const hFrac = (counts["H"] ?? 0) / totalAtoms;
  return elements.includes("H") && hFrac > 0.05;
}

export function estimateFormationEnergy(formula: string): number {
  const compType = classifyCompoundType(formula);
  if (compType === "intermetallic") {
    if (isHydrideIntermetallic(formula)) {
      return computeHydrideFormationEnergy(formula);
    }
    return computeMiedemaFormationEnergy(formula);
  }
  const ionicE = computeIonicFormationEnergy(formula);
  if (compType === "mixed") {
    const counts = parseFormulaCounts(formula);
    const totalAtoms = getTotalAtoms(counts);
    const elements = Object.keys(counts);

    let ionicAnionFrac = 0;
    let metallicFrac = 0;
    for (const el of elements) {
      const f = counts[el] / totalAtoms;
      if (OXIDE_ANIONS.has(el) || CHALCOGENIDE_ANIONS.has(el) || PNICTIDE_ANIONS.has(el)) {
        ionicAnionFrac += f;
      } else {
        metallicFrac += f;
      }
    }
    const ionicWeight = Math.max(0.1, Math.min(0.9, ionicAnionFrac / (ionicAnionFrac + metallicFrac + 0.01)));
    const miedemaE = computeMiedemaFormationEnergy(formula);
    return ionicWeight * ionicE + (1 - ionicWeight) * miedemaE;
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

export interface PhaseDiagramResult {
  elements: string[];
  stablePhases: PhaseDiagramEntry[];
  unstablePhases: PhaseDiagramEntry[];
  hullVertices: HullVertex[];
  phaseBoundaries: { from: string; to: string; energy: number }[];
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

function compositionToX(counts: Record<string, number>, total: number, elements: string[]): number {
  if (elements.length === 2) {
    return (counts[elements[0]] || 0) / total;
  }
  let x = 0;
  for (let i = 0; i < elements.length; i++) {
    const frac = (counts[elements[i]] || 0) / total;
    x += frac * (i + 1);
  }
  return x / elements.length;
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

export function computeConvexHull(
  formula: string,
  elements: string[],
  formationEnergies: { formula: string; energy: number }[]
): ConvexHullResult {
  if (elements.length < 2) {
    return { energyAboveHull: 0, hullVertices: [], decompositionProducts: [], isOnHull: true };
  }

  const targetCounts = parseFormulaCounts(formula);
  const targetTotal = getTotalAtoms(targetCounts);
  const targetFormE = computeMiedemaFormationEnergy(formula);

  const allPoints: { x: number; y: number; label: string }[] = [];

  for (const el of elements) {
    const elCounts: Record<string, number> = { [el]: 1 };
    allPoints.push({ x: compositionToX(elCounts, 1, elements), y: 0, label: el });
  }

  for (const entry of formationEnergies) {
    const entryCounts = parseFormulaCounts(entry.formula);
    const entryTotal = getTotalAtoms(entryCounts);
    const x = compositionToX(entryCounts, entryTotal, elements);
    allPoints.push({ x, y: entry.energy, label: entry.formula });
  }

  const targetX = compositionToX(targetCounts, targetTotal, elements);
  allPoints.push({ x: targetX, y: targetFormE, label: formula });

  const hull = lowerConvexHull2D(allPoints);

  const hullEnergy = interpolateHullEnergy(targetX, hull);
  const energyAboveHull = Math.max(0, targetFormE - hullEnergy);

  const isOnHull = energyAboveHull < 0.005;

  const hullVertices: HullVertex[] = hull.map(p => ({
    composition: p.label,
    energy: p.y,
  }));

  const decompositionProducts: string[] = [];
  if (!isOnHull) {
    for (let i = 0; i < hull.length - 1; i++) {
      if (targetX >= hull[i].x && targetX <= hull[i + 1].x) {
        if (hull[i].label !== formula) decompositionProducts.push(hull[i].label);
        if (hull[i + 1].label !== formula) decompositionProducts.push(hull[i + 1].label);
        break;
      }
    }
    if (decompositionProducts.length === 0 && hull.length > 0) {
      decompositionProducts.push(hull[0].label);
    }
  }

  return { energyAboveHull, hullVertices, decompositionProducts, isOnHull };
}

export async function getCompetingPhases(
  formula: string
): Promise<ConvexHullResult> {
  const elements = parseFormulaElements(formula);

  const allMaterials = await storage.getMaterials(500, 0);
  const competingFormulas: { formula: string; energy: number }[] = [];

  for (const mat of allMaterials) {
    const matElements = parseFormulaElements(mat.formula);
    const isSubset = matElements.every(el => elements.includes(el));
    if (isSubset && mat.formula !== formula) {
      const energy = mat.formationEnergy ?? computeMiedemaFormationEnergy(mat.formula);
      competingFormulas.push({ formula: mat.formula, energy });
    }
  }

  const candidates = await storage.getSuperconductorCandidates(200);
  for (const cand of candidates) {
    const candElements = parseFormulaElements(cand.formula);
    const isSubset = candElements.every(el => elements.includes(el));
    if (isSubset && cand.formula !== formula && !competingFormulas.some(cf => cf.formula === cand.formula)) {
      const energy = computeMiedemaFormationEnergy(cand.formula);
      competingFormulas.push({ formula: cand.formula, energy });
    }
  }

  const result = computeConvexHull(formula, elements, competingFormulas);

  return result;
}

export function assessMetastability(
  formula: string,
  eAboveHull: number
): MetastabilityAssessment {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);

  let avgMeltingPoint = 0;
  let count = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.meltingPoint) {
      avgMeltingPoint += data.meltingPoint * (counts[el] || 1);
      count += (counts[el] || 1);
    }
  }
  avgMeltingPoint = count > 0 ? avgMeltingPoint / count : 1000;

  const kB = 8.617e-5;
  const attemptFrequency = 1e13;
  const kineticBarrier = eAboveHull > 0
    ? Math.max(0.1, Math.min(2.5, eAboveHull * 6 + 0.2 * Math.log(Math.max(avgMeltingPoint, 300) / 300)))
    : 1.5;

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

  const isMetastable = eAboveHull > 0.005 && eAboveHull <= 0.2 && kineticBarrier > 0.5;

  const decompositionPathway: string[] = [];
  if (eAboveHull > 0) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const binary = `${elements[i]}${elements[j]}`;
        const binaryE = computeMiedemaFormationEnergy(binary);
        if (binaryE < 0) {
          decompositionPathway.push(binary);
        }
      }
    }
    if (decompositionPathway.length === 0) {
      decompositionPathway.push(...elements);
    }
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

  const hullPoints: { x: number; y: number; label: string }[] = [];
  for (const entry of allEntries) {
    const counts = parseFormulaCounts(entry.formula);
    const total = getTotalAtoms(counts);
    const x = elementSet.length >= 2 ? compositionToX(counts, total, elementSet) : 0;
    hullPoints.push({ x, y: entry.formationEnergy, label: entry.formula });
  }

  const hull = lowerConvexHull2D(hullPoints);
  const hullLabels = new Set(hull.map(h => h.label));

  for (const entry of allEntries) {
    entry.isStable = hullLabels.has(entry.formula);
  }

  const stablePhases = allEntries.filter(e => e.isStable);
  const unstablePhases = allEntries.filter(e => !e.isStable);

  const hullVertices: HullVertex[] = hull.map(p => ({
    composition: p.label,
    energy: p.y,
  }));

  const phaseBoundaries: { from: string; to: string; energy: number }[] = [];
  for (let i = 0; i < hull.length - 1; i++) {
    phaseBoundaries.push({
      from: hull[i].label,
      to: hull[i + 1].label,
      energy: (hull[i].y + hull[i + 1].y) / 2,
    });
  }

  return {
    elements: elementSet,
    stablePhases,
    unstablePhases,
    hullVertices,
    phaseBoundaries,
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

  if (hullDistance <= 0.005) {
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
  const metastability = assessMetastability(formula, hullResult.energyAboveHull);

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
