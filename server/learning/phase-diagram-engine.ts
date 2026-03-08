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
  const R_P = 0;

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
      const nwsAvg = (nwsA + nwsB) / 2;

      const fAB = 2 * fractions[elA] * fractions[elB];

      const isTransitionA = (dA.atomicNumber >= 21 && dA.atomicNumber <= 30) ||
                            (dA.atomicNumber >= 39 && dA.atomicNumber <= 48) ||
                            (dA.atomicNumber >= 72 && dA.atomicNumber <= 80);
      const isTransitionB = (dB.atomicNumber >= 21 && dB.atomicNumber <= 30) ||
                            (dB.atomicNumber >= 39 && dB.atomicNumber <= 48) ||
                            (dB.atomicNumber >= 72 && dB.atomicNumber <= 80);

      let Q = Q_P;
      if (isTransitionA && isTransitionB) {
        Q = Q_P;
      } else if (!isTransitionA && !isTransitionB) {
        Q = Q_P * 0.73;
      }

      const vAvg = (vA * fractions[elA] + vB * fractions[elB]) / (fractions[elA] + fractions[elB]);
      const interfaceEnergy = -P * deltaPhi * deltaPhi + Q * deltaNws * deltaNws - R_P;
      const contribution = fAB * vAvg * interfaceEnergy / (nwsAvg * nwsAvg);

      deltaH += contribution;
    }
  }

  const efPerAtom = deltaH / totalAtoms;
  return Math.max(-4.0, Math.min(2.0, efPerAtom));
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
    allPoints.push({ x: elements.indexOf(el) === 0 ? 1.0 : 0.0, y: 0, label: el });
  }
  if (elements.length === 2) {
    allPoints.push({ x: 0.0, y: 0, label: elements[1] });
  }

  for (const entry of formationEnergies) {
    const entryCounts = parseFormulaCounts(entry.formula);
    const entryTotal = getTotalAtoms(entryCounts);
    let x: number;
    if (elements.length === 2) {
      x = (entryCounts[elements[0]] || 0) / entryTotal;
    } else {
      x = (entryCounts[elements[0]] || 0) / entryTotal;
    }
    allPoints.push({ x, y: entry.energy, label: entry.formula });
  }

  let targetX: number;
  if (elements.length === 2) {
    targetX = (targetCounts[elements[0]] || 0) / targetTotal;
  } else {
    targetX = (targetCounts[elements[0]] || 0) / targetTotal;
  }
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
    ? Math.max(0.1, Math.min(2.5, eAboveHull * 6 + 0.2 * Math.log(avgMeltingPoint / 300)))
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
    const x = elementSet.length >= 2 ? (entry.fractions[elementSet[0]] || 0) : 0;
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
  verdict: "stable" | "near-hull" | "metastable" | "unstable";
  reason: string;
  hullDistance: number;
  formationEnergy: number;
  kineticBarrier?: number;
}

export async function passesStabilityGate(formula: string): Promise<StabilityGateResult> {
  const formationEnergy = computeMiedemaFormationEnergy(formula);
  let hullDistance: number;
  let decompositionProducts: string[] = [];

  try {
    const hullResult = await getCompetingPhases(formula);
    hullDistance = hullResult.energyAboveHull;
    decompositionProducts = hullResult.decompositionProducts;
  } catch {
    hullDistance = Math.max(0, formationEnergy * 0.3);
  }

  if (hullDistance > 0.50) {
    return {
      pass: false,
      verdict: "unstable",
      reason: `hull distance ${hullDistance.toFixed(4)} eV/atom > 0.50 threshold` +
        (decompositionProducts.length > 0 ? `, decomposes to ${decompositionProducts.join("+")}` : ""),
      hullDistance,
      formationEnergy,
    };
  }

  if (hullDistance > 0.25) {
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
    return {
      pass: false,
      verdict: "unstable",
      reason: `hull distance ${hullDistance.toFixed(4)} eV/atom > 0.25, kinetic barrier ${metastabilityCheck.kineticBarrier.toFixed(3)} eV <= 0.2 eV` +
        (decompositionProducts.length > 0 ? `, decomposes to ${decompositionProducts.join("+")}` : ""),
      hullDistance,
      formationEnergy,
    };
  }

  if (hullDistance > 0.1) {
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
    return {
      pass: false,
      verdict: "unstable",
      reason: `hull distance ${hullDistance.toFixed(4)} eV/atom > 0.1, kinetic barrier ${metastabilityCheck.kineticBarrier.toFixed(3)} eV <= 0.3 eV` +
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

  return {
    pass: false,
    verdict: "unstable",
    reason: `hull distance ${hullDistance.toFixed(4)} eV/atom in metastable range but kinetic barrier ${metastability.kineticBarrier.toFixed(3)} eV <= 0.5 eV`,
    hullDistance,
    formationEnergy,
    kineticBarrier: metastability.kineticBarrier,
  };
}

export async function runConvexHullAnalysis(
  emit: EventEmitter,
  formula: string
): Promise<ConvexHullResult> {
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

  return hullResult;
}
