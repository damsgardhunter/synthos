import { ELEMENTAL_DATA, getElementData, isTransitionMetal, isRareEarth, isActinide } from "./elemental-data";
import { normalizeFormula } from "./utils";
import type { EventEmitter } from "./engine";

function parseFormulaElements(formula: string): string[] {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
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
  return Object.values(counts).reduce((s, n) => s + n, 0) || 1;
}

function buildFormula(counts: Record<string, number>): string {
  const parts: string[] = [];
  for (const [el, n] of Object.entries(counts)) {
    if (n <= 0) continue;
    if (Number.isInteger(n) && n === 1) parts.push(el);
    else if (Number.isInteger(n)) parts.push(`${el}${n}`);
    else parts.push(`${el}${Math.round(n * 100) / 100}`);
  }
  return parts.join("");
}

export type PrototypeType =
  | "rocksalt" | "perovskite" | "hexagonal" | "layered"
  | "bcc" | "fcc" | "spinel" | "fluorite" | "rutile" | "pyrite";

const PROTOTYPE_LATTICE: Record<PrototypeType, { a: number; c_over_a: number; crystalSystem: string }> = {
  rocksalt:   { a: 4.2,  c_over_a: 1.0,  crystalSystem: "cubic" },
  perovskite: { a: 3.9,  c_over_a: 1.0,  crystalSystem: "cubic" },
  hexagonal:  { a: 3.2,  c_over_a: 1.63, crystalSystem: "hexagonal" },
  layered:    { a: 3.8,  c_over_a: 3.2,  crystalSystem: "tetragonal" },
  bcc:        { a: 3.3,  c_over_a: 1.0,  crystalSystem: "cubic" },
  fcc:        { a: 3.6,  c_over_a: 1.0,  crystalSystem: "cubic" },
  spinel:     { a: 8.1,  c_over_a: 1.0,  crystalSystem: "cubic" },
  fluorite:   { a: 5.5,  c_over_a: 1.0,  crystalSystem: "cubic" },
  rutile:     { a: 4.6,  c_over_a: 0.64, crystalSystem: "tetragonal" },
  pyrite:     { a: 5.4,  c_over_a: 1.0,  crystalSystem: "cubic" },
};

export function assignPrototype(formula: string): PrototypeType {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const hasO = elements.includes("O");
  const hasH = elements.includes("H");
  const hasS = elements.includes("S") || elements.includes("Se") || elements.includes("Te");
  const hasB = elements.includes("B");
  const hasN = elements.includes("N");
  const hasC = elements.includes("C");

  const metalEls = elements.filter(e =>
    isTransitionMetal(e) || isRareEarth(e) || isActinide(e) ||
    ["Al", "Mg", "Ca", "Sr", "Ba", "Li", "Na", "K", "Ga", "In", "Tl", "Sn", "Pb", "Bi"].includes(e)
  );
  const metalFrac = metalEls.reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
  const nonMetalEls = elements.filter(e => !metalEls.includes(e));

  if (hasO && elements.length >= 3) {
    const oCount = counts["O"] || 0;
    const nonOCount = totalAtoms - oCount;
    const ratio = oCount / Math.max(1, nonOCount);

    if (elements.includes("Cu") && elements.length >= 3) return "layered";

    if (ratio >= 1.3 && ratio <= 1.7 && nonMetalEls.length <= 1) return "perovskite";

    if (metalEls.length >= 2 && oCount >= 4) return "spinel";

    if (ratio >= 1.8 && ratio <= 2.2) return "fluorite";

    if (ratio >= 0.9 && ratio <= 1.1 && elements.length === 2) return "rocksalt";

    if (ratio >= 1.8 && ratio <= 2.2 && elements.length === 2) return "rutile";

    return "perovskite";
  }

  if (hasS && metalEls.length >= 1) {
    const sCount = (counts["S"] || 0) + (counts["Se"] || 0) + (counts["Te"] || 0);
    const metalCount = metalEls.reduce((s, e) => s + (counts[e] || 0), 0);
    if (sCount >= metalCount * 2) return "pyrite";
    if (sCount >= metalCount) return "layered";
    return "rocksalt";
  }

  if (hasH && metalEls.length >= 1) {
    const hCount = counts["H"] || 0;
    const metalCount = metalEls.reduce((s, e) => s + (counts[e] || 0), 0);
    if (hCount / Math.max(1, metalCount) >= 3) return "fcc";
    return "rocksalt";
  }

  if ((hasB || hasN || hasC) && metalEls.length >= 1) {
    return "hexagonal";
  }

  if (metalFrac >= 0.95) {
    if (elements.length >= 1) {
      const avgVE = elements.reduce((s, e) => {
        const d = getElementData(e);
        return s + (d?.valenceElectrons ?? 4) * (counts[e] || 1);
      }, 0) / totalAtoms;
      if (avgVE <= 6) return "bcc";
      return "fcc";
    }
  }

  return "bcc";
}

export interface DistortedLattice {
  formula: string;
  distortionType: string;
  magnitude: number;
  latticeParams: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  energyPenalty: number;
}

function estimateDistortionEnergy(distortionType: string, magnitude: number, formula: string): number {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  let avgBulkModulus = 0;
  let count = 0;
  for (const el of elements) {
    const d = getElementData(el);
    if (d?.bulkModulus) {
      avgBulkModulus += d.bulkModulus * (counts[el] || 1);
      count += (counts[el] || 1);
    }
  }
  avgBulkModulus = count > 0 ? avgBulkModulus / count : 100;

  const stiffnessFactor = avgBulkModulus / 100;
  const strainSquared = magnitude * magnitude;

  switch (distortionType) {
    case "tetragonal":
      return 0.05 * stiffnessFactor * strainSquared * 100;
    case "orthorhombic":
      return 0.08 * stiffnessFactor * strainSquared * 100;
    case "monoclinic":
      return 0.12 * stiffnessFactor * (magnitude / 5) * (magnitude / 5);
    default:
      return 0.1 * stiffnessFactor * strainSquared * 100;
  }
}

export function generateDistortedLattices(formula: string, prototype: PrototypeType): DistortedLattice[] {
  const base = PROTOTYPE_LATTICE[prototype];
  const results: DistortedLattice[] = [];
  const MAX_ENERGY = 0.5;

  const tetragonalDistortions = [0.10, 0.15, 0.20, 0.25, 0.30];
  for (const mag of tetragonalDistortions) {
    for (const sign of [1, -1]) {
      const caRatio = base.c_over_a * (1 + sign * mag);
      const a = base.a;
      const c = a * caRatio;
      const energy = estimateDistortionEnergy("tetragonal", mag, formula);
      if (energy <= MAX_ENERGY) {
        results.push({
          formula,
          distortionType: `tetragonal c/a ${sign > 0 ? "+" : "-"}${(mag * 100).toFixed(0)}%`,
          magnitude: mag,
          latticeParams: { a, b: a, c, alpha: 90, beta: 90, gamma: 90 },
          energyPenalty: Math.round(energy * 1000) / 1000,
        });
      }
    }
  }

  const orthoDistortions = [0.02, 0.05, 0.08, 0.10];
  for (const mag of orthoDistortions) {
    const a = base.a;
    const b = a * (1 + mag);
    const c = a * base.c_over_a;
    const energy = estimateDistortionEnergy("orthorhombic", mag, formula);
    if (energy <= MAX_ENERGY) {
      results.push({
        formula,
        distortionType: `orthorhombic b/a +${(mag * 100).toFixed(0)}%`,
        magnitude: mag,
        latticeParams: { a, b, c, alpha: 90, beta: 90, gamma: 90 },
        energyPenalty: Math.round(energy * 1000) / 1000,
      });
    }
  }

  const monoclinicTilts = [85, 87, 89, 91, 93, 95];
  for (const beta of monoclinicTilts) {
    if (beta === 90) continue;
    const tiltMag = Math.abs(90 - beta);
    const energy = estimateDistortionEnergy("monoclinic", tiltMag, formula);
    if (energy <= MAX_ENERGY) {
      results.push({
        formula,
        distortionType: `monoclinic beta=${beta}deg`,
        magnitude: tiltMag,
        latticeParams: { a: base.a, b: base.a, c: base.a * base.c_over_a, alpha: 90, beta, gamma: 90 },
        energyPenalty: Math.round(energy * 1000) / 1000,
      });
    }
  }

  return results;
}

export interface LayeredStructure {
  formula: string;
  layerType: string;
  layerCount: number;
  dimensionality: string;
}

export function generateLayeredStructures(formula: string, prototype: PrototypeType): LayeredStructure[] {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const results: LayeredStructure[] = [];

  const hasO = elements.includes("O");
  const metalEls = elements.filter(e =>
    isTransitionMetal(e) || isRareEarth(e) || isActinide(e) ||
    ["Ca", "Sr", "Ba", "La", "Y"].includes(e)
  );
  const bSiteEls = elements.filter(e => isTransitionMetal(e));

  if (hasO && metalEls.length >= 1 && bSiteEls.length >= 1) {
    const aSite = metalEls.find(e => ["Ca", "Sr", "Ba", "La", "Y"].includes(e)) || metalEls[0];
    const bSite = bSiteEls.find(e => e !== aSite) || bSiteEls[0];

    for (const n of [1, 2, 3]) {
      const rpFormula = `${aSite}${n + 1}${bSite}${n}O${3 * n + 1}`;
      results.push({
        formula: normalizeFormula(rpFormula),
        layerType: `Ruddlesden-Popper n=${n}`,
        layerCount: n,
        dimensionality: n === 1 ? "quasi-2D" : n === 2 ? "quasi-2D" : "3D",
      });
    }
  }

  if (metalEls.length >= 1) {
    const spacerEls = [
      { el: "Sr", anion: "O", type: "rocksalt-block" },
      { el: "Ba", anion: "O", type: "rocksalt-block" },
      { el: "Ca", anion: "F", type: "fluorite-block" },
    ];
    for (const spacer of spacerEls) {
      if (!elements.includes(spacer.el)) {
        const newCounts = { ...counts };
        newCounts[spacer.el] = (newCounts[spacer.el] || 0) + 2;
        newCounts[spacer.anion] = (newCounts[spacer.anion] || 0) + 2;
        const spacerFormula = buildFormula(newCounts);
        results.push({
          formula: normalizeFormula(spacerFormula),
          layerType: `${spacer.type} (${spacer.el}${spacer.anion})`,
          layerCount: 1,
          dimensionality: "quasi-2D",
        });
      }
    }
  }

  if (elements.length >= 2) {
    for (let layers = 2; layers <= 4; layers++) {
      const slFormula = `${formula}_SL${layers}`;
      const baseCounts = { ...counts };
      for (const el of Object.keys(baseCounts)) {
        baseCounts[el] = baseCounts[el] * layers;
      }
      results.push({
        formula: normalizeFormula(buildFormula(baseCounts)),
        layerType: `superlattice ${layers}-layer`,
        layerCount: layers,
        dimensionality: layers <= 2 ? "quasi-2D" : "3D",
      });
    }
  }

  return results;
}

export interface VacancyStructure {
  formula: string;
  vacancyType: string;
  concentration: number;
  expectedEffect: string;
}

function estimateVacancyFormationEnergy(element: string, concentration: number, formula: string): number {
  const d = getElementData(element);
  if (!d) return 0.3;

  let baseEnergy = 0.15;

  if (d.bulkModulus && d.bulkModulus > 150) baseEnergy += 0.1;
  if (d.meltingPoint && d.meltingPoint > 2000) baseEnergy += 0.1;
  if (isTransitionMetal(element)) baseEnergy += 0.05;

  return baseEnergy * concentration * 4;
}

export function generateVacancyStructures(formula: string, prototype: PrototypeType): VacancyStructure[] {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const results: VacancyStructure[] = [];
  const MAX_ENERGY = 0.5;

  const vacancyConcentrations = [0.05, 0.10, 0.15, 0.20, 0.25];

  for (const el of elements) {
    const originalCount = counts[el];
    if (originalCount < 2) continue;

    for (const conc of vacancyConcentrations) {
      const removed = Math.max(1, Math.round(originalCount * conc));
      const newCount = originalCount - removed;
      if (newCount < 1) continue;

      const energy = estimateVacancyFormationEnergy(el, conc, formula);
      if (energy > MAX_ENERGY) continue;

      const newCounts = { ...counts, [el]: newCount };
      const vacFormula = normalizeFormula(buildFormula(newCounts));

      let effect = "unknown";
      if (el === "O" || el === "N") effect = "hole doping, increased carrier concentration";
      else if (isTransitionMetal(el)) effect = "disorder scattering, possible local moment formation";
      else if (isRareEarth(el)) effect = "charge reservoir modification";
      else effect = "lattice strain, phonon softening";

      results.push({
        formula: vacFormula,
        vacancyType: `${el}-vacancy ${(conc * 100).toFixed(0)}%`,
        concentration: conc,
        expectedEffect: effect,
      });
    }
  }

  if (elements.length >= 2) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const elA = elements[i];
        const elB = elements[j];
        if (counts[elA] < 2 || counts[elB] < 2) continue;

        const swapCount = 1;
        const newCounts = { ...counts };
        newCounts[elA] = counts[elA] - swapCount;
        newCounts[elB] = counts[elB] - swapCount;
        const mixKey = `${elA}${elB}mix`;
        newCounts[elA] += swapCount;
        newCounts[elB] += swapCount;

        results.push({
          formula: normalizeFormula(buildFormula(newCounts)),
          vacancyType: `anti-site ${elA}/${elB} swap`,
          concentration: swapCount / getTotalAtoms(counts),
          expectedEffect: `site disorder between ${elA} and ${elB}, modified local electronic structure`,
        });
      }
    }
  }

  return results;
}

export interface StrainedVariant {
  formula: string;
  substrate: string;
  strainPercent: number;
  strainType: string;
  expectedTcChange: number;
}

const SUBSTRATES: { name: string; latticeA: number }[] = [
  { name: "SrTiO3", latticeA: 3.905 },
  { name: "LaAlO3", latticeA: 3.789 },
  { name: "MgO", latticeA: 4.212 },
  { name: "Si", latticeA: 5.431 },
  { name: "Al2O3", latticeA: 4.758 },
];

function estimateTcChangeFromStrain(strainPercent: number, formula: string): number {
  const absStrain = Math.abs(strainPercent);
  const elements = parseFormulaElements(formula);

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se"));

  if (isCuprate) {
    if (strainPercent < 0) return Math.round(absStrain * 3);
    return Math.round(-absStrain * 2);
  }

  if (isPnictide) {
    if (absStrain < 2) return Math.round(absStrain * 2);
    return Math.round(-absStrain * 1.5);
  }

  if (absStrain < 1) return Math.round(absStrain * 1.5);
  if (absStrain < 3) return Math.round(absStrain * 0.5);
  return Math.round(-absStrain * 0.8);
}

export function generateStrainedVariants(formula: string, prototype: PrototypeType): StrainedVariant[] {
  const base = PROTOTYPE_LATTICE[prototype];
  const filmA = base.a;
  const results: StrainedVariant[] = [];

  for (const sub of SUBSTRATES) {
    const mismatch = ((sub.latticeA - filmA) / filmA) * 100;
    const absMismatch = Math.abs(mismatch);

    if (absMismatch > 10) continue;

    const strainType = mismatch > 0 ? "tensile" : "compressive";
    const tcChange = estimateTcChangeFromStrain(mismatch, formula);

    results.push({
      formula,
      substrate: sub.name,
      strainPercent: Math.round(mismatch * 100) / 100,
      strainType,
      expectedTcChange: tcChange,
    });
  }

  return results;
}

export interface StructuralMutationResult {
  distorted: DistortedLattice[];
  layered: LayeredStructure[];
  vacancy: VacancyStructure[];
  strained: StrainedVariant[];
  filteredByEnergy: number;
  totalGenerated: number;
}

export function runStructuralMutations(
  topCandidates: { formula: string; predictedTc?: number | null }[],
  emit?: EventEmitter
): StructuralMutationResult {
  const allDistorted: DistortedLattice[] = [];
  const allLayered: LayeredStructure[] = [];
  const allVacancy: VacancyStructure[] = [];
  const allStrained: StrainedVariant[] = [];
  let filteredByEnergy = 0;
  const seenFormulas = new Set<string>();

  for (const candidate of topCandidates) {
    const prototype = assignPrototype(candidate.formula);

    const distortedRaw = generateDistortedLattices(candidate.formula, prototype);
    const distortedBefore = distortedRaw.length;
    const distorted = distortedRaw.filter(d => d.energyPenalty <= 0.5);
    filteredByEnergy += distortedBefore - distorted.length;

    for (const d of distorted) {
      const key = `distort-${d.formula}-${d.distortionType}`;
      if (!seenFormulas.has(key)) {
        seenFormulas.add(key);
        allDistorted.push(d);
      }
    }

    const layered = generateLayeredStructures(candidate.formula, prototype);
    for (const l of layered) {
      if (!seenFormulas.has(l.formula)) {
        seenFormulas.add(l.formula);
        allLayered.push(l);
      }
    }

    const vacancy = generateVacancyStructures(candidate.formula, prototype);
    for (const v of vacancy) {
      if (!seenFormulas.has(v.formula)) {
        seenFormulas.add(v.formula);
        allVacancy.push(v);
      }
    }

    const strained = generateStrainedVariants(candidate.formula, prototype);
    for (const s of strained) {
      const key = `strain-${s.formula}-${s.substrate}`;
      if (!seenFormulas.has(key)) {
        seenFormulas.add(key);
        allStrained.push(s);
      }
    }
  }

  const totalGenerated = allDistorted.length + allLayered.length + allVacancy.length + allStrained.length;

  if (emit) {
    emit("log", {
      phase: "engine",
      event: "Structural mutations complete",
      detail: `Structural mutations: ${allDistorted.length} distorted, ${allLayered.length} layered, ${allVacancy.length} vacancy, ${allStrained.length} strained variants generated (${filteredByEnergy} filtered by energy)`,
      dataSource: "Structural Mutator",
    });
  }

  return {
    distorted: allDistorted,
    layered: allLayered,
    vacancy: allVacancy,
    strained: allStrained,
    filteredByEnergy,
    totalGenerated,
  };
}
