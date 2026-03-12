import { ELEMENTAL_DATA, getElementData, isTransitionMetal, isRareEarth, isActinide } from "./elemental-data";
import { normalizeFormula, parseFormulaCounts } from "./utils";
import type { EventEmitter } from "./engine";

function parseFormulaElements(formula: string): string[] {
  return Object.keys(parseFormulaCounts(formula));
}

function getTotalAtoms(counts: Record<string, number>): number {
  return Object.values(counts).reduce((s, n) => s + n, 0);
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
  | "bcc" | "fcc" | "spinel" | "fluorite" | "rutile" | "pyrite" | "clathrate";

const PROTOTYPE_LATTICE: Record<PrototypeType, { a: number; c_over_a: number; crystalSystem: string; refRadius: number }> = {
  rocksalt:   { a: 4.2,  c_over_a: 1.0,  crystalSystem: "cubic",      refRadius: 140 },
  perovskite: { a: 3.9,  c_over_a: 1.0,  crystalSystem: "cubic",      refRadius: 135 },
  hexagonal:  { a: 3.2,  c_over_a: 1.63, crystalSystem: "hexagonal",   refRadius: 120 },
  layered:    { a: 3.8,  c_over_a: 3.2,  crystalSystem: "tetragonal",  refRadius: 130 },
  bcc:        { a: 3.3,  c_over_a: 1.0,  crystalSystem: "cubic",      refRadius: 140 },
  fcc:        { a: 3.6,  c_over_a: 1.0,  crystalSystem: "cubic",      refRadius: 140 },
  spinel:     { a: 8.1,  c_over_a: 1.0,  crystalSystem: "cubic",      refRadius: 130 },
  fluorite:   { a: 5.5,  c_over_a: 1.0,  crystalSystem: "cubic",      refRadius: 135 },
  rutile:     { a: 4.6,  c_over_a: 0.64, crystalSystem: "tetragonal",  refRadius: 130 },
  pyrite:     { a: 5.4,  c_over_a: 1.0,  crystalSystem: "cubic",      refRadius: 130 },
  clathrate:  { a: 5.0,  c_over_a: 1.0,  crystalSystem: "cubic",      refRadius: 125 },
};

function computeScaledLatticeA(formula: string, prototype: PrototypeType): number {
  const base = PROTOTYPE_LATTICE[prototype];
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  let weightedRadius = 0;
  let totalWeight = 0;
  for (const el of elements) {
    const d = getElementData(el);
    const radius = d?.atomicRadius ?? base.refRadius;
    const weight = counts[el] || 1;
    weightedRadius += radius * weight;
    totalWeight += weight;
  }
  if (totalWeight === 0) return base.a;
  const avgRadius = weightedRadius / totalWeight;
  return base.a * (avgRadius / base.refRadius);
}

export function assignPrototype(formula: string): PrototypeType {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  if (totalAtoms === 0 || elements.length === 0) return "bcc";

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

    const RESERVOIR_ELS = ["Ba", "Sr", "La", "Y", "Ca", "Tl", "Bi", "Hg"];
    const hasReservoir = RESERVOIR_ELS.some(r => elements.includes(r));
    if (elements.includes("Cu") && hasReservoir && elements.length >= 3) return "layered";

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
    const hRatio = hCount / Math.max(1, metalCount);
    if (hRatio >= 8) return "clathrate";
    if (hRatio >= 3) return "fcc";
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

export type StabilityTier = "stable" | "metastable" | "highly-unstable";

export interface MutationSource {
  parentFormula: string;
  parentTc: number;
}

export interface DistortedLattice extends MutationSource {
  formula: string;
  distortionType: string;
  magnitude: number;
  latticeParams: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  spaceGroup: string;
  energyPenalty: number;
  stabilityTier: StabilityTier;
}

const PROTOTYPE_SPACE_GROUPS: Record<PrototypeType, string> = {
  rocksalt: "Fm-3m", perovskite: "Pm-3m", hexagonal: "P6_3/mmc",
  layered: "I4/mmm", bcc: "Im-3m", fcc: "Fm-3m", spinel: "Fd-3m",
  fluorite: "Fm-3m", rutile: "P4_2/mnm", pyrite: "Pa-3", clathrate: "Pm-3n",
};

function distortedSpaceGroup(prototype: PrototypeType, distortionType: string): string {
  const base = PROTOTYPE_SPACE_GROUPS[prototype];
  if (distortionType === "tetragonal") {
    switch (prototype) {
      case "perovskite": return "P4/mmm";
      case "rocksalt": case "fcc": case "fluorite": return "I4/mmm";
      case "bcc": return "I4/mmm";
      case "spinel": return "I4_1/amd";
      default: return base;
    }
  }
  if (distortionType === "orthorhombic") {
    switch (prototype) {
      case "perovskite": return "Pnma";
      case "rocksalt": case "fcc": return "Fmmm";
      case "bcc": return "Immm";
      default: return "Pmmm";
    }
  }
  if (distortionType === "monoclinic") {
    switch (prototype) {
      case "perovskite": return "P2_1/m";
      case "layered": return "C2/m";
      default: return "P2/m";
    }
  }
  return base;
}

function classifyStability(energy: number): StabilityTier {
  if (energy <= 0.05) return "stable";
  if (energy <= 0.15) return "metastable";
  return "highly-unstable";
}

interface PrecomputedElastic {
  stiffnessFactor: number;
}

function precomputeElasticData(formula: string): PrecomputedElastic {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  let avgBulkModulus = 0;
  let count = 0;
  for (const el of elements) {
    const d = getElementData(el);
    if (d?.bulkModulus) {
      const w = counts[el] || 1;
      avgBulkModulus += d.bulkModulus * w;
      count += w;
    }
  }
  avgBulkModulus = count > 0 ? avgBulkModulus / count : 100;
  return { stiffnessFactor: avgBulkModulus / 100 };
}

function estimateDistortionEnergyFast(distortionType: string, magnitude: number, elastic: PrecomputedElastic): number {
  const { stiffnessFactor } = elastic;
  const strainSquared = magnitude * magnitude;

  switch (distortionType) {
    case "tetragonal":
      return 0.05 * stiffnessFactor * strainSquared * 100;
    case "orthorhombic":
      return 0.08 * stiffnessFactor * strainSquared * 100;
    case "monoclinic": {
      const normalizedTilt = magnitude / 5;
      const harmonic = normalizedTilt * normalizedTilt;
      const anharmonic = magnitude > 5 ? 0.02 * Math.pow(normalizedTilt, 4) : 0;
      return 0.12 * stiffnessFactor * (harmonic + anharmonic);
    }
    default:
      return 0.1 * stiffnessFactor * strainSquared * 100;
  }
}

const MAX_ENERGY = 0.2;

export function generateDistortedLattices(
  formula: string,
  prototype: PrototypeType,
  source: MutationSource,
  seen?: Set<string>,
): DistortedLattice[] {
  const base = PROTOTYPE_LATTICE[prototype];
  const scaledA = computeScaledLatticeA(formula, prototype);
  const elastic = precomputeElasticData(formula);
  const results: DistortedLattice[] = [];

  const tetragonalDistortions = [0.10, 0.15, 0.20, 0.25, 0.30];
  for (const mag of tetragonalDistortions) {
    for (const sign of [1, -1]) {
      const key = `distort-${formula}-tetragonal-${sign > 0 ? "+" : "-"}${mag}`;
      if (seen?.has(key)) continue;
      const energy = estimateDistortionEnergyFast("tetragonal", mag, elastic);
      if (energy > MAX_ENERGY) continue;
      seen?.add(key);
      const a = scaledA;
      const c = a * base.c_over_a * (1 + sign * mag);
      results.push({
        ...source,
        formula,
        distortionType: `tetragonal c/a ${sign > 0 ? "+" : "-"}${(mag * 100).toFixed(0)}%`,
        magnitude: mag,
        latticeParams: { a, b: a, c, alpha: 90, beta: 90, gamma: 90 },
        spaceGroup: distortedSpaceGroup(prototype, "tetragonal"),
        energyPenalty: Math.round(energy * 1000) / 1000,
        stabilityTier: classifyStability(energy),
      });
    }
  }

  const orthoDistortions = [0.02, 0.05, 0.08, 0.10];
  for (const mag of orthoDistortions) {
    const key = `distort-${formula}-ortho-${mag}`;
    if (seen?.has(key)) continue;
    const energy = estimateDistortionEnergyFast("orthorhombic", mag, elastic);
    if (energy > MAX_ENERGY) continue;
    seen?.add(key);
    const a = scaledA;
    const b = a * (1 + mag);
    const c = a * base.c_over_a;
    results.push({
      ...source,
      formula,
      distortionType: `orthorhombic b/a +${(mag * 100).toFixed(0)}%`,
      magnitude: mag,
      latticeParams: { a, b, c, alpha: 90, beta: 90, gamma: 90 },
      spaceGroup: distortedSpaceGroup(prototype, "orthorhombic"),
      energyPenalty: Math.round(energy * 1000) / 1000,
      stabilityTier: classifyStability(energy),
    });
  }

  const monoclinicTilts = [85, 87, 89, 91, 93, 95];
  for (const beta of monoclinicTilts) {
    if (beta === 90) continue;
    const tiltMag = Math.abs(90 - beta);
    const key = `distort-${formula}-mono-${beta}`;
    if (seen?.has(key)) continue;
    const energy = estimateDistortionEnergyFast("monoclinic", tiltMag, elastic);
    if (energy > MAX_ENERGY) continue;
    seen?.add(key);
    results.push({
      ...source,
      formula,
      distortionType: `monoclinic beta=${beta}deg`,
      magnitude: tiltMag,
      latticeParams: { a: scaledA, b: scaledA, c: scaledA * base.c_over_a, alpha: 90, beta, gamma: 90 },
      spaceGroup: distortedSpaceGroup(prototype, "monoclinic"),
      energyPenalty: Math.round(energy * 1000) / 1000,
      stabilityTier: classifyStability(energy),
    });
  }

  return results;
}

export interface LayeredStructure extends MutationSource {
  formula: string;
  layerType: string;
  layerCount: number;
  dimensionality: string;
}

export function generateLayeredStructures(
  formula: string,
  prototype: PrototypeType,
  source: MutationSource,
  seen?: Set<string>,
): LayeredStructure[] {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const results: LayeredStructure[] = [];

  const hasO = elements.includes("O");
  const RP_A_SITE_ELS = ["Ca", "Sr", "Ba", "La", "Y", "K", "Na", "Li", "Rb", "Cs"];
  const metalEls = elements.filter(e =>
    isTransitionMetal(e) || isRareEarth(e) || isActinide(e) ||
    RP_A_SITE_ELS.includes(e)
  );
  const aSiteCandidates = elements.filter(e => RP_A_SITE_ELS.includes(e) || isRareEarth(e));
  const bSiteEls = elements.filter(e => isTransitionMetal(e));

  if (hasO && aSiteCandidates.length >= 1 && bSiteEls.length >= 1) {
    const aSite = aSiteCandidates.sort((a, b) => {
      const rA = getElementData(a)?.atomicRadius ?? 0;
      const rB = getElementData(b)?.atomicRadius ?? 0;
      return rB - rA;
    })[0];

    const bSite = bSiteEls.find(e => e !== aSite) || bSiteEls[0];
    if (bSite === aSite) { /* skip RP if no distinct B-site */ }
    else {
      const aSiteRadius = getElementData(aSite)?.atomicRadius ?? 100;
      const bSiteRadius = getElementData(bSite)?.atomicRadius ?? 100;
      if (aSiteRadius > bSiteRadius) {
        for (const n of [1, 2, 3]) {
          const rpF = normalizeFormula(`${aSite}${n + 1}${bSite}${n}O${3 * n + 1}`);
          if (seen?.has(rpF)) continue;
          seen?.add(rpF);
          results.push({
            ...source,
            formula: rpF,
            layerType: `Ruddlesden-Popper n=${n}`,
            layerCount: n,
            dimensionality: n === 1 ? "quasi-2D" : n === 2 ? "quasi-2D" : "3D",
          });
        }
      }
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
        const spacerF = normalizeFormula(buildFormula(newCounts));
        if (seen?.has(spacerF)) continue;
        seen?.add(spacerF);
        results.push({
          ...source,
          formula: spacerF,
          layerType: `${spacer.type} (${spacer.el}${spacer.anion})`,
          layerCount: 1,
          dimensionality: "quasi-2D",
        });
      }
    }
  }

  if (elements.length >= 2) {
    for (let layers = 2; layers <= 4; layers++) {
      const baseCounts = { ...counts };
      for (const el of Object.keys(baseCounts)) {
        baseCounts[el] = baseCounts[el] * layers;
      }
      const slF = `${normalizeFormula(buildFormula(baseCounts))}_SL${layers}`;
      if (seen?.has(slF)) continue;
      seen?.add(slF);
      results.push({
        ...source,
        formula: slF,
        layerType: `superlattice ${layers}-layer`,
        layerCount: layers,
        dimensionality: layers <= 2 ? "quasi-2D" : "3D",
      });
    }
  }

  return results;
}

export interface VacancyStructure extends MutationSource {
  formula: string;
  vacancyType: string;
  concentration: number;
  expectedEffect: string;
}

function estimateVacancyFormationEnergy(element: string, concentration: number, formula: string): number {
  const d = getElementData(element);
  if (!d) return 0.3;

  const allElements = parseFormulaElements(formula);
  const allCounts = parseFormulaCounts(formula);
  let avgMeltingPt = 0;
  let totalW = 0;
  for (const el of allElements) {
    const ed = getElementData(el);
    if (ed?.meltingPoint) {
      const w = allCounts[el] || 1;
      avgMeltingPt += ed.meltingPoint * w;
      totalW += w;
    }
  }
  avgMeltingPt = totalW > 0 ? avgMeltingPt / totalW : 1500;
  const cohesiveScale = avgMeltingPt / 1500;

  let baseEnergy = 0.10 * cohesiveScale;

  if (d.bulkModulus && d.bulkModulus > 150) baseEnergy += 0.05;
  if (isTransitionMetal(element)) baseEnergy += 0.03;
  if (isRareEarth(element)) baseEnergy += 0.02;

  return baseEnergy * concentration * 20;
}

export function generateVacancyStructures(
  formula: string,
  prototype: PrototypeType,
  source: MutationSource,
  seen?: Set<string>,
): VacancyStructure[] {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const results: VacancyStructure[] = [];
  const VAC_MAX_ENERGY = 0.5;

  const vacancyConcentrations = [0.05, 0.10, 0.15, 0.20, 0.25];

  for (const el of elements) {
    const originalCount = counts[el];
    if (originalCount < 2) continue;

    for (const conc of vacancyConcentrations) {
      const removed = Math.max(1, Math.round(originalCount * conc));
      const newCount = originalCount - removed;
      if (newCount < 1) continue;

      const energy = estimateVacancyFormationEnergy(el, conc, formula);
      if (energy > VAC_MAX_ENERGY) continue;

      const newCounts = { ...counts, [el]: newCount };
      const vacFormula = normalizeFormula(buildFormula(newCounts));
      if (seen?.has(vacFormula)) continue;
      seen?.add(vacFormula);

      let effect = "unknown";
      if (el === "O" || el === "N") effect = "hole doping, increased carrier concentration";
      else if (isTransitionMetal(el)) effect = "disorder scattering, possible local moment formation";
      else if (isRareEarth(el)) effect = "charge reservoir modification";
      else effect = "lattice strain, phonon softening";

      results.push({
        ...source,
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
        newCounts[elB] = counts[elB] + swapCount;
        const swapF = normalizeFormula(buildFormula(newCounts));
        newCounts[elA] = counts[elA] + swapCount;
        newCounts[elB] = counts[elB] - swapCount;
        const reverseF = normalizeFormula(buildFormula(newCounts));

        if (!seen?.has(swapF)) {
          seen?.add(swapF);
          results.push({
            ...source,
            formula: swapF,
            vacancyType: `anti-site ${elA}->${elB} swap`,
            concentration: swapCount / Math.max(1, getTotalAtoms(counts)),
            expectedEffect: `site disorder: ${elA} replaced by ${elB} at one site, modified local electronic structure`,
          });
        }
        if (!seen?.has(reverseF)) {
          seen?.add(reverseF);
          results.push({
            ...source,
            formula: reverseF,
            vacancyType: `anti-site ${elB}->${elA} swap`,
            concentration: swapCount / Math.max(1, getTotalAtoms(counts)),
            expectedEffect: `site disorder: ${elB} replaced by ${elA} at one site, modified local electronic structure`,
          });
        }
      }
    }
  }

  return results;
}

export interface StrainedVariant extends MutationSource {
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

function estimateTcChangeFromStrain(strainPercent: number, formula: string, parentTc: number): number {
  const absStrain = Math.abs(strainPercent);
  const elements = parseFormulaElements(formula);

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se"));

  const maxBoost = Math.max(1, parentTc * 0.15);

  let rawChange: number;
  if (isCuprate) {
    rawChange = strainPercent < 0
      ? absStrain * 2
      : -absStrain * 1.5;
  } else if (isPnictide) {
    rawChange = absStrain < 2
      ? absStrain * 1.5
      : -absStrain * 1.2;
  } else {
    if (absStrain < 1) rawChange = absStrain * 1.0;
    else if (absStrain < 3) rawChange = absStrain * 0.3;
    else rawChange = -absStrain * 0.5;
  }

  return Math.round(Math.max(-maxBoost, Math.min(maxBoost, rawChange)));
}

const MAX_STRAIN_PERCENT = 4;

export function generateStrainedVariants(
  formula: string,
  prototype: PrototypeType,
  parentTc: number = 30,
  seen?: Set<string>,
): StrainedVariant[] {
  const filmA = computeScaledLatticeA(formula, prototype);
  const results: StrainedVariant[] = [];
  const source: MutationSource = { parentFormula: formula, parentTc };

  for (const sub of SUBSTRATES) {
    const mismatch = ((sub.latticeA - filmA) / filmA) * 100;
    const absMismatch = Math.abs(mismatch);

    if (absMismatch > MAX_STRAIN_PERCENT) continue;

    const key = `strain-${formula}-${sub.name}`;
    if (seen?.has(key)) continue;
    seen?.add(key);

    const strainType = mismatch > 0 ? "tensile" : "compressive";
    const tcChange = estimateTcChangeFromStrain(mismatch, formula, parentTc);

    results.push({
      ...source,
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
  const seen = new Set<string>();
  const protoCache = new Map<string, PrototypeType>();

  for (const candidate of topCandidates) {
    const f = candidate.formula;
    let prototype = protoCache.get(f);
    if (!prototype) {
      prototype = assignPrototype(f);
      protoCache.set(f, prototype);
    }

    const parentTc = candidate.predictedTc ?? 30;
    const source: MutationSource = { parentFormula: f, parentTc };

    const distorted = generateDistortedLattices(f, prototype, source, seen);
    allDistorted.push(...distorted);

    const layered = generateLayeredStructures(f, prototype, source, seen);
    allLayered.push(...layered);

    const vacancy = generateVacancyStructures(f, prototype, source, seen);
    allVacancy.push(...vacancy);

    const strained = generateStrainedVariants(f, prototype, parentTc, seen);
    allStrained.push(...strained);
  }

  const totalGenerated = allDistorted.length + allLayered.length + allVacancy.length + allStrained.length;
  const highlyUnstableCount = allDistorted.filter(d => d.stabilityTier === "highly-unstable").length;

  if (emit) {
    emit("log", {
      phase: "engine",
      event: "Structural mutations complete",
      detail: `Structural mutations: ${allDistorted.length} distorted (${highlyUnstableCount} highly-unstable), ${allLayered.length} layered, ${allVacancy.length} vacancy, ${allStrained.length} strained variants generated`,
      dataSource: "Structural Mutator",
    });
  }

  return {
    distorted: allDistorted,
    layered: allLayered,
    vacancy: allVacancy,
    strained: allStrained,
    filteredByEnergy: 0,
    totalGenerated,
  };
}
