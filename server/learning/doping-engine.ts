import { parseFormulaElements } from "./physics-engine";
import { normalizeFormula, isValidFormula } from "./utils";
import { getElementData } from "./elemental-data";
import { runXTBOptimization } from "../dft/qe-dft-engine";

export type DopingCharacter = "electron" | "hole" | "isovalent" | "vacancy-hole" | "interstitial-electron";

export interface DopingSpec {
  type: "substitutional" | "vacancy" | "interstitial";
  base: string;
  dopant?: string;
  site?: string;
  fraction: number;
  resultFormula: string;
  supercellSize: number;
  rationale: string;
  dopingCharacter: DopingCharacter;
  valenceChange: number;
  carrierDensity: number;
  relaxation?: RelaxationMetrics;
}

export interface RelaxationMetrics {
  converged: boolean;
  latticeStrain: number;
  bondVariance: number;
  meanDisplacement: number;
  maxDisplacement: number;
  volumeChange: number;
  energyPerAtom: number;
  wallTimeMs: number;
}

export interface DopingResult {
  baseFormula: string;
  variants: DopingSpec[];
  totalGenerated: number;
  validGenerated: number;
  wallTimeMs: number;
}

export interface DopingEngineStats {
  totalBaseMaterials: number;
  totalVariantsGenerated: number;
  substitutionalCount: number;
  vacancyCount: number;
  interstitialCount: number;
  validVariants: number;
  electronDopedCount: number;
  holeDopedCount: number;
  relaxationsCompleted: number;
  avgLatticeStrain: number;
  recentResults: Array<{ base: string; variants: number; timestamp: number }>;
}

const stats: DopingEngineStats = {
  totalBaseMaterials: 0,
  totalVariantsGenerated: 0,
  substitutionalCount: 0,
  vacancyCount: 0,
  interstitialCount: 0,
  validVariants: 0,
  electronDopedCount: 0,
  holeDopedCount: 0,
  relaxationsCompleted: 0,
  avgLatticeStrain: 0,
  recentResults: [],
};

let totalStrainSum = 0;

const MAX_RECENT = 100;

const COMMON_OXIDATION_STATES: Record<string, number> = {
  H: 1, Li: 1, Na: 1, K: 1, Rb: 1, Cs: 1,
  Be: 2, Mg: 2, Ca: 2, Sr: 2, Ba: 2,
  Sc: 3, Y: 3, La: 3, Ce: 3, Pr: 3, Nd: 3, Gd: 3,
  Ti: 4, Zr: 4, Hf: 4,
  V: 5, Nb: 5, Ta: 5,
  Cr: 3, Mo: 6, W: 6,
  Mn: 2, Fe: 3, Co: 3, Ni: 2, Cu: 2, Zn: 2,
  Ru: 4, Rh: 3, Pd: 2, Ir: 4, Pt: 4,
  Al: 3, Ga: 3, In: 3, Tl: 1,
  B: 3, C: 4, Si: 4, Ge: 4, Sn: 4, Pb: 2,
  N: -3, P: -3, As: -3, Sb: -3, Bi: 3,
  O: -2, S: -2, Se: -2, Te: -2,
  F: -1, Cl: -1, Br: -1, I: -1,
  Re: 7, Os: 4,
};

const ELECTRON_DOPING_PAIRS: Array<{ from: string; to: string }> = [
  { from: "O", to: "F" },
  { from: "Fe", to: "Co" },
  { from: "Ti", to: "Nb" },
  { from: "Ti", to: "V" },
  { from: "Cu", to: "Zn" },
  { from: "Ni", to: "Cu" },
  { from: "Mn", to: "Fe" },
  { from: "Cr", to: "Mn" },
  { from: "N", to: "O" },
  { from: "S", to: "Cl" },
  { from: "Se", to: "Br" },
  { from: "Zr", to: "Nb" },
  { from: "Hf", to: "Ta" },
  { from: "Mo", to: "W" },
  { from: "Al", to: "Si" },
  { from: "Ga", to: "Ge" },
  { from: "Sn", to: "Sb" },
  { from: "In", to: "Sn" },
  { from: "Ca", to: "Sc" },
  { from: "Sr", to: "Y" },
];

const HOLE_DOPING_PAIRS: Array<{ from: string; to: string }> = [
  { from: "La", to: "Sr" },
  { from: "Ba", to: "K" },
  { from: "Y", to: "Ca" },
  { from: "La", to: "Ba" },
  { from: "Ce", to: "La" },
  { from: "Sr", to: "K" },
  { from: "Ca", to: "Na" },
  { from: "Fe", to: "Mn" },
  { from: "Nb", to: "Ti" },
  { from: "Co", to: "Fe" },
  { from: "Cu", to: "Ni" },
  { from: "Bi", to: "Pb" },
  { from: "Pb", to: "Tl" },
  { from: "Sn", to: "In" },
  { from: "Ga", to: "Zn" },
  { from: "Al", to: "Mg" },
  { from: "Nd", to: "Sr" },
  { from: "Gd", to: "Ca" },
  { from: "Ta", to: "Zr" },
  { from: "V", to: "Ti" },
];

function getOxidationState(el: string): number {
  return COMMON_OXIDATION_STATES[el] ?? 0;
}

function classifyDopingCharacter(site: string, dopant: string, type: "substitutional" | "vacancy" | "interstitial"): { character: DopingCharacter; valenceChange: number } {
  if (type === "vacancy") {
    const siteOx = getOxidationState(site);
    return { character: "vacancy-hole", valenceChange: -siteOx };
  }
  if (type === "interstitial") {
    const dopantOx = getOxidationState(dopant);
    return { character: "interstitial-electron", valenceChange: dopantOx };
  }

  const siteOx = getOxidationState(site);
  const dopantOx = getOxidationState(dopant);
  const delta = dopantOx - siteOx;

  if (delta > 0) return { character: "electron", valenceChange: delta };
  if (delta < 0) return { character: "hole", valenceChange: delta };
  return { character: "isovalent", valenceChange: 0 };
}

function estimateUnitCellVolume(counts: Record<string, number>): number {
  const totalAtoms = getTotalAtoms(counts);
  let avgRadius = 0;
  let totalWeight = 0;
  for (const [el, n] of Object.entries(counts)) {
    const data = getElementData(el);
    const r = data?.atomicRadius ?? 130;
    avgRadius += r * n;
    totalWeight += n;
  }
  avgRadius = totalWeight > 0 ? avgRadius / totalWeight : 130;
  const latticeParam = avgRadius * 2.8 / 100;
  const volumePerAtom = latticeParam ** 3;
  return volumePerAtom * totalAtoms;
}

function computeCarrierDensity(valenceChange: number, nDopedAtoms: number, cellVolumeNm3: number): number {
  if (cellVolumeNm3 <= 0) return 0;
  const totalChargeChange = Math.abs(valenceChange) * nDopedAtoms;
  const volumeCm3 = cellVolumeNm3 * 1e-21;
  return totalChargeChange / volumeCm3;
}

function parseFormulaCounts(formula: string): Record<string, number> {
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

function countsToFormula(counts: Record<string, number>): string {
  const sorted = Object.entries(counts)
    .filter(([, n]) => n > 0.001)
    .sort(([a], [b]) => a.localeCompare(b));
  return sorted.map(([el, n]) => {
    const rounded = Math.round(n * 100) / 100;
    if (Math.abs(rounded - 1) < 0.01) return el;
    if (Number.isInteger(rounded)) return `${el}${rounded}`;
    return `${el}${rounded}`;
  }).join("");
}

function getTotalAtoms(counts: Record<string, number>): number {
  return Object.values(counts).reduce((s, n) => s + n, 0);
}

const SC_DOPANT_MAP: Record<string, string[]> = {
  La: ["Sr", "Ba", "Ca", "Ce", "Y"],
  Sr: ["La", "Ba", "Ca", "K"],
  Ba: ["La", "Sr", "K", "Ca"],
  Y: ["La", "Ce", "Ca", "Ba"],
  Ca: ["Sr", "La", "Ba", "Na"],
  Ti: ["Nb", "V", "Zr", "Hf"],
  Fe: ["Co", "Ni", "Mn", "Cu"],
  Co: ["Fe", "Ni", "Mn"],
  Ni: ["Cu", "Co", "Fe", "Pd"],
  Cu: ["Ni", "Zn", "Co"],
  Nb: ["Ti", "Ta", "V", "Mo"],
  Zr: ["Ti", "Hf", "Nb"],
  Mn: ["Fe", "Co", "Cr"],
  Bi: ["Sb", "Pb", "Tl"],
  Pb: ["Bi", "Sn", "Tl"],
  Sn: ["In", "Pb", "Ge"],
  In: ["Sn", "Ga", "Tl"],
  Ga: ["In", "Al"],
  Al: ["Ga", "In", "B"],
  B: ["C", "N", "Al"],
  Se: ["Te", "S"],
  Te: ["Se", "S"],
  As: ["P", "Sb"],
  P: ["As", "N"],
  Hf: ["Zr", "Ti"],
  Ta: ["Nb", "V"],
  Mo: ["W", "Nb"],
  W: ["Mo", "Ta"],
  Cr: ["V", "Mn"],
  V: ["Nb", "Ti", "Cr"],
  Ru: ["Os", "Ir"],
  Pd: ["Pt", "Ni"],
  Pt: ["Pd", "Ir"],
  Re: ["Tc", "Mo"],
  Ir: ["Rh", "Pt"],
  Rh: ["Ir", "Co"],
  Ge: ["Si", "Sn"],
  Si: ["Ge", "C"],
  N: ["C", "B"],
  C: ["N", "B"],
  O: ["F", "N"],
  F: ["O", "Cl"],
};

const INTERSTITIAL_DOPANTS: Record<string, string[]> = {
  layered: ["Li", "Na", "K", "Ca"],
  cage: ["H", "Li", "Na"],
  chalcogenide: ["Li", "Na", "K", "Cu"],
  pnictide: ["Li", "Na", "H"],
  oxide: ["H", "Li", "F"],
  general: ["Li", "H", "Na", "F"],
};

const VACANCY_TARGETS = ["O", "F", "S", "Se", "Te", "N", "Cl"];

const DOPING_FRACTIONS = [0.05, 0.10, 0.15, 0.20];

function classifyLayeredOrCage(formula: string): string {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
  const hasPnictogen = elements.some(e => ["As", "P", "Sb", "Bi"].includes(e));
  const hasOxygen = elements.includes("O");
  const hFrac = (counts["H"] || 0) / totalAtoms;

  if (hFrac > 0.3) return "cage";
  if (hasChalcogen) return "chalcogenide";
  if (hasPnictogen) return "pnictide";
  if (hasOxygen) return "oxide";

  const layeredElements = ["Bi", "Sb", "Se", "Te", "S", "As", "P"];
  if (elements.some(e => layeredElements.includes(e))) return "layered";

  return "general";
}

function getSupercellMultiplier(totalAtoms: number): number {
  if (totalAtoms <= 4) return 8;
  if (totalAtoms <= 8) return 4;
  if (totalAtoms <= 12) return 2;
  return 1;
}

function getDopantPriority(site: string, dopant: string, elements: string[]): number {
  if (elements.includes(dopant)) return -1;

  for (const pair of ELECTRON_DOPING_PAIRS) {
    if (pair.from === site && pair.to === dopant) return 10;
  }
  for (const pair of HOLE_DOPING_PAIRS) {
    if (pair.from === site && pair.to === dopant) return 10;
  }

  const siteOx = getOxidationState(site);
  const dopantOx = getOxidationState(dopant);
  const delta = Math.abs(dopantOx - siteOx);
  if (delta === 1) return 8;
  if (delta === 0) return 5;
  if (delta === 2) return 6;
  return 3;
}

function generateSubstitutionalVariants(
  formula: string,
  counts: Record<string, number>,
  maxVariants: number = 8
): DopingSpec[] {
  const variants: DopingSpec[] = [];
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const supercellMult = getSupercellMultiplier(totalAtoms);
  const cellVolume = estimateUnitCellVolume(counts);

  for (const site of elements) {
    const siteCount = counts[site];
    if (siteCount < 0.5) continue;

    const dopants = SC_DOPANT_MAP[site];
    if (!dopants) continue;

    const siteData = getElementData(site);
    if (!siteData) continue;

    const sortedDopants = [...dopants]
      .map(d => ({ dopant: d, priority: getDopantPriority(site, d, elements) }))
      .filter(d => d.priority >= 0)
      .sort((a, b) => b.priority - a.priority)
      .map(d => d.dopant);

    for (const dopant of sortedDopants) {
      if (variants.length >= maxVariants) break;

      const dopantData = getElementData(dopant);
      if (!dopantData) continue;

      const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
        ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
        : 0.5;
      if (radiusDiff > 0.3) continue;

      const { character, valenceChange } = classifyDopingCharacter(site, dopant, "substitutional");

      const fractions = radiusDiff < 0.15
        ? DOPING_FRACTIONS
        : DOPING_FRACTIONS.filter(f => f <= 0.10);

      for (const fraction of fractions) {
        if (variants.length >= maxVariants) break;

        const supercellCounts: Record<string, number> = {};
        for (const [el, n] of Object.entries(counts)) {
          supercellCounts[el] = n * supercellMult;
        }

        const sitesInSupercell = supercellCounts[site];
        const nReplace = Math.max(1, Math.round(sitesInSupercell * fraction));
        if (nReplace >= sitesInSupercell) continue;

        supercellCounts[site] = sitesInSupercell - nReplace;
        supercellCounts[dopant] = (supercellCounts[dopant] || 0) + nReplace;

        const gcd = findGCD(Object.values(supercellCounts).filter(v => v > 0).map(v => Math.round(v)));
        const reduced: Record<string, number> = {};
        for (const [el, n] of Object.entries(supercellCounts)) {
          if (n > 0) reduced[el] = n / gcd;
        }

        const resultFormula = countsToFormula(reduced);
        if (!isValidFormula(resultFormula)) continue;

        const supercellVolume = cellVolume * supercellMult;
        const carrierDensity = computeCarrierDensity(valenceChange, nReplace, supercellVolume);

        const dopingLabel = character === "electron" ? "electron-doping" :
          character === "hole" ? "hole-doping" : "isovalent";
        const chargeInfo = valenceChange !== 0
          ? ` [${dopingLabel}: delta_q=${valenceChange > 0 ? "+" : ""}${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3]`
          : " [isovalent substitution]";

        const rationale = `${dopant} substitution at ${site} site (${(fraction * 100).toFixed(0)}%): `
          + `radius match ${(1 - radiusDiff).toFixed(2)}, `
          + `replaces ${nReplace}/${sitesInSupercell} ${site} atoms in ${supercellMult > 1 ? supercellMult + "x supercell" : "unit cell"}`
          + chargeInfo;

        variants.push({
          type: "substitutional",
          base: formula,
          dopant,
          site,
          fraction,
          resultFormula: normalizeFormula(resultFormula),
          supercellSize: supercellMult,
          rationale,
          dopingCharacter: character,
          valenceChange,
          carrierDensity,
        });
      }
    }
    if (variants.length >= maxVariants) break;
  }

  return variants;
}

function generateVacancyVariants(
  formula: string,
  counts: Record<string, number>,
  maxVariants: number = 4
): DopingSpec[] {
  const variants: DopingSpec[] = [];
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const supercellMult = getSupercellMultiplier(totalAtoms);
  const cellVolume = estimateUnitCellVolume(counts);

  const vacancySites = elements.filter(e => VACANCY_TARGETS.includes(e));
  if (vacancySites.length === 0) return [];

  for (const site of vacancySites) {
    const siteCount = counts[site];
    if (siteCount < 1) continue;

    const { character, valenceChange } = classifyDopingCharacter(site, "", "vacancy");

    const fracs = [0.05, 0.10, 0.15];
    for (const fraction of fracs) {
      if (variants.length >= maxVariants) break;

      const supercellCounts: Record<string, number> = {};
      for (const [el, n] of Object.entries(counts)) {
        supercellCounts[el] = n * supercellMult;
      }

      const sitesInSupercell = supercellCounts[site];
      const nRemove = Math.max(1, Math.round(sitesInSupercell * fraction));
      if (nRemove >= sitesInSupercell) continue;

      supercellCounts[site] = sitesInSupercell - nRemove;

      const gcd = findGCD(Object.values(supercellCounts).filter(v => v > 0).map(v => Math.round(v)));
      const reduced: Record<string, number> = {};
      for (const [el, n] of Object.entries(supercellCounts)) {
        if (n > 0) reduced[el] = n / gcd;
      }

      const resultFormula = countsToFormula(reduced);
      if (!isValidFormula(resultFormula)) continue;

      const supercellVolume = cellVolume * supercellMult;
      const carrierDensity = computeCarrierDensity(valenceChange, nRemove, supercellVolume);

      const carrierType = valenceChange < 0 ? "hole" : "electron";

      variants.push({
        type: "vacancy",
        base: formula,
        site,
        fraction,
        resultFormula: normalizeFormula(resultFormula),
        supercellSize: supercellMult,
        rationale: `${site} vacancy doping (${(fraction * 100).toFixed(0)}%): removed ${nRemove}/${sitesInSupercell} ${site} atoms — creates ${carrierType} carriers (delta_q=${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3)`,
        dopingCharacter: character,
        valenceChange,
        carrierDensity,
      });
    }
  }

  return variants;
}

function generateInterstitialVariants(
  formula: string,
  counts: Record<string, number>,
  maxVariants: number = 4
): DopingSpec[] {
  const variants: DopingSpec[] = [];
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const supercellMult = getSupercellMultiplier(totalAtoms);
  const structureType = classifyLayeredOrCage(formula);
  const cellVolume = estimateUnitCellVolume(counts);

  const dopantPool = INTERSTITIAL_DOPANTS[structureType] || INTERSTITIAL_DOPANTS.general;
  const availableDopants = dopantPool.filter(d => !elements.includes(d));

  for (const dopant of availableDopants) {
    if (variants.length >= maxVariants) break;

    const { character, valenceChange } = classifyDopingCharacter("", dopant, "interstitial");

    const fracs = [0.05, 0.10];
    for (const fraction of fracs) {
      if (variants.length >= maxVariants) break;

      const supercellCounts: Record<string, number> = {};
      for (const [el, n] of Object.entries(counts)) {
        supercellCounts[el] = n * supercellMult;
      }

      const totalInSupercell = getTotalAtoms(supercellCounts);
      const nInsert = Math.max(1, Math.round(totalInSupercell * fraction));

      supercellCounts[dopant] = (supercellCounts[dopant] || 0) + nInsert;

      const gcd = findGCD(Object.values(supercellCounts).filter(v => v > 0).map(v => Math.round(v)));
      const reduced: Record<string, number> = {};
      for (const [el, n] of Object.entries(supercellCounts)) {
        if (n > 0) reduced[el] = n / gcd;
      }

      const resultFormula = countsToFormula(reduced);
      if (!isValidFormula(resultFormula)) continue;

      const totalNew = getTotalAtoms(reduced);
      if (totalNew > 20) continue;

      const supercellVolume = cellVolume * supercellMult;
      const carrierDensity = computeCarrierDensity(valenceChange, nInsert, supercellVolume);

      variants.push({
        type: "interstitial",
        base: formula,
        dopant,
        fraction,
        resultFormula: normalizeFormula(resultFormula),
        supercellSize: supercellMult,
        rationale: `${dopant} interstitial insertion (${(fraction * 100).toFixed(0)}%): ${nInsert} atoms into ${structureType} structure — electron-doping (delta_q=+${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3)`,
        dopingCharacter: character,
        valenceChange,
        carrierDensity,
      });
    }
  }

  return variants;
}

function findGCD(nums: number[]): number {
  if (nums.length === 0) return 1;
  const gcd2 = (a: number, b: number): number => {
    a = Math.abs(Math.round(a));
    b = Math.abs(Math.round(b));
    while (b > 0) {
      [a, b] = [b, a % b];
    }
    return a || 1;
  };
  return nums.reduce((acc, n) => gcd2(acc, n), nums[0]);
}

export async function relaxDopedStructure(formula: string): Promise<RelaxationMetrics | null> {
  try {
    const optResult = await runXTBOptimization(formula, 0);
    if (!optResult || !optResult.converged) return null;

    const atoms = optResult.optimizedAtoms;
    if (atoms.length < 2) return null;

    const dist = optResult.distortion;

    const latticeStrain = dist?.latticeDistortion?.strainMagnitude ?? 0;
    const volumeChange = dist?.latticeDistortion?.volumeChangePct ?? 0;
    const meanDisplacement = dist?.atomicDistortion?.meanDisplacement ?? 0;
    const maxDisplacement = dist?.atomicDistortion?.maxDisplacement ?? 0;

    let bondVariance = 0;
    if (atoms.length >= 2) {
      const distances: number[] = [];
      for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
          const dx = atoms[i].x - atoms[j].x;
          const dy = atoms[i].y - atoms[j].y;
          const dz = atoms[i].z - atoms[j].z;
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (d < 3.5) {
            distances.push(d);
          }
        }
      }
      if (distances.length > 1) {
        const mean = distances.reduce((s, d) => s + d, 0) / distances.length;
        bondVariance = Math.sqrt(
          distances.reduce((s, d) => s + (d - mean) ** 2, 0) / distances.length
        );
      }
    }

    const energyPerAtom = atoms.length > 0 ? (optResult.optimizedEnergy * 27.2114) / atoms.length : 0;

    return {
      converged: optResult.converged,
      latticeStrain,
      bondVariance,
      meanDisplacement,
      maxDisplacement,
      volumeChange,
      energyPerAtom,
      wallTimeMs: optResult.wallTimeSeconds * 1000,
    };
  } catch {
    return null;
  }
}

export function generateDopedVariants(
  formula: string,
  maxTotal: number = 12
): DopingResult {
  const start = Date.now();
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);

  if (elements.length === 0 || totalAtoms < 2 || elements.length > 5) {
    return { baseFormula: formula, variants: [], totalGenerated: 0, validGenerated: 0, wallTimeMs: Date.now() - start };
  }

  const subMax = Math.ceil(maxTotal * 0.5);
  const vacMax = Math.ceil(maxTotal * 0.25);
  const intMax = maxTotal - subMax - vacMax;

  const substitutional = generateSubstitutionalVariants(formula, counts, subMax);
  const vacancy = generateVacancyVariants(formula, counts, vacMax);
  const interstitial = generateInterstitialVariants(formula, counts, intMax);

  const allVariants = [...substitutional, ...vacancy, ...interstitial];

  const seen = new Set<string>();
  seen.add(normalizeFormula(formula));
  const unique = allVariants.filter(v => {
    if (seen.has(v.resultFormula)) return false;
    seen.add(v.resultFormula);
    return true;
  });

  stats.totalBaseMaterials++;
  stats.totalVariantsGenerated += unique.length;
  stats.substitutionalCount += substitutional.filter(v => unique.includes(v)).length;
  stats.vacancyCount += vacancy.filter(v => unique.includes(v)).length;
  stats.interstitialCount += interstitial.filter(v => unique.includes(v)).length;
  stats.validVariants += unique.length;
  stats.electronDopedCount += unique.filter(v =>
    v.dopingCharacter === "electron" || v.dopingCharacter === "interstitial-electron"
  ).length;
  stats.holeDopedCount += unique.filter(v =>
    v.dopingCharacter === "hole" || v.dopingCharacter === "vacancy-hole"
  ).length;

  stats.recentResults.push({ base: formula, variants: unique.length, timestamp: Date.now() });
  if (stats.recentResults.length > MAX_RECENT) {
    stats.recentResults = stats.recentResults.slice(-MAX_RECENT);
  }

  return {
    baseFormula: formula,
    variants: unique,
    totalGenerated: allVariants.length,
    validGenerated: unique.length,
    wallTimeMs: Date.now() - start,
  };
}

export async function generateDopedVariantsWithRelaxation(
  formula: string,
  maxTotal: number = 12,
  maxRelaxations: number = 4
): Promise<DopingResult> {
  const result = generateDopedVariants(formula, maxTotal);

  const toRelax = result.variants
    .filter(v => v.dopingCharacter !== "isovalent")
    .sort((a, b) => Math.abs(b.carrierDensity) - Math.abs(a.carrierDensity))
    .slice(0, maxRelaxations);

  for (const variant of toRelax) {
    const metrics = await relaxDopedStructure(variant.resultFormula);
    if (metrics) {
      variant.relaxation = metrics;
      stats.relaxationsCompleted++;
      totalStrainSum += metrics.latticeStrain;
      stats.avgLatticeStrain = stats.relaxationsCompleted > 0
        ? totalStrainSum / stats.relaxationsCompleted
        : 0;
    }
  }

  return result;
}

export function runDopingBatch(
  formulas: string[],
  maxVariantsPerBase: number = 8,
  maxTotalDoped: number = 50,
  excludeSet?: Set<string>
): { dopedFormulas: string[]; specs: DopingSpec[]; stats: { basesProcessed: number; totalVariants: number; substitutional: number; vacancy: number; interstitial: number; electronDoped: number; holeDoped: number } } {
  const dopedFormulas: string[] = [];
  const specs: DopingSpec[] = [];
  let subCount = 0, vacCount = 0, intCount = 0;
  let eDoped = 0, hDoped = 0;

  for (const base of formulas) {
    if (dopedFormulas.length >= maxTotalDoped) break;

    const result = generateDopedVariants(base, maxVariantsPerBase);
    for (const v of result.variants) {
      if (dopedFormulas.length >= maxTotalDoped) break;
      if (excludeSet && excludeSet.has(v.resultFormula)) continue;

      dopedFormulas.push(v.resultFormula);
      specs.push(v);
      if (v.type === "substitutional") subCount++;
      else if (v.type === "vacancy") vacCount++;
      else intCount++;
      if (v.dopingCharacter === "electron" || v.dopingCharacter === "interstitial-electron") eDoped++;
      if (v.dopingCharacter === "hole" || v.dopingCharacter === "vacancy-hole") hDoped++;
    }
  }

  return {
    dopedFormulas,
    specs,
    stats: {
      basesProcessed: formulas.length,
      totalVariants: dopedFormulas.length,
      substitutional: subCount,
      vacancy: vacCount,
      interstitial: intCount,
      electronDoped: eDoped,
      holeDoped: hDoped,
    },
  };
}

export function getDopingEngineStats(): DopingEngineStats {
  return { ...stats };
}

export function getDopingRecommendations(formula: string): {
  substitutional: Array<{ dopant: string; site: string; rationale: string; dopingType: string; valenceChange: number }>;
  vacancy: Array<{ site: string; rationale: string; dopingType: string; valenceChange: number }>;
  interstitial: Array<{ dopant: string; rationale: string; dopingType: string; valenceChange: number }>;
} {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const structureType = classifyLayeredOrCage(formula);

  const sub: Array<{ dopant: string; site: string; rationale: string; dopingType: string; valenceChange: number }> = [];

  for (const site of elements) {
    for (const pair of ELECTRON_DOPING_PAIRS) {
      if (pair.from === site && !elements.includes(pair.to)) {
        const { valenceChange } = classifyDopingCharacter(site, pair.to, "substitutional");
        const siteData = getElementData(site);
        const dopantData = getElementData(pair.to);
        if (siteData && dopantData) {
          const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
            ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
            : 0.5;
          sub.push({
            dopant: pair.to,
            site,
            rationale: `${pair.to} replaces ${site}: electron-doping (${site}${getOxidationState(site) > 0 ? "+" + getOxidationState(site) : getOxidationState(site)} -> ${pair.to}${getOxidationState(pair.to) > 0 ? "+" + getOxidationState(pair.to) : getOxidationState(pair.to)}), radius match ${((1 - radiusDiff) * 100).toFixed(0)}%`,
            dopingType: "electron",
            valenceChange,
          });
        }
      }
    }

    for (const pair of HOLE_DOPING_PAIRS) {
      if (pair.from === site && !elements.includes(pair.to)) {
        const { valenceChange } = classifyDopingCharacter(site, pair.to, "substitutional");
        const siteData = getElementData(site);
        const dopantData = getElementData(pair.to);
        if (siteData && dopantData) {
          const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
            ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
            : 0.5;
          sub.push({
            dopant: pair.to,
            site,
            rationale: `${pair.to} replaces ${site}: hole-doping (${site}${getOxidationState(site) > 0 ? "+" + getOxidationState(site) : getOxidationState(site)} -> ${pair.to}${getOxidationState(pair.to) > 0 ? "+" + getOxidationState(pair.to) : getOxidationState(pair.to)}), radius match ${((1 - radiusDiff) * 100).toFixed(0)}%`,
            dopingType: "hole",
            valenceChange,
          });
        }
      }
    }

    if (sub.filter(s => s.site === site).length === 0) {
      const dopants = SC_DOPANT_MAP[site];
      if (!dopants) continue;
      const siteData = getElementData(site);
      if (!siteData) continue;
      for (const dopant of dopants.slice(0, 2)) {
        if (elements.includes(dopant)) continue;
        const dopantData = getElementData(dopant);
        if (!dopantData) continue;
        const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
          ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
          : 0.5;
        if (radiusDiff <= 0.3) {
          const { character, valenceChange } = classifyDopingCharacter(site, dopant, "substitutional");
          sub.push({
            dopant,
            site,
            rationale: `${dopant} replaces ${site}: ${character}-doping (delta_q=${valenceChange}), radius match ${((1 - radiusDiff) * 100).toFixed(0)}%`,
            dopingType: character,
            valenceChange,
          });
        }
      }
    }
  }

  const vac: Array<{ site: string; rationale: string; dopingType: string; valenceChange: number }> = [];
  for (const site of elements) {
    if (VACANCY_TARGETS.includes(site) && counts[site] >= 1) {
      const { valenceChange } = classifyDopingCharacter(site, "", "vacancy");
      const carrierType = valenceChange < 0 ? "hole" : "electron";
      vac.push({
        site,
        rationale: `${site} vacancy: creates ${carrierType} carriers (removes ${site}${getOxidationState(site) > 0 ? "+" : ""}${getOxidationState(site)} charge)`,
        dopingType: "vacancy-" + carrierType,
        valenceChange,
      });
    }
  }

  const intPool = INTERSTITIAL_DOPANTS[structureType] || INTERSTITIAL_DOPANTS.general;
  const int: Array<{ dopant: string; rationale: string; dopingType: string; valenceChange: number }> = [];
  for (const dopant of intPool) {
    if (!elements.includes(dopant)) {
      const { valenceChange } = classifyDopingCharacter("", dopant, "interstitial");
      int.push({
        dopant,
        rationale: `${dopant} intercalation into ${structureType} lattice — donates ${valenceChange} electron(s), enhances electron-phonon coupling`,
        dopingType: "interstitial-electron",
        valenceChange,
      });
    }
  }

  return { substitutional: sub, vacancy: vac, interstitial: int.slice(0, 3) };
}
