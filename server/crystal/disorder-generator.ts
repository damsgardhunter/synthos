import { getElementData } from "../learning/elemental-data";

export type DisorderType = "vacancy" | "substitution" | "interstitial" | "site-mixing";

export interface DisorderSpec {
  type: DisorderType;
  element: string;
  fraction: number;
  substituent?: string;
}

export interface DisorderedAtom {
  element: string;
  x: number;
  y: number;
  z: number;
  originalElement?: string;
  isDefect: boolean;
  defectType?: DisorderType;
}

export interface DisorderedStructure {
  base: string;
  disorder: DisorderSpec;
  atoms: DisorderedAtom[];
  supercellSize: [number, number, number];
  latticeA: number;
  latticeB: number;
  latticeC: number;
  totalAtoms: number;
  defectCount: number;
  defectFraction: number;
  formationEnergyEstimate: number;
  tcModifierEstimate: number;
  notes: string;
  generatedAt: number;
}

export interface DisorderGeneratorStats {
  totalGenerated: number;
  byType: Record<DisorderType, number>;
  avgDefectFraction: number;
  avgTcModifier: number;
  bestTcModifier: number;
  bestFormula: string;
  bestDisorderType: DisorderType | "";
  recentGenerations: Array<{
    base: string;
    type: DisorderType;
    element: string;
    fraction: number;
    defectCount: number;
    tcModifier: number;
  }>;
  topCandidates: Array<{
    base: string;
    type: DisorderType;
    element: string;
    fraction: number;
    tcModifier: number;
    formationEnergy: number;
  }>;
}

const SUBSTITUTION_MAP: Record<string, string[]> = {
  Sr: ["La", "Ca", "Ba", "K", "Na", "Eu"],
  Ba: ["Sr", "La", "Ca", "K", "Cs"],
  La: ["Sr", "Ce", "Nd", "Y", "Pr"],
  Y: ["La", "Gd", "Sc", "Lu", "Er"],
  Ca: ["Sr", "Ba", "Na", "K"],
  Fe: ["Co", "Ni", "Mn", "Cr", "Cu", "Ru"],
  Co: ["Fe", "Ni", "Mn", "Rh"],
  Ni: ["Co", "Cu", "Fe", "Pd"],
  Cu: ["Ni", "Zn", "Fe", "Ag"],
  Ti: ["V", "Zr", "Nb", "Hf"],
  V: ["Ti", "Cr", "Nb", "Ta"],
  Nb: ["V", "Ta", "Mo", "Ti"],
  Ta: ["Nb", "V", "W", "Hf"],
  Mo: ["W", "Cr", "Nb", "Re"],
  W: ["Mo", "Ta", "Re", "Cr"],
  Zr: ["Hf", "Ti", "Nb"],
  Hf: ["Zr", "Ti", "Ta"],
  Mn: ["Fe", "Cr", "Co"],
  Cr: ["V", "Mo", "Mn"],
  Bi: ["Sb", "Pb", "Tl"],
  Pb: ["Bi", "Sn", "Tl"],
  Se: ["Te", "S"],
  Te: ["Se", "S"],
  S: ["Se", "Te"],
  As: ["P", "Sb"],
  P: ["As", "N"],
  O: ["F", "N", "S"],
  N: ["O", "C", "P"],
  Al: ["Ga", "In", "B"],
  Ga: ["Al", "In"],
  Si: ["Ge", "Sn"],
  Ge: ["Si", "Sn"],
  B: ["C", "Al", "N"],
  C: ["B", "N"],
  Li: ["Na", "K"],
  Na: ["Li", "K"],
  K: ["Na", "Rb", "Cs"],
  Mg: ["Ca", "Be", "Zn"],
  Zn: ["Cu", "Mg", "Cd"],
  Sn: ["Ge", "Pb", "In"],
  In: ["Ga", "Tl", "Sn"],
  Ru: ["Os", "Fe", "Rh"],
  H: ["D", "Li"],
};

const INTERSTITIAL_CANDIDATES = ["H", "O", "N", "Li", "F", "C", "B"];

const KNOWN_LATTICE: Record<string, { a: number; b: number; c: number }> = {
  "SrTiO3": { a: 3.905, b: 3.905, c: 3.905 },
  "BaTiO3": { a: 3.992, b: 3.992, c: 4.036 },
  "LaAlO3": { a: 3.789, b: 3.789, c: 3.789 },
  "FeSe": { a: 3.765, b: 3.765, c: 5.518 },
  "FeTe": { a: 3.822, b: 3.822, c: 6.272 },
  "MgB2": { a: 3.086, b: 3.086, c: 3.524 },
  "YBa2Cu3O7": { a: 3.820, b: 3.886, c: 11.680 },
  "La2CuO4": { a: 3.787, b: 3.787, c: 13.226 },
  "BaFe2As2": { a: 3.963, b: 3.963, c: 13.017 },
  "NbSe2": { a: 3.445, b: 3.445, c: 12.547 },
  "Nb3Sn": { a: 5.289, b: 5.289, c: 5.289 },
  "NbN": { a: 4.394, b: 4.394, c: 4.394 },
  "H3S": { a: 3.089, b: 3.089, c: 3.089 },
  "LaH10": { a: 5.100, b: 5.100, c: 5.100 },
  "LaFeAsO": { a: 4.035, b: 4.035, c: 8.741 },
  "NdNiO2": { a: 3.921, b: 3.921, c: 3.281 },
  "Bi2Sr2CaCu2O8": { a: 3.814, b: 3.814, c: 30.89 },
  "V3Si": { a: 4.722, b: 4.722, c: 4.722 },
  "CaH6": { a: 3.520, b: 3.520, c: 3.520 },
};

const PROTOTYPE_SITES: Record<string, Array<{ element: string; frac: [number, number, number] }>> = {
  "perovskite": [
    { element: "A", frac: [0, 0, 0] },
    { element: "B", frac: [0.5, 0.5, 0.5] },
    { element: "X", frac: [0.5, 0.5, 0] },
    { element: "X", frac: [0.5, 0, 0.5] },
    { element: "X", frac: [0, 0.5, 0.5] },
  ],
  "rocksalt": [
    { element: "A", frac: [0, 0, 0] },
    { element: "B", frac: [0.5, 0.5, 0.5] },
  ],
  "fluorite": [
    { element: "A", frac: [0, 0, 0] },
    { element: "X", frac: [0.25, 0.25, 0.25] },
    { element: "X", frac: [0.75, 0.75, 0.75] },
  ],
  "hcp": [
    { element: "A", frac: [0, 0, 0] },
    { element: "A", frac: [1 / 3, 2 / 3, 0.5] },
  ],
  "bcc": [
    { element: "A", frac: [0, 0, 0] },
    { element: "A", frac: [0.5, 0.5, 0.5] },
  ],
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

function estimateLattice(formula: string): { a: number; b: number; c: number } {
  if (KNOWN_LATTICE[formula]) return KNOWN_LATTICE[formula];
  const comp = parseFormula(formula);
  const elements = Object.keys(comp);
  const totalAtoms = Object.values(comp).reduce((s, n) => s + n, 0);
  let avgRadius = 0;
  for (const el of elements) {
    const data = getElementData(el);
    avgRadius += ((data?.atomicRadius ?? 150) / 100) * (comp[el] / totalAtoms);
  }
  const a = Math.max(2.5, Math.min(8.0, avgRadius * 3.0));
  const hasLayered = elements.some(el => ["Se", "Te", "S", "As"].includes(el));
  const c = hasLayered ? a * 2.5 : a;
  return { a, b: a, c };
}

function classifyPrototype(formula: string): string {
  const comp = parseFormula(formula);
  const elements = Object.keys(comp);
  const totalAtoms = Object.values(comp).reduce((s, n) => s + n, 0);
  const hasO = elements.includes("O");
  const hasTM = elements.some(el => {
    const d = getElementData(el);
    return d && d.atomicNumber >= 21 && d.atomicNumber <= 30;
  });
  if (hasO && hasTM && elements.length >= 3 && totalAtoms >= 5) return "perovskite";
  if (elements.length === 2 && totalAtoms <= 3) return "rocksalt";
  if (elements.length === 2) return "hcp";
  return "bcc";
}

function buildUnitCellSites(formula: string): Array<{ element: string; frac: [number, number, number] }> {
  const comp = parseFormula(formula);
  const elements = Object.keys(comp);
  const totalAtoms = Object.values(comp).reduce((s, n) => s + n, 0);
  const sites: Array<{ element: string; frac: [number, number, number] }> = [];

  const roundedCounts: Record<string, number> = {};
  for (const el of elements) {
    roundedCounts[el] = Math.max(1, Math.round(comp[el]));
  }

  const totalSites = Object.values(roundedCounts).reduce((s, n) => s + n, 0);
  let siteIdx = 0;

  for (const el of elements) {
    const count = roundedCounts[el];
    for (let i = 0; i < count; i++) {
      const fx = ((siteIdx * 7 + 3) % totalSites) / totalSites;
      const fy = ((siteIdx * 11 + 5) % totalSites) / totalSites;
      const fz = ((siteIdx * 13 + 7) % totalSites) / totalSites;
      sites.push({ element: el, frac: [fx, fy, fz] });
      siteIdx++;
    }
  }

  return sites;
}

function buildSupercell(
  formula: string,
  size: [number, number, number]
): { atoms: DisorderedAtom[]; lattice: { a: number; b: number; c: number } } {
  const lat = estimateLattice(formula);
  const unitSites = buildUnitCellSites(formula);

  const atoms: DisorderedAtom[] = [];
  const [nx, ny, nz] = size;

  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++) {
        for (const site of unitSites) {
          atoms.push({
            element: site.element,
            x: (ix + site.frac[0]) * lat.a,
            y: (iy + site.frac[1]) * lat.b,
            z: (iz + site.frac[2]) * lat.c,
            isDefect: false,
          });
        }
      }
    }
  }

  return {
    atoms,
    lattice: { a: lat.a * nx, b: lat.b * ny, c: lat.c * nz },
  };
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

function introduceVacancy(
  atoms: DisorderedAtom[],
  element: string,
  fraction: number,
  rng: () => number
): { modified: DisorderedAtom[]; removed: number } {
  const targetIndices = atoms
    .map((a, i) => ({ a, i }))
    .filter(({ a }) => a.element === element)
    .map(({ i }) => i);

  const removeN = Math.max(1, Math.round(targetIndices.length * fraction));
  const shuffled = targetIndices.slice().sort(() => rng() - 0.5);
  const removeSet = new Set(shuffled.slice(0, removeN));

  const modified = atoms.filter((_, i) => !removeSet.has(i));
  return { modified, removed: removeSet.size };
}

function introduceSubstitution(
  atoms: DisorderedAtom[],
  element: string,
  substituent: string,
  fraction: number,
  rng: () => number
): { modified: DisorderedAtom[]; substituted: number } {
  const targetIndices = atoms
    .map((a, i) => ({ a, i }))
    .filter(({ a }) => a.element === element)
    .map(({ i }) => i);

  const subN = Math.max(1, Math.round(targetIndices.length * fraction));
  const shuffled = targetIndices.slice().sort(() => rng() - 0.5);
  const subSet = new Set(shuffled.slice(0, subN));

  const modified = atoms.map((a, i) => {
    if (subSet.has(i)) {
      return {
        ...a,
        element: substituent,
        originalElement: a.element,
        isDefect: true,
        defectType: "substitution" as DisorderType,
      };
    }
    return a;
  });

  return { modified, substituted: subSet.size };
}

function introduceInterstitial(
  atoms: DisorderedAtom[],
  element: string,
  fraction: number,
  lattice: { a: number; b: number; c: number },
  rng: () => number
): { modified: DisorderedAtom[]; added: number } {
  const totalSites = atoms.length;
  const addN = Math.max(1, Math.round(totalSites * fraction));
  const newAtoms = [...atoms];

  for (let i = 0; i < addN; i++) {
    const x = rng() * lattice.a;
    const y = rng() * lattice.b;
    const z = rng() * lattice.c;

    let tooClose = false;
    for (const existing of newAtoms) {
      const dx = existing.x - x;
      const dy = existing.y - y;
      const dz = existing.z - z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 1.2) {
        tooClose = true;
        break;
      }
    }
    if (tooClose) continue;

    newAtoms.push({
      element,
      x,
      y,
      z,
      isDefect: true,
      defectType: "interstitial",
    });
  }

  return { modified: newAtoms, added: newAtoms.length - atoms.length };
}

function introduceSiteMixing(
  atoms: DisorderedAtom[],
  element: string,
  fraction: number,
  rng: () => number
): { modified: DisorderedAtom[]; mixed: number } {
  const targetIndices = atoms
    .map((a, i) => ({ a, i }))
    .filter(({ a }) => a.element === element)
    .map(({ i }) => i);

  const otherElements = [...new Set(atoms.map(a => a.element).filter(e => e !== element))];
  if (otherElements.length === 0) return { modified: atoms, mixed: 0 };

  const mixN = Math.max(1, Math.round(targetIndices.length * fraction));
  const shuffled = targetIndices.slice().sort(() => rng() - 0.5);
  const mixSet = new Set(shuffled.slice(0, mixN));

  const modified = atoms.map((a, i) => {
    if (mixSet.has(i)) {
      const replacement = otherElements[Math.floor(rng() * otherElements.length)];
      return {
        ...a,
        element: replacement,
        originalElement: a.element,
        isDefect: true,
        defectType: "site-mixing" as DisorderType,
      };
    }
    return a;
  });

  return { modified, mixed: mixSet.size };
}

function estimateFormationEnergy(
  disorderType: DisorderType,
  element: string,
  fraction: number
): number {
  const data = getElementData(element);
  const ie = (data?.firstIonizationEnergy ?? 7) / 13.6;
  const radius = (data?.atomicRadius ?? 150) / 100;

  let Ef: number;
  switch (disorderType) {
    case "vacancy":
      Ef = ie * 0.4 + radius * 0.3 + fraction * 2.0;
      break;
    case "substitution":
      Ef = ie * 0.2 + 0.5 + fraction * 1.5;
      break;
    case "interstitial":
      Ef = ie * 0.3 + (1.0 / radius) * 0.5 + fraction * 3.0;
      break;
    case "site-mixing":
      Ef = 0.8 + fraction * 2.5;
      break;
    default:
      Ef = 2.0;
  }

  return Math.max(0.1, Math.min(6.0, Ef));
}

function estimateTcModifier(
  disorderType: DisorderType,
  element: string,
  fraction: number,
  baseFormula: string
): number {
  const comp = parseFormula(baseFormula);
  const elements = Object.keys(comp);

  const isCuprate = elements.includes("Cu") && elements.includes("O") &&
    elements.some(e => ["La", "Y", "Ba", "Sr", "Bi", "Ca"].includes(e));
  const isIronBased = elements.includes("Fe") &&
    elements.some(e => ["As", "Se", "Te", "P"].includes(e));

  let modifier = 1.0;

  switch (disorderType) {
    case "vacancy":
      if (element === "O" && isCuprate) {
        modifier = fraction <= 0.05 ? 1.0 + fraction * 3.0 : 1.15 - (fraction - 0.05) * 4.0;
      } else {
        modifier = 1.0 - fraction * 1.5;
      }
      break;

    case "substitution":
      if (isCuprate && element === "La") {
        modifier = fraction <= 0.15 ? 1.0 + fraction * 2.0 : 1.3 - (fraction - 0.15) * 2.0;
      } else if (isIronBased && element === "Fe") {
        modifier = fraction <= 0.08 ? 1.0 + fraction * 3.5 : 1.28 - (fraction - 0.08) * 3.0;
      } else {
        modifier = 1.0 + fraction * 0.5 * (fraction < 0.1 ? 1 : -1);
      }
      break;

    case "interstitial":
      if (element === "H") {
        modifier = 1.0 + fraction * 2.0;
      } else if (element === "O" && !isCuprate) {
        modifier = 1.0 - fraction * 0.8;
      } else {
        modifier = 1.0 + fraction * 0.3;
      }
      break;

    case "site-mixing":
      modifier = 1.0 - fraction * 0.8;
      break;
  }

  return Math.max(0.3, Math.min(1.5, modifier));
}

function generateNotes(
  disorderType: DisorderType,
  element: string,
  fraction: number,
  tcMod: number,
  baseFormula: string
): string {
  const pctStr = (fraction * 100).toFixed(1);
  const direction = tcMod >= 1.0 ? "enhances" : "suppresses";

  switch (disorderType) {
    case "vacancy":
      return `${pctStr}% ${element} vacancies in ${baseFormula}: ${direction} Tc by ${Math.abs((tcMod - 1) * 100).toFixed(0)}%. ` +
        `Vacancies modify carrier concentration and can create resonant impurity states near the Fermi level.`;

    case "substitution":
      return `${pctStr}% ${element} substitution in ${baseFormula}: ${direction} Tc by ${Math.abs((tcMod - 1) * 100).toFixed(0)}%. ` +
        `Chemical doping tunes the Fermi level and can optimize nesting or coupling.`;

    case "interstitial":
      return `${pctStr}% interstitial ${element} in ${baseFormula}: ${direction} Tc by ${Math.abs((tcMod - 1) * 100).toFixed(0)}%. ` +
        `Interstitials provide additional electron donors/acceptors and modify the phonon spectrum.`;

    case "site-mixing":
      return `${pctStr}% ${element} site-mixing in ${baseFormula}: ${direction} Tc by ${Math.abs((tcMod - 1) * 100).toFixed(0)}%. ` +
        `Random alloy disorder broadens band structure and acts as pair-breaking scatterer in anisotropic SCs.`;
  }
}

function chooseSuperCellSize(formula: string): [number, number, number] {
  const comp = parseFormula(formula);
  const atomsPerCell = Object.values(comp).reduce((s, n) => s + Math.round(n), 0);
  if (atomsPerCell * 64 <= 200) return [4, 4, 4];
  if (atomsPerCell * 27 <= 200) return [3, 3, 3];
  return [2, 2, 2];
}

const stats: DisorderGeneratorStats = {
  totalGenerated: 0,
  byType: { vacancy: 0, substitution: 0, interstitial: 0, "site-mixing": 0 },
  avgDefectFraction: 0,
  avgTcModifier: 0,
  bestTcModifier: 0,
  bestFormula: "",
  bestDisorderType: "",
  recentGenerations: [],
  topCandidates: [],
};

let totalDefectFractionSum = 0;
let totalTcModifierSum = 0;

function updateStats(result: DisorderedStructure) {
  stats.totalGenerated++;
  stats.byType[result.disorder.type]++;
  totalDefectFractionSum += result.defectFraction;
  totalTcModifierSum += result.tcModifierEstimate;
  stats.avgDefectFraction = totalDefectFractionSum / stats.totalGenerated;
  stats.avgTcModifier = totalTcModifierSum / stats.totalGenerated;

  if (result.tcModifierEstimate > stats.bestTcModifier) {
    stats.bestTcModifier = result.tcModifierEstimate;
    stats.bestFormula = result.base;
    stats.bestDisorderType = result.disorder.type;
  }

  stats.recentGenerations.unshift({
    base: result.base,
    type: result.disorder.type,
    element: result.disorder.element,
    fraction: result.disorder.fraction,
    defectCount: result.defectCount,
    tcModifier: result.tcModifierEstimate,
  });
  if (stats.recentGenerations.length > 20) stats.recentGenerations.length = 20;

  const entry = {
    base: result.base,
    type: result.disorder.type,
    element: result.disorder.element,
    fraction: result.disorder.fraction,
    tcModifier: result.tcModifierEstimate,
    formationEnergy: result.formationEnergyEstimate,
  };

  stats.topCandidates.push(entry);
  stats.topCandidates.sort((a, b) => b.tcModifier - a.tcModifier);
  if (stats.topCandidates.length > 15) stats.topCandidates.length = 15;
}

export function generateDisorderedStructure(
  baseFormula: string,
  disorder: DisorderSpec,
  supercellSize?: [number, number, number]
): DisorderedStructure {
  const size = supercellSize || chooseSuperCellSize(baseFormula);
  const { atoms: pristineAtoms, lattice } = buildSupercell(baseFormula, size);
  const seed = baseFormula.split("").reduce((s, c) => s + c.charCodeAt(0), 0) +
    disorder.element.charCodeAt(0) + Math.round(disorder.fraction * 1000);
  const rng = seededRandom(seed);

  let resultAtoms: DisorderedAtom[];
  let defectCount: number;

  const comp = parseFormula(baseFormula);
  const elementsInFormula = Object.keys(comp);

  switch (disorder.type) {
    case "vacancy": {
      if (!pristineAtoms.some(a => a.element === disorder.element)) {
        const closest = elementsInFormula[0] || disorder.element;
        const { modified, removed } = introduceVacancy(pristineAtoms, closest, disorder.fraction, rng);
        resultAtoms = modified;
        defectCount = removed;
      } else {
        const { modified, removed } = introduceVacancy(pristineAtoms, disorder.element, disorder.fraction, rng);
        resultAtoms = modified;
        defectCount = removed;
      }
      break;
    }

    case "substitution": {
      const sub = disorder.substituent ||
        (SUBSTITUTION_MAP[disorder.element] || [])[0] || "X";
      const targetEl = pristineAtoms.some(a => a.element === disorder.element)
        ? disorder.element : elementsInFormula[0];
      const { modified, substituted } = introduceSubstitution(
        pristineAtoms, targetEl, sub, disorder.fraction, rng
      );
      resultAtoms = modified;
      defectCount = substituted;
      break;
    }

    case "interstitial": {
      const { modified, added } = introduceInterstitial(
        pristineAtoms, disorder.element, disorder.fraction, lattice, rng
      );
      resultAtoms = modified;
      defectCount = added;
      break;
    }

    case "site-mixing": {
      const targetEl = pristineAtoms.some(a => a.element === disorder.element)
        ? disorder.element : elementsInFormula[0];
      const { modified, mixed } = introduceSiteMixing(
        pristineAtoms, targetEl, disorder.fraction, rng
      );
      resultAtoms = modified;
      defectCount = mixed;
      break;
    }

    default:
      resultAtoms = pristineAtoms;
      defectCount = 0;
  }

  const formationEnergy = estimateFormationEnergy(disorder.type, disorder.element, disorder.fraction);
  const tcModifier = estimateTcModifier(disorder.type, disorder.element, disorder.fraction, baseFormula);
  const defectFraction = defectCount / Math.max(1, pristineAtoms.length);
  const notes = generateNotes(disorder.type, disorder.element, disorder.fraction, tcModifier, baseFormula);

  const result: DisorderedStructure = {
    base: baseFormula,
    disorder,
    atoms: resultAtoms,
    supercellSize: size,
    latticeA: lattice.a,
    latticeB: lattice.b,
    latticeC: lattice.c,
    totalAtoms: resultAtoms.length,
    defectCount,
    defectFraction,
    formationEnergyEstimate: formationEnergy,
    tcModifierEstimate: tcModifier,
    notes,
    generatedAt: Date.now(),
  };

  updateStats(result);
  return result;
}

export function generateAllDisorderVariants(
  baseFormula: string,
  fractions?: number[]
): DisorderedStructure[] {
  const comp = parseFormula(baseFormula);
  const elements = Object.keys(comp);
  const frac = fractions || [0.02, 0.05, 0.10];
  const results: DisorderedStructure[] = [];

  for (const el of elements) {
    for (const f of frac) {
      results.push(generateDisorderedStructure(baseFormula, {
        type: "vacancy",
        element: el,
        fraction: f,
      }));
    }
  }

  for (const el of elements) {
    const subs = SUBSTITUTION_MAP[el];
    if (!subs || subs.length === 0) continue;
    for (const sub of subs.slice(0, 2)) {
      for (const f of frac.slice(0, 2)) {
        results.push(generateDisorderedStructure(baseFormula, {
          type: "substitution",
          element: el,
          fraction: f,
          substituent: sub,
        }));
      }
    }
  }

  for (const intEl of INTERSTITIAL_CANDIDATES.slice(0, 3)) {
    if (elements.includes(intEl)) continue;
    for (const f of frac.slice(0, 2)) {
      results.push(generateDisorderedStructure(baseFormula, {
        type: "interstitial",
        element: intEl,
        fraction: f,
      }));
    }
  }

  for (const el of elements) {
    results.push(generateDisorderedStructure(baseFormula, {
      type: "site-mixing",
      element: el,
      fraction: 0.05,
    }));
  }

  return results;
}

export function suggestDisorders(baseFormula: string): DisorderSpec[] {
  const comp = parseFormula(baseFormula);
  const elements = Object.keys(comp);
  const suggestions: DisorderSpec[] = [];

  const isCuprate = elements.includes("Cu") && elements.includes("O") &&
    elements.some(e => ["La", "Y", "Ba", "Sr"].includes(e));
  const isIronBased = elements.includes("Fe") &&
    elements.some(e => ["As", "Se", "Te"].includes(e));

  if (isCuprate) {
    suggestions.push({ type: "vacancy", element: "O", fraction: 0.05 });
    const alkEarth = elements.find(e => ["La", "Y"].includes(e));
    if (alkEarth) {
      suggestions.push({ type: "substitution", element: alkEarth, fraction: 0.10, substituent: "Sr" });
    }
    suggestions.push({ type: "substitution", element: "Cu", fraction: 0.05, substituent: "Ni" });
  }

  if (isIronBased) {
    suggestions.push({ type: "substitution", element: "Fe", fraction: 0.05, substituent: "Co" });
    const pnictogen = elements.find(e => ["As", "P"].includes(e));
    if (pnictogen) {
      suggestions.push({ type: "substitution", element: pnictogen, fraction: 0.03, substituent: pnictogen === "As" ? "P" : "As" });
    }
    suggestions.push({ type: "vacancy", element: elements.find(e => ["Se", "Te"].includes(e)) || "Se", fraction: 0.03 });
  }

  for (const el of elements) {
    if (!suggestions.some(s => s.type === "vacancy" && s.element === el)) {
      suggestions.push({ type: "vacancy", element: el, fraction: 0.05 });
      break;
    }
  }

  if (!suggestions.some(s => s.type === "interstitial")) {
    const intEl = INTERSTITIAL_CANDIDATES.find(e => !elements.includes(e)) || "H";
    suggestions.push({ type: "interstitial", element: intEl, fraction: 0.03 });
  }

  if (!suggestions.some(s => s.type === "site-mixing")) {
    suggestions.push({ type: "site-mixing", element: elements[0], fraction: 0.05 });
  }

  return suggestions;
}

export function getDisorderGeneratorStats(): DisorderGeneratorStats {
  return { ...stats };
}
