import { normalizeFormula, classifyFamily } from "./utils";
import { extractFeatures } from "./ml-predictor";
import { gbPredict } from "./gradient-boost";
import { ELEMENTAL_DATA, getElementData } from "./elemental-data";

const MAX_FORMULAS = 2000;

const ELECTRONEGATIVITY: Record<string, number> = {
  H: 2.20, Li: 0.98, Be: 1.57, B: 2.04, C: 2.55, N: 3.04, O: 3.44, F: 3.98,
  Na: 0.93, Mg: 1.31, Al: 1.61, Si: 1.90, P: 2.19, S: 2.58, Cl: 3.16,
  K: 0.82, Ca: 1.00, Sc: 1.36, Ti: 1.54, V: 1.63, Cr: 1.66, Mn: 1.55, Fe: 1.83,
  Co: 1.88, Ni: 1.91, Cu: 1.90, Zn: 1.65, Ga: 1.81, Ge: 2.01, As: 2.18, Se: 2.55,
  Br: 2.96, Rb: 0.82, Sr: 0.95, Y: 1.22, Zr: 1.33, Nb: 1.60, Mo: 2.16,
  Tc: 1.90, Ru: 2.20, Rh: 2.28, Pd: 2.20, Ag: 1.93, Cd: 1.69, In: 1.78, Sn: 1.96,
  Sb: 2.05, Te: 2.10, I: 2.66, Cs: 0.79, Ba: 0.89, La: 1.10, Ce: 1.12,
  Pr: 1.13, Nd: 1.14, Sm: 1.17, Eu: 1.20, Gd: 1.20, Tb: 1.10, Dy: 1.22, Ho: 1.23,
  Er: 1.24, Tm: 1.25, Yb: 1.10, Lu: 1.27, Hf: 1.30, Ta: 1.50, W: 2.36, Re: 1.90,
  Os: 2.20, Ir: 2.20, Pt: 2.28, Au: 2.54, Hg: 2.00, Tl: 1.62, Pb: 2.33, Bi: 2.02,
  Th: 1.30, U: 1.38,
};

const MAX_OXIDATION_STATE: Record<string, number> = {
  H: 1, Li: 1, Na: 1, K: 1, Rb: 1, Cs: 1,
  Be: 2, Mg: 2, Ca: 2, Sr: 2, Ba: 2,
  Sc: 3, Y: 3, La: 3, Ce: 4, Pr: 4, Nd: 3, Sm: 3, Eu: 3, Gd: 3,
  Tb: 4, Dy: 3, Ho: 3, Er: 3, Tm: 3, Yb: 3, Lu: 3,
  Ti: 4, Zr: 4, Hf: 4,
  V: 5, Nb: 5, Ta: 5,
  Cr: 6, Mo: 6, W: 6,
  Mn: 7, Re: 7,
  Fe: 3, Ru: 8, Os: 8,
  Co: 3, Rh: 4, Ir: 4,
  Ni: 3, Pd: 4, Pt: 4,
  Cu: 2, Ag: 1, Au: 3,
  Zn: 2, Cd: 2, Hg: 2,
  B: 3, Al: 3, Ga: 3, In: 3, Tl: 3,
  C: 4, Si: 4, Ge: 4, Sn: 4, Pb: 4,
  N: 5, P: 5, As: 5, Sb: 5, Bi: 5,
  O: 2, S: 6, Se: 6, Te: 6,
  F: 1, Cl: 7, Br: 5, I: 7,
  Th: 4, U: 6,
};

const MIN_OXIDATION_STATE: Record<string, number> = {
  H: -1, Li: 0, Na: 0, K: 0, Rb: 0, Cs: 0,
  Be: 0, Mg: 0, Ca: 0, Sr: 0, Ba: 0,
  Sc: 0, Y: 0, La: 0, Ce: 0, Pr: 0, Nd: 0, Sm: 0, Eu: 0, Gd: 0,
  Ti: 0, Zr: 0, Hf: 0, V: 0, Nb: 0, Ta: 0,
  Cr: 0, Mo: 0, W: 0, Mn: 0, Re: 0,
  Fe: 0, Ru: 0, Os: 0, Co: 0, Rh: 0, Ir: 0,
  Ni: 0, Pd: 0, Pt: 0, Cu: 0, Ag: 0, Au: 0,
  Zn: 0, Cd: 0, Hg: 0,
  B: -3, Al: 0, Ga: 0, In: 0, Tl: 0,
  C: -4, Si: -4, Ge: -4, Sn: -4, Pb: -4,
  N: -3, P: -3, As: -3, Sb: -3, Bi: -3,
  O: -2, S: -2, Se: -2, Te: -2,
  F: -1, Cl: -1, Br: -1, I: -1,
  Th: 0, U: 0,
};

const CARBIDE_FORMERS = new Set([
  "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Fe", "Co", "Ni",
  "Sc", "Y", "La", "Ce", "Th", "U", "B", "Si", "Al", "Ca", "Sr", "Ba",
]);

const NITRIDE_FORMERS = new Set([
  "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Fe", "Co", "Mn",
  "Sc", "Y", "La", "Ce", "Al", "Ga", "In", "Si", "Ge", "B", "Ca", "Sr", "Ba", "Li",
]);

const BORIDE_FORMERS = new Set([
  "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Fe", "Co", "Ni", "Mn",
  "Sc", "Y", "La", "Ce", "Mg", "Ca", "Sr", "Ba", "Al", "Re", "Os", "Ru",
]);

const HYDRIDE_FORMERS = new Set([
  "La", "Y", "Ca", "Sr", "Ba", "Sc", "Ti", "Zr", "Hf", "V", "Nb", "Ta",
  "Cr", "Mo", "W", "Mn", "Fe", "Co", "Ni", "Pd", "Cu",
  "Mg", "Li", "Na", "K", "Rb", "Cs", "Ce", "Pr", "Nd", "Th", "U",
  "Al", "Ga", "In", "Sn", "Pb", "Bi",
]);

const ELEMENT_COMPATIBILITY: Record<string, Set<string>> = {
  C: CARBIDE_FORMERS,
  N: NITRIDE_FORMERS,
  B: BORIDE_FORMERS,
  H: HYDRIDE_FORMERS,
};

const SIMILAR_ELEMENTS: Record<string, string[]> = {
  Ti: ["Zr", "Hf", "V", "Nb"],
  Zr: ["Ti", "Hf", "Nb", "Ta"],
  Hf: ["Ti", "Zr", "Ta"],
  V: ["Nb", "Ta", "Ti", "Cr", "Mo"],
  Nb: ["V", "Ta", "Ti", "Zr", "Mo"],
  Ta: ["Nb", "V", "Hf", "W"],
  Cr: ["Mo", "W", "V", "Mn"],
  Mo: ["W", "Cr", "Nb", "V", "Re"],
  W: ["Mo", "Cr", "Ta", "Re"],
  Mn: ["Fe", "Cr", "Re", "Co"],
  Fe: ["Co", "Ni", "Mn", "Ru"],
  Co: ["Fe", "Ni", "Rh", "Ir"],
  Ni: ["Co", "Pd", "Pt", "Cu", "Fe"],
  Cu: ["Ag", "Au", "Ni", "Zn"],
  Zn: ["Cd", "Cu", "Ga"],
  Ru: ["Os", "Fe", "Rh", "Ir"],
  Rh: ["Ir", "Co", "Pd"],
  Pd: ["Pt", "Ni", "Rh"],
  Ag: ["Cu", "Au"],
  Os: ["Ru", "Ir", "Re"],
  Ir: ["Rh", "Os", "Pt"],
  Pt: ["Pd", "Ni", "Ir"],
  Au: ["Ag", "Cu"],
  Sc: ["Y", "La", "Ti"],
  Y: ["Sc", "La", "Lu", "Gd"],
  La: ["Y", "Ce", "Sc", "Ba"],
  Ce: ["La", "Pr", "Nd"],
  Pr: ["Ce", "Nd", "La"],
  Nd: ["Pr", "Ce", "Sm"],
  Sm: ["Nd", "Eu", "Gd"],
  Eu: ["Sm", "Gd", "Ba", "Sr"],
  Gd: ["Sm", "Tb", "Y"],
  Tb: ["Gd", "Dy"],
  Dy: ["Tb", "Ho", "Y"],
  Ho: ["Dy", "Er"],
  Er: ["Ho", "Tm"],
  Tm: ["Er", "Yb"],
  Yb: ["Tm", "Lu", "Eu"],
  Lu: ["Yb", "Y", "Sc"],
  Li: ["Na", "K"],
  Na: ["Li", "K"],
  K: ["Na", "Rb", "Ca"],
  Rb: ["K", "Cs"],
  Cs: ["Rb", "K", "Ba"],
  Be: ["Mg", "Al"],
  Mg: ["Be", "Ca", "Al", "Zn"],
  Ca: ["Sr", "Ba", "Mg"],
  Sr: ["Ca", "Ba", "Eu"],
  Ba: ["Sr", "Ca", "La", "Eu"],
  Al: ["Ga", "In", "Mg", "Be"],
  Ga: ["Al", "In", "Zn"],
  In: ["Ga", "Al", "Tl", "Sn"],
  Tl: ["In", "Pb", "Bi"],
  Si: ["Ge", "C", "Sn"],
  Ge: ["Si", "Sn"],
  Sn: ["Ge", "Pb", "In"],
  Pb: ["Sn", "Bi", "Tl"],
  As: ["P", "Sb", "Se"],
  Sb: ["As", "Bi"],
  Bi: ["Sb", "Pb", "Tl"],
  Se: ["Te", "S"],
  Te: ["Se", "S"],
  S: ["Se", "Te", "O"],
  O: ["S", "Se"],
  B: ["C", "N", "Al"],
  C: ["B", "Si", "N"],
  N: ["C", "B", "P"],
  H: ["Li", "B", "F"],
  P: ["As", "N"],
  F: ["Cl", "O"],
  Th: ["U", "La", "Ce"],
  U: ["Th", "Np"],
  Re: ["Mn", "Os", "W", "Mo"],
};

const DOPANT_ELEMENTS = ["Li", "Na", "K", "Mg", "Ca", "Sr", "Ba", "La", "Y", "Sc", "H", "B", "N", "F"];

const HIGH_COUPLING_METALS = new Set(["Nb", "Ta", "Mo", "V", "Ti", "Hf", "W", "Re", "Zr"]);
const LIGHT_PHONON_ELEMENTS = new Set(["B", "C", "N", "H"]);
const CHARGE_RESERVOIR_ELEMENTS = new Set(["La", "Y", "Ca", "Sr", "Sc", "Ce", "Nd"]);
const WEAK_SC_ELEMENTS = new Set(["Zn", "Tl", "Pd", "Cd", "Hg", "Au", "Ag", "Bi", "Pb"]);

function elementSubstitutionDistance(elA: string, elB: string): number {
  const enA = ELECTRONEGATIVITY[elA] ?? 2.0;
  const enB = ELECTRONEGATIVITY[elB] ?? 2.0;
  const dataA = getElementData(elA);
  const dataB = getElementData(elB);
  const radiusA = dataA?.atomicRadius ?? 150;
  const radiusB = dataB?.atomicRadius ?? 150;
  const zA = dataA?.atomicNumber ?? 50;
  const zB = dataB?.atomicNumber ?? 50;

  const enDiff = Math.abs(enA - enB);
  const radiusDiff = Math.abs(radiusA - radiusB) / 100;
  const periodDiff = Math.abs(Math.floor(Math.log2(zA + 1)) - Math.floor(Math.log2(zB + 1)));

  return enDiff + radiusDiff + periodDiff * 0.5;
}

const MIN_SUBSTITUTION_DISTANCE = 1.0;

function hasSuperconductingPotential(formula: string): boolean {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const hasHighCoupling = elements.some(el => HIGH_COUPLING_METALS.has(el));
  const hasLightPhonon = elements.some(el => LIGHT_PHONON_ELEMENTS.has(el));
  const weakCount = elements.filter(el => WEAK_SC_ELEMENTS.has(el)).length;
  if (weakCount > 1) return false;
  if (!hasHighCoupling && !hasLightPhonon) return false;
  return true;
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

function countsToFormula(counts: Record<string, number>): string {
  const elements = Object.keys(counts).filter(el => counts[el] > 0);
  elements.sort((a, b) => (ELECTRONEGATIVITY[a] ?? 2.0) - (ELECTRONEGATIVITY[b] ?? 2.0));
  let result = "";
  for (const el of elements) {
    const c = counts[el];
    if (c === 1) result += el;
    else if (Number.isInteger(c)) result += `${el}${c}`;
    else result += `${el}${Math.round(c * 100) / 100}`;
  }
  return result;
}

function reduceStoichiometry(counts: Record<string, number>): Record<string, number> {
  const vals = Object.values(counts).filter(v => v > 0);
  if (vals.length === 0) return counts;
  const allInts = vals.every(v => Number.isInteger(v) && v >= 1);
  if (!allInts) return counts;
  let g = vals[0];
  for (let i = 1; i < vals.length; i++) {
    let a = g, b = vals[i];
    while (b > 0) { const t = b; b = a % b; a = t; }
    g = a;
  }
  if (g > 1) {
    const reduced: Record<string, number> = {};
    for (const [el, n] of Object.entries(counts)) {
      reduced[el] = n / g;
    }
    return reduced;
  }
  return counts;
}

function canonicalize(formula: string): string {
  const counts = parseFormulaCounts(formula);
  const reduced = reduceStoichiometry(counts);
  return countsToFormula(reduced);
}

function passesValenceFilter(formula: string): boolean {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length === 0) return false;

  for (const el of elements) {
    const count = counts[el];
    if (count <= 0 || !Number.isFinite(count)) return false;
    if (!ELEMENTAL_DATA[el]) return false;
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms < 2 || totalAtoms > 50) return false;

  let maxChargeSum = 0;
  let minChargeSum = 0;
  for (const el of elements) {
    const maxOx = MAX_OXIDATION_STATE[el] ?? 4;
    const minOx = MIN_OXIDATION_STATE[el] ?? -2;
    maxChargeSum += maxOx * counts[el];
    minChargeSum += minOx * counts[el];
  }

  if (minChargeSum > 0 && maxChargeSum > 0) return false;
  if (maxChargeSum < 0 && minChargeSum < 0) return false;

  return minChargeSum <= 0 && maxChargeSum >= 0;
}

function passesCompatibilityFilter(formula: string): boolean {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  for (const [anion, formers] of Object.entries(ELEMENT_COMPATIBILITY)) {
    if (!counts[anion] || counts[anion] <= 0) continue;
    const otherElements = elements.filter(el => el !== anion);
    if (otherElements.length === 0) continue;
    const hasCompatible = otherElements.some(el => formers.has(el));
    if (!hasCompatible) return false;
  }

  return true;
}

export function generateElementSubstitutions(baseFormulas: string[], count: number = 200): string[] {
  const results = new Set<string>();

  for (const base of baseFormulas) {
    const counts = parseFormulaCounts(base);
    const elements = Object.keys(counts);

    for (const el of elements) {
      const similar = SIMILAR_ELEMENTS[el] || [];
      for (const sub of similar) {
        if (elements.includes(sub)) continue;
        if (elementSubstitutionDistance(el, sub) < MIN_SUBSTITUTION_DISTANCE) continue;
        const newCounts = { ...counts };
        newCounts[sub] = newCounts[el];
        delete newCounts[el];
        const formula = canonicalize(countsToFormula(newCounts));
        if (formula && formula.length > 1) results.add(formula);
        if (results.size >= count) break;
      }

      if (results.size >= count) break;

      for (const sub of similar) {
        if (elements.includes(sub)) continue;
        if (elementSubstitutionDistance(el, sub) < MIN_SUBSTITUTION_DISTANCE) continue;
        for (const el2 of elements) {
          if (el2 === el) continue;
          const similar2 = SIMILAR_ELEMENTS[el2] || [];
          for (const sub2 of similar2) {
            if (sub2 === sub || elements.includes(sub2)) continue;
            if (elementSubstitutionDistance(el2, sub2) < MIN_SUBSTITUTION_DISTANCE) continue;
            const newCounts = { ...counts };
            newCounts[sub] = newCounts[el];
            delete newCounts[el];
            newCounts[sub2] = newCounts[el2];
            delete newCounts[el2];
            const formula = canonicalize(countsToFormula(newCounts));
            if (formula && formula.length > 1) results.add(formula);
            if (results.size >= count) break;
          }
          if (results.size >= count) break;
        }
        if (results.size >= count) break;
      }
      if (results.size >= count) break;
    }

    for (const el of elements) {
      const stoich = counts[el];
      for (const variation of [stoich + 1, stoich - 1, stoich * 2, Math.max(1, Math.round(stoich / 2))]) {
        if (variation < 1 || variation > 12 || variation === stoich) continue;
        const newCounts = { ...counts, [el]: variation };
        const formula = canonicalize(countsToFormula(newCounts));
        if (formula && formula.length > 1) results.add(formula);
        if (results.size >= count) break;
      }
      if (results.size >= count) break;
    }

    if (results.size >= count) break;
  }

  return Array.from(results).slice(0, count);
}

export function generateCompositionInterpolations(formula1: string, formula2: string, steps: number = 10): string[] {
  const counts1 = parseFormulaCounts(formula1);
  const counts2 = parseFormulaCounts(formula2);
  const allElements = Array.from(new Set([...Object.keys(counts1), ...Object.keys(counts2)]));
  const results: string[] = [];

  for (let i = 1; i < steps; i++) {
    const frac = i / steps;
    const interpolated: Record<string, number> = {};
    for (const el of allElements) {
      const v1 = counts1[el] || 0;
      const v2 = counts2[el] || 0;
      const interp = v1 * (1 - frac) + v2 * frac;
      const rounded = Math.round(interp);
      if (rounded > 0) interpolated[el] = rounded;
    }
    if (Object.keys(interpolated).length >= 2) {
      const formula = canonicalize(countsToFormula(interpolated));
      if (formula && formula.length > 1) results.push(formula);
    }
  }

  return results;
}

export function generateRandomDopedVariants(baseFormulas: string[], dopantElements: string[] = DOPANT_ELEMENTS, count: number = 50): string[] {
  const results = new Set<string>();

  for (const base of baseFormulas) {
    const counts = parseFormulaCounts(base);
    const elements = Object.keys(counts);
    const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

    for (const dopant of dopantElements) {
      if (elements.includes(dopant)) continue;

      for (const dopantCount of [1, 2, 3]) {
        if (totalAtoms + dopantCount > 20) continue;
        const newCounts = { ...counts, [dopant]: dopantCount };
        const formula = canonicalize(countsToFormula(newCounts));
        if (formula && formula.length > 1) results.add(formula);
        if (results.size >= count) break;
      }
      if (results.size >= count) break;

      for (const el of elements) {
        const stoich = counts[el];
        if (stoich <= 1) continue;
        for (const replace of [1, Math.floor(stoich / 2)]) {
          if (replace >= stoich || replace <= 0) continue;
          const newCounts = { ...counts };
          newCounts[el] = stoich - replace;
          newCounts[dopant] = replace;
          const formula = canonicalize(countsToFormula(newCounts));
          if (formula && formula.length > 1) results.add(formula);
          if (results.size >= count) break;
        }
        if (results.size >= count) break;
      }
      if (results.size >= count) break;
    }
    if (results.size >= count) break;
  }

  return Array.from(results).slice(0, count);
}

export function generateCompositionSweep(elements: string[], maxAtoms: number = 8): string[] {
  const results: string[] = [];
  const n = elements.length;
  if (n < 2 || n > 5) return results;

  const maxPerElement = Math.min(maxAtoms, 6);

  function enumerate(idx: number, current: number[]): boolean {
    if (results.length >= MAX_FORMULAS) return true;

    if (idx === n) {
      const total = current.reduce((s, v) => s + v, 0);
      if (total < 2 || total > maxAtoms) return false;
      const zeroCount = current.filter(v => v === 0).length;
      if (zeroCount > n - 2) return false;

      const counts: Record<string, number> = {};
      for (let i = 0; i < n; i++) {
        if (current[i] > 0) counts[elements[i]] = current[i];
      }
      if (Object.keys(counts).length < 2) return false;

      const formula = canonicalize(countsToFormula(counts));
      if (formula && formula.length > 1) results.push(formula);
      return results.length >= MAX_FORMULAS;
    }

    for (let v = 0; v <= maxPerElement; v++) {
      current.push(v);
      if (enumerate(idx + 1, current)) return true;
      current.pop();
    }
    return false;
  }

  enumerate(0, []);
  return results;
}

export interface MassiveGenerationStats {
  totalGenerated: number;
  uniqueAfterDedup: number;
  passedValenceFilter: number;
  passedCompatibilityFilter: number;
  passedPreScreen: number;
}

export interface RapidScreenResult {
  formula: string;
  predictedTc: number;
  gbScore: number;
}

export function rapidGBScreen(formulas: string[]): RapidScreenResult[] {
  const results: RapidScreenResult[] = [];

  for (const formula of formulas) {
    try {
      const features = extractFeatures(formula);
      const gb = gbPredict(features);
      if (gb.tcPredicted >= 5) {
        results.push({
          formula,
          predictedTc: gb.tcPredicted,
          gbScore: gb.score,
        });
      }
    } catch {
      continue;
    }
  }

  results.sort((a, b) => b.predictedTc - a.predictedTc);
  return results;
}

export function runMassiveGeneration(
  topCandidates: { formula: string; predictedTc?: number }[],
  focusArea: string
): { formulas: string[]; stats: MassiveGenerationStats } {
  const baseFormulas = topCandidates.map(c => c.formula);
  const allGenerated = new Set<string>();
  const stats: MassiveGenerationStats = {
    totalGenerated: 0,
    uniqueAfterDedup: 0,
    passedValenceFilter: 0,
    passedCompatibilityFilter: 0,
    passedPreScreen: 0,
  };

  const focusElements: Record<string, string[][]> = {
    Carbides: [["Nb", "C"], ["Ti", "C"], ["Mo", "C"], ["V", "C"], ["Zr", "C"], ["Hf", "C"], ["Ta", "C"], ["W", "C"]],
    Borides: [["Nb", "B"], ["Ti", "B"], ["Zr", "B"], ["Mo", "B"], ["V", "B"], ["Ta", "B"], ["Mg", "B"], ["Ca", "B"]],
    Nitrides: [["Nb", "N"], ["Ti", "N"], ["Zr", "N"], ["V", "N"], ["Mo", "N"], ["Ta", "N"], ["Hf", "N"]],
    Hydrides: [["La", "H"], ["Y", "H"], ["Ca", "H"], ["Sr", "H"], ["Ba", "H"], ["Th", "H"], ["Sc", "H"]],
    Intermetallics: [["Nb", "Ge"], ["Nb", "Sn"], ["V", "Si"], ["Nb", "Al"], ["Mo", "Ge"], ["V", "Ga"]],
    Cuprates: [["Ba", "Cu", "O"], ["La", "Cu", "O"], ["Y", "Ba", "Cu", "O"], ["Bi", "Sr", "Cu", "O"]],
    Pnictides: [["Ba", "Fe", "As"], ["Sr", "Fe", "As"], ["La", "Fe", "As"], ["Fe", "Se"]],
    Chalcogenides: [["Fe", "Se"], ["Nb", "Se"], ["Mo", "Se"], ["Fe", "Te"]],
    Sulfides: [["Mo", "S"], ["Nb", "S"], ["Ta", "S"], ["Ti", "S"]],
    Oxides: [["Sr", "Ti", "O"], ["Ba", "Ti", "O"], ["La", "Mn", "O"]],
    Alloys: [["Nb", "Ti"], ["Nb", "Zr"], ["Mo", "Re"], ["V", "Ti"]],
    Kagome: [["K", "V", "Sb"], ["Rb", "V", "Sb"], ["Cs", "V", "Sb"], ["K", "V", "Bi"], ["Cs", "Ti", "Sb"], ["Ba", "V", "Sb"], ["Ca", "V", "Sb"], ["Sr", "V", "Bi"]],
    "Mixed-mechanism": [["La", "Fe", "As", "O"], ["Ba", "Fe", "As"], ["Sr", "Fe", "As"], ["Fe", "Se"], ["Fe", "Te"], ["La", "Ni", "O"], ["Nd", "Ni", "O"], ["La", "Cu", "O"], ["Y", "Cu", "O"], ["Ba", "Co", "As"], ["Sr", "Co", "As"], ["Ca", "Fe", "As"]],
    Layered: [["Nb", "Se"], ["Ta", "Se"], ["Mo", "S"], ["W", "Se"], ["Ti", "Se"], ["Nb", "S"], ["Mo", "Se"], ["Ta", "S"]],
    "Layered-chalcogenide": [["Nb", "Se"], ["Ta", "Se"], ["Mo", "S"], ["W", "Se"], ["Ti", "Se"], ["Nb", "S"], ["Mo", "Se"], ["Ta", "S"], ["Li", "Nb", "Se"], ["Cu", "Nb", "Se"]],
    "Layered-pnictide": [["La", "Fe", "As"], ["Ba", "Fe", "As"], ["Sr", "Fe", "As"], ["Ce", "Fe", "As"], ["La", "Co", "As"], ["Ba", "Co", "As"], ["La", "Fe", "P"], ["Ba", "Ni", "As"]],
    "Intercalated-layered": [["Li", "Nb", "Se"], ["Na", "Nb", "Se"], ["K", "Mo", "S"], ["Ca", "Ta", "Se"], ["Li", "C"], ["K", "C"], ["Ca", "C"], ["Sr", "Nb", "Se"]],
  };

  const scBiasedSeeds: string[][] = [
    ["Nb", "B", "C"], ["Ta", "B", "C"], ["V", "B", "N"], ["Ti", "B", "N"],
    ["Nb", "C", "N"], ["Ta", "C", "N"],
    ["La", "Nb", "B"], ["Sc", "Nb", "C"],
  ];

  const unconventionalSeeds: string[][] = [
    ["Nb", "Se"], ["Ta", "Se"], ["Nb", "S"], ["Mo", "Se"],
    ["Fe", "Se"], ["Fe", "Te"], ["Fe", "As"],
    ["Ba", "Fe", "As"], ["Sr", "Fe", "As"], ["La", "Fe", "P"],
    ["K", "V", "Sb"], ["Cs", "V", "Bi"], ["Rb", "V", "Sb"],
    ["La", "Ni", "O"], ["Sr", "Co", "O"], ["Ba", "Cu", "O"],
    ["Ca", "H"], ["La", "H"], ["Y", "H"],
    ["Nb", "Se", "S"], ["Ta", "Se", "Te"], ["Mo", "S", "Se"],
  ];

  const focusPairs = focusElements[focusArea] || focusElements["Carbides"];
  const biasedSubset = scBiasedSeeds.sort(() => Math.random() - 0.5).slice(0, 4);
  const unconvSubset = unconventionalSeeds.sort(() => Math.random() - 0.5).slice(0, 8);
  const combinedPairs = [...focusPairs, ...biasedSubset, ...unconvSubset];
  const seedFormulas: string[] = [];
  for (const pair of combinedPairs) {
    const stoichs = [[1, 1], [1, 2], [2, 1], [3, 1], [1, 3], [2, 3], [3, 2]];
    for (const s of stoichs) {
      if (pair.length === 2) {
        seedFormulas.push(`${pair[0]}${s[0]}${pair[1]}${s[1]}`);
      } else if (pair.length === 3) {
        seedFormulas.push(`${pair[0]}${s[0]}${pair[1]}${s[1]}${pair[2]}${Math.max(1, s[1])}`);
      } else if (pair.length === 4) {
        seedFormulas.push(`${pair[0]}${s[0]}${pair[1]}${Math.max(1, s[0])}${pair[2]}${s[1]}${pair[3]}${Math.max(1, s[1] + s[0])}`);
      }
    }
  }

  const allSeeds = [...baseFormulas, ...seedFormulas];

  const substitutions = generateElementSubstitutions(allSeeds, 400);
  for (const f of substitutions) allGenerated.add(f);

  const sorted = topCandidates
    .filter(c => c.predictedTc != null)
    .sort((a, b) => (b.predictedTc ?? 0) - (a.predictedTc ?? 0));
  for (let i = 0; i < sorted.length && i < sorted.length - 1; i++) {
    for (let j = i + 1; j < sorted.length && j < i + 3; j++) {
      const interps = generateCompositionInterpolations(sorted[i].formula, sorted[j].formula, 10);
      for (const f of interps) allGenerated.add(f);
    }
  }

  const doped = generateRandomDopedVariants(allSeeds.slice(0, 20), DOPANT_ELEMENTS, 200);
  for (const f of doped) allGenerated.add(f);

  const hcMetals = ["Nb", "Ta", "Mo", "V", "Ti", "Hf", "W", "Re", "Zr"];
  const lpElements = ["B", "C", "N"];
  const reservoirs = ["La", "Y", "Ca", "Sr", "Sc", "Ce"];
  for (let i = 0; i < 100; i++) {
    const m1 = hcMetals[Math.floor(Math.random() * hcMetals.length)];
    const lp = lpElements[Math.floor(Math.random() * lpElements.length)];
    const s1 = Math.floor(Math.random() * 3) + 1;
    const s2 = Math.floor(Math.random() * 3) + 1;
    allGenerated.add(canonicalize(`${m1}${s1}${lp}${s2}`));
    if (Math.random() > 0.5) {
      const m2 = hcMetals[Math.floor(Math.random() * hcMetals.length)];
      if (m2 !== m1) {
        allGenerated.add(canonicalize(`${m1}${s1}${m2}1${lp}${s2}`));
      }
    }
    if (Math.random() > 0.6) {
      const res = reservoirs[Math.floor(Math.random() * reservoirs.length)];
      allGenerated.add(canonicalize(`${res}1${m1}${s1}${lp}${s2}`));
    }
  }

  for (const pair of focusPairs) {
    if (allGenerated.size >= MAX_FORMULAS) break;
    const swept = generateCompositionSweep(pair, 6);
    for (const f of swept) {
      allGenerated.add(f);
      if (allGenerated.size >= MAX_FORMULAS) break;
    }
  }

  stats.totalGenerated = allGenerated.size;

  const dedupedMap = new Map<string, string>();
  const allGeneratedArr = Array.from(allGenerated);
  for (const f of allGeneratedArr) {
    const canonical = canonicalize(f);
    if (!dedupedMap.has(canonical)) {
      dedupedMap.set(canonical, f);
    }
  }
  const uniqueFormulas = Array.from(dedupedMap.keys());
  stats.uniqueAfterDedup = uniqueFormulas.length;

  const valenceFiltered: string[] = [];
  for (const f of uniqueFormulas) {
    if (passesValenceFilter(f)) {
      valenceFiltered.push(f);
    }
  }
  stats.passedValenceFilter = valenceFiltered.length;

  const compatFiltered: string[] = [];
  for (const f of valenceFiltered) {
    if (passesCompatibilityFilter(f) && hasSuperconductingPotential(f)) {
      compatFiltered.push(f);
    }
  }
  stats.passedCompatibilityFilter = compatFiltered.length;

  const screened = rapidGBScreen(compatFiltered);
  const top50 = screened.slice(0, 50);
  stats.passedPreScreen = top50.length;

  return {
    formulas: top50.map(r => r.formula),
    stats,
  };
}
