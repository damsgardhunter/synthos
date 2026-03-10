import { parseFormulaElements } from "./physics-engine";
import { normalizeFormula, isValidFormula } from "./utils";
import { getElementData } from "./elemental-data";

export interface DopingSpec {
  type: "substitutional" | "vacancy" | "interstitial";
  base: string;
  dopant?: string;
  site?: string;
  fraction: number;
  resultFormula: string;
  supercellSize: number;
  rationale: string;
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
  recentResults: Array<{ base: string; variants: number; timestamp: number }>;
}

const stats: DopingEngineStats = {
  totalBaseMaterials: 0,
  totalVariantsGenerated: 0,
  substitutionalCount: 0,
  vacancyCount: 0,
  interstitialCount: 0,
  validVariants: 0,
  recentResults: [],
};

const MAX_RECENT = 100;

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

function generateSubstitutionalVariants(
  formula: string,
  counts: Record<string, number>,
  maxVariants: number = 8
): DopingSpec[] {
  const variants: DopingSpec[] = [];
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const supercellMult = getSupercellMultiplier(totalAtoms);

  for (const site of elements) {
    const siteCount = counts[site];
    if (siteCount < 0.5) continue;

    const dopants = SC_DOPANT_MAP[site];
    if (!dopants) continue;

    const siteData = getElementData(site);
    if (!siteData) continue;

    for (const dopant of dopants) {
      if (elements.includes(dopant)) continue;
      if (variants.length >= maxVariants) break;

      const dopantData = getElementData(dopant);
      if (!dopantData) continue;

      const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
        ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
        : 0.5;
      if (radiusDiff > 0.3) continue;

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

        const rationale = `${dopant} substitution at ${site} site (${(fraction * 100).toFixed(0)}%): `
          + `radius match ${(1 - radiusDiff).toFixed(2)}, `
          + `replaces ${nReplace}/${sitesInSupercell} ${site} atoms in ${supercellMult > 1 ? supercellMult + "x supercell" : "unit cell"}`;

        variants.push({
          type: "substitutional",
          base: formula,
          dopant,
          site,
          fraction,
          resultFormula: normalizeFormula(resultFormula),
          supercellSize: supercellMult,
          rationale,
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

  const vacancySites = elements.filter(e => VACANCY_TARGETS.includes(e));
  if (vacancySites.length === 0) return [];

  for (const site of vacancySites) {
    const siteCount = counts[site];
    if (siteCount < 1) continue;

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

      variants.push({
        type: "vacancy",
        base: formula,
        site,
        fraction,
        resultFormula: normalizeFormula(resultFormula),
        supercellSize: supercellMult,
        rationale: `${site} vacancy doping (${(fraction * 100).toFixed(0)}%): removed ${nRemove}/${sitesInSupercell} ${site} atoms — creates carrier doping via ${site} vacancies`,
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

  const dopantPool = INTERSTITIAL_DOPANTS[structureType] || INTERSTITIAL_DOPANTS.general;
  const availableDopants = dopantPool.filter(d => !elements.includes(d));

  for (const dopant of availableDopants) {
    if (variants.length >= maxVariants) break;

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

      variants.push({
        type: "interstitial",
        base: formula,
        dopant,
        fraction,
        resultFormula: normalizeFormula(resultFormula),
        supercellSize: supercellMult,
        rationale: `${dopant} interstitial insertion (${(fraction * 100).toFixed(0)}%): ${nInsert} atoms into ${structureType} structure — common intercalation dopant for ${structureType} materials`,
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
  stats.substitutionalCount += substitutional.filter(v => !seen.has(v.resultFormula) || unique.includes(v)).length;
  stats.vacancyCount += vacancy.filter(v => unique.includes(v)).length;
  stats.interstitialCount += interstitial.filter(v => unique.includes(v)).length;
  stats.validVariants += unique.length;

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

export function runDopingBatch(
  formulas: string[],
  maxVariantsPerBase: number = 8,
  maxTotalDoped: number = 50,
  excludeSet?: Set<string>
): { dopedFormulas: string[]; specs: DopingSpec[]; stats: { basesProcessed: number; totalVariants: number; substitutional: number; vacancy: number; interstitial: number } } {
  const dopedFormulas: string[] = [];
  const specs: DopingSpec[] = [];
  let subCount = 0, vacCount = 0, intCount = 0;

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
    },
  };
}

export function getDopingEngineStats(): DopingEngineStats {
  return { ...stats };
}

export function getDopingRecommendations(formula: string): {
  substitutional: Array<{ dopant: string; site: string; rationale: string }>;
  vacancy: Array<{ site: string; rationale: string }>;
  interstitial: Array<{ dopant: string; rationale: string }>;
} {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const structureType = classifyLayeredOrCage(formula);

  const sub: Array<{ dopant: string; site: string; rationale: string }> = [];
  for (const site of elements) {
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
        sub.push({
          dopant,
          site,
          rationale: `${dopant} replaces ${site}: ionic radius match ${((1 - radiusDiff) * 100).toFixed(0)}%, common SC dopant pair`,
        });
      }
    }
  }

  const vac: Array<{ site: string; rationale: string }> = [];
  for (const site of elements) {
    if (VACANCY_TARGETS.includes(site) && counts[site] >= 1) {
      vac.push({
        site,
        rationale: `${site} vacancy creates hole/electron carriers — effective for oxide/chalcogenide SC`,
      });
    }
  }

  const intPool = INTERSTITIAL_DOPANTS[structureType] || INTERSTITIAL_DOPANTS.general;
  const int: Array<{ dopant: string; rationale: string }> = [];
  for (const dopant of intPool) {
    if (!elements.includes(dopant)) {
      int.push({
        dopant,
        rationale: `${dopant} intercalation into ${structureType} lattice — enhances electron-phonon coupling`,
      });
    }
  }

  return { substitutional: sub, vacancy: vac, interstitial: int.slice(0, 3) };
}
