const TRANSITION_METALS = new Set([
  "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
  "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
  "La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Th","U","Np","Pu",
]);

const METALS = new Set([
  ...TRANSITION_METALS,
  "Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba",
  "Al","Ga","In","Tl","Sn","Pb","Bi",
]);

const REGEX_FAMILIES: Record<string, RegExp> = {
  "Mixed-mechanism": /(?:La|Ce|Pr|Nd|Sm|Gd|Ba|Sr|Ca|Y)(?:Fe|Ni|Co|Cu).*(?:As|Se|Te|O)|(?:Fe|Ni|Co|Cu)(?:As|Se|Te)(?:O|F)?|mixed.?mechanism/i,
  "Hydrides": /H\d|LaH|YH|CeH|CaH|BaH|SrH|MgH|hydride/i,
  "Cuprates": /(?:La|Y|Ba|Sr|Bi|Tl|Hg|Ca|Nd|Pr|Sm|Eu|Gd).*Cu.*O|cuprate|YBCO|BSCCO/i,
  "Heavy Fermions": /(?:Ce|Yb|U|Pu|Np)(?:In|Co|Rh|Ir|Pd|Pt|Ni|Cu|Al|Ga|Ge|Si|Sn|Sb|Bi)\d|CeCoIn|UPt|UBe|CeCu/i,
  "Pnictides": /Fe.*As|Ba.*Fe.*As|Sr.*Fe.*As|La.*Fe.*As|Fe.*P(?:[^btdm]|$)|pnictide/i,
  "Chalcogenides": /(?:Fe|Nb|Ta|Mo|W|Bi|Sb|Cu|Cd|Zn|Sn|Pb|In|Ga|Ti|Zr|Hf|V|Cr|Mn|Co|Ni|Pd|Pt|Re|Ir)(?:Se|Te)\d*|FeSe|FeTe|NbSe|TaSe|chalcogenide/i,
  "Silicides": /(?:Fe|Co|Ni|Mn|Cr|V|Ti|Nb|Mo|W|Ru|Os|Ir|Pt|Pd|Re)Si\d|silicide|FeSi|CoSi|MnSi/i,
  "Phosphides": /(?:Fe|Co|Ni|Mn|Ga|In|Zn|Cd|Al|B)P\d*(?:[^btdm]|$)|phosphide|GaP|InP/i,
  "Intermetallics": /(?:Nb|V|Nb|Ta).*(?:Sn|Ge|Si|Ga|Al)|Nb.*Ti|NbSn|V3Si|Nb3Ge|Nb3Al|intermetallic/i,
  "Kagome": /(?:K|Rb|Cs|Ca|Sr|Ba)(?:V|Ti|Cr|Mn|Fe|Co|Ni)3(?:Sb|Bi|Sn|Ge|As)\d|kagome/i,
  "Layered-chalcogenide": /(?:Nb|Ta|Mo|W|Ti|Zr|Hf|V|Re)(?:Se|S|Te)2|(?:Li|Na|K|Ca|Sr|Cu)(?:Nb|Ta|Mo|W|Ti)(?:Se|S|Te)2/i,
  "Layered-pnictide": /(?:La|Ce|Pr|Nd|Sm|Gd|Y)(?:Fe|Co|Ni|Mn|Cr)(?:As|P|Sb)(?:O|F)|(?:Ba|Sr|Ca)(?:Fe|Co|Ni)2(?:As|P)2/i,
  "Intercalated-layered": /(?:Li|Na|K|Ca|Rb|Cs|Cu|Ag|NH4|TMA).*(?:Se|S|Te)2|(?:Li|Na|K|Rb|Cs)C(?:6|8|12)/i,
  "Nickelates": /(?:La|Nd|Pr|Sm|Gd|Y)NiO|(?:La|Nd|Pr).*Ni.*O|nickelate/i,
  "Borocarbides": /(?:Y|Lu|Er|Ho|Dy|Tm|Sc)Ni2B2C|(?:La|Ce|Pr|Nd)Ni2B2C|borocarbide/i,
  "Clathrates": /(?:Ba|Sr|K|Na|Eu).*(?:Si|Ge|Sn).*(?:46|clathrate)|clathrate/i,
};

export function classifyFamily(formula: string): string {
  for (const [family, pattern] of Object.entries(REGEX_FAMILIES)) {
    if (pattern.test(formula)) return family;
  }

  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const has = (el: string) => (counts[el] ?? 0) > 0;
  const hasMetal = elements.some(el => METALS.has(el));

  if (has("Ni") && has("O") && elements.some(el => ["La","Nd","Pr","Sm","Gd","Y"].includes(el))) return "Nickelates";
  if (has("B") && has("C") && has("Ni") && hasMetal) return "Borocarbides";

  const PNICTIDE_TM = ["Fe","Co","Ni","Mn","Cr"];
  const PNICTIDE_ANION = ["As","P","Sb"];
  const PNICTIDE_SPACER = ["Li","Na","K","Ba","Sr","Ca","La","Ce","Pr","Nd","Sm","Gd","Y"];
  const hasPnictideTM = PNICTIDE_TM.some(el => has(el));
  const hasPnictideAnion = PNICTIDE_ANION.some(el => has(el));
  const hasSpacer = PNICTIDE_SPACER.some(el => has(el));
  if (hasPnictideTM && hasPnictideAnion && (hasSpacer || elements.length >= 3)) return "Pnictides";

  if (has("H") && hasMetal && !has("O") && !has("S") && !has("Se") && elements.length <= 3) return "Hydrides";
  if (has("S") && hasMetal && !has("O") && !has("Se") && !has("Te") && elements.length >= 3) return "Sulfides";
  if (has("B") && hasMetal && !has("O") && !has("N")) return "Borides";
  if (has("C") && hasMetal && !has("O") && !has("H") && !has("Se") && !has("Te") && !has("As")) return "Carbides";
  if (has("N") && hasMetal && !has("O") && !has("H") && !has("Se") && !has("Te") && !has("As")) return "Nitrides";
  if (has("O") && hasMetal) return "Oxides";

  if (elements.length >= 2 && elements.every(el => METALS.has(el))) return "Alloys";

  return "Other";
}

export function safeNumber(val: unknown, fallback: number = 0): number {
  if (val === null || val === undefined) return fallback;
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

export function safeDivide(numerator: number, denominator: number, fallback: number = 0): number {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) return fallback;
  const result = numerator / denominator;
  return Number.isFinite(result) ? result : fallback;
}

export function safeFixed(val: number, digits: number = 2): number {
  if (!Number.isFinite(val)) return 0;
  return Number(val.toFixed(digits));
}

function expandParentheses(formula: string): string {
  let result = formula;
  let safety = 20;
  while (/[(\[]/.test(result) && safety-- > 0) {
    result = result.replace(/[(\[]([^()\[\]]+)[)\]](\d*\.?\d*)/g, (_m, inner, mult) => {
      const factor = mult ? parseFloat(mult) : 1;
      return inner.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_: string, el: string, n: string) => {
        const count = n ? parseFloat(n) : 1;
        const newCount = count * factor;
        return newCount === 1 ? el : `${el}${newCount}`;
      });
    });
  }
  return result;
}

const UNICODE_SUBSCRIPT_MAP: Record<string, string> = {
  "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
  "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
};

function replaceUnicodeSubscripts(s: string): string {
  return s.replace(/[₀₁₂₃₄₅₆₇₈₉]/g, (c) => UNICODE_SUBSCRIPT_MAP[c] ?? c);
}

export function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const counts: Record<string, number> = {};
  let cleaned = replaceUnicodeSubscripts(formula);
  cleaned = expandParentheses(cleaned);
  const regex = /([A-Z][a-z]?)(\d+\.?\d*|\.\d+)?/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const raw = match[2];
    const count = raw ? Number(raw) : 1;
    if (!count || isNaN(count)) continue;
    counts[el] = (counts[el] || 0) + count;
  }
  return counts;
}

export function getCompositionHash(formula: string): string {
  const counts = parseFormulaCounts(formula);
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  if (total < 1e-6) return formula;
  const ratios = Object.entries(counts)
    .map(([el, n]) => ({ el, ratio: Math.round((n / total) * 1000) / 1000 }))
    .sort((a, b) => a.el.localeCompare(b.el));
  return ratios.map(r => `${r.el}:${r.ratio}`).join("-");
}

function integerGcd(a: number, b: number): number {
  a = Math.abs(Math.round(a));
  b = Math.abs(Math.round(b));
  while (b > 0) { const t = b; b = a % b; a = t; }
  return a;
}

function toIntegerCounts(vals: number[]): number[] {
  const maxDec = vals.reduce((mx, v) => {
    const s = v.toFixed(6).replace(/0+$/, "");
    const dot = s.indexOf(".");
    return Math.max(mx, dot >= 0 ? s.length - dot - 1 : 0);
  }, 0);
  const scale = Math.pow(10, Math.min(maxDec, 6));
  return vals.map(v => Math.round(v * scale));
}

export function getPrototypeHash(formula: string): string {
  const counts = parseFormulaCounts(formula);
  const vals = Object.values(counts);
  if (vals.length === 0) return formula;
  const intVals = toIntegerCounts(vals);
  let g = intVals[0];
  for (let i = 1; i < intVals.length; i++) {
    g = integerGcd(g, intVals[i]);
  }
  if (g < 1) g = 1;
  const normalized = intVals.map(v => Math.round(v / g)).sort((a, b) => a - b);
  return normalized.join(":");
}

const ELECTRONEGATIVITY: Record<string, number> = {
  H: 2.20, He: 0, Li: 0.98, Be: 1.57, B: 2.04, C: 2.55, N: 3.04, O: 3.44, F: 3.98, Ne: 0,
  Na: 0.93, Mg: 1.31, Al: 1.61, Si: 1.90, P: 2.19, S: 2.58, Cl: 3.16, Ar: 0,
  K: 0.82, Ca: 1.00, Sc: 1.36, Ti: 1.54, V: 1.63, Cr: 1.66, Mn: 1.55, Fe: 1.83,
  Co: 1.88, Ni: 1.91, Cu: 1.90, Zn: 1.65, Ga: 1.81, Ge: 2.01, As: 2.18, Se: 2.55,
  Br: 2.96, Kr: 0, Rb: 0.82, Sr: 0.95, Y: 1.22, Zr: 1.33, Nb: 1.60, Mo: 2.16,
  Tc: 1.90, Ru: 2.20, Rh: 2.28, Pd: 2.20, Ag: 1.93, Cd: 1.69, In: 1.78, Sn: 1.96,
  Sb: 2.05, Te: 2.10, I: 2.66, Xe: 0, Cs: 0.79, Ba: 0.89, La: 1.10, Ce: 1.12,
  Pr: 1.13, Nd: 1.14, Sm: 1.17, Eu: 1.20, Gd: 1.20, Tb: 1.10, Dy: 1.22, Ho: 1.23,
  Er: 1.24, Tm: 1.25, Yb: 1.10, Lu: 1.27, Hf: 1.30, Ta: 1.50, W: 2.36, Re: 1.90,
  Os: 2.20, Ir: 2.20, Pt: 2.28, Au: 2.54, Hg: 2.00, Tl: 1.62, Pb: 2.33, Bi: 2.02,
  Po: 2.00, At: 2.20, Fr: 0.70, Ra: 0.90, Ac: 1.10,
  Th: 1.30, Pa: 1.50, U: 1.38, Np: 1.36, Pu: 1.28, Am: 1.30, Pm: 1.13,
};

export function getElectronegativitySpread(counts: Record<string, number>): number {
  const elements = Object.keys(counts).filter(el => ELECTRONEGATIVITY[el] !== undefined && ELECTRONEGATIVITY[el] > 0);
  if (elements.length < 2) return 0;

  let minEn = Infinity;
  let maxEn = -Infinity;
  for (const el of elements) {
    const en = ELECTRONEGATIVITY[el];
    if (en < minEn) minEn = en;
    if (en > maxEn) maxEn = en;
  }
  return maxEn - minEn;
}

export function getWeightedElectronegativity(counts: Record<string, number>): number {
  let totalWeight = 0;
  let weightedSum = 0;
  for (const [el, n] of Object.entries(counts)) {
    const en = ELECTRONEGATIVITY[el] ?? 0;
    if (en > 0) {
      weightedSum += en * n;
      totalWeight += n;
    }
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 2.0;
}

const VALID_ELEMENTS = new Set([
  ...Object.keys(ELECTRONEGATIVITY),
  "Rn", "Cm", "Bk", "Cf",
  "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
  "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]);

const NOBLE_GASES = new Set(["He", "Ne", "Ar", "Kr", "Xe", "Rn"]);

export function isValidFormula(formula: string): boolean {
  if (typeof formula !== "string") return false;
  let cleaned = replaceUnicodeSubscripts(formula);
  cleaned = cleaned.replace(/[()[\]{},;δ−+\-\.x\s]/g, "");
  cleaned = cleaned.replace(/\d+/g, " ").trim();
  const elementTokens = cleaned.match(/[A-Z][a-z]?/g);
  if (!elementTokens || elementTokens.length === 0) return false;
  for (const el of elementTokens) {
    if (!VALID_ELEMENTS.has(el)) {
      return false;
    }
  }
  const uniqueElements = [...new Set(elementTokens)];
  if (uniqueElements.some(el => NOBLE_GASES.has(el))) {
    return false;
  }
  return true;
}

export function normalizeFormula(raw: string): string {
  if (typeof raw !== "string") raw = String(raw ?? "");
  raw = raw.replace(/\s+/g, "");
  if (!isValidFormula(raw)) return raw;

  const counts = parseFormulaCounts(raw);
  const elements = Object.keys(counts);
  if (elements.length === 0) return raw;

  let vals = Object.values(counts);
  const hasFractions = vals.some(v => !Number.isInteger(v) || v < 1);

  if (hasFractions && vals.every(v => v > 0)) {
    const decimalPlaces = vals.map(v => {
      const s = v.toString();
      const dot = s.indexOf(".");
      return dot >= 0 ? s.length - dot - 1 : 0;
    });
    const maxDec = Math.max(...decimalPlaces);
    const pow10 = Math.pow(10, maxDec);
    for (const el of elements) {
      counts[el] = Math.round(counts[el] * pow10);
    }
    vals = Object.values(counts);
  }

  const allInts = vals.every(v => Number.isInteger(v) && v >= 1);

  if (allInts) {
    let g = Math.round(vals[0]);
    for (let i = 1; i < vals.length; i++) {
      g = integerGcd(g, Math.round(vals[i]));
    }
    if (g > 1) {
      for (const el of elements) counts[el] = Math.round(counts[el] / g);
    }
  }

  vals = Object.values(counts);
  const totalAtoms = vals.reduce((s, v) => s + v, 0);
  if (totalAtoms > 50) {
    const elKeys = elements.slice();
    const rawVals = elKeys.map(el => counts[el]);
    const intVals2 = toIntegerCounts(rawVals);
    let g2 = intVals2[0];
    for (let i = 1; i < intVals2.length; i++) {
      g2 = integerGcd(g2, intVals2[i]);
    }
    if (g2 > 0) {
      for (let i = 0; i < elKeys.length; i++) {
        counts[elKeys[i]] = Math.round(intVals2[i] / g2);
      }
    }
  }

  elements.sort((a, b) => a.localeCompare(b));

  let result = "";
  for (const el of elements) {
    const c = counts[el];
    if (c === 1) {
      result += el;
    } else if (Number.isInteger(c)) {
      result += `${el}${c}`;
    } else {
      const rounded = Math.round(c * 100) / 100;
      result += `${el}${rounded}`;
    }
  }
  return result;
}

export function isIsostructuralDuplicate(formulaA: string, formulaB: string): boolean {
  if (formulaA === formulaB) return true;
  const countsA = parseFormulaCounts(formulaA);
  const countsB = parseFormulaCounts(formulaB);
  const elsA = Object.keys(countsA).sort();
  const elsB = Object.keys(countsB).sort();
  if (elsA.length !== elsB.length) return false;

  function gcdReduce(vals: number[]): number[] {
    if (vals.length === 0) return vals;
    let g = vals[0];
    for (let i = 1; i < vals.length; i++) {
      let a = g, b = vals[i];
      while (b > 0.001) { const t = b; b = a % b; a = t; }
      g = a;
    }
    if (g < 0.001) g = 1;
    return vals.map(v => Math.round(v / g));
  }

  const valsA = gcdReduce(elsA.map(el => countsA[el]).sort((a, b) => a - b));
  const valsB = gcdReduce(elsB.map(el => countsB[el]).sort((a, b) => a - b));
  return valsA.every((v, i) => v === valsB[i]);
}

const FORBIDDEN_WORDS: [RegExp, string][] = [
  [/\bbreakthrough\b/gi, "notable finding"],
  [/\bconfirmed\b/gi, "verified"],
  [/\bbreakthroughs\b/gi, "notable findings"],
];

export function sanitizeForbiddenWords(text: string): string {
  let result = text;
  for (const [pattern, replacement] of FORBIDDEN_WORDS) {
    result = result.replace(pattern, replacement);
  }
  return result;
}

export const NONMETALS = new Set([
  "H", "He", "B", "C", "N", "O", "F", "Ne",
  "Si", "P", "S", "Cl", "Ar",
  "Ge", "As", "Se", "Br", "Kr",
  "Te", "I", "Xe",
  "At", "Rn",
]);

export function getMetalElements(elements: string[]): string[] {
  return elements.filter(e => !NONMETALS.has(e));
}

export function getHydrogenMetalRatio(counts: Record<string, number>, elements: string[]): number {
  const hCount = counts["H"] || 0;
  const metalElements = getMetalElements(elements);
  const metalAtomCount = metalElements.reduce((s, e) => s + (counts[e] || 0), 0);
  return metalAtomCount > 0 ? hCount / metalAtomCount : 0;
}

export function computeOrbitalHc2(tc: number, lambdaP: number, coherenceLengthNm: number): number {
  const PHI0 = 2.07e-15;
  const xiM = coherenceLengthNm * 1e-9;
  if (xiM <= 0) return 0;
  const orbitalLimit = PHI0 / (2 * Math.PI * xiM * xiM);
  const pauliLimit = 1.86 * tc * Math.sqrt(1 + lambdaP);
  return Math.min(orbitalLimit, pauliLimit);
}
