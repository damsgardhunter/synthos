const MATERIAL_FAMILIES: Record<string, RegExp> = {
  "Hydrides": /H\d|LaH|YH|CeH|CaH|BaH|SrH|MgH|hydride/i,
  "Cuprates": /(?:La|Y|Ba|Sr|Bi|Tl|Hg|Ca|Nd|Pr|Sm|Eu|Gd).*Cu.*O|cuprate|YBCO|BSCCO/i,
  "Pnictides": /Fe.*As|Ba.*Fe.*As|Sr.*Fe.*As|La.*Fe.*As|Fe.*P(?:[^btdm]|$)|pnictide/i,
  "Chalcogenides": /(?:Fe|Nb|Ta|Mo|W|Bi|Sb|Cu|Cd|Zn|Sn|Pb|In|Ga|Ti|Zr|Hf|V|Cr|Mn|Co|Ni|Pd|Pt|Re|Ir)(?:Se|Te)\d*|FeSe|FeTe|NbSe|TaSe|TaS|NbS|MoS|WS|chalcogenide/i,
  "Borides": /(?:Mg|Ti|Zr|Hf|V|Nb|Ta|Cr|Mo|W|Mn|Fe|Co|Ni|La|Y|Ca|Sr|Sc|Al|Re|Ru|Os)B\d|MgB\d*|boride/i,
  "Carbides": /(?:Ti|Zr|Hf|V|Nb|Ta|Cr|Mo|W|Fe|Si|Sc|Y|La)C\d*(?:[^aeioulrs]|$)|carbide|SiC/i,
  "Nitrides": /(?:Ti|Zr|Hf|V|Nb|Ta|Cr|Mo|W|Al|Ga|In|Si|B|Sc|Y|La)N\d*(?:[^abeiodr]|$)|nitride|BN|GaN|AlN/i,
  "Oxides": /(?:Sr|Ba|Pb|Bi|La|Y|Nd|Ca|Mg|Ti|Zr|Mn|Co|Ni|Fe|V|Cr|W|Mo).*O\d|oxide|perovskite|SrTiO|BaTiO/i,
  "Intermetallics": /Nb.*Sn|Nb.*Ti|V.*Si|Nb.*Ge|intermetallic/i,
};

export function classifyFamily(formula: string): string {
  for (const [family, pattern] of Object.entries(MATERIAL_FAMILIES)) {
    if (pattern.test(formula)) return family;
  }
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

function parseFormulaCounts(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const cleaned = formula.replace(/[₀-₉]/g, (c) => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + count;
  }
  return counts;
}

export function getCompositionHash(formula: string): string {
  const counts = parseFormulaCounts(formula);
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  if (total === 0) return formula;
  const ratios = Object.entries(counts)
    .map(([el, n]) => ({ el, ratio: Math.round((n / total) * 1000) / 1000 }))
    .sort((a, b) => a.el.localeCompare(b.el));
  return ratios.map(r => `${r.el}:${r.ratio}`).join("-");
}

export function getPrototypeHash(formula: string): string {
  const counts = parseFormulaCounts(formula);
  const vals = Object.values(counts);
  if (vals.length === 0) return formula;
  let g = vals[0];
  for (let i = 1; i < vals.length; i++) {
    let a = g, b = vals[i];
    while (b > 0.001) { const t = b; b = a % b; a = t; }
    g = a;
  }
  if (g < 0.001) g = 1;
  const normalized = vals.map(v => Math.round(v / g)).sort((a, b) => a - b);
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
  Po: 2.00, At: 2.20, Th: 1.30, U: 1.38,
};

export function normalizeFormula(raw: string): string {
  const counts = parseFormulaCounts(raw);
  const elements = Object.keys(counts);
  if (elements.length === 0) return raw;

  const vals = Object.values(counts);
  const allInts = vals.every(v => Number.isInteger(v) && v >= 1);

  if (allInts) {
    let g = vals[0];
    for (let i = 1; i < vals.length; i++) {
      let a = g, b = vals[i];
      while (b > 0) { const t = b; b = a % b; a = t; }
      g = a;
    }
    if (g > 1) {
      for (const el of elements) counts[el] = counts[el] / g;
    }
  }

  elements.sort((a, b) => (ELECTRONEGATIVITY[a] ?? 2.0) - (ELECTRONEGATIVITY[b] ?? 2.0));

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
  const valsA = elsA.map(el => countsA[el]).sort((a, b) => a - b);
  const valsB = elsB.map(el => countsB[el]).sort((a, b) => a - b);
  return valsA.every((v, i) => v === valsB[i]);
}
