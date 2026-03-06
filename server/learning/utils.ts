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
