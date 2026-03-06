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
