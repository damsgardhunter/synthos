const MATERIAL_FAMILIES: Record<string, RegExp> = {
  "Hydrides": /H\d|LaH|YH|CeH|CaH|BaH|SrH|MgH|hydride/i,
  "Cuprates": /Cu.*O|Ba.*Cu|Sr.*Cu|La.*Cu|cuprate|YBCO|BSCCO/i,
  "Pnictides": /Fe.*As|Ba.*Fe|Sr.*Fe|La.*Fe.*As|pnictide/i,
  "Chalcogenides": /Se|Te|FeSe|FeTe|chalcogenide/i,
  "Borides": /B\d|MgB|boride/i,
  "Carbides": /C\d|carbide|SiC/i,
  "Nitrides": /N\d|nitride|BN|GaN|AlN/i,
  "Oxides": /O\d|oxide|perovskite|SrTiO|BaTiO/i,
  "Intermetallics": /Nb.*Sn|Nb.*Ti|V.*Si|intermetallic/i,
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
