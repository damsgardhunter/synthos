import { readFileSync, writeFileSync } from 'fs';

const path = 'server/learning/materials-project-client.ts';
let src = readFileSync(path, 'utf8');

function rep(from, to, label) {
  if (src.includes(from)) { src = src.replace(from, to); return true; }
  const srcLF = src.replace(/\r\n/g, '\n');
  const fromLF = from.replace(/\r\n/g, '\n');
  if (srcLF.includes(fromLF)) {
    const toLF = to.replace(/\r\n/g, '\n');
    const replaced = srcLF.replace(fromLF, toLF);
    src = replaced.replace(/\n/g, '\r\n');
    return true;
  }
  console.error('FAILED:', label ?? JSON.stringify(from.slice(0, 80)));
  return false;
}

// Insert MPStructureData interface + fetchMPStructureData after fetchMagnetism
rep(
`export async function fetchPhonon(formula: string): Promise<MPPhononData | null> {`,
`export interface MPStructureData {
  latticeParams: { a: number; b: number; c: number };
  atomicPositions: { element: string; x: number; y: number; z: number }[];
  spaceGroup: string | null;
}

export async function fetchMPStructureData(formula: string): Promise<MPStructureData | null> {
  const cached = await getCachedData(formula, "structure");
  if (cached) return cached as MPStructureData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/summary/", {
    formula: normalizedFormula,
    _limit: "1",
    fields: "structure,material_id,symmetry,formula_pretty",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const s = entry.structure;
  if (!s?.lattice || !Array.isArray(s.sites) || s.sites.length === 0) return null;

  const result: MPStructureData = {
    latticeParams: {
      a: s.lattice.a ?? 5,
      b: s.lattice.b ?? 5,
      c: s.lattice.c ?? 5,
    },
    atomicPositions: s.sites.map((site: any) => ({
      element: site.species?.[0]?.element ?? "X",
      x: site.abc?.[0] ?? 0,
      y: site.abc?.[1] ?? 0,
      z: site.abc?.[2] ?? 0,
    })),
    spaceGroup: entry.symmetry?.symbol ?? null,
  };

  await setCachedData(formula, "structure", result, entry.material_id);
  return result;
}

export async function fetchPhonon(formula: string): Promise<MPPhononData | null> {`,
'insert fetchMPStructureData'
);

writeFileSync(path, src, 'utf8');
console.log('Done. Checking...');

const checks = [
  'export interface MPStructureData',
  'export async function fetchMPStructureData',
  'await getCachedData(formula, "structure")',
  'fields: "structure,material_id,symmetry,formula_pretty"',
  'atomicPositions: s.sites.map',
];
let ok = true;
for (const c of checks) {
  const found = src.includes(c);
  console.log((found ? '✓ ' : '✗ MISSING: ') + c);
  if (!found) ok = false;
}
process.exit(ok ? 0 : 1);
