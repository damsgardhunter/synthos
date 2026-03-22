import { readFileSync, writeFileSync } from 'fs';

const path = 'server/learning/graph-neural-net.ts';
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

// ── 1. Add import for prefetchStructures at the top (after existing imports) ─
rep(
`import { parseFormulaCounts as parseFormulaCountsCanonical } from "./utils";`,
`import { parseFormulaCounts as parseFormulaCountsCanonical } from "./utils";
import { prefetchStructures } from "./structure-resolver";`,
'add prefetchStructures import'
);

// ── 2. Add structure pre-fetch before the graph cache build loop ──────────────
rep(
`  const graphCache = new Map<string, CrystalGraph>();
  const origEmbeddings = new Map<string, number[][]>();
  for (let si = 0; si < trainingData.length; si++) {`,
`  const graphCache = new Map<string, CrystalGraph>();
  const origEmbeddings = new Map<string, number[][]>();

  // Pre-fetch real atomic positions from Materials Project for formulas that
  // lack a prototype-based graph. Real positions activate buildCrystalGraph's
  // physics-consistent distance calculation path instead of heuristic distances.
  const formulasNeedingStructure = [...new Set(
    trainingData.filter(s => !s.prototype).map(s => s.formula)
  )];
  const mpStructureMap = formulasNeedingStructure.length > 0
    ? await prefetchStructures(formulasNeedingStructure)
    : new Map<string, any>();

  for (let si = 0; si < trainingData.length; si++) {`,
'add mpStructureMap pre-fetch'
);

// ── 3. Pass real structure data into buildCrystalGraph ────────────────────────
rep(
`        : buildCrystalGraph(sample.formula, sample.structure, sample.pressureGpa, sampleHints);`,
`        : buildCrystalGraph(sample.formula, sample.structure ?? mpStructureMap.get(sample.formula), sample.pressureGpa, sampleHints);`,
'wire mpStructureMap into buildCrystalGraph'
);

writeFileSync(path, src, 'utf8');
console.log('Done. Checking...');

const checks = [
  'import { prefetchStructures } from "./structure-resolver"',
  'const mpStructureMap = formulasNeedingStructure.length > 0',
  'sample.structure ?? mpStructureMap.get(sample.formula)',
];
let ok = true;
for (const c of checks) {
  const found = src.includes(c);
  console.log((found ? '✓ ' : '✗ MISSING: ') + c);
  if (!found) ok = false;
}
process.exit(ok ? 0 : 1);
