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

// 1. assignSiteLabels: sort element keys
rep(
  `  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);`,
  `  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts).sort();`,
  'assignSiteLabels elements.sort()'
);

// 2. buildPrototypeGraph: sort element keys
rep(
  `  const rawCounts = parseFormulaCounts(formula);
  const elements = Object.keys(rawCounts);
  const protoInfo = PROTOTYPE_COORDINATIONS[prototype];`,
  `  const rawCounts = parseFormulaCounts(formula);
  const elements = Object.keys(rawCounts).sort();
  const protoInfo = PROTOTYPE_COORDINATIONS[prototype];`,
  'buildPrototypeGraph elements.sort()'
);

// 3. buildCrystalGraph: sort element keys
rep(
  `  const rawCounts = parseFormulaCounts(formula);
  const elements = Object.keys(rawCounts);
  const { normalized, multiplicities } = normalizeFormulaCounts(rawCounts);`,
  `  const rawCounts = parseFormulaCounts(formula);
  const elements = Object.keys(rawCounts).sort();
  const { normalized, multiplicities } = normalizeFormulaCounts(rawCounts);`,
  'buildCrystalGraph elements.sort()'
);

// 4. buildPrototypeGraph: sort adjacency + edges before buildEdgeIndex
rep(
  `  const edgeIndex = buildEdgeIndex(nodes, edges);
  const globalFeatures = computeGlobalCompositionFeatures(rawCounts, hints, pressureGpa);
  const threeBodyFeatures = compute3BodyFeatures({ nodes, edges, threeBodyFeatures: [], adjacency, edgeIndex, formula, prototype, globalFeatures });
  return { nodes, edges, threeBodyFeatures, adjacency, edgeIndex, formula, prototype, pressureGpa, globalFeatures };`,
  `  // Normalize ordering for permutation invariance
  edges.sort((a, b) => a.source !== b.source ? a.source - b.source : a.target - b.target);
  for (let i = 0; i < adjacency.length; i++) adjacency[i].sort((a, b) => a - b);
  const edgeIndex = buildEdgeIndex(nodes, edges);
  const globalFeatures = computeGlobalCompositionFeatures(rawCounts, hints, pressureGpa);
  const threeBodyFeatures = compute3BodyFeatures({ nodes, edges, threeBodyFeatures: [], adjacency, edgeIndex, formula, prototype, globalFeatures });
  return { nodes, edges, threeBodyFeatures, adjacency, edgeIndex, formula, prototype, pressureGpa, globalFeatures };`,
  'buildPrototypeGraph sort edges+adjacency'
);

// 5. buildCrystalGraph: sort adjacency + edges before buildEdgeIndex
rep(
  `  const edgeIndex = buildEdgeIndex(nodes, edges);
  const globalFeatures = computeGlobalCompositionFeatures(rawCounts, enrichedHints, pressureGpa);
  const partialGraph: CrystalGraph = { nodes, edges, threeBodyFeatures: [], adjacency, edgeIndex, formula, pressureGpa, globalFeatures };
  partialGraph.threeBodyFeatures = compute3BodyFeatures(partialGraph);
  return partialGraph;`,
  `  // Normalize ordering for permutation invariance
  edges.sort((a, b) => a.source !== b.source ? a.source - b.source : a.target - b.target);
  for (let i = 0; i < adjacency.length; i++) adjacency[i].sort((a, b) => a - b);
  const edgeIndex = buildEdgeIndex(nodes, edges);
  const globalFeatures = computeGlobalCompositionFeatures(rawCounts, enrichedHints, pressureGpa);
  const partialGraph: CrystalGraph = { nodes, edges, threeBodyFeatures: [], adjacency, edgeIndex, formula, pressureGpa, globalFeatures };
  partialGraph.threeBodyFeatures = compute3BodyFeatures(partialGraph);
  return partialGraph;`,
  'buildCrystalGraph sort edges+adjacency'
);

writeFileSync(path, src, 'utf8');
console.log('Done. Checking...');

const checks = [
  'const elements = Object.keys(counts).sort();',
  'const elements = Object.keys(rawCounts).sort();',
  'edges.sort((a, b) => a.source !== b.source ? a.source - b.source : a.target - b.target);',
  'for (let i = 0; i < adjacency.length; i++) adjacency[i].sort((a, b) => a - b);',
];
let ok = true;
// Check each appears at least once (elements.sort appears twice — once per function)
for (const c of checks) {
  const count = src.split(c).length - 1;
  const found = count > 0;
  console.log((found ? `✓ (×${count}) ` : '✗ MISSING: ') + c);
  if (!found) ok = false;
}
process.exit(ok ? 0 : 1);
