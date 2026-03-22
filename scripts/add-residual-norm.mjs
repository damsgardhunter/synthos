import { readFileSync, writeFileSync } from 'fs';

const path = 'server/learning/graph-neural-net.ts';
let src = readFileSync(path, 'utf8');

// Helper: replace with CRLF-aware matching
function rep(from, to) {
  // Try literal first
  if (src.includes(from)) { src = src.replace(from, to); return true; }
  // Try normalizing CRLF → LF in source temporarily
  const srcLF = src.replace(/\r\n/g, '\n');
  const fromLF = from.replace(/\r\n/g, '\n');
  if (srcLF.includes(fromLF)) {
    // Replace in CRLF source by rebuilding the CRLF version of `to`
    const toLF = to.replace(/\r\n/g, '\n');
    const replaced = srcLF.replace(fromLF, toLF);
    src = replaced.replace(/\n/g, '\r\n');
    return true;
  }
  console.error('FAILED to replace:', JSON.stringify(from.slice(0, 80)));
  return false;
}

// ── 1. Add layerNormBackward just before layerNorm ──────────────────────────
const lnBackward = `/** Backward pass through LayerNorm (no learned scale/bias).
 *  preNorm: the pre-norm input x (same as vec passed to layerNorm).
 *  dLdY:    upstream gradient ∂L/∂y where y = (x − μ) / σ.
 *  Returns: ∂L/∂x = (1/σ) * [dL/dy − mean(dL/dy) − ŷ·mean(dL/dy·ŷ)]
 */
function layerNormBackward(preNorm: number[], dLdY: number[], eps: number = 1e-5): number[] {
  const n = preNorm.length;
  if (n === 0) return [];
  let mean = 0;
  for (let i = 0; i < n; i++) mean += preNorm[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) { const d = preNorm[i] - mean; variance += d * d; }
  variance /= n;
  const std = Math.sqrt(variance + eps);
  let sumDy = 0, sumDyY = 0;
  for (let i = 0; i < n; i++) {
    const yHat = (preNorm[i] - mean) / std;
    sumDy  += dLdY[i];
    sumDyY += dLdY[i] * yHat;
  }
  const dLdX = new Array(n);
  for (let i = 0; i < n; i++) {
    const yHat = (preNorm[i] - mean) / std;
    dLdX[i] = (dLdY[i] - sumDy / n - yHat * sumDyY / n) / std;
  }
  return dLdX;
}

`;

if (!src.includes('function layerNormBackward')) {
  rep(
    'function layerNorm(vec: number[], eps: number = 1e-5): number[] {',
    lnBackward + 'function layerNorm(vec: number[], eps: number = 1e-5): number[] {'
  );
}

// ── 2. AttnLayerCache: add preNormActs field ─────────────────────────────────
rep(
`interface AttnLayerCache {
  inputEmbs: number[][];
  preActs: number[][];
  attnWts: number[][];
  neighborLists: number[][];
}`,
`interface AttnLayerCache {
  inputEmbs: number[][];
  preActs: number[][];
  preNormActs: number[][];  // h + activation(update) before layerNorm
  attnWts: number[][];
  neighborLists: number[][];
}`
);

// ── 3. CGCNNLayerCache: add preNormActs field ─────────────────────────────────
rep(
`  cutoffWts: number[][];
  totalWeights: number[];`,
`  preNormActs: number[][];  // h + aggUpdate before layerNorm
  cutoffWts: number[][];
  totalWeights: number[];`
);

// ── 4. attnMessagePassCached: add preNormActs array declaration ───────────────
rep(
`  const preActs: number[][] = [];
  const attnWts: number[][] = [];
  const neighborLists: number[][] = [];
  const newEmbeddings: number[][] = [];`,
`  const preActs: number[][] = [];
  const preNormActs: number[][] = [];
  const attnWts: number[][] = [];
  const neighborLists: number[][] = [];
  const newEmbeddings: number[][] = [];`
);

// isolated-node path in attnMessagePassCached (has preActs.push and attnWts.push)
rep(
`    if (neighbors.length === 0) {
      newEmbeddings.push([...inputEmbs[i]]);
      preActs.push([...inputEmbs[i]]);
      attnWts.push([]);
      continue;
    }`,
`    if (neighbors.length === 0) {
      newEmbeddings.push([...inputEmbs[i]]);
      preActs.push([...inputEmbs[i]]);
      preNormActs.push([...inputEmbs[i]]);
      attnWts.push([]);
      continue;
    }`
);

// end of node loop + return in attnMessagePassCached
rep(
`    const combined = [...inputEmbs[i], ...aggMessage];
    const pre = matVecMul(W_update, combined);
    preActs.push([...pre]);
    if (useLeakyMsg) {
      newEmbeddings.push(pre.map(v => v >= 0 ? v : 0.01 * v));
    } else {
      newEmbeddings.push(pre.map(v => v >= 0 ? v : 0));
    }
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return { embeddings: newEmbeddings, cache: { inputEmbs, preActs, attnWts, neighborLists } };`,
`    const combined = [...inputEmbs[i], ...aggMessage];
    const pre = matVecMul(W_update, combined);
    preActs.push([...pre]);
    const hUpd = useLeakyMsg
      ? pre.map(v => v >= 0 ? v : 0.01 * v)
      : pre.map(v => v >= 0 ? v : 0);
    // Residual: h = input + activation(update)
    const hRes = inputEmbs[i].map((v, k) => v + (hUpd[k] ?? 0));
    preNormActs.push(hRes);
    // LayerNorm after residual
    newEmbeddings.push(layerNorm(hRes));
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return { embeddings: newEmbeddings, cache: { inputEmbs, preActs, preNormActs, attnWts, neighborLists } };`
);

// ── 5. attentionMessagePassingLayer inference: residual + layerNorm ───────────
rep(
`    const combined = [...embeddings[i], ...aggMessage];
    const updated = fusedUpdate(W_update, combined);
    // No layerNorm here — the training path (attnMessagePassCached) does not apply
    // layerNorm, so applying it at inference would create a train/inference mismatch
    // that causes validation R² to be far worse than training R².
    // Activation explosion is prevented by the W_update He-initialization and the
    // residual gates (sigmoid-gated skip connections) that bound the norm growth.
    newEmbeddings.push(updated);`,
`    const combined = [...embeddings[i], ...aggMessage];
    const hUpd = fusedUpdate(W_update, combined);
    // Residual + LayerNorm — consistent with training path (attnMessagePassCached).
    const hRes = embeddings[i].map((v, k) => v + (hUpd[k] ?? 0));
    newEmbeddings.push(layerNorm(hRes));`
);

// ── 6. attnLayerBackward: prepend layerNorm backward + residual gradient ──────
rep(
`    const neighbors = cache.neighborLists[i];
    if (!neighbors || neighbors.length === 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) dLdInput[i][k] += dLdOutput[i][k];
      continue;
    }

    const pre = cache.preActs[i];
    const dPre = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      const mask = useLeaky ? (pre[k] >= 0 ? 1.0 : 0.01) : (pre[k] >= 0 ? 1.0 : 0.0);
      dPre[k] = dLdOutput[i][k] * mask;
    }`,
`    const neighbors = cache.neighborLists[i];
    if (!neighbors || neighbors.length === 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) dLdInput[i][k] += dLdOutput[i][k];
      continue;
    }

    // Backprop through layerNorm(h_res): upstream dLdOutput → dLd(h_res)
    const dLdPreNorm = cache.preNormActs?.[i]
      ? layerNormBackward(cache.preNormActs[i], dLdOutput[i])
      : [...dLdOutput[i]];
    // Residual pass-through: h_res = input + hUpd → ∂L/∂input += dLdPreNorm
    for (let k = 0; k < HIDDEN_DIM; k++) dLdInput[i][k] += dLdPreNorm[k];

    const pre = cache.preActs[i];
    const dPre = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      const mask = useLeaky ? (pre[k] >= 0 ? 1.0 : 0.01) : (pre[k] >= 0 ? 1.0 : 0.0);
      dPre[k] = dLdPreNorm[k] * mask;  // gradient flows through activation from dLdPreNorm
    }`
);

// ── 7. cgcnnConvCached: add preNormActs declaration ───────────────────────────
rep(
`  const gateRaws: number[][][] = [];
  const valueRaws: number[][][] = [];
  const concats: number[][][] = [];
  const cutoffWts: number[][] = [];
  const totalWeights: number[] = [];
  const newEmbeddings: number[][] = [];`,
`  const gateRaws: number[][][] = [];
  const valueRaws: number[][][] = [];
  const concats: number[][][] = [];
  const preNormActs: number[][] = [];
  const cutoffWts: number[][] = [];
  const totalWeights: number[] = [];
  const newEmbeddings: number[][] = [];`
);

// isolated-node path in cgcnnConvCached
rep(
`    if (!neighbors || neighbors.length === 0) {
      newEmbeddings.push([...inputEmbs[i]]);
      gateRaws.push(nodeGates);
      valueRaws.push(nodeValues);
      concats.push(nodeConcats);
      cutoffWts.push(nodeCutoffs);
      totalWeights.push(0);
      continue;
    }`,
`    if (!neighbors || neighbors.length === 0) {
      newEmbeddings.push([...inputEmbs[i]]);
      preNormActs.push([...inputEmbs[i]]);
      gateRaws.push(nodeGates);
      valueRaws.push(nodeValues);
      concats.push(nodeConcats);
      cutoffWts.push(nodeCutoffs);
      totalWeights.push(0);
      continue;
    }`
);

// end of cgcnnConvCached node loop + return
rep(
`    const updated: number[] = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      updated[k] = (inputEmbs[i][k] ?? 0) + aggUpdate[k];
    }
    newEmbeddings.push(updated);
  }

  for (let i = 0; i < nNodes; i++) graph.nodes[i].embedding = newEmbeddings[i];

  return {
    embeddings: newEmbeddings,
    cache: { inputEmbs, gateRaws, valueRaws, concats, cutoffWts, totalWeights, adaptiveLogits },
  };`,
`    const hRes: number[] = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      hRes[k] = (inputEmbs[i][k] ?? 0) + aggUpdate[k];
    }
    preNormActs.push(hRes);
    // LayerNorm after residual — consistent with cgcnnConvolutionLayer inference path
    newEmbeddings.push(layerNorm(hRes));
  }

  for (let i = 0; i < nNodes; i++) graph.nodes[i].embedding = newEmbeddings[i];

  return {
    embeddings: newEmbeddings,
    cache: { inputEmbs, gateRaws, valueRaws, concats, preNormActs, cutoffWts, totalWeights, adaptiveLogits },
  };`
);

// ── 8. cgcnnConvolutionLayer inference: layerNorm after existing residual ──────
rep(
`    const updated: number[] = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      updated[k] = (embeddings[i][k] ?? 0) + aggUpdate[k];
    }
    newEmbeddings.push(updated);
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return newEmbeddings;
}`,
`    const hRes: number[] = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      hRes[k] = (embeddings[i][k] ?? 0) + aggUpdate[k];
    }
    // LayerNorm after residual — consistent with cgcnnConvCached training path
    newEmbeddings.push(layerNorm(hRes));
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return newEmbeddings;
}`
);

// ── 9. cgcnnLayerBackward: prepend layerNorm backward ─────────────────────────
rep(
`  for (let i = 0; i < nNodes; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      dLdInput[i][k] += dLdOutput[i][k];
    }

    const totalW = cache.totalWeights[i];`,
`  // Pre-compute layerNorm-backward per node, then add residual pass-through.
  const dLdEff: number[][] = Array.from({ length: nNodes }, (_, i) => {
    const pn = (cache as any).preNormActs?.[i] as number[] | undefined;
    return pn ? layerNormBackward(pn, dLdOutput[i]) : [...dLdOutput[i]];
  });

  for (let i = 0; i < nNodes; i++) {
    // Residual pass-through: ∂L/∂input += dLdEff[i] (h_res = input + aggUpdate)
    for (let k = 0; k < HIDDEN_DIM; k++) dLdInput[i][k] += dLdEff[i][k];

    const totalW = cache.totalWeights[i];`
);

// Replace dLdOutput[i][k] → dLdEff[i][k] inside the per-edge loop (if not already done)
if (src.includes('const dAgg = dLdOutput[i][k] / totalW;')) {
  src = src.replace('        const dAgg = dLdOutput[i][k] / totalW;', '        const dAgg = dLdEff[i][k] / totalW;');
}

writeFileSync(path, src, 'utf8');
console.log('Done. Checking replacements...');

const checks = [
  'function layerNormBackward',
  'preNormActs: number[][];  // h + activation',
  'preNormActs: number[][];  // h + aggUpdate',
  'preNormActs.push(hRes)',
  'layerNorm(hRes)',
  'layerNormBackward(cache.preNormActs[i]',
  'const dLdEff: number[][]',
  'const dAgg = dLdEff[i][k] / totalW',
  'preNormActs, attnWts, neighborLists',
  'preNormActs, cutoffWts',
];
let ok = true;
for (const c of checks) {
  const found = src.includes(c);
  console.log((found ? '✓ ' : '✗ MISSING: ') + c);
  if (!found) ok = false;
}
process.exit(ok ? 0 : 1);
