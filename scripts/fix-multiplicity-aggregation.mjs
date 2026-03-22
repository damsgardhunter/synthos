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

// ── 1. attentionMessagePassingLayer (inference): bias score by log(mult_j) ────
rep(
`      attentionScores.push(score);
      messages.push(matVecMul(W_message, embeddings[j]));
    }

    const attentionWeights = softmax(attentionScores);`,
`      // Bias attention by log(multiplicity): higher-multiplicity neighbors aggregate more
      attentionScores.push(score + Math.log(graph.nodes[j].multiplicity ?? 1));
      messages.push(matVecMul(W_message, embeddings[j]));
    }

    const attentionWeights = softmax(attentionScores);`,
'attentionMessagePassingLayer score+log(mult)'
);

// ── 2. attnMessagePassCached (training): bias score by log(mult_j) ─────────
rep(
`      attentionScores.push(score);
      messages.push(matVecMul(W_message, inputEmbs[j]));
    }

    const aw = softmax(attentionScores);`,
`      // Bias attention by log(multiplicity): higher-multiplicity neighbors aggregate more
      attentionScores.push(score + Math.log(graph.nodes[j].multiplicity ?? 1));
      messages.push(matVecMul(W_message, inputEmbs[j]));
    }

    const aw = softmax(attentionScores);`,
'attnMessagePassCached score+log(mult)'
);

// ── 3. cgcnnConvolutionLayer (inference): scale message and totalWeight by mult_j ──
rep(
`      const hj = embeddings[j];
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggUpdate[k] += (h_ij[k] ?? 0) * (hj[k] ?? 0) * cw;
      }
      totalWeight += cw;`,
`      const hj = embeddings[j];
      const multJ = graph.nodes[j].multiplicity ?? 1;
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggUpdate[k] += (h_ij[k] ?? 0) * (hj[k] ?? 0) * cw * multJ;
      }
      totalWeight += cw * multJ;`,
'cgcnnConvolutionLayer mult scale'
);

// ── 4. cgcnnConvCached (training): scale message and totalWeight by mult_j ───
rep(
`      const hj = inputEmbs[j];

      nodeFilterPreActs.push([...z_ij]);
      nodeFilterH1s.push([...h1_ij]);
      nodeFilterOuts.push([...h_ij]);
      nodeRbfs.push([...rbf_ij]);

      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggUpdate[k] += (h_ij[k] + adaptLogit) * (hj[k] ?? 0) * cw;
      }
      totalWeight += cw;`,
`      const hj = inputEmbs[j];
      const multJ = graph.nodes[j].multiplicity ?? 1;

      nodeFilterPreActs.push([...z_ij]);
      nodeFilterH1s.push([...h1_ij]);
      nodeFilterOuts.push([...h_ij]);
      nodeRbfs.push([...rbf_ij]);

      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggUpdate[k] += (h_ij[k] + adaptLogit) * (hj[k] ?? 0) * cw * multJ;
      }
      totalWeight += cw * multJ;`,
'cgcnnConvCached mult scale'
);

// ── 5. messagePassingLayer (basic): replace uniform 1/nCount with mult-weighted mean ──
rep(
`    const aggMessage = initVector(HIDDEN_DIM);
    const nCount = Math.max(1, neighbors.length);
    for (const j of neighbors) {
      const msg = matVecMul(W_message, embeddings[j]);
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggMessage[k] += msg[k] / nCount;
      }
    }`,
`    const aggMessage = initVector(HIDDEN_DIM);
    let totalMult = 0;
    for (const j of neighbors) totalMult += (graph.nodes[j].multiplicity ?? 1);
    const denom = Math.max(1, totalMult);
    for (const j of neighbors) {
      const msg = matVecMul(W_message, embeddings[j]);
      const mw = (graph.nodes[j].multiplicity ?? 1) / denom;
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggMessage[k] += msg[k] * mw;
      }
    }`,
'messagePassingLayer mult-weighted mean'
);

writeFileSync(path, src, 'utf8');
console.log('Done. Checking...');

const checks = [
  'attentionScores.push(score + Math.log(graph.nodes[j].multiplicity ?? 1));',
  'const multJ = graph.nodes[j].multiplicity ?? 1;',
  'totalWeight += cw * multJ;',
  'totalMult += (graph.nodes[j].multiplicity ?? 1)',
  'const mw = (graph.nodes[j].multiplicity ?? 1) / denom;',
];
let ok = true;
for (const c of checks) {
  const count = src.split(c).length - 1;
  const found = count > 0;
  console.log((found ? `✓ (×${count}) ` : '✗ MISSING: ') + c);
  if (!found) ok = false;
}
process.exit(ok ? 0 : 1);
