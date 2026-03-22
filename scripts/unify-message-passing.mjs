/**
 * Consolidates 5 message-passing implementations into 2 unified functions
 * (attnMessagePass + cgcnnConv), each with a mode: 'train' | 'infer' parameter.
 *
 * Fixes addressed at the same time:
 *   - cgcnnConvCached was missing layerNorm(hRes) (train/infer mismatch bug)
 *   - cgcnnConvolutionLayer was missing layerNorm(hRes) (same bug)
 *   - messagePassingLayer was dead code — removed
 */
import { readFileSync, writeFileSync } from 'fs';

const path = 'server/learning/graph-neural-net.ts';
let src = readFileSync(path, 'utf8');
// Normalise to LF for all matching; convert back to CRLF at write time.
const hasCRLF = src.includes('\r\n');
if (hasCRLF) src = src.replace(/\r\n/g, '\n');

// ─────────────────────────────────────────────────────────────────────────────
// Helper: replace the first match of `from` with `to`.
// ─────────────────────────────────────────────────────────────────────────────
function rep(from, to, label) {
  if (src.includes(from)) { src = src.replace(from, to); return true; }
  console.error('FAILED:', label ?? JSON.stringify(from.slice(0, 80)));
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// The two new unified functions (already LF-normalised).
// ─────────────────────────────────────────────────────────────────────────────

const NEW_ATTN = `\
/**
 * Unified attention message-passing layer.
 * mode='train': deep-copies embeddings, populates cache for backward pass.
 * mode='infer': references embeddings directly, returns cache:null.
 * Dropout (msgRng) is applied in both modes — training uses it for
 * regularisation; inference uses it for MC-dropout uncertainty estimation.
 */
function attnMessagePass(
  graph: CrystalGraph,
  W_message: number[][], W_update: number[][],
  W_query: number[][], W_key: number[][],
  useLeakyMsg: boolean,
  mode: 'train' | 'infer',
  msgRng?: () => number,
): { embeddings: number[][]; cache: AttnLayerCache | null } {
  const nNodes = graph.nodes.length;
  // Deep-copy in train mode so backward pass has original pre-update inputs.
  const inputEmbs = mode === 'train'
    ? graph.nodes.map(n => [...n.embedding])
    : graph.nodes.map(n => n.embedding);

  // Cache arrays — only populated in train mode.
  const preActs: number[][] = [];
  const preNormActs: number[][] = [];
  const attnWts: number[][] = [];
  const neighborLists: number[][] = [];
  const newEmbeddings: number[][] = [];

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    if (mode === 'train') neighborLists.push([...neighbors]);
    if (neighbors.length === 0) {
      newEmbeddings.push([...inputEmbs[i]]);
      if (mode === 'train') {
        preActs.push([...inputEmbs[i]]);
        preNormActs.push([...inputEmbs[i]]);
        attnWts.push([]);
      }
      continue;
    }

    const query = layerNorm(matVecMul(W_query, inputEmbs[i]));
    const attentionScores: number[] = [];
    const messages: number[][] = [];

    for (const j of neighbors) {
      const key = layerNorm(matVecMul(W_key, inputEmbs[j]));
      const score = dotProduct(query, key);
      attentionScores.push(score + Math.log(graph.nodes[j].multiplicity ?? 1));
      messages.push(matVecMul(W_message, inputEmbs[j]));
    }

    // Message dropout (inverted-dropout scaling keeps expected value constant).
    if (msgRng) {
      const msgScale = 1.0 / (1.0 - MSG_DROPOUT_RATE);
      for (let n = 0; n < messages.length; n++) {
        if (msgRng() < MSG_DROPOUT_RATE) {
          messages[n] = new Array(HIDDEN_DIM).fill(0);
        } else {
          for (let k = 0; k < HIDDEN_DIM; k++) messages[n][k] *= msgScale;
        }
      }
    }

    const aw = softmax(attentionScores);
    if (mode === 'train') attnWts.push(aw);

    const aggMessage = initVector(HIDDEN_DIM);
    for (let n = 0; n < neighbors.length; n++) {
      const w = aw[n];
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggMessage[k] += (messages[n][k] ?? 0) * w;
      }
    }

    const combined = [...inputEmbs[i], ...aggMessage];
    const pre = matVecMul(W_update, combined);
    if (mode === 'train') preActs.push([...pre]);
    const hUpd = useLeakyMsg
      ? pre.map(v => v >= 0 ? v : 0.01 * v)
      : pre.map(v => v >= 0 ? v : 0);
    const hRes = inputEmbs[i].map((v, k) => v + (hUpd[k] ?? 0));
    if (mode === 'train') preNormActs.push(hRes);
    newEmbeddings.push(layerNorm(hRes));
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  const cache = mode === 'train'
    ? { inputEmbs, preActs, preNormActs, attnWts, neighborLists }
    : null;
  return { embeddings: newEmbeddings, cache };
}`;

const NEW_CGCNN = `\
/**
 * Unified SchNet-style continuous-filter convolution layer.
 * h_ij = W_filter2 @ silu(W_filter1 @ rbf_ij + b_filter1) + b_filter2
 * message = h_ij ⊙ h_j;  h_i' = layerNorm(h_i + mean_j(message * cutoff * mult_j))
 *
 * mode='train': deep-copies embeddings, populates cache for backward pass.
 * mode='infer': references embeddings directly, returns cache:null.
 */
function cgcnnConv(
  graph: CrystalGraph,
  W_filter1: number[][], b_filter1: number[],
  W_filter2: number[][], b_filter2: number[],
  adaptiveLogits: number[][] | undefined,
  mode: 'train' | 'infer',
): { embeddings: number[][]; cache: CGCNNLayerCache | null } {
  const nNodes = graph.nodes.length;
  // Deep-copy in train mode so backward pass has original pre-update inputs.
  const inputEmbs = mode === 'train'
    ? graph.nodes.map(n => [...n.embedding])
    : graph.nodes.map(n => n.embedding);

  // Cache arrays — only populated in train mode.
  const filterPreActs: number[][][] = [];
  const filterH1s: number[][][] = [];
  const filterOuts: number[][][] = [];
  const rbfs: number[][][] = [];
  const preNormActs: number[][] = [];
  const cutoffWts: number[][] = [];
  const totalWeights: number[] = [];
  const newEmbeddings: number[][] = [];

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    const nodeFilterPreActs: number[][] = [];
    const nodeFilterH1s: number[][] = [];
    const nodeFilterOuts: number[][] = [];
    const nodeRbfs: number[][] = [];
    const nodeCutoffs: number[] = [];

    if (!neighbors || neighbors.length === 0) {
      newEmbeddings.push([...inputEmbs[i]]);
      if (mode === 'train') {
        preNormActs.push([...inputEmbs[i]]);
        filterPreActs.push(nodeFilterPreActs);
        filterH1s.push(nodeFilterH1s);
        filterOuts.push(nodeFilterOuts);
        rbfs.push(nodeRbfs);
        cutoffWts.push(nodeCutoffs);
        totalWeights.push(0);
      }
      continue;
    }

    const aggUpdate = initVector(HIDDEN_DIM);
    let totalWeight = 0;

    for (let nIdx = 0; nIdx < neighbors.length; nIdx++) {
      const j = neighbors[nIdx];
      const edgeFeat = getEdgeFromIndex(graph.edgeIndex, nNodes, i, j);
      const distance = edgeFeat?.distance ?? 2.5;
      const cw = cosineCutoff(distance);
      if (mode === 'train') nodeCutoffs.push(cw);
      if (cw <= 0) {
        if (mode === 'train') {
          nodeFilterPreActs.push([]);
          nodeFilterH1s.push([]);
          nodeFilterOuts.push([]);
          nodeRbfs.push([]);
        }
        continue;
      }

      const edgeVec = edgeFeat?.features ?? initVector(EDGE_DIM);
      const rbf_ij: number[] = new Array(N_GAUSSIAN_BASIS);
      for (let k = 0; k < N_GAUSSIAN_BASIS; k++) rbf_ij[k] = edgeVec[k] ?? 0;

      const z_ij = vecAdd(matVecMul(W_filter1, rbf_ij), b_filter1);
      const h1_ij: number[] = new Array(HIDDEN_DIM);
      for (let k = 0; k < HIDDEN_DIM; k++) h1_ij[k] = silu(z_ij[k] ?? 0);
      const h_ij = vecAdd(matVecMul(W_filter2, h1_ij), b_filter2);

      const adaptLogit = adaptiveLogits?.[i]?.[nIdx] ?? 0;
      const hj = inputEmbs[j];
      const multJ = graph.nodes[j].multiplicity ?? 1;

      if (mode === 'train') {
        nodeFilterPreActs.push([...z_ij]);
        nodeFilterH1s.push([...h1_ij]);
        nodeFilterOuts.push([...h_ij]);
        nodeRbfs.push([...rbf_ij]);
      }

      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggUpdate[k] += (h_ij[k] + adaptLogit) * (hj[k] ?? 0) * cw * multJ;
      }
      totalWeight += cw * multJ;
    }

    if (mode === 'train') {
      filterPreActs.push(nodeFilterPreActs);
      filterH1s.push(nodeFilterH1s);
      filterOuts.push(nodeFilterOuts);
      rbfs.push(nodeRbfs);
      cutoffWts.push(nodeCutoffs);
      totalWeights.push(totalWeight);
    }

    if (totalWeight > 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) aggUpdate[k] /= totalWeight;
    }

    const hRes: number[] = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      hRes[k] = (inputEmbs[i][k] ?? 0) + aggUpdate[k];
    }
    if (mode === 'train') preNormActs.push(hRes);
    // LayerNorm after residual — same in both modes (fixes prior train/infer mismatch).
    newEmbeddings.push(layerNorm(hRes));
  }

  for (let i = 0; i < nNodes; i++) graph.nodes[i].embedding = newEmbeddings[i];

  const cache = mode === 'train'
    ? { inputEmbs, filterPreActs, filterH1s, filterOuts, rbfs, preNormActs, cutoffWts, totalWeights, adaptiveLogits }
    : null;
  return { embeddings: newEmbeddings, cache };
}`;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Replace the entire 5-function block with the two unified functions.
//     Block starts at the first SchNet doc comment and ends with cgcnnConvCached's }.
// ─────────────────────────────────────────────────────────────────────────────
const BLOCK_START = `\n/**\n * SchNet continuous filter convolution (inference, non-cached).\n * h_ij = W_filter2 @ silu(W_filter1 @ rbf_ij + b_filter1) + b_filter2\n * message = h_ij * h_j (element-wise);  h_i' = h_i + mean_j(message * cutoff)\n */`;
const BLOCK_END   = `\nfunction attnLayerBackward(`;

const startIdx = src.indexOf(BLOCK_START);
const endIdx   = src.indexOf(BLOCK_END);

if (startIdx === -1) { console.error('FAILED: could not find block START'); process.exit(1); }
if (endIdx   === -1) { console.error('FAILED: could not find block END');   process.exit(1); }

src = src.slice(0, startIdx)
  + '\n\n' + NEW_ATTN + '\n\n' + NEW_CGCNN + '\n'
  + src.slice(endIdx);

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Update inference call sites.
// ─────────────────────────────────────────────────────────────────────────────
const inferAttnPairs = [
  ['attentionMessagePassingLayer(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true, dropoutRng);',
   'attnMessagePass(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true, \'infer\', dropoutRng);'],
  ['attentionMessagePassingLayer(graph, weights.W_message2, weights.W_update2, weights.W_attn_query2, weights.W_attn_key2, false, dropoutRng);',
   'attnMessagePass(graph, weights.W_message2, weights.W_update2, weights.W_attn_query2, weights.W_attn_key2, false, \'infer\', dropoutRng);'],
  ['attentionMessagePassingLayer(graph, weights.W_message3, weights.W_update3, weights.W_attn_query3, weights.W_attn_key3, false, dropoutRng);',
   'attnMessagePass(graph, weights.W_message3, weights.W_update3, weights.W_attn_query3, weights.W_attn_key3, false, \'infer\', dropoutRng);'],
  ['attentionMessagePassingLayer(graph, weights.W_message4, weights.W_update4, weights.W_attn_query4, weights.W_attn_key4, false, dropoutRng);',
   'attnMessagePass(graph, weights.W_message4, weights.W_update4, weights.W_attn_query4, weights.W_attn_key4, false, \'infer\', dropoutRng);'],
];
for (const [from, to] of inferAttnPairs) rep(from, to, 'infer attn call site');

rep(
  'cgcnnConvolutionLayer(graph, weights.W_filter1, weights.b_filter1, weights.W_filter2, weights.b_filter2, adaptLogits);',
  'cgcnnConv(graph, weights.W_filter1, weights.b_filter1, weights.W_filter2, weights.b_filter2, adaptLogits, \'infer\');',
  'infer cgcnn call site'
);

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Update training call sites.
//     Cast return type so TypeScript knows cache is AttnLayerCache (not | null).
// ─────────────────────────────────────────────────────────────────────────────
const trainAttnPairs = [
  ['const { cache: ac0 } = attnMessagePassCached(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true, msgRng);',
   'const { cache: ac0 } = attnMessagePass(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true, \'train\', msgRng) as { embeddings: number[][]; cache: AttnLayerCache };'],
  ['const { cache: ac1 } = attnMessagePassCached(graph, weights.W_message2, weights.W_update2, weights.W_attn_query2, weights.W_attn_key2, false, msgRng);',
   'const { cache: ac1 } = attnMessagePass(graph, weights.W_message2, weights.W_update2, weights.W_attn_query2, weights.W_attn_key2, false, \'train\', msgRng) as { embeddings: number[][]; cache: AttnLayerCache };'],
  ['const { cache: ac2 } = attnMessagePassCached(graph, weights.W_message3, weights.W_update3, weights.W_attn_query3, weights.W_attn_key3, false, msgRng);',
   'const { cache: ac2 } = attnMessagePass(graph, weights.W_message3, weights.W_update3, weights.W_attn_query3, weights.W_attn_key3, false, \'train\', msgRng) as { embeddings: number[][]; cache: AttnLayerCache };'],
  ['const { cache: ac3 } = attnMessagePassCached(graph, weights.W_message4, weights.W_update4, weights.W_attn_query4, weights.W_attn_key4, false, msgRng);',
   'const { cache: ac3 } = attnMessagePass(graph, weights.W_message4, weights.W_update4, weights.W_attn_query4, weights.W_attn_key4, false, \'train\', msgRng) as { embeddings: number[][]; cache: AttnLayerCache };'],
];
for (const [from, to] of trainAttnPairs) rep(from, to, 'train attn call site');

rep(
  'const { cache: cgcnnC } = cgcnnConvCached(graph, weights.W_filter1, weights.b_filter1, weights.W_filter2, weights.b_filter2, adaptLogits);',
  'const { cache: cgcnnC } = cgcnnConv(graph, weights.W_filter1, weights.b_filter1, weights.W_filter2, weights.b_filter2, adaptLogits, \'train\') as { embeddings: number[][]; cache: CGCNNLayerCache };',
  'train cgcnn call site'
);

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Write out.
// ─────────────────────────────────────────────────────────────────────────────
if (hasCRLF) src = src.replace(/\n/g, '\r\n');
writeFileSync(path, src, 'utf8');
console.log('Done. Verifying...');

const checks = [
  // New functions present
  'function attnMessagePass(',
  'function cgcnnConv(',
  "mode: 'train' | 'infer'",
  // Old functions absent
  // (we check absence by confirming the defining `export function` strings are gone)
  // Call sites updated
  "attnMessagePass(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true, 'infer', dropoutRng)",
  "attnMessagePass(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true, 'train', msgRng)",
  "cgcnnConv(graph, weights.W_filter1, weights.b_filter1, weights.W_filter2, weights.b_filter2, adaptLogits, 'infer')",
  "cgcnnConv(graph, weights.W_filter1, weights.b_filter1, weights.W_filter2, weights.b_filter2, adaptLogits, 'train')",
  // layerNorm in CGCNN (previously missing)
  '// LayerNorm after residual — same in both modes',
  // Old functions gone
];

const absent = [
  'export function cgcnnConvolutionLayer(',
  'export function attentionMessagePassingLayer(',
  'export function messagePassingLayer(',
  'function attnMessagePassCached(',
  'function cgcnnConvCached(',
];

let ok = true;
for (const c of checks) {
  const count = src.split(c).length - 1;
  const found = count > 0;
  console.log((found ? `✓ (×${count}) ` : '✗ MISSING: ') + c);
  if (!found) ok = false;
}
for (const c of absent) {
  const count = src.split(c).length - 1;
  const gone = count === 0;
  console.log((gone ? '✓ removed: ' : `✗ STILL PRESENT (×${count}): `) + c);
  if (!gone) ok = false;
}
process.exit(ok ? 0 : 1);
