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

// Insert invalidation block right before the closing `}` of the batch loop,
// which immediately follows the dense_skip_gate sub-block close.
rep(
`        weights.dense_skip_gate = Math.max(-4.6, Math.min(4.6, weights.dense_skip_gate));
      }
    }

    // Progress bar — log every ~20% of epochs, always log first and last.`,
`        weights.dense_skip_gate = Math.max(-4.6, Math.min(4.6, weights.dense_skip_gate));
      }
      // Invalidate flat matrix caches after Adam in-place mutations.
      // adamUpdate mutates w[i][j] in-place so the WeakMap-cached Float32Array goes stale;
      // without invalidation every forward pass from batch 2 onward reads pre-update weights.
      for (const wMat of [
        weights.W_mlp2, weights.W_mlp2_var, weights.W_mlp1, weights.W_cls1,
        weights.W_message,  weights.W_update,  weights.W_message2, weights.W_update2,
        weights.W_message3, weights.W_update3, weights.W_message4, weights.W_update4,
        weights.W_filter1, weights.W_filter2, weights.W_input_proj,
        weights.W_elem_feat, weights.W_graph_adapt,
      ]) { invalidateFlatCache(wMat); }
    }

    // Progress bar — log every ~20% of epochs, always log first and last.`,
'per-batch flat cache invalidation'
);

writeFileSync(path, src, 'utf8');
console.log('Done. Checking...');

const check = 'Invalidate flat matrix caches after Adam in-place mutations.';
const found = src.includes(check);
console.log((found ? '✓ ' : '✗ MISSING: ') + check);

// Verify it appears only ONCE (we only want one insertion, not duplicating the end-of-training one)
const count = src.split('invalidateFlatCache(wMat)').length - 1;
console.log(`invalidateFlatCache(wMat) appears ${count} times (expected 2: per-batch + end-of-training)`);

process.exit(found ? 0 : 1);
