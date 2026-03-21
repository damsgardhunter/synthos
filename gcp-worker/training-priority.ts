/**
 * Shared training priority coordination between GCP worker loops.
 *
 * When GNN major training is active (large ensemble on 14k+ samples):
 *   - XGB and ML loops skip new job claims until GNN finishes
 *   - DFT loop caps concurrent jobs at DFT_GNN_CONCURRENT (default 2)
 *     so running vc-relax jobs can finish but new ones don't compete
 *
 * A mutex (acquireGNNTrainingSlot) ensures only ONE GNN training runs at
 * a time — startup corpus training and dispatched jobs can't overlap.
 *
 * This gives GNN ~24 of the 32 vCPUs during training instead of competing
 * equally with 7 DFT jobs + XGB + ML for the same cores.
 */

let _gnnMajorTrainingActive = false;
let _xgbTrainingActive = false;

// How many DFT slots to allow while GNN is training (default 2).
export const DFT_GNN_CONCURRENT = parseInt(process.env.DFT_GNN_CONCURRENT ?? "2", 10);

// ── GNN training mutex ────────────────────────────────────────────────────────
// Prevents a dispatched job and the startup corpus training from both launching
// 5 worker threads simultaneously (which would pin all 32 vCPUs and slow both).

let _gnnTrainingSlotTaken = false;
let _gnnSlotBusyLastLogMs = 0;

/**
 * Try to acquire the exclusive GNN training slot.
 * Returns true if acquired (caller should train then call releaseGNNTrainingSlot).
 * Returns false if another GNN training is already running — caller should skip.
 */
export function acquireGNNTrainingSlot(label: string): boolean {
  if (_gnnTrainingSlotTaken) {
    const now = Date.now();
    if (now - _gnnSlotBusyLastLogMs >= 5 * 60 * 1000) {
      console.log(`[Priority] GNN training slot busy — ${label} will skip this cycle`);
      _gnnSlotBusyLastLogMs = now;
    }
    return false;
  }
  _gnnTrainingSlotTaken = true;
  _gnnMajorTrainingActive = true;
  console.log(`[Priority] GNN training slot acquired by ${label} — XGB/ML pausing, DFT capped at ${DFT_GNN_CONCURRENT} slots`);
  return true;
}

export function releaseGNNTrainingSlot(label: string): void {
  _gnnTrainingSlotTaken = false;
  _gnnMajorTrainingActive = false;
  console.log(`[Priority] GNN training slot released by ${label} — all loops resuming`);
}

export function isGNNMajorTrainingActive(): boolean {
  return _gnnMajorTrainingActive;
}

// ── XGB active flag ───────────────────────────────────────────────────────────

export function setXGBTrainingActive(active: boolean): void {
  _xgbTrainingActive = active;
}

export function isXGBTrainingActive(): boolean {
  return _xgbTrainingActive;
}

/**
 * Waits until XGB is not actively training (or until maxWaitMs elapses).
 * Called by GNN before acquiring the training slot so XGB finishes its current
 * job and frees CPU before GNN saturates all cores.
 */
export async function waitForXGBIdle(maxWaitMs = 10 * 60_000): Promise<void> {
  if (!_xgbTrainingActive) return;
  const deadline = Date.now() + maxWaitMs;
  console.log("[Priority] Waiting for XGB to finish current job before GNN major training...");
  while (_xgbTrainingActive && Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 3_000));
  }
  if (_xgbTrainingActive) {
    console.log("[Priority] XGB still active after timeout — proceeding with GNN training anyway");
  }
}
