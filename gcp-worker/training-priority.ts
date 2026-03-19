/**
 * Shared training priority coordination between GCP worker loops.
 *
 * When GNN major training is active (large ensemble on 14k+ samples):
 *   - XGB and ML loops skip new job claims until GNN finishes
 *   - DFT loop caps concurrent jobs at DFT_GNN_CONCURRENT (default 2)
 *     so running vc-relax jobs can finish but new ones don't compete
 *
 * This gives GNN ~24 of the 32 vCPUs during training instead of competing
 * equally with 7 DFT jobs + XGB + ML for the same cores.
 */

let _gnnMajorTrainingActive = false;
let _xgbTrainingActive = false;

// How many DFT slots to allow while GNN is training (default 2).
export const DFT_GNN_CONCURRENT = parseInt(process.env.DFT_GNN_CONCURRENT ?? "2", 10);

export function setGNNMajorTrainingActive(active: boolean): void {
  _gnnMajorTrainingActive = active;
  if (active) {
    console.log("[Priority] GNN major training started — XGB/ML pausing, DFT capped at " + DFT_GNN_CONCURRENT + " slots");
  } else {
    console.log("[Priority] GNN major training complete — all loops resuming normal operation");
  }
}

export function isGNNMajorTrainingActive(): boolean {
  return _gnnMajorTrainingActive;
}

export function setXGBTrainingActive(active: boolean): void {
  _xgbTrainingActive = active;
}

export function isXGBTrainingActive(): boolean {
  return _xgbTrainingActive;
}

/**
 * Waits until XGB is not actively training (or until maxWaitMs elapses).
 * Called by GNN before launching its worker threads so XGB gets to finish
 * its current job and free CPU before GNN saturates all cores.
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
