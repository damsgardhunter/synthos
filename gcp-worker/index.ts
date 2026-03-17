/**
 * Quantum Alchemy Engine — GCP Worker Entry Point
 *
 * Runs on the GCP instance (34.42.172.122, 32 vCPUs, 128 GB RAM).
 * Connects to the shared Neon DB and processes:
 *   - DFT jobs  (dft_jobs table)  — Quantum ESPRESSO calculations
 *   - GNN jobs  (gnn_training_jobs table) — full ensemble training
 *
 * Environment variables required (set in /etc/quantum-alchemy.env):
 *   DATABASE_URL          — Neon Postgres connection string (shared with local server)
 *   QE_BIN_DIR            — Path to QE binaries (default: /usr/bin)
 *   OMP_NUM_THREADS       — Threads per QE run (default: 8 for 4 concurrent × 8 = 32 vCPUs)
 *   PSEUDO_DIR            — Pseudopotential directory
 *   ENABLE_DFT_WORKER     — "true" to run DFT loop (default: true)
 *   ENABLE_GNN_WORKER     — "true" to run GNN loop (default: true)
 *
 * xTB setup on GCP (xtb-dist/ is gitignored — install system xTB instead):
 *   sudo apt-get install -y xtb          # or: conda install -c conda-forge xtb
 *   Then add to /etc/quantum-alchemy.env:
 *   XTB_BIN=/usr/bin/xtb
 *   XTBHOME=/usr/share/xtb
 *   XTBPATH=/usr/share/xtb
 */

import { startDFTLoop, stopDFTLoop } from "./dft-loop";
import { startGNNLoop, stopGNNLoop } from "./gnn-loop";
import { startXGBLoop, stopXGBLoop } from "./xgb-loop";
import { startMLLoop, stopMLLoop } from "./ml-loop";

// Default OMP_NUM_THREADS: 4 concurrent QE jobs × 8 threads each = 32 vCPUs
if (!process.env.OMP_NUM_THREADS) {
  process.env.OMP_NUM_THREADS = "8";
}

const ENABLE_DFT = process.env.ENABLE_DFT_WORKER !== "false";
const ENABLE_GNN = process.env.ENABLE_GNN_WORKER !== "false";
const ENABLE_XGB = process.env.ENABLE_XGB_WORKER !== "false";
const ENABLE_ML  = process.env.ENABLE_ML_WORKER  !== "false";

console.log("=".repeat(60));
console.log("  Quantum Alchemy Engine — GCP Worker");
console.log(`  DFT loop : ${ENABLE_DFT ? "ENABLED" : "disabled"}`);
console.log(`  GNN loop : ${ENABLE_GNN ? "ENABLED" : "disabled"}`);
console.log(`  XGB loop : ${ENABLE_XGB ? "ENABLED" : "disabled"}`);
console.log(`  ML  loop : ${ENABLE_ML  ? "ENABLED" : "disabled"}`);
console.log(`  OMP_NUM_THREADS = ${process.env.OMP_NUM_THREADS}`);
console.log(`  QE_BIN_DIR      = ${process.env.QE_BIN_DIR ?? "(auto)"}`);
console.log("=".repeat(60));

// Stagger loop starts to avoid saturating the Neon connection pool.
// All loops fire their first DB poll at startup; launching them simultaneously
// exhausts pg-pool and causes ETIMEDOUT errors on the initial poll.
// GNN goes first (it has a warmup phase), others are delayed so they start
// after the pool has settled.
function delay(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

const loops: Promise<void>[] = [];

if (ENABLE_GNN) loops.push(startGNNLoop());
if (ENABLE_DFT) loops.push(delay(3_000).then(() => startDFTLoop()));
if (ENABLE_XGB) loops.push(delay(6_000).then(() => startXGBLoop()));
if (ENABLE_ML)  loops.push(delay(9_000).then(() => startMLLoop()));

if (loops.length === 0) {
  console.error("No workers enabled — set ENABLE_DFT_WORKER=true or ENABLE_GNN_WORKER=true");
  process.exit(1);
}

function shutdown(signal: string) {
  console.log(`\n[GCP-Worker] Received ${signal} — shutting down gracefully...`);
  stopDFTLoop();
  stopGNNLoop();
  stopXGBLoop();
  stopMLLoop();
  // Allow in-flight jobs ~30s to finish
  setTimeout(() => {
    console.log("[GCP-Worker] Shutdown complete");
    process.exit(0);
  }, 30_000);
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT",  () => shutdown("SIGINT"));

Promise.all(loops).catch(err => {
  console.error("[GCP-Worker] Fatal error:", err);
  process.exit(1);
});
