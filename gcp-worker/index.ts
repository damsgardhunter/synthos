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
import { startSuperConIngestion } from "../server/learning/supercon-db-ingestion";
import { startJarvisIngestion } from "../server/learning/jarvis-ingestion";
import { startThreeDSCIngestion } from "../server/learning/threedsc-ingestion";
import { startCODCachePopulation, printSpaceGroupCoverage } from "../server/learning/space-group-explorer";

// Default OMP_NUM_THREADS: 7 concurrent QE jobs × 4 threads = 28 vCPUs (main instance).
// On a dedicated DFT-only instance, set OMP_NUM_THREADS=3 and DFT_MAX_CONCURRENT=5.
// QE small systems (≤16 atoms) don't scale past 4 OMP threads — beyond that
// MPI/OpenMP overhead dominates, so halving threads doubles slot count for free.
if (!process.env.OMP_NUM_THREADS) {
  process.env.OMP_NUM_THREADS = "4";
}

// Suppress the graph-neural-net.ts local startup (Phases 1-3 trained on SUPERCON seed data).
// The GCP worker runs its own full-corpus startup via runStartupFullCorpusTraining() in
// gnn-loop.ts — the local startup would waste ~10 min and saturate the DB pool.
// This must be set before the isMainThread setTimeout in graph-neural-net.ts fires.
process.env.OFFLOAD_GNN_TO_GCP = "true";

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

// Print space group coverage at startup (informational only — no I/O).
printSpaceGroupCoverage();

// Start background data ingestion tasks (non-blocking — fire-and-forget).
// These run at lowest priority, yielding every 80 ms between batches.
// SuperCon: ingests ~33k entries from local CSV / NIMS API / Hamidieh fallback.
// JARVIS:   ingests supercon_chem (16k), supercon_3d/2d (1.2k), dft3d metallic negatives.
// COD:      pre-populates structural data cache for high-relevance space groups.
delay(15_000).then(() => {
  startSuperConIngestion();
  startCODCachePopulation(5);  // SGs with relevance score ≥ 5
});
// JARVIS starts 45 s after SuperCon to avoid DB connection contention on startup.
delay(60_000).then(() => {
  startJarvisIngestion();
});
// 3DSC_MP starts 90 s after SuperCon (staggered behind JARVIS).
delay(105_000).then(() => {
  startThreeDSCIngestion();
});

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
