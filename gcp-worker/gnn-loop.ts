/**
 * GCP GNN Training Loop
 * Polls gnn_training_jobs for queued jobs, trains the full GNN ensemble,
 * writes weights back, then fetches the next batch of MP data so each
 * subsequent training cycle has a progressively richer dataset.
 *
 * On the dedicated GNN GCP instance all 5 ensemble members train in parallel
 * via worker_threads (trainEnsembleParallel), cutting wall time by ~5×.
 * Falls back to sequential trainEnsemble if any worker thread fails.
 */
import { db, isConnectionError } from "../server/db";
import { storage } from "../server/storage";
import { acquireGNNTrainingSlot, releaseGNNTrainingSlot, waitForXGBIdle, isGNNTrainingSlotFree } from "./training-priority";
import { fileURLToPath } from "url";
import path from "path";
import { spawn, ChildProcess } from "child_process";
import {
  ENSEMBLE_SIZE,
  splitTrainValidation,
} from "../server/learning/graph-neural-net";
import type { TrainingSample } from "../server/learning/graph-neural-net";
import { fetchMPBatchFromAPI } from "../server/learning/materials-project-client";

// ── Python GNN service (FastAPI on localhost:8765) ────────────────────────────
const GNN_SERVICE_URL  = process.env.GNN_SERVICE_URL ?? "http://127.0.0.1:8765";
const GNN_SERVICE_PORT = process.env.GNN_SERVICE_PORT ?? "8765";
const GNN_PY_SCRIPT    = path.join(path.dirname(fileURLToPath(import.meta.url)), "../gnn/server.py");

let _pythonProcess: ChildProcess | null = null;

/** Spawn the FastAPI GNN service and wait until it responds to /health. */
async function spawnPythonGNNService(): Promise<void> {
  if (_pythonProcess && !_pythonProcess.killed) return; // already running

  const python = process.env.PYTHON_BIN ?? "python3";
  console.log(`[GNN-GCP] Spawning Python GNN service: ${python} ${GNN_PY_SCRIPT}`);

  _pythonProcess = spawn(python, [GNN_PY_SCRIPT], {
    env: {
      ...process.env,
      GNN_SERVICE_PORT,
      GNN_WEIGHTS_DIR: process.env.GNN_WEIGHTS_DIR ?? "/opt/qae/gnn_weights",
      GNN_LOG_LEVEL: "INFO",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  _pythonProcess.stdout?.on("data", (d: Buffer) => process.stdout.write(`[py-gnn] ${d}`));
  _pythonProcess.stderr?.on("data", (d: Buffer) => process.stderr.write(`[py-gnn] ${d}`));
  _pythonProcess.on("exit", (code) => {
    console.warn(`[GNN-GCP] Python GNN service exited with code ${code}`);
    _pythonProcess = null;
  });

  // Wait up to 90 s for the service to start accepting requests
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 2000));
    try {
      const res = await fetch(`${GNN_SERVICE_URL}/health`);
      if (res.ok) {
        const body = await res.json() as any;
        console.log(`[GNN-GCP] Python GNN service ready — device=${body.device} models_loaded=${body.models_loaded}`);
        return;
      }
    } catch { /* not ready yet */ }
  }
  throw new Error("Python GNN service did not become ready within 90s");
}

/** POST training data to the Python service, returns parsed response. */
async function callPythonTrain(
  jobId:             number,
  trainingData:      TrainingSample[],
  startupValR2?:     number,
  maxPretrainEpochs: number = 15,
): Promise<{
  status: "done" | "discarded" | "failed";
  reason?: string;
  r2: number; mae: number; rmse: number;
  trainR2: number; trainMae: number;
  valN: number;
  ci95Coverage: number; ci95Width: number;
  wallSeconds: number;
  modelPath?: string;
  nSamples: number;
}> {
  const body = {
    job_id:              jobId,
    training_data:       trainingData.map(s => ({
      ...s,
      lambda: (s as any).lambda,   // preserve field name for Python
    })),
    max_pretrain_epochs: maxPretrainEpochs,
    startup_val_r2:      startupValR2 ?? null,
  };

  const res = await fetch(`${GNN_SERVICE_URL}/train`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
    signal:  AbortSignal.timeout(60 * 60_000), // 60-min timeout for large corpus
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "(no body)");
    throw new Error(`Python /train failed ${res.status}: ${text.slice(0, 200)}`);
  }

  const data = await res.json() as any;
  const m    = data.metrics ?? {};
  return {
    status:       data.status,
    reason:       data.reason,
    r2:           m.r2        ?? 0,
    mae:          m.mae       ?? 0,
    rmse:         m.rmse      ?? 0,
    trainR2:      m.train_r2  ?? 0,
    trainMae:     m.train_mae ?? 0,
    valN:         m.val_n     ?? 0,
    ci95Coverage: m.ci95_coverage ?? 0,
    ci95Width:    m.ci95_width    ?? 0,
    wallSeconds:  m.wall_seconds  ?? 0,
    modelPath:    m.model_path,
    nSamples:     m.n_samples ?? trainingData.length,
  };
}

/** Store the PyTorch model path in system_state for observability. */
async function recordPytorchModelPath(jobId: number, modelPath: string): Promise<void> {
  try {
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('gnn_pytorch_model_path', '{"jobId":${jobId},"path":${JSON.stringify(modelPath)}}', NOW())
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()`
    );
  } catch { /* non-fatal */ }
}

// trainEnsembleParallel and spawnModelWorker removed —
// training is now delegated to the Python FastAPI service (gnn/server.py)
// via callPythonTrain() defined above.

const POLL_INTERVAL_MS = 10_000;
const MP_BATCH_SIZE = 500;          // records per progressive fetch
const MP_MAX_CACHE = 10_000;        // stop fetching after this many cached records
const MP_SKIP_STATE_KEY = "mp_batch_skip";
let running = true;

// In-memory skip counter — primary source of truth. DB write is best-effort.
let _mpSkip = 0;
let _mpSkipLoaded = false;
// Mutex: prevents concurrent GNN job completions from each firing a duplicate MP fetch.
let _mpFetchInProgress = false;

// ── MP progressive fetch ─────────────────────────────────────────────────────

async function getMPSkip(): Promise<number> {
  if (_mpSkipLoaded) return _mpSkip;
  try {
    const rows = await db.execute(
      `SELECT value FROM system_state WHERE key = '${MP_SKIP_STATE_KEY}'`
    );
    const row = (rows as any).rows?.[0] ?? (Array.isArray(rows) ? rows[0] : undefined);
    _mpSkip = Number((row?.value as any)?.skip ?? 0);
  } catch {
    // DB not yet available — start from 0
  }
  _mpSkipLoaded = true;
  return _mpSkip;
}

async function setMPSkip(skip: number): Promise<void> {
  _mpSkip = skip; // always update in-memory immediately
  try {
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('${MP_SKIP_STATE_KEY}', '{"skip":${skip}}', NOW())
       ON CONFLICT (key) DO UPDATE SET value = '{"skip":${skip}}', updated_at = NOW()`
    );
  } catch (err: any) {
    console.warn(`[GNN-GCP] setMPSkip DB write failed (using in-memory ${skip}): ${err.message?.slice(0,80)}`);
  }
}

async function getCachedMPCount(): Promise<number> {
  try {
    const rows = await db.execute(
      `SELECT COUNT(*)::int AS count FROM mp_material_cache WHERE data_type = 'summary'`
    );
    const row = (rows as any).rows?.[0] ?? (Array.isArray(rows) ? rows[0] : undefined);
    return Number(row?.count ?? 0);
  } catch {
    return 0;
  }
}

async function fetchNextMPBatch(): Promise<void> {
  // Mutex: if a fetch is already in progress (e.g. from a concurrent GNN job completion), skip.
  if (_mpFetchInProgress) {
    console.log(`[GNN-GCP] MP fetch already in progress — skipping duplicate`);
    return;
  }
  _mpFetchInProgress = true;
  try {
    const cached = await getCachedMPCount();
    if (cached >= MP_MAX_CACHE) {
      console.log(`[GNN-GCP] MP cache full (${cached} records) — skipping fetch`);
      return;
    }

    const skip = await getMPSkip();
    console.log(`[GNN-GCP] Fetching MP batch: skip=${skip}, limit=${MP_BATCH_SIZE} (cache has ${cached} records)`);

    const records = await fetchMPBatchFromAPI(MP_BATCH_SIZE, skip);

    if (records.length === 0) {
      // Likely reached the end of MP results — reset to start cycling through again
      console.log(`[GNN-GCP] MP API returned 0 records at skip=${skip} — resetting to 0`);
      await setMPSkip(0);
      return;
    }

    await setMPSkip(skip + records.length);
    console.log(`[GNN-GCP] MP batch done: +${records.length} records cached (total ~${cached + records.length}, next skip=${skip + records.length})`);
  } catch (err: any) {
    console.warn(`[GNN-GCP] MP batch fetch failed: ${err.message}`);
  } finally {
    _mpFetchInProgress = false;
  }
}

// computeMetrics and computeCalibration removed —
// metrics are now computed inside the Python service (gnn/server.py)
// and returned directly in the callPythonTrain() response.

// Kept only for the CI95 calibration metric storage helper below.
// ── CI95 stub (values come from Python service) ───────────────────────────────
// (calibration values come from Python service response)

async function storeCalibrationMetric(
  jobId: number,
  coverage: number,
  meanWidth: number,
  n: number,
): Promise<void> {
  try {
    await db.execute(
      `INSERT INTO system_metrics (metric_name, metric_value, metadata, recorded_at)
       VALUES (
         'gnn_ci95_coverage',
         ${coverage.toFixed(4)},
         '{"jobId":${jobId},"meanWidthK":${meanWidth.toFixed(2)},"n":${n}}',
         NOW()
       )` as any
    );
  } catch (err: any) {
    // system_metrics may not exist yet — not fatal
    console.warn(`[GNN-GCP] Could not store calibration metric: ${err.message?.slice(0, 80)}`);
  }
}

// ── QE dataset augmentation (DFT-verified SC samples) ────────────────────────

async function loadQEDatasetSamples(existingFormulas: Set<string>): Promise<TrainingSample[]> {
  try {
    const rows = await db.execute(
      `SELECT material, tc, formation_energy, band_gap, lambda, omega_log, mu_star
       FROM quantum_engine_dataset
       WHERE tc >= 1.0 AND lambda IS NOT NULL AND lambda > 0
       ORDER BY tc DESC
       LIMIT 5000`
    );
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    for (const row of items) {
      if (!row.material || existingFormulas.has(row.material)) continue;
      // omega_log in quantum_engine_dataset is stored in cm⁻¹; convert to Kelvin (×1.4388)
      // so TrainingSample.omegaLog is always in Kelvin (matching GNN internal units).
      const omegaLogCm1 = row.omega_log != null ? Number(row.omega_log) : null;
      const omegaLogK = omegaLogCm1 != null && omegaLogCm1 > 0 ? omegaLogCm1 * 1.4388 : undefined;
      const muStar = row.mu_star != null ? Number(row.mu_star) : undefined;
      const bgRaw = row.band_gap != null ? Number(row.band_gap) : null;
      samples.push({
        formula: row.material,
        tc: Number(row.tc) || 0,
        formationEnergy: row.formation_energy != null ? Number(row.formation_energy) : undefined,
        bandgap: bgRaw != null && Number.isFinite(bgRaw) && bgRaw >= 0 ? bgRaw : undefined,
        lambda: row.lambda != null ? Number(row.lambda) : undefined,
        omegaLog: omegaLogK != null && Number.isFinite(omegaLogK) ? omegaLogK : undefined,
        muStar: muStar != null && Number.isFinite(muStar) && muStar > 0 ? muStar : undefined,
        dataConfidence: "physics-enriched", // QE entries with lambda/omega_log/mu_star
        structure: undefined,
        prototype: undefined,
        sourceTag: 'qe-dft',
      });
    }
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadQEDatasetSamples failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── SuperCon external entries (NIMS + JARVIS SC samples) ─────────────────────
// Loads superconductors from supercon_external_entries, which contains both the
// NIMS SuperCon database (~21k entries) and JARVIS SC datasets (SuperCon-Chem,
// SuperCon-2D, etc.). The local server dispatch payload often contains only the
// core SUPERCON_TRAINING_DATA (~274 samples); loading from the DB here ensures
// every GNN training job uses the full 16k+ SC corpus.

async function loadSuperconExternalSamples(existingFormulas: Set<string>): Promise<TrainingSample[]> {
  try {
    // LEFT JOIN with jarvis-dft3d by shared external_id (JVASP-XXXXX) to pull
    // formation_energy_per_atom, bandgap_ev, and other DFT properties that are
    // absent from jarvis-supercon_3d entries. ~907/1058 SC3D entries match.
    // No LIMIT — loads all sources including 3DSC, NIMS, JARVIS-Chem, etc.
    const rows = await db.execute(
      `SELECT sc.formula, sc.tc, sc.lambda, sc.space_group, sc.crystal_system,
              sc.source,
              (sc.raw_data->>'wlog_K')::real                              AS omega_log_k,
              (sc.raw_data->>'mu_star')::real                             AS raw_mu_star,
              sc.raw_data->>'data_confidence'                             AS data_confidence,
              (dft.raw_data->>'formation_energy_per_atom')::real          AS fe_per_atom,
              (dft.raw_data->>'bandgap_ev')::real                        AS bandgap_ev
       FROM supercon_external_entries sc
       LEFT JOIN supercon_external_entries dft
         ON dft.external_id = sc.external_id AND dft.source = 'jarvis-dft3d'
       WHERE sc.is_superconductor = true AND sc.tc > 0
       ORDER BY sc.source, sc.tc DESC`
    );
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    const srcCounts: Record<string, number> = {};
    for (const row of items) {
      const formula = row.formula as string;
      if (!formula || existingFormulas.has(formula)) continue;
      const tc = Number(row.tc);
      if (!Number.isFinite(tc) || tc <= 0) continue;
      const lambda = row.lambda != null ? Number(row.lambda) : undefined;
      // wlog_K from JARVIS-SuperCon-3D is already in Kelvin
      const omegaLogRaw = row.omega_log_k != null ? Number(row.omega_log_k) : null;
      const omegaLog = omegaLogRaw != null && Number.isFinite(omegaLogRaw) && omegaLogRaw > 0
        ? omegaLogRaw : undefined;
      const muStarRaw = row.raw_mu_star != null ? Number(row.raw_mu_star) : null;
      const muStar = muStarRaw != null && Number.isFinite(muStarRaw) && muStarRaw > 0
        ? muStarRaw : undefined;
      const spaceGroup = row.space_group as string | null;
      const crystalSystem = row.crystal_system as string | null;
      // Formation energy from DFT3D JOIN (per_atom), with sanity bounds
      const feRaw = row.fe_per_atom != null ? Number(row.fe_per_atom) : null;
      const formationEnergy = feRaw != null && Number.isFinite(feRaw) && feRaw > -20 && feRaw < 5
        ? feRaw : undefined;
      // Bandgap from DFT3D JOIN
      const bgRaw = row.bandgap_ev != null ? Number(row.bandgap_ev) : null;
      const bandgap = bgRaw != null && Number.isFinite(bgRaw) && bgRaw >= 0 ? bgRaw : undefined;
      // data_confidence from JARVIS SC3D raw_data — "dft-verified" unlocks 5× loss weight
      const dataConfidence = row.data_confidence === "dft-verified" ? "dft-verified" as const : undefined;
      const src = (row.source as string) ?? 'unknown';
      srcCounts[src] = (srcCounts[src] ?? 0) + 1;
      // Normalize source tag to one of the GNN's known one-hot buckets
      const sourceTag = src === 'hamidieh' ? 'hamidieh'
        : src.startsWith('jarvis-supercon') ? 'jarvis-sc'
        : src === '3dsc-mp' ? '3dsc-mp'
        : src === 'jarvis-dft3d' ? 'contrast-jarvis'
        : src;
      samples.push({
        formula,
        tc,
        formationEnergy,
        bandgap,
        dataConfidence,
        lambda: lambda != null && Number.isFinite(lambda) ? lambda : undefined,
        omegaLog,
        muStar,
        structure: spaceGroup || crystalSystem
          ? { spaceGroup: spaceGroup ?? undefined, crystalSystem: crystalSystem ?? undefined, dimensionality: undefined }
          : undefined,
        prototype: undefined,
        sourceTag,
      });
    }
    const srcSummary = Object.entries(srcCounts).map(([k, v]) => `${k}=${v}`).join(', ');
    console.log(`[GNN-GCP] loadSuperconExternalSamples: ${samples.length} total (${srcSummary})`);
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadSuperconExternalSamples failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── JARVIS DFT3D metallic negatives (Tc=0 contrast from JARVIS) ──────────────
// Loads metallic (non-SC) materials from the JARVIS DFT3D dataset stored in
// supercon_external_entries. These Tc=0 contrast examples teach the GNN that
// most metals don't superconduct.

async function loadJarvisDFT3DContrast(existingFormulas: Set<string>, scCount: number): Promise<TrainingSample[]> {
  try {
    const rows = await db.execute(
      `SELECT formula, space_group, crystal_system,
              (raw_data->>'formation_energy_per_atom')::real  AS fe_per_atom,
              (raw_data->>'bandgap_ev')::real                 AS bandgap_ev
       FROM supercon_external_entries
       WHERE source IN ('jarvis-dft3d', 'JARVIS-DFT3D-Metallic') AND (tc IS NULL OR tc = 0)
       ORDER BY RANDOM()
       LIMIT ${Math.min(scCount, 20000)}`
    );
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    for (const row of items) {
      const formula = row.formula as string;
      if (!formula || existingFormulas.has(formula)) continue;
      const spaceGroup = row.space_group as string | null;
      const crystalSystem = row.crystal_system as string | null;
      const feRaw = row.fe_per_atom != null ? Number(row.fe_per_atom) : null;
      const formationEnergy = feRaw != null && Number.isFinite(feRaw) && feRaw > -20 && feRaw < 5
        ? feRaw : undefined;
      const bgRaw = row.bandgap_ev != null ? Number(row.bandgap_ev) : null;
      const bandgap = bgRaw != null && Number.isFinite(bgRaw) && bgRaw >= 0 ? bgRaw : undefined;
      samples.push({
        formula,
        tc: 0,
        formationEnergy,
        bandgap,
        structure: spaceGroup || crystalSystem
          ? { spaceGroup: spaceGroup ?? undefined, crystalSystem: crystalSystem ?? undefined, dimensionality: undefined }
          : undefined,
        prototype: undefined,
        sourceTag: 'contrast-jarvis',
      });
    }
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadJarvisDFT3DContrast failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── MP augmentation (Tc=0 contrast examples) ─────────────────────────────────

async function loadMPContrastSamples(existingFormulas: Set<string>, scCount: number): Promise<TrainingSample[]> {
  try {
    const rows = await db.execute(
      `SELECT formula, data FROM mp_material_cache WHERE data_type = 'summary' LIMIT 10000`
    );
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const maxContrast = scCount;  // 1:1 cap — avoids Tc=0 majority diluting SC signal
    const samples: TrainingSample[] = [];
    for (const row of items) {
      if (samples.length >= maxContrast) break;
      const formula = row.formula as string;
      if (!formula || existingFormulas.has(formula)) continue;
      const d = row.data as any;
      if (!d) continue;
      // Default to 0 (metallic) when field is absent — we only seed metallic formulas,
      // so missing bandGap means the API didn't return it, not that it's an insulator.
      if (d.bandGap != null && d.bandGap > 0.1) continue;
      samples.push({ formula, tc: 0, formationEnergy: d.formationEnergyPerAtom ?? undefined, structure: undefined, prototype: undefined, sourceTag: 'contrast-mp' });
    }
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadMPContrastSamples failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── Job processing ────────────────────────────────────────────────────────────

async function processNextGnnJob(): Promise<boolean> {
  // Silent slot check — avoids the claim→augment→requeue loop without spamming
  // acquire/release log messages every poll cycle when the slot is free but idle.
  if (!isGNNTrainingSlotFree()) {
    return false; // startup training holds the slot; try again next poll cycle
  }

  const rows = await db.execute(
    `UPDATE gnn_training_jobs
     SET status = 'running', started_at = NOW()
     WHERE id = (
       SELECT id FROM gnn_training_jobs
       WHERE status = 'queued'
       ORDER BY created_at ASC
       LIMIT 1
       FOR UPDATE SKIP LOCKED
     )
     RETURNING id, training_data, dataset_size, dft_samples`
  );

  const job = (rows as any).rows?.[0] ?? (Array.isArray(rows) ? rows[0] : undefined);
  if (!job) return false;

  const jobId: number = job.id;
  let trainingData: TrainingSample[] = job.training_data as TrainingSample[];
  const datasetSize: number = job.dataset_size ?? trainingData.length;

  const existingFormulas = new Set(trainingData.map(s => s.formula));

  // Pre-warm the Neon connection before augmentation queries. Neon may cold-start
  // after inactivity; a failed warm-up here causes silent empty results below.
  try {
    await db.execute("SELECT 1");
  } catch (warmErr: any) {
    console.warn(`[GNN-GCP] DB pre-warm failed for job #${jobId}: ${warmErr.message?.slice(0, 80)} — augmentation queries may return empty`);
    // Small delay to let the pool recover before continuing
    await new Promise(r => setTimeout(r, 3000));
  }

  // Augment with NIMS + JARVIS superconductors from supercon_external_entries.
  // The local server dispatch payload often contains only the core SUPERCON seed
  // (~274 samples); the full 16k+ SC corpus lives in the shared Neon DB.
  const externalSCSamples = await loadSuperconExternalSamples(existingFormulas);
  if (externalSCSamples.length > 0) {
    trainingData = [...trainingData, ...externalSCSamples];
    externalSCSamples.forEach(s => existingFormulas.add(s.formula));
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${externalSCSamples.length} NIMS+JARVIS SC entries — total ${trainingData.length}`);
  }

  // Augment with DFT-verified SC entries from quantum_engine_dataset
  const qeSamples = await loadQEDatasetSamples(existingFormulas);
  if (qeSamples.length > 0) {
    trainingData = [...trainingData, ...qeSamples];
    qeSamples.forEach(s => existingFormulas.add(s.formula));
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${qeSamples.length} QE dataset SC entries — total ${trainingData.length}`);
  }

  // ── Fix #6: Cross-reference BCS physics from quantum_engine_dataset ────
  // supercon_external_entries has 0 omega_log/mu_star, but quantum_engine_dataset
  // has 2000 entries with all BCS parameters. Enrich by formula match.
  const physicsMap = new Map<string, { lambda?: number; omegaLog?: number; muStar?: number }>();
  for (const s of qeSamples) {
    if (s.lambda || s.omegaLog || s.muStar) {
      physicsMap.set(s.formula, { lambda: s.lambda, omegaLog: s.omegaLog, muStar: s.muStar });
    }
  }
  let enrichedCount = 0;
  for (const s of trainingData) {
    if (!s.lambda && !s.omegaLog && !s.muStar) {
      const physics = physicsMap.get(s.formula);
      if (physics) {
        if (physics.lambda && !s.lambda) s.lambda = physics.lambda;
        if (physics.omegaLog && !s.omegaLog) s.omegaLog = physics.omegaLog;
        if (physics.muStar && !s.muStar) s.muStar = physics.muStar;
        enrichedCount++;
      }
    }
  }
  if (enrichedCount > 0) {
    console.log(`[GNN-GCP] BCS physics enrichment: ${enrichedCount} samples gained lambda/omegaLog/muStar from QE dataset`);
  }

  // Augment with Tc=0 contrast: JARVIS DFT3D metallic negatives (structure-enriched)
  // plus MP metallic cache. 1:1 cap vs SC count to avoid label imbalance.
  const scCount = trainingData.filter(s => s.tc > 0).length;
  const jarvisContrast = await loadJarvisDFT3DContrast(existingFormulas, scCount);
  if (jarvisContrast.length > 0) {
    trainingData = [...trainingData, ...jarvisContrast];
    jarvisContrast.forEach(s => existingFormulas.add(s.formula));
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${jarvisContrast.length} JARVIS DFT3D contrast (Tc=0) — total ${trainingData.length}`);
  }

  const remainingContrastSlots = Math.max(0, scCount - jarvisContrast.length);
  const mpContrast = await loadMPContrastSamples(existingFormulas, remainingContrastSlots);
  if (mpContrast.length > 0) {
    trainingData = [...trainingData, ...mpContrast];
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${mpContrast.length} MP contrast samples (Tc=0) — total ${trainingData.length}`);
  }

  // Split BEFORE training so validation data is never seen by the models.
  // SC samples (Tc>0) are stratified: 80% train / 20% val.
  // Tc=0 contrast goes entirely into training — we validate on SC targets only.
  const scSamples  = trainingData.filter(s => s.tc > 0);
  const nonScSamples = trainingData.filter(s => s.tc === 0);
  const { train: scTrain, validation: scVal } = splitTrainValidation(scSamples, 0.20, 42);
  // Cap Tc=0 contrast samples to at most 1× SC training count to prevent
  // the model from collapsing to predict 0 (contrast imbalance was 2.4:1).
  const cappedNonSc = nonScSamples.slice(0, scTrain.length);
  const trainSet = [...scTrain, ...cappedNonSc];
  const valSet   = scVal; // held-out SC samples only — these are the discovery targets

  console.log(
    `[GNN-GCP] Starting training job #${jobId} — ${datasetSize} seed + ${externalSCSamples.length} NIMS+JARVIS + ${qeSamples.length} QE + ${jarvisContrast.length + mpContrast.length} contrast (capped ${cappedNonSc.length})` +
    ` | train=${trainSet.length} val=${valSet.length} (SC holdout)`
  );
  const startMs = Date.now();

  // Load startup R² for quality gate comparison
  let startupValR2 = -Infinity;
  try {
    const stateRow = await db.execute(
      `SELECT value FROM system_state WHERE key = '${STARTUP_CORPUS_STATE_KEY}'`
    );
    const stateVal = (stateRow as any).rows?.[0]?.value ?? (Array.isArray(stateRow) ? stateRow[0]?.value : undefined);
    if (stateVal) {
      const parsed = typeof stateVal === 'string' ? JSON.parse(stateVal) : stateVal;
      startupValR2 = parsed.valR2 ?? -Infinity;
    }
  } catch { /* non-fatal */ }

  try {
    await waitForXGBIdle();
    if (!acquireGNNTrainingSlot(`Job#${jobId}`)) {
      await db.execute(`UPDATE gnn_training_jobs SET status = 'queued', started_at = NULL WHERE id = ${jobId}`);
      return false;
    }
    let pyResult: Awaited<ReturnType<typeof callPythonTrain>>;
    try {
      console.log(`[GNN-GCP] Delegating job #${jobId} to Python GNN service…`);
      pyResult = await callPythonTrain(
        jobId, trainingData, Number.isFinite(startupValR2) ? startupValR2 : undefined,
      );
    } finally {
      releaseGNNTrainingSlot(`Job#${jobId}`);
    }

    const { r2, mae, rmse, trainR2, trainMae, valN, ci95Coverage, ci95Width, wallSeconds } = pyResult;
    const wallSec = wallSeconds.toFixed(1);

    // Yield + warm DB pool before writing (Python service may have held the connection idle)
    await new Promise(r => setTimeout(r, 50));
    try { await db.execute("SELECT 1"); } catch { /* pool will reconnect */ }

    if (pyResult.status === "discarded") {
      console.warn(`[GNN-GCP] Job #${jobId} discarded by Python service: ${pyResult.reason}`);
      await db.execute(`UPDATE gnn_training_jobs SET status = 'discarded' WHERE id = ${jobId}`);
    } else if (pyResult.status === "failed") {
      throw new Error(pyResult.reason ?? "Python service returned status=failed");
    } else {
      // PyTorch weights live in a .pt file on disk; TS DB stores metadata only.
      if (pyResult.modelPath) await recordPytorchModelPath(jobId, pyResult.modelPath);
      await storage.updateGnnTrainingJob(jobId, {
        status:      "done",
        weights:     [] as any,  // PyTorch weights stored in .pt file, not DB JSON
        r2,
        mae,
        rmse,
        trainR2,
        trainMae,
        valN,
        completedAt: new Date(),
      } as any);
    }

    const coverageStr = ci95Coverage > 0
      ? `CI95-cov=${(ci95Coverage * 100).toFixed(1)}% width=${ci95Width.toFixed(1)}K`
      : "CI95-cov=N/A";

    console.log(
      `[GNN-GCP] Job #${jobId} complete in ${wallSec}s` +
      ` | VAL(n=${valN}): R²=${r2.toFixed(3)} MAE=${mae.toFixed(1)}K RMSE=${rmse.toFixed(1)}K` +
      ` | TRAIN(sample): R²=${trainR2.toFixed(3)} MAE=${trainMae.toFixed(1)}K` +
      ` | overfit-gap=${(trainR2 - r2).toFixed(3)}` +
      ` | ${coverageStr}`
    );

    if (ci95Coverage > 0) {
      if (ci95Coverage < 0.80) {
        console.warn(`[GNN-GCP] ⚠ CI95 under-coverage (${(ci95Coverage * 100).toFixed(1)}%) — ensemble uncertainty is underestimated`);
      } else if (ci95Coverage > 0.99) {
        console.warn(`[GNN-GCP] ⚠ CI95 over-coverage (${(ci95Coverage * 100).toFixed(1)}%) — ensemble intervals are too wide`);
      }
      await storeCalibrationMetric(jobId, ci95Coverage, ci95Width, valN);
    }

    // Fetch next MP batch after every successful training job so the next
    // cycle's training payload is richer. Cached in mp_material_cache — the
    // local server reads from there when building the next training payload.
    fetchNextMPBatch().catch(() => {});

  } catch (err: any) {
    const jobErrMsg = err instanceof Error ? (err.message || err.constructor?.name || "(empty Error)") : String(err ?? "unknown");
    console.error(`[GNN-GCP] Job #${jobId} failed: ${jobErrMsg}`);
    await storage.updateGnnTrainingJob(jobId, {
      status: "failed",
      errorMessage: err.message?.slice(0, 1000) ?? "unknown error",
      completedAt: new Date(),
    }).catch(() => {});
  }

  return true;
}

// ── Startup full corpus training ──────────────────────────────────────────────
// Runs once per GCP worker lifetime (guarded by system_state). Trains the GNN
// ensemble on the full NIMS + JARVIS SC corpus (~15k entries) plus QE/MP contrast.
// Subsequent dispatched retrains use only 5k samples so they finish in minutes.

const STARTUP_CORPUS_STATE_KEY = 'gnn_startup_corpus_training';
// Per-model checkpoint keys written to system_state as each worker completes.
// These let a restarted process skip models that already finished.
const STARTUP_CKPT_KEY_PREFIX = 'gnn_startup_ckpt_';
const STARTUP_CKPT_TTL_HOURS = 20; // checkpoints older than this are stale

// Increment this whenever the GNN forward pass formula changes in a way that
// makes old weights incompatible with the new code (e.g. hard-cap → sigmoid).
// When the stored version doesn't match, startup retrains from scratch.
const GNN_MODEL_VERSION = 14; // v14: hard BCE labels + v15 dedicated cls head (migrateWeights handles new fields — no retrain needed)

// In-process guard: run corpus training exactly once per GCP worker process lifetime.
let _startupCorpusRan = false;

// ── Startup checkpoint helpers ────────────────────────────────────────────────

/** Returns loaded ensemble weights if startup training completed within TTL, else null. */
async function checkStartupCompletedRecently(): Promise<GNNWeights[] | null> {
  try {
    const stateRows = await db.execute(
      `SELECT value FROM system_state WHERE key = '${STARTUP_CORPUS_STATE_KEY}'`
    );
    const stateRow = (stateRows as any).rows?.[0] ?? (Array.isArray(stateRows) ? stateRows[0] : undefined);
    if (!stateRow?.value) return null;
    const state = typeof stateRow.value === 'string' ? JSON.parse(stateRow.value) : stateRow.value;
    if (!state?.doneAt) return null;
    // Version check: if stored weights were trained with an older forward pass
    // formula, they're incompatible with the current code and must be retrained.
    if ((state.modelVersion ?? 1) !== GNN_MODEL_VERSION) {
      console.log(`[GNN-GCP] Startup weights are model version ${state.modelVersion ?? 1} but code requires v${GNN_MODEL_VERSION} — will retrain from scratch`);
      return null;
    }
    const ageHours = (Date.now() - new Date(state.doneAt).getTime()) / 3_600_000;
    if (ageHours > 8) {
      console.log(`[GNN-GCP] Startup training record is ${ageHours.toFixed(1)}h old — will retrain`);
      return null;
    }
    // Load weights from the most recent completed job
    const jobRows = await db.execute(
      `SELECT weights FROM gnn_training_jobs WHERE status = 'done' AND weights IS NOT NULL ORDER BY completed_at DESC LIMIT 1`
    );
    const jobRow = (jobRows as any).rows?.[0] ?? (Array.isArray(jobRows) ? jobRows[0] : undefined);
    if (!jobRow?.weights) return null;
    const weights = typeof jobRow.weights === 'string' ? JSON.parse(jobRow.weights) : jobRow.weights;
    if (!Array.isArray(weights) || weights.length === 0) return null;
    console.log(`[GNN-GCP] Startup training completed ${ageHours.toFixed(1)}h ago (v${GNN_MODEL_VERSION}) — restoring ${weights.length} models from DB`);
    return weights as GNNWeights[];
  } catch {
    return null;
  }
}

/**
 * Loads any per-model checkpoints saved from a previous interrupted run.
 * Returns a map of modelIndex → GNNWeights for models that already finished.
 */
async function loadStartupModelCheckpoints(): Promise<Map<number, GNNWeights>> {
  const result = new Map<number, GNNWeights>();
  const n = Math.min(ENSEMBLE_SIZE, ENSEMBLE_SEEDS.length);
  try {
    const keys = Array.from({ length: n }, (_, i) => `'${STARTUP_CKPT_KEY_PREFIX}${i}'`).join(', ');
    const rows = await db.execute(
      `SELECT key, value, updated_at FROM system_state WHERE key IN (${keys})`
    );
    const rowArr: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const cutoff = Date.now() - STARTUP_CKPT_TTL_HOURS * 3_600_000;
    for (const row of rowArr) {
      const updatedAt = new Date(row.updated_at ?? 0).getTime();
      if (updatedAt < cutoff) continue; // stale checkpoint
      const idx = parseInt(String(row.key).replace(STARTUP_CKPT_KEY_PREFIX, ''), 10);
      if (!Number.isFinite(idx)) continue;
      const model = typeof row.value === 'string' ? JSON.parse(row.value) : row.value;
      if (model && typeof model === 'object') {
        result.set(idx, model as GNNWeights);
        console.log(`[GNN-GCP] Restored checkpoint for model ${idx}`);
      }
    }
  } catch (err: any) {
    console.warn(`[GNN-GCP] Checkpoint load failed (non-fatal): ${err.message?.slice(0, 80)}`);
  }
  return result;
}

/** Saves a single trained model to system_state as a checkpoint. */
async function saveStartupModelCheckpoint(model: GNNWeights, modelIndex: number): Promise<void> {
  try {
    const key = `${STARTUP_CKPT_KEY_PREFIX}${modelIndex}`;
    // GNN weights are pure JSON numbers — no single-quote injection risk.
    const value = JSON.stringify(model);
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('${key}', '${value}'::jsonb, NOW())
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()`
    );
    console.log(`[GNN-GCP] Checkpointed model ${modelIndex} (${(value.length / 1024).toFixed(0)} KB)`);
  } catch (err: any) {
    console.warn(`[GNN-GCP] Checkpoint save failed for model ${modelIndex}: ${err.message?.slice(0, 80)}`);
  }
}

/** Clears per-model checkpoints after a full successful startup training. */
async function clearStartupModelCheckpoints(): Promise<void> {
  const n = Math.min(ENSEMBLE_SIZE, ENSEMBLE_SEEDS.length);
  try {
    const keys = Array.from({ length: n }, (_, i) => `'${STARTUP_CKPT_KEY_PREFIX}${i}'`).join(', ');
    await db.execute(`DELETE FROM system_state WHERE key IN (${keys})`);
  } catch { /* non-fatal */ }
}

async function runStartupFullCorpusTraining(): Promise<void> {
  // Skip if already ran in this process (e.g. called twice).
  if (_startupCorpusRan) {
    console.log('[GNN-GCP] Startup full corpus training already ran this session — skipping');
    return;
  }
  _startupCorpusRan = true;

  // ── Load partial checkpoints from a previous interrupted run ─────────────
  // Always run startup training on every GCP restart — the full corpus
  // training is the most reliable baseline and must not be skipped.
  // Checkpoints handle interrupted training; the "skip if recent" guard was
  // removed because it caused startup to skip when the GCP worker was
  // restarted after a code deployment.
  const checkpointedModels = await loadStartupModelCheckpoints();
  const allIndices = Array.from({ length: Math.min(ENSEMBLE_SIZE, ENSEMBLE_SEEDS.length) }, (_, i) => i);
  const indicesToTrain = allIndices.filter(i => !checkpointedModels.has(i));

  if (checkpointedModels.size > 0) {
    console.log(`[GNN-GCP] Found ${checkpointedModels.size} checkpointed model(s) — only training ${indicesToTrain.length} remaining`);
  }

  console.log('[GNN-GCP] ════════════════════════════════════════════════════════');
  console.log('[GNN-GCP]  STARTUP PHASE 3 — Full Corpus Training');
  console.log('[GNN-GCP]  Loading full SC corpus (NIMS + JARVIS + 3DSC — no limit)');
  console.log('[GNN-GCP] ════════════════════════════════════════════════════════');

  // Pre-warm DB connection before heavy queries
  try { await db.execute("SELECT 1"); } catch { await new Promise(r => setTimeout(r, 3000)); }

  const existingFormulas = new Set<string>();

  // Load full SC corpus from supercon_external_entries
  const externalSC = await loadSuperconExternalSamples(existingFormulas);
  externalSC.forEach(s => existingFormulas.add(s.formula));

  // Augment with DFT-verified QE entries
  const qeSamples = await loadQEDatasetSamples(existingFormulas);
  qeSamples.forEach(s => existingFormulas.add(s.formula));
  if (qeSamples.length > 0) console.log(`[GNN-GCP]  +${qeSamples.length} QE DFT-verified entries`);

  let trainingData: TrainingSample[] = [...externalSC, ...qeSamples];

  // Contrast examples: JARVIS metallic negatives + MP
  const scCount = trainingData.filter(s => s.tc > 0).length;
  const jarvisContrast = await loadJarvisDFT3DContrast(existingFormulas, scCount);
  jarvisContrast.forEach(s => existingFormulas.add(s.formula));
  const remainingContrast = Math.max(0, scCount - jarvisContrast.length);
  const mpContrast = await loadMPContrastSamples(existingFormulas, remainingContrast);
  trainingData = [...trainingData, ...jarvisContrast, ...mpContrast];
  if (jarvisContrast.length + mpContrast.length > 0) {
    console.log(`[GNN-GCP]  +${jarvisContrast.length} JARVIS metallic + ${mpContrast.length} MP contrast (Tc=0)`);
  }

  // Train/val split
  const scSamples = trainingData.filter(s => s.tc > 0);
  const nonScSamples = trainingData.filter(s => s.tc === 0);
  const { train: scTrain, validation: scVal } = splitTrainValidation(scSamples, 0.20, 42);
  const cappedNonSc = nonScSamples.slice(0, scTrain.length);
  const trainSet = [...scTrain, ...cappedNonSc];
  const valSet = scVal;


  console.log(`[GNN-GCP]  Dataset: ${scSamples.length} SC + ${cappedNonSc.length} contrast | train=${trainSet.length} val=${valSet.length}`);
  console.log(`[GNN-GCP]  Delegating startup training to Python GNN service…`);

  let r2 = 0, mae = 0, rmse = 0, trainR2 = 0, trainMae = 0, valN = 0;
  let startupModelPath: string | undefined;
  let pyResult: Awaited<ReturnType<typeof callPythonTrain>> | undefined;
  try {
    await waitForXGBIdle();
    if (!acquireGNNTrainingSlot('STARTUP')) {
      console.log('[GNN-GCP] Startup corpus training deferred — dispatched job is training, will retry in 5 min');
      await new Promise(r => setTimeout(r, 5 * 60_000));
      if (!acquireGNNTrainingSlot('STARTUP-retry')) {
        console.log('[GNN-GCP] Startup corpus training skipped — GNN still busy');
        return;
      }
    }
    try {
      // Insert a placeholder job so the Python service result has a job ID to reference
      const placeholderJob = await storage.insertGnnTrainingJob({
        status: "running" as any,
        trainingData: [] as any,
        datasetSize: trainSet.length,
        dftSamples: qeSamples.length,
      });
      const startupJobId = placeholderJob.id;

      pyResult = await callPythonTrain(startupJobId, trainingData, undefined, 15);
      ({ r2, mae, rmse, trainR2, trainMae, valN, startupModelPath } = {
        ...pyResult, startupModelPath: pyResult.modelPath,
      });
    } finally {
      try {
        const deleted = await db.execute(
          `DELETE FROM gnn_training_jobs WHERE status IN ('queued', 'running') RETURNING id`
        );
        const deletedIds = ((deleted as any).rows ?? (Array.isArray(deleted) ? deleted : [])).map((r: any) => r.id);
        if (deletedIds.length > 0) {
          console.log(`[GNN-GCP] Cleared ${deletedIds.length} stale job(s) before slot release — startup weights preserved`);
        }
      } catch { /* non-fatal */ }
      releaseGNNTrainingSlot('STARTUP');
    }
  } catch (err: any) {
    console.error(`[GNN-GCP] Startup corpus training failed: ${err.message}`);
    return;
  }

  const wallSec = pyResult ? pyResult.wallSeconds.toFixed(1) : "?";
  const trainMaeV = trainMae;

  console.log('[GNN-GCP] ────────────────────────────────────────────────────────');
  console.log(`[GNN-GCP]  Startup training complete in ${wallSec}s`);
  console.log(`[GNN-GCP]  VAL  (n=${valN}): R²=${r2.toFixed(3)}  MAE=${mae.toFixed(1)}K  RMSE=${rmse.toFixed(1)}K`);
  console.log(`[GNN-GCP]  TRAIN(sample): R²=${trainR2.toFixed(3)}  MAE=${trainMaeV.toFixed(1)}K  overfit-gap=${(trainR2 - r2).toFixed(3)}`);
  if (startupModelPath) console.log(`[GNN-GCP]  PyTorch ensemble saved: ${startupModelPath}`);
  console.log('[GNN-GCP] ════════════════════════════════════════════════════════');

  if (pyResult?.status === "discarded") {
    console.warn(`[GNN-GCP] Startup weights NOT stored — ${pyResult.reason}`);
  } else if (pyResult?.status === "done") {
    // Record metrics in DB as a completed job (weights: [] since .pt lives on disk)
    try {
      await db.execute("SELECT 1");
      const insertedJob = await storage.insertGnnTrainingJob({
        status: "queued" as any,
        trainingData: [] as any,
        datasetSize: trainSet.length,
        dftSamples: qeSamples.length,
      });
      await storage.updateGnnTrainingJob(insertedJob.id, {
        status: "done",
        weights: [] as any,   // PyTorch weights live in .pt file on disk
        r2, mae, rmse,
        trainR2,
        trainMae,
        valN,
        completedAt: new Date(),
      } as any);
      if (startupModelPath) await recordPytorchModelPath(insertedJob.id, startupModelPath);
      console.log(`[GNN-GCP] Startup metrics stored as job #${insertedJob.id}`);
    } catch (err: any) {
      console.warn(`[GNN-GCP] Failed to store startup metrics in DB: ${err.message?.slice(0, 120)}`);
    }
  }

  // Mark startup complete in system_state and clear per-model checkpoints
  try {
    const nowIso = new Date().toISOString();
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('${STARTUP_CORPUS_STATE_KEY}', '{"doneAt":"${nowIso}","trainN":${trainSet.length},"valR2":${r2.toFixed(4)},"modelVersion":${GNN_MODEL_VERSION}}', NOW())
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()`
    );
  } catch { /* non-fatal */ }
  // Per-model checkpoints are superseded by the full job stored above.
  await clearStartupModelCheckpoints();
}

// ── Main loop ─────────────────────────────────────────────────────────────────

export async function startGNNLoop(): Promise<void> {
  console.log(`[GNN-GCP] GNN training worker started — poll every ${POLL_INTERVAL_MS / 1000}s`);
  console.log(`[GNN-GCP] Progressive MP fetch: ${MP_BATCH_SIZE} records/cycle, max cache ${MP_MAX_CACHE}`);

  // ── Start Python GNN service ──────────────────────────────────────────────
  console.log("[GNN-GCP] Starting Python GNN service (FastAPI)…");
  try {
    await spawnPythonGNNService();
  } catch (err: any) {
    console.error(`[GNN-GCP] FATAL: Could not start Python GNN service: ${err.message}`);
    console.error("[GNN-GCP] Ensure gnn/server.py dependencies are installed:");
    console.error("  pip install fastapi uvicorn torch torch-geometric");
    throw err;
  }
  console.log("[GNN-GCP] Python GNN service ready on " + GNN_SERVICE_URL);

  // Reset any jobs left in 'running' state from a previous crashed/restarted process.
  // Without this, the local server sees an active job and won't dispatch new ones.
  try {
    const reset = await db.execute(
      `UPDATE gnn_training_jobs SET status = 'queued', started_at = NULL
       WHERE status = 'running' AND started_at < NOW() - INTERVAL '5 minutes'
       RETURNING id`
    );
    const resetIds: number[] = ((reset as any).rows ?? (Array.isArray(reset) ? reset : [])).map((r: any) => r.id);
    if (resetIds.length > 0) {
      console.log(`[GNN-GCP] Reset ${resetIds.length} stale running job(s) back to queued: #${resetIds.join(', #')}`);
    }
  } catch (err: any) {
    console.warn(`[GNN-GCP] Stale job reset failed (non-fatal): ${err.message?.slice(0, 80)}`);
  }

  // Phase 3 startup: train on full NIMS+JARVIS corpus as a background task.
  // Delayed 30s so XGB/ML/DFT loops can claim their DB connections first.
  // Regular dispatched jobs use a 5k-sample subset; startup gets the full corpus once.
  // Fire-and-forget so job polling starts immediately after startup training acquires the slot.
  new Promise<void>(r => setTimeout(r, 30_000))
    .then(() => runStartupFullCorpusTraining())
    .catch(err => console.error("[GNN-GCP] Startup corpus training error:", err?.message ?? String(err)));

  // Delay job polling by 35s so the startup corpus training acquires the slot first.
  // Without this delay the poll loop grabs a queued job immediately, wins the slot race,
  // and startup training gets deferred for 5+ minutes.
  await new Promise(r => setTimeout(r, 35_000));
  console.log("[GNN-GCP] Job polling starting — startup training should hold the slot");

  let pollCount = 0;
  let lastHeartbeatMs = Date.now();
  const HEARTBEAT_INTERVAL_MS = 5 * 60_000; // log alive every 5 minutes

  while (running) {
    try {
      const processed = await processNextGnnJob();
      pollCount++;
      if (!processed) {
        const now = Date.now();
        if (now - lastHeartbeatMs >= HEARTBEAT_INTERVAL_MS) {
          const reason = !isGNNTrainingSlotFree() ? 'startup training in progress' : 'waiting for local server to dispatch';
          console.log(`[GNN-GCP] Polling — idle (${pollCount} polls since start). ${reason}.`);
          lastHeartbeatMs = now;
        }
      } else {
        pollCount = 0;
        lastHeartbeatMs = Date.now();
      }
      await new Promise(r => setTimeout(r, processed ? 1000 : POLL_INTERVAL_MS));
    } catch (err: any) {
      const msg = err instanceof Error
        ? (err.stack || err.message || err.constructor?.name || "(empty Error)")
        : (err != null ? String(err) : "unknown");
      console.error(`[GNN-GCP] Loop error (${err?.constructor?.name ?? typeof err}): ${msg || "(no message)"}`);
      await new Promise(r => setTimeout(r, isConnectionError(err) ? 5000 : POLL_INTERVAL_MS));
    }
  }
}

export function stopGNNLoop() {
  running = false;
  if (_pythonProcess && !_pythonProcess.killed) {
    console.log("[GNN-GCP] Stopping Python GNN service…");
    _pythonProcess.kill("SIGTERM");
  }
}
