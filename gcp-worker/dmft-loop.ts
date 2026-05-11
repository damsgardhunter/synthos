/**
 * DMFT Service Loop — Manages the DMFT Docker container lifecycle
 * alongside the GNN/DFT workers on the gnn-training VM.
 *
 * Responsibilities:
 *   1. Start the DMFT Docker container (if not already running)
 *   2. Health-check the service at /health
 *   3. Log DMFT service status and job activity
 *   4. Restart container if it crashes
 *   5. Expose status for the main worker process
 *
 * The DMFT service runs alongside GNN training — both share the same VM.
 * DMFT uses 12 MPI ranks (12 cores), GNN uses GPU + ~8 cores.
 * On c2-standard-30 (30 vCPUs) this leaves headroom for the OS and Node.js.
 */

import { execSync, spawn } from "child_process";

let _running = false;
let _dmftHealthy = false;
let _lastHealthCheck: number = 0;
let _consecutiveFailures = 0;
const HEALTH_INTERVAL_MS = 60_000;        // check every 60s
const DMFT_SERVICE_URL = process.env.DMFT_SERVICE_URL || "http://localhost:8780";
const DMFT_COMPOSE_FILE = "dmft/docker-compose.yml";
const MAX_CONSECUTIVE_FAILURES = 5;

export function isDMFTHealthy(): boolean {
  return _dmftHealthy;
}

export async function startDMFTLoop(): Promise<void> {
  _running = true;
  console.log("[DMFT-GCP] DMFT service loop starting");
  console.log(`[DMFT-GCP] Service URL: ${DMFT_SERVICE_URL}`);
  console.log(`[DMFT-GCP] Compose file: ${DMFT_COMPOSE_FILE}`);

  // Only manage the Docker container if we're on the same machine (localhost URL).
  // If DMFT_SERVICE_URL points to a remote VM, we just health-check it.
  const isLocal = DMFT_SERVICE_URL.includes("localhost") || DMFT_SERVICE_URL.includes("127.0.0.1");

  // Phase 1: Ensure Docker container is running (only if local)
  if (isLocal) {
    await ensureContainerRunning();
  } else {
    console.log(`[DMFT-GCP] Remote DMFT service at ${DMFT_SERVICE_URL} — skipping container management`);
  }

  // Phase 2: Wait for service to become healthy
  const healthy = await waitForHealth(120_000); // 2 min timeout
  if (healthy) {
    console.log("[DMFT-GCP] DMFT service is healthy and ready for jobs");
  } else {
    console.log("[DMFT-GCP] DMFT service not healthy after 2 min — will keep retrying in background");
  }

  // Phase 3: Periodic health check loop
  while (_running) {
    await new Promise(r => setTimeout(r, HEALTH_INTERVAL_MS));
    if (!_running) break;

    try {
      const health = await checkHealth();
      if (health.healthy) {
        if (!_dmftHealthy) {
          console.log(`[DMFT-GCP] DMFT service recovered: ${JSON.stringify(health.components)}`);
        }
        _dmftHealthy = true;
        _consecutiveFailures = 0;

        // Log job activity if any
        if (health.jobs_pending > 0 || health.current_job) {
          console.log(
            `[DMFT-GCP] Jobs: pending=${health.jobs_pending}, ` +
            `current=${health.current_job || "none"}, ` +
            `completed=${health.total_completed}`
          );
        }
      } else {
        _dmftHealthy = false;
        _consecutiveFailures++;
        console.log(`[DMFT-GCP] Health check failed (${_consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES})`);

        if (_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES && isLocal) {
          console.log("[DMFT-GCP] Too many failures — restarting container");
          await restartContainer();
          _consecutiveFailures = 0;
        } else if (_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
          console.log("[DMFT-GCP] Remote DMFT service unresponsive — cannot restart from here");
          _consecutiveFailures = 0; // reset to avoid spamming
        }
      }
    } catch (err: any) {
      _dmftHealthy = false;
      _consecutiveFailures++;
      console.log(`[DMFT-GCP] Health check error: ${err.message} (${_consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES})`);

      if (_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES && isLocal) {
        console.log("[DMFT-GCP] Too many failures — restarting container");
        await restartContainer();
        _consecutiveFailures = 0;
      } else if (_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        console.log("[DMFT-GCP] Remote DMFT service unresponsive — cannot restart from here");
        _consecutiveFailures = 0;
      }
    }
  }

  console.log("[DMFT-GCP] DMFT loop stopped");
}

export function stopDMFTLoop(): void {
  _running = false;
  console.log("[DMFT-GCP] Stop requested");
}

// ── Docker container management ─────────────────────────────────────────────

async function ensureContainerRunning(): Promise<void> {
  try {
    // Check if container exists and is running
    const status = execSync(
      'docker inspect --format="{{.State.Status}}" qae-dmft 2>/dev/null || echo "missing"',
      { timeout: 10_000 },
    ).toString().trim();

    if (status === "running") {
      console.log("[DMFT-GCP] Container already running");
      return;
    }

    if (status === "exited" || status === "created") {
      console.log(`[DMFT-GCP] Container exists but ${status} — starting`);
      execSync(`docker compose -f ${DMFT_COMPOSE_FILE} start`, { timeout: 30_000 });
      return;
    }

    // Container doesn't exist — try docker compose up
    console.log("[DMFT-GCP] Starting DMFT container via docker compose...");
    const proc = spawn(
      "docker", ["compose", "-f", DMFT_COMPOSE_FILE, "up", "-d", "--build"],
      { stdio: ["ignore", "pipe", "pipe"] },
    );

    // Capture build output for logging
    proc.stdout?.on("data", (d: Buffer) => {
      const line = d.toString().trim();
      if (line) console.log(`[DMFT-GCP] docker: ${line}`);
    });
    proc.stderr?.on("data", (d: Buffer) => {
      const line = d.toString().trim();
      if (line) console.log(`[DMFT-GCP] docker: ${line}`);
    });

    await new Promise<void>((resolve, reject) => {
      proc.on("close", (code) => {
        if (code === 0) {
          console.log("[DMFT-GCP] Container started successfully");
          resolve();
        } else {
          console.error(`[DMFT-GCP] docker compose up failed with exit ${code}`);
          resolve(); // don't crash the worker, just mark unhealthy
        }
      });
      proc.on("error", (err) => {
        console.error(`[DMFT-GCP] docker compose error: ${err.message}`);
        resolve();
      });
    });
  } catch (err: any) {
    console.error(`[DMFT-GCP] Failed to start container: ${err.message}`);
    // Check if Docker is even installed
    try {
      execSync("docker --version", { timeout: 5_000 });
    } catch {
      console.error("[DMFT-GCP] Docker not found — DMFT service cannot run");
    }
  }
}

async function restartContainer(): Promise<void> {
  try {
    console.log("[DMFT-GCP] Restarting DMFT container...");
    execSync(`docker compose -f ${DMFT_COMPOSE_FILE} restart`, { timeout: 60_000 });
    console.log("[DMFT-GCP] Container restarted");
    // Wait a bit for service to come up
    await new Promise(r => setTimeout(r, 15_000));
  } catch (err: any) {
    console.error(`[DMFT-GCP] Restart failed: ${err.message}`);
  }
}

// ── Health checking ─────────────────────────────────────────────────────────

interface DMFTHealth {
  healthy: boolean;
  components?: Record<string, string>;
  jobs_pending?: number;
  current_job?: string | null;
  total_completed?: number;
}

async function checkHealth(): Promise<DMFTHealth> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10_000);

  try {
    const res = await fetch(`${DMFT_SERVICE_URL}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!res.ok) {
      return { healthy: false };
    }

    const data = await res.json() as any;
    return {
      healthy: data.status === "healthy",
      components: data.components,
      jobs_pending: data.jobs_pending,
      current_job: data.current_job,
      total_completed: data.total_completed,
    };
  } catch {
    clearTimeout(timeout);
    return { healthy: false };
  }
}

async function waitForHealth(timeoutMs: number): Promise<boolean> {
  const start = Date.now();
  const interval = 5_000;

  while (Date.now() - start < timeoutMs) {
    const health = await checkHealth();
    if (health.healthy) {
      _dmftHealthy = true;
      return true;
    }
    console.log("[DMFT-GCP] Waiting for service to become healthy...");
    await new Promise(r => setTimeout(r, interval));
  }

  return false;
}
