import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";
import { isMainThread } from "worker_threads";

// DATABASE_URL must point to the Neon *pooler* endpoint (hostname contains "-pooler").
// The pooler (pgBouncer) is always-on and never scales to zero, eliminating cold-start
// connection timeouts. Direct endpoint connections are unreliable even on paid plans.
const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  // 15 connections — engine cycle phases + DFT refill + API routes all compete.
  // 10 was too few: 6-query Promise.all in engine-memory + 2 DFT refill queries
  // + engine cycle ops routinely exceeded 10 and caused cascade timeouts.
  max: 10,  // reduced from 20: when GB training blocks event loop for ~60s, all connections go zombie simultaneously.
            // With max=20, pool tries to create 20 new connections at once after training → Neon throttles
            // (authentication timeouts). With max=10, only 10 simultaneous reconnects → Neon handles cleanly.
            // Compensated by: Phase 7 skips during SG sweep, DFT refill skips during sweep.
  // Recycle idle connections after 10s — pgBouncer (Neon serverless) drops idle connections
  // after ~15-20s. Setting our local pool timeout below that threshold means we discard
  // connections before pgBouncer kills them, preventing "Connection terminated" zombie errors.
  idleTimeoutMillis: 10_000,
  // 25s: GB training is offloaded to GCP — no more 60s local event-loop blocks.
  // SG sweep blocks up to ~20s (extractFeatures sequential batches). After a block,
  // socket.setTimeout(20s) fires and destroys zombies; pool reconnects in ~1-2s per
  // slot. 25s gives 5s margin (25-20=5s) — sufficient for the pool's max=10 config
  // (vs old max=20 which needed more margin). On genuine failure callers wait 25s
  // (vs 60s before), cutting reconnection storm duration by 58%.
  connectionTimeoutMillis: 25_000,
  // 15s: raised from 10s — gives large queries (getCandidatesForDFTRefill, getSuperconductorCandidates)
  // time to complete cleanly rather than being killed mid-execution, which returns the
  // connection to the pool in an uncertain state and causes ECONNRESET on the next user.
  query_timeout: 15_000,
  keepAlive: true,
  // 1s: lowered from 5s — Windows TCP keepalive fires after the initial delay,
  // then waits for system-level probe intervals (~60-70s total) before declaring
  // a silently-dropped Neon connection dead. At 5s initial delay the full hang
  // was ~70s, causing /api/research-logs to block for 70s before returning 500.
  // At 1s the first keepalive fires quickly, cutting the hang to ~10-20s.
  keepAliveInitialDelayMillis: 1_000,
});

pool.on("error", (err) => {
  console.error("[DB] Pool error:", err.message);
});

// NOTE: The custom socket.setTimeout(20_000) handler was removed. It fired during
// SG sweep bursts (when the event loop was temporarily blocked), destroying healthy
// connections that had been idle for 20s, which cascaded into pool exhaustion and
// process crashes. The pool already handles zombie connections safely via:
//   - idleTimeoutMillis: 10_000  — pg-pool discards idle connections after 10s
//   - keepAlive: true / keepAliveInitialDelayMillis: 1_000 — OS-level probes detect dead sockets
//   - connectionTimeoutMillis: 25_000 — callers get a clear error if pool is exhausted
// These built-in mechanisms are sufficient without the risky manual socket destroy.

export const db = drizzle(pool, { schema });

/**
 * Returns true if an error string looks like a transient connection problem
 * that will resolve itself once pg-pool opens a fresh connection.
 */
export function isConnectionError(err: any): boolean {
  const msg: string = (err?.message ?? String(err ?? "")).toLowerCase();
  return msg.includes("connection terminated") ||
    msg.includes("connection timeout") ||
    msg.includes("etimedout") ||
    msg.includes("timeout") ||
    msg.includes("econnreset") ||
    msg.includes("econnrefused") ||
    msg.includes("socket hang up") ||
    msg.includes("the client is closed") ||
    msg.includes("connection ended unexpectedly") ||
    msg.includes("before secure tls connection") ||
    msg.includes("socket disconnected") ||
    err?.constructor?.name === "AggregateError";
}

/**
 * Drains all idle pool connections and releases them with destroyConnection=true.
 * Call this after any long-running synchronous operation (e.g. SG sweep) that
 * blocks the Node.js event loop for >10s — during the block, idle timeout timers
 * and socket timeout callbacks cannot fire, so zombie connections accumulate.
 * Draining proactively clears them before the next DB write attempt.
 */
export async function drainIdleConnections(): Promise<void> {
  const idleCount = pool.idleCount;
  if (idleCount === 0) return;
  console.log(`[DB] Draining ${idleCount} idle connections after blocking operation`);
  const drains: Promise<void>[] = [];
  for (let i = 0; i < idleCount; i++) {
    drains.push(
      pool.connect()
        .then(client => { client.release(true); })
        .catch(() => { /* ignore — connection already dead */ })
    );
  }
  await Promise.allSettled(drains);
}

/**
 * Probe DB connectivity with a SELECT 1, retrying up to maxAttempts times.
 * Uses a standalone pg.Client (NOT the shared pool) so the probe never
 * competes with pool-slot reconnection storms after an event-loop block.
 * When the pool is stuck (all 10 slots in TCP CONNECTING), pool.connect()
 * queues behind those attempts and hangs for 5+ minutes. A standalone client
 * bypasses the pool entirely, completing in ≤15s regardless of pool state.
 * On success, drains the shared pool to flush zombie connections.
 */
export async function probeDBConnection(
  maxAttempts = 5,
  delayMs = 10_000
): Promise<boolean> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const client = new pg.Client({
      connectionString: process.env.DATABASE_URL,
      connectionTimeoutMillis: 15_000,
    });
    try {
      await client.connect();
      await client.query("SELECT 1");
      await client.end();
      console.log(`[DB] Probe OK on attempt ${attempt}/${maxAttempts} — flushing pool zombies`);
      // Pool may still have slots stuck in TCP CONNECTING. Drain them now so
      // the next pool.connect() call gets a fresh connection immediately.
      await drainIdleConnections().catch(() => {});
      return true;
    } catch (err: any) {
      console.warn(`[DB] Probe failed attempt ${attempt}/${maxAttempts}: ${err?.message}`);
      try { await client.end(); } catch {}
      if (attempt < maxAttempts) await new Promise(r => setTimeout(r, delayMs));
    }
  }
  return false;
}

if (isMainThread) {
  // Verify connectivity at startup — pooler should respond immediately.
  pool.connect().then(client => {
    client.query("SELECT 1").then(() => {
      console.log("[DB] Neon pooler connected");
      client.release();
    }).catch(() => client.release());
  }).catch(err => {
    console.warn("[DB] Startup connection failed:", err.message);
  });
}
