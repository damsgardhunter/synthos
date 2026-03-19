import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";
import { isMainThread } from "worker_threads";

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  // 5 max connections per process (local + GCP = 10 total) — keeps simultaneous
  // TLS handshakes low during Neon cold-start. 10/process was overwhelming Neon.
  max: 5,
  // Keep idle connections for 90s — shorter than 120s so stale connections
  // get recycled before they accumulate, but longer than the 45s keepalive interval.
  idleTimeoutMillis: 90_000,
  // Give Neon up to 8s to connect; fail fast so pool slots are freed quickly.
  // Neon cold-start typically takes 5-7s; 8s gives a small buffer while halving
  // the worst-case stall when the pool is saturated (was 20s, now 8s).
  connectionTimeoutMillis: 8_000,
  // Client-side query timeout — kills queries that never reach the server (dead
  // TCP connection). statement_timeout=15s is server-side and doesn't fire when
  // the TCP connection is silently dropped (common in WSL2→Neon serverless path).
  // 30s is above the 15s statement timeout so legitimate slow queries are killed
  // server-side first; this only fires for truly stuck connections.
  query_timeout: 30_000,
  keepAlive: true,
  keepAliveInitialDelayMillis: 5_000,
});

// Kill any query taking longer than 15s at the DB level so slow Neon queries
// can't hold a pool connection and starve other requests. 15s (not 8s) because
// Neon cold-start can add 10-15s of latency to the first query on a new connection.
pool.on("connect", (client) => {
  client.query("SET statement_timeout = 15000").catch(() => {});
});

pool.on("error", (err) => {
  // Log pool errors but don't crash — the next query will get a fresh connection.
  console.error("[DB] Pool error:", err.message);
});

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

// Only run warm-up and keepalive in the main thread — worker threads (GNN training)
// load this module transitively but do pure in-memory computation and don't need
// a live DB connection. Skipping these in workers prevents Neon from being
// overwhelmed with 5 simultaneous cold-start connection attempts.
if (isMainThread) {
  // Warm the pool at startup so the first API calls don't hit a cold Neon connection.
  pool.connect().then(client => {
    client.query("SELECT 1").then(() => {
      console.log("[DB] Neon connection warmed up");
      client.release();
    }).catch(() => client.release());
  }).catch(err => {
    console.warn("[DB] Startup warm-up failed (Neon may be cold-starting):", err.message);
  });

  // Keepalive ping every 60s — Neon suspends after 5 min inactivity on serverless.
  // Uses a reschedule-after-completion pattern (not setInterval) to prevent concurrent
  // keepalive loops from piling up and saturating the 5-connection pool.
  // Retry backoff: 5s+10s — total retry budget ~31s, well under the 60s interval.
  let keepaliveRunning = false;
  async function keepalivePing(): Promise<void> {
    if (keepaliveRunning) return; // previous ping still in progress — skip
    keepaliveRunning = true;
    try {
      await pool.query("SELECT 1");
    } catch (err: any) {
      console.warn("[DB] Keepalive ping failed:", err.message?.slice(0, 80));
      for (let attempt = 1; attempt <= 2; attempt++) {
        await new Promise(r => setTimeout(r, attempt * 5000)); // 5s, 10s
        try {
          const client = await pool.connect();
          await client.query("SELECT 1");
          client.release();
          console.log(`[DB] Keepalive reconnect succeeded (attempt ${attempt})`);
          break;
        } catch (err2: any) {
          console.warn(`[DB] Keepalive reconnect attempt ${attempt}/2 failed: ${err2.message?.slice(0, 80)}`);
        }
      }
    } finally {
      keepaliveRunning = false;
    }
  }
  setInterval(keepalivePing, 60 * 1000);
}
