import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";
import { isMainThread } from "worker_threads";

// DATABASE_URL must point to the Neon *pooler* endpoint (hostname contains "-pooler").
// The pooler (pgBouncer) is always-on and never scales to zero, eliminating cold-start
// connection timeouts. Direct endpoint connections are unreliable even on paid plans.
const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  // 10 connections per process — pooler handles fan-out to actual compute,
  // so higher limits are safe and reduce queuing under bursts.
  max: 10,
  // Recycle idle connections after 60s — pooler connections are cheap to re-establish.
  idleTimeoutMillis: 60_000,
  // Pooler is always warm — 3s is generous; should connect in <100ms normally.
  connectionTimeoutMillis: 3_000,
  // Kill stuck queries that never return — pooler doesn't buffer forever.
  query_timeout: 30_000,
  keepAlive: true,
  keepAliveInitialDelayMillis: 5_000,
});

pool.on("error", (err) => {
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
