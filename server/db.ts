import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  max: 10,
  // Keep idle connections for 2 minutes — longer than the 60s keepalive ping interval
  // so there's always a live connection in the pool and pg-pool never needs to open
  // a new TCP connection mid-poll (which risks ETIMEDOUT on Neon cold-start).
  idleTimeoutMillis: 120_000,
  // Give Neon up to 60s to cold-start; Neon free-tier can take 30-45s after suspension.
  connectionTimeoutMillis: 60_000,
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
    err?.constructor?.name === "AggregateError";
}

// Warm the pool at startup so the first API calls don't hit a cold Neon connection.
pool.connect().then(client => {
  client.query("SELECT 1").then(() => {
    console.log("[DB] Neon connection warmed up");
    client.release();
  }).catch(() => client.release());
}).catch(err => {
  console.warn("[DB] Startup warm-up failed (Neon may be cold-starting):", err.message);
});

// Keepalive ping every 45s — Neon suspends after 5 min inactivity on serverless.
// On failure, retry up to 3× with backoff so a single cold-start (30-60s) doesn't
// leave the pool dead until the next ping cycle 60s later.
async function keepalivePing(): Promise<void> {
  try {
    await pool.query("SELECT 1");
  } catch (err: any) {
    console.warn("[DB] Keepalive ping failed:", err.message?.slice(0, 80));
    for (let attempt = 1; attempt <= 3; attempt++) {
      await new Promise(r => setTimeout(r, attempt * 5000)); // 5s, 10s, 15s
      try {
        const client = await pool.connect();
        await client.query("SELECT 1");
        client.release();
        console.log(`[DB] Keepalive reconnect succeeded (attempt ${attempt})`);
        return;
      } catch (err2: any) {
        console.warn(`[DB] Keepalive reconnect attempt ${attempt}/3 failed: ${err2.message?.slice(0, 80)}`);
      }
    }
  }
}
setInterval(keepalivePing, 45 * 1000);
