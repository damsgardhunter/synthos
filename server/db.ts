import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  max: 10,
  // Recycle idle connections after 10s — well before Neon terminates them (~60s TCP idle).
  idleTimeoutMillis: 10000,
  // Give Neon up to 20s to cold-start before failing.
  connectionTimeoutMillis: 20000,
  keepAlive: true,
  keepAliveInitialDelayMillis: 5000,
});

// Kill any query taking longer than 8s at the DB level so slow Neon queries
// can't hold a pool connection and starve other requests.
pool.on("connect", (client) => {
  client.query("SET statement_timeout = 8000").catch(() => {});
});

pool.on("error", (err) => {
  // Log pool errors but don't crash — the next query will get a fresh connection.
  console.error("[DB] Pool error:", err.message);
});

export const db = drizzle(pool, { schema });

// Warm the pool at startup so the first API calls don't hit a cold Neon connection.
pool.connect().then(client => {
  client.query("SELECT 1").then(() => {
    console.log("[DB] Neon connection warmed up");
    client.release();
  }).catch(() => client.release());
}).catch(err => {
  console.warn("[DB] Startup warm-up failed (Neon may be cold-starting):", err.message);
});

// Keepalive ping every 2 minutes — Neon suspends after 5 min inactivity;
// 2 min gives a comfortable margin even if one ping is missed.
setInterval(() => {
  pool.query("SELECT 1").catch(err => {
    console.warn("[DB] Keepalive ping failed:", err.message);
  });
}, 2 * 60 * 1000);
