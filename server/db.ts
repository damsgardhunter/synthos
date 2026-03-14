import { drizzle } from "drizzle-orm/node-postgres";
import pg from "pg";
import * as schema from "@shared/schema";

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  max: 20,
  idleTimeoutMillis: 30000,
  // Neon serverless cold-starts can take 10-20s — give it 30s to wake up.
  connectionTimeoutMillis: 30000,
  // Keep TCP connections alive so Neon doesn't auto-suspend between requests.
  keepAlive: true,
  keepAliveInitialDelayMillis: 10000,
});

pool.on("error", (err) => {
  // Log pool errors but don't crash — the next request will retry.
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
