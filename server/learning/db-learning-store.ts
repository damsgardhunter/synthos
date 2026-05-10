/**
 * db-learning-store.ts — Buffered write-through for persistent learning tables
 *
 * Each store follows the same pattern proven in engine.ts's _logWriteBuffer:
 *   - In-memory buffer accumulates entries
 *   - Flushed every FLUSH_INTERVAL_MS or when batch hits FLUSH_BATCH_SIZE
 *   - Circuit breaker backs off after consecutive failures
 *   - Non-blocking: callers never await the DB write
 *
 * The in-memory caches in engine.ts / cross-engine-hub.ts / etc. remain the
 * hot path for reads.  These tables are the durable backing store so that
 * knowledge survives eviction and restarts.
 */

import { db } from "../db";
import { isConnectionError } from "../db";
import {
  crossEngineInsightsLog,
  pressureObservationsLog,
  formulaScreenLog,
  engineInsightsLog,
  cycleDiagnosticReportsLog,
} from "@shared/schema";
import type {
  InsertCrossEngineInsightLog,
  InsertPressureObservationLog,
  InsertFormulaScreenLogEntry,
  InsertEngineInsightLog,
  InsertCycleDiagnosticReportLog,
} from "@shared/schema";
import { sql, desc } from "drizzle-orm";

// ── Generic buffered writer ────────────────────────────────────────────────

interface BufferedWriterOpts<T> {
  name: string;
  flushIntervalMs?: number;
  flushBatchSize?: number;
  failThreshold?: number;
  backoffMs?: number;
  insertFn: (batch: T[]) => Promise<void>;
}

class BufferedWriter<T> {
  private buffer: T[] = [];
  private failures = 0;
  private backoffUntil = 0;
  private readonly name: string;
  private readonly flushBatchSize: number;
  private readonly failThreshold: number;
  private readonly backoffMs: number;
  private readonly insertFn: (batch: T[]) => Promise<void>;

  constructor(opts: BufferedWriterOpts<T>) {
    this.name = opts.name;
    this.flushBatchSize = opts.flushBatchSize ?? 50;
    this.failThreshold = opts.failThreshold ?? 5;
    this.backoffMs = opts.backoffMs ?? 30_000;
    this.insertFn = opts.insertFn;

    setInterval(() => { this.flush().catch(() => {}); }, opts.flushIntervalMs ?? 3_000);
  }

  push(entry: T): void {
    if (Date.now() < this.backoffUntil) return;
    this.buffer.push(entry);
    if (this.buffer.length >= this.flushBatchSize) {
      this.flush().catch(() => {});
    }
  }

  pushMany(entries: T[]): void {
    if (Date.now() < this.backoffUntil || entries.length === 0) return;
    this.buffer.push(...entries);
    if (this.buffer.length >= this.flushBatchSize) {
      this.flush().catch(() => {});
    }
  }

  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;
    if (Date.now() < this.backoffUntil) { this.buffer.length = 0; return; }
    const batch = this.buffer.splice(0);
    try {
      await this.insertFn(batch);
      this.failures = 0;
    } catch (e: any) {
      if (isConnectionError(e)) {
        await new Promise(r => setTimeout(r, 500));
        try { await this.insertFn(batch); this.failures = 0; return; } catch { /* fall through */ }
      }
      this.failures++;
      if (this.failures >= this.failThreshold) {
        this.backoffUntil = Date.now() + this.backoffMs;
        console.error(`[${this.name}] Circuit open for ${this.backoffMs / 1000}s after ${this.failures} failures`);
        this.failures = 0;
      } else {
        console.error(`[${this.name}] Write failed: ${e?.message?.slice(0, 120) ?? "unknown"}`);
      }
    }
  }
}

// ── Writers for each table ─────────────────────────────────────────────────

export const crossEngineWriter = new BufferedWriter<InsertCrossEngineInsightLog>({
  name: "CrossEngineInsights",
  insertFn: async (batch) => { await db.insert(crossEngineInsightsLog).values(batch); },
});

export const pressureObsWriter = new BufferedWriter<InsertPressureObservationLog>({
  name: "PressureObservations",
  insertFn: async (batch) => { await db.insert(pressureObservationsLog).values(batch); },
});

export const formulaScreenWriter = new BufferedWriter<InsertFormulaScreenLogEntry>({
  name: "FormulaScreenLog",
  insertFn: async (batch) => { await db.insert(formulaScreenLog).values(batch); },
});

export const engineInsightWriter = new BufferedWriter<InsertEngineInsightLog>({
  name: "EngineInsights",
  insertFn: async (batch) => { await db.insert(engineInsightsLog).values(batch); },
});

export const cycleDiagWriter = new BufferedWriter<InsertCycleDiagnosticReportLog>({
  name: "CycleDiagnostics",
  insertFn: async (batch) => { await db.insert(cycleDiagnosticReportsLog).values(batch); },
});

// ── Read helpers — load from DB on startup to warm in-memory caches ────────

export async function loadRecentCrossEngineInsights(limit = 100_000): Promise<{ formula: string; engine: string; insightData: any; createdAt: Date }[]> {
  try {
    return await db.select().from(crossEngineInsightsLog).orderBy(desc(crossEngineInsightsLog.createdAt)).limit(limit);
  } catch { return []; }
}

export async function loadRecentPressureObservations(limit = 100_000): Promise<{ formula: string; pressureGpa: number; tc: number; stable: boolean | null; enthalpy: number | null }[]> {
  try {
    return await db.select({
      formula: pressureObservationsLog.formula,
      pressureGpa: pressureObservationsLog.pressureGpa,
      tc: pressureObservationsLog.tc,
      stable: pressureObservationsLog.stable,
      enthalpy: pressureObservationsLog.enthalpy,
    }).from(pressureObservationsLog).orderBy(desc(pressureObservationsLog.createdAt)).limit(limit);
  } catch { return []; }
}

export async function loadScreenedFormulas(): Promise<Set<string>> {
  try {
    const rows = await db.select({ formula: formulaScreenLog.formula }).from(formulaScreenLog);
    return new Set(rows.map(r => r.formula));
  } catch { return new Set(); }
}

export async function loadRejectedFormulas(): Promise<Map<string, { reason: string; tc: number; lambda?: number; timestamp: number }>> {
  try {
    const rows = await db.select().from(formulaScreenLog).orderBy(desc(formulaScreenLog.createdAt)).limit(1_000_000);
    const map = new Map<string, { reason: string; tc: number; lambda?: number; timestamp: number }>();
    for (const r of rows) {
      if (r.status === "rejected" && !map.has(r.formula)) {
        map.set(r.formula, {
          reason: r.reason ?? "unknown",
          tc: r.tc ?? 0,
          lambda: r.lambda ?? undefined,
          timestamp: r.createdAt.getTime(),
        });
      }
    }
    return map;
  } catch { return new Map(); }
}

export async function loadRecentInsights(limit = 50_000): Promise<string[]> {
  try {
    const rows = await db.select({ text: engineInsightsLog.insightText })
      .from(engineInsightsLog)
      .orderBy(desc(engineInsightsLog.createdAt))
      .limit(limit);
    return rows.map(r => r.text).reverse(); // oldest-first for ring buffer consistency
  } catch { return []; }
}

export async function getFormulaScreenCount(): Promise<number> {
  try {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(formulaScreenLog);
    return Number(count);
  } catch { return 0; }
}
