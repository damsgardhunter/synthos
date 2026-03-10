import express, { type Request, Response, NextFunction } from "express";
import { createServer } from "http";

const app = express();
const httpServer = createServer(app);

declare module "http" {
  interface IncomingMessage {
    rawBody: unknown;
  }
}

app.use(
  express.json({
    verify: (req, _res, buf) => {
      req.rawBody = buf;
    },
  }),
);

app.use(express.urlencoded({ extended: false }));

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  const port = parseInt(process.env.PORT || "5000", 10);

  let initialized = false;

  app.get("/api/health", (_req, res) => {
    res.json({ status: initialized ? "ready" : "starting", port });
  });

  app.use((req, res, next) => {
    if (initialized) return next();
    if (req.path.startsWith("/api/")) return next();
    res.status(200).send(`<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MatSci Supercomputer - Starting</title>
<meta http-equiv="refresh" content="10">
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0a0a0a;color:#e0e0e0;font-family:'Open Sans',system-ui,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh}
.container{text-align:center;max-width:480px;padding:2rem}.title{font-size:1.5rem;font-weight:700;margin-bottom:1rem;color:#fff}
.subtitle{font-size:0.95rem;color:#999;margin-bottom:2rem}
.spinner{width:48px;height:48px;border:4px solid #333;border-top-color:#3b82f6;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 1.5rem}
@keyframes spin{to{transform:rotate(360deg)}}</style></head>
<body><div class="container"><div class="spinner"></div><div class="title">Initializing MatSci Supercomputer</div>
<div class="subtitle">Loading ML models, physics engines, and database. This page will refresh automatically.</div></div></body></html>`);
  });

  httpServer.listen(
    {
      port,
      host: "0.0.0.0",
      reusePort: true,
    },
    () => {
      log(`port ${port} open (initializing...)`);
      setTimeout(initializeApp, 100);
    },
  );

  const yield_ = () => new Promise<void>(r => setTimeout(r, 50));

  async function initializeApp() {
    try {
      await yield_();
      const t2 = Date.now();
      if (process.env.NODE_ENV === "production") {
        const { serveStatic } = await import("./static");
        serveStatic(app);
      } else {
        const { setupVite } = await import("./vite");
        await setupVite(httpServer, app);
      }
      console.log(`[startup] vite/static: ${Date.now() - t2}ms`);
      initialized = true;
      log(`frontend ready on port ${port}`);

      await yield_();
      const t0 = Date.now();
      const { seedDatabase } = await import("./seed");
      await yield_();
      await seedDatabase();
      console.log(`[startup] seedDatabase: ${Date.now() - t0}ms`);
      await yield_();
      const { storage: storageInstance } = await import("./storage");

      await yield_();
      const t1 = Date.now();
      const { registerRoutes } = await import("./routes");
      await yield_();
      await registerRoutes(httpServer, app);
      console.log(`[startup] registerRoutes: ${Date.now() - t1}ms`);

      app.use((err: any, _req: Request, res: Response, next: NextFunction) => {
        const status = err.status || err.statusCode || 500;
        const message = err.message || "Internal Server Error";
        console.error("Internal Server Error:", err);
        if (res.headersSent) {
          return next(err);
        }
        return res.status(status).json({ message });
      });

      log(`serving on port ${port}`);

      try {
        const removed = await storageInstance.deduplicateSuperconductorCandidates();
        if (removed > 0) log(`Deduplicated SC candidates: removed ${removed} duplicate rows`, "startup");
      } catch {}

      const { db } = await import("./db");
      const { sql } = await import("drizzle-orm");

      try {
        await db.execute(sql`UPDATE superconductor_candidates SET upper_critical_field = 300 WHERE upper_critical_field > 300`);
        await db.execute(sql`UPDATE superconductor_candidates SET coherence_length = 1.0 WHERE coherence_length > 0 AND coherence_length < 1.0`);
        await db.execute(sql`UPDATE superconductor_candidates SET ensemble_score = 0.5 WHERE xgboost_score IS NULL AND neural_net_score IS NULL AND ensemble_score > 0.5`);
        await db.execute(sql`UPDATE superconductor_candidates SET ensemble_score = 0.95 WHERE ensemble_score > 0.95`);
        await db.execute(sql`UPDATE superconductor_candidates SET room_temp_viable = false WHERE room_temp_viable = true AND (predicted_tc < 293 OR pressure_gpa > 50 OR pressure_gpa IS NULL OR meissner_effect = false OR zero_resistance = false)`);
        await db.execute(sql`UPDATE superconductor_candidates SET ambient_pressure_stable = false WHERE ambient_pressure_stable = true AND (pressure_gpa > 1 OR pressure_gpa IS NULL)`);

        const overComplex = await db.execute(sql`DELETE FROM superconductor_candidates WHERE array_length(regexp_split_to_array(formula, '[A-Z]'), 1) - 1 > 5`);
        const deletedCount = overComplex.rowCount ?? 0;
        if (deletedCount > 0) log(`Purged ${deletedCount} over-complex candidates (>5 elements) on startup`, "startup");

        log("Applied bulk corrections on startup", "startup");
      } catch (err: any) {
        log(`Bulk correction error: ${err.message?.slice(0, 100)}`, "startup");
      }

      try {
        const logCount = await db.execute(sql`SELECT COUNT(*) as cnt FROM research_logs`);
        const totalLogs = Number(logCount.rows[0]?.cnt ?? 0);
        if (totalLogs > 20000) {
          await db.execute(sql`DELETE FROM research_logs WHERE timestamp < (SELECT timestamp FROM research_logs ORDER BY timestamp DESC OFFSET 10000 LIMIT 1)`);
          log(`Trimmed research_logs from ${totalLogs} to ~10000`, "startup");
        }
      } catch {}

      try {
        const { startEngine } = await import("./learning/engine");
        startEngine().then(() => {
          log("Learning engine auto-started", "startup");
        }).catch((err: any) => {
          log(`Engine auto-start failed: ${err.message}`, "startup");
        });
      } catch (err: any) {
        log(`Engine import failed: ${err.message}`, "startup");
      }
    } catch (err: any) {
      console.error("[startup] Fatal initialization error:", err);
    }
  }
})();
