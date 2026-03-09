import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { serveStatic } from "./static";
import { createServer } from "http";
import { db } from "./db";
import { sql } from "drizzle-orm";

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
  const { seedDatabase } = await import("./seed");
  await seedDatabase();
  const { storage } = await import("./storage");

  await registerRoutes(httpServer, app);

  app.use((err: any, _req: Request, res: Response, next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    console.error("Internal Server Error:", err);

    if (res.headersSent) {
      return next(err);
    }

    return res.status(status).json({ message });
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (process.env.NODE_ENV === "production") {
    serveStatic(app);
  } else {
    const { setupVite } = await import("./vite");
    await setupVite(httpServer, app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || "5000", 10);
  httpServer.listen(
    {
      port,
      host: "0.0.0.0",
      reusePort: true,
    },
    async () => {
      log(`serving on port ${port}`);

      try {
        const removed = await storage.deduplicateSuperconductorCandidates();
        if (removed > 0) log(`Deduplicated SC candidates: removed ${removed} duplicate rows`, "startup");
      } catch {}

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
        await startEngine();
        log("Learning engine auto-started", "startup");
      } catch (err: any) {
        log(`Engine auto-start failed: ${err.message}`, "startup");
      }
    },
  );
})();
