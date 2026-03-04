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
  const removed = await storage.deduplicateSuperconductorCandidates();
  if (removed > 0) {
    log(`Deduplicated SC candidates: removed ${removed} duplicate rows`, "startup");
  }

  try {
    const { computeElectronicStructure, computePhononSpectrum, computeElectronPhononCoupling, assessCorrelationStrength } = await import("./learning/physics-engine");
    const allCandidates = await storage.getSuperconductorCandidates(2000);
    let corrected = 0;
    for (const c of allCandidates) {
      const currentTc = c.predictedTc ?? 0;
      const hasPhysics = c.electronPhononCoupling != null;

      const corrStrength = c.correlationStrength ?? 0;
      const hasMott = Array.isArray(c.competingPhases) && (c.competingPhases as any[]).some((p: any) => p.type === "Mott");
      const isMottInsulator = (hasMott && corrStrength > 0.7) || corrStrength > 0.85;
      const isStronglyCorrelated = corrStrength > 0.7;
      const electronic = computeElectronicStructure(c.formula, null);
      const phonon = computePhononSpectrum(c.formula, electronic);
      const coupling = computeElectronPhononCoupling(electronic, phonon, c.formula);

      const omegaLogK = coupling.omegaLog * 1.44;
      const denom = coupling.lambda - coupling.muStar * (1 + 0.62 * coupling.lambda);
      let eliashbergTc = 0;
      if (Math.abs(denom) > 1e-6 && coupling.lambda > 0.2) {
        const exponent = -1.04 * (1 + coupling.lambda) / denom;
        eliashbergTc = (omegaLogK / 1.2) * Math.exp(exponent);
        if (!Number.isFinite(eliashbergTc) || eliashbergTc < 0) eliashbergTc = 0;
      }

      const isNonMetallic = electronic.metallicity < 0.4;
      if (isNonMetallic) {
        eliashbergTc = eliashbergTc * Math.max(0.02, electronic.metallicity);
      } else if (isMottInsulator) {
        eliashbergTc = eliashbergTc * 0.05;
      } else if (isStronglyCorrelated) {
        eliashbergTc = eliashbergTc * 0.3;
      }

      const updates: any = {};
      if (hasPhysics && Math.abs(coupling.lambda - (c.electronPhononCoupling ?? 0)) > 0.3) {
        updates.electronPhononCoupling = coupling.lambda;
        updates.logPhononFrequency = coupling.omegaLog;
        updates.coulombPseudopotential = coupling.muStar;
      }

      if (!hasPhysics && currentTc > 200) {
        updates.electronPhononCoupling = coupling.lambda;
        updates.logPhononFrequency = coupling.omegaLog;
        updates.coulombPseudopotential = coupling.muStar;
      }

      let mcMillanCap = 350;
      if (coupling.lambda > 0.2 && eliashbergTc > 0) {
        const multiplier = coupling.lambda > 2.5 ? 1.3 : coupling.lambda > 2.0 ? 1.5 : coupling.lambda > 1.5 ? 1.8 : 2.0;
        mcMillanCap = Math.round(Math.min(350, eliashbergTc * multiplier));
      }
      if (coupling.lambda < 0.2 && currentTc > 30) {
        updates.predictedTc = Math.round(Math.max(1, currentTc * 0.05));
      } else if (currentTc > mcMillanCap) {
        const downBlend = currentTc > mcMillanCap * 2 ? 0.8 : 0.7;
        updates.predictedTc = Math.round((1 - downBlend) * currentTc + downBlend * eliashbergTc);
      } else if (eliashbergTc > 0 && eliashbergTc < currentTc && currentTc > 50) {
        const downBlend = eliashbergTc < currentTc * 0.5 ? 0.7 : 0.5;
        updates.predictedTc = Math.round((1 - downBlend) * currentTc + downBlend * eliashbergTc);
      }

      if ((c.upperCriticalField ?? 0) > 300) {
        updates.upperCriticalField = 300;
      }
      if ((c.coherenceLength ?? 0) > 0 && (c.coherenceLength ?? 0) < 1.0) {
        updates.coherenceLength = 1.0;
      }

      if (c.xgboostScore == null && c.neuralNetScore == null && (c.ensembleScore ?? 0) > 0.5) {
        updates.ensembleScore = Math.min(0.5, c.ensembleScore ?? 0.3);
      }

      const pressure = c.pressureGpa ?? 999;
      if (c.roomTempViable && pressure > 50) {
        updates.roomTempViable = false;
      }
      if (c.ambientPressureStable && pressure > 1) {
        updates.ambientPressureStable = false;
      }

      if (Object.keys(updates).length > 0) {
        await storage.updateSuperconductorCandidate(c.id, updates);
        corrected++;
      }
    }
    if (corrected > 0) {
      log(`Retroactively corrected ${corrected} inflated Tc candidates using Eliashberg physics`, "startup");
    }

    await db.execute(sql`UPDATE superconductor_candidates SET upper_critical_field = 300 WHERE upper_critical_field > 300`);
    await db.execute(sql`UPDATE superconductor_candidates SET coherence_length = 1.0 WHERE coherence_length > 0 AND coherence_length < 1.0`);
    await db.execute(sql`UPDATE superconductor_candidates SET ensemble_score = 0.5 WHERE xgboost_score IS NULL AND neural_net_score IS NULL AND ensemble_score > 0.5`);
    await db.execute(sql`UPDATE superconductor_candidates SET room_temp_viable = false WHERE room_temp_viable = true AND (predicted_tc < 293 OR pressure_gpa > 50 OR pressure_gpa IS NULL OR meissner_effect = false OR zero_resistance = false)`);
    await db.execute(sql`UPDATE superconductor_candidates SET ambient_pressure_stable = false WHERE ambient_pressure_stable = true AND (pressure_gpa > 1 OR pressure_gpa IS NULL)`);
    log("Applied bulk corrections: Hc2 cap 300T, coherence min 1nm, ensemble cap for unscored, pressure-gated roomTempViable/ambientPressure", "startup");

    const highTcCandidates = await storage.getSuperconductorCandidatesByTc(100);
    let bulkCorrected = 0;
    for (const c of highTcCandidates) {
      const tc = c.predictedTc ?? 0;
      if (tc <= 100) continue;
      const lambda = c.electronPhononCoupling ?? 0;
      const electronic2 = computeElectronicStructure(c.formula, null);
      const phonon2 = computePhononSpectrum(c.formula, electronic2);
      const coupling2 = computeElectronPhononCoupling(electronic2, phonon2, c.formula);
      const omK = coupling2.omegaLog * 1.44;
      const dn = coupling2.lambda - coupling2.muStar * (1 + 0.62 * coupling2.lambda);
      let eliTc = 0;
      if (Math.abs(dn) > 1e-6 && coupling2.lambda > 0.2) {
        eliTc = (omK / 1.2) * Math.exp(-1.04 * (1 + coupling2.lambda) / dn);
        if (!Number.isFinite(eliTc) || eliTc < 0) eliTc = 0;
      }
      if (electronic2.metallicity < 0.4) eliTc *= Math.max(0.02, electronic2.metallicity);
      const corr2 = assessCorrelationStrength(c.formula);
      if (corr2.ratio > 0.85) eliTc *= 0.05;
      else if (corr2.ratio > 0.7) eliTc *= 0.3;
      const mult = coupling2.lambda > 2.5 ? 1.2 : coupling2.lambda > 2.0 ? 1.3 : coupling2.lambda > 1.5 ? 1.5 : 1.8;
      const absMax = Math.min(350, Math.round(eliTc * mult));
      if (tc > absMax && absMax > 0) {
        const blendDown = 0.8;
        const newTc = Math.round((1 - blendDown) * tc + blendDown * eliTc);
        await storage.updateSuperconductorCandidate(c.id, { predictedTc: Math.min(newTc, absMax) });
        log(`Bulk Tc correction: ${c.formula} ${tc}K -> ${Math.min(newTc, absMax)}K (eliashberg=${Math.round(eliTc)}K, cap=${absMax}K, lambda=${coupling2.lambda.toFixed(2)})`, "startup");
        bulkCorrected++;
      }
    }
    if (bulkCorrected > 0) {
      log(`Bulk-corrected ${bulkCorrected} high-Tc candidates via Eliashberg physics`, "startup");
    }

    const tcQueryResult = await storage.getSuperconductorCandidatesByTc(1);
    const topTcNow = tcQueryResult.length > 0 ? (tcQueryResult[0].predictedTc ?? 0) : 0;
    const allForPhysCheck = await storage.getSuperconductorCandidatesByTc(50);
    const physicsValidated = allForPhysCheck.filter(c => (c.verificationStage ?? 0) >= 1 && c.electronPhononCoupling != null);
    const topPhysTcNow = physicsValidated.reduce((mx, c) => Math.max(mx, c.predictedTc ?? 0), 0);
    log(`After corrections: actual best Tc=${topTcNow}K, best physics-validated Tc=${topPhysTcNow}K`, "startup");

    const snapshots = await storage.getConvergenceSnapshots(500);
    let snapFixed = 0;
    for (const s of snapshots) {
      const needsFix = (s.bestTc ?? 0) > topTcNow + 10 || (s.bestPhysicsTc ?? 0) > topPhysTcNow + 10;
      if (needsFix) {
        await storage.deleteConvergenceSnapshotByCycle(s.cycle);
        await storage.insertConvergenceSnapshot({
          id: s.id + "-fix2",
          cycle: s.cycle,
          bestTc: Math.min(s.bestTc ?? 0, topTcNow),
          bestPhysicsTc: s.bestPhysicsTc != null ? Math.min(s.bestPhysicsTc, topPhysTcNow) : null,
          bestScore: s.bestScore,
          avgTopScore: s.avgTopScore,
          candidatesTotal: s.candidatesTotal,
          pipelinePassRate: s.pipelinePassRate,
          novelInsightCount: s.novelInsightCount,
          topFormula: s.topFormula,
          strategyFocus: s.strategyFocus,
          familyDiversity: s.familyDiversity,
          duplicatesSkipped: s.duplicatesSkipped,
        });
        snapFixed++;
      }
    }
    if (snapFixed > 0) {
      log(`Corrected ${snapFixed} convergence snapshots to match actual best Tc=${topTcNow}K, PhysTc=${topPhysTcNow}K`, "startup");
    }
  } catch (err: any) {
    log(`Tc correction error: ${err.message}`, "startup");
  }
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
        const { startEngine } = await import("./learning/engine");
        await startEngine();
        log("Learning engine auto-started", "startup");
      } catch (err: any) {
        log(`Engine auto-start failed: ${err.message}`, "startup");
      }
    },
  );
})();
