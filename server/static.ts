import express, { type Express } from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = typeof import.meta?.url === "string" ? path.dirname(fileURLToPath(import.meta.url)) : process.cwd();

export function serveStatic(app: Express) {
  // In ESM __dirname points to dist/, so "public" resolves to dist/public.
  // In CJS fallback __dirname is cwd (project root), so we need dist/public explicitly.
  const distPath = typeof import.meta?.url === "string"
    ? path.resolve(__dirname, "public")
    : path.resolve(__dirname, "dist/public");
  if (!fs.existsSync(distPath)) {
    throw new Error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`,
    );
  }

  app.use(express.static(distPath));

  // fall through to index.html if the file doesn't exist
  app.use("/{*path}", (_req, res) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}
