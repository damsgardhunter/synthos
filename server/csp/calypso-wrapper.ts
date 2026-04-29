/**
 * CALYPSO (Crystal structure AnaLYsis by Particle Swarm Optimization) wrapper.
 *
 * CALYPSO uses PSO to explore the energy landscape. Best for Stage 2
 * basin refinement where we want to converge on low-enthalpy motifs.
 *
 * Binary: calypso.x (Fortran) — reads input.dat, generates POSCAR files
 * in results/ directory. We use ICode=8 (QE mode) but only harvest the
 * generated structures — relaxation is done by our own pipeline.
 *
 * Reference: Wang et al., Comput. Phys. Commun. 183 (2012) 2063
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import type { CSPCandidate, CSPEngine, CSPEngineConfig } from "./csp-types";
import { parseMultiplePOSCARS, writePOSCAR } from "./poscar-io";

const CALYPSO_BIN = process.env.CALYPSO_BIN ?? "/opt/calypso/calypso.x";

function generateCalypsoInput(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  popSize: number,
  nGenerations: number,
  seed: number,
): string {
  const atomCounts = elements.map(e => Math.round(counts[e]));
  const totalAtoms = atomCounts.reduce((s, n) => s + n, 0);

  return `SystemName = ${elements.join("")}
NumberOfSpecies = ${elements.length}
NameOfAtoms = ${elements.join(" ")}
NumberOfAtoms = ${atomCounts.join(" ")}
NumberOfFormula = 1 1
Volume = 0
@DistanceOfIon
${generateDistanceMatrix(elements)}
@End
Ialgo = 2
PsoRatio = 0.6
PopSize = ${popSize}
MaxStep = ${nGenerations}
ICode = 1
NumberOfLocalOptim = 0
Pressure = ${pressureGPa.toFixed(1)}
Fmax = 0.01
RandomSeed = ${seed}
Split = T
PSTRESS = ${(pressureGPa * 10).toFixed(1)}
`;
}

function generateDistanceMatrix(elements: string[]): string {
  const radii: Record<string, number> = {
    H: 0.31, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66,
    Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, Ca: 1.76, Sc: 1.70,
    Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26,
    Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19,
    Se: 1.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54,
    Sn: 1.39, Sb: 1.39, Te: 1.38, Ba: 2.15, La: 2.07, Hf: 1.75,
    Ta: 1.70, W: 1.62, Re: 1.51, Pb: 1.46, Bi: 1.48, Th: 2.06,
  };
  const lines: string[] = [];
  for (const el1 of elements) {
    const row: string[] = [];
    for (const el2 of elements) {
      const r1 = radii[el1] ?? 1.2;
      const r2 = radii[el2] ?? 1.2;
      row.push(((r1 + r2) * 0.8).toFixed(2));
    }
    lines.push(row.join("  "));
  }
  return lines.join("\n");
}

export const calypsoEngine: CSPEngine = {
  name: "calypso",

  isAvailable(): boolean {
    try {
      if (!fs.existsSync(CALYPSO_BIN)) return false;
      execSync(`${CALYPSO_BIN} --version 2>&1 || true`, { timeout: 5000 });
      return true;
    } catch {
      return false;
    }
  },

  async generateStructures(
    elements: string[],
    counts: Record<string, number>,
    config: CSPEngineConfig,
  ): Promise<CSPCandidate[]> {
    const baseSeed = config.baseSeed ?? Math.floor(Math.random() * 1e8);
    const workDir = config.workDir;
    fs.mkdirSync(workDir, { recursive: true });

    // CALYPSO parameters scale with budget
    const popSize = Math.max(6, Math.min(30, Math.floor(config.maxStructures / 3)));
    const nGenerations = Math.max(2, Math.ceil(config.maxStructures / popSize));

    const inputContent = generateCalypsoInput(
      elements, counts, config.pressureGPa, popSize, nGenerations, baseSeed
    );
    fs.writeFileSync(path.join(workDir, "input.dat"), inputContent);

    // Inject seed structures if available (cross-seeding from other engines)
    if (config.seedStructures && config.seedStructures.length > 0) {
      const seedDir = path.join(workDir, "results");
      fs.mkdirSync(seedDir, { recursive: true });
      for (let i = 0; i < Math.min(config.seedStructures.length, popSize); i++) {
        const poscar = writePOSCAR(config.seedStructures[i]);
        fs.writeFileSync(path.join(seedDir, `POSCAR_${i + 1}`), poscar);
      }
      console.log(`[CALYPSO] Injected ${Math.min(config.seedStructures.length, popSize)} seed structures`);
    }

    console.log(`[CALYPSO] Starting PSO: pop=${popSize}, gen=${nGenerations}, P=${config.pressureGPa} GPa, seed=${baseSeed}`);

    try {
      execSync(
        `cd ${workDir} && ${CALYPSO_BIN} 2>&1 || true`,
        { cwd: workDir, timeout: config.timeoutMs, maxBuffer: 10 * 1024 * 1024 }
      );
    } catch (err: any) {
      console.log(`[CALYPSO] Process ended: ${err.message?.slice(0, 80)}`);
    }

    // Harvest generated structures from results/
    const candidates: CSPCandidate[] = [];
    const resultsDir = path.join(workDir, "results");
    if (fs.existsSync(resultsDir)) {
      const files = fs.readdirSync(resultsDir).filter(f => f.startsWith("POSCAR"));
      for (const file of files) {
        try {
          const content = fs.readFileSync(path.join(resultsDir, file), "utf-8");
          const parsed = parseMultiplePOSCARS(content, "calypso", config.pressureGPa, baseSeed, 1);
          candidates.push(...parsed);
        } catch {}
      }
    }

    // Also check for gatheredPOSCARS (some CALYPSO versions)
    const gatheredPath = path.join(workDir, "gatheredPOSCARS");
    if (fs.existsSync(gatheredPath)) {
      const content = fs.readFileSync(gatheredPath, "utf-8");
      const parsed = parseMultiplePOSCARS(content, "calypso", config.pressureGPa, baseSeed, 1);
      candidates.push(...parsed);
    }

    console.log(`[CALYPSO] Harvested ${candidates.length} structures`);
    return candidates;
  },
};
