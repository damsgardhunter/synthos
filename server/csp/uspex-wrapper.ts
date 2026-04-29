/**
 * USPEX (Universal Structure Predictor: Evolutionary Xtallography) wrapper.
 *
 * USPEX uses evolutionary algorithms (genetic operators: heredity, mutation,
 * lattice mutation, permutation) to search the enthalpy landscape.
 * Best for Stage 2-3 where we want to refine within known basins.
 *
 * Binary: USPEX (Python + Fortran) — reads INPUT.txt, generates structures
 * in results1/gatheredPOSCARS. Supports seed injection via Seeds/POSCARS.
 *
 * Reference: Oganov & Glass, J. Chem. Phys. 124 (2006) 244704
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import type { CSPCandidate, CSPEngine, CSPEngineConfig } from "./csp-types";
import { parseMultiplePOSCARS, writePOSCAR } from "./poscar-io";

const USPEX_BIN = process.env.USPEX_BIN ?? "/opt/uspex/USPEX";

function generateUSPEXInput(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  popSize: number,
  nGenerations: number,
  seed: number,
): string {
  const atomCounts = elements.map(e => Math.round(counts[e]));

  return `PARAMETERS EVOLUTIONARY ALGORITHM
{
-- System
atomType = ${elements.join(" ")}
numSpecies = ${atomCounts.join(" ")}
ExternalPressure = ${pressureGPa.toFixed(1)}

-- Population
populationSize = ${popSize}
numGenerations = ${nGenerations}
stopCrit = ${Math.max(3, Math.floor(nGenerations / 2))}

-- Variation operators
fracGene = 0.40
fracRand = 0.20
fracPerm = 0.10
fracAtomsMut = 0.10
fracLatMut = 0.10
fracRotMut = 0.10

-- Structure generation
minDistMatrixType = 1

-- Calculation
abinitioCode = 1
numParallelCalcs = 1
commandExecutable = echo skip

-- We only want structure generation, no actual relaxation
-- (our pipeline handles relaxation). Setting numCalcSteps=0
-- and using a dummy commandExecutable.

howManyOptSteps = 0

-- Seed
RandSeed = ${seed}
}
`;
}

export const uspexEngine: CSPEngine = {
  name: "uspex",

  isAvailable(): boolean {
    try {
      if (!fs.existsSync(USPEX_BIN)) return false;
      const result = execSync(`${USPEX_BIN} --version 2>&1 || true`, { timeout: 10000 });
      return result.toString().toLowerCase().includes("uspex");
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

    const popSize = Math.max(6, Math.min(30, Math.floor(config.maxStructures / 3)));
    const nGenerations = Math.max(2, Math.ceil(config.maxStructures / popSize));

    const inputContent = generateUSPEXInput(
      elements, counts, config.pressureGPa, popSize, nGenerations, baseSeed
    );
    fs.writeFileSync(path.join(workDir, "INPUT.txt"), inputContent);

    // Cross-seeding: write seed structures to Seeds/POSCARS
    // USPEX natively supports this — it uses seed structures as the
    // initial population alongside randomly generated ones.
    if (config.seedStructures && config.seedStructures.length > 0) {
      const seedDir = path.join(workDir, "Seeds");
      fs.mkdirSync(seedDir, { recursive: true });
      let seedPOSCARs = "";
      for (const seed of config.seedStructures.slice(0, popSize)) {
        seedPOSCARs += writePOSCAR(seed) + "\n";
      }
      fs.writeFileSync(path.join(seedDir, "POSCARS"), seedPOSCARs);
      console.log(`[USPEX] Injected ${Math.min(config.seedStructures.length, popSize)} seed structures via Seeds/POSCARS`);
    }

    console.log(`[USPEX] Starting EA: pop=${popSize}, gen=${nGenerations}, P=${config.pressureGPa} GPa, seed=${baseSeed}`);

    // Create Specific/ directory (USPEX requires it)
    fs.mkdirSync(path.join(workDir, "Specific"), { recursive: true });
    fs.writeFileSync(path.join(workDir, "Specific", "INCAR_1"), "# dummy\n");

    try {
      execSync(
        `cd ${workDir} && ${USPEX_BIN} -r 2>&1 || true`,
        { cwd: workDir, timeout: config.timeoutMs, maxBuffer: 10 * 1024 * 1024 }
      );
    } catch (err: any) {
      console.log(`[USPEX] Process ended: ${err.message?.slice(0, 80)}`);
    }

    // Harvest structures from results1/gatheredPOSCARS
    const candidates: CSPCandidate[] = [];
    const gatheredPath = path.join(workDir, "results1", "gatheredPOSCARS");
    if (fs.existsSync(gatheredPath)) {
      const content = fs.readFileSync(gatheredPath, "utf-8");
      candidates.push(...parseMultiplePOSCARS(content, "uspex", config.pressureGPa, baseSeed, 1));
    }

    // Also check extended gatheredPOSCARS paths
    for (const altPath of [
      path.join(workDir, "gatheredPOSCARS"),
      path.join(workDir, "results1", "POSCARS"),
    ]) {
      if (fs.existsSync(altPath) && candidates.length === 0) {
        const content = fs.readFileSync(altPath, "utf-8");
        candidates.push(...parseMultiplePOSCARS(content, "uspex", config.pressureGPa, baseSeed, 1));
      }
    }

    console.log(`[USPEX] Harvested ${candidates.length} structures`);
    return candidates;
  },
};
