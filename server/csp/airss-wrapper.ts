/**
 * AIRSS (Ab Initio Random Structure Searching) wrapper.
 *
 * Uses `buildcell` to generate sensible random structures.
 * buildcell reads a `.cell` file with composition/symmetry constraints
 * and outputs a randomized structure in the same format.
 *
 * AIRSS is ideal for Stage 1 broad exploration because:
 * - buildcell is extremely fast (~10ms per structure)
 * - Generates physically sensible structures (respects min distances, symmetry)
 * - Good at sampling diverse structural motifs
 * - No evolutionary bias — true exploration of configuration space
 *
 * Reference: Pickard & Needs, J. Phys.: Condens. Matter 23 (2011) 053201
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import type { CSPCandidate, CSPEngine, CSPEngineConfig } from "./csp-types";
import { latticeParamsToVectors, parsePOSCAR } from "./poscar-io";
import { cellVolumeFromVectors } from "./csp-types";

// Binary paths — try env var, then common install locations
const BUILDCELL_BIN = process.env.AIRSS_BIN
  ?? process.env.BUILDCELL_BIN
  ?? "/usr/local/bin/buildcell";

// Covalent radii for minimum separation (Angstrom)
const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71,
  O: 0.66, F: 0.57, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07,
  S: 1.05, Cl: 1.02, K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53,
  Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22,
  Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20, Rb: 2.20, Sr: 1.95,
  Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54, Ru: 1.46, Rh: 1.42, Pd: 1.39,
  Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39,
  Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04, Hf: 1.75, Ta: 1.70, W: 1.62,
  Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36, Pb: 1.46, Bi: 1.48,
  Th: 2.06,
};

// Estimated volume per atom (Angstrom^3/atom) for target volume calculation
const VOL_PER_ATOM: Record<string, number> = {
  H: 3.5, Li: 21.3, Be: 8.1, B: 7.3, C: 5.7, N: 6.0, O: 5.5,
  Na: 39.5, Mg: 23.2, Al: 16.6, Si: 20.0, P: 17.4, S: 15.6,
  K: 75.5, Ca: 43.6, Sc: 25.0, Ti: 17.6, V: 13.8, Cr: 11.9,
  Mn: 12.2, Fe: 11.8, Co: 11.1, Ni: 10.9, Cu: 11.8, Zn: 15.2,
  Ga: 19.6, Ge: 22.6, As: 21.5, Se: 27.3, Rb: 92.0, Sr: 56.4,
  Y: 33.0, Zr: 23.3, Nb: 18.0, Mo: 15.6, Ru: 13.6, Rh: 13.7,
  Pd: 14.7, Ag: 17.1, Sn: 27.1, Sb: 30.3, Te: 34.0, Cs: 117.0,
  Ba: 63.4, La: 37.2, Ce: 34.4, Hf: 22.3, Ta: 18.0, W: 15.8,
  Re: 14.7, Pb: 30.3, Bi: 35.4, Th: 32.8,
};

function estimateVolume(elements: string[], counts: Record<string, number>, _pressureGPa: number): number {
  let vol = 0;
  for (const el of elements) {
    const v = VOL_PER_ATOM[el] ?? 15;
    vol += v * Math.round(counts[el]);
  }
  // Do NOT compress for pressure here — buildcell needs ambient-like volume
  // to successfully pack atoms. The DFT pipeline handles compression later
  // via Murnaghan EOS / literature lattice override.
  return vol;
}

/**
 * Generate the AIRSS .cell input file.
 */
function generateCellInput(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  seed: number,
): string {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  const targetVol = estimateVolume(elements, counts, pressureGPa);

  // Target volume with ±20% range for buildcell to explore
  const volMin = targetVol * 0.80;
  const volMax = targetVol * 1.20;

  let cell = `#TARGVOL=${((volMin + volMax) / 2).toFixed(1)}\n`;

  // Minimum separations — use scaled covalent radii.
  // Keep global MINSEP modest so buildcell can pack hydrogen-rich compounds.
  const hasH = elements.includes("H");
  const globalMinSep = hasH ? 0.8 : 1.2;
  cell += `#MINSEP=${globalMinSep.toFixed(2)}\n`;

  // Element-specific minimum separations (more relaxed for H pairs)
  for (let i = 0; i < elements.length; i++) {
    for (let j = i; j < elements.length; j++) {
      const ri = COVALENT_RADII[elements[i]] ?? 1.2;
      const rj = COVALENT_RADII[elements[j]] ?? 1.2;
      // H-H can be very close in cage structures (~1.0-1.5 Å)
      const isHH = elements[i] === "H" && elements[j] === "H";
      const isMH = elements[i] === "H" || elements[j] === "H";
      const scale = isHH ? 0.5 : (isMH ? 0.6 : 0.75);
      const minDist = (ri + rj) * scale;
      cell += `#MINSEP=${elements[i]}-${elements[j]}=${minDist.toFixed(2)}\n`;
    }
  }

  // Symmetry — allow low-symmetry SGs (1-4 ops) for easier packing of
  // complex stoichiometries like H9Na2Y, plus high-symmetry for diversity.
  cell += `#SYMMOPS=1-48\n`;

  // Volume range
  cell += `#VARVOL=${volMin.toFixed(1)}-${volMax.toFixed(1)}\n`;

  // Seed for reproducibility
  cell += `#SEED=${seed}\n`;

  // Lattice block (buildcell needs a template cell to randomize)
  const a = Math.pow(targetVol, 1 / 3);
  cell += `\n%BLOCK LATTICE_CART\n`;
  cell += `${a.toFixed(4)} 0.0000 0.0000\n`;
  cell += `0.0000 ${a.toFixed(4)} 0.0000\n`;
  cell += `0.0000 0.0000 ${a.toFixed(4)}\n`;
  cell += `%ENDBLOCK LATTICE_CART\n`;

  // Atomic positions block (buildcell randomizes these)
  cell += `\n%BLOCK POSITIONS_FRAC\n`;
  for (const el of elements) {
    for (let i = 0; i < Math.round(counts[el]); i++) {
      cell += `${el} 0.0 0.0 0.0 # ${el}${i + 1}\n`;
    }
  }
  cell += `%ENDBLOCK POSITIONS_FRAC\n`;

  // Pressure constraint (for enthalpy optimization context)
  if (pressureGPa > 0) {
    cell += `\n%BLOCK EXTERNAL_PRESSURE\n`;
    cell += `${pressureGPa.toFixed(1)} 0.0 0.0\n`;
    cell += `0.0 ${pressureGPa.toFixed(1)} 0.0\n`;
    cell += `0.0 0.0 ${pressureGPa.toFixed(1)}\n`;
    cell += `%ENDBLOCK EXTERNAL_PRESSURE\n`;
  }

  return cell;
}

/**
 * Parse AIRSS .cell output format into a CSPCandidate.
 */
function parseCellOutput(
  content: string,
  elements: string[],
  seed: number,
  pressureGPa: number,
  stage: 1 | 2 | 3,
): CSPCandidate | null {
  // Parse lattice vectors
  const latMatch = content.match(
    /%BLOCK\s+LATTICE_CART\s*\n([\s\S]*?)%ENDBLOCK\s+LATTICE_CART/i
  );
  if (!latMatch) return null;

  const latLines = latMatch[1].trim().split("\n");
  if (latLines.length < 3) return null;

  const vecs: [number, number, number][] = [];
  for (const line of latLines) {
    const parts = line.trim().split(/\s+/).map(Number);
    if (parts.length >= 3 && !parts.some(isNaN)) {
      vecs.push([parts[0], parts[1], parts[2]]);
    }
  }
  if (vecs.length < 3) return null;

  // Parse fractional positions
  const posMatch = content.match(
    /%BLOCK\s+POSITIONS_FRAC\s*\n([\s\S]*?)%ENDBLOCK\s+POSITIONS_FRAC/i
  );
  if (!posMatch) return null;

  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];
  for (const line of posMatch[1].trim().split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 4 && /^[A-Z][a-z]?$/.test(parts[0])) {
      const x = parseFloat(parts[1]);
      const y = parseFloat(parts[2]);
      const z = parseFloat(parts[3]);
      if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
        positions.push({ element: parts[0], x: ((x % 1) + 1) % 1, y: ((y % 1) + 1) % 1, z: ((z % 1) + 1) % 1 });
      }
    }
  }

  if (positions.length === 0) return null;

  const a = Math.sqrt(vecs[0][0] ** 2 + vecs[0][1] ** 2 + vecs[0][2] ** 2);
  const b = Math.sqrt(vecs[1][0] ** 2 + vecs[1][1] ** 2 + vecs[1][2] ** 2);
  const c = Math.sqrt(vecs[2][0] ** 2 + vecs[2][1] ** 2 + vecs[2][2] ** 2);
  const volume = cellVolumeFromVectors(vecs);

  return {
    latticeA: a,
    latticeB: b,
    latticeC: c,
    cOverA: c / a,
    positions,
    prototype: "AIRSS-buildcell",
    crystalSystem: "unknown",
    spaceGroup: "",
    source: `AIRSS buildcell (seed=${seed})`,
    confidence: 0.4,
    isMetallic: null,
    sourceEngine: "airss",
    generationStage: stage,
    seed,
    pressureGPa,
    cellVolume: volume,
    cellVectors: vecs,
    latticeParams: { a, b, c, alpha: 90, beta: 90, gamma: 90 },
    relaxationLevel: "raw",
  };
}

// ---------------------------------------------------------------------------
// AIRSS Engine
// ---------------------------------------------------------------------------

let _airssAvailable: boolean | null = null;

export const airssEngine: CSPEngine = {
  name: "airss",

  isAvailable(): boolean {
    if (_airssAvailable !== null) return _airssAvailable;
    try {
      execSync(`${BUILDCELL_BIN} < /dev/null 2>&1 || true`, { timeout: 5000 });
      _airssAvailable = true;
      console.log(`[AIRSS] buildcell found at ${BUILDCELL_BIN}`);
    } catch {
      _airssAvailable = false;
    }
    return _airssAvailable;
  },

  async generateStructures(
    elements: string[],
    counts: Record<string, number>,
    config: CSPEngineConfig,
  ): Promise<CSPCandidate[]> {
    const candidates: CSPCandidate[] = [];
    const baseSeed = config.baseSeed ?? Math.floor(Math.random() * 1e8);
    const workDir = config.workDir;

    fs.mkdirSync(workDir, { recursive: true });

    console.log(`[AIRSS] Generating ${config.maxStructures} structures for ${elements.join("")} at ${config.pressureGPa} GPa (baseSeed=${baseSeed})`);

    for (let i = 0; i < config.maxStructures; i++) {
      const seed = baseSeed + i;
      try {
        const cellInput = generateCellInput(elements, counts, config.pressureGPa, seed);
        const inputPath = path.join(workDir, `airss_${seed}.cell`);
        fs.writeFileSync(inputPath, cellInput);

        const result = execSync(
          `${BUILDCELL_BIN} < ${inputPath} 2>&1`,
          { cwd: workDir, timeout: config.timeoutMs / config.maxStructures, maxBuffer: 1024 * 1024 }
        );

        const output = result.toString("utf-8");
        const candidate = parseCellOutput(output, elements, seed, config.pressureGPa, 1);
        if (candidate) {
          if (config.seedStructures && config.seedStructures.length > 0) {
            candidate.parentSeeds = config.seedStructures.map(s => s.seed);
          }
          candidates.push(candidate);
        } else if (i === 0) {
          // Log first parse failure to diagnose format issues
          console.log(`[AIRSS] buildcell ran but parse failed. Output (first 300 chars): ${output.slice(0, 300)}`);
          // Keep the input file for debugging
          console.log(`[AIRSS] Input file: ${inputPath}`);
        }

        // Clean up input file (keep first one if it failed for debugging)
        if (candidate || i > 0) {
          try { fs.unlinkSync(inputPath); } catch {}
        }
      } catch (err: any) {
        if (i === 0) {
          console.log(`[AIRSS] First structure exception: ${err.message?.slice(0, 200)}`);
          // Also log stderr if available
          if (err.stderr) console.log(`[AIRSS] stderr: ${err.stderr.toString().slice(0, 200)}`);
          if (err.stdout) console.log(`[AIRSS] stdout: ${err.stdout.toString().slice(0, 200)}`);
        }
      }
    }

    console.log(`[AIRSS] Generated ${candidates.length}/${config.maxStructures} structures`);
    return candidates;
  },
};
