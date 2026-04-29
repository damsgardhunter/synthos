/**
 * POSCAR (VASP) format reader/writer for cross-engine structure exchange.
 *
 * POSCAR is the natural interchange format because:
 * - USPEX generates/reads POSCAR natively (gatheredPOSCARS, Seeds/POSCARS)
 * - CALYPSO generates/reads POSCAR natively (results/ directory)
 * - AIRSS .cell files can be converted, but we write POSCAR directly
 * - QE input is built from positions + lattice vectors (same data)
 *
 * Format reference:
 * ```
 * Comment line (formula, engine, seed, enthalpy)
 * 1.0            ← universal scaling factor (lattice in Angstrom)
 *   ax ay az     ← lattice vector a
 *   bx by bz     ← lattice vector b
 *   cx cy cz     ← lattice vector c
 * El1 El2 ...    ← species names
 * N1  N2  ...    ← counts per species
 * Direct         ← fractional coordinates
 *   x1 y1 z1
 *   x2 y2 z2
 *   ...
 * ```
 */

import type { CSPCandidate, CSPEngineName, RelaxationLevel } from "./csp-types";
import { cellVolumeFromVectors } from "./csp-types";

// ---------------------------------------------------------------------------
// POSCAR Writer
// ---------------------------------------------------------------------------

export function writePOSCAR(candidate: CSPCandidate): string {
  const elements: string[] = [];
  const countMap: Record<string, number> = {};
  for (const pos of candidate.positions) {
    if (!countMap[pos.element]) {
      elements.push(pos.element);
      countMap[pos.element] = 0;
    }
    countMap[pos.element]++;
  }

  const vecs = candidate.cellVectors ?? latticeParamsToVectors(
    candidate.latticeA,
    candidate.latticeB ?? candidate.latticeA,
    candidate.latticeC ?? candidate.latticeA * (candidate.cOverA ?? 1.0),
    candidate.latticeParams?.alpha ?? 90,
    candidate.latticeParams?.beta ?? 90,
    candidate.latticeParams?.gamma ?? (candidate.crystalSystem === "hexagonal" ? 120 : 90),
  );

  // Comment line encodes metadata for round-tripping
  const meta = [
    candidate.source,
    `engine=${candidate.sourceEngine}`,
    `seed=${candidate.seed}`,
    `stage=${candidate.generationStage}`,
    candidate.enthalpyPerAtom != null ? `H=${candidate.enthalpyPerAtom.toFixed(4)}eV/at` : "",
    candidate.pressureGPa > 0 ? `P=${candidate.pressureGPa}GPa` : "",
  ].filter(Boolean).join(" | ");

  let out = `${meta}\n`;
  out += `1.0\n`;
  for (const v of vecs) {
    out += `  ${v[0].toFixed(10)}  ${v[1].toFixed(10)}  ${v[2].toFixed(10)}\n`;
  }
  out += elements.join("  ") + "\n";
  out += elements.map(e => countMap[e]).join("  ") + "\n";
  out += "Direct\n";

  // Write positions grouped by species (POSCAR convention)
  for (const el of elements) {
    for (const pos of candidate.positions) {
      if (pos.element === el) {
        out += `  ${pos.x.toFixed(10)}  ${pos.y.toFixed(10)}  ${pos.z.toFixed(10)}\n`;
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// POSCAR Parser
// ---------------------------------------------------------------------------

export function parsePOSCAR(
  content: string,
  sourceEngine: CSPEngineName = "uspex",
  pressureGPa: number = 0,
  seed: number = 0,
  stage: 1 | 2 | 3 = 1,
): CSPCandidate | null {
  const lines = content.trim().split("\n");
  if (lines.length < 8) return null;

  const comment = lines[0];
  const scale = parseFloat(lines[1]);
  if (isNaN(scale) || scale <= 0) return null;

  // Parse lattice vectors
  const vecs: [number, number, number][] = [];
  for (let i = 2; i <= 4; i++) {
    const parts = lines[i].trim().split(/\s+/).map(Number);
    if (parts.length < 3 || parts.some(isNaN)) return null;
    vecs.push([parts[0] * scale, parts[1] * scale, parts[2] * scale]);
  }

  // Species names (line 5) and counts (line 6)
  const speciesLine = lines[5].trim().split(/\s+/);
  const countsLine = lines[6].trim().split(/\s+/).map(Number);

  // Check if line 5 is species names or counts (POSCAR format ambiguity)
  let species: string[];
  let speciesCounts: number[];
  let coordStart: number;

  if (speciesLine.every(s => /^[A-Z][a-z]?$/.test(s))) {
    // Line 5 = species, line 6 = counts
    species = speciesLine;
    speciesCounts = countsLine;
    coordStart = 8; // line 7 = Direct/Cartesian, line 8+ = coords
  } else {
    // Old POSCAR without species names — not supported here
    return null;
  }

  if (species.length !== speciesCounts.length) return null;

  // Coordinate type
  const coordType = lines[7]?.trim().toLowerCase() ?? "direct";
  const isDirect = coordType.startsWith("d");

  // Parse positions
  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];
  let atomIdx = 0;
  for (let s = 0; s < species.length; s++) {
    for (let i = 0; i < speciesCounts[s]; i++) {
      const lineIdx = coordStart + atomIdx;
      if (lineIdx >= lines.length) break;
      const parts = lines[lineIdx].trim().split(/\s+/).map(Number);
      if (parts.length < 3 || parts.some(isNaN)) { atomIdx++; continue; }

      let x = parts[0], y = parts[1], z = parts[2];

      if (!isDirect) {
        // Cartesian → fractional: multiply by inverse of lattice matrix
        // For simplicity, use the scaled vectors directly
        const inv = invertMatrix3x3(vecs);
        if (inv) {
          const fx = inv[0][0] * x + inv[0][1] * y + inv[0][2] * z;
          const fy = inv[1][0] * x + inv[1][1] * y + inv[1][2] * z;
          const fz = inv[2][0] * x + inv[2][1] * y + inv[2][2] * z;
          x = fx; y = fy; z = fz;
        }
      }

      // Wrap to [0, 1)
      x = ((x % 1) + 1) % 1;
      y = ((y % 1) + 1) % 1;
      z = ((z % 1) + 1) % 1;

      positions.push({ element: species[s], x, y, z });
      atomIdx++;
    }
  }

  if (positions.length === 0) return null;

  // Extract lattice parameters from vectors
  const a = Math.sqrt(vecs[0][0] ** 2 + vecs[0][1] ** 2 + vecs[0][2] ** 2);
  const b = Math.sqrt(vecs[1][0] ** 2 + vecs[1][1] ** 2 + vecs[1][2] ** 2);
  const c = Math.sqrt(vecs[2][0] ** 2 + vecs[2][1] ** 2 + vecs[2][2] ** 2);
  const alpha = Math.acos(dot3(vecs[1], vecs[2]) / (b * c)) * 180 / Math.PI;
  const beta = Math.acos(dot3(vecs[0], vecs[2]) / (a * c)) * 180 / Math.PI;
  const gamma = Math.acos(dot3(vecs[0], vecs[1]) / (a * b)) * 180 / Math.PI;

  const volume = cellVolumeFromVectors(vecs);

  // Parse metadata from comment line
  const seedMatch = comment.match(/seed=(\d+)/);
  const enthalpyMatch = comment.match(/H=([-\d.]+)/);
  const parsedSeed = seedMatch ? parseInt(seedMatch[1]) : seed;
  const parsedEnthalpy = enthalpyMatch ? parseFloat(enthalpyMatch[1]) : undefined;

  return {
    // StructureCandidate base fields
    latticeA: a,
    latticeB: b,
    latticeC: c,
    cOverA: c / a,
    positions,
    prototype: `${sourceEngine}-generated`,
    crystalSystem: guessCrystalSystem(a, b, c, alpha, beta, gamma),
    spaceGroup: "",
    source: comment.slice(0, 80),
    confidence: 0.5,
    isMetallic: null,

    // CSPCandidate extensions
    sourceEngine,
    generationStage: stage,
    seed: parsedSeed,
    pressureGPa: pressureGPa,
    cellVolume: volume,
    cellVectors: vecs,
    latticeParams: { a, b, c, alpha, beta, gamma },
    relaxationLevel: "raw",
    enthalpyPerAtom: parsedEnthalpy,
  };
}

// ---------------------------------------------------------------------------
// Batch POSCAR parsing (for USPEX gatheredPOSCARS, CALYPSO results/)
// ---------------------------------------------------------------------------

/**
 * Parse a concatenated POSCAR file (multiple structures separated by blank lines
 * or by the comment line of the next structure).
 */
export function parseMultiplePOSCARS(
  content: string,
  sourceEngine: CSPEngineName,
  pressureGPa: number,
  baseSeed: number = 0,
  stage: 1 | 2 | 3 = 1,
): CSPCandidate[] {
  const candidates: CSPCandidate[] = [];

  // Split on boundaries: a POSCAR starts with a comment line followed by
  // a line with a single number (the scaling factor)
  const blocks: string[] = [];
  let currentBlock: string[] = [];

  const lines = content.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const nextLine = (i + 1 < lines.length) ? lines[i + 1].trim() : "";

    // Detect POSCAR boundary: a line followed by a line that's a single float
    if (currentBlock.length > 7 && /^[0-9.]+$/.test(nextLine) && line !== "") {
      // Current line is the last coord line; next line starts a new POSCAR
      currentBlock.push(lines[i]);
      blocks.push(currentBlock.join("\n"));
      currentBlock = [];
      continue;
    }

    if (line !== "" || currentBlock.length > 0) {
      currentBlock.push(lines[i]);
    }
  }
  if (currentBlock.length > 7) {
    blocks.push(currentBlock.join("\n"));
  }

  for (let i = 0; i < blocks.length; i++) {
    const parsed = parsePOSCAR(blocks[i], sourceEngine, pressureGPa, baseSeed + i, stage);
    if (parsed) candidates.push(parsed);
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/**
 * Convert lattice parameters (a, b, c, alpha, beta, gamma) to 3x3 vectors.
 * Uses the standard crystallographic convention:
 *   a along x, b in xy-plane, c general.
 */
export function latticeParamsToVectors(
  a: number, b: number, c: number,
  alphaDeg: number, betaDeg: number, gammaDeg: number,
): [number, number, number][] {
  const toRad = Math.PI / 180;
  const alpha = alphaDeg * toRad;
  const beta = betaDeg * toRad;
  const gamma = gammaDeg * toRad;

  const cosAlpha = Math.cos(alpha);
  const cosBeta = Math.cos(beta);
  const cosGamma = Math.cos(gamma);
  const sinGamma = Math.sin(gamma);

  const va: [number, number, number] = [a, 0, 0];
  const vb: [number, number, number] = [b * cosGamma, b * sinGamma, 0];
  const cx = c * cosBeta;
  const cy = c * (cosAlpha - cosBeta * cosGamma) / sinGamma;
  const cz = Math.sqrt(Math.max(0, c * c - cx * cx - cy * cy));
  const vc: [number, number, number] = [cx, cy, cz];

  return [va, vb, vc];
}

function dot3(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function invertMatrix3x3(m: [number, number, number][]): [number, number, number][] | null {
  const [a, b, c] = m;
  const det = a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
  if (Math.abs(det) < 1e-12) return null;
  const inv = 1 / det;
  return [
    [(b[1] * c[2] - b[2] * c[1]) * inv, (a[2] * c[1] - a[1] * c[2]) * inv, (a[1] * b[2] - a[2] * b[1]) * inv],
    [(b[2] * c[0] - b[0] * c[2]) * inv, (a[0] * c[2] - a[2] * c[0]) * inv, (a[2] * b[0] - a[0] * b[2]) * inv],
    [(b[0] * c[1] - b[1] * c[0]) * inv, (a[1] * c[0] - a[0] * c[1]) * inv, (a[0] * b[1] - a[1] * b[0]) * inv],
  ];
}

function guessCrystalSystem(
  a: number, b: number, c: number,
  alpha: number, beta: number, gamma: number,
): string {
  const tol = 1.5; // degrees
  const ltol = 0.05; // fractional lattice tolerance
  const isRightAngle = (x: number) => Math.abs(x - 90) < tol;
  const is120 = (x: number) => Math.abs(x - 120) < tol;
  const eq = (x: number, y: number) => Math.abs(x - y) / Math.max(x, y, 0.01) < ltol;

  if (eq(a, b) && eq(b, c) && isRightAngle(alpha) && isRightAngle(beta) && isRightAngle(gamma)) return "cubic";
  if (eq(a, b) && isRightAngle(alpha) && isRightAngle(beta) && is120(gamma)) return "hexagonal";
  if (eq(a, b) && isRightAngle(alpha) && isRightAngle(beta) && isRightAngle(gamma)) return "tetragonal";
  if (isRightAngle(alpha) && isRightAngle(beta) && isRightAngle(gamma)) return "orthorhombic";
  if (isRightAngle(alpha) && isRightAngle(gamma)) return "monoclinic";
  return "triclinic";
}
