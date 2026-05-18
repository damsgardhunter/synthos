/**
 * AIRSS (Ab Initio Random Structure Searching) wrapper.
 *
 * Uses `buildcell` to generate sensible random structures with:
 * - Formula-unit (Z) sweeps: Z = 1, 2, 3, 4, 6, 8
 * - Volume ensemble: multiple TARGVOL values per composition
 * - Pressure-aware MINSEP: element-pair distances scaled by pressure
 * - Screening tiers: preview (2500), standard (5000), deep (10000)
 *
 * Reference: Pickard & Needs, J. Phys.: Condens. Matter 23 (2011) 053201
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import type { CSPCandidate, CSPEngine, CSPEngineConfig, ScreeningTierConfig } from "./csp-types";
import {
  cellVolumeFromVectors,
  COVALENT_RADII,
  getGeneratorMinsep,
  pressureVolumeEnsemble,
  SCREENING_TIERS,
} from "./csp-types";
import { loadVolumeBias } from "./volume-bias-cache";
import { estimateBulkModulusFromElements } from "../dft/vegard-lattice";
import { lookupKnownStructure, knownStructureVolumePerAtom } from "../learning/known-structures";

const BUILDCELL_BIN = process.env.AIRSS_BIN
  ?? process.env.BUILDCELL_BIN
  ?? "/usr/local/bin/buildcell";

// Deduplication cache for noisy logs: AIRSS calls generateCellInput once per
// (Z, volume) combination per formula (≈20× per material), and the LLM-advised
// MINSEP computation is identical for every call. Track emitted messages so we
// log each unique line once per process. Capped at 256 entries; oldest evicted.
const LOGGED_MESSAGES = new Set<string>();
const LOGGED_MESSAGE_ORDER: string[] = [];
function logOnce(message: string): void {
  if (LOGGED_MESSAGES.has(message)) return;
  LOGGED_MESSAGES.add(message);
  LOGGED_MESSAGE_ORDER.push(message);
  if (LOGGED_MESSAGE_ORDER.length > 256) {
    const evicted = LOGGED_MESSAGE_ORDER.shift();
    if (evicted) LOGGED_MESSAGES.delete(evicted);
  }
  console.log(message);
}

// Volume per atom estimates (Angstrom^3/atom)
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

export function estimateVolumePerAtom(elements: string[], counts: Record<string, number>): number {
  let totalVol = 0;
  let totalAtoms = 0;
  for (const el of elements) {
    const n = Math.round(counts[el]);
    totalVol += (VOL_PER_ATOM[el] ?? 15) * n;
    totalAtoms += n;
  }
  return totalAtoms > 0 ? totalVol / totalAtoms : 15;
}

/**
 * Generate a .cell input file for buildcell.
 */
function generateCellInput(
  elements: string[],
  counts: Record<string, number>,
  z: number,
  targetVolume: number,
  pressureGPa: number,
  advisedPairDistances?: Record<string, number>,
  advisedLattice?: { a: number; b: number; c: number },
): string {
  // Scale counts by Z
  const scaledCounts: Record<string, number> = {};
  let totalAtoms = 0;
  for (const el of elements) {
    scaledCounts[el] = Math.round(counts[el]) * z;
    totalAtoms += scaledCounts[el];
  }

  const cellVol = targetVolume * totalAtoms;
  // Cell SHAPE: honor the LLM-advised lattice anisotropy. A cubic a×a×a seed
  // forced layered compounds — cuprates have c/a ≈ 3-3.5 — into a cube, so
  // buildcell could not form the CuO2 planes and produced garbage geometry.
  // Preserve the advised a:b:c ratios (clamped against LLM hallucination),
  // scaled to the target cell volume; buildcell rescales the LATTICE_CART to
  // #TARGVOL while keeping the shape. Falls back to cubic with no advice.
  let lattA: number, lattB: number, lattC: number;
  if (advisedLattice && advisedLattice.a > 0 && advisedLattice.b > 0 && advisedLattice.c > 0) {
    const clampRatio = (r: number) => Math.max(0.25, Math.min(8.0, r));
    const rB = clampRatio(advisedLattice.b / advisedLattice.a);
    const rC = clampRatio(advisedLattice.c / advisedLattice.a);
    lattA = Math.pow(cellVol / (rB * rC), 1 / 3);
    lattB = lattA * rB;
    lattC = lattA * rC;
  } else {
    lattA = lattB = lattC = Math.pow(cellVol, 1 / 3);
  }

  // Global MINSEP — the closest any two atoms may approach in buildcell.
  // Don't use per-pair #MINSEP syntax — buildcell can't satisfy multiple pair
  // constraints simultaneously in small high-P cells. Floor: the covalent-
  // radii heuristic (getGeneratorMinsep). The LLM may advise tighter per-pair
  // minima, but each advised distance is first cross-checked against the
  // covalent-radii sum for that element pair — a hallucinated value (too
  // small → atom overlap, or too large → buildcell can't place atoms) is
  // logged and ignored, never allowed to drive MINSEP.
  const elementSymbols = Array.from(new Set(elements));
  let heuristicMin = Infinity;
  for (const el1 of elementSymbols) {
    for (const el2 of elementSymbols) {
      heuristicMin = Math.min(heuristicMin, getGeneratorMinsep(el1, el2, pressureGPa));
    }
  }
  if (!Number.isFinite(heuristicMin) || heuristicMin <= 0) heuristicMin = 0.7;
  let globalMin = heuristicMin;
  if (advisedPairDistances && Object.keys(advisedPairDistances).length > 0) {
    const plausible: number[] = [];
    const suspect: string[] = [];
    for (const [pair, dist] of Object.entries(advisedPairDistances)) {
      if (typeof dist !== "number" || !(dist > 0)) continue;
      const els = pair.split(/[-_/ ]+/).filter(Boolean);
      const covSum = (COVALENT_RADII[els[0]] ?? 1.0) + (COVALENT_RADII[els[1] ?? els[0]] ?? 1.0);
      // A valid minimum-separation sits within 0.35x (extreme compression) to
      // 1.5x (no closer than a long bond) of the covalent-radii sum.
      const lo = 0.35 * covSum, hi = 1.5 * covSum;
      if (dist >= lo && dist <= hi) {
        plausible.push(dist);
      } else {
        suspect.push(`${pair}=${dist.toFixed(2)} (physical range ${lo.toFixed(2)}-${hi.toFixed(2)})`);
      }
    }
    if (suspect.length > 0) {
      logOnce(`[AIRSS] MINSEP validation for ${elements.join("")}: ${suspect.length} advised pair distance(s) implausible vs covalent radii — ignored: ${suspect.join("; ")}`);
    }
    if (plausible.length > 0) {
      const advisedMin = Math.min(...plausible);
      const advisedMinsep = advisedMin * 0.75;
      globalMin = Math.max(heuristicMin, advisedMinsep);
      logOnce(`[AIRSS] LLM-advised MINSEP for ${elements.join("")}: smallest plausible pair=${advisedMin.toFixed(2)} A x 0.75 = ${advisedMinsep.toFixed(2)} A, heuristic floor=${heuristicMin.toFixed(2)} A -> MINSEP=${globalMin.toFixed(2)} A`);
    } else {
      logOnce(`[AIRSS] MINSEP for ${elements.join("")}: no plausible advised pair distances — using heuristic floor ${heuristicMin.toFixed(2)} A`);
    }
  }

  let cell = `#TARGVOL=${cellVol.toFixed(1)}\n`;
  cell += `#MINSEP=${Math.max(0.5, globalMin).toFixed(2)}\n`;

  cell += `\n%BLOCK LATTICE_CART\n`;
  cell += `${lattA.toFixed(4)} 0.0000 0.0000\n`;
  cell += `0.0000 ${lattB.toFixed(4)} 0.0000\n`;
  cell += `0.0000 0.0000 ${lattC.toFixed(4)}\n`;
  cell += `%ENDBLOCK LATTICE_CART\n`;

  cell += `\n%BLOCK POSITIONS_FRAC\n`;
  for (const el of elements) {
    for (let i = 0; i < scaledCounts[el]; i++) {
      cell += `${el} 0.0 0.0 0.0\n`;
    }
  }
  cell += `%ENDBLOCK POSITIONS_FRAC\n`;

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
 * Parse buildcell output (.cell format) into a CSPCandidate.
 */
function parseCellOutput(
  content: string,
  elements: string[],
  seed: number,
  pressureGPa: number,
  stage: 1 | 2 | 3,
  z: number,
): CSPCandidate | null {
  // Parse lattice vectors — handle both LATTICE_CART and LATTICE_ABC
  let vecs: [number, number, number][] = [];

  const latCartMatch = content.match(
    /%BLOCK\s+LATTICE_CART\s*\n([\s\S]*?)%ENDBLOCK\s+LATTICE_CART/i
  );
  const latAbcMatch = !latCartMatch
    ? content.match(/%BLOCK\s+LATTICE_ABC\s*\n([\s\S]*?)%ENDBLOCK\s+LATTICE_ABC/i)
    : null;

  if (latCartMatch) {
    for (const line of latCartMatch[1].trim().split("\n")) {
      const parts = line.trim().split(/\s+/).map(Number);
      if (parts.length >= 3 && !parts.some(isNaN)) {
        vecs.push([parts[0], parts[1], parts[2]]);
      }
    }
  } else if (latAbcMatch) {
    const abcLines = latAbcMatch[1].trim().split("\n").filter(l => l.trim().length > 0);
    if (abcLines.length >= 2) {
      const [aVal, bVal, cVal] = abcLines[0].split(/\s+/).map(Number);
      const [alpha, beta, gamma] = abcLines[1].split(/\s+/).map(Number);
      if ([aVal, bVal, cVal, alpha, beta, gamma].every(v => !isNaN(v))) {
        const toRad = (d: number) => (d * Math.PI) / 180;
        const cosA = Math.cos(toRad(alpha)), cosB = Math.cos(toRad(beta));
        const cosG = Math.cos(toRad(gamma)), sinG = Math.sin(toRad(gamma));
        vecs = [
          [aVal, 0, 0],
          [bVal * cosG, bVal * sinG, 0],
          [cVal * cosB, cVal * (cosA - cosB * cosG) / sinG,
           cVal * Math.sqrt(Math.max(0, 1 - cosA ** 2 - cosB ** 2 - cosG ** 2 + 2 * cosA * cosB * cosG)) / sinG],
        ];
      }
    }
  }
  if (vecs.length < 3) return null;

  // Parse positions
  const posMatch = content.match(
    /%BLOCK\s+POSITIONS_FRAC\s*\n([\s\S]*?)%ENDBLOCK\s+POSITIONS_FRAC/i
  );
  if (!posMatch) return null;

  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];
  for (const line of posMatch[1].trim().split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 4 && /^[A-Z][a-z]?$/.test(parts[0])) {
      const x = parseFloat(parts[1]), y = parseFloat(parts[2]), zz = parseFloat(parts[3]);
      if (!isNaN(x) && !isNaN(y) && !isNaN(zz)) {
        positions.push({
          element: parts[0],
          x: ((x % 1) + 1) % 1,
          y: ((y % 1) + 1) % 1,
          z: ((zz % 1) + 1) % 1,
        });
      }
    }
  }
  if (positions.length === 0) return null;

  const a = Math.sqrt(vecs[0][0] ** 2 + vecs[0][1] ** 2 + vecs[0][2] ** 2);
  const b = Math.sqrt(vecs[1][0] ** 2 + vecs[1][1] ** 2 + vecs[1][2] ** 2);
  const c = Math.sqrt(vecs[2][0] ** 2 + vecs[2][1] ** 2 + vecs[2][2] ** 2);
  const volume = cellVolumeFromVectors(vecs);

  return {
    latticeA: a, latticeB: b, latticeC: c, cOverA: c / a,
    positions,
    prototype: "AIRSS-buildcell",
    crystalSystem: "unknown",
    spaceGroup: "",
    source: `AIRSS (seed=${seed}, Z=${z})`,
    confidence: 0.4,
    isMetallic: null,
    sourceEngine: "airss",
    generationStage: 1,
    seed,
    pressureGPa,
    cellVolume: volume,
    cellVectors: vecs,
    latticeParams: { a, b, c, alpha: 90, beta: 90, gamma: 90 },
    relaxationLevel: "raw",
  };
}

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

    if (config.structureAdvice) {
      const adv = config.structureAdvice;
      const pairs = adv.pairDistances ?? {};
      const pairStr = Object.keys(pairs).length > 0
        ? Object.entries(pairs).map(([k, v]) => `${k}=${(v as number).toFixed(2)}`).join(", ")
        : "none";
      const lat = adv.estimatedLattice;
      const latStr = lat && lat.a > 0 ? `${lat.a.toFixed(2)}x${lat.b.toFixed(2)}x${lat.c.toFixed(2)} A` : "none";
      const altSG = (adv.alternativeSpaceGroups ?? []).length > 0
        ? ` (alt: ${adv.alternativeSpaceGroups!.join(",")})` : "";
      console.log(`[AIRSS] LLM structure advice for ${elements.join("")}: type=${adv.structureType ?? "?"}, SG=${adv.likelySpaceGroup ?? "?"}${altSG}, estimatedLattice=${latStr}`);
      console.log(`[AIRSS]   advised pair distances (A): ${pairStr}`);
    }

    // Determine screening tier from budget
    let tierConfig: ScreeningTierConfig;
    if (config.maxStructures >= 10000) tierConfig = SCREENING_TIERS.deep;
    else if (config.maxStructures >= 5000) tierConfig = SCREENING_TIERS.standard;
    else if (config.maxStructures >= 2500) tierConfig = SCREENING_TIERS.preview;
    else {
      // Small budget (called from qe-worker inline) — use minimal sweep
      tierConfig = {
        tier: "preview",
        maxAtoms: 20,
        zValues: [1, 2],
        airssBudgetPerCombo: Math.ceil(config.maxStructures / 2),
        airssTotalCap: config.maxStructures,
        pyxtalBudgetPerZ: 0,
        pyxtalTotalCap: 0,
        volumeEnsemble: [0.85, 1.0, 1.15],
        dft0Floor: 3, dft0Cap: 8, dft0ClusterFraction: 0.03,
      };
    }

    const formulaAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

    // Persisted volume-bias correction from a prior F6 CHGNet batch — see
    // server/csp/volume-bias-cache.ts. Applied as a multiplier on whichever
    // volume base we pick below.
    const biasFormula = elements
      .map(el => { const n = Math.round(counts[el] ?? 0); return el + (n > 1 ? n : ""); })
      .join("");
    const volumeBias = loadVolumeBias(biasFormula, config.pressureGPa);

    // Volume targeting. Priority:
    //  1. Known structure at a comparable pressure → its EXACT cell volume.
    //     The elemental-volume sum overestimates compound volumes ~30-60%
    //     and the LLM lattice cannot predict a high-pressure volume at all
    //     (it returned ~40 Å³/atom for YBeH8 @ 180 GPa, ~10× too large) —
    //     either way CHGNet's drift gate then rejected ~40% of candidates.
    //  2. Otherwise: ambient elemental sum compressed by the Murnaghan EOS
    //     with the same element-based bulk modulus the Vegard path uses.
    // The LLM lattice is still used only for cell SHAPE in generateCellInput.
    let volPerAtom: number;
    let volumeTargets: number[];
    let volSource: string;
    const knownStruct = lookupKnownStructure(biasFormula);
    const knownVpa = knownStruct ? knownStructureVolumePerAtom(knownStruct) : NaN;
    if (knownStruct && Number.isFinite(knownVpa) && knownVpa > 0
        && Math.abs((knownStruct.pressureGPa ?? 0) - config.pressureGPa) < 30) {
      volPerAtom = knownVpa * volumeBias;
      volumeTargets = tierConfig.volumeEnsemble.map(f => volPerAtom * f);
      volSource = `known-structure ${knownVpa.toFixed(1)} A^3/atom`;
    } else {
      const fractions: Record<string, number> = {};
      for (const el of elements) fractions[el] = Math.round(counts[el] ?? 0);
      const b0 = estimateBulkModulusFromElements(elements, fractions);
      const ambientVPA = estimateVolumePerAtom(elements, counts);
      volPerAtom = ambientVPA * volumeBias;
      volumeTargets = pressureVolumeEnsemble(volPerAtom, config.pressureGPa, tierConfig.volumeEnsemble, b0);
      volSource = `EOS ambient ${ambientVPA.toFixed(1)} A^3/atom, B0=${b0.toFixed(0)} GPa`;
    }
    console.log(`[AIRSS] Volume target for ${biasFormula} @ ${config.pressureGPa} GPa: ${volSource}` +
      `${Math.abs(volumeBias - 1) > 0.05 ? ` (volume-bias x${volumeBias.toFixed(2)})` : ""}, ${volumeTargets.length} ensemble points`);

    // Filter Z values by atom count cap
    const validZ = tierConfig.zValues.filter(z => formulaAtoms * z <= tierConfig.maxAtoms);
    if (validZ.length === 0) validZ.push(1); // always try Z=1

    const totalCombos = validZ.length * volumeTargets.length;
    const budgetPerCombo = Math.min(
      tierConfig.airssBudgetPerCombo,
      Math.ceil(tierConfig.airssTotalCap / totalCombos),
    );

    console.log(`[AIRSS] Generating for ${elements.join("")} at ${config.pressureGPa} GPa: Z=[${validZ.join(",")}], ${volumeTargets.length} volumes, ${budgetPerCombo}/combo, cap=${tierConfig.airssTotalCap} (${tierConfig.tier} tier)`);

    let totalGenerated = 0;
    let seedCounter = 0;

    for (const z of validZ) {
      for (const targetVol of volumeTargets) {
        if (totalGenerated >= tierConfig.airssTotalCap) break;

        const batchBudget = Math.min(budgetPerCombo, tierConfig.airssTotalCap - totalGenerated);
        let batchGenerated = 0;

        for (let i = 0; i < batchBudget; i++) {
          const seed = baseSeed + seedCounter++;
          try {
            const cellInput = generateCellInput(elements, counts, z, targetVol, config.pressureGPa, config.structureAdvice?.pairDistances, config.structureAdvice?.estimatedLattice);
            const inputPath = path.join(workDir, `airss_${seed}.cell`);
            fs.writeFileSync(inputPath, cellInput);

            const result = execSync(
              `${BUILDCELL_BIN} < ${inputPath} 2>&1`,
              { cwd: workDir, timeout: 10000, maxBuffer: 1024 * 1024 }
            );

            const output = result.toString("utf-8");
            const candidate = parseCellOutput(output, elements, seed, config.pressureGPa, 1, z);
            if (candidate) {
              candidates.push(candidate);
              batchGenerated++;
              totalGenerated++;
            }

            try { fs.unlinkSync(inputPath); } catch {}
          } catch (err: any) {
            if (batchGenerated === 0 && i === 0) {
              console.log(`[AIRSS] Z=${z} vol=${targetVol.toFixed(1)} first fail: ${err.message?.slice(0, 100)}`);
            }
          }
        }
      }
      if (totalGenerated >= tierConfig.airssTotalCap) break;
    }

    console.log(`[AIRSS] Generated ${totalGenerated} structures (Z sweep: ${validZ.join(",")}, ${volumeTargets.length} volumes)`);
    return candidates;
  },
};
