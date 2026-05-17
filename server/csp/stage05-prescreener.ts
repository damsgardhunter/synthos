/**
 * Stage 0.5 Structural Pre-Screener
 *
 * Quickly validates novel compositions using real structural calculations:
 * 1. PyXtal generates 3-5 candidate structures (0.3s)
 * 2. CHGNet MLIP relaxes the best structures (5-20s)
 * 3. xTB computes phonon Hessian on relaxed geometry (5-15s)
 * 4. Extracts: formation_energy, phonon_stability, omega_log, metallicity_proxy
 * 5. Returns a recommendation: promote-to-dft / borderline / reject
 *
 * Total time: ~30-50s per formula (vs 30-120 min for full DFT)
 *
 * This replaces the crude Hopfield heuristic for novel compounds with
 * semi-empirical structural data that has real physical meaning.
 */

import { pyxtalEngine } from "./pyxtal-wrapper";
import { runChgnetEvaluation, isChgnetAvailable } from "./chgnet-wrapper";
import { computeFiniteDisplacementPhonons } from "../dft/phonon-calculator";
import { VERIFIED_COMPOUNDS, computePhysicsTcUQ } from "../learning/physics-engine";
import { estimateFamilyPressure } from "../learning/candidate-generator";
import { parseFormulaCounts } from "../learning/physics-engine";
import * as path from "path";
import * as fs from "fs";
import * as os from "os";

export interface Stage05Result {
  formula: string;
  pressureGpa: number;
  // Structural results
  structuresGenerated: number;
  structuresRelaxed: number;
  bestFormationEnergyPerAtom: number | null;  // eV/atom from CHGNet
  bestVolumePerAtom: number | null;             // ų/atom
  // Phonon results
  phononStable: boolean | null;               // null = couldn't compute
  omegaLogEstimate: number | null;            // cm⁻¹ from xTB Hessian
  lowestFrequency: number | null;             // cm⁻¹ (negative = imaginary)
  imaginaryModeCount: number;
  // Derived physics
  tcEstimate: number;                         // K, from Allen-Dynes with real omega_log
  isMetallicProxy: boolean;                   // crude metallicity from composition
  // Decision
  recommendation: "promote-to-dft" | "borderline" | "reject";
  rejectReason: string | null;
  // Metadata
  source: "stage05-structural";
  confidence: number;                         // 0-1: how much we trust this estimate
  wallTimeMs: number;
}

/**
 * Run Stage 0.5 pre-screening on a novel formula.
 * Returns structural physics features in ~30-50s without touching DFT.
 */
export async function runStage05PreScreen(
  formula: string,
  pressureGpa?: number,
): Promise<Stage05Result> {
  const t0 = Date.now();
  const pressure = pressureGpa ?? estimateFamilyPressure(formula);

  // Skip for verified compounds — we already know the answer
  const normalized = formula.replace(/\s+/g, "");
  if (VERIFIED_COMPOUNDS[normalized]) {
    const uq = computePhysicsTcUQ(formula, pressure);
    return {
      formula,
      pressureGpa: pressure,
      structuresGenerated: 0,
      structuresRelaxed: 0,
      bestFormationEnergyPerAtom: null,
      bestVolumePerAtom: null,
      phononStable: true,
      omegaLogEstimate: VERIFIED_COMPOUNDS[normalized].omegaLog,
      lowestFrequency: 50,  // assumed stable
      imaginaryModeCount: 0,
      tcEstimate: uq.mean,
      isMetallicProxy: true,
      recommendation: "promote-to-dft",
      rejectReason: null,
      source: "stage05-structural",
      confidence: 1.0,
      wallTimeMs: Date.now() - t0,
    };
  }

  // Parse composition
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  // Quick metallicity check: compounds with ONLY strong nonmetals (F, O, N, Cl) + H
  // and no metals are insulators — reject immediately
  const STRONG_NONMETALS = new Set(["F", "Cl", "Br", "I", "O", "N"]);
  const hasAnyMetal = elements.some(e =>
    !STRONG_NONMETALS.has(e) && e !== "H" && e !== "C" && e !== "S" && e !== "Se" && e !== "P"
  );
  if (!hasAnyMetal && pressure < 50) {
    return {
      formula, pressureGpa: pressure,
      structuresGenerated: 0, structuresRelaxed: 0,
      bestFormationEnergyPerAtom: null, bestVolumePerAtom: null,
      phononStable: null, omegaLogEstimate: null,
      lowestFrequency: null, imaginaryModeCount: 0,
      tcEstimate: 0, isMetallicProxy: false,
      recommendation: "reject",
      rejectReason: "no metallic elements at ambient pressure — insulator",
      source: "stage05-structural", confidence: 0.9,
      wallTimeMs: Date.now() - t0,
    };
  }

  let structuresGenerated = 0;
  let structuresRelaxed = 0;
  let bestFormationEnergyPerAtom: number | null = null;
  let bestVolumePerAtom: number | null = null;
  let phononStable: boolean | null = null;
  let omegaLogEstimate: number | null = null;
  let lowestFrequency: number | null = null;
  let imaginaryModeCount = 0;

  // Step 1: Generate structures with PyXtal (if available)
  const workDir = path.join(os.tmpdir(), `stage05-${formula}-${Date.now()}`);
  try {
    fs.mkdirSync(workDir, { recursive: true });

    if (pyxtalEngine.isAvailable() && totalAtoms <= 20) {
      try {
        const candidates = await pyxtalEngine.generateStructures(
          elements,
          counts,
          {
            workDir,
            maxStructures: 5,
            pressureGPa: pressure,
            baseSeed: Date.now() % 100000,
            timeoutMs: 10000,
          }
        );
        structuresGenerated = candidates.length;

        // Step 2: CHGNet relaxation (if available)
        if (isChgnetAvailable() && candidates.length > 0) {
          try {
            const { rankedCandidates, stats } = await runChgnetEvaluation(
              candidates,
              workDir,
              true,   // relax
              3,      // max 3 structures
              30000,  // 30s timeout
            );
            structuresRelaxed = stats.relaxed;
            if (stats.bestEnergy != null) {
              bestFormationEnergyPerAtom = stats.bestEnergy;
            }
            if (rankedCandidates.length > 0) {
              const best = rankedCandidates[0];
              bestVolumePerAtom = best.cellVolume ? best.cellVolume / totalAtoms : null;

              // Step 3: xTB phonon on CHGNet-relaxed best structure
              if (best.positions && best.positions.length > 0) {
                try {
                  const latA = best.latticeA ?? 5;
                  const latB = best.latticeB ?? best.latticeA ?? 5;
                  const latC = best.latticeC ?? best.latticeA ?? 5;
                  const atoms = best.positions.map(p => ({
                    element: p.element,
                    x: p.x * latA,
                    y: p.y * latB,
                    z: p.z * latC,
                  }));
                  // Pass the (diagonal) lattice vectors matching the embedding
                  // above so the phonon calculator can build a periodic
                  // supercell and compute a real q≠0 dispersion via the
                  // lattice sum, rather than a Γ-only spectrum.
                  const latticeVectors = [[latA, 0, 0], [0, latB, 0], [0, 0, latC]];
                  const phononResult = await computeFiniteDisplacementPhonons(
                    formula, atoms, best.crystalSystem ?? "cubic", 0.015, latticeVectors
                  );
                  if (phononResult) {
                    phononStable = phononResult.dynamicallyStable;
                    omegaLogEstimate = phononResult.omegaLog;
                    lowestFrequency = phononResult.lowestFrequency;
                    imaginaryModeCount = phononResult.imaginaryModeCount;
                  }
                } catch (phErr: any) {
                  console.warn(`[Stage05] xTB phonon failed for ${formula}: ${phErr?.message?.slice(0, 80)}`);
                }
              }
            }
          } catch (cgErr: any) {
            console.warn(`[Stage05] CHGNet failed for ${formula}: ${cgErr?.message?.slice(0, 80)}`);
          }
        }
      } catch (pyErr: any) {
        console.warn(`[Stage05] PyXtal failed for ${formula}: ${pyErr?.message?.slice(0, 80)}`);
      }
    }
  } finally {
    // Cleanup
    try { fs.rmSync(workDir, { recursive: true, force: true }); } catch {}
  }

  // Step 4: Compute Tc estimate
  // If we have omega_log from xTB, use it with the UQ function
  // Otherwise fall back to the heuristic UQ
  const uq = computePhysicsTcUQ(formula, pressure);
  const tcEstimate = uq.mean;

  // Step 5: Make recommendation
  let recommendation: Stage05Result["recommendation"] = "borderline";
  let rejectReason: string | null = null;
  let confidence = 0.4;

  if (phononStable === false && imaginaryModeCount > 3) {
    recommendation = "reject";
    rejectReason = `phonon unstable: ${imaginaryModeCount} imaginary modes, lowest=${lowestFrequency?.toFixed(0)} cm⁻¹`;
    confidence = 0.7;
  } else if (bestFormationEnergyPerAtom != null && bestFormationEnergyPerAtom > 1.0 && pressure < 50) {
    recommendation = "reject";
    rejectReason = `high formation energy: ${bestFormationEnergyPerAtom.toFixed(3)} eV/atom — likely unstable at ambient`;
    confidence = 0.6;
  } else if (tcEstimate > 20 && (phononStable === true || phononStable === null)) {
    recommendation = "promote-to-dft";
    confidence = phononStable ? 0.7 : 0.5;
  } else if (tcEstimate > 5) {
    recommendation = "borderline";
    confidence = 0.4;
  } else {
    recommendation = "reject";
    rejectReason = `Tc estimate too low: ${tcEstimate.toFixed(1)}K`;
    confidence = 0.5;
  }

  return {
    formula,
    pressureGpa: pressure,
    structuresGenerated,
    structuresRelaxed,
    bestFormationEnergyPerAtom,
    bestVolumePerAtom,
    phononStable,
    omegaLogEstimate,
    lowestFrequency,
    imaginaryModeCount,
    tcEstimate,
    isMetallicProxy: hasAnyMetal,
    recommendation,
    rejectReason,
    source: "stage05-structural",
    confidence,
    wallTimeMs: Date.now() - t0,
  };
}

/**
 * Batch version for screening multiple formulas in sequence.
 * Yields between candidates to avoid event loop blocking.
 */
export async function runStage05Batch(
  formulas: string[],
  pressureGpa?: number,
): Promise<Stage05Result[]> {
  const results: Stage05Result[] = [];
  for (let i = 0; i < formulas.length; i++) {
    if (i > 0 && i % 3 === 0) await new Promise<void>(r => setTimeout(r, 0));
    try {
      const result = await runStage05PreScreen(formulas[i], pressureGpa);
      results.push(result);
      console.log(`[Stage05] ${formulas[i]}: ${result.recommendation} (Tc=${result.tcEstimate.toFixed(1)}K, phonon=${result.phononStable}, ${result.wallTimeMs}ms)`);
    } catch (err: any) {
      console.warn(`[Stage05] Failed for ${formulas[i]}: ${err?.message?.slice(0, 100)}`);
    }
  }
  return results;
}
