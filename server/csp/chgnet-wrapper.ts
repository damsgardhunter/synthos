/**
 * CHGNet MLIP Wrapper for F6 Pre-Relaxation
 *
 * CHGNet is a universal machine learning interatomic potential trained on
 * Materials Project data. It provides:
 * - Energy prediction (~1 meV/atom accuracy for known chemistries)
 * - Force prediction
 * - Stress prediction
 * - Structure relaxation (100-1000× faster than DFT)
 *
 * Used in the candidate funnel as F6:
 * - Evaluate energy for all post-F5 candidates (~0.1s per structure)
 * - Optionally relax promising candidates (~1-5s per structure)
 * - Rank by MLIP energy → send lowest-energy to DFT
 *
 * CHGNet runs inference only (no training) to avoid disrupting
 * the DFT pipeline on the GCP worker.
 *
 * Installation: pip install chgnet
 * Reference: Deng et al., Nature Machine Intelligence 5 (2023) 1031
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import type { CSPCandidate } from "./csp-types";
import { writePOSCAR } from "./poscar-io";

const PYTHON_BIN = process.env.PYTHON_BIN ?? "python3";

// ---------------------------------------------------------------------------
// Availability check
// ---------------------------------------------------------------------------

let _chgnetAvailable: boolean | null = null;

export function isChgnetAvailable(): boolean {
  if (_chgnetAvailable !== null) return _chgnetAvailable;
  try {
    const result = execSync(
      `${PYTHON_BIN} -c "from chgnet.model import CHGNet; print('ok')" 2>&1`,
      { timeout: 30000 }
    );
    _chgnetAvailable = result.toString().trim().includes("ok");
    if (_chgnetAvailable) console.log("[CHGNet] Available via " + PYTHON_BIN);
  } catch {
    _chgnetAvailable = false;
  }
  return _chgnetAvailable;
}

// ---------------------------------------------------------------------------
// Python script generation
// ---------------------------------------------------------------------------

function generateChgnetScript(
  poscarDir: string,
  outputPath: string,
  doRelax: boolean,
  maxStructures: number,
): string {
  return `#!/usr/bin/env python3
"""CHGNet MLIP energy evaluation with full relaxation."""
import os, sys, json, warnings, traceback, time
warnings.filterwarnings("ignore")

# Limit threads to avoid competing with QE on shared worker
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

try:
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer
    from pymatgen.core import Structure
    import numpy as np
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}"}))
    sys.exit(1)

try:
    # Load pre-trained model (inference only, no training)
    model = CHGNet.load()
    optimizer = StructOptimizer()

    poscar_dir = ${JSON.stringify(poscarDir.replace(/\\/g, "/"))}
    output_path = ${JSON.stringify(outputPath.replace(/\\/g, "/"))}
    do_relax = ${doRelax ? "True" : "False"}
    max_structures = ${maxStructures}

    results = []
    files = sorted([f for f in os.listdir(poscar_dir) if f.endswith(".vasp") or f.startswith("POSCAR")])[:max_structures]

    for i, fname in enumerate(files):
        fpath = os.path.join(poscar_dir, fname)
        try:
            struct = Structure.from_file(fpath)
            n_atoms = len(struct)
            t0 = time.time()

            # Single-point energy prediction (~0.1s)
            prediction = model.predict_structure(struct)
            energy_per_atom = float(prediction["e"])
            forces = prediction["f"]
            max_force = float(max(abs(f).max() for f in forces)) if forces is not None else None

            result = {
                "file": fname,
                "n_atoms": n_atoms,
                "energy_per_atom_ev": energy_per_atom,
                "total_energy_ev": energy_per_atom * n_atoms,
                "max_force_ev_ang": max_force,
                "relaxed": False,
            }

            # Full relaxation for ALL candidates
            # Scale steps with atom count: more atoms need more steps to converge
            # Small cells (< 10 atoms): 100 steps, large cells (50+ atoms): 500 steps
            max_steps = min(500, max(100, n_atoms * 10))
            fmax_target = 0.02  # tighter than before (was 0.05)

            try:
                relax_result = optimizer.relax(
                    struct,
                    fmax=fmax_target,
                    steps=max_steps,
                    verbose=False,
                )
                relaxed_struct = relax_result["final_structure"]
                relaxed_pred = model.predict_structure(relaxed_struct)

                relaxed_e = float(relaxed_pred["e"])
                relaxed_forces = relaxed_pred["f"]
                relaxed_max_force = float(max(abs(f).max() for f in relaxed_forces)) if relaxed_forces is not None else None

                # Track volume change during relaxation
                orig_vol = struct.volume
                relaxed_vol = relaxed_struct.volume
                vol_change_pct = (relaxed_vol - orig_vol) / orig_vol * 100

                result["relaxed"] = True
                result["relaxed_energy_per_atom_ev"] = relaxed_e
                result["relaxed_total_energy_ev"] = relaxed_e * len(relaxed_struct)
                result["relaxed_max_force"] = relaxed_max_force
                result["relaxed_volume"] = relaxed_vol
                result["volume_change_pct"] = round(vol_change_pct, 2)
                result["relax_steps"] = max_steps
                result["relax_time_s"] = round(time.time() - t0, 2)

                # Extract relaxed lattice parameters
                latt = relaxed_struct.lattice
                result["relaxed_lattice_a"] = round(latt.a, 4)
                result["relaxed_lattice_b"] = round(latt.b, 4)
                result["relaxed_lattice_c"] = round(latt.c, 4)
                result["relaxed_alpha"] = round(latt.alpha, 2)
                result["relaxed_beta"] = round(latt.beta, 2)
                result["relaxed_gamma"] = round(latt.gamma, 2)

                # Write relaxed structure as POSCAR for DFT follow-up
                relaxed_path = os.path.join(poscar_dir, f"relaxed_{fname}")
                relaxed_struct.to(fmt="poscar", filename=relaxed_path)
                result["relaxed_file"] = f"relaxed_{fname}"

            except Exception as relax_err:
                result["relax_error"] = str(relax_err)[:100]
                result["relax_time_s"] = round(time.time() - t0, 2)

            results.append(result)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"CHGNET_PROGRESS {i+1}/{len(files)} last_relax={elapsed:.1f}s", flush=True)

        except Exception as struct_err:
            results.append({
                "file": fname,
                "error": str(struct_err)[:100],
            })

    # Sort by relaxed energy (lowest first), fall back to single-point
    valid_results = [r for r in results if "energy_per_atom_ev" in r]
    valid_results.sort(key=lambda r: r.get("relaxed_energy_per_atom_ev", r["energy_per_atom_ev"]))

    # Compute stats
    relaxed_energies = [r["relaxed_energy_per_atom_ev"] for r in valid_results if r.get("relaxed")]
    unrelaxed_energies = [r["energy_per_atom_ev"] for r in valid_results]
    total_relax_time = sum(r.get("relax_time_s", 0) for r in valid_results)

    best_e = relaxed_energies[0] if relaxed_energies else (unrelaxed_energies[0] if unrelaxed_energies else None)
    worst_e = relaxed_energies[-1] if relaxed_energies else (unrelaxed_energies[-1] if unrelaxed_energies else None)

    output = {
        "total": len(files),
        "evaluated": len(valid_results),
        "relaxed": len(relaxed_energies),
        "failed": len(results) - len(valid_results),
        "results": valid_results,
        "best_energy": best_e,
        "worst_energy": worst_e,
        "total_relax_time_s": round(total_relax_time, 1),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"CHGNET_DONE evaluated={len(valid_results)} relaxed={len(relaxed_energies)} failed={len(results) - len(valid_results)} best={best_e:.4f} worst={worst_e:.4f} time={total_relax_time:.0f}s")

except Exception as fatal:
    print(f"CHGNET_FATAL {traceback.format_exc()}")
    sys.exit(1)
`;
}

// ---------------------------------------------------------------------------
// Main interface
// ---------------------------------------------------------------------------

export interface ChgnetResult {
  file: string;
  nAtoms: number;
  energyPerAtomEv: number;
  totalEnergyEv: number;
  maxForceEvAng: number | null;
  relaxed: boolean;
  relaxedEnergyPerAtomEv?: number;
  relaxedMaxForce?: number;
  relaxedFile?: string;
  relaxedVolume?: number;
  volumeChangePct?: number;
  relaxedLatticeA?: number;
  relaxedLatticeB?: number;
  relaxedLatticeC?: number;
  relaxTimeS?: number;
}

/**
 * Run CHGNet energy evaluation on a batch of candidates.
 *
 * @param candidates - CSP candidates to evaluate
 * @param workDir - Working directory for POSCAR files
 * @param doRelax - Whether to also relax promising candidates
 * @param maxStructures - Max structures to evaluate
 * @param timeoutMs - Timeout for the entire batch
 * @returns Candidates sorted by MLIP energy (lowest first), with energy attached
 */
export async function runChgnetEvaluation(
  candidates: CSPCandidate[],
  workDir: string,
  doRelax: boolean = false,
  maxStructures: number = 300,
  timeoutMs: number = 300000, // 5 min default
): Promise<{
  rankedCandidates: CSPCandidate[];
  results: ChgnetResult[];
  stats: { evaluated: number; relaxed: number; failed: number; bestEnergy: number | null; totalRelaxTimeS: number };
}> {
  if (!isChgnetAvailable()) {
    console.log("[CHGNet] Not available — skipping F6 MLIP evaluation");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 } };
  }

  const poscarDir = path.join(workDir, "chgnet_poscars");
  fs.mkdirSync(poscarDir, { recursive: true });

  // Write candidates as POSCAR files
  const candidateMap = new Map<string, CSPCandidate>();
  const toEvaluate = candidates.slice(0, maxStructures);

  for (let i = 0; i < toEvaluate.length; i++) {
    const fname = `POSCAR_${String(i).padStart(4, "0")}.vasp`;
    try {
      fs.writeFileSync(path.join(poscarDir, fname), writePOSCAR(toEvaluate[i]));
      candidateMap.set(fname, toEvaluate[i]);
    } catch {}
  }

  if (candidateMap.size === 0) {
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 } };
  }

  // Generate and run CHGNet script
  const outputPath = path.join(workDir, "chgnet_results.json");
  const scriptPath = path.join(workDir, "chgnet_eval.py");
  const script = generateChgnetScript(poscarDir, outputPath, doRelax, maxStructures);
  fs.writeFileSync(scriptPath, script);

  console.log(`[CHGNet] Evaluating ${candidateMap.size} candidates (relax=${doRelax}, timeout=${Math.round(timeoutMs / 1000)}s)`);

  try {
    const result = execSync(
      `${PYTHON_BIN} ${scriptPath} 2>&1`,
      { cwd: workDir, timeout: Math.round(timeoutMs), maxBuffer: 10 * 1024 * 1024 }
    );
    const output = result.toString();

    // Log progress
    const doneMatch = output.match(/CHGNET_DONE evaluated=(\d+) relaxed=(\d+) failed=(\d+) best=([-\d.]+) worst=([-\d.]+) time=(\d+)s/);
    if (doneMatch) {
      console.log(`[CHGNet] Done: ${doneMatch[1]} evaluated, ${doneMatch[2]} relaxed, ${doneMatch[3]} failed, best=${doneMatch[4]} eV/atom, worst=${doneMatch[5]} eV/atom, time=${doneMatch[6]}s`);
    }
    if (output.includes("CHGNET_FATAL")) {
      console.log(`[CHGNet] Fatal error: ${output.slice(output.indexOf("CHGNET_FATAL"), output.indexOf("CHGNET_FATAL") + 200)}`);
      return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 } };
    }
  } catch (err: any) {
    console.log(`[CHGNet] Evaluation failed: ${err.message?.slice(0, 100)}`);
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 } };
  }

  // Parse results
  if (!fs.existsSync(outputPath)) {
    console.log("[CHGNet] No output file produced");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 } };
  }

  let parsed: any;
  try {
    parsed = JSON.parse(fs.readFileSync(outputPath, "utf-8"));
  } catch {
    console.log("[CHGNet] Failed to parse output JSON");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 } };
  }

  const results: ChgnetResult[] = (parsed.results ?? []).map((r: any) => ({
    file: r.file,
    nAtoms: r.n_atoms ?? 0,
    energyPerAtomEv: r.energy_per_atom_ev ?? 0,
    totalEnergyEv: r.total_energy_ev ?? 0,
    maxForceEvAng: r.max_force_ev_ang ?? null,
    relaxed: r.relaxed ?? false,
    relaxedEnergyPerAtomEv: r.relaxed_energy_per_atom_ev,
    relaxedMaxForce: r.relaxed_max_force,
    relaxedFile: r.relaxed_file,
    relaxedVolume: r.relaxed_volume,
    volumeChangePct: r.volume_change_pct,
    relaxedLatticeA: r.relaxed_lattice_a,
    relaxedLatticeB: r.relaxed_lattice_b,
    relaxedLatticeC: r.relaxed_lattice_c,
    relaxTimeS: r.relax_time_s,
  }));

  // Attach MLIP energy + relaxed geometry to candidates and sort by energy.
  // DRIFT GATE: track both raw CSP and CHGNet-relaxed structures.
  // If CHGNet drifts too far (volume change > 30%, distance collapse, etc.),
  // keep the raw CSP geometry instead — ML potentials can collapse or bias
  // structures incorrectly, especially for high-pressure hydrides.
  const ranked: CSPCandidate[] = [];
  let driftRejected = 0;
  for (const r of results) {
    const candidate = candidateMap.get(r.file);
    if (candidate) {
      // Attach relaxed energy (always prefer relaxed over single-point)
      candidate.enthalpyPerAtom = r.relaxedEnergyPerAtomEv ?? r.energyPerAtomEv;
      candidate.enthalpy = (r.relaxedEnergyPerAtomEv ?? r.energyPerAtomEv) * r.nAtoms;

      // Use relaxed force if available, otherwise single-point force
      const forceEvAng = r.relaxedMaxForce ?? r.maxForceEvAng;
      if (forceEvAng != null) {
        // Convert eV/Å to Ry/bohr (1 Ry/bohr = 25.711 eV/Å)
        candidate.postRelaxForce = forceEvAng / 25.711;
      }

      // --- CHGNet drift gate ---
      // Save pre-MLIP snapshot before potentially overwriting
      if (r.relaxed && r.relaxedLatticeA != null) {
        candidate.preMLIPLatticeA = candidate.latticeA;
        candidate.preMLIPLatticeB = candidate.latticeB;
        candidate.preMLIPLatticeC = candidate.latticeC;
        candidate.preMLIPVolume = candidate.cellVolume;
        candidate.mlipVolumeChangePct = r.volumeChangePct ?? 0;

        const volChangePct = Math.abs(r.volumeChangePct ?? 0);

        // Check for excessive drift
        let driftTooHigh = false;
        let driftReason = "";

        // Volume change > 30% — MLIP likely collapsed or expanded unreasonably
        if (volChangePct > 30) {
          driftTooHigh = true;
          driftReason = `volume change ${(r.volumeChangePct ?? 0).toFixed(1)}% > 30%`;
        }

        // Minimum distance collapse check: if relaxed lattice is very small,
        // MLIP may have crushed the structure
        if (r.relaxedLatticeA != null && r.relaxedLatticeA < candidate.latticeA * 0.5) {
          driftTooHigh = true;
          driftReason = `lattice collapsed (${candidate.latticeA.toFixed(2)} → ${r.relaxedLatticeA.toFixed(2)} Å)`;
        }

        if (driftTooHigh) {
          // Keep raw CSP geometry, mark as drift-rejected
          candidate.mlipDriftRejected = true;
          candidate.relaxationLevel = "raw";
          driftRejected++;
          console.log(`[CHGNet] Drift gate REJECTED for ${r.file}: ${driftReason} — keeping raw CSP geometry`);
        } else {
          // Accept CHGNet-relaxed geometry
          candidate.latticeA = r.relaxedLatticeA;
          if (r.relaxedLatticeB != null) candidate.latticeB = r.relaxedLatticeB;
          if (r.relaxedLatticeC != null) candidate.latticeC = r.relaxedLatticeC;
          if (r.relaxedVolume != null) candidate.cellVolume = r.relaxedVolume;
          candidate.mlipDriftRejected = false;
          candidate.relaxationLevel = "mlip-relaxed";
        }
      }

      ranked.push(candidate);
    }
  }

  if (driftRejected > 0) {
    console.log(`[CHGNet] Drift gate: ${driftRejected}/${results.length} candidates had excessive MLIP drift — raw CSP geometry preserved`);
  }

  // Add unevaluated candidates at the end (worst rank)
  const evaluatedSet = new Set(ranked);
  for (const c of candidates) {
    if (!evaluatedSet.has(c)) ranked.push(c);
  }

  // Cleanup
  try { fs.rmSync(poscarDir, { recursive: true, force: true }); } catch {}
  try { fs.unlinkSync(scriptPath); } catch {}
  try { fs.unlinkSync(outputPath); } catch {}

  return {
    rankedCandidates: ranked,
    results,
    stats: {
      evaluated: parsed.evaluated ?? 0,
      relaxed: parsed.relaxed ?? 0,
      failed: parsed.failed ?? 0,
      bestEnergy: parsed.best_energy ?? null,
      totalRelaxTimeS: parsed.total_relax_time_s ?? 0,
    },
  };
}
