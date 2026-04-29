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
"""CHGNet MLIP energy evaluation and optional relaxation."""
import os, sys, json, warnings, traceback
warnings.filterwarnings("ignore")

# Limit threads to avoid competing with QE on shared worker
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

try:
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer
    from pymatgen.core import Structure
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}"}))
    sys.exit(1)

try:
    # Load pre-trained model (inference only, no training)
    model = CHGNet.load()

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

            # Single-point energy prediction (~0.1s)
            prediction = model.predict_structure(struct)
            energy_per_atom = float(prediction["e"])
            forces = prediction["f"]
            max_force = float(max(abs(f).max() for f in forces)) if forces is not None else None
            stress = prediction.get("s", None)

            result = {
                "file": fname,
                "n_atoms": n_atoms,
                "energy_per_atom_ev": energy_per_atom,
                "total_energy_ev": energy_per_atom * n_atoms,
                "max_force_ev_ang": max_force,
                "relaxed": False,
            }

            # Optional relaxation for top candidates
            if do_relax and max_force is not None and max_force > 0.05:
                try:
                    optimizer = StructOptimizer()
                    relax_result = optimizer.relax(
                        struct,
                        fmax=0.05,
                        steps=50,
                        verbose=False,
                    )
                    relaxed_struct = relax_result["final_structure"]
                    relaxed_pred = model.predict_structure(relaxed_struct)
                    result["relaxed"] = True
                    result["relaxed_energy_per_atom_ev"] = float(relaxed_pred["e"])
                    result["relaxed_total_energy_ev"] = float(relaxed_pred["e"]) * len(relaxed_struct)
                    relaxed_forces = relaxed_pred["f"]
                    result["relaxed_max_force"] = float(max(abs(f).max() for f in relaxed_forces)) if relaxed_forces is not None else None

                    # Write relaxed structure
                    relaxed_path = os.path.join(poscar_dir, f"relaxed_{fname}")
                    relaxed_struct.to(fmt="poscar", filename=relaxed_path)
                    result["relaxed_file"] = f"relaxed_{fname}"
                except Exception as relax_err:
                    result["relax_error"] = str(relax_err)[:80]

            results.append(result)

            if (i + 1) % 50 == 0:
                print(f"CHGNET_PROGRESS {i+1}/{len(files)}", flush=True)

        except Exception as struct_err:
            results.append({
                "file": fname,
                "error": str(struct_err)[:80],
            })

    # Sort by energy (lowest first)
    valid_results = [r for r in results if "energy_per_atom_ev" in r]
    valid_results.sort(key=lambda r: r.get("relaxed_energy_per_atom_ev", r["energy_per_atom_ev"]))

    output = {
        "total": len(files),
        "evaluated": len(valid_results),
        "failed": len(results) - len(valid_results),
        "results": valid_results,
        "best_energy": valid_results[0]["energy_per_atom_ev"] if valid_results else None,
        "worst_energy": valid_results[-1]["energy_per_atom_ev"] if valid_results else None,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"CHGNET_DONE evaluated={len(valid_results)} failed={len(results) - len(valid_results)} best={output['best_energy']:.4f} worst={output['worst_energy']:.4f}")

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
  stats: { evaluated: number; failed: number; bestEnergy: number | null };
}> {
  if (!isChgnetAvailable()) {
    console.log("[CHGNet] Not available — skipping F6 MLIP evaluation");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, failed: 0, bestEnergy: null } };
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
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, failed: 0, bestEnergy: null } };
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
      { cwd: workDir, timeout: timeoutMs, maxBuffer: 10 * 1024 * 1024 }
    );
    const output = result.toString();

    // Log progress
    const doneMatch = output.match(/CHGNET_DONE evaluated=(\d+) failed=(\d+) best=([-\d.]+) worst=([-\d.]+)/);
    if (doneMatch) {
      console.log(`[CHGNet] Done: ${doneMatch[1]} evaluated, ${doneMatch[2]} failed, best=${doneMatch[3]} eV/atom, worst=${doneMatch[4]} eV/atom`);
    }
    if (output.includes("CHGNET_FATAL")) {
      console.log(`[CHGNet] Fatal error: ${output.slice(output.indexOf("CHGNET_FATAL"), output.indexOf("CHGNET_FATAL") + 200)}`);
      return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, failed: 0, bestEnergy: null } };
    }
  } catch (err: any) {
    console.log(`[CHGNet] Evaluation failed: ${err.message?.slice(0, 100)}`);
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, failed: 0, bestEnergy: null } };
  }

  // Parse results
  if (!fs.existsSync(outputPath)) {
    console.log("[CHGNet] No output file produced");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, failed: 0, bestEnergy: null } };
  }

  let parsed: any;
  try {
    parsed = JSON.parse(fs.readFileSync(outputPath, "utf-8"));
  } catch {
    console.log("[CHGNet] Failed to parse output JSON");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, failed: 0, bestEnergy: null } };
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
  }));

  // Attach MLIP energy to candidates and sort by energy
  const ranked: CSPCandidate[] = [];
  for (const r of results) {
    const candidate = candidateMap.get(r.file);
    if (candidate) {
      // Attach energy to candidate
      candidate.enthalpyPerAtom = r.relaxedEnergyPerAtomEv ?? r.energyPerAtomEv;
      candidate.enthalpy = (r.relaxedEnergyPerAtomEv ?? r.energyPerAtomEv) * r.nAtoms;
      if (r.maxForceEvAng != null) {
        // Convert eV/Å to Ry/bohr (1 Ry/bohr = 25.711 eV/Å)
        candidate.postRelaxForce = r.maxForceEvAng / 25.711;
      }
      ranked.push(candidate);
    }
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
      failed: parsed.failed ?? 0,
      bestEnergy: parsed.best_energy ?? null,
    },
  };
}
