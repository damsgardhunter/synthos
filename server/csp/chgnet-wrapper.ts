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
import { execSync, exec } from "child_process";
import { promisify } from "util";
import type { CSPCandidate } from "./csp-types";

const execAsync = promisify(exec);
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
  pressureGPa: number,
  relaxTopN: number,
): string {
  return `#!/usr/bin/env python3
"""CHGNet MLIP energy evaluation with constant-pressure cell relaxation."""
import os, sys, json, warnings, traceback, time
warnings.filterwarnings("ignore")

# Limit threads to avoid competing with QE on shared worker
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

try:
    from chgnet.model import CHGNet
    from pymatgen.core import Structure
    import numpy as np
except ImportError as e:
    print(json.dumps({"error": f"Import failed: {e}"}))
    sys.exit(1)

# --- Constant-pressure relaxation backend ---------------------------------
# CHGNet's StructOptimizer.relax() minimizes the energy at AMBIENT pressure
# only — it has no external-pressure term. Relaxing a high-pressure structure
# that way decompresses it to its ambient volume (~2x larger), which made the
# drift gate reject most high-P candidates and corrupted the volume-bias
# cache. Instead we drive an ASE cell filter that minimizes the ENTHALPY
# H = E + P*V at the target pressure. FrechetCellFilter is ASE >= 3.23;
# ExpCellFilter is the older equivalent. If neither (or the CHGNet ASE
# calculator) is importable, fall back to StructOptimizer (ambient) with a
# warning so the degraded behaviour is visible in the logs.
_CellFilter = None
_FIRE = None
_AseAdaptor = None
_CHGNetCalculator = None
try:
    from ase.optimize import FIRE as _FIRE
    from pymatgen.io.ase import AseAtomsAdaptor as _AseAdaptor
    from chgnet.model.dynamics import CHGNetCalculator as _CHGNetCalculator
    try:
        from ase.filters import FrechetCellFilter as _CellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter as _CellFilter
except Exception:
    _CellFilter = None

# ASE cell filters take scalar_pressure in eV/Angstrom^3. 1 eV/A^3 = 160.21766208 GPa.
GPA_TO_EV_A3 = 1.0 / 160.21766208

try:
    # Load pre-trained model (inference only, no training)
    model = CHGNet.load()

    poscar_dir = ${JSON.stringify(poscarDir.replace(/\\/g, "/"))}
    output_path = ${JSON.stringify(outputPath.replace(/\\/g, "/"))}
    do_relax = ${doRelax ? "True" : "False"}
    max_structures = ${maxStructures}
    # Only the top relax_top_n candidates (by preScore — input order) get the
    # expensive cell relaxation; the rest get a cheap single-point energy. This
    # keeps the batch inside its wall-time budget instead of timing out.
    relax_top_n = ${Number.isFinite(relaxTopN) ? Math.max(0, Math.round(relaxTopN)) : 1_000_000}
    pressure_gpa = ${Number.isFinite(pressureGPa) ? pressureGPa : 0}
    scalar_pressure = pressure_gpa * GPA_TO_EV_A3

    # Pick the relaxation backend: constant-pressure ASE path if available,
    # else ambient StructOptimizer fallback.
    ase_calc = None
    struct_optimizer = None
    if _CellFilter is not None and _CHGNetCalculator is not None and _FIRE is not None and _AseAdaptor is not None:
        try:
            ase_calc = _CHGNetCalculator(model=model)
        except Exception:
            ase_calc = None
    if ase_calc is None:
        from chgnet.model.dynamics import StructOptimizer
        struct_optimizer = StructOptimizer()
        if pressure_gpa > 0:
            print("CHGNET_WARN constant-pressure relaxation unavailable (ASE cell filter/calculator missing) - relaxing at ambient P; high-P volumes will be wrong", flush=True)
    print(f"CHGNET_INFO relax_mode={'const-P' if ase_calc is not None else 'ambient-fallback'} pressure_gpa={pressure_gpa}", flush=True)

    def relax_structure(struct, fmax, steps):
        """Relax cell + positions. ASE path minimizes H = E + P*V at the
        target pressure; the fallback minimizes E at ambient pressure."""
        if ase_calc is not None:
            atoms = _AseAdaptor.get_atoms(struct)
            atoms.calc = ase_calc
            ucf = _CellFilter(atoms, scalar_pressure=scalar_pressure) if scalar_pressure > 0 else _CellFilter(atoms)
            opt = _FIRE(ucf, logfile=None)
            opt.run(fmax=fmax, steps=steps)
            return _AseAdaptor.get_structure(atoms)
        rr = struct_optimizer.relax(struct, fmax=fmax, steps=steps, verbose=False)
        return rr["final_structure"]

    results = []
    files = sorted([f for f in os.listdir(poscar_dir) if f.endswith(".vasp") or f.startswith("POSCAR")])[:max_structures]

    def write_output():
        """Serialize current results to disk atomically. Called periodically so
        a timeout kill still leaves a usable partial file rather than nothing."""
        valid = [r for r in results if "energy_per_atom_ev" in r]
        valid.sort(key=lambda r: r.get("relaxed_energy_per_atom_ev", r["energy_per_atom_ev"]))
        rlx = [r["relaxed_energy_per_atom_ev"] for r in valid if r.get("relaxed")]
        unrlx = [r["energy_per_atom_ev"] for r in valid]
        trt = sum(r.get("relax_time_s", 0) for r in valid)
        be = rlx[0] if rlx else (unrlx[0] if unrlx else None)
        we = rlx[-1] if rlx else (unrlx[-1] if unrlx else None)
        out = {
            "total": len(files),
            "evaluated": len(valid),
            "relaxed": len(rlx),
            "failed": len(results) - len(valid),
            "results": valid,
            "best_energy": be,
            "worst_energy": we,
            "total_relax_time_s": round(trt, 1),
        }
        tmp = output_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, output_path)
        return out, be, we, trt

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

            if do_relax and i < relax_top_n:
                # Scale steps with atom count: more atoms need more steps to
                # converge. Small cells (< 10 atoms): 100 steps, large cells
                # (50+ atoms): 500 steps.
                max_steps = min(500, max(100, n_atoms * 10))
                fmax_target = 0.02

                try:
                    relaxed_struct = relax_structure(struct, fmax_target, max_steps)
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

            if (i + 1) % 8 == 0:
                write_output()
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"CHGNET_PROGRESS {i+1}/{len(files)} last_relax={elapsed:.1f}s", flush=True)

        except Exception as struct_err:
            results.append({
                "file": fname,
                "error": str(struct_err)[:100],
            })

    # Final write (also captures the trailing structures since the last checkpoint)
    output, best_e, worst_e, total_relax_time = write_output()

    if best_e is not None and worst_e is not None:
        print(f"CHGNET_DONE evaluated={output['evaluated']} relaxed={output['relaxed']} failed={output['failed']} best={best_e:.4f} worst={worst_e:.4f} time={total_relax_time:.0f}s")
    else:
        print(f"CHGNET_DONE evaluated=0 relaxed=0 failed={output['failed']} best=0.0 worst=0.0 time=0s")

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
  pressureGPa: number = 0,
  relaxTopN: number = 1_000_000, // relax all by default; callers cap this
): Promise<{
  rankedCandidates: CSPCandidate[];
  results: ChgnetResult[];
  stats: { evaluated: number; relaxed: number; failed: number; bestEnergy: number | null; totalRelaxTimeS: number };
  /**
   * Recommended multiplier for the NEXT CSP batch's target volume. 1.0 means
   * no systematic bias detected; >1 means CSP volumes were too compressed,
   * <1 means too expanded. Damped relative to the raw CHGNet drift.
   */
  volumeBiasFactor: number;
}> {
  if (!isChgnetAvailable()) {
    console.log("[CHGNet] Not available — skipping F6 MLIP evaluation");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 }, volumeBiasFactor: 1.0 };
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
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 }, volumeBiasFactor: 1.0 };
  }

  // Generate and run CHGNet script
  const outputPath = path.join(workDir, "chgnet_results.json");
  const scriptPath = path.join(workDir, "chgnet_eval.py");
  const script = generateChgnetScript(poscarDir, outputPath, doRelax, maxStructures, pressureGPa, relaxTopN);
  fs.writeFileSync(scriptPath, script);

  const relaxCount = doRelax ? Math.min(candidateMap.size, relaxTopN) : 0;
  console.log(`[CHGNet] Evaluating ${candidateMap.size} candidates (relax top ${relaxCount}, single-point ${candidateMap.size - relaxCount}, timeout=${Math.round(timeoutMs / 1000)}s)`);

  // IMPORTANT: this MUST be an async spawn. execSync froze the Node event loop
  // for the entire batch (relax=true batches run multiple hours), which stalled
  // every other timer in the worker — the 90 s Vegard/candidate-fetch timeout
  // could not even fire. The Python script checkpoints results to disk every
  // few structures, so a timeout here still leaves a usable partial file: log
  // and fall through to the parse step rather than discarding everything.
  try {
    const result = await execAsync(
      `${PYTHON_BIN} ${scriptPath} 2>&1`,
      { cwd: workDir, timeout: Math.round(timeoutMs), maxBuffer: 10 * 1024 * 1024 }
    );
    const output = result.stdout.toString();

    // Log progress
    const doneMatch = output.match(/CHGNET_DONE evaluated=(\d+) relaxed=(\d+) failed=(\d+) best=([-\d.]+) worst=([-\d.]+) time=(\d+)s/);
    if (doneMatch) {
      console.log(`[CHGNet] Done: ${doneMatch[1]} evaluated, ${doneMatch[2]} relaxed, ${doneMatch[3]} failed, best=${doneMatch[4]} eV/atom, worst=${doneMatch[5]} eV/atom, time=${doneMatch[6]}s`);
    }
    if (output.includes("CHGNET_FATAL")) {
      console.log(`[CHGNet] Fatal error: ${output.slice(output.indexOf("CHGNET_FATAL"), output.indexOf("CHGNET_FATAL") + 200)}`);
      return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 }, volumeBiasFactor: 1.0 };
    }
  } catch (err: any) {
    // Timeout or non-zero exit. The script checkpoints to disk, so partial
    // results may still be on disk — fall through to the parse step.
    console.log(`[CHGNet] Evaluation did not complete cleanly (${err.message?.slice(0, 80)}) — attempting to salvage checkpointed results`);
  }

  // Parse results
  if (!fs.existsSync(outputPath)) {
    console.log("[CHGNet] No output file produced");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 }, volumeBiasFactor: 1.0 };
  }

  let parsed: any;
  try {
    parsed = JSON.parse(fs.readFileSync(outputPath, "utf-8"));
  } catch {
    console.log("[CHGNet] Failed to parse output JSON");
    return { rankedCandidates: candidates, results: [], stats: { evaluated: 0, relaxed: 0, failed: 0, bestEnergy: null, totalRelaxTimeS: 0 }, volumeBiasFactor: 1.0 };
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
  // If CHGNet drifts too far (volume change > threshold, distance collapse,
  // etc.), keep the raw CSP geometry — ML potentials can over-compress or
  // bias structures, especially outside their training distribution.
  //
  // Pressure-aware threshold: CHGNet was trained mostly on ambient-P data
  // from Materials Project. At high P (>50 GPa) MLIP volumes are unreliable
  // and we want a TIGHTER threshold (30%) to reject more aggressively. At
  // ambient P, MLIP relaxation is more trustworthy and a 30% reject was
  // catching real over-expanded starting structures — loosen to 40%.
  const driftVolPctThreshold = pressureGPa >= 50 ? 30 : 40;
  const ranked: CSPCandidate[] = [];
  let driftRejected = 0;
  let totalCompressionPct = 0;
  let totalExpansionPct = 0;
  let nCompressions = 0;
  let nExpansions = 0;
  for (const r of results) {
    const candidate = candidateMap.get(r.file);
    if (candidate) {
      // Use relaxed force if available, otherwise single-point force
      const forceEvAng = r.relaxedMaxForce ?? r.maxForceEvAng;
      if (forceEvAng != null) {
        // Convert eV/Å to Ry/bohr (1 Ry/bohr = 25.711 eV/Å)
        candidate.postRelaxForce = forceEvAng / 25.711;
      }

      // --- CHGNet drift gate ---
      let useRelaxedEnergy = true;
      // Save pre-MLIP snapshot before potentially overwriting
      if (r.relaxed && r.relaxedLatticeA != null) {
        candidate.preMLIPLatticeA = candidate.latticeA;
        candidate.preMLIPLatticeB = candidate.latticeB;
        candidate.preMLIPLatticeC = candidate.latticeC;
        candidate.preMLIPVolume = candidate.cellVolume;
        candidate.mlipVolumeChangePct = r.volumeChangePct ?? 0;

        const volChangePctRaw = r.volumeChangePct ?? 0;
        const volChangePct = Math.abs(volChangePctRaw);
        if (volChangePctRaw < 0) { totalCompressionPct += volChangePct; nCompressions++; }
        else if (volChangePctRaw > 0) { totalExpansionPct += volChangePct; nExpansions++; }

        // Check for excessive drift
        let driftTooHigh = false;
        let driftReason = "";

        if (volChangePct > driftVolPctThreshold) {
          driftTooHigh = true;
          driftReason = `volume change ${volChangePctRaw.toFixed(1)}% > ${driftVolPctThreshold}% (P=${pressureGPa} GPa threshold)`;
        }

        // Minimum distance collapse check: if relaxed lattice is very small,
        // MLIP may have crushed the structure
        if (r.relaxedLatticeA != null && r.relaxedLatticeA < candidate.latticeA * 0.5) {
          driftTooHigh = true;
          driftReason = `lattice collapsed (${candidate.latticeA.toFixed(2)} → ${r.relaxedLatticeA.toFixed(2)} Å)`;
        }

        if (driftTooHigh) {
          // Keep raw CSP geometry — and use SINGLE-POINT energy to match
          // (the relaxed energy doesn't correspond to a geometry we kept).
          candidate.mlipDriftRejected = true;
          candidate.relaxationLevel = "raw";
          useRelaxedEnergy = false;
          driftRejected++;
          console.log(`[CHGNet] Drift gate REJECTED for ${r.file}: ${driftReason} — keeping raw CSP geometry, using single-point energy`);
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

      // Attach energy: use single-point when drift-rejected (matches the raw
      // geometry we kept); otherwise use relaxed.
      const energyToUse = useRelaxedEnergy
        ? (r.relaxedEnergyPerAtomEv ?? r.energyPerAtomEv)
        : r.energyPerAtomEv;
      candidate.enthalpyPerAtom = energyToUse;
      candidate.enthalpy = energyToUse * r.nAtoms;

      ranked.push(candidate);
    }
  }

  if (driftRejected > 0) {
    console.log(`[CHGNet] Drift gate: ${driftRejected}/${results.length} candidates had excessive MLIP drift — raw CSP geometry preserved`);
  }
  // Systematic-volume-bias hint: if most candidates compress or expand by
  // similar amounts, the CSP volume prior is off. We turn the hint into a
  // damped correction multiplier (volumeBiasFactor) that the funnel persists
  // so the NEXT CSP batch for this composition starts at corrected volumes.
  // Damping (0.7 of the raw drift) avoids overshoot — a single CHGNet volume
  // is not DFT-exact, and the correction compounds across runs anyway.
  let volumeBiasFactor = 1.0;
  if (nCompressions + nExpansions >= 10) {
    const total = nCompressions + nExpansions;
    const avgComp = nCompressions > 0 ? totalCompressionPct / nCompressions : 0;
    const avgExp = nExpansions > 0 ? totalExpansionPct / nExpansions : 0;
    if (nCompressions > 0.7 * total && avgComp > 15) {
      // Relaxed volumes smaller than raw → CSP volumes too large → shrink.
      const observed = Math.max(0.4, 1 - avgComp / 100);
      volumeBiasFactor = 1 + 0.7 * (observed - 1);
      console.log(`[CHGNet] Volume-bias hint: ${nCompressions}/${total} candidates compressed by avg ${avgComp.toFixed(1)}% — CSP starting volumes look systematically too expanded (next-batch TARGVOL ×${volumeBiasFactor.toFixed(2)})`);
    } else if (nExpansions > 0.7 * total && avgExp > 15) {
      // Relaxed volumes larger than raw → CSP volumes too small → grow.
      const observed = 1 + avgExp / 100;
      volumeBiasFactor = 1 + 0.7 * (observed - 1);
      console.log(`[CHGNet] Volume-bias hint: ${nExpansions}/${total} candidates expanded by avg ${avgExp.toFixed(1)}% — CSP starting volumes look systematically too compressed (next-batch TARGVOL ×${volumeBiasFactor.toFixed(2)})`);
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
      relaxed: parsed.relaxed ?? 0,
      failed: parsed.failed ?? 0,
      bestEnergy: parsed.best_energy ?? null,
      totalRelaxTimeS: parsed.total_relax_time_s ?? 0,
    },
    volumeBiasFactor,
  };
}
