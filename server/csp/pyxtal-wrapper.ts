/**
 * PyXtal wrapper for crystal structure prediction.
 *
 * PyXtal (Python Crystallography library) generates random crystal structures
 * with full space-group symmetry constraints. Unlike AIRSS buildcell (which
 * generates structures with random symmetry), PyXtal can target specific
 * space groups or explore across all 230 space groups systematically.
 *
 * Replaces USPEX in the pipeline — fully open source (MIT license),
 * pip-installable, no compilation needed.
 *
 * Usage: generates a Python script, runs it via the system Python,
 * parses the POSCAR output. Same wrapping pattern as xTB.
 *
 * Reference: Fredericks et al., Comput. Phys. Commun. 261 (2021) 107810
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import type { CSPCandidate, CSPEngine, CSPEngineConfig } from "./csp-types";
import { parsePOSCAR } from "./poscar-io";

const PYTHON_BIN = process.env.PYTHON_BIN ?? "python3";

// Common superconductor-relevant space groups to explore.
// Weighted by frequency in known superconductors.
const SC_SPACE_GROUPS = [
  225, // Fm-3m (LaH10, NaCl, fluorite)
  229, // Im-3m (CaH6, H3S, BCC)
  194, // P63/mmc (YH9, MgB2, HCP)
  139, // I4/mmm (BaFe2As2, ThCr2Si2)
  221, // Pm-3m (perovskite)
  223, // Pm-3n (A15, Nb3Sn)
  166, // R-3m (Bi2Te3)
  129, // P4/nmm (FeSe, LiFeAs)
  191, // P6/mmm (AlB2, CsV3Sb5)
  227, // Fd-3m (diamond, spinel)
  62,  // Pnma (GdFeO3)
  12,  // C2/m (monoclinic)
  123, // P4/mmm
  148, // R-3
  164, // P-3m1
  216, // F-43m (half-Heusler)
  63,  // Cmcm
  14,  // P21/c
  2,   // P-1 (triclinic — maximum exploration)
  1,   // P1 (no symmetry — pure random)
];

/**
 * Generate the Python script that uses PyXtal to create random structures.
 */
function generatePyXtalScript(
  elements: string[],
  counts: Record<string, number>,
  nStructures: number,
  pressureGPa: number,
  baseSeed: number,
  outputDir: string,
  spaceGroups?: number[],
): string {
  const baseComp = elements.map(e => Math.round(counts[e]));
  const sgs = spaceGroups ?? SC_SPACE_GROUPS;

  // Try multiple formula unit multiplicities (Z = 1, 2, 4) to find
  // compositions compatible with more space groups.
  // E.g., Nb3Ge [3,1] can't fit SG 223, but [6,2] (Z=2) uses 2a+6c perfectly.
  // PyXtal will try each Z and use the first that works for the chosen SG.
  const zValues = [1, 2, 4].filter(z => {
    const total = baseComp.reduce((s, n) => s + n * z, 0);
    return total <= 20; // keep cell size manageable for DFT
  });

  // Volume factor: use ambient-like volume for structure generation.
  // Trying to pack atoms into compressed high-P volumes fails because
  // most space groups can't fit the composition. DFT handles compression.
  // Use factor slightly > 1.0 to give PyXtal room for valid packings.
  const volFactor = 1.1;

  return `#!/usr/bin/env python3
"""PyXtal structure generation for ${elements.join("")} at ${pressureGPa} GPa."""
import os, sys, random, warnings
warnings.filterwarnings("ignore")

try:
    from pyxtal import pyxtal
except ImportError:
    print("PYXTAL_NOT_INSTALLED", file=sys.stderr)
    sys.exit(1)

elements = ${JSON.stringify(elements)}
base_composition = ${JSON.stringify(baseComp)}
z_values = ${JSON.stringify(zValues)}
n_structures = ${nStructures}
base_seed = ${baseSeed}
vol_factor = ${volFactor}
output_dir = ${JSON.stringify(outputDir.replace(/\\/g, "/"))}
space_groups = ${JSON.stringify(sgs)}

os.makedirs(output_dir, exist_ok=True)

generated = 0
attempts = 0
max_attempts = n_structures * 50  # Many SGs can't fit complex stoichiometry — need many retries

random.seed(base_seed)

# For complex stoichiometries (like H9Na2Y), low-symmetry SGs succeed more often.
# After 30% of budget, add P1 (no symmetry) to ensure we get SOME structures.
total_atoms = sum(composition)
is_complex = max(composition) >= 6 or total_atoms >= 10

while generated < n_structures and attempts < max_attempts:
    attempts += 1

    # After many failed attempts, fall back to low-symmetry SGs
    if attempts > max_attempts * 0.3 and generated == 0:
        sg = random.choice([1, 2, 4, 14, 62])  # P1, P-1, P21, P21/c, Pnma
    else:
        sg = random.choice(space_groups)
    seed = base_seed + attempts

    # Try multiple volume factors if packing is tight
    vf = vol_factor + random.uniform(-0.2, 0.4)

    try:
        crystal = None
        # Try each Z multiplier until one is compatible with this SG
        for z in z_values:
            comp = [n * z for n in base_composition]
            try:
                crystal = pyxtal()
                crystal.from_random(
                    dim=3,
                    group=sg,
                    species=elements,
                    numIons=comp,
                    factor=max(0.9, vf),
                    seed=seed,
                )
                if crystal.valid:
                    break
            except Exception:
                crystal = None
                continue
        if crystal is None:
            continue

        if crystal.valid:
            # Write POSCAR directly (avoid pymatgen .to() API compatibility issues)
            poscar_path = os.path.join(output_dir, f"POSCAR_{generated:04d}")
            try:
                struct = crystal.to_pymatgen_structure()
                lattice = struct.lattice
                # Build POSCAR manually
                species_order = []
                species_counts = {}
                for site in struct:
                    el = str(site.specie)
                    if el not in species_counts:
                        species_order.append(el)
                        species_counts[el] = 0
                    species_counts[el] += 1
                lines = []
                lines.append(f"PyXtal SG={sg} seed={seed} P={pressureGPa}GPa")
                lines.append("1.0")
                for v in lattice.matrix:
                    lines.append(f"  {v[0]:.10f}  {v[1]:.10f}  {v[2]:.10f}")
                lines.append("  ".join(species_order))
                lines.append("  ".join(str(species_counts[s]) for s in species_order))
                lines.append("Direct")
                for s in species_order:
                    for site in struct:
                        if str(site.specie) == s:
                            fc = site.frac_coords
                            lines.append(f"  {fc[0]:.10f}  {fc[1]:.10f}  {fc[2]:.10f}")
                with open(poscar_path, "w") as f:
                    f.write("\\n".join(lines) + "\\n")
                generated += 1
            except Exception as e2:
                if attempts <= 3:
                    print(f"PYXTAL_DEBUG attempt={attempts} sg={sg} write_error={str(e2)[:60]}", flush=True)
        elif attempts <= 3:
            print(f"PYXTAL_DEBUG attempt={attempts} sg={sg} valid=False factor={vf:.2f}", flush=True)

    except Exception as e:
        if attempts <= 3:
            print(f"PYXTAL_DEBUG attempt={attempts} sg={sg} error={str(e)[:80]}", flush=True)

print(f"PYXTAL_DONE generated={generated} attempts={attempts}")
`;
}

let _pyxtalAvailable: boolean | null = null;

export const pyxtalEngine: CSPEngine = {
  name: "uspex", // Slot into USPEX's position in the pipeline weights

  isAvailable(): boolean {
    if (_pyxtalAvailable !== null) return _pyxtalAvailable;
    try {
      const result = execSync(
        `${PYTHON_BIN} -c "from pyxtal import pyxtal; print('ok')" 2>&1`,
        { timeout: 10000 }
      );
      _pyxtalAvailable = result.toString().trim().includes("ok");
      if (_pyxtalAvailable) {
        console.log(`[PyXtal] Available via ${PYTHON_BIN}`);
      }
    } catch {
      _pyxtalAvailable = false;
    }
    return _pyxtalAvailable;
  },

  async generateStructures(
    elements: string[],
    counts: Record<string, number>,
    config: CSPEngineConfig,
  ): Promise<CSPCandidate[]> {
    const baseSeed = config.baseSeed ?? Math.floor(Math.random() * 1e8);
    const workDir = config.workDir;
    fs.mkdirSync(workDir, { recursive: true });

    const outputDir = path.join(workDir, "structures");
    const scriptPath = path.join(workDir, "generate.py");

    // For cross-seeded runs, bias toward space groups from seed structures
    let targetSGs: number[] | undefined;
    if (config.seedStructures && config.seedStructures.length > 0) {
      const seedSGs = config.seedStructures
        .map(s => s.spaceGroup)
        .filter(Boolean)
        .map(sg => {
          // Try to extract SG number from string like "Fm-3m" or "225"
          const n = parseInt(sg);
          return isNaN(n) ? null : n;
        })
        .filter((n): n is number => n !== null);
      if (seedSGs.length > 0) {
        // Mix seed SGs (60%) with general exploration (40%)
        const generalSGs = SC_SPACE_GROUPS.slice(0, 10);
        targetSGs = [...seedSGs, ...seedSGs, ...seedSGs, ...generalSGs];
      }
    }

    const script = generatePyXtalScript(
      elements, counts, config.maxStructures, config.pressureGPa,
      baseSeed, outputDir, targetSGs
    );
    fs.writeFileSync(scriptPath, script);

    console.log(`[PyXtal] Generating ${config.maxStructures} structures for ${elements.join("")} at ${config.pressureGPa} GPa (seed=${baseSeed})`);

    try {
      const result = execSync(
        `${PYTHON_BIN} ${scriptPath} 2>&1`,
        { cwd: workDir, timeout: config.timeoutMs, maxBuffer: 5 * 1024 * 1024 }
      );
      const output = result.toString();
      // Log debug lines from the Python script
      const debugLines = output.split("\n").filter(l => l.startsWith("PYXTAL_DEBUG"));
      if (debugLines.length > 0) {
        console.log(`[PyXtal] Debug: ${debugLines.slice(0, 5).join(" | ")}`);
      }
      const match = output.match(/PYXTAL_DONE generated=(\d+) attempts=(\d+)/);
      if (match) {
        console.log(`[PyXtal] Done: ${match[1]} structures from ${match[2]} attempts`);
      } else {
        // No DONE marker — script may have crashed
        console.log(`[PyXtal] Script output (first 500 chars): ${output.slice(0, 500)}`);
      }
      if (output.includes("PYXTAL_NOT_INSTALLED")) {
        console.log("[PyXtal] Not installed — skipping");
        _pyxtalAvailable = false;
        return [];
      }
    } catch (err: any) {
      console.log(`[PyXtal] Generation failed: ${err.message?.slice(0, 100)}`);
    }

    // Harvest generated POSCAR files
    const candidates: CSPCandidate[] = [];
    if (fs.existsSync(outputDir)) {
      const files = fs.readdirSync(outputDir)
        .filter(f => f.startsWith("POSCAR"))
        .sort();

      for (const file of files) {
        try {
          const content = fs.readFileSync(path.join(outputDir, file), "utf-8");
          const parsed = parsePOSCAR(content, "uspex", config.pressureGPa, baseSeed, 1);
          if (parsed) {
            parsed.prototype = "PyXtal-random";
            parsed.source = `PyXtal (${content.split("\n")[0].slice(0, 60)})`;
            candidates.push(parsed);
          }
        } catch {}
      }
    }

    console.log(`[PyXtal] Harvested ${candidates.length} valid structures`);
    return candidates;
  },
};
