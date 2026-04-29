/**
 * PyXtal wrapper for crystal structure prediction.
 *
 * Generates symmetry-constrained random structures across all 230 space
 * groups with formula-unit (Z) sweeps. Scales from 250 (preview) to
 * 1000 (deep) structures per composition.
 *
 * Reference: Fredericks et al., Comput. Phys. Commun. 261 (2021) 107810
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import type { CSPCandidate, CSPEngine, CSPEngineConfig, ScreeningTierConfig } from "./csp-types";
import { SCREENING_TIERS } from "./csp-types";
import { parsePOSCAR } from "./poscar-io";

const PYTHON_BIN = process.env.PYTHON_BIN ?? "python3";

// Tiered space group sampling for PyXtal structure generation.
// Distribution: 40% high-symmetry hydride SGs, 30% medium, 20% low, 10% P1/P-1
const SG_HIGH_SYMMETRY = [
  225, // Fm-3m (LaH10, NaCl)
  229, // Im-3m (CaH6, H3S, BCC)
  194, // P63/mmc (YH9, MgB2)
  139, // I4/mmm (BaFe2As2)
  221, // Pm-3m (perovskite)
  223, // Pm-3n (A15, Nb3Sn)
  166, // R-3m (Bi2Te3)
  191, // P6/mmm (AlB2)
  227, // Fd-3m (spinel)
  216, // F-43m (half-Heusler)
];
const SG_MEDIUM_SYMMETRY = [
  12,  // C2/m (monoclinic layered)
  62,  // Pnma (GdFeO3, orthorhombic)
  129, // P4/nmm (FeSe, LiFeAs)
  123, // P4/mmm
  63,  // Cmcm
  136, // P42/mnm (rutile)
  148, // R-3
  164, // P-3m1
  140, // I4/mcm
  127, // P4/mbm
  15,  // C2/c
  71,  // Immm
  65,  // Cmmm
  167, // R-3c
  87,  // I4/m
];
const SG_LOW_SYMMETRY = [
  14,  // P21/c (most common monoclinic)
  19,  // P212121 (orthorhombic chiral)
  33,  // Pna21
  29,  // Pca21
  4,   // P21
  9,   // Cc
  11,  // P21/m
  61,  // Pbca
  56,  // Pccn
  60,  // Pbcn
];
const SG_FALLBACK = [1, 2]; // P1, P-1

/**
 * Sample a space group from the tiered distribution.
 * 40% high-symmetry, 30% medium, 20% low, 10% P1/P-1
 */
function sampleSpaceGroup(rng: () => number): number {
  const r = rng();
  if (r < 0.40) return SG_HIGH_SYMMETRY[Math.floor(rng() * SG_HIGH_SYMMETRY.length)];
  if (r < 0.70) return SG_MEDIUM_SYMMETRY[Math.floor(rng() * SG_MEDIUM_SYMMETRY.length)];
  if (r < 0.90) return SG_LOW_SYMMETRY[Math.floor(rng() * SG_LOW_SYMMETRY.length)];
  return SG_FALLBACK[Math.floor(rng() * SG_FALLBACK.length)];
}

// All SGs combined for the Python script
const ALL_SPACE_GROUPS = [...SG_HIGH_SYMMETRY, ...SG_MEDIUM_SYMMETRY, ...SG_LOW_SYMMETRY, ...SG_FALLBACK];

/**
 * Generate the Python script for PyXtal structure generation.
 */
function generatePyXtalScript(
  elements: string[],
  baseComp: number[],
  zValues: number[],
  nStructures: number,
  pressureGPa: number,
  baseSeed: number,
  outputDir: string,
  maxAtoms: number,
  spaceGroups?: number[],
): string {
  const sgs = spaceGroups ?? ALL_SPACE_GROUPS;

  return `#!/usr/bin/env python3
import os, sys, random, traceback, warnings
warnings.filterwarnings("ignore")

try:
    from pyxtal import pyxtal
except ImportError:
    print("PYXTAL_NOT_INSTALLED")
    sys.exit(1)

try:
    import pymatgen
except ImportError:
    print("PYXTAL_ERROR pymatgen not installed")
    sys.exit(1)

try:
    elements = ${JSON.stringify(elements)}
    base_composition = ${JSON.stringify(baseComp)}
    z_values = ${JSON.stringify(zValues)}
    n_structures = ${nStructures}
    base_seed = ${baseSeed}
    pressure_gpa = ${pressureGPa}
    max_atoms = ${maxAtoms}
    output_dir = ${JSON.stringify(outputDir.replace(/\\/g, "/"))}
    space_groups = ${JSON.stringify(sgs)}

    os.makedirs(output_dir, exist_ok=True)

    generated = 0
    attempts = 0
    max_attempts = n_structures * 50
    random.seed(base_seed)

    while generated < n_structures and attempts < max_attempts:
        attempts += 1

        # Tiered SG sampling: 40% high, 30% medium, 20% low, 10% P1/P-1
        # After many failed attempts, shift more budget to low-symmetry
        high_sgs = ${JSON.stringify(SG_HIGH_SYMMETRY)}
        med_sgs = ${JSON.stringify(SG_MEDIUM_SYMMETRY)}
        low_sgs = ${JSON.stringify(SG_LOW_SYMMETRY)}

        if attempts > max_attempts * 0.5 and generated == 0:
            sg = random.choice([1, 2, 4, 14, 62])  # desperation fallback
        else:
            r = random.random()
            if r < 0.40:
                sg = random.choice(high_sgs)
            elif r < 0.70:
                sg = random.choice(med_sgs)
            elif r < 0.90:
                sg = random.choice(low_sgs)
            else:
                sg = random.choice([1, 2])
        seed = base_seed + attempts
        # Pressure-aware volume jitter
        if pressure_gpa < 20:
            vf = 1.1 + random.uniform(-0.25, 0.40)   # 0.85-1.50
        elif pressure_gpa < 100:
            vf = 1.0 + random.uniform(-0.25, 0.25)    # 0.75-1.25
        elif pressure_gpa < 200:
            vf = 0.85 + random.uniform(-0.20, 0.25)   # 0.65-1.10
        else:
            vf = 0.75 + random.uniform(-0.20, 0.25)   # 0.55-1.00

        try:
            crystal = None
            for z in z_values:
                comp = [n * z for n in base_composition]
                total = sum(comp)
                if total > max_atoms:
                    continue
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
                # Reject H₂-molecular structures: check if any H-H pair < 0.90 Å
                # This prevents wasting DFT on structures that are just molecular hydrogen
                struct_check = crystal.to_pymatgen()
                has_h2 = False
                h_sites = [s for s in struct_check if str(s.specie) == "H"]
                if len(h_sites) >= 2:
                    for i_h in range(len(h_sites)):
                        for j_h in range(i_h + 1, len(h_sites)):
                            d_hh = struct_check.lattice.get_all_distances(
                                [h_sites[i_h].frac_coords], [h_sites[j_h].frac_coords]
                            )[0][0]
                            if d_hh < 0.90:
                                has_h2 = True
                                break
                        if has_h2:
                            break
                if has_h2:
                    continue  # skip this H₂-like structure

                poscar_path = os.path.join(output_dir, f"POSCAR_{generated:04d}")
                try:
                    struct = struct_check
                    lattice = struct.lattice
                    species_order = []
                    species_counts = {}
                    for site in struct:
                        el = str(site.specie)
                        if el not in species_counts:
                            species_order.append(el)
                            species_counts[el] = 0
                        species_counts[el] += 1
                    lines = []
                    lines.append(f"PyXtal SG={sg} seed={seed} P={pressure_gpa}GPa Z={z}")
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
except Exception as fatal:
    print(f"PYXTAL_FATAL {traceback.format_exc()}")
    sys.exit(1)
`;
}

let _pyxtalAvailable: boolean | null = null;

export const pyxtalEngine: CSPEngine = {
  name: "uspex", // Slots into USPEX's pipeline weight position

  isAvailable(): boolean {
    if (_pyxtalAvailable !== null) return _pyxtalAvailable;
    try {
      const result = execSync(
        `${PYTHON_BIN} -c "from pyxtal import pyxtal; print('ok')" 2>&1`,
        { timeout: 10000 }
      );
      _pyxtalAvailable = result.toString().trim().includes("ok");
      if (_pyxtalAvailable) console.log(`[PyXtal] Available via ${PYTHON_BIN}`);
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

    // Determine tier from budget
    let tierConfig: ScreeningTierConfig;
    if (config.maxStructures >= 1000) tierConfig = SCREENING_TIERS.deep;
    else if (config.maxStructures >= 500) tierConfig = SCREENING_TIERS.standard;
    else if (config.maxStructures >= 250) tierConfig = SCREENING_TIERS.preview;
    else {
      // Small budget (inline qe-worker call) — minimal config
      tierConfig = {
        tier: "preview",
        maxAtoms: 20,
        zValues: [1, 2, 4],
        airssBudgetPerCombo: 0,
        airssTotalCap: 0,
        pyxtalBudgetPerZ: Math.ceil(config.maxStructures / 3),
        pyxtalTotalCap: config.maxStructures,
        volumeEnsemble: [1.0],
        dft0Floor: 3, dft0Cap: 8, dft0ClusterFraction: 0.03,
      };
    }

    const formulaAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
    const baseComp = elements.map(e => Math.round(counts[e]));
    const validZ = tierConfig.zValues.filter(z => formulaAtoms * z <= tierConfig.maxAtoms);
    if (validZ.length === 0) validZ.push(1);

    const outputDir = path.join(workDir, "structures");
    const scriptPath = path.join(workDir, "generate.py");

    // For cross-seeded runs, bias SGs from seed structures
    let targetSGs: number[] | undefined;
    if (config.seedStructures && config.seedStructures.length > 0) {
      const seedSGs = config.seedStructures
        .map(s => parseInt(s.spaceGroup))
        .filter(n => !isNaN(n));
      if (seedSGs.length > 0) {
        targetSGs = [...seedSGs, ...seedSGs, ...seedSGs, ...SG_HIGH_SYMMETRY];
      }
    }

    const script = generatePyXtalScript(
      elements, baseComp, validZ, config.maxStructures,
      config.pressureGPa, baseSeed, outputDir, tierConfig.maxAtoms, targetSGs
    );
    fs.writeFileSync(scriptPath, script);

    console.log(`[PyXtal] Generating ${config.maxStructures} structures for ${elements.join("")} at ${config.pressureGPa} GPa (Z=[${validZ.join(",")}], maxAtoms=${tierConfig.maxAtoms}, ${tierConfig.tier} tier)`);

    try {
      const result = execSync(
        `${PYTHON_BIN} ${scriptPath} 2>&1`,
        { cwd: workDir, timeout: config.timeoutMs, maxBuffer: 5 * 1024 * 1024 }
      );
      const output = result.toString();
      const debugLines = output.split("\n").filter(l => l.startsWith("PYXTAL_DEBUG"));
      if (debugLines.length > 0) {
        console.log(`[PyXtal] Debug: ${debugLines.slice(0, 5).join(" | ")}`);
      }
      const match = output.match(/PYXTAL_DONE generated=(\d+) attempts=(\d+)/);
      if (match) {
        console.log(`[PyXtal] Done: ${match[1]} structures from ${match[2]} attempts`);
      } else {
        console.log(`[PyXtal] Script output (first 500 chars): ${output.slice(0, 500)}`);
      }
      if (output.includes("PYXTAL_NOT_INSTALLED")) {
        _pyxtalAvailable = false;
        return [];
      }
    } catch (err: any) {
      console.log(`[PyXtal] Generation failed: ${err.message?.slice(0, 100)}`);
    }

    // Harvest POSCAR files
    const candidates: CSPCandidate[] = [];
    if (fs.existsSync(outputDir)) {
      const files = fs.readdirSync(outputDir).filter(f => f.startsWith("POSCAR")).sort();
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
