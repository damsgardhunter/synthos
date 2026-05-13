/**
 * DMFT-Ready Bundle Exporter
 *
 * Packages everything solid_dmft needs from a completed DFT run into a
 * standardized HDF5 bundle:
 *
 *   - H(k) on a uniform k-mesh (from Wannier90 DMFT-projector _hr.dat)
 *   - Correlated subspace definition (which orbitals, which atoms)
 *   - Hubbard U and J values (from Hubbard workflow + ACBN0)
 *   - Magnetic ground state
 *   - Structure, pressure, electronic info
 *
 * The bundle is self-contained: the DMFT service can run solid_dmft from
 * this file alone, with no access to the original DFT working directory.
 *
 * Also useful as an export for external collaborators using eDMFT or DCore.
 *
 * Bundle format: HDF5 (written via a Python helper since Node.js HDF5
 * libraries are unreliable). Falls back to JSON if Python/h5py unavailable.
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import type { HubbardWorkflowResult, HubbardSiteConfig } from "./hubbard-workflow";
import type { MagneticGroundStateResult } from "./magnetic-ground-state";
import type { ACBN0Result } from "./acbn0-pipeline";
import { countDMFTOrbitals, DMFT_PROJECTIONS } from "./epw-pipeline";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DMFTBundleInput {
  formula: string;
  elements: string[];
  counts: Record<string, number>;
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  latticeVectors: number[][];     // 3×3 Angstrom
  pressureGpa: number;
  fermiEnergy: number;            // eV
  qualityTier: string;

  // From Hubbard workflow
  hubbardWorkflow: HubbardWorkflowResult;

  // From ACBN0 (optional — provides first-principles U)
  acbn0?: ACBN0Result;

  // From magnetic ground-state search
  magneticGroundState?: MagneticGroundStateResult;

  // Wannier90 DMFT-projector output directory
  // Contains: {prefix}_hr.dat, {prefix}.win, {prefix}_centres.xyz
  wannier90Dir: string;
  wannier90Prefix: string;
}

export interface DMFTBundleResult {
  /** Path to the generated bundle file (HDF5 or JSON fallback) */
  bundlePath: string;
  /** Format: "hdf5" or "json" */
  format: "hdf5" | "json";
  /** Number of correlated shells */
  nCorrelatedShells: number;
  /** Total correlated orbitals */
  nCorrelatedOrbitals: number;
  /** Whether H(k) was successfully parsed from _hr.dat */
  hamiltonianParsed: boolean;
  /** Warnings */
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Wannier90 _hr.dat parser
// ---------------------------------------------------------------------------

interface HrData {
  /** Number of Wannier functions */
  numWann: number;
  /** Number of R-vectors */
  nRpts: number;
  /** R-vector degeneracies */
  degeneracies: number[];
  /** H(R) matrix elements: Map<"rx_ry_rz", [i, j, re, im][]> */
  hR: Map<string, Array<{ i: number; j: number; re: number; im: number }>>;
}

function parseWannier90Hr(hrPath: string): HrData | null {
  if (!fs.existsSync(hrPath)) return null;

  const lines = fs.readFileSync(hrPath, "utf-8").split("\n");
  if (lines.length < 3) return null;

  // Line 0: comment
  // Line 1: num_wann
  // Line 2: nrpts
  const numWann = parseInt(lines[1].trim(), 10);
  const nRpts = parseInt(lines[2].trim(), 10);
  if (isNaN(numWann) || isNaN(nRpts)) return null;

  // Next ceil(nRpts/15) lines: degeneracies
  const degLines = Math.ceil(nRpts / 15);
  const degeneracies: number[] = [];
  for (let i = 0; i < degLines; i++) {
    const vals = lines[3 + i].trim().split(/\s+/).map(Number);
    degeneracies.push(...vals);
  }

  // Remaining lines: R_x R_y R_z i j Re(H) Im(H)
  const dataStart = 3 + degLines;
  const hR = new Map<string, Array<{ i: number; j: number; re: number; im: number }>>();

  for (let ln = dataStart; ln < lines.length; ln++) {
    const parts = lines[ln].trim().split(/\s+/);
    if (parts.length < 7) continue;
    const [rx, ry, rz, i, j, re, im] = parts.map(Number);
    const key = `${rx}_${ry}_${rz}`;
    if (!hR.has(key)) hR.set(key, []);
    hR.get(key)!.push({ i, j, re, im });
  }

  return { numWann, nRpts, degeneracies, hR };
}

// ---------------------------------------------------------------------------
// Correlated subspace builder
// ---------------------------------------------------------------------------

interface CorrelatedShell {
  /** Atom index (0-based) in the unit cell */
  atom: number;
  /** Angular momentum quantum number: 2=d, 3=f */
  l: number;
  /** Dimension of the correlated subspace (2l+1) */
  dim: number;
  /** Sort index (groups equivalent atoms) */
  sort: number;
  /** Element symbol */
  element: string;
  /** U value (eV) */
  U: number;
  /** J value (eV) */
  J: number;
}

function buildCorrelatedSubspace(
  input: DMFTBundleInput,
): CorrelatedShell[] {
  const shells: CorrelatedShell[] = [];
  const { elements, counts, positions, hubbardWorkflow, acbn0 } = input;

  // Map element to Hubbard site config
  const siteMap = new Map<string, HubbardSiteConfig>();
  for (const site of hubbardWorkflow.sites) {
    if (site.needsU) siteMap.set(site.element, site);
  }

  // Build shells from atom positions
  let sortIndex = 0;
  const sortMap = new Map<string, number>();

  for (let atomIdx = 0; atomIdx < positions.length; atomIdx++) {
    const el = positions[atomIdx].element;
    const site = siteMap.get(el);
    if (!site) continue;

    // Determine l from manifold
    let l: number;
    switch (site.orbitalManifold) {
      case "3d": case "4d": case "5d": l = 2; break;
      case "4f": case "5f": l = 3; break;
      default: continue;
    }

    // Sort index: same element type → same sort
    if (!sortMap.has(el)) {
      sortMap.set(el, sortIndex++);
    }

    // U/J: prefer ACBN0 first-principles values, fall back to workflow
    let U = site.uEffective;
    let J = site.hunds;
    if (acbn0?.converged && acbn0.hubbardU[el] != null) {
      U = acbn0.hubbardU[el];
    }

    shells.push({
      atom: atomIdx,
      l,
      dim: 2 * l + 1,
      sort: sortMap.get(el)!,
      element: el,
      U,
      J,
    });
  }

  return shells;
}

// ---------------------------------------------------------------------------
// Bundle JSON structure (intermediate, before HDF5 conversion)
// ---------------------------------------------------------------------------

interface DMFTBundleJSON {
  hamiltonian: {
    /** H(R) from Wannier90 _hr.dat: flat array of [rx,ry,rz,i,j,re,im] */
    hr_data: number[][];
    num_wann: number;
    n_rpts: number;
    degeneracies: number[];
    kmesh: [number, number, number];
  } | null;
  correlated_subspace: {
    corr_shells: Array<{ atom: number; l: number; dim: number; sort: number; element: string }>;
    corr_to_inequiv: number[];
    n_corr_shells: number;
    n_inequiv_shells: number;
  };
  interaction: {
    U_values: number[];
    J_values: number[];
    hubbard_kind: number;
  };
  structure: {
    formula: string;
    elements: string[];
    lattice_vectors: number[][];
    positions: Array<{ element: string; x: number; y: number; z: number }>;
    pressure_gpa: number;
  };
  electronic: {
    fermi_energy: number;
    n_electrons: number;
    magnetic_ordering: string;
    correlation_regime: string;
  };
  metadata: {
    source: string;
    created_at: string;
    qe_quality_tier: string;
    wannier_mode: string;
  };
}

// ---------------------------------------------------------------------------
// Main export function
// ---------------------------------------------------------------------------

export async function exportDMFTBundle(
  input: DMFTBundleInput,
): Promise<DMFTBundleResult> {
  const warnings: string[] = [];

  // 1. Build correlated subspace
  const shells = buildCorrelatedSubspace(input);
  if (shells.length === 0) {
    warnings.push("No correlated shells found — this material may not need DMFT");
  }

  const dmftOrbitals = countDMFTOrbitals(input.elements, input.counts);

  // 2. Parse Wannier90 H(R) from _hr.dat
  const hrPath = path.join(input.wannier90Dir, `${input.wannier90Prefix}_hr.dat`);
  const hrData = parseWannier90Hr(hrPath);
  const hamiltonianParsed = hrData !== null;
  if (!hamiltonianParsed) {
    warnings.push(`Wannier90 _hr.dat not found at ${hrPath} — bundle will lack H(k)`);
  }

  // 3. Build corr_to_inequiv mapping (group by sort index)
  const corrToInequiv = shells.map(s => s.sort);
  const nInequiv = new Set(corrToInequiv).size;

  // 4. Determine magnetic ordering
  let magneticOrdering = "NM";
  if (input.magneticGroundState?.searchPerformed) {
    magneticOrdering = input.magneticGroundState.groundState;
  }

  // 5. Estimate n_electrons in correlated subspace
  // Rough estimate: for d-shells, count valence electrons from the periodic table
  const VALENCE_D_ELECTRONS: Record<string, number> = {
    Sc: 1, Ti: 2, V: 3, Cr: 5, Mn: 5, Fe: 6, Co: 7, Ni: 8, Cu: 9, Zn: 10,
    Y: 1, Zr: 2, Nb: 4, Mo: 5, Ru: 7, Rh: 8, Pd: 10,
    La: 0, Hf: 2, Ta: 3, W: 4, Re: 5, Os: 6, Ir: 7, Pt: 9,
    Ce: 1, Pr: 2, Nd: 3, Sm: 5, Eu: 6, Gd: 7, Tb: 8, Dy: 9,
    Th: 0, U: 2,
  };
  let nElectrons = 0;
  for (const shell of shells) {
    nElectrons += VALENCE_D_ELECTRONS[shell.element] ?? 0;
  }

  // 6. Determine k-mesh from Wannier90 .win file (parse mp_grid line)
  let kmesh: [number, number, number] = [8, 8, 8]; // default
  const winPath = path.join(input.wannier90Dir, `${input.wannier90Prefix}.win`);
  if (fs.existsSync(winPath)) {
    const winContent = fs.readFileSync(winPath, "utf-8");
    const mpMatch = winContent.match(/mp_grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)/);
    if (mpMatch) {
      kmesh = [parseInt(mpMatch[1]), parseInt(mpMatch[2]), parseInt(mpMatch[3])];
    }
  }

  // 7. Assemble JSON bundle
  const bundle: DMFTBundleJSON = {
    hamiltonian: hrData ? {
      hr_data: flattenHrData(hrData),
      num_wann: hrData.numWann,
      n_rpts: hrData.nRpts,
      degeneracies: hrData.degeneracies,
      kmesh,
    } : null,
    correlated_subspace: {
      corr_shells: shells.map(s => ({
        atom: s.atom, l: s.l, dim: s.dim, sort: s.sort, element: s.element,
      })),
      corr_to_inequiv: corrToInequiv,
      n_corr_shells: shells.length,
      n_inequiv_shells: nInequiv,
    },
    interaction: {
      U_values: shells.map(s => s.U),
      J_values: shells.map(s => s.J),
      hubbard_kind: input.hubbardWorkflow.hubbardKind,
    },
    structure: {
      formula: input.formula,
      elements: input.elements,
      lattice_vectors: input.latticeVectors,
      positions: input.positions,
      pressure_gpa: input.pressureGpa,
    },
    electronic: {
      fermi_energy: input.fermiEnergy,
      n_electrons: nElectrons,
      magnetic_ordering: magneticOrdering,
      correlation_regime: input.hubbardWorkflow.correlationRegime,
    },
    metadata: {
      source: "qae-dmft-bundle-v1",
      created_at: new Date().toISOString(),
      qe_quality_tier: input.qualityTier,
      wannier_mode: "dmft_projector",
    },
  };

  // 8. Write bundle — try HDF5 via Python helper, fall back to JSON
  const bundleDir = path.dirname(input.wannier90Dir);
  const bundleBase = `${input.formula}_${Math.round(input.pressureGpa)}GPa_dmft_bundle`;
  let bundlePath: string;
  let format: "hdf5" | "json";

  // Write JSON first (always, as intermediate)
  const jsonPath = path.join(bundleDir, `${bundleBase}.json`);
  fs.writeFileSync(jsonPath, JSON.stringify(bundle, null, 2));

  // Try HDF5 conversion
  const h5Path = path.join(bundleDir, `${bundleBase}.h5`);
  const converted = tryConvertToHDF5(jsonPath, h5Path);
  if (converted) {
    bundlePath = h5Path;
    format = "hdf5";
    console.log(`[DMFT-Bundle] Wrote HDF5 bundle: ${h5Path}`);
  } else {
    bundlePath = jsonPath;
    format = "json";
    warnings.push("HDF5 conversion unavailable — bundle written as JSON (convert with dmft/convert-bundle.py)");
    console.log(`[DMFT-Bundle] Wrote JSON bundle: ${jsonPath}`);
  }

  return {
    bundlePath,
    format,
    nCorrelatedShells: shells.length,
    nCorrelatedOrbitals: dmftOrbitals.total,
    hamiltonianParsed,
    warnings,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function flattenHrData(hr: HrData): number[][] {
  const rows: number[][] = [];
  hr.hR.forEach((entries, key) => {
    const [rx, ry, rz] = key.split("_").map(Number);
    for (const e of entries) {
      rows.push([rx, ry, rz, e.i, e.j, e.re, e.im]);
    }
  });
  return rows;
}

/**
 * Try to convert JSON bundle to HDF5 using a Python one-liner.
 * Returns true on success. Silently fails if Python/h5py unavailable.
 */
function tryConvertToHDF5(jsonPath: string, h5Path: string): boolean {
  const pythonScript = `
import json, sys, os
try:
    import h5py, numpy as np
except ImportError:
    sys.exit(1)

with open(sys.argv[1]) as f:
    d = json.load(f)

with h5py.File(sys.argv[2], 'w') as hf:
    # Hamiltonian
    if d.get('hamiltonian'):
        h = d['hamiltonian']
        hg = hf.create_group('hamiltonian')
        hr = np.array(h['hr_data'])
        hg.create_dataset('hr_data', data=hr)
        hg.create_dataset('num_wann', data=h['num_wann'])
        hg.create_dataset('n_rpts', data=h['n_rpts'])
        hg.create_dataset('degeneracies', data=np.array(h['degeneracies']))
        hg.create_dataset('kmesh', data=np.array(h['kmesh']))

        # Fourier transform H(R) -> H(k) on the k-mesh for direct TRIQS use
        nw = h['num_wann']
        km = h['kmesh']
        nk = km[0] * km[1] * km[2]
        hk = np.zeros((nk, nw, nw), dtype=complex)
        # Build R-vectors and H(R) matrices, preserving the INSERTION ORDER
        # from _hr.dat. The degeneracy array is in file order (Wannier90
        # writes deg[0] for the first R encountered, deg[1] for the second,
        # etc.). If we sorted r_vecs.keys() before iterating, the degeneracy
        # index would no longer align with the R-vector — silently corrupting
        # H(k) normalization.
        r_vecs = {}
        r_insertion_order = {}  # R -> file index (matches degeneracy array)
        for row in h['hr_data']:
            rx, ry, rz, i, j, re, im = row[0], row[1], row[2], int(row[3])-1, int(row[4])-1, row[5], row[6]
            key = (int(rx), int(ry), int(rz))
            if key not in r_vecs:
                r_vecs[key] = np.zeros((nw, nw), dtype=complex)
                r_insertion_order[key] = len(r_insertion_order)
            r_vecs[key][i, j] = complex(re, im)

        # FT: H(k) = sum_R exp(ik.R) H(R) / deg(R)
        degs = h['degeneracies']
        if len(r_vecs) != len(degs):
            print(f"  WARNING: {len(r_vecs)} unique R-vectors vs {len(degs)} degeneracies — _hr.dat ordering mismatch")
        kidx = 0
        for ik1 in range(km[0]):
            for ik2 in range(km[1]):
                for ik3 in range(km[2]):
                    kfrac = np.array([ik1/km[0], ik2/km[1], ik3/km[2]])
                    # Iterate in INSERTION ORDER so deg-index matches R
                    for R in r_vecs.keys():
                        ri = r_insertion_order[R]
                        phase = np.exp(2j * np.pi * np.dot(kfrac, R))
                        deg = degs[ri] if ri < len(degs) else 1
                        hk[kidx] += phase * r_vecs[R] / deg
                    kidx += 1

        # Sanity check: H(k) must be Hermitian for a physical Hamiltonian.
        herm_residual = float(np.max(np.abs(hk - hk.conj().transpose(0, 2, 1))))
        herm_rel = herm_residual / (float(np.max(np.abs(hk))) + 1e-30)
        if herm_rel > 1e-6:
            print(f"  WARNING: H(k) Hermiticity residual {herm_residual:.2e} "
                  f"(relative {herm_rel:.2e}) — _hr.dat may be incomplete or "
                  f"degeneracy alignment is wrong")

        hg.create_dataset('hk', data=hk)
        kpts = []
        for ik1 in range(km[0]):
            for ik2 in range(km[1]):
                for ik3 in range(km[2]):
                    kpts.append([ik1/km[0], ik2/km[1], ik3/km[2]])
        hg.create_dataset('kpoints', data=np.array(kpts))

    # Correlated subspace
    cs = d['correlated_subspace']
    cg = hf.create_group('correlated_subspace')
    cg.create_dataset('corr_shells', data=json.dumps(cs['corr_shells']))
    if cs['corr_shells']:
        max_dim = max(s['dim'] for s in cs['corr_shells'])
        nk_total = int(np.prod(d['hamiltonian']['kmesh'])) if d.get('hamiltonian') else 1
        nw_total = d['hamiltonian']['num_wann'] if d.get('hamiltonian') else max_dim
        n_sh = len(cs['corr_shells'])
        # Identity projectors (DMFT projector mode = no rotation needed)
        proj = np.zeros((nk_total, n_sh, max_dim, nw_total), dtype=complex)
        orb_offset = 0
        for si, sh in enumerate(cs['corr_shells']):
            for m in range(sh['dim']):
                if orb_offset + m < nw_total:
                    proj[:, si, m, orb_offset + m] = 1.0
            orb_offset += sh['dim']
        cg.create_dataset('proj_mat', data=proj)
    cg.create_dataset('corr_to_inequiv', data=np.array(cs['corr_to_inequiv']))

    # Interaction
    ig = hf.create_group('interaction')
    ig.create_dataset('U_values', data=np.array(d['interaction']['U_values']))
    ig.create_dataset('J_values', data=np.array(d['interaction']['J_values']))
    ig.create_dataset('hubbard_kind', data=d['interaction']['hubbard_kind'])

    # Structure
    sg = hf.create_group('structure')
    sg.create_dataset('formula', data=d['structure']['formula'])
    sg.create_dataset('elements', data=json.dumps(d['structure']['elements']))
    sg.create_dataset('lattice_vectors', data=np.array(d['structure']['lattice_vectors']))
    pos = d['structure']['positions']
    sg.create_dataset('positions', data=np.array([[p['x'], p['y'], p['z']] for p in pos]))
    sg.create_dataset('pressure_gpa', data=d['structure']['pressure_gpa'])

    # Electronic
    eg = hf.create_group('electronic')
    eg.create_dataset('fermi_energy', data=d['electronic']['fermi_energy'])
    eg.create_dataset('n_electrons', data=d['electronic']['n_electrons'])
    eg.create_dataset('magnetic_ordering', data=d['electronic']['magnetic_ordering'])
    eg.create_dataset('correlation_regime', data=d['electronic']['correlation_regime'])

    # Metadata
    mg = hf.create_group('metadata')
    for k, v in d['metadata'].items():
        mg.create_dataset(k, data=str(v))

print('OK')
`;

  try {
    const pythonBin = process.env.PYTHON_BIN || "python3";
    const tmpScript = path.join(path.dirname(jsonPath), "_convert_bundle.py");
    fs.writeFileSync(tmpScript, pythonScript);
    const result = execSync(
      `${pythonBin} "${tmpScript}" "${jsonPath}" "${h5Path}"`,
      { timeout: 60_000, stdio: ["pipe", "pipe", "pipe"] },
    );
    fs.unlinkSync(tmpScript);
    return result.toString().trim() === "OK";
  } catch {
    return false;
  }
}

/**
 * Check if a material is DMFT-eligible based on its DFT results.
 *
 * DMFT is computationally expensive (hours per material). Only run it for:
 *   1. Materials with correlated d/f electrons (Hubbard workflow says so)
 *   2. Publication-quality structures (force < 0.001)
 *   3. Converged electronic structure (SCF + metallic or near-insulator)
 */
export function isDMFTEligible(
  hubbardWorkflow: HubbardWorkflowResult | undefined,
  qualityTier: string | undefined,
): { eligible: boolean; reason: string } {
  if (!hubbardWorkflow) {
    return { eligible: false, reason: "No Hubbard workflow result" };
  }
  if (!hubbardWorkflow.applyDFTplusU) {
    return { eligible: false, reason: "No correlated orbitals requiring DFT+U" };
  }
  if (hubbardWorkflow.correlatedSiteCount === 0) {
    return { eligible: false, reason: "Zero correlated sites" };
  }

  // Only DMFT for well-converged structures
  const goodTiers = new Set(["publication_ready", "final_converged"]);
  if (!qualityTier || !goodTiers.has(qualityTier)) {
    return { eligible: false, reason: `Quality tier ${qualityTier} insufficient — need final_converged or publication_ready` };
  }

  // Check correlation regime — DMFT is most valuable for strongly correlated
  const strongRegimes = new Set(["Mott-proximate", "strongly-correlated", "moderately-correlated"]);
  if (!strongRegimes.has(hubbardWorkflow.correlationRegime)) {
    return { eligible: false, reason: `Correlation regime '${hubbardWorkflow.correlationRegime}' too weak for DMFT` };
  }

  return {
    eligible: true,
    reason: `${hubbardWorkflow.correlatedSiteCount} correlated sites, regime=${hubbardWorkflow.correlationRegime}`,
  };
}
