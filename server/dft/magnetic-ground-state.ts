/**
 * Magnetic ground-state search module.
 *
 * Before phonon calculations, the pipeline must determine the correct magnetic
 * ordering. Running phonons on the wrong magnetic state produces phonon
 * frequencies that look fine but correspond to a metastable state — the true
 * ground state may have different symmetry, different phonon spectra, and
 * different electron-phonon coupling.
 *
 * The problem is worst for:
 *   - Iron-based superconductors: FM, AFM-stripe, AFM-checkerboard, AFM-bicollinear
 *     are all close in energy. BaFe2As2 is AFM-stripe, but FM SCF converges fine.
 *   - Nickel-based: NdNiO2 is paramagnetic but AFM fluctuations matter.
 *   - Mn compounds: complex non-collinear orderings common.
 *   - Cr compounds: spin-density-wave ground states.
 *   - Hydrides with TM: e.g., FeH, NiH — magnetic state affects hydrogen
 *     phonon frequencies by 10-30%.
 *
 * Strategy:
 *   For each magnetic candidate, run short SCF calculations with different
 *   spin orderings (NM, FM, AFM variants) and pick the lowest-energy state.
 *   This adds 2-4 SCF calculations but prevents silent convergence to a
 *   metastable magnetic state.
 *
 * @see I. I. Mazin et al., Phys. Rev. Lett. 101, 057003 (2008) — Fe pnictide magnetism
 * @see M. D. Johannes & I. I. Mazin, Phys. Rev. B 79, 220510 (2009) — stripe AFM
 * @see P. J. Hirschfeld et al., Rep. Prog. Phys. 74, 124508 (2011) — Fe-SC magnetism
 */

// ─── Magnetic ordering types ─────────────────────────────────────────

export type MagneticOrdering = "NM" | "FM" | "AFM-checkerboard" | "AFM-stripe" | "AFM-layered" | "AFM-alternating" | "AFM-double-stripe" | "ferrimagnetic" | "noncollinear-canted" | "noncollinear-spiral";

export interface MagneticTrialConfig {
  /** Label for this trial */
  ordering: MagneticOrdering;
  /** nspin value for QE (1 for NM, 2 for collinear magnetic; ignored when noncolin=true) */
  nspin: 1 | 2;
  /** Whether this trial requires noncolin=.true. in QE */
  noncolin?: boolean;
  /** starting_magnetization lines for QE input */
  magnetizationBlock: string;
  /** Expected computational cost relative to NM */
  costFactor: number;
  /** Physical motivation for trying this ordering */
  rationale: string;
  /** conv_thr override for this trial (default uses the triage value) */
  convThr?: number;
}

export interface MagneticTrialResult {
  /** Which ordering was tried */
  ordering: MagneticOrdering;
  /** Total energy (eV) — the comparison metric. The docstring previously
   *  said Ry, but parseSCFOutput converts Ry→eV before returning, and
   *  qe-worker.ts:6386 passes that value here unchanged. See line 636
   *  below for the unit-conversion that was using RY_TO_MEV instead of
   *  EV_TO_MEV — wrong by 13605× until fixed. */
  totalEnergy: number | null;
  /** Total magnetization from QE output (Bohr magneton) */
  totalMagnetization: number | null;
  /** Absolute magnetization from QE output */
  absoluteMagnetization: number | null;
  /** Whether SCF strictly converged */
  converged: boolean;
  /** Last scf accuracy (Ry) — for accepting near-converged trials */
  lastScfAccuracyRy: number | null;
  /** Wall time for this trial (ms) */
  wallTimeMs: number;
}

export interface MagneticGroundStateResult {
  /** Whether a magnetic search was performed */
  searchPerformed: boolean;
  /** The winning magnetic ordering */
  groundState: MagneticOrdering;
  /** All trial results, sorted by energy */
  trials: MagneticTrialResult[];
  /** Energy difference between ground state and next-best (eV/atom).
   *  Docstring previously claimed Ry/atom, but the value is actually in
   *  eV/atom because MagneticTrialResult.totalEnergy is in eV. Consumers
   *  at qe-worker.ts:6422 that multiplied by 13605.7 (Ry→meV) were also
   *  buggy — see that fix below. */
  energyGapPerAtom: number;
  /** Whether the ground state is clearly separated (gap > 5 meV/atom).
   *  Threshold defined by NEAR_DEGENERATE_THRESHOLD_MEV. */
  wellSeparated: boolean;
  /** QE magnetization block for the winning state */
  winningMagBlock: string;
  /** nspin for the winning state */
  winningNspin: 1 | 2;
  /** Whether the winning state requires non-collinear mode */
  winningNoncolin: boolean;
  /** Whether a tighter-convergence rerun was performed for near-degenerate states */
  tightConvergenceRerun: boolean;
  /** Whether a non-collinear test phase was triggered */
  noncollinearTestTriggered: boolean;
  /** Whether a spiral-magnet test was flagged */
  spiralMagnetFlagged: boolean;
  /** Total magnetization of the ground state */
  groundStateMagnetization: number | null;
  /** Human-readable notes */
  notes: string[];
}

// ─── Magnetic element classification ─────────────────────────────────

/** Elements with established magnetic moments in compounds (μB per atom).
 *  Lanthanides Pr/Tb/Dy/Ho/Er/Tm moved here from WEAK_MAGNETIC — their Ln³⁺
 *  free-ion moments are 3-11 μB (Ho³⁺ is 10.6 μB, the largest of any element).
 *  Previously classified as "weak" → started SCF with mag=0.3 μB which was
 *  too small to break symmetry, often converging to the wrong magnetic state.
 *  Actinides U/Np/Pu/Am/Cm have large 5f moments in compounds (heavy-fermion
 *  SC parent compounds like UPt3, NpBe13, PuCoGa5 are all magnetic).  */
const STRONG_MAGNETIC: Record<string, number> = {
  // 3d
  Fe: 2.2, Co: 1.7, Ni: 0.6, Mn: 3.0, Cr: 1.5, V: 0.5,
  // 4f (Ln³⁺ effective moments)
  Pr: 3.6, Nd: 3.0, Sm: 1.0, Eu: 7.0, Gd: 7.0,
  Tb: 9.7, Dy: 10.6, Ho: 10.6, Er: 9.6, Tm: 7.6,
  // 5f
  U: 3.0, Np: 2.5, Pu: 4.0, Am: 5.0, Cm: 7.0,
};

/** Elements that can develop induced moments in compounds */
const WEAK_MAGNETIC = new Set([
  "Ti", "Sc", "Cu", "Ru", "Rh", "Pd", "Os", "Ir", "Pt",
  "Ce", "Yb",  // 4f¹ and 4f¹³/¹⁴ — moments depend on valence; small intrinsic
]);

/** Anion/ligand elements that mediate superexchange */
const EXCHANGE_MEDIATORS = new Set(["O", "F", "S", "Se", "Te", "As", "P", "N"]);

// ─── Convergence thresholds ──────────────────────────────────────────

/** Initial triage conv_thr — tightened from 1e-5 to 1e-6 for reliable energy ordering */
export const TRIAGE_CONV_THR = 1e-6;

/** Tight conv_thr for re-running near-degenerate candidates */
export const TIGHT_CONV_THR = 1e-7;

/** Energy gap threshold (meV/atom) below which states are "nearly degenerate" */
export const NEAR_DEGENERATE_THRESHOLD_MEV = 5.0;

/** Energy gap threshold (meV/atom) for triggering tight-convergence rerun */
export const TIGHT_RERUN_THRESHOLD_MEV = 3.0;

/** Conversion: 1 Ry = 13605.7 meV */
const RY_TO_MEV = 13605.7;

// ─── Spiral-magnet detection ─────────────────────────────────────────

/** Element pairs that commonly form helimagnetic / spin-spiral ground states */
const SPIRAL_MAGNET_PAIRS: Array<{ mag: string; partner: string; system: string }> = [
  { mag: "Fe", partner: "Ge", system: "FeGe (B20 helimagnet)" },
  { mag: "Mn", partner: "Si", system: "MnSi (B20 helimagnet)" },
  { mag: "Mn", partner: "Ge", system: "MnGe (B20 helimagnet)" },
  { mag: "Cr", partner: "Al", system: "Cr-alloy (SDW / spin spiral)" },
  { mag: "Cr", partner: "Mn", system: "CrMn (competing exchange → spiral)" },
  { mag: "Fe", partner: "Sn", system: "Fe-Sn (kagome helimagnet)" },
];

/**
 * Check if composition matches a known spiral-magnet pattern.
 */
function detectSpiralMagnetism(elements: string[]): { flagged: boolean; systems: string[] } {
  const systems: string[] = [];
  for (const pair of SPIRAL_MAGNET_PAIRS) {
    if (elements.includes(pair.mag) && elements.includes(pair.partner)) {
      systems.push(pair.system);
    }
  }
  return { flagged: systems.length > 0, systems };
}

// ─── Magnetic ground-state search logic ──────────────────────────────

/**
 * Classify the magnetic landscape for a given composition.
 *
 * Returns the set of magnetic orderings that should be tried before
 * committing to phonon calculations.
 *
 * Decision logic:
 *   1. No magnetic elements → NM only (skip search)
 *   2. Single magnetic species, no mediator → FM + NM
 *   3. Single magnetic species + mediator (O, S, As) → FM + AFM + NM
 *   4. Multiple magnetic species → FM + multiple AFM patterns + NM
 *   5. Known families (cuprate, pnictide) → targeted AFM patterns
 *   6. Spiral-magnet pairs (FeGe, MnSi, Cr-alloy) → non-collinear spiral test
 *   7. Fe-pnictide → double-stripe / bicollinear ordering
 *   8. When SOC handler requires full SOC → all trials run non-collinear from start
 *
 * @param socRequiresNoncolin - if true, SOC handler mandates non-collinear mode
 */
export function classifyMagneticLandscape(
  elements: string[],
  counts: Record<string, number>,
  socRequiresNoncolin: boolean = false,
): MagneticTrialConfig[] {
  const configs: MagneticTrialConfig[] = [];
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  const strongMagElements = elements.filter(el => el in STRONG_MAGNETIC);
  const weakMagElements = elements.filter(el => WEAK_MAGNETIC.has(el));
  const hasMediators = elements.some(el => EXCHANGE_MEDIATORS.has(el));
  const allMagElements = [...strongMagElements, ...weakMagElements];

  // No magnetic elements at all → skip search
  if (allMagElements.length === 0) {
    return [];
  }

  // Always try NM as baseline
  configs.push({
    ordering: "NM",
    nspin: 1,
    magnetizationBlock: "",
    costFactor: 1.0,
    rationale: "Non-magnetic baseline for energy comparison",
  });

  // Always try FM
  const fmLines = buildMagnetizationBlock(elements, counts, "FM");
  configs.push({
    ordering: "FM",
    nspin: 2,
    magnetizationBlock: fmLines,
    costFactor: 1.5,
    rationale: "Ferromagnetic ordering — all moments parallel",
  });

  // Known family patterns
  const hasFe = elements.includes("Fe");
  const hasCu = elements.includes("Cu");
  const hasMn = elements.includes("Mn");
  const hasCr = elements.includes("Cr");
  const hasNi = elements.includes("Ni");
  const hasAs = elements.includes("As");
  const hasP = elements.includes("P");
  const hasO = elements.includes("O");
  const hasSe = elements.includes("Se");
  const hasTe = elements.includes("Te");
  const hasS = elements.includes("S");

  // [MagSearch] Fe-pnictide/chalcogenide: stripe AFM is almost always the ground state
  if (hasFe && (hasAs || hasP || hasSe || hasTe || hasS)) {
    configs.push({
      ordering: "AFM-stripe",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "AFM-stripe"),
      costFactor: 1.5,
      rationale: "Stripe AFM — ground state for most Fe-pnictides (Mazin et al., PRL 101, 057003 (2008))",
    });
    configs.push({
      ordering: "AFM-checkerboard",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "AFM-checkerboard"),
      costFactor: 1.5,
      rationale: "Checkerboard AFM — competitor to stripe in Fe-pnictides",
    });
    // [MagSearch] Double-stripe / bicollinear ordering for Fe-pnictides
    // (↑↑↓↓ along one axis, ↑↓ along the other — relevant for FeTe, some 122s)
    configs.push({
      ordering: "AFM-double-stripe",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "AFM-double-stripe"),
      costFactor: 1.5,
      rationale: "Double-stripe (bicollinear) AFM — ground state for FeTe, competitor in Fe-pnictides (Bao et al., PRL 102, 247001 (2009))",
    });
  }

  // Cuprate: layered AFM (Neel order in CuO2 planes)
  else if (hasCu && hasO) {
    configs.push({
      ordering: "AFM-layered",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "AFM-layered"),
      costFactor: 1.5,
      rationale: "Layered AFM — Neel order in CuO2 planes (parent cuprate state)",
    });
  }

  // Mn-O, Cr-O: multiple AFM patterns possible
  else if ((hasMn || hasCr) && hasO) {
    configs.push({
      ordering: "AFM-checkerboard",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "AFM-checkerboard"),
      costFactor: 1.5,
      rationale: `${hasMn ? "Mn" : "Cr"}-oxide: checkerboard AFM via superexchange`,
    });
    configs.push({
      ordering: "AFM-alternating",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "AFM-alternating"),
      costFactor: 1.5,
      rationale: `${hasMn ? "Mn" : "Cr"}-oxide: alternating AFM pattern`,
    });
  }

  // Ni compounds (nickelates)
  else if (hasNi && hasO) {
    configs.push({
      ordering: "AFM-checkerboard",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "AFM-checkerboard"),
      costFactor: 1.5,
      rationale: "Ni-oxide: checkerboard AFM (NiO-type superexchange)",
    });
  }

  // General: any magnetic element + exchange mediator → try AFM
  else if (strongMagElements.length > 0 && hasMediators) {
    configs.push({
      ordering: "AFM-alternating",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "AFM-alternating"),
      costFactor: 1.5,
      rationale: "Generic AFM — superexchange via anion mediator",
    });
  }

  // Multiple magnetic species → try ferrimagnetic
  if (strongMagElements.length >= 2) {
    configs.push({
      ordering: "ferrimagnetic",
      nspin: 2,
      magnetizationBlock: buildMagnetizationBlock(elements, counts, "ferrimagnetic"),
      costFactor: 1.5,
      rationale: "Ferrimagnetic — unequal opposing moments on different magnetic sublattices",
    });
  }

  // [MagSearch] Spiral-magnet detection: FeGe, MnSi, Cr-alloy systems
  const spiral = detectSpiralMagnetism(elements);
  if (spiral.flagged) {
    configs.push({
      ordering: "noncollinear-spiral",
      nspin: 2,
      noncolin: true,
      magnetizationBlock: buildNoncollinearMagBlock(elements, counts, "spiral"),
      costFactor: 2.5,
      rationale: `[MagSearch] Spiral-magnet candidate: ${spiral.systems.join(", ")}. ` +
        `Non-collinear spin-spiral test with canted initial moments.`,
    });
  }

  // [MagSearch] SOC handler mandates non-collinear → convert all collinear trials
  // and add a canted non-collinear test
  if (socRequiresNoncolin) {
    for (const cfg of configs) {
      if (cfg.nspin === 2 && !cfg.noncolin) {
        // Convert collinear trials to non-collinear angle format
        cfg.noncolin = true;
        cfg.magnetizationBlock = convertToAngleFormat(cfg.magnetizationBlock);
        cfg.costFactor *= 1.5; // spinor basis overhead
        cfg.rationale = `[SOC-noncolin] ${cfg.rationale}`;
      }
    }
  }

  // QE's `starting_magnetization(i)` is per-species, not per-atom. For a cell
  // with only ONE strong-magnetic species (e.g., BaFe2As2 → species Ba, Fe,
  // As — only Fe is magnetic), every collinear AFM-* trial produces the
  // identical FM input because all Fe atoms share `starting_magnetization(Fe)`
  // and there's nothing to alternate against. Without species-splitting (Fe1,
  // Fe2 pointing to the same pseudo with opposite signs — not implemented
  // here yet), running 3 AFM-* trials wastes 3× SCF time and returns
  // identical energies. Deduplicate by magnetization-block content; keep
  // only the first occurrence and mark the duplicates as dropped.
  const seenBlocks = new Set<string>();
  const dedupedConfigs: MagneticTrialConfig[] = [];
  const droppedOrderings: MagneticOrdering[] = [];
  for (const cfg of configs) {
    // Non-collinear blocks legitimately differ in angle1/angle2 and should
    // not be deduplicated on the collinear magnetization line alone.
    const key = cfg.noncolin ? `nc:${cfg.magnetizationBlock}` : cfg.magnetizationBlock;
    if (seenBlocks.has(key)) {
      droppedOrderings.push(cfg.ordering);
      continue;
    }
    seenBlocks.add(key);
    dedupedConfigs.push(cfg);
  }
  if (droppedOrderings.length > 0 && strongMagElements.length <= 1) {
    console.warn(`[MagSearch] Single magnetic species (${strongMagElements.join(",") || "weak-mag only"}): ` +
      `AFM trials ${droppedOrderings.join(", ")} collapse to FM (QE starting_magnetization is per-species). ` +
      `True AFM requires species-splitting (e.g. Fe1/Fe2 → opposite signs); skipped duplicates to save SCF time.`);
  }

  return dedupedConfigs;
}

/**
 * Build the starting_magnetization block for a given magnetic ordering.
 */
function buildMagnetizationBlock(
  elements: string[],
  counts: Record<string, number>,
  ordering: MagneticOrdering,
): string {
  let lines = "";
  let magSpeciesIdx = 0;

  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    const strongMag = STRONG_MAGNETIC[el];
    const isWeak = WEAK_MAGNETIC.has(el);
    let mag = strongMag ?? (isWeak ? 0.3 : 0.0);

    switch (ordering) {
      case "NM":
        mag = 0.0;
        break;
      case "FM":
        // All positive — keep mag as-is
        break;
      case "AFM-checkerboard":
        if (mag > 0) {
          mag = magSpeciesIdx % 2 === 1 ? -mag : mag;
          magSpeciesIdx++;
        }
        break;
      case "AFM-stripe":
        // For Fe-pnictides, stripe means (↑↑↓↓) pattern
        // In species-based seeding, alternate pairs
        if (mag > 0) {
          const pair = Math.floor(magSpeciesIdx / 2);
          mag = pair % 2 === 1 ? -mag : mag;
          magSpeciesIdx++;
        }
        break;
      case "AFM-layered":
        // For cuprates, half the magnetic species up, half down
        if (mag > 0) {
          const nMagSpecies = elements.filter(e => e in STRONG_MAGNETIC || WEAK_MAGNETIC.has(e)).length;
          mag = magSpeciesIdx >= Math.ceil(nMagSpecies / 2) ? -mag : mag;
          magSpeciesIdx++;
        }
        break;
      case "AFM-alternating":
        if (mag > 0) {
          mag = magSpeciesIdx % 2 === 1 ? -mag : mag;
          magSpeciesIdx++;
        }
        break;
      case "AFM-double-stripe":
        // Double-stripe / bicollinear: (↑↑↓↓) pattern — groups of 2
        // For species-based seeding with limited species, use alternating
        // with a 2-period grouping
        if (mag > 0) {
          const group = Math.floor(magSpeciesIdx / 2) % 2;
          mag = group === 1 ? -mag : mag;
          magSpeciesIdx++;
        }
        break;
      case "ferrimagnetic":
        // First magnetic species keeps full moment, others get reduced opposing
        if (mag > 0) {
          if (magSpeciesIdx > 0) {
            mag = -mag * 0.5; // Reduced opposing moment
          }
          magSpeciesIdx++;
        }
        break;
      case "noncollinear-canted":
      case "noncollinear-spiral":
        // Handled by buildNoncollinearMagBlock instead
        break;
    }

    lines += `  starting_magnetization(${i + 1}) = ${mag.toFixed(1)},\n`;
  }

  return lines;
}

/**
 * Build non-collinear magnetization block using angle1/angle2 format.
 * Used for spiral and canted spin tests.
 *
 * @param mode - "spiral" for rotating angles, "canted" for tilted FM
 */
function buildNoncollinearMagBlock(
  elements: string[],
  counts: Record<string, number>,
  mode: "spiral" | "canted",
): string {
  let lines = "";
  let magIdx = 0;

  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    const strongMag = STRONG_MAGNETIC[el];
    const isWeak = WEAK_MAGNETIC.has(el);
    // Non-magnetic atoms (ligands like O, N, As, Te, halogens) should start
    // with zero magnetization. The old 0.1 default seeded spurious moments
    // on ligands that QE then had to relax away — wasted iterations, and
    // for some structures (e.g., cuprates near AFM Néel point) could trap
    // SCF in a wrong magnetic configuration. Magnetic species below
    // dominate symmetry-breaking via their strong_mag / weak_mag values.
    const mag = strongMag ?? (isWeak ? 0.3 : 0.0);

    lines += `  starting_magnetization(${i + 1}) = ${mag.toFixed(2)},\n`;

    if (mode === "spiral" && (strongMag || isWeak)) {
      // Rotate angle2 (azimuthal) by 90° per magnetic species to seed a spiral
      const angle1 = 90.0; // equatorial plane
      const angle2 = (magIdx * 90.0) % 360.0;
      lines += `  angle1(${i + 1}) = ${angle1.toFixed(1)},\n`;
      lines += `  angle2(${i + 1}) = ${angle2.toFixed(1)},\n`;
      magIdx++;
    } else if (mode === "canted" && (strongMag || isWeak)) {
      // Tilt from FM by 30° per species to allow relaxation to canted state
      const angle1 = magIdx * 30.0;
      const angle2 = magIdx * 45.0;
      lines += `  angle1(${i + 1}) = ${angle1.toFixed(1)},\n`;
      lines += `  angle2(${i + 1}) = ${angle2.toFixed(1)},\n`;
      magIdx++;
    } else {
      lines += `  angle1(${i + 1}) = 0.0,\n`;
      lines += `  angle2(${i + 1}) = 0.0,\n`;
    }
  }
  return lines;
}

/**
 * Convert a collinear starting_magnetization block to non-collinear angle format.
 * Positive mag → angle1=0 (spin up), negative mag → angle1=180 (spin down).
 * Used when SOC handler mandates noncolin=.true.
 */
function convertToAngleFormat(collinearBlock: string): string {
  const lines = collinearBlock.split("\n").filter(l => l.trim().length > 0);
  let result = "";
  for (const line of lines) {
    const match = line.match(/starting_magnetization\((\d+)\)\s*=\s*([-\d.]+)/);
    if (match) {
      const idx = match[1];
      const mag = parseFloat(match[2]);
      const absMag = Math.abs(mag);
      const angle1 = mag < 0 ? 180.0 : 0.0;
      result += `  starting_magnetization(${idx}) = ${absMag.toFixed(1)},\n`;
      result += `  angle1(${idx}) = ${angle1.toFixed(1)},\n`;
      result += `  angle2(${idx}) = 0.0,\n`;
    }
  }
  return result;
}

/**
 * Select the magnetic ground state from a set of trial results.
 *
 * The ground state is the ordering with the lowest total energy.
 * If energies are within ~5 meV/atom, the states are considered nearly degenerate
 * and a non-collinear canted-spin test should be run before committing to phonons.
 * If within ~3 meV/atom, re-run top 2 candidates with conv_thr=1e-7 to confirm.
 */
export function selectMagneticGroundState(
  trials: MagneticTrialResult[],
  configs: MagneticTrialConfig[],
  totalAtoms: number,
): MagneticGroundStateResult {
  const notes: string[] = [];

  // Accept strictly-converged OR near-converged trials (accuracy ≤ 1e-3 Ry
  // is good enough to rank FM vs AFM, which typically differ by tens of meV
  // — 1e-3 Ry ≈ 13.6 meV, still well under realistic FM-AFM gaps).
  // Originally set to 1e-4 (May 2026), but observed HgBa2CuO4 NM trial
  // landing at accuracy=8.8e-4 (just above 1e-4) and being rejected, forcing
  // a fallback to FM-with-broadened-seed even though NM was clearly the
  // best result. 1e-3 is the right threshold for trial RANKING (we'd still
  // require tighter convergence for the WINNING state's full vc-relax).
  const NEAR_CONVERGED_RY = 1e-3;
  const isUsableTrial = (t: MagneticTrialResult): boolean =>
    t.totalEnergy !== null && (
      t.converged
      || (t.lastScfAccuracyRy !== null && t.lastScfAccuracyRy < NEAR_CONVERGED_RY)
    );
  const convergedTrials = trials.filter(isUsableTrial);

  if (convergedTrials.length === 0) {
    // Distinguish two failure modes:
    //   (a) "ran but didn't converge" — at least one trial got past SCF parse
    //       and produced an accuracy reading. Magnetic ordering is hard but
    //       the binary works; falling back to FM with broadened seeding is
    //       reasonable.
    //   (b) "trials crashed instantly" — every trial returned with totalEnergy
    //       null AND lastScfAccuracyRy null AND short wallTime. The SCF
    //       infrastructure is broken (e.g. La4Ni3O10 May-2026: ylmr2 error
    //       from missing lmaxx=6 in QE binary). Falling back to FM and
    //       continuing wastes another hour of downstream compute on a
    //       material the worker physically can't handle.
    const everyTrialCrashedInstantly = trials.length > 0 && trials.every(t =>
      t.totalEnergy === null
      && t.lastScfAccuracyRy === null
      && t.wallTimeMs < 60_000  // < 60s = crashed before SCF could start
    );
    if (everyTrialCrashedInstantly) {
      notes.push(`[MagSearch] All ${trials.length} mag trials crashed instantly (< 60s each, no SCF accuracy) — magnetic search infrastructure broken; not falling back to FM`);
      return {
        searchPerformed: true,
        groundState: "NM",       // safest default — downstream can refuse to spin-polarize
        trials,
        energyGapPerAtom: 0,
        wellSeparated: false,
        winningMagBlock: "",
        winningNspin: 1,         // tell downstream NOT to spin-polarize
        winningNoncolin: false,
        tightConvergenceRerun: false,
        noncollinearTestTriggered: false,
        spiralMagnetFlagged: false,
        groundStateMagnetization: null,
        notes,
      };
    }
    notes.push("[MagSearch] No magnetic trial converged — falling back to FM with broadened seeding");
    const fmConfig = configs.find(c => c.ordering === "FM") ?? configs[0];
    return {
      searchPerformed: true,
      groundState: "FM",
      trials,
      energyGapPerAtom: 0,
      wellSeparated: false,
      winningMagBlock: fmConfig?.magnetizationBlock ?? "",
      winningNspin: 2,
      winningNoncolin: fmConfig?.noncolin ?? false,
      tightConvergenceRerun: false,
      noncollinearTestTriggered: false,
      spiralMagnetFlagged: false,
      groundStateMagnetization: null,
      notes,
    };
  }

  // Sort by energy (lowest first)
  const sorted = [...convergedTrials].sort((a, b) => (a.totalEnergy ?? 0) - (b.totalEnergy ?? 0));
  const winner = sorted[0];
  const winnerConfig = configs.find(c => c.ordering === winner.ordering)!;

  // Energy gap to next-best. totalEnergy is in eV (per MagneticTrialResult
  // docstring above), so eV→meV is ×1000, NOT ×RY_TO_MEV (13605.7).
  // The previous Ry→meV conversion reported every gap 13605× too large,
  // breaking ALL three downstream gates:
  //   (1) wellSeparated (gap > 5 meV/atom) was always true — even truly
  //       degenerate FM/AFM trials looked "clearly separated"
  //   (2) needsTightRerun (gap < 3 meV/atom) was always false — tight
  //       reruns never triggered for near-degenerate magnetic states
  //   (3) needsNoncollinearTest (gap < 5 meV/atom) was always false —
  //       non-collinear / spiral candidates never got that test
  // For BaFe2As2 with real FM-AFM gap ~10 meV/atom, the bug reported
  // ~136 eV/atom — clearly absurd if anyone had spot-checked the log.
  let energyGap = 0;
  let energyGapMeV = 0;
  if (sorted.length >= 2) {
    energyGap = ((sorted[1].totalEnergy ?? 0) - (sorted[0].totalEnergy ?? 0)) / totalAtoms;
    energyGapMeV = energyGap * 1000;  // eV → meV
  }
  const wellSeparated = energyGapMeV > NEAR_DEGENERATE_THRESHOLD_MEV;

  // [MagSearch] Detect if tight-convergence rerun is needed
  const needsTightRerun = sorted.length >= 2 && energyGapMeV < TIGHT_RERUN_THRESHOLD_MEV && energyGapMeV > 0;

  // [MagSearch] Detect if non-collinear canted test should be triggered
  // When collinear FM/AFM/NM energies are within 5 meV/atom, the true ground state
  // may be non-collinear (canted, spiral) — flag for non-collinear test
  const collinearTrials = sorted.filter(t =>
    !t.ordering.startsWith("noncollinear")
  );
  const needsNoncollinearTest = collinearTrials.length >= 2 && energyGapMeV < NEAR_DEGENERATE_THRESHOLD_MEV;

  // Check if any spiral trial was included
  const hasSpiralTrial = sorted.some(t => t.ordering === "noncollinear-spiral");

  notes.push(
    `[MagSearch] Magnetic ground state: ${winner.ordering} ` +
    `(E = ${winner.totalEnergy?.toFixed(6)} eV)`
  );

  if (sorted.length >= 2) {
    notes.push(
      `[MagSearch] Energy gap to next state (${sorted[1].ordering}): ` +
      `${(energyGap * 1000).toFixed(2)} mRy/atom ` +
      `(${energyGapMeV.toFixed(1)} meV/atom)` +
      (wellSeparated ? " — well separated" : " — nearly degenerate, phonons may be sensitive to ordering")
    );
  }

  if (winner.ordering === "NM" && sorted.length >= 2) {
    notes.push(
      "[MagSearch] Non-magnetic state is the ground state — no spontaneous magnetism. " +
      "nspin=1 is correct for phonon calculations."
    );
  }

  if (needsTightRerun) {
    notes.push(
      `[MagSearch] TIGHT-CONVERGENCE RERUN NEEDED: energy gap ${energyGapMeV.toFixed(1)} meV/atom ` +
      `< ${TIGHT_RERUN_THRESHOLD_MEV} meV threshold. Re-run ${sorted[0].ordering} and ${sorted[1].ordering} ` +
      `with conv_thr=${TIGHT_CONV_THR} to confirm energy ordering.`
    );
  }

  if (needsNoncollinearTest) {
    notes.push(
      `[MagSearch] NON-COLLINEAR TEST RECOMMENDED: collinear states within ` +
      `${energyGapMeV.toFixed(1)} meV/atom — run canted-spin non-collinear test ` +
      `before committing to phonons. True ground state may be non-collinear.`
    );
  }

  if (!wellSeparated && sorted.length >= 2) {
    notes.push(
      `[MagSearch] WARNING: Magnetic states nearly degenerate (<${NEAR_DEGENERATE_THRESHOLD_MEV} meV/atom). ` +
      `Phonon spectra may differ qualitatively between orderings. ` +
      `Consider running phonons for both ${sorted[0].ordering} and ${sorted[1].ordering}.`
    );
  }

  if (winner.totalMagnetization !== null) {
    notes.push(
      `Total magnetization: ${winner.totalMagnetization.toFixed(3)} Bohr mag/cell` +
      (winner.absoluteMagnetization !== null
        ? `, absolute: ${winner.absoluteMagnetization.toFixed(3)} Bohr mag/cell`
        : "")
    );
  }

  // Summary of all trials
  for (const trial of sorted) {
    const eStr = trial.totalEnergy !== null ? `${trial.totalEnergy.toFixed(6)} eV` : "N/A";
    const mStr = trial.totalMagnetization !== null ? `M=${trial.totalMagnetization.toFixed(2)}` : "";
    notes.push(
      `  Trial ${trial.ordering}: E=${eStr} ${mStr} ` +
      `[${trial.converged ? "converged" : "FAILED"}, ${trial.wallTimeMs}ms]`
    );
  }

  return {
    searchPerformed: true,
    groundState: winner.ordering,
    trials,
    energyGapPerAtom: Number(Math.abs(energyGap).toFixed(6)),
    wellSeparated,
    winningMagBlock: winnerConfig?.magnetizationBlock ?? "",
    winningNspin: winnerConfig?.nspin ?? 2,
    winningNoncolin: winnerConfig?.noncolin ?? false,
    tightConvergenceRerun: needsTightRerun,
    noncollinearTestTriggered: needsNoncollinearTest,
    spiralMagnetFlagged: hasSpiralTrial,
    groundStateMagnetization: winner.totalMagnetization,
    notes,
  };
}

/**
 * Determine whether a magnetic ground-state search is warranted.
 *
 * Skip the search for:
 *   - Materials with no magnetic elements
 *   - Pure metals without exchange mediators (simple FM/NM)
 *   - Materials where magnetic ordering is well-established
 *
 * Always search for:
 *   - Fe + pnictogen/chalcogen (pnictide SC candidates)
 *   - Cu + O (cuprate SC candidates)
 *   - Mn or Cr compounds (complex magnetic landscapes)
 *   - Multiple TM species (competing exchange interactions)
 */
export function shouldSearchMagneticGS(
  elements: string[],
  counts: Record<string, number>,
): { shouldSearch: boolean; reason: string } {
  const strongMag = elements.filter(el => el in STRONG_MAGNETIC);
  const weakMag = elements.filter(el => WEAK_MAGNETIC.has(el));
  const hasMediators = elements.some(el => EXCHANGE_MEDIATORS.has(el));

  if (strongMag.length === 0 && weakMag.length === 0) {
    return { shouldSearch: false, reason: "No magnetic elements" };
  }

  // Always search for known complex magnetic systems
  const hasFe = elements.includes("Fe");
  const hasCu = elements.includes("Cu");
  const hasMn = elements.includes("Mn");
  const hasCr = elements.includes("Cr");
  const hasNi = elements.includes("Ni");
  const hasAs = elements.includes("As");
  const hasP = elements.includes("P");
  const hasO = elements.includes("O");
  const hasSe = elements.includes("Se");
  const hasTe = elements.includes("Te");

  if (hasFe && (hasAs || hasP || hasSe || hasTe)) {
    return { shouldSearch: true, reason: "Fe-pnictide/chalcogenide — stripe vs checkerboard AFM competition" };
  }
  if (hasCu && hasO) {
    return { shouldSearch: true, reason: "Cu-O compound — Neel AFM ground state expected" };
  }
  if ((hasMn || hasCr) && hasO) {
    return { shouldSearch: true, reason: `${hasMn ? "Mn" : "Cr"}-oxide — complex magnetic landscape` };
  }
  if (hasNi && hasO) {
    return { shouldSearch: true, reason: "Ni-oxide — competing magnetic orderings in nickelates" };
  }

  // Multiple strong magnetic species
  if (strongMag.length >= 2) {
    return { shouldSearch: true, reason: `Multiple magnetic species (${strongMag.join(", ")}) — competing exchange` };
  }

  // Single magnetic element + exchange mediator
  if (strongMag.length === 1 && hasMediators) {
    return { shouldSearch: true, reason: `${strongMag[0]} + anion mediator — superexchange may favor AFM` };
  }

  // Single magnetic element, no mediator → FM is almost always correct
  if (strongMag.length === 1 && !hasMediators) {
    return { shouldSearch: false, reason: `Single magnetic species (${strongMag[0]}) without mediator — FM assumed` };
  }

  // Weak magnetic only
  if (weakMag.length > 0 && strongMag.length === 0) {
    return { shouldSearch: false, reason: `Only weak magnetic elements (${weakMag.join(", ")}) — NM/FM sufficient` };
  }

  // [MagSearch] Spiral-magnet pairs (FeGe, MnSi, etc.)
  const spiral = detectSpiralMagnetism(elements);
  if (spiral.flagged) {
    return { shouldSearch: true, reason: `[MagSearch] Spiral-magnet candidate: ${spiral.systems.join(", ")}` };
  }

  return { shouldSearch: false, reason: "Default: no complex magnetic landscape detected" };
}

/**
 * Generate a non-collinear canted-spin trial config for near-degenerate systems.
 *
 * Called when collinear triage shows energies within 5 meV/atom — the true
 * ground state may be a canted or non-collinear state that collinear SCF cannot reach.
 */
export function buildCantedNoncollinearTrial(
  elements: string[],
  counts: Record<string, number>,
): MagneticTrialConfig {
  return {
    ordering: "noncollinear-canted",
    nspin: 2,
    noncolin: true,
    magnetizationBlock: buildNoncollinearMagBlock(elements, counts, "canted"),
    costFactor: 2.5,
    rationale: "[MagSearch] Canted non-collinear test — triggered by near-degenerate collinear states (<5 meV/atom gap)",
  };
}

/**
 * Parse total and absolute magnetization from QE SCF output.
 *
 * QE prints:
 *   total magnetization       =     X.XX Bohr mag/cell
 *   absolute magnetization    =     X.XX Bohr mag/cell
 */
export function parseMagnetizationFromOutput(output: string): {
  totalMagnetization: number | null;
  absoluteMagnetization: number | null;
} {
  let totalMag: number | null = null;
  let absMag: number | null = null;

  // Take the LAST occurrence (final SCF step)
  const totalRegex = /total magnetization\s*=\s*([-\d.]+)\s*Bohr/g;
  let totalMatch: RegExpExecArray | null;
  while ((totalMatch = totalRegex.exec(output)) !== null) {
    const val = parseFloat(totalMatch[1]);
    if (Number.isFinite(val)) totalMag = val;
  }

  const absRegex = /absolute magnetization\s*=\s*([-\d.]+)\s*Bohr/g;
  let absMatch: RegExpExecArray | null;
  while ((absMatch = absRegex.exec(output)) !== null) {
    const val = parseFloat(absMatch[1]);
    if (Number.isFinite(val)) absMag = val;
  }

  return { totalMagnetization: totalMag, absoluteMagnetization: absMag };
}
