/**
 * Hubbard U workflow for correlated-electron DFT calculations.
 *
 * Plain GGA (PBE) gives qualitatively wrong electronic structure for any
 * material with localized d or f electrons: band gaps are underestimated,
 * orbital ordering is wrong, and magnetic moments are too small. This
 * propagates directly into N(E_F), lambda, and Tc predictions.
 *
 * DFT+U adds an on-site Coulomb correction to localized orbitals:
 *   E_DFT+U = E_DFT + U/2 * Σ_m [n_m - n_m²]
 * where n_m are the occupation numbers of the correlated orbitals.
 *
 * This module provides:
 *   1. Composition-aware U value selection (not just element-specific)
 *   2. Determination of which orbitals need +U correction
 *   3. QE input block generation for all calculation types
 *   4. Validation and warnings for unusual U values
 *
 * U value selection hierarchy:
 *   1. Material-specific override (known compounds with validated U)
 *   2. Oxidation-state-aware U from literature tables
 *   3. Element-specific default from ELEMENTAL_DATA
 *
 * Key references:
 * @see V. I. Anisimov et al., Phys. Rev. B 44, 943 (1991) — original DFT+U
 * @see S. L. Dudarev et al., Phys. Rev. B 57, 1505 (1998) — simplified rotationally-invariant DFT+U
 * @see M. Cococcioni & S. de Gironcoli, Phys. Rev. B 71, 035105 (2005) — linear-response U
 * @see I. Timrov et al., Phys. Rev. B 98, 085127 (2018) — HP code for self-consistent U
 * @see B. Himmetoglu et al., Int. J. Quantum Chem. 114, 14 (2014) — DFT+U review
 */

import {
  getHubbardU,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "../learning/elemental-data";
import { parseFormulaElements } from "../learning/physics-engine";

// ─── Types ───────────────────────────────────────────────────────────

export interface HubbardSiteConfig {
  /** Element symbol */
  element: string;
  /** Species index in QE (1-based) */
  speciesIndex: number;
  /** Effective U value (eV) — Dudarev U_eff = U - J */
  uEffective: number;
  /** Hund's coupling J (eV) — nonzero for Liechtenstein kind=1 */
  hunds: number;
  /** Which orbital manifold gets the correction */
  orbitalManifold: "3d" | "4d" | "5d" | "4f" | "5f" | "3p" | "none";
  /** Source of U value */
  source: "material-specific" | "oxidation-aware" | "element-default" | "none";
  /** Whether this element genuinely needs +U */
  needsU: boolean;
}

export interface HubbardWorkflowResult {
  /** Whether DFT+U should be applied */
  applyDFTplusU: boolean;
  /** Per-site Hubbard configurations */
  sites: HubbardSiteConfig[];
  /**
   * QE ≥7.1 HUBBARD card block — goes AFTER ATOMIC_SPECIES, not in &SYSTEM.
   * Old syntax (lda_plus_u in &SYSTEM) is rejected by QE 7.1+.
   * New syntax uses: HUBBARD (ortho-atomic) / U Element-manifold value
   */
  qeHubbardCard: string;
  /** @deprecated kept empty for compatibility — Hubbard params now go in qeHubbardCard */
  qeSystemBlock: string;
  /** Correlation regime from analysis */
  correlationRegime: string;
  /** Material class patterns detected */
  materialPatterns: string[];
  /** Human-readable notes */
  notes: string[];
  /** Total number of correlated sites */
  correlatedSiteCount: number;
  /** Whether DFT+U should also be used in vc-relax (not just SCF) */
  applyToVCRelax: boolean;
  /** Whether DFT+U should be used in phonon calculations */
  applyToPhonons: boolean;
  /**
   * DFT+U formulation:
   *   0 = Dudarev simplified (default — U_eff = U - J, J channel lost)
   *   1 = Liechtenstein (explicit U and J — needed for orbital ordering,
   *       Hund's-driven physics in nickelates, ruthenates, multiorbital systems)
   */
  hubbardKind: 0 | 1;
}

// ─── Composition-aware U values ──────────────────────────────────────

/**
 * Material-specific U overrides for well-studied compounds.
 * These take priority over element defaults when the formula matches.
 *
 * Sources:
 *   - VASP wiki recommended values
 *   - Materials Project GGA+U settings (Jain et al., APL Mat. 1, 011002 (2013))
 *   - Specific DFT+U studies in the literature
 */
const MATERIAL_SPECIFIC_U: Array<{
  pattern: RegExp;
  overrides: Record<string, { u: number; orbital: string; source: string }>;
}> = [
  // Iron oxides: Fe³⁺ in octahedral coordination
  {
    pattern: /^Fe[23]?O/,
    overrides: {
      Fe: { u: 4.3, orbital: "3d", source: "Materials Project Fe-oxide (Jain et al. 2013)" },
    },
  },
  // Cuprates: Cu²⁺ in square-planar CuO₂
  {
    pattern: /(La|Y|Ba|Sr|Bi|Tl|Hg).*(Cu).*O/,
    overrides: {
      Cu: { u: 5.0, orbital: "3d", source: "Cuprate CuO₂ planes (Anisimov et al. 1991)" },
    },
  },
  // Nickelates: Ni¹⁺/Ni²⁺ in NiO₂ planes
  {
    pattern: /(Nd|La|Pr).*(Ni).*O/,
    overrides: {
      Ni: { u: 5.1, orbital: "3d", source: "Infinite-layer nickelate (Lechermann 2020)" },
    },
  },
  // Iron pnictides: Fe²⁺ in tetrahedral coordination
  {
    pattern: /(Ba|Sr|Ca|La|Nd|Sm).*Fe.*As|Fe(Se|Te|S)/,
    overrides: {
      Fe: { u: 3.0, orbital: "3d", source: "Fe-pnictide tetrahedral (Nakamura et al. 2009) — lower than octahedral" },
    },
  },
  // NiO: well-studied Mott insulator
  {
    pattern: /^NiO$/,
    overrides: {
      Ni: { u: 6.4, orbital: "3d", source: "NiO Mott insulator (Dudarev et al. 1998)" },
    },
  },
  // MnO: antiferromagnetic Mott insulator
  {
    pattern: /^MnO$/,
    overrides: {
      Mn: { u: 3.9, orbital: "3d", source: "MnO (Cococcioni & de Gironcoli 2005)" },
    },
  },
  // CoO
  {
    pattern: /^CoO$/,
    overrides: {
      Co: { u: 5.0, orbital: "3d", source: "CoO (Anisimov et al. 1993)" },
    },
  },
  // CeO₂: Ce⁴⁺ with empty 4f
  {
    pattern: /^CeO2$/,
    overrides: {
      Ce: { u: 5.0, orbital: "4f", source: "CeO₂ (Loschen et al. 2007)" },
    },
  },
  // LaNiO₃ perovskite
  {
    pattern: /^LaNiO3$/,
    overrides: {
      Ni: { u: 4.0, orbital: "3d", source: "LaNiO₃ perovskite (Park et al. 2012)" },
    },
  },
];

/**
 * Oxidation-state-aware U values for d-block elements.
 *
 * When no material-specific override matches, we estimate U based on
 * the likely oxidation state inferred from the chemical environment.
 *
 * Key insight: U increases with oxidation state because higher charge
 * means more localized d electrons and stronger on-site repulsion.
 *
 * Structure: [low-oxidation U, high-oxidation U, threshold atom ratio]
 * If O/F/Cl ratio per TM > threshold, assume high oxidation state.
 */
const OXIDATION_AWARE_U: Record<string, { lowOx: number; highOx: number; orbital: string }> = {
  // 3d metals
  Ti: { lowOx: 2.5, highOx: 3.5, orbital: "3d" },
  V:  { lowOx: 2.5, highOx: 3.5, orbital: "3d" },
  Cr: { lowOx: 2.5, highOx: 3.5, orbital: "3d" },
  Mn: { lowOx: 3.0, highOx: 4.0, orbital: "3d" },
  Fe: { lowOx: 3.0, highOx: 4.3, orbital: "3d" },
  Co: { lowOx: 3.0, highOx: 5.0, orbital: "3d" },
  Ni: { lowOx: 3.5, highOx: 6.0, orbital: "3d" },
  Cu: { lowOx: 4.0, highOx: 5.5, orbital: "3d" },
  // 4d metals
  Zr: { lowOx: 2.0, highOx: 3.0, orbital: "4d" },
  Nb: { lowOx: 1.5, highOx: 2.5, orbital: "4d" },
  Mo: { lowOx: 2.0, highOx: 3.5, orbital: "4d" },
  Ru: { lowOx: 2.5, highOx: 3.5, orbital: "4d" },
  Rh: { lowOx: 2.5, highOx: 4.0, orbital: "4d" },
  Pd: { lowOx: 2.0, highOx: 3.5, orbital: "4d" },
  // 5d metals — generally smaller U (more delocalized)
  Hf: { lowOx: 1.5, highOx: 2.5, orbital: "5d" },
  Ta: { lowOx: 1.5, highOx: 2.5, orbital: "5d" },
  W:  { lowOx: 1.5, highOx: 2.5, orbital: "5d" },
  Os: { lowOx: 2.0, highOx: 3.0, orbital: "5d" },
  Ir: { lowOx: 2.0, highOx: 3.0, orbital: "5d" },
  Pt: { lowOx: 2.0, highOx: 3.5, orbital: "5d" },
  // 4f lanthanides — large U for localized f electrons
  Ce: { lowOx: 4.5, highOx: 6.0, orbital: "4f" },
  Pr: { lowOx: 5.0, highOx: 6.5, orbital: "4f" },
  Nd: { lowOx: 5.5, highOx: 6.5, orbital: "4f" },
  Sm: { lowOx: 5.5, highOx: 7.0, orbital: "4f" },
  Eu: { lowOx: 6.0, highOx: 7.5, orbital: "4f" },
  Gd: { lowOx: 6.0, highOx: 7.0, orbital: "4f" },
  Tb: { lowOx: 5.5, highOx: 7.0, orbital: "4f" },
  Dy: { lowOx: 5.5, highOx: 7.0, orbital: "4f" },
  // 5f actinides
  Th: { lowOx: 3.0, highOx: 5.0, orbital: "5f" },
  U:  { lowOx: 3.0, highOx: 4.5, orbital: "5f" },
};

/** Electronegative anions that indicate oxidized TM environment */
const OXIDIZING_ANIONS = new Set(["O", "F", "Cl", "Br"]);

/**
 * Hund's coupling J values (eV) for elements where orbital ordering matters.
 * Used when Liechtenstein (kind=1) is selected. Values from constrained-RPA
 * and linear-response calculations in the literature.
 *
 * @see A. I. Liechtenstein et al., Phys. Rev. B 52, R5467 (1995)
 * @see F. Aryasetiawan et al., Phys. Rev. B 74, 125106 (2006) — cRPA J values
 * @see L. Vaugier et al., Phys. Rev. B 86, 165105 (2012) — cRPA for 3d oxides
 */
const HUNDS_J: Record<string, number> = {
  // 3d — J ~ 0.7-1.0 eV from cRPA
  Ti: 0.64, V: 0.68, Cr: 0.70, Mn: 0.75, Fe: 0.89,
  Co: 0.92, Ni: 0.95, Cu: 0.98,
  // 4d — J ~ 0.3-0.6 eV (more delocalized)
  Zr: 0.35, Nb: 0.40, Mo: 0.45, Ru: 0.50, Rh: 0.55, Pd: 0.60,
  // Lanthanides — J ~ 0.6-0.7 eV for 4f
  Ce: 0.68, Pr: 0.65, Nd: 0.63, Sm: 0.60, Eu: 0.60,
  Gd: 0.60, Tb: 0.62, Dy: 0.63, Ho: 0.65, Er: 0.66, Tm: 0.67,
};

/**
 * Detect whether Liechtenstein (kind=1) formulation is needed.
 *
 * Kind=1 matters when Hund's coupling J drives the physics:
 *   - Nickelates: orbital ordering in NiO₂ planes depends on J
 *   - Ruthenates (Sr₂RuO₄): multiorbital Hund's metal, J controls
 *     orbital-selective correlations
 *   - Chromates/vanadates: orbital ordering transitions
 *   - Any system where U/J ratio matters for ground-state selection
 *
 * @see A. Georges et al., Annu. Rev. Condens. Matter Phys. 4, 137 (2013) — Hund's metals
 * @see L. de' Medici et al., Phys. Rev. Lett. 107, 256401 (2011) — Hund's coupling
 */
function needsLiechtenstein(
  elements: string[],
  materialPatterns: string[],
): boolean {
  // Nickelates: J drives orbital ordering in NiO₂ planes
  if (elements.includes("Ni") && elements.includes("O") &&
      elements.some(e => ["La", "Nd", "Pr", "Sr"].includes(e))) return true;
  // Ruthenates: multiorbital Hund's metal
  if (elements.includes("Ru") && elements.includes("O") &&
      elements.some(e => ["Sr", "Ca", "Ba"].includes(e))) return true;
  // Vanadates/chromates with orbital ordering
  if ((elements.includes("V") || elements.includes("Cr")) &&
      elements.includes("O") && elements.length <= 4) return true;
  // Cobaltates: spin-state transitions driven by J
  if (elements.includes("Co") && elements.includes("O") &&
      elements.some(e => ["La", "Sr", "Na"].includes(e))) return true;
  // Heavy-fermion with strong J channel
  if (materialPatterns.some(p => p.includes("heavy-fermion") || p.includes("Kondo"))) return true;
  return false;
}

/** Elements where d/f electrons are localized enough to need +U */
function needsHubbardU(element: string): boolean {
  // 3d transition metals (most important for SC)
  if (["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"].includes(element)) return true;
  // 4d with localization tendency
  if (["Zr", "Nb", "Mo", "Ru", "Rh", "Pd"].includes(element)) return true;
  // 5d — generally more delocalized, +U mainly for oxides
  if (["Hf", "Ta", "W", "Os", "Ir", "Pt"].includes(element)) return true;
  // Lanthanides — always need +U for 4f
  if (isRareEarth(element)) return true;
  // Actinides — always need +U for 5f
  if (isActinide(element)) return true;
  return false;
}

/**
 * Estimate whether the TM is in a high-oxidation environment.
 *
 * Heuristic: if the ratio of electronegative anions to TM atoms > 2,
 * the TM is likely in a high oxidation state (e.g., Fe₂O₃ → Fe³⁺).
 */
function estimateHighOxidation(
  elements: string[],
  counts: Record<string, number>,
  targetElement: string,
): boolean {
  const anionCount = elements
    .filter(el => OXIDIZING_ANIONS.has(el))
    .reduce((s, el) => s + (counts[el] || 0), 0);
  const tmCount = counts[targetElement] || 1;
  return (anionCount / tmCount) > 2.0;
}

function getOrbitalManifold(element: string): "3d" | "4d" | "5d" | "4f" | "5f" | "3p" | "none" {
  const row3d = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"];
  const row4d = ["Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"];
  const row5d = ["La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"];
  const f4 = ["Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"];
  const f5 = ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm"];
  if (row3d.includes(element)) return "3d";
  if (row4d.includes(element)) return "4d";
  if (row5d.includes(element)) return "5d";
  if (f4.includes(element)) return "4f";
  if (f5.includes(element)) return "5f";
  return "none";
}

// ─── Main workflow ───────────────────────────────────────────────────

function parseFormulaCounts(formula: string): Record<string, number> {
  let cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  let result = cleaned.replace(/\[/g, "(").replace(/\]/g, ")");
  const parenRegex = /\(([^()]+)\)(\d*\.?\d*)/;
  let iterations = 0;
  while (result.includes("(") && iterations < 20) {
    const prev = result;
    result = result.replace(parenRegex, (_, group: string, mult: string) => {
      const m = mult ? parseFloat(mult) : 1;
      if (isNaN(m) || m <= 0) return group;
      if (m === 1) return group;
      return group.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_x: string, el: string, num: string) => {
        const n = num ? parseFloat(num) : 1;
        const newN = (isNaN(n) || n <= 0 ? 1 : n) * m;
        return newN === 1 ? el : `${el}${newN}`;
      });
    });
    if (result === prev) break;
    iterations++;
  }
  result = result.replace(/[()]/g, "");
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(result)) !== null) {
    const val = m[2] ? parseFloat(m[2]) : 1;
    counts[m[1]] = (counts[m[1]] || 0) + (isNaN(val) || val <= 0 ? 1 : val);
  }
  return counts;
}

/**
 * Determine whether a material is a hydrogen-dominated system where
 * phonon-mediated BCS pairing dominates and DFT+U is secondary.
 */
function isHydrogenDominated(elements: string[], counts: Record<string, number>): boolean {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const hCount = counts["H"] || 0;
  return (hCount / totalAtoms) > 0.6;
}

/**
 * Main entry point: analyze a material and produce a complete Hubbard U workflow.
 *
 * This replaces the ad-hoc DFT+U detection in qe-worker.ts with a systematic
 * approach that considers composition, oxidation state, and material class.
 */
export function analyzeHubbardWorkflow(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  correlationRegime: string,
  materialPatterns: string[],
): HubbardWorkflowResult {
  const notes: string[] = [];
  const sites: HubbardSiteConfig[] = [];
  let correlatedSiteCount = 0;

  // Check for material-specific U overrides first
  const materialOverrides: Record<string, { u: number; orbital: string; source: string }> = {};
  for (const entry of MATERIAL_SPECIFIC_U) {
    if (entry.pattern.test(formula)) {
      Object.assign(materialOverrides, entry.overrides);
      notes.push(`Material-specific U override matched: ${entry.pattern.source}`);
    }
  }

  // Build per-site Hubbard configuration
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    const needs = needsHubbardU(el);
    const orbital = getOrbitalManifold(el);

    if (!needs) {
      sites.push({
        element: el,
        speciesIndex: i + 1,
        uEffective: 0,
        hunds: 0,
        orbitalManifold: orbital,
        source: "none",
        needsU: false,
      });
      continue;
    }

    let uVal: number;
    let source: "material-specific" | "oxidation-aware" | "element-default";

    // Priority 1: material-specific override
    if (materialOverrides[el]) {
      uVal = materialOverrides[el].u;
      source = "material-specific";
      notes.push(`${el}: U=${uVal.toFixed(1)} eV (${materialOverrides[el].source})`);
    }
    // Priority 2: oxidation-state-aware
    else if (OXIDATION_AWARE_U[el]) {
      const isHighOx = estimateHighOxidation(elements, counts, el);
      const oxData = OXIDATION_AWARE_U[el];
      uVal = isHighOx ? oxData.highOx : oxData.lowOx;
      source = "oxidation-aware";
      notes.push(`${el}: U=${uVal.toFixed(1)} eV (${isHighOx ? "high" : "low"}-oxidation ${oxData.orbital})`);
    }
    // Priority 3: element default
    else {
      uVal = getHubbardU(el) ?? 0;
      source = "element-default";
      if (uVal > 0) {
        notes.push(`${el}: U=${uVal.toFixed(1)} eV (element default from ELEMENTAL_DATA)`);
      }
    }

    if (uVal > 0) correlatedSiteCount++;

    sites.push({
      element: el,
      speciesIndex: i + 1,
      uEffective: uVal,
      hunds: 0, // Dudarev simplified: J=0, U_eff = U - J
      orbitalManifold: orbital,
      source: uVal > 0 ? source : "none",
      needsU: uVal > 0,
    });
  }

  // Decision: should DFT+U be applied?
  const hasCorrelatedSites = correlatedSiteCount > 0;
  const isHDominated = isHydrogenDominated(elements, counts);

  // Broadened trigger: apply DFT+U whenever there are d/f electrons that need it,
  // not just for "strongly-correlated" / "Mott-proximate" regimes.
  // The key insight: even "moderately-correlated" materials like NbN, MoS2,
  // or LaFeAsO need +U for correct band ordering.
  const applyDFTplusU = hasCorrelatedSites && !isHDominated && (
    correlationRegime === "Mott-proximate" ||
    correlationRegime === "strongly-correlated" ||
    correlationRegime === "moderately-correlated" ||
    // Also apply for any material with significant d/f orbital character
    // even if the correlation score is low (e.g., metallic 4d/5d systems)
    sites.some(s => s.needsU && (s.orbitalManifold === "4f" || s.orbitalManifold === "5f")) ||
    sites.some(s => s.needsU && s.uEffective >= 3.0)
  );

  if (!applyDFTplusU && hasCorrelatedSites) {
    notes.push(
      `Correlated sites present but DFT+U not applied: ` +
      `${isHDominated ? "hydrogen-dominated (phonon BCS)" : `regime=${correlationRegime}, U values below threshold`}`
    );
  }

  // Determine whether to apply to vc-relax and phonons
  // For strongly/Mott: always apply to vc-relax (geometry depends on U)
  // For moderately: apply to vc-relax if any U >= 3.0 eV (significant effect)
  const applyToVCRelax = applyDFTplusU && (
    correlationRegime === "Mott-proximate" ||
    correlationRegime === "strongly-correlated" ||
    sites.some(s => s.uEffective >= 3.0)
  );

  // Apply to phonons whenever DFT+U is used in SCF
  // ph.x inherits the SCF density, but explicit lda_plus_u in the phonon
  // SCF ensures consistency
  const applyToPhonons = applyDFTplusU;

  if (applyToVCRelax) {
    notes.push(
      "DFT+U applied to vc-relax: structural relaxation with +U gives correct geometry " +
      "for localized-electron systems. Without this, relaxed structure is suboptimal."
    );
  }

  // Determine DFT+U formulation: Dudarev (kind=0) vs Liechtenstein (kind=1)
  const useLiechtenstein = applyDFTplusU && needsLiechtenstein(elements, materialPatterns);
  const hubbardKind: 0 | 1 = useLiechtenstein ? 1 : 0;

  // Assign Hund's J when using Liechtenstein
  if (useLiechtenstein) {
    for (const site of sites) {
      if (site.needsU && HUNDS_J[site.element] != null) {
        site.hunds = HUNDS_J[site.element];
      }
    }
    notes.push(
      `Liechtenstein (kind=1) formulation selected: Hund's coupling J is physically ` +
      `important for this system (orbital ordering / Hund's metal physics). ` +
      `Ref: Liechtenstein et al., PRB 52, R5467 (1995); Georges et al., ARCMP 4, 137 (2013).`
    );
  }

  // Build QE ≥7.1 HUBBARD card block (goes after ATOMIC_SPECIES, NOT in &SYSTEM).
  // Old syntax (lda_plus_u, Hubbard_U in &SYSTEM) was removed in QE 7.1.
  // New syntax: separate HUBBARD card with projector type + per-element U/J lines.
  //
  // Format:
  //   HUBBARD (ortho-atomic)
  //   U Cu-3d 5.0
  //   J Cu-3d 0.98
  let qeHubbardCard = "";
  const qeBlock = ""; // empty — no Hubbard params in &SYSTEM for QE ≥7.1
  if (applyDFTplusU) {
    // QE ≥7.1 recommends "ortho-atomic" universally — it's more accurate
    // than plain "atomic" because the projectors are properly orthogonalized
    // across overlapping orbitals (avoids over/undercounting near bonded
    // atoms). The old conditional `kind===1 ? "ortho-atomic" : "ortho-atomic"`
    // was dead code from a refactor; both formulations (Dudarev kind=0,
    // Liechtenstein kind=1) work with the orthogonalized projector.
    // See: QE 7.1 Release Notes; Mahajan et al., PRB 104, 134402 (2021).
    const projector = "ortho-atomic";
    qeHubbardCard += `HUBBARD (${projector})\n`;
    for (const site of sites) {
      if (site.uEffective > 0 && site.orbitalManifold !== "none") {
        qeHubbardCard += `U ${site.element}-${site.orbitalManifold} ${site.uEffective.toFixed(1)}\n`;
        if (hubbardKind === 1 && site.hunds > 0) {
          qeHubbardCard += `J ${site.element}-${site.orbitalManifold} ${site.hunds.toFixed(2)}\n`;
        }
      }
    }
    const kindLabel = hubbardKind === 0 ? "Dudarev" : "Liechtenstein";
    notes.push(
      `DFT+U enabled (${kindLabel}): ${correlatedSiteCount} correlated site(s), ` +
      `regime=${correlationRegime}. QE ≥7.1 HUBBARD card syntax. ` +
      `Ref: ${hubbardKind === 0 ? "Dudarev et al., PRB 57, 1505 (1998)" : "Liechtenstein et al., PRB 52, R5467 (1995)"}.`
    );
  }

  // Warnings
  const highUSites = sites.filter(s => s.uEffective > 6.0);
  if (highUSites.length > 0) {
    notes.push(
      `WARNING: Large U values (>6 eV) on ${highUSites.map(s => `${s.element}=${s.uEffective}`).join(", ")}. ` +
      `These may over-localize electrons. Consider self-consistent U via hp.x (ACBN0). ` +
      `Ref: Timrov et al., PRB 98, 085127 (2018).`
    );
  }

  const mixed4d5d = sites.filter(s => s.needsU && (s.orbitalManifold === "4d" || s.orbitalManifold === "5d"));
  if (mixed4d5d.length > 0 && !applyDFTplusU) {
    notes.push(
      `Note: 4d/5d elements (${mixed4d5d.map(s => s.element).join(", ")}) present but DFT+U skipped. ` +
      `These elements are more delocalized than 3d — PBE may be adequate for metallic compounds.`
    );
  }

  return {
    applyDFTplusU,
    sites,
    qeHubbardCard,
    qeSystemBlock: qeBlock, // empty for QE ≥7.1
    correlationRegime,
    materialPatterns,
    notes,
    correlatedSiteCount,
    applyToVCRelax,
    applyToPhonons,
    hubbardKind,
  };
}
