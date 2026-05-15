/**
 * Spin-Orbit Coupling (SOC) handler for heavy-element DFT calculations.
 *
 * For any candidate containing heavy elements (5d, 6p, lanthanides, actinides),
 * non-relativistic DFT gets band ordering wrong — SOC splits degenerate states,
 * reshapes the Fermi surface, and changes N(E_F). This propagates directly into
 * the electron-phonon coupling lambda and the Coulomb pseudopotential mu*.
 *
 * SOC effects by element group:
 *   - 5d transition metals (Hf-Hg): SOC ~ 0.3-1.5 eV splitting, inverts
 *     band ordering at high-symmetry points (e.g. Pt, Ir, Os)
 *   - 6p metals (Tl, Pb, Bi): SOC ~ 1-3 eV, critical for topological
 *     classification (Bi2Se3, Bi2Te3 are topological ONLY with SOC)
 *   - Lanthanides (La-Lu): 4f SOC ~ 0.1-0.5 eV, modest but 5d contribution
 *     matters for La-based hydrides (LaH10)
 *   - Actinides (Th, U): 5f SOC ~ 0.5-2.0 eV, large and essential
 *   - 4d metals (Nb, Mo, Ru, Rh, Pd): SOC ~ 0.05-0.3 eV, usually small
 *     but can matter for narrow-gap systems
 *
 * Implementation strategy:
 *   - Scalar-relativistic PPs handle the major relativistic effects (mass-velocity,
 *     Darwin term) without requiring noncolin/lspinorb in QE. These are the
 *     standard for all elements Z > 36.
 *   - Full SOC (noncolin=.true., lspinorb=.true.) is reserved for heavy elements
 *     where band-ordering errors exceed ~0.1 eV. This doubles the computational
 *     cost (spinor wavefunctions) but is essential for correct physics.
 *   - SOC-corrected N(E_F) feeds into the mu* and Tc calculations.
 *
 * QE requirements for SOC:
 *   - Fully-relativistic PPs (filename contains "rel-" or "_rel")
 *   - noncolin = .true. in &SYSTEM
 *   - lspinorb = .true. in &SYSTEM
 *   - nspin is ignored when noncolin=.true. (QE uses 4-component spinors)
 *   - starting_magnetization still accepted for initial spin direction
 *   - nbnd must be doubled (spinor basis)
 *
 * @see A. Dal Corso, Comp. Mat. Sci. 95, 337 (2014) — thermo_pw SOC
 * @see G. Kresse & D. Joubert, Phys. Rev. B 59, 1758 (1999) — PAW + SOC
 * @see A. H. MacDonald et al., J. Phys. F 10, 2005 (1980) — relativistic bands
 */

// ─── SOC classification ─────────────────────────────────────────────

/**
 * Elements where SOC is large enough to change band ordering at the Fermi level.
 * Approximate SOC energy scale in eV for the valence shell.
 */
export const SOC_ELEMENTS: Record<string, { socEV: number; shell: string; priority: "critical" | "recommended" | "optional" }> = {
  // 6p metals — SOC is critical, changes topology
  Tl: { socEV: 1.0, shell: "6p", priority: "critical" },
  Pb: { socEV: 1.3, shell: "6p", priority: "critical" },
  Bi: { socEV: 1.5, shell: "6p", priority: "critical" },
  Po: { socEV: 2.0, shell: "6p", priority: "critical" },

  // 5d transition metals — SOC reshapes Fermi surface
  Hf: { socEV: 0.3, shell: "5d", priority: "recommended" },
  Ta: { socEV: 0.4, shell: "5d", priority: "recommended" },
  W:  { socEV: 0.5, shell: "5d", priority: "recommended" },
  Re: { socEV: 0.6, shell: "5d", priority: "recommended" },
  Os: { socEV: 0.7, shell: "5d", priority: "recommended" },
  Ir: { socEV: 0.9, shell: "5d", priority: "recommended" },
  Pt: { socEV: 1.0, shell: "5d", priority: "recommended" },
  Au: { socEV: 0.6, shell: "5d", priority: "recommended" },
  Hg: { socEV: 0.5, shell: "5d", priority: "recommended" },

  // Actinides — very large 5f SOC (1-2 eV across the series). Pa onward
  // have partially-filled 5f and need full SOC + DFT+U for correct physics;
  // Th (5f^0) is borderline but the 6d/7s mixing has substantial SOC too.
  Th: { socEV: 0.8, shell: "5f/6d", priority: "critical" },
  Pa: { socEV: 1.0, shell: "5f", priority: "critical" },
  U:  { socEV: 1.2, shell: "5f", priority: "critical" },
  Np: { socEV: 1.4, shell: "5f", priority: "critical" },
  Pu: { socEV: 1.5, shell: "5f", priority: "critical" },
  Am: { socEV: 1.6, shell: "5f", priority: "critical" },
  Cm: { socEV: 1.7, shell: "5f", priority: "critical" },

  // Lanthanides — differentiated by 4f occupation:
  //   La (4f^0, [Xe]5d^1 6s^2): empty 4f, SOC on conduction 5d is small → scalar-rel fine
  La: { socEV: 0.15, shell: "5d", priority: "optional" },
  //   Ce (4f^1): single 4f electron ON the Fermi level, SOC + crystal-field competition
  //   is critical → needs full SOC (noncolin + lspinorb) + DFT+U on f-states
  Ce: { socEV: 0.30, shell: "4f", priority: "critical" },
  //   Pr-Sm (4f^2 – 4f^5): partially filled 4f with increasing SOC → full SOC + U
  Pr: { socEV: 0.25, shell: "4f", priority: "critical" },
  Nd: { socEV: 0.28, shell: "4f", priority: "critical" },
  Pm: { socEV: 0.30, shell: "4f", priority: "critical" },
  Sm: { socEV: 0.32, shell: "4f", priority: "critical" },
  //   Eu-Gd (4f^7, half-filled): strong exchange splitting, SOC secondary
  Eu: { socEV: 0.18, shell: "4f", priority: "recommended" },
  Gd: { socEV: 0.22, shell: "4f", priority: "recommended" },
  //   Tb-Tm (4f^8 – 4f^12): SOC increases with occupation, heavy ones need full SOC + U
  Tb: { socEV: 0.30, shell: "4f", priority: "recommended" },
  Dy: { socEV: 0.35, shell: "4f", priority: "critical" },
  Ho: { socEV: 0.38, shell: "4f", priority: "critical" },
  Er: { socEV: 0.40, shell: "4f", priority: "critical" },
  Tm: { socEV: 0.42, shell: "4f", priority: "critical" },
  //   Yb-Lu (4f^13-14): 4f below Fermi level, filled/near-full → scalar-rel usually fine
  Yb: { socEV: 0.15, shell: "4f", priority: "optional" },
  Lu: { socEV: 0.12, shell: "5d", priority: "optional" },

  // Heavy s-block — minor SOC but flags relativistic PP need
  Cs: { socEV: 0.05, shell: "6s", priority: "optional" },
  Ba: { socEV: 0.08, shell: "6s", priority: "optional" },

  // 4d metals — small SOC, include for completeness
  Nb: { socEV: 0.05, shell: "4d", priority: "optional" },
  Mo: { socEV: 0.08, shell: "4d", priority: "optional" },
  Tc: { socEV: 0.10, shell: "4d", priority: "optional" },
  Ru: { socEV: 0.12, shell: "4d", priority: "optional" },
  Rh: { socEV: 0.14, shell: "4d", priority: "optional" },
  Pd: { socEV: 0.15, shell: "4d", priority: "optional" },
  Ag: { socEV: 0.10, shell: "4d", priority: "optional" },
  Cd: { socEV: 0.08, shell: "4d", priority: "optional" },
  In: { socEV: 0.10, shell: "5p", priority: "optional" },
  Sn: { socEV: 0.15, shell: "5p", priority: "optional" },
  Sb: { socEV: 0.20, shell: "5p", priority: "optional" },
  Te: { socEV: 0.25, shell: "5p", priority: "optional" },
  I:  { socEV: 0.30, shell: "5p", priority: "optional" },
};

/**
 * Fully-relativistic pseudopotential URLs from PSLibrary.
 * These include SOC in the PP generation (rel- prefix).
 * Required when lspinorb=.true. is set in QE.
 *
 * Only populated for elements where SOC priority is "critical" or "recommended".
 * For "optional" elements, the standard scalar-relativistic PP is sufficient.
 */
const GH_BASE = "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS";

export const RELATIVISTIC_PP_URLS: Record<string, string> = {
  // 6p — critical SOC
  Tl: `${GH_BASE}/Tl.pbe-dn-kjpaw_psl.1.0.0.UPF`,   // rel PP when available
  Pb: `${GH_BASE}/Pb.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Bi: `${GH_BASE}/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  // 5d — recommended SOC
  Hf: `${GH_BASE}/Hf.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ta: `${GH_BASE}/Ta.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  W:  `${GH_BASE}/W.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Re: `${GH_BASE}/Re.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Os: `${GH_BASE}/Os.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ir: `${GH_BASE}/Ir.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Pt: `${GH_BASE}/Pt.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Au: `${GH_BASE}/Au.pbe-nd-kjpaw_psl.1.0.0.UPF`,
  Hg: `${GH_BASE}/Hg.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  // Actinides — critical SOC
  Th: `${GH_BASE}/Th.pbe-spfn-kjpaw_psl.1.0.0.UPF`,
  U:  `${GH_BASE}/U.pbe-spfn-kjpaw_psl.1.0.0.UPF`,
  // Lanthanides — differentiated by 4f occupation
  La: `${GH_BASE}/La.pbe-spfn-kjpaw_psl.1.0.0.UPF`,
  Ce: `${GH_BASE}/Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Pr: `${GH_BASE}/Pr.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Nd: `${GH_BASE}/Nd.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Sm: `${GH_BASE}/Sm.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Eu: `${GH_BASE}/Eu.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Gd: `${GH_BASE}/Gd.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Tb: `${GH_BASE}/Tb.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Dy: `${GH_BASE}/Dy.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Ho: `${GH_BASE}/Ho.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Er: `${GH_BASE}/Er.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Tm: `${GH_BASE}/Tm.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
};

// ─── SOC analysis result ─────────────────────────────────────────────

export interface SOCAnalysis {
  /** Whether this material needs SOC for correct physics */
  needsSOC: boolean;
  /** Whether full non-collinear SOC should be enabled */
  enableFullSOC: boolean;
  /** Whether scalar-relativistic PP is sufficient (no lspinorb needed) */
  scalarRelSufficient: boolean;
  /** Maximum SOC energy scale in the material (eV) */
  maxSOCEnergy: number;
  /** Elements triggering SOC */
  socElements: Array<{ element: string; socEV: number; priority: string }>;
  /** Estimated impact on N(E_F): fractional change */
  estimatedDOSImpact: number;
  /** Estimated Tc impact from SOC-corrected N(E_F) */
  estimatedTcImpactK: number;
  /** QE flags to add to &SYSTEM */
  qeSystemFlags: string;
  /** Whether starting_magnetization should use angle format (theta, phi) */
  useAngleMagnetization: boolean;
  /** Whether DFT+U is needed for 4f states (lanthanide SOC) */
  needsDFTplusU: boolean;
  /** Elements requiring Hubbard U correction */
  hubbardUElements: string[];
  /** Cost multiplier for SOC calculation (typically 2-4x) */
  costMultiplier: number;
  /** Human-readable notes */
  notes: string[];
}

/**
 * Analyze whether a material needs spin-orbit coupling treatment.
 *
 * Decision tree:
 *   1. Any 6p element (Tl, Pb, Bi) or actinide → full SOC (critical)
 *   2. Any 5d element with SOC > 0.3 eV → full SOC (recommended)
 *   3. Multiple heavy elements → full SOC (cumulative effect)
 *   4. Lanthanides — differentiated by 4f occupation:
 *      a. La (empty 4f): scalar-relativistic sufficient
 *      b. Ce (4f^1 on Fermi level): full SOC + DFT+U critical
 *      c. Pr-Sm (4f^2-5): full SOC + DFT+U critical
 *      d. Eu-Gd (half-filled 4f): SOC recommended, exchange splitting dominates
 *      e. Tb-Tm (4f^8-12): full SOC + U for heavy ones (Dy-Tm critical)
 *      f. Yb-Lu (full/near-full 4f): scalar-relativistic sufficient
 *   5. Only 4d/5p elements → scalar-relativistic sufficient
 */
export function analyzeSOCRequirement(
  elements: string[],
  counts: Record<string, number>,
  pressureGpa: number = 0,
): SOCAnalysis {
  const socElements: Array<{ element: string; socEV: number; priority: string }> = [];
  let maxSOC = 0;
  let hasCritical = false;
  let hasRecommended = false;
  let totalSOCWeight = 0;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  for (const el of elements) {
    const soc = SOC_ELEMENTS[el];
    if (soc) {
      socElements.push({ element: el, socEV: soc.socEV, priority: soc.priority });
      maxSOC = Math.max(maxSOC, soc.socEV);
      if (soc.priority === "critical") hasCritical = true;
      if (soc.priority === "recommended") hasRecommended = true;
      // Weight by fraction in composition
      const frac = (counts[el] || 1) / totalAtoms;
      totalSOCWeight += soc.socEV * frac;
    }
  }

  const notes: string[] = [];

  if (socElements.length === 0) {
    return {
      needsSOC: false,
      enableFullSOC: false,
      scalarRelSufficient: true,
      maxSOCEnergy: 0,
      socElements: [],
      estimatedDOSImpact: 0,
      estimatedTcImpactK: 0,
      qeSystemFlags: "",
      useAngleMagnetization: false,
      needsDFTplusU: false,
      hubbardUElements: [],
      costMultiplier: 1.0,
      notes: ["No heavy elements — SOC negligible"],
    };
  }

  // Lanthanide-specific: flag 4f elements that need DFT+U alongside SOC.
  // Tb (4f⁹) is past half-filled, so it belongs with the SOC+U group (same
  // regime as Dy 4f¹⁰), NOT with the half-filled exchange-dominated set.
  // Only Eu (4f⁷) and Gd (4f⁷) are truly half-filled.
  const LANTHANIDE_4F_SOC = new Set(["Ce", "Pr", "Nd", "Pm", "Sm", "Tb", "Dy", "Ho", "Er", "Tm"]);
  const LANTHANIDE_HALF_FILLED = new Set(["Eu", "Gd"]);
  const lanthanidesWith4fSOC = elements.filter(el => LANTHANIDE_4F_SOC.has(el));
  const lanthanidesHalfFilled = elements.filter(el => LANTHANIDE_HALF_FILLED.has(el));

  // Actinide analog: partially-filled 5f elements need DFT+U just like the
  // 4f lanthanides. Without this, U/Pu/Np-containing heavy-fermion candidates
  // run with PBE-only 5f states — wrong localization, wrong band structure,
  // wrong Tc. Th (5f⁰) is empty-f and skipped; Am/Cm (5f⁷ half-filled) get
  // the same DFT+U treatment as Eu/Gd (where exchange dominates but U still
  // helps fix the localized-vs-itinerant 5f balance).
  const ACTINIDE_5F_SOC = new Set(["Pa", "U", "Np", "Pu", "Bk", "Cf"]);
  const ACTINIDE_HALF_FILLED = new Set(["Am", "Cm"]);
  const actinidesWith5fSOC = elements.filter(el => ACTINIDE_5F_SOC.has(el));
  const actinidesHalfFilled = elements.filter(el => ACTINIDE_HALF_FILLED.has(el));

  // Decision: full SOC vs scalar-relativistic
  const enableFullSOC = hasCritical || (hasRecommended && socElements.length >= 2) || totalSOCWeight > 0.3;
  const scalarRelSufficient = !enableFullSOC;

  // Estimate DOS impact
  // SOC splits degenerate bands, redistributing spectral weight near E_F.
  // For Bi/Pb, this can change N(E_F) by 10-30%.
  // For 5d metals, typically 5-15%.
  // For lanthanides, typically 2-8%.
  let dosImpact = 0;
  if (hasCritical) {
    dosImpact = 0.15 + totalSOCWeight * 0.1;
  } else if (hasRecommended) {
    dosImpact = 0.08 + totalSOCWeight * 0.05;
  } else {
    dosImpact = 0.03 + totalSOCWeight * 0.02;
  }
  dosImpact = Math.min(0.35, dosImpact);

  // Estimate Tc impact: d(Tc) ~ Tc * d(N_EF)/N_EF * sensitivity
  // For Allen-Dynes, Tc is roughly proportional to exp(-1/lambda),
  // and lambda ~ N(E_F) * <I^2>/M*omega^2, so a 15% change in N(E_F)
  // gives roughly a 15% change in lambda, which for strong coupling
  // (lambda~2) changes Tc by ~10-20 K.
  const tcImpact = dosImpact * 80; // rough: 80 K per unit dosImpact

  // Build QE flags
  let qeFlags = "";
  if (enableFullSOC) {
    qeFlags = "  noncolin = .true.,\n  lspinorb = .true.,\n";
    notes.push(
      `Full SOC enabled: max SOC energy ${maxSOC.toFixed(2)} eV ` +
      `(${socElements.map(e => `${e.element}:${e.socEV}eV`).join(", ")}). ` +
      `Non-collinear spinor wavefunctions will be used (2x cost). ` +
      `Ref: Dal Corso, CMS 95, 337 (2014).`
    );
  } else if (socElements.length > 0) {
    notes.push(
      `Scalar-relativistic treatment sufficient: SOC elements present ` +
      `(${socElements.map(e => `${e.element}:${e.socEV}eV`).join(", ")}) ` +
      `but all below full-SOC threshold. Standard PPs include scalar-relativistic effects.`
    );
  }

  // [SOC] Lanthanide 4f-specific notes
  if (lanthanidesWith4fSOC.length > 0) {
    notes.push(
      `[SOC] Lanthanide 4f SOC+U required for: ${lanthanidesWith4fSOC.join(", ")}. ` +
      `4f electrons are at/near Fermi level — SOC + crystal-field competition is critical. ` +
      `Add Hubbard U on f-states via QE ≥7.1 HUBBARD card (U ~ 4-6 eV for 4f).`
    );
  }
  if (lanthanidesHalfFilled.length > 0) {
    notes.push(
      `[SOC] Half-filled 4f lanthanides: ${lanthanidesHalfFilled.join(", ")}. ` +
      `Exchange splitting dominates over SOC — full SOC recommended but not critical. ` +
      `DFT+U still advised for correct 4f positioning.`
    );
  }

  // [SOC] Actinide 5f-specific notes (analog of the 4f lanthanide treatment).
  // Without these, U/Pu/Np-containing heavy-fermion candidates ran with no
  // Hubbard U on the 5f manifold — wrong 5f localization, wrong Tc.
  if (actinidesWith5fSOC.length > 0) {
    notes.push(
      `[SOC] Actinide 5f SOC+U required for: ${actinidesWith5fSOC.join(", ")}. ` +
      `5f electrons are partially filled and strongly correlated — SOC + Hubbard U ` +
      `are both essential. Add U on 5f via QE ≥7.1 HUBBARD card (U ~ 3-5 eV for 5f).`
    );
  }
  if (actinidesHalfFilled.length > 0) {
    notes.push(
      `[SOC] Half-filled 5f actinides: ${actinidesHalfFilled.join(", ")}. ` +
      `Hund's exchange splitting dominates (~7-10 eV between majority/minority spin) ` +
      `but Hubbard U still needed to fix localized-vs-itinerant 5f balance.`
    );
  }

  // [SOC] When full SOC is enabled, downstream magnetic search must use non-collinear mode
  if (enableFullSOC) {
    notes.push(
      `[SOC] Full SOC enabled → magnetic ground-state search MUST use non-collinear mode ` +
      `(noncolin=.true.). Collinear nspin=2 contradicts the spinor basis. ` +
      `Use angle1/angle2 format for starting_magnetization.`
    );
  }

  if (dosImpact > 0.1) {
    notes.push(
      `SOC estimated to change N(E_F) by ~${(dosImpact * 100).toFixed(0)}%, ` +
      `which could shift Tc by ~${tcImpact.toFixed(0)} K. ` +
      `Non-relativistic band ordering may be qualitatively wrong for this composition.`
    );
  }

  // Pressure note: under extreme pressure, bandwidth widens and
  // SOC effects become relatively less important
  if (pressureGpa > 200 && !hasCritical) {
    notes.push(
      `At ${pressureGpa} GPa, bandwidth broadening reduces relative SOC importance. ` +
      `Scalar-relativistic may be adequate even for 5d elements.`
    );
  }

  // Cost multiplier: full SOC roughly doubles SCF cost (spinor basis)
  // plus additional memory for 2x bands
  const costMultiplier = enableFullSOC ? 2.5 : 1.0;

  return {
    needsSOC: socElements.length > 0,
    enableFullSOC,
    scalarRelSufficient,
    maxSOCEnergy: maxSOC,
    socElements,
    estimatedDOSImpact: Number(dosImpact.toFixed(4)),
    estimatedTcImpactK: Number(tcImpact.toFixed(1)),
    qeSystemFlags: qeFlags,
    useAngleMagnetization: enableFullSOC,
    needsDFTplusU: lanthanidesWith4fSOC.length > 0 || lanthanidesHalfFilled.length > 0
                || actinidesWith5fSOC.length > 0 || actinidesHalfFilled.length > 0,
    hubbardUElements: [
      ...lanthanidesWith4fSOC, ...lanthanidesHalfFilled,
      ...actinidesWith5fSOC, ...actinidesHalfFilled,
    ],
    costMultiplier,
    notes,
  };
}

/**
 * Generate SOC-aware magnetization lines for non-collinear calculations.
 *
 * When noncolin=.true., QE expects angle1(i) and angle2(i) instead of
 * starting_magnetization(i). These define the initial spin direction:
 *   angle1 = polar angle (0 = +z, 180 = -z)
 *   angle2 = azimuthal angle
 *
 * For FM: all atoms angle1=0
 * For AFM: alternate angle1=0 and angle1=180
 */
export function generateNoncollinearMagLines(
  elements: string[],
  magneticElements: Record<string, number>,
  isAFM: boolean,
): string {
  let lines = "";
  let magIdx = 0;

  for (let i = 0; i < elements.length; i++) {
    const el = elements[i];
    const mag = magneticElements[el];
    if (mag !== undefined && mag > 0) {
      const angle1 = (isAFM && magIdx % 2 === 1) ? 180.0 : 0.0;
      lines += `  starting_magnetization(${i + 1}) = ${mag.toFixed(1)},\n`;
      lines += `  angle1(${i + 1}) = ${angle1.toFixed(1)},\n`;
      lines += `  angle2(${i + 1}) = 0.0,\n`;
      magIdx++;
    } else {
      lines += `  starting_magnetization(${i + 1}) = 0.1,\n`;
      lines += `  angle1(${i + 1}) = 0.0,\n`;
      lines += `  angle2(${i + 1}) = 0.0,\n`;
    }
  }
  return lines;
}
