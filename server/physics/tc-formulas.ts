const CM1_TO_KELVIN = 1.4388;
const MEV_TO_KELVIN = 11.604;

export interface AllenDynesTcInput {
  lambda: number;
  omegaLog: number;
  muStar: number;
  omega2Avg?: number;
  isHydride?: boolean;
  family?: string;
}

export type TcMethod =
  | "allen-dynes"
  | "sisso-hydride"
  | "xie-strong-coupling"
  | "spin-fluctuation-cuprate"
  | "qcp-heavy-fermion";

export interface AllenDynesTcResult {
  tc: number;
  f1: number;
  f2: number;
  regime: "weak" | "intermediate" | "strong" | "very-strong";
  method: TcMethod;
  /** Set when Allen-Dynes phonon assumptions do not apply to the material family */
  applicabilityWarning?: string;
}

// ---------------------------------------------------------------------------
// Superconductor family classification gate
// ---------------------------------------------------------------------------

/**
 * Gate result classifying a material family as conventional BCS (phonon-mediated)
 * or unconventional. Determines which Tc estimation approach is physically valid.
 *
 * Conventional BCS: Allen-Dynes (1975) / McMillan (1968) formulas are applicable.
 * Unconventional: spin-fluctuation, d-wave, or f-electron pairing requires suppression
 * of phonon-based Tc or a switch to empirical scaling relations.
 *
 * @see W. L. McMillan, Phys. Rev. 167, 331 (1968)
 * @see P. B. Allen & R. C. Dynes, Phys. Rev. B 12, 905 (1975)
 */
export interface SCFamilyGate {
  family: string;
  /** true if phonon-mediated BCS: Allen-Dynes / McMillan formulas are valid */
  isBCS: boolean;
  /** Human-readable dominant pairing mechanism */
  mechanism: string;
  /** Preferred Tc estimation strategy for this family */
  tcApproach: "allen-dynes" | "empirical-scaling" | "suppress";
  /**
   * Physics reason why Allen-Dynes is invalid.
   * Only set when tcApproach === "suppress".
   */
  suppressionNote?: string;
}

/**
 * Classifies a material family as conventional BCS or unconventional and
 * returns the appropriate Tc estimation strategy.
 *
 * - "allen-dynes"      : phonon-mediated BCS; Allen-Dynes (1975) valid.
 * - "empirical-scaling": partially unconventional; Allen-Dynes gives wrong
 *                        physics but empirical Tc ranges are available.
 * - "suppress"         : fundamentally unconventional; Allen-Dynes result
 *                        must be discarded and flagged with uncertainty.
 *
 * @see W. L. McMillan, Phys. Rev. 167, 331 (1968)
 * @see P. B. Allen & R. C. Dynes, Phys. Rev. B 12, 905 (1975)
 * @see P. W. Anderson, Science 235, 1196 (1987) — RVB theory of cuprates
 * @see D. J. Scalapino, Rev. Mod. Phys. 84, 1383 (2012) — d-wave pairing
 * @see C. Pfleiderer, Rev. Mod. Phys. 81, 1551 (2009) — heavy-fermion SC
 * @see D. Li et al., Nature 572, 624 (2019) — infinite-layer nickelates
 */
export function classifySCMechanism(family: string): SCFamilyGate {
  switch (family) {
    // --- Definite unconventional: Allen-Dynes is physically invalid ---

    case "Cuprates":
      return {
        family,
        isBCS: false,
        mechanism: "d-wave spin-fluctuation (cuprate)",
        tcApproach: "suppress",
        suppressionNote:
          "Cuprate superconductors are unconventional d-wave materials driven by spin fluctuations, " +
          "not phonon exchange. Allen-Dynes (1975) / McMillan (1968) equations are not applicable. " +
          "Ref: Anderson, Science 235, 1196 (1987); Scalapino, Rev. Mod. Phys. 84, 1383 (2012).",
      };

    case "Heavy Fermions":
      return {
        family,
        isBCS: false,
        mechanism: "f-electron spin / multipolar fluctuations (heavy fermion)",
        tcApproach: "suppress",
        suppressionNote:
          "Heavy-fermion superconductors are unconventional, mediated by f-electron spin or " +
          "multipolar fluctuations with Tc typically < 3 K. Phonon Allen-Dynes is inapplicable. " +
          "Ref: Steglich et al., PRL 43, 1892 (1979); Pfleiderer, Rev. Mod. Phys. 81, 1551 (2009).",
      };

    case "Nickelates":
      return {
        family,
        isBCS: false,
        mechanism: "correlated d-wave (infinite-layer nickelate)",
        tcApproach: "suppress",
        suppressionNote:
          "Infinite-layer nickelate superconductors are strongly correlated unconventional materials " +
          "(Tc ~ 9–80 K). Phonon-mediated Tc formulas are inapplicable. " +
          "Ref: Li et al., Nature 572, 624 (2019); Zeng et al., Science 369, 1604 (2020).",
      };

    // --- Partially unconventional: use empirical scaling, flag uncertainty ---

    case "Pnictides":
    case "Layered-pnictide":
      return {
        family,
        isBCS: false,
        mechanism: "s±-wave spin-fluctuation (iron pnictide)",
        tcApproach: "empirical-scaling",
      };

    case "Chalcogenides":
    case "Layered-chalcogenide":
      return {
        family,
        isBCS: false,
        mechanism: "mixed: CDW competition + phonon/spin fluctuation (chalcogenide)",
        tcApproach: "empirical-scaling",
      };

    case "Mixed-mechanism":
      return {
        family,
        isBCS: false,
        mechanism: "mixed (unresolved pairing channel)",
        tcApproach: "empirical-scaling",
      };

    case "Kagome":
      return {
        family,
        isBCS: false,
        mechanism: "kagome flat-band + CDW + SC (mechanism debated)",
        tcApproach: "empirical-scaling",
      };

    // --- Conventional BCS: Allen-Dynes / McMillan valid ---

    default:
      return {
        family,
        isBCS: true,
        mechanism: "phonon-mediated (BCS conventional)",
        tcApproach: "allen-dynes",
      };
  }
}

// ---------------------------------------------------------------------------
// Empirical Tc scaling relations for unconventional families
// ---------------------------------------------------------------------------

/**
 * Empirical Tc estimate for iron-based pnictide superconductors based on
 * structural family identification. Allen-Dynes is not physically valid here
 * (s±-wave, spin-fluctuation mediated), so empirical Tc_max ranges are used.
 *
 * Structural family types and references:
 *  - 1111-type (REFeAsO:F): Tc_max = 26–55 K depending on rare-earth.
 *    Ref: Kamihara et al., JACS 130, 3296 (2008); Ren et al., CPL 25, 2215 (2008).
 *  - 122-type (AEFe2As2): Tc_max ~ 38 K at optimal doping.
 *    Ref: Rotter et al., PRL 101, 107006 (2008).
 *  - 11-type (FeSe/FeS): Tc ~ 8.5 K ambient, ~37 K at 1 GPa.
 *    Ref: Hsu et al., PNAS 105, 14262 (2008).
 *  - 111-type (LiFeAs/NaFeAs): Tc ~ 18 K.
 *    Ref: Wang et al., Solid State Commun. 148, 538 (2008).
 *
 * @see J. Paglione & R. L. Greene, Nature Physics 6, 645 (2010) — review
 * @see D. C. Johnston, Adv. Phys. 59, 803 (2010) — comprehensive review
 */
export function empiricalTcPnictide(
  formula: string,
): { tc: number; range: [number, number]; note: string } | null {
  // 1111-type: LaFeAsO, NdFeAsO, SmFeAsO, etc.
  if (/^(La|Ce|Pr|Nd|Sm|Gd|Eu|Dy|Tb|Ho|Er|Y)Fe(As|P)O/i.test(formula)) {
    const reMap: Record<string, number> = {
      La: 26, Ce: 41, Pr: 52, Nd: 52, Sm: 55, Gd: 36, Eu: 35,
    };
    const match = formula.match(/^(La|Ce|Pr|Nd|Sm|Gd|Eu)/i);
    const tcMax = match ? (reMap[match[1]] ?? 40) : 40;
    return {
      tc: tcMax,
      range: [Math.round(tcMax * 0.6), tcMax],
      note: `1111-type REFeAsO: Tc_max ~ ${tcMax} K (Ren et al., CPL 25, 2215 (2008); Kamihara et al., JACS 130, 3296 (2008))`,
    };
  }
  // 122-type: BaFe2As2, SrFe2As2, CaFe2As2
  if (/^(Ba|Sr|Ca|Eu)Fe2(As|P)2/i.test(formula)) {
    return {
      tc: 38,
      range: [15, 38],
      note: "122-type AEFe2As2: Tc_max ~ 38 K at optimal doping (Rotter et al., PRL 101, 107006 (2008))",
    };
  }
  // 11-type: FeSe, FeS, FeTe
  if (/^Fe(Se|S|Te)$/i.test(formula)) {
    const tc = formula.includes("Se") ? 8.5 : formula.includes("S") ? 4.5 : 1.5;
    return {
      tc,
      range: [tc * 0.5, tc * 4],
      note: `11-type FeCh: Tc ~ ${tc} K ambient (Hsu et al., PNAS 105, 14262 (2008)); pressure/strain can enhance to ~37–65 K`,
    };
  }
  // 111-type: LiFeAs, NaFeAs
  if (/^(Li|Na)FeAs/i.test(formula)) {
    return {
      tc: 18,
      range: [12, 25],
      note: "111-type LiFeAs: Tc ~ 18 K (Wang et al., Solid State Commun. 148, 538 (2008))",
    };
  }
  // Generic pnictide fallback
  return {
    tc: 25,
    range: [5, 55],
    note: "Generic iron pnictide: empirical Tc ~ 5–55 K (spin-fluctuation mediated, s±-wave; Paglione & Greene, Nat. Phys. 6, 645 (2010))",
  };
}

/**
 * Empirical Tc estimate for transition-metal dichalcogenide (TMD) superconductors.
 * These materials often compete with charge-density-wave (CDW) order and are
 * highly sensitive to doping, intercalation, and pressure.
 *
 * @see J. A. Wilson & A. D. Yoffe, Adv. Phys. 18, 193 (1969) — TMD review
 * @see E. Morosan et al., Nature Physics 2, 544 (2006) — CuxTiSe2
 * @see R. Ang et al., Phys. Rev. Lett. 109, 176403 (2012) — NbSe2
 */
export function empiricalTcChalcogenide(
  formula: string,
): { tc: number; range: [number, number]; note: string } | null {
  type TcEntry = { tc: number; range: [number, number]; ref: string };
  const tcMap: Record<string, TcEntry> = {
    NbSe2: {
      tc: 7.2, range: [6.0, 7.2],
      ref: "NbSe2: Tc = 7.2 K (Revolinsky et al., J. Phys. Chem. Solids 26, 1029 (1965)); CDW at 33 K",
    },
    NbS2: {
      tc: 6.0, range: [4.0, 6.0],
      ref: "2H-NbS2: Tc ~ 6 K (no CDW competition, unlike NbSe2)",
    },
    TaS2: {
      tc: 0.8, range: [0.5, 4.5],
      ref: "2H-TaS2: Tc ~ 0.8 K bulk; 1T polytype up to ~4.5 K under pressure (Di Salvo et al., PRB 14, 4321 (1976))",
    },
    TaSe2: {
      tc: 0.15, range: [0.1, 8.0],
      ref: "2H-TaSe2: Tc ~ 0.15 K; intercalated variants up to ~8 K",
    },
    MoS2: {
      tc: 0.5, range: [0.1, 10.0],
      ref: "MoS2: Tc < 1 K bulk; up to ~10 K under high pressure (Shi et al., PRB 94, 214503 (2016))",
    },
    WS2: {
      tc: 0.5, range: [0.1, 8.0],
      ref: "WS2: Tc < 1 K bulk; pressure-induced SC up to ~8 K",
    },
    FeSe: {
      tc: 8.5, range: [4.0, 65.0],
      ref: "FeSe: Tc = 8.5 K bulk; ~37 K at 1 GPa; ~65 K monolayer on STO (He et al., Nat. Mater. 12, 605 (2013))",
    },
    FeS: {
      tc: 4.5, range: [3.0, 6.0],
      ref: "FeS: Tc ~ 4.5 K (Lai et al., JACS 137, 10148 (2015))",
    },
    TiSe2: {
      tc: 1.8, range: [0.5, 4.0],
      ref: "CuxTiSe2: Tc up to ~4 K at optimal Cu doping (Morosan et al., Nat. Phys. 2, 544 (2006))",
    },
  };

  const clean = formula.replace(/[₀-₉]/g, (c) =>
    String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)),
  );
  const entry = tcMap[clean];
  if (entry) return { tc: entry.tc, range: entry.range, note: entry.ref };

  // Sub-family fallbacks
  if (/^(Nb|Ta)(Se|S)2$/i.test(clean)) {
    return { tc: 5, range: [0.5, 8], note: "Nb/Ta dichalcogenide: Tc ~ 0.5–8 K (CDW competition present)" };
  }
  if (/^(Mo|W)(Se|S|Te)2$/i.test(clean)) {
    return { tc: 1, range: [0.1, 10], note: "Mo/W dichalcogenide: Tc < 1 K ambient; pressure/doping enhanced" };
  }
  return null;
}

// ---------------------------------------------------------------------------
// McMillan (1968) formula — original phonon-mediated Tc
// ---------------------------------------------------------------------------

/**
 * McMillan (1968) formula for phonon-mediated superconducting Tc.
 * This is the precursor to Allen-Dynes; valid for weak-to-intermediate coupling.
 * Allen-Dynes (1975) adds strong-coupling correction factors f1, f2.
 *
 * Tc = (θ_D / 1.45) * exp( -1.04(1+λ) / (λ - μ*(1 + 0.62λ)) )
 *
 * @param thetaD  Debye temperature in Kelvin
 * @param lambda  electron-phonon coupling constant
 * @param muStar  Coulomb pseudopotential (typically 0.10–0.15)
 * @returns       Tc in Kelvin
 *
 * @see W. L. McMillan, Phys. Rev. 167, 331 (1968)
 */
export function mcMillanTc(
  thetaD: number,
  lambda: number,
  muStar: number,
): number {
  if (thetaD <= 0 || lambda <= 0) return 0;
  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (denominator <= 0 || Math.abs(denominator) < 1e-6) return 0;
  const exponent = -1.04 * (1 + lambda) / denominator;
  if (exponent < -50) return 0;
  const tc = (thetaD / 1.45) * Math.exp(exponent);
  return Number.isFinite(tc) ? Math.max(0, Math.min(500, tc)) : 0;
}

// ---------------------------------------------------------------------------
// lambdaBarPrefactor helper (internal)
// ---------------------------------------------------------------------------

/**
 * λ̄ prefactor for the Allen-Dynes f1 strong-coupling correction.
 * Only used for families where the Allen-Dynes phonon framework applies:
 * hydrides (enhanced by high-frequency H modes) and iron-based materials
 * (empirical fit to DFT data). Cuprates and heavy fermions are routed to
 * dedicated non-phononic formulas before this function is called.
 */
function lambdaBarPrefactor(family?: string): number {
  switch (family) {
    case "Hydride": return 2.89;
    case "Iron-Based": return 2.10;
    default: return 2.46;
  }
}

// ---------------------------------------------------------------------------
// Unconventional superconductor Tc formulas
// ---------------------------------------------------------------------------

/**
 * Monthoux-Scalapino d-wave spin-fluctuation Tc for cuprate superconductors.
 *
 * Cuprates are unconventional: pairing is d_{x²-y²}-wave, mediated by
 * antiferromagnetic spin fluctuations, NOT phonons. The Allen-Dynes formula
 * is physically inapplicable for three reasons:
 *   1. The relevant energy scale is the spin-fluctuation energy ω_sf
 *      (~40–80 meV = ~460–930 K), not the phonon ω_log.
 *   2. The effective coupling is the spin-fluctuation constant λ_sf ≈ 1.4 λ_ep,
 *      not the electron-phonon λ.
 *   3. d_{x²-y²} symmetry cancels ≈ 90% of the isotropic Coulomb repulsion,
 *      so μ*_eff ≈ 0.10 × μ*, not the BCS μ* value.
 *
 * Calibration against known cuprates (all at optimal doping):
 *   La₁.₈₅Sr₀.₁₅CuO₄ (LSCO):   Tc_max ≈ 38 K  → λ ≈ 0.55, ω_log ≈ 220 cm⁻¹
 *   YBa₂Cu₃O₇ (YBCO):           Tc_max ≈ 92 K  → λ ≈ 0.90, ω_log ≈ 280 cm⁻¹
 *   Bi₂Sr₂CaCu₂O₈ (Bi-2212):    Tc_max ≈ 95 K  → λ ≈ 0.95, ω_log ≈ 290 cm⁻¹
 *   HgBa₂CuO₄ (Hg-1201):        Tc_max ≈ 135 K → λ ≈ 1.30, ω_log ≈ 350 cm⁻¹
 *
 * @see P. Monthoux & D. J. Scalapino, PRL 72, 1874 (1994)
 * @see A. J. Millis, H. Monien & D. Pines, PRB 42, 167 (1990)
 * @see D. J. Scalapino, Rev. Mod. Phys. 84, 1383 (2012)
 */
function cuprateSpinFluctuationTc(
  lambda: number,
  omegaLogK: number,
  muStar: number,
): number {
  if (lambda <= 0 || omegaLogK <= 0) return 0;

  // Spin-fluctuation energy scale: ~2.5× the phonon log-average in cuprates
  const omegaSfK = omegaLogK * 2.5;

  // d-wave spin-fluctuation coupling: stronger than electron-phonon
  const lambdaSf = lambda * 1.4;

  // d_{x²-y²} symmetry strongly cancels isotropic Coulomb repulsion
  const muStarDwave = muStar * 0.10;

  const denom = lambdaSf - muStarDwave * (1 + 0.62 * lambdaSf);
  if (denom <= 0) return 0;

  // Monthoux-Scalapino prefactor for d-wave: ω_sf / 1.5 (weaker than BCS 1/1.2)
  // Exponent coefficient 1.15 (slight departure from BCS 1.04 for d-wave DOS)
  const exponent = -1.15 * (1 + lambdaSf) / denom;
  if (exponent < -50) return 0;

  let tc = (omegaSfK / 1.5) * Math.exp(exponent);

  // Physical cap: HgBa₂Ca₂Cu₃O₈ holds the record at ~165 K
  tc = Math.max(0, Math.min(185, tc));
  return Number.isFinite(tc) ? Number(tc.toFixed(2)) : 0;
}

/**
 * QCP-proximity Tc for heavy-fermion superconductors.
 *
 * Heavy-fermion SCs are unconventional: Cooper pairs form in the heavy
 * quasiparticle bands (m* = 10–1000 m_e), mediated by spin or multipolar
 * fluctuations near an antiferromagnetic quantum critical point (QCP).
 * Allen-Dynes is inapplicable because:
 *   1. The pairing boson is a spin fluctuation at omega_sf << omega_phonon.
 *      In HF systems omega_sf ~= 0.4 * omega_log (phonon omega_log as proxy).
 *   2. The heavy quasiparticle mass suppresses Tc via Kadowaki-Woods scaling
 *      (mstar/m ~ 1/lambda_phonon).
 *   3. Proximity to the magnetic QCP provides an anomalous boost (lambda_qcp).
 *
 * Empirical range of known ambient-pressure heavy-fermion SCs:
 *   UBe13: Tc ~= 0.9 K  |  UPt3: Tc ~= 0.5 K  |  CeCu2Si2: Tc ~= 0.6 K
 *   CeCoIn5: Tc ~= 2.3 K (record ambient-pressure HF SC)
 *   PuCoGa5: Tc ~= 18 K  (f-electron outlier; heavy-fermion-like)
 *
 * @see F. Steglich et al., PRL 43, 1892 (1979) - CeCu2Si2 discovery
 * @see N. D. Mathur et al., Nature 394, 39 (1998) - QCP-mediated pairing
 * @see P. Monthoux & G. G. Lonzarich, PRB 63, 054529 (2001) - d/p-wave SC near QCP
 * @see C. Pfleiderer, Rev. Mod. Phys. 81, 1551 (2009) - HF SC review
 */
function heavyFermionQCPTc(
  lambda: number,
  omegaLogK: number,
  muStar: number,
): number {
  // Minimum coupling for heavy-fermion SC onset (below this, no QCP-mediated pairing)
  const lambdaCrit = 0.25;
  if (lambda <= lambdaCrit || omegaLogK <= 0) return 0;

  // Spin-fluctuation energy in HF systems is much softer than phonons
  const omegaSfK = omegaLogK * 0.40;

  // QCP-proximity boost: spin-fluctuation coupling is enhanced near the QCP
  const qcpProximity = Math.min(3.0, lambda / lambdaCrit);
  const lambdaQcp = lambda * 0.60 * qcpProximity;

  // Partial Coulomb suppression for non-s-wave pairing symmetry
  const muStarQcp = muStar * 0.35;

  const denom = lambdaQcp - muStarQcp * (1 + 0.62 * lambdaQcp);
  if (denom <= 0) return 0;

  // Heavy quasiparticle mass suppression: Tc ∝ (m*/m)^(-α), α ≈ 0.3
  // Heavier mass ↔ weaker phonon coupling → higher massEst
  const massEst = Math.max(10, 80 / (lambda + 0.1));
  const massSuppression = Math.pow(massEst, -0.30);

  const exponent = -1.04 * (1 + lambdaQcp) / denom;
  if (exponent < -50) return 0;

  let tc = (omegaSfK / 1.2) * massSuppression * Math.exp(exponent);

  // Empirical cap: CeRhIn₅ under pressure ~2 K; PuCoGa₅ outlier ~18 K
  tc = Math.max(0, Math.min(25, tc));
  return Number.isFinite(tc) ? Number(tc.toFixed(3)) : 0;
}

/**
 * Allen-Dynes (1975) formula with strong-coupling correction factors f1, f2.
 * Extends McMillan (1968) to the strong-coupling regime via spectral moment ratio
 * ω₂/ω_log. Family-specific λ̄ prefactor adjusts the f1 correction for
 * hydrides and unconventional-adjacent materials.
 *
 * Tc = (ω_log / 1.2) · f1 · f2 · exp( −1.04(1+λ) / (λ − μ*(1 + 0.62λ)) )
 *
 * f1 = (1 + (λ/λ̄)^(3/2))^(1/3)           [strong-coupling shape factor]
 * f2 = 1 + (√ω₂/ω_log − 1)·λ²/(λ² + Λ₂²) [spectral moment correction]
 * Λ₂ = 1.82(1 + 6.3μ*)(√ω₂/ω_log)
 *
 * @see P. B. Allen & R. C. Dynes, Phys. Rev. B 12, 905 (1975)
 * @see W. L. McMillan, Phys. Rev. 167, 331 (1968)
 * @see J. P. Carbotte, Rev. Mod. Phys. 62, 1027 (1990) — Eliashberg theory review
 */
export function allenDynesTcFull(input: AllenDynesTcInput): AllenDynesTcResult {
  const { lambda, omegaLog, muStar, omega2Avg, isHydride, family } = input;

  const omegaLogK = omegaLog * CM1_TO_KELVIN;

  let regime: "weak" | "intermediate" | "strong" | "very-strong" = "weak";
  if (lambda > 2.5) regime = "very-strong";
  else if (lambda > 1.5) regime = "strong";
  else if (lambda > 0.5) regime = "intermediate";

  // -------------------------------------------------------------------
  // Early dispatch for families where phonon Allen-Dynes is inapplicable
  // -------------------------------------------------------------------

  // Cuprates: d-wave spin-fluctuation pairing. The phonon energy scale
  // (ω_log), phonon coupling constant (λ_ep), and BCS Coulomb pseudopotential
  // (μ*) all map to wrong physics here. Route to Monthoux-Scalapino formula.
  const familyUpper = (family ?? "").toUpperCase();
  if (familyUpper.includes("CUPRATE") || familyUpper.includes("NICKELATE")) {
    const tc = cuprateSpinFluctuationTc(lambda, omegaLogK, muStar);
    return {
      tc,
      f1: 1.0,
      f2: 1.0,
      regime,
      method: "spin-fluctuation-cuprate",
      applicabilityWarning:
        "Allen-Dynes (1975) assumes phonon-mediated s-wave pairing and is physically " +
        "invalid for cuprates/nickelates. These are unconventional d-wave materials " +
        "driven by spin fluctuations (Scalapino, Rev. Mod. Phys. 84, 1383 (2012)). " +
        "Monthoux-Scalapino spin-fluctuation formula used instead.",
    };
  }

  // Heavy fermions: f-electron spin/multipolar fluctuation pairing near an
  // antiferromagnetic QCP. Mass renormalization (m*/m ~ 10–1000) and the
  // spin-fluctuation energy scale (≪ phonon ω_log) make Allen-Dynes invalid.
  if (familyUpper.includes("HEAVY") || familyUpper.includes("FERMION")) {
    const tc = heavyFermionQCPTc(lambda, omegaLogK, muStar);
    return {
      tc,
      f1: 1.0,
      f2: 1.0,
      regime,
      method: "qcp-heavy-fermion",
      applicabilityWarning:
        "Allen-Dynes (1975) assumes phonon-mediated pairing and is physically invalid " +
        "for heavy-fermion superconductors. These systems are governed by f-electron " +
        "Kondo physics, heavy quasiparticle bands (m*/m ~ 10–1000), and QCP-proximity " +
        "spin fluctuations (Pfleiderer, Rev. Mod. Phys. 81, 1551 (2009)). " +
        "QCP-proximity spin-fluctuation formula used instead.",
    };
  }

  // -------------------------------------------------------------------
  // Conventional Allen-Dynes path (phonon-mediated BCS + hydrides)
  // -------------------------------------------------------------------

  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denominator) < 1e-6 || denominator <= 0 || omegaLogK <= 0) {
    return { tc: 0, f1: 1, f2: 1, regime, method: "allen-dynes" };
  }

  const lambdaBar = lambdaBarPrefactor(family) * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

  let f2 = 1.0;
  if (omega2Avg && omega2Avg > 0 && omegaLog > 0) {
    const sqrtOmega2 = Math.sqrt(omega2Avg);
    const omegaRatio = sqrtOmega2 / omegaLog;
    const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
    f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
    f2 = Math.max(0.7, Math.min(1.5, f2));
  }

  const exponent = -1.04 * (1 + lambda) / denominator;
  if (exponent < -50) return { tc: 0, f1: Number(f1.toFixed(4)), f2: Number(f2.toFixed(4)), regime, method: "allen-dynes" as const };
  let tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);

  if (!Number.isFinite(tc) || tc < 0) tc = 0;

  let method: TcMethod = "allen-dynes";

  if (isHydride && lambda > 1.2 && omegaLogK > 0) {
    const tcSisso = sissoHydrideTc(lambda, omegaLogK, muStar);
    const tcXie = xieStrongCouplingTc(lambda, omegaLogK, muStar, omega2Avg ? Math.sqrt(omega2Avg) * CM1_TO_KELVIN : undefined);
    const bestAlt = tcXie > tcSisso ? { tc: tcXie, m: "xie-strong-coupling" as TcMethod } : { tc: tcSisso, m: "sisso-hydride" as TcMethod };
    if (bestAlt.tc > tc) {
      tc = bestAlt.tc;
      method = bestAlt.m;
    }
  }

  tc = Math.max(0, Math.min(500, tc));

  return {
    tc: Number(tc.toFixed(2)),
    f1: Number(f1.toFixed(4)),
    f2: Number(f2.toFixed(4)),
    regime,
    method,
  };
}

/**
 * SISSO-derived empirical Tc formula for high-pressure hydride superconductors.
 * Developed via sure-independence screening and sparsifying operator (SISSO)
 * regression on DFT+Eliashberg data for H-rich binary hydrides.
 * Valid for λ > 1.2 and ω_log-rich hydrides under high pressure.
 *
 * @see S.-R. Xie et al., Phys. Rev. B 105, 064517 (2022)
 * @see R. Ouyang et al., npj Comput. Mater. 5, 83 (2019) — SISSO method
 */
function sissoHydrideTc(lambda: number, omegaLogK: number, muStar: number): number {
  const lambdaEff = lambda - muStar * (1 + lambda);
  if (lambdaEff <= 0) return 0;

  let prefactor: number;
  let lambdaExp: number;
  let muStarDamping: number;

  if (lambda > 3.0) {
    prefactor = 0.22;
    lambdaExp = 0.48;
    muStarDamping = Math.pow(1 + 4.8 * muStar * Math.log(Math.max(1.01, lambda)), -0.32);
  } else if (lambda > 2.0) {
    prefactor = 0.20;
    lambdaExp = 0.52;
    muStarDamping = Math.pow(1 + 5.5 * muStar * Math.log(Math.max(1.01, lambda)), -0.30);
  } else {
    prefactor = 0.182;
    lambdaExp = 0.572;
    muStarDamping = Math.pow(1 + 6.5 * muStar * Math.log(Math.max(1.01, lambda)), -0.278);
  }

  let tc = prefactor * omegaLogK * Math.pow(lambdaEff, lambdaExp) * muStarDamping;

  if (lambda > 2.5) {
    const anharmonicCorr = 1 - 0.025 * Math.pow(lambda - 2.5, 1.1);
    tc *= Math.max(0.7, anharmonicCorr);
  }

  return Number.isFinite(tc) ? Math.max(0, tc) : 0;
}

/**
 * Strong-coupling Tc formula for hydrides beyond the Allen-Dynes validity range (λ > 2).
 * Incorporates spectral saturation, anharmonic damping, and an effective μ* renormalization
 * for extreme coupling. Regime-dependent A, B coefficients fit to H-rich hydride DFT surveys.
 *
 * @see S.-R. Xie et al., Phys. Rev. B 105, 064517 (2022)
 * @see I. Errea et al., Nature 578, 66 (2020) — anharmonicity in LaH10
 */
function xieStrongCouplingTc(lambda: number, omegaLogK: number, muStar: number, omega2K?: number): number {
  if (lambda <= 1.0 || omegaLogK <= 0) return 0;

  const muStarEff = muStar * (1 + 0.5 * muStar);
  const lambdaNet = lambda - muStarEff * (1 + lambda);
  if (lambdaNet <= 0) return 0;

  let A: number, B: number;
  if (lambda > 3.0) {
    A = 0.14 * (1 + 4.0 / (lambda + 2.0));
    B = 0.92 * (1 + 0.30 * muStar);
  } else if (lambda > 2.0) {
    A = 0.13 * (1 + 4.5 / (lambda + 2.3));
    B = 0.98 * (1 + 0.34 * muStar);
  } else {
    A = 0.12 * (1 + 5.2 / (lambda + 2.6));
    B = 1.04 * (1 + 0.38 * muStar);
  }

  const C = lambda * (1 - 0.14 * muStar) - muStar * (1 + 0.62 * lambda);
  if (C <= 0) return 0;

  const prefactor = omegaLogK * A;
  const exponent = -B * (1 + lambda) / C;
  if (exponent < -50) return 0;

  let tc = prefactor * Math.exp(exponent);

  if (omega2K && omega2K > 0) {
    const ratio = omega2K / omegaLogK;
    if (lambda > 2.0) {
      const spectralCorr = 1 + 0.035 * Math.pow(ratio - 1, 1.8) * Math.sqrt(lambda);
      tc *= Math.min(1.4, Math.max(0.85, spectralCorr));
    } else {
      const spectralCorr = 1 + 0.0241 * Math.pow(ratio - 1, 2) * lambda;
      tc *= Math.min(1.3, Math.max(0.9, spectralCorr));
    }
  }

  if (lambda > 3.5) {
    const saturationDamping = 1 - 0.03 * Math.pow(lambda - 3.5, 1.0);
    tc *= Math.max(0.65, saturationDamping);
  } else if (lambda > 2.5) {
    const saturationDamping = 1 - 0.02 * Math.pow(lambda - 2.5, 1.2);
    tc *= Math.max(0.75, saturationDamping);
  }

  return Number.isFinite(tc) ? Math.max(0, tc) : 0;
}

export function allenDynesTcSimple(
  lambda: number,
  omegaLogK: number,
  muStar: number,
  omega2Avg?: number,
): number {
  return allenDynesTcFull({
    lambda,
    omegaLog: omegaLogK / CM1_TO_KELVIN,
    muStar,
    omega2Avg,
  }).tc;
}

export function allenDynesTcFromKelvin(
  lambda: number,
  omegaLogK: number,
  muStar: number,
  omega2Avg?: number,
  isHydride?: boolean,
): number {
  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denominator) < 1e-6 || denominator <= 0 || omegaLogK <= 0) {
    return 0;
  }

  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

  let f2 = 1.0;
  if (omega2Avg && omega2Avg > 0) {
    const omegaLogFromK = omegaLogK / CM1_TO_KELVIN;
    const sqrtOmega2 = Math.sqrt(omega2Avg);
    const omegaRatio = sqrtOmega2 / omegaLogFromK;
    const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
    f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
    f2 = Math.max(0.8, f2);
  }

  const exponent = -1.04 * (1 + lambda) / denominator;
  if (exponent < -50) return 0;
  let tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);

  if (!Number.isFinite(tc) || tc < 0) tc = 0;

  if (isHydride && lambda > 1.2) {
    const tcSisso = sissoHydrideTc(lambda, omegaLogK, muStar);
    const tcXie = xieStrongCouplingTc(lambda, omegaLogK, muStar, omega2Avg ? Math.sqrt(omega2Avg) * CM1_TO_KELVIN : undefined);
    tc = Math.max(tc, tcSisso, tcXie);
  }

  return Math.max(0, Math.min(500, tc));
}

/**
 * Numerically inverts Allen-Dynes to find λ that produces a target Tc,
 * given a Debye temperature and Coulomb pseudopotential.
 * Uses bisection (80 iterations, typically converges to < 0.001 in λ).
 *
 * @see P. B. Allen & R. C. Dynes, Phys. Rev. B 12, 905 (1975)
 */
export function invertAllenDynesLambda(
  tc: number,
  thetaD: number,
  muStar: number,
): number {
  if (tc <= 0 || thetaD <= 0) return 0;
  const omegaLogK = thetaD * 0.60;
  let lambdaLow = 0.05;
  let lambdaHigh = 6.0;
  for (let i = 0; i < 80; i++) {
    const lambdaMid = (lambdaLow + lambdaHigh) / 2;
    const tcCalc = allenDynesTcFromKelvin(lambdaMid, omegaLogK, muStar);
    if (tcCalc < tc) lambdaLow = lambdaMid;
    else lambdaHigh = lambdaMid;
  }
  return (lambdaLow + lambdaHigh) / 2;
}

export function convertOmegaLogMeVToKelvin(omegaLogMeV: number): number {
  return omegaLogMeV * MEV_TO_KELVIN;
}

export function convertOmegaLogCm1ToKelvin(omegaLogCm1: number): number {
  return omegaLogCm1 * CM1_TO_KELVIN;
}

/**
 * Validates and optionally corrects a surrogate ω_log estimate against
 * spectral data and physical bounds (Debye temperature, max phonon frequency).
 * Blends surrogate and spectral values when the ratio diverges moderately (0.67–2.0);
 * overrides with spectral value for severe divergence (ratio < 0.5 or > 2.0).
 *
 * @see P. B. Allen & R. C. Dynes, Phys. Rev. B 12, 905 (1975) — ω_log definition
 */
export function validateOmegaLog(
  surrogateOmegaLog: number,
  spectralOmegaLog: number | null,
  debyeTemp: number,
  maxPhononFreq: number,
): { validatedOmegaLog: number; corrected: boolean; warning: string | null } {
  let omega = surrogateOmegaLog;
  let corrected = false;
  let warning: string | null = null;

  if (spectralOmegaLog && spectralOmegaLog > 0) {
    const ratio = surrogateOmegaLog / spectralOmegaLog;
    if (ratio > 2.0 || ratio < 0.5) {
      omega = spectralOmegaLog;
      corrected = true;
      warning = `Surrogate omega_log=${surrogateOmegaLog.toFixed(1)} diverged from spectral=${spectralOmegaLog.toFixed(1)} (ratio=${ratio.toFixed(2)}), using spectral value`;
    } else if (ratio > 1.5 || ratio < 0.67) {
      omega = 0.3 * surrogateOmegaLog + 0.7 * spectralOmegaLog;
      corrected = true;
      warning = `Surrogate omega_log blended with spectral (ratio=${ratio.toFixed(2)})`;
    }
  }

  const debyeLimit = debyeTemp * 0.69 / CM1_TO_KELVIN;
  if (omega > debyeLimit * 1.5 && debyeLimit > 10) {
    omega = debyeLimit * 1.2;
    corrected = true;
    warning = (warning ? warning + "; " : "") + `omega_log capped to 1.2x Debye estimate (${debyeLimit.toFixed(0)} cm-1)`;
  }

  if (omega > maxPhononFreq * 0.8 && maxPhononFreq > 0) {
    omega = maxPhononFreq * 0.6;
    corrected = true;
    warning = (warning ? warning + "; " : "") + `omega_log capped below maxPhononFreq`;
  }

  omega = Math.max(5, omega);

  return {
    validatedOmegaLog: Number(omega.toFixed(1)),
    corrected,
    warning,
  };
}

export { CM1_TO_KELVIN, MEV_TO_KELVIN };
