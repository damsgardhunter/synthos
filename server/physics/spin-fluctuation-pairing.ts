/**
 * Spin-fluctuation pairing channel for unconventional superconductors.
 *
 * For cuprates, iron-pnictides, nickelates, and heavy-fermion systems,
 * superconductivity is NOT phonon-mediated — it's driven by spin fluctuations.
 * The peak of the dynamic spin susceptibility χ(q,ω) at some characteristic
 * wavevector Q and frequency ω_sf sets the pairing scale, in the same role
 * that α²F(ω) plays for phonon-mediated superconductors.
 *
 * This module provides:
 *   1. Pairing channel classifier — determines which mechanism dominates
 *      using magnetic ground state, electronic structure, and DFT+U data
 *   2. Static Lindhard susceptibility χ₀(q) — from the non-interacting
 *      band structure. Cheap. Catches Fermi surface nesting.
 *   3. RPA-enhanced susceptibility χ_RPA(q,ω) — using Hubbard U from
 *      the DFT+U workflow. Captures Stoner enhancement and spin-fluctuation
 *      peaks. Standard spin-fluctuation theory (Berk-Schrieffer/Moriya/Scalapino).
 *   4. Spin-fluctuation coupling I²χ(ω) — the "spin analog of α²F".
 *      Integrates to give λ_sf and ω_sf for Tc estimation.
 *   5. Spin-fluctuation Tc estimator — Eliashberg-like equation with I²χ
 *      instead of α²F. Combined with phonon Tc for total prediction.
 *
 * @see N. F. Berk & J. R. Schrieffer, PRL 17, 433 (1966) — spin fluctuation pairing
 * @see T. Moriya, Spin Fluctuations in Itinerant Electron Magnetism (Springer, 1985)
 * @see D. J. Scalapino, Rev. Mod. Phys. 84, 1383 (2012) — d-wave pairing review
 * @see K. Kuroki et al., PRL 101, 087004 (2008) — s± pairing in pnictides
 * @see P. Monthoux et al., Nature 450, 1177 (2007) — SC near magnetic instabilities
 */

import type { ElectronicStructure } from "../learning/physics-engine";
import type { MagneticGroundStateResult } from "../dft/magnetic-ground-state";

// ─── Pairing Channel Classification ──────────────────────────────────

export type PairingChannel =
  | "phonon-bcs"           // conventional: Allen-Dynes/Eliashberg applies
  | "spin-fluctuation"     // d-wave or s±: χ(q,ω) dominates
  | "orbital-fluctuation"  // Hund's metals: orbital channel
  | "mixed-phonon-spin"    // both channels active (iron pnictides)
  | "unknown";

export interface PairingClassification {
  /** Dominant pairing channel */
  dominantChannel: PairingChannel;
  /** Confidence in the classification (0-1) */
  confidence: number;
  /** Score for each channel (0-1) — higher = more likely */
  channelScores: {
    phonon: number;
    spinFluctuation: number;
    orbital: number;
  };
  /** Evidence used for classification */
  evidence: string[];
  /** Whether spin susceptibility calculation is warranted */
  shouldComputeSusceptibility: boolean;
  /** Pairing symmetry prediction */
  pairingSymmetry: "s-wave" | "d-wave" | "s±-wave" | "p-wave" | "unknown";
}

/**
 * Classify the likely pairing channel for a material.
 *
 * Uses existing pipeline data:
 *   - Magnetic ground state (from magnetic-ground-state.ts)
 *   - Electronic structure (from physics-engine.ts)
 *   - DFT+U regime and Hubbard U values
 *   - Material family classification
 *
 * Decision logic:
 *   1. AFM ground state + insulating parent → spin fluctuation (cuprate/pnictide)
 *   2. FM ground state + high DOS → phonon BCS (conventional)
 *   3. Large U/W ratio + flat bands → mixed or orbital
 *   4. No magnetic moment + metallic → phonon BCS
 */
export function classifyPairingChannel(
  electronic: ElectronicStructure,
  magneticGS: MagneticGroundStateResult | null,
  correlationRegime: string,
  materialPatterns: string[],
  hubbardU: number, // average U across correlated sites
  bandwidth: number, // estimated bandwidth W (eV)
): PairingClassification {
  const evidence: string[] = [];
  let phononScore = 0.5; // start neutral
  let spinScore = 0.1;
  let orbitalScore = 0.05;

  // ── Magnetic ground state signals ──
  if (magneticGS?.searchPerformed) {
    const gs = magneticGS.groundState;

    if (gs.startsWith("AFM")) {
      spinScore += 0.35;
      phononScore -= 0.15;
      evidence.push(`AFM ground state (${gs}) — strong spin-fluctuation signal`);

      if (magneticGS.wellSeparated) {
        spinScore += 0.1;
        evidence.push("AFM state well-separated from FM — robust magnetic ordering");
      }

      // AFM + near insulating → cuprate/pnictide class
      if (electronic.metallicity < 0.3) {
        spinScore += 0.15;
        evidence.push("Near-insulating + AFM — Mott-proximate, spin-fluctuation pairing likely");
      }
    } else if (gs === "FM") {
      // FM suppresses spin-fluctuation pairing (no Q≠0 peak in χ)
      phononScore += 0.15;
      spinScore -= 0.1;
      evidence.push("FM ground state — spin-fluctuation pairing suppressed, phonon channel favored");
    } else if (gs === "NM") {
      // Non-magnetic — conventional phonon pairing most likely
      phononScore += 0.2;
      evidence.push("NM ground state — conventional phonon BCS likely");
    }

    // Absolute magnetization signal
    if (magneticGS.groundStateMagnetization != null) {
      const absMag = Math.abs(magneticGS.groundStateMagnetization);
      if (absMag > 1.0) {
        spinScore += 0.1;
        evidence.push(`Large magnetic moment (${absMag.toFixed(2)} μ_B) — strong spin channel`);
      }
    }
  }

  // ── Electronic structure signals ──

  // Flat bands at Fermi level → enhanced DOS → stronger coupling in ALL channels
  if (electronic.flatBandIndicator > 0.5) {
    spinScore += 0.1;
    orbitalScore += 0.1;
    evidence.push(`Flat bands at E_F (score=${electronic.flatBandIndicator.toFixed(2)}) — enhances spin/orbital channels`);
  }

  // Nesting features → peaks in χ(q) → spin fluctuations
  if (electronic.nestingScore > 0.5) {
    spinScore += 0.15;
    evidence.push(`Strong Fermi surface nesting (score=${electronic.nestingScore.toFixed(2)}) — χ(q) peak expected`);
  }

  // Van Hove proximity → logarithmic DOS divergence → all channels enhanced
  if (electronic.vanHoveProximity > 0.5) {
    spinScore += 0.1;
    phononScore += 0.05;
    evidence.push(`Near Van Hove singularity (proximity=${electronic.vanHoveProximity.toFixed(2)})`);
  }

  // Multi-orbital character → orbital fluctuations
  const dFrac = electronic.orbitalFractions.d + electronic.orbitalFractions.f;
  if (dFrac > 0.5) {
    orbitalScore += 0.1 * dFrac;
    evidence.push(`Strong d/f orbital character (${(dFrac * 100).toFixed(0)}%) — orbital fluctuations possible`);
  }

  // ── Correlation regime signals ──

  const UoverW = bandwidth > 0 ? hubbardU / bandwidth : 0;

  if (correlationRegime === "Mott-proximate") {
    spinScore += 0.25;
    phononScore -= 0.2;
    evidence.push(`Mott-proximate (U/W=${UoverW.toFixed(2)}) — spin fluctuations dominate near Mott transition`);
  } else if (correlationRegime === "strongly-correlated") {
    spinScore += 0.15;
    phononScore -= 0.1;
    evidence.push(`Strongly correlated (U/W=${UoverW.toFixed(2)}) — spin fluctuations significant`);
  } else if (correlationRegime === "moderately-correlated") {
    spinScore += 0.05;
    evidence.push(`Moderately correlated — mixed channels possible`);
  }

  // ── Material pattern signals ──

  for (const pattern of materialPatterns) {
    if (pattern.includes("cuprate") || pattern.includes("Mott")) {
      spinScore += 0.2;
      phononScore -= 0.15;
      evidence.push(`Cuprate/Mott pattern — d-wave spin-fluctuation pairing`);
    }
    if (pattern.includes("Fe-pnictide") || pattern.includes("spin-fluctuation")) {
      spinScore += 0.15;
      evidence.push(`Fe-pnictide pattern — s± spin-fluctuation pairing`);
    }
    if (pattern.includes("heavy-fermion") || pattern.includes("Kondo")) {
      spinScore += 0.15;
      orbitalScore += 0.1;
      evidence.push(`Heavy-fermion/Kondo pattern — spin + orbital channels`);
    }
    if (pattern.includes("superhydride") || pattern.includes("hydride-phonon")) {
      phononScore += 0.3;
      spinScore -= 0.1;
      evidence.push(`Hydride pattern — phonon BCS dominates`);
    }
  }

  // ── Normalize and classify ──
  phononScore = Math.max(0, Math.min(1, phononScore));
  spinScore = Math.max(0, Math.min(1, spinScore));
  orbitalScore = Math.max(0, Math.min(1, orbitalScore));

  const total = phononScore + spinScore + orbitalScore;
  if (total > 0) {
    phononScore /= total;
    spinScore /= total;
    orbitalScore /= total;
  }

  let dominantChannel: PairingChannel;
  let pairingSymmetry: "s-wave" | "d-wave" | "s±-wave" | "p-wave" | "unknown";

  if (spinScore > 0.45 && spinScore > phononScore) {
    if (materialPatterns.some(p => p.includes("cuprate"))) {
      dominantChannel = "spin-fluctuation";
      pairingSymmetry = "d-wave";
    } else if (materialPatterns.some(p => p.includes("Fe-pnictide"))) {
      dominantChannel = "mixed-phonon-spin";
      pairingSymmetry = "s±-wave";
    } else {
      dominantChannel = "spin-fluctuation";
      pairingSymmetry = "d-wave";
    }
  } else if (phononScore > 0.5) {
    dominantChannel = "phonon-bcs";
    pairingSymmetry = "s-wave";
  } else if (orbitalScore > 0.3) {
    dominantChannel = "orbital-fluctuation";
    pairingSymmetry = "unknown";
  } else if (spinScore > 0.25 && phononScore > 0.25) {
    dominantChannel = "mixed-phonon-spin";
    pairingSymmetry = "s±-wave";
  } else {
    dominantChannel = "unknown";
    pairingSymmetry = "unknown";
  }

  const maxScore = Math.max(phononScore, spinScore, orbitalScore);
  const confidence = maxScore > 0.6 ? 0.8 + (maxScore - 0.6) * 0.5 : maxScore * 1.3;

  return {
    dominantChannel,
    confidence: Math.min(0.95, Math.max(0.1, confidence)),
    channelScores: {
      phonon: Number(phononScore.toFixed(3)),
      spinFluctuation: Number(spinScore.toFixed(3)),
      orbital: Number(orbitalScore.toFixed(3)),
    },
    evidence,
    shouldComputeSusceptibility: spinScore > 0.25,
    pairingSymmetry,
  };
}

// ─── Spin Susceptibility (Lindhard + RPA) ────────────────────────────

export interface SpinSusceptibilityResult {
  /** Static Lindhard susceptibility χ₀(Q) at the nesting wavevector */
  chi0AtQ: number;
  /** RPA-enhanced susceptibility χ_RPA(Q) = χ₀ / (1 - U·χ₀) */
  chiRPAAtQ: number;
  /** Stoner enhancement factor S = 1 / (1 - U·χ₀) */
  stonerFactor: number;
  /** Whether the system is near a magnetic instability (S > 5) */
  nearInstability: boolean;
  /** Characteristic spin-fluctuation frequency ω_sf (meV) */
  omegaSF: number;
  /** Spin-fluctuation coupling constant λ_sf */
  lambdaSF: number;
  /** Spin-fluctuation Tc estimate (K) */
  tcSpinFluctuation: number;
  /** Method used */
  method: "lindhard-rpa" | "empirical-scaling";
  /** Notes */
  notes: string[];
}

/**
 * Estimate spin susceptibility and spin-fluctuation Tc.
 *
 * Uses the RPA approximation:
 *   χ_RPA(q) = χ₀(q) / (1 - U · χ₀(q))
 *
 * where χ₀(q) is the Lindhard (non-interacting) susceptibility and U is
 * the Hubbard interaction from DFT+U.
 *
 * The spin-fluctuation coupling constant is:
 *   λ_sf = N(E_F) · <I²> · ∫ Im[χ(q,ω)] / ω dω
 *
 * which we approximate as:
 *   λ_sf ≈ N(E_F) · U² · χ_RPA(Q) / ω_sf
 *
 * The characteristic frequency ω_sf is estimated from the Stoner factor:
 *   ω_sf ≈ W / S  (bandwidth / Stoner enhancement)
 *
 * @see T. Moriya & K. Ueda, Adv. Phys. 49, 555 (2000)
 * @see P. Monthoux & D. J. Scalapino, PRL 72, 1874 (1994)
 */
export function computeSpinSusceptibility(
  electronic: ElectronicStructure,
  hubbardU: number,     // eV — average U for correlated sites
  bandwidth: number,    // eV — estimated bandwidth W
  nestingScore: number, // 0-1 — from electronic structure
  pairingSymmetry: "s-wave" | "d-wave" | "s±-wave" | "p-wave" | "unknown",
): SpinSusceptibilityResult {
  const notes: string[] = [];
  const N_EF = Math.max(0.1, electronic.densityOfStatesAtFermi);

  // Step 1: Estimate Lindhard susceptibility χ₀ at nesting vector Q
  // For a nested Fermi surface, χ₀(Q) ∝ N(E_F) × log(W / T)
  // For a poorly nested surface, χ₀(Q) ≈ N(E_F)
  const nestingEnhancement = 1.0 + nestingScore * 2.0; // 1x (no nesting) to 3x (perfect nesting)
  const chi0AtQ = N_EF * nestingEnhancement;

  // Step 2: RPA enhancement
  // χ_RPA = χ₀ / (1 - U·χ₀)
  // Stoner factor S = 1 / (1 - U·χ₀)
  // When U·χ₀ → 1, the system is at a magnetic instability (Stoner criterion)
  const Uchi0 = hubbardU * chi0AtQ;
  const stonerDenom = Math.max(0.05, 1 - Uchi0); // floor at 0.05 to prevent divergence
  const stonerFactor = 1 / stonerDenom;
  const chiRPAAtQ = chi0AtQ * stonerFactor;
  const nearInstability = stonerFactor > 5;

  notes.push(`Lindhard χ₀(Q) = ${chi0AtQ.toFixed(3)} (nesting enhancement = ${nestingEnhancement.toFixed(2)})`);
  notes.push(`Stoner factor S = ${stonerFactor.toFixed(2)} (U·χ₀ = ${Uchi0.toFixed(3)})`);

  if (nearInstability) {
    notes.push(`WARNING: Near magnetic instability (S > 5) — strong spin fluctuations expected`);
  }

  // Step 3: Characteristic spin-fluctuation frequency
  // ω_sf ≈ W / S for itinerant systems
  // For localized systems (Kondo), ω_sf ≈ T_K (Kondo temperature)
  const omegaSF_eV = Math.max(0.001, bandwidth / stonerFactor);
  const omegaSF_meV = omegaSF_eV * 1000;
  const omegaSF_K = omegaSF_eV * 11604; // eV to K

  notes.push(`Spin-fluctuation energy: ω_sf = ${omegaSF_meV.toFixed(1)} meV = ${omegaSF_K.toFixed(0)} K`);

  // Step 4: Spin-fluctuation coupling constant λ_sf
  // λ_sf ≈ N(E_F) · U² · χ_RPA(Q) / ω_sf
  // Normalized so that λ_sf ≈ 1 for cuprates near optimal doping
  const I2 = hubbardU * hubbardU; // |I|² = U² (interaction matrix element squared)
  const lambdaSF = N_EF * I2 * chiRPAAtQ / Math.max(0.01, omegaSF_eV);
  // Renormalize: the raw formula gives huge numbers; scale by bandwidth
  const lambdaSF_renorm = lambdaSF * omegaSF_eV / (bandwidth * bandwidth);

  notes.push(`Spin-fluctuation λ_sf = ${lambdaSF_renorm.toFixed(3)}`);

  // Step 5: Tc from spin fluctuations
  // For d-wave: Tc ≈ ω_sf · exp(-1 / λ_sf_eff)
  // The d-wave gap equation has a stronger coupling than s-wave because the
  // repulsive interaction at Q is attractive in the d-wave channel.
  //
  // For s±-wave (pnictides): similar but with sign change between electron/hole pockets
  let tcSF = 0;
  const lambdaEff = lambdaSF_renorm;

  if (lambdaEff > 0.1) {
    // d-wave Monthoux-Scalapino formula
    const muStarSF = pairingSymmetry === "d-wave" ? 0.0 // d-wave: Coulomb repulsion cancels by symmetry
      : pairingSymmetry === "s±-wave" ? 0.05 // s±: partial cancellation
      : 0.10; // unknown: conventional μ*

    const denom = lambdaEff - muStarSF * (1 + 0.62 * lambdaEff);
    if (denom > 0) {
      const exponent = -1.04 * (1 + lambdaEff) / denom;
      if (exponent > -50) {
        tcSF = (omegaSF_K / 1.2) * Math.exp(exponent);
        tcSF = Math.max(0, Math.min(300, tcSF)); // cap at 300 K
      }
    }

    notes.push(`Tc(spin-fluctuation) = ${tcSF.toFixed(1)} K (${pairingSymmetry}, μ*_sf=${muStarSF})`);
  } else {
    notes.push(`λ_sf too small (${lambdaEff.toFixed(3)}) for meaningful spin-fluctuation Tc`);
  }

  return {
    chi0AtQ: Number(chi0AtQ.toFixed(4)),
    chiRPAAtQ: Number(chiRPAAtQ.toFixed(4)),
    stonerFactor: Number(stonerFactor.toFixed(3)),
    nearInstability,
    omegaSF: Number(omegaSF_meV.toFixed(2)),
    lambdaSF: Number(lambdaEff.toFixed(4)),
    tcSpinFluctuation: Number(tcSF.toFixed(2)),
    method: "lindhard-rpa",
    notes,
  };
}

// ─── Combined Tc (phonon + spin fluctuation) ─────────────────────────

export interface CombinedTcResult {
  /** Phonon-mediated Tc (from Allen-Dynes/Eliashberg) */
  tcPhonon: number;
  /** Spin-fluctuation Tc */
  tcSpinFluctuation: number;
  /** Combined best estimate of Tc */
  tcCombined: number;
  /** How the channels were combined */
  combinationMethod: "constructive" | "destructive" | "single-channel" | "max-of-channels";
  /** Dominant channel for this material */
  dominantChannel: PairingChannel;
  /** Pairing symmetry */
  pairingSymmetry: string;
  /** Notes */
  notes: string[];
}

/**
 * Combine phonon and spin-fluctuation Tc estimates.
 *
 * The interaction between channels depends on the pairing symmetry:
 *   - s-wave + phonon: constructive (both attract in s-wave)
 *   - d-wave + phonon: destructive (phonons are pair-breaking in d-wave)
 *   - s± + phonon: weakly constructive (phonons help intraband)
 *
 * For materials where both channels are significant, the combined Tc
 * is not simply the sum — it depends on the interference.
 *
 * @see A. V. Chubukov et al., Phys. Rev. B 78, 134512 (2008)
 */
export function combineTcEstimates(
  tcPhonon: number,
  tcSpinFluctuation: number,
  pairing: PairingClassification,
): CombinedTcResult {
  const notes: string[] = [];
  let tcCombined: number;
  let combinationMethod: "constructive" | "destructive" | "single-channel" | "max-of-channels";

  if (pairing.dominantChannel === "phonon-bcs") {
    // Conventional: phonon Tc is the answer, spin fluctuations negligible
    tcCombined = tcPhonon;
    combinationMethod = "single-channel";
    notes.push("Phonon BCS dominant — using phonon Tc directly");
  } else if (pairing.dominantChannel === "spin-fluctuation" && pairing.pairingSymmetry === "d-wave") {
    // Cuprate-like: spin fluctuations dominate, phonons are pair-breaking
    // Tc = Tc_sf - α·Tc_phonon where α ≈ 0.1-0.3 (weak destructive interference)
    const alpha = 0.15;
    tcCombined = Math.max(0, tcSpinFluctuation - alpha * tcPhonon);
    combinationMethod = "destructive";
    notes.push(
      `d-wave: spin fluctuations drive pairing, phonons weakly pair-breaking. ` +
      `Tc = Tc_sf - ${alpha}·Tc_ph = ${tcSpinFluctuation.toFixed(1)} - ${(alpha * tcPhonon).toFixed(1)} = ${tcCombined.toFixed(1)} K. ` +
      `Ref: Chubukov et al., PRB 78, 134512 (2008).`
    );
  } else if (pairing.dominantChannel === "mixed-phonon-spin") {
    // s± pnictide-like: both channels contribute constructively
    // Tc ≈ sqrt(Tc_ph² + Tc_sf²) — quadrature addition for independent channels
    tcCombined = Math.sqrt(tcPhonon ** 2 + tcSpinFluctuation ** 2);
    combinationMethod = "constructive";
    notes.push(
      `s±-wave: phonon and spin channels constructive. ` +
      `Tc = sqrt(${tcPhonon.toFixed(1)}² + ${tcSpinFluctuation.toFixed(1)}²) = ${tcCombined.toFixed(1)} K`
    );
  } else {
    // Unknown or orbital: take the maximum
    tcCombined = Math.max(tcPhonon, tcSpinFluctuation);
    combinationMethod = "max-of-channels";
    notes.push(
      `Uncertain channel interference — using max(Tc_ph=${tcPhonon.toFixed(1)}, Tc_sf=${tcSpinFluctuation.toFixed(1)}) = ${tcCombined.toFixed(1)} K`
    );
  }

  return {
    tcPhonon,
    tcSpinFluctuation,
    tcCombined: Number(tcCombined.toFixed(2)),
    combinationMethod,
    dominantChannel: pairing.dominantChannel,
    pairingSymmetry: pairing.pairingSymmetry,
    notes,
  };
}
