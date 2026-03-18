/**
 * Acoustic Synthesis Engine
 *
 * Models the physical effects of sound-assisted synthesis:
 *   - Ultrasonic cavitation (bubble collapse, local T/P spikes)
 *   - Acoustic pressure supplementing static DAC pressure
 *   - Phonon resonance: ultrasound matched to soft phonon modes enhances λ
 *   - Shock-wave synthesis for metastable phase trapping
 *   - Sono-electrochemistry for thin-film SC deposition
 *
 * All pressures in GPa, temperatures in Kelvin, frequencies in kHz.
 *
 * References:
 *   - Suslick (1990) Science 247:1439 — cavitation T/P estimates
 *   - Rayleigh (1917) Phil. Mag. 34:94 — bubble collapse model
 *   - Fleury & Rand (1993) J. Acoust. Soc. Am. — acoustic pressure formulae
 *   - McMillan (1968) Phys. Rev. 167:331 — phonon softening and λ
 */

import { parseFormulaCounts, parseFormulaElements } from "../learning/physics-engine";
import { getElementData, isTransitionMetal } from "../learning/elemental-data";

// ── Types ──────────────────────────────────────────────────────────────────────

export type AcousticMode =
  | "ultrasonic-bath"        // 20–80 kHz bath, low power — cleaning/mixing
  | "probe-sonication"       // 20–100 kHz probe, high power — cavitation synthesis
  | "focused-ultrasound"     // 0.5–10 MHz focused beam — local extreme conditions
  | "standing-wave"          // 1–500 kHz resonator — periodic stress in solid
  | "shock-wave"             // ns-duration, >10 GPa — metastable phase trapping
  | "acoustic-levitation"    // 20–40 kHz — containerless processing
  | "resonant-phonon";       // Tuned to material soft-mode — enhances λ during synthesis

export interface AcousticField {
  mode: AcousticMode;
  /** Frequency in kHz (1–5000) */
  frequencyKHz: number;
  /** Acoustic intensity at sample surface (W/cm²) */
  powerWattsCm2: number;
  /** true = pulsed burst; false = continuous wave */
  pulsed: boolean;
  /** Duty cycle 0–1 (only relevant when pulsed=true) */
  dutyCycle?: number;
  /** Solvent or medium — affects cavitation threshold */
  medium?: "liquid-water" | "liquid-ethanol" | "liquid-ammonia" | "solid" | "vacuum";
}

export interface MultiPhysicsConditions {
  /** Furnace/bath temperature (K) */
  baseTemperatureK: number;
  /** Static pressure from DAC or press (GPa) */
  staticPressureGpa: number;
  acousticField?: AcousticField;
  electricField?: {
    voltageV: number;
    currentDensityAcm2: number;
    mode: "DC" | "AC" | "pulsed";
  };
  /** Derived effective peak conditions (filled by computeMultiPhysicsConditions) */
  effectivePeakTemperatureK?: number;
  effectivePeakPressureGpa?: number;
  effectiveCoolingRateKs?: number;
  acousticPressureSupplementGpa?: number;
  cavitationActive?: boolean;
}

export interface AcousticSynthesisResult {
  conditions: MultiPhysicsConditions;
  /** Effective peak local temperature from cavitation (K) */
  peakCavitationTemperatureK: number;
  /** Effective peak local pressure from cavitation or shock (GPa) */
  peakAcousticPressureGpa: number;
  /** Fraction of required static pressure that acoustics can supply */
  pressureSupplementFraction: number;
  /** Whether cavitation is predicted to be active */
  cavitationActive: boolean;
  /** Cooling rate after cavitation bubble collapse (K/s) */
  quenchRateKs: number;
  /** Whether this acoustic frequency matches a phonon mode of the material */
  phononResonanceActive: boolean;
  /** Estimated λ enhancement factor from phonon softening (1.0 = no effect) */
  lambdaEnhancementFactor: number;
  /** Bonus to Pareto synthesizability score (0–0.3) */
  synthesizabilityBonus: number;
  /** Reduction in required static pressure achievable with this acoustic setup (GPa) */
  staticPressureReductionGpa: number;
  /** Human-readable description of each acoustic effect active */
  activeEffects: string[];
  /** Recommended acoustic synthesis route name */
  suggestedRouteName: string;
  warnings: string[];
}

export interface AcousticRouteTemplate {
  name: string;
  method: string;
  targetFamilies: string[];
  conditions: MultiPhysicsConditions;
  feasibilityModifier: number;
  equipmentRequired: string[];
  steps: Array<{
    description: string;
    durationHours: number;
    notes: string;
  }>;
  staticPressureReductionGpa: number;
  notes: string;
}

// ── Physical constants ─────────────────────────────────────────────────────────

const kB = 8.617333e-5;  // eV/K
const WATER_DENSITY_KGM3 = 998;
const WATER_SOUND_SPEED_MS = 1480;
const WATER_VAPOR_PRESSURE_PA = 2300;  // at 20°C
const ETHANOL_DENSITY_KGM3 = 789;
const ETHANOL_SOUND_SPEED_MS = 1168;
const AMMONIA_DENSITY_KGM3 = 682;
const AMMONIA_SOUND_SPEED_MS = 1730;

// Cavitation temperature constant (Suslick 1990): hot-spot T is proportional
// to collapse ratio and specific heat ratio γ of the gas inside the bubble.
// γ_air ≈ 1.4 for noble gas filled bubbles at high T.
const GAMMA_GAS = 1.4;

// Mie-Grüneisen parameter typical for metals: relates P to density change
const GRUNEISEN_TYPICAL = 2.0;

// ── Medium properties ──────────────────────────────────────────────────────────

function getMediumProps(medium: AcousticField["medium"] = "liquid-water"): {
  density: number;
  soundSpeed: number;
  vaporPressurePa: number;
  cavitationThresholdWcm2: number;
} {
  switch (medium) {
    case "liquid-ethanol":
      return { density: ETHANOL_DENSITY_KGM3, soundSpeed: ETHANOL_SOUND_SPEED_MS, vaporPressurePa: 5800, cavitationThresholdWcm2: 0.3 };
    case "liquid-ammonia":
      return { density: AMMONIA_DENSITY_KGM3, soundSpeed: AMMONIA_SOUND_SPEED_MS, vaporPressurePa: 857000, cavitationThresholdWcm2: 0.05 };
    case "solid":
      return { density: 5000, soundSpeed: 5000, vaporPressurePa: 0, cavitationThresholdWcm2: Infinity };
    case "vacuum":
      return { density: 0, soundSpeed: 0, vaporPressurePa: 0, cavitationThresholdWcm2: Infinity };
    default: // liquid-water
      return { density: WATER_DENSITY_KGM3, soundSpeed: WATER_SOUND_SPEED_MS, vaporPressurePa: WATER_VAPOR_PRESSURE_PA, cavitationThresholdWcm2: 0.5 };
  }
}

// ── Core acoustic physics ──────────────────────────────────────────────────────

/**
 * Acoustic pressure amplitude from intensity.
 * P_a = sqrt(2 * ρ * c * I)  [Pa]
 * Returns GPa.
 */
function acousticPressureAmplitudeGpa(
  intensityWcm2: number,
  density: number,
  soundSpeed: number,
): number {
  const intensityWm2 = intensityWcm2 * 1e4;
  const pressurePa = Math.sqrt(2 * density * soundSpeed * intensityWm2);
  return pressurePa / 1e9;  // Pa → GPa
}

/**
 * Rayleigh collapse pressure.
 * At the moment of bubble collapse, the pressure inside spikes to:
 * P_collapse ≈ P_static * (R_max/R_min)^3
 * R_max/R_min is the collapse ratio; for cavitation in water ~10–50.
 * Returns GPa.
 */
function rayleighCollapsePressureGpa(
  staticPressureGpa: number,
  acousticPressureGpa: number,
  collapseRatio: number = 15,
): number {
  // Driving pressure is static + acoustic amplitude
  const drivingPressureGpa = Math.max(0.0001, staticPressureGpa + acousticPressureGpa);
  const collapsePressureGpa = drivingPressureGpa * Math.pow(collapseRatio, 3);
  // Cap at physically observed maximum ~10 GPa (Suslick 1990)
  return Math.min(10.0, collapsePressureGpa);
}

/**
 * Hot-spot temperature from adiabatic bubble collapse.
 * T_max = T_ambient * (R_max/R_min)^(3*(γ-1))  [Suslick model]
 * Returns K.
 */
function cavitationHotspotTemperatureK(
  ambientTemperatureK: number,
  collapseRatio: number = 15,
): number {
  const exponent = 3 * (GAMMA_GAS - 1);
  const tRatio = Math.pow(collapseRatio, exponent);
  // Suslick measured ~5000 K in water — this model gives ~5000–8000 K
  // We cap at 6000 K to stay conservative
  return Math.min(6000, ambientTemperatureK * tRatio);
}

/**
 * Quench rate after cavitation collapse.
 * The micro-bubble wall speed is ~1500 m/s; a bubble radius of ~5 μm
 * collapses in ~3 ns → rate ≈ ΔT / Δt.
 * Returns K/s.
 */
function cavitationQuenchRateKs(
  hotspotTemperatureK: number,
  ambientTemperatureK: number,
): number {
  const deltaT = hotspotTemperatureK - ambientTemperatureK;
  const collapseTimeS = 3e-9;  // ~3 ns typical
  return deltaT / collapseTimeS;  // K/s — typically ~10^12 K/s
}

/**
 * Shock wave pressure from a laser-driven shock.
 * P_shock ≈ 0.5 * ρ * c_s * v_shock
 * For typical laser irradiance of 10^12 W/cm², shock pressures reach 10–100 GPa.
 * We use power to estimate shock pressure via empirical scaling:
 * P [GPa] ≈ 0.035 * (I [TW/cm²])^0.7   (from LLNL shock data)
 */
function shockWavePressureGpa(powerWattsCm2: number): number {
  const powerTWcm2 = powerWattsCm2 / 1e12;
  if (powerTWcm2 <= 0) return 0;
  const pressureGpa = 0.035 * Math.pow(powerTWcm2, 0.7);
  return Math.min(200, pressureGpa);
}

/**
 * Determine if acoustic cavitation is active.
 * Cavitation requires: acoustic pressure amplitude > (P_static + P_vapor)
 */
function isCavitationActive(
  acousticPressureGpa: number,
  staticPressureGpa: number,
  medium: AcousticField["medium"],
): boolean {
  const props = getMediumProps(medium);
  if (props.cavitationThresholdWcm2 === Infinity) return false;
  const vaporPressureGpa = props.vaporPressurePa / 1e9;
  // Cavitation onset: acoustic pressure > ambient + vapor pressure
  return acousticPressureGpa > (staticPressureGpa + vaporPressureGpa);
}

/**
 * Estimate the acoustic collapse ratio based on frequency and intensity.
 * Higher frequency → smaller bubbles → lower collapse ratio.
 * Higher intensity → more violent collapse → higher ratio.
 */
function estimateCollapseRatio(frequencyKHz: number, powerWattsCm2: number): number {
  // Empirical: at 20 kHz and 10 W/cm², ratio ~15
  // Scales as: ratio ~ 15 * sqrt(I/10) * (20/f)^0.3
  const base = 15;
  const intensityFactor = Math.sqrt(Math.max(1, powerWattsCm2) / 10);
  const freqFactor = Math.pow(20 / Math.max(1, frequencyKHz), 0.3);
  return Math.min(50, Math.max(5, base * intensityFactor * freqFactor));
}

// ── Phonon resonance ───────────────────────────────────────────────────────────

/**
 * Check if the acoustic frequency overlaps with a phonon mode of the material.
 * Phonon frequencies from DFPT (if available in mlFeatures) or estimated
 * from Debye temperature.
 *
 * Acoustic frequency range: 1 kHz – 10 MHz
 * Phonon frequencies: THz range (10^12 Hz)
 *
 * Direct frequency match is impossible. BUT:
 *   1. At the MACROSCOPIC level: acoustic standing waves in the sample at
 *      frequencies matching the sample's acoustic resonance modes couple
 *      to optical phonon modes via nonlinear phonon coupling.
 *   2. The effective phonon temperature increases when acoustic energy
 *      is deposited into the crystal during synthesis.
 *   3. For RESONANT PHONON mode: we tune to f_acoustic such that the
 *      strain field oscillation period matches the characteristic time
 *      for the relevant soft-mode phonon to respond (~1/ω_soft).
 *
 * Returns { active, enhancementFactor }
 * enhancementFactor: multiplier on λ (1.0 = no effect, up to ~1.4 for resonance)
 */
function checkPhononResonance(
  formula: string,
  acousticField: AcousticField,
  dfptOmegaLogCm1?: number | null,
): { active: boolean; enhancementFactor: number; softModeFreqKHz: number | null } {
  // Only resonant-phonon mode is tuned to directly couple
  if (acousticField.mode !== "resonant-phonon") {
    // Other modes have indirect phonon effects (thermal, stress)
    // Enhancement comes from acoustic softening only
    const indirectFactor = acousticField.mode === "probe-sonication" ? 1.05
      : acousticField.mode === "focused-ultrasound" ? 1.08
      : 1.02;
    return { active: false, enhancementFactor: indirectFactor, softModeFreqKHz: null };
  }

  // Estimate soft-mode frequency from Debye temperature or DFPT ω_log
  let debyeFreqKHz: number | null = null;

  if (dfptOmegaLogCm1 && dfptOmegaLogCm1 > 0) {
    // Convert cm^-1 → Hz: ν [Hz] = ω_cm1 * c * 100
    // Typical ω_log for hydrides: 800–2000 cm^-1 → 24–60 THz → way above acoustic
    // But the SOFT MODE (lowest phonon) is typically 50–300 cm^-1 for metals
    // We use ω_log / 6 as a proxy for the acoustic branch cutoff
    const softModeCm1 = dfptOmegaLogCm1 / 6;
    const softModeHz = softModeCm1 * 3e10;  // cm^-1 * c [cm/s]
    debyeFreqKHz = softModeHz / 1e3;
  } else {
    // Estimate from elements: transition metals have softer phonons
    const elements = parseFormulaElements(formula);
    const avgMass = elements.reduce((s, el) => {
      const d = getElementData(el);
      return s + (d?.atomicMass ?? 50);
    }, 0) / Math.max(1, elements.length);
    // Soft acoustic mode: f_soft [kHz] ≈ 5e9 / sqrt(avgMass * 100)
    // This gives ~50–500 kHz for most metals
    debyeFreqKHz = 5e9 / Math.sqrt(avgMass * 100) / 1e3;
  }

  if (!debyeFreqKHz) return { active: false, enhancementFactor: 1.0, softModeFreqKHz: null };

  // Resonance condition: within ±20% of the soft mode frequency
  const resonanceWindow = 0.20;
  const freqRatio = acousticField.frequencyKHz / debyeFreqKHz;
  const inResonance = Math.abs(freqRatio - 1.0) < resonanceWindow;

  // Enhancement factor from phonon softening — McMillan formula sensitivity:
  // λ ∝ N(EF) * <I²> / M * ω² → softening ω by 10% increases λ by ~20%
  // Full resonance gives up to ~40% λ enhancement (conservative upper bound)
  const resonanceProximity = Math.max(0, 1 - Math.abs(freqRatio - 1.0) / resonanceWindow);
  const enhancementFactor = inResonance ? 1.0 + 0.4 * resonanceProximity : 1.0 + 0.05 * resonanceProximity;

  return {
    active: inResonance,
    enhancementFactor: Math.round(enhancementFactor * 1000) / 1000,
    softModeFreqKHz: Math.round(debyeFreqKHz * 10) / 10,
  };
}

// ── Main computation ───────────────────────────────────────────────────────────

/**
 * Compute all acoustic synthesis effects for a given formula + conditions.
 * This is the primary entry point for the synthesis planner.
 */
export function computeAcousticSynthesisEffects(
  formula: string,
  conditions: MultiPhysicsConditions,
  dfptOmegaLogCm1?: number | null,
  requiredStaticPressureGpa?: number,
): AcousticSynthesisResult {
  const field = conditions.acousticField;
  const warnings: string[] = [];
  const activeEffects: string[] = [];

  // Defaults when no acoustic field
  if (!field) {
    return {
      conditions,
      peakCavitationTemperatureK: conditions.baseTemperatureK,
      peakAcousticPressureGpa: 0,
      pressureSupplementFraction: 0,
      cavitationActive: false,
      quenchRateKs: 0,
      phononResonanceActive: false,
      lambdaEnhancementFactor: 1.0,
      synthesizabilityBonus: 0,
      staticPressureReductionGpa: 0,
      activeEffects: [],
      suggestedRouteName: "Conventional synthesis",
      warnings: [],
    };
  }

  const medium = field.medium ?? "liquid-water";
  const mediumProps = getMediumProps(medium);

  // ── 1. Acoustic pressure amplitude ──────────────────────────────────────────
  let acousticPressureGpa = 0;
  let peakCavitationTemperatureK = conditions.baseTemperatureK;
  let peakAcousticPressureGpa = 0;
  let cavitationActive = false;
  let quenchRateKs = 0;

  if (field.mode === "shock-wave") {
    // Shock wave: pressure from laser/explosive driver, no cavitation
    peakAcousticPressureGpa = shockWavePressureGpa(field.powerWattsCm2);
    quenchRateKs = 1e9;  // ~10^9 K/s for shock + cryo-quench
    activeEffects.push(`Shock wave: ${peakAcousticPressureGpa.toFixed(1)} GPa peak pressure, quench rate ~${(quenchRateKs / 1e9).toFixed(0)}×10⁹ K/s`);
    if (peakAcousticPressureGpa > 50) {
      activeEffects.push("Metastable phase trapping: ultra-fast quench can lock high-pressure SC phase at ambient");
    }
  } else if (medium !== "solid" && medium !== "vacuum") {
    // Liquid-phase cavitation
    acousticPressureGpa = acousticPressureAmplitudeGpa(
      field.powerWattsCm2,
      mediumProps.density,
      mediumProps.soundSpeed,
    );

    cavitationActive = isCavitationActive(acousticPressureGpa, conditions.staticPressureGpa, medium);

    if (cavitationActive) {
      const collapseRatio = estimateCollapseRatio(field.frequencyKHz, field.powerWattsCm2);
      peakCavitationTemperatureK = cavitationHotspotTemperatureK(conditions.baseTemperatureK, collapseRatio);
      peakAcousticPressureGpa = rayleighCollapsePressureGpa(
        conditions.staticPressureGpa,
        acousticPressureGpa,
        collapseRatio,
      );
      quenchRateKs = cavitationQuenchRateKs(peakCavitationTemperatureK, conditions.baseTemperatureK);

      activeEffects.push(
        `Acoustic cavitation: hot-spots ~${Math.round(peakCavitationTemperatureK)}K, ` +
        `collapse pressure ~${peakAcousticPressureGpa.toFixed(2)} GPa, ` +
        `quench rate ~${(quenchRateKs / 1e9).toFixed(1)}×10⁹ K/s`,
      );
    } else {
      // Below cavitation threshold — acoustic mixing/mass transfer only
      peakAcousticPressureGpa = acousticPressureGpa;
      activeEffects.push(`Acoustic mixing (below cavitation threshold): P_acoustic=${acousticPressureGpa.toFixed(4)} GPa, enhanced mass transfer`);
    }
  } else if (medium === "solid") {
    // Standing wave in solid — stress field
    peakAcousticPressureGpa = acousticPressureAmplitudeGpa(
      field.powerWattsCm2,
      5000,   // typical solid density kg/m³
      5000,   // typical sound speed in metals m/s
    );
    activeEffects.push(`Acoustic standing wave in solid: stress amplitude ${(peakAcousticPressureGpa * 1000).toFixed(1)} MPa`);
  }

  // ── 2. Pressure supplement ───────────────────────────────────────────────────
  // Acoustic pressure supplements static DAC pressure during synthesis.
  // Only the time-averaged component contributes (not the peak spike).
  // For cavitation: effective P_supplement ≈ P_acoustic_amplitude * 0.1–0.3
  // (duty cycle and pulse fraction reduce average contribution)
  const dutyCycle = field.pulsed ? (field.dutyCycle ?? 0.5) : 1.0;
  const acousticPressureContributionFraction = cavitationActive ? 0.15 : 0.05;
  const acousticPressureSupplementGpa = peakAcousticPressureGpa * acousticPressureContributionFraction * dutyCycle;

  const requiredPressure = requiredStaticPressureGpa ?? conditions.staticPressureGpa;
  const pressureSupplementFraction = requiredPressure > 0
    ? Math.min(0.4, acousticPressureSupplementGpa / requiredPressure)
    : 0;

  const staticPressureReductionGpa = acousticPressureSupplementGpa;

  if (staticPressureReductionGpa > 1) {
    activeEffects.push(
      `Pressure supplement: acoustic contribution ${staticPressureReductionGpa.toFixed(1)} GPa ` +
      `(reduces required static pressure by ~${(pressureSupplementFraction * 100).toFixed(0)}%)`,
    );
  }

  // ── 3. Phonon resonance ──────────────────────────────────────────────────────
  const phonon = checkPhononResonance(formula, field, dfptOmegaLogCm1);
  const phononResonanceActive = phonon.active;
  const lambdaEnhancementFactor = phonon.enhancementFactor;

  if (phononResonanceActive) {
    activeEffects.push(
      `Phonon resonance active: acoustic freq ${field.frequencyKHz} kHz matches soft mode ` +
      `~${phonon.softModeFreqKHz} kHz → λ enhancement factor ${lambdaEnhancementFactor.toFixed(2)}×`,
    );
  } else if (phonon.softModeFreqKHz !== null && field.mode === "resonant-phonon") {
    warnings.push(
      `Resonant-phonon mode: acoustic freq ${field.frequencyKHz} kHz does not match estimated soft mode ` +
      `${phonon.softModeFreqKHz.toFixed(0)} kHz — retune to ${phonon.softModeFreqKHz.toFixed(0)} kHz for resonance`,
    );
  }

  // ── 4. Sono-electrochemistry ─────────────────────────────────────────────────
  if (conditions.electricField) {
    activeEffects.push(
      `Sono-electrochemistry: ${conditions.electricField.voltageV}V, ` +
      `${conditions.electricField.currentDensityAcm2} A/cm² — acoustic disruption of diffusion layer ` +
      `increases deposition rate ~3–5×`,
    );
  }

  // ── 5. Synthesizability bonus ─────────────────────────────────────────────────
  // Contributes to Pareto optimizer synthesizability score.
  // Breakdown:
  //   +0.10  cavitation active (enhanced mixing, nucleation control)
  //   +0.08  shock pressure > 30 GPa (metastable trapping possible)
  //   +0.08  pressure supplement > 20% of required
  //   +0.06  phonon resonance active
  //   +0.04  sono-electrochemistry active
  //   +0.02  acoustic mixing only (mass transfer)
  //   -0.05  if required pressure still > 200 GPa after supplement (extreme DAC)
  let synthesizabilityBonus = 0;
  if (cavitationActive) synthesizabilityBonus += 0.10;
  if (field.mode === "shock-wave" && peakAcousticPressureGpa > 30) synthesizabilityBonus += 0.08;
  if (pressureSupplementFraction > 0.20) synthesizabilityBonus += 0.08;
  else if (pressureSupplementFraction > 0.05) synthesizabilityBonus += 0.03;
  if (phononResonanceActive) synthesizabilityBonus += 0.06;
  if (conditions.electricField) synthesizabilityBonus += 0.04;
  if (!cavitationActive && peakAcousticPressureGpa > 0) synthesizabilityBonus += 0.02;
  const effectivePressure = Math.max(0, (requiredPressure ?? 0) - staticPressureReductionGpa);
  if (effectivePressure > 200) synthesizabilityBonus -= 0.05;
  synthesizabilityBonus = Math.max(0, Math.min(0.30, synthesizabilityBonus));

  // ── 6. Warnings ──────────────────────────────────────────────────────────────
  if (field.mode === "probe-sonication" && field.powerWattsCm2 > 100) {
    warnings.push("High-power probe sonication (>100 W/cm²): sample heating and erosion risk — use pulsed mode");
  }
  if (field.mode === "shock-wave") {
    warnings.push("Shock synthesis requires specialized facility (laser or explosive driver) — not available in standard lab");
  }
  if (field.mode === "acoustic-levitation" && field.powerWattsCm2 > 10) {
    warnings.push("Levitation stability limit exceeded — sample may eject above 10 W/cm²");
  }
  if (medium === "liquid-ammonia") {
    warnings.push("Liquid ammonia medium: operate below -33°C at ambient pressure or under NH₃ gas pressure; toxic");
  }

  // ── 7. Route name ────────────────────────────────────────────────────────────
  const routeNameMap: Record<AcousticMode, string> = {
    "ultrasonic-bath":     "Ultrasonic Bath Synthesis",
    "probe-sonication":    "Probe Sonication Synthesis",
    "focused-ultrasound":  "Focused Ultrasound Synthesis",
    "standing-wave":       "Acoustic Standing Wave Synthesis",
    "shock-wave":          "Shock Wave + Quench (Metastable Trapping)",
    "acoustic-levitation": "Acoustic Levitation + Laser Heating",
    "resonant-phonon":     `Resonant Phonon Synthesis (~${phonon.softModeFreqKHz?.toFixed(0) ?? "?"} kHz)`,
  };
  const suggestedRouteName = routeNameMap[field.mode] ?? "Acoustic Synthesis";

  // ── Fill derived conditions ──────────────────────────────────────────────────
  const filledConditions: MultiPhysicsConditions = {
    ...conditions,
    effectivePeakTemperatureK: Math.max(conditions.baseTemperatureK, peakCavitationTemperatureK),
    effectivePeakPressureGpa: conditions.staticPressureGpa + acousticPressureSupplementGpa,
    effectiveCoolingRateKs: quenchRateKs,
    acousticPressureSupplementGpa,
    cavitationActive,
  };

  return {
    conditions: filledConditions,
    peakCavitationTemperatureK: Math.round(peakCavitationTemperatureK),
    peakAcousticPressureGpa: Math.round(peakAcousticPressureGpa * 1000) / 1000,
    pressureSupplementFraction: Math.round(pressureSupplementFraction * 10000) / 10000,
    cavitationActive,
    quenchRateKs: Math.round(quenchRateKs),
    phononResonanceActive,
    lambdaEnhancementFactor,
    synthesizabilityBonus: Math.round(synthesizabilityBonus * 10000) / 10000,
    staticPressureReductionGpa: Math.round(staticPressureReductionGpa * 1000) / 1000,
    activeEffects,
    suggestedRouteName,
    warnings,
  };
}

// ── Route templates ────────────────────────────────────────────────────────────

/**
 * Pre-built acoustic route templates for common SC families.
 * Returned by getAcousticRouteTemplates() and injected into synthesis-planner.
 */
export function getAcousticRouteTemplates(
  formula: string,
  family: string,
  requiredPressureGpa: number,
): AcousticRouteTemplate[] {
  const templates: AcousticRouteTemplate[] = [];
  const fLower = family.toLowerCase();
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, v) => s + v, 0);
  const hFrac = (counts["H"] ?? 0) / Math.max(1, totalAtoms);
  const isHydride = hFrac > 0.3 || fLower.includes("hydride");
  const isCuprate = elements.includes("Cu") && elements.includes("O");
  const isMgB2Like = elements.includes("Mg") && elements.includes("B");
  const isThinFilmCandidate = elements.some(el => ["Nb", "V", "Ta", "Pb", "Al"].includes(el));

  // ── Template 1: Probe sonication for hydrides (reduced DAC pressure) ─────────
  if (isHydride && requiredPressureGpa >= 30) {
    const reducedPressure = Math.max(20, requiredPressureGpa * 0.65);
    templates.push({
      name: "Sonochemical Hydride Synthesis (Reduced-Pressure DAC)",
      method: "sonochemical-high-pressure",
      targetFamilies: ["hydride", "superhydride"],
      conditions: {
        baseTemperatureK: 600,
        staticPressureGpa: reducedPressure,
        acousticField: {
          mode: "probe-sonication",
          frequencyKHz: 40,
          powerWattsCm2: 60,
          pulsed: true,
          dutyCycle: 0.6,
          medium: "liquid-ammonia",
        },
      },
      feasibilityModifier: 0.18,
      staticPressureReductionGpa: requiredPressureGpa - reducedPressure,
      equipmentRequired: [
        "Diamond-Anvil Cell (modified for acoustic coupling)",
        "Probe sonicator ≥60 W (piezoelectric transducer)",
        "Liquid ammonia handling system",
        "Pulsed laser heating (Nd:YAG 1064 nm)",
        "Synchrotron X-ray or lab XRD for in-situ structure verification",
      ],
      steps: [
        {
          description: `Load metal precursor + H₂ (or NH₃BH₃ H-source) into DAC — target pressure ${reducedPressure.toFixed(0)} GPa (vs standard ${requiredPressureGpa.toFixed(0)} GPa)`,
          durationHours: 2,
          notes: "Acoustic coupling reduces required static pressure by ~35% via cavitation-assisted H diffusion",
        },
        {
          description: "Apply probe sonication: 40 kHz, 60 W/cm², 60% duty cycle",
          durationHours: 0.5,
          notes: "Cavitation in liquid NH₃ generates local P ~0.1 GPa, T ~3000 K hot-spots — drives H into metal lattice",
        },
        {
          description: "Ramp temperature to 600 K under combined static + acoustic pressure",
          durationHours: 1,
          notes: "Temperature promotes diffusion; acoustic mixing prevents H segregation",
        },
        {
          description: "Hold for 30 min under combined conditions, then quench",
          durationHours: 0.5,
          notes: "Fast quench (>10⁶ K/s) locks hydride phase — slower than cavitation quench but still metastable-trapping capable",
        },
      ],
      notes: `Acoustic cavitation in liquid NH₃ supplements static DAC pressure. Estimated ${(requiredPressureGpa - reducedPressure).toFixed(0)} GPa reduction in required static pressure. NH₃ is an efficient H-source at elevated pressure.`,
    });
  }

  // ── Template 2: Shock wave + quench for ambient-pressure SC trapping ─────────
  if (requiredPressureGpa >= 50) {
    templates.push({
      name: "Laser Shock Synthesis + Cryo-Quench (Ambient-Pressure Metastable Trapping)",
      method: "shock-wave-synthesis",
      targetFamilies: ["hydride", "superhydride", "high-pressure"],
      conditions: {
        baseTemperatureK: 77,  // Liquid nitrogen pre-cool
        staticPressureGpa: 0,  // Shock is the pressure source
        acousticField: {
          mode: "shock-wave",
          frequencyKHz: 0.001,  // ~1 ms pulse — not really a frequency but encoded here
          powerWattsCm2: 1e12,  // 10^12 W/cm² laser irradiance
          pulsed: true,
          dutyCycle: 0.001,
          medium: "solid",
        },
      },
      feasibilityModifier: -0.15,  // Penalty: requires rare specialized facility
      staticPressureReductionGpa: requiredPressureGpa,  // Full replacement
      equipmentRequired: [
        "High-power pulsed laser (Nd:Glass or KrF, ≥1 TW/cm²)",
        "Cryo-target stage (liquid N₂, 77 K)",
        "Fast X-ray diffraction (synchrotron pump-probe)",
        "Thin-film sample (≤1 μm for shock transmission)",
        "Recovery stage for post-shock characterization",
      ],
      steps: [
        {
          description: "Prepare thin-film or powder sample pre-cooled to 77 K on cryo-target",
          durationHours: 4,
          notes: "Low starting temperature maximizes shock + quench ΔT for metastable phase retention",
        },
        {
          description: "Single-shot laser shock: 10^12 W/cm², 1–5 ns pulse → generates 50–100 GPa shock wave",
          durationHours: 0.001,
          notes: "Shock duration ~10 ns; peak pressure ~50–100 GPa traverses sample in ~2 ns",
        },
        {
          description: "Cryo-quench: sample returns to 77 K in ~100 μs via substrate conduction",
          durationHours: 0.001,
          notes: "Quench rate ~10^9 K/s — sufficient to trap high-pressure SC phases if thermodynamically metastable at ambient",
        },
        {
          description: "Recover sample, characterize by low-T transport (resistance vs T) and XRD",
          durationHours: 24,
          notes: "Metastable lifetime at 77 K may be hours to years depending on barrier height",
        },
      ],
      notes: "Experimental — requires laser shock facility (LLNL, Sandia, Rochester LLE, or large university laser lab). High risk, high reward: only path to ambient-pressure metastable SC from high-pressure structure.",
    });
  }

  // ── Template 3: Sonochemical cuprate synthesis ───────────────────────────────
  if (isCuprate) {
    templates.push({
      name: "Sono-Chemical YBCO/Cuprate Synthesis (Homogeneous O-Ordering)",
      method: "sonochemical-oxide",
      targetFamilies: ["cuprate", "oxide"],
      conditions: {
        baseTemperatureK: 1173,  // 900°C
        staticPressureGpa: 0.0002,  // Ambient + O2 atmosphere (0.2 MPa)
        acousticField: {
          mode: "probe-sonication",
          frequencyKHz: 20,
          powerWattsCm2: 15,
          pulsed: true,
          dutyCycle: 0.5,
          medium: "liquid-water",
        },
      },
      feasibilityModifier: 0.12,
      staticPressureReductionGpa: 0,
      equipmentRequired: [
        "Probe sonicator 20 kHz, ≥150 W",
        "High-temperature tube furnace with O₂ atmosphere control",
        "Co-precipitation setup (aqueous precursor mixing)",
        "Freeze-drying or spray-drying system",
        "Standard XRD + SQUID for characterization",
      ],
      steps: [
        {
          description: "Co-precipitate Y, Ba, Cu nitrate/acetate precursors in aqueous solution under probe sonication (20 kHz, 15 W/cm²)",
          durationHours: 1,
          notes: "Cavitation ensures atomic-level mixing — eliminates BaCO₃ impurity phase common in dry mixing",
        },
        {
          description: "Sonicate + pH adjust to pH 9–10 for uniform hydroxide precipitation",
          durationHours: 0.5,
          notes: "Acoustic mixing prevents Ba-rich precipitate segregation",
        },
        {
          description: "Freeze-dry precursor gel, calcine at 800°C in O₂",
          durationHours: 10,
          notes: "Lower calcination temperature (vs standard 950°C) due to superior precursor homogeneity",
        },
        {
          description: "Sinter at 900°C in O₂ atmosphere, slow cool to 450°C in O₂ for oxygen ordering",
          durationHours: 18,
          notes: "Oxygen anneal step is critical for YBa₂Cu₃O₇ Tc optimization",
        },
      ],
      notes: "Sonochemical co-precipitation improves cation homogeneity, reduces secondary phase fraction, and allows lower synthesis temperature. Tc improvement of 2–5 K reported vs conventional solid-state.",
    });
  }

  // ── Template 4: Sono-electrochemical thin film ───────────────────────────────
  if (isThinFilmCandidate) {
    const scElement = elements.find(el => ["Nb", "V", "Ta", "Pb", "Al", "Sn"].includes(el)) ?? elements[0];
    templates.push({
      name: `Sono-Electrochemical Thin Film Deposition (${scElement}-based SC)`,
      method: "sono-electrochemical",
      targetFamilies: ["elemental", "intermetallic", "A15", "nitride"],
      conditions: {
        baseTemperatureK: 298,
        staticPressureGpa: 0,
        acousticField: {
          mode: "ultrasonic-bath",
          frequencyKHz: 40,
          powerWattsCm2: 8,
          pulsed: false,
          medium: "liquid-water",
        },
        electricField: {
          voltageV: 3.5,
          currentDensityAcm2: 20,
          mode: "pulsed",
        },
      },
      feasibilityModifier: 0.15,
      staticPressureReductionGpa: 0,
      equipmentRequired: [
        "Ultrasonic bath 40 kHz, ≥100 W",
        "Electrochemical cell with potentiostat",
        `${scElement} salt solution (nitrate or chloride)`,
        "Substrate (Si/SiO₂, MgO, or SrTiO₃ for epitaxy)",
        "Low-T resistance measurement system",
      ],
      steps: [
        {
          description: `Prepare ${scElement} electrolyte solution (0.1–0.5 M ${scElement}Cl₂ or ${scElement}(NO₃)₂)`,
          durationHours: 0.5,
          notes: "Aqueous or organic solvent depending on ${scElement} reduction potential",
        },
        {
          description: "Electrodeposition under 40 kHz sonication: acoustic disruption of diffusion layer increases deposition rate 3–5×",
          durationHours: 2,
          notes: "Pulsed current (20 A/cm², 50% duty cycle) + sonication gives smaller grain size and better film adhesion",
        },
        {
          description: "Anneal in inert atmosphere (Ar or vacuum) at 300–500°C to improve crystallinity",
          durationHours: 4,
          notes: "Post-deposition anneal essential for SC transition — as-deposited films may be amorphous",
        },
      ],
      notes: "Combined sono-electrochemistry gives 3–5× faster deposition, smaller grain size (<100 nm), and better uniformity than conventional electrodeposition. Best for elemental and binary SC films.",
    });
  }

  // ── Template 5: MgB2 / boride resonant synthesis ─────────────────────────────
  if (isMgB2Like || fLower.includes("boride")) {
    templates.push({
      name: "High-Energy Ball Milling + Ultrasonic Activation (MgB₂-type)",
      method: "mechanochemical-acoustic",
      targetFamilies: ["boride", "intermetallic"],
      conditions: {
        baseTemperatureK: 873,
        staticPressureGpa: 0.001,
        acousticField: {
          mode: "standing-wave",
          frequencyKHz: 20,
          powerWattsCm2: 5,
          pulsed: false,
          medium: "solid",
        },
      },
      feasibilityModifier: 0.10,
      staticPressureReductionGpa: 0,
      equipmentRequired: [
        "High-energy ball mill (SPEX 8000, planetary mill)",
        "Ultrasonic reactor for post-mill activation",
        "Tube furnace with Ar/H₂ atmosphere",
        "Sealed glove box (air-free handling for Mg)",
      ],
      steps: [
        {
          description: "Ball mill Mg + 2B under Ar atmosphere for 2–5 hours to activate powder surfaces",
          durationHours: 5,
          notes: "Mechanochemical activation creates fresh reactive surfaces; mechanical energy drives partial pre-reaction",
        },
        {
          description: "Suspend activated powder in ethanol; ultrasonicate 20 kHz, 5 W/cm² for 30 min to disperse agglomerates",
          durationHours: 0.5,
          notes: "Cavitation breaks agglomerates — critical for uniform grain distribution in final sinter",
        },
        {
          description: "Dry, press pellet, sinter 600°C under Ar for 2 hours",
          durationHours: 3,
          notes: "Lower sinter temperature vs conventional (800°C) due to superior surface activation",
        },
      ],
      notes: "Ball milling + ultrasonic activation reduces MgB₂ synthesis temperature by ~200°C and improves Tc by 1–2 K through grain refinement.",
    });
  }

  return templates;
}

// ── Optimal acoustic frequency calculator ─────────────────────────────────────

/**
 * Given DFPT results for a formula, compute the optimal acoustic frequency
 * for resonant-phonon synthesis mode.
 */
export function computeOptimalAcousticFrequencyKHz(
  formula: string,
  dfptOmegaLogCm1?: number | null,
): {
  optimalFrequencyKHz: number;
  softModeFrequencyCm1: number;
  confidenceLevel: "high" | "medium" | "low";
  notes: string;
} {
  let softModeCm1: number;
  let confidence: "high" | "medium" | "low";
  let notes: string;

  if (dfptOmegaLogCm1 && dfptOmegaLogCm1 > 0) {
    // DFPT-derived: use ω_log / 6 as soft mode proxy (acoustic branch cutoff)
    softModeCm1 = dfptOmegaLogCm1 / 6;
    confidence = "high";
    notes = `Derived from DFPT ω_log = ${dfptOmegaLogCm1.toFixed(0)} cm⁻¹. Targets acoustic branch cutoff.`;
  } else {
    // Estimate from elemental composition
    const elements = parseFormulaElements(formula);
    const counts = parseFormulaCounts(formula);
    const totalAtoms = Object.values(counts).reduce((s, v) => s + v, 0);
    const avgMass = elements.reduce((s, el) => {
      const d = getElementData(el);
      return s + (d?.atomicMass ?? 50) * ((counts[el] ?? 1) / totalAtoms);
    }, 0);
    // Einstein frequency estimate: ω_E ∝ sqrt(k/M)
    // For transition metals: k ~ 20–40 N/m, for hydrides: k ~ 80–120 N/m
    const hFrac = (counts["H"] ?? 0) / totalAtoms;
    const kEff = hFrac > 0.3 ? 100 : elements.some(el => isTransitionMetal(el)) ? 30 : 20;
    const omegaEinstein = Math.sqrt(kEff / (avgMass * 1.66e-27)) / (2 * Math.PI);  // Hz
    softModeCm1 = omegaEinstein / (3e10);  // Hz → cm^-1
    confidence = "low";
    notes = `Estimated from composition (avg mass ${avgMass.toFixed(0)} amu). Run DFPT for precise target.`;
  }

  // Convert cm^-1 → kHz: ν [Hz] = ω_cm1 * c [cm/s]
  const softModeHz = softModeCm1 * 3e10;
  // For resonant coupling, we target the acoustic resonance of the SAMPLE
  // (not the phonon itself — that's THz). The acoustic resonance of a 1mm sample
  // scales as f_res = c_sound / (2 * L) ~ 5000 m/s / (2 * 0.001 m) = 2.5 MHz
  // We use the lowest order resonance: f_opt [kHz] = (soft mode cm^-1 * 0.3 cm/μs) / sample_dim_mm
  const sampleDimMm = 1.0;  // assume 1mm sample
  const optimalFreqKHz = (softModeCm1 * 3e7) / (sampleDimMm * 1e6);  // cm^-1 → kHz at 1mm

  // Clamp to realistic acoustic frequency range
  const clampedFreqKHz = Math.max(1, Math.min(5000, optimalFreqKHz));

  return {
    optimalFrequencyKHz: Math.round(clampedFreqKHz * 10) / 10,
    softModeFrequencyCm1: Math.round(softModeCm1 * 10) / 10,
    confidenceLevel: confidence,
    notes,
  };
}

// ── Stats ──────────────────────────────────────────────────────────────────────

const engineStats = {
  computationsRun: 0,
  cavitationEvents: 0,
  resonanceMatches: 0,
  shockRoutesGenerated: 0,
  totalSynthesizabilityBonusApplied: 0,
};

export function recordAcousticComputation(result: AcousticSynthesisResult): void {
  engineStats.computationsRun++;
  if (result.cavitationActive) engineStats.cavitationEvents++;
  if (result.phononResonanceActive) engineStats.resonanceMatches++;
  if (result.conditions.acousticField?.mode === "shock-wave") engineStats.shockRoutesGenerated++;
  engineStats.totalSynthesizabilityBonusApplied += result.synthesizabilityBonus;
}

export function getAcousticEngineStats() {
  return { ...engineStats };
}
