/**
 * Screening Tier Assignment
 *
 * Determines how many candidates to generate (preview/standard/deep/publication)
 * based on the material's characteristics, priority, and prior knowledge.
 *
 * Decision factors:
 * - Priority level from job queue (9999 = critical benchmark, 50 = normal discovery)
 * - Known structure (has DFT-quality Wyckoff? → lighter search, positions already good)
 * - Material family (hydrides need more exploration than known intermetallics)
 * - Pressure (high-P structures are harder → more candidates)
 * - Prior DFT results (already ran successfully? → don't re-explore from scratch)
 * - Element count (more complex stoichiometry → more candidates needed)
 * - Adaptive learning data (if we've learned this family, search smarter not harder)
 */

import type { ScreeningTier, ScreeningTierConfig } from "./csp-types";
import { SCREENING_TIERS } from "./csp-types";
import { classifyFamily, classifyPressureBin, type MaterialFamily } from "./adaptive-learning";
import { hasKnownStructure } from "../learning/known-structures";

export interface TierDecision {
  tier: ScreeningTier;
  config: ScreeningTierConfig;
  reason: string;
  airssBudget: number;
  pyxtalBudget: number;
  timeoutMs: number;
}

/**
 * Assign a screening tier to a material.
 *
 * @param formula - Chemical formula
 * @param elements - Element list
 * @param pressureGPa - Target pressure
 * @param priority - Job queue priority (higher = more important)
 * @param hasCompletedDFT - Whether this formula has already completed DFT before
 * @param jobType - "scf" | "scf_tsc" | "scf_retry"
 */
export function assignTier(
  formula: string,
  elements: string[],
  pressureGPa: number,
  priority: number = 50,
  hasCompletedDFT: boolean = false,
  jobType: string = "scf",
): TierDecision {
  const family = classifyFamily(elements);
  const isKnown = hasKnownStructure(formula);
  const nElements = elements.length;
  const hasH = elements.includes("H");

  // --- Critical benchmarks (priority 9999) ---
  // These are LaH10, CaH6, Nb3Sn etc. — we have literature positions,
  // no need for massive structure search. Use preview or skip CSP entirely.
  if (priority >= 9000) {
    if (isKnown) {
      return makeTier("preview", "critical benchmark with known structure — minimal CSP", 30, 10);
    }
    return makeTier("preview", "critical benchmark — preview CSP", 50, 20);
  }

  // --- Already completed DFT successfully ---
  // Re-runs (retries, TSC follow-ups) don't need full structure search
  if (hasCompletedDFT || jobType === "scf_retry") {
    return makeTier("preview", "re-run/retry — positions already established", 20, 10);
  }

  // --- TSC candidates (topological superconductor screening) ---
  // These are speculative — need more exploration
  if (jobType === "scf_tsc") {
    return makeTier("standard", "TSC candidate — broader exploration", 100, 40);
  }

  // --- Known structure exists ---
  // Literature positions are DFT-quality, CSP is supplementary
  if (isKnown) {
    return makeTier("preview", "known structure — CSP supplementary only", 30, 10);
  }

  // --- High-pressure hydrides (most important for discovery) ---
  if (hasH && pressureGPa >= 100) {
    if (nElements >= 3) {
      // Ternary+ high-P hydrides: maximum exploration
      return makeTier("deep", "ternary high-P hydride — deep exploration", 200, 80);
    }
    // Binary high-P hydrides
    return makeTier("standard", "binary high-P hydride — standard exploration", 100, 40);
  }

  // --- Moderate-pressure hydrides ---
  if (hasH && pressureGPa >= 20) {
    return makeTier("standard", "moderate-P hydride — standard", 100, 40);
  }

  // --- Ambient-pressure hydrides ---
  if (hasH) {
    return makeTier("preview", "ambient hydride — preview", 50, 20);
  }

  // --- Complex stoichiometry (4+ elements) ---
  if (nElements >= 4) {
    return makeTier("standard", "complex stoichiometry — broader search", 100, 40);
  }

  // --- Iron pnictides, chalcogenides ---
  if (family === "pnictide" || family === "chalcogenide") {
    return makeTier("standard", `${family} — standard exploration`, 80, 30);
  }

  // --- Default: simple binary/ternary compounds ---
  return makeTier("preview", "standard material — preview", 50, 20);
}

function makeTier(
  tier: ScreeningTier,
  reason: string,
  airssBudget: number,
  pyxtalBudget: number,
): TierDecision {
  const config = SCREENING_TIERS[tier];

  // Timeout scales with budget: ~2s per AIRSS structure, ~5s per PyXtal
  const timeoutAirss = Math.max(30000, airssBudget * 2000);
  const timeoutPyxtal = Math.max(30000, pyxtalBudget * 5000);
  const timeoutMs = Math.max(timeoutAirss, timeoutPyxtal);

  return {
    tier,
    config,
    reason,
    airssBudget: Math.min(airssBudget, config.airssTotalCap),
    pyxtalBudget: Math.min(pyxtalBudget, config.pyxtalTotalCap),
    timeoutMs,
  };
}

/**
 * Log the tier assignment decision.
 */
export function logTierDecision(formula: string, decision: TierDecision): void {
  console.log(`[CSP-Tier] ${formula}: ${decision.tier} tier (AIRSS=${decision.airssBudget}, PyXtal=${decision.pyxtalBudget}) — ${decision.reason}`);
}
