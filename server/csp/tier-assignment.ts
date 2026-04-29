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
  // no need for massive structure search. Minimal CSP.
  if (priority >= 9000 && isKnown) {
    return makeTier("preview", "critical benchmark with known structure — minimal CSP");
  }
  if (priority >= 9000) {
    return makeTier("preview", "critical benchmark — preview CSP");
  }

  // --- Already completed DFT successfully ---
  if (hasCompletedDFT || jobType === "scf_retry") {
    return makeTier("preview", "re-run/retry — positions already established");
  }

  // --- Known structure exists ---
  if (isKnown) {
    return makeTier("preview", "known structure — CSP supplementary only");
  }

  // --- High-pressure hydrides (most important for discovery) ---
  if (hasH && pressureGPa >= 100) {
    if (nElements >= 3) {
      // Ternary+ high-P hydrides: maximum exploration
      // deep tier = AIRSS up to 10,000, PyXtal up to 1,000
      return makeTier("deep", "ternary high-P hydride — deep exploration");
    }
    // Binary high-P hydrides
    return makeTier("standard", "binary high-P hydride — standard exploration");
  }

  // --- TSC candidates (topological superconductor screening) ---
  if (jobType === "scf_tsc") {
    return makeTier("standard", "TSC candidate — broader exploration");
  }

  // --- Moderate-pressure hydrides ---
  if (hasH && pressureGPa >= 20) {
    return makeTier("standard", "moderate-P hydride — standard");
  }

  // --- Ambient-pressure hydrides ---
  if (hasH) {
    return makeTier("preview", "ambient hydride — preview");
  }

  // --- Complex stoichiometry (4+ elements) ---
  if (nElements >= 4) {
    return makeTier("standard", "complex stoichiometry — broader search");
  }

  // --- Iron pnictides, chalcogenides ---
  if (family === "pnictide" || family === "chalcogenide") {
    return makeTier("standard", `${family} — standard exploration`);
  }

  // --- Default: simple binary/ternary compounds ---
  return makeTier("preview", "standard material — preview");
}

function makeTier(
  tier: ScreeningTier,
  reason: string,
): TierDecision {
  const config = SCREENING_TIERS[tier];

  // Use the tier's full budget caps — these are the designed values:
  // preview: AIRSS=2500, PyXtal=250
  // standard: AIRSS=5000, PyXtal=500
  // deep: AIRSS=10000, PyXtal=1000
  // publication: AIRSS=10000, PyXtal=1000
  const airssBudget = config.airssTotalCap;
  const pyxtalBudget = config.pyxtalTotalCap;

  // Timeout scaling:
  // AIRSS: ~100ms per structure, cap at 10 min
  // PyXtal: includes SG pre-filtering (~30s) + generation (~2-5s per structure).
  // Pre-filtering tests each SG once — adds ~30-60s upfront but makes
  // generation 5× faster by only sampling compatible SGs.
  // Cap at 10 min for preview, 20 min for deep.
  const timeoutAirss = Math.min(600000, Math.max(30000, airssBudget * 100));
  const timeoutPyxtal = Math.min(
    tier === "deep" || tier === "publication" ? 1200000 : 600000, // 20 min deep, 10 min preview
    Math.max(60000, pyxtalBudget * 2000 + 60000), // 2s per target + 60s for pre-filter
  );
  const timeoutMs = Math.max(timeoutAirss, timeoutPyxtal);

  return {
    tier,
    config,
    reason,
    airssBudget,
    pyxtalBudget,
    timeoutMs,
  };
}

/**
 * Log the tier assignment decision.
 */
export function logTierDecision(formula: string, decision: TierDecision): void {
  console.log(`[CSP-Tier] ${formula}: ${decision.tier} tier (AIRSS=${decision.airssBudget}, PyXtal=${decision.pyxtalBudget}) — ${decision.reason}`);
}
