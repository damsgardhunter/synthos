/**
 * LLM-guided structure advisor for CSP candidate generation.
 *
 * Before generating random candidates, asks gpt-4o-mini (cheap, fast) for
 * structural hints about the target composition: expected crystal structure
 * type, likely space group, approximate lattice parameters, per-pair minimum
 * distances, and element coordination roles.
 *
 * These hints feed into the CSP generators:
 *   - AIRSS: per-pair MINSEP values instead of global minimum
 *   - PyXtal: bias toward likely space groups
 *   - Cage seeder: element role assignments (cage-center vs vertex vs network)
 *   - Volume ensemble: lattice estimate cross-check
 *
 * The random generators still produce thousands of unique structures — the
 * hints just set better parameter bounds so fewer candidates are born dead.
 *
 * Results are cached to disk per formula — each composition is queried once, ever.
 * Falls back to empty hints (no bias) if OpenAI is unavailable.
 */

import OpenAI from "openai";
import * as fs from "fs";
import * as path from "path";
import { openaiCircuitOpen, recordOpenAIFail, recordOpenAISuccess } from "../learning/openai-circuit";

// ─── Types ───────────────────────────────────────────────────────────

export interface StructureAdvice {
  /** Expected crystal structure type */
  structureType: string;
  /** Most likely space group number (0 = unknown) */
  likelySpaceGroup: number;
  /** Alternative space groups worth trying (weighted) */
  alternativeSpaceGroups: number[];
  /** Estimated lattice parameters in Å */
  estimatedLattice: { a: number; b: number; c: number };
  /** Per-pair minimum distances in Å (for MINSEP) */
  pairDistances: Record<string, number>;
  /** Element coordination hints */
  coordinationHints: Record<string, {
    role: string;
    typicalCoordination: number;
    nearestNeighbor: string;
  }>;
  /** Free-form reasoning from the model */
  reasoning: string;
  /** Whether this advice came from cache vs fresh call */
  fromCache: boolean;
  /** Confidence: "high" if model was specific, "low" if hedging */
  confidence: "high" | "medium" | "low";
}

const EMPTY_ADVICE: StructureAdvice = {
  structureType: "unknown",
  likelySpaceGroup: 0,
  alternativeSpaceGroups: [],
  estimatedLattice: { a: 0, b: 0, c: 0 },
  pairDistances: {},
  coordinationHints: {},
  reasoning: "",
  fromCache: false,
  confidence: "low",
};

// ─── OpenAI client ───────────────────────────────────────────────────

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  timeout: 15_000,
  maxRetries: 0,
});

// ─── Disk cache ──────────────────────────────────────────────────────

const CACHE_DIR = "/tmp/qe_calculations/structure_advice_cache";

function getCachePath(formula: string, pressureGpa: number): string {
  const pressureBucket = Math.round(pressureGpa / 10) * 10; // bucket by 10 GPa
  return path.join(CACHE_DIR, `${formula}_${pressureBucket}GPa.json`);
}

function loadFromCache(formula: string, pressureGpa: number): StructureAdvice | null {
  try {
    const cachePath = getCachePath(formula, pressureGpa);
    if (!fs.existsSync(cachePath)) return null;
    const data = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
    if (!data.structureType || !data.pairDistances) return null;
    return { ...data, fromCache: true };
  } catch {
    return null;
  }
}

function saveToCache(formula: string, pressureGpa: number, advice: StructureAdvice): void {
  try {
    if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
    const cachePath = getCachePath(formula, pressureGpa);
    fs.writeFileSync(cachePath, JSON.stringify(advice, null, 2));
  } catch {}
}

// ─── Prompt ──────────────────────────────────────────────────────────

function buildPrompt(formula: string, pressureGpa: number, elements: string[]): string {
  const pressureStr = pressureGpa > 0 ? `at ${pressureGpa} GPa pressure` : "at ambient pressure";
  return `You are a computational materials scientist. Given the composition ${formula} ${pressureStr}, predict the most likely crystal structure.

Return ONLY valid JSON with these fields:
{
  "structureType": "clathrate-cage" | "perovskite" | "layered" | "A15" | "BCC" | "FCC" | "HCP" | "rocksalt" | "fluorite" | "spinel" | "pyrochlore" | "heusler" | "tetragonal-bodycentered" | "hexagonal-layered" | "cage-hydride" | "molecular" | "other",
  "likelySpaceGroup": <integer space group number, 0 if unsure>,
  "alternativeSpaceGroups": [<up to 3 alternative SG numbers>],
  "estimatedLattice": { "a": <Å>, "b": <Å>, "c": <Å> },
  "pairDistances": { "${elements.map((a, i) => elements.slice(i).map(b => `${a}-${b}`)).flat().join('": <Å>, "')}": <Å> },
  "coordinationHints": {
    ${elements.map(el => `"${el}": { "role": "<cage-center|cage-vertex|cage-network|framework|interstitial|layer-atom|chain-atom>", "typicalCoordination": <integer>, "nearestNeighbor": "<element>" }`).join(",\n    ")}
  },
  "reasoning": "<1-2 sentence explanation of why this structure>",
  "confidence": "high" | "medium" | "low"
}

Be specific about pair distances — these will be used as minimum interatomic distances for structure generation. For hydrides under pressure, H-H distances are typically 1.0-1.5 Å in cage structures, 0.74 Å in H₂ molecules. Metal-H distances depend on the metal size and pressure.

For space groups: use the international number (1-230). Common hydride space groups: 225 (Fm-3m, clathrate), 229 (Im-3m, sodalite/BCC), 194 (P63/mmc, hex), 139 (I4/mmm), 221 (Pm-3m, perovskite).`;
}

// ─── Main entry ──────────────────────────────────────────────────────

/**
 * Get structure advice for a composition. Returns cached result if available,
 * otherwise makes one OpenAI call. Falls back to empty advice on failure.
 */
export async function getStructureAdvice(
  formula: string,
  pressureGpa: number,
  elements: string[],
): Promise<StructureAdvice> {
  // Check disk cache first
  const cached = loadFromCache(formula, pressureGpa);
  if (cached) {
    console.log(`[StructAdvisor] ${formula} @ ${pressureGpa} GPa: loaded from cache (${cached.structureType}, SG=${cached.likelySpaceGroup})`);
    return cached;
  }

  // Check circuit breaker
  if (openaiCircuitOpen()) {
    console.log(`[StructAdvisor] ${formula}: OpenAI circuit open, using empty advice`);
    return { ...EMPTY_ADVICE };
  }

  // Make the call
  const startMs = Date.now();
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "You are a computational materials scientist specializing in crystal structure prediction and high-pressure physics. Always respond with valid JSON only.",
        },
        {
          role: "user",
          content: buildPrompt(formula, pressureGpa, elements),
        },
      ],
      temperature: 0.2, // low temperature for deterministic structural predictions
      max_tokens: 800,
      response_format: { type: "json_object" },
    });

    recordOpenAISuccess();
    const content = response.choices[0]?.message?.content ?? "";
    const elapsed = Date.now() - startMs;

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      console.log(`[StructAdvisor] ${formula}: JSON parse failed (${elapsed}ms)`);
      recordOpenAIFail();
      return { ...EMPTY_ADVICE };
    }

    // Validate and sanitize
    const advice: StructureAdvice = {
      structureType: typeof parsed.structureType === "string" ? parsed.structureType : "unknown",
      likelySpaceGroup: typeof parsed.likelySpaceGroup === "number" && parsed.likelySpaceGroup >= 0 && parsed.likelySpaceGroup <= 230
        ? Math.round(parsed.likelySpaceGroup) : 0,
      alternativeSpaceGroups: Array.isArray(parsed.alternativeSpaceGroups)
        ? parsed.alternativeSpaceGroups.filter((sg: any) => typeof sg === "number" && sg >= 1 && sg <= 230).slice(0, 5)
        : [],
      estimatedLattice: validateLattice(parsed.estimatedLattice),
      pairDistances: validatePairDistances(parsed.pairDistances),
      coordinationHints: typeof parsed.coordinationHints === "object" && parsed.coordinationHints
        ? parsed.coordinationHints : {},
      reasoning: typeof parsed.reasoning === "string" ? parsed.reasoning.slice(0, 500) : "",
      fromCache: false,
      confidence: ["high", "medium", "low"].includes(parsed.confidence) ? parsed.confidence : "low",
    };

    // Cache to disk
    saveToCache(formula, pressureGpa, advice);

    const pairStr = Object.entries(advice.pairDistances).map(([k, v]) => `${k}=${v}`).join(", ");
    console.log(
      `[StructAdvisor] ${formula} @ ${pressureGpa} GPa: ${advice.structureType}, SG=${advice.likelySpaceGroup}, ` +
      `a=${advice.estimatedLattice.a.toFixed(2)} Å, pairs=[${pairStr}], conf=${advice.confidence} (${elapsed}ms)`
    );

    return advice;
  } catch (err: any) {
    recordOpenAIFail();
    const elapsed = Date.now() - startMs;
    console.log(`[StructAdvisor] ${formula}: OpenAI call failed (${elapsed}ms): ${err.message?.slice(0, 100)}`);
    return { ...EMPTY_ADVICE };
  }
}

// ─── Validation helpers ──────────────────────────────────────────────

function validateLattice(raw: any): { a: number; b: number; c: number } {
  if (!raw || typeof raw !== "object") return { a: 0, b: 0, c: 0 };
  const a = typeof raw.a === "number" && raw.a > 0.5 && raw.a < 30 ? raw.a : 0;
  const b = typeof raw.b === "number" && raw.b > 0.5 && raw.b < 30 ? raw.b : a;
  const c = typeof raw.c === "number" && raw.c > 0.5 && raw.c < 50 ? raw.c : a;
  return { a, b, c };
}

function validatePairDistances(raw: any): Record<string, number> {
  if (!raw || typeof raw !== "object") return {};
  const result: Record<string, number> = {};
  for (const [pair, dist] of Object.entries(raw)) {
    if (typeof dist === "number" && dist > 0.3 && dist < 10) {
      result[pair] = Number(dist.toFixed(2));
    }
  }
  return result;
}
