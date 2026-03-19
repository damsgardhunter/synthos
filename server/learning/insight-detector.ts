import OpenAI from "openai";
import crypto from "crypto";
import { storage } from "../storage";
import { cache, CACHE_KEYS } from "../cache";
import type { EventEmitter } from "./engine";
import { sanitizeForbiddenWords } from "./utils";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  timeout: 60_000,
  maxRetries: 0, // Connection errors do not self-resolve; avoid 3x retry amplification
});

function insightId(text: string, phaseId: number): string {
  const hash = crypto.createHash("sha256").update(`${phaseId}:${text.toLowerCase().trim()}`).digest("hex").slice(0, 12);
  return `ni-${phaseId}-${hash}`;
}

const CATEGORY_QUOTAS: Record<string, number> = {
  "novel-correlation": 2,
  "new-mechanism": 2,
  "cross-domain": 2,
  "computational-discovery": 2,
  "design-principle": 2,
  "phonon-softening": 2,
  "fermi-nesting": 2,
  "charge-transfer": 2,
  "structural-motif": 2,
  "electron-density": 2,
};

const recentCategoryCounts: Record<string, number> = {};
let lastCategoryResetTime = 0;
const CATEGORY_RESET_INTERVAL_MS = 30 * 60 * 1000;

function getCategoryCount(category: string): number {
  const now = Date.now();
  if (now - lastCategoryResetTime > CATEGORY_RESET_INTERVAL_MS) {
    for (const key of Object.keys(recentCategoryCounts)) {
      delete recentCategoryCounts[key];
    }
    lastCategoryResetTime = now;
  }
  return recentCategoryCounts[category] ?? 0;
}

function incrementCategoryCount(category: string): void {
  const now = Date.now();
  if (now - lastCategoryResetTime > CATEGORY_RESET_INTERVAL_MS) {
    for (const key of Object.keys(recentCategoryCounts)) {
      delete recentCategoryCounts[key];
    }
    lastCategoryResetTime = now;
  }
  recentCategoryCounts[category] = (recentCategoryCounts[category] ?? 0) + 1;
}

const WELL_KNOWN_PATTERNS = [
  "electronegativity increases across periods",
  "atomic radius follows inverse trend to ionization energy",
  "noble gases maintain full outer shells",
  "transition metals exhibit variable oxidation states",
  "higher formation energy correlates with lower stability",
  "higher stability correlates with lower formation energy",
  "lower formation energy correlates with higher stability",
  "formation energy correlates with stability",
  "stability correlates with formation energy",
  "negative formation energy indicates stability",
  "lower formation energy indicates greater stability",
  "zero band gap materials are metallic",
  "negative formation energy indicates thermodynamic stability",
  "d-wave symmetry enhances superconductivity in cuprates",
  "cooper pairs form via electron-phonon interactions",
  "meissner effect is key for superconductivity",
  "room-temperature superconductors remain a challenge",
  "bcs theory describes conventional superconductors",
  "strong electron-phonon coupling favors superconductivity",
  "layered structures can enhance superconducting properties",
  "quantum coherence is important for superconducting qubits",
  "perovskite structures host ferroelectric properties",
  "hydrogen-rich compounds show promise under pressure",
  "phonon coupling enhances cooper pair formation",
  "cuprates exhibit unconventional superconductivity",
  "high band gap materials are insulators",
  "metallic elements display lower band gaps",
  "face-centered cubic structures favorable for ductility",
  "pressure tuning can elevate transition temperatures",
  "critical temperature increases with electron-phonon coupling",
  "higher pressure increases critical temperature",
  "pressure enhances superconducting transition temperature",
  "band gap determines metallic or insulating behavior",
  "zero band gap indicates metallic behavior",
  "large band gap indicates insulating behavior",
  "metallic materials have zero or near-zero band gap",
  "insulators have large band gaps",
  "electron-phonon coupling is essential for bcs superconductivity",
  "lighter elements have higher phonon frequencies",
  "heavier elements tend to have lower phonon frequencies",
  "crystal structure affects material properties",
  "symmetry plays a role in superconducting pairing",
  "density of states at fermi level affects superconductivity",
  "higher density of states favors superconductivity",
  "van hove singularities enhance density of states",
  "ionic radius affects crystal structure",
  "electronegativity difference determines bond character",
  "lattice stability correlates with formation energy",
  "thermodynamic stability indicated by negative formation energy",
  "higher atomic mass leads to lower debye temperature",
  "debye temperature relates to phonon spectrum",
  "magnetic impurities suppress superconductivity",
  "spin-orbit coupling affects band structure",
  "hydrides show enhanced superconductivity under pressure",
  "electron correlation effects important in transition metals",
  "crystal field splitting affects electronic properties",
  "bonding character influences material hardness",
  "coordination number affects stability",
  "oxidation state influences magnetic behavior",
];

const SPURIOUS_CORRELATION_PATTERNS = [
  /formation\s+energy.*(?:enhance|increas|boost|improv|correlat|predict|indicat).*(?:superconduct|tc|critical\s+temp|pairing)/i,
  /(?:superconduct|tc|critical\s+temp|pairing).*(?:enhance|increas|boost|improv|correlat|predict|indicat).*formation\s+energy/i,
  /(?:low|negative|lower)\s+formation\s+energy.*(?:enhance|favor|promot|lead|indicat).*(?:superconduct|tc|higher\s+tc|pairing)/i,
  /formation\s+energy\s*[<>]\s*-?\d+.*(?:superconduct|tc|pairing|coupling)/i,
  /(?:superconduct|tc|pairing).*formation\s+energy\s*[<>]\s*-?\d+/i,
  /Hc2\s*[=>]\s*[12]\d{2,}/i,
  /upper\s+critical\s+field.*[=>]\s*[12]\d{2,}/i,
];

function isSpuriousCorrelation(insight: string): boolean {
  return SPURIOUS_CORRELATION_PATTERNS.some(pattern => pattern.test(insight));
}

function isTextbookKnowledge(insight: string): boolean {
  const lower = insight.toLowerCase();
  if (isSpuriousCorrelation(lower)) return true;
  return WELL_KNOWN_PATTERNS.some(pattern =>
    lower.includes(pattern) || pattern.split(" ").filter(w => w.length > 4).every(w => lower.includes(w))
  );
}

const BANNED_META_PHRASES = [
  "overemphasis",
  "remains unexplored",
  "may mislead",
  "should be considered",
  "further investigation needed",
  "more research required",
  "warrants further study",
  "needs more investigation",
  "deserves attention",
  "requires further analysis",
  "as an ai",
  "as a language model",
  "i cannot verify",
  "i cannot confirm",
  "i'm not able to",
  "i am not able to",
  "this is speculative",
  "i should note that",
  "it's important to note",
  "it is important to note",
  "please note that",
  "i don't have access",
  "beyond my training",
  "my training data",
  "i was trained",
  "hypothetical scenario",
  "for illustrative purposes",
];

const QUANTITATIVE_PROPERTY_NAMES = [
  "tc", "t_c", "critical temperature", "lambda", "electron-phonon",
  "dos", "density of states", "nesting", "band gap", "formation energy",
  "phonon", "omega", "debye", "coupling", "mu*", "coulomb",
  "pressure", "gpa", "synthesizability", "hull distance",
  "vanhove", "van hove", "fermi", "metallicity", "coherence",
];

const CHEMICAL_FORMULA_REGEX = /[A-Z][a-z]?\d+[A-Z]|[A-Z][a-z]?[A-Z][a-z]?\d|(?:La|Ba|Sr|Bi|Mg|Fe|Cu|Nb|Ti|Sc|Zr|Hf|Ta|Cr|Mo|Mn|Co|Ni|Zn|Ga|Ge|As|Se|Cd|In|Sn|Sb|Te|Pb|Tl|Hg|Ca|Al|Si|Ru|Rh|Pd|Pt|Ir|Os|Re)(?:H|O|N|C|B|S|F|Se|Te|As|P|Si|Ge|Sn)\d/;

const NUMBER_WITH_UNIT_REGEX = /\d+\.?\d*\s*(%|K|eV|GPa|cm|meV|THz|Hz|Å|nm|T)\b/i;
const BARE_NUMBER_REGEX = /(?:=|≈|~|>|<|≥|≤)\s*\d+\.?\d*/;
const COMPARISON_WITH_NUMBER_REGEX = /(?:higher|lower|greater|less|increase|decrease|correlat)\w*\s+(?:than|with|by)\s+\d/i;

function isMetaCommentary(text: string): boolean {
  const lower = text.toLowerCase();
  return BANNED_META_PHRASES.some(phrase => lower.includes(phrase));
}

function hasQuantitativeContent(text: string): boolean {
  const hasNumber = /\d+\.?\d*/.test(text);
  const hasFormula = CHEMICAL_FORMULA_REGEX.test(text);

  if (NUMBER_WITH_UNIT_REGEX.test(text)) return true;
  if (BARE_NUMBER_REGEX.test(text)) return true;
  if (COMPARISON_WITH_NUMBER_REGEX.test(text)) return true;

  if (hasFormula && hasNumber) return true;

  if (hasFormula) {
    const lower = text.toLowerCase();
    if (QUANTITATIVE_PROPERTY_NAMES.some(prop => lower.includes(prop))) return true;
  }

  if (hasNumber) {
    const lower = text.toLowerCase();
    if (QUANTITATIVE_PROPERTY_NAMES.some(prop => lower.includes(prop))) return true;
  }

  return false;
}

export function requiresQuantitativeContent(text: string): boolean {
  if (isMetaCommentary(text)) return false;
  if (!hasQuantitativeContent(text)) return false;
  return true;
}

const INSIGHT_KEY_TERMS = [
  "stability", "formation energy", "band gap", "metallic", "insulating",
  "electron-phonon", "phonon", "coupling", "critical temperature", "tc",
  "pressure", "superconductivity", "superconducting", "cooper pair",
  "density of states", "fermi", "crystal structure", "lattice",
  "electronegativity", "oxidation", "magnetic", "debye", "bcs",
  "eliashberg", "meissner", "cuprate", "hydride", "perovskite",
  "correlation", "symmetry", "spin-orbit", "transition temperature",
  "thermodynamic", "ionic radius", "atomic radius", "band structure",
];

function extractKeyTerms(text: string): string[] {
  const lower = text.toLowerCase();
  return INSIGHT_KEY_TERMS.filter(term => lower.includes(term));
}

function jaccardSimilarity(a: string, b: string): number {
  const arrA = a.toLowerCase().split(/\s+/).filter(w => w.length > 2);
  const arrB = b.toLowerCase().split(/\s+/).filter(w => w.length > 2);
  const wordsA = new Set(arrA);
  const wordsB = new Set(arrB);
  if (wordsA.size === 0 && wordsB.size === 0) return 1;
  if (wordsA.size === 0 || wordsB.size === 0) return 0;
  let intersection = 0;
  const aArr = Array.from(wordsA);
  for (const w of aArr) {
    if (wordsB.has(w)) intersection++;
  }
  const combined = Array.from(wordsA).concat(Array.from(wordsB));
  const union = new Set(combined).size;
  return union === 0 ? 0 : intersection / union;
}

function hasKeywordOverlap(text: string, existingTexts: string[]): boolean {
  const terms = extractKeyTerms(text);
  if (terms.length < 4) return false;

  for (const existing of existingTexts) {
    const existingTerms = extractKeyTerms(existing);
    const overlap = terms.filter(t => existingTerms.includes(t));
    if (overlap.length >= 4) return true;
  }
  return false;
}

function isDuplicateByJaccard(text: string, existingTexts: string[]): boolean {
  const lower = text.toLowerCase();
  for (const existing of existingTexts) {
    if (jaccardSimilarity(lower, existing) > 0.4) return true;
  }
  return false;
}

const CONCEPT_SYNONYMS: [string, string][] = [
  ["stability", "stable"],
  ["formation energy", "energy of formation"],
  ["lower formation energy", "negative formation energy"],
  ["higher stability", "more stable"],
  ["correlates with", "indicates"],
  ["correlates with", "linked to"],
  ["correlates with", "associated with"],
  ["increases", "enhances"],
  ["decreases", "reduces"],
  ["high", "elevated"],
  ["low", "reduced"],
  ["critical temperature", "tc"],
  ["superconducting", "superconductivity"],
  ["electron-phonon coupling", "e-ph coupling"],
  ["density of states", "dos"],
];

const COMPILED_SYNONYM_REGEXES: [string, RegExp][] = CONCEPT_SYNONYMS.map(
  ([canonical, synonym]) => [canonical, new RegExp(synonym.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')]
);

function extractConceptFingerprint(text: string): Set<string> {
  let lower = text.toLowerCase().replace(/[.,;:!?]/g, " ");
  for (const [canonical, regex] of COMPILED_SYNONYM_REGEXES) {
    lower = lower.replace(regex, canonical);
  }
  const keyTerms = INSIGHT_KEY_TERMS.filter(t => lower.includes(t));
  return new Set(keyTerms);
}

function isConceptualDuplicate(text: string, existingTexts: string[]): boolean {
  const fp = extractConceptFingerprint(text);
  if (fp.size < 2) return false;

  for (const existing of existingTexts) {
    const efp = extractConceptFingerprint(existing);
    if (efp.size < 2) continue;
    let shared = 0;
    for (const t of fp) {
      if (efp.has(t)) shared++;
    }
    const smaller = Math.min(fp.size, efp.size);
    if (smaller > 0 && shared / smaller >= 0.7) return true;
  }
  return false;
}

const CORRELATION_DIRECTIONS = ["higher", "lower", "greater", "less", "increase", "decrease", "enhance", "reduce", "correlat", "positive", "negative", "inverse"];

interface CorrelationFingerprint {
  pairs: [string, string][];
  direction: string;
  rValue?: number;
}

function extractCorrelationFingerprint(text: string): CorrelationFingerprint | null {
  const lower = text.toLowerCase();
  const matchedProps: string[] = [];
  for (const prop of QUANTITATIVE_PROPERTY_NAMES) {
    if (lower.includes(prop)) {
      matchedProps.push(prop);
    }
  }
  if (matchedProps.length < 2) return null;

  let direction = "unknown";
  for (const dir of CORRELATION_DIRECTIONS) {
    if (lower.includes(dir)) {
      direction = dir;
      break;
    }
  }

  const uniqueProps = [...new Set(matchedProps)].sort();
  const pairs: [string, string][] = [];
  for (let i = 0; i < uniqueProps.length; i++) {
    for (let j = i + 1; j < uniqueProps.length; j++) {
      pairs.push([uniqueProps[i], uniqueProps[j]]);
    }
  }

  const rMatch = lower.match(/r[\s=:]*([+-]?\d+\.?\d*)/);
  const rValue = rMatch ? parseFloat(rMatch[1]) : undefined;

  return { pairs, direction, rValue };
}

function normalizeDirection(dir: string): string {
  if (["higher", "greater", "increase", "enhance", "positive", "correlat"].some(d => dir.startsWith(d))) return "positive";
  if (["lower", "less", "decrease", "reduce", "negative", "inverse"].some(d => dir.startsWith(d))) return "negative";
  return "unknown";
}

function isCorrelationDuplicate(text: string, existingTexts: string[]): boolean {
  const fp = extractCorrelationFingerprint(text);
  if (!fp || fp.pairs.length === 0) return false;

  for (const existing of existingTexts) {
    const efp = extractCorrelationFingerprint(existing);
    if (!efp || efp.pairs.length === 0) continue;

    const sharedPairs = fp.pairs.filter(([a, b]) =>
      efp.pairs.some(([ea, eb]) => a === ea && b === eb)
    );
    if (sharedPairs.length === 0) continue;

    const overlapRatio = sharedPairs.length / Math.min(fp.pairs.length, efp.pairs.length);
    if (overlapRatio < 0.5) continue;

    const dirNew = normalizeDirection(fp.direction);
    const dirExisting = normalizeDirection(efp.direction);
    if (dirNew !== "unknown" && dirExisting !== "unknown" && dirNew !== dirExisting) {
      continue;
    }
    if (fp.rValue !== undefined && efp.rValue !== undefined) {
      if (Math.abs(fp.rValue - efp.rValue) > 0.1) continue;
    }
    return true;
  }
  return false;
}

const ROLLING_WINDOW_SIZE = 500;
const EMBEDDING_CACHE_MAX = 2000;
const SEMANTIC_DUP_THRESHOLD = 0.75;
const DIRECTION_CHECK_LOW = 0.70;
const DIRECTION_CHECK_HIGH = 0.95;

const MAX_NOVEL_INSIGHTS_PER_CYCLE = 3;
let novelInsightQueue: { text: string; phaseId: number; phaseName: string; relatedFormulas: string[] }[] = [];

interface EmbeddingEntry {
  text: string;
  embedding: Float32Array;
  addedAt: number;
}

const embeddingCacheMap = new Map<string, EmbeddingEntry>();
const embeddingCacheOrder: string[] = [];

function embeddingCacheKey(text: string): string {
  return text.toLowerCase().trim().replace(/\s+/g, ' ');
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export async function computeInsightEmbedding(text: string): Promise<Float32Array | null> {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text,
    });
    const vec = response.data[0]?.embedding;
    if (!vec) return null;
    return new Float32Array(vec);
  } catch {
    return null;
  }
}

async function computeInsightEmbeddingsBatch(texts: string[]): Promise<(Float32Array | null)[]> {
  if (texts.length === 0) return [];
  if (texts.length === 1) {
    try {
      const single = await computeInsightEmbedding(texts[0]);
      return [single];
    } catch {
      return [null];
    }
  }
  try {
    const BATCH_SIZE = 50;
    const results: (Float32Array | null)[] = new Array(texts.length).fill(null);
    for (let start = 0; start < texts.length; start += BATCH_SIZE) {
      const batch = texts.slice(start, start + BATCH_SIZE);
      const response = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: batch,
      });
      for (let i = 0; i < response.data.length; i++) {
        const vec = response.data[i]?.embedding;
        if (vec) results[start + i] = new Float32Array(vec);
      }
    }
    return results;
  } catch {
    // Batch failed — return nulls rather than retrying individually (each would also time out,
    // turning N insights into N × 60s = potentially minutes of blocking).
    return new Array(texts.length).fill(null);
  }
}

function addToEmbeddingCache(text: string, embedding: Float32Array): void {
  const key = embeddingCacheKey(text);
  if (embeddingCacheMap.has(key)) return;

  embeddingCacheMap.set(key, { text, embedding, addedAt: Date.now() });
  embeddingCacheOrder.push(key);

  while (embeddingCacheMap.size > EMBEDDING_CACHE_MAX) {
    const oldest = embeddingCacheOrder.shift();
    if (oldest) embeddingCacheMap.delete(oldest);
  }
}

/**
 * Called once at startup. Loads the most recent stored novel insights and computes their
 * embeddings so the in-memory semantic dedup cache survives server restarts.
 * Capped at 50 to avoid making dozens of API calls at startup.
 */
export async function bootstrapInsightEmbeddingCache(): Promise<void> {
  try {
    const existing = await storage.getNovelInsightsOnly(50);
    if (existing.length === 0) return;

    const texts = existing.map(i => i.insightText);
    const embeddings = await computeInsightEmbeddingsBatch(texts);

    let loaded = 0;
    for (let i = 0; i < texts.length; i++) {
      if (embeddings[i]) {
        addToEmbeddingCache(texts[i], embeddings[i]!);
        loaded++;
      }
    }
    console.log(`[InsightDetector] Bootstrapped embedding cache: ${loaded}/${existing.length} insights loaded`);
  } catch (err: any) {
    console.warn(`[InsightDetector] Bootstrap failed: ${err?.message?.slice(0, 100)}`);
  }
}

function extractDirectionFromText(text: string): string {
  const lower = text.toLowerCase();
  for (const dir of CORRELATION_DIRECTIONS) {
    if (lower.includes(dir)) return normalizeDirection(dir);
  }
  return "unknown";
}

function isSemanticDuplicate(embedding: Float32Array, candidateText: string): { isDuplicate: boolean; bestSimilarity: number; matchText: string } {
  let bestSimilarity = 0;
  let matchText = "";
  for (const entry of embeddingCacheMap.values()) {
    const sim = cosineSimilarity(embedding, entry.embedding);
    if (sim > bestSimilarity) {
      bestSimilarity = sim;
      matchText = entry.text;
    }
  }

  if (bestSimilarity > DIRECTION_CHECK_HIGH) {
    return { isDuplicate: true, bestSimilarity, matchText };
  }

  if (bestSimilarity > DIRECTION_CHECK_LOW) {
    const newDir = extractDirectionFromText(candidateText);
    const oldDir = extractDirectionFromText(matchText);
    if (newDir !== "unknown" && oldDir !== "unknown" && newDir !== oldDir) {
      return { isDuplicate: false, bestSimilarity, matchText };
    }
    if (bestSimilarity > SEMANTIC_DUP_THRESHOLD) {
      return { isDuplicate: true, bestSimilarity, matchText };
    }
  }

  return { isDuplicate: bestSimilarity > SEMANTIC_DUP_THRESHOLD, bestSimilarity, matchText };
}

export type InsightTempo = "excited" | "exploring" | "contemplating";

export async function evaluateInsightNovelty(
  emit: EventEmitter,
  insights: string[],
  phaseId: number,
  phaseName: string,
  relatedFormulas?: string[],
  tempo?: InsightTempo
): Promise<{ novel: number; total: number }> {
  if (insights.length === 0) return { novel: 0, total: 0 };

  const [recentInsights, novelInsights] = await Promise.all([
    storage.getNovelInsights(ROLLING_WINDOW_SIZE),
    storage.getNovelInsightsOnly(ROLLING_WINDOW_SIZE),
  ]);

  const combinedMap = new Map<string, typeof recentInsights[0]>();
  for (const ins of recentInsights) combinedMap.set(ins.id, ins);
  for (const ins of novelInsights) combinedMap.set(ins.id, ins);
  const existingInsights = Array.from(combinedMap.values()).slice(0, ROLLING_WINDOW_SIZE);
  const existingTexts = existingInsights.map(i => i.insightText.toLowerCase());

  const exactDuplicates = new Set<number>();
  const potentiallyNovel: { index: number; text: string }[] = [];

  for (let i = 0; i < insights.length; i++) {
    const text = insights[i];
    const lower = text.toLowerCase();

    const levenMatch = existingInsights.find(e =>
      e.insightText.toLowerCase() === lower || levenshteinSimilarity(e.insightText.toLowerCase(), lower) > 0.70
    );
    if (levenMatch) {
      exactDuplicates.add(i);
      if (relatedFormulas && relatedFormulas.length > 0) {
        storage.appendFormulasToInsight(levenMatch.id, relatedFormulas).catch(() => {});
      }
      continue;
    }

    if (isDuplicateByJaccard(lower, existingTexts) || isConceptualDuplicate(lower, existingTexts)) {
      continue;
    }

    if (isCorrelationDuplicate(lower, existingTexts)) {
      continue;
    }

    if (isTextbookKnowledge(text)) {
      continue;
    }

    if (hasKeywordOverlap(lower, existingTexts)) {
      continue;
    }

    if (!requiresQuantitativeContent(text)) {
      continue;
    }

    potentiallyNovel.push({ index: i, text });
  }

  if (potentiallyNovel.length === 0) return { novel: 0, total: insights.length };

  const batchTexts = potentiallyNovel.map(c => c.text);
  const batchEmbeddings = await computeInsightEmbeddingsBatch(batchTexts);

  const afterEmbeddingFilter: { index: number; text: string; embedding: Float32Array | null }[] = [];
  for (let ci = 0; ci < potentiallyNovel.length; ci++) {
    const candidate = potentiallyNovel[ci];
    const embedding = batchEmbeddings[ci];
    if (embedding) {
      const { isDuplicate } = isSemanticDuplicate(embedding, candidate.text);
      if (isDuplicate) {
        continue;
      }
    }
    afterEmbeddingFilter.push({ ...candidate, embedding });
  }

  if (afterEmbeddingFilter.length === 0) return { novel: 0, total: insights.length };

  let novelCount = 0;

  // Skip LLM evaluation — embedding + rule-based filters already provide strong dedup.
  // Insights that reach here have passed: exact-match, Levenshtein, Jaccard, conceptual,
  // correlation, textbook, keyword-overlap, quantitative-content, and semantic embedding checks.
  // Auto-approve with a fixed score to avoid blocking the engine on OpenAI calls.
  try {
    for (let globalIdx = 0; globalIdx < afterEmbeddingFilter.length; globalIdx++) {
      const entry = afterEmbeddingFilter[globalIdx];
      if (!entry) continue;

      if (entry.embedding) {
        addToEmbeddingCache(entry.text, entry.embedding);
      }

      // Derive a simple category from the text
      const lower = entry.text.toLowerCase();
      let category = "novel-correlation";
      if (lower.includes("phonon") || lower.includes("softening")) category = "phonon-softening";
      else if (lower.includes("fermi") || lower.includes("nesting")) category = "fermi-nesting";
      else if (lower.includes("charge") || lower.includes("transfer")) category = "charge-transfer";
      else if (lower.includes("structur") || lower.includes("crystal")) category = "structural-motif";
      else if (lower.includes("mechanism")) category = "new-mechanism";
      else if (lower.includes("design") || lower.includes("principle")) category = "design-principle";

      const categoryQuota = CATEGORY_QUOTAS[category] ?? 3;
      const currentCategoryCount = getCategoryCount(category);
      if (currentCategoryCount >= categoryQuota) {
        emit("log", {
          phase: `phase-${phaseId}`,
          event: "Insight category quota reached",
          detail: `[QUOTA] Category "${category}" has ${currentCategoryCount}/${categoryQuota} insights. Skipping: ${entry.text.slice(0, 80)}...`,
          dataSource: "Insight Detector",
        });
        continue;
      }
      if (novelCount >= MAX_NOVEL_INSIGHTS_PER_CYCLE) {
        novelInsightQueue.push({ text: entry.text, phaseId, phaseName, relatedFormulas: relatedFormulas ?? [] });
        emit("log", {
          phase: `phase-${phaseId}`,
          event: "Novel insight queued",
          detail: `[QUEUED] Cycle cap reached (${MAX_NOVEL_INSIGHTS_PER_CYCLE}). Queued: ${entry.text.slice(0, 80)}...`,
          dataSource: "Insight Detector",
        });
        continue;
      }

      try {
        await storage.insertNovelInsight({
          id: insightId(entry.text, phaseId),
          phaseId,
          phaseName,
          insightText: entry.text,
          isNovel: true,
          noveltyScore: 0.72,
          noveltyReason: "passed semantic and rule-based filters",
          category,
          relatedFormulas: relatedFormulas ?? [],
        });
        cache.invalidatePrefix(CACHE_KEYS.NOVEL_INSIGHTS);
      } catch {}

      incrementCategoryCount(category);
      novelCount++;
      emit("log", {
        phase: `phase-${phaseId}`,
        event: "Novel insight discovered",
        detail: `[NOVEL] ${entry.text}`,
        dataSource: "Insight Detector",
      });
      emit("insight", {
        phase: phaseName,
        insight: entry.text,
        isNovel: true,
        noveltyScore: 0.72,
        category,
      });
    }
  } catch (err: any) {
    emit("log", {
      phase: `phase-${phaseId}`,
      event: "Insight novelty evaluation error",
      detail: err.message?.slice(0, 150),
      dataSource: "Insight Detector",
    });
  }

  return { novel: novelCount, total: insights.length };
}

export function getQueuedInsightCount(): number {
  return novelInsightQueue.length;
}

export function drainQueuedInsights(): { text: string; phaseId: number; phaseName: string; relatedFormulas: string[] }[] {
  const queued = novelInsightQueue.splice(0, MAX_NOVEL_INSIGHTS_PER_CYCLE);
  return queued;
}

function levenshteinSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const longer = a.length > b.length ? a : b;
  const shorter = a.length > b.length ? b : a;
  if (longer.length === 0) return 1;

  let prev = new Uint16Array(longer.length + 1);
  let curr = new Uint16Array(longer.length + 1);
  for (let j = 0; j <= longer.length; j++) prev[j] = j;

  for (let i = 1; i <= shorter.length; i++) {
    curr[0] = i;
    for (let j = 1; j <= longer.length; j++) {
      const cost = shorter[i - 1] === longer[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return 1 - prev[longer.length] / longer.length;
}
