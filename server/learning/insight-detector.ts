import OpenAI from "openai";
import crypto from "crypto";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

function insightId(text: string, phaseId: number): string {
  const hash = crypto.createHash("sha256").update(`${phaseId}:${text.toLowerCase().trim()}`).digest("hex").slice(0, 12);
  return `ni-${phaseId}-${hash}`;
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

function isTextbookKnowledge(insight: string): boolean {
  const lower = insight.toLowerCase();
  return WELL_KNOWN_PATTERNS.some(pattern =>
    lower.includes(pattern) || pattern.split(" ").filter(w => w.length > 4).every(w => lower.includes(w))
  );
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

export async function evaluateInsightNovelty(
  emit: EventEmitter,
  insights: string[],
  phaseId: number,
  phaseName: string,
  relatedFormulas?: string[]
): Promise<{ novel: number; total: number }> {
  if (insights.length === 0) return { novel: 0, total: 0 };

  const [recentInsights, novelInsights] = await Promise.all([
    storage.getNovelInsights(200),
    storage.getNovelInsightsOnly(300),
  ]);

  const combinedMap = new Map<string, typeof recentInsights[0]>();
  for (const ins of recentInsights) combinedMap.set(ins.id, ins);
  for (const ins of novelInsights) combinedMap.set(ins.id, ins);
  const existingInsights = Array.from(combinedMap.values());
  const existingTexts = existingInsights.map(i => i.insightText.toLowerCase());

  const exactDuplicates = new Set<number>();
  const potentiallyNovel: { index: number; text: string }[] = [];

  for (let i = 0; i < insights.length; i++) {
    const text = insights[i];
    const lower = text.toLowerCase();

    if (existingTexts.some(e => e === lower || levenshteinSimilarity(e, lower) > 0.70)) {
      exactDuplicates.add(i);
      continue;
    }

    if (isTextbookKnowledge(text)) {
      try {
        await storage.insertNovelInsight({
          id: insightId(text, phaseId),
          phaseId,
          phaseName,
          insightText: text,
          isNovel: false,
          noveltyScore: 0.1,
          noveltyReason: "Restates well-known textbook knowledge",
          category: "textbook",
          relatedFormulas: relatedFormulas ?? [],
        });
      } catch {}
      continue;
    }

    if (hasKeywordOverlap(lower, existingTexts)) {
      try {
        await storage.insertNovelInsight({
          id: insightId(text, phaseId),
          phaseId,
          phaseName,
          insightText: text,
          isNovel: false,
          noveltyScore: 0.15,
          noveltyReason: "Semantically similar to existing insight (keyword overlap)",
          category: "known-pattern",
          relatedFormulas: relatedFormulas ?? [],
        });
      } catch {}
      continue;
    }

    potentiallyNovel.push({ index: i, text });
  }

  if (potentiallyNovel.length === 0) return { novel: 0, total: insights.length };

  let novelCount = 0;

  try {
    const recentKnown = existingInsights.slice(0, 50).map(i => i.insightText);

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a materials science novelty evaluator. Given a list of scientific insights, determine which ones are genuinely NOVEL (not just restating textbook knowledge or well-known facts).

A NOVEL insight is one that:
1. Identifies a NEW correlation between material properties not widely published
2. Proposes an unexpected mechanism or relationship
3. Connects disparate domains (e.g., linking crystal symmetry to a specific superconducting property in a new way)
4. Identifies a pattern in computational results that contradicts conventional wisdom
5. Suggests a new material design principle not in standard references

A NON-NOVEL insight merely restates:
- Standard periodic table trends
- Well-known BCS/Eliashberg theory
- Basic chemistry (formation energy = stability)
- Known facts about specific material families
- General statements about superconductivity
- Any variation of "stability correlates with formation energy" or vice versa
- Any variation of "band gap determines metallic/insulating behavior"
- Any variation of "pressure increases critical temperature"
- Any variation of "electron-phonon coupling favors superconductivity"

Previously discovered insights for deduplication:
${recentKnown.join("\n")}

Return JSON with 'evaluations' array, each with:
- 'index' (number matching input index)
- 'isNovel' (boolean)
- 'noveltyScore' (0-1, where 1 = groundbreaking, 0.5 = interesting, 0.1 = textbook)
- 'reason' (under 100 chars explaining why novel or not)
- 'category' (one of: 'novel-correlation', 'new-mechanism', 'cross-domain', 'computational-discovery', 'design-principle', 'textbook', 'known-pattern', 'incremental')`,
        },
        {
          role: "user",
          content: `Evaluate these insights for novelty:\n${potentiallyNovel.map((p, i) => `${i}. "${p.text}"`).join("\n")}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 600,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return { novel: 0, total: insights.length };

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      return { novel: 0, total: insights.length };
    }

    const evaluations = parsed.evaluations ?? [];

    for (const ev of evaluations) {
      const entry = potentiallyNovel[ev.index];
      if (!entry) continue;

      const isNovel = ev.isNovel === true && (ev.noveltyScore ?? 0) >= 0.4;

      try {
        await storage.insertNovelInsight({
          id: insightId(entry.text, phaseId),
          phaseId,
          phaseName,
          insightText: entry.text,
          isNovel,
          noveltyScore: ev.noveltyScore ?? 0.2,
          noveltyReason: ev.reason ?? "",
          category: ev.category ?? "known-pattern",
          relatedFormulas: relatedFormulas ?? [],
        });
      } catch {}

      if (isNovel) {
        novelCount++;
        emit("log", {
          phase: `phase-${phaseId}`,
          event: "Novel insight discovered",
          detail: `[NOVEL ${((ev.noveltyScore ?? 0) * 100).toFixed(0)}%] ${entry.text}`,
          dataSource: "Insight Detector",
        });
        emit("insight", {
          phase: phaseName,
          insight: entry.text,
          isNovel: true,
          noveltyScore: ev.noveltyScore,
          category: ev.category,
        });
      }
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

function levenshteinSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const longer = a.length > b.length ? a : b;
  const shorter = a.length > b.length ? b : a;
  if (longer.length === 0) return 1;

  const matrix: number[][] = [];
  for (let i = 0; i <= shorter.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= longer.length; j++) {
    matrix[0][j] = j;
  }
  for (let i = 1; i <= shorter.length; i++) {
    for (let j = 1; j <= longer.length; j++) {
      const cost = shorter[i - 1] === longer[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,
        matrix[i][j - 1] + 1,
        matrix[i - 1][j - 1] + cost
      );
    }
  }
  return 1 - matrix[shorter.length][longer.length] / longer.length;
}
