import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { trackDuplicatesSkipped } from "./strategy-analyzer";
import { normalizeFormula } from "./utils";
import { passesValenceFilter, passesCompositionComplexityFilter } from "./candidate-generator";
import { setActiveApplication } from "./gradient-boost";
import { rlAgent } from "./rl-agent";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

function repairTruncatedJSON(raw: string): { materials: GeneratedFormula[] } | null {
  try {
    const materialsMatch = raw.match(/"materials"\s*:\s*\[/);
    if (materialsMatch) {
      const arrayStart = raw.indexOf("[", materialsMatch.index!);
      let depth = 0;
      let lastCompleteObj = -1;

      for (let i = arrayStart; i < raw.length; i++) {
        if (raw[i] === "{") depth++;
        if (raw[i] === "}") {
          depth--;
          if (depth === 0) lastCompleteObj = i;
        }
      }

      if (lastCompleteObj > arrayStart) {
        const repairedArray = raw.substring(arrayStart, lastCompleteObj + 1) + "]";
        const repaired = `{"materials": ${repairedArray}}`;
        const parsed = JSON.parse(repaired);
        if (Array.isArray(parsed.materials) && parsed.materials.length > 0) return parsed;
      }
    }

    const arrayMatch = raw.match(/\[\s*\{/);
    if (arrayMatch) {
      const arrayStart = raw.indexOf("[", arrayMatch.index!);
      let depth = 0;
      let lastCompleteObj = -1;
      for (let i = arrayStart; i < raw.length; i++) {
        if (raw[i] === "{") depth++;
        if (raw[i] === "}") {
          depth--;
          if (depth === 0) lastCompleteObj = i;
        }
      }
      if (lastCompleteObj > arrayStart) {
        const repairedArray = raw.substring(arrayStart, lastCompleteObj + 1) + "]";
        const parsed = JSON.parse(`{"materials": ${repairedArray}}`);
        if (Array.isArray(parsed.materials) && parsed.materials.length > 0) return parsed;
      }
    }

    const objectMatch = raw.match(/\{\s*"(?:name|formula)"/);
    if (objectMatch) {
      const objStart = raw.indexOf("{", objectMatch.index!);
      let depth = 0;
      let lastComplete = -1;
      for (let i = objStart; i < raw.length; i++) {
        if (raw[i] === "{") depth++;
        if (raw[i] === "}") { depth--; if (depth === 0) { lastComplete = i; break; } }
      }
      if (lastComplete > objStart) {
        const singleObj = raw.substring(objStart, lastComplete + 1);
        const parsed = JSON.parse(singleObj);
        if (parsed.formula) return { materials: [parsed] };
      }
    }

    return null;
  } catch {
    return null;
  }
}

const TARGET_APPLICATIONS = [
  "high-temperature superconductor",
  "ultra-hard coating material",
  "thermoelectric generator",
  "transparent conductor",
  "topological insulator",
  "solid-state battery electrolyte",
  "photovoltaic absorber",
  "catalytic surface material",
  "heavy fermion superconductor",
  "kagome lattice metal",
  "MXene-based compound",
  "clathrate hydride superconductor",
  "spin-orbit coupled material",
];

const APPLICATION_FAMILY_MAP: Record<string, string[]> = {
  "high-temperature superconductor": ["cuprate", "hydride", "layered-pnictide"],
  "topological insulator": ["kagome-metal", "chalcogenide"],
  "heavy fermion superconductor": ["intermetallic", "oxide-perovskite"],
  "kagome lattice metal": ["kagome-metal"],
  "clathrate hydride superconductor": ["hydride"],
  "spin-orbit coupled material": ["chalcogenide", "kagome-metal"],
  "ultra-hard coating material": ["boride", "intermetallic"],
  "MXene-based compound": ["intermetallic", "boride"],
};

let applicationIndex = 0;
let applicationSuccessStreak: Record<string, number> = {};
let recentlyGenerated: string[] = [];
let recentlyGeneratedLoaded = false;
let inverseDesignMode = false;
let boundaryHuntingMode = false;
let chemicalSpaceExpansionMode = false;

async function loadRecentlyGenerated(): Promise<void> {
  if (recentlyGeneratedLoaded) return;
  try {
    const recent = await storage.getTopPredictionFormulas(50);
    recentlyGenerated = recent;
    recentlyGeneratedLoaded = true;
  } catch {
    recentlyGeneratedLoaded = true;
  }
}

export function setInverseDesignMode(enabled: boolean): void {
  inverseDesignMode = enabled;
}

export function setBoundaryHuntingMode(enabled: boolean): void {
  boundaryHuntingMode = enabled;
}

export function setChemicalSpaceExpansionMode(enabled: boolean): void {
  chemicalSpaceExpansionMode = enabled;
}

export function getGenerationModes(): { inverseDesign: boolean; boundaryHunting: boolean; chemicalSpaceExpansion: boolean } {
  return { inverseDesign: inverseDesignMode, boundaryHunting: boundaryHuntingMode, chemicalSpaceExpansion: chemicalSpaceExpansionMode };
}

export function getNextTargetApplication(): string {
  try {
    const stats = rlAgent.getStats();

    if (stats.recentAvgReward > 0.3 && stats.topElementGroups.length > 0) {
      const topFamilies = stats.topElementGroups.slice(0, 3).map(g => g.name.toLowerCase());

      let bestApp: string | null = null;
      let bestScore = -1;

      for (const app of TARGET_APPLICATIONS) {
        const mappedFamilies = APPLICATION_FAMILY_MAP[app];
        if (!mappedFamilies) continue;

        let overlap = 0;
        for (const fam of mappedFamilies) {
          if (topFamilies.some(tf => fam.includes(tf) || tf.includes(fam))) overlap++;
        }

        const streak = applicationSuccessStreak[app] || 0;
        const streakBonus = Math.min(0.3, streak * 0.1);
        const score = overlap + streakBonus;

        if (score > bestScore) {
          bestScore = score;
          bestApp = app;
        }
      }

      if (bestApp && bestScore > 0) {
        return bestApp;
      }
    }
  } catch {}

  const app = TARGET_APPLICATIONS[applicationIndex % TARGET_APPLICATIONS.length];
  applicationIndex++;
  return app;
}

export function recordApplicationSuccess(application: string, count: number): void {
  if (count > 0) {
    applicationSuccessStreak[application] = (applicationSuccessStreak[application] || 0) + count;
  } else {
    applicationSuccessStreak[application] = 0;
  }
}

interface GeneratedFormula {
  name: string;
  formula: string;
  predictedProperties: {
    bandGap?: number;
    formationEnergy?: number;
    stability?: number;
    hardness?: string;
    conductivity?: string;
    description: string;
  };
  confidence: number;
  targetApplication: string;
  notes: string;
}

export async function generateNovelFormulas(
  emit: EventEmitter,
  insights: string[],
  targetApp?: string,
  strategyHint?: string,
  rejectionCategories?: Record<string, number>
): Promise<number> {
  await loadRecentlyGenerated();
  const application = targetApp || getNextTargetApplication();
  setActiveApplication(application);
  let generated = 0;

  emit("log", {
    phase: "phase-6",
    event: "Formula generation started",
    detail: `Generating novel materials for: ${application}${strategyHint ? ` (strategy: ${strategyHint})` : ""}`,
    dataSource: "OpenAI NLP",
  });

  const insightContext =
    insights.length > 0
      ? `Known patterns from analysis:\n${insights.slice(0, 5).join("\n")}`
      : "Use general materials science knowledge.";

  let strategyContext = strategyHint
    ? `\n\nCurrent research strategy prioritizes: ${strategyHint}. Approximately 70% of generated candidates should come from these prioritized material families. The remaining 30% should explore other chemical spaces to maintain diversity and avoid confirmation bias. Balance focused exploitation with broad exploration.`
    : "";

  if (inverseDesignMode) {
    strategyContext += `\n\nINVERSE DESIGN MODE ACTIVE: Instead of predicting "good superconductors", design materials that maximize electron-boson coupling (lambda > 2.0). Focus on: high DOS at Fermi level (>5 states/eV/atom), mixed stiff-soft bonding networks, cage/clathrate structures with light intercalants, and flat bands near Ef. Optimize for pairing susceptibility, not just Tc.`;
  }

  if (boundaryHuntingMode) {
    strategyContext += `\n\nBOUNDARY HUNTING MODE ACTIVE: Design materials at the edge of phase instabilities. Target: (1) compositions near magnetic quantum critical points (almost-ferromagnetic metals, doped antiferromagnets), (2) materials at structural phase boundaries (tolerance factor ~0.85 or ~1.05), (3) systems near metal-insulator transitions (correlated electron systems with U/W ~ 1), (4) compounds prone to charge density wave instabilities. Place compositions AT the boundary, not safely inside a stable phase.`;
  }

  if (chemicalSpaceExpansionMode) {
    strategyContext += `\n\nCHEMICAL SPACE EXPANSION MODE ACTIVE: The search has stagnated severely. You MUST incorporate elements rarely used in superconductor design. Specifically include at least 2 of: Sc, Hf, Zr, Ta, Re, Os, Ir, Ru, Rh, Pd, Pt, Ga, Ge, In, Tl, Cd, Ag, Au, Th, U, Ce, Pr, Nd, Sm, Eu, Gd, Dy, Er, Yb, Lu. Design ternary and quaternary compositions mixing these rare elements with known SC-active elements (Cu, Fe, Nb, B, N, H). Prioritize unexplored stoichiometries and crystal structure types (Laves phases, sigma phases, Heusler alloys, skutterudites, filled pyrochlores).`;
  }

  if (rejectionCategories) {
    const totalRejects = Object.values(rejectionCategories).reduce((s, v) => s + v, 0);
    if (totalRejects > 20) {
      const sorted = Object.entries(rejectionCategories).sort((a, b) => b[1] - a[1]);
      const topRejects = sorted.slice(0, 5).map(([cat, n]) => `${cat}: ${n}`).join(", ");
      const chemRejects = rejectionCategories["chemistry_reject"] || 0;
      const stabilityRejects = rejectionCategories["stability_reject"] || 0;
      const failRate = chemRejects + stabilityRejects;
      strategyContext += `\n\nREJECTION FEEDBACK: Out of ${totalRejects} total rejections, the top categories are: ${topRejects}.`;
      if (failRate > totalRejects * 0.5) {
        strategyContext += ` WARNING: Over 50% of rejections are chemistry/stability failures. The current element combinations are producing unphysical or unstable compositions. Shift toward well-established structural families (perovskites, layered cuprates, clathrate hydrides, ThCr2Si2-type) and avoid exotic multi-element combinations that lack known analogues.`;
      }
    }
  }

  let exclusionContext = "";
  try {
    const topFormulas = await storage.getTopPredictionFormulas(20);
    const exclusionSet = new Set<string>();
    for (const f of topFormulas) exclusionSet.add(f);
    for (const f of recentlyGenerated) exclusionSet.add(f);
    const allExclusions = Array.from(exclusionSet);
    if (allExclusions.length > 0) {
      exclusionContext = `\n\nDo NOT generate any of these already-known compositions: ${allExclusions.slice(0, 25).join(", ")}. Generate genuinely novel compositions not yet explored.`;
    }
  } catch {}

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a materials science AI that generates novel chemical compositions optimized for strong pairing susceptibility. Focus on: high DOS at Fermi level, strong electron-phonon coupling, favorable nesting, proximity to quantum critical points, and mixed stiff-soft bonding. Each material must be chemically plausible. Stoichiometry: prefer integers (MgB2, LaH10) but fractions to 2 decimal places are allowed for doped systems (Ba0.6K0.4BiO3, La1.85Sr0.15CuO4). No fractions below 0.05. Max 5 distinct elements. Return JSON: {"materials": [...]}`,
        },
        {
          role: "assistant",
          content: `{"materials": [{"name": "Strontium-doped lanthanum cuprate", "formula": "La1.85Sr0.15CuO4", "predictedProperties": {"bandGap": null, "formationEnergy": -2.1, "stability": 0.8, "description": "Hole-doped 214 cuprate with optimal carrier concentration"}, "confidence": 0.75, "notes": "Strong dx2-y2 pairing channel, VHS near Ef at optimal doping"}, {"name": "Yttrium decahydride", "formula": "YH10", "predictedProperties": {"bandGap": null, "formationEnergy": -0.3, "stability": 0.4, "description": "Sodalite-cage clathrate hydride"}, "confidence": 0.6, "notes": "High phonon frequency H cage, strong e-ph lambda > 2.5"}]}`,
        },
        {
          role: "user",
          content: `Target application: ${application}\n\n${insightContext}${strategyContext}${exclusionContext}\n\nGenerate novel material candidates.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1600,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      emit("log", { phase: "phase-6", event: "Generator empty response", detail: "No content from OpenAI", dataSource: "OpenAI NLP" });
      return 0;
    }

    let parsed: { materials: GeneratedFormula[] };
    try {
      parsed = JSON.parse(content);
    } catch (parseErr) {
      parsed = repairTruncatedJSON(content)!;
      if (!parsed) {
        emit("log", { phase: "phase-6", event: "Generator JSON parse error", detail: content.slice(0, 200), dataSource: "OpenAI NLP" });
        return 0;
      }
      emit("log", { phase: "phase-6", event: "Generator JSON repaired", detail: `Recovered ${parsed.materials?.length ?? 0} materials from truncated response`, dataSource: "OpenAI NLP" });
    }
    const candidates = parsed.materials ?? [];

    const processingSet = new Set<string>();
    for (const f of recentlyGenerated) processingSet.add(f);

    for (const candidate of candidates) {
      if (!candidate.formula || !candidate.name) continue;
      candidate.formula = normalizeFormula(candidate.formula);

      if (!passesValenceFilter(candidate.formula) || !passesCompositionComplexityFilter(candidate.formula)) {
        emit("log", { phase: "phase-6", event: "NLP formula rejected by filter", detail: `${candidate.formula}: failed valence or complexity filter`, dataSource: "OpenAI NLP" });
        continue;
      }

      if (processingSet.has(candidate.formula)) {
        trackDuplicatesSkipped(1);
        continue;
      }
      processingSet.add(candidate.formula);

      const existing = await storage.getNovelPredictionByFormula(candidate.formula);
      if (existing) {
        trackDuplicatesSkipped(1);
        const newConf = Math.min(1, Math.max(0, candidate.confidence ?? 0.5));
        if (newConf > (existing.confidence ?? 0)) {
          await storage.updateNovelPrediction(existing.id, {
            confidence: newConf,
            predictedProperties: candidate.predictedProperties || existing.predictedProperties,
            notes: candidate.notes || existing.notes,
          });
          emit("log", { phase: "phase-6", event: "Novel prediction upgraded", detail: `${candidate.formula}: confidence ${(existing.confidence ?? 0).toFixed(2)} -> ${newConf.toFixed(2)}`, dataSource: "OpenAI NLP" });
        }
        continue;
      }

      const id = `novel-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      try {
        const inserted = await storage.insertNovelPrediction({
          id,
          name: candidate.name,
          formula: candidate.formula,
          predictedProperties: candidate.predictedProperties || { description: "AI-generated candidate" },
          confidence: Math.min(1, Math.max(0, candidate.confidence ?? 0.5)),
          targetApplication: application,
          status: "predicted",
          notes: candidate.notes || "Generated by MatSci-∞ AI engine",
        });

        if (!inserted) {
          trackDuplicatesSkipped(1);
          continue;
        }

        generated++;

        recentlyGenerated.push(candidate.formula);
        if (recentlyGenerated.length > 100) recentlyGenerated.splice(0, recentlyGenerated.length - 100);

        emit("prediction", {
          id,
          name: candidate.name,
          formula: candidate.formula,
          confidence: candidate.confidence,
          targetApplication: application,
        });
      } catch (e: any) {
        if (e.message?.includes("unique") || e.message?.includes("duplicate") || e.code === "23505") {
          trackDuplicatesSkipped(1);
        } else {
          emit("log", { phase: "phase-6", event: "Prediction insert failed", detail: `${candidate.formula}: ${e.message?.slice(0, 100) || "unknown"}`, dataSource: "OpenAI NLP" });
        }
      }
    }

    if (generated > 0) {
      emit("log", {
        phase: "phase-6",
        event: "Novel materials generated",
        detail: `Created ${generated} new candidates for ${application}`,
        dataSource: "OpenAI NLP",
      });
      emit("progress", { phase: 6, newItems: generated });
    }
  } catch (err: any) {
    emit("log", {
      phase: "phase-6",
      event: "Formula generation error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "OpenAI NLP",
    });
  }

  recordApplicationSuccess(application, generated);
  return generated;
}
