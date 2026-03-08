import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { trackDuplicatesSkipped } from "./strategy-analyzer";
import { normalizeFormula } from "./utils";
import { passesValenceFilter, passesCompositionComplexityFilter } from "./candidate-generator";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

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

let applicationIndex = 0;
const recentlyGenerated: string[] = [];
let inverseDesignMode = false;
let boundaryHuntingMode = false;
let chemicalSpaceExpansionMode = false;

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
  const app = TARGET_APPLICATIONS[applicationIndex % TARGET_APPLICATIONS.length];
  applicationIndex++;
  return app;
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
  strategyHint?: string
): Promise<number> {
  const application = targetApp || getNextTargetApplication();
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
    ? `\n\nCurrent research strategy prioritizes: ${strategyHint}. When generating candidates, prefer compositions from these material families if relevant to the target application. IMPORTANT: Do NOT generate materials from families not listed in the strategy priorities unless exploring novel chemical space. If a family (e.g., Hydrides, Carbides) is not listed above, reduce generation probability for that family.`
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

  let exclusionContext = "";
  try {
    const topFormulas = await storage.getTopPredictionFormulas(20);
    const allExclusions = [...new Set([...topFormulas, ...recentlyGenerated])];
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
          content: `You are a materials science AI that generates novel chemical compositions optimized for strong pairing susceptibility. Rather than directly targeting high Tc, focus on compositions with: high density of states at the Fermi level, strong electron-phonon or electron-boson coupling channels, favorable Fermi surface nesting, proximity to quantum critical points, and mixed stiff-soft bonding networks. Each material should be chemically plausible (valid stoichiometry, realistic oxidation states). IMPORTANT: Use only integer stoichiometry (e.g. NbC, MgB2, LaH10). Do NOT use fractional atomic counts like Fe0.07C or Mn0.5Ni0.5. All atom counts must be whole numbers >= 1. Return JSON with key 'materials' containing an array of objects, each with: 'name' (descriptive), 'formula' (chemical formula with integer stoichiometry), 'predictedProperties' (object with 'bandGap' number or null, 'formationEnergy' number or null, 'stability' number 0-1, 'description' string under 100 chars), 'confidence' (0-1 float), 'notes' (brief rationale under 150 chars focusing on pairing mechanism).`,
        },
        {
          role: "user",
          content: `Target application: ${application}\n\n${insightContext}${strategyContext}${exclusionContext}\n\nGenerate novel material candidates.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 800,
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
      emit("log", { phase: "phase-6", event: "Generator JSON parse error", detail: content.slice(0, 200), dataSource: "OpenAI NLP" });
      return 0;
    }
    const candidates = parsed.materials ?? [];

    for (const candidate of candidates) {
      if (!candidate.formula || !candidate.name) continue;
      candidate.formula = normalizeFormula(candidate.formula);

      if (!passesValenceFilter(candidate.formula) || !passesCompositionComplexityFilter(candidate.formula)) {
        emit("log", { phase: "phase-6", event: "NLP formula rejected by filter", detail: `${candidate.formula}: failed valence or complexity filter`, dataSource: "OpenAI NLP" });
        continue;
      }

      if (recentlyGenerated.includes(candidate.formula)) {
        trackDuplicatesSkipped(1);
        continue;
      }

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
        await storage.insertNovelPrediction({
          id,
          name: candidate.name,
          formula: candidate.formula,
          predictedProperties: candidate.predictedProperties || { description: "AI-generated candidate" },
          confidence: Math.min(1, Math.max(0, candidate.confidence ?? 0.5)),
          targetApplication: application,
          status: "predicted",
          notes: candidate.notes || "Generated by MatSci-∞ AI engine",
        });
        generated++;

        recentlyGenerated.push(candidate.formula);
        if (recentlyGenerated.length > 50) recentlyGenerated.shift();

        emit("prediction", {
          id,
          name: candidate.name,
          formula: candidate.formula,
          confidence: candidate.confidence,
          targetApplication: application,
        });
      } catch (e: any) {
        emit("log", { phase: "phase-6", event: "Prediction insert failed", detail: `${candidate.formula}: ${e.message?.slice(0, 100) || "unknown"}`, dataSource: "OpenAI NLP" });
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

  return generated;
}
