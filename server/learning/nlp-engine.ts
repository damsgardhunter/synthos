import OpenAI from "openai";
import { storage } from "../storage";
import { batchProcess } from "../replit_integrations/batch/utils";
import type { Material } from "@shared/schema";
import type { EventEmitter } from "./engine";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

const PHYSICS_RULES: { pattern: RegExp; valid: boolean; reason: string }[] = [
  { pattern: /higher\s+formation\s+energy.*(?:correlat|indicat|lead|suggest).*(?:stab|more stable)/i, valid: false, reason: "Higher formation energy does NOT correlate with stability" },
  { pattern: /(?:increas|higher|more)\s+formation\s+energy.*(?:more|higher|greater)\s+stabil/i, valid: false, reason: "Higher formation energy means LESS stability" },
  { pattern: /(?:negative|lower)\s+band\s*gap.*(?:metal|conduct)/i, valid: false, reason: "Band gap cannot be negative" },
  { pattern: /(?:superconducti|superconduct).*(?:increas|higher).*(?:temperatur|heat)/i, valid: false, reason: "Superconductivity is suppressed, not enhanced, by higher temperature" },
  { pattern: /(?:insulator|semiconductor).*(?:zero|no)\s+(?:resistance|resistiv)/i, valid: false, reason: "Insulators cannot have zero resistance" },
];

function validatePhysicsRules(insight: string): { valid: boolean; reason?: string } {
  for (const rule of PHYSICS_RULES) {
    if (rule.pattern.test(insight) && !rule.valid) {
      return { valid: false, reason: rule.reason };
    }
  }
  return { valid: true };
}

const MIN_DATASET_FOR_INSIGHTS = 100;

function computeDatasetStatistics(materials: Material[]): string {
  const withBG = materials.filter(m => m.bandGap !== null && m.bandGap !== undefined);
  const withFE = materials.filter(m => m.formationEnergy !== null && m.formationEnergy !== undefined);
  const withStab = materials.filter(m => m.stability !== null && m.stability !== undefined);

  const stats: string[] = [];
  if (withBG.length >= 10) {
    const bgVals = withBG.map(m => m.bandGap as number);
    const avgBG = bgVals.reduce((s, v) => s + v, 0) / bgVals.length;
    const metals = bgVals.filter(v => v < 0.1).length;
    stats.push(`Band gap: avg=${avgBG.toFixed(2)} eV, metals=${metals}/${bgVals.length}, range=[${Math.min(...bgVals).toFixed(1)}, ${Math.max(...bgVals).toFixed(1)}]`);
  }
  if (withFE.length >= 10) {
    const feVals = withFE.map(m => m.formationEnergy as number);
    const avgFE = feVals.reduce((s, v) => s + v, 0) / feVals.length;
    const negFE = feVals.filter(v => v < 0).length;
    stats.push(`Formation energy: avg=${avgFE.toFixed(2)} eV/atom, negative=${negFE}/${feVals.length} (lower=more stable), range=[${Math.min(...feVals).toFixed(1)}, ${Math.max(...feVals).toFixed(1)}]`);
  }
  if (withStab.length >= 10) {
    const stabVals = withStab.map(m => m.stability as number);
    const avgStab = stabVals.reduce((s, v) => s + v, 0) / stabVals.length;
    stats.push(`Stability: avg=${avgStab.toFixed(3)}, range=[${Math.min(...stabVals).toFixed(2)}, ${Math.max(...stabVals).toFixed(2)}]`);
  }
  return stats.join("\n");
}

export async function analyzeBondingPatterns(
  emit: EventEmitter,
  materials: Material[]
): Promise<string[]> {
  if (materials.length === 0) return [];

  const allMats = await storage.getMaterials(200, 0);
  const dataset = allMats.length >= MIN_DATASET_FOR_INSIGHTS ? allMats : materials;

  if (dataset.length < 30) {
    emit("log", {
      phase: "phase-3",
      event: "Bonding analysis deferred",
      detail: `Dataset too small (${dataset.length} materials, need 30+). Skipping insight generation to avoid hallucinations.`,
      dataSource: "Statistical Analysis",
    });
    return [];
  }

  const dataStats = computeDatasetStatistics(dataset);

  emit("log", {
    phase: "phase-3",
    event: "Bonding statistical analysis started",
    detail: `Analyzing bonding patterns across ${dataset.length} materials. ${dataStats.split("\n")[0] || ""}`,
    dataSource: "Statistical Analysis",
  });

  const materialSummaries = dataset.slice(0, 100).map((m) => ({
    name: m.name,
    formula: m.formula,
    spacegroup: m.spacegroup,
    bandGap: m.bandGap,
    formationEnergy: m.formationEnergy,
    stability: m.stability,
  }));

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            `You are a materials science AI. Analyze the provided materials data and identify bonding patterns, structural trends, and property correlations. IMPORTANT PHYSICS RULES that MUST be followed:
- Lower (more negative) formation energy = MORE stable (thermodynamically favorable)
- Higher formation energy = LESS stable
- Band gap cannot be negative
- Superconductivity occurs at LOW temperatures, not high
Return a JSON object with a single key 'insights' containing an array of 3-5 concise scientific insight strings (each under 120 characters). Only include insights supported by statistical evidence in the data.`,
        },
        {
          role: "user",
          content: `Dataset statistics (${dataset.length} materials):\n${dataStats}\n\nSample data:\n${JSON.stringify(materialSummaries.slice(0, 50), null, 2)}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      emit("log", { phase: "phase-3", event: "NLP returned empty response", detail: "No content in OpenAI response", dataSource: "OpenAI NLP" });
      return [];
    }

    let parsed: { insights: string[] };
    try {
      parsed = JSON.parse(content);
    } catch (parseErr) {
      emit("log", { phase: "phase-3", event: "NLP JSON parse error", detail: content.slice(0, 200), dataSource: "OpenAI NLP" });
      return [];
    }

    const rawInsights = parsed.insights ?? [];
    const insights = rawInsights.filter(insight => {
      const check = validatePhysicsRules(insight);
      if (!check.valid) {
        emit("log", {
          phase: "phase-3",
          event: "Insight rejected (physics violation)",
          detail: `"${insight}" — ${check.reason}`,
          dataSource: "Physics Validator",
        });
        return false;
      }
      return true;
    });

    if (insights.length > 0) {
      emit("log", {
        phase: "phase-3",
        event: "Bonding patterns discovered",
        detail: insights[0],
        dataSource: "Statistical Analysis",
      });
      emit("insight", { phase: 3, insights });
    }

    return insights;
  } catch (err: any) {
    emit("log", {
      phase: "phase-3",
      event: "NLP analysis error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "Statistical Analysis",
    });
    return [];
  }
}

export async function analyzePropertyPredictionPatterns(
  emit: EventEmitter,
  materials: Material[]
): Promise<string[]> {
  if (materials.length === 0) return [];

  const allMats = await storage.getMaterials(200, 0);
  const dataset = allMats.length >= MIN_DATASET_FOR_INSIGHTS ? allMats : materials;

  if (dataset.length < 30) {
    emit("log", {
      phase: "phase-5",
      event: "Prediction analysis deferred",
      detail: `Dataset too small (${dataset.length} materials, need 30+). Skipping to avoid unreliable patterns.`,
      dataSource: "Statistical Analysis",
    });
    return [];
  }

  const dataStats = computeDatasetStatistics(dataset);

  emit("log", {
    phase: "phase-5",
    event: "Property prediction statistical analysis started",
    detail: `Analyzing ${dataset.length} materials for predictive patterns. ${dataStats.split("\n")[0] || ""}`,
    dataSource: "Statistical Analysis",
  });

  const materialSummaries = dataset.slice(0, 100).map((m) => ({
    formula: m.formula,
    bandGap: m.bandGap,
    formationEnergy: m.formationEnergy,
    stability: m.stability,
    source: m.source,
  }));

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            `You are a materials science AI specializing in property prediction. Analyze the provided materials data and identify patterns that could predict properties of unknown materials. Focus on relationships between composition and band gap, formation energy, and stability.
IMPORTANT PHYSICS RULES that MUST be followed:
- Lower (more negative) formation energy = MORE stable (thermodynamically favorable)
- Higher formation energy = LESS stable (this is a fundamental thermodynamic law)
- Band gap is always >= 0
- Metals have zero or near-zero band gap; insulators have large band gap
Return a JSON object with 'insights' (array of 3-5 concise prediction rules, each under 120 chars, supported by statistical evidence) and 'applications' (array of objects with 'pattern' and 'targetProperty' keys).`,
        },
        {
          role: "user",
          content: `Dataset statistics (${dataset.length} materials):\n${dataStats}\n\nSample data:\n${JSON.stringify(materialSummaries.slice(0, 50), null, 2)}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 600,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      emit("log", { phase: "phase-5", event: "NLP returned empty response", detail: "No content in prediction response", dataSource: "Statistical Analysis" });
      return [];
    }

    let parsed: { insights: string[]; applications?: { pattern: string; targetProperty: string }[] };
    try {
      parsed = JSON.parse(content);
    } catch (parseErr) {
      emit("log", { phase: "phase-5", event: "NLP JSON parse error", detail: content.slice(0, 200), dataSource: "Statistical Analysis" });
      return [];
    }

    const rawInsights = parsed.insights ?? [];
    const insights = rawInsights.filter(insight => {
      const check = validatePhysicsRules(insight);
      if (!check.valid) {
        emit("log", {
          phase: "phase-5",
          event: "Insight rejected (physics violation)",
          detail: `"${insight}" — ${check.reason}`,
          dataSource: "Physics Validator",
        });
        return false;
      }
      return true;
    });

    if (insights.length > 0) {
      emit("log", {
        phase: "phase-5",
        event: "Prediction patterns discovered",
        detail: insights[0],
        dataSource: "Statistical Analysis",
      });
      emit("insight", { phase: 5, insights });
    }

    return insights;
  } catch (err: any) {
    emit("log", {
      phase: "phase-5",
      event: "Property prediction error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "Statistical Analysis",
    });
    return [];
  }
}

export async function classifyMaterialApplications(
  emit: EventEmitter,
  materials: Material[]
): Promise<Map<string, string>> {
  const results = new Map<string, string>();
  if (materials.length === 0) return results;

  const batch = materials.slice(0, 10);
  try {
    const classified = await batchProcess(
      batch,
      async (mat) => {
        const response = await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [
            {
              role: "system",
              content:
                'Classify this material into one application category: "energy", "aerospace", "electronics", "biomedical", "construction", or "catalysis". Return JSON with "category" key only.',
            },
            {
              role: "user",
              content: `Material: ${mat.name} (${mat.formula}), band gap: ${mat.bandGap ?? "unknown"} eV, formation energy: ${mat.formationEnergy ?? "unknown"} eV/atom`,
            },
          ],
          response_format: { type: "json_object" },
          max_completion_tokens: 50,
        });
        const content = response.choices[0]?.message?.content;
        if (!content) return { id: mat.id, category: "unknown" };
        const parsed = JSON.parse(content) as { category: string };
        return { id: mat.id, category: parsed.category || "unknown" };
      },
      { concurrency: 2, retries: 3 }
    );

    for (const c of classified) {
      if (c) results.set(c.id, c.category);
    }
  } catch (err: any) {
    emit("log", {
      phase: "phase-5",
      event: "Classification error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "OpenAI NLP",
    });
  }

  return results;
}
