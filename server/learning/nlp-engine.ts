import OpenAI from "openai";
import { storage } from "../storage";
import { batchProcess } from "../replit_integrations/batch/utils";
import type { Material } from "@shared/schema";
import type { EventEmitter } from "./engine";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export async function analyzeBondingPatterns(
  emit: EventEmitter,
  materials: Material[]
): Promise<string[]> {
  if (materials.length === 0) return [];

  emit("log", {
    phase: "phase-3",
    event: "NLP bonding analysis started",
    detail: `Analyzing bonding patterns across ${materials.length} materials`,
    dataSource: "OpenAI NLP",
  });

  const materialSummaries = materials.slice(0, 15).map((m) => ({
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
            "You are a materials science AI. Analyze the provided materials data and identify bonding patterns, structural trends, and property correlations. Return a JSON object with a single key 'insights' containing an array of 3-5 concise scientific insight strings (each under 120 characters).",
        },
        {
          role: "user",
          content: `Analyze bonding patterns in these materials:\n${JSON.stringify(materialSummaries, null, 2)}`,
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
    const insights = parsed.insights ?? [];

    if (insights.length > 0) {
      emit("log", {
        phase: "phase-3",
        event: "Bonding patterns discovered",
        detail: insights[0],
        dataSource: "OpenAI NLP",
      });
      emit("insight", { phase: 3, insights });
    }

    return insights;
  } catch (err: any) {
    emit("log", {
      phase: "phase-3",
      event: "NLP analysis error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "OpenAI NLP",
    });
    return [];
  }
}

export async function analyzePropertyPredictionPatterns(
  emit: EventEmitter,
  materials: Material[]
): Promise<string[]> {
  if (materials.length === 0) return [];

  emit("log", {
    phase: "phase-5",
    event: "Property prediction analysis started",
    detail: `Analyzing ${materials.length} materials for predictive patterns`,
    dataSource: "OpenAI NLP",
  });

  const materialSummaries = materials.slice(0, 20).map((m) => ({
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
            "You are a materials science AI specializing in property prediction. Analyze the provided materials data and identify patterns that could predict properties of unknown materials. Focus on relationships between composition and band gap, formation energy, and stability. Return a JSON object with 'insights' (array of 3-5 concise prediction rules, each under 120 chars) and 'applications' (array of objects with 'pattern' and 'targetProperty' keys).",
        },
        {
          role: "user",
          content: `Find property prediction patterns:\n${JSON.stringify(materialSummaries, null, 2)}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 600,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      emit("log", { phase: "phase-5", event: "NLP returned empty response", detail: "No content in prediction response", dataSource: "OpenAI NLP" });
      return [];
    }

    let parsed: { insights: string[]; applications?: { pattern: string; targetProperty: string }[] };
    try {
      parsed = JSON.parse(content);
    } catch (parseErr) {
      emit("log", { phase: "phase-5", event: "NLP JSON parse error", detail: content.slice(0, 200), dataSource: "OpenAI NLP" });
      return [];
    }
    const insights = parsed.insights ?? [];

    if (insights.length > 0) {
      emit("log", {
        phase: "phase-5",
        event: "Prediction patterns discovered",
        detail: insights[0],
        dataSource: "OpenAI NLP",
      });
      emit("insight", { phase: 5, insights });
    }

    return insights;
  } catch (err: any) {
    emit("log", {
      phase: "phase-5",
      event: "Property prediction error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "OpenAI NLP",
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
