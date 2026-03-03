import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { extractFeatures, runMLPrediction } from "./ml-predictor";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export async function runSuperconductorResearch(
  emit: EventEmitter,
  allInsights: string[]
): Promise<{ generated: number; insights: string[] }> {
  let generated = 0;
  const newInsights: string[] = [];

  emit("log", {
    phase: "phase-7",
    event: "Superconductor research cycle started",
    detail: "XGBoost+NN ensemble targeting room-temperature superconductivity",
    dataSource: "SC Research",
  });

  const materials = await storage.getMaterials(50, 0);

  const mlResult = await runMLPrediction(emit, materials);
  newInsights.push(...mlResult.insights);

  for (const candidate of mlResult.candidates) {
    const id = `sc-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    try {
      await storage.insertSuperconductorCandidate({
        id,
        name: candidate.name || "Unknown",
        formula: candidate.formula || "Unknown",
        predictedTc: candidate.predictedTc ?? null,
        pressureGpa: candidate.pressureGpa ?? null,
        meissnerEffect: candidate.meissnerEffect ?? false,
        zeroResistance: candidate.zeroResistance ?? false,
        cooperPairMechanism: candidate.cooperPairMechanism ?? null,
        crystalStructure: candidate.crystalStructure ?? null,
        quantumCoherence: candidate.quantumCoherence ?? null,
        stabilityScore: candidate.stabilityScore ?? null,
        synthesisPath: candidate.synthesisPath ?? null,
        mlFeatures: candidate.mlFeatures ?? null,
        xgboostScore: candidate.xgboostScore ?? null,
        neuralNetScore: candidate.neuralNetScore ?? null,
        ensembleScore: candidate.ensembleScore ?? null,
        roomTempViable: candidate.roomTempViable ?? false,
        status: candidate.status || "theoretical",
        notes: candidate.notes ?? null,
      });
      generated++;

      emit("prediction", {
        type: "superconductor",
        id,
        name: candidate.name,
        formula: candidate.formula,
        predictedTc: candidate.predictedTc,
        ensembleScore: candidate.ensembleScore,
        roomTempViable: candidate.roomTempViable,
        meissnerEffect: candidate.meissnerEffect,
      });
    } catch (e: any) {
      emit("log", { phase: "phase-7", event: "SC candidate insert error", detail: `${candidate.formula}: ${e.message?.slice(0, 100)}`, dataSource: "SC Research" });
    }
  }

  if (generated > 0) {
    emit("log", {
      phase: "phase-7",
      event: "Superconductor candidates generated",
      detail: `${generated} candidates, ${mlResult.candidates.filter(c => c.roomTempViable).length} room-temp viable, ${mlResult.candidates.filter(c => c.meissnerEffect).length} with Meissner effect`,
      dataSource: "SC Research",
    });
  }

  try {
    const existingCandidates = await storage.getSuperconductorCandidates(10);
    if (existingCandidates.length > 0) {
      const novelResult = await generateNovelSuperconductors(emit, existingCandidates, allInsights);
      generated += novelResult;
    }
  } catch (err: any) {
    emit("log", { phase: "phase-7", event: "Novel SC generation error", detail: err.message?.slice(0, 200), dataSource: "SC Research" });
  }

  return { generated, insights: newInsights };
}

async function generateNovelSuperconductors(
  emit: EventEmitter,
  existingCandidates: any[],
  allInsights: string[]
): Promise<number> {
  let generated = 0;

  const bestCandidates = existingCandidates
    .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0))
    .slice(0, 5)
    .map(c => ({
      formula: c.formula,
      predictedTc: c.predictedTc,
      ensembleScore: c.ensembleScore,
      meissnerEffect: c.meissnerEffect,
      cooperPairMechanism: c.cooperPairMechanism,
    }));

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are the discovery module of a superconductor AI. Based on the best candidates found so far and known patterns, propose 2-3 completely NEW chemical formulas that could be room-temperature superconductors.

Requirements for ideal candidates:
- Critical temperature (Tc) >= 293K (room temperature, 20C)
- Achievable at low or ambient pressure (< 10 GPa ideal, < 50 GPa acceptable)
- Must exhibit the Meissner effect (complete expulsion of magnetic flux from interior)
- Must exhibit zero electrical resistance below Tc
- Must support Cooper pair formation (phonon-mediated or unconventional)
- Should maintain quantum coherence for potential qubit applications
- Must be thermodynamically stable or metastable at operating conditions

Consider these strategies:
1. Hydrogen-rich compounds under moderate pressure (superhydrides)
2. Layered cuprate-like structures with enhanced Cu-O planes
3. Iron-based pnictide/chalcogenide variants with optimized doping
4. Nickelate superconductors (infinite-layer structures)
5. Topological superconductors with Majorana fermion potential

For each candidate, also describe the synthesis pathway - exactly how it would be made in a lab.

Return JSON with 'candidates' array:
- 'name' (descriptive)
- 'formula' (chemical formula)
- 'predictedTc' (Kelvin)
- 'pressureGpa' (required pressure)
- 'meissnerEffect' (boolean)
- 'zeroResistance' (boolean)
- 'cooperPairMechanism' (description)
- 'crystalStructure' (predicted)
- 'quantumCoherence' (0-1)
- 'roomTempViable' (boolean)
- 'synthesisPath' (object with 'method', 'steps' array, 'precursors' array, 'conditions' object)
- 'reasoning' (string under 200 chars explaining why this could work)`,
        },
        {
          role: "user",
          content: `Best candidates so far:\n${JSON.stringify(bestCandidates, null, 2)}\n\nPatterns discovered:\n${allInsights.slice(-8).join("\n")}\n\nPropose novel room-temperature superconductor candidates that improve upon these.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return 0;

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      return 0;
    }

    const newCandidates = parsed.candidates ?? [];

    for (const c of newCandidates) {
      if (!c.formula) continue;
      const id = `sc-novel-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const features = extractFeatures(c.formula);

      try {
        await storage.insertSuperconductorCandidate({
          id,
          name: c.name || c.formula,
          formula: c.formula,
          predictedTc: c.predictedTc ?? null,
          pressureGpa: c.pressureGpa ?? null,
          meissnerEffect: c.meissnerEffect ?? false,
          zeroResistance: c.zeroResistance ?? false,
          cooperPairMechanism: c.cooperPairMechanism ?? null,
          crystalStructure: c.crystalStructure ?? null,
          quantumCoherence: c.quantumCoherence ?? null,
          stabilityScore: features.cooperPairStrength,
          synthesisPath: c.synthesisPath ?? null,
          mlFeatures: features as any,
          xgboostScore: null,
          neuralNetScore: null,
          ensembleScore: c.quantumCoherence ?? 0.5,
          roomTempViable: c.roomTempViable ?? false,
          status: "novel-design",
          notes: c.reasoning ?? null,
        });
        generated++;

        emit("prediction", {
          type: "novel-superconductor",
          id,
          name: c.name,
          formula: c.formula,
          predictedTc: c.predictedTc,
          roomTempViable: c.roomTempViable,
          meissnerEffect: c.meissnerEffect,
          hasSynthesisPath: !!c.synthesisPath,
        });
      } catch (e: any) {
        emit("log", { phase: "phase-7", event: "Novel SC insert error", detail: `${c.formula}: ${e.message?.slice(0, 100)}`, dataSource: "SC Research" });
      }
    }

    if (generated > 0) {
      emit("log", {
        phase: "phase-7",
        event: "Novel superconductors designed",
        detail: `${generated} new designs with synthesis pathways, targeting Tc >= 293K`,
        dataSource: "SC Research",
      });
    }
  } catch (err: any) {
    emit("log", { phase: "phase-7", event: "Novel SC generation error", detail: err.message?.slice(0, 200), dataSource: "SC Research" });
  }

  return generated;
}
