import OpenAI from "openai";
import { storage } from "../storage";
import type { Material } from "@shared/schema";
import type { EventEmitter } from "./engine";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export async function discoverSynthesisProcesses(
  emit: EventEmitter,
  materials: Material[]
): Promise<number> {
  if (materials.length === 0) return 0;
  let discovered = 0;

  emit("log", {
    phase: "phase-8",
    event: "Synthesis discovery started",
    detail: `Analyzing creation processes for ${materials.length} materials`,
    dataSource: "Synthesis Engine",
  });

  const batch = materials.slice(0, 8);
  const materialList = batch.map(m => `${m.name} (${m.formula})`).join(", ");

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a materials synthesis expert. For each material, determine exactly how it is created in a laboratory. Think like a chemist: what raw materials (precursors) go in, what conditions (temperature, pressure, atmosphere, time) are needed, what equipment is used, what are the step-by-step synthesis instructions.

Consider all synthesis methods: solid-state reaction, sol-gel, chemical vapor deposition (CVD), hydrothermal, sputtering, molecular beam epitaxy, arc melting, ball milling, Czochralski growth, zone melting, electrodeposition, etc.

For each material think about the fundamental chemistry: just as water is made by combining hydrogen and oxygen (2H2 + O2 -> 2H2O via combustion or electrolysis reversal), every material has a creation pathway from simpler building blocks.

Return JSON with key 'processes' containing an array of objects:
- 'materialName' (string)
- 'formula' (string)
- 'method' (primary synthesis method name)
- 'conditions' (object with 'temperature' in Celsius, 'pressure' in atm, 'atmosphere' string, 'duration' string)
- 'steps' (array of 3-6 step strings describing the process)
- 'precursors' (array of starting material strings, e.g. "Y2O3", "BaCO3", "CuO")
- 'equipment' (array of required equipment strings)
- 'difficulty' ("easy"|"moderate"|"hard"|"extreme")
- 'timeEstimate' (string like "24 hours", "3 days")
- 'safetyNotes' (string, brief safety considerations)
- 'yieldPercent' (number 0-100, typical yield)`,
        },
        {
          role: "user",
          content: `Determine the complete synthesis process for each material: ${materialList}`,
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
      emit("log", { phase: "phase-8", event: "Synthesis parse error", detail: content.slice(0, 200), dataSource: "Synthesis Engine" });
      return 0;
    }

    const processes = parsed.processes ?? [];

    for (const proc of processes) {
      if (!proc.formula || !proc.method) continue;
      const matchedMat = batch.find(m => m.formula === proc.formula || m.name === proc.materialName);
      const id = `synth-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

      try {
        await storage.insertSynthesisProcess({
          id,
          materialId: matchedMat?.id ?? null,
          materialName: proc.materialName || proc.formula,
          formula: proc.formula,
          method: proc.method,
          conditions: proc.conditions || {},
          steps: proc.steps || [],
          precursors: proc.precursors || [],
          equipment: proc.equipment || [],
          difficulty: proc.difficulty || "moderate",
          timeEstimate: proc.timeEstimate || null,
          safetyNotes: proc.safetyNotes || null,
          yieldPercent: proc.yieldPercent ?? null,
        });
        discovered++;
      } catch (e: any) {
        emit("log", { phase: "phase-8", event: "Synthesis insert error", detail: `${proc.formula}: ${e.message?.slice(0, 100)}`, dataSource: "Synthesis Engine" });
      }
    }

    if (discovered > 0) {
      emit("log", {
        phase: "phase-8",
        event: "Synthesis processes discovered",
        detail: `Mapped creation pathways for ${discovered} materials`,
        dataSource: "Synthesis Engine",
      });
      emit("progress", { phase: 8, newItems: discovered });
    }
  } catch (err: any) {
    emit("log", {
      phase: "phase-8",
      event: "Synthesis discovery error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "Synthesis Engine",
    });
  }

  return discovered;
}

export async function discoverChemicalReactions(
  emit: EventEmitter,
  focusArea: string = "superconductor synthesis"
): Promise<number> {
  let discovered = 0;

  emit("log", {
    phase: "phase-9",
    event: "Chemical reaction discovery started",
    detail: `Learning lab processes relevant to: ${focusArea}`,
    dataSource: "Reaction Engine",
  });

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a chemistry expert specializing in laboratory processes for advanced materials synthesis. Generate chemical reactions that are critical for creating superconducting materials and their precursors.

Include reactions for:
- Oxide formation (calcination, oxidation)
- Reduction reactions (hydrogen reduction, carbothermic reduction)
- High-pressure synthesis (diamond anvil cell reactions)
- Hydride formation (hydrogenation under pressure)
- Crystal growth reactions
- Doping reactions (substitutional, interstitial)
- Thin film deposition reactions (CVD precursor reactions)
- Electrochemical reactions relevant to material processing
- Solid-state reactions between precursor powders

For each reaction, explain the exact mechanism: what bonds break, what bonds form, how electrons rearrange, what drives the reaction thermodynamically.

Return JSON with 'reactions' array of objects:
- 'name' (descriptive name)
- 'equation' (balanced chemical equation string)
- 'reactionType' (e.g. "solid-state", "oxidation", "reduction", "hydrogenation", "decomposition", "precipitation", "CVD")
- 'reactants' (array of objects with 'formula' and 'role' keys)
- 'products' (array of objects with 'formula' and 'role' keys)
- 'conditions' (object with 'temperature', 'pressure', 'atmosphere', 'catalyst' keys)
- 'energetics' (object with 'deltaH' kJ/mol, 'deltaG' kJ/mol, 'activationEnergy' kJ/mol)
- 'mechanism' (string explaining bond breaking/forming, 2-3 sentences)
- 'relevanceToSuperconductor' (0-1 float, how relevant to room-temp superconductor discovery)
- 'labProcess' (string describing the real lab procedure in 1-2 sentences)`,
        },
        {
          role: "user",
          content: `Generate 4-6 chemical reactions critical for ${focusArea}. Focus on processes that could help create a room-temperature superconductor - including precursor synthesis, high-pressure reactions, and doping processes.`,
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
      emit("log", { phase: "phase-9", event: "Reaction parse error", detail: content.slice(0, 200), dataSource: "Reaction Engine" });
      return 0;
    }

    const reactions = parsed.reactions ?? [];

    for (const rxn of reactions) {
      if (!rxn.equation || !rxn.name) continue;
      const id = `rxn-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

      try {
        await storage.insertChemicalReaction({
          id,
          name: rxn.name,
          equation: rxn.equation,
          reactionType: rxn.reactionType || "unknown",
          reactants: rxn.reactants || [],
          products: rxn.products || [],
          conditions: rxn.conditions || {},
          energetics: rxn.energetics || null,
          mechanism: rxn.mechanism || null,
          relevanceToSuperconductor: rxn.relevanceToSuperconductor ?? 0,
          labProcess: rxn.labProcess || null,
          source: "AI Analysis",
        });
        discovered++;
      } catch (e: any) {
        emit("log", { phase: "phase-9", event: "Reaction insert error", detail: `${rxn.name}: ${e.message?.slice(0, 100)}`, dataSource: "Reaction Engine" });
      }
    }

    if (discovered > 0) {
      emit("log", {
        phase: "phase-9",
        event: "Chemical reactions catalogued",
        detail: `Learned ${discovered} lab processes for ${focusArea}`,
        dataSource: "Reaction Engine",
      });
      emit("progress", { phase: 9, newItems: discovered });
    }
  } catch (err: any) {
    emit("log", {
      phase: "phase-9",
      event: "Reaction discovery error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "Reaction Engine",
    });
  }

  return discovered;
}

const REACTION_TOPICS = [
  "superconductor synthesis",
  "cuprate oxide formation",
  "hydride compression under extreme pressure",
  "thin film superconductor deposition",
  "crystal growth for quantum materials",
  "doping reactions for Tc enhancement",
  "precursor powder preparation for YBCO",
  "hydrogen loading into metal lattices",
  "pnictide superconductor synthesis",
  "topological material fabrication",
];
let topicIndex = 0;

export function getNextReactionTopic(): string {
  const t = REACTION_TOPICS[topicIndex % REACTION_TOPICS.length];
  topicIndex++;
  return t;
}
