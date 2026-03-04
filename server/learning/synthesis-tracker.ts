import OpenAI from "openai";
import { storage } from "../storage";
import type { Material } from "@shared/schema";
import type { EventEmitter } from "./engine";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

const SYNTHESIS_VALIDATION_BOUNDS = {
  temperature: { min: 0, max: 4000 },
  pressure: { min: 0, max: 500 },
};

interface SynthesisValidationResult {
  valid: boolean;
  rejectionReasons: string[];
}

function validateSynthesisConditions(proc: any): SynthesisValidationResult {
  const rejectionReasons: string[] = [];
  const conditions = proc.conditions || {};

  if (conditions.temperature !== undefined && conditions.temperature !== null) {
    const tempC = typeof conditions.temperature === "number" ? conditions.temperature : parseFloat(conditions.temperature);
    if (!isNaN(tempC)) {
      const tempK = tempC + 273.15;
      if (tempK < SYNTHESIS_VALIDATION_BOUNDS.temperature.min) {
        rejectionReasons.push(`Synthesis temperature ${tempC}C (${tempK.toFixed(0)}K) below absolute zero`);
      }
      if (tempK > SYNTHESIS_VALIDATION_BOUNDS.temperature.max) {
        rejectionReasons.push(`Synthesis temperature ${tempC}C (${tempK.toFixed(0)}K) exceeds ${SYNTHESIS_VALIDATION_BOUNDS.temperature.max}K limit`);
      }
    }
  }

  if (conditions.pressure !== undefined && conditions.pressure !== null) {
    const pressureAtm = typeof conditions.pressure === "number" ? conditions.pressure : parseFloat(conditions.pressure);
    if (!isNaN(pressureAtm)) {
      const pressureGpa = pressureAtm * 0.000101325;
      if (pressureGpa < SYNTHESIS_VALIDATION_BOUNDS.pressure.min) {
        rejectionReasons.push(`Synthesis pressure ${pressureAtm} atm is negative`);
      }
      if (pressureGpa > SYNTHESIS_VALIDATION_BOUNDS.pressure.max) {
        rejectionReasons.push(`Synthesis pressure ${pressureGpa.toFixed(2)} GPa exceeds ${SYNTHESIS_VALIDATION_BOUNDS.pressure.max} GPa limit`);
      }
    }
  }

  if (conditions.duration !== undefined && conditions.duration !== null) {
    const durationStr = String(conditions.duration).toLowerCase();
    const numMatch = durationStr.match(/[\d.]+/);
    if (numMatch) {
      const durationVal = parseFloat(numMatch[0]);
      if (durationVal <= 0) {
        rejectionReasons.push(`Synthesis duration must be positive, got: ${conditions.duration}`);
      }
    }
  }

  if (proc.yieldPercent !== undefined && proc.yieldPercent !== null) {
    if (proc.yieldPercent < 0 || proc.yieldPercent > 100) {
      rejectionReasons.push(`Yield ${proc.yieldPercent}% outside valid range [0, 100]`);
    }
  }

  return { valid: rejectionReasons.length === 0, rejectionReasons };
}

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

  const batch = materials.slice(0, 4);
  const materialList = batch.map(m => `${m.name} (${m.formula})`).join(", ");

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a materials synthesis expert drawing from peer-reviewed literature (Journal of the American Chemical Society, Nature Materials, Acta Materialia, Journal of Materials Science) and established laboratory procedures.

For each material, determine EXACTLY how it is created in a real laboratory. Provide precise, quantitative details a lab technician could follow:

CRITICAL DETAILS TO INCLUDE:
- EXACT temperatures in Celsius (not ranges - give specific values, e.g. "950C" not "high temperature")
- EXACT heating durations (e.g. "hold at 950C for 12 hours" not "heat for a long time")
- Heating rates (e.g. "ramp at 5C/min from room temperature to 950C")
- Cooling rates (e.g. "furnace cool to room temperature over 6 hours" or "quench in liquid nitrogen")
- Atmosphere (e.g. "flowing oxygen at 50 mL/min" or "argon with 5% hydrogen")
- Pressure (e.g. "1 atm" or "200 MPa in hot isostatic press")
- Intermediate grinding/mixing steps with specifics (e.g. "ball mill at 300 rpm for 4 hours with zirconia media")
- Number of sintering cycles if applicable

Consider all synthesis methods: solid-state reaction, sol-gel, chemical vapor deposition (CVD), hydrothermal, sputtering, molecular beam epitaxy (MBE), arc melting, ball milling, Czochralski growth, Bridgman method, zone melting, electrodeposition, co-precipitation, spray pyrolysis, pulsed laser deposition (PLD), etc.

Return JSON with key 'processes' containing an array of objects:
- 'materialName' (string)
- 'formula' (string)
- 'method' (primary synthesis method name)
- 'conditions' (object with:
    'temperature' (number in Celsius, the peak temperature),
    'heatingRate' (string, e.g. "5C/min"),
    'holdTime' (string, e.g. "12 hours at 950C"),
    'coolingMethod' (string, e.g. "furnace cool over 8 hours"),
    'pressure' (number in atm),
    'atmosphere' (string, be specific),
    'intermediateSteps' (string, e.g. "regrind after first sintering"),
    'duration' (string, total process time)
  )
- 'steps' (array of 4-8 detailed step strings, each step should be specific enough to follow in a lab)
- 'precursors' (array of starting material strings with purity, e.g. "Y2O3 (99.99% purity)")
- 'equipment' (array of required equipment strings)
- 'difficulty' ("easy"|"moderate"|"hard"|"extreme")
- 'timeEstimate' (string like "24 hours total including cooling")
- 'safetyNotes' (string, specific hazards and precautions)
- 'yieldPercent' (number 0-100, typical yield from literature)`,
        },
        {
          role: "user",
          content: `Determine the complete, detailed synthesis process for each material with exact temperatures, heating times, and conditions that a real lab could follow: ${materialList}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 3500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return 0;

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      const truncatedResponse = content.length > 3000;
      if (truncatedResponse) {
        emit("log", { phase: "phase-8", event: "Synthesis response truncated", detail: `Response was ${content.length} chars - increasing detail may need more tokens. Retrying with fewer materials next cycle.`, dataSource: "Synthesis Engine" });
      } else {
        emit("log", { phase: "phase-8", event: "Synthesis parse error", detail: content.slice(0, 200), dataSource: "Synthesis Engine" });
      }
      return 0;
    }

    const processes = parsed.processes ?? [];

    for (const proc of processes) {
      if (!proc.formula || !proc.method) continue;

      const synthValidation = validateSynthesisConditions(proc);
      if (!synthValidation.valid) {
        emit("log", { phase: "phase-8", event: "Synthesis process rejected", detail: `${proc.formula}: ${synthValidation.rejectionReasons.join("; ")}`, dataSource: "Synthesis Engine" });
        continue;
      }

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
        detail: `Mapped ${discovered} detailed creation pathways with exact temperatures, durations, and conditions`,
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
  focusArea: string = "general materials chemistry"
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
          content: `You are a chemistry expert with deep knowledge from peer-reviewed sources (JACS, Angewandte Chemie, Chemical Reviews, Inorganic Chemistry). Generate chemical reactions that are fundamental to materials science and laboratory practice.

For each reaction provide PRECISE quantitative details:
- Exact balanced equation with states (s, l, g, aq)
- Specific temperature ranges from published literature
- Exact thermodynamic values from NIST or CRC Handbook data
- Mechanism at the atomic level: which bonds break (bond dissociation energy), which form, electron transfer details
- Real laboratory procedure: what a chemist actually does step by step

Include reactions across ALL areas of materials chemistry, not just superconductors:
- Oxide synthesis (calcination, oxidation, thermal decomposition)
- Reduction reactions (hydrogen reduction, carbothermic, metallothermic)
- Acid-base reactions in materials processing
- Precipitation and co-precipitation for nanoparticles
- Sol-gel hydrolysis and condensation
- Combustion synthesis (self-propagating high-temperature)
- Electrochemical reactions (electrodeposition, anodization)
- Vapor-phase reactions (CVD, ALD precursor chemistry)
- Solid-state diffusion reactions
- High-pressure synthesis reactions
- Polymerization reactions for polymer materials
- Corrosion and oxidation reactions
- Battery electrode reactions (charge/discharge chemistry)

Return JSON with 'reactions' array of objects:
- 'name' (descriptive name)
- 'equation' (balanced chemical equation with states, e.g. "2H2(g) + O2(g) -> 2H2O(l)")
- 'reactionType' (e.g. "solid-state", "oxidation", "reduction", "hydrogenation", "decomposition", "precipitation", "CVD", "sol-gel", "combustion", "electrochemical")
- 'reactants' (array of objects with 'formula', 'role', and 'state' keys)
- 'products' (array of objects with 'formula', 'role', and 'state' keys)
- 'conditions' (object with 'temperature' string with exact value and unit, 'pressure' string, 'atmosphere' string, 'catalyst' string or null, 'duration' string)
- 'energetics' (object with 'deltaH' kJ/mol from published data, 'deltaG' kJ/mol, 'activationEnergy' kJ/mol)
- 'mechanism' (string explaining bond breaking/forming at atomic level, 3-4 sentences)
- 'relevanceToSuperconductor' (0-1 float)
- 'labProcess' (string describing the real lab procedure in 2-3 sentences with specific details)
- 'source' (string citing the type of source, e.g. "NIST Chemistry WebBook", "CRC Handbook", "Atkins Physical Chemistry")`,
        },
        {
          role: "user",
          content: `Generate 5-7 important chemical reactions for the topic: ${focusArea}. Include exact thermodynamic values and detailed mechanisms. These should be real, well-documented reactions from established chemistry literature.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 3500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return 0;

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      const truncated = content.length > 3000;
      if (truncated) {
        emit("log", { phase: "phase-9", event: "Reaction response truncated", detail: `Response was ${content.length} chars - may need reduced detail per reaction`, dataSource: "Reaction Engine" });
      } else {
        emit("log", { phase: "phase-9", event: "Reaction parse error", detail: content.slice(0, 200), dataSource: "Reaction Engine" });
      }
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
          source: rxn.source || "Chemistry Literature",
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
        detail: `Learned ${discovered} reactions with thermodynamic data for ${focusArea}`,
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
  "fundamental oxide formation and thermal decomposition",
  "metal reduction reactions in extractive metallurgy",
  "sol-gel chemistry for ceramic nanoparticles",
  "electrochemical reactions in lithium-ion batteries",
  "combustion synthesis and self-propagating reactions",
  "chemical vapor deposition precursor reactions",
  "corrosion and passivation of structural metals",
  "precipitation reactions for catalyst preparation",
  "polymer synthesis and cross-linking reactions",
  "superconductor precursor oxide formation",
  "cuprate oxide formation for YBCO synthesis",
  "hydride compression reactions under extreme pressure",
  "thin film deposition chemistry (PLD, sputtering)",
  "crystal growth reactions from solution and melt",
  "doping reactions for semiconductor modification",
  "high-temperature solid-state sintering reactions",
  "acid-base reactions in materials processing",
  "photocatalytic reactions on semiconductor surfaces",
  "hydrogen evolution and oxygen evolution reactions",
  "atomic layer deposition half-reactions",
];
let topicIndex = 0;

export function getNextReactionTopic(): string {
  const t = REACTION_TOPICS[topicIndex % REACTION_TOPICS.length];
  topicIndex++;
  return t;
}
