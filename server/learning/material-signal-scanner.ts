import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { classifyFamily } from "./utils";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export interface MaterialSignal {
  id: string;
  name: string;
  description: string;
  keywords: string[];
  elementHints: string[];
  propertyChecks: (candidate: any) => boolean;
  verificationPrompt: (formula: string) => string;
}

const MATERIAL_SIGNALS: MaterialSignal[] = [
  {
    id: "next-gen-energy",
    name: "Next-Generation Energy Material",
    description: "Highly efficient materials for solar cells (improved perovskites) or high-capacity, fast-charging, safe battery materials (solid-state electrolytes)",
    keywords: ["perovskite", "photovoltaic", "solar", "electrolyte", "battery", "cathode", "anode", "lithium", "sodium", "solid-state"],
    elementHints: ["Li", "Na", "K", "Pb", "Sn", "I", "Br", "Cl", "Ti", "Zr", "La", "Ce", "Ba", "Sr", "Cs"],
    propertyChecks: (c) => {
      const fam = classifyFamily(c.formula);
      return fam === "Perovskite" || fam === "Oxide" || /Li|Na/.test(c.formula);
    },
    verificationPrompt: (formula) =>
      `Could ${formula} function as a next-generation energy material — specifically as an efficient solar cell absorber (perovskite-type or similar) or as a solid-state battery electrolyte with high ionic conductivity? Consider its crystal structure plausibility, electronic band gap suitability (1.1-1.8 eV for solar, wide gap for electrolytes), and ionic transport characteristics.`,
  },
  {
    id: "carbon-nanomaterial",
    name: "Advanced Carbon Nanomaterial",
    description: "Graphene and carbon nanotube derivatives offering unprecedented strength and electrical conductivity for aerospace, electronics, and composites",
    keywords: ["graphene", "nanotube", "CNT", "fullerene", "carbon fiber", "diamond-like"],
    elementHints: ["C", "B", "N"],
    propertyChecks: (c) => {
      const f = c.formula;
      return /^C\d*$/.test(f) || /^(B|N)?C\d+/.test(f) || /^C\d+(B|N)/.test(f);
    },
    verificationPrompt: (formula) =>
      `Could ${formula} be classified as an advanced carbon nanomaterial — a graphene derivative, carbon nanotube variant, or novel sp2/sp3 carbon allotrope? Consider whether this composition could form a stable low-dimensional carbon structure with exceptional mechanical strength or electrical conductivity.`,
  },
  {
    id: "self-healing",
    name: "Self-Healing Material",
    description: "Smart polymers or metals that can automatically repair themselves after damage, extending product and infrastructure lifespan",
    keywords: ["self-heal", "shape-memory", "reversible", "dynamic bond", "vitrimers"],
    elementHints: ["Ni", "Ti", "Cu", "Zn", "Fe"],
    propertyChecks: (c) => {
      const f = c.formula;
      return (/NiTi/.test(f) || /CuZn/.test(f) || /CuAl/.test(f)) && !/(O\d|H\d)/.test(f);
    },
    verificationPrompt: (formula) =>
      `Could ${formula} exhibit self-healing or shape-memory properties? Consider whether this composition could form a martensitic phase transformation system (like NiTi), a shape-memory alloy, or a metallic glass with self-healing capability through diffusion-mediated crack closure.`,
  },
  {
    id: "biocompatible",
    name: "Biocompatible/Biodegradable Material",
    description: "Materials for medical implants, tissue engineering, and drug delivery that integrate with or safely degrade within the human body",
    keywords: ["biocompatible", "biodegradable", "implant", "scaffold", "hydroxyapatite", "bioglass"],
    elementHints: ["Ti", "Ca", "P", "Mg", "Zn", "Fe", "Si", "Zr", "Ta", "Nb"],
    propertyChecks: (c) => {
      const f = c.formula;
      return /Ti.*(O|N|Al|V|Zr|Nb|Ta)/.test(f) || /Ca.*P/.test(f) || /Mg.*(Zn|Ca|Sr)/.test(f) || /Zr.*O/.test(f);
    },
    verificationPrompt: (formula) =>
      `Could ${formula} be a biocompatible or biodegradable material suitable for medical implants, tissue engineering scaffolds, or drug delivery? Consider its potential for corrosion resistance in physiological environments, non-toxicity of constituent elements, and mechanical compatibility with bone or soft tissue.`,
  },
  {
    id: "high-temp-ceramic",
    name: "High-Temperature Ceramic/Alloy",
    description: "Lightweight materials that withstand extreme heat and environments, crucial for jet engines and space exploration",
    keywords: ["refractory", "ultra-high temperature", "UHTC", "thermal barrier", "superalloy", "turbine"],
    elementHints: ["Hf", "Zr", "Ta", "W", "Mo", "Re", "Nb", "Ti", "Si", "B", "C", "N"],
    propertyChecks: (c) => {
      const f = c.formula;
      return /(Hf|Zr|Ta|W|Mo)(C|B|N|Si)/.test(f) || /(Hf|Zr|Ta)(C|B)\d*$/.test(f) || /Si.*C.*N/.test(f);
    },
    verificationPrompt: (formula) =>
      `Could ${formula} function as a high-temperature ceramic or refractory alloy capable of withstanding temperatures above 1500C? Consider its melting point potential, oxidation resistance, mechanical stability at elevated temperatures, and weight-to-strength ratio for aerospace applications.`,
  },
  {
    id: "sustainable-eco",
    name: "Sustainable/Eco-friendly Material",
    description: "Biodegradable materials, recyclable composites, and materials with low carbon footprints to combat pollution",
    keywords: ["biodegradable", "recyclable", "green", "sustainable", "eco-friendly", "low-carbon"],
    elementHints: ["Fe", "Al", "Si", "Ca", "Mg", "Na", "K", "O", "C", "N"],
    propertyChecks: (c) => {
      const f = c.formula;
      const earthAbundant = ["Fe", "Al", "Si", "Ca", "Mg", "Na", "K", "Ti", "Mn"];
      const toxic = ["Cd", "Hg", "Tl", "Pb", "As", "Be"];
      const hasToxic = toxic.some(e => f.includes(e));
      const hasAbundant = earthAbundant.filter(e => f.includes(e)).length >= 2;
      return hasAbundant && !hasToxic;
    },
    verificationPrompt: (formula) =>
      `Could ${formula} be classified as a sustainable or eco-friendly material? Consider whether its constituent elements are earth-abundant and non-toxic, whether it could be produced with low energy input, and whether it could serve as a replacement for environmentally harmful materials in construction, packaging, or industrial applications.`,
  },
  {
    id: "metamaterial",
    name: "Metamaterial",
    description: "Artificially engineered materials with properties not found in nature, such as negative refractive index for cloaking or advanced optics",
    keywords: ["metamaterial", "negative refraction", "cloaking", "photonic crystal", "plasmonic", "left-handed"],
    elementHints: ["Au", "Ag", "Cu", "Al", "Si", "Ge", "SrTiO3"],
    propertyChecks: (c) => {
      const f = c.formula;
      return /(Au|Ag).*(Si|Ge|Al|Cu)/.test(f) || /Sr.*Ti.*O/.test(f);
    },
    verificationPrompt: (formula) =>
      `Could ${formula} serve as a building block for metamaterial applications — exhibiting unusual electromagnetic properties such as negative refractive index, strong plasmonic resonance, or photonic band gap behavior? Consider its dielectric properties, plasmonic response of metallic components, and structural periodicity potential.`,
  },
  {
    id: "quantum-topological",
    name: "Quantum Material/Topological Insulator",
    description: "Materials with unique electronic states useful for quantum computing, spintronics, and topological electronics",
    keywords: ["topological", "Dirac", "Weyl", "spin-orbit", "quantum spin Hall", "Majorana", "spintronics"],
    elementHints: ["Bi", "Sb", "Te", "Se", "Sn", "Pb", "Hg", "Mn", "Cr", "V"],
    propertyChecks: (c) => {
      const f = c.formula;
      return /(Bi|Sb).*(Te|Se)/.test(f) || /(Sn|Pb).*(Te|Se)/.test(f) || /(Hg|Cd)Te/.test(f) || /Mn.*(Bi|Sb)/.test(f);
    },
    verificationPrompt: (formula) =>
      `Could ${formula} be a topological insulator or quantum material? Consider whether its heavy elements could produce strong spin-orbit coupling, whether its band structure could support topological surface states or Dirac cone dispersion, and whether it belongs to a known topological material family (Bi2Se3-type, SnTe-type, Heusler).`,
  },
  {
    id: "semiconductor",
    name: "Traditional Semiconductor",
    description: "Materials for conventional and next-generation semiconductor applications including transistors, LEDs, and power electronics",
    keywords: ["semiconductor", "transistor", "LED", "photoconductor", "band gap", "doping", "p-n junction"],
    elementHints: ["Si", "Ge", "Ga", "As", "In", "P", "N", "Al", "Zn", "Cd", "Se", "Te", "Sn"],
    propertyChecks: (c) => {
      const f = c.formula;
      return /(Ga|In|Al).*(As|N|P|Sb)/.test(f) || /(Zn|Cd).*(O|S|Se|Te)/.test(f) || /Si.*Ge/.test(f) || /Si.*C/.test(f);
    },
    verificationPrompt: (formula) =>
      `Could ${formula} function as a traditional or next-generation semiconductor material? Consider its potential band gap (0.5-3.5 eV range), crystal structure suitability for doping, carrier mobility prospects, and whether it could serve in transistors, LEDs, power electronics, or optoelectronic devices.`,
  },
];

const scannedThisCycle = new Set<string>();
const rejectedCooldown = new Map<string, number>();
const REJECTED_COOLDOWN_CYCLES = 20;
const MAX_VERIFICATIONS_PER_CYCLE = 8;

interface SignalMatch {
  signal: MaterialSignal;
  formula: string;
  candidateName: string;
  matchScore: number;
  matchReasons: string[];
}

function scoreCandidate(candidate: any, signal: MaterialSignal): SignalMatch | null {
  const formula: string = candidate.formula || "";
  if (!formula) return null;

  let score = 0;
  const reasons: string[] = [];

  const elementMatches = signal.elementHints.filter(el => formula.includes(el));
  if (elementMatches.length >= 2) {
    score += 0.4;
    reasons.push(`Contains signal elements: ${elementMatches.join(", ")}`);
  } else if (elementMatches.length === 1) {
    score += 0.15;
    reasons.push(`Contains signal element: ${elementMatches[0]}`);
  }

  if (signal.propertyChecks(candidate)) {
    score += 0.4;
    reasons.push("Matches structural/property pattern");
  }

  const notes = ((candidate.notes || "") + " " + (candidate.predictedProperties?.description || "")).toLowerCase();
  const keywordMatches = signal.keywords.filter(kw => notes.includes(kw.toLowerCase()));
  if (keywordMatches.length > 0) {
    score += 0.2;
    reasons.push(`Keyword matches: ${keywordMatches.join(", ")}`);
  }

  if (score < 0.35) return null;

  return {
    signal,
    formula,
    candidateName: candidate.name || formula,
    matchScore: Math.min(1, score),
    matchReasons: reasons,
  };
}

async function verifyWithOpenAI(match: SignalMatch): Promise<{ valid: number; reasoning: string }> {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a materials science expert verifier. You will be asked whether a specific chemical composition could realistically serve a particular materials application. Respond with ONLY valid JSON: {"valid": 1, "reasoning": "..."} if the material is a plausible candidate, or {"valid": 0, "reasoning": "..."} if it is not. The reasoning should be 1-3 sentences explaining your assessment. Be scientifically rigorous — do not confirm implausible claims.`,
        },
        {
          role: "user",
          content: match.signal.verificationPrompt(match.formula),
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 300,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return { valid: 0, reasoning: "OpenAI returned empty response" };

    const parsed = JSON.parse(content);
    return {
      valid: parsed.valid === 1 ? 1 : 0,
      reasoning: parsed.reasoning || "No reasoning provided",
    };
  } catch (err: any) {
    return { valid: 0, reasoning: `Verification error: ${err.message?.slice(0, 100) || "Unknown"}` };
  }
}

export async function scanMaterialSignals(
  emit: EventEmitter,
  broadcast: (type: string, data: any) => void,
  cycleNumber: number,
): Promise<number> {
  scannedThisCycle.clear();
  let signalsDetected = 0;

  try {
    const candidates = await storage.getSuperconductorCandidates(50);
    const novelPredictions = await storage.getNovelPredictions(30);

    const allCandidates = [
      ...candidates.map(c => ({
        formula: c.formula,
        name: c.formula,
        notes: "",
        predictedProperties: { description: classifyFamily(c.formula) },
        source: "superconductor" as const,
      })),
      ...novelPredictions.map(np => ({
        formula: np.formula,
        name: np.name,
        notes: np.notes || "",
        predictedProperties: np.predictedProperties as any,
        source: "novel" as const,
      })),
    ];

    const matches: SignalMatch[] = [];
    for (const candidate of allCandidates) {
      if (scannedThisCycle.has(candidate.formula)) continue;
      scannedThisCycle.add(candidate.formula);

      for (const signal of MATERIAL_SIGNALS) {
        const match = scoreCandidate(candidate, signal);
        if (match) {
          matches.push(match);
        }
      }
    }

    matches.sort((a, b) => b.matchScore - a.matchScore);

    const existingMilestones = await storage.getMilestones(500);
    let verificationsThisCycle = 0;

    for (const match of matches) {
      if (verificationsThisCycle >= MAX_VERIFICATIONS_PER_CYCLE) break;

      const dedupeKey = `${match.signal.id}::${match.formula}`;

      const alreadyDiscovered = existingMilestones.some(m =>
        m.type === `signal-${match.signal.id}` && m.relatedFormula === match.formula
      );
      if (alreadyDiscovered) continue;

      const cooldownUntil = rejectedCooldown.get(dedupeKey);
      if (cooldownUntil && cycleNumber < cooldownUntil) continue;

      emit("log", {
        phase: "signal-scanner",
        event: `Signal triggered: ${match.signal.name}`,
        detail: `${match.formula} matched with score ${match.matchScore.toFixed(2)}. Reasons: ${match.matchReasons.join("; ")}. Verifying with AI...`,
        dataSource: "Signal Scanner",
      });

      verificationsThisCycle++;
      const verification = await verifyWithOpenAI(match);
      signalsDetected++;

      if (verification.valid === 1) {
        emit("log", {
          phase: "signal-scanner",
          event: `Signal VERIFIED: ${match.formula} as ${match.signal.name}`,
          detail: `AI Verification: ${verification.reasoning}`,
          dataSource: "Signal Scanner",
        });

        const milestoneId = `ms-sig-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        try {
          await storage.insertMilestone({
            id: milestoneId,
            cycle: cycleNumber,
            type: `signal-${match.signal.id}`,
            title: `${match.signal.name}: ${match.formula}`,
            description: `${match.formula} identified as a potential ${match.signal.name.toLowerCase()}. ${verification.reasoning}`,
            significance: 2,
            relatedFormula: match.formula,
          });

          broadcast("milestone", {
            id: milestoneId,
            cycle: cycleNumber,
            type: `signal-${match.signal.id}`,
            title: `${match.signal.name}: ${match.formula}`,
            description: `${match.formula} identified as a potential ${match.signal.name.toLowerCase()}. ${verification.reasoning}`,
            significance: 2,
            relatedFormula: match.formula,
          });
        } catch {}

        const novelId = `signal-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        try {
          const existing = await storage.getNovelPredictionByFormula(match.formula);
          if (!existing) {
            await storage.insertNovelPrediction({
              id: novelId,
              name: `${match.candidateName} (${match.signal.name})`,
              formula: match.formula,
              predictedProperties: {
                description: `Potential ${match.signal.name.toLowerCase()}`,
                signalType: match.signal.id,
                verificationReasoning: verification.reasoning,
                matchScore: match.matchScore,
              },
              confidence: Math.min(0.85, match.matchScore),
              targetApplication: match.signal.name,
              status: "signal-verified",
              notes: `Signal scanner: ${match.matchReasons.join("; ")}. AI verification: ${verification.reasoning}`,
            });

            emit("prediction", {
              id: novelId,
              name: `${match.candidateName} (${match.signal.name})`,
              formula: match.formula,
              confidence: Math.min(0.85, match.matchScore),
              targetApplication: match.signal.name,
            });
          } else {
            await storage.updateNovelPrediction(existing.id, {
              notes: `${existing.notes || ""} | Signal: ${match.signal.name} verified. ${verification.reasoning}`,
            });
          }
        } catch {}

        storage.insertResearchLog({
          phase: "signal-scanner",
          event: `Material discovery signal: ${match.signal.name}`,
          detail: `${match.formula} verified as potential ${match.signal.name.toLowerCase()} (score: ${match.matchScore.toFixed(2)}). ${verification.reasoning}`,
          dataSource: "Signal Scanner",
        }).catch(() => {});

      } else {
        rejectedCooldown.set(dedupeKey, cycleNumber + REJECTED_COOLDOWN_CYCLES);

        emit("log", {
          phase: "signal-scanner",
          event: `Signal NOT verified: ${match.formula} as ${match.signal.name}`,
          detail: `AI determined this is NOT a valid ${match.signal.name.toLowerCase()}. Reasoning: ${verification.reasoning}. The signal detection heuristic may need refinement, or this composition lacks the required properties for this application category.`,
          dataSource: "Signal Scanner",
        });

        storage.insertResearchLog({
          phase: "signal-scanner",
          event: `Signal rejected: ${match.signal.name}`,
          detail: `${match.formula} failed AI verification as ${match.signal.name.toLowerCase()}. ${verification.reasoning}. Signal match score was ${match.matchScore.toFixed(2)} (${match.matchReasons.join("; ")}), but AI assessment found this candidate does not meet the criteria. This may indicate a false positive in the signal detection heuristic or an edge-case composition.`,
          dataSource: "Signal Scanner",
        }).catch(() => {});
      }
    }

    if (signalsDetected > 0) {
      emit("log", {
        phase: "signal-scanner",
        event: `Signal scan complete`,
        detail: `Evaluated ${topMatches.length} signal matches across ${scannedThisCycle.size} candidates in cycle ${cycleNumber}.`,
        dataSource: "Signal Scanner",
      });
    }

  } catch (err: any) {
    emit("log", {
      phase: "signal-scanner",
      event: "Signal scanner error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "Signal Scanner",
    });
  }

  return signalsDetected;
}

export function getSignalDefinitions(): MaterialSignal[] {
  return MATERIAL_SIGNALS;
}
