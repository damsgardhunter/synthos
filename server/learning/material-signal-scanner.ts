import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { classifyFamily } from "./utils";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

interface CandidateData {
  formula: string;
  name: string;
  notes: string;
  predictedProperties: any;
  source: "superconductor" | "novel";
  predictedTc?: number | null;
  stabilityScore?: number | null;
  electronPhononCoupling?: number | null;
  dimensionality?: string | null;
  ambientPressureStable?: boolean | null;
  pressureGpa?: number | null;
  crystalStructure?: string | null;
  correlationStrength?: number | null;
  fermiSurfaceTopology?: string | null;
  bandGap?: number | null;
  z2Score?: number | null;
  chernScore?: number | null;
  topologicalScore?: number | null;
  socStrength?: number | null;
  dosAtEF?: number | null;
  formationEnergy?: number | null;
  decompositionEnergy?: number | null;
}

function getNum(val: any): number | null {
  if (val == null) return null;
  const n = Number(val);
  return isNaN(n) ? null : n;
}

export interface MaterialSignal {
  id: string;
  name: string;
  description: string;
  keywords: string[];
  elementHints: string[];
  formulaChecks: (formula: string) => boolean;
  datapointChecks: (c: CandidateData) => { score: number; reasons: string[] };
  verificationPrompt: (formula: string, datapoints: string[]) => string;
}

const MATERIAL_SIGNALS: MaterialSignal[] = [
  {
    id: "next-gen-energy",
    name: "Next-Generation Energy Material",
    description: "Highly efficient materials for solar cells (improved perovskites) or high-capacity, fast-charging, safe battery materials (solid-state electrolytes)",
    keywords: ["perovskite", "photovoltaic", "solar", "electrolyte", "battery", "cathode", "anode", "lithium", "sodium", "solid-state"],
    elementHints: ["Li", "Na", "K", "Pb", "Sn", "I", "Br", "Cl", "Ti", "Zr", "La", "Ce", "Ba", "Sr", "Cs"],
    formulaChecks: (f) => {
      const fam = classifyFamily(f);
      return fam === "Perovskite" || fam === "Oxide" || /Li|Na/.test(f);
    },
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      const bg = c.bandGap;
      if (bg != null) {
        if (bg >= 1.1 && bg <= 1.8) {
          score += 0.3;
          reasons.push(`Band gap ${bg.toFixed(2)} eV in optimal solar absorber range (1.1-1.8 eV)`);
        } else if (bg > 3.0) {
          score += 0.25;
          reasons.push(`Wide band gap ${bg.toFixed(2)} eV suitable for solid-state electrolyte`);
        }
      }
      if (c.stabilityScore != null && c.stabilityScore > 0.7) {
        score += 0.1;
        reasons.push(`High stability score ${c.stabilityScore.toFixed(2)}`);
      }
      if (c.ambientPressureStable === true || (c.pressureGpa != null && c.pressureGpa < 1)) {
        score += 0.1;
        reasons.push("Ambient pressure stable");
      }
      if (c.formationEnergy != null && c.formationEnergy < -0.3) {
        score += 0.05;
        reasons.push(`Favorable formation energy ${c.formationEnergy.toFixed(2)} eV/atom`);
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} function as a next-generation energy material — specifically as an efficient solar cell absorber (perovskite-type or similar) or as a solid-state battery electrolyte with high ionic conductivity? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider its crystal structure plausibility, electronic band gap suitability (1.1-1.8 eV for solar, wide gap for electrolytes), and ionic transport characteristics.`,
  },
  {
    id: "carbon-nanomaterial",
    name: "Advanced Carbon Nanomaterial",
    description: "Graphene and carbon nanotube derivatives offering unprecedented strength and electrical conductivity for aerospace, electronics, and composites",
    keywords: ["graphene", "nanotube", "CNT", "fullerene", "carbon fiber", "diamond-like"],
    elementHints: ["C", "B", "N"],
    formulaChecks: (f) => /^C\d*$/.test(f) || /^(B|N)?C\d+/.test(f) || /^C\d+(B|N)/.test(f),
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      const dim = (c.dimensionality || "").toLowerCase();
      if (dim.includes("2d") || dim.includes("1d") || dim.includes("quasi-2d")) {
        score += 0.25;
        reasons.push(`Low-dimensional structure (${c.dimensionality}) favorable for nanomaterial`);
      }
      if (c.electronPhononCoupling != null && c.electronPhononCoupling > 0.5) {
        score += 0.15;
        reasons.push(`Strong electron-phonon coupling lambda=${c.electronPhononCoupling.toFixed(2)} indicates good conductivity`);
      }
      if (c.dosAtEF != null && c.dosAtEF > 3) {
        score += 0.1;
        reasons.push(`High DOS at Fermi level (${c.dosAtEF.toFixed(1)} states/eV) indicates metallic character`);
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} be classified as an advanced carbon nanomaterial — a graphene derivative, carbon nanotube variant, or novel sp2/sp3 carbon allotrope? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider whether this composition could form a stable low-dimensional carbon structure with exceptional mechanical strength or electrical conductivity.`,
  },
  {
    id: "self-healing",
    name: "Self-Healing Material",
    description: "Smart polymers or metals that can automatically repair themselves after damage, extending product and infrastructure lifespan",
    keywords: ["self-heal", "shape-memory", "reversible", "dynamic bond", "vitrimers"],
    elementHints: ["Ni", "Ti", "Cu", "Zn", "Fe"],
    formulaChecks: (f) => (/NiTi/.test(f) || /CuZn/.test(f) || /CuAl/.test(f) || /FeNi/.test(f) || /TiNb/.test(f)) && !/(O\d|H\d)/.test(f),
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      if (c.stabilityScore != null && c.stabilityScore > 0.6) {
        score += 0.15;
        reasons.push(`Stability score ${c.stabilityScore.toFixed(2)} supports structural integrity`);
      }
      const cs = (c.crystalStructure || "").toLowerCase();
      if (cs.includes("b2") || cs.includes("b19") || cs.includes("bcc") || cs.includes("martensite")) {
        score += 0.25;
        reasons.push(`Crystal structure ${c.crystalStructure} associated with martensitic transformation`);
      }
      if (c.ambientPressureStable === true) {
        score += 0.1;
        reasons.push("Ambient pressure stable — practical for structural applications");
      }
      if (c.correlationStrength != null && c.correlationStrength < 0.3) {
        score += 0.05;
        reasons.push("Low correlation strength consistent with metallic bonding in shape-memory alloys");
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} exhibit self-healing or shape-memory properties? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider whether this composition could form a martensitic phase transformation system (like NiTi), a shape-memory alloy, or a metallic glass with self-healing capability through diffusion-mediated crack closure.`,
  },
  {
    id: "biocompatible",
    name: "Biocompatible/Biodegradable Material",
    description: "Materials for medical implants, tissue engineering, and drug delivery that integrate with or safely degrade within the human body",
    keywords: ["biocompatible", "biodegradable", "implant", "scaffold", "hydroxyapatite", "bioglass"],
    elementHints: ["Ti", "Ca", "P", "Mg", "Zn", "Fe", "Si", "Zr", "Ta", "Nb"],
    formulaChecks: (f) => /Ti.*(O|N|Al|V|Zr|Nb|Ta)/.test(f) || /Ca.*P/.test(f) || /Mg.*(Zn|Ca|Sr)/.test(f) || /Zr.*O/.test(f),
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      if (c.ambientPressureStable === true || (c.pressureGpa != null && c.pressureGpa < 1)) {
        score += 0.15;
        reasons.push("Ambient pressure stable — required for biomedical applications");
      }
      if (c.stabilityScore != null && c.stabilityScore > 0.7) {
        score += 0.15;
        reasons.push(`High stability ${c.stabilityScore.toFixed(2)} — resists degradation in physiological environment`);
      }
      const toxic = ["Cd", "Hg", "Tl", "Pb", "As", "Be", "Cr", "Co"];
      const hasToxic = toxic.some(e => c.formula.includes(e));
      if (!hasToxic) {
        score += 0.1;
        reasons.push("No toxic elements detected — biocompatibility favorable");
      }
      if (c.decompositionEnergy != null && c.decompositionEnergy > 0.1) {
        score += 0.1;
        reasons.push(`Decomposition energy ${c.decompositionEnergy.toFixed(2)} eV indicates controlled degradation potential`);
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} be a biocompatible or biodegradable material suitable for medical implants, tissue engineering scaffolds, or drug delivery? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider its potential for corrosion resistance in physiological environments, non-toxicity of constituent elements, and mechanical compatibility with bone or soft tissue.`,
  },
  {
    id: "high-temp-ceramic",
    name: "High-Temperature Ceramic/Alloy",
    description: "Lightweight materials that withstand extreme heat and environments, crucial for jet engines and space exploration",
    keywords: ["refractory", "ultra-high temperature", "UHTC", "thermal barrier", "superalloy", "turbine"],
    elementHints: ["Hf", "Zr", "Ta", "W", "Mo", "Re", "Nb", "Ti", "Si", "B", "C", "N"],
    formulaChecks: (f) => /(Hf|Zr|Ta|W|Mo)(C|B|N|Si)/.test(f) || /(Hf|Zr|Ta)(C|B)\d*$/.test(f) || /Si.*C.*N/.test(f),
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      if (c.formationEnergy != null && c.formationEnergy < -1.0) {
        score += 0.2;
        reasons.push(`Very negative formation energy ${c.formationEnergy.toFixed(2)} eV/atom — strong bonding indicates high melting point`);
      } else if (c.formationEnergy != null && c.formationEnergy < -0.5) {
        score += 0.1;
        reasons.push(`Negative formation energy ${c.formationEnergy.toFixed(2)} eV/atom — moderate bond strength`);
      }
      if (c.stabilityScore != null && c.stabilityScore > 0.8) {
        score += 0.15;
        reasons.push(`High thermodynamic stability (${c.stabilityScore.toFixed(2)})`);
      }
      if (c.electronPhononCoupling != null && c.electronPhononCoupling < 0.3) {
        score += 0.1;
        reasons.push(`Low electron-phonon coupling (${c.electronPhononCoupling.toFixed(2)}) consistent with stiff lattice`);
      }
      if (c.bandGap != null && c.bandGap > 2.0) {
        score += 0.1;
        reasons.push(`Wide band gap ${c.bandGap.toFixed(2)} eV typical of ceramic character`);
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} function as a high-temperature ceramic or refractory alloy capable of withstanding temperatures above 1500C? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider its melting point potential, oxidation resistance, mechanical stability at elevated temperatures, and weight-to-strength ratio for aerospace applications.`,
  },
  {
    id: "sustainable-eco",
    name: "Sustainable/Eco-friendly Material",
    description: "Biodegradable materials, recyclable composites, and materials with low carbon footprints to combat pollution",
    keywords: ["biodegradable", "recyclable", "green", "sustainable", "eco-friendly", "low-carbon"],
    elementHints: ["Fe", "Al", "Si", "Ca", "Mg", "Na", "K", "O", "C", "N"],
    formulaChecks: (f) => {
      const earthAbundant = ["Fe", "Al", "Si", "Ca", "Mg", "Na", "K", "Ti", "Mn"];
      const toxic = ["Cd", "Hg", "Tl", "Pb", "As", "Be"];
      const hasToxic = toxic.some(e => f.includes(e));
      const hasAbundant = earthAbundant.filter(e => f.includes(e)).length >= 2;
      return hasAbundant && !hasToxic;
    },
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      if (c.ambientPressureStable === true || (c.pressureGpa != null && c.pressureGpa < 1)) {
        score += 0.15;
        reasons.push("Ambient pressure stable — no extreme conditions needed for use");
      }
      if (c.formationEnergy != null && c.formationEnergy > -0.5 && c.formationEnergy < 0) {
        score += 0.15;
        reasons.push(`Low-energy synthesis feasible (formation energy ${c.formationEnergy.toFixed(2)} eV/atom)`);
      }
      if (c.stabilityScore != null && c.stabilityScore > 0.5) {
        score += 0.1;
        reasons.push(`Moderate stability (${c.stabilityScore.toFixed(2)}) — durable yet potentially recyclable`);
      }
      const rare = ["Rh", "Ir", "Os", "Ru", "Re", "Au", "Pt", "Pd"];
      const hasRare = rare.some(e => c.formula.includes(e));
      if (!hasRare) {
        score += 0.1;
        reasons.push("No rare/precious elements — low resource footprint");
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} be classified as a sustainable or eco-friendly material? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider whether its constituent elements are earth-abundant and non-toxic, whether it could be produced with low energy input, and whether it could serve as a replacement for environmentally harmful materials in construction, packaging, or industrial applications.`,
  },
  {
    id: "metamaterial",
    name: "Metamaterial",
    description: "Artificially engineered materials with properties not found in nature, such as negative refractive index for cloaking or advanced optics",
    keywords: ["metamaterial", "negative refraction", "cloaking", "photonic crystal", "plasmonic", "left-handed"],
    elementHints: ["Au", "Ag", "Cu", "Al", "Si", "Ge", "SrTiO3"],
    formulaChecks: (f) => /(Au|Ag).*(Si|Ge|Al|Cu)/.test(f) || /Sr.*Ti.*O/.test(f),
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      if (c.bandGap != null && c.bandGap > 0 && c.bandGap < 0.5) {
        score += 0.15;
        reasons.push(`Narrow band gap ${c.bandGap.toFixed(2)} eV — semimetallic character suitable for plasmonic response`);
      }
      if (c.dosAtEF != null && c.dosAtEF > 5) {
        score += 0.15;
        reasons.push(`High DOS at Fermi level (${c.dosAtEF.toFixed(1)}) supports strong optical response`);
      }
      const hasNoble = ["Au", "Ag", "Cu"].some(e => c.formula.includes(e));
      const hasDielectric = ["Si", "Ge", "Ti", "O", "Al"].some(e => c.formula.includes(e));
      if (hasNoble && hasDielectric) {
        score += 0.2;
        reasons.push("Metal-dielectric combination ideal for plasmonic metamaterial");
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} serve as a building block for metamaterial applications — exhibiting unusual electromagnetic properties such as negative refractive index, strong plasmonic resonance, or photonic band gap behavior? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider its dielectric properties, plasmonic response of metallic components, and structural periodicity potential.`,
  },
  {
    id: "quantum-topological",
    name: "Quantum Material/Topological Insulator",
    description: "Materials with unique electronic states useful for quantum computing, spintronics, and topological electronics",
    keywords: ["topological", "Dirac", "Weyl", "spin-orbit", "quantum spin Hall", "Majorana", "spintronics"],
    elementHints: ["Bi", "Sb", "Te", "Se", "Sn", "Pb", "Hg", "Mn", "Cr", "V"],
    formulaChecks: (f) => /(Bi|Sb).*(Te|Se)/.test(f) || /(Sn|Pb).*(Te|Se)/.test(f) || /(Hg|Cd)Te/.test(f) || /Mn.*(Bi|Sb)/.test(f),
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      if (c.z2Score != null && c.z2Score > 0.5) {
        score += 0.3;
        reasons.push(`Z2 invariant score ${c.z2Score.toFixed(2)} indicates topological character`);
      }
      if (c.chernScore != null && c.chernScore > 0) {
        score += 0.25;
        reasons.push(`Non-zero Chern number (${c.chernScore.toFixed(2)}) indicates topological band structure`);
      }
      if (c.socStrength != null && c.socStrength > 0.3) {
        score += 0.15;
        reasons.push(`Strong spin-orbit coupling (${c.socStrength.toFixed(2)}) — essential for topological states`);
      }
      if (c.topologicalScore != null && c.topologicalScore > 0.5) {
        score += 0.2;
        reasons.push(`Topological score ${c.topologicalScore.toFixed(2)} from engine analysis`);
      }
      const fst = (c.fermiSurfaceTopology || "").toLowerCase();
      if (fst.includes("dirac") || fst.includes("weyl") || fst.includes("topological")) {
        score += 0.2;
        reasons.push(`Fermi surface topology "${c.fermiSurfaceTopology}" indicates exotic electronic states`);
      }
      const dim = (c.dimensionality || "").toLowerCase();
      if (dim.includes("2d") || dim.includes("quasi-2d")) {
        score += 0.1;
        reasons.push(`${c.dimensionality} dimensionality favorable for 2D topological states`);
      }
      if (c.bandGap != null && c.bandGap > 0 && c.bandGap < 0.5) {
        score += 0.1;
        reasons.push(`Narrow band gap ${c.bandGap.toFixed(2)} eV consistent with topological insulator`);
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} be a topological insulator or quantum material? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider whether its heavy elements could produce strong spin-orbit coupling, whether its band structure could support topological surface states or Dirac cone dispersion, and whether it belongs to a known topological material family (Bi2Se3-type, SnTe-type, Heusler).`,
  },
  {
    id: "semiconductor",
    name: "Traditional Semiconductor",
    description: "Materials for conventional and next-generation semiconductor applications including transistors, LEDs, and power electronics",
    keywords: ["semiconductor", "transistor", "LED", "photoconductor", "band gap", "doping", "p-n junction"],
    elementHints: ["Si", "Ge", "Ga", "As", "In", "P", "N", "Al", "Zn", "Cd", "Se", "Te", "Sn"],
    formulaChecks: (f) => /(Ga|In|Al).*(As|N|P|Sb)/.test(f) || /(Zn|Cd).*(O|S|Se|Te)/.test(f) || /Si.*Ge/.test(f) || /Si.*C/.test(f),
    datapointChecks: (c) => {
      let score = 0;
      const reasons: string[] = [];
      if (c.bandGap != null) {
        if (c.bandGap >= 0.5 && c.bandGap <= 3.5) {
          score += 0.3;
          reasons.push(`Band gap ${c.bandGap.toFixed(2)} eV in semiconductor range (0.5-3.5 eV)`);
        }
        if (c.bandGap > 3.0) {
          const hasConductive = ["Sn", "In", "Zn", "Cd", "Ga"].some(e => c.formula.includes(e));
          if (hasConductive) {
            score += 0.15;
            reasons.push(`Transparent conductor candidate: wide gap ${c.bandGap.toFixed(2)} eV + conductive elements`);
          }
        }
      }
      if (c.correlationStrength != null && c.correlationStrength < 0.3) {
        score += 0.1;
        reasons.push(`Low correlation strength (${c.correlationStrength.toFixed(2)}) — band theory applicable`);
      }
      if (c.stabilityScore != null && c.stabilityScore > 0.6) {
        score += 0.1;
        reasons.push(`Good stability (${c.stabilityScore.toFixed(2)}) for device integration`);
      }
      if (c.ambientPressureStable === true) {
        score += 0.05;
        reasons.push("Ambient pressure stable — practical for device fabrication");
      }
      return { score, reasons };
    },
    verificationPrompt: (formula, datapoints) =>
      `Could ${formula} function as a traditional or next-generation semiconductor material? ${datapoints.length > 0 ? `Measured properties: ${datapoints.join(". ")}.` : ""} Consider its potential band gap (0.5-3.5 eV range), crystal structure suitability for doping, carrier mobility prospects, and whether it could serve in transistors, LEDs, power electronics, or optoelectronic devices.`,
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
  family: string;
}

function scoreCandidate(candidate: CandidateData, signal: MaterialSignal): SignalMatch | null {
  const formula = candidate.formula || "";
  if (!formula) return null;

  let score = 0;
  const reasons: string[] = [];

  const elementMatches = signal.elementHints.filter(el => formula.includes(el));
  if (elementMatches.length >= 2) {
    score += 0.3;
    reasons.push(`Contains signal elements: ${elementMatches.join(", ")}`);
  } else if (elementMatches.length === 1) {
    score += 0.1;
    reasons.push(`Contains signal element: ${elementMatches[0]}`);
  }

  if (signal.formulaChecks(formula)) {
    score += 0.2;
    reasons.push("Matches formula/structural pattern");
  }

  const dp = signal.datapointChecks(candidate);
  if (dp.score > 0) {
    score += Math.min(0.5, dp.score);
    reasons.push(...dp.reasons);
  }

  const notes = ((candidate.notes || "") + " " + (candidate.predictedProperties?.description || "")).toLowerCase();
  const keywordMatches = signal.keywords.filter(kw => notes.includes(kw.toLowerCase()));
  if (keywordMatches.length > 0) {
    score += 0.15;
    reasons.push(`Keyword matches: ${keywordMatches.join(", ")}`);
  }

  if (score < 0.35) return null;

  return {
    signal,
    formula,
    candidateName: candidate.name || formula,
    matchScore: Math.min(1, score),
    matchReasons: reasons,
    family: classifyFamily(formula),
  };
}

async function verifyWithOpenAI(match: SignalMatch): Promise<{ valid: number; reasoning: string }> {
  try {
    const datapointReasons = match.matchReasons.filter(r =>
      !r.startsWith("Contains signal element") && !r.startsWith("Matches formula") && !r.startsWith("Keyword match")
    );
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a materials science expert verifier. You will be asked whether a specific chemical composition could realistically serve a particular materials application. Respond with ONLY valid JSON: {"valid": 1, "reasoning": "..."} if the material is a plausible candidate, or {"valid": 0, "reasoning": "..."} if it is not. The reasoning should be 1-3 sentences explaining your assessment. Be scientifically rigorous — do not confirm implausible claims.`,
        },
        {
          role: "user",
          content: match.signal.verificationPrompt(match.formula, datapointReasons),
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

    const allCandidates: CandidateData[] = [
      ...candidates.map(c => {
        const mlf = (c.mlFeatures as any) || {};
        return {
          formula: c.formula,
          name: c.formula,
          notes: c.notes || "",
          predictedProperties: { description: classifyFamily(c.formula) },
          source: "superconductor" as const,
          predictedTc: c.predictedTc,
          stabilityScore: c.stabilityScore,
          electronPhononCoupling: c.electronPhononCoupling,
          dimensionality: c.dimensionality,
          ambientPressureStable: c.ambientPressureStable,
          pressureGpa: c.pressureGpa,
          crystalStructure: c.crystalStructure,
          correlationStrength: c.correlationStrength,
          fermiSurfaceTopology: c.fermiSurfaceTopology,
          bandGap: getNum(mlf.bandGap),
          z2Score: getNum(mlf.z2Score),
          chernScore: getNum(mlf.chernScore),
          topologicalScore: getNum(mlf.topologicalScore),
          socStrength: getNum(mlf.socStrength),
          dosAtEF: getNum(mlf.dosAtEF),
          formationEnergy: getNum(mlf.formationEnergy),
          decompositionEnergy: c.decompositionEnergy,
        };
      }),
      ...novelPredictions.map(np => {
        const pp = (np.predictedProperties as any) || {};
        return {
          formula: np.formula,
          name: np.name,
          notes: np.notes || "",
          predictedProperties: pp,
          source: "novel" as const,
          predictedTc: null,
          stabilityScore: getNum(pp.stability),
          electronPhononCoupling: null,
          dimensionality: null,
          ambientPressureStable: null,
          pressureGpa: null,
          crystalStructure: null,
          correlationStrength: null,
          fermiSurfaceTopology: null,
          bandGap: getNum(pp.bandGap),
          z2Score: null,
          chernScore: null,
          topologicalScore: null,
          socStrength: null,
          dosAtEF: null,
          formationEnergy: getNum(pp.formationEnergy),
          decompositionEnergy: null,
        };
      }),
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
    const existingSet = new Set(existingMilestones.map(m => `${m.type}::${m.relatedFormula}`));

    const eligible: SignalMatch[] = [];
    for (const match of matches) {
      const dedupeKey = `${match.signal.id}::${match.formula}`;
      if (existingSet.has(`signal-${match.signal.id}::${match.formula}`)) continue;
      const cooldownUntil = rejectedCooldown.get(dedupeKey);
      if (cooldownUntil && cycleNumber < cooldownUntil) continue;
      eligible.push(match);
    }

    const familyBuckets = new Map<string, SignalMatch[]>();
    for (const match of eligible) {
      const bucket = familyBuckets.get(match.family) || [];
      bucket.push(match);
      familyBuckets.set(match.family, bucket);
    }

    const roundRobinOrder: SignalMatch[] = [];
    const familyKeys = [...familyBuckets.keys()].sort();
    const familyPointers = new Map<string, number>(familyKeys.map(k => [k, 0]));
    let placed = 0;
    while (placed < eligible.length && roundRobinOrder.length < MAX_VERIFICATIONS_PER_CYCLE) {
      let anyAdded = false;
      for (const fk of familyKeys) {
        if (roundRobinOrder.length >= MAX_VERIFICATIONS_PER_CYCLE) break;
        const bucket = familyBuckets.get(fk)!;
        const ptr = familyPointers.get(fk)!;
        if (ptr < bucket.length) {
          roundRobinOrder.push(bucket[ptr]);
          familyPointers.set(fk, ptr + 1);
          placed++;
          anyAdded = true;
        }
      }
      if (!anyAdded) break;
    }

    let verificationsThisCycle = 0;

    for (const match of roundRobinOrder) {

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
            description: `${match.formula} identified as a potential ${match.signal.name.toLowerCase()}. Datapoints: ${match.matchReasons.join("; ")}. ${verification.reasoning}`,
            significance: 2,
            relatedFormula: match.formula,
          });

          broadcast("milestone", {
            id: milestoneId,
            cycle: cycleNumber,
            type: `signal-${match.signal.id}`,
            title: `${match.signal.name}: ${match.formula}`,
            description: `${match.formula} identified as a potential ${match.signal.name.toLowerCase()}. Datapoints: ${match.matchReasons.join("; ")}. ${verification.reasoning}`,
            significance: 2,
            relatedFormula: match.formula,
          });
        } catch {}

        const novelId = `signal-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const signalMeta = {
          signalId: match.signal.id,
          signalName: match.signal.name,
          matchScore: Math.round(match.matchScore * 1000) / 1000,
          family: match.family,
          datapoints: match.matchReasons,
          verificationReasoning: verification.reasoning,
          verifiedAt: new Date().toISOString(),
          cycle: cycleNumber,
        };
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
                datapoints: match.matchReasons,
              },
              confidence: Math.min(0.85, match.matchScore),
              targetApplication: match.signal.name,
              status: "signal-verified",
              notes: `Signal scanner: ${match.matchReasons.join("; ")}. AI verification: ${verification.reasoning}`,
              signalMetadata: signalMeta,
            });

            emit("prediction", {
              id: novelId,
              name: `${match.candidateName} (${match.signal.name})`,
              formula: match.formula,
              confidence: Math.min(0.85, match.matchScore),
              targetApplication: match.signal.name,
            });
          } else {
            const existingSignals = Array.isArray((existing as any).signalMetadata) ? (existing as any).signalMetadata : (existing as any).signalMetadata ? [(existing as any).signalMetadata] : [];
            existingSignals.push(signalMeta);
            await storage.updateNovelPrediction(existing.id, {
              notes: `${existing.notes || ""} | Signal: ${match.signal.name} verified. ${verification.reasoning}`,
              signalMetadata: existingSignals,
            });
          }
        } catch {}

        storage.insertResearchLog({
          phase: "signal-scanner",
          event: `Material discovery signal: ${match.signal.name}`,
          detail: `${match.formula} verified as potential ${match.signal.name.toLowerCase()} (score: ${match.matchScore.toFixed(2)}). Datapoints: ${match.matchReasons.join("; ")}. ${verification.reasoning}`,
          dataSource: "Signal Scanner",
        }).catch(() => {});

      } else {
        rejectedCooldown.set(dedupeKey, cycleNumber + REJECTED_COOLDOWN_CYCLES);

        emit("log", {
          phase: "signal-scanner",
          event: `Signal NOT verified: ${match.formula} as ${match.signal.name}`,
          detail: `AI determined this is NOT a valid ${match.signal.name.toLowerCase()}. Reasoning: ${verification.reasoning}. Datapoints considered: ${match.matchReasons.join("; ")}. The signal detection heuristic may need refinement, or this composition lacks the required properties for this application category.`,
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
        detail: `Evaluated ${verificationsThisCycle} signal matches across ${scannedThisCycle.size} candidates in cycle ${cycleNumber}.`,
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
