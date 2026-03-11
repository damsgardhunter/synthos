import { storage } from "../storage";
import { ELEMENTAL_DATA, getMeltingPoint } from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";
import type { SynthesisRoute, ReactionStep, ThermodynamicScoring } from "./reaction-pathway";

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  return parseNestedFormula(cleaned);
}

function parseNestedFormula(s: string): Record<string, number> {
  const counts: Record<string, number> = {};
  let i = 0;
  while (i < s.length) {
    if (s[i] === '(') {
      let depth = 1;
      let j = i + 1;
      while (j < s.length && depth > 0) {
        if (s[j] === '(') depth++;
        else if (s[j] === ')') depth--;
        j++;
      }
      const inner = parseNestedFormula(s.substring(i + 1, j - 1));
      let numStr = '';
      while (j < s.length && (s[j] >= '0' && s[j] <= '9' || s[j] === '.')) {
        numStr += s[j]; j++;
      }
      const mult = numStr ? parseFloat(numStr) : 1;
      for (const [el, cnt] of Object.entries(inner)) {
        counts[el] = (counts[el] || 0) + cnt * mult;
      }
      i = j;
    } else if (s[i] >= 'A' && s[i] <= 'Z') {
      let el = s[i]; i++;
      while (i < s.length && s[i] >= 'a' && s[i] <= 'z') { el += s[i]; i++; }
      let numStr = '';
      while (i < s.length && (s[i] >= '0' && s[i] <= '9' || s[i] === '.')) { numStr += s[i]; i++; }
      const num = numStr ? parseFloat(numStr) : 1;
      counts[el] = (counts[el] || 0) + num;
    } else { i++; }
  }
  return counts;
}

function parseFormulaElements(formula: string): string[] {
  return Object.keys(parseFormulaCounts(formula));
}

interface SimilarityScore {
  formula: string;
  method: string;
  score: number;
  elementOverlap: number;
  familyMatch: boolean;
  sharedElements: string[];
  sourceId: string;
}

interface ReactionSimilarity {
  reactionId: string;
  equation: string;
  reactionType: string;
  score: number;
  relevantReactants: string[];
  relevantProducts: string[];
  conditions: any;
  energetics: any;
}

const FAMILY_SIMILARITY: Record<string, string[]> = {
  Cuprates: ["Oxides", "Pnictides", "Nickelates"],
  Pnictides: ["Cuprates", "Chalcogenides"],
  Chalcogenides: ["Pnictides", "Sulfides"],
  Hydrides: ["Intermetallics"],
  Intermetallics: ["Borides", "Alloys", "Hydrides"],
  Borides: ["Carbides", "Nitrides", "Intermetallics"],
  Carbides: ["Borides", "Nitrides"],
  Nitrides: ["Borides", "Carbides"],
  Oxides: ["Cuprates", "Perovskites"],
  Sulfides: ["Chalcogenides", "Selenides"],
  Kagome: ["Intermetallics", "Pnictides"],
  Alloys: ["Intermetallics"],
};

function computeElementJaccard(elsA: string[], elsB: string[]): number {
  const setA = new Set(elsA);
  const setB = new Set(elsB);
  const intersection = [...setA].filter(el => setB.has(el));
  const union = new Set([...setA, ...setB]);
  return union.size > 0 ? intersection.length / union.size : 0;
}

function computeChemicalSimilarity(
  targetElements: string[],
  targetFamily: string,
  sourceFormula: string,
  sourceMethod: string,
): SimilarityScore {
  const sourceElements = parseFormulaElements(sourceFormula);
  const sourceFamily = classifyFamily(sourceFormula);
  const elementOverlap = computeElementJaccard(targetElements, sourceElements);
  const sharedElements = targetElements.filter(el => sourceElements.includes(el));
  const familyMatch = targetFamily === sourceFamily;

  const familyRelated = FAMILY_SIMILARITY[targetFamily]?.includes(sourceFamily) ?? false;

  let pettiforDist = 0;
  let pettiforCount = 0;
  for (const elA of targetElements) {
    for (const elB of sourceElements) {
      const dA = ELEMENTAL_DATA[elA];
      const dB = ELEMENTAL_DATA[elB];
      if (dA && dB) {
        pettiforDist += Math.abs(dA.pettiforScale - dB.pettiforScale);
        pettiforCount++;
      }
    }
  }
  const avgPettiforDist = pettiforCount > 0 ? pettiforDist / pettiforCount : 50;
  const pettiforSimilarity = Math.max(0, 1 - avgPettiforDist / 100);

  let enDist = 0;
  let enCount = 0;
  for (const elA of targetElements) {
    for (const elB of sourceElements) {
      const dA = ELEMENTAL_DATA[elA];
      const dB = ELEMENTAL_DATA[elB];
      if (dA?.paulingElectronegativity != null && dB?.paulingElectronegativity != null) {
        enDist += Math.abs(dA.paulingElectronegativity - dB.paulingElectronegativity);
        enCount++;
      }
    }
  }
  const avgENDist = enCount > 0 ? enDist / enCount : 2.0;
  const enSimilarity = Math.max(0, 1 - avgENDist / 3.0);

  let score = elementOverlap * 0.40
    + (familyMatch ? 0.25 : familyRelated ? 0.12 : 0)
    + pettiforSimilarity * 0.20
    + enSimilarity * 0.15;

  score = Math.max(0, Math.min(1, score));

  return {
    formula: sourceFormula,
    method: sourceMethod,
    score,
    elementOverlap,
    familyMatch,
    sharedElements,
    sourceId: "",
  };
}

function scaleMeltingPointRatio(sourceElements: string[], targetElements: string[]): number {
  function avgMp(els: string[]): number {
    let sum = 0;
    let count = 0;
    for (const el of els) {
      const mp = getMeltingPoint(el);
      if (mp != null) { sum += mp; count++; }
    }
    return count > 0 ? sum / count : 1500;
  }
  const sourceMp = avgMp(sourceElements);
  const targetMp = avgMp(targetElements);
  return sourceMp > 0 ? targetMp / sourceMp : 1.0;
}

function adaptPrecursors(targetElements: string[], sourceMethod: string): string[] {
  const PRECURSOR_MAP: Record<string, Record<string, string>> = {
    "solid-state": {
      Y: "Y2O3", Ba: "BaCO3", Cu: "CuO", La: "La2O3", Sr: "SrCO3", Ca: "CaCO3",
      Fe: "Fe2O3", Nb: "Nb2O5", Ti: "TiO2", Mg: "MgO", B: "B2O3", V: "V2O5",
      Mo: "MoO3", W: "WO3", Cr: "Cr2O3", Mn: "MnO2", Co: "CoO", Ni: "NiO",
      Zn: "ZnO", Al: "Al2O3", Ga: "Ga2O3", In: "In2O3", Sn: "SnO2", Pb: "PbO",
      Bi: "Bi2O3", Zr: "ZrO2", Hf: "HfO2", Ta: "Ta2O5", Ce: "CeO2", Nd: "Nd2O3",
      P: "P2O5", Si: "SiO2", Ge: "GeO2", Ru: "RuO2", Re: "Re2O7",
    },
    "arc-melting": {},
    "high-pressure": {
      H: "NH3BH3", La: "La (elemental)", Y: "Y (elemental)", Ca: "CaH2",
      Sr: "SrH2", Ba: "BaH2", Th: "ThH4", Sc: "Sc (elemental)",
    },
    "CVD": {
      Ti: "TiCl4", Zr: "ZrCl4", Hf: "HfCl4", Nb: "NbCl5", Ta: "TaCl5",
      Cu: "Cu(hfac)2", La: "La(thd)3", Y: "Y(thd)3", Ba: "Ba(thd)2",
      Sr: "Sr(thd)2", Fe: "Fe(CO)5", Ni: "Ni(CO)4",
    },
    "ball-milling": {},
    "sputtering": {},
  };

  const methodMap = PRECURSOR_MAP[sourceMethod] || {};
  return targetElements.map(el => {
    if (methodMap[el]) return methodMap[el];
    if (sourceMethod === "arc-melting" || sourceMethod === "sputtering" || sourceMethod === "ball-milling") {
      return `${el} (elemental, 99.9%)`;
    }
    return `${el} precursor`;
  });
}

function adaptStep(
  step: ReactionStep,
  targetFormula: string,
  targetElements: string[],
  sourceElements: string[],
  mpRatio: number,
): ReactionStep {
  const adaptedTemp = step.temperature > 50
    ? Math.round(step.temperature * mpRatio)
    : step.temperature;

  const targetHasO = targetElements.includes("O");
  const sourceHasO = sourceElements.includes("O");
  let atmosphere = step.atmosphere;
  if (sourceHasO && !targetHasO && atmosphere.toLowerCase().includes("o2")) {
    atmosphere = "Ar";
  } else if (!sourceHasO && targetHasO && atmosphere === "Ar") {
    atmosphere = "flowing O2";
  }

  const adaptedProducts = step.products.map(p => {
    for (const srcEl of sourceElements) {
      if (p.includes(srcEl) && !targetElements.includes(srcEl)) {
        return targetFormula;
      }
    }
    if (p.includes("film") || p.includes("powder") || p.includes("pellet") ||
        p.includes("substrate") || p.includes("assembly") || p.includes("weighed") ||
        p.includes("cleaned") || p.includes("compressed") || p.includes("mixed") ||
        p.includes("calcined") || p.includes("reground") || p.includes("milled") ||
        p.includes("button") || p.includes("vapor")) {
      return p;
    }
    return targetFormula;
  });

  return {
    ...step,
    temperature: adaptedTemp,
    atmosphere,
    products: adaptedProducts,
    notes: step.notes
      ? `[Adapted] ${step.notes.replace(/\d+\s*C/g, `${adaptedTemp} C`)}`
      : `[Adapted from analogy] Step ${step.stepNumber}`,
  };
}

function computeAdaptedThermodynamics(
  formula: string,
  steps: ReactionStep[],
): ThermodynamicScoring {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) fractions[el] = (counts[el] || 0) / totalAtoms;

  let deltaH = 0;
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const dA = ELEMENTAL_DATA[elements[i]];
      const dB = ELEMENTAL_DATA[elements[j]];
      if (!dA || !dB) continue;
      const phiA = dA.miedemaPhiStar, phiB = dB.miedemaPhiStar;
      const nwsA = dA.miedemaNws13, nwsB = dB.miedemaNws13;
      const vA = dA.miedemaV23, vB = dB.miedemaV23;
      if (phiA == null || phiB == null || nwsA == null || nwsB == null || vA == null || vB == null) continue;
      const fAB = 2 * (fractions[elements[i]] || 0) * (fractions[elements[j]] || 0);
      const vAvg = (vA + vB) / 2;
      const nwsAvgInv = 2 / (1 / nwsA + 1 / nwsB);
      const interfaceEnergy = (-14.1 * (phiA - phiB) ** 2 + 9.4 * (nwsA - nwsB) ** 2) / nwsAvgInv;
      deltaH += fAB * vAvg * interfaceEnergy;
    }
  }
  const gibbsFreeEnergy = deltaH / Math.max(1, totalAtoms);

  const maxTemp = Math.max(...steps.map(s => s.temperature), 300);
  let maxMp = 0;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp != null && mp > maxMp) maxMp = mp;
  }
  const tammTemp = maxMp > 0 ? 0.57 * maxMp : 1200;
  const kineticBarrier = Math.max(0, Math.min(3.0, ((tammTemp - maxTemp) / tammTemp) * 2.0));

  const kB = 8.617e-5;
  const arrheniusRate = maxTemp > 0
    ? Math.exp(Math.max(-50, Math.min(50, -kineticBarrier / (kB * maxTemp))))
    : 0;

  const gibbsFactor = gibbsFreeEnergy < 0 ? 1.0 : Math.max(0, 1.0 - gibbsFreeEnergy * 0.5);
  const barrierFactor = Math.max(0, 1.0 - kineticBarrier * 0.3);
  const rateFactor = Math.min(1.0, arrheniusRate * 10);
  const overallFeasibility = Math.max(0, Math.min(1,
    gibbsFactor * 0.35 + barrierFactor * 0.25 + rateFactor * 0.2 + 0.5 * 0.2
  ));

  return {
    gibbsFreeEnergy: Number(gibbsFreeEnergy.toFixed(4)),
    kineticBarrier: Number(kineticBarrier.toFixed(4)),
    arrheniusRate: Number(arrheniusRate.toFixed(6)),
    metastableQuenchFeasibility: 0.5,
    overallFeasibility: Number(overallFeasibility.toFixed(4)),
  };
}

function transferRoute(
  sourceFormula: string,
  sourceMethod: string,
  sourceSteps: ReactionStep[],
  sourcePrecursors: string[],
  targetFormula: string,
  similarity: number,
): SynthesisRoute | null {
  const targetElements = parseFormulaElements(targetFormula);
  const sourceElements = parseFormulaElements(sourceFormula);

  if (targetElements.length === 0 || sourceElements.length === 0) return null;

  const mpRatio = scaleMeltingPointRatio(sourceElements, targetElements);
  const clampedRatio = Math.max(0.5, Math.min(2.0, mpRatio));

  const adaptedSteps = sourceSteps.map(step =>
    adaptStep(step, targetFormula, targetElements, sourceElements, clampedRatio)
  );

  const adaptedPrecursors = adaptPrecursors(targetElements, sourceMethod);

  const thermo = computeAdaptedThermodynamics(targetFormula, adaptedSteps);

  const maxTemp = Math.max(...adaptedSteps.map(s => s.temperature), 0);
  const maxPressure = Math.max(...adaptedSteps.map(s => s.pressure), 0);

  const adjustedFeasibility = thermo.overallFeasibility * Math.max(0.3, similarity);

  return {
    routeId: `at-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    routeName: `Analogy: ${sourceMethod} (from ${sourceFormula})`,
    method: sourceMethod,
    steps: adaptedSteps,
    precursors: adaptedPrecursors,
    thermodynamics: thermo,
    totalDuration: adaptedSteps.length > 3 ? "~48 hours (estimated)" : "~24 hours (estimated)",
    maxTemperature: maxTemp,
    maxPressure: maxPressure,
    difficulty: maxTemp > 1400 || maxPressure > 100 ? "hard" : maxTemp > 800 ? "moderate" : "easy",
    equipment: inferEquipment(sourceMethod),
    feasibilityScore: Number(adjustedFeasibility.toFixed(4)),
    notes: `Transferred from ${sourceFormula} synthesis (similarity: ${(similarity * 100).toFixed(0)}%). Temperatures scaled by ${clampedRatio.toFixed(2)}x melting point ratio.`,
  };
}

function inferEquipment(method: string): string[] {
  const EQUIPMENT: Record<string, string[]> = {
    "solid-state": ["Box furnace", "Planetary ball mill", "Uniaxial press", "Agate mortar"],
    "arc-melting": ["Arc furnace", "Water-cooled Cu hearth", "Ar glovebox", "Quartz tube sealer"],
    "high-pressure": ["Diamond anvil cell", "YAG laser heating", "Ruby fluorescence gauge", "Synchrotron XRD"],
    "ball-milling": ["Planetary ball mill", "WC balls and jar", "Cold isostatic press", "Tube furnace"],
    "CVD": ["MOCVD reactor", "Mass flow controllers", "Substrate heater", "Vacuum pump"],
    "sputtering": ["Magnetron sputtering system", "RF/DC power supplies", "Substrate heater", "Turbo pump"],
    "sol-gel": ["Magnetic stirrer", "Drying oven", "Muffle furnace", "Centrifuge"],
  };
  return EQUIPMENT[method] || ["General laboratory equipment"];
}

export interface AnalogyTransferResult {
  routes: SynthesisRoute[];
  analogues: {
    sourceFormula: string;
    sourceMethod: string;
    similarity: number;
    sharedElements: string[];
  }[];
  reactionsApplied: {
    reactionName: string;
    reactionType: string;
    adaptedConditions: any;
  }[];
}

export async function findAnalogousSyntheses(
  formula: string,
  family?: string,
): Promise<SimilarityScore[]> {
  const targetElements = parseFormulaElements(formula);
  const targetFamily = family || classifyFamily(formula);
  if (targetElements.length === 0) return [];

  const allProcesses = await storage.getSynthesisProcesses(200);

  const scored: SimilarityScore[] = [];
  for (const proc of allProcesses) {
    if (!proc.formula || proc.formula === formula) continue;
    const sim = computeChemicalSimilarity(
      targetElements,
      targetFamily,
      proc.formula,
      proc.method || "unknown",
    );
    sim.sourceId = proc.id;
    if (sim.score >= 0.15) {
      scored.push(sim);
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 10);
}

export async function findRelevantReactions(
  formula: string,
): Promise<ReactionSimilarity[]> {
  const targetElements = parseFormulaElements(formula);
  if (targetElements.length === 0) return [];

  const allReactions = await storage.getChemicalReactions(200);

  const relevant: ReactionSimilarity[] = [];
  for (const rxn of allReactions) {
    if (!rxn.equation) continue;

    const reactants = Array.isArray(rxn.reactants) ? rxn.reactants : [];
    const products = Array.isArray(rxn.products) ? rxn.products : [];

    const allFormulas = [
      ...reactants.map((r: any) => r.formula || ""),
      ...products.map((p: any) => p.formula || ""),
      rxn.equation,
    ].join(" ");

    const matchedElements = targetElements.filter(el =>
      new RegExp(`\\b${el}\\b|${el}[0-9(]|[)0-9]${el}`).test(allFormulas)
    );

    if (matchedElements.length === 0) continue;

    const elementCoverage = matchedElements.length / targetElements.length;
    const typeBonus = ["solid-state", "hydrogenation", "reduction", "oxidation", "high-pressure"]
      .includes(rxn.reactionType || "") ? 0.1 : 0;
    const relevanceBonus = (rxn.relevanceToSuperconductor ?? 0) * 0.2;

    const score = elementCoverage * 0.6 + typeBonus + relevanceBonus;

    if (score >= 0.15) {
      relevant.push({
        reactionId: rxn.id,
        equation: rxn.equation,
        reactionType: rxn.reactionType || "unknown",
        score,
        relevantReactants: reactants
          .filter((r: any) => matchedElements.some(el => (r.formula || "").includes(el)))
          .map((r: any) => r.formula),
        relevantProducts: products
          .filter((p: any) => matchedElements.some(el => (p.formula || "").includes(el)))
          .map((p: any) => p.formula),
        conditions: rxn.conditions,
        energetics: rxn.energetics,
      });
    }
  }

  relevant.sort((a, b) => b.score - a.score);
  return relevant.slice(0, 8);
}

function buildRouteFromReaction(
  targetFormula: string,
  reaction: ReactionSimilarity,
): SynthesisRoute | null {
  const targetElements = parseFormulaElements(targetFormula);
  const conditions = reaction.conditions || {};

  let temperature = 800;
  if (conditions.temperature) {
    const tempStr = String(conditions.temperature);
    const tempMatch = tempStr.match(/(\d+)/);
    if (tempMatch) temperature = parseInt(tempMatch[1]);
  }

  let pressure = 0;
  if (conditions.pressure) {
    const pressStr = String(conditions.pressure);
    const pressMatch = pressStr.match(/([\d.]+)/);
    if (pressMatch) {
      const val = parseFloat(pressMatch[1]);
      if (pressStr.toLowerCase().includes("gpa")) pressure = val;
      else if (pressStr.toLowerCase().includes("atm")) pressure = val * 0.000101325;
      else pressure = val;
    }
  }

  const atmosphere = conditions.atmosphere || "Ar";
  const duration = conditions.duration || "12 hours";

  const precursors = adaptPrecursors(targetElements, reaction.reactionType);

  const steps: ReactionStep[] = [
    {
      stepNumber: 1,
      reactants: precursors,
      products: ["mixed precursors"],
      temperature: 25,
      pressure: 0,
      atmosphere: "air",
      reactionType: "preparation",
      duration: "2 hours",
      notes: `[From reaction: ${reaction.equation.slice(0, 60)}] Prepare precursors`,
    },
    {
      stepNumber: 2,
      reactants: ["mixed precursors"],
      products: [targetFormula],
      temperature,
      pressure,
      atmosphere,
      reactionType: reaction.reactionType,
      duration,
      notes: `[Analogy from ${reaction.reactionType} reaction] React at ${temperature} C`,
    },
  ];

  if (temperature > 500) {
    steps.push({
      stepNumber: 3,
      reactants: [targetFormula],
      products: [`${targetFormula} (annealed)`],
      temperature: Math.round(temperature * 0.6),
      pressure: 0,
      atmosphere,
      reactionType: "annealing",
      duration: "6 hours",
      notes: `Post-reaction anneal at ${Math.round(temperature * 0.6)} C`,
    });
  }

  const thermo = computeAdaptedThermodynamics(targetFormula, steps);

  return {
    routeId: `rxn-at-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    routeName: `Reaction Analogy: ${reaction.reactionType} (${reaction.equation.slice(0, 40)})`,
    method: reaction.reactionType,
    steps,
    precursors,
    thermodynamics: thermo,
    totalDuration: "~24 hours (estimated)",
    maxTemperature: temperature,
    maxPressure: pressure,
    difficulty: temperature > 1400 || pressure > 50 ? "hard" : "moderate",
    equipment: inferEquipment(reaction.reactionType),
    feasibilityScore: Number((thermo.overallFeasibility * reaction.score).toFixed(4)),
    notes: `Adapted from known reaction: ${reaction.equation}. Original conditions: ${JSON.stringify(conditions).slice(0, 120)}`,
  };
}

export async function proposeAnalogousRoutes(
  formula: string,
): Promise<AnalogyTransferResult> {
  const result: AnalogyTransferResult = {
    routes: [],
    analogues: [],
    reactionsApplied: [],
  };

  const analogues = await findAnalogousSyntheses(formula);

  const processedFormulas = new Set<string>();
  for (const analogue of analogues.slice(0, 5)) {
    if (analogue.score < 0.2) continue;
    if (processedFormulas.has(analogue.formula)) continue;
    processedFormulas.add(analogue.formula);

    const processes = await storage.getSynthesisProcessesByFormula(analogue.formula);
    if (processes.length === 0) continue;

    const seenMethods = new Set<string>();
    for (const proc of processes.slice(0, 2)) {
      if (seenMethods.has(proc.method || "")) continue;
      seenMethods.add(proc.method || "");
      const steps: ReactionStep[] = [];
      const rawSteps = proc.steps || [];

      for (let si = 0; si < rawSteps.length; si++) {
        const stepText = rawSteps[si];
        steps.push({
          stepNumber: si + 1,
          reactants: si === 0 ? (proc.precursors || []) : [`intermediate-${si}`],
          products: si === rawSteps.length - 1 ? [proc.formula || "product"] : [`intermediate-${si + 1}`],
          temperature: extractTemperature(stepText, proc.conditions),
          pressure: extractPressure(stepText, proc.conditions),
          atmosphere: extractAtmosphere(stepText),
          reactionType: inferReactionType(stepText),
          duration: extractDuration(stepText),
          notes: stepText,
        });
      }

      if (steps.length === 0) continue;

      const route = transferRoute(
        analogue.formula,
        proc.method || "solid-state",
        steps,
        proc.precursors || [],
        formula,
        analogue.score,
      );

      if (route && route.feasibilityScore > 0.1) {
        result.routes.push(route);
        result.analogues.push({
          sourceFormula: analogue.formula,
          sourceMethod: proc.method || "unknown",
          similarity: analogue.score,
          sharedElements: analogue.sharedElements,
        });
      }
    }
  }

  const relevantReactions = await findRelevantReactions(formula);
  for (const rxn of relevantReactions.slice(0, 3)) {
    if (rxn.score < 0.25) continue;

    const route = buildRouteFromReaction(formula, rxn);
    if (route && route.feasibilityScore > 0.1) {
      result.routes.push(route);
      result.reactionsApplied.push({
        reactionName: rxn.equation.slice(0, 60),
        reactionType: rxn.reactionType,
        adaptedConditions: rxn.conditions,
      });
    }
  }

  result.routes.sort((a, b) => b.feasibilityScore - a.feasibilityScore);

  return result;
}

function extractTemperature(stepText: string, conditions: any): number {
  const match = stepText.match(/(\d{2,4})\s*[°]?\s*C/i);
  if (match) return parseInt(match[1]);
  const matchK = stepText.match(/(\d{3,4})\s*K/i);
  if (matchK) return Math.round(parseInt(matchK[1]) - 273);
  if (conditions?.temperature != null) {
    const t = typeof conditions.temperature === "number" ? conditions.temperature : parseFloat(conditions.temperature);
    if (!isNaN(t)) return t;
  }
  return 25;
}

function extractPressure(stepText: string, conditions: any): number {
  const matchGpa = stepText.match(/([\d.]+)\s*GPa/i);
  if (matchGpa) return parseFloat(matchGpa[1]);
  const matchMpa = stepText.match(/([\d.]+)\s*MPa/i);
  if (matchMpa) return parseFloat(matchMpa[1]) * 0.001;
  const matchAtm = stepText.match(/([\d.]+)\s*atm/i);
  if (matchAtm) return parseFloat(matchAtm[1]) * 0.000101325;
  if (conditions?.pressure != null) {
    const rawP = typeof conditions.pressure === "number" ? conditions.pressure : parseFloat(String(conditions.pressure));
    if (!isNaN(rawP)) {
      const unitStr = String(conditions.pressure).toLowerCase();
      if (unitStr.includes("gpa")) return rawP;
      if (unitStr.includes("mpa")) return rawP * 0.001;
      if (unitStr.includes("bar") || unitStr.includes("kbar")) {
        return unitStr.includes("kbar") ? rawP * 0.1 : rawP * 0.0001;
      }
      if (rawP > 1000) return rawP * 0.000101325;
      if (rawP > 50) return rawP * 0.001;
      return rawP * 0.000101325;
    }
  }
  return 0;
}

function extractAtmosphere(stepText: string): string {
  const lower = stepText.toLowerCase();
  if (lower.includes("oxygen") || lower.includes("o2 flow") || lower.includes("flowing o2")) return "flowing O2";
  if (lower.includes("argon") || lower.includes("ar ")) return "Ar";
  if (lower.includes("nitrogen") || lower.includes("n2")) return "N2";
  if (lower.includes("hydrogen") || lower.includes("h2")) return "H2/Ar";
  if (lower.includes("vacuum")) return "vacuum";
  if (lower.includes("air")) return "air";
  return "Ar";
}

function inferReactionType(stepText: string): string {
  const lower = stepText.toLowerCase();
  if (lower.includes("sinter")) return "sintering";
  if (lower.includes("calcin")) return "calcination";
  if (lower.includes("anneal")) return "annealing";
  if (lower.includes("grind") || lower.includes("mill") || lower.includes("mix")) return "mixing";
  if (lower.includes("press")) return "pressing";
  if (lower.includes("melt") || lower.includes("arc")) return "arc-melting";
  if (lower.includes("laser") || lower.includes("heat")) return "heating";
  if (lower.includes("compress")) return "compression";
  if (lower.includes("quench") || lower.includes("cool")) return "cooling";
  if (lower.includes("weigh") || lower.includes("load") || lower.includes("prepar")) return "preparation";
  return "processing";
}

function extractDuration(stepText: string): string {
  const match = stepText.match(/(\d+)\s*(hours?|h\b|minutes?|min|days?|d\b)/i);
  if (match) return `${match[1]} ${match[2]}`;
  return "variable";
}
