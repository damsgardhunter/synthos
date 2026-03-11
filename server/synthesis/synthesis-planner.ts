import { matchTemplates, type TemplateMatch, type ReactionTemplate } from "./reaction-templates";
import { findBestPrecursors, computePrecursorAvailabilityScore, type PrecursorSelection, type PrecursorAvailabilityResult } from "./precursor-database";
import { computeReactionFeasibility, computeSynthesisTemperature, assessPressureRequirement, type ReactionFeasibilityResult } from "./thermodynamic-feasibility";
import { classifyFamily } from "../learning/utils";
import { getMeltingPoint } from "../learning/elemental-data";

export interface SynthesisStep {
  stepNumber: number;
  reactants: string[];
  products: string[];
  temperature: number;
  pressure: number;
  atmosphere: string;
  reactionType: string;
  duration: string;
  notes: string;
}

export interface SynthesisRoute {
  routeId: string;
  formula: string;
  method: string;
  routeName: string;
  steps: SynthesisStep[];
  precursors: string[];
  precursorDetails: PrecursorSelection[];
  feasibilityScore: number;
  templateMatchScore: number;
  precursorAvailabilityScore: number;
  thermodynamicFeasibilityScore: number;
  complexityPenalty: number;
  totalCostEstimate: string;
  equipmentList: string[];
  safetyNotes: string[];
  estimatedYield: string;
  maxTemperature: number;
  maxPressure: number;
  totalDuration: string;
  difficulty: string;
  reasoning: string[];
}

export interface SynthesisPlanResult {
  formula: string;
  family: string;
  routes: SynthesisRoute[];
  bestRoute: SynthesisRoute | null;
  summary: string;
  thermodynamics: ReactionFeasibilityResult;
  precursorAvailability: PrecursorAvailabilityResult;
}

export interface SynthesisPlanOptions {
  formationEnergy?: number | null;
  maxRoutes?: number;
  preferredMethods?: string[];
  includeMultiStep?: boolean;
}

interface MaterialNode {
  formula: string;
  elements: string[];
  isIntermediate: boolean;
}

interface ReactionEdge {
  from: string[];
  to: string;
  method: string;
  temperature: number;
  pressure: number;
}

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

const KNOWN_BINARY_INTERMEDIATES: Record<string, string[]> = {
  "Ba-Cu": ["BaCu"],
  "Y-Cu": ["YCu"],
  "Y-Ba": ["YBa"],
  "La-Cu": ["LaCu"],
  "Fe-As": ["FeAs", "Fe2As"],
  "Fe-Se": ["FeSe"],
  "Ba-Fe": ["BaFe2"],
  "Sr-Fe": ["SrFe2"],
  "Nb-Sn": ["Nb3Sn"],
  "Nb-Ti": ["NbTi"],
  "Nb-Ge": ["Nb3Ge"],
  "V-Si": ["V3Si"],
  "Mg-B": ["MgB2"],
  "La-H": ["LaH2", "LaH3"],
  "Y-H": ["YH2", "YH3"],
  "Ca-H": ["CaH2"],
  "Ba-H": ["BaH2"],
  "Ce-Co": ["CeCo"],
  "Ce-In": ["CeIn3"],
  "Bi-Sr": ["BiSr"],
  "Bi-Cu": ["BiCu"],
  "Ti-O": ["TiO2"],
  "Sr-Ti": ["SrTiO3"],
  "Ba-Ti": ["BaTiO3"],
};

function getIntermediateKey(el1: string, el2: string): string {
  return [el1, el2].sort().join("-");
}

function buildMaterialGraph(formula: string): { nodes: MaterialNode[]; edges: ReactionEdge[] } {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const nodes: MaterialNode[] = [];
  const edges: ReactionEdge[] = [];

  nodes.push({ formula, elements, isIntermediate: false });

  for (const el of elements) {
    nodes.push({ formula: el, elements: [el], isIntermediate: false });
  }

  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const key = getIntermediateKey(elements[i], elements[j]);
      const intermediates = KNOWN_BINARY_INTERMEDIATES[key];
      if (intermediates) {
        for (const inter of intermediates) {
          nodes.push({ formula: inter, elements: [elements[i], elements[j]], isIntermediate: true });
          edges.push({
            from: [elements[i], elements[j]],
            to: inter,
            method: "solid-state",
            temperature: 800,
            pressure: 0,
          });
        }
      }
    }
  }

  edges.push({
    from: elements,
    to: formula,
    method: "direct",
    temperature: 1000,
    pressure: 0,
  });

  return { nodes, edges };
}

function decomposeIntoIntermediateRoutes(formula: string, elements: string[]): string[][] {
  const routes: string[][] = [];

  routes.push([elements.join(" + ") + " → " + formula]);

  if (elements.length >= 3) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const key = getIntermediateKey(elements[i], elements[j]);
        const intermediates = KNOWN_BINARY_INTERMEDIATES[key];
        if (intermediates && intermediates.length > 0) {
          const inter = intermediates[0];
          const remaining = elements.filter((_, idx) => idx !== i && idx !== j);
          const step1 = `${elements[i]} + ${elements[j]} → ${inter}`;
          const step2 = `${inter} + ${remaining.join(" + ")} → ${formula}`;
          routes.push([step1, step2]);
        }
      }
    }
  }

  if (elements.length >= 4) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const key1 = getIntermediateKey(elements[i], elements[j]);
        const inter1List = KNOWN_BINARY_INTERMEDIATES[key1];
        if (!inter1List || inter1List.length === 0) continue;

        const remaining = elements.filter((_, idx) => idx !== i && idx !== j);
        for (let k = 0; k < remaining.length; k++) {
          for (let l = k + 1; l < remaining.length; l++) {
            const key2 = getIntermediateKey(remaining[k], remaining[l]);
            const inter2List = KNOWN_BINARY_INTERMEDIATES[key2];
            if (!inter2List || inter2List.length === 0) continue;

            const inter1 = inter1List[0];
            const inter2 = inter2List[0];
            const leftover = remaining.filter((_, idx) => idx !== k && idx !== l);
            const step1 = `${elements[i]} + ${elements[j]} → ${inter1}`;
            const step2 = `${remaining[k]} + ${remaining[l]} → ${inter2}`;
            const step3parts = [inter1, inter2, ...leftover].join(" + ");
            const step3 = `${step3parts} → ${formula}`;
            routes.push([step1, step2, step3]);
          }
        }
      }
    }
  }

  return routes;
}

function buildStepsFromTemplate(
  formula: string,
  elements: string[],
  template: ReactionTemplate,
  precursorSelections: PrecursorSelection[],
  thermoResult: ReactionFeasibilityResult,
  intermediateRoute?: string[]
): SynthesisStep[] {
  const steps: SynthesisStep[] = [];
  const precursorFormulas = precursorSelections.map(s => s.precursor.formula);
  const synthTemp = thermoResult.synthesisTemperature;
  const pressure = thermoResult.pressureRequirement;

  if (intermediateRoute && intermediateRoute.length > 1) {
    let stepNum = 1;

    for (let i = 0; i < intermediateRoute.length; i++) {
      const routeStep = intermediateRoute[i];
      const [reactantPart, productPart] = routeStep.split(" → ");
      const reactants = reactantPart.split(" + ").map(r => r.trim());
      const product = productPart.trim();
      const isLast = i === intermediateRoute.length - 1;

      const stepTemp = isLast ? synthTemp : Math.round(synthTemp * 0.7);

      if (i === 0) {
        steps.push({
          stepNumber: stepNum++,
          reactants: precursorFormulas,
          products: ["mixed precursors"],
          temperature: 25,
          pressure: 0,
          atmosphere: template.atmosphere[0] || "air",
          reactionType: "preparation",
          duration: "2 hours",
          notes: `Weigh and mix precursors for ${reactants.join(", ")}`,
        });
      }

      steps.push({
        stepNumber: stepNum++,
        reactants: i === 0 ? precursorFormulas : [intermediateRoute[i - 1].split(" → ")[1].trim()],
        products: [product],
        temperature: stepTemp,
        pressure: template.method === "high-pressure" ? pressure : 0,
        atmosphere: template.atmosphere[0] || "Ar",
        reactionType: template.method,
        duration: `${Math.round(template.durationRange[0] + (template.durationRange[1] - template.durationRange[0]) * 0.3)} hours`,
        notes: `${template.name}: ${routeStep}`,
      });

      if (!isLast) {
        steps.push({
          stepNumber: stepNum++,
          reactants: [product],
          products: [`${product} (ground)`],
          temperature: 25,
          pressure: 0,
          atmosphere: "air",
          reactionType: "grinding",
          duration: "1 hour",
          notes: `Grind intermediate ${product} and prepare for next step`,
        });
      }
    }
  } else {
    let stepNum = 1;

    steps.push({
      stepNumber: stepNum++,
      reactants: precursorFormulas,
      products: ["mixed precursors"],
      temperature: 25,
      pressure: 0,
      atmosphere: template.atmosphere[0] || "air",
      reactionType: "preparation",
      duration: "2 hours",
      notes: `Weigh stoichiometric amounts and prepare for ${template.method}`,
    });

    if (template.method === "solid-state" || template.method === "ball-milling") {
      steps.push({
        stepNumber: stepNum++,
        reactants: ["mixed precursors"],
        products: ["ground mixture"],
        temperature: 25,
        pressure: template.method === "ball-milling" ? 0 : 0,
        atmosphere: "Ar",
        reactionType: template.method === "ball-milling" ? "ball-milling" : "grinding",
        duration: template.method === "ball-milling" ? "20 hours" : "4 hours",
        notes: template.method === "ball-milling"
          ? "High-energy ball mill at 400 rpm, WC balls, ball:powder 10:1"
          : "Ball mill or mortar grinding for homogenization",
      });

      const calcineTemp = Math.round(synthTemp * 0.7);
      steps.push({
        stepNumber: stepNum++,
        reactants: ["ground mixture"],
        products: ["calcined powder"],
        temperature: calcineTemp,
        pressure: 0,
        atmosphere: elements.includes("O") ? "flowing O2" : "Ar",
        reactionType: "calcination",
        duration: "12 hours",
        notes: `Calcine at ${calcineTemp} K, ramp 5 K/min`,
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["calcined powder"],
        products: ["pellet"],
        temperature: 25,
        pressure: 200,
        atmosphere: "air",
        reactionType: "pressing",
        duration: "30 minutes",
        notes: "Uniaxial press at 200 MPa",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["pellet"],
        products: [formula],
        temperature: synthTemp,
        pressure: 0,
        atmosphere: elements.includes("O") ? "flowing O2" : "Ar/5% H2",
        reactionType: "sintering",
        duration: `${Math.round(template.durationRange[0] + (template.durationRange[1] - template.durationRange[0]) * 0.4)} hours`,
        notes: `Sinter at ${synthTemp} K, ramp 3 K/min, furnace cool`,
      });
    } else if (template.method === "arc-melting") {
      steps.push({
        stepNumber: stepNum++,
        reactants: precursorFormulas,
        products: ["arc-melted button"],
        temperature: 3000,
        pressure: 0,
        atmosphere: "ultra-high purity Ar",
        reactionType: "arc-melting",
        duration: "5 minutes per melt, 4 flips",
        notes: "Arc-melt on water-cooled Cu hearth, flip and remelt 4x",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["arc-melted button"],
        products: [formula],
        temperature: Math.round(synthTemp * 0.6),
        pressure: 0,
        atmosphere: "sealed quartz tube under Ar",
        reactionType: "annealing",
        duration: "7 days",
        notes: `Wrap in Ta foil, seal in evacuated quartz tube, anneal at ${Math.round(synthTemp * 0.6)} K`,
      });
    } else if (template.method === "high-pressure") {
      steps.push({
        stepNumber: stepNum++,
        reactants: precursorFormulas,
        products: ["sample assembly"],
        temperature: 25,
        pressure: 0,
        atmosphere: elements.includes("H") ? "Ar glovebox" : "air",
        reactionType: "sample loading",
        duration: "2 hours",
        notes: pressure > 50
          ? "Load into DAC gasket with pressure medium"
          : "Load into multi-anvil capsule",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["sample assembly"],
        products: ["compressed sample"],
        temperature: 25,
        pressure,
        atmosphere: "compressed Ne or He",
        reactionType: "compression",
        duration: "4 hours",
        notes: `Compress to ${pressure} GPa`,
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["compressed sample"],
        products: [formula],
        temperature: synthTemp,
        pressure,
        atmosphere: "under pressure",
        reactionType: "laser-heating",
        duration: "30 minutes",
        notes: `Laser-heat to ${synthTemp} K at ${pressure} GPa`,
      });

      if (pressure > 50) {
        steps.push({
          stepNumber: stepNum++,
          reactants: [formula],
          products: [`${formula} (quenched)`],
          temperature: 25,
          pressure: 0,
          atmosphere: "ambient",
          reactionType: "pressure-quench",
          duration: "2 hours",
          notes: `Decompress from ${pressure} GPa; assess metastable retention`,
        });
      }
    } else if (template.method === "sputtering" || template.method === "PLD" || template.method === "MBE") {
      steps.push({
        stepNumber: stepNum++,
        reactants: ["substrate"],
        products: ["cleaned substrate"],
        temperature: 25,
        pressure: 0,
        atmosphere: "clean room",
        reactionType: "substrate preparation",
        duration: "1 hour",
        notes: "Ultrasonic clean, O2 plasma clean",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: precursorFormulas,
        products: [`${formula} film`],
        temperature: Math.round((template.temperatureRange[0] + template.temperatureRange[1]) / 2),
        pressure: 0.005,
        atmosphere: template.atmosphere[0] || "Ar",
        reactionType: template.method,
        duration: "3 hours",
        notes: `Deposit ${formula} thin film via ${template.method}`,
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: [`${formula} film`],
        products: [formula],
        temperature: Math.round(synthTemp * 0.5),
        pressure: 0,
        atmosphere: elements.includes("O") ? "O2" : "Ar",
        reactionType: "post-annealing",
        duration: "2 hours",
        notes: "Post-deposition anneal to optimize stoichiometry",
      });
    } else if (template.method === "CVD") {
      steps.push({
        stepNumber: stepNum++,
        reactants: ["substrate"],
        products: ["cleaned substrate"],
        temperature: 25,
        pressure: 0,
        atmosphere: "clean room",
        reactionType: "substrate preparation",
        duration: "1 hour",
        notes: "Ultrasonic clean in acetone, IPA, DI water",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: precursorFormulas,
        products: [`${formula} film`],
        temperature: 700,
        pressure: 0.01,
        atmosphere: "carrier gas (Ar/O2)",
        reactionType: "CVD",
        duration: "2 hours",
        notes: "Chemical vapor deposition at 10 mTorr",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: [`${formula} film`],
        products: [formula],
        temperature: 500,
        pressure: 0.0001,
        atmosphere: "flowing O2",
        reactionType: "post-annealing",
        duration: "1 hour",
        notes: "Post-anneal at ambient pressure in O2",
      });
    } else if (template.method === "sol-gel") {
      steps.push({
        stepNumber: stepNum++,
        reactants: precursorFormulas,
        products: ["precursor solution"],
        temperature: 80,
        pressure: 0,
        atmosphere: "air",
        reactionType: "dissolution",
        duration: "4 hours",
        notes: "Dissolve precursors in citric acid/nitric acid solution, stir at 80 C",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["precursor solution"],
        products: ["gel"],
        temperature: 200,
        pressure: 0,
        atmosphere: "air",
        reactionType: "gelation",
        duration: "12 hours",
        notes: "Heat to form gel, dry at 200 C",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["gel"],
        products: ["calcined powder"],
        temperature: Math.round(synthTemp * 0.6),
        pressure: 0,
        atmosphere: "air",
        reactionType: "calcination",
        duration: "6 hours",
        notes: `Calcine gel at ${Math.round(synthTemp * 0.6)} K`,
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["calcined powder"],
        products: [formula],
        temperature: synthTemp,
        pressure: 0,
        atmosphere: elements.includes("O") ? "flowing O2" : "Ar",
        reactionType: "sintering",
        duration: "12 hours",
        notes: `Final sinter at ${synthTemp} K`,
      });
    } else if (template.method === "flux-growth") {
      steps.push({
        stepNumber: stepNum++,
        reactants: [...precursorFormulas, "flux (Sn, In, or Pb)"],
        products: ["flux mixture"],
        temperature: 25,
        pressure: 0,
        atmosphere: "Ar glovebox",
        reactionType: "preparation",
        duration: "2 hours",
        notes: "Mix precursors with self-flux in alumina crucible",
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["flux mixture"],
        products: ["melt"],
        temperature: Math.round(synthTemp * 1.2),
        pressure: 0,
        atmosphere: "sealed quartz tube",
        reactionType: "melting",
        duration: "6 hours",
        notes: `Heat to ${Math.round(synthTemp * 1.2)} K, hold 6h for homogenization`,
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: ["melt"],
        products: [`${formula} crystals + flux`],
        temperature: synthTemp,
        pressure: 0,
        atmosphere: "sealed quartz tube",
        reactionType: "slow cooling",
        duration: "7-14 days",
        notes: `Cool at 1-2 K/h from ${Math.round(synthTemp * 1.2)} K to ${synthTemp} K`,
      });

      steps.push({
        stepNumber: stepNum++,
        reactants: [`${formula} crystals + flux`],
        products: [formula],
        temperature: 25,
        pressure: 0,
        atmosphere: "air",
        reactionType: "flux removal",
        duration: "4 hours",
        notes: "Centrifuge or acid etch to remove excess flux",
      });
    } else {
      steps.push({
        stepNumber: stepNum++,
        reactants: precursorFormulas,
        products: [formula],
        temperature: synthTemp,
        pressure,
        atmosphere: template.atmosphere[0] || "Ar",
        reactionType: template.method,
        duration: `${Math.round((template.durationRange[0] + template.durationRange[1]) / 2)} hours`,
        notes: `${template.name} synthesis`,
      });
    }
  }

  return steps;
}

function estimateYield(feasibility: number, method: string, nSteps: number): string {
  let baseYield = feasibility * 100;

  if (method === "solid-state") baseYield *= 0.9;
  else if (method === "arc-melting") baseYield *= 0.85;
  else if (method === "high-pressure") baseYield *= 0.5;
  else if (method === "sputtering" || method === "PLD" || method === "MBE") baseYield *= 0.7;
  else if (method === "CVD") baseYield *= 0.65;
  else if (method === "sol-gel") baseYield *= 0.8;
  else if (method === "flux-growth") baseYield *= 0.6;
  else if (method === "ball-milling") baseYield *= 0.85;

  baseYield *= Math.pow(0.95, nSteps - 1);
  baseYield = Math.max(5, Math.min(95, baseYield));

  return `${Math.round(baseYield)}%`;
}

function estimateTotalDuration(steps: SynthesisStep[]): string {
  let totalHours = 0;
  for (const step of steps) {
    const match = step.duration.match(/(\d+)/);
    if (match) {
      totalHours += parseInt(match[1], 10);
    }
    if (step.duration.includes("days")) {
      const daysMatch = step.duration.match(/(\d+)/);
      if (daysMatch) totalHours += parseInt(daysMatch[1], 10) * 24;
    }
  }
  if (totalHours > 48) {
    return `~${Math.round(totalHours / 24)} days`;
  }
  return `~${totalHours} hours`;
}

function computeComplexityPenalty(nSteps: number, nElements: number, pressure: number, method: string): number {
  let penalty = 1.0;

  if (nSteps > 5) penalty *= 0.9;
  if (nSteps > 8) penalty *= 0.85;

  if (nElements > 4) penalty *= 0.95;
  if (nElements > 6) penalty *= 0.9;

  if (pressure > 100) penalty *= 0.7;
  else if (pressure > 50) penalty *= 0.8;
  else if (pressure > 10) penalty *= 0.9;

  if (method === "MBE") penalty *= 0.85;
  if (method === "high-pressure") penalty *= 0.9;

  return Math.max(0.3, penalty);
}

function buildSynthesisRoute(
  formula: string,
  elements: string[],
  templateMatch: TemplateMatch,
  precursorSelections: PrecursorSelection[],
  precursorResult: PrecursorAvailabilityResult,
  thermoResult: ReactionFeasibilityResult,
  intermediateRoute?: string[]
): SynthesisRoute {
  const template = templateMatch.template;
  const steps = buildStepsFromTemplate(
    formula, elements, template, precursorSelections, thermoResult, intermediateRoute
  );

  const pressure = thermoResult.pressureRequirement;
  const complexityPenalty = computeComplexityPenalty(
    steps.length, elements.length, pressure, template.method
  );

  const compositeScore =
    thermoResult.overallFeasibility * 0.30 +
    precursorResult.overallScore * 0.20 +
    templateMatch.score * 0.30 +
    complexityPenalty * 0.20;

  const safetyNotes: string[] = [];
  for (const sel of precursorSelections) {
    if (sel.precursor.safetyNotes && sel.precursor.safetyNotes !== "Safe" && sel.precursor.safetyNotes !== "Low toxicity") {
      safetyNotes.push(`${sel.precursor.formula}: ${sel.precursor.safetyNotes}`);
    }
  }
  if (pressure > 50) safetyNotes.push("High-pressure experiment: DAC safety protocols required");
  if (template.method === "arc-melting") safetyNotes.push("Arc furnace: UV protection and electrical safety required");

  const maxTemp = Math.max(...steps.map(s => s.temperature));
  const maxPressure = Math.max(...steps.map(s => s.pressure));

  return {
    routeId: `plan-${template.id}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    formula,
    method: template.method,
    routeName: intermediateRoute && intermediateRoute.length > 1
      ? `${template.name} (multi-step via intermediates)`
      : template.name,
    steps,
    precursors: precursorSelections.map(s => s.precursor.formula),
    precursorDetails: precursorSelections,
    feasibilityScore: Number(Math.max(0, Math.min(1, compositeScore)).toFixed(4)),
    templateMatchScore: templateMatch.score,
    precursorAvailabilityScore: precursorResult.overallScore,
    thermodynamicFeasibilityScore: thermoResult.overallFeasibility,
    complexityPenalty: Number(complexityPenalty.toFixed(4)),
    totalCostEstimate: precursorResult.costEstimate,
    equipmentList: [...template.equipment],
    safetyNotes,
    estimatedYield: estimateYield(compositeScore, template.method, steps.length),
    maxTemperature: maxTemp,
    maxPressure,
    totalDuration: estimateTotalDuration(steps),
    difficulty: template.difficulty,
    reasoning: templateMatch.reasoning,
  };
}

export function planSynthesisRoutes(
  formula: string,
  options: SynthesisPlanOptions = {}
): SynthesisPlanResult {
  const {
    formationEnergy = null,
    maxRoutes = 8,
    preferredMethods,
    includeMultiStep = true,
  } = options;

  const family = classifyFamily(formula);
  const elements = parseFormulaElements(formula);

  const templateMatches = matchTemplates(formula, family, formationEnergy);

  const thermoResult = computeReactionFeasibility(
    formula,
    formationEnergy,
    elements
  );

  const allRoutes: SynthesisRoute[] = [];
  const seenMethods = new Set<string>();

  const topTemplates = templateMatches
    .filter(tm => tm.score > 0.15)
    .slice(0, 10);

  for (const tm of topTemplates) {
    const precursorSelections = findBestPrecursors(elements, tm.template.method);
    const precursorResult = computePrecursorAvailabilityScore(precursorSelections);

    const route = buildSynthesisRoute(
      formula, elements, tm, precursorSelections, precursorResult, thermoResult
    );
    allRoutes.push(route);
    seenMethods.add(tm.template.method);

    if (includeMultiStep && elements.length >= 3) {
      const intermediateRoutes = decomposeIntoIntermediateRoutes(formula, elements);
      const multiStepRoutes = intermediateRoutes.filter(r => r.length > 1);

      for (const intRoute of multiStepRoutes.slice(0, 2)) {
        const multiRoute = buildSynthesisRoute(
          formula, elements, tm, precursorSelections, precursorResult, thermoResult, intRoute
        );
        multiRoute.routeId = `plan-multi-${tm.template.id}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        allRoutes.push(multiRoute);
      }
    }
  }

  if (preferredMethods) {
    for (const method of preferredMethods) {
      if (!seenMethods.has(method)) {
        const methodTemplate = templateMatches.find(tm => tm.template.method === method);
        if (methodTemplate) {
          const precursorSelections = findBestPrecursors(elements, method);
          const precursorResult = computePrecursorAvailabilityScore(precursorSelections);
          const route = buildSynthesisRoute(
            formula, elements, methodTemplate, precursorSelections, precursorResult, thermoResult
          );
          allRoutes.push(route);
        }
      }
    }
  }

  allRoutes.sort((a, b) => b.feasibilityScore - a.feasibilityScore);

  const rankedRoutes = allRoutes.slice(0, maxRoutes);
  const bestRoute = rankedRoutes.length > 0 ? rankedRoutes[0] : null;

  const bestPrecursorSelections = bestRoute
    ? findBestPrecursors(elements, bestRoute.method)
    : findBestPrecursors(elements, "solid-state");
  const precursorAvailability = computePrecursorAvailabilityScore(bestPrecursorSelections);

  const summary = bestRoute
    ? `Best route: ${bestRoute.routeName} (feasibility ${(bestRoute.feasibilityScore * 100).toFixed(1)}%), ` +
      `${bestRoute.steps.length} steps, max ${bestRoute.maxTemperature} K, ` +
      `${bestRoute.totalDuration}, cost: ${bestRoute.totalCostEstimate}, ` +
      `yield: ${bestRoute.estimatedYield}. ` +
      `${rankedRoutes.length} total routes evaluated.`
    : `No viable synthesis route found for ${formula}`;

  return {
    formula,
    family,
    routes: rankedRoutes,
    bestRoute,
    summary,
    thermodynamics: thermoResult,
    precursorAvailability,
  };
}

let plannerStats = {
  totalPlans: 0,
  totalRoutes: 0,
  methodBreakdown: {} as Record<string, number>,
  avgFeasibility: 0,
  feasibilitySum: 0,
};

export function planAndTrack(formula: string, options?: SynthesisPlanOptions): SynthesisPlanResult {
  const result = planSynthesisRoutes(formula, options);

  plannerStats.totalPlans++;
  plannerStats.totalRoutes += result.routes.length;

  if (result.bestRoute) {
    plannerStats.feasibilitySum += result.bestRoute.feasibilityScore;
    plannerStats.avgFeasibility = plannerStats.feasibilitySum / plannerStats.totalPlans;

    const method = result.bestRoute.method;
    plannerStats.methodBreakdown[method] = (plannerStats.methodBreakdown[method] || 0) + 1;
  }

  return result;
}

export function getSynthesisPlannerStats() {
  return {
    totalPlans: plannerStats.totalPlans,
    totalRoutes: plannerStats.totalRoutes,
    avgFeasibility: Number(plannerStats.avgFeasibility.toFixed(4)),
    methodBreakdown: { ...plannerStats.methodBreakdown },
  };
}
