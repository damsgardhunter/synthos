import { matchTemplates, type TemplateMatch, type ReactionTemplate } from "./reaction-templates";
import { findBestPrecursors, computePrecursorAvailabilityScore, type PrecursorSelection, type PrecursorAvailabilityResult } from "./precursor-database";
import { computeReactionFeasibility, computeSynthesisTemperature, assessPressureRequirement, type ReactionFeasibilityResult } from "./thermodynamic-feasibility";
import { classifyFamily } from "../learning/utils";
import { getMeltingPoint } from "../learning/elemental-data";
import { proposeAnalogousRoutes, type AnalogyTransferResult } from "./synthesis-analogy-engine";
import { getAcousticRouteTemplates, computeAcousticSynthesisEffects, type AcousticRouteTemplate, type AcousticSynthesisResult, type MultiPhysicsConditions } from "../physics/acoustic-synthesis-engine";

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
  /** Acoustic/multi-physics conditions for acoustic-assisted routes */
  multiPhysicsConditions?: MultiPhysicsConditions;
  /** Full acoustic synthesis analysis for this route */
  acousticSynthesisResult?: AcousticSynthesisResult;
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
  pressureGpa?: number; // actual DFT/candidate pressure — overrides heuristic in thermodynamic-feasibility
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
        pressure: 0.2,
        atmosphere: "air",
        reactionType: "pressing",
        duration: "30 minutes",
        notes: "Uniaxial press at 0.2 GPa",
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

// Literature-grounded yield ranges [min%, max%] per synthesis method.
// The feasibility score (0–1) modulates position within the range; it does NOT
// act as a direct yield multiplier because a 0–1 ML score ≠ a 0–100 % mass yield.
//
// References:
//   solid-state:   West (2014) "Solid State Chemistry and its Applications", 2nd ed.;
//                  typical 60–90 % — incomplete sintering kinetics, grinding losses.
//   arc-melting:   Canfield & Fisk (1992) Phil. Mag. B 65:1117;
//                  80–95 % — mostly mass-conserving; volatile elements increase loss.
//   high-pressure: Hemley et al. (1997) PNAS 94:2176;
//                  70–92 % — small sample but high conversion efficiency in DAC/anvil.
//   sputtering:    Puurunen (2005) J. Appl. Phys. 97:121301;
//                  40–70 % — geometric deposition efficiency, chamber wall losses.
//   PLD:           Eason ed. (2007) "Pulsed Laser Deposition of Thin Films";
//                  50–75 % — plume angular distribution limits substrate coverage.
//   MBE:           Arthur (2002) Surf. Sci. 500:189;
//                  70–88 % — controlled flux but sticking coefficient < 1.
//   CVD:           Pierson (1999) "Handbook of CVD", 2nd ed.;
//                  70–85 % — optimised precursors give >80 %; simple oxides/nitrides.
//   sol-gel:       Brinker & Scherer (1990) "Sol-Gel Science";
//                  50–80 % — calcination shrinkage and densification losses.
//   flux-growth:   Canfield (2020) Rev. Sci. Instrum. 91:103903;
//                  20–60 % — crystallographic yield after decanting and acid etching.
//   ball-milling:  Suryanarayana (2001) Prog. Mater. Sci. 46:1;
//                  90–98 % — near-quantitative mechanical mixing, minimal loss.
const METHOD_YIELD_RANGE: Record<string, readonly [number, number]> = {
  "solid-state":   [60, 90],
  "arc-melting":   [80, 95],
  "high-pressure": [70, 92],
  "sputtering":    [40, 70],
  "PLD":           [50, 75],
  "MBE":           [70, 88],
  "CVD":           [70, 85],
  "sol-gel":       [50, 80],
  "flux-growth":   [20, 60],
  "ball-milling":  [90, 98],
} as const;
const DEFAULT_YIELD_RANGE: readonly [number, number] = [50, 85];

// Per-step yield loss: ~3 % per transfer/firing step beyond the first.
// Source: empirical solid-state mass-loss data compiled in Ruiz-Hitzky et al. (2013)
// Adv. Mater. 25:998 and West (2014) §3.2; typical range 2–5 %, 3 % is conservative.
const STEP_YIELD_LOSS_FRACTION = 0.03;

function estimateYield(feasibility: number, method: string, nSteps: number): string {
  const [minY, maxY] = METHOD_YIELD_RANGE[method] ?? DEFAULT_YIELD_RANGE;
  const f = Math.max(0, Math.min(1, feasibility));

  // Feasibility shifts position within the literature baseline range.
  // f = 0 → minY (worst-case for that method), f = 1 → maxY (best-case).
  let yieldPct = minY + f * (maxY - minY);

  // Compound per-step loss over each step beyond the first.
  const extraSteps = Math.max(0, nSteps - 1);
  yieldPct *= Math.pow(1 - STEP_YIELD_LOSS_FRACTION, extraSteps);

  // Clamp: 0 % and 100 % are not physically meaningful at planning stage.
  yieldPct = Math.max(5, Math.min(95, yieldPct));

  return `${Math.round(yieldPct)}%`;
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

/** Convert an AcousticRouteTemplate to a SynthesisRoute for ranking alongside conventional routes. */
function adaptAcousticRoute(
  formula: string,
  template: AcousticRouteTemplate,
  acousticResult: AcousticSynthesisResult,
  thermoResult: ReactionFeasibilityResult,
): SynthesisRoute {
  const steps: SynthesisStep[] = template.steps.map((s, i) => ({
    stepNumber: i + 1,
    reactants: [formula],
    products: i === template.steps.length - 1 ? [formula] : ["intermediate"],
    temperature: acousticResult.conditions.baseTemperatureK,
    pressure: acousticResult.conditions.staticPressureGpa,
    atmosphere: template.conditions.acousticField?.medium === "liquid-ammonia" ? "liquid NH₃ under pressure" : "controlled atmosphere",
    reactionType: template.method,
    duration: `${s.durationHours} hours`,
    notes: s.description + (s.notes ? ` — ${s.notes}` : ""),
  }));

  // Acoustic feasibility = thermodynamic base + template modifier + synthesizability bonus
  const baseFeasibility = Math.max(0.1, thermoResult.overallFeasibility);
  const feasibility = Math.min(0.98, baseFeasibility + template.feasibilityModifier + acousticResult.synthesizabilityBonus);

  const safetyNotes = [...acousticResult.warnings];
  if (template.conditions.staticPressureGpa > 50) {
    safetyNotes.push("High-pressure experiment: DAC safety protocols required");
  }

  const maxTemp = Math.max(
    acousticResult.conditions.baseTemperatureK,
    acousticResult.peakCavitationTemperatureK,
  );

  return {
    routeId: `acoustic-${template.method}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    formula,
    method: template.method,
    routeName: template.name,
    steps,
    precursors: [],
    precursorDetails: [],
    feasibilityScore: Number(feasibility.toFixed(4)),
    templateMatchScore: 0.5,
    precursorAvailabilityScore: 0.5,
    thermodynamicFeasibilityScore: thermoResult.overallFeasibility,
    complexityPenalty: 0.85,
    totalCostEstimate: template.method === "shock-wave-synthesis" ? "$$$$ (specialist facility)" : "$$$ (specialized equipment)",
    equipmentList: template.equipmentRequired,
    safetyNotes,
    estimatedYield: template.method === "shock-wave-synthesis" ? "~20-40%" : "~60-75%",
    maxTemperature: maxTemp,
    maxPressure: Math.max(
      template.conditions.staticPressureGpa,
      acousticResult.peakAcousticPressureGpa,
    ),
    totalDuration: `~${template.steps.reduce((s, step) => s + step.durationHours, 0).toFixed(1)} hours`,
    difficulty: template.method === "shock-wave-synthesis" ? "Expert" : "Advanced",
    reasoning: [
      template.notes,
      ...acousticResult.activeEffects,
      acousticResult.staticPressureReductionGpa > 1
        ? `Static pressure reduction: ${acousticResult.staticPressureReductionGpa.toFixed(1)} GPa`
        : "",
      acousticResult.phononResonanceActive
        ? `Phonon resonance: λ × ${acousticResult.lambdaEnhancementFactor.toFixed(2)}`
        : "",
    ].filter(Boolean),
    multiPhysicsConditions: acousticResult.conditions,
    acousticSynthesisResult: acousticResult,
  };
}

/** Map an analogy engine route (reaction-pathway.ts types) to the planner's SynthesisRoute. */
function adaptAnalogyRoute(formula: string, ar: AnalogyTransferResult["routes"][number]): SynthesisRoute {
  return {
    routeId: ar.routeId,
    formula,
    method: ar.method,
    routeName: ar.routeName,
    steps: ar.steps as unknown as SynthesisStep[], // ReactionStep is structurally identical to SynthesisStep
    precursors: ar.precursors,
    precursorDetails: [],
    feasibilityScore: ar.feasibilityScore,
    templateMatchScore: 0,
    precursorAvailabilityScore: 0.6,
    thermodynamicFeasibilityScore: ar.thermodynamics?.overallFeasibility ?? ar.feasibilityScore,
    complexityPenalty: 0.1,
    totalCostEstimate: "$$ (analogy-transferred)",
    equipmentList: ar.equipment ?? [],
    safetyNotes: [],
    estimatedYield: "~70%",
    maxTemperature: ar.maxTemperature,
    maxPressure: ar.maxPressure,
    totalDuration: ar.totalDuration,
    difficulty: ar.difficulty,
    reasoning: ar.notes ? [ar.notes] : [],
  };
}

export async function planSynthesisRoutes(
  formula: string,
  options: SynthesisPlanOptions = {}
): Promise<SynthesisPlanResult> {
  const {
    formationEnergy = null,
    maxRoutes = 8,
    preferredMethods,
    includeMultiStep = true,
    pressureGpa,
  } = options;

  const family = classifyFamily(formula);
  const elements = parseFormulaElements(formula);

  const templateMatches = matchTemplates(formula, family, formationEnergy);

  const thermoResult = computeReactionFeasibility(
    formula,
    formationEnergy,
    elements,
    pressureGpa, // pass actual DFT pressure — prevents ambient-pressure synthesis of 128 GPa materials
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

  // Merge analogy-transferred routes from structurally similar known compounds.
  // These carry real experimental conditions scaled by melting-point ratio — higher
  // information value than pure template routes for novel compositions.
  try {
    const analogyResult = await proposeAnalogousRoutes(formula);
    for (const ar of analogyResult.routes) {
      if (!seenMethods.has(ar.method)) {
        allRoutes.push(adaptAnalogyRoute(formula, ar));
        seenMethods.add(ar.method);
      }
    }
  } catch { /* non-fatal — analogy engine may have no data for this formula */ }

  // Inject acoustic/multi-physics routes.
  // For any compound requiring > 10 GPa, or any hydride, acoustic routes are generated.
  // These can supplement or partially replace static DAC pressure via cavitation.
  try {
    const requiredPressure = thermoResult.pressureRequirement;
    const acousticTemplates = getAcousticRouteTemplates(formula, family, requiredPressure);
    for (const at of acousticTemplates) {
      if (!seenMethods.has(at.method)) {
        const acousticResult = computeAcousticSynthesisEffects(
          formula,
          at.conditions,
          null,
          requiredPressure,
        );
        allRoutes.push(adaptAcousticRoute(formula, at, acousticResult, thermoResult));
        seenMethods.add(at.method);
      }
    }
  } catch { /* non-fatal — acoustic engine may not apply to this formula */ }

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

const PLANNER_STATS_WINDOW = 10_000;

let plannerStats = {
  totalPlans: 0,
  totalRoutes: 0,
  methodBreakdown: {} as Record<string, number>,
  // Rolling window of the last PLANNER_STATS_WINDOW feasibility scores.
  // Using a window instead of a running sum prevents unbounded float growth
  // (feasibilitySum would lose precision after ~10^9 additions at 864 DFT/day).
  feasibilityWindow: [] as number[],
};

export async function planAndTrack(formula: string, options?: SynthesisPlanOptions): Promise<SynthesisPlanResult> {
  const result = await planSynthesisRoutes(formula, options);

  plannerStats.totalPlans++;
  plannerStats.totalRoutes += result.routes.length;

  if (result.bestRoute) {
    plannerStats.feasibilityWindow.push(result.bestRoute.feasibilityScore);
    if (plannerStats.feasibilityWindow.length > PLANNER_STATS_WINDOW) {
      plannerStats.feasibilityWindow.shift();
    }

    const method = result.bestRoute.method;
    plannerStats.methodBreakdown[method] = (plannerStats.methodBreakdown[method] || 0) + 1;
  }

  return result;
}

export function getSynthesisPlannerStats() {
  const w = plannerStats.feasibilityWindow;
  const avgFeasibility = w.length > 0
    ? w.reduce((s, v) => s + v, 0) / w.length
    : 0;
  return {
    totalPlans: plannerStats.totalPlans,
    totalRoutes: plannerStats.totalRoutes,
    avgFeasibility: Number(avgFeasibility.toFixed(4)),
    methodBreakdown: { ...plannerStats.methodBreakdown },
  };
}
