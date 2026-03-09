import { isTransitionMetal, isRareEarth, isActinide, getMeltingPoint } from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";

export interface HeuristicRoute {
  rule: string;
  method: string;
  precursors: string[];
  product: string;
  equation: string;
  steps: string[];
  temperature: number;
  pressure: number;
  atmosphere: string;
  difficulty: string;
  confidence: number;
  notes: string;
}

interface FormulaBreakdown {
  elements: string[];
  counts: Record<string, number>;
  totalAtoms: number;
  hasH: boolean;
  hasO: boolean;
  hasN: boolean;
  hasC: boolean;
  hasB: boolean;
  hasS: boolean;
  hasSe: boolean;
  hasTe: boolean;
  hasF: boolean;
  metals: string[];
  nonmetals: string[];
  transitionMetals: string[];
  rareEarths: string[];
  hFraction: number;
}

const NONMETALS = new Set(["H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "I", "Xe", "At", "Rn", "Te", "As"]);
const ALKALI = new Set(["Li", "Na", "K", "Rb", "Cs"]);
const ALKALINE_EARTH = new Set(["Be", "Mg", "Ca", "Sr", "Ba"]);
const METALLOIDS = new Set(["B", "Si", "Ge", "As", "Sb", "Te"]);

function parseFormula(formula: string): FormulaBreakdown {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const metals = elements.filter(e => !NONMETALS.has(e) && !METALLOIDS.has(e));
  const nonmetals = elements.filter(e => NONMETALS.has(e) || METALLOIDS.has(e));
  const transitionMetals = elements.filter(e => isTransitionMetal(e));
  const rareEarths = elements.filter(e => isRareEarth(e) || isActinide(e));
  const hCount = counts["H"] ?? 0;

  return {
    elements, counts, totalAtoms,
    hasH: !!counts["H"], hasO: !!counts["O"], hasN: !!counts["N"],
    hasC: !!counts["C"], hasB: !!counts["B"], hasS: !!counts["S"],
    hasSe: !!counts["Se"], hasTe: !!counts["Te"], hasF: !!counts["F"],
    metals, nonmetals, transitionMetals, rareEarths,
    hFraction: hCount / Math.max(1, totalAtoms),
  };
}

function maxMeltingPoint(elements: string[]): number {
  let maxMp = 800;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp && mp > maxMp) maxMp = mp;
  }
  return maxMp;
}

function formatEquation(reactants: string[], products: string[]): string {
  return `${reactants.join(" + ")} → ${products.join(" + ")}`;
}

function ruleHydrogenation(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (!fb.hasH || fb.metals.length === 0) return null;

  const metalPart = fb.metals.map(m => {
    const c = fb.counts[m];
    return c > 1 ? `${m}${c}` : m;
  }).join("");

  const hCount = fb.counts["H"] ?? 0;
  const isHighPressure = fb.hFraction > 0.5 || hCount > 6;

  const precursors = [metalPart, `H₂`];
  const pressure = isHighPressure ? 150 : (hCount > 3 ? 10 : 1);
  const temperature = isHighPressure ? 1500 : 600;

  return {
    rule: "hydrogenation",
    method: isHighPressure ? "diamond-anvil-cell" : "gas-phase-hydrogenation",
    precursors,
    product: formula,
    equation: formatEquation(precursors, [formula]),
    steps: [
      `Load ${metalPart} into ${isHighPressure ? "DAC sample chamber" : "pressure vessel"}`,
      `Pressurize with H₂ to ${pressure} GPa`,
      `Heat to ${temperature}K under ${pressure} GPa H₂`,
      `Hold for ${isHighPressure ? "1-6 hours" : "12-24 hours"} to allow hydrogen absorption`,
      isHighPressure ? "Characterize in-situ via XRD at synchrotron" : "Slowly cool and depressurize",
    ],
    temperature,
    pressure,
    atmosphere: "H₂",
    difficulty: isHighPressure ? "very-hard" : "moderate",
    confidence: isHighPressure ? 0.5 : 0.7,
    notes: isHighPressure
      ? `Superhydride synthesis requires DAC at ${pressure} GPa. Product may be metastable at ambient pressure.`
      : `Standard hydrogenation of ${metalPart}. H₂ overpressure promotes complete uptake.`,
  };
}

function ruleAlloying(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (fb.metals.length < 2 || fb.nonmetals.length > 0) return null;
  if (fb.hasH || fb.hasO || fb.hasN || fb.hasS || fb.hasSe || fb.hasTe || fb.hasB || fb.hasC || fb.hasF) return null;

  const precursors = fb.metals.map(m => m);
  const maxMp = maxMeltingPoint(fb.metals);
  const temperature = Math.round(maxMp * 1.1);

  return {
    rule: "alloying",
    method: "arc-melting",
    precursors,
    product: formula,
    equation: formatEquation(precursors, [formula]),
    steps: [
      `Weigh stoichiometric amounts of ${precursors.join(", ")} (>99.9% purity)`,
      "Place on water-cooled Cu hearth in arc-melting furnace",
      `Purge chamber 3x with Ar, maintain Ar flow`,
      `Melt Ti getter to scavenge residual O₂`,
      `Arc-melt constituents at ~${temperature}K, flip and re-melt 4-5 times for homogeneity`,
      "Allow to cool on hearth under Ar",
      "Anneal in sealed quartz tube at 0.7·T_melt for 1 week if ordered phase required",
    ],
    temperature,
    pressure: 0,
    atmosphere: "Ar",
    difficulty: "moderate",
    confidence: 0.8,
    notes: `Intermetallic alloy formed by arc melting. ${fb.metals.length > 2 ? "Multi-component alloy may require extended annealing." : "Binary alloy synthesis is straightforward."}`,
  };
}

function ruleSolidState(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (fb.metals.length < 1 || fb.elements.length < 2) return null;
  if (fb.hasH) return null;

  const precursors: string[] = [];
  const steps: string[] = [];

  for (const m of fb.metals) {
    if (fb.hasO) {
      if (ALKALINE_EARTH.has(m)) {
        precursors.push(`${m}CO₃`);
      } else if (ALKALI.has(m)) {
        precursors.push(`${m}₂CO₃`);
      } else {
        precursors.push(`${m}₂O₃`);
      }
    } else {
      precursors.push(m);
    }
  }
  for (const nm of fb.nonmetals) {
    if (nm === "O") continue;
    if (nm === "B") precursors.push("B");
    else if (nm === "C") precursors.push("C");
    else if (nm === "N") precursors.push("N source (urea or NH₃ gas)");
    else if (nm === "S") precursors.push("S");
    else if (nm === "Se") precursors.push("Se");
    else if (nm === "Te") precursors.push("Te");
    else if (nm === "P") precursors.push("P (red)");
    else if (nm === "As") precursors.push("As");
    else if (nm === "F") precursors.push(`${fb.metals[0] ?? ""}F₂`);
    else precursors.push(nm);
  }

  const maxMp = maxMeltingPoint(fb.metals);
  const temperature = Math.round(maxMp * 0.7);
  const isCuprate = fb.transitionMetals.includes("Cu") && fb.hasO;

  steps.push(`Dry precursors (${precursors.join(", ")}) at 200°C for 12h`);
  steps.push("Grind in agate mortar for 30 min or ball-mill for 2h");
  steps.push("Press into pellets at 5-10 MPa");
  steps.push(`Calcine at ${Math.round(temperature * 0.7)}K for 12h in air`);
  steps.push("Re-grind, re-pelletize");
  steps.push(`Sinter at ${temperature}K for 24-48h in ${fb.hasO ? "flowing O₂" : "sealed tube under Ar"}`);
  if (isCuprate) {
    steps.push("Slow-cool at 1K/min through 700-400K range for oxygen ordering");
    steps.push("Anneal at 450°C in flowing O₂ for 48h for optimal oxygen stoichiometry");
  } else {
    steps.push("Furnace-cool to room temperature");
  }

  return {
    rule: "solid-state-reaction",
    method: "solid-state",
    precursors,
    product: formula,
    equation: formatEquation(precursors, fb.hasO ? [formula] : [formula]),
    steps,
    temperature,
    pressure: 0,
    atmosphere: fb.hasO ? "O₂" : "Ar",
    difficulty: "moderate",
    confidence: 0.75,
    notes: isCuprate
      ? "Cuprate synthesis requires careful oxygen annealing for optimal Tc."
      : `Standard ceramic/solid-state synthesis. Multiple grind-sinter cycles improve phase purity.`,
  };
}

function rulePrecursorDecomposition(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (!fb.hasO || fb.metals.length === 0) return null;

  const decomposableMetals = fb.metals.filter(m => ALKALINE_EARTH.has(m) || ALKALI.has(m) || isRareEarth(m));
  if (decomposableMetals.length === 0) return null;

  const precursors: string[] = [];
  const byproducts: string[] = [];
  const steps: string[] = [];

  for (const m of decomposableMetals) {
    if (ALKALINE_EARTH.has(m)) {
      precursors.push(`${m}CO₃`);
      byproducts.push("CO₂");
    } else if (ALKALI.has(m)) {
      precursors.push(`${m}₂CO₃`);
      byproducts.push("CO₂");
    } else {
      precursors.push(`${m}(NO₃)₃`);
      byproducts.push("NO₂", "O₂");
    }
  }

  const otherMetals = fb.metals.filter(m => !decomposableMetals.includes(m));
  for (const m of otherMetals) {
    precursors.push(`${m}₂O₃`);
  }

  const temperature = 1100;

  steps.push(`Dissolve ${precursors.join(", ")} in dilute HNO₃ or water`);
  steps.push("Mix solutions in stoichiometric ratio");
  steps.push("Evaporate to dryness at 80°C");
  steps.push(`Decompose precursors at 600K in air (releases ${byproducts.join(", ")})`);
  steps.push("Grind decomposed powder, press pellets");
  steps.push(`Sinter at ${temperature}K for 24h in flowing O₂`);
  steps.push("Cool slowly, characterize by XRD");

  return {
    rule: "precursor-decomposition",
    method: "decomposition",
    precursors,
    product: formula,
    equation: formatEquation(precursors, [formula, ...byproducts]),
    steps,
    temperature,
    pressure: 0,
    atmosphere: "air → O₂",
    difficulty: "easy",
    confidence: 0.7,
    notes: `Carbonate/nitrate decomposition route yields fine-grained oxide. Better homogeneity than direct solid-state.`,
  };
}

function ruleSolGel(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (!fb.hasO || fb.metals.length < 2) return null;
  if (fb.hasH && fb.hFraction > 0.3) return null;

  const precursors: string[] = [];
  for (const m of fb.metals) {
    if (isTransitionMetal(m)) {
      precursors.push(`${m} acetylacetonate`);
    } else if (ALKALINE_EARTH.has(m) || ALKALI.has(m)) {
      precursors.push(`${m} acetate`);
    } else if (isRareEarth(m)) {
      precursors.push(`${m}(NO₃)₃·6H₂O`);
    } else {
      precursors.push(`${m} alkoxide`);
    }
  }

  return {
    rule: "sol-gel",
    method: "sol-gel",
    precursors,
    product: formula,
    equation: formatEquation(precursors, [formula, "organics"]),
    steps: [
      `Dissolve ${precursors.join(", ")} in 2-methoxyethanol`,
      "Add citric acid (1:1 molar ratio to total metals) as chelating agent",
      "Stir at 60°C for 4h to form a clear sol",
      "Gel at 80°C for 12h",
      "Dry gel at 200°C, grind to powder",
      "Calcine at 600°C for 4h to burn off organics",
      "Press into pellets, sinter at 900°C for 12h in O₂",
    ],
    temperature: 900,
    pressure: 0,
    atmosphere: "O₂",
    difficulty: "moderate",
    confidence: 0.65,
    notes: "Sol-gel route provides atomic-level mixing for complex oxides. Better stoichiometric control than solid-state.",
  };
}

function ruleBallMilling(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (fb.metals.length < 2) return null;
  if (fb.hasO || fb.hasN || fb.hasH || fb.hasSe || fb.hasTe || fb.hasS) return null;

  const precursors = fb.elements.map(e => e);

  return {
    rule: "ball-milling",
    method: "mechanical-alloying",
    precursors,
    product: formula,
    equation: formatEquation(precursors, [formula]),
    steps: [
      `Load ${precursors.join(", ")} powder in stoichiometric ratio into WC vial`,
      "Add WC balls (ball-to-powder ratio 10:1)",
      "Seal vial under Ar in glovebox",
      "Mill at 300 rpm for 20h with 30-min rest intervals",
      "Characterize by XRD; if not single phase, continue milling",
      "Consolidate by SPS at 0.7·T_melt for 5 min under 50 MPa",
    ],
    temperature: Math.round(maxMeltingPoint(fb.metals) * 0.7),
    pressure: 0.05,
    atmosphere: "Ar",
    difficulty: "moderate",
    confidence: 0.6,
    notes: "Mechanical alloying achieves non-equilibrium phases. Useful for metastable intermetallics and amorphous precursors.",
  };
}

function ruleBorideSynthesis(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (!fb.hasB || fb.metals.length === 0) return null;
  if (fb.hasH) return null;

  const bCount = fb.counts["B"] ?? 1;
  const metalOxides = fb.metals.map(m => `${m}₂O₃`);
  const precursors = [...metalOxides, bCount > 2 ? "amorphous B" : "B₂O₃", "C (excess)"];
  const metalNames = fb.metals.join(", ");

  return {
    rule: "boride-synthesis",
    method: "borothermal-reduction",
    precursors,
    product: formula,
    equation: formatEquation(precursors, [formula, "CO"]),
    steps: [
      `Mix ${metalNames} oxide(s) with B₂O₃ and excess carbon in stoichiometric ratio`,
      "Ball-mill for 4h under Ar",
      "Press into pellets at 20 MPa",
      "React at 1800K under flowing Ar for 6h",
      "Cool under Ar, wash with dilute HCl to remove oxide residues",
      "Characterize by XRD and confirm boride phase",
    ],
    temperature: 1800,
    pressure: 0,
    atmosphere: "Ar",
    difficulty: "hard",
    confidence: 0.65,
    notes: `Carbothermal/borothermal reduction of ${metalNames} oxide(s). High temperature required for refractory boride formation.`,
  };
}

function ruleChalcogenide(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  const chalcogen = fb.hasSe ? "Se" : fb.hasTe ? "Te" : fb.hasS ? "S" : null;
  if (!chalcogen || fb.metals.length === 0) return null;
  if (fb.hasH || fb.hasO) return null;

  const precursors = [...fb.metals, chalcogen];
  const maxMp = maxMeltingPoint(fb.metals);
  const temperature = Math.min(maxMp, 1200);

  return {
    rule: "chalcogenide-synthesis",
    method: "sealed-tube",
    precursors,
    product: formula,
    equation: formatEquation(precursors, [formula]),
    steps: [
      `Weigh stoichiometric amounts of ${precursors.join(", ")} (all >99.9%)`,
      "Load into quartz ampoule in Ar glovebox",
      `Seal ampoule under vacuum (10⁻³ mbar)`,
      `Heat slowly (1K/min) to ${temperature}K to avoid ${chalcogen} explosion`,
      `Hold at ${temperature}K for 48-72h with occasional rocking`,
      "Cool at 2K/min to room temperature",
      "Break ampoule, characterize product by XRD",
    ],
    temperature,
    pressure: 0,
    atmosphere: "vacuum (sealed tube)",
    difficulty: "moderate",
    confidence: 0.75,
    notes: `Sealed-tube synthesis for ${chalcogen}-containing compound. Slow heating critical to prevent violent reaction with volatile ${chalcogen}.`,
  };
}

function rulePnictide(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  const hasAs = fb.elements.includes("As");
  const hasP = fb.elements.includes("P");
  if (!hasAs && !hasP) return null;
  if (fb.metals.length === 0 || fb.hasH) return null;

  const pnictogen = hasAs ? "As" : "P";
  const precursors = [...fb.metals, pnictogen];
  const temperature = hasAs ? 1100 : 1000;

  return {
    rule: "pnictide-synthesis",
    method: "sealed-tube",
    precursors,
    product: formula,
    equation: formatEquation(precursors, [formula]),
    steps: [
      `Weigh ${precursors.join(", ")} in stoichiometric ratio (handle ${pnictogen} in fume hood)`,
      `Load into alumina crucible inside quartz tube`,
      `Evacuate and seal tube`,
      `Ramp at 0.5K/min to ${temperature}K (${pnictogen} is volatile and ${hasAs ? "toxic" : "pyrophoric"})`,
      `Hold at ${temperature}K for 48h`,
      "Slow-cool at 1K/min to 700K, then furnace-cool",
      "Grind, re-pelletize, re-anneal for improved phase purity",
    ],
    temperature,
    pressure: 0,
    atmosphere: "vacuum (sealed tube)",
    difficulty: hasAs ? "hard" : "moderate",
    confidence: 0.7,
    notes: hasAs
      ? "Arsenide synthesis requires strict safety protocols. As vapor is highly toxic."
      : "Phosphide synthesis under sealed-tube conditions. P vapor is pyrophoric.",
  };
}

function ruleThinFilm(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (fb.elements.length < 2 || fb.elements.length > 4) return null;
  if (fb.hasH) return null;

  const targets = fb.metals.map(m => m);
  const isOxide = fb.hasO;

  return {
    rule: "thin-film-deposition",
    method: isOxide ? "pulsed-laser-deposition" : "sputtering",
    precursors: [isOxide ? `${formula} ceramic target` : `${targets.join("-")} alloy target`],
    product: `${formula} thin film`,
    equation: `${formula} target → ${formula} thin film`,
    steps: isOxide ? [
      `Prepare ${formula} ceramic target by solid-state synthesis`,
      `Mount target in PLD chamber, load substrate (SrTiO₃ or LaAlO₃)`,
      `Evacuate to 10⁻⁶ mbar, heat substrate to 750°C`,
      `Introduce O₂ to 0.3 mbar partial pressure`,
      `Ablate target with KrF excimer laser (248 nm, 2 J/cm²)`,
      `Deposit at 1-2 Hz for 30 min (~100 nm film)`,
      `Cool in 500 mbar O₂ at 5°C/min for optimal oxygenation`,
    ] : [
      `Install ${targets.join("-")} target in magnetron sputtering system`,
      `Load substrate, evacuate to 10⁻⁷ mbar`,
      `Introduce Ar to 5×10⁻³ mbar, ignite plasma`,
      `Pre-sputter for 5 min to clean target surface`,
      `Deposit at 50W DC/RF for 1h (target thickness ~200 nm)`,
      `Post-anneal at 500°C in vacuum for 2h if ordered phase needed`,
    ],
    temperature: isOxide ? 1023 : 773,
    pressure: 0,
    atmosphere: isOxide ? "O₂ (0.3 mbar)" : "Ar (5×10⁻³ mbar)",
    difficulty: "hard",
    confidence: 0.55,
    notes: `Thin film route for ${formula}. Films allow strain-engineering and are ideal for transport measurements.`,
  };
}

function ruleNitrideSynthesis(formula: string, fb: FormulaBreakdown): HeuristicRoute | null {
  if (!fb.hasN || fb.metals.length === 0) return null;
  if (fb.hasH || fb.hasO) return null;

  const metalNames = fb.metals.join(", ");
  const metalPrecursors = fb.metals.map(m => `${m}₂O₃ or ${m}Cl₃`);
  const temperature = Math.min(maxMeltingPoint(fb.metals), 1500);

  return {
    rule: "nitride-synthesis",
    method: "ammonolysis",
    precursors: [...metalPrecursors, "NH₃ gas"],
    product: formula,
    equation: formatEquation([`${metalNames} oxide(s)`, "NH₃"], [formula, "H₂O"]),
    steps: [
      `Mix ${metalNames} oxide powders in stoichiometric ratio`,
      "Load mixed powder in alumina boat",
      "Place in tube furnace with gas flow capability",
      `Ramp to ${temperature}K under flowing NH₃ (100 mL/min)`,
      `Hold for 12-24h for complete nitridation`,
      "Cool under NH₃ to 400K, then switch to N₂",
      "Characterize by XRD and nitrogen analysis",
    ],
    temperature,
    pressure: 0,
    atmosphere: "NH₃",
    difficulty: "moderate",
    confidence: 0.65,
    notes: `Ammonolysis converts ${metalNames} oxides/chlorides to nitrides. Extended reaction times needed for bulk samples.`,
  };
}

const ALL_RULES = [
  ruleHydrogenation,
  ruleAlloying,
  ruleSolidState,
  rulePrecursorDecomposition,
  ruleSolGel,
  ruleBallMilling,
  ruleBorideSynthesis,
  ruleChalcogenide,
  rulePnictide,
  ruleThinFilm,
  ruleNitrideSynthesis,
];

let stats = { totalGenerated: 0, formulasProcessed: 0, ruleHits: {} as Record<string, number> };

export function generateHeuristicRoutes(formula: string, trackStats = false): HeuristicRoute[] {
  const fb = parseFormula(formula);
  if (fb.elements.length < 2) return [];

  const routes: HeuristicRoute[] = [];
  for (const rule of ALL_RULES) {
    try {
      const route = rule(formula, fb);
      if (route) {
        routes.push(route);
        if (trackStats) {
          stats.ruleHits[route.rule] = (stats.ruleHits[route.rule] || 0) + 1;
        }
      }
    } catch (_) {}
  }

  routes.sort((a, b) => b.confidence - a.confidence);
  if (trackStats) {
    stats.totalGenerated += routes.length;
    stats.formulasProcessed++;
  }
  return routes;
}

export function getHeuristicGeneratorStats() {
  return { ...stats, totalRules: ALL_RULES.length };
}
