import { ELEMENTAL_DATA, getMeltingPoint } from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";

export interface ReactionStep {
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

export interface ThermodynamicScoring {
  gibbsFreeEnergy: number;
  kineticBarrier: number;
  arrheniusRate: number;
  metastableQuenchFeasibility: number;
  overallFeasibility: number;
}

export interface SynthesisRoute {
  routeId: string;
  routeName: string;
  method: string;
  steps: ReactionStep[];
  precursors: string[];
  thermodynamics: ThermodynamicScoring;
  totalDuration: string;
  maxTemperature: number;
  maxPressure: number;
  difficulty: string;
  equipment: string[];
  feasibilityScore: number;
  notes: string;
}

export interface SynthesisPathwayResult {
  formula: string;
  family: string;
  routes: SynthesisRoute[];
  bestRoute: SynthesisRoute | null;
  summary: string;
}

export interface SynthesisPathwayStats {
  totalPathwaysComputed: number;
  totalRoutesGenerated: number;
  avgFeasibility: number;
  methodBreakdown: Record<string, number>;
  familyBreakdown: Record<string, number>;
}

const pathwayCache = new Map<string, SynthesisPathwayResult>();
let totalPathwaysComputed = 0;
let totalRoutesGenerated = 0;
const methodCounts: Record<string, number> = {};
const familyCounts: Record<string, number> = {};
const feasibilityValues: number[] = [];

function parseFormulaCounts(formula: string): Record<string, number> {
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
  return counts;
}

function parseFormulaElements(formula: string): string[] {
  const counts = parseFormulaCounts(formula);
  return Object.keys(counts);
}

const COMMON_PRECURSORS: Record<string, string[]> = {
  Y: ["Y2O3", "YCl3"],
  Ba: ["BaCO3", "BaO", "Ba(NO3)2"],
  Cu: ["CuO", "Cu2O", "Cu(NO3)2"],
  La: ["La2O3", "LaCl3"],
  Sr: ["SrCO3", "SrO"],
  Ca: ["CaCO3", "CaO"],
  Fe: ["Fe2O3", "Fe", "FeCl3"],
  As: ["As2O3", "As"],
  Se: ["Se", "SeO2"],
  Te: ["Te", "TeO2"],
  Nb: ["Nb2O5", "Nb", "NbCl5"],
  Ti: ["TiO2", "Ti", "TiCl4"],
  Mg: ["MgO", "Mg", "MgCl2"],
  B: ["B2O3", "B", "H3BO3"],
  Sn: ["SnO2", "Sn"],
  Pb: ["PbO", "Pb(NO3)2"],
  Bi: ["Bi2O3", "Bi"],
  Tl: ["Tl2O3", "TlNO3"],
  Hg: ["HgO", "HgCl2"],
  Zr: ["ZrO2", "ZrCl4"],
  Hf: ["HfO2", "HfCl4"],
  V: ["V2O5", "V"],
  Cr: ["Cr2O3", "Cr"],
  Mn: ["MnO2", "MnCO3"],
  Co: ["CoO", "Co(NO3)2"],
  Ni: ["NiO", "Ni(NO3)2"],
  Zn: ["ZnO", "Zn"],
  Al: ["Al2O3", "Al"],
  Ga: ["Ga2O3", "Ga"],
  In: ["In2O3", "In"],
  Ge: ["GeO2", "Ge"],
  Si: ["SiO2", "Si"],
  P: ["P2O5", "NH4H2PO4"],
  S: ["S", "Na2S"],
  N: ["N2", "NH3", "Li3N"],
  O: ["O2"],
  H: ["H2", "CaH2", "NaH", "LiH"],
  Li: ["Li2CO3", "LiOH", "Li"],
  Na: ["Na2CO3", "NaOH", "Na"],
  K: ["K2CO3", "KOH", "K"],
  Cs: ["Cs2CO3", "CsCl"],
  Rb: ["Rb2CO3", "RbCl"],
  W: ["WO3", "W"],
  Mo: ["MoO3", "Mo"],
  Ru: ["RuO2", "RuCl3"],
  Rh: ["Rh2O3", "RhCl3"],
  Pd: ["PdCl2", "Pd"],
  Ir: ["IrO2", "IrCl3"],
  Pt: ["PtCl2", "Pt"],
  Au: ["HAuCl4", "Au"],
  Ag: ["AgNO3", "Ag"],
  Cd: ["CdO", "CdCl2"],
  Re: ["Re2O7", "Re"],
  Os: ["OsO4", "Os"],
  Ta: ["Ta2O5", "Ta"],
  Sc: ["Sc2O3"],
  Th: ["ThO2"],
  U: ["UO2", "U3O8"],
  Ce: ["CeO2", "Ce2O3"],
  Pr: ["Pr6O11", "PrCl3"],
  Nd: ["Nd2O3", "NdCl3"],
  Sm: ["Sm2O3"],
  Eu: ["Eu2O3"],
  Gd: ["Gd2O3"],
  Dy: ["Dy2O3"],
  Ho: ["Ho2O3"],
  Er: ["Er2O3"],
  Yb: ["Yb2O3"],
  Lu: ["Lu2O3"],
};

function getPrecursorsForElement(el: string): string[] {
  return COMMON_PRECURSORS[el] || [`${el} (elemental)`];
}

function selectPrecursors(elements: string[], method: string): string[] {
  const precursors: string[] = [];
  for (const el of elements) {
    const options = getPrecursorsForElement(el);
    if (method === "solid-state" || method === "ball-milling") {
      const oxide = options.find(p => p.includes("O") && !p.includes("NO3") && !p.includes("Cl"));
      precursors.push(oxide || options[0]);
    } else if (method === "arc-melting") {
      const elemental = options.find(p => p === el || p.includes("elemental"));
      precursors.push(elemental || options[0]);
    } else if (method === "CVD" || method === "sputtering") {
      const halide = options.find(p => p.includes("Cl"));
      precursors.push(halide || options[0]);
    } else if (method === "high-pressure") {
      precursors.push(options[0]);
    } else {
      precursors.push(options[0]);
    }
  }
  return precursors;
}

function computeMiedemaFormationEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return 0;

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }

  let deltaH = 0;
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const dA = ELEMENTAL_DATA[elements[i]];
      const dB = ELEMENTAL_DATA[elements[j]];
      if (!dA || !dB) continue;

      const phiA = dA.miedemaPhiStar;
      const phiB = dB.miedemaPhiStar;
      const nwsA = dA.miedemaNws13;
      const nwsB = dB.miedemaNws13;
      const vA = dA.miedemaV23;
      const vB = dB.miedemaV23;

      if (phiA == null || phiB == null || nwsA == null || nwsB == null || vA == null || vB == null) continue;

      const deltaPhi = phiA - phiB;
      const deltaNws = nwsA - nwsB;
      const nwsAvg = (nwsA + nwsB) / 2;
      const fAB = 2 * fractions[elements[i]] * fractions[elements[j]];
      const vAvg = (vA * fractions[elements[i]] + vB * fractions[elements[j]]) / (fractions[elements[i]] + fractions[elements[j]]);
      const interfaceEnergy = -14.1 * deltaPhi * deltaPhi + 9.4 * deltaNws * deltaNws;
      deltaH += fAB * vAvg * interfaceEnergy / (nwsAvg * nwsAvg);
    }
  }

  return deltaH / totalAtoms;
}

function computeGibbsFreeEnergy(formula: string, temperature: number): number {
  const dH = computeMiedemaFormationEnergy(formula);
  const elements = parseFormulaElements(formula);
  const nElements = elements.length;
  const configEntropy = nElements > 1 ? 8.314e-3 * Math.log(nElements) * nElements : 0;
  const dG = dH - temperature * configEntropy * 1e-3;
  return dG;
}

function computeKineticBarrier(elements: string[], temperature: number): number {
  let maxMp = 0;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp !== null && mp > maxMp) maxMp = mp;
  }
  const tammTemperature = maxMp > 0 ? 0.57 * maxMp : 1200;
  const barrier = Math.max(0.1, (tammTemperature - temperature) / tammTemperature);
  return Math.max(0, Math.min(3.0, barrier * 2.0));
}

function computeArrheniusRate(barrierEv: number, temperature: number): number {
  const kB = 8.617e-5;
  if (temperature <= 0) return 0;
  const exponent = -barrierEv / (kB * temperature);
  return Math.exp(Math.max(-50, Math.min(50, exponent)));
}

function computeMetastableQuenchFeasibility(formula: string, pressure: number): number {
  const elements = parseFormulaElements(formula);
  const hasH = elements.includes("H");
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hFrac = (counts["H"] || 0) / totalAtoms;

  let feasibility = 0.7;

  if (hasH && hFrac > 0.5) {
    feasibility -= 0.3;
    if (pressure > 100) feasibility += 0.15;
  }

  if (pressure > 50) {
    feasibility -= 0.1;
    if (pressure > 200) feasibility -= 0.2;
  }

  const dH = computeMiedemaFormationEnergy(formula);
  if (dH > 0.5) feasibility -= 0.2;
  if (dH < -0.5) feasibility += 0.1;

  return Math.max(0, Math.min(1, feasibility));
}

function computeThermodynamicScoring(formula: string, steps: ReactionStep[]): ThermodynamicScoring {
  const elements = parseFormulaElements(formula);
  const maxTemp = Math.max(...steps.map(s => s.temperature), 300);
  const maxPressure = Math.max(...steps.map(s => s.pressure), 0);

  const gibbs = computeGibbsFreeEnergy(formula, maxTemp);
  const barrier = computeKineticBarrier(elements, maxTemp);
  const rate = computeArrheniusRate(barrier, maxTemp);
  const quench = computeMetastableQuenchFeasibility(formula, maxPressure);

  const gibbsFactor = gibbs < 0 ? 1.0 : Math.max(0, 1.0 - gibbs * 0.5);
  const barrierFactor = Math.max(0, 1.0 - barrier * 0.3);
  const rateFactor = Math.min(1.0, rate * 10);
  const overallFeasibility = Math.max(0, Math.min(1, (gibbsFactor * 0.35 + barrierFactor * 0.25 + rateFactor * 0.2 + quench * 0.2)));

  return {
    gibbsFreeEnergy: Number(gibbs.toFixed(4)),
    kineticBarrier: Number(barrier.toFixed(4)),
    arrheniusRate: Number(rate.toFixed(6)),
    metastableQuenchFeasibility: Number(quench.toFixed(4)),
    overallFeasibility: Number(overallFeasibility.toFixed(4)),
  };
}

function generateSolidStateRoute(formula: string, elements: string[], counts: Record<string, number>): SynthesisRoute {
  const precursors = selectPrecursors(elements, "solid-state");
  let maxMp = 0;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp !== null && mp > maxMp) maxMp = mp;
  }
  const sinterTemp = Math.round(Math.max(800, Math.min(1600, 0.57 * maxMp)));
  const calcineTemp = Math.round(sinterTemp * 0.7);

  const steps: ReactionStep[] = [
    {
      stepNumber: 1,
      reactants: precursors,
      products: ["mixed powder"],
      temperature: 25,
      pressure: 0,
      atmosphere: "air",
      reactionType: "mixing",
      duration: "4 hours",
      notes: "Ball mill precursor powders with zirconia media at 300 rpm",
    },
    {
      stepNumber: 2,
      reactants: ["mixed powder"],
      products: ["calcined powder"],
      temperature: calcineTemp,
      pressure: 0,
      atmosphere: elements.includes("O") ? "flowing O2" : "Ar",
      reactionType: "calcination",
      duration: "12 hours",
      notes: `Ramp at 5 C/min to ${calcineTemp} C, hold 12h`,
    },
    {
      stepNumber: 3,
      reactants: ["calcined powder"],
      products: ["reground powder"],
      temperature: 25,
      pressure: 0,
      atmosphere: "air",
      reactionType: "grinding",
      duration: "2 hours",
      notes: "Regrind calcined powder in agate mortar",
    },
    {
      stepNumber: 4,
      reactants: ["reground powder"],
      products: ["pellet"],
      temperature: 25,
      pressure: 200,
      atmosphere: "air",
      reactionType: "pressing",
      duration: "30 minutes",
      notes: "Uniaxial press at 200 MPa into 13mm pellet",
    },
    {
      stepNumber: 5,
      reactants: ["pellet"],
      products: [formula],
      temperature: sinterTemp,
      pressure: 0,
      atmosphere: elements.includes("O") ? "flowing O2 at 50 mL/min" : "Ar/5% H2",
      reactionType: "sintering",
      duration: "24 hours",
      notes: `Ramp at 3 C/min to ${sinterTemp} C, hold 24h, furnace cool`,
    },
  ];

  const thermo = computeThermodynamicScoring(formula, steps);
  const equipment = ["Planetary ball mill", "Box furnace", "Uniaxial press", "Agate mortar", "Tube furnace"];

  return {
    routeId: `ss-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    routeName: "Solid-State Reaction",
    method: "solid-state",
    steps,
    precursors,
    thermodynamics: thermo,
    totalDuration: `~48 hours (including cooling)`,
    maxTemperature: sinterTemp,
    maxPressure: 200,
    difficulty: sinterTemp > 1400 ? "hard" : sinterTemp > 1000 ? "moderate" : "easy",
    equipment,
    feasibilityScore: thermo.overallFeasibility,
    notes: `Standard solid-state synthesis at ${sinterTemp} C`,
  };
}

function generateArcMeltingRoute(formula: string, elements: string[]): SynthesisRoute {
  const precursors = selectPrecursors(elements, "arc-melting");

  const steps: ReactionStep[] = [
    {
      stepNumber: 1,
      reactants: precursors,
      products: ["weighed elements"],
      temperature: 25,
      pressure: 0,
      atmosphere: "argon glovebox",
      reactionType: "preparation",
      duration: "1 hour",
      notes: "Weigh stoichiometric amounts in argon glovebox",
    },
    {
      stepNumber: 2,
      reactants: ["weighed elements"],
      products: ["arc-melted button"],
      temperature: 3000,
      pressure: 0,
      atmosphere: "ultra-high purity Ar",
      reactionType: "arc-melting",
      duration: "5 minutes per melt, 4 flips",
      notes: "Arc-melt on water-cooled Cu hearth, flip and remelt 4x for homogeneity",
    },
    {
      stepNumber: 3,
      reactants: ["arc-melted button"],
      products: [formula],
      temperature: 900,
      pressure: 0,
      atmosphere: "sealed quartz tube under Ar",
      reactionType: "annealing",
      duration: "7 days",
      notes: "Wrap in Ta foil, seal in evacuated quartz tube, anneal at 900 C for 7 days",
    },
  ];

  const thermo = computeThermodynamicScoring(formula, steps);
  const equipment = ["Arc furnace", "Water-cooled Cu hearth", "Ar glovebox", "Analytical balance", "Quartz tube sealer"];

  return {
    routeId: `am-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    routeName: "Arc Melting",
    method: "arc-melting",
    steps,
    precursors,
    thermodynamics: thermo,
    totalDuration: "~8 days",
    maxTemperature: 3000,
    maxPressure: 0,
    difficulty: "moderate",
    equipment,
    feasibilityScore: thermo.overallFeasibility * 0.95,
    notes: "Arc melting with annealing for intermetallics",
  };
}

function generateHighPressureRoute(formula: string, elements: string[], targetPressureGpa: number): SynthesisRoute {
  const precursors = selectPrecursors(elements, "high-pressure");
  const pressure = Math.max(1, targetPressureGpa);
  const hasH = elements.includes("H");

  const hpTemp = hasH ? Math.round(Math.min(2000, 300 + pressure * 5)) : Math.round(Math.min(2500, 800 + pressure * 3));

  const steps: ReactionStep[] = [
    {
      stepNumber: 1,
      reactants: precursors,
      products: ["sample assembly"],
      temperature: 25,
      pressure: 0,
      atmosphere: hasH ? "Ar glovebox" : "air",
      reactionType: "preparation",
      duration: "2 hours",
      notes: hasH
        ? "Load precursors with hydrogen source (ammonia borane or LiH) into diamond anvil cell gasket"
        : "Prepare sample in Re or Ta gasket for DAC",
    },
    {
      stepNumber: 2,
      reactants: ["sample assembly"],
      products: ["compressed sample"],
      temperature: 25,
      pressure,
      atmosphere: "compressed Ne or He medium",
      reactionType: "compression",
      duration: "4 hours",
      notes: `Compress to ${pressure} GPa in diamond anvil cell with Ne pressure medium`,
    },
    {
      stepNumber: 3,
      reactants: ["compressed sample"],
      products: [formula],
      temperature: hpTemp,
      pressure,
      atmosphere: "under pressure",
      reactionType: "laser-heating",
      duration: "30 minutes",
      notes: `Laser-heat to ${hpTemp} K at ${pressure} GPa using YAG or CO2 laser`,
    },
  ];

  if (pressure > 50) {
    steps.push({
      stepNumber: 4,
      reactants: [formula],
      products: [`${formula} (quenched)`],
      temperature: 25,
      pressure: 0,
      atmosphere: "ambient",
      reactionType: "pressure-quench",
      duration: "2 hours",
      notes: `Decompress from ${pressure} GPa to ambient over 2h; assess metastable retention`,
    });
  }

  const thermo = computeThermodynamicScoring(formula, steps);
  const equipment = [
    "Diamond anvil cell (DAC)",
    "YAG/CO2 laser heating system",
    "Ruby fluorescence pressure gauge",
    "X-ray diffraction (synchrotron)",
  ];
  if (pressure > 100) equipment.push("Multi-megabar DAC");

  const difficulty = pressure > 200 ? "extreme" : pressure > 100 ? "hard" : pressure > 20 ? "moderate" : "moderate";

  return {
    routeId: `hp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    routeName: `High-Pressure Synthesis (${pressure} GPa)`,
    method: "high-pressure",
    steps,
    precursors,
    thermodynamics: thermo,
    totalDuration: pressure > 50 ? "~8 hours" : "~6 hours",
    maxTemperature: hpTemp,
    maxPressure: pressure,
    difficulty,
    equipment,
    feasibilityScore: thermo.overallFeasibility * (pressure > 200 ? 0.5 : pressure > 100 ? 0.7 : 0.9),
    notes: `High-pressure synthesis at ${pressure} GPa, ${hpTemp} K`,
  };
}

function generateBallMillingRoute(formula: string, elements: string[]): SynthesisRoute {
  const precursors = selectPrecursors(elements, "ball-milling");

  const steps: ReactionStep[] = [
    {
      stepNumber: 1,
      reactants: precursors,
      products: ["milled powder"],
      temperature: 25,
      pressure: 0,
      atmosphere: "Ar-filled milling jar",
      reactionType: "ball-milling",
      duration: "20 hours",
      notes: "High-energy ball milling at 400 rpm with WC balls, ball:powder ratio 10:1",
    },
    {
      stepNumber: 2,
      reactants: ["milled powder"],
      products: ["pellet"],
      temperature: 25,
      pressure: 300,
      atmosphere: "air",
      reactionType: "pressing",
      duration: "30 minutes",
      notes: "Cold isostatic press at 300 MPa",
    },
    {
      stepNumber: 3,
      reactants: ["pellet"],
      products: [formula],
      temperature: 800,
      pressure: 0,
      atmosphere: "Ar/5% H2",
      reactionType: "sintering",
      duration: "6 hours",
      notes: "Sinter at 800 C for 6h under reducing atmosphere",
    },
  ];

  const thermo = computeThermodynamicScoring(formula, steps);
  const equipment = ["Planetary ball mill", "WC balls and jar", "Cold isostatic press", "Tube furnace"];

  return {
    routeId: `bm-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    routeName: "Mechanochemical Ball Milling",
    method: "ball-milling",
    steps,
    precursors,
    thermodynamics: thermo,
    totalDuration: "~28 hours",
    maxTemperature: 800,
    maxPressure: 300,
    difficulty: "moderate",
    equipment,
    feasibilityScore: thermo.overallFeasibility * 0.9,
    notes: "Mechanochemical synthesis with post-sintering",
  };
}

function generateCVDRoute(formula: string, elements: string[]): SynthesisRoute {
  const precursors = selectPrecursors(elements, "CVD");

  const steps: ReactionStep[] = [
    {
      stepNumber: 1,
      reactants: ["substrate (SrTiO3 or MgO)"],
      products: ["cleaned substrate"],
      temperature: 25,
      pressure: 0,
      atmosphere: "clean room",
      reactionType: "preparation",
      duration: "1 hour",
      notes: "Ultrasonic clean in acetone, IPA, DI water; O2 plasma clean 5 min",
    },
    {
      stepNumber: 2,
      reactants: precursors,
      products: ["vapor precursors"],
      temperature: 200,
      pressure: 0.001,
      atmosphere: "carrier gas (Ar/O2)",
      reactionType: "vaporization",
      duration: "continuous",
      notes: "Vaporize precursors in separate bubblers with controlled flow rates",
    },
    {
      stepNumber: 3,
      reactants: ["vapor precursors", "cleaned substrate"],
      products: [`${formula} thin film`],
      temperature: 700,
      pressure: 0.01,
      atmosphere: "O2/Ar mixed flow",
      reactionType: "CVD",
      duration: "2 hours",
      notes: "Deposit at 700 C, 10 mTorr total pressure, growth rate ~1 nm/min",
    },
    {
      stepNumber: 4,
      reactants: [`${formula} thin film`],
      products: [formula],
      temperature: 500,
      pressure: 760,
      atmosphere: "flowing O2",
      reactionType: "post-annealing",
      duration: "1 hour",
      notes: "Post-anneal in 1 atm O2 at 500 C for 1h to optimize oxygen stoichiometry",
    },
  ];

  const thermo = computeThermodynamicScoring(formula, steps);
  const equipment = ["MOCVD reactor", "Mass flow controllers", "Substrate heater", "Vacuum pump", "Precursor bubblers"];

  return {
    routeId: `cvd-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    routeName: "Chemical Vapor Deposition",
    method: "CVD",
    steps,
    precursors,
    thermodynamics: thermo,
    totalDuration: "~5 hours",
    maxTemperature: 700,
    maxPressure: 760,
    difficulty: "hard",
    equipment,
    feasibilityScore: thermo.overallFeasibility * 0.8,
    notes: "Thin film synthesis via CVD",
  };
}

function generateSputteringRoute(formula: string, elements: string[]): SynthesisRoute {
  const precursors = elements.map(el => `${el} target (99.99%)`);

  const steps: ReactionStep[] = [
    {
      stepNumber: 1,
      reactants: ["substrate"],
      products: ["cleaned substrate"],
      temperature: 25,
      pressure: 0,
      atmosphere: "clean room",
      reactionType: "preparation",
      duration: "30 minutes",
      notes: "Clean substrate ultrasonically, load into sputtering chamber",
    },
    {
      stepNumber: 2,
      reactants: precursors,
      products: [`${formula} film`],
      temperature: 500,
      pressure: 0.005,
      atmosphere: "Ar + reactive gas",
      reactionType: "sputtering",
      duration: "3 hours",
      notes: "Co-sputter from elemental targets, substrate at 500 C, 5 mTorr Ar",
    },
    {
      stepNumber: 3,
      reactants: [`${formula} film`],
      products: [formula],
      temperature: 600,
      pressure: 0,
      atmosphere: elements.includes("O") ? "O2" : "Ar",
      reactionType: "annealing",
      duration: "2 hours",
      notes: "Post-deposition anneal at 600 C for 2h",
    },
  ];

  const thermo = computeThermodynamicScoring(formula, steps);
  const equipment = ["Magnetron sputtering system", "RF/DC power supplies", "Substrate heater", "Turbo pump"];

  return {
    routeId: `sp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    routeName: "Magnetron Sputtering",
    method: "sputtering",
    steps,
    precursors,
    thermodynamics: thermo,
    totalDuration: "~6 hours",
    maxTemperature: 600,
    maxPressure: 0.005,
    difficulty: "hard",
    equipment,
    feasibilityScore: thermo.overallFeasibility * 0.75,
    notes: "Thin film synthesis via magnetron sputtering",
  };
}

function selectRoutesForFamily(formula: string, family: string, elements: string[], counts: Record<string, number>): SynthesisRoute[] {
  const routes: SynthesisRoute[] = [];
  const hasH = elements.includes("H");
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hFrac = (counts["H"] || 0) / totalAtoms;

  routes.push(generateSolidStateRoute(formula, elements, counts));

  if (family === "Intermetallics" || family === "Alloys" || family === "Borides" || family === "Silicides") {
    routes.push(generateArcMeltingRoute(formula, elements));
  }

  if (hasH && hFrac > 0.3) {
    routes.push(generateHighPressureRoute(formula, elements, hFrac > 0.6 ? 150 : 50));
  } else if (family === "Hydrides") {
    routes.push(generateHighPressureRoute(formula, elements, 100));
  }

  routes.push(generateBallMillingRoute(formula, elements));

  if (family === "Cuprates" || family === "Pnictides" || family === "Chalcogenides" || family === "Oxides") {
    routes.push(generateCVDRoute(formula, elements));
  }

  if (elements.length <= 4) {
    routes.push(generateSputteringRoute(formula, elements));
  }

  if (!hasH && elements.length >= 3) {
    const pressure = 5 + Math.random() * 15;
    routes.push(generateHighPressureRoute(formula, elements, pressure));
  }

  return routes;
}

export function computeSynthesisPathway(formula: string, analogyRoutes?: SynthesisRoute[]): SynthesisPathwayResult {
  if (!analogyRoutes || analogyRoutes.length === 0) {
    const cached = pathwayCache.get(formula);
    if (cached) return cached;
  }

  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const family = classifyFamily(formula);

  const routes = selectRoutesForFamily(formula, family, elements, counts);

  if (analogyRoutes && analogyRoutes.length > 0) {
    routes.push(...analogyRoutes);
  }

  routes.sort((a, b) => b.feasibilityScore - a.feasibilityScore);

  const bestRoute = routes.length > 0 ? routes[0] : null;

  const bestMethod = bestRoute ? bestRoute.method : "unknown";
  const bestFeasibility = bestRoute ? bestRoute.feasibilityScore : 0;
  const analogyCount = (analogyRoutes || []).length;
  const summary = bestRoute
    ? `Best route: ${bestRoute.routeName} (feasibility ${(bestFeasibility * 100).toFixed(1)}%), ${bestRoute.steps.length} steps, max ${bestRoute.maxTemperature} C, ${bestRoute.totalDuration}${analogyCount > 0 ? ` (+${analogyCount} analogy-transferred)` : ""}`
    : `No viable synthesis route found for ${formula}`;

  const result: SynthesisPathwayResult = {
    formula,
    family,
    routes,
    bestRoute,
    summary,
  };

  if (!analogyRoutes || analogyRoutes.length === 0) {
    pathwayCache.set(formula, result);
  }

  totalPathwaysComputed++;
  totalRoutesGenerated += routes.length;
  methodCounts[bestMethod] = (methodCounts[bestMethod] || 0) + 1;
  familyCounts[family] = (familyCounts[family] || 0) + 1;
  if (bestFeasibility > 0) feasibilityValues.push(bestFeasibility);

  if (pathwayCache.size > 500) {
    const keys = Array.from(pathwayCache.keys());
    for (let i = 0; i < 100; i++) {
      pathwayCache.delete(keys[i]);
    }
  }

  return result;
}

export function getSynthesisPathwayStats(): SynthesisPathwayStats {
  const avgFeasibility = feasibilityValues.length > 0
    ? feasibilityValues.reduce((a, b) => a + b, 0) / feasibilityValues.length
    : 0;

  return {
    totalPathwaysComputed,
    totalRoutesGenerated,
    avgFeasibility: Number(avgFeasibility.toFixed(4)),
    methodBreakdown: { ...methodCounts },
    familyBreakdown: { ...familyCounts },
  };
}

export function triggerSynthesisPathwayForCandidate(
  formula: string,
  predictedTc: number,
  verificationStage: number,
): SynthesisPathwayResult | null {
  if (verificationStage < 4 && predictedTc < 100) {
    return null;
  }
  return computeSynthesisPathway(formula);
}
