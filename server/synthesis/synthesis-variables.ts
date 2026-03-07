export interface SynthesisVariable {
  parameter: string;
  category: SynthesisCategory;
  units: string;
  grid: number[];
  importance: string;
  applicableMaterials: string[];
}

export interface DiscreteVariable {
  parameter: string;
  category: SynthesisCategory;
  units: string;
  options: string[];
  importance: string;
  applicableMaterials: string[];
}

export type SynthesisCategory =
  | "thermal"
  | "pressure"
  | "electrical"
  | "magnetic"
  | "mechanical"
  | "thermal-cycling"
  | "atmosphere"
  | "structural"
  | "doping"
  | "interface-layer";

export interface SynthesisConditionSet {
  synthesisTemperature: number;
  annealTemperature: number;
  annealTime: number;
  coolingRate: number;
  synthesisPressure: number;
  pressureRampRate: number;
  currentDensity: number;
  electricField: number;
  magneticField: number;
  mechanicalStress: number;
  latticeStrain: number;
  thermalCycles: number;
  cycleTemperatureRange: string;
  oxygenPartialPressure: number;
  hydrogenPressure: number;
  gasEnvironment: string;
  grainSize: number;
  defectDensity: number;
  dopantConcentration: number;
  vacancyFraction: number;
  layerThickness: number;
  interlayerSpacing: number;
  feasibilityScore: number;
  method: string;
  notes: string;
}

export const THERMAL_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "synthesisTemperature",
    category: "thermal",
    units: "K",
    grid: [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1700, 2000, 2500],
    importance: "Controls crystal phase formation, diffusion rates, defect density",
    applicableMaterials: ["all"],
  },
  {
    parameter: "annealTemperature",
    category: "thermal",
    units: "K",
    grid: [300, 500, 700, 900, 1100, 1300, 1500],
    importance: "Modifies grain size, oxygen ordering, lattice strain",
    applicableMaterials: ["cuprates", "oxides", "ceramics", "pnictides"],
  },
  {
    parameter: "annealTime",
    category: "thermal",
    units: "hours",
    grid: [0.5, 1, 2, 5, 10, 24, 72],
    importance: "Allows dopant diffusion and ordering transitions",
    applicableMaterials: ["all"],
  },
  {
    parameter: "coolingRate",
    category: "thermal",
    units: "K/s",
    grid: [0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
    importance: "<1 K/s equilibrium crystals; 1-50 moderate defects; 100-1000 metastable; >5000 amorphous",
    applicableMaterials: ["all"],
  },
];

export const PRESSURE_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "synthesisPressure",
    category: "pressure",
    units: "GPa",
    grid: [0, 0.1, 1, 2, 5, 10, 20, 50, 100, 150, 200, 300],
    importance: "Strongly affects bonding and phonon frequencies. <1 normal; 1-10 HP ceramics; 50-200 hydride SC",
    applicableMaterials: ["all"],
  },
  {
    parameter: "pressureRampRate",
    category: "pressure",
    units: "GPa/min",
    grid: [0.1, 0.5, 1, 2, 5, 10, 20],
    importance: "Rapid compression can trap metastable phases",
    applicableMaterials: ["hydrides", "high-pressure"],
  },
];

export const ELECTRICAL_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "currentDensity",
    category: "electrical",
    units: "A/cm2",
    grid: [0, 1, 5, 10, 20, 50, 100, 200],
    importance: "Spark plasma sintering, grain boundary alignment, dopant migration",
    applicableMaterials: ["ceramics", "intermetallics", "all"],
  },
  {
    parameter: "electricField",
    category: "electrical",
    units: "V/cm",
    grid: [0, 10, 50, 100, 500, 1000, 5000],
    importance: "Induces ion migration, changes electronic ordering",
    applicableMaterials: ["oxides", "cuprates", "ferroelectrics"],
  },
];

export const MAGNETIC_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "magneticField",
    category: "magnetic",
    units: "Tesla",
    grid: [0, 0.1, 0.5, 1, 2, 5, 10, 15, 20],
    importance: "Grain alignment, spin ordering during crystal growth",
    applicableMaterials: ["cuprates", "pnictides", "heavy-fermion", "magnetic"],
  },
];

export const MECHANICAL_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "mechanicalStress",
    category: "mechanical",
    units: "GPa",
    grid: [0, 0.1, 0.5, 1, 2, 5, 10, 20],
    importance: "Distorts lattice, modifies electron bands, can enhance Tc",
    applicableMaterials: ["all"],
  },
  {
    parameter: "latticeStrain",
    category: "mechanical",
    units: "%",
    grid: [0, 0.1, 0.5, 1, 2, 5],
    importance: "Strain-enhanced Tc in epitaxial films and heterostructures",
    applicableMaterials: ["thin-films", "cuprates", "pnictides"],
  },
];

export const THERMAL_CYCLING_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "thermalCycles",
    category: "thermal-cycling",
    units: "count",
    grid: [0, 1, 2, 5, 10, 20, 50],
    importance: "Causes phase transitions, strain accumulation, defect annealing",
    applicableMaterials: ["cuprates", "ceramics", "intermetallics"],
  },
];

export const THERMAL_CYCLING_RANGES = [
  "300-800 K",
  "300-1000 K",
  "500-1200 K",
];

export const ATMOSPHERE_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "oxygenPartialPressure",
    category: "atmosphere",
    units: "atm",
    grid: [0, 1e-6, 1e-4, 1e-2, 0.1, 1],
    importance: "Critical for cuprates and oxide superconductors",
    applicableMaterials: ["cuprates", "oxides"],
  },
  {
    parameter: "hydrogenPressure",
    category: "atmosphere",
    units: "GPa",
    grid: [0, 0.1, 1, 5, 10, 50, 100],
    importance: "Essential for hydride superconductors",
    applicableMaterials: ["hydrides", "superhydrides"],
  },
];

export const GAS_ENVIRONMENTS = [
  "vacuum",
  "argon",
  "nitrogen",
  "oxygen",
  "hydrogen",
  "ammonia",
];

export const STRUCTURAL_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "grainSize",
    category: "structural",
    units: "nm",
    grid: [10, 50, 100, 500, 1000, 5000],
    importance: "Smaller grains increase boundary density, affect pinning and critical current",
    applicableMaterials: ["all"],
  },
  {
    parameter: "defectDensity",
    category: "structural",
    units: "defects/cm3",
    grid: [1e14, 1e16, 1e18, 1e20],
    importance: "Scatter electrons, alter pairing interactions, create pinning centers",
    applicableMaterials: ["all"],
  },
];

export const DOPING_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "dopantConcentration",
    category: "doping",
    units: "fraction",
    grid: [0, 0.01, 0.02, 0.05, 0.1, 0.2],
    importance: "Controls carrier density, optimal doping for max Tc in cuprates/pnictides",
    applicableMaterials: ["cuprates", "pnictides", "oxides", "all"],
  },
  {
    parameter: "vacancyFraction",
    category: "doping",
    units: "fraction",
    grid: [0, 0.01, 0.02, 0.05, 0.1],
    importance: "Vacancies influence pairing symmetry and DOS at Fermi level",
    applicableMaterials: ["cuprates", "pnictides", "oxides"],
  },
];

export const INTERFACE_VARIABLES: SynthesisVariable[] = [
  {
    parameter: "layerThickness",
    category: "interface-layer",
    units: "nm",
    grid: [1, 2, 5, 10, 20],
    importance: "Controls 2D confinement, interface effects, and superlattice coupling",
    applicableMaterials: ["cuprates", "pnictides", "carbides", "nitrides", "thin-films"],
  },
  {
    parameter: "interlayerSpacing",
    category: "interface-layer",
    units: "angstrom",
    grid: [3, 4, 5, 6, 7],
    importance: "Affects interlayer coupling, c-axis transport, and Cooper pair tunneling",
    applicableMaterials: ["cuprates", "pnictides", "layered"],
  },
];

export const ALL_NUMERIC_VARIABLES: SynthesisVariable[] = [
  ...THERMAL_VARIABLES,
  ...PRESSURE_VARIABLES,
  ...ELECTRICAL_VARIABLES,
  ...MAGNETIC_VARIABLES,
  ...MECHANICAL_VARIABLES,
  ...THERMAL_CYCLING_VARIABLES,
  ...ATMOSPHERE_VARIABLES,
  ...STRUCTURAL_VARIABLES,
  ...DOPING_VARIABLES,
  ...INTERFACE_VARIABLES,
];

export function getVariablesByCategory(category: SynthesisCategory): SynthesisVariable[] {
  return ALL_NUMERIC_VARIABLES.filter(v => v.category === category);
}

export function getApplicableVariables(materialClass: string): SynthesisVariable[] {
  const mc = materialClass.toLowerCase();
  return ALL_NUMERIC_VARIABLES.filter(v =>
    v.applicableMaterials.includes("all") ||
    v.applicableMaterials.some(a => mc.includes(a))
  );
}

export function getParameterSpace(): {
  totalVariables: number;
  categories: { name: string; count: number; parameters: string[] }[];
  totalGridPoints: number;
  discreteVariables: { name: string; options: string[] }[];
} {
  const categoryMap = new Map<string, SynthesisVariable[]>();
  for (const v of ALL_NUMERIC_VARIABLES) {
    const existing = categoryMap.get(v.category) || [];
    existing.push(v);
    categoryMap.set(v.category, existing);
  }

  const categories = Array.from(categoryMap.entries()).map(([name, vars]) => ({
    name,
    count: vars.length,
    parameters: vars.map(v => v.parameter),
  }));

  const totalGridPoints = ALL_NUMERIC_VARIABLES.reduce((sum, v) => sum + v.grid.length, 0);

  return {
    totalVariables: ALL_NUMERIC_VARIABLES.length,
    categories,
    totalGridPoints,
    discreteVariables: [
      { name: "gasEnvironment", options: GAS_ENVIRONMENTS },
      { name: "cycleTemperatureRange", options: THERMAL_CYCLING_RANGES },
    ],
  };
}
