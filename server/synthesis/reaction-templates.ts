import { ELEMENTAL_DATA, getMeltingPoint } from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";

export interface ReactionTemplate {
  id: string;
  method: string;
  name: string;
  description: string;
  temperatureRange: [number, number];
  pressureRange: [number, number];
  atmosphere: string[];
  equipment: string[];
  durationRange: [number, number];
  difficulty: "easy" | "moderate" | "hard" | "extreme";
  precursorStrategy: "oxide" | "elemental" | "halide" | "carbonate" | "organometallic" | "hydride" | "mixed";
  applicableFamilies: string[];
  requiredElements?: string[];
  excludedElements?: string[];
  minElements?: number;
  maxElements?: number;
  notes: string;
}

export interface TemplateMatch {
  template: ReactionTemplate;
  score: number;
  familyMatchScore: number;
  elementCompatibilityScore: number;
  thermodynamicScore: number;
  pressureFeasibilityScore: number;
  reasoning: string[];
}

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
  return Object.keys(parseFormulaCounts(formula));
}

const REACTION_TEMPLATES: ReactionTemplate[] = [
  {
    id: "hydride-dac",
    method: "high-pressure",
    name: "Hydride Formation (DAC)",
    description: "Metal + H2 at extreme pressure in diamond anvil cell to form metal hydride",
    temperatureRange: [300, 2500],
    pressureRange: [50, 400],
    atmosphere: ["H2", "NH3BH3 decomposition"],
    equipment: ["Diamond anvil cell (DAC)", "YAG/CO2 laser heating system", "Ruby fluorescence pressure gauge", "Synchrotron XRD beamline", "Cryostat for Tc measurement"],
    durationRange: [2, 12],
    difficulty: "extreme",
    precursorStrategy: "elemental",
    applicableFamilies: ["hydride", "superhydride", "hydrogen-rich"],
    requiredElements: ["H"],
    notes: "High-pressure hydrogen-rich compound synthesis. Requires synchrotron access for in-situ characterization.",
  },
  {
    id: "hydride-moderate-pressure",
    method: "high-pressure",
    name: "Moderate-Pressure Hydride Synthesis",
    description: "Metal + H2 at moderate pressures using multi-anvil press",
    temperatureRange: [400, 1500],
    pressureRange: [1, 50],
    atmosphere: ["H2", "Ar/H2 mix"],
    equipment: ["Multi-anvil press", "Walker module", "H2 gas loading system", "XRD"],
    durationRange: [4, 48],
    difficulty: "hard",
    precursorStrategy: "elemental",
    applicableFamilies: ["hydride", "hydrogen-rich"],
    requiredElements: ["H"],
    notes: "Moderate pressure synthesis for lower-hydrogen hydrides amenable to multi-anvil press.",
  },
  {
    id: "alloy-arc-melting",
    method: "arc-melting",
    name: "Arc Melting Alloy Formation",
    description: "A + B → AB via arc melting on water-cooled Cu hearth under Ar",
    temperatureRange: [2000, 3500],
    pressureRange: [0, 0],
    atmosphere: ["ultra-high purity Ar"],
    equipment: ["Arc furnace", "Water-cooled Cu hearth", "Ar glovebox", "Analytical balance", "Quartz tube sealer"],
    durationRange: [8, 240],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["intermetallic", "A15", "Laves", "Heusler", "boride", "carbide", "silicide", "heavy-fermion"],
    excludedElements: ["H", "O"],
    minElements: 2,
    maxElements: 4,
    notes: "Standard route for intermetallic compounds. Multiple re-melting and annealing improves homogeneity.",
  },
  {
    id: "oxide-solid-state",
    method: "solid-state",
    name: "Solid-State Oxide Synthesis",
    description: "Metal oxides/carbonates → complex oxide via grinding, calcination, sintering",
    temperatureRange: [800, 1600],
    pressureRange: [0, 200],
    atmosphere: ["flowing O2", "air"],
    equipment: ["Planetary ball mill", "Box furnace", "Tube furnace", "Uniaxial press", "Agate mortar"],
    durationRange: [24, 120],
    difficulty: "easy",
    precursorStrategy: "oxide",
    applicableFamilies: ["cuprate", "oxide", "perovskite", "ruthenate", "titanate", "bismuthate", "nickelate"],
    requiredElements: ["O"],
    notes: "Classical ceramic synthesis. Multiple grinding-sintering cycles improve phase purity.",
  },
  {
    id: "solid-state-diffusion",
    method: "solid-state",
    name: "Solid-State Diffusion Reaction",
    description: "A + B + C → ABC via iterative grinding and furnace heating",
    temperatureRange: [600, 1400],
    pressureRange: [0, 300],
    atmosphere: ["Ar", "Ar/5% H2", "vacuum", "N2"],
    equipment: ["Planetary ball mill", "Box furnace", "Uniaxial press", "Tube furnace", "Agate mortar"],
    durationRange: [24, 168],
    difficulty: "easy",
    precursorStrategy: "oxide",
    applicableFamilies: ["intermetallic", "pnictide", "chalcogenide", "borocarbide", "nitride", "silicide", "carbide"],
    minElements: 2,
    notes: "General solid-state method for non-oxide compounds. Atmosphere chosen based on element sensitivity.",
  },
  {
    id: "thin-film-sputtering",
    method: "sputtering",
    name: "Magnetron Sputtering Thin Film",
    description: "Co-sputtering from elemental targets onto heated substrate",
    temperatureRange: [25, 800],
    pressureRange: [0, 0.01],
    atmosphere: ["Ar", "Ar + reactive gas (O2/N2)"],
    equipment: ["Magnetron sputtering system", "RF/DC power supplies", "Substrate heater", "Turbo pump", "Thickness monitor"],
    durationRange: [2, 8],
    difficulty: "hard",
    precursorStrategy: "elemental",
    applicableFamilies: ["cuprate", "nitride", "oxide", "intermetallic", "pnictide", "MgB2-type"],
    maxElements: 5,
    notes: "Thin film route enabling epitaxial growth and strain engineering. Suitable for studying interface effects.",
  },
  {
    id: "thin-film-cvd",
    method: "CVD",
    name: "Chemical Vapor Deposition",
    description: "Metal-organic or halide precursors → thin film via vapor-phase reaction",
    temperatureRange: [200, 900],
    pressureRange: [0, 0.01],
    atmosphere: ["carrier gas (Ar/O2)", "O2/Ar mixed flow"],
    equipment: ["MOCVD reactor", "Mass flow controllers", "Substrate heater", "Vacuum pump", "Precursor bubblers"],
    durationRange: [3, 10],
    difficulty: "hard",
    precursorStrategy: "organometallic",
    applicableFamilies: ["cuprate", "oxide", "nitride", "perovskite"],
    requiredElements: ["O"],
    maxElements: 5,
    notes: "MOCVD provides excellent thickness control and conformal coverage. Requires volatile precursors.",
  },
  {
    id: "thin-film-pld",
    method: "PLD",
    name: "Pulsed Laser Deposition",
    description: "Ablation of polycrystalline target with UV laser onto heated substrate",
    temperatureRange: [400, 850],
    pressureRange: [0, 0.001],
    atmosphere: ["O2 background (0.1-1 mbar)", "Ar background"],
    equipment: ["Excimer laser (KrF/ArF)", "Vacuum chamber", "Substrate heater", "RHEED", "Target carousel"],
    durationRange: [2, 6],
    difficulty: "hard",
    precursorStrategy: "oxide",
    applicableFamilies: ["cuprate", "oxide", "perovskite", "nickelate", "ruthenate", "bismuthate"],
    requiredElements: ["O"],
    notes: "Excellent stoichiometry transfer from target to film. Ideal for complex oxide heterostructures.",
  },
  {
    id: "high-pressure-general",
    method: "high-pressure",
    name: "High-Pressure Synthesis (Multi-Anvil/DAC)",
    description: "Synthesis under high pressure for metastable or high-density phases",
    temperatureRange: [500, 2500],
    pressureRange: [5, 200],
    atmosphere: ["compressed Ne or He medium", "sealed capsule"],
    equipment: ["Multi-anvil press", "Diamond anvil cell", "Walker module", "Synchrotron XRD"],
    durationRange: [2, 24],
    difficulty: "hard",
    precursorStrategy: "mixed",
    applicableFamilies: ["hydride", "boride", "nitride", "carbide", "oxide", "intermetallic", "clathrate"],
    notes: "General high-pressure route. Multi-anvil up to ~25 GPa, DAC for higher pressures.",
  },
  {
    id: "ball-milling",
    method: "ball-milling",
    name: "Mechanochemical Ball Milling",
    description: "High-energy milling for mechanical alloying and amorphization",
    temperatureRange: [25, 800],
    pressureRange: [0, 300],
    atmosphere: ["Ar-filled milling jar"],
    equipment: ["Planetary ball mill", "WC or steel balls and jar", "Cold isostatic press", "Tube furnace", "Ar glovebox"],
    durationRange: [8, 72],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["intermetallic", "boride", "silicide", "carbide", "MgB2-type", "A15", "Chevrel"],
    excludedElements: ["H"],
    notes: "Mechanical alloying produces nanocrystalline and metastable phases. Post-sintering often required.",
  },
  {
    id: "sol-gel-cuprate",
    method: "sol-gel",
    name: "Sol-Gel Synthesis",
    description: "Metal nitrate/acetate solution → gel → calcination → phase-pure complex oxide",
    temperatureRange: [400, 1100],
    pressureRange: [0, 0],
    atmosphere: ["air", "flowing O2"],
    equipment: ["Hot plate with stirrer", "Drying oven", "Box furnace", "pH meter", "Beakers and flasks"],
    durationRange: [24, 96],
    difficulty: "moderate",
    precursorStrategy: "mixed",
    applicableFamilies: ["cuprate", "oxide", "perovskite", "bismuthate", "titanate", "ruthenate"],
    requiredElements: ["O"],
    notes: "Sol-gel produces nanoscale mixing for better phase purity. Citrate or Pechini method common for cuprates.",
  },
  {
    id: "flux-growth",
    method: "flux-growth",
    name: "Flux Growth (Single Crystal)",
    description: "Growth from high-temperature self-flux or salt flux for single crystals",
    temperatureRange: [700, 1400],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "flowing O2", "sealed quartz tube"],
    equipment: ["Box furnace with programmable controller", "Alumina crucibles", "Pt crucibles", "Centrifuge", "Wire saw"],
    durationRange: [48, 720],
    difficulty: "moderate",
    precursorStrategy: "oxide",
    applicableFamilies: ["cuprate", "pnictide", "chalcogenide", "heavy-fermion", "oxide", "intermetallic", "boride"],
    minElements: 2,
    notes: "Self-flux (e.g., Sn, In, Pb, Bi) or salt flux. Slow cooling yields mm-scale single crystals.",
  },
  {
    id: "czochralski",
    method: "Czochralski",
    name: "Czochralski Crystal Growth",
    description: "Pull single crystal from melt for congruently melting compounds",
    temperatureRange: [1000, 2500],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "N2"],
    equipment: ["Czochralski puller", "Ir/Pt crucible", "Induction heating", "Seed rod"],
    durationRange: [24, 168],
    difficulty: "hard",
    precursorStrategy: "elemental",
    applicableFamilies: ["intermetallic", "oxide", "boride"],
    minElements: 2,
    maxElements: 3,
    notes: "Produces large single crystals of congruently melting phases. Limited to certain compositions.",
  },
  {
    id: "bridgman",
    method: "Bridgman",
    name: "Bridgman Crystal Growth",
    description: "Directional solidification in sealed ampoule",
    temperatureRange: [400, 1800],
    pressureRange: [0, 0],
    atmosphere: ["sealed evacuated quartz ampoule"],
    equipment: ["Bridgman furnace", "Quartz ampoules", "Vacuum sealer", "Glovebox"],
    durationRange: [48, 336],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["chalcogenide", "pnictide", "intermetallic", "heavy-fermion", "topological"],
    minElements: 2,
    maxElements: 4,
    notes: "Sealed tube directional growth for air-sensitive or volatile compositions.",
  },
  {
    id: "vapor-transport",
    method: "vapor-transport",
    name: "Chemical Vapor Transport",
    description: "Crystal growth via transport agent in sealed tube along temperature gradient",
    temperatureRange: [500, 1100],
    pressureRange: [0, 0],
    atmosphere: ["sealed tube with transport agent (I2, TeCl4)"],
    equipment: ["Two-zone tube furnace", "Quartz ampoules", "Vacuum sealer", "Glovebox"],
    durationRange: [168, 504],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["chalcogenide", "pnictide", "topological", "TMD"],
    minElements: 2,
    maxElements: 3,
    notes: "Produces high-quality single crystals of layered materials. Growth over days to weeks.",
  },
  {
    id: "mbe",
    method: "MBE",
    name: "Molecular Beam Epitaxy",
    description: "Atomic-layer-precision film growth from elemental sources in ultra-high vacuum",
    temperatureRange: [200, 800],
    pressureRange: [0, 0],
    atmosphere: ["ultra-high vacuum (< 10^-9 Torr)"],
    equipment: ["MBE system", "Knudsen cells", "RHEED", "Mass spectrometer", "Substrate heater"],
    durationRange: [4, 24],
    difficulty: "extreme",
    precursorStrategy: "elemental",
    applicableFamilies: ["cuprate", "oxide", "nickelate", "pnictide", "topological", "perovskite"],
    maxElements: 5,
    notes: "Highest quality epitaxial films. Layer-by-layer growth monitored by RHEED oscillations.",
  },
  {
    id: "hydrothermal",
    method: "hydrothermal",
    name: "Hydrothermal Synthesis",
    description: "Aqueous solution reaction in sealed autoclave at elevated temperature/pressure",
    temperatureRange: [100, 350],
    pressureRange: [0.1, 20],
    atmosphere: ["aqueous solution", "mineralizer solution"],
    equipment: ["Teflon-lined autoclave", "Drying oven", "pH meter", "Centrifuge"],
    durationRange: [12, 168],
    difficulty: "easy",
    precursorStrategy: "mixed",
    applicableFamilies: ["oxide", "chalcogenide", "hydroxide"],
    notes: "Low-temperature route for metastable phases and nanostructures. Limited to hydrolytically stable products.",
  },
  {
    id: "electrochemical",
    method: "electrochemical",
    name: "Electrochemical Synthesis",
    description: "Electrodeposition or electrochemical intercalation for thin films and layered compounds",
    temperatureRange: [25, 100],
    pressureRange: [0, 0],
    atmosphere: ["electrolyte solution"],
    equipment: ["Potentiostat/galvanostat", "Three-electrode cell", "Reference electrode", "Working electrode substrate"],
    durationRange: [1, 48],
    difficulty: "moderate",
    precursorStrategy: "mixed",
    applicableFamilies: ["oxide", "chalcogenide", "nitride"],
    notes: "Precise control of composition via applied potential. Suitable for thin films and intercalation.",
  },
  {
    id: "spark-plasma-sintering",
    method: "SPS",
    name: "Spark Plasma Sintering",
    description: "Rapid densification using pulsed current and uniaxial pressure",
    temperatureRange: [400, 1600],
    pressureRange: [30, 100],
    atmosphere: ["vacuum", "Ar"],
    equipment: ["SPS system", "Graphite dies", "Pyrometer", "Vacuum pump"],
    durationRange: [0.5, 4],
    difficulty: "moderate",
    precursorStrategy: "mixed",
    applicableFamilies: ["intermetallic", "boride", "carbide", "silicide", "oxide", "MgB2-type", "Chevrel"],
    notes: "Ultra-fast sintering preserves metastable phases and nanostructure. Excellent densification.",
  },
  {
    id: "ammonolysis-nitride",
    method: "solid-state",
    name: "Ammonolysis Nitride Synthesis",
    description: "Metal oxide/chloride + NH3 gas → metal nitride + H2O/HCl",
    temperatureRange: [600, 1500],
    pressureRange: [0, 0],
    atmosphere: ["flowing NH3", "NH3/H2 mix"],
    equipment: ["Tube furnace", "Alumina boat", "Mass flow controllers", "NH3 cracker/scrubber", "XRD"],
    durationRange: [12, 48],
    difficulty: "moderate",
    precursorStrategy: "oxide",
    applicableFamilies: ["nitride", "perovskite-nitride", "oxynitride"],
    requiredElements: ["N"],
    excludedElements: ["H"],
    notes: "Primary route for bulk nitrides. NH3 acts as both nitrogen source and reducing agent.",
  },
  {
    id: "borothermal-reduction",
    method: "arc-melting",
    name: "Borothermal/Carbothermal Reduction",
    description: "Metal oxide + B/C → metal boride/carbide + CO/B2O3",
    temperatureRange: [1400, 2200],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "vacuum"],
    equipment: ["High-temperature vacuum furnace", "Graphite crucible", "Optical pyrometer", "Ball mill"],
    durationRange: [4, 24],
    difficulty: "hard",
    precursorStrategy: "oxide",
    applicableFamilies: ["boride", "carbide", "refractory-intermetallic"],
    requiredElements: ["B"],
    notes: "Used for refractory borides when elemental metal is unavailable or too expensive.",
  },
  {
    id: "alkali-intercalation",
    method: "solid-state",
    name: "Liquid/Vapor Alkali Intercalation",
    description: "Layered compound + Alkali metal → Intercalated superconductor (e.g., K-FeSe)",
    temperatureRange: [50, 600],
    pressureRange: [0, 0],
    atmosphere: ["ultra-high purity Ar", "vacuum"],
    equipment: ["Sealed Pyrex/Quartz tube", "Ar glovebox", "Centrifuge (for liquid metal removal)", "Low-temp furnace"],
    durationRange: [12, 168],
    difficulty: "hard",
    precursorStrategy: "elemental",
    applicableFamilies: ["pnictide", "chalcogenide", "graphite-intercalation", "TMD", "fulleride"],
    requiredElements: ["Li", "Na", "K", "Rb", "Cs"],
    notes: "Critical for inducing superconductivity in layered insulators or semiconductors. Highly air-sensitive.",
  },
  {
    id: "topochemical-reduction",
    method: "solid-state",
    name: "Topochemical Oxygen Reduction",
    description: "Complex oxide + CaH2/LiH → reduced oxide with novel coordination (e.g., Ni1+)",
    temperatureRange: [200, 500],
    pressureRange: [0, 0],
    atmosphere: ["sealed evacuated tube", "flowing Ar"],
    equipment: ["Low-temp tube furnace", "Glovebox", "Quartz ampoules", "Vacuum sealer"],
    durationRange: [24, 120],
    difficulty: "hard",
    precursorStrategy: "oxide",
    applicableFamilies: ["nickelate", "cuprate", "cobaltate", "infinite-layer"],
    requiredElements: ["O"],
    notes: "Low-temperature transition-metal reduction. Essential for synthesizing 'infinite-layer' nickelate superconductors.",
  },
  {
    id: "molten-salt-electro-growth",
    method: "electrochemical",
    name: "Molten Salt Electrocrystallization",
    description: "Electrochemical reduction in molten salt flux to grow crystals",
    temperatureRange: [300, 900],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "N2"],
    equipment: ["Vertical furnace", "Crucible cell", "Potentiostat", "Pt/Au electrodes"],
    durationRange: [24, 168],
    difficulty: "hard",
    precursorStrategy: "oxide",
    applicableFamilies: ["bismuthate", "boride", "oxide-bronze"],
    minElements: 2,
    notes: "Specifically famous for growing high-quality crystals of BKBO (Ba-K-Bi-O) superconductors.",
  },
  {
    id: "metal-flux-extraction",
    method: "flux-growth",
    name: "Reactive Metal Flux Synthesis",
    description: "Reaction in molten Al, Ga, or Sn flux followed by chemical/centrifugal removal",
    temperatureRange: [600, 1200],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "sealed quartz tube"],
    equipment: ["Box furnace", "Canfield crucible set (alumina)", "Centrifuge", "Analytical balance"],
    durationRange: [48, 240],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["pnictide", "silicide", "germanide", "heavy-fermion"],
    minElements: 3,
    notes: "The flux acts as both solvent and reactant. Excellent for discovering new ternary silicides.",
  },
  {
    id: "combustion-synthesis",
    method: "solid-state",
    name: "Solution Combustion Synthesis",
    description: "Metal nitrates + fuel (glycine/urea) → self-sustained redox flame → oxide powder",
    temperatureRange: [25, 1200],
    pressureRange: [0, 0],
    atmosphere: ["air"],
    equipment: ["Muffle furnace", "Hot plate", "Large alumina crucible", "Fume hood"],
    durationRange: [1, 4],
    difficulty: "easy",
    precursorStrategy: "mixed",
    applicableFamilies: ["oxide", "perovskite", "ferrite"],
    requiredElements: ["O"],
    notes: "Fast, exothermic route for highly porous, fine-grained oxide powders. Requires careful safety handling.",
  },
  {
    id: "microwave-sintering",
    method: "solid-state",
    name: "Microwave-Assisted Solid State",
    description: "Direct microwave coupling for ultra-fast volumetric heating and reaction",
    temperatureRange: [25, 1400],
    pressureRange: [0, 0],
    atmosphere: ["air", "Ar", "vacuum"],
    equipment: ["Multimode microwave furnace", "Susceptor (SiC)", "Infrared pyrometer", "Alumina insulation"],
    durationRange: [0.1, 2],
    difficulty: "moderate",
    precursorStrategy: "mixed",
    applicableFamilies: ["oxide", "pnictide", "chalcogenide", "intermetallic"],
    notes: "Reduces reaction time from days to minutes. Can prevent volatile loss due to rapid heating.",
  }
];

function computeFamilyMatchScore(template: ReactionTemplate, family: string): number {
  const normalizedFamily = family.toLowerCase();
  const matchingFamilies = template.applicableFamilies.map(f => f.toLowerCase());

  if (matchingFamilies.some(f => normalizedFamily.includes(f) || f.includes(normalizedFamily))) {
    return 1.0;
  }

  const familyAliases: Record<string, string[]> = {
    cuprate: ["oxide", "perovskite", "high-tc"],
    pnictide: ["iron-based", "fe-based", "arsenide"],
    chalcogenide: ["selenide", "telluride", "sulfide"],
    hydride: ["superhydride", "hydrogen-rich", "clathrate"],
    intermetallic: ["alloy", "A15", "Laves", "Heusler"],
    boride: ["borocarbide", "MgB2-type"],
    "heavy-fermion": ["Kondo", "Ce-based", "U-based"],
    topological: ["TI", "Weyl", "Dirac"],
  };

  for (const [key, aliases] of Object.entries(familyAliases)) {
    const allNames = [key, ...aliases].map(n => n.toLowerCase());
    const familyMatch = allNames.some(n => normalizedFamily.includes(n) || n.includes(normalizedFamily));
    const templateMatch = matchingFamilies.some(f => allNames.some(n => f.includes(n) || n.includes(f)));
    if (familyMatch && templateMatch) return 0.8;
  }

  return 0.1;
}

function computeElementCompatibilityScore(template: ReactionTemplate, elements: string[]): number {
  let score = 1.0;

  if (template.requiredElements) {
    const hasRequired = template.requiredElements.every(el => elements.includes(el));
    if (!hasRequired) score *= 0.05;
  }

  if (template.excludedElements) {
    const hasExcluded = template.excludedElements.some(el => elements.includes(el));
    if (hasExcluded) score *= 0.1;
  }

  if (template.minElements && elements.length < template.minElements) {
    score *= 0.3;
  }
  if (template.maxElements && elements.length > template.maxElements) {
    score *= 0.5;
  }

  const hasVolatile = elements.some(el => ["Hg", "Cd", "Zn", "As", "Se", "Te", "P", "S"].includes(el));
  if (hasVolatile && (template.method === "arc-melting" || template.method === "SPS")) {
    score *= 0.4;
  }

  const hasRefractory = elements.some(el => {
    const mp = getMeltingPoint(el);
    return mp !== null && mp > 2500;
  });
  if (hasRefractory && template.method === "sol-gel") {
    score *= 0.5;
  }

  const hasAirSensitive = elements.some(el => ["Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba", "La", "Ce", "Eu", "Yb"].includes(el));
  if (hasAirSensitive && template.method === "hydrothermal") {
    score *= 0.6;
  }

  return Math.max(0, Math.min(1, score));
}

function computeThermodynamicMatchScore(template: ReactionTemplate, formationEnergy: number | null, elements: string[]): number {
  let score = 0.5;

  if (formationEnergy !== null) {
    if (formationEnergy < -0.5) {
      score = 0.9;
    } else if (formationEnergy < 0) {
      score = 0.7;
    } else if (formationEnergy < 0.3) {
      score = 0.5;
    } else if (formationEnergy < 1.0) {
      score = 0.3;
    } else {
      score = 0.1;
    }

    if (formationEnergy > 0 && (template.method === "high-pressure" || template.method === "SPS")) {
      score += 0.2;
    }
    if (formationEnergy > 0.5 && template.method === "ball-milling") {
      score += 0.15;
    }
  }

  let maxMp = 0;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp !== null && mp > maxMp) maxMp = mp;
  }

  const tammTemp = maxMp > 0 ? 0.57 * maxMp : 1000;
  const templateMidTemp = (template.temperatureRange[0] + template.temperatureRange[1]) / 2;

  if (template.method === "solid-state" || template.method === "ball-milling") {
    if (templateMidTemp >= tammTemp * 0.8) {
      score += 0.1;
    } else {
      score -= 0.1;
    }
  }

  return Math.max(0, Math.min(1, score));
}

function computePressureFeasibilityScore(template: ReactionTemplate, family: string, elements: string[]): number {
  const normalizedFamily = family.toLowerCase();
  let score = 0.8;

  const isHydride = normalizedFamily.includes("hydride") || normalizedFamily.includes("hydrogen");
  const hasH = elements.includes("H");
  const counts = parseFormulaCounts(elements.join(""));

  if (isHydride || hasH) {
    if (template.method === "high-pressure") {
      score = 0.95;
    } else {
      score = 0.3;
    }
  }

  if (template.pressureRange[1] > 100) {
    score *= 0.7;
  }
  if (template.pressureRange[1] > 200) {
    score *= 0.6;
  }

  if (template.method === "high-pressure" && !isHydride && !hasH) {
    score *= 0.6;
  }

  return Math.max(0, Math.min(1, score));
}

function buildMatchReasoning(template: ReactionTemplate, familyScore: number, elementScore: number, thermoScore: number, pressureScore: number, family: string): string[] {
  const reasons: string[] = [];

  if (familyScore >= 0.8) {
    reasons.push(`${template.method} is well-suited for ${family} materials`);
  } else if (familyScore >= 0.5) {
    reasons.push(`${template.method} has partial compatibility with ${family} family`);
  }

  if (elementScore >= 0.8) {
    reasons.push("Element composition is compatible with this synthesis method");
  } else if (elementScore < 0.3) {
    reasons.push("Element compatibility issues reduce feasibility");
  }

  if (thermoScore >= 0.7) {
    reasons.push("Thermodynamically favorable formation");
  } else if (thermoScore < 0.4) {
    reasons.push("Thermodynamic barriers may require high-energy methods");
  }

  if (pressureScore >= 0.8) {
    reasons.push("Pressure requirements are appropriate for this method");
  } else if (pressureScore < 0.4) {
    reasons.push("Pressure constraints limit feasibility");
  }

  reasons.push(`Equipment: ${template.equipment.slice(0, 3).join(", ")}`);
  reasons.push(`Temperature: ${template.temperatureRange[0]}-${template.temperatureRange[1]} K, Duration: ${template.durationRange[0]}-${template.durationRange[1]} hours`);

  return reasons;
}

export function matchTemplates(
  formula: string,
  family?: string,
  formationEnergy?: number | null
): TemplateMatch[] {
  const resolvedFamily = family || classifyFamily(formula);
  const elements = parseFormulaElements(formula);
  const resolvedEnergy = formationEnergy ?? null;

  const matches: TemplateMatch[] = [];

  for (const template of REACTION_TEMPLATES) {
    const familyMatchScore = computeFamilyMatchScore(template, resolvedFamily);
    const elementCompatibilityScore = computeElementCompatibilityScore(template, elements);
    const thermodynamicScore = computeThermodynamicMatchScore(template, resolvedEnergy, elements);
    const pressureFeasibilityScore = computePressureFeasibilityScore(template, resolvedFamily, elements);

    const score =
      familyMatchScore * 0.35 +
      elementCompatibilityScore * 0.25 +
      thermodynamicScore * 0.25 +
      pressureFeasibilityScore * 0.15;

    const reasoning = buildMatchReasoning(template, familyMatchScore, elementCompatibilityScore, thermodynamicScore, pressureFeasibilityScore, resolvedFamily);

    matches.push({
      template,
      score: Number(score.toFixed(4)),
      familyMatchScore: Number(familyMatchScore.toFixed(4)),
      elementCompatibilityScore: Number(elementCompatibilityScore.toFixed(4)),
      thermodynamicScore: Number(thermodynamicScore.toFixed(4)),
      pressureFeasibilityScore: Number(pressureFeasibilityScore.toFixed(4)),
      reasoning,
    });
  }

  matches.sort((a, b) => b.score - a.score);

  return matches;
}

export function getTemplateById(id: string): ReactionTemplate | undefined {
  return REACTION_TEMPLATES.find(t => t.id === id);
}

export function getTemplatesByMethod(method: string): ReactionTemplate[] {
  return REACTION_TEMPLATES.filter(t => t.method === method);
}

export function getAllTemplates(): ReactionTemplate[] {
  return [...REACTION_TEMPLATES];
}

export function getTopTemplates(formula: string, family?: string, formationEnergy?: number | null, limit: number = 5): TemplateMatch[] {
  return matchTemplates(formula, family, formationEnergy).slice(0, limit);
}
