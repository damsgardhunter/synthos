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
  },
  {
    id: "high-pressure-floating-zone",
    method: "floating-zone",
    name: "High-Pressure Optical Floating Zone",
    description: "Crucible-free crystal growth using focused light under high gas pressure",
    temperatureRange: [1000, 3000],
    pressureRange: [1, 150],
    atmosphere: ["O2", "Ar", "Ar/H2 mix"],
    equipment: ["Optical floating zone furnace", "Xenon/Halogen lamps", "High-pressure gas system", "Shaft rotation controllers"],
    durationRange: [12, 120],
    difficulty: "extreme",
    precursorStrategy: "oxide",
    applicableFamilies: ["cuprate", "ruthenate", "nickelate", "intermetallic", "oxide"],
    notes: "Ideal for high-purity crystals of incongruently melting oxides. High pressure stabilizes high oxidation states.",
  },
  {
    id: "reactive-sputtering-nitride",
    method: "sputtering",
    name: "Reactive Magnetron Sputtering",
    description: "Sputtering metal target in a partial N2/O2 atmosphere to form compounds in-situ",
    temperatureRange: [25, 900],
    pressureRange: [0, 0.05],
    atmosphere: ["Ar/N2 mix", "Ar/O2 mix"],
    equipment: ["DC/RF Sputter system", "Mass flow controllers", "Substrate heater", "Plasma monitor"],
    durationRange: [1, 6],
    difficulty: "hard",
    precursorStrategy: "elemental",
    applicableFamilies: ["nitride", "oxynitride", "oxide", "MgB2-type"],
    requiredElements: ["N"],
    notes: "Enables precise control over nitrogen/oxygen stoichiometry in thin films via gas flow ratios.",
  },
  {
    id: "spin-coating-sol-gel",
    method: "sol-gel",
    name: "Spin-Coating Deposition",
    description: "Thin film formation via solution casting on rotating substrate followed by annealing",
    temperatureRange: [300, 900],
    pressureRange: [0, 0],
    atmosphere: ["air", "O2", "N2"],
    equipment: ["Spin coater", "Hot plate", "Clean room environment", "Annealing furnace"],
    durationRange: [2, 12],
    difficulty: "easy",
    precursorStrategy: "organometallic",
    applicableFamilies: ["oxide", "cuprate", "perovskite", "bismuthate"],
    notes: "Low-cost route for large-area superconducting films. Multiple layers can be stacked.",
  },
  {
    id: "chemical-bath-deposition",
    method: "hydrothermal",
    name: "Chemical Bath Deposition (CBD)",
    description: "Controlled chemical reaction in aqueous solution to deposit films on substrates",
    temperatureRange: [25, 100],
    pressureRange: [0, 0],
    atmosphere: ["aqueous solution"],
    equipment: ["Jacketed reaction vessel", "Stirrer", "Substrate holders", "pH controller"],
    durationRange: [1, 24],
    difficulty: "easy",
    precursorStrategy: "mixed",
    applicableFamilies: ["chalcogenide", "oxide", "hydroxide"],
    requiredElements: ["S", "Se"],
    notes: "Extremely low-temperature method. Suitable for sensitive substrates and large-scale coatings.",
  },
  {
    id: "zone-refining",
    method: "Bridgman",
    name: "Zone Refining / Zone Melting",
    description: "Purification and crystal growth by moving a molten zone along a rod",
    temperatureRange: [500, 2800],
    pressureRange: [0, 10],
    atmosphere: ["Ar", "vacuum", "H2"],
    equipment: ["Zone melting furnace", "Induction/Resistance heater", "Quartz/Alumina boat", "Precision drive"],
    durationRange: [24, 240],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["intermetallic", "chalcogenide", "boride", "silicide"],
    notes: "Used to obtain ultra-high purity intermetallics by segregating impurities to the end of the rod.",
  },
  {
    id: "topochemical-fluorination",
    method: "solid-state",
    name: "Topochemical Fluorination",
    description: "Low-temp insertion of fluorine into an oxide lattice using XeF2 or PVDF",
    temperatureRange: [150, 400],
    pressureRange: [0, 0],
    atmosphere: ["F2 gas", "sealed tube with fluorinating agent"],
    equipment: ["Monel/Teflon-lined reactor", "Fluorine gas handling system", "Quartz ampoules"],
    durationRange: [12, 120],
    difficulty: "extreme",
    precursorStrategy: "oxide",
    applicableFamilies: ["oxyfluoride", "cuprate", "iron-pnictide"],
    requiredElements: ["F"],
    notes: "Unique route to induce superconductivity by hole/electron doping via fluorine insertion.",
  },
  {
    id: "metal-organic-decomposition",
    method: "CVD",
    name: "Metal-Organic Decomposition (MOD)",
    description: "Solution of metal-organic salts printed/coated and thermally decomposed to oxide",
    temperatureRange: [400, 1000],
    pressureRange: [0, 0],
    atmosphere: ["O2", "Ar/H2O mix"],
    equipment: ["Solution preparation lab", "Slot-die or Inkjet coater", "Reel-to-reel furnace"],
    durationRange: [5, 48],
    difficulty: "moderate",
    precursorStrategy: "organometallic",
    applicableFamilies: ["cuprate", "REBCO-tape", "perovskite"],
    requiredElements: ["O"],
    notes: "Industry standard for 'Second Generation' superconducting tapes (coated conductors).",
  },
  {
    id: "high-pressure-belt-press",
    method: "high-pressure",
    name: "Belt-Type High-Pressure Synthesis",
    description: "Bulk synthesis in a large-volume belt press for industrial scale HP phases",
    temperatureRange: [500, 2000],
    pressureRange: [1, 10],
    atmosphere: ["sealed pyrophyllite cell"],
    equipment: ["Belt press", "Hydraulic ram", "Internal carbon heater", "Pyrophyllite gaskets"],
    durationRange: [1, 10],
    difficulty: "hard",
    precursorStrategy: "mixed",
    applicableFamilies: ["boride", "diamond-like", "intermetallic", "MgB2-type"],
    notes: "Bridges the gap between lab-scale multi-anvil and industrial production. Used for synthetic diamond/cBN.",
  },
  {
    id: "arc-melt-suction-casting",
    method: "arc-melting",
    name: "Arc-Melt Suction Casting",
    description: "Melting followed by rapid suction into a Cu mold to form glassy or nanocrystalline rods",
    temperatureRange: [1500, 3500],
    pressureRange: [0, 0],
    atmosphere: ["Ar"],
    equipment: ["Arc furnace with suction attachment", "Cu mold", "Vacuum reservoir", "Fast-acting valve"],
    durationRange: [0.1, 1],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["amorphous-superconductor", "intermetallic", "heavy-fermion"],
    notes: "Crucial for obtaining bulk metallic glasses or preventing phase separation in intermetallics.",
  },
  {
    id: "solvothermal-microwave",
    method: "hydrothermal",
    name: "Microwave-Solvothermal Synthesis",
    description: "Organic solvent reaction in pressure vessels heated by microwave radiation",
    temperatureRange: [100, 300],
    pressureRange: [1, 5],
    atmosphere: ["organic solvent (alcohol, ethylene glycol)"],
    equipment: ["Microwave digestion system", "Teflon bomb", "Pressure/Temp sensors"],
    durationRange: [0.5, 6],
    difficulty: "easy",
    precursorStrategy: "mixed",
    applicableFamilies: ["chalcogenide", "oxide", "pnictide", "nanostructure"],
    notes: "Extremely fast kinetics compared to standard hydrothermal. Promotes uniform nucleation.",
  },
  {
    id: "vapor-phase-epitaxy",
    method: "CVD",
    name: "Vapor Phase Epitaxy (VPE)",
    description: "Growth of thin crystalline layers from the vapor phase onto a substrate",
    temperatureRange: [500, 1100],
    pressureRange: [0, 1],
    atmosphere: ["H2 carrier", "hydride gas (AsH3, PH3)"],
    equipment: ["VPE reactor", "Gas scrubbing system", "RF heating coil", "Optical pyrometer"],
    durationRange: [2, 10],
    difficulty: "hard",
    precursorStrategy: "mixed",
    applicableFamilies: ["pnictide", "chalcogenide", "nitride"],
    notes: "Commonly used for pnictide films. Highly hazardous due to toxic hydride gases.",
  },
  {
    id: "thermal-evaporation",
    method: "sputtering",
    name: "Co-Evaporation / Thermal Evaporation",
    description: "Heating materials in crucibles via resistive heating to form thin films",
    temperatureRange: [25, 600],
    pressureRange: [0, 0],
    atmosphere: ["High vacuum (< 10^-6 Torr)"],
    equipment: ["Evaporation chamber", "W/Ta boats", "Quartz crystal monitor", "Shutter system"],
    durationRange: [1, 4],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["intermetallic", "chalcogenide", "lead-type", "alloy"],
    notes: "Simple thin film route. Best for low-melting-point elements like Sn, In, Pb, or Al.",
  },
  {
    id: "melt-spinning",
    method: "arc-melting",
    name: "Melt Spinning",
    description: "Molten metal jet quenched on a rapidly rotating Cu wheel to form ribbons",
    temperatureRange: [1000, 2500],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "He"],
    equipment: ["Melt spinner", "Induction coil", "Cu chill block wheel", "Quartz nozzle"],
    durationRange: [0.1, 2],
    difficulty: "hard",
    precursorStrategy: "elemental",
    applicableFamilies: ["amorphous-superconductor", "intermetallic", "A15", "metastable"],
    notes: "Standard for producing amorphous superconducting ribbons. Cooling rates up to 10^6 K/s.",
  },
  {
    id: "flux-growth-centrifugal",
    method: "flux-growth",
    name: "Centrifugal Flux Growth",
    description: "Crystals grown in flux and separated at high-temp using a centrifuge",
    temperatureRange: [600, 1300],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "sealed ampoule"],
    equipment: ["Box furnace", "High-temperature centrifuge", "Quartz/Alumina crucibles", "Strain-gauge"],
    durationRange: [24, 168],
    difficulty: "hard",
    precursorStrategy: "elemental",
    applicableFamilies: ["heavy-fermion", "pnictide", "chalcogenide", "intermetallic"],
    notes: "The centrifuge spin decants the liquid flux from the crystals while still inside the furnace.",
  },
  {
    id: "atomic-layer-deposition",
    method: "CVD",
    name: "Atomic Layer Deposition (ALD)",
    description: "Sequential, self-limiting surface reactions to deposit monolayers",
    temperatureRange: [100, 450],
    pressureRange: [0, 0.01],
    atmosphere: ["inert purge gas (N2/Ar)", "reactant pulse"],
    equipment: ["ALD reactor", "Fast-switching valves", "Precursor manifolds", "Vacuum system"],
    durationRange: [4, 24],
    difficulty: "hard",
    precursorStrategy: "organometallic",
    applicableFamilies: ["nitride", "oxide", "topological"],
    notes: "Unmatched conformal coating and thickness control. Essential for Josephson junction barriers.",
  },
  {
    id: "liquid-phase-epitaxy",
    method: "flux-growth",
    name: "Liquid Phase Epitaxy (LPE)",
    description: "Growth of epitaxial layers from a supersaturated liquid melt onto a substrate",
    temperatureRange: [400, 1000],
    pressureRange: [0, 0],
    atmosphere: ["H2", "Ar"],
    equipment: ["LPE slider system", "Multi-bin crucible", "Precision substrate holder", "Tube furnace"],
    durationRange: [4, 48],
    difficulty: "hard",
    precursorStrategy: "mixed",
    applicableFamilies: ["cuprate", "oxide", "perovskite", "pnictide"],
    notes: "Fast growth rate for high-quality thick crystalline films. Used for early HTS film research.",
  },
  {
    id: "hot-isostatic-pressing",
    method: "solid-state",
    name: "Hot Isostatic Pressing (HIP)",
    description: "Simultaneous application of high temperature and high gas pressure to densify bulk",
    temperatureRange: [500, 2000],
    pressureRange: [50, 200],
    atmosphere: ["Ar", "N2"],
    equipment: ["HIP vessel", "Internal furnace", "Compressor system", "Encapsulation tools"],
    durationRange: [4, 24],
    difficulty: "hard",
    precursorStrategy: "mixed",
    applicableFamilies: ["intermetallic", "boride", "oxide", "MgB2-type", "A15"],
    notes: "Eliminates porosity and heals microcracks in bulk superconductors. Critical for wire performance.",
  },
  {
    id: "amorphous-precursor-reaction",
    method: "solid-state",
    name: "Amorphous Precursor Route",
    description: "Reaction of amorphous/nanocrystalline powders to lower synthesis temperature",
    temperatureRange: [300, 800],
    pressureRange: [0, 0],
    atmosphere: ["vacuum", "Ar"],
    equipment: ["Ball mill", "Low-temp furnace", "Glovebox", "Pellet press"],
    durationRange: [12, 120],
    difficulty: "moderate",
    precursorStrategy: "mixed",
    applicableFamilies: ["intermetallic", "pnictide", "metastable"],
    notes: "Bypasses high-temp barriers by using high-energy starting materials. Stabilizes metastable phases.",
  },
  {
    id: "metal-vapor-annealing",
    method: "vapor-transport",
    name: "Metal Vapor Annealing",
    description: "Annealing a compound in a specific metal vapor to tune stoichiometry (e.g., Mg in MgB2)",
    temperatureRange: [400, 950],
    pressureRange: [0, 1],
    atmosphere: ["sealed tube with excess metal"],
    equipment: ["Two-zone furnace", "Stainless steel or Quartz ampoule", "Glovebox"],
    durationRange: [12, 72],
    difficulty: "moderate",
    precursorStrategy: "mixed",
    applicableFamilies: ["MgB2-type", "pnictide", "chalcogenide"],
    notes: "Prevents loss of volatile components or forces over-stoichiometry to improve Tc.",
  },
  {
    id: "self-propagating-synthesis",
    method: "solid-state",
    name: "Self-Propagating High-Temp Synthesis (SHS)",
    description: "Redox reaction ignited locally that propagates as a combustion wave",
    temperatureRange: [1000, 3500],
    pressureRange: [0, 100],
    atmosphere: ["Ar", "air", "vacuum"],
    equipment: ["SHS reactor", "Ignition source (W wire)", "Pellet press", "High-speed camera"],
    durationRange: [0.01, 0.1],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["boride", "silicide", "nitride", "carbide", "intermetallic"],
    notes: "Extremely fast. Heat of reaction drives the process. Can produce very high purity borides.",
  },
  {
    id: "pulsed-electron-deposition",
    method: "PLD",
    name: "Pulsed Electron Deposition (PED)",
    description: "High-power pulsed electron beam ablates target to form thin films",
    temperatureRange: [25, 800],
    pressureRange: [0, 0.001],
    atmosphere: ["O2", "Ar", "N2"],
    equipment: ["Pulsed electron source", "Vacuum chamber", "Target carousel", "Substrate heater"],
    durationRange: [2, 6],
    difficulty: "hard",
    precursorStrategy: "oxide",
    applicableFamilies: ["oxide", "cuprate", "nitride", "pnictide"],
    notes: "Alternative to PLD using electrons instead of lasers. Highly efficient for complex targets.",
  },
  {
    id: "laser-floating-zone",
    method: "floating-zone",
    name: "Laser-Heated Floating Zone",
    description: "Floating zone growth using high-power CO2 lasers as the heat source",
    temperatureRange: [500, 3500],
    pressureRange: [0, 10],
    atmosphere: ["air", "O2", "Ar"],
    equipment: ["CO2 Laser system", "Focusing optics", "Floating zone chamber", "Pyrometer"],
    durationRange: [4, 48],
    difficulty: "hard",
    precursorStrategy: "mixed",
    applicableFamilies: ["oxide", "boride", "intermetallic", "fiber-crystal"],
    notes: "Precise control over temperature gradients. Can grow very small diameter crystal fibers.",
  },
  {
    id: "arc-melting-chill-block",
    method: "arc-melting",
    name: "Chill-Block Melt Spinning",
    description: "Induction melting in a quartz tube followed by ejection onto a Cu wheel",
    temperatureRange: [1000, 2500],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "He"],
    equipment: ["Melt spinning system", "Induction generator", "Cu wheel", "Pressure controller"],
    durationRange: [0.1, 1],
    difficulty: "hard",
    precursorStrategy: "elemental",
    applicableFamilies: ["intermetallic", "amorphous-superconductor", "A15"],
    notes: "Optimized for ribbon production of brittle intermetallics to study grain boundary effects.",
  },
  {
    id: "ion-beam-sputtering",
    method: "sputtering",
    name: "Ion Beam Sputtering (IBS)",
    description: "Independent ion beam source used to sputter targets in a vacuum",
    temperatureRange: [25, 700],
    pressureRange: [0, 0],
    atmosphere: ["Ar", "O2 plasma"],
    equipment: ["Kaufman ion source", "UHV chamber", "Target holder", "Neutralizer"],
    durationRange: [4, 12],
    difficulty: "hard",
    precursorStrategy: "mixed",
    applicableFamilies: ["oxide", "nitride", "metal-multilayer"],
    notes: "Produces very dense films with low impurity content compared to magnetron sputtering.",
  },
  {
    id: "microwave-hydrothermal",
    method: "hydrothermal",
    name: "Microwave-Assisted Hydrothermal",
    description: "Rapid hydrothermal reaction using microwave heating for phase control",
    temperatureRange: [100, 250],
    pressureRange: [1, 10],
    atmosphere: ["aqueous solution"],
    equipment: ["Microwave autoclave", "Pressure sensors", "Centrifuge", "Drying oven"],
    durationRange: [0.5, 4],
    difficulty: "easy",
    precursorStrategy: "mixed",
    applicableFamilies: ["oxide", "hydroxide", "chalcogenide"],
    notes: "Significantly reduces time to synthesize superconducting oxides/hydroxides like FeSe-based.",
  },
  {
    id: "salt-flux-electrolysis",
    method: "electrochemical",
    name: "Salt-Flux Electrolysis",
    description: "Deposition from a molten salt electrolyte at high temperatures",
    temperatureRange: [400, 1000],
    pressureRange: [0, 0],
    atmosphere: ["Ar"],
    equipment: ["High-temp furnace", "Electrolysis cell", "Potentiostat", "Graphite anode"],
    durationRange: [12, 72],
    difficulty: "hard",
    precursorStrategy: "oxide",
    applicableFamilies: ["boride", "silicide", "oxide"],
    notes: "Used to grow crystals of refractory superconductors like Titanium Boride.",
  },
  {
    id: "metal-vapor-deposition",
    method: "mbe",
    name: "Metal Vapor Deposition (MVD)",
    description: "Co-deposition of metal vapors in UHV to form alloys and compounds",
    temperatureRange: [25, 500],
    pressureRange: [0, 0],
    atmosphere: ["Vacuum (< 10^-8 Torr)"],
    equipment: ["E-beam evaporator", "Quartz monitor", "Substrate rotator"],
    durationRange: [2, 8],
    difficulty: "moderate",
    precursorStrategy: "elemental",
    applicableFamilies: ["alloy", "intermetallic", "thin-film"],
    notes: "Precision control for simple metal-superconductor thin films.",
  },
  {
    id: "topochemical-intercalation",
    method: "solid-state",
    name: "Soft-Chemistry Intercalation",
    description: "Intercalation of molecules or ions into a host at or near room temp",
    temperatureRange: [20, 150],
    pressureRange: [0, 0],
    atmosphere: ["N2", "Ar", "solution"],
    equipment: ["Stirrer", "Glovebox", "Centrifuge", "Schlenk line"],
    durationRange: [24, 240],
    difficulty: "moderate",
    precursorStrategy: "mixed",
    applicableFamilies: ["TMD", "chalcogenide", "graphite-intercalation"],
    notes: "Common for organic molecular intercalation into superconducting layered materials.",
  },
  {
    id: "hybrid-phys-chem-deposition",
    method: "CVD",
    name: "Hybrid Physical-Chemical Vapor Deposition (HPCVD)",
    description: "Combination of thermal evaporation and CVD for complex phases (e.g., MgB2)",
    temperatureRange: [600, 800],
    pressureRange: [0.01, 1],
    atmosphere: ["H2", "B2H6 gas"],
    equipment: ["HPCVD reactor", "Induction heater", "Gas cabinets", "Safety scrubbers"],
    durationRange: [1, 4],
    difficulty: "extreme",
    precursorStrategy: "mixed",
    applicableFamilies: ["MgB2-type"],
    requiredElements: ["B"],
    notes: "Considered the gold standard for producing the highest Tc and Jc MgB2 thin films.",
  },
  {
    id: "ultrasonic-spray-pyrolysis",
    method: "CVD",
    name: "Ultrasonic Spray Pyrolysis",
    description: "Aerosol of precursor solution sprayed onto heated substrate and decomposed",
    temperatureRange: [300, 800],
    pressureRange: [0, 0],
    atmosphere: ["air", "Ar/O2"],
    equipment: ["Ultrasonic atomizer", "Carrier gas supply", "Substrate heater", "Exhaust hood"],
    durationRange: [1, 5],
    difficulty: "easy",
    precursorStrategy: "mixed",
    applicableFamilies: ["oxide", "cuprate", "bismuthate"],
    requiredElements: ["O"],
    notes: "Scalable, low-cost atmospheric pressure thin-film deposition technique.",
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
