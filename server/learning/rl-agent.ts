const ELEMENT_GROUPS = [
  { name: "alkali", elements: ["Li", "Na", "K", "Rb", "Cs"] },
  { name: "alkaline-earth", elements: ["Be", "Mg", "Ca", "Sr", "Ba"] },
  { name: "3d-transition", elements: ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"] },
  { name: "4d-transition", elements: ["Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag"] },
  { name: "5d-transition", elements: ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"] },
  { name: "lanthanide", elements: ["La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy", "Er", "Yb", "Lu"] },
  { name: "p-block-metal", elements: ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi"] },
  { name: "metalloid", elements: ["B", "Si", "Ge", "As", "Sb", "Te", "Se"] },
  { name: "nonmetal", elements: ["H", "C", "N", "O", "F", "P", "S", "Cl"] },
] as const;

const STOICH_TEMPLATES = [
  { name: "binary-metal-rich", pattern: "A3B", nElements: 2 },
  { name: "binary-balanced", pattern: "AB", nElements: 2 },
  { name: "binary-anion-rich", pattern: "AB3", nElements: 2 },
  { name: "ternary-122", pattern: "AB2C2", nElements: 3 },
  { name: "ternary-perovskite", pattern: "ABC3", nElements: 3 },
  { name: "ternary-balanced", pattern: "A2BC", nElements: 3 },
  { name: "quaternary", pattern: "ABCD", nElements: 4 },
  { name: "hydride-rich", pattern: "AH3", nElements: 2 },
  { name: "ternary-hydride", pattern: "ABH4", nElements: 3 },
  { name: "boride-carbide", pattern: "A2B3C", nElements: 3 },
] as const;

const STRUCTURE_TYPES = [
  "A15", "AlB2", "NaCl", "Perovskite", "ThCr2Si2",
  "Heusler", "BCC", "FCC", "Layered", "Kagome",
  "HexBoride", "MX2", "Anti-perovskite", "CsCl",
  "Cu2Mg-Laves", "Fluorite", "Cr3Si", "Ni3Sn", "Fe3C", "Spinel",
  "Clathrate-H32", "Skutterudite", "BiS2-layered", "Kagome-variant",
  "Chevrel", "Pyrite", "Wurtzite", "Antifluorite",
  "Laves-C14", "Laves-C15", "HfFe6Ge6", "CeCu2Si2",
  "PuCoGa5-115", "Infinite-layer", "T-prime",
] as const;

const LAYERING_DIMENSIONS = [
  { name: "3D-isotropic", dim: 3, description: "Fully 3D metallic bonding" },
  { name: "quasi-2D", dim: 2, description: "Layered with weak interlayer coupling" },
  { name: "quasi-1D", dim: 1, description: "Chain-like with 1D channels" },
  { name: "mixed-dim", dim: 2.5, description: "Intermediate dimensionality" },
] as const;

const HYDROGEN_DENSITIES = [
  { name: "no-hydrogen", ratio: 0, description: "No hydrogen content" },
  { name: "low-H", ratio: 2, description: "Interstitial hydrogen (H/M ~ 1-2)" },
  { name: "medium-H", ratio: 4, description: "Moderate hydrogen (H/M ~ 3-4)" },
  { name: "high-H", ratio: 6, description: "High hydrogen at ambient limits (H/M ~ 5-6)" },
] as const;

const ELECTRON_COUNTS = [
  { name: "low-VEC", vec: 3, description: "Low valence electron count (1-4)" },
  { name: "mid-VEC", vec: 5, description: "Mid valence electron count (5-7)" },
  { name: "high-VEC", vec: 9, description: "High valence electron count (8-10)" },
  { name: "very-high-VEC", vec: 12, description: "Very high VEC (11+)" },
] as const;

const ORBITAL_CONFIGS = [
  { name: "s-dominant", orbital: "s", description: "s-orbital dominated bonding" },
  { name: "p-dominant", orbital: "p", description: "p-orbital dominated bonding" },
  { name: "d-dominant", orbital: "d", description: "d-orbital dominated bonding" },
  { name: "f-dominant", orbital: "f", description: "f-orbital dominated bonding" },
  { name: "sp-hybrid", orbital: "sp", description: "sp hybridized bonding" },
  { name: "sd-hybrid", orbital: "sd", description: "sd hybridized bonding" },
] as const;

interface RLState {
  bestTc: number;
  avgRecentTc: number;
  recentRewardTrend: number;
  familyDiversity: number;
  stagnationCycles: number;
  explorationBudgetUsed: number;
  elementSuccessEntropy: number;
  cycleNumber: number;
}

const CHEMICAL_FAMILY_ACTIONS = [
  { name: "hydride", hostGroups: [5, 1], anionGroups: [8], biasStructures: [6, 7, 8, 20] },
  { name: "intermetallic", hostGroups: [2, 3, 4], anionGroups: [6, 7], biasStructures: [0, 14, 5, 21, 28, 29] },
  { name: "layered-pnictide", hostGroups: [1, 5], anionGroups: [7], biasStructures: [4, 8, 22, 31] },
  { name: "boride", hostGroups: [1, 2], anionGroups: [7], biasStructures: [8, 10] },
  { name: "cuprate", hostGroups: [1, 5], anionGroups: [8], biasStructures: [3, 8, 33, 34] },
  { name: "chalcogenide", hostGroups: [2, 3, 4], anionGroups: [7], biasStructures: [11, 8, 25] },
  { name: "kagome-metal", hostGroups: [2, 3], anionGroups: [6, 7], biasStructures: [9, 23, 30] },
  { name: "oxide-perovskite", hostGroups: [1, 5], anionGroups: [8], biasStructures: [3, 12, 34] },
] as const;

interface RLAction {
  elementGroup1: number;
  elementGroup2: number;
  stoichTemplate: number;
  structureType: number;
  layeringDimension: number;
  hydrogenDensity: number;
  electronCount: number;
  orbitalConfiguration: number;
  chemicalFamily: number;
}

interface Experience {
  state: RLState;
  action: RLAction;
  reward: number;
  timestamp: number;
}

interface ElementPairPrior {
  el1: string;
  el2: string;
  bias: number;
  reason: string;
}

const KNOWN_PAIR_PRIORS: ElementPairPrior[] = [
  { el1: "La", el2: "H", bias: 0.6, reason: "LaH10 high-Tc hydride" },
  { el1: "Y", el2: "H", bias: 0.5, reason: "YH6/YH9 high-Tc hydride" },
  { el1: "Ca", el2: "H", bias: 0.4, reason: "CaH6 superconducting hydride" },
  { el1: "Ba", el2: "H", bias: 0.35, reason: "BaH hydride family" },
  { el1: "Fe", el2: "As", bias: 0.45, reason: "Iron pnictide superconductors" },
  { el1: "Fe", el2: "Se", bias: 0.4, reason: "FeSe superconductor family" },
  { el1: "Cu", el2: "O", bias: 0.5, reason: "Cuprate superconductors" },
  { el1: "Nb", el2: "B", bias: 0.35, reason: "MgB2-type boride superconductors" },
  { el1: "Nb", el2: "N", bias: 0.3, reason: "NbN conventional superconductor" },
  { el1: "Nb", el2: "Ge", bias: 0.3, reason: "Nb3Ge A15 superconductor" },
  { el1: "Nb", el2: "Sn", bias: 0.3, reason: "Nb3Sn A15 superconductor" },
  { el1: "Y", el2: "Ba", bias: 0.45, reason: "YBCO cuprate superconductor" },
  { el1: "Bi", el2: "Se", bias: 0.3, reason: "Topological superconductor candidate" },
  { el1: "Bi", el2: "Sr", bias: 0.35, reason: "BSCCO cuprate superconductor" },
  { el1: "Mg", el2: "B", bias: 0.4, reason: "MgB2 conventional superconductor" },
  { el1: "La", el2: "Cu", bias: 0.35, reason: "LSCO cuprate family" },
  { el1: "Ir", el2: "H", bias: 0.3, reason: "Iridium hydride candidate" },
  { el1: "Ce", el2: "H", bias: 0.35, reason: "Cerium hydride candidate" },
  { el1: "Th", el2: "H", bias: 0.3, reason: "Thorium hydride candidate" },
  { el1: "V", el2: "Si", bias: 0.25, reason: "V3Si A15 superconductor" },
];

interface PolicyWeights {
  elementGroup: number[];
  stoichTemplate: number[];
  structureType: number[];
  elementPairBias: number[][];
  elementPairSpecific: Map<string, number>;
  layeringDimension: number[];
  hydrogenDensity: number[];
  electronCount: number[];
  orbitalConfiguration: number[];
  chemicalFamily: number[];
}

export interface PhysicsAwareRewardContext {
  lambda?: number;
  metallicity?: number;
  nestingScore?: number;
  vanHoveProximity?: number;
  dimensionality?: number;
  hydrogenRatio?: number;
  orbitalCharacter?: string;
  correlationStrength?: number;
  phononFrequency?: number;
  bandFlatness?: number;
  motifName?: string;
  chemicalFamily?: string;
  electronCountValid?: boolean;
  vecPerTM?: number;
  dElectronCount?: number;
  synthesisScore?: number;
}

const KNOWN_SC_MOTIFS = new Set([
  "A15", "perovskite", "ThCr2Si2", "NaCl-type", "AlB2-type",
  "clathrate", "layered-cuprate", "BCC-hydride", "FCC-hydride",
  "Heusler", "Chevrel", "skutterudite", "pyrochlore", "anti-perovskite",
  "infinite-layer", "Ruddlesden-Popper", "nickelate", "MgB2-sigma",
  "TMD", "fullerene", "heavy-fermion", "borocarbide", "BiS2-layer",
  "1T-prime-TMD", "carbon-clathrate", "oxide-interface", "kagome",
  "Clathrate-H32", "Kagome-variant", "Laves-C14", "Laves-C15",
  "HfFe6Ge6", "CeCu2Si2", "PuCoGa5-115", "T-prime",
  "pyrite", "wurtzite", "antifluorite",
]);

const KNOWN_FAMILIES = new Set([
  "hydride", "intermetallic", "layered-pnictide", "boride",
  "cuprate", "chalcogenide", "kagome-metal", "oxide-perovskite",
]);

const MOTIF_FAMILY_MAP: Record<string, string[]> = {
  "A15": ["intermetallic"],
  "perovskite": ["oxide-perovskite", "cuprate"],
  "ThCr2Si2": ["layered-pnictide", "intermetallic"],
  "AlB2-type": ["boride"],
  "MgB2-sigma": ["boride"],
  "clathrate": ["hydride"],
  "BCC-hydride": ["hydride"],
  "FCC-hydride": ["hydride"],
  "layered-cuprate": ["cuprate"],
  "infinite-layer": ["cuprate"],
  "Ruddlesden-Popper": ["cuprate", "oxide-perovskite"],
  "nickelate": ["cuprate", "oxide-perovskite"],
  "Heusler": ["intermetallic"],
  "Chevrel": ["chalcogenide", "intermetallic"],
  "skutterudite": ["intermetallic"],
  "pyrochlore": ["oxide-perovskite"],
  "anti-perovskite": ["oxide-perovskite", "intermetallic"],
  "TMD": ["chalcogenide"],
  "1T-prime-TMD": ["chalcogenide"],
  "BiS2-layer": ["chalcogenide"],
  "fullerene": ["intermetallic"],
  "heavy-fermion": ["intermetallic"],
  "borocarbide": ["boride", "intermetallic"],
  "carbon-clathrate": ["intermetallic"],
  "oxide-interface": ["oxide-perovskite"],
  "kagome": ["kagome-metal"],
  "Clathrate-H32": ["hydride"],
  "Kagome-variant": ["kagome-metal"],
  "Laves-C14": ["intermetallic"],
  "Laves-C15": ["intermetallic"],
  "HfFe6Ge6": ["kagome-metal", "intermetallic"],
  "CeCu2Si2": ["intermetallic"],
  "PuCoGa5-115": ["intermetallic"],
  "T-prime": ["cuprate", "oxide-perovskite"],
  "pyrite": ["chalcogenide"],
  "wurtzite": ["chalcogenide"],
  "antifluorite": ["intermetallic"],
};

function computeMotifValidityScore(context: PhysicsAwareRewardContext): number {
  if (!context.motifName) return 0;
  if (KNOWN_SC_MOTIFS.has(context.motifName)) return 1.0;
  const motifArray = Array.from(KNOWN_SC_MOTIFS);
  for (let i = 0; i < motifArray.length; i++) {
    if (context.motifName.toLowerCase().includes(motifArray[i].toLowerCase())) return 0.7;
  }
  return 0.2;
}

function computeFamilyConsistencyScore(context: PhysicsAwareRewardContext): number {
  if (!context.chemicalFamily && !context.motifName) return 0;
  if (context.chemicalFamily && KNOWN_FAMILIES.has(context.chemicalFamily)) {
    if (context.motifName) {
      const allowedFamilies = MOTIF_FAMILY_MAP[context.motifName];
      if (allowedFamilies && allowedFamilies.includes(context.chemicalFamily)) {
        return 1.0;
      }
      if (allowedFamilies) return 0.2;
    }
    return 0.6;
  }
  if (context.chemicalFamily) return 0.3;
  return 0;
}

function computeElectronCountStabilityScore(context: PhysicsAwareRewardContext): number {
  if (context.electronCountValid === true) return 1.0;
  if (context.electronCountValid === false) return 0.0;

  let score = 0.5;
  if (context.vecPerTM !== undefined) {
    if (context.vecPerTM > 12) return 0.0;
    if (context.vecPerTM >= 4 && context.vecPerTM <= 10) score = 0.8;
    else if (context.vecPerTM >= 2 && context.vecPerTM <= 12) score = 0.5;
    else score = 0.2;
  }
  if (context.dElectronCount !== undefined) {
    if (context.motifName) {
      const motifLower = context.motifName.toLowerCase();
      if (motifLower.includes("cuprate") || motifLower.includes("infinite-layer")) {
        if (context.dElectronCount === 9) score = Math.max(score, 1.0);
        else if (context.dElectronCount === 8 || context.dElectronCount === 10) score = Math.max(score, 0.6);
        else score = Math.min(score, 0.3);
      }
      if (motifLower.includes("pnictide") || motifLower === "thcr2si2") {
        if (context.dElectronCount === 6) score = Math.max(score, 1.0);
        else if (context.dElectronCount >= 5 && context.dElectronCount <= 7) score = Math.max(score, 0.7);
        else score = Math.min(score, 0.3);
      }
    }
  }
  return score;
}

function softmax(logits: number[], temperature: number = 1.0): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp((l - maxLogit) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function sampleFromDistribution(probs: number[]): number {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r <= cumulative) return i;
  }
  return probs.length - 1;
}

function stateToFeatures(state: RLState): number[] {
  return [
    state.bestTc / 400,
    state.avgRecentTc / 200,
    state.recentRewardTrend,
    state.familyDiversity / 20,
    Math.min(1, state.stagnationCycles / 30),
    state.explorationBudgetUsed,
    state.elementSuccessEntropy,
    Math.min(1, state.cycleNumber / 5000),
  ];
}

function computePhysicsPrincipleReward(context: PhysicsAwareRewardContext): number {
  let reward = 0;

  if (context.lambda !== undefined) {
    if (context.lambda >= 0.5 && context.lambda <= 3.0) {
      reward += 0.3 * Math.min(1, context.lambda / 2.0);
    }
    if (context.lambda > 3.5) {
      reward -= 0.2;
    }
  }

  if (context.metallicity !== undefined) {
    if (context.metallicity >= 0.5) {
      reward += 0.2 * context.metallicity;
    } else if (context.metallicity < 0.2) {
      reward -= 0.3;
    }
  }

  if (context.nestingScore !== undefined && context.nestingScore > 0.5) {
    reward += 0.15 * context.nestingScore;
  }

  if (context.vanHoveProximity !== undefined && context.vanHoveProximity > 0.7) {
    reward += 0.2 * context.vanHoveProximity;
  }

  if (context.dimensionality !== undefined) {
    if (context.dimensionality >= 1.5 && context.dimensionality <= 2.5) {
      reward += 0.15;
    }
  }

  if (context.hydrogenRatio !== undefined && context.hydrogenRatio >= 6) {
    const hBonus = Math.min(0.3, (context.hydrogenRatio - 5) * 0.05);
    reward += hBonus;
  }

  if (context.orbitalCharacter !== undefined) {
    if (context.orbitalCharacter === "d" || context.orbitalCharacter === "sd") {
      reward += 0.1;
    }
  }

  if (context.bandFlatness !== undefined && context.bandFlatness > 0.5) {
    reward += 0.1 * context.bandFlatness;
  }

  if (context.phononFrequency !== undefined) {
    if (context.phononFrequency > 500 && context.phononFrequency < 2000) {
      reward += 0.1;
    }
  }

  if (context.correlationStrength !== undefined) {
    if (context.correlationStrength > 0.3 && context.correlationStrength < 0.8) {
      reward += 0.1;
    } else if (context.correlationStrength >= 0.8) {
      reward -= 0.1;
    }
  }

  const motifScore = computeMotifValidityScore(context);
  reward += motifScore * 0.25;

  const familyScore = computeFamilyConsistencyScore(context);
  reward += familyScore * 0.2;

  const electronScore = computeElectronCountStabilityScore(context);
  reward += electronScore * 0.15;

  return reward;
}

function inferLayeringFromAction(action: RLAction): string {
  const structName = STRUCTURE_TYPES[action.structureType];
  const layeringChoice = LAYERING_DIMENSIONS[action.layeringDimension];
  const layeredStructures = ["Layered", "MX2", "ThCr2Si2"];
  const chainStructures = ["Kagome"];
  if (layeringChoice.dim === 2 || layeredStructures.includes(structName)) return "quasi-2D";
  if (layeringChoice.dim === 1 || chainStructures.includes(structName)) return "quasi-1D";
  if (layeringChoice.dim === 2.5) return "mixed-dim";
  return "3D-isotropic";
}

export class RLChemicalSpaceAgent {
  private policy: PolicyWeights;
  private replayBuffer: Experience[] = [];
  private maxReplaySize = 2000;
  private learningRate = 0.01;
  private gamma = 0.99;
  private epsilon = 0.15;
  private epsilonDecay = 0.9995;
  private minEpsilon = 0.05;
  private temperature = 1.0;
  private temperatureDecay = 0.999;
  private minTemperature = 0.3;
  private totalUpdates = 0;
  private elementSuccessRates: Map<string, { successes: number; total: number }> = new Map();
  private pairSuccessRates: Map<string, { successes: number; total: number; avgTc: number }> = new Map();
  private bestActionSequence: { action: RLAction; reward: number }[] = [];

  constructor() {
    this.policy = {
      elementGroup: new Array(ELEMENT_GROUPS.length).fill(0),
      stoichTemplate: new Array(STOICH_TEMPLATES.length).fill(0),
      structureType: new Array(STRUCTURE_TYPES.length).fill(0),
      elementPairBias: Array.from(
        { length: ELEMENT_GROUPS.length },
        () => new Array(ELEMENT_GROUPS.length).fill(0)
      ),
      elementPairSpecific: new Map<string, number>(),
      layeringDimension: new Array(LAYERING_DIMENSIONS.length).fill(0),
      hydrogenDensity: new Array(HYDROGEN_DENSITIES.length).fill(0),
      electronCount: new Array(ELECTRON_COUNTS.length).fill(0),
      orbitalConfiguration: new Array(ORBITAL_CONFIGS.length).fill(0),
      chemicalFamily: new Array(CHEMICAL_FAMILY_ACTIONS.length).fill(0),
    };

    this.initializePriors();
  }

  private makeElementPairKey(el1: string, el2: string): string {
    return [el1, el2].sort().join("-");
  }

  private initializePriors(): void {
    const tmBias = 0.3;
    this.policy.elementGroup[2] = tmBias;
    this.policy.elementGroup[3] = tmBias + 0.1;
    this.policy.elementGroup[5] = 0.2;
    this.policy.elementGroup[7] = 0.15;
    this.policy.elementGroup[8] = 0.1;

    this.policy.stoichTemplate[3] = 0.2;
    this.policy.stoichTemplate[4] = 0.15;
    this.policy.stoichTemplate[7] = 0.3;
    this.policy.stoichTemplate[8] = 0.2;

    this.policy.structureType[0] = 0.2;
    this.policy.structureType[3] = 0.15;
    this.policy.structureType[4] = 0.2;
    this.policy.structureType[8] = 0.1;

    this.policy.layeringDimension[0] = 0.1;
    this.policy.layeringDimension[1] = 0.25;
    this.policy.layeringDimension[3] = 0.1;

    this.policy.hydrogenDensity[0] = 0.0;
    this.policy.hydrogenDensity[2] = 0.2;
    this.policy.hydrogenDensity[3] = 0.3;

    this.policy.electronCount[1] = 0.15;
    this.policy.electronCount[2] = 0.2;

    this.policy.orbitalConfiguration[2] = 0.25;
    this.policy.orbitalConfiguration[5] = 0.15;

    this.policy.chemicalFamily[0] = 0.3;
    this.policy.chemicalFamily[1] = 0.2;
    this.policy.chemicalFamily[2] = 0.25;
    this.policy.chemicalFamily[3] = 0.2;
    this.policy.chemicalFamily[4] = 0.3;

    for (const prior of KNOWN_PAIR_PRIORS) {
      const key = this.makeElementPairKey(prior.el1, prior.el2);
      this.policy.elementPairSpecific.set(key, prior.bias);
    }
  }

  selectAction(state: RLState): RLAction {
    const stateFeatures = stateToFeatures(state);
    const stagnationBoost = Math.min(0.5, state.stagnationCycles * 0.02);

    const effectiveEpsilon = Math.min(this.epsilon + stagnationBoost, 0.5);

    if (Math.random() < effectiveEpsilon) {
      const famIdx = Math.floor(Math.random() * CHEMICAL_FAMILY_ACTIONS.length);
      return {
        chemicalFamily: famIdx,
        elementGroup1: Math.floor(Math.random() * ELEMENT_GROUPS.length),
        elementGroup2: Math.floor(Math.random() * ELEMENT_GROUPS.length),
        stoichTemplate: Math.floor(Math.random() * STOICH_TEMPLATES.length),
        structureType: Math.floor(Math.random() * STRUCTURE_TYPES.length),
        layeringDimension: Math.floor(Math.random() * LAYERING_DIMENSIONS.length),
        hydrogenDensity: Math.floor(Math.random() * HYDROGEN_DENSITIES.length),
        electronCount: Math.floor(Math.random() * ELECTRON_COUNTS.length),
        orbitalConfiguration: Math.floor(Math.random() * ORBITAL_CONFIGS.length),
      };
    }

    const contextBias = this.computeContextBias(stateFeatures);

    const famLogits = this.policy.chemicalFamily.map((w, i) => w + (contextBias.family?.[i] ?? 0));
    const famProbs = softmax(famLogits, this.temperature);
    const chemicalFamily = sampleFromDistribution(famProbs);

    const selectedFamily = CHEMICAL_FAMILY_ACTIONS[chemicalFamily];

    const hostSet = new Set(selectedFamily.hostGroups as readonly number[]);
    const anionSet = new Set(selectedFamily.anionGroups as readonly number[]);
    const structSet = new Set(selectedFamily.biasStructures as readonly number[]);

    const elLogits = this.policy.elementGroup.map((w, i) => {
      let bias = w + contextBias.element[i];
      if (hostSet.has(i)) bias += 0.4;
      return bias;
    });
    const elProbs = softmax(elLogits, this.temperature);
    const elementGroup1 = sampleFromDistribution(elProbs);

    const pairLogits = this.policy.elementPairBias[elementGroup1].map(
      (w, i) => {
        let bias = w + this.policy.elementGroup[i] + contextBias.element[i];
        if (anionSet.has(i)) bias += 0.3;
        return bias;
      }
    );
    const pairProbs = softmax(pairLogits, this.temperature);
    const elementGroup2 = sampleFromDistribution(pairProbs);

    const stoichLogits = this.policy.stoichTemplate.map((w, i) => w + contextBias.stoich[i]);
    const stoichProbs = softmax(stoichLogits, this.temperature);
    const stoichTemplate = sampleFromDistribution(stoichProbs);

    const structLogits = this.policy.structureType.map((w, i) => {
      let bias = w + contextBias.struct[i];
      if (structSet.has(i)) bias += 0.3;
      return bias;
    });
    const structProbs = softmax(structLogits, this.temperature);
    const structureType = sampleFromDistribution(structProbs);

    const layerLogits = this.policy.layeringDimension.map((w, i) => w + contextBias.layering[i]);
    const layerProbs = softmax(layerLogits, this.temperature);
    const layeringDimension = sampleFromDistribution(layerProbs);

    const hDensLogits = this.policy.hydrogenDensity.map((w, i) => {
      let bias = w + contextBias.hDensity[i];
      if (chemicalFamily === 0 && i >= 2) bias += 0.4;
      return bias;
    });
    const hDensProbs = softmax(hDensLogits, this.temperature);
    const hydrogenDensity = sampleFromDistribution(hDensProbs);

    const vecLogits = this.policy.electronCount.map((w, i) => w + contextBias.eCount[i]);
    const vecProbs = softmax(vecLogits, this.temperature);
    const electronCount = sampleFromDistribution(vecProbs);

    const orbLogits = this.policy.orbitalConfiguration.map((w, i) => w + contextBias.orbital[i]);
    const orbProbs = softmax(orbLogits, this.temperature);
    const orbitalConfiguration = sampleFromDistribution(orbProbs);

    return {
      chemicalFamily, elementGroup1, elementGroup2, stoichTemplate, structureType,
      layeringDimension, hydrogenDensity, electronCount, orbitalConfiguration,
    };
  }

  private computeContextBias(stateFeatures: number[]): {
    element: number[];
    stoich: number[];
    struct: number[];
    layering: number[];
    hDensity: number[];
    eCount: number[];
    orbital: number[];
    family: number[];
  } {
    const stagnation = stateFeatures[4];
    const bestTcNorm = stateFeatures[0];

    const elementBias = new Array(ELEMENT_GROUPS.length).fill(0);
    if (stagnation > 0.3) {
      for (let i = 0; i < elementBias.length; i++) {
        elementBias[i] += (Math.random() - 0.5) * stagnation * 0.5;
      }
    }

    for (const [key, stats] of this.elementSuccessRates) {
      const idx = ELEMENT_GROUPS.findIndex(g => g.name === key);
      if (idx >= 0 && stats.total > 5) {
        const successRate = stats.successes / stats.total;
        elementBias[idx] += (successRate - 0.5) * 0.3;
      }
    }

    for (let gi = 0; gi < ELEMENT_GROUPS.length; gi++) {
      let groupPairBoost = 0;
      let pairCount = 0;
      for (const el of ELEMENT_GROUPS[gi].elements) {
        for (const [pairKey, bias] of this.policy.elementPairSpecific) {
          const pairParts = pairKey.split("-");
          if ((pairParts[0] === el || pairParts[1] === el) && Math.abs(bias) > 0.05) {
            groupPairBoost += bias;
            pairCount++;
          }
        }
      }
      if (pairCount > 0) {
        elementBias[gi] += (groupPairBoost / pairCount) * 0.4;
      }
    }

    const stoichBias = new Array(STOICH_TEMPLATES.length).fill(0);
    if (bestTcNorm > 0.3) {
      stoichBias[7] += 0.2;
      stoichBias[8] += 0.15;
    }

    const structBias = new Array(STRUCTURE_TYPES.length).fill(0);

    const layeringBias = new Array(LAYERING_DIMENSIONS.length).fill(0);
    if (bestTcNorm > 0.2) {
      layeringBias[1] += 0.15;
    }
    if (stagnation > 0.4) {
      for (let i = 0; i < layeringBias.length; i++) {
        layeringBias[i] += (Math.random() - 0.5) * stagnation * 0.3;
      }
    }

    const hDensityBias = new Array(HYDROGEN_DENSITIES.length).fill(0);
    if (bestTcNorm > 0.4) {
      hDensityBias[2] += 0.15;
      hDensityBias[3] += 0.25;
    }

    const eCountBias = new Array(ELECTRON_COUNTS.length).fill(0);
    if (bestTcNorm > 0.15) {
      eCountBias[1] += 0.1;
      eCountBias[2] += 0.15;
    }

    const orbitalBias = new Array(ORBITAL_CONFIGS.length).fill(0);
    if (bestTcNorm > 0.1) {
      orbitalBias[2] += 0.15;
    }

    const familyBias = new Array(CHEMICAL_FAMILY_ACTIONS.length).fill(0);
    if (bestTcNorm > 0.3) {
      familyBias[0] += 0.2;
      familyBias[4] += 0.15;
    }
    if (stagnation > 0.3) {
      for (let i = 0; i < familyBias.length; i++) {
        familyBias[i] += (Math.random() - 0.5) * stagnation * 0.4;
      }
    }

    return {
      element: elementBias,
      stoich: stoichBias,
      struct: structBias,
      layering: layeringBias,
      hDensity: hDensityBias,
      eCount: eCountBias,
      orbital: orbitalBias,
      family: familyBias,
    };
  }

  updatePolicy(state: RLState, action: RLAction, reward: number): void {
    this.replayBuffer.push({
      state,
      action,
      reward,
      timestamp: Date.now(),
    });

    if (this.replayBuffer.length > this.maxReplaySize) {
      this.replayBuffer = this.replayBuffer.slice(-this.maxReplaySize);
    }

    if (reward > 0) {
      this.bestActionSequence.push({ action, reward });
      if (this.bestActionSequence.length > 50) {
        this.bestActionSequence.sort((a, b) => b.reward - a.reward);
        this.bestActionSequence = this.bestActionSequence.slice(0, 50);
      }
    }

    const lr = this.learningRate / (1 + this.totalUpdates * 0.0001);

    const advantageReward = reward - this.getBaselineReward();

    this.policy.elementGroup[action.elementGroup1] += lr * advantageReward;
    this.policy.elementGroup[action.elementGroup2] += lr * advantageReward * 0.7;
    this.policy.stoichTemplate[action.stoichTemplate] += lr * advantageReward;
    this.policy.structureType[action.structureType] += lr * advantageReward;

    this.policy.layeringDimension[action.layeringDimension] += lr * advantageReward * 0.8;
    this.policy.hydrogenDensity[action.hydrogenDensity] += lr * advantageReward * 0.9;
    this.policy.electronCount[action.electronCount] += lr * advantageReward * 0.7;
    this.policy.orbitalConfiguration[action.orbitalConfiguration] += lr * advantageReward * 0.7;

    if (action.chemicalFamily !== undefined && action.chemicalFamily < this.policy.chemicalFamily.length) {
      this.policy.chemicalFamily[action.chemicalFamily] += lr * advantageReward * 1.0;
    }

    this.policy.elementPairBias[action.elementGroup1][action.elementGroup2] += lr * advantageReward * 0.5;
    this.policy.elementPairBias[action.elementGroup2][action.elementGroup1] += lr * advantageReward * 0.5;

    if (this.totalUpdates % 20 === 0 && this.replayBuffer.length >= 32) {
      this.replayBatch(32);
    }

    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
    this.temperature = Math.max(this.minTemperature, this.temperature * this.temperatureDecay);
    this.totalUpdates++;
  }

  private getBaselineReward(): number {
    if (this.replayBuffer.length < 10) return 0;
    const recent = this.replayBuffer.slice(-50);
    return recent.reduce((s, e) => s + e.reward, 0) / recent.length;
  }

  private replayBatch(batchSize: number): void {
    const buffer = this.replayBuffer;
    const indices: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      indices.push(Math.floor(Math.random() * buffer.length));
    }

    const baseline = this.getBaselineReward();
    const lr = this.learningRate * 0.5 / (1 + this.totalUpdates * 0.0001);

    for (const idx of indices) {
      const exp = buffer[idx];
      const advantage = exp.reward - baseline;
      const decay = Math.pow(this.gamma, (buffer.length - idx) / buffer.length);

      this.policy.elementGroup[exp.action.elementGroup1] += lr * advantage * decay;
      this.policy.stoichTemplate[exp.action.stoichTemplate] += lr * advantage * decay;
      this.policy.structureType[exp.action.structureType] += lr * advantage * decay;
      this.policy.layeringDimension[exp.action.layeringDimension] += lr * advantage * decay * 0.8;
      this.policy.hydrogenDensity[exp.action.hydrogenDensity] += lr * advantage * decay * 0.9;
      this.policy.electronCount[exp.action.electronCount] += lr * advantage * decay * 0.7;
      this.policy.orbitalConfiguration[exp.action.orbitalConfiguration] += lr * advantage * decay * 0.7;
      if (exp.action.chemicalFamily !== undefined && exp.action.chemicalFamily < this.policy.chemicalFamily.length) {
        this.policy.chemicalFamily[exp.action.chemicalFamily] += lr * advantage * decay;
      }
    }
  }

  private rejectionCategoryCounts: Record<string, number> = {};

  recordElementOutcome(elements: string[], tc: number, passed: boolean, rejectCategory?: string): void {
    const safeTc = (tc != null && Number.isFinite(tc)) ? tc : 0;

    if (rejectCategory) {
      this.rejectionCategoryCounts[rejectCategory] = (this.rejectionCategoryCounts[rejectCategory] || 0) + 1;
    }

    for (const el of elements) {
      const group = ELEMENT_GROUPS.find(g => (g.elements as readonly string[]).includes(el));
      if (!group) continue;

      const stats = this.elementSuccessRates.get(group.name) || { successes: 0, total: 0 };
      stats.total++;
      if (passed || safeTc > 20) stats.successes++;
      this.elementSuccessRates.set(group.name, stats);
    }

    if (elements.length >= 2) {
      for (let i = 0; i < elements.length; i++) {
        for (let j = i + 1; j < elements.length; j++) {
          const pair = this.makeElementPairKey(elements[i], elements[j]);
          const stats = this.pairSuccessRates.get(pair) || { successes: 0, total: 0, avgTc: 0 };
          stats.total++;
          if (safeTc > 20) stats.successes++;
          stats.avgTc = (stats.avgTc * (stats.total - 1) + safeTc) / stats.total;
          this.pairSuccessRates.set(pair, stats);

          const currentBias = this.policy.elementPairSpecific.get(pair) ?? 0;
          const lr = this.learningRate * 0.3;
          if (tc > 50) {
            const tcBonus = Math.min(0.5, (tc - 50) / 400);
            this.policy.elementPairSpecific.set(pair, currentBias + lr * tcBonus);
          } else if (tc < 5 && stats.total > 5) {
            this.policy.elementPairSpecific.set(pair, Math.max(-0.3, currentBias - lr * 0.1));
          }

          if (rejectCategory === "chemistry_reject" || rejectCategory === "stability_reject") {
            this.policy.elementPairSpecific.set(pair, Math.max(-0.5, currentBias - lr * 0.15));
          }
        }
      }
    }
  }

  generateCandidatesFromAction(action: RLAction, count: number = 20): string[] {
    const group1 = ELEMENT_GROUPS[action.elementGroup1];
    const group2 = ELEMENT_GROUPS[action.elementGroup2];
    const template = STOICH_TEMPLATES[action.stoichTemplate];
    const hDensity = HYDROGEN_DENSITIES[action.hydrogenDensity];
    const vecTarget = ELECTRON_COUNTS[action.electronCount];
    const orbConfig = ORBITAL_CONFIGS[action.orbitalConfiguration];
    const layering = inferLayeringFromAction(action);

    const candidates: string[] = [];
    const seen = new Set<string>();

    const pairWeightedElements1 = this.getWeightedElements(group1.elements, group2.elements);
    const pairWeightedElements2 = this.getWeightedElements(group2.elements, group1.elements);

    for (let attempt = 0; attempt < count * 5 && candidates.length < count; attempt++) {
      const el1 = this.sampleWeightedElement(pairWeightedElements1, group1.elements);
      const el2 = this.sampleWeightedElement(pairWeightedElements2, group2.elements);
      if (el1 === el2) continue;

      let formula: string;
      const pattern = template.pattern;

      if (hDensity.ratio >= 4 && !pattern.includes("H")) {
        const hCount = Math.min(hDensity.ratio, 6);
        if (template.nElements === 2) {
          formula = `${el1}${el2}H${hCount}`;
        } else {
          formula = `${el1}${el2}H${hCount}`;
        }
      } else if (template.nElements === 2) {
        formula = applyBinaryPattern(el1, el2, pattern);
      } else if (template.nElements === 3) {
        let el3: string;
        if (hDensity.ratio >= 2) {
          el3 = "H";
        } else {
          const thirdGroupIdx = selectThirdGroupByOrbital(orbConfig.orbital, vecTarget.vec);
          const group3 = ELEMENT_GROUPS[thirdGroupIdx];
          el3 = group3.elements[Math.floor(Math.random() * group3.elements.length)];
        }
        if (el3 === el1 || el3 === el2) continue;
        formula = applyTernaryPattern(el1, el2, el3, pattern);
      } else {
        const g3 = ELEMENT_GROUPS[Math.floor(Math.random() * ELEMENT_GROUPS.length)];
        const g4 = ELEMENT_GROUPS[Math.floor(Math.random() * ELEMENT_GROUPS.length)];
        const el3 = g3.elements[Math.floor(Math.random() * g3.elements.length)];
        const el4 = g4.elements[Math.floor(Math.random() * g4.elements.length)];
        if (new Set([el1, el2, el3, el4]).size < 4) continue;
        formula = `${el1}${el2}${el3}${el4}`;
      }

      if (layering === "quasi-2D" && !formula.includes("O") && Math.random() < 0.3) {
        formula = formula + "O";
      }

      if (!seen.has(formula)) {
        seen.add(formula);
        candidates.push(formula);
      }
    }

    return candidates;
  }

  private getWeightedElements(
    elements: readonly string[],
    partnerElements: readonly string[]
  ): Map<string, number> {
    const weights = new Map<string, number>();
    for (const el of elements) {
      let w = 1.0;
      for (const partner of partnerElements) {
        const key = this.makeElementPairKey(el, partner);
        const pairBias = this.policy.elementPairSpecific.get(key) ?? 0;
        w += pairBias;
      }
      const pairStats = this.pairSuccessRates;
      for (const partner of partnerElements) {
        const key = this.makeElementPairKey(el, partner);
        const stats = pairStats.get(key);
        if (stats && stats.total >= 3) {
          const successRate = stats.successes / stats.total;
          w += (successRate - 0.3) * 0.5;
        }
      }
      weights.set(el, Math.max(0.1, w));
    }
    return weights;
  }

  private sampleWeightedElement(
    weights: Map<string, number>,
    elements: readonly string[]
  ): string {
    const totalWeight = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    let r = Math.random() * totalWeight;
    for (const el of elements) {
      const w = weights.get(el) ?? 1.0;
      r -= w;
      if (r <= 0) return el;
    }
    return elements[elements.length - 1];
  }

  computeReward(
    tcPredicted: number,
    bestTcBefore: number,
    pipelinePassed: boolean,
    stabilityScore: number,
    noveltyBonus: number = 0,
    physicsContext?: PhysicsAwareRewardContext
  ): number {
    const W_TC = 0.35;
    const W_STABILITY = 0.20;
    const W_MOTIF = 0.15;
    const W_ELECTRON_TOPOLOGY = 0.10;
    const W_NOVELTY = 0.10;
    const W_SYNTHESIS = 0.10;

    const safeTc = (tcPredicted != null && Number.isFinite(tcPredicted)) ? tcPredicted : 0;
    const safeBestBefore = (bestTcBefore != null && Number.isFinite(bestTcBefore)) ? bestTcBefore : 0;
    const tcNorm = Math.min(1, safeTc / 400);
    let tcScore = tcNorm * 2.0;
    if (safeTc > safeBestBefore) {
      const improvement = (safeTc - safeBestBefore) / Math.max(1, safeBestBefore);
      tcScore += improvement * 3.0;
    }
    if (pipelinePassed) {
      tcScore += 0.5;
    }
    if (safeTc < 5) {
      tcScore -= 0.5;
    }

    const stabilityComponent = stabilityScore;

    let motifScore = 0;
    if (physicsContext) {
      motifScore = computeMotifValidityScore(physicsContext);
      const familyScore = computeFamilyConsistencyScore(physicsContext);
      motifScore = motifScore * 0.6 + familyScore * 0.4;
    }

    let electronTopologyScore = 0;
    if (physicsContext) {
      const electronScore = computeElectronCountStabilityScore(physicsContext);
      let topologyBonus = 0;
      if (physicsContext.nestingScore !== undefined && physicsContext.nestingScore > 0.5) {
        topologyBonus += physicsContext.nestingScore * 0.3;
      }
      if (physicsContext.vanHoveProximity !== undefined && physicsContext.vanHoveProximity > 0.7) {
        topologyBonus += physicsContext.vanHoveProximity * 0.3;
      }
      if (physicsContext.bandFlatness !== undefined && physicsContext.bandFlatness > 0.5) {
        topologyBonus += physicsContext.bandFlatness * 0.2;
      }
      if (physicsContext.dimensionality !== undefined) {
        if (physicsContext.dimensionality >= 1.5 && physicsContext.dimensionality <= 2.5) {
          topologyBonus += 0.2;
        }
      }
      electronTopologyScore = electronScore * 0.5 + Math.min(1, topologyBonus) * 0.5;
    }

    const noveltyComponent = Math.min(1, noveltyBonus);

    let synthesisComponent = 0;
    if (physicsContext) {
      if (physicsContext.synthesisScore !== undefined) {
        synthesisComponent = Math.min(1, physicsContext.synthesisScore);
      } else {
        synthesisComponent = 0.5;
        if (physicsContext.lambda !== undefined && physicsContext.lambda >= 0.5 && physicsContext.lambda <= 3.0) {
          synthesisComponent += 0.15;
        }
        if (physicsContext.metallicity !== undefined && physicsContext.metallicity >= 0.5) {
          synthesisComponent += 0.1 * physicsContext.metallicity;
        }
        if (physicsContext.orbitalCharacter !== undefined) {
          if (physicsContext.orbitalCharacter === "d" || physicsContext.orbitalCharacter === "sd") {
            synthesisComponent += 0.1;
          }
        }
        if (physicsContext.phononFrequency !== undefined && physicsContext.phononFrequency > 500 && physicsContext.phononFrequency < 2000) {
          synthesisComponent += 0.1;
        }
        if (physicsContext.correlationStrength !== undefined) {
          if (physicsContext.correlationStrength > 0.3 && physicsContext.correlationStrength < 0.8) {
            synthesisComponent += 0.1;
          } else if (physicsContext.correlationStrength >= 0.8) {
            synthesisComponent -= 0.15;
          }
        }
        synthesisComponent = Math.min(1, Math.max(0, synthesisComponent));
      }
    }

    const reward =
      W_TC * tcScore +
      W_STABILITY * stabilityComponent +
      W_MOTIF * motifScore +
      W_ELECTRON_TOPOLOGY * electronTopologyScore +
      W_NOVELTY * noveltyComponent +
      W_SYNTHESIS * synthesisComponent;

    if (physicsContext) {
      const physicsBonus = computePhysicsPrincipleReward(physicsContext);
      return reward + physicsBonus * 0.15;
    }

    return reward;
  }

  getActionDescription(action: RLAction): string {
    const g1 = ELEMENT_GROUPS[action.elementGroup1].name;
    const g2 = ELEMENT_GROUPS[action.elementGroup2].name;
    const st = STOICH_TEMPLATES[action.stoichTemplate].name;
    const str = STRUCTURE_TYPES[action.structureType];
    const lay = LAYERING_DIMENSIONS[action.layeringDimension].name;
    const hd = HYDROGEN_DENSITIES[action.hydrogenDensity].name;
    const vec = ELECTRON_COUNTS[action.electronCount].name;
    const orb = ORBITAL_CONFIGS[action.orbitalConfiguration].name;
    return `${g1}+${g2} / ${st} / ${str} | dim=${lay} H=${hd} VEC=${vec} orb=${orb}`;
  }

  getStats(): {
    totalUpdates: number;
    epsilon: number;
    temperature: number;
    replayBufferSize: number;
    topElementGroups: { name: string; weight: number }[];
    topStoichTemplates: { name: string; weight: number }[];
    topStructureTypes: { name: string; weight: number }[];
    elementSuccessRates: { group: string; rate: number; total: number }[];
    topPairs: { pair: string; avgTc: number; count: number }[];
    recentAvgReward: number;
    topLayeringDimensions: { name: string; weight: number }[];
    topHydrogenDensities: { name: string; weight: number }[];
    topElectronCounts: { name: string; weight: number }[];
    topOrbitalConfigs: { name: string; weight: number }[];
    rejectionCategories: Record<string, number>;
  } {
    const elWeights = this.policy.elementGroup.map((w, i) => ({
      name: ELEMENT_GROUPS[i].name,
      weight: Math.round(w * 1000) / 1000,
    }));
    elWeights.sort((a, b) => b.weight - a.weight);

    const stWeights = this.policy.stoichTemplate.map((w, i) => ({
      name: STOICH_TEMPLATES[i].name,
      weight: Math.round(w * 1000) / 1000,
    }));
    stWeights.sort((a, b) => b.weight - a.weight);

    const strWeights = this.policy.structureType.map((w, i) => ({
      name: STRUCTURE_TYPES[i],
      weight: Math.round(w * 1000) / 1000,
    }));
    strWeights.sort((a, b) => b.weight - a.weight);

    const elSuccess: { group: string; rate: number; total: number }[] = [];
    for (const [group, stats] of this.elementSuccessRates) {
      elSuccess.push({
        group,
        rate: Math.round((stats.successes / Math.max(1, stats.total)) * 1000) / 1000,
        total: stats.total,
      });
    }
    elSuccess.sort((a, b) => b.rate - a.rate);

    const topPairs: { pair: string; avgTc: number; count: number }[] = [];
    for (const [pair, stats] of this.pairSuccessRates) {
      if (stats.total >= 3) {
        topPairs.push({
          pair,
          avgTc: Math.round(stats.avgTc * 10) / 10,
          count: stats.total,
        });
      }
    }
    topPairs.sort((a, b) => b.avgTc - a.avgTc);

    const recent = this.replayBuffer.slice(-50);
    const recentAvgReward = recent.length > 0
      ? recent.reduce((s, e) => s + e.reward, 0) / recent.length
      : 0;

    const layWeights = this.policy.layeringDimension.map((w, i) => ({
      name: LAYERING_DIMENSIONS[i].name,
      weight: Math.round(w * 1000) / 1000,
    }));
    layWeights.sort((a, b) => b.weight - a.weight);

    const hdWeights = this.policy.hydrogenDensity.map((w, i) => ({
      name: HYDROGEN_DENSITIES[i].name,
      weight: Math.round(w * 1000) / 1000,
    }));
    hdWeights.sort((a, b) => b.weight - a.weight);

    const ecWeights = this.policy.electronCount.map((w, i) => ({
      name: ELECTRON_COUNTS[i].name,
      weight: Math.round(w * 1000) / 1000,
    }));
    ecWeights.sort((a, b) => b.weight - a.weight);

    const orbWeights = this.policy.orbitalConfiguration.map((w, i) => ({
      name: ORBITAL_CONFIGS[i].name,
      weight: Math.round(w * 1000) / 1000,
    }));
    orbWeights.sort((a, b) => b.weight - a.weight);

    return {
      totalUpdates: this.totalUpdates,
      epsilon: Math.round(this.epsilon * 1000) / 1000,
      temperature: Math.round(this.temperature * 1000) / 1000,
      replayBufferSize: this.replayBuffer.length,
      topElementGroups: elWeights.slice(0, 5),
      topStoichTemplates: stWeights.slice(0, 5),
      topStructureTypes: strWeights.slice(0, 5),
      elementSuccessRates: elSuccess.slice(0, 5),
      topPairs: topPairs.slice(0, 10),
      recentAvgReward: Math.round(recentAvgReward * 1000) / 1000,
      topLayeringDimensions: layWeights,
      topHydrogenDensities: hdWeights,
      topElectronCounts: ecWeights,
      topOrbitalConfigs: orbWeights,
      rejectionCategories: { ...this.rejectionCategoryCounts },
    };
  }

  getElementGroups(): typeof ELEMENT_GROUPS { return ELEMENT_GROUPS; }
  getStoichTemplates(): typeof STOICH_TEMPLATES { return STOICH_TEMPLATES; }
  getStructureTypes(): typeof STRUCTURE_TYPES { return STRUCTURE_TYPES; }
  getLayeringDimensions(): typeof LAYERING_DIMENSIONS { return LAYERING_DIMENSIONS; }
  getHydrogenDensities(): typeof HYDROGEN_DENSITIES { return HYDROGEN_DENSITIES; }
  getElectronCounts(): typeof ELECTRON_COUNTS { return ELECTRON_COUNTS; }
  getOrbitalConfigs(): typeof ORBITAL_CONFIGS { return ORBITAL_CONFIGS; }
}

function selectThirdGroupByOrbital(orbitalPref: string, vecTarget: number): number {
  switch (orbitalPref) {
    case "d":
    case "sd":
      if (vecTarget >= 8) return 2;
      return 3;
    case "f":
      return 5;
    case "p":
    case "sp":
      return 6;
    case "s":
      return 0;
    default:
      return Math.floor(Math.random() * ELEMENT_GROUPS.length);
  }
}

function applyBinaryPattern(el1: string, el2: string, pattern: string): string {
  switch (pattern) {
    case "A3B": return `${el1}3${el2}`;
    case "AB": return `${el1}${el2}`;
    case "AB3": return `${el1}${el2}3`;
    case "AH3": return `${el1}${el2}3`;
    default: return `${el1}${el2}`;
  }
}

function applyTernaryPattern(el1: string, el2: string, el3: string, pattern: string): string {
  switch (pattern) {
    case "AB2C2": return `${el1}${el2}2${el3}2`;
    case "ABC3": return `${el1}${el2}${el3}3`;
    case "A2BC": return `${el1}2${el2}${el3}`;
    case "ABH4": return `${el1}${el2}${el3}4`;
    case "A2B3C": return `${el1}2${el2}3${el3}`;
    default: return `${el1}${el2}${el3}`;
  }
}

export const rlAgent = new RLChemicalSpaceAgent();
