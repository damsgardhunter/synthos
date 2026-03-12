export type InstructionType =
  | "create_lattice"
  | "set_symmetry"
  | "add_sublattice"
  | "populate_sites"
  | "add_hydrogen_cage"
  | "add_channel_network"
  | "add_intercalation_layer"
  | "add_charge_reservoir"
  | "apply_strain"
  | "optimize_dos"
  | "dope_sites"
  | "set_stoichiometry"
  | "add_interface"
  | "add_phonon_enhancer"
  | "apply_pressure_stabilization";

const INSTRUCTION_PRIORITY: Record<InstructionType, number> = {
  create_lattice: 0,
  set_symmetry: 0,
  populate_sites: 1,
  add_sublattice: 1,
  add_hydrogen_cage: 1,
  set_stoichiometry: 1,
  dope_sites: 2,
  add_interface: 2,
  add_charge_reservoir: 2,
  add_intercalation_layer: 2,
  add_channel_network: 2,
  apply_strain: 3,
  add_phonon_enhancer: 3,
  apply_pressure_stabilization: 3,
  optimize_dos: 4,
};

function reorderByPriority(instructions: ProgramInstruction[]): void {
  instructions.sort((a, b) => {
    const pa = INSTRUCTION_PRIORITY[a.type] ?? 3;
    const pb = INSTRUCTION_PRIORITY[b.type] ?? 3;
    if (pa !== pb) return pa - pb;
    return a.order - b.order;
  });
  for (let i = 0; i < instructions.length; i++) {
    instructions[i].order = i;
  }
}

const TRANSITION_METALS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt"];
const RARE_EARTHS = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"];
const LIGHT_METALS = ["Li", "Na", "Mg", "Al", "Ca", "K", "Sr", "Ba"];

function chemistryAwareFallback(existingElements: string[]): string {
  const hasTM = existingElements.some(el => TRANSITION_METALS.includes(el));
  const hasRE = existingElements.some(el => RARE_EARTHS.includes(el));
  const hasH = existingElements.includes("H");

  if (hasTM) {
    const pool = TRANSITION_METALS.filter(el => !existingElements.includes(el));
    if (pool.length > 0) return pool[Math.floor(Math.random() * pool.length)];
  }
  if (hasRE) {
    const pool = RARE_EARTHS.filter(el => !existingElements.includes(el));
    if (pool.length > 0) return pool[Math.floor(Math.random() * pool.length)];
  }
  if (hasH) {
    const pool = [...LIGHT_METALS, ...TRANSITION_METALS].filter(el => !existingElements.includes(el));
    if (pool.length > 0) return pool[Math.floor(Math.random() * pool.length)];
  }
  return "Y";
}

export interface ProgramInstruction {
  type: InstructionType;
  params: Record<string, number | string | string[] | boolean>;
  order: number;
}

export interface DesignProgram {
  id: string;
  name: string;
  instructions: ProgramInstruction[];
  outputFormula: string;
  outputPrototype: string;
  metadata: {
    complexity: number;
    expressiveness: number;
    generatedAt: number;
    parentId: string | null;
    generation: number;
    mutationHistory: string[];
  };
  featureVector: number[];
}

export interface ProgramExecutionResult {
  formula: string;
  prototype: string;
  elements: string[];
  stoichiometry: Record<string, number>;
  latticeType: string;
  symmetryGroup: string;
  hydrogenFraction: number;
  layerCount: number;
  channelDensity: number;
  strainApplied: number;
  dopingLevel: number;
  interfaceCount: number;
  pressureGpa: number;
  featureVector: number[];
  complexity: number;
  stabilityPenalty: number;
  predictedTc?: number;
  executionError?: string;
}

export type ComponentType =
  | "electron_source"
  | "phonon_mediator"
  | "charge_reservoir"
  | "structural_backbone"
  | "hydrogen_cage"
  | "intercalation_host"
  | "dopant_site"
  | "interface_layer"
  | "strain_buffer"
  | "topological_surface"
  | "pairing_channel"
  | "dos_enhancer";

export type EdgeType =
  | "bonding"
  | "electron_transfer"
  | "phonon_coupling"
  | "charge_transfer"
  | "structural"
  | "proximity"
  | "epitaxial"
  | "hybridization"
  | "sublattice"
  | "interlayer"
  | "pairing";

export interface GraphNode {
  id: string;
  type: ComponentType;
  element: string;
  properties: {
    electronCount: number;
    atomicMass: number;
    electronegativity: number;
    ionicRadius: number;
    oxidationState: number;
    orbitalCharacter: string;
  };
  position: [number, number];
  weight: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: EdgeType;
  strength: number;
  properties: {
    bondLength: number;
    overlapIntegral: number;
    couplingConstant: number;
  };
}

export interface DesignGraph {
  id: string;
  name: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  outputFormula: string;
  metadata: {
    nodeCount: number;
    edgeCount: number;
    connectivity: number;
    avgDegree: number;
    clusteringCoeff: number;
    generatedAt: number;
    parentId: string | null;
    generation: number;
  };
  embedding: number[];
}

export interface GraphAnalysis {
  centralNodes: { id: string; element: string; centrality: number }[];
  bottleneckEdges: { source: string; target: string; betweenness: number }[];
  communities: { id: number; members: string[]; avgWeight: number }[];
  pathLengths: { avg: number; max: number; diameter: number };
  spectralGap: number;
  isConnected: boolean;
}

export interface DesignRepresentationStats {
  programs: {
    total: number;
    avgComplexity: number;
    avgInstructions: number;
    bestFormula: string;
    bestTc: number;
    instructionFrequency: Record<string, number>;
    generationDistribution: Record<number, number>;
  };
  graphs: {
    total: number;
    avgNodes: number;
    avgEdges: number;
    avgConnectivity: number;
    bestFormula: string;
    bestTc: number;
    componentFrequency: Record<string, number>;
    edgeTypeFrequency: Record<string, number>;
  };
  crossRepresentation: {
    programToGraphConversions: number;
    graphToProgramConversions: number;
    avgFeatureCorrelation: number;
  };
}

const ELEMENT_DB: Record<string, { mass: number; eneg: number; radius: number; electrons: number; orbital: string }> = {
  H: { mass: 1.008, eneg: 2.20, radius: 0.25, electrons: 1, orbital: "s" },
  Li: { mass: 6.941, eneg: 0.98, radius: 1.45, electrons: 1, orbital: "s" },
  Be: { mass: 9.012, eneg: 1.57, radius: 1.05, electrons: 2, orbital: "s" },
  B: { mass: 10.81, eneg: 2.04, radius: 0.85, electrons: 3, orbital: "p" },
  C: { mass: 12.01, eneg: 2.55, radius: 0.70, electrons: 4, orbital: "p" },
  N: { mass: 14.01, eneg: 3.04, radius: 0.65, electrons: 5, orbital: "p" },
  O: { mass: 16.00, eneg: 3.44, radius: 0.60, electrons: 6, orbital: "p" },
  Al: { mass: 26.98, eneg: 1.61, radius: 1.25, electrons: 3, orbital: "p" },
  Si: { mass: 28.09, eneg: 1.90, radius: 1.10, electrons: 4, orbital: "p" },
  S: { mass: 32.07, eneg: 2.58, radius: 1.00, electrons: 6, orbital: "p" },
  Ca: { mass: 40.08, eneg: 1.00, radius: 1.80, electrons: 2, orbital: "s" },
  Sc: { mass: 44.96, eneg: 1.36, radius: 1.60, electrons: 1, orbital: "d" },
  Ti: { mass: 47.87, eneg: 1.54, radius: 1.40, electrons: 2, orbital: "d" },
  V: { mass: 50.94, eneg: 1.63, radius: 1.35, electrons: 3, orbital: "d" },
  Nb: { mass: 92.91, eneg: 1.60, radius: 1.45, electrons: 4, orbital: "d" },
  Mo: { mass: 95.94, eneg: 2.16, radius: 1.45, electrons: 5, orbital: "d" },
  Sr: { mass: 87.62, eneg: 0.95, radius: 2.00, electrons: 2, orbital: "s" },
  Y: { mass: 88.91, eneg: 1.22, radius: 1.80, electrons: 1, orbital: "d" },
  Zr: { mass: 91.22, eneg: 1.33, radius: 1.55, electrons: 2, orbital: "d" },
  Sn: { mass: 118.7, eneg: 1.96, radius: 1.45, electrons: 4, orbital: "p" },
  Ba: { mass: 137.3, eneg: 0.89, radius: 2.15, electrons: 2, orbital: "s" },
  La: { mass: 138.9, eneg: 1.10, radius: 1.95, electrons: 1, orbital: "d" },
  Ce: { mass: 140.1, eneg: 1.12, radius: 1.85, electrons: 1, orbital: "f" },
  Hf: { mass: 178.5, eneg: 1.30, radius: 1.55, electrons: 2, orbital: "d" },
  Ta: { mass: 180.9, eneg: 1.50, radius: 1.45, electrons: 3, orbital: "d" },
  W: { mass: 183.8, eneg: 2.36, radius: 1.35, electrons: 4, orbital: "d" },
  Re: { mass: 186.2, eneg: 1.90, radius: 1.35, electrons: 5, orbital: "d" },
  Bi: { mass: 209.0, eneg: 2.02, radius: 1.60, electrons: 5, orbital: "p" },
  Cu: { mass: 63.55, eneg: 1.90, radius: 1.35, electrons: 1, orbital: "d" },
  Fe: { mass: 55.85, eneg: 1.83, radius: 1.40, electrons: 2, orbital: "d" },
  Co: { mass: 58.93, eneg: 1.88, radius: 1.35, electrons: 2, orbital: "d" },
  Ni: { mass: 58.69, eneg: 1.91, radius: 1.35, electrons: 2, orbital: "d" },
  Se: { mass: 78.96, eneg: 2.55, radius: 1.15, electrons: 6, orbital: "p" },
  Te: { mass: 127.6, eneg: 2.10, radius: 1.40, electrons: 6, orbital: "p" },
  Sb: { mass: 121.8, eneg: 2.05, radius: 1.45, electrons: 5, orbital: "p" },
  As: { mass: 74.92, eneg: 2.18, radius: 1.15, electrons: 5, orbital: "p" },
  Ge: { mass: 72.63, eneg: 2.01, radius: 1.25, electrons: 4, orbital: "p" },
  Mg: { mass: 24.31, eneg: 1.31, radius: 1.50, electrons: 2, orbital: "s" },
  Eu: { mass: 152.0, eneg: 1.20, radius: 1.85, electrons: 2, orbital: "f" },
  Nd: { mass: 144.2, eneg: 1.14, radius: 1.85, electrons: 4, orbital: "f" },
  Pr: { mass: 140.9, eneg: 1.13, radius: 1.85, electrons: 3, orbital: "f" },
  Th: { mass: 232.0, eneg: 1.30, radius: 1.80, electrons: 2, orbital: "f" },
  Pb: { mass: 207.2, eneg: 2.33, radius: 1.75, electrons: 4, orbital: "p" },
  In: { mass: 114.8, eneg: 1.78, radius: 1.55, electrons: 3, orbital: "p" },
  Tl: { mass: 204.4, eneg: 1.62, radius: 1.70, electrons: 3, orbital: "p" },
  Hg: { mass: 200.6, eneg: 2.00, radius: 1.50, electrons: 2, orbital: "d" },
  P: { mass: 30.97, eneg: 2.19, radius: 1.00, electrons: 5, orbital: "p" },
  Cl: { mass: 35.45, eneg: 3.16, radius: 0.99, electrons: 7, orbital: "p" },
  Br: { mass: 79.90, eneg: 2.96, radius: 1.14, electrons: 7, orbital: "p" },
};

function getElementInfo(el: string) {
  return ELEMENT_DB[el] ?? { mass: 50, eneg: 1.5, radius: 1.2, electrons: 2, orbital: "d" };
}

const LATTICE_TYPES = ["cubic", "tetragonal", "hexagonal", "orthorhombic", "monoclinic", "triclinic", "rhombohedral"];
const SYMMETRY_GROUPS = ["Pm-3m", "Im-3m", "Fm-3m", "P6/mmm", "I4/mmm", "P4/nmm", "Cmcm", "P63/mmc", "R-3m"];

const LATTICE_SYMMETRY_MAP: Record<string, string[]> = {
  "cubic": ["Pm-3m", "Im-3m", "Fm-3m"],
  "tetragonal": ["I4/mmm", "P4/nmm"],
  "hexagonal": ["P6/mmm", "P63/mmc"],
  "orthorhombic": ["Cmcm"],
  "rhombohedral": ["R-3m"],
  "monoclinic": ["P2_1/c"],
  "triclinic": ["P-1"],
};

export function compatibleSymmetryGroups(latticeType: string): string[] {
  return LATTICE_SYMMETRY_MAP[latticeType] ?? SYMMETRY_GROUPS;
}

const CAGE_FIXED_COUNTS: Record<string, number> = {
  "sodalite": 10, "clathrate": 10, "pyritohedron": 12,
  "truncated-octahedron": 24, "icosahedron": 12,
};

function resolveHydrogenCageCount(cageType: string | undefined, currentCount: number): number {
  if (cageType && CAGE_FIXED_COUNTS[cageType]) {
    return CAGE_FIXED_COUNTS[cageType];
  }
  return Math.max(3, Math.min(16,
    Math.round(currentCount + (Math.random() - 0.5) * 4)));
}

const INSTRUCTION_TEMPLATES: Record<string, ProgramInstruction[]> = {
  "hydride-cage": [
    { type: "create_lattice", params: { latticeType: "cubic", a: 5.0 }, order: 0 },
    { type: "set_symmetry", params: { group: "Im-3m" }, order: 1 },
    { type: "populate_sites", params: { elements: ["La"], site: "vertex" }, order: 2 },
    { type: "add_sublattice", params: { elements: ["La"], pattern: "bcc-center", count: 1 }, order: 3 },
    { type: "add_hydrogen_cage", params: { count: 10, cageType: "clathrate", bondLength: 1.1 }, order: 4 },
    { type: "add_sublattice", params: { elements: ["H"], pattern: "cage-interstitial", count: 4 }, order: 5 },
    { type: "add_channel_network", params: { levels: 2, branchingFactor: 6, channelWidth: 1.0 }, order: 6 },
    { type: "add_intercalation_layer", params: { element: "Y", spacing: 2.8 }, order: 7 },
    { type: "add_charge_reservoir", params: { elements: ["La", "Y"], layers: 2 }, order: 8 },
    { type: "optimize_dos", params: { targetDOS: 5.0, method: "van-hove" }, order: 9 },
    { type: "add_phonon_enhancer", params: { mode: "H-stretching", frequency: 1200 }, order: 10 },
    { type: "add_sublattice", params: { elements: ["H"], pattern: "octahedral-interstitial", count: 3 }, order: 11 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 150, method: "chemical-precompression" }, order: 12 },
    { type: "add_interface", params: { type: "stabilization-layer", width: 1 }, order: 13 },
    { type: "apply_strain", params: { type: "hydrostatic", magnitude: 0.01 }, order: 14 },
    { type: "dope_sites", params: { dopant: "Y", fraction: 0.05, site: "vertex" }, order: 15 },
    { type: "add_phonon_enhancer", params: { mode: "cage-rattling", frequency: 800 }, order: 16 },
    { type: "add_sublattice", params: { elements: ["La"], pattern: "face-center", count: 2 }, order: 17 },
    { type: "optimize_dos", params: { targetDOS: 6.5, method: "flat-band" }, order: 18 },
    { type: "dope_sites", params: { dopant: "Ce", fraction: 0.03, site: "bcc-center" }, order: 19 },
    { type: "apply_strain", params: { type: "biaxial", magnitude: 0.005 }, order: 20 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 100, method: "lattice-stiffening" }, order: 21 },
    { type: "add_channel_network", params: { levels: 3, branchingFactor: 4, channelWidth: 0.9 }, order: 22 },
  ],
  "layered-cuprate": [
    { type: "create_lattice", params: { latticeType: "tetragonal", a: 3.9, c: 11.7 }, order: 0 },
    { type: "set_symmetry", params: { group: "I4/mmm" }, order: 1 },
    { type: "add_sublattice", params: { elements: ["Cu", "O"], pattern: "planar", count: 2 }, order: 2 },
    { type: "add_charge_reservoir", params: { elements: ["Ba", "La"], layers: 2 }, order: 3 },
    { type: "add_intercalation_layer", params: { element: "O", spacing: 2.4 }, order: 4 },
    { type: "add_sublattice", params: { elements: ["O"], pattern: "apical", count: 1 }, order: 5 },
    { type: "dope_sites", params: { dopant: "Sr", fraction: 0.15, site: "reservoir" }, order: 6 },
    { type: "add_interface", params: { type: "charge-transfer", width: 1 }, order: 7 },
    { type: "add_sublattice", params: { elements: ["Cu", "O"], pattern: "planar", count: 2 }, order: 8 },
    { type: "add_charge_reservoir", params: { elements: ["La"], layers: 1 }, order: 9 },
    { type: "add_intercalation_layer", params: { element: "O", spacing: 2.0 }, order: 10 },
    { type: "optimize_dos", params: { targetDOS: 8.0, method: "saddle-point" }, order: 11 },
    { type: "add_phonon_enhancer", params: { mode: "breathing", frequency: 400 }, order: 12 },
    { type: "add_interface", params: { type: "epitaxial", width: 1 }, order: 13 },
    { type: "apply_strain", params: { type: "biaxial", magnitude: 0.02 }, order: 14 },
    { type: "dope_sites", params: { dopant: "Ca", fraction: 0.10, site: "spacer" }, order: 15 },
    { type: "add_sublattice", params: { elements: ["Cu", "O"], pattern: "planar", count: 1 }, order: 16 },
    { type: "add_phonon_enhancer", params: { mode: "apical-oxygen", frequency: 500 }, order: 17 },
    { type: "optimize_dos", params: { targetDOS: 10.0, method: "flat-band" }, order: 18 },
    { type: "add_channel_network", params: { levels: 2, branchingFactor: 4, channelWidth: 1.2 }, order: 19 },
    { type: "dope_sites", params: { dopant: "Nd", fraction: 0.04, site: "reservoir" }, order: 20 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 5, method: "epitaxial-strain" }, order: 21 },
    { type: "apply_strain", params: { type: "uniaxial", magnitude: 0.008 }, order: 22 },
  ],
  "a15-compound": [
    { type: "create_lattice", params: { latticeType: "cubic", a: 5.3 }, order: 0 },
    { type: "set_symmetry", params: { group: "Pm-3m" }, order: 1 },
    { type: "populate_sites", params: { elements: ["Nb"], site: "chain" }, order: 2 },
    { type: "populate_sites", params: { elements: ["Sn"], site: "body-center" }, order: 3 },
    { type: "add_sublattice", params: { elements: ["Nb"], pattern: "1d-chain", count: 3 }, order: 4 },
    { type: "add_sublattice", params: { elements: ["Nb"], pattern: "1d-chain-y", count: 3 }, order: 5 },
    { type: "add_sublattice", params: { elements: ["Nb"], pattern: "1d-chain-z", count: 3 }, order: 6 },
    { type: "add_phonon_enhancer", params: { mode: "soft-phonon", frequency: 80 }, order: 7 },
    { type: "optimize_dos", params: { targetDOS: 6.0, method: "van-hove" }, order: 8 },
    { type: "add_channel_network", params: { levels: 2, branchingFactor: 3, channelWidth: 1.5 }, order: 9 },
    { type: "add_charge_reservoir", params: { elements: ["Sn"], layers: 1 }, order: 10 },
    { type: "add_interface", params: { type: "chain-coupling", width: 1 }, order: 11 },
    { type: "apply_strain", params: { type: "uniaxial", magnitude: 0.015 }, order: 12 },
    { type: "dope_sites", params: { dopant: "Al", fraction: 0.08, site: "body-center" }, order: 13 },
    { type: "add_phonon_enhancer", params: { mode: "martensitic-precursor", frequency: 120 }, order: 14 },
    { type: "add_sublattice", params: { elements: ["Sn"], pattern: "corner", count: 1 }, order: 15 },
    { type: "optimize_dos", params: { targetDOS: 7.5, method: "nesting-peak" }, order: 16 },
    { type: "dope_sites", params: { dopant: "Ge", fraction: 0.05, site: "chain" }, order: 17 },
    { type: "apply_strain", params: { type: "hydrostatic", magnitude: 0.01 }, order: 18 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 20, method: "chemical-precompression" }, order: 19 },
    { type: "add_interface", params: { type: "anti-phase-boundary", width: 1 }, order: 20 },
    { type: "add_channel_network", params: { levels: 3, branchingFactor: 3, channelWidth: 1.2 }, order: 21 },
  ],
  "hea-superconductor": [
    { type: "create_lattice", params: { latticeType: "cubic", a: 3.2 }, order: 0 },
    { type: "set_symmetry", params: { group: "Im-3m" }, order: 1 },
    { type: "populate_sites", params: { elements: ["Nb", "Ti", "Zr", "Hf", "V"], site: "random" }, order: 2 },
    { type: "add_sublattice", params: { elements: ["Nb", "V"], pattern: "bcc-center", count: 2 }, order: 3 },
    { type: "add_sublattice", params: { elements: ["Ti", "Zr"], pattern: "bcc-corner", count: 2 }, order: 4 },
    { type: "add_sublattice", params: { elements: ["Hf"], pattern: "interstitial", count: 1 }, order: 5 },
    { type: "apply_strain", params: { type: "hydrostatic", magnitude: 0.02 }, order: 6 },
    { type: "optimize_dos", params: { targetDOS: 4.0, method: "disorder-enhanced" }, order: 7 },
    { type: "add_phonon_enhancer", params: { mode: "disorder-softening", frequency: 200 }, order: 8 },
    { type: "add_charge_reservoir", params: { elements: ["Nb", "V"], layers: 1 }, order: 9 },
    { type: "dope_sites", params: { dopant: "Mo", fraction: 0.12, site: "random" }, order: 10 },
    { type: "add_interface", params: { type: "grain-boundary", width: 2 }, order: 11 },
    { type: "add_channel_network", params: { levels: 3, branchingFactor: 4, channelWidth: 1.0 }, order: 12 },
    { type: "add_intercalation_layer", params: { element: "N", spacing: 1.8 }, order: 13 },
    { type: "apply_strain", params: { type: "biaxial", magnitude: 0.01 }, order: 14 },
    { type: "add_phonon_enhancer", params: { mode: "lattice-anharmonicity", frequency: 160 }, order: 15 },
    { type: "dope_sites", params: { dopant: "Ta", fraction: 0.06, site: "random" }, order: 16 },
    { type: "optimize_dos", params: { targetDOS: 5.5, method: "cocktail-effect" }, order: 17 },
    { type: "add_interface", params: { type: "phase-boundary", width: 1 }, order: 18 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 15, method: "chemical-precompression" }, order: 19 },
    { type: "add_sublattice", params: { elements: ["Nb"], pattern: "cluster-center", count: 1 }, order: 20 },
    { type: "apply_strain", params: { type: "uniaxial", magnitude: 0.008 }, order: 21 },
  ],
  "topological-sc": [
    { type: "create_lattice", params: { latticeType: "rhombohedral", a: 4.14, c: 28.6 }, order: 0 },
    { type: "set_symmetry", params: { group: "R-3m" }, order: 1 },
    { type: "add_sublattice", params: { elements: ["Bi", "Se"], pattern: "quintuple-layer", count: 3 }, order: 2 },
    { type: "add_sublattice", params: { elements: ["Bi", "Se"], pattern: "quintuple-layer", count: 2 }, order: 3 },
    { type: "add_interface", params: { type: "topological-surface", width: 2 }, order: 4 },
    { type: "add_intercalation_layer", params: { element: "Cu", spacing: 3.2 }, order: 5 },
    { type: "dope_sites", params: { dopant: "Cu", fraction: 0.10, site: "intercalated" }, order: 6 },
    { type: "add_sublattice", params: { elements: ["Se"], pattern: "vdw-gap", count: 1 }, order: 7 },
    { type: "add_charge_reservoir", params: { elements: ["Bi"], layers: 1 }, order: 8 },
    { type: "add_intercalation_layer", params: { element: "Nb", spacing: 3.0 }, order: 9 },
    { type: "optimize_dos", params: { targetDOS: 3.0, method: "dirac-cone" }, order: 10 },
    { type: "add_phonon_enhancer", params: { mode: "surface-phonon", frequency: 150 }, order: 11 },
    { type: "add_interface", params: { type: "topological-bulk-boundary", width: 1 }, order: 12 },
    { type: "add_sublattice", params: { elements: ["Te"], pattern: "substitutional", count: 1 }, order: 13 },
    { type: "apply_strain", params: { type: "biaxial", magnitude: 0.03 }, order: 14 },
    { type: "add_charge_reservoir", params: { elements: ["Sr"], layers: 1 }, order: 15 },
    { type: "dope_sites", params: { dopant: "Nb", fraction: 0.06, site: "vdw-gap" }, order: 16 },
    { type: "add_channel_network", params: { levels: 2, branchingFactor: 3, channelWidth: 1.0 }, order: 17 },
    { type: "optimize_dos", params: { targetDOS: 4.0, method: "spin-orbit-enhanced" }, order: 18 },
    { type: "add_phonon_enhancer", params: { mode: "Kohn-anomaly", frequency: 100 }, order: 19 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 8, method: "chemical-precompression" }, order: 20 },
    { type: "apply_strain", params: { type: "uniaxial", magnitude: 0.01 }, order: 21 },
  ],
  "phonon-optimized": [
    { type: "create_lattice", params: { latticeType: "hexagonal", a: 3.1, c: 3.5 }, order: 0 },
    { type: "set_symmetry", params: { group: "P6/mmm" }, order: 1 },
    { type: "populate_sites", params: { elements: ["Mg"], site: "vertex" }, order: 2 },
    { type: "populate_sites", params: { elements: ["B"], site: "honeycomb" }, order: 3 },
    { type: "add_sublattice", params: { elements: ["B"], pattern: "graphene-like", count: 2 }, order: 4 },
    { type: "add_phonon_enhancer", params: { mode: "E2g", frequency: 600 }, order: 5 },
    { type: "add_sublattice", params: { elements: ["B"], pattern: "graphene-like", count: 2 }, order: 6 },
    { type: "add_intercalation_layer", params: { element: "Mg", spacing: 3.5 }, order: 7 },
    { type: "add_charge_reservoir", params: { elements: ["Mg"], layers: 1 }, order: 8 },
    { type: "optimize_dos", params: { targetDOS: 7.0, method: "sigma-band" }, order: 9 },
    { type: "add_interface", params: { type: "substrate-coupling", width: 1 }, order: 10 },
    { type: "add_phonon_enhancer", params: { mode: "bond-stretching", frequency: 700 }, order: 11 },
    { type: "add_channel_network", params: { levels: 2, branchingFactor: 6, channelWidth: 0.8 }, order: 12 },
    { type: "apply_strain", params: { type: "biaxial", magnitude: 0.025 }, order: 13 },
    { type: "dope_sites", params: { dopant: "Al", fraction: 0.06, site: "vertex" }, order: 14 },
    { type: "add_sublattice", params: { elements: ["C"], pattern: "substitutional", count: 1 }, order: 15 },
    { type: "add_phonon_enhancer", params: { mode: "acoustic-optical-coupling", frequency: 450 }, order: 16 },
    { type: "optimize_dos", params: { targetDOS: 8.5, method: "two-gap" }, order: 17 },
    { type: "dope_sites", params: { dopant: "Li", fraction: 0.04, site: "intercalated" }, order: 18 },
    { type: "apply_strain", params: { type: "hydrostatic", magnitude: 0.01 }, order: 19 },
    { type: "add_interface", params: { type: "hetero-bilayer", width: 1 }, order: 20 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 10, method: "lattice-stiffening" }, order: 21 },
  ],
  "pressure-hydride": [
    { type: "create_lattice", params: { latticeType: "cubic", a: 3.7 }, order: 0 },
    { type: "set_symmetry", params: { group: "Fm-3m" }, order: 1 },
    { type: "populate_sites", params: { elements: ["La"], site: "fcc" }, order: 2 },
    { type: "add_hydrogen_cage", params: { count: 10, cageType: "sodalite", bondLength: 0.98 }, order: 3 },
    { type: "add_sublattice", params: { elements: ["H"], pattern: "cage-interstitial", count: 4 }, order: 4 },
    { type: "add_sublattice", params: { elements: ["H"], pattern: "octahedral-void", count: 3 }, order: 5 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 150, method: "chemical-precompression" }, order: 6 },
    { type: "add_channel_network", params: { levels: 3, branchingFactor: 4, channelWidth: 1.2 }, order: 7 },
    { type: "add_intercalation_layer", params: { element: "Y", spacing: 2.5 }, order: 8 },
    { type: "add_charge_reservoir", params: { elements: ["La", "Y"], layers: 2 }, order: 9 },
    { type: "optimize_dos", params: { targetDOS: 6.0, method: "van-hove" }, order: 10 },
    { type: "add_phonon_enhancer", params: { mode: "H-stretching", frequency: 1500 }, order: 11 },
    { type: "add_interface", params: { type: "stabilization-layer", width: 1 }, order: 12 },
    { type: "add_sublattice", params: { elements: ["H"], pattern: "tetrahedral-void", count: 2 }, order: 13 },
    { type: "add_phonon_enhancer", params: { mode: "H-libration", frequency: 900 }, order: 14 },
    { type: "apply_strain", params: { type: "isotropic", magnitude: 0.015 }, order: 15 },
    { type: "dope_sites", params: { dopant: "Ce", fraction: 0.08, site: "fcc" }, order: 16 },
    { type: "optimize_dos", params: { targetDOS: 7.0, method: "flat-band" }, order: 17 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 120, method: "hydrogen-bonding-network" }, order: 18 },
    { type: "dope_sites", params: { dopant: "Th", fraction: 0.03, site: "fcc" }, order: 19 },
    { type: "add_channel_network", params: { levels: 4, branchingFactor: 6, channelWidth: 0.9 }, order: 20 },
    { type: "apply_strain", params: { type: "biaxial", magnitude: 0.008 }, order: 21 },
    { type: "add_interface", params: { type: "caging-boundary", width: 2 }, order: 22 },
  ],
  "charge-transfer": [
    { type: "create_lattice", params: { latticeType: "orthorhombic", a: 3.82, b: 3.89, c: 11.68 }, order: 0 },
    { type: "set_symmetry", params: { group: "Cmcm" }, order: 1 },
    { type: "add_sublattice", params: { elements: ["Cu", "O"], pattern: "planar", count: 2 }, order: 2 },
    { type: "add_charge_reservoir", params: { elements: ["Y", "Ba"], layers: 3 }, order: 3 },
    { type: "add_intercalation_layer", params: { element: "O", spacing: 2.0 }, order: 4 },
    { type: "add_sublattice", params: { elements: ["O"], pattern: "chain", count: 1 }, order: 5 },
    { type: "add_sublattice", params: { elements: ["Cu", "O"], pattern: "planar", count: 2 }, order: 6 },
    { type: "add_charge_reservoir", params: { elements: ["Ba"], layers: 2 }, order: 7 },
    { type: "add_intercalation_layer", params: { element: "O", spacing: 2.2 }, order: 8 },
    { type: "optimize_dos", params: { targetDOS: 6.0, method: "saddle-point" }, order: 9 },
    { type: "add_interface", params: { type: "charge-transfer", width: 2 }, order: 10 },
    { type: "dope_sites", params: { dopant: "Ca", fraction: 0.12, site: "reservoir" }, order: 11 },
    { type: "add_phonon_enhancer", params: { mode: "apical-oxygen", frequency: 350 }, order: 12 },
    { type: "add_interface", params: { type: "block-layer", width: 1 }, order: 13 },
    { type: "apply_strain", params: { type: "uniaxial", magnitude: 0.02 }, order: 14 },
    { type: "add_channel_network", params: { levels: 2, branchingFactor: 4, channelWidth: 1.4 }, order: 15 },
    { type: "add_phonon_enhancer", params: { mode: "buckling", frequency: 280 }, order: 16 },
    { type: "dope_sites", params: { dopant: "Sr", fraction: 0.08, site: "spacer" }, order: 17 },
    { type: "optimize_dos", params: { targetDOS: 7.5, method: "nesting-peak" }, order: 18 },
    { type: "add_sublattice", params: { elements: ["O"], pattern: "apical", count: 1 }, order: 19 },
    { type: "apply_strain", params: { type: "biaxial", magnitude: 0.01 }, order: 20 },
    { type: "apply_pressure_stabilization", params: { targetPressure: 3, method: "epitaxial-strain" }, order: 21 },
    { type: "dope_sites", params: { dopant: "La", fraction: 0.04, site: "reservoir" }, order: 22 },
  ],
};

export function generateDesignProgram(
  strategyType: string,
  elementPool: string[],
  generation: number = 0,
  parentId: string | null = null,
): DesignProgram {
  const templateKey = selectTemplateForStrategy(strategyType);
  const templateInstructions = INSTRUCTION_TEMPLATES[templateKey] ?? INSTRUCTION_TEMPLATES["hydride-cage"];

  const instructions = templateInstructions.map(inst => {
    const mutated = JSON.parse(JSON.stringify(inst)) as ProgramInstruction;

    if (mutated.type === "populate_sites" && Array.isArray(mutated.params.elements)) {
      const origElements = mutated.params.elements as string[];
      const isHEA = templateKey === "hea-superconductor";
      const usedElements = new Set<string>();
      const newElements = origElements.map(el => {
        if (Math.random() < 0.4 && elementPool.length > 0) {
          let candidates = elementPool.filter(e => {
            const orig = getElementInfo(el);
            const candidate = getElementInfo(e);
            return Math.abs(orig.radius - candidate.radius) < 0.25 &&
              Math.abs(orig.eneg - candidate.eneg) < 0.8;
          });
          if (isHEA) {
            const unique = candidates.filter(e => !usedElements.has(e));
            if (unique.length > 0) candidates = unique;
          }
          if (candidates.length > 0) {
            const picked = candidates[Math.floor(Math.random() * candidates.length)];
            usedElements.add(picked);
            return picked;
          }
        }
        usedElements.add(el);
        return el;
      });
      mutated.params.elements = newElements;
    }

    if (mutated.type === "dope_sites" && typeof mutated.params.dopant === "string") {
      if (Math.random() < 0.3 && elementPool.length > 0) {
        mutated.params.dopant = elementPool[Math.floor(Math.random() * elementPool.length)];
      }
      mutated.params.fraction = Math.max(0.01, Math.min(0.30,
        ((mutated.params.fraction as number) || 0.1) + (Math.random() - 0.5) * 0.05));
    }

    if (mutated.type === "add_hydrogen_cage" && typeof mutated.params.count === "number") {
      mutated.params.count = resolveHydrogenCageCount(
        mutated.params.cageType as string | undefined,
        mutated.params.count as number
      );
    }

    if (mutated.type === "add_channel_network") {
      const levels = typeof mutated.params.levels === "number" ? mutated.params.levels : 2;
      const branching = typeof mutated.params.branchingFactor === "number" ? mutated.params.branchingFactor : 3;
      const width = typeof mutated.params.channelWidth === "number" ? mutated.params.channelWidth : 1.0;
      mutated.params.levels = Math.max(1, Math.min(5,
        Math.round(levels + (Math.random() - 0.5) * 2)));
      mutated.params.branchingFactor = Math.max(2, Math.min(6,
        Math.round(branching + (Math.random() - 0.5) * 2)));
      mutated.params.channelWidth = Math.max(0.3, Math.min(2.5, width + (Math.random() - 0.5) * 0.4));
    }

    return mutated;
  });

  if (Math.random() < 0.4 && generation > 0) {
    const templateElements = templateInstructions
      .flatMap(inst => {
        const els: string[] = [];
        if (Array.isArray(inst.params.elements)) els.push(...(inst.params.elements as string[]));
        if (typeof inst.params.element === "string") els.push(inst.params.element);
        if (typeof inst.params.dopant === "string") els.push(inst.params.dopant);
        return els;
      })
      .filter((v, i, a) => a.indexOf(v) === i);
    const pickElement = (pool: string[]): string => {
      if (pool.length > 0) return pool[Math.floor(Math.random() * pool.length)];
      if (templateElements.length > 0) return templateElements[Math.floor(Math.random() * templateElements.length)];
      return chemistryAwareFallback(templateElements);
    };

    const templateLattice = templateInstructions.find(i => i.type === "create_lattice")?.params.latticeType as string | undefined;
    const isLayered = templateLattice === "tetragonal" || templateLattice === "hexagonal" || templateLattice === "orthorhombic" || templateLattice === "rhombohedral";
    const maxInterfaceWidth = isLayered ? 3 : 2;

    const extraInstructions: ProgramInstruction[] = [
      { type: "apply_strain", params: { type: "biaxial", magnitude: 0.01 + Math.random() * 0.04 }, order: 0 },
      { type: "add_interface", params: { type: "heterostructure", width: 1 + Math.floor(Math.random() * maxInterfaceWidth) }, order: 0 },
      { type: "add_phonon_enhancer", params: { mode: "soft-phonon", frequency: 50 + Math.random() * 200 }, order: 0 },
      { type: "add_sublattice", params: { elements: [pickElement(elementPool)], pattern: "interstitial", count: 1 + Math.floor(Math.random() * 3) }, order: 0 },
      { type: "add_intercalation_layer", params: { element: pickElement(elementPool), spacing: 1.5 + Math.random() * 2.5 }, order: 0 },
      { type: "add_charge_reservoir", params: { elements: [pickElement(elementPool)], layers: 1 + Math.floor(Math.random() * 2) }, order: 0 },
    ];
    const shuffled = extraInstructions.sort(() => Math.random() - 0.5);
    const numExtra = Math.random() < 0.4 ? 3 : 2;
    for (let i = 0; i < Math.min(numExtra, shuffled.length); i++) {
      instructions.push({ ...shuffled[i], order: instructions.length });
    }
    reorderByPriority(instructions);
  }

  const result = executeDesignProgram({ id: "", name: "", instructions, outputFormula: "", outputPrototype: "", metadata: { complexity: 0, expressiveness: 0, generatedAt: 0, parentId: null, generation: 0, mutationHistory: [] }, featureVector: [] });

  const program: DesignProgram = {
    id: `prog-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    name: `${templateKey}-gen${generation}`,
    instructions,
    outputFormula: result.formula,
    outputPrototype: result.prototype,
    metadata: {
      complexity: result.complexity,
      expressiveness: instructions.length / 20,
      generatedAt: Date.now(),
      parentId,
      generation,
      mutationHistory: [],
    },
    featureVector: result.featureVector,
  };

  return program;
}

function selectTemplateForStrategy(strategyType: string): string {
  const mapping: Record<string, string[]> = {
    "hydride-cage-optimizer": ["hydride-cage", "pressure-hydride"],
    "layered-intercalation": ["layered-cuprate", "charge-transfer"],
    "high-entropy-alloy": ["hea-superconductor", "a15-compound"],
    "light-element-phonon": ["phonon-optimized", "a15-compound"],
    "topological-edge": ["topological-sc"],
    "pressure-stabilized": ["pressure-hydride", "hydride-cage"],
    "electron-phonon-resonance": ["a15-compound", "phonon-optimized"],
    "charge-transfer-layer": ["charge-transfer", "layered-cuprate"],

    "conventional-BCS": ["a15-compound", "phonon-optimized"],
    "strong-coupling-metal": ["a15-compound", "hea-superconductor"],
    "light-element-compound": ["phonon-optimized", "a15-compound"],
    "hydride-moderate-pressure": ["hydride-cage", "pressure-hydride"],
    "superhydride": ["pressure-hydride", "hydride-cage"],
    "unconventional-layered": ["layered-cuprate", "charge-transfer"],
    "kagome-flat-band": ["hea-superconductor", "a15-compound"],
    "topological-SC": ["topological-sc"],
  };

  const candidates = mapping[strategyType] ?? Object.keys(INSTRUCTION_TEMPLATES);
  return candidates[Math.floor(Math.random() * candidates.length)];
}

export function executeDesignProgram(program: DesignProgram): ProgramExecutionResult {
  const elements: string[] = [];
  const stoichiometry: Record<string, number> = {};
  let latticeType = "cubic";
  let symmetryGroup = "Pm-3m";
  let hydrogenFraction = 0;
  let layerCount = 0;
  let channelDensity = 0;
  let strainApplied = 0;
  let dopingLevel = 0;
  let interfaceCount = 0;
  let pressureGpa = 0;

  for (const inst of program.instructions.sort((a, b) => a.order - b.order)) {
    switch (inst.type) {
      case "create_lattice":
        latticeType = (inst.params.latticeType as string) || "cubic";
        break;
      case "set_symmetry":
        symmetryGroup = (inst.params.group as string) || "Pm-3m";
        break;
      case "add_sublattice":
      case "populate_sites": {
        const els = (inst.params.elements as string[]) || [];
        for (const el of els) {
          if (!elements.includes(el)) elements.push(el);
          stoichiometry[el] = (stoichiometry[el] || 0) + (inst.params.count as number || 1);
        }
        break;
      }
      case "add_hydrogen_cage": {
        if (!elements.includes("H")) elements.push("H");
        const hCount = (inst.params.count as number) || 6;
        stoichiometry["H"] = (stoichiometry["H"] || 0) + hCount;
        break;
      }
      case "add_channel_network":
        channelDensity = ((inst.params.levels as number) || 3) * ((inst.params.branchingFactor as number) || 3) / 10;
        break;
      case "add_intercalation_layer": {
        layerCount++;
        const intEl = (inst.params.element as string) || "O";
        if (!elements.includes(intEl)) elements.push(intEl);
        const intSpacing = (inst.params.spacing as number) || 2.0;
        const intCount = Math.max(1, Math.round(3.0 / intSpacing));
        stoichiometry[intEl] = (stoichiometry[intEl] || 0) + intCount;
        break;
      }
      case "add_charge_reservoir": {
        const resEls = (inst.params.elements as string[]) || [];
        const resLayers = (inst.params.layers as number) || 1;
        layerCount += resLayers;
        for (const el of resEls) {
          if (!elements.includes(el)) elements.push(el);
          stoichiometry[el] = (stoichiometry[el] || 0) + resLayers;
        }
        break;
      }
      case "apply_strain":
        strainApplied = (inst.params.magnitude as number) || 0.01;
        break;
      case "dope_sites": {
        const dopant = (inst.params.dopant as string) || "Sr";
        if (!elements.includes(dopant)) elements.push(dopant);
        dopingLevel = (inst.params.fraction as number) || 0.10;
        const nonHAtoms = Object.entries(stoichiometry)
          .filter(([el]) => el !== "H" && el !== dopant)
          .reduce((s, [, c]) => s + c, 0);
        const hostBasis = Math.max(1, nonHAtoms);
        const dopantCount = Math.max(1, Math.round(dopingLevel * hostBasis));
        stoichiometry[dopant] = (stoichiometry[dopant] || 0) + dopantCount;
        break;
      }
      case "add_interface":
        interfaceCount++;
        break;
      case "apply_pressure_stabilization":
        pressureGpa = Math.max(pressureGpa, (inst.params.targetPressure as number) || 0);
        break;
      case "add_phonon_enhancer":
      case "optimize_dos":
      case "set_stoichiometry":
        break;
    }
  }

  let executionError: string | undefined;
  if (elements.length === 0) {
    executionError = "No element-producing instructions in design program";
    return {
      formula: "",
      prototype: symmetryGroup,
      elements: [],
      stoichiometry: {},
      latticeType,
      symmetryGroup,
      hydrogenFraction: 0,
      layerCount,
      channelDensity,
      strainApplied,
      dopingLevel,
      interfaceCount,
      pressureGpa,
      featureVector: [],
      complexity: 0,
      stabilityPenalty: 1,
      executionError,
    };
  }

  const totalAtoms = Object.values(stoichiometry).reduce((a, b) => a + b, 0);
  hydrogenFraction = (stoichiometry["H"] || 0) / Math.max(1, totalAtoms);

  const formula = elements.map(el => {
    const count = stoichiometry[el] || 1;
    return count === 1 ? el : `${el}${count}`;
  }).join("");

  const featureVector = computeProgramFeatureVector(elements, stoichiometry, {
    latticeType, symmetryGroup, hydrogenFraction, layerCount, channelDensity,
    strainApplied, dopingLevel, interfaceCount, pressureGpa,
  });

  const complexity = program.instructions.length * 0.15 +
    elements.length * 0.1 +
    layerCount * 0.2 +
    interfaceCount * 0.15 +
    (channelDensity > 0 ? 0.2 : 0) +
    (dopingLevel > 0 ? 0.1 : 0);

  let stabilityPenalty = 0;
  if (channelDensity > 1.5) {
    stabilityPenalty += Math.min(0.5, (channelDensity - 1.5) * 0.25);
  }
  if (strainApplied > 0.05) {
    stabilityPenalty += Math.min(0.3, (strainApplied - 0.05) * 2.0);
  }
  if (interfaceCount > 3) {
    stabilityPenalty += (interfaceCount - 3) * 0.1;
  }
  stabilityPenalty = Math.min(1, stabilityPenalty);

  return {
    formula,
    prototype: symmetryGroup,
    elements,
    stoichiometry,
    latticeType,
    symmetryGroup,
    hydrogenFraction,
    layerCount,
    channelDensity,
    strainApplied,
    dopingLevel,
    interfaceCount,
    pressureGpa,
    featureVector,
    complexity: Math.min(1, complexity),
    stabilityPenalty,
  };
}

function softNorm(x: number, scale: number): number {
  return x / (x + scale);
}

function computeProgramFeatureVector(
  elements: string[],
  stoichiometry: Record<string, number>,
  props: Record<string, any>,
): number[] {
  const vec: number[] = [];

  const totalMass = elements.reduce((s, el) => s + getElementInfo(el).mass * (stoichiometry[el] || 1), 0);
  const avgEneg = elements.reduce((s, el) => s + getElementInfo(el).eneg, 0) / Math.max(1, elements.length);
  const avgRadius = elements.reduce((s, el) => s + getElementInfo(el).radius, 0) / Math.max(1, elements.length);
  const totalElectrons = elements.reduce((s, el) => s + getElementInfo(el).electrons * (stoichiometry[el] || 1), 0);

  vec.push(softNorm(totalMass, 500));
  vec.push(avgEneg / 4);
  vec.push(avgRadius / 2.5);
  vec.push(softNorm(totalElectrons, 200));
  vec.push(softNorm(elements.length, 8));
  vec.push(props.hydrogenFraction);
  vec.push(softNorm(props.layerCount, 4));
  vec.push(softNorm(props.channelDensity, 1.5));
  vec.push(Math.min(1, props.strainApplied * 10));
  vec.push(props.dopingLevel);
  vec.push(softNorm(props.interfaceCount, 3));

  const dOrbitalCount = elements.filter(el => getElementInfo(el).orbital === "d").length;
  const fOrbitalCount = elements.filter(el => getElementInfo(el).orbital === "f").length;
  vec.push(dOrbitalCount / Math.max(1, elements.length));
  vec.push(fOrbitalCount / Math.max(1, elements.length));

  const latticeIdx = LATTICE_TYPES.indexOf(props.latticeType);
  vec.push((latticeIdx >= 0 ? latticeIdx : 0) / LATTICE_TYPES.length);

  const symIdx = SYMMETRY_GROUPS.indexOf(props.symmetryGroup);
  vec.push((symIdx >= 0 ? symIdx : 0) / SYMMETRY_GROUPS.length);

  const totalAtoms = Object.values(stoichiometry).reduce((a, b) => a + b, 0);
  vec.push(softNorm(totalAtoms, 30));
  vec.push(softNorm(props.pressureGpa || 0, 200));

  return vec;
}

export function mutateDesignProgram(parent: DesignProgram, elementPool: string[]): DesignProgram {
  const child = JSON.parse(JSON.stringify(parent)) as DesignProgram;
  child.id = `prog-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  child.metadata.parentId = parent.id;
  child.metadata.generation = parent.metadata.generation + 1;
  child.metadata.generatedAt = Date.now();
  child.metadata.mutationHistory = [...parent.metadata.mutationHistory];

  const mutationType = Math.random();

  if (mutationType < 0.3) {
    const idx = Math.floor(Math.random() * child.instructions.length);
    const inst = child.instructions[idx];
    if (inst.type === "populate_sites" && Array.isArray(inst.params.elements)) {
      const elIdx = Math.floor(Math.random() * (inst.params.elements as string[]).length);
      if (elementPool.length > 0) {
        (inst.params.elements as string[])[elIdx] = elementPool[Math.floor(Math.random() * elementPool.length)];
        child.metadata.mutationHistory.push(`swap-element-${inst.params.elements[elIdx]}`);
      }
    }
  } else if (mutationType < 0.5) {
    const parentElements = child.instructions
      .flatMap(inst => {
        const els: string[] = [];
        if (Array.isArray(inst.params.elements)) els.push(...(inst.params.elements as string[]));
        if (typeof inst.params.element === "string") els.push(inst.params.element);
        if (typeof inst.params.dopant === "string") els.push(inst.params.dopant);
        return els;
      })
      .filter((v, i, a) => a.indexOf(v) === i);
    const pickEl = (pool: string[]): string => {
      if (pool.length > 0) return pool[Math.floor(Math.random() * pool.length)];
      if (parentElements.length > 0) return parentElements[Math.floor(Math.random() * parentElements.length)];
      return chemistryAwareFallback(parentElements);
    };
    const parentLattice = child.instructions.find(i => i.type === "create_lattice")?.params.latticeType as string | undefined;
    const layered = parentLattice === "tetragonal" || parentLattice === "hexagonal" || parentLattice === "orthorhombic" || parentLattice === "rhombohedral";
    const maxIW = layered ? 3 : 2;

    const insertable: ProgramInstruction[] = [
      { type: "apply_strain", params: { type: "biaxial", magnitude: Math.random() * 0.05 }, order: 0 },
      { type: "dope_sites", params: { dopant: pickEl(elementPool), fraction: 0.05 + Math.random() * 0.15, site: "random" }, order: 0 },
      { type: "add_phonon_enhancer", params: { mode: "soft-phonon", frequency: 50 + Math.random() * 300 }, order: 0 },
      { type: "add_interface", params: { type: "heterostructure", width: 1 + Math.floor(Math.random() * maxIW) }, order: 0 },
      { type: "add_sublattice", params: { elements: [pickEl(elementPool)], pattern: "interstitial", count: 1 + Math.floor(Math.random() * 3) }, order: 0 },
      { type: "add_charge_reservoir", params: { elements: [pickEl(elementPool)], layers: 1 + Math.floor(Math.random() * 2) }, order: 0 },
      { type: "add_channel_network", params: { levels: 1 + Math.floor(Math.random() * 4), branchingFactor: 2 + Math.floor(Math.random() * 5), channelWidth: 0.5 + Math.random() * 1.5 }, order: 0 },
      { type: "optimize_dos", params: { targetDOS: 2.0 + Math.random() * 6.0, method: "van-hove" }, order: 0 },
      { type: "apply_pressure_stabilization", params: { targetPressure: 10 + Math.random() * 200, method: "chemical-precompression" }, order: 0 },
      { type: "add_intercalation_layer", params: { element: pickEl(elementPool), spacing: 1.0 + Math.random() * 3.0 }, order: 0 },
    ];
    const numToInsert = Math.random() < 0.3 ? 3 : Math.random() < 0.5 ? 2 : 1;
    const shuffled = insertable.sort(() => Math.random() - 0.5);
    for (let i = 0; i < Math.min(numToInsert, shuffled.length); i++) {
      child.instructions.push({ ...shuffled[i], order: child.instructions.length });
    }
    reorderByPriority(child.instructions);
    child.metadata.mutationHistory.push(`insert-${numToInsert}-instructions`);
  } else if (mutationType < 0.65 && child.instructions.length > 3) {
    const removeIdx = 2 + Math.floor(Math.random() * (child.instructions.length - 2));
    child.instructions.splice(removeIdx, 1);
    child.metadata.mutationHistory.push("remove-instruction");
  } else if (mutationType < 0.8) {
    for (const inst of child.instructions) {
      if (inst.type === "add_hydrogen_cage" && typeof inst.params.count === "number") {
        inst.params.count = resolveHydrogenCageCount(
          inst.params.cageType as string | undefined,
          inst.params.count
        );
        child.metadata.mutationHistory.push("tweak-h-count");
      }
      if (inst.type === "add_channel_network") {
        const levels = typeof inst.params.levels === "number" ? inst.params.levels : 2;
        const branching = typeof inst.params.branchingFactor === "number" ? inst.params.branchingFactor : 3;
        inst.params.levels = Math.max(1, Math.min(5, levels + Math.round((Math.random() - 0.5) * 2)));
        inst.params.branchingFactor = Math.max(2, Math.min(6, branching + Math.round((Math.random() - 0.5) * 2)));
        child.metadata.mutationHistory.push("tweak-channel");
      }
    }
  } else {
    if (child.instructions.length >= 2) {
      const i = Math.floor(Math.random() * (child.instructions.length - 1));
      const j = i + 1;
      [child.instructions[i], child.instructions[j]] = [child.instructions[j], child.instructions[i]];
      child.instructions[i].order = i;
      child.instructions[j].order = j;
      child.metadata.mutationHistory.push("reorder-instructions");
    }
  }

  reorderByPriority(child.instructions);
  reconcileStructuralConsistency(child.instructions);

  const result = executeDesignProgram(child);
  child.outputFormula = result.formula;
  child.outputPrototype = result.prototype;
  child.featureVector = result.featureVector;
  child.metadata.complexity = result.complexity;

  return child;
}

const CUBIC_CAGE_GROUPS = new Set(["Pm-3m", "Im-3m", "Fm-3m"]);

function reconcileStructuralConsistency(instructions: ProgramInstruction[]): void {
  const latticeInst = instructions.find(i => i.type === "create_lattice");
  const symmetryInst = instructions.find(i => i.type === "set_symmetry");
  if (!latticeInst) return;

  const latticeType = (latticeInst.params.latticeType as string) || "cubic";
  const validGroups = compatibleSymmetryGroups(latticeType);
  const isLayered = latticeType === "tetragonal" || latticeType === "hexagonal" || latticeType === "orthorhombic" || latticeType === "rhombohedral";
  const isCubic = latticeType === "cubic";

  if (symmetryInst) {
    const currentGroup = symmetryInst.params.group as string;
    if (!validGroups.includes(currentGroup)) {
      symmetryInst.params.group = validGroups[0];
    }
  }

  const assignedGroup = symmetryInst
    ? (symmetryInst.params.group as string)
    : validGroups[0];

  if (!isCubic || !CUBIC_CAGE_GROUPS.has(assignedGroup)) {
    for (let i = instructions.length - 1; i >= 0; i--) {
      if (instructions[i].type === "add_hydrogen_cage") {
        instructions.splice(i, 1);
      }
    }
  }

  const maxWidth = isLayered ? 3 : 2;
  for (const inst of instructions) {
    if (inst.type === "add_interface" && typeof inst.params.width === "number") {
      inst.params.width = Math.min(inst.params.width, maxWidth);
    }
  }

  for (let i = 0; i < instructions.length; i++) {
    instructions[i].order = i;
  }
}

export function crossoverPrograms(parent1: DesignProgram, parent2: DesignProgram): DesignProgram {
  const child = JSON.parse(JSON.stringify(parent1)) as DesignProgram;
  child.id = `prog-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  child.metadata.parentId = parent1.id;
  child.metadata.generation = Math.max(parent1.metadata.generation, parent2.metadata.generation) + 1;
  child.metadata.generatedAt = Date.now();
  child.metadata.mutationHistory = ["crossover"];

  const crossoverPoint = Math.floor(Math.random() * Math.min(child.instructions.length, parent2.instructions.length));
  const p2Tail = parent2.instructions.slice(crossoverPoint).map((inst, i) => ({
    ...JSON.parse(JSON.stringify(inst)),
    order: crossoverPoint + i,
  }));

  child.instructions = [...child.instructions.slice(0, crossoverPoint), ...p2Tail]
    .map((inst, idx) => ({ ...inst, order: idx }));

  reorderByPriority(child.instructions);
  reconcileStructuralConsistency(child.instructions);

  const result = executeDesignProgram(child);
  child.outputFormula = result.formula;
  child.outputPrototype = result.prototype;
  child.featureVector = result.featureVector;
  child.metadata.complexity = result.complexity;

  return child;
}

export function generateDesignGraph(
  strategyType: string,
  elementPool: string[],
  generation: number = 0,
  parentId: string | null = null,
): DesignGraph {
  const archetype = selectGraphArchetype(strategyType);
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  const usedElements = elementPool.length > 0
    ? elementPool.slice(0, Math.min(8, elementPool.length))
    : ["Nb", "Sn", "H"];

  for (const template of archetype.nodeTemplates) {
    const el = template.preferredElement && usedElements.includes(template.preferredElement)
      ? template.preferredElement
      : selectElementForRole(template.type, usedElements);

    const info = getElementInfo(el);
    const node: GraphNode = {
      id: `n-${nodes.length}`,
      type: template.type,
      element: el,
      properties: {
        electronCount: info.electrons,
        atomicMass: info.mass,
        electronegativity: info.eneg,
        ionicRadius: info.radius,
        oxidationState: estimateOxidationState(el, template.type),
        orbitalCharacter: info.orbital,
      },
      position: template.position,
      weight: template.weight,
    };
    nodes.push(node);
  }

  for (const template of archetype.edgeTemplates) {
    if (template.sourceIdx < nodes.length && template.targetIdx < nodes.length) {
      const source = nodes[template.sourceIdx];
      const target = nodes[template.targetIdx];
      const bondLen = (source.properties.ionicRadius + target.properties.ionicRadius) * 0.8;
      const overlap = Math.max(0, 1 - Math.abs(source.properties.electronegativity - target.properties.electronegativity) / 3);

      edges.push({
        source: source.id,
        target: target.id,
        type: template.type,
        strength: template.strength * (0.8 + Math.random() * 0.4),
        properties: {
          bondLength: bondLen,
          overlapIntegral: overlap,
          couplingConstant: overlap * template.strength,
        },
      });
    }
  }

  if (Math.random() < 0.3 && generation > 0 && nodes.length >= 3) {
    const i = Math.floor(Math.random() * nodes.length);
    let j = Math.floor(Math.random() * nodes.length);
    while (j === i) j = Math.floor(Math.random() * nodes.length);
    const exists = edges.some(e =>
      (e.source === nodes[i].id && e.target === nodes[j].id) ||
      (e.source === nodes[j].id && e.target === nodes[i].id));
    const dx = nodes[i].position[0] - nodes[j].position[0];
    const dy = nodes[i].position[1] - nodes[j].position[1];
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (!exists && dist < 0.5) {
      const proxStrength = (0.3 + Math.random() * 0.4) * (1 - dist);
      const proxOverlap = Math.max(0.05, 0.3 * (1 - dist));
      const proxRadiusSum = nodes[i].properties.ionicRadius + nodes[j].properties.ionicRadius;
      edges.push({
        source: nodes[i].id,
        target: nodes[j].id,
        type: "proximity",
        strength: proxStrength,
        properties: {
          bondLength: abstractDistToBondLength(dist, proxRadiusSum),
          overlapIntegral: proxOverlap,
          couplingConstant: proxOverlap * proxStrength,
        },
      });
    }
  }

  const formula = buildFormulaFromGraph(nodes);
  const embedding = computeGraphEmbedding(nodes, edges);
  const connectivity = edges.length > 0 ? (2 * edges.length) / (nodes.length * (nodes.length - 1)) : 0;
  const avgDegree = nodes.length > 0 ? (2 * edges.length) / nodes.length : 0;

  return {
    id: `graph-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    name: `${strategyType}-graph-gen${generation}`,
    nodes,
    edges,
    outputFormula: formula,
    metadata: {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      connectivity: Math.round(connectivity * 1000) / 1000,
      avgDegree: Math.round(avgDegree * 100) / 100,
      clusteringCoeff: computeClusteringCoeff(nodes, edges),
      generatedAt: Date.now(),
      parentId,
      generation,
    },
    embedding,
  };
}

interface NodeTemplate {
  type: ComponentType;
  preferredElement?: string;
  position: [number, number];
  weight: number;
}

interface EdgeTemplate {
  sourceIdx: number;
  targetIdx: number;
  type: EdgeType;
  strength: number;
}

interface GraphArchetype {
  nodeTemplates: NodeTemplate[];
  edgeTemplates: EdgeTemplate[];
}

function selectGraphArchetype(strategyType: string): GraphArchetype {
  const archetypes: Record<string, GraphArchetype> = {
    "hydride-cage-optimizer": {
      nodeTemplates: [
        { type: "structural_backbone", preferredElement: "La", position: [0.5, 0.3], weight: 0.9 },
        { type: "hydrogen_cage", preferredElement: "H", position: [0.3, 0.7], weight: 0.8 },
        { type: "hydrogen_cage", preferredElement: "H", position: [0.7, 0.7], weight: 0.8 },
        { type: "phonon_mediator", position: [0.5, 0.5], weight: 0.7 },
        { type: "dos_enhancer", position: [0.5, 0.9], weight: 0.6 },
      ],
      edgeTemplates: [
        { sourceIdx: 0, targetIdx: 1, type: "bonding", strength: 0.9 },
        { sourceIdx: 0, targetIdx: 2, type: "bonding", strength: 0.9 },
        { sourceIdx: 1, targetIdx: 2, type: "phonon_coupling", strength: 0.7 },
        { sourceIdx: 1, targetIdx: 3, type: "phonon_coupling", strength: 0.8 },
        { sourceIdx: 2, targetIdx: 3, type: "phonon_coupling", strength: 0.8 },
        { sourceIdx: 3, targetIdx: 4, type: "electron_transfer", strength: 0.6 },
      ],
    },
    "layered-intercalation": {
      nodeTemplates: [
        { type: "electron_source", preferredElement: "Cu", position: [0.3, 0.3], weight: 0.9 },
        { type: "pairing_channel", preferredElement: "O", position: [0.7, 0.3], weight: 0.8 },
        { type: "charge_reservoir", preferredElement: "La", position: [0.5, 0.6], weight: 0.7 },
        { type: "intercalation_host", preferredElement: "Sr", position: [0.5, 0.9], weight: 0.6 },
        { type: "dos_enhancer", position: [0.2, 0.7], weight: 0.5 },
      ],
      edgeTemplates: [
        { sourceIdx: 0, targetIdx: 1, type: "bonding", strength: 0.95 },
        { sourceIdx: 0, targetIdx: 2, type: "charge_transfer", strength: 0.8 },
        { sourceIdx: 1, targetIdx: 2, type: "electron_transfer", strength: 0.7 },
        { sourceIdx: 2, targetIdx: 3, type: "structural", strength: 0.6 },
        { sourceIdx: 0, targetIdx: 4, type: "hybridization", strength: 0.5 },
      ],
    },
    "high-entropy-alloy": {
      nodeTemplates: [
        { type: "structural_backbone", preferredElement: "Nb", position: [0.5, 0.2], weight: 0.8 },
        { type: "electron_source", preferredElement: "Ti", position: [0.2, 0.5], weight: 0.7 },
        { type: "phonon_mediator", preferredElement: "Zr", position: [0.8, 0.5], weight: 0.7 },
        { type: "dos_enhancer", preferredElement: "V", position: [0.3, 0.8], weight: 0.6 },
        { type: "strain_buffer", preferredElement: "Hf", position: [0.7, 0.8], weight: 0.6 },
      ],
      edgeTemplates: [
        { sourceIdx: 0, targetIdx: 1, type: "bonding", strength: 0.8 },
        { sourceIdx: 0, targetIdx: 2, type: "bonding", strength: 0.8 },
        { sourceIdx: 1, targetIdx: 2, type: "bonding", strength: 0.7 },
        { sourceIdx: 1, targetIdx: 3, type: "hybridization", strength: 0.6 },
        { sourceIdx: 2, targetIdx: 4, type: "structural", strength: 0.6 },
        { sourceIdx: 3, targetIdx: 4, type: "proximity", strength: 0.5 },
      ],
    },
    "topological-edge": {
      nodeTemplates: [
        { type: "topological_surface", preferredElement: "Bi", position: [0.5, 0.2], weight: 0.9 },
        { type: "electron_source", preferredElement: "Se", position: [0.3, 0.5], weight: 0.8 },
        { type: "electron_source", preferredElement: "Te", position: [0.7, 0.5], weight: 0.7 },
        { type: "dopant_site", preferredElement: "Cu", position: [0.5, 0.8], weight: 0.5 },
      ],
      edgeTemplates: [
        { sourceIdx: 0, targetIdx: 1, type: "bonding", strength: 0.9 },
        { sourceIdx: 0, targetIdx: 2, type: "bonding", strength: 0.8 },
        { sourceIdx: 1, targetIdx: 2, type: "proximity", strength: 0.5 },
        { sourceIdx: 0, targetIdx: 3, type: "electron_transfer", strength: 0.4 },
      ],
    },
    default: {
      nodeTemplates: [
        { type: "structural_backbone", position: [0.5, 0.3], weight: 0.8 },
        { type: "electron_source", position: [0.3, 0.7], weight: 0.7 },
        { type: "phonon_mediator", position: [0.7, 0.7], weight: 0.6 },
      ],
      edgeTemplates: [
        { sourceIdx: 0, targetIdx: 1, type: "bonding", strength: 0.8 },
        { sourceIdx: 0, targetIdx: 2, type: "bonding", strength: 0.7 },
        { sourceIdx: 1, targetIdx: 2, type: "phonon_coupling", strength: 0.5 },
      ],
    },
  };

  return archetypes[strategyType] ?? archetypes.default;
}

const ROLE_HEAVY_FALLBACKS: Record<string, string[]> = {
  structural_backbone: ["Nb", "Ta", "La", "Y", "Hf"],
  electron_source: ["Nb", "V", "Cu", "Fe"],
  charge_reservoir: ["La", "Ba", "Sr"],
  dos_enhancer: ["Nb", "V", "Mo", "W"],
  strain_buffer: ["Hf", "Zr", "Ti"],
};

function selectElementForRole(role: ComponentType, pool: string[]): string {
  const rolePreferences: Record<string, string[]> = {
    electron_source: ["Cu", "Fe", "Ni", "Co", "Nb", "V"],
    phonon_mediator: ["H", "B", "C", "N", "O"],
    charge_reservoir: ["La", "Ba", "Sr", "Ca", "Y"],
    structural_backbone: ["La", "Y", "Nb", "Ti", "Zr"],
    hydrogen_cage: ["H"],
    intercalation_host: ["Sr", "Ba", "Ca"],
    dopant_site: ["Sr", "Cu", "Fe", "Co"],
    interface_layer: ["O", "Se", "S"],
    strain_buffer: ["Hf", "Zr", "Ti"],
    topological_surface: ["Bi", "Sb", "Sn"],
    pairing_channel: ["O", "As", "Se"],
    dos_enhancer: ["V", "Nb", "Mo", "W"],
  };

  const prefs = rolePreferences[role] ?? ["Nb"];
  for (const el of prefs) {
    if (pool.includes(el)) return el;
  }

  if (pool.length > 0) {
    const heavyFallbacks = ROLE_HEAVY_FALLBACKS[role];
    if (heavyFallbacks) {
      const heavyInPool = pool.filter(el => {
        const info = getElementInfo(el);
        return info.mass > 40;
      });
      if (heavyInPool.length > 0) return heavyInPool[Math.floor(Math.random() * heavyInPool.length)];
    }
    return pool[Math.floor(Math.random() * pool.length)];
  }

  const fallbacks = ROLE_HEAVY_FALLBACKS[role];
  return fallbacks ? fallbacks[0] : prefs[0];
}

function estimateOxidationState(el: string, role: ComponentType): number {
  if (el === "H") {
    if (role === "hydrogen_cage" || role === "phonon_mediator") return -0.5;
    return 1;
  }

  const states: Record<string, number> = {
    Li: 1, Be: 2, B: 3, C: 4, N: -3, O: -2, Al: 3, Si: 4, S: -2,
    Ca: 2, Sc: 3, Ti: 4, V: 5, Nb: 5, Mo: 6, Sr: 2, Y: 3, Zr: 4,
    Sn: 4, Ba: 2, La: 3, Ce: 3, Hf: 4, Ta: 5, W: 6, Re: 7, Bi: 3,
    Cu: 2, Fe: 3, Co: 2, Ni: 2, Se: -2, Te: -2, Sb: 3, As: -3,
    Ge: 4, Mg: 2, Eu: 2, Nd: 3, Pr: 3, Th: 4, Pb: 2, In: 3,
    Tl: 1, Hg: 2, P: -3, Cl: -1, Br: -1,
  };
  return states[el] ?? 0;
}

function buildFormulaFromGraph(nodes: GraphNode[]): string {
  const counts: Record<string, number> = {};
  for (const node of nodes) {
    if (node.element === "interface" || node.element === "intercalation") continue;
    counts[node.element] = (counts[node.element] || 0) + 1;
  }
  return Object.entries(counts)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([el, count]) => count === 1 ? el : `${el}${count}`)
    .join("");
}

function computeGraphEmbedding(nodes: GraphNode[], edges: GraphEdge[]): number[] {
  const vec: number[] = [];

  vec.push(softNorm(nodes.length, 8));
  vec.push(softNorm(edges.length, 12));

  const avgMass = nodes.reduce((s, n) => s + n.properties.atomicMass, 0) / Math.max(1, nodes.length);
  vec.push(softNorm(avgMass, 120));

  const avgEneg = nodes.reduce((s, n) => s + n.properties.electronegativity, 0) / Math.max(1, nodes.length);
  vec.push(avgEneg / 4);

  const avgStrength = edges.reduce((s, e) => s + e.strength, 0) / Math.max(1, edges.length);
  vec.push(Math.min(1, avgStrength));

  const typeCount: Record<string, number> = {};
  for (const n of nodes) typeCount[n.type] = (typeCount[n.type] || 0) + 1;
  vec.push(softNorm(Object.keys(typeCount).length, 10));

  const edgeTypeCount: Record<string, number> = {};
  for (const e of edges) edgeTypeCount[e.type] = (edgeTypeCount[e.type] || 0) + 1;
  vec.push(softNorm(Object.keys(edgeTypeCount).length, 6));

  const avgWeight = nodes.reduce((s, n) => s + n.weight, 0) / Math.max(1, nodes.length);
  vec.push(Math.min(1, avgWeight));

  const hNodes = nodes.filter(n => n.element === "H").length;
  vec.push(hNodes / Math.max(1, nodes.length));

  const dOrbital = nodes.filter(n => n.properties.orbitalCharacter === "d").length;
  vec.push(dOrbital / Math.max(1, nodes.length));

  const avgCoupling = edges.reduce((s, e) => s + e.properties.couplingConstant, 0) / Math.max(1, edges.length);
  vec.push(Math.min(1, avgCoupling));

  const connectivity = edges.length > 0 ? (2 * edges.length) / (nodes.length * Math.max(1, nodes.length - 1)) : 0;
  vec.push(Math.min(1, connectivity));

  return vec;
}

function computeClusteringCoeff(nodes: GraphNode[], edges: GraphEdge[]): number {
  if (nodes.length < 3) return 0;

  const adj = new Map<string, Set<string>>();
  for (const n of nodes) adj.set(n.id, new Set());
  for (const e of edges) {
    adj.get(e.source)?.add(e.target);
    adj.get(e.target)?.add(e.source);
  }

  let totalCoeff = 0;
  for (const n of nodes) {
    const neighbors = Array.from(adj.get(n.id) || []);
    if (neighbors.length < 2) continue;
    let triangles = 0;
    for (let i = 0; i < neighbors.length; i++) {
      for (let j = i + 1; j < neighbors.length; j++) {
        if (adj.get(neighbors[i])?.has(neighbors[j])) triangles++;
      }
    }
    const possible = (neighbors.length * (neighbors.length - 1)) / 2;
    totalCoeff += triangles / possible;
  }

  return Math.round((totalCoeff / nodes.length) * 1000) / 1000;
}

const UNIT_CELL_SCALE_ANGSTROM = 5.0;
const MIN_NODE_SEPARATION = 0.15;

function placeNodePosition(existingNodes: GraphNode[]): [number, number] {
  const maxAttempts = 20;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const pos: [number, number] = [Math.random(), Math.random()];
    let tooClose = false;
    for (const n of existingNodes) {
      const dx = pos[0] - n.position[0];
      const dy = pos[1] - n.position[1];
      if (Math.sqrt(dx * dx + dy * dy) < MIN_NODE_SEPARATION) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) return pos;
  }
  const angle = Math.random() * 2 * Math.PI;
  const r = 0.3 + Math.random() * 0.15;
  return [0.5 + r * Math.cos(angle), 0.5 + r * Math.sin(angle)];
}

function abstractDistToBondLength(
  abstractDist: number,
  radiusSum: number,
): number {
  const physicalDist = abstractDist * UNIT_CELL_SCALE_ANGSTROM;
  const minBond = radiusSum * 0.85;
  const maxBond = radiusSum * 1.8;
  return Math.max(minBond, Math.min(maxBond, physicalDist));
}

function nextNodeId(nodes: GraphNode[]): string {
  let maxIdx = -1;
  for (const n of nodes) {
    const m = n.id.match(/^n-(\d+)$/);
    if (m) maxIdx = Math.max(maxIdx, parseInt(m[1], 10));
  }
  return `n-${maxIdx + 1}`;
}

function ensureGraphConnectivity(nodes: GraphNode[], edges: GraphEdge[]): void {
  if (nodes.length < 2) return;

  const adj = new Map<string, Set<string>>();
  for (const n of nodes) adj.set(n.id, new Set());
  for (const e of edges) {
    adj.get(e.source)?.add(e.target);
    adj.get(e.target)?.add(e.source);
  }

  const visited = new Set<string>();
  const components: string[][] = [];

  for (const n of nodes) {
    if (visited.has(n.id)) continue;
    const component: string[] = [];
    const stack = [n.id];
    while (stack.length > 0) {
      const curr = stack.pop()!;
      if (visited.has(curr)) continue;
      visited.add(curr);
      component.push(curr);
      for (const neighbor of adj.get(curr) || []) {
        if (!visited.has(neighbor)) stack.push(neighbor);
      }
    }
    components.push(component);
  }

  if (components.length <= 1) return;

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  for (let c = 1; c < components.length; c++) {
    let bestDist = Infinity;
    let bestI = components[0][0];
    let bestJ = components[c][0];

    for (const idA of components[0]) {
      const nA = nodeMap.get(idA)!;
      for (const idB of components[c]) {
        const nB = nodeMap.get(idB)!;
        const dx = nA.position[0] - nB.position[0];
        const dy = nA.position[1] - nB.position[1];
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestDist) {
          bestDist = d;
          bestI = idA;
          bestJ = idB;
        }
      }
    }

    const nI = nodeMap.get(bestI)!;
    const nJ = nodeMap.get(bestJ)!;
    const bridgeRadiusSum = nI.properties.ionicRadius + nJ.properties.ionicRadius;
    const bondLen = abstractDistToBondLength(bestDist, bridgeRadiusSum);
    const overlap = Math.max(0.05, 0.3 * (1 - Math.min(1, bestDist)));
    const strength = 0.3 + Math.random() * 0.3;
    edges.push({
      source: bestI,
      target: bestJ,
      type: "proximity",
      strength,
      properties: { bondLength: bondLen, overlapIntegral: overlap, couplingConstant: overlap * strength },
    });

    components[0].push(...components[c]);
  }
}

export function mutateDesignGraph(parent: DesignGraph, elementPool: string[]): DesignGraph {
  const child = JSON.parse(JSON.stringify(parent)) as DesignGraph;
  child.id = `graph-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  child.metadata.parentId = parent.id;
  child.metadata.generation = parent.metadata.generation + 1;
  child.metadata.generatedAt = Date.now();

  const mutationType = Math.random();

  if (mutationType < 0.25 && child.nodes.length > 0) {
    const idx = Math.floor(Math.random() * child.nodes.length);
    const newEl = elementPool.length > 0
      ? elementPool[Math.floor(Math.random() * elementPool.length)]
      : child.nodes[idx].element;
    child.nodes[idx].element = newEl;
    const info = getElementInfo(newEl);
    child.nodes[idx].properties = {
      electronCount: info.electrons,
      atomicMass: info.mass,
      electronegativity: info.eneg,
      ionicRadius: info.radius,
      oxidationState: estimateOxidationState(newEl, child.nodes[idx].type),
      orbitalCharacter: info.orbital,
    };
  } else if (mutationType < 0.45) {
    const types: ComponentType[] = ["electron_source", "phonon_mediator", "dos_enhancer", "dopant_site", "strain_buffer"];
    const newType = types[Math.floor(Math.random() * types.length)];
    const el = selectElementForRole(newType, elementPool);
    const info = getElementInfo(el);
    const newNode: GraphNode = {
      id: nextNodeId(child.nodes),
      type: newType,
      element: el,
      properties: {
        electronCount: info.electrons,
        atomicMass: info.mass,
        electronegativity: info.eneg,
        ionicRadius: info.radius,
        oxidationState: estimateOxidationState(el, newType),
        orbitalCharacter: info.orbital,
      },
      position: placeNodePosition(child.nodes),
      weight: 0.3 + Math.random() * 0.5,
    };
    child.nodes.push(newNode);

    const candidates = child.nodes
      .slice(0, child.nodes.length - 1)
      .map((n, idx) => {
        const ddx = newNode.position[0] - n.position[0];
        const ddy = newNode.position[1] - n.position[1];
        return { idx, dist: Math.sqrt(ddx * ddx + ddy * ddy) };
      })
      .filter(c => c.dist < 0.5)
      .sort((a, b) => a.dist - b.dist);
    const connectTo = candidates.length > 0
      ? candidates[0].idx
      : Math.floor(Math.random() * (child.nodes.length - 1));
    const connDist = candidates.length > 0 ? candidates[0].dist : 0.4;
    const edgeTypes: EdgeType[] = ["bonding", "electron_transfer", "proximity", "hybridization"];
    const connTarget = child.nodes[connectTo];
    const radiusSum = newNode.properties.ionicRadius + connTarget.properties.ionicRadius;
    const connBondLen = abstractDistToBondLength(connDist, radiusSum);
    const connOverlap = Math.max(0.05, 0.3 * (1 - connDist));
    const connStrength = 0.3 + Math.random() * 0.5;
    child.edges.push({
      source: newNode.id,
      target: connTarget.id,
      type: edgeTypes[Math.floor(Math.random() * edgeTypes.length)],
      strength: connStrength,
      properties: { bondLength: connBondLen, overlapIntegral: connOverlap, couplingConstant: connOverlap * connStrength },
    });
  } else if (mutationType < 0.6 && child.nodes.length > 3) {
    const removeIdx = Math.floor(Math.random() * child.nodes.length);
    const removedId = child.nodes[removeIdx].id;
    child.nodes.splice(removeIdx, 1);
    child.edges = child.edges.filter(e => e.source !== removedId && e.target !== removedId);
    ensureGraphConnectivity(child.nodes, child.edges);
  } else if (mutationType < 0.8 && child.nodes.length >= 2) {
    const i = Math.floor(Math.random() * child.nodes.length);
    let j = Math.floor(Math.random() * child.nodes.length);
    while (j === i) j = Math.floor(Math.random() * child.nodes.length);
    const exists = child.edges.some(e =>
      (e.source === child.nodes[i].id && e.target === child.nodes[j].id) ||
      (e.source === child.nodes[j].id && e.target === child.nodes[i].id));
    const edx = child.nodes[i].position[0] - child.nodes[j].position[0];
    const edy = child.nodes[i].position[1] - child.nodes[j].position[1];
    const edist = Math.sqrt(edx * edx + edy * edy);
    if (!exists && edist < 0.5) {
      const edgeTypes: EdgeType[] = ["bonding", "phonon_coupling", "charge_transfer", "hybridization"];
      const eStr = (0.3 + Math.random() * 0.5) * (1 - edist);
      const eOverlap = Math.max(0.05, 0.3 * (1 - edist));
      const eRadiusSum = child.nodes[i].properties.ionicRadius + child.nodes[j].properties.ionicRadius;
      const eBondLen = abstractDistToBondLength(edist, eRadiusSum);
      child.edges.push({
        source: child.nodes[i].id,
        target: child.nodes[j].id,
        type: edgeTypes[Math.floor(Math.random() * edgeTypes.length)],
        strength: eStr,
        properties: { bondLength: eBondLen, overlapIntegral: eOverlap, couplingConstant: eOverlap * eStr },
      });
    }
  } else if (child.edges.length > 2) {
    const removeIdx = Math.floor(Math.random() * child.edges.length);
    child.edges.splice(removeIdx, 1);
  }

  child.outputFormula = buildFormulaFromGraph(child.nodes);
  child.embedding = computeGraphEmbedding(child.nodes, child.edges);
  child.metadata.nodeCount = child.nodes.length;
  child.metadata.edgeCount = child.edges.length;
  child.metadata.connectivity = child.nodes.length > 1
    ? Math.round((2 * child.edges.length) / (child.nodes.length * (child.nodes.length - 1)) * 1000) / 1000
    : 0;
  child.metadata.avgDegree = child.nodes.length > 0
    ? Math.round((2 * child.edges.length) / child.nodes.length * 100) / 100
    : 0;
  child.metadata.clusteringCoeff = computeClusteringCoeff(child.nodes, child.edges);

  return child;
}

export function analyzeGraph(graph: DesignGraph): GraphAnalysis {
  const adj = new Map<string, Set<string>>();
  for (const n of graph.nodes) adj.set(n.id, new Set());
  for (const e of graph.edges) {
    adj.get(e.source)?.add(e.target);
    adj.get(e.target)?.add(e.source);
  }

  const centralNodes = graph.nodes.map(n => ({
    id: n.id,
    element: n.element,
    centrality: (adj.get(n.id)?.size ?? 0) / Math.max(1, graph.nodes.length - 1),
  })).sort((a, b) => b.centrality - a.centrality).slice(0, 3);

  const weightMap = new Map<string, number>();
  for (const n of graph.nodes) weightMap.set(n.id, n.weight);

  const visited = new Set<string>();
  const communities: { id: number; members: string[]; avgWeight: number }[] = [];
  let communityId = 0;
  const nodeCommunity = new Map<string, number>();

  for (const n of graph.nodes) {
    if (visited.has(n.id)) continue;
    const members: string[] = [];
    const queue = [n.id];
    while (queue.length > 0) {
      const curr = queue.shift()!;
      if (visited.has(curr)) continue;
      visited.add(curr);
      members.push(curr);
      for (const neighbor of adj.get(curr) ?? []) {
        if (!visited.has(neighbor)) queue.push(neighbor);
      }
    }
    const cId = communityId++;
    for (const m of members) nodeCommunity.set(m, cId);
    const totalWeight = members.reduce((s, id) => s + (weightMap.get(id) || 0), 0);
    communities.push({
      id: cId,
      members,
      avgWeight: totalWeight / Math.max(1, members.length),
    });
  }

  const isConnected = communities.length <= 1;

  const bottleneckEdges = graph.edges.map(e => {
    const srcComm = nodeCommunity.get(e.source) ?? -1;
    const tgtComm = nodeCommunity.get(e.target) ?? -1;
    const isBridge = srcComm !== tgtComm;

    const srcDeg = adj.get(e.source)?.size ?? 0;
    const tgtDeg = adj.get(e.target)?.size ?? 0;
    const minDeg = Math.min(srcDeg, tgtDeg);
    const isNarrow = minDeg <= 2;

    let betweenness = 0;
    if (isBridge) {
      const srcCommSize = communities.find(c => c.id === srcComm)?.members.length ?? 1;
      const tgtCommSize = communities.find(c => c.id === tgtComm)?.members.length ?? 1;
      betweenness = (srcCommSize * tgtCommSize) / Math.max(1, graph.nodes.length * (graph.nodes.length - 1) / 2);
    } else if (isNarrow) {
      betweenness = e.strength * (1 / Math.max(1, minDeg));
    } else {
      betweenness = e.strength / Math.max(1, graph.edges.length);
    }
    return { source: e.source, target: e.target, betweenness };
  }).sort((a, b) => b.betweenness - a.betweenness).slice(0, 3);

  const totalPossiblePairs = graph.nodes.length * (graph.nodes.length - 1) / 2;

  let maxDist = 0;
  let totalDist = 0;
  let reachablePairs = 0;
  for (const start of graph.nodes) {
    const dist = new Map<string, number>();
    dist.set(start.id, 0);
    const q = [start.id];
    while (q.length > 0) {
      const curr = q.shift()!;
      const d = dist.get(curr)!;
      for (const neighbor of adj.get(curr) ?? []) {
        if (!dist.has(neighbor)) {
          dist.set(neighbor, d + 1);
          q.push(neighbor);
          if (d + 1 > maxDist) maxDist = d + 1;
          totalDist += d + 1;
          reachablePairs++;
        }
      }
    }
  }

  const unreachablePairs = totalPossiblePairs - (reachablePairs / 2);
  const disconnectedPenalty = graph.nodes.length * 2;
  const effectiveTotalDist = totalDist / 2 + unreachablePairs * disconnectedPenalty;
  const avgPath = totalPossiblePairs > 0 ? effectiveTotalDist / totalPossiblePairs : 0;

  const spectralGap = computeFiedlerValue(graph.nodes, graph.edges, adj);

  return {
    centralNodes,
    bottleneckEdges,
    communities,
    pathLengths: { avg: Math.round(avgPath * 100) / 100, max: maxDist, diameter: maxDist },
    spectralGap: Math.round(spectralGap * 1000) / 1000,
    isConnected,
  };
}

function computeFiedlerValue(
  nodes: GraphNode[],
  edges: GraphEdge[],
  _adj: Map<string, Set<string>>,
): number {
  const n = nodes.length;
  if (n < 2) return 0;

  const idxMap = new Map<string, number>();
  for (let i = 0; i < n; i++) idxMap.set(nodes[i].id, i);

  const degree = new Float64Array(n);
  const adjIdx: number[][] = Array.from({ length: n }, () => []);
  for (const e of edges) {
    const si = idxMap.get(e.source);
    const ti = idxMap.get(e.target);
    if (si === undefined || ti === undefined) continue;
    degree[si]++;
    degree[ti]++;
    adjIdx[si].push(ti);
    adjIdx[ti].push(si);
  }

  let maxDeg = 0;
  for (let i = 0; i < n; i++) if (degree[i] > maxDeg) maxDeg = degree[i];
  const shift = maxDeg + 1;

  const shiftedLaplacianMultiply = (v: Float64Array): Float64Array => {
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      result[i] = (shift - degree[i]) * v[i];
      for (const j of adjIdx[i]) {
        result[i] += v[j];
      }
    }
    return result;
  };

  const projectOutConstant = (v: Float64Array): void => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += v[i];
    const avg = sum / n;
    for (let i = 0; i < n; i++) v[i] -= avg;
  };

  const normalize = (v: Float64Array): number => {
    let norm = 0;
    for (let i = 0; i < n; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return 0;
    for (let i = 0; i < n; i++) v[i] /= norm;
    return norm;
  };

  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = Math.random() - 0.5;
  projectOutConstant(v);
  normalize(v);

  let eigenvalue = 0;
  const maxIter = 80;
  for (let iter = 0; iter < maxIter; iter++) {
    const Mv = shiftedLaplacianMultiply(v);
    projectOutConstant(Mv);

    let rayleigh = 0;
    let vNormSq = 0;
    for (let i = 0; i < n; i++) {
      rayleigh += v[i] * Mv[i];
      vNormSq += v[i] * v[i];
    }
    eigenvalue = vNormSq > 1e-12 ? rayleigh / vNormSq : 0;

    for (let i = 0; i < n; i++) v[i] = Mv[i];
    projectOutConstant(v);
    const norm = normalize(v);
    if (norm < 1e-12) break;
  }

  const fiedler = shift - eigenvalue;
  return Math.max(0, fiedler);
}

export function programToGraph(program: DesignProgram): DesignGraph {
  const result = executeDesignProgram(program);
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  const elementNodeIds: Record<string, string[]> = {};

  for (const el of result.elements) {
    const info = getElementInfo(el);
    const role = inferRoleFromElement(el, result);
    const stoich = result.stoichiometry[el] || 1;
    const totalAtoms = Object.values(result.stoichiometry).reduce((a, b) => a + b, 0);
    const siteCount = Math.min(stoich, 4);
    elementNodeIds[el] = [];

    for (let s = 0; s < siteCount; s++) {
      const nodeId = `n-${nodes.length}`;
      elementNodeIds[el].push(nodeId);
      const angle = (2 * Math.PI * nodes.length) / (result.elements.length * 2);
      nodes.push({
        id: nodeId,
        type: role,
        element: el,
        properties: {
          electronCount: info.electrons,
          atomicMass: info.mass,
          electronegativity: info.eneg,
          ionicRadius: info.radius,
          oxidationState: estimateOxidationState(el, role),
          orbitalCharacter: info.orbital,
        },
        position: [0.5 + 0.3 * Math.cos(angle + s * 0.3), 0.5 + 0.3 * Math.sin(angle + s * 0.3)],
        weight: 1 / Math.max(1, totalAtoms),
      });

      if (s > 0) {
        edges.push({
          source: elementNodeIds[el][0],
          target: nodeId,
          type: "sublattice",
          strength: 0.9,
          properties: {
            bondLength: info.radius * 2,
            overlapIntegral: 0.8,
            couplingConstant: 0.3,
          },
        });
      }
    }
  }

  const structuralNodeTypes: Array<{ type: ComponentType; label: string; condition: boolean }> = [
    { type: "phonon_mediator", label: "phonon-bridge", condition: result.elements.some(el => ["H", "B", "C", "N", "O"].includes(el)) },
    { type: "charge_reservoir", label: "charge-layer", condition: result.elements.some(el => ["La", "Y", "Ba", "Sr", "Ca", "Ce"].includes(el)) },
    { type: "structural_backbone", label: "lattice-frame", condition: true },
    { type: "pairing_channel", label: "cooper-channel", condition: result.elements.length >= 2 && ["tetragonal", "orthorhombic", "hexagonal", "monoclinic", "rhombohedral"].includes(result.latticeType) },
    { type: "dos_enhancer", label: "dos-peak", condition: result.elements.some(el => ["V", "Nb", "Ta", "Mo", "W", "Fe", "Co", "Ni"].includes(el)) },
    { type: "strain_buffer", label: "strain-layer", condition: result.elements.length >= 3 },
  ];

  let fnodeCounter = 0;
  for (const snt of structuralNodeTypes) {
    if (!snt.condition) continue;
    const nodeId = `fn-${snt.label}-${fnodeCounter++}`;
    nodes.push({
      id: nodeId,
      type: snt.type,
      element: snt.label,
      properties: {
        electronCount: 0,
        atomicMass: 0,
        electronegativity: 0,
        ionicRadius: 0,
        oxidationState: 0,
        orbitalCharacter: "functional",
      },
      position: [0.1 + (fnodeCounter * 0.13) % 0.8, 0.1 + (fnodeCounter * 0.21) % 0.8],
      weight: 0.1,
    });

    const relatedElements = result.elements.filter(el => {
      if (snt.type === "phonon_mediator") return ["H", "B", "C", "N", "O"].includes(el);
      if (snt.type === "charge_reservoir") return ["La", "Y", "Ba", "Sr", "Ca", "Ce"].includes(el);
      if (snt.type === "dos_enhancer") return ["V", "Nb", "Ta", "Mo", "W", "Fe", "Co", "Ni"].includes(el);
      return true;
    });

    for (const el of relatedElements) {
      const elNodes = elementNodeIds[el];
      if (elNodes && elNodes.length > 0) {
        const elInfo = getElementInfo(el);
        const edgeType: EdgeType = snt.type === "phonon_mediator" ? "phonon_coupling"
          : snt.type === "charge_reservoir" ? "charge_transfer"
          : snt.type === "pairing_channel" ? "pairing"
          : "bonding";
        const ionicR = elInfo.radius;
        const eneg = elInfo.eneg;
        edges.push({
          source: elNodes[0],
          target: nodeId,
          type: edgeType,
          strength: Math.max(0.3, Math.min(0.95, 1.0 - ionicR / 3)),
          properties: {
            bondLength: ionicR * 2 + 0.5,
            overlapIntegral: Math.max(0.1, Math.min(0.9, eneg / 4)),
            couplingConstant: Math.max(0.05, Math.min(0.8, 1.0 / (1.0 + ionicR))),
          },
        });
      }
    }
  }

  let ifaceCounter = 0;
  for (const inst of program.instructions) {
    if (inst.type === "add_interface" || inst.type === "add_intercalation_layer") {
      const ifaceLabel = inst.type === "add_interface" ? "interface" : "intercalation";
      const nodeId = `fn-${ifaceLabel}-${ifaceCounter++}`;
      nodes.push({
        id: nodeId,
        type: "topological_surface",
        element: ifaceLabel,
        properties: {
          electronCount: 0, atomicMass: 0, electronegativity: 0,
          ionicRadius: 0, oxidationState: 0, orbitalCharacter: "interface",
        },
        position: [0.2 + (ifaceCounter * 0.17) % 0.6, 0.8 - (ifaceCounter * 0.23) % 0.6],
        weight: 0.08,
      });

      const pressureGpa = result.pressureGpa;
      const interlayerBondLength = pressureGpa > 0
        ? Math.max(1.0, 3.0 * Math.exp(-pressureGpa / 100))
        : 3.0;
      const interlayerOverlap = pressureGpa > 0
        ? Math.min(0.8, 0.3 + pressureGpa / 500)
        : 0.3;
      const interlayerCoupling = pressureGpa > 0
        ? Math.min(0.6, 0.15 + pressureGpa / 400)
        : 0.15;
      const INTERFACE_ROLES: Set<string> = new Set(["charge_reservoir", "structural_backbone", "phonon_mediator"]);
      const targetNodeIds: string[] = [];
      for (const n of nodes) {
        if (n.id.startsWith("fn-")) {
          if (INTERFACE_ROLES.has(n.type)) targetNodeIds.push(n.id);
        } else {
          targetNodeIds.push(n.id);
        }
      }
      for (const srcId of targetNodeIds) {
        if (srcId === nodeId) continue;
        edges.push({
          source: srcId,
          target: nodeId,
          type: "interlayer",
          strength: 0.5,
          properties: { bondLength: interlayerBondLength, overlapIntegral: interlayerOverlap, couplingConstant: interlayerCoupling },
        });
      }
    }
  }

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const allElementNodes = Object.values(elementNodeIds).flat();
  for (let i = 0; i < allElementNodes.length; i++) {
    for (let j = i + 1; j < allElementNodes.length; j++) {
      const ni = nodeMap.get(allElementNodes[i]);
      const nj = nodeMap.get(allElementNodes[j]);
      if (!ni || !nj) continue;
      if (ni.element === nj.element) continue;
      const enegDiff = Math.abs(ni.properties.electronegativity - nj.properties.electronegativity);
      const edgeType: EdgeType = enegDiff > 1.5 ? "charge_transfer" : enegDiff > 0.5 ? "bonding" : "hybridization";
      edges.push({
        source: allElementNodes[i],
        target: allElementNodes[j],
        type: edgeType,
        strength: Math.max(0.2, 1 - enegDiff / 4),
        properties: {
          bondLength: ni.properties.ionicRadius + nj.properties.ionicRadius,
          overlapIntegral: Math.max(0, 1 - enegDiff / 4),
          couplingConstant: Math.max(0.1, 0.5 - enegDiff / 8),
        },
      });
    }
  }

  return {
    id: `graph-from-prog-${program.id}`,
    name: `converted-${program.name}`,
    nodes,
    edges,
    outputFormula: result.formula,
    metadata: {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      connectivity: nodes.length > 1 ? Math.round((2 * edges.length) / (nodes.length * (nodes.length - 1)) * 1000) / 1000 : 0,
      avgDegree: nodes.length > 0 ? Math.round((2 * edges.length) / nodes.length * 100) / 100 : 0,
      clusteringCoeff: computeClusteringCoeff(nodes, edges),
      generatedAt: Date.now(),
      parentId: program.id,
      generation: program.metadata.generation,
    },
    embedding: computeGraphEmbedding(nodes, edges),
  };
}

function inferRoleFromElement(el: string, result: ProgramExecutionResult): ComponentType {
  if (el === "H") return "hydrogen_cage";
  if (el === "O") return "pairing_channel";
  if (["La", "Y", "Ba", "Sr", "Ca"].includes(el)) return "charge_reservoir";
  if (["Cu", "Fe", "Ni", "Co"].includes(el)) return "electron_source";
  if (["Nb", "Ti", "Zr", "Hf", "Ta"].includes(el)) return "structural_backbone";
  if (["B", "C", "N"].includes(el)) return "phonon_mediator";
  if (["Bi", "Sb", "Sn", "Pb"].includes(el)) return "topological_surface";
  if (["Se", "Te", "As", "S"].includes(el)) return "pairing_channel";
  return "dos_enhancer";
}

function inferLatticeFromGraph(graph: DesignGraph): string {
  const hasIntercalation = graph.nodes.some(n => n.element === "intercalation" || n.element === "interface");
  const hasInterlayerEdges = graph.edges.some(e => e.type === "interlayer");
  const hasChargeReservoir = graph.nodes.some(n => n.type === "charge_reservoir");
  const hasPairingChannel = graph.nodes.some(n => n.type === "pairing_channel");
  const hasHydrogenCage = graph.nodes.some(n => n.type === "hydrogen_cage");
  const elementNodes = graph.nodes.filter(n => !n.id.startsWith("fn-"));
  const hasHeavyTM = elementNodes.some(n => ["Cu", "Fe", "Co", "Ni"].includes(n.element));

  if (hasIntercalation || hasInterlayerEdges) {
    if (hasChargeReservoir && hasHeavyTM) return "tetragonal";
    return "hexagonal";
  }
  if (hasPairingChannel && hasChargeReservoir) return "tetragonal";
  if (hasHydrogenCage && elementNodes.length <= 3) return "cubic";
  if (elementNodes.length >= 5) return "orthorhombic";
  return "cubic";
}

export function graphToProgram(graph: DesignGraph): DesignProgram {
  const instructions: ProgramInstruction[] = [];
  let order = 0;

  const backboneNodes = graph.nodes.filter(n =>
    n.type === "structural_backbone" || n.type === "electron_source");

  const latticeType = inferLatticeFromGraph(graph);

  if (backboneNodes.length > 0) {
    const latticeParams: Record<string, number | string> = { latticeType };
    if (latticeType === "cubic") {
      latticeParams.a = 4.0 + backboneNodes.length * 0.3;
    } else if (latticeType === "tetragonal") {
      latticeParams.a = 3.8 + backboneNodes.length * 0.2;
      latticeParams.c = 10.0 + backboneNodes.length * 0.5;
    } else if (latticeType === "hexagonal") {
      latticeParams.a = 3.0 + backboneNodes.length * 0.2;
      latticeParams.c = 3.5 + backboneNodes.length * 0.3;
    } else if (latticeType === "orthorhombic") {
      latticeParams.a = 3.8 + backboneNodes.length * 0.1;
      latticeParams.b = 3.9 + backboneNodes.length * 0.15;
      latticeParams.c = 11.0 + backboneNodes.length * 0.4;
    } else {
      latticeParams.a = 4.0 + backboneNodes.length * 0.3;
    }
    instructions.push({
      type: "create_lattice",
      params: latticeParams,
      order: order++,
    });
    const validGroups = compatibleSymmetryGroups(latticeType);
    instructions.push({
      type: "set_symmetry",
      params: { group: validGroups[backboneNodes.length % validGroups.length] },
      order: order++,
    });
    instructions.push({
      type: "populate_sites",
      params: { elements: backboneNodes.map(n => n.element), site: "vertex" },
      order: order++,
    });
  }

  const hNodes = graph.nodes.filter(n => n.type === "hydrogen_cage");
  if (hNodes.length > 0) {
    const formulaMatch = graph.outputFormula.match(/H(\d+)/);
    const formulaHCount = formulaMatch ? parseInt(formulaMatch[1], 10) : 0;

    let cageType = "clathrate";
    let hCount: number;
    if (formulaHCount > 0) {
      hCount = formulaHCount;
      for (const [cage, fixedCount] of Object.entries(CAGE_FIXED_COUNTS)) {
        if (fixedCount === formulaHCount) { cageType = cage; break; }
      }
    } else {
      hCount = CAGE_FIXED_COUNTS[cageType];
    }

    instructions.push({
      type: "add_hydrogen_cage",
      params: { count: hCount, cageType, bondLength: 1.1 },
      order: order++,
    });
  }

  const reservoirNodes = graph.nodes.filter(n =>
    n.type === "charge_reservoir" || n.type === "intercalation_host");
  if (reservoirNodes.length > 0) {
    instructions.push({
      type: "add_charge_reservoir",
      params: { elements: reservoirNodes.map(n => n.element), layers: reservoirNodes.length },
      order: order++,
    });
  }

  const dopantNodes = graph.nodes.filter(n => n.type === "dopant_site");
  if (dopantNodes.length > 0) {
    instructions.push({
      type: "dope_sites",
      params: { dopant: dopantNodes[0].element, fraction: 0.10, site: "random" },
      order: order++,
    });
  }

  const phononNodes = graph.nodes.filter(n => n.type === "phonon_mediator");
  if (phononNodes.length > 0) {
    instructions.push({
      type: "add_phonon_enhancer",
      params: { mode: "soft-phonon", frequency: 200 },
      order: order++,
    });
  }

  const strainNodes = graph.nodes.filter(n => n.type === "strain_buffer");
  if (strainNodes.length > 0) {
    instructions.push({
      type: "apply_strain",
      params: { type: "hydrostatic", magnitude: 0.02 },
      order: order++,
    });
  }

  const dosNodes = graph.nodes.filter(n => n.type === "dos_enhancer");
  if (dosNodes.length > 0) {
    instructions.push({
      type: "optimize_dos",
      params: { targetDOS: 4.0, method: "van-hove" },
      order: order++,
    });
  }

  if (instructions.length === 0) {
    const fallbackLattice = inferLatticeFromGraph(graph);
    const fallbackGroups = compatibleSymmetryGroups(fallbackLattice);
    instructions.push(
      { type: "create_lattice", params: { latticeType: fallbackLattice, a: 5.0 }, order: 0 },
      { type: "set_symmetry", params: { group: fallbackGroups[0] }, order: 1 },
      { type: "populate_sites", params: { elements: graph.nodes.filter(n => !n.id.startsWith("fn-")).map(n => n.element).slice(0, 3), site: "random" }, order: 2 },
    );
  }

  const program: DesignProgram = {
    id: `prog-from-graph-${graph.id}`,
    name: `converted-${graph.name}`,
    instructions,
    outputFormula: graph.outputFormula,
    outputPrototype: "Pm-3m",
    metadata: {
      complexity: Math.min(1, instructions.length * 0.12 + graph.metadata.nodeCount * 0.03 + graph.metadata.edgeCount * 0.01),
      expressiveness: Math.min(1, (graph.metadata.nodeCount / Math.max(1, instructions.length)) * 0.15),
      generatedAt: Date.now(),
      parentId: graph.id,
      generation: graph.metadata.generation,
      mutationHistory: ["graph-to-program"],
    },
    featureVector: [],
  };

  const result = executeDesignProgram(program);
  program.outputFormula = result.formula;
  program.outputPrototype = result.prototype;
  program.featureVector = result.featureVector;

  return program;
}

const REGISTRY_CAP = 200;

interface ScoredProgram { program: DesignProgram; tc: number; }
interface ScoredGraph { graph: DesignGraph; tc: number; }

class DesignRegistry {
  private programsScored: ScoredProgram[] = [];
  private graphsScored: ScoredGraph[] = [];
  private _programBestTc = 0;
  private _programBestFormula = "";
  private _graphBestTc = 0;
  private _graphBestFormula = "";
  private _conversions = 0;
  private progToGraphMap = new Map<string, string>();

  private elitePrune<T extends { tc: number }>(arr: T[], cap: number): void {
    if (arr.length <= cap) return;
    arr.sort((a, b) => b.tc - a.tc);
    arr.length = cap;
  }

  registerProgram(program: DesignProgram, tc: number): void {
    this.programsScored.push({ program, tc });
    this.elitePrune(this.programsScored, REGISTRY_CAP);
    if (tc > this._programBestTc) {
      this._programBestTc = tc;
      this._programBestFormula = program.outputFormula;
    }
  }

  registerGraph(graph: DesignGraph, tc: number): void {
    this.graphsScored.push({ graph, tc });
    this.elitePrune(this.graphsScored, REGISTRY_CAP);
    if (tc > this._graphBestTc) {
      this._graphBestTc = tc;
      this._graphBestFormula = graph.outputFormula;
    }
  }

  linkProgramToGraph(programId: string, graphId: string): void {
    this.progToGraphMap.set(programId, graphId);
  }

  recordConversion(): void { this._conversions++; }

  get programs(): DesignProgram[] { return this.programsScored.map(s => s.program); }
  get graphs(): DesignGraph[] { return this.graphsScored.map(s => s.graph); }
  get programBestTc(): number { return this._programBestTc; }
  get programBestFormula(): string { return this._programBestFormula; }
  get graphBestTc(): number { return this._graphBestTc; }
  get graphBestFormula(): string { return this._graphBestFormula; }
  get totalConversions(): number { return this._conversions; }

  findMatchingGraph(prog: DesignProgram): DesignGraph | undefined {
    const linkedId = this.progToGraphMap.get(prog.id);
    if (linkedId) {
      const found = this.graphsScored.find(s => s.graph.id === linkedId);
      if (found) return found.graph;
    }
    return this.graphsScored.find(s => s.graph.metadata.parentId === prog.id)?.graph;
  }
}

const registry = new DesignRegistry();

export function registerProgram(program: DesignProgram, tc: number): void {
  registry.registerProgram(program, tc);
}

export function registerGraph(graph: DesignGraph, tc: number): void {
  registry.registerGraph(graph, tc);
}

export function linkProgramToGraph(programId: string, graphId: string): void {
  registry.linkProgramToGraph(programId, graphId);
}

export function recordConversion(): void { registry.recordConversion(); }

export function getDesignRepresentationStats(): DesignRepresentationStats {
  const programs = registry.programs;
  const graphs = registry.graphs;

  const instrFreq: Record<string, number> = {};
  const genDist: Record<number, number> = {};
  for (const p of programs) {
    for (const inst of p.instructions) instrFreq[inst.type] = (instrFreq[inst.type] || 0) + 1;
    genDist[p.metadata.generation] = (genDist[p.metadata.generation] || 0) + 1;
  }

  const compFreq: Record<string, number> = {};
  const edgeFreq: Record<string, number> = {};
  for (const g of graphs) {
    for (const n of g.nodes) compFreq[n.type] = (compFreq[n.type] || 0) + 1;
    for (const e of g.edges) edgeFreq[e.type] = (edgeFreq[e.type] || 0) + 1;
  }

  let avgFeatureCorr = 0;
  if (programs.length > 0 && graphs.length > 0) {
    let corrSum = 0;
    let corrCount = 0;
    for (const prog of programs) {
      if (prog.featureVector.length === 0) continue;
      const matchingGraph = registry.findMatchingGraph(prog);
      if (!matchingGraph) continue;
      const gNodes = matchingGraph.nodes;
      const elementNodes = gNodes.filter(n => !n.id.startsWith("fn-"));
      const totalGraphMass = elementNodes.reduce((s, n) => s + (n.properties.atomicMass || 0), 0);
      const avgGraphEneg = elementNodes.length > 0
        ? elementNodes.reduce((s, n) => s + (n.properties.electronegativity || 0), 0) / elementNodes.length
        : 0;
      const avgGraphRadius = elementNodes.length > 0
        ? elementNodes.reduce((s, n) => s + (n.properties.ionicRadius || 0), 0) / elementNodes.length
        : 0;
      const totalGraphElectrons = elementNodes.reduce((s, n) => s + (n.properties.electronCount || 0), 0);
      const uniqueElements = new Set(elementNodes.map(n => n.element)).size;
      const hNodes = elementNodes.filter(n => n.element === "H");
      const hydrogenFraction = elementNodes.length > 0 ? hNodes.length / elementNodes.length : 0;
      const interlayerEdges = matchingGraph.edges.filter(e => e.type === "interlayer");
      const layerProxy = interlayerEdges.length > 0 ? Math.min(1, interlayerEdges.length / 4) : 0;

      const gvec: number[] = [
        softNorm(totalGraphMass, 500),
        avgGraphEneg / 4,
        avgGraphRadius / 2.5,
        softNorm(totalGraphElectrons, 200),
        softNorm(uniqueElements, 8),
        hydrogenFraction,
        layerProxy,
      ];

      const pv = prog.featureVector.slice(0, gvec.length);
      while (pv.length < gvec.length) pv.push(0);
      let dot = 0, magP = 0, magG = 0;
      for (let i = 0; i < gvec.length; i++) {
        dot += pv[i] * gvec[i];
        magP += pv[i] * pv[i];
        magG += gvec[i] * gvec[i];
      }
      const denom = Math.sqrt(magP) * Math.sqrt(magG);
      if (denom > 0) {
        corrSum += dot / denom;
        corrCount++;
      }
    }
    avgFeatureCorr = corrCount > 0 ? Math.round(corrSum / corrCount * 100) / 100 : 0;
  }

  return {
    programs: {
      total: programs.length,
      avgComplexity: programs.length > 0
        ? Math.round(programs.reduce((s, p) => s + p.metadata.complexity, 0) / programs.length * 100) / 100
        : 0,
      avgInstructions: programs.length > 0
        ? Math.round(programs.reduce((s, p) => s + p.instructions.length, 0) / programs.length * 10) / 10
        : 0,
      bestFormula: registry.programBestFormula,
      bestTc: Math.round(registry.programBestTc * 10) / 10,
      instructionFrequency: instrFreq,
      generationDistribution: genDist,
    },
    graphs: {
      total: graphs.length,
      avgNodes: graphs.length > 0
        ? Math.round(graphs.reduce((s, g) => s + g.metadata.nodeCount, 0) / graphs.length * 10) / 10
        : 0,
      avgEdges: graphs.length > 0
        ? Math.round(graphs.reduce((s, g) => s + g.metadata.edgeCount, 0) / graphs.length * 10) / 10
        : 0,
      avgConnectivity: graphs.length > 0
        ? Math.round(graphs.reduce((s, g) => s + g.metadata.connectivity, 0) / graphs.length * 1000) / 1000
        : 0,
      bestFormula: registry.graphBestFormula,
      bestTc: Math.round(registry.graphBestTc * 10) / 10,
      componentFrequency: compFreq,
      edgeTypeFrequency: edgeFreq,
    },
    crossRepresentation: {
      programToGraphConversions: registry.totalConversions,
      graphToProgramConversions: registry.totalConversions,
      avgFeatureCorrelation: avgFeatureCorr,
    },
  };
}
