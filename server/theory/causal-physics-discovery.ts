import { computeMultiScaleFeatures, type MultiScaleFeatures, computeCrossScaleCoupling } from "./multi-scale-engine";
import { computePairingProfile, type PairingProfile } from "../physics/pairing-mechanisms";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";

export type CausalVariableCategory =
  | "atomic_structure" | "electronic_structure" | "phonon_properties"
  | "pairing_interactions" | "thermodynamic_conditions" | "superconducting_properties"
  | "topological" | "synthesis" | "intervention";

export interface CausalVariable {
  name: string;
  category: CausalVariableCategory;
  description: string;
  unit: string;
  isIntervention: boolean;
  range: [number, number];
}

const CAUSAL_VARIABLES: CausalVariable[] = [
  { name: "atomic_mass_avg", category: "atomic_structure", description: "Average atomic mass", unit: "amu", isIntervention: false, range: [1, 260] },
  { name: "electronegativity_spread", category: "atomic_structure", description: "Electronegativity range", unit: "Pauling", isIntervention: false, range: [0, 3] },
  { name: "coordination_number", category: "atomic_structure", description: "Average coordination", unit: "dimensionless", isIntervention: false, range: [1, 12] },
  { name: "bond_length_dist", category: "atomic_structure", description: "Bond length distribution", unit: "Angstrom", isIntervention: false, range: [0, 5] },
  { name: "charge_transfer", category: "atomic_structure", description: "Charge transfer magnitude", unit: "e", isIntervention: false, range: [0, 3] },
  { name: "hydrogen_density", category: "atomic_structure", description: "Hydrogen fraction", unit: "fraction", isIntervention: true, range: [0, 1] },
  { name: "DOS_EF", category: "electronic_structure", description: "DOS at Fermi level", unit: "states/eV", isIntervention: false, range: [0, 10] },
  { name: "bandwidth", category: "electronic_structure", description: "Electronic bandwidth", unit: "eV", isIntervention: false, range: [0, 15] },
  { name: "van_hove_distance", category: "electronic_structure", description: "Van Hove singularity proximity", unit: "eV", isIntervention: false, range: [0, 2] },
  { name: "band_flatness", category: "electronic_structure", description: "Band flatness parameter", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "nesting_score", category: "electronic_structure", description: "Fermi surface nesting", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "mott_proximity", category: "electronic_structure", description: "Proximity to Mott transition", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "phonon_freq", category: "phonon_properties", description: "Characteristic phonon frequency", unit: "cm-1", isIntervention: false, range: [10, 3000] },
  { name: "debye_temp", category: "phonon_properties", description: "Debye temperature", unit: "K", isIntervention: false, range: [50, 2000] },
  { name: "phonon_softening", category: "phonon_properties", description: "Phonon softening parameter", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "anharmonicity", category: "phonon_properties", description: "Anharmonic contribution", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "lambda", category: "pairing_interactions", description: "Electron-phonon coupling", unit: "dimensionless", isIntervention: false, range: [0, 5] },
  { name: "mu_star", category: "pairing_interactions", description: "Coulomb pseudopotential", unit: "dimensionless", isIntervention: false, range: [0, 0.3] },
  { name: "spin_fluct_strength", category: "pairing_interactions", description: "Spin fluctuation coupling", unit: "dimensionless", isIntervention: false, range: [0, 2] },
  { name: "orbital_fluct", category: "pairing_interactions", description: "Orbital fluctuation strength", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "pairing_symmetry", category: "pairing_interactions", description: "Dominant pairing symmetry (0=s,1=d,2=p)", unit: "enum", isIntervention: false, range: [0, 2] },
  { name: "pressure", category: "thermodynamic_conditions", description: "Applied pressure", unit: "GPa", isIntervention: true, range: [0, 500] },
  { name: "temperature", category: "thermodynamic_conditions", description: "Temperature", unit: "K", isIntervention: true, range: [0, 500] },
  { name: "strain", category: "thermodynamic_conditions", description: "Applied strain", unit: "fraction", isIntervention: true, range: [-0.1, 0.1] },
  { name: "doping", category: "thermodynamic_conditions", description: "Doping level", unit: "fraction", isIntervention: true, range: [0, 0.5] },
  { name: "defect_density", category: "thermodynamic_conditions", description: "Defect concentration", unit: "cm-3", isIntervention: true, range: [0, 1e20] },
  { name: "Tc", category: "superconducting_properties", description: "Critical temperature", unit: "K", isIntervention: false, range: [0, 300] },
  { name: "coupling_strength", category: "superconducting_properties", description: "Composite coupling", unit: "dimensionless", isIntervention: false, range: [0, 10] },
  { name: "layeredness", category: "atomic_structure", description: "Structural layeredness", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "dimensionality", category: "atomic_structure", description: "Structural dimensionality", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "topology_z2", category: "topological", description: "Z2 topological invariant", unit: "dimensionless", isIntervention: false, range: [0, 1] },
  { name: "berry_phase", category: "topological", description: "Berry phase proxy", unit: "radians", isIntervention: false, range: [0, Math.PI] },
];

export interface CausalEdge {
  source: string;
  target: string;
  strength: number;
  direction: "forward" | "bidirectional";
  mechanism: string;
  confidence: number;
  evidenceCount: number;
  validated: boolean;
}

export interface CausalGraph {
  nodes: string[];
  edges: CausalEdge[];
  discoveredAt: number;
  method: string;
  datasetSize: number;
}

export interface OntologyNode {
  variable: string;
  category: CausalVariableCategory;
  level: number;
  parents: string[];
  children: string[];
}

const PHYSICS_ONTOLOGY: OntologyNode[] = [
  { variable: "atomic_mass_avg", category: "atomic_structure", level: 0, parents: [], children: ["phonon_freq", "debye_temp"] },
  { variable: "coordination_number", category: "atomic_structure", level: 0, parents: [], children: ["bandwidth", "DOS_EF"] },
  { variable: "charge_transfer", category: "atomic_structure", level: 0, parents: [], children: ["DOS_EF", "nesting_score"] },
  { variable: "hydrogen_density", category: "atomic_structure", level: 0, parents: [], children: ["phonon_freq", "lambda"] },
  { variable: "electronegativity_spread", category: "atomic_structure", level: 0, parents: [], children: ["charge_transfer", "band_flatness"] },
  { variable: "layeredness", category: "atomic_structure", level: 0, parents: [], children: ["dimensionality", "nesting_score", "bandwidth"] },
  { variable: "dimensionality", category: "atomic_structure", level: 1, parents: ["layeredness"], children: ["DOS_EF", "nesting_score"] },
  { variable: "pressure", category: "thermodynamic_conditions", level: 0, parents: [], children: ["phonon_freq", "bandwidth", "lambda"] },
  { variable: "strain", category: "thermodynamic_conditions", level: 0, parents: [], children: ["bandwidth", "DOS_EF", "phonon_freq"] },
  { variable: "doping", category: "thermodynamic_conditions", level: 0, parents: [], children: ["DOS_EF", "nesting_score", "mu_star"] },
  { variable: "defect_density", category: "thermodynamic_conditions", level: 0, parents: [], children: ["phonon_softening", "mu_star"] },
  { variable: "DOS_EF", category: "electronic_structure", level: 2, parents: ["coordination_number", "charge_transfer", "doping", "strain", "dimensionality"], children: ["lambda", "nesting_score"] },
  { variable: "bandwidth", category: "electronic_structure", level: 2, parents: ["coordination_number", "pressure", "strain", "layeredness"], children: ["mott_proximity", "DOS_EF"] },
  { variable: "band_flatness", category: "electronic_structure", level: 2, parents: ["electronegativity_spread"], children: ["van_hove_distance", "DOS_EF"] },
  { variable: "van_hove_distance", category: "electronic_structure", level: 2, parents: ["band_flatness"], children: ["DOS_EF"] },
  { variable: "nesting_score", category: "electronic_structure", level: 2, parents: ["DOS_EF", "layeredness", "dimensionality", "doping", "charge_transfer"], children: ["spin_fluct_strength", "lambda"] },
  { variable: "mott_proximity", category: "electronic_structure", level: 2, parents: ["bandwidth"], children: ["spin_fluct_strength", "mu_star"] },
  { variable: "phonon_freq", category: "phonon_properties", level: 1, parents: ["atomic_mass_avg", "pressure", "hydrogen_density", "strain"], children: ["lambda", "debye_temp"] },
  { variable: "debye_temp", category: "phonon_properties", level: 1, parents: ["atomic_mass_avg", "phonon_freq"], children: ["Tc"] },
  { variable: "phonon_softening", category: "phonon_properties", level: 1, parents: ["defect_density"], children: ["lambda", "anharmonicity"] },
  { variable: "anharmonicity", category: "phonon_properties", level: 2, parents: ["phonon_softening"], children: ["lambda"] },
  { variable: "lambda", category: "pairing_interactions", level: 3, parents: ["DOS_EF", "phonon_freq", "nesting_score", "hydrogen_density", "pressure", "phonon_softening", "anharmonicity"], children: ["Tc", "coupling_strength"] },
  { variable: "mu_star", category: "pairing_interactions", level: 3, parents: ["doping", "mott_proximity", "defect_density"], children: ["Tc"] },
  { variable: "spin_fluct_strength", category: "pairing_interactions", level: 3, parents: ["nesting_score", "mott_proximity"], children: ["Tc", "pairing_symmetry"] },
  { variable: "orbital_fluct", category: "pairing_interactions", level: 3, parents: ["nesting_score"], children: ["pairing_symmetry"] },
  { variable: "pairing_symmetry", category: "pairing_interactions", level: 3, parents: ["spin_fluct_strength", "orbital_fluct"], children: ["Tc"] },
  { variable: "topology_z2", category: "topological", level: 2, parents: [], children: ["pairing_symmetry", "Tc"] },
  { variable: "berry_phase", category: "topological", level: 2, parents: [], children: ["topology_z2"] },
  { variable: "coupling_strength", category: "superconducting_properties", level: 4, parents: ["lambda"], children: ["Tc"] },
  { variable: "Tc", category: "superconducting_properties", level: 5, parents: ["lambda", "mu_star", "debye_temp", "spin_fluct_strength", "pairing_symmetry", "coupling_strength", "topology_z2"], children: [] },
];

export interface CausalDataRecord {
  formula: string;
  material_family: string;
  [key: string]: number | string;
}

export function buildCausalDataRecord(formula: string): CausalDataRecord {
  let multiScale: MultiScaleFeatures | null = null;
  try { multiScale = computeMultiScaleFeatures(formula); } catch {}

  let features: Record<string, number> = {};
  try { features = extractFeatures(formula); } catch {}

  let gb: { tcPredicted: number } = { tcPredicted: 0 };
  try { gb = gbPredict(features); } catch {}

  let pairing: PairingProfile | null = null;
  try { pairing = computePairingProfile(formula); } catch {}

  const lambda = features.electronPhononLambda ?? 0.5;
  const DOS_EF = multiScale?.electronic?.DOS_EF ?? 1.0;
  const omega_log = features.logPhononFreq ?? 400;
  const bandwidth = multiScale?.electronic?.bandwidth ?? 2.0;
  const nesting = multiScale?.electronic?.nesting_score ?? 0.1;
  const mott = multiScale?.electronic?.mott_proximity ?? 0;

  const hasH = formula.includes("H");
  const hasCu = formula.includes("Cu");
  const hasFe = formula.includes("Fe");
  let family = "other";
  if (hasH) family = "hydride";
  else if (hasCu && formula.includes("O")) family = "cuprate";
  else if (hasFe && (formula.includes("As") || formula.includes("Se"))) family = "iron-based";
  else family = "intermetallic";

  return {
    formula,
    material_family: family,
    atomic_mass_avg: multiScale?.atomic?.atomic_mass_avg ?? 50,
    electronegativity_spread: multiScale?.atomic?.electronegativity_spread ?? 1.5,
    coordination_number: multiScale?.atomic?.coordination_number ?? 6,
    bond_length_dist: multiScale?.atomic?.bond_length_distribution ?? 0.2,
    charge_transfer: multiScale?.atomic?.charge_transfer ?? 0.3,
    hydrogen_density: hasH ? 0.3 + Math.random() * 0.5 : 0,
    DOS_EF,
    bandwidth,
    van_hove_distance: multiScale?.electronic?.van_hove_distance ?? 0.5,
    band_flatness: multiScale?.electronic?.band_flatness ?? 0.2,
    nesting_score: nesting,
    mott_proximity: mott,
    phonon_freq: omega_log,
    debye_temp: omega_log * 1.44,
    phonon_softening: 0.1 + Math.random() * 0.3,
    anharmonicity: 0.05 + Math.random() * 0.2,
    lambda,
    mu_star: features.muStarEstimate ?? 0.10,
    spin_fluct_strength: pairing?.spin?.chiQ ?? nesting * 0.5,
    orbital_fluct: pairing?.orbital?.orbitalFluctuation ?? 0.1,
    pairing_symmetry: pairing?.pairingSymmetry === "d-wave" ? 1 : 0,
    pressure: features.avgBulkModulus ? features.avgBulkModulus * 0.1 : 0,
    temperature: 300,
    strain: 0,
    doping: 0,
    defect_density: 1e15,
    Tc: gb.tcPredicted,
    coupling_strength: lambda * DOS_EF,
    layeredness: multiScale?.mesoscopic?.layeredness ?? 0.1,
    dimensionality: multiScale?.mesoscopic?.dimensionality ?? 0.5,
    topology_z2: features.z2Score ?? 0,
    berry_phase: features.berryPhaseProxy ?? 0,
  };
}

function computeCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i] - mx;
    const yi = y[i] - my;
    num += xi * yi;
    dx += xi * xi;
    dy += yi * yi;
  }
  const denom = Math.sqrt(dx * dy);
  return denom > 1e-10 ? num / denom : 0;
}

function partialCorrelation(x: number[], y: number[], z: number[]): number {
  const rxy = computeCorrelation(x, y);
  const rxz = computeCorrelation(x, z);
  const ryz = computeCorrelation(y, z);
  const denom = Math.sqrt((1 - rxz * rxz) * (1 - ryz * ryz));
  if (denom < 1e-10) return 0;
  return (rxy - rxz * ryz) / denom;
}

function getColumn(dataset: CausalDataRecord[], varName: string): number[] {
  return dataset.map(r => {
    const v = r[varName];
    return typeof v === "number" ? v : 0;
  });
}

const FORBIDDEN_EDGES = new Set([
  "atomic_mass_avg->doping",
  "doping->atomic_mass_avg",
  "atomic_mass_avg->pressure",
  "pressure->atomic_mass_avg",
  "atomic_mass_avg->strain",
  "strain->atomic_mass_avg",
  "atomic_mass_avg->defect_density",
  "defect_density->atomic_mass_avg",
  "atomic_mass_avg->temperature",
  "temperature->atomic_mass_avg",
  "coordination_number->doping",
  "doping->coordination_number",
  "coordination_number->pressure",
  "pressure->coordination_number",
  "electronegativity_spread->doping",
  "doping->electronegativity_spread",
  "electronegativity_spread->pressure",
  "pressure->electronegativity_spread",
  "layeredness->doping",
  "doping->layeredness",
  "hydrogen_density->doping",
  "doping->hydrogen_density",
  "doping->phonon_freq",
  "phonon_freq->doping",
  "Tc->lambda",
  "Tc->DOS_EF",
  "Tc->phonon_freq",
  "Tc->pressure",
  "Tc->doping",
  "Tc->atomic_mass_avg",
]);

function isEdgeForbidden(source: string, target: string): boolean {
  return FORBIDDEN_EDGES.has(`${source}->${target}`);
}

function isOntologyAllowed(source: string, target: string): boolean {
  const srcNode = PHYSICS_ONTOLOGY.find(o => o.variable === source);
  const tgtNode = PHYSICS_ONTOLOGY.find(o => o.variable === target);
  if (!srcNode || !tgtNode) return true;
  if (srcNode.children.includes(target)) return true;
  if (tgtNode.children.includes(source)) return false;
  if (srcNode.level <= tgtNode.level) return true;
  return false;
}

export function discoverCausalGraph(dataset: CausalDataRecord[], significanceThreshold: number = 0.15): CausalGraph {
  const variables = CAUSAL_VARIABLES.map(v => v.name);
  const edges: CausalEdge[] = [];
  const columns: Map<string, number[]> = new Map();

  for (const v of variables) {
    columns.set(v, getColumn(dataset, v));
  }

  for (let i = 0; i < variables.length; i++) {
    for (let j = i + 1; j < variables.length; j++) {
      const vi = variables[i];
      const vj = variables[j];
      const xi = columns.get(vi)!;
      const xj = columns.get(vj)!;
      const corr = computeCorrelation(xi, xj);

      if (Math.abs(corr) < significanceThreshold) continue;

      let isConditionallyIndependent = false;
      for (let k = 0; k < variables.length; k++) {
        if (k === i || k === j) continue;
        const xk = columns.get(variables[k])!;
        const partialCorr = partialCorrelation(xi, xj, xk);
        if (Math.abs(partialCorr) < significanceThreshold * 0.5) {
          isConditionallyIndependent = true;
          break;
        }
      }

      if (isConditionallyIndependent) continue;

      const ontologyI = PHYSICS_ONTOLOGY.find(o => o.variable === vi);
      const ontologyJ = PHYSICS_ONTOLOGY.find(o => o.variable === vj);
      let source = vi, target = vj;
      let direction: "forward" | "bidirectional" = "forward";

      if (ontologyI && ontologyJ) {
        if (ontologyI.level < ontologyJ.level) {
          source = vi; target = vj;
        } else if (ontologyJ.level < ontologyI.level) {
          source = vj; target = vi;
        } else {
          direction = "bidirectional";
        }
      }

      if (ontologyI && ontologyI.children.includes(vj)) {
        source = vi; target = vj; direction = "forward";
      } else if (ontologyJ && ontologyJ.children.includes(vi)) {
        source = vj; target = vi; direction = "forward";
      }

      if (isEdgeForbidden(source, target)) continue;
      if (!isOntologyAllowed(source, target)) continue;

      const mechanism = inferMechanism(source, target);

      edges.push({
        source, target,
        strength: Math.abs(corr),
        direction,
        mechanism,
        confidence: Math.min(1, Math.abs(corr) * 1.5),
        evidenceCount: dataset.length,
        validated: false,
      });
    }
  }

  edges.sort((a, b) => b.strength - a.strength);

  return {
    nodes: variables.filter(v => edges.some(e => e.source === v || e.target === v)),
    edges: edges.slice(0, 50),
    discoveredAt: Date.now(),
    method: "PC-algorithm with physics ontology and forbidden-edge priors",
    datasetSize: dataset.length,
  };
}

function inferMechanism(source: string, target: string): string {
  const mechanisms: Record<string, string> = {
    "pressure_phonon_freq": "Pressure increases lattice stiffness, shifting phonon frequencies",
    "hydrogen_density_phonon_freq": "Dense hydrogen networks create high-frequency optical phonon modes",
    "hydrogen_density_lambda": "Hydrogen sigma-bonding states enhance electron-phonon matrix elements",
    "phonon_freq_lambda": "Phonon frequency determines electron-phonon coupling via Eliashberg theory",
    "DOS_EF_lambda": "Higher DOS at Fermi level increases phase space for electron-phonon scattering",
    "lambda_Tc": "Electron-phonon coupling directly determines Tc via McMillan/Allen-Dynes equation",
    "mu_star_Tc": "Coulomb pseudopotential suppresses pairing, reducing Tc",
    "nesting_score_spin_fluct_strength": "Fermi surface nesting enhances spin susceptibility at nesting vector",
    "spin_fluct_strength_Tc": "Spin fluctuations provide alternative pairing channel (d-wave)",
    "mott_proximity_spin_fluct_strength": "Proximity to Mott transition enhances spin correlations",
    "band_flatness_DOS_EF": "Flat bands create Van Hove singularities with large DOS peaks",
    "layeredness_nesting_score": "Layered structures promote 2D Fermi surface nesting",
    "dimensionality_DOS_EF": "Reduced dimensionality concentrates DOS",
    "debye_temp_Tc": "Debye temperature sets energy scale for phonon-mediated pairing",
    "coupling_strength_Tc": "Composite coupling directly determines critical temperature",
    "topology_z2_pairing_symmetry": "Nontrivial topology constrains pairing symmetry",
    "pressure_bandwidth": "Pressure increases orbital overlap, widening bands",
    "doping_DOS_EF": "Doping shifts Fermi level, changing DOS",
    "defect_density_phonon_softening": "Defects break translational symmetry, softening phonons",
    "strain_phonon_freq": "Lattice strain modifies force constants and phonon frequencies",
    "phonon_softening_lambda": "Soft phonon modes enhance electron-phonon coupling",
    "charge_transfer_DOS_EF": "Charge redistribution modifies electronic structure near EF",
  };
  const key = `${source}_${target}`;
  return mechanisms[key] ?? `${source} influences ${target} through physics coupling`;
}

export interface InterventionResult {
  variable: string;
  originalValue: number;
  modifiedValue: number;
  changePercent: number;
  effects: { variable: string; originalValue: number; newValue: number; changePercent: number; causalPathLength: number }[];
  tcChange: number;
  tcOriginal: number;
  tcNew: number;
  causalPathway: string[];
}

export function simulateIntervention(
  record: CausalDataRecord,
  variable: string,
  newValue: number,
  graph: CausalGraph,
): InterventionResult {
  const original = { ...record };
  const modified = { ...record };
  const originalValue = typeof original[variable] === "number" ? original[variable] as number : 0;
  modified[variable] = newValue;

  const effects: InterventionResult["effects"] = [];
  const causalPathway: string[] = [variable];
  const visited = new Set<string>([variable]);

  function propagate(fromVar: string, depth: number) {
    if (depth > 5) return;
    const outEdges = graph.edges.filter(e => e.source === fromVar && !visited.has(e.target));

    for (const edge of outEdges) {
      visited.add(edge.target);
      const origVal = typeof original[edge.target] === "number" ? original[edge.target] as number : 0;
      const sourceOrigVal = typeof original[edge.source] === "number" ? original[edge.source] as number : 0;
      const sourceNewVal = typeof modified[edge.source] === "number" ? modified[edge.source] as number : 0;

      let sourceChange: number;
      if (Math.abs(sourceOrigVal) < 1e-10) {
        sourceChange = sourceNewVal > 0 ? 1.0 : sourceNewVal < 0 ? -1.0 : 0;
      } else {
        sourceChange = (sourceNewVal - sourceOrigVal) / Math.abs(sourceOrigVal);
      }
      const elasticity = edge.strength * 0.8;
      const delta = sourceChange * elasticity;
      const newVal = Math.abs(origVal) > 1e-10
        ? origVal * (1 + delta)
        : origVal + delta * (edge.strength > 0 ? 1 : -1);
      modified[edge.target] = newVal;

      effects.push({
        variable: edge.target,
        originalValue: origVal,
        newValue: Math.round(newVal * 1000) / 1000,
        changePercent: origVal !== 0 ? Math.round(((newVal - origVal) / Math.abs(origVal)) * 10000) / 100 : 0,
        causalPathLength: depth + 1,
      });

      causalPathway.push(edge.target);
      propagate(edge.target, depth + 1);
    }
  }

  propagate(variable, 0);

  const tcOriginal = typeof original.Tc === "number" ? original.Tc as number : 0;
  let tcNew = typeof modified.Tc === "number" ? modified.Tc as number : tcOriginal;

  if (!visited.has("Tc") || Math.abs(tcNew - tcOriginal) < 0.01) {
    const lam = typeof modified.lambda === "number" ? modified.lambda as number : 0.5;
    const omega = typeof modified.phonon_freq === "number" ? modified.phonon_freq as number : 400;
    const muStar = typeof modified.mu_star === "number" ? modified.mu_star as number : 0.1;
    const muEff = muStar * (1 + 0.62 * lam);
    const denom = lam - muEff;
    if (denom > 0.05) {
      const lambdaBar = 2.46 * (1 + 3.8 * muStar);
      const f1 = Math.pow(1 + Math.pow(lam / lambdaBar, 3 / 2), 1 / 3);
      tcNew = (omega * 1.4388 / 1.2) * f1 * Math.exp(-1.04 * (1 + lam) / denom);
      tcNew = Math.max(0, Math.min(400, tcNew));
    }
  }

  return {
    variable,
    originalValue,
    modifiedValue: newValue,
    changePercent: originalValue !== 0 ? Math.round(((newValue - originalValue) / Math.abs(originalValue)) * 10000) / 100 : 0,
    effects,
    tcChange: Math.round((tcNew - tcOriginal) * 100) / 100,
    tcOriginal: Math.round(tcOriginal * 100) / 100,
    tcNew: Math.round(tcNew * 100) / 100,
    causalPathway,
  };
}

export interface CounterfactualResult {
  question: string;
  variable: string;
  modification: string;
  originalTc: number;
  counterfactualTc: number;
  tcDelta: number;
  tcDeltaPercent: number;
  propagatedEffects: { variable: string; change: number }[];
  designImplication: string;
}

export function runCounterfactual(
  record: CausalDataRecord,
  variable: string,
  modificationPercent: number,
  graph: CausalGraph,
): CounterfactualResult {
  const origVal = typeof record[variable] === "number" ? record[variable] as number : 0;
  const newVal = origVal * (1 + modificationPercent / 100);

  const intervention = simulateIntervention(record, variable, newVal, graph);

  const question = `What if ${variable} were ${modificationPercent > 0 ? "increased" : "decreased"} by ${Math.abs(modificationPercent)}%?`;
  const modification = `${variable}: ${origVal.toFixed(2)} -> ${newVal.toFixed(2)}`;

  const propagated = intervention.effects.map(e => ({
    variable: e.variable,
    change: e.changePercent,
  }));

  let implication = "No significant design implication.";
  if (Math.abs(intervention.tcChange) > 5) {
    if (intervention.tcChange > 0) {
      implication = `Increasing ${variable} by ${Math.abs(modificationPercent)}% could raise Tc by ${intervention.tcChange.toFixed(1)}K. Target materials with higher ${variable}.`;
    } else {
      implication = `Increasing ${variable} by ${Math.abs(modificationPercent)}% would reduce Tc by ${Math.abs(intervention.tcChange).toFixed(1)}K. Avoid materials with excessive ${variable}.`;
    }
  } else if (Math.abs(intervention.tcChange) > 1) {
    implication = `Moderate Tc sensitivity to ${variable}. Consider as secondary optimization parameter.`;
  }

  return {
    question,
    variable,
    modification,
    originalTc: intervention.tcOriginal,
    counterfactualTc: intervention.tcNew,
    tcDelta: intervention.tcChange,
    tcDeltaPercent: intervention.tcOriginal > 0 ? Math.round((intervention.tcChange / intervention.tcOriginal) * 10000) / 100 : 0,
    propagatedEffects: propagated,
    designImplication: implication,
  };
}

export interface CausalMechanismHypothesis {
  id: string;
  statement: string;
  causalChain: string[];
  mechanism: string;
  supportingEdges: { source: string; target: string; strength: number }[];
  confidence: number;
  testableIntervention: string;
  materialFamilies: string[];
  discoveredAt: number;
}

function generateMechanismHypotheses(graph: CausalGraph, dataset: CausalDataRecord[]): CausalMechanismHypothesis[] {
  const hypotheses: CausalMechanismHypothesis[] = [];

  const pathsToTc: string[][] = [];
  function findPaths(current: string, path: string[], visited: Set<string>) {
    if (current === "Tc") {
      pathsToTc.push([...path, "Tc"]);
      return;
    }
    if (path.length > 5 || visited.has(current)) return;
    visited.add(current);
    const outEdges = graph.edges.filter(e => e.source === current);
    for (const edge of outEdges) {
      findPaths(edge.target, [...path, current], new Set(visited));
    }
  }

  const roots = graph.nodes.filter(n => !graph.edges.some(e => e.target === n));
  for (const root of roots.slice(0, 10)) {
    findPaths(root, [], new Set());
  }

  for (const path of pathsToTc.slice(0, 15)) {
    if (path.length < 3) continue;

    const edges = [];
    let avgStrength = 0;
    for (let i = 0; i < path.length - 1; i++) {
      const edge = graph.edges.find(e => e.source === path[i] && e.target === path[i + 1]);
      if (edge) {
        edges.push({ source: edge.source, target: edge.target, strength: edge.strength });
        avgStrength += edge.strength;
      }
    }
    avgStrength = edges.length > 0 ? avgStrength / edges.length : 0;

    if (avgStrength < 0.15) continue;

    const chainDesc = path.join(" -> ");
    const mechanism = edges.map(e => inferMechanism(e.source, e.target)).join("; ");

    const families = [...new Set(dataset.map(r => r.material_family as string))];
    const relevantFamilies = families.filter(fam => {
      const famData = dataset.filter(r => r.material_family === fam);
      if (famData.length < 3) return false;
      const startVar = path[0];
      const vals = famData.map(r => typeof r[startVar] === "number" ? r[startVar] as number : 0);
      const variance = vals.reduce((s, v) => s + (v - vals.reduce((a,b)=>a+b,0)/vals.length) ** 2, 0) / vals.length;
      return variance > 0.01;
    });

    const intervention = CAUSAL_VARIABLES.find(v => v.name === path[0] && v.isIntervention);
    const testable = intervention
      ? `Modify ${path[0]} and observe cascading effect through ${path.slice(1, -1).join(", ")} to Tc`
      : `Compare materials with varying ${path[0]} levels to validate causal chain`;

    hypotheses.push({
      id: `hyp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      statement: `Tc is causally influenced by ${path[0]} through the chain: ${chainDesc}`,
      causalChain: path,
      mechanism,
      supportingEdges: edges,
      confidence: Math.min(1, avgStrength * 1.2),
      testableIntervention: testable,
      materialFamilies: relevantFamilies,
      discoveredAt: Date.now(),
    });
  }

  return hypotheses.sort((a, b) => b.confidence - a.confidence);
}

export interface CausalRule {
  antecedent: string;
  consequent: string;
  mechanism: string;
  strength: number;
  causalChain: string[];
  validatedAcross: string[];
}

function extractCausalRules(graph: CausalGraph, dataset: CausalDataRecord[]): CausalRule[] {
  const rules: CausalRule[] = [];
  const families = [...new Set(dataset.map(r => r.material_family as string))];

  for (const edge of graph.edges.filter(e => e.strength > 0.2)) {
    const validFamilies: string[] = [];
    for (const fam of families) {
      const famData = dataset.filter(r => r.material_family === fam);
      if (famData.length < 3) continue;
      const x = famData.map(r => typeof r[edge.source] === "number" ? r[edge.source] as number : 0);
      const y = famData.map(r => typeof r[edge.target] === "number" ? r[edge.target] as number : 0);
      const corr = computeCorrelation(x, y);
      if (Math.abs(corr) > 0.1) validFamilies.push(fam);
    }

    rules.push({
      antecedent: `high ${edge.source}`,
      consequent: `${edge.strength > 0 ? "higher" : "lower"} ${edge.target}`,
      mechanism: edge.mechanism,
      strength: edge.strength,
      causalChain: [edge.source, edge.target],
      validatedAcross: validFamilies,
    });
  }

  return rules.sort((a, b) => b.strength - a.strength);
}

export interface CrossFamilyValidation {
  family: string;
  edgeCount: number;
  avgEdgeStrength: number;
  consistentEdges: number;
  consistencyScore: number;
  dominantPathway: string;
}

function validateAcrossFamilies(graph: CausalGraph, dataset: CausalDataRecord[]): CrossFamilyValidation[] {
  const families = [...new Set(dataset.map(r => r.material_family as string))];
  const results: CrossFamilyValidation[] = [];

  for (const fam of families) {
    const famData = dataset.filter(r => r.material_family === fam);
    if (famData.length < 5) continue;

    let consistentEdges = 0;
    let totalStrength = 0;

    for (const edge of graph.edges) {
      const x = famData.map(r => typeof r[edge.source] === "number" ? r[edge.source] as number : 0);
      const y = famData.map(r => typeof r[edge.target] === "number" ? r[edge.target] as number : 0);
      const localCorr = computeCorrelation(x, y);
      if (Math.sign(localCorr) === Math.sign(edge.strength) && Math.abs(localCorr) > 0.1) {
        consistentEdges++;
      }
      totalStrength += Math.abs(localCorr);
    }

    const tcEdges = graph.edges.filter(e => e.target === "Tc").sort((a, b) => b.strength - a.strength);
    const dominantPath = tcEdges.length > 0 ? `${tcEdges[0].source} -> Tc` : "unknown";

    results.push({
      family: fam,
      edgeCount: graph.edges.length,
      avgEdgeStrength: graph.edges.length > 0 ? totalStrength / graph.edges.length : 0,
      consistentEdges,
      consistencyScore: graph.edges.length > 0 ? consistentEdges / graph.edges.length : 0,
      dominantPathway: dominantPath,
    });
  }

  return results.sort((a, b) => b.consistencyScore - a.consistencyScore);
}

export interface DesignGuidance {
  variable: string;
  direction: "maximize" | "minimize";
  causalImpactOnTc: number;
  mechanism: string;
  rank: number;
}

function generateDesignGuidance(graph: CausalGraph, dataset: CausalDataRecord[]): DesignGuidance[] {
  const guidance: DesignGuidance[] = [];
  const interventionVars = CAUSAL_VARIABLES.filter(v => v.isIntervention);

  for (const iv of interventionVars) {
    const tcEdges = findCausalPathStrength(graph, iv.name, "Tc");
    if (tcEdges.totalStrength < 0.05) continue;

    const x = getColumn(dataset, iv.name);
    const tc = getColumn(dataset, "Tc");
    const corr = computeCorrelation(x, tc);

    guidance.push({
      variable: iv.name,
      direction: corr > 0 ? "maximize" : "minimize",
      causalImpactOnTc: Math.round(tcEdges.totalStrength * 1000) / 1000,
      mechanism: tcEdges.pathway.join(" -> "),
      rank: 0,
    });
  }

  guidance.sort((a, b) => b.causalImpactOnTc - a.causalImpactOnTc);
  guidance.forEach((g, i) => g.rank = i + 1);

  return guidance;
}

function findCausalPathStrength(graph: CausalGraph, from: string, to: string): { totalStrength: number; pathway: string[] } {
  const bestPath: { path: string[]; strength: number } = { path: [], strength: 0 };

  function dfs(current: string, path: string[], minStrength: number, visited: Set<string>) {
    if (current === to) {
      if (minStrength > bestPath.strength) {
        bestPath.path = [...path, to];
        bestPath.strength = minStrength;
      }
      return;
    }
    if (path.length > 6 || visited.has(current)) return;
    visited.add(current);

    const outEdges = graph.edges.filter(e => e.source === current);
    for (const edge of outEdges) {
      dfs(edge.target, [...path, current], Math.min(minStrength, edge.strength), new Set(visited));
    }
  }

  dfs(from, [], 1.0, new Set());
  return { totalStrength: bestPath.strength, pathway: bestPath.path };
}

export interface PressureRegimeComparison {
  ambientEdges: CausalEdge[];
  highPressureEdges: CausalEdge[];
  survivingMechanisms: { source: string; target: string; ambientStrength: number; hpStrength: number }[];
  newMechanisms: { source: string; target: string; strength: number; regime: string }[];
  decompressionInsight: string;
}

function comparePressureRegimes(dataset: CausalDataRecord[], graph: CausalGraph): PressureRegimeComparison {
  const ambient = dataset.filter(r => (r.pressure as number) < 20);
  const highP = dataset.filter(r => (r.pressure as number) >= 20);

  let ambientGraph: CausalGraph = { nodes: [], edges: [], discoveredAt: 0, method: "", datasetSize: 0 };
  let hpGraph: CausalGraph = { nodes: [], edges: [], discoveredAt: 0, method: "", datasetSize: 0 };

  if (ambient.length >= 10) ambientGraph = discoverCausalGraph(ambient, 0.2);
  if (highP.length >= 10) hpGraph = discoverCausalGraph(highP, 0.2);

  const surviving: PressureRegimeComparison["survivingMechanisms"] = [];
  const newMechs: PressureRegimeComparison["newMechanisms"] = [];

  for (const ae of ambientGraph.edges) {
    const hpe = hpGraph.edges.find(e => e.source === ae.source && e.target === ae.target);
    if (hpe) {
      surviving.push({ source: ae.source, target: ae.target, ambientStrength: ae.strength, hpStrength: hpe.strength });
    }
  }

  for (const hpe of hpGraph.edges) {
    if (!ambientGraph.edges.some(e => e.source === hpe.source && e.target === hpe.target)) {
      newMechs.push({ source: hpe.source, target: hpe.target, strength: hpe.strength, regime: "high-pressure" });
    }
  }
  for (const ae of ambientGraph.edges) {
    if (!hpGraph.edges.some(e => e.source === ae.source && e.target === ae.target)) {
      newMechs.push({ source: ae.source, target: ae.target, strength: ae.strength, regime: "ambient" });
    }
  }

  let insight = "Insufficient data for pressure regime comparison.";
  if (surviving.length > 0) {
    const strongest = surviving.sort((a, b) => b.ambientStrength - a.ambientStrength)[0];
    insight = `Mechanism ${strongest.source} -> ${strongest.target} survives across pressure regimes (ambient: ${strongest.ambientStrength.toFixed(2)}, high-P: ${strongest.hpStrength.toFixed(2)}). This pathway may be critical for ambient-pressure superconductivity.`;
  }

  return {
    ambientEdges: ambientGraph.edges.slice(0, 10),
    highPressureEdges: hpGraph.edges.slice(0, 10),
    survivingMechanisms: surviving,
    newMechanisms: newMechs.slice(0, 10),
    decompressionInsight: insight,
  };
}

let discoveredGraphs: CausalGraph[] = [];
let discoveredHypotheses: CausalMechanismHypothesis[] = [];
let discoveredRules: CausalRule[] = [];
let totalDiscoveryRuns = 0;

export function runCausalDiscovery(dataset: CausalDataRecord[]): {
  graph: CausalGraph;
  hypotheses: CausalMechanismHypothesis[];
  rules: CausalRule[];
  crossFamilyValidation: CrossFamilyValidation[];
  designGuidance: DesignGuidance[];
  pressureComparison: PressureRegimeComparison;
} {
  totalDiscoveryRuns++;

  const graph = discoverCausalGraph(dataset);
  const hypotheses = generateMechanismHypotheses(graph, dataset);
  const rules = extractCausalRules(graph, dataset);
  const crossFamilyValidation = validateAcrossFamilies(graph, dataset);
  const designGuidance = generateDesignGuidance(graph, dataset);
  const pressureComparison = comparePressureRegimes(dataset, graph);

  discoveredGraphs.push(graph);
  if (discoveredGraphs.length > 20) discoveredGraphs = discoveredGraphs.slice(-20);

  for (const h of hypotheses) {
    if (!discoveredHypotheses.some(dh => dh.statement === h.statement)) {
      discoveredHypotheses.push(h);
    }
  }
  if (discoveredHypotheses.length > 50) {
    discoveredHypotheses.sort((a, b) => b.confidence - a.confidence);
    discoveredHypotheses = discoveredHypotheses.slice(0, 50);
  }

  for (const r of rules) {
    if (!discoveredRules.some(dr => dr.antecedent === r.antecedent && dr.consequent === r.consequent)) {
      discoveredRules.push(r);
    }
  }
  if (discoveredRules.length > 100) {
    discoveredRules.sort((a, b) => b.strength - a.strength);
    discoveredRules = discoveredRules.slice(0, 100);
  }

  return { graph, hypotheses, rules, crossFamilyValidation, designGuidance, pressureComparison };
}

export function generateCausalDataset(count: number = 60): CausalDataRecord[] {
  const formulas = [
    "MgB2", "NbSn3", "LaH10", "YBa2Cu3O7", "FeSe", "Nb3Ge",
    "CaH6", "SrTiO3", "Bi2Sr2CaCu2O8", "LaFeAsO", "NbN", "VN",
    "TiN", "ZrN", "HfN", "MoC", "NbC", "TaC", "ScH3",
    "BaFe2As2", "LiFeAs", "FeTeSe", "KFe2Se2",
    "NbTiZrHfV", "TaNbHfZrTi", "MoNbTaVW", "LaBH8", "YH6",
    "CeH9", "ThH10", "PrH9", "NdH9", "EuH6",
    "Tl2Ba2CaCu2O8", "HgBa2Ca2Cu3O8", "La2CuO4",
    "PbMo6S8", "LuNi2B2C", "YNi2B2C", "CaC6",
  ];

  const records: CausalDataRecord[] = [];
  for (let i = 0; i < Math.min(count, formulas.length); i++) {
    try {
      const rec = buildCausalDataRecord(formulas[i]);
      if ((rec.Tc as number) <= 0 && (rec.lambda as number) > 0.1) {
        const lam = rec.lambda as number;
        const omega = rec.phonon_freq as number;
        const muStar = rec.mu_star as number;
        const muEff = muStar * (1 + 0.62 * lam);
        const denom = lam - muEff;
        if (denom > 0.05) {
          const lambdaBar2 = 2.46 * (1 + 3.8 * (muStar as number));
          const f1r = Math.pow(1 + Math.pow(lam / lambdaBar2, 3 / 2), 1 / 3);
          rec.Tc = (omega * 1.4388 / 1.2) * f1r * Math.exp(-1.04 * (1 + lam) / denom);
          rec.Tc = Math.max(0, Math.min(300, rec.Tc as number)) + (Math.random() - 0.5) * 2;
        }
      }
      records.push(rec);
    } catch {}
  }

  while (records.length < count) {
    const lambda = 0.3 + Math.random() * 2.5;
    const omega = 100 + Math.random() * 1500;
    const muStar = 0.08 + Math.random() * 0.07;
    const DOS = 0.5 + Math.random() * 4;
    const bw = 1 + Math.random() * 8;
    const nest = Math.random() * 0.8;
    const press = Math.random() * 200;
    const muEff = muStar * (1 + 0.62 * lambda);
    const denom = lambda - muEff;
    let Tc = 0;
    if (denom > 0.05) {
      const lambdaBarS = 2.46 * (1 + 3.8 * muStar);
      const f1s = Math.pow(1 + Math.pow(lambda / lambdaBarS, 3 / 2), 1 / 3);
      Tc = (omega * 1.4388 / 1.2) * f1s * Math.exp(-1.04 * (1 + lambda) / denom);
      Tc = Math.max(0, Math.min(300, Tc));
    }
    Tc += (Math.random() - 0.5) * 5;
    Tc = Math.max(0, Tc);

    records.push({
      formula: `CausalSynth${records.length}`,
      material_family: ["hydride", "cuprate", "iron-based", "intermetallic", "other"][Math.floor(Math.random() * 5)],
      atomic_mass_avg: 20 + Math.random() * 200,
      electronegativity_spread: 0.5 + Math.random() * 2,
      coordination_number: 4 + Math.random() * 8,
      bond_length_dist: 0.1 + Math.random() * 0.4,
      charge_transfer: Math.random() * 1.5,
      hydrogen_density: Math.random() * 0.8,
      DOS_EF: DOS,
      bandwidth: bw,
      van_hove_distance: Math.random() * 1.5,
      band_flatness: Math.random() * 0.5,
      nesting_score: nest,
      mott_proximity: Math.random() * 0.5,
      phonon_freq: omega,
      debye_temp: omega * 1.44,
      phonon_softening: Math.random() * 0.4,
      anharmonicity: Math.random() * 0.3,
      lambda,
      mu_star: muStar,
      spin_fluct_strength: nest * 0.5 + Math.random() * 0.3,
      orbital_fluct: Math.random() * 0.3,
      pairing_symmetry: Math.random() > 0.7 ? 1 : 0,
      pressure: press,
      temperature: 300,
      strain: (Math.random() - 0.5) * 0.05,
      doping: Math.random() * 0.3,
      defect_density: Math.random() * 1e18,
      Tc,
      coupling_strength: lambda * DOS,
      layeredness: Math.random() * 0.8,
      dimensionality: 0.2 + Math.random() * 0.6,
      topology_z2: Math.random() > 0.7 ? 1 : 0,
      berry_phase: Math.random() * Math.PI,
    });
  }

  return records;
}

export interface CausalDiscoveryStats {
  totalRuns: number;
  totalGraphsDiscovered: number;
  totalHypotheses: number;
  totalCausalRules: number;
  causalVariableCount: number;
  interventionVariableCount: number;
  ontologyNodeCount: number;
  topHypotheses: {
    id: string;
    statement: string;
    causalChain: string[];
    confidence: number;
    materialFamilies: string[];
  }[];
  topEdges: {
    source: string;
    target: string;
    strength: number;
    mechanism: string;
  }[];
  variableCategories: Record<string, number>;
  latestGraphSize: { nodes: number; edges: number };
}

export function getCausalDiscoveryStats(): CausalDiscoveryStats {
  const catCounts: Record<string, number> = {};
  for (const v of CAUSAL_VARIABLES) {
    catCounts[v.category] = (catCounts[v.category] || 0) + 1;
  }

  const latestGraph = discoveredGraphs.length > 0 ? discoveredGraphs[discoveredGraphs.length - 1] : null;

  return {
    totalRuns: totalDiscoveryRuns,
    totalGraphsDiscovered: discoveredGraphs.length,
    totalHypotheses: discoveredHypotheses.length,
    totalCausalRules: discoveredRules.length,
    causalVariableCount: CAUSAL_VARIABLES.length,
    interventionVariableCount: CAUSAL_VARIABLES.filter(v => v.isIntervention).length,
    ontologyNodeCount: PHYSICS_ONTOLOGY.length,
    topHypotheses: discoveredHypotheses.slice(0, 8).map(h => ({
      id: h.id,
      statement: h.statement,
      causalChain: h.causalChain,
      confidence: Math.round(h.confidence * 1000) / 1000,
      materialFamilies: h.materialFamilies,
    })),
    topEdges: (latestGraph?.edges ?? []).slice(0, 10).map(e => ({
      source: e.source,
      target: e.target,
      strength: Math.round(e.strength * 1000) / 1000,
      mechanism: e.mechanism,
    })),
    variableCategories: catCounts,
    latestGraphSize: {
      nodes: latestGraph?.nodes.length ?? 0,
      edges: latestGraph?.edges.length ?? 0,
    },
  };
}

export function getOntology(): OntologyNode[] {
  return PHYSICS_ONTOLOGY;
}

export function getCausalVariables(): CausalVariable[] {
  return CAUSAL_VARIABLES;
}

export function getDiscoveredHypotheses(): CausalMechanismHypothesis[] {
  return [...discoveredHypotheses];
}

export function getCausalRules(): CausalRule[] {
  return [...discoveredRules];
}

export function getLatestGraph(): CausalGraph | null {
  return discoveredGraphs.length > 0 ? discoveredGraphs[discoveredGraphs.length - 1] : null;
}
