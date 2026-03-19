import { computeMultiScaleFeatures, type MultiScaleFeatures, computeCrossScaleCoupling, type CrossScaleCoupling } from "./multi-scale-engine";
import { computePairingProfile, type PairingProfile } from "../physics/pairing-mechanisms";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";
import { parseFormulaCounts } from "../learning/physics-engine";

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

const CAUSAL_VAR_META = new Map<string, CausalVariable>(
  CAUSAL_VARIABLES.map(v => [v.name, v])
);

function describeVariable(varName: string): string {
  const meta = CAUSAL_VAR_META.get(varName);
  if (!meta) return varName;
  return meta.unit !== "dimensionless" && meta.unit !== "enum"
    ? `${meta.description} (${meta.unit})`
    : meta.description;
}

function groupByFamily(dataset: CausalDataRecord[]): Map<string, CausalDataRecord[]> {
  const map = new Map<string, CausalDataRecord[]>();
  for (const r of dataset) {
    const fam = r.material_family as string;
    if (!map.has(fam)) map.set(fam, []);
    map.get(fam)!.push(r);
  }
  return map;
}

let hypCounter = 0;

const CATEGORY_COUNTS: Record<string, number> = {};
const INTERVENTION_COUNT: number = (() => {
  let count = 0;
  for (const v of CAUSAL_VARIABLES) {
    CATEGORY_COUNTS[v.category] = (CATEGORY_COUNTS[v.category] || 0) + 1;
    if (v.isIntervention) count++;
  }
  return count;
})();

function formatMechanism(raw: string): string {
  if (!raw || raw.length === 0) return "Unknown mechanism";
  return raw.charAt(0).toUpperCase() + raw.slice(1);
}

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
  { variable: "pairing_symmetry", category: "pairing_interactions", level: 3, parents: ["spin_fluct_strength", "orbital_fluct", "topology_z2"], children: ["Tc"] },
  { variable: "berry_phase", category: "topological", level: 0, parents: [], children: ["topology_z2"] },
  { variable: "topology_z2", category: "topological", level: 0, parents: ["berry_phase"], children: ["pairing_symmetry", "Tc"] },
  { variable: "coupling_strength", category: "superconducting_properties", level: 4, parents: ["lambda"], children: ["Tc"] },
  { variable: "Tc", category: "superconducting_properties", level: 5, parents: ["lambda", "mu_star", "debye_temp", "spin_fluct_strength", "pairing_symmetry", "coupling_strength", "topology_z2"], children: [] },
];

export interface CausalDataRecord {
  formula: string;
  material_family: string;
  [key: string]: number | string;
}

function encodePairingSymmetry(sym: string | undefined | null): number {
  if (!sym) return 0;
  const s = sym.toLowerCase();
  if (s.includes("triplet") || s.includes("p-wave") || s.includes("odd-parity")) return 2;
  if (s.includes("d-wave") || s.includes("dx2") || s.includes("d_x2")) return 1;
  return 0;
}

export async function buildCausalDataRecord(formula: string): Promise<CausalDataRecord> {
  let multiScale: MultiScaleFeatures | null = null;
  try { multiScale = computeMultiScaleFeatures(formula); } catch {}

  let features: Record<string, number> = {};
  try { features = await extractFeatures(formula); } catch {}

  let gb: { tcPredicted: number } = { tcPredicted: 0 };
  try { gb = await gbPredict(features); } catch {}

  let pairing: PairingProfile | null = null;
  try { pairing = computePairingProfile(formula); } catch {}

  let crossScale: CrossScaleCoupling | null = null;
  if (multiScale) {
    try { crossScale = computeCrossScaleCoupling(multiScale); } catch {}
  }

  const lambda = features.electronPhononLambda ?? 0.5;
  const DOS_EF = multiScale?.electronic?.DOS_EF ?? 1.0;
  const omega_log = features.logPhononFreq ?? 400;
  const bandwidth = multiScale?.electronic?.bandwidth ?? 2.0;
  const nesting = multiScale?.electronic?.nesting_score ?? 0.1;
  const mott = multiScale?.electronic?.mott_proximity ?? 0;

  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, c) => s + c, 0) || 1;
  const hCount = counts["H"] || 0;
  const hAtomicFraction = hCount / totalAtoms;

  const hasH = hCount > 0;
  const hasCu = formula.includes("Cu");
  const hasFe = formula.includes("Fe");
  let family = "other";
  if (hasH) family = "hydride";
  else if (hasCu && formula.includes("O")) family = "cuprate";
  else if (hasFe && (formula.includes("As") || formula.includes("Se"))) family = "iron-based";
  else family = "intermetallic";

  let hydrogenDensity: number;
  if ((features as any).hydrogenRatio != null && (features as any).hydrogenRatio > 0) {
    hydrogenDensity = Math.min(1.0, (features as any).hydrogenRatio / 10.0);
  } else if (hasH) {
    hydrogenDensity = Math.min(1.0, hAtomicFraction * 1.5);
  } else {
    hydrogenDensity = 0;
  }

  const phononSoftening = (features as any).phononSofteningIndex != null
    ? (features as any).phononSofteningIndex
    : crossScale
      ? Math.min(1.0, (1.0 / Math.max(0.1, crossScale.bond_stiffness_vs_phonon)) * 0.3
          + (multiScale?.electronic?.nesting_score ?? 0) * 0.3)
      : 0.15;

  const epMassRatio = crossScale?.electron_phonon_mass_ratio ?? 1.0;
  const anharmonicityVal = (features as any).anharmonicityFlag != null
    ? ((features as any).anharmonicityFlag ? 0.5 : 0.15)
      * (1.0 + epMassRatio * 0.05)
    : crossScale
      ? Math.min(0.8, epMassRatio * 0.08 + (hasH ? hAtomicFraction * 0.4 : 0))
      : 0.1 + (hasH ? hAtomicFraction * 0.3 : 0);

  return {
    formula,
    material_family: family,
    atomic_mass_avg: multiScale?.atomic?.atomic_mass_avg ?? 50,
    electronegativity_spread: multiScale?.atomic?.electronegativity_spread ?? 1.5,
    coordination_number: multiScale?.atomic?.coordination_number ?? 6,
    bond_length_dist: multiScale?.atomic?.bond_length_distribution ?? 0.2,
    charge_transfer: multiScale?.atomic?.charge_transfer ?? 0.3,
    hydrogen_density: hydrogenDensity,
    DOS_EF,
    bandwidth,
    van_hove_distance: multiScale?.electronic?.van_hove_distance ?? 0.5,
    band_flatness: multiScale?.electronic?.band_flatness ?? 0.2,
    nesting_score: nesting,
    mott_proximity: mott,
    phonon_freq: omega_log,
    debye_temp: omega_log * 1.44,
    phonon_softening: phononSoftening,
    anharmonicity: anharmonicityVal,
    lambda,
    mu_star: features.muStarEstimate ?? 0.10,
    spin_fluct_strength: pairing?.spin?.chiQ ?? nesting * 0.5,
    orbital_fluct: pairing?.orbital?.orbitalFluctuation ?? 0.1,
    pairing_symmetry: encodePairingSymmetry(pairing?.pairingSymmetry),
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

function computePearsonCorrelation(x: number[], y: number[]): number {
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
  if (dx < 1e-10 || dy < 1e-10) return 0;
  const denom = Math.sqrt(dx) * Math.sqrt(dy);
  if (denom < 1e-10) return 0;
  const r = num / denom;
  return Math.max(-1, Math.min(1, r));
}

function toRanks(x: number[]): number[] {
  const n = x.length;
  const indexed = x.map((val, i) => ({ val, i }));
  indexed.sort((a, b) => a.val - b.val);
  const ranks = new Array(n);
  let pos = 0;
  while (pos < n) {
    let end = pos + 1;
    while (end < n && indexed[end].val === indexed[pos].val) end++;
    const avgRank = (pos + end - 1) / 2;
    for (let k = pos; k < end; k++) ranks[indexed[k].i] = avgRank;
    pos = end;
  }
  return ranks;
}

function computeSpearmanCorrelation(x: number[], y: number[]): number {
  if (x.length < 3) return 0;
  return computePearsonCorrelation(toRanks(x), toRanks(y));
}

function computeCorrelation(x: number[], y: number[]): number {
  const pearson = computePearsonCorrelation(x, y);
  const spearman = computeSpearmanCorrelation(x, y);
  return Math.abs(spearman) > Math.abs(pearson)
    ? spearman
    : pearson;
}

function residualizeMultivariate(v: number[], conditioningSet: number[][]): number[] {
  const n = v.length;
  const p = conditioningSet.length;
  if (p === 0) {
    const mv = v.reduce((a, b) => a + b, 0) / n;
    return v.map(vi => vi - mv);
  }

  const means = conditioningSet.map(c => c.reduce((a, b) => a + b, 0) / n);
  const mv = v.reduce((a, b) => a + b, 0) / n;

  const centered: number[][] = conditioningSet.map((c, ci) => c.map(val => val - means[ci]));
  const vc = v.map(val => val - mv);

  const eps = 1e-10;

  const gram: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const crossVec: number[] = new Array(p).fill(0);

  for (let a = 0; a < p; a++) {
    for (let b = a; b < p; b++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += centered[a][i] * centered[b][i];
      gram[a][b] = dot;
      gram[b][a] = dot;
    }
    let dotV = 0;
    for (let i = 0; i < n; i++) dotV += centered[a][i] * vc[i];
    crossVec[a] = dotV;
  }

  for (let a = 0; a < p; a++) {
    gram[a][a] += eps;
  }

  const aug: number[][] = gram.map((row, i) => [...row, crossVec[i]]);
  for (let col = 0; col < p; col++) {
    let maxRow = col;
    for (let row = col + 1; row < p; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    if (maxRow !== col) [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < eps) continue;

    for (let row = col + 1; row < p; row++) {
      const factor = aug[row][col] / pivot;
      for (let c = col; c <= p; c++) aug[row][c] -= factor * aug[col][c];
    }
  }

  const coeffs = new Array(p).fill(0);
  for (let row = p - 1; row >= 0; row--) {
    if (Math.abs(aug[row][row]) < eps) continue;
    let sum = aug[row][p];
    for (let col = row + 1; col < p; col++) sum -= aug[row][col] * coeffs[col];
    coeffs[row] = sum / aug[row][row];
  }

  const residual = new Array(n);
  for (let i = 0; i < n; i++) {
    let predicted = 0;
    for (let a = 0; a < p; a++) predicted += coeffs[a] * centered[a][i];
    residual[i] = vc[i] - predicted;
  }
  return residual;
}

function partialCorrelation(x: number[], y: number[], conditioningSet: number[][]): number | null {
  const n = x.length;
  const minSamples = conditioningSet.length + 4;
  if (n < minSamples) return null;

  const rxPearson = residualizeMultivariate(x, conditioningSet);
  const ryPearson = residualizeMultivariate(y, conditioningSet);
  const pearson = computePearsonCorrelation(rxPearson, ryPearson);

  const rankedCond = conditioningSet.map(c => toRanks(c));
  const rxSpearman = residualizeMultivariate(toRanks(x), rankedCond);
  const rySpearman = residualizeMultivariate(toRanks(y), rankedCond);
  const spearman = computePearsonCorrelation(rxSpearman, rySpearman);

  return Math.abs(spearman) > Math.abs(pearson) ? spearman : pearson;
}

function getColumn(dataset: CausalDataRecord[], varName: string): number[] {
  return dataset.map(r => {
    const v = r[varName];
    return typeof v === "number" ? v : 0;
  });
}

export type CausalDiscoveryMode = "physics" | "discovery_process";

const PHYSICS_FORBIDDEN_EDGES = new Set([
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

const DISCOVERY_PROCESS_FORBIDDEN_EDGES = new Set([
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
  "Tc->atomic_mass_avg",
]);

const REQUIRED_EDGES: { source: string; target: string; mechanism: string; minStrength: number }[] = [
  { source: "phonon_freq", target: "debye_temp", mechanism: "Debye temperature is proportional to characteristic phonon frequency (T_D ≈ ℏω/k_B)", minStrength: 0.8 },
  { source: "lambda", target: "Tc", mechanism: "Electron-phonon coupling directly determines Tc via McMillan/Allen-Dynes equation", minStrength: 0.6 },
  { source: "lambda", target: "coupling_strength", mechanism: "Coupling strength is a direct function of lambda and DOS(E_F)", minStrength: 0.7 },
  { source: "DOS_EF", target: "lambda", mechanism: "DOS at Fermi level enters the electron-phonon coupling integral", minStrength: 0.5 },
  { source: "atomic_mass_avg", target: "phonon_freq", mechanism: "Phonon frequency scales as 1/√M (harmonic oscillator)", minStrength: 0.5 },
  { source: "mu_star", target: "Tc", mechanism: "Coulomb pseudopotential suppresses pairing, reducing Tc", minStrength: 0.4 },
  { source: "debye_temp", target: "Tc", mechanism: "Debye temperature sets the energy scale for phonon-mediated pairing", minStrength: 0.5 },
  { source: "berry_phase", target: "topology_z2", mechanism: "Z2 invariant is determined by Berry phase winding", minStrength: 0.6 },
  { source: "topology_z2", target: "pairing_symmetry", mechanism: "Nontrivial topology constrains allowed pairing channels", minStrength: 0.4 },
  { source: "nesting_score", target: "spin_fluct_strength", mechanism: "Fermi surface nesting enhances spin susceptibility at nesting vector", minStrength: 0.5 },
  { source: "hydrogen_density", target: "phonon_freq", mechanism: "Light hydrogen atoms create high-frequency phonon modes", minStrength: 0.5 },
  { source: "defect_density", target: "phonon_softening", mechanism: "Defects break translational symmetry, softening phonon modes", minStrength: 0.3 },
  { source: "phonon_softening", target: "anharmonicity", mechanism: "Soft phonon modes are inherently anharmonic", minStrength: 0.4 },
];

const CATEGORY_PRECEDENCE: Record<CausalVariableCategory, number> = {
  "atomic_structure": 0,
  "topological": 1,
  "thermodynamic_conditions": 2,
  "phonon_properties": 3,
  "electronic_structure": 4,
  "pairing_interactions": 5,
  "superconducting_properties": 6,
  "synthesis": 7,
  "intervention": 8,
};

function isEdgeForbidden(source: string, target: string, mode: CausalDiscoveryMode = "physics"): boolean {
  const edges = mode === "discovery_process" ? DISCOVERY_PROCESS_FORBIDDEN_EDGES : PHYSICS_FORBIDDEN_EDGES;
  return edges.has(`${source}->${target}`);
}

function isEdgeRequired(source: string, target: string): { source: string; target: string; mechanism: string; minStrength: number } | null {
  return REQUIRED_EDGES.find(e => e.source === source && e.target === target) ?? null;
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

function resolveDirection(
  vi: string, vj: string,
  ontologyI: OntologyNode | undefined,
  ontologyJ: OntologyNode | undefined
): { source: string; target: string; direction: "forward" | "bidirectional" } {
  if (ontologyI && ontologyI.children.includes(vj)) {
    return { source: vi, target: vj, direction: "forward" };
  }
  if (ontologyJ && ontologyJ.children.includes(vi)) {
    return { source: vj, target: vi, direction: "forward" };
  }

  if (ontologyI && ontologyJ) {
    if (ontologyI.level < ontologyJ.level) {
      return { source: vi, target: vj, direction: "forward" };
    }
    if (ontologyJ.level < ontologyI.level) {
      return { source: vj, target: vi, direction: "forward" };
    }

    const catI = CATEGORY_PRECEDENCE[ontologyI.category] ?? 99;
    const catJ = CATEGORY_PRECEDENCE[ontologyJ.category] ?? 99;
    if (catI < catJ) {
      return { source: vi, target: vj, direction: "forward" };
    }
    if (catJ < catI) {
      return { source: vj, target: vi, direction: "forward" };
    }

    if (ontologyI.parents.length < ontologyJ.parents.length) {
      return { source: vi, target: vj, direction: "forward" };
    }
    if (ontologyJ.parents.length < ontologyI.parents.length) {
      return { source: vj, target: vi, direction: "forward" };
    }
  }

  return { source: vi, target: vj, direction: "bidirectional" };
}

function getConditioningSubsets(indices: number[], size: number, cap: number = 200): number[][] {
  const results: number[][] = [];
  const n = indices.length;
  if (size === 0 || n === 0) return [];

  const generate = (start: number, current: number[]): boolean => {
    if (current.length === size) {
      results.push(current.slice());
      return results.length < cap;
    }
    const remaining = size - current.length;
    for (let i = start; i <= n - remaining; i++) {
      current.push(indices[i]);
      if (!generate(i + 1, current)) return false;
      current.pop();
    }
    return true;
  };
  generate(0, []);
  return results;
}

export function discoverCausalGraph(
  dataset: CausalDataRecord[],
  significanceThreshold: number = 0.15,
  mode: CausalDiscoveryMode = "physics"
): CausalGraph {
  const variables = CAUSAL_VARIABLES.map(v => v.name);
  const edges: CausalEdge[] = [];
  const columns: Map<string, number[]> = new Map();
  const addedEdgeKeys = new Set<string>();

  for (const v of variables) {
    columns.set(v, getColumn(dataset, v));
  }

  const maxConditioningDepth = dataset.length < 8 ? 0 : Math.min(3, Math.floor(Math.log2(dataset.length)) - 2);

  for (let i = 0; i < variables.length; i++) {
    for (let j = i + 1; j < variables.length; j++) {
      const vi = variables[i];
      const vj = variables[j];
      const xi = columns.get(vi)!;
      const xj = columns.get(vj)!;
      const corr = computeCorrelation(xi, xj);

      if (Math.abs(corr) < significanceThreshold) continue;

      const candidateConditioners: number[] = [];
      for (let k = 0; k < variables.length; k++) {
        if (k === i || k === j) continue;
        candidateConditioners.push(k);
      }

      let isConditionallyIndependent = false;
      let ciTestsSurvived = 0;
      let ciTestsTotal = 0;

      for (let depth = 1; depth <= maxConditioningDepth && !isConditionallyIndependent; depth++) {
        const subsets = getConditioningSubsets(candidateConditioners, depth);
        for (const subset of subsets) {
          const condCols = subset.map(k => columns.get(variables[k])!);
          const partialCorr = partialCorrelation(xi, xj, condCols);
          if (partialCorr === null) continue;
          ciTestsTotal++;
          if (Math.abs(partialCorr) < significanceThreshold * 0.5) {
            isConditionallyIndependent = true;
            break;
          }
          ciTestsSurvived++;
        }
      }

      if (isConditionallyIndependent) continue;

      const ontologyI = PHYSICS_ONTOLOGY.find(o => o.variable === vi);
      const ontologyJ = PHYSICS_ONTOLOGY.find(o => o.variable === vj);
      const resolved = resolveDirection(vi, vj, ontologyI, ontologyJ);

      if (isEdgeForbidden(resolved.source, resolved.target, mode)) continue;
      if (!isOntologyAllowed(resolved.source, resolved.target)) continue;

      const mechanism = inferMechanism(resolved.source, resolved.target);
      const survivalRatio = ciTestsTotal > 0 ? ciTestsSurvived / ciTestsTotal : 1;

      edges.push({
        source: resolved.source,
        target: resolved.target,
        strength: Math.abs(corr),
        direction: resolved.direction,
        mechanism,
        confidence: Math.min(1, Math.abs(corr) * 1.5 * (0.5 + 0.5 * survivalRatio)),
        evidenceCount: ciTestsSurvived + 1,
        validated: false,
      });
      addedEdgeKeys.add(`${resolved.source}->${resolved.target}`);
    }
  }

  for (const req of REQUIRED_EDGES) {
    const key = `${req.source}->${req.target}`;
    if (addedEdgeKeys.has(key)) continue;
    if (isEdgeForbidden(req.source, req.target, mode)) continue;

    const xs = columns.get(req.source);
    const xt = columns.get(req.target);
    const empiricalStrength = xs && xt ? Math.abs(computeCorrelation(xs, xt)) : 0;
    const strength = Math.max(req.minStrength, empiricalStrength);

    edges.push({
      source: req.source,
      target: req.target,
      strength,
      direction: "forward",
      mechanism: req.mechanism,
      confidence: Math.min(1, strength * 1.2),
      evidenceCount: dataset.length,
      validated: true,
    });
    addedEdgeKeys.add(key);
  }

  if (mode === "discovery_process") {
    const interventionFeedbackEdges: { source: string; target: string; mechanism: string }[] = [
      { source: "Tc", target: "pressure", mechanism: "Observed Tc guides next-iteration pressure selection in experimental feedback loop" },
      { source: "Tc", target: "doping", mechanism: "Observed Tc guides next-iteration doping level in experimental feedback loop" },
    ];
    for (const fb of interventionFeedbackEdges) {
      const key = `${fb.source}->${fb.target}`;
      if (addedEdgeKeys.has(key)) continue;
      const xs = columns.get(fb.source);
      const xt = columns.get(fb.target);
      const empiricalStrength = xs && xt ? Math.abs(computeCorrelation(xs, xt)) : 0;
      if (empiricalStrength > significanceThreshold) {
        edges.push({
          source: fb.source,
          target: fb.target,
          strength: empiricalStrength,
          direction: "forward",
          mechanism: fb.mechanism,
          confidence: Math.min(1, empiricalStrength * 1.2),
          evidenceCount: dataset.length,
          validated: false,
        });
        addedEdgeKeys.add(key);
      }
    }
  }

  edges.sort((a, b) => b.strength - a.strength);

  return {
    nodes: variables.filter(v => edges.some(e => e.source === v || e.target === v)),
    edges: edges.slice(0, 60),
    discoveredAt: Date.now(),
    method: mode === "discovery_process"
      ? "PC-algorithm with intervention feedback loops, multi-variable conditioning, physics ontology"
      : "PC-algorithm with multi-variable conditioning, physics ontology, required/forbidden-edge priors",
    datasetSize: dataset.length,
  };
}

const KNOWN_MECHANISMS: Record<string, string> = {
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
  "phonon_freq_debye_temp": "Debye temperature is proportional to characteristic phonon frequency (T_D ≈ ℏω/k_B)",
  "atomic_mass_avg_phonon_freq": "Phonon frequency scales inversely with square root of atomic mass",
  "phonon_softening_anharmonicity": "Soft phonon modes are inherently anharmonic near structural instabilities",
  "anharmonicity_lambda": "Anharmonic phonon corrections modify the effective electron-phonon coupling",
  "berry_phase_topology_z2": "Z2 topological invariant is determined by Berry phase winding across the BZ",
  "coordination_number_bandwidth": "Higher coordination increases orbital overlap, widening the bandwidth",
  "coordination_number_DOS_EF": "Coordination geometry shapes the crystal field splitting and DOS at E_F",
  "electronegativity_spread_charge_transfer": "Electronegativity differences drive interatomic charge redistribution",
  "electronegativity_spread_band_flatness": "Large electronegativity contrasts localize bands, increasing flatness",
  "bandwidth_mott_proximity": "Narrow bandwidth increases U/W ratio, pushing toward Mott localization",
  "topology_z2_Tc": "Nontrivial topology can protect or enhance superconducting pairing",
  "orbital_fluct_pairing_symmetry": "Orbital fluctuations favor specific pairing symmetry channels",
  "spin_fluct_strength_pairing_symmetry": "Strong spin fluctuations favor d-wave over s-wave pairing",
  "layeredness_dimensionality": "Layered crystal structure reduces effective dimensionality",
  "layeredness_bandwidth": "Interlayer coupling modulates the electronic bandwidth",
  "charge_transfer_nesting_score": "Charge transfer modifies Fermi surface topology, affecting nesting",
  "doping_nesting_score": "Doping shifts the Fermi level, tuning nesting conditions",
  "doping_mu_star": "Doping modifies screening, changing the Coulomb pseudopotential",
  "defect_density_mu_star": "Defects enhance pair-breaking scattering, effectively increasing mu*",
  "strain_bandwidth": "Strain modifies orbital overlaps and bond angles, changing bandwidth",
  "strain_DOS_EF": "Strain shifts band positions relative to Fermi level",
  "pressure_lambda": "Pressure modifies lattice dynamics and electron-phonon matrix elements",
  "lambda_coupling_strength": "Coupling strength is directly proportional to lambda × DOS(E_F)",
};

function inferMechanism(source: string, target: string): string {
  const key = `${source}_${target}`;
  const known = KNOWN_MECHANISMS[key];
  if (known) return known;

  const srcVar = CAUSAL_VARIABLES.find(v => v.name === source);
  const tgtVar = CAUSAL_VARIABLES.find(v => v.name === target);
  if (srcVar && tgtVar) {
    const srcCat = srcVar.category.replace(/_/g, " ");
    const tgtCat = tgtVar.category.replace(/_/g, " ");
    const coupling = srcCat === tgtCat ? "intra-scale" : `${srcCat}-to-${tgtCat}`;
    return `${srcVar.description} (${srcVar.unit}) modulates ${tgtVar.description} via ${coupling} coupling`;
  }
  return `${source} influences ${target} through physics coupling`;
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
  hypothesisType: "design" | "observation";
  materialFamilies: string[];
  discoveredAt: number;
}

function generateMechanismHypotheses(graph: CausalGraph, dataset: CausalDataRecord[]): CausalMechanismHypothesis[] {
  const hypotheses: CausalMechanismHypothesis[] = [];
  const minFamilySamples = Math.max(5, Math.ceil(dataset.length * 0.05));

  const pathsToTc: string[][] = [];
  function findPaths(current: string, path: string[], visited: Set<string>) {
    if (current === "Tc") {
      pathsToTc.push([...path, "Tc"]);
      return;
    }
    if (path.length > 6 || visited.has(current)) return;
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

  const familyMap = groupByFamily(dataset);
  const totalTests = pathsToTc.length * familyMap.size;
  const bonferroniAlpha = totalTests > 0 ? 0.05 / totalTests : 0.05;
  const correctedMinVariance = 0.01 + bonferroniAlpha * 10;

  pathsToTc.sort((a, b) => b.length - a.length);
  for (const path of pathsToTc.slice(0, 25)) {
    if (path.length < 2) continue;

    const edges = [];
    let minStrength = Infinity;
    for (let i = 0; i < path.length - 1; i++) {
      const edge = graph.edges.find(e => e.source === path[i] && e.target === path[i + 1]);
      if (edge) {
        edges.push({ source: edge.source, target: edge.target, strength: edge.strength });
        if (edge.strength < minStrength) minStrength = edge.strength;
      }
    }
    if (edges.length === 0) continue;
    const bottleneckStrength = isFinite(minStrength) ? minStrength : 0;

    if (bottleneckStrength < 0.15) continue;

    const chainDesc = path.join(" -> ");
    const mechanism = edges.map(e => inferMechanism(e.source, e.target)).join("; ");
    const relevantFamilies: string[] = [];
    for (const [fam, famData] of familyMap) {
      if (famData.length < minFamilySamples) continue;
      const startVar = path[0];
      const vals = famData.map(r => typeof r[startVar] === "number" ? r[startVar] as number : 0);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
      if (variance > correctedMinVariance) relevantFamilies.push(fam);
    }

    const rootVar = CAUSAL_VARIABLES.find(v => v.name === path[0]);
    const isDesign = rootVar?.isIntervention === true;
    const hypothesisType: "design" | "observation" = isDesign ? "design" : "observation";

    const rootLabel = describeVariable(path[0]);
    const testable = isDesign
      ? `Modify ${rootLabel} and observe cascading effect through ${path.slice(1, -1).map(describeVariable).join(", ")} to Tc`
      : `Compare materials with varying ${rootLabel} levels to validate causal chain`;

    const statement = isDesign
      ? `Design: Tc can be tuned by adjusting ${rootLabel} through the chain: ${chainDesc}`
      : `Observation: Tc correlates with ${rootLabel} through the chain: ${chainDesc}`;

    hypCounter++;
    hypotheses.push({
      id: `hyp-${Date.now()}-${hypCounter}`,
      statement,
      causalChain: path,
      mechanism,
      supportingEdges: edges,
      confidence: Math.min(1, bottleneckStrength),
      testableIntervention: testable,
      hypothesisType,
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
  const familyMap = groupByFamily(dataset);
  const minFamilySamples = Math.max(5, Math.ceil(dataset.length * 0.05));

  const strengths = graph.edges.map(e => e.strength);
  const adaptiveThreshold = strengths.length > 2
    ? Math.max(0.1, strengths.sort((a, b) => b - a)[Math.min(Math.floor(strengths.length * 0.3), strengths.length - 1)])
    : 0.2;

  const familyColumns = new Map<string, Map<string, number[]>>();
  const allVars = new Set<string>();
  for (const e of graph.edges) { allVars.add(e.source); allVars.add(e.target); }
  for (const [fam, famData] of familyMap) {
    if (famData.length < minFamilySamples) continue;
    const cols = new Map<string, number[]>();
    for (const v of allVars) {
      cols.set(v, famData.map(r => typeof r[v] === "number" ? r[v] as number : 0));
    }
    familyColumns.set(fam, cols);
  }

  for (const edge of graph.edges.filter(e => e.strength > adaptiveThreshold)) {
    const validFamilies: string[] = [];
    for (const [fam, cols] of familyColumns) {
      const x = cols.get(edge.source);
      const y = cols.get(edge.target);
      if (!x || !y) continue;
      const corr = computeCorrelation(x, y);
      const corrThreshold = Math.max(0.05, adaptiveThreshold * 0.5);
      if (Math.abs(corr) > corrThreshold && Math.sign(corr) === Math.sign(edge.strength)) validFamilies.push(fam);
    }

    const srcDesc = describeVariable(edge.source);
    const tgtDesc = describeVariable(edge.target);
    rules.push({
      antecedent: `Increased ${srcDesc}`,
      consequent: `${edge.strength > 0 ? "higher" : "lower"} ${tgtDesc}`,
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
  sampleSize: number;
  edgeCount: number;
  avgEdgeStrength: number;
  consistentEdges: number;
  consistencyScore: number;
  statisticalConfidence: number;
  dominantPathway: string;
}

function sampleSizeConfidence(n: number): number {
  if (n < 10) return 0;
  if (n >= 100) return 1;
  return Math.min(1, (n - 10) / 90 * 0.7 + 0.3);
}

function validateAcrossFamilies(graph: CausalGraph, dataset: CausalDataRecord[]): CrossFamilyValidation[] {
  const familyMap = groupByFamily(dataset);
  const results: CrossFamilyValidation[] = [];

  const allVars = new Set<string>();
  for (const e of graph.edges) { allVars.add(e.source); allVars.add(e.target); }

  const tcEdges = graph.edges.filter(e => e.target === "Tc").sort((a, b) => b.strength - a.strength);
  const dominantPath = tcEdges.length > 0 ? `${tcEdges[0].source} -> Tc` : "unknown";

  for (const [fam, famData] of familyMap) {
    if (famData.length < 10) continue;

    const cols = new Map<string, number[]>();
    for (const v of allVars) {
      cols.set(v, famData.map(r => typeof r[v] === "number" ? r[v] as number : 0));
    }

    let consistentEdges = 0;
    let totalStrength = 0;

    for (const edge of graph.edges) {
      const x = cols.get(edge.source);
      const y = cols.get(edge.target);
      if (!x || !y) continue;
      const localCorr = computeCorrelation(x, y);
      if (Math.sign(localCorr) === Math.sign(edge.strength) && Math.abs(localCorr) > 0.1) {
        consistentEdges++;
      }
      totalStrength += Math.abs(localCorr);
    }

    const consistencyScore = graph.edges.length > 0 ? consistentEdges / graph.edges.length : 0;
    const sizeConf = sampleSizeConfidence(famData.length);
    const statisticalConfidence = Math.round(consistencyScore * sizeConf * 1000) / 1000;

    results.push({
      family: fam,
      sampleSize: famData.length,
      edgeCount: graph.edges.length,
      avgEdgeStrength: graph.edges.length > 0 ? totalStrength / graph.edges.length : 0,
      consistentEdges,
      consistencyScore,
      statisticalConfidence,
      dominantPathway: dominantPath,
    });
  }

  return results.sort((a, b) => b.statisticalConfidence - a.statisticalConfidence);
}

export interface DesignGuidance {
  variable: string;
  variableLabel: string;
  direction: "maximize" | "minimize";
  recommendation: string;
  causalImpactOnTc: number;
  mechanism: string;
  rank: number;
}

function generateDesignGuidance(graph: CausalGraph, dataset: CausalDataRecord[]): DesignGuidance[] {
  const guidance: DesignGuidance[] = [];
  const interventionVars = CAUSAL_VARIABLES.filter(v => v.isIntervention);

  const interventionColumns = new Map<string, number[]>();
  for (const iv of interventionVars) {
    interventionColumns.set(iv.name, getColumn(dataset, iv.name));
  }
  const tc = getColumn(dataset, "Tc");

  for (const iv of interventionVars) {
    const tcEdges = findCausalPathStrength(graph, iv.name, "Tc");
    if (tcEdges.totalStrength < 0.05) continue;

    const x = interventionColumns.get(iv.name)!;

    const conditioningSet: number[][] = [];
    for (const other of interventionVars) {
      if (other.name !== iv.name) {
        conditioningSet.push(interventionColumns.get(other.name)!);
      }
    }

    let corr: number;
    const partial = partialCorrelation(x, tc, conditioningSet);
    if (partial !== null) {
      corr = partial;
    } else {
      corr = computeCorrelation(x, tc);
    }

    const dir = corr > 0 ? "maximize" : "minimize";
    const label = describeVariable(iv.name);
    guidance.push({
      variable: iv.name,
      variableLabel: label,
      direction: dir,
      recommendation: `${dir === "maximize" ? "Maximize" : "Minimize"} ${label}`,
      causalImpactOnTc: Math.round(tcEdges.totalStrength * 1000) / 1000,
      mechanism: tcEdges.pathway.length > 0 ? tcEdges.pathway.join(" -> ") : "No direct causal pathway found",
      rank: 0,
    });
  }

  guidance.sort((a, b) => b.causalImpactOnTc - a.causalImpactOnTc);
  guidance.forEach((g, i) => g.rank = i + 1);

  return guidance;
}

function findCausalPathStrength(graph: CausalGraph, from: string, to: string): { totalStrength: number; pathway: string[] } {
  const adjacency = new Map<string, { target: string; strength: number }[]>();
  for (const edge of graph.edges) {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
    adjacency.get(edge.source)!.push({ target: edge.target, strength: edge.strength });
  }

  let bestStrength = 0;
  let bestPath: string[] = [];
  const visited = new Set<string>();
  const path: string[] = [from];

  function dfs(current: string, minStrength: number) {
    if (current === to) {
      if (minStrength > bestStrength) {
        bestStrength = minStrength;
        bestPath = [...path];
      }
      return;
    }
    if (path.length >= 6 || minStrength <= bestStrength) return;

    const neighbors = adjacency.get(current);
    if (!neighbors) return;

    for (const { target, strength } of neighbors) {
      if (visited.has(target)) continue;
      const edgeMin = Math.min(minStrength, strength);
      if (edgeMin <= bestStrength) continue;

      visited.add(target);
      path.push(target);
      dfs(target, edgeMin);
      path.pop();
      visited.delete(target);
    }
  }

  visited.add(from);
  dfs(from, 1.0);
  return { totalStrength: bestStrength, pathway: bestPath };
}

export interface PressureRegimeComparison {
  ambientEdges: CausalEdge[];
  highPressureEdges: CausalEdge[];
  survivingMechanisms: { source: string; target: string; ambientStrength: number; hpStrength: number }[];
  newMechanisms: { source: string; target: string; strength: number; regime: string }[];
  decompressionInsight: string;
}

function computePressureThreshold(dataset: CausalDataRecord[]): number {
  const pressures = dataset
    .map(r => r.pressure as number)
    .filter(p => typeof p === "number" && isFinite(p))
    .sort((a, b) => a - b);

  if (pressures.length < 4) return 20;

  const median = pressures.length % 2 === 0
    ? (pressures[pressures.length / 2 - 1] + pressures[pressures.length / 2]) / 2
    : pressures[Math.floor(pressures.length / 2)];

  if (median < 1) {
    const p75 = pressures[Math.floor(pressures.length * 0.75)];
    return Math.max(5, p75);
  }

  return Math.max(5, median);
}

function comparePressureRegimes(dataset: CausalDataRecord[], graph: CausalGraph): PressureRegimeComparison {
  const threshold = computePressureThreshold(dataset);
  const ambient = dataset.filter(r => (r.pressure as number) < threshold);
  const highP = dataset.filter(r => (r.pressure as number) >= threshold);

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

  let insight = "";
  const ambientInsufficient = ambient.length < 10;
  const hpInsufficient = highP.length < 10;

  if (ambientInsufficient && hpInsufficient) {
    insight = `Insufficient data in both regimes (ambient: ${ambient.length} records, high-pressure: ${highP.length} records; minimum 10 required each). No causal graph could be constructed for either regime.`;
  } else if (ambientInsufficient) {
    insight = `Insufficient ambient-pressure data (${ambient.length} records; minimum 10 required). High-pressure graph has ${hpGraph.edges.length} edges but comparison is not possible without an ambient baseline.`;
  } else if (hpInsufficient) {
    insight = `Insufficient high-pressure data (${highP.length} records; minimum 10 required). Ambient graph has ${ambientGraph.edges.length} edges but comparison is not possible without a high-pressure counterpart.`;
  } else if (surviving.length > 0) {
    const strongest = surviving.sort((a, b) => b.ambientStrength - a.ambientStrength)[0];
    insight = `Mechanism ${strongest.source} -> ${strongest.target} survives across pressure regimes (ambient: ${strongest.ambientStrength.toFixed(2)}, high-P: ${strongest.hpStrength.toFixed(2)}). Threshold: ${threshold.toFixed(1)} GPa. This pathway may be critical for ambient-pressure superconductivity.`;
  } else {
    insight = `No shared mechanisms found between ambient (${ambientGraph.edges.length} edges) and high-pressure (${hpGraph.edges.length} edges) regimes at threshold ${threshold.toFixed(1)} GPa. The superconducting mechanisms may be fundamentally different across pressure regimes.`;
  }

  return {
    ambientEdges: ambientGraph.edges.slice(0, 10),
    highPressureEdges: hpGraph.edges.slice(0, 10),
    survivingMechanisms: surviving,
    newMechanisms: newMechs.slice(0, 10),
    decompressionInsight: insight,
  };
}

class CausalDiscoveryStore {
  private graphs: CausalGraph[] = [];
  private hypotheses: Map<string, CausalMechanismHypothesis> = new Map();
  private rules: Map<string, CausalRule> = new Map();
  private latestGuidance: DesignGuidance[] = [];
  private _totalRuns = 0;

  private static readonly MAX_GRAPHS = 20;
  private static readonly MAX_HYPOTHESES = 50;
  private static readonly MAX_RULES = 100;

  get totalRuns() { return this._totalRuns; }

  addGraph(graph: CausalGraph) {
    this.graphs.push(graph);
    if (this.graphs.length > CausalDiscoveryStore.MAX_GRAPHS) {
      this.graphs.splice(0, this.graphs.length - CausalDiscoveryStore.MAX_GRAPHS);
    }
  }

  addHypotheses(hypotheses: CausalMechanismHypothesis[]) {
    for (const h of hypotheses) {
      const existing = this.hypotheses.get(h.statement);
      if (!existing || h.confidence > existing.confidence) {
        this.hypotheses.set(h.statement, h);
      }
    }
    this.evictLowestHypotheses();
  }

  addRules(rules: CausalRule[]) {
    for (const r of rules) {
      const key = `${r.antecedent}|${r.consequent}`;
      const existing = this.rules.get(key);
      if (!existing || r.strength > existing.strength) {
        this.rules.set(key, r);
      }
    }
    this.evictLowestRules();
  }

  private evictLowestHypotheses() {
    while (this.hypotheses.size > CausalDiscoveryStore.MAX_HYPOTHESES) {
      let worstKey = "";
      let worstConf = Infinity;
      for (const [key, h] of this.hypotheses) {
        if (h.confidence < worstConf) {
          worstConf = h.confidence;
          worstKey = key;
        }
      }
      this.hypotheses.delete(worstKey);
    }
  }

  private evictLowestRules() {
    while (this.rules.size > CausalDiscoveryStore.MAX_RULES) {
      let worstKey = "";
      let worstStr = Infinity;
      for (const [key, r] of this.rules) {
        if (r.strength < worstStr) {
          worstStr = r.strength;
          worstKey = key;
        }
      }
      this.rules.delete(worstKey);
    }
  }

  setGuidance(guidance: DesignGuidance[]) { this.latestGuidance = guidance; }

  incrementRuns() { this._totalRuns++; }

  getGraphs(): CausalGraph[] { return this.graphs; }
  getLatestGraph(): CausalGraph | null { return this.graphs.length > 0 ? this.graphs[this.graphs.length - 1] : null; }
  getHypotheses(): CausalMechanismHypothesis[] {
    return [...this.hypotheses.values()].sort((a, b) => b.confidence - a.confidence);
  }
  getRules(): CausalRule[] {
    return [...this.rules.values()].sort((a, b) => b.strength - a.strength);
  }
  getGuidance(): DesignGuidance[] { return this.latestGuidance; }

  reset() {
    this.graphs = [];
    this.hypotheses.clear();
    this.rules.clear();
    this.latestGuidance = [];
    this._totalRuns = 0;
  }
}

const discoveryStore = new CausalDiscoveryStore();

export async function runCausalDiscovery(dataset: CausalDataRecord[]): Promise<{
  graph: CausalGraph;
  hypotheses: CausalMechanismHypothesis[];
  rules: CausalRule[];
  crossFamilyValidation: CrossFamilyValidation[];
  designGuidance: DesignGuidance[];
  pressureComparison: PressureRegimeComparison;
}> {
  discoveryStore.incrementRuns();

  // Yield between each major phase so HTTP requests aren't blocked during
  // the 3-8s causal graph discovery run.
  const yieldNow = () => new Promise<void>(r => setTimeout(r, 0));

  const graph = discoverCausalGraph(dataset);
  await yieldNow();
  const hypotheses = generateMechanismHypotheses(graph, dataset);
  await yieldNow();
  const rules = extractCausalRules(graph, dataset);
  await yieldNow();
  const crossFamilyValidation = validateAcrossFamilies(graph, dataset);
  await yieldNow();
  const designGuidance = generateDesignGuidance(graph, dataset);
  await yieldNow();
  const pressureComparison = comparePressureRegimes(dataset, graph);

  discoveryStore.addGraph(graph);
  discoveryStore.addHypotheses(hypotheses);
  discoveryStore.addRules(rules);
  discoveryStore.setGuidance(designGuidance);

  return { graph, hypotheses, rules, crossFamilyValidation, designGuidance, pressureComparison };
}

export async function generateCausalDataset(count: number = 60): Promise<CausalDataRecord[]> {
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
      const rec = await buildCausalDataRecord(formulas[i]);
      if ((rec.Tc as number) <= 0 && (rec.lambda as number) > 0.1) {
        const lam = rec.lambda as number;
        const omega = rec.phonon_freq as number;
        const muStar = rec.mu_star as number;
        const muEff = muStar * (1 + 0.62 * lam);
        const denom = lam - muEff;
        const denomMin = 0.05 + 0.1 * lam;
        if (denom > denomMin) {
          const exponent = -1.04 * (1 + lam) / denom;
          if (exponent <= -0.1) {
            const lambdaBar2 = 2.46 * (1 + 3.8 * (muStar as number));
            const f1r = Math.pow(1 + Math.pow(lam / lambdaBar2, 3 / 2), 1 / 3);
            rec.Tc = (omega * 1.4388 / 1.2) * f1r * Math.exp(exponent);
            rec.Tc = Math.max(0, Math.min(300, rec.Tc as number));
            rec.Tc += (Math.random() - 0.5) * 2;
            rec.Tc = Math.max(0, rec.Tc as number);
          }
        }
      }
      records.push(rec);
    } catch {}
  }

  while (records.length < count) {
    const atomicMass = 20 + Math.random() * 200;
    const omega = Math.max(50, 1600 - atomicMass * 4 + (Math.random() - 0.5) * 300);
    const press = Math.random() * 200;

    const hydrogenDensity = Math.max(0, Math.min(1,
      (press > 100 ? 0.3 : 0.05) + Math.random() * 0.5 + (atomicMass < 50 ? 0.2 : 0)
    ));

    const DOS = Math.max(0.5, 0.5 + Math.random() * 3.5 + hydrogenDensity * 1.5 + (Math.random() - 0.5) * 0.5);
    const bw = Math.max(1, 9 - DOS * 0.8 + (Math.random() - 0.5) * 2);
    const nest = Math.max(0, Math.min(0.8,
      (bw < 4 ? 0.4 : 0.1) + (Math.random() - 0.5) * 0.3
    ));

    const lambda = Math.max(0.3, DOS * 0.4 + hydrogenDensity * 0.8 + (Math.random() - 0.5) * 0.4);
    const muStar = 0.08 + Math.random() * 0.07;

    const enSpread = Math.max(0.5, Math.min(2.5,
      1.0 + (atomicMass > 100 ? 0.5 : -0.2) + (Math.random() - 0.5) * 0.6
    ));
    const coordNum = Math.max(4, Math.min(12,
      6 + (press > 50 ? 2 : 0) + (Math.random() - 0.5) * 3
    ));
    const bondLengthDist = Math.max(0.1, Math.min(0.5,
      0.3 - press * 0.0005 + (Math.random() - 0.5) * 0.1
    ));

    const phonSoft = Math.max(0, Math.min(0.4,
      lambda * 0.1 + (Math.random() - 0.5) * 0.1
    ));
    const anharm = Math.max(0, Math.min(0.3,
      phonSoft * 0.5 + Math.random() * 0.1
    ));

    const muEff = muStar * (1 + 0.62 * lambda);
    const denom = lambda - muEff;
    const denomMin = 0.05 + 0.1 * lambda;
    let Tc = 0;
    if (denom > denomMin) {
      const exponent = -1.04 * (1 + lambda) / denom;
      if (exponent <= -0.1) {
        const lambdaBarS = 2.46 * (1 + 3.8 * muStar);
        const f1s = Math.pow(1 + Math.pow(lambda / lambdaBarS, 3 / 2), 1 / 3);
        Tc = (omega * 1.4388 / 1.2) * f1s * Math.exp(exponent);
        Tc = Math.max(0, Math.min(300, Tc));
      }
    }
    Tc += (Math.random() - 0.5) * 5;
    Tc = Math.max(0, Tc);

    const doping = Math.max(0, Math.min(0.3, 0.1 + (Math.random() - 0.5) * 0.15));

    records.push({
      formula: `CausalSynth${records.length}`,
      material_family: ["hydride", "cuprate", "iron-based", "intermetallic", "other"][Math.floor(Math.random() * 5)],
      atomic_mass_avg: atomicMass,
      electronegativity_spread: enSpread,
      coordination_number: coordNum,
      bond_length_dist: bondLengthDist,
      charge_transfer: Math.max(0, enSpread * 0.4 + (Math.random() - 0.5) * 0.3),
      hydrogen_density: hydrogenDensity,
      DOS_EF: DOS,
      bandwidth: bw,
      van_hove_distance: Math.max(0, 1.5 - DOS * 0.2 + (Math.random() - 0.5) * 0.4),
      band_flatness: Math.max(0, Math.min(0.5, 0.5 - bw * 0.04 + (Math.random() - 0.5) * 0.1)),
      nesting_score: nest,
      mott_proximity: Math.max(0, Math.min(0.5, nest * 0.3 + (Math.random() - 0.5) * 0.15)),
      phonon_freq: omega,
      debye_temp: omega * 1.44,
      phonon_softening: phonSoft,
      anharmonicity: anharm,
      lambda,
      mu_star: muStar,
      spin_fluct_strength: nest * 0.5 + Math.random() * 0.3,
      orbital_fluct: Math.max(0, nest * 0.2 + (Math.random() - 0.5) * 0.1),
      pairing_symmetry: nest > 0.5 ? 1 : (Math.random() > 0.8 ? 2 : 0),
      pressure: press,
      temperature: 300,
      strain: (Math.random() - 0.5) * 0.05,
      doping,
      defect_density: Math.max(0, Math.random() * 1e18 * (1 - doping * 2)),
      Tc,
      coupling_strength: lambda * DOS,
      layeredness: Math.max(0, Math.min(0.8, nest * 0.6 + (Math.random() - 0.5) * 0.2)),
      dimensionality: Math.max(0.2, Math.min(0.8, 0.5 + (press > 100 ? 0.15 : -0.1) + (Math.random() - 0.5) * 0.2)),
      topology_z2: nest > 0.4 && DOS > 2 ? 1 : 0,
      berry_phase: Math.max(0, Math.min(Math.PI, nest > 0.4 ? Math.PI * 0.7 + (Math.random() - 0.5) * 0.5 : Math.random() * Math.PI * 0.3)),
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
  const latestGraph = discoveryStore.getLatestGraph();
  const allHypotheses = discoveryStore.getHypotheses();
  const allRules = discoveryStore.getRules();

  const sortedEdges = latestGraph
    ? [...latestGraph.edges].sort((a, b) => b.strength - a.strength)
    : [];

  return {
    totalRuns: discoveryStore.totalRuns,
    totalGraphsDiscovered: discoveryStore.getGraphs().length,
    totalHypotheses: allHypotheses.length,
    totalCausalRules: allRules.length,
    causalVariableCount: CAUSAL_VARIABLES.length,
    interventionVariableCount: INTERVENTION_COUNT,
    ontologyNodeCount: PHYSICS_ONTOLOGY.length,
    topHypotheses: allHypotheses.slice(0, 8).map(h => ({
      id: h.id,
      statement: h.statement,
      causalChain: h.causalChain,
      confidence: Math.round(h.confidence * 1000) / 1000,
      materialFamilies: h.materialFamilies,
    })),
    topEdges: sortedEdges.slice(0, 10).map(e => ({
      source: e.source,
      target: e.target,
      strength: Math.round(e.strength * 1000) / 1000,
      mechanism: formatMechanism(e.mechanism),
    })),
    variableCategories: CATEGORY_COUNTS,
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
  return discoveryStore.getHypotheses();
}

export function getCausalRules(): CausalRule[] {
  return discoveryStore.getRules();
}

export function getLatestGraph(): CausalGraph | null {
  return discoveryStore.getLatestGraph();
}

export function getDesignGuidance(): DesignGuidance[] {
  return discoveryStore.getGuidance();
}
