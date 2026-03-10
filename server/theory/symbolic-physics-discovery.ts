import { computeMultiScaleFeatures, type MultiScaleFeatures, type CrossScaleCoupling, computeCrossScaleCoupling } from "./multi-scale-engine";
import { computePairingFeatureVector } from "../physics/pairing-mechanisms";
import { computeFermiSurface } from "../physics/fermi-surface-engine";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";

type OpName = "+" | "-" | "*" | "/" | "^" | "sqrt" | "exp" | "log" | "sin" | "abs";

interface OpNode { type: "op"; op: OpName; children: ExprNode[]; }
interface VarNode { type: "var"; name: string; }
interface ConstNode { type: "const"; value: number; }
type ExprNode = OpNode | VarNode | ConstNode;

const BINARY_OPS: OpName[] = ["+", "-", "*", "/", "^"];
const UNARY_OPS: OpName[] = ["sqrt", "exp", "log", "sin", "abs"];

function isUnary(op: OpName): boolean {
  return UNARY_OPS.includes(op);
}

function treeDepth(node: ExprNode): number {
  if (node.type !== "op") return 1;
  return 1 + Math.max(...node.children.map(treeDepth));
}

function treeSize(node: ExprNode): number {
  if (node.type !== "op") return 1;
  return 1 + node.children.reduce((s, c) => s + treeSize(c), 0);
}

function cloneTree(node: ExprNode): ExprNode {
  if (node.type === "const") return { type: "const", value: node.value };
  if (node.type === "var") return { type: "var", name: node.name };
  return { type: "op", op: node.op, children: node.children.map(cloneTree) };
}

function evaluateNode(node: ExprNode, row: Record<string, number>): number {
  if (node.type === "const") return node.value;
  if (node.type === "var") return row[node.name] ?? 0;
  const vals = node.children.map(c => evaluateNode(c, row));
  switch (node.op) {
    case "+": return vals[0] + vals[1];
    case "-": return vals[0] - vals[1];
    case "*": return vals[0] * vals[1];
    case "/": return Math.abs(vals[1]) < 1e-10 ? 1e4 : vals[0] / vals[1];
    case "^": {
      if (Math.abs(vals[1]) > 5) return 1e4;
      if (vals[0] < 0 && Math.abs(vals[1] - Math.round(vals[1])) > 1e-6) return 1e4;
      return Math.pow(Math.abs(vals[0]) + 1e-10, vals[1]);
    }
    case "sqrt": return vals[0] >= 0 ? Math.sqrt(vals[0]) : 1e4;
    case "exp": return Math.abs(vals[0]) > 50 ? 1e4 : Math.exp(vals[0]);
    case "log": return vals[0] > 1e-10 ? Math.log(vals[0]) : -1e4;
    case "sin": return Math.sin(vals[0]);
    case "abs": return Math.abs(vals[0]);
    default: return 0;
  }
}

function nodeToString(node: ExprNode): string {
  if (node.type === "const") return node.value.toFixed(3);
  if (node.type === "var") return node.name;
  if (isUnary(node.op)) return `${node.op}(${nodeToString(node.children[0])})`;
  return `(${nodeToString(node.children[0])} ${node.op} ${nodeToString(node.children[1])})`;
}

function randomChoice<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

export type PhysicsUnit =
  | "energy" | "frequency" | "length" | "charge" | "temperature"
  | "pressure" | "dimensionless" | "coupling" | "density" | "mass"
  | "velocity" | "field" | "unknown";

export interface UnitSpec {
  energy: number;
  frequency: number;
  length: number;
  temperature: number;
  pressure: number;
  mass: number;
  charge: number;
}

const VARIABLE_UNITS: Record<string, UnitSpec> = {
  lambda: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  DOS_EF: { energy: -1, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  omega_log: { energy: 0, frequency: 1, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  Tc: { energy: 0, frequency: 0, length: 0, temperature: 1, pressure: 0, mass: 0, charge: 0 },
  bandwidth: { energy: 1, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  pressure: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 1, mass: 0, charge: 0 },
  bulk_modulus: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 1, mass: 0, charge: 0 },
  mu_star: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  nesting_score: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  band_flatness: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  charge_transfer: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 1 },
  atomic_mass_avg: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 1, charge: 0 },
  electronegativity_spread: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  mott_proximity: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  layeredness: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  strain_sensitivity: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  dimensionality: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  U_over_W: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  DOS_lambda: { energy: -1, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  phonon_over_bandwidth: { energy: -1, frequency: 1, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  pressure_over_bulk: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
  coupling_strength: { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
};

const ZERO_UNIT: UnitSpec = { energy: 0, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 };

const ALLOWED_FRAC_EXPS = new Set([1/3, 2/3, 1/2, 3/2]);

function getVariableUnit(name: string): UnitSpec {
  return VARIABLE_UNITS[name] ?? { ...ZERO_UNIT };
}

function unitsMatch(a: UnitSpec, b: UnitSpec): boolean {
  return a.energy === b.energy && a.frequency === b.frequency &&
    a.length === b.length && a.temperature === b.temperature &&
    a.pressure === b.pressure && a.mass === b.mass && a.charge === b.charge;
}

function isDimensionless(u: UnitSpec): boolean {
  return unitsMatch(u, ZERO_UNIT);
}

function propagateUnits(node: ExprNode): UnitSpec | null {
  if (node.type === "const") return { ...ZERO_UNIT };
  if (node.type === "var") return getVariableUnit(node.name);

  const childUnits = node.children.map(c => propagateUnits(c));
  if (childUnits.some(u => u === null)) return null;

  const u = childUnits as UnitSpec[];

  switch (node.op) {
    case "+":
    case "-":
      if (!unitsMatch(u[0], u[1])) return null;
      return { ...u[0] };
    case "*":
      return {
        energy: u[0].energy + u[1].energy,
        frequency: u[0].frequency + u[1].frequency,
        length: u[0].length + u[1].length,
        temperature: u[0].temperature + u[1].temperature,
        pressure: u[0].pressure + u[1].pressure,
        mass: u[0].mass + u[1].mass,
        charge: u[0].charge + u[1].charge,
      };
    case "/":
      return {
        energy: u[0].energy - u[1].energy,
        frequency: u[0].frequency - u[1].frequency,
        length: u[0].length - u[1].length,
        temperature: u[0].temperature - u[1].temperature,
        pressure: u[0].pressure - u[1].pressure,
        mass: u[0].mass - u[1].mass,
        charge: u[0].charge - u[1].charge,
      };
    case "^":
      if (isDimensionless(u[0])) return { ...ZERO_UNIT };
      if (node.children[1].type === "const") {
        const exp = node.children[1].value;
        if (Number.isInteger(exp) || ALLOWED_FRAC_EXPS.has(exp)) {
          return {
            energy: u[0].energy * exp,
            frequency: u[0].frequency * exp,
            length: u[0].length * exp,
            temperature: u[0].temperature * exp,
            pressure: u[0].pressure * exp,
            mass: u[0].mass * exp,
            charge: u[0].charge * exp,
          };
        }
      }
      return null;
    case "sqrt":
    case "exp":
    case "log":
    case "sin":
    case "abs":
      if (node.op === "abs") return { ...u[0] };
      if (node.op === "sqrt") {
        const allEven = Object.values(u[0]).every(v => v % 2 === 0);
        if (!allEven && !isDimensionless(u[0])) return null;
        if (allEven) {
          return {
            energy: u[0].energy / 2,
            frequency: u[0].frequency / 2,
            length: u[0].length / 2,
            temperature: u[0].temperature / 2,
            pressure: u[0].pressure / 2,
            mass: u[0].mass / 2,
            charge: u[0].charge / 2,
          };
        }
        return { ...ZERO_UNIT };
      }
      if (!isDimensionless(u[0])) return null;
      return { ...ZERO_UNIT };
    default:
      return null;
  }
}

export interface SymbolicTerm {
  name: string;
  expression: string;
  tree: ExprNode;
  variables: string[];
  category: "basic" | "quadratic" | "interaction" | "ratio" | "transform" | "physics";
  physicsInspired: boolean;
}

const PHYSICS_VARIABLES = [
  "lambda", "DOS_EF", "omega_log", "bandwidth", "pressure",
  "mu_star", "nesting_score", "band_flatness", "charge_transfer",
  "atomic_mass_avg", "electronegativity_spread", "mott_proximity",
  "layeredness", "strain_sensitivity", "dimensionality",
  "U_over_W", "DOS_lambda", "phonon_over_bandwidth",
  "pressure_over_bulk", "coupling_strength",
];

function buildFeatureLibrary(): SymbolicTerm[] {
  const library: SymbolicTerm[] = [];

  for (const v of PHYSICS_VARIABLES) {
    library.push({
      name: v,
      expression: v,
      tree: { type: "var", name: v },
      variables: [v],
      category: "basic",
      physicsInspired: false,
    });
  }

  for (const v of PHYSICS_VARIABLES) {
    library.push({
      name: `${v}_sq`,
      expression: `${v}^2`,
      tree: { type: "op", op: "^", children: [{ type: "var", name: v }, { type: "const", value: 2 }] },
      variables: [v],
      category: "quadratic",
      physicsInspired: false,
    });
  }

  const interactionPairs = [
    ["lambda", "omega_log"], ["DOS_EF", "lambda"], ["lambda", "bandwidth"],
    ["nesting_score", "DOS_EF"], ["charge_transfer", "lambda"],
    ["mott_proximity", "bandwidth"], ["layeredness", "nesting_score"],
    ["strain_sensitivity", "lambda"], ["dimensionality", "DOS_EF"],
    ["coupling_strength", "omega_log"],
  ];

  for (const [a, b] of interactionPairs) {
    library.push({
      name: `${a}_x_${b}`,
      expression: `${a} * ${b}`,
      tree: { type: "op", op: "*", children: [{ type: "var", name: a }, { type: "var", name: b }] },
      variables: [a, b],
      category: "interaction",
      physicsInspired: true,
    });
  }

  const ratioPairs = [
    ["lambda", "mu_star"], ["omega_log", "bandwidth"], ["pressure", "pressure_over_bulk"],
    ["DOS_EF", "bandwidth"], ["charge_transfer", "electronegativity_spread"],
    ["nesting_score", "band_flatness"],
  ];

  for (const [a, b] of ratioPairs) {
    library.push({
      name: `${a}_over_${b}`,
      expression: `${a} / ${b}`,
      tree: { type: "op", op: "/", children: [{ type: "var", name: a }, { type: "var", name: b }] },
      variables: [a, b],
      category: "ratio",
      physicsInspired: true,
    });
  }

  const transformVars = ["lambda", "DOS_EF", "omega_log", "bandwidth", "coupling_strength"];
  for (const v of transformVars) {
    library.push({
      name: `sqrt_${v}`,
      expression: `sqrt(${v})`,
      tree: { type: "op", op: "sqrt", children: [{ type: "var", name: v }] },
      variables: [v],
      category: "transform",
      physicsInspired: false,
    });
    library.push({
      name: `log_${v}`,
      expression: `log(${v})`,
      tree: { type: "op", op: "log", children: [{ type: "var", name: v }] },
      variables: [v],
      category: "transform",
      physicsInspired: false,
    });
  }

  library.push({
    name: "mcmillan_core",
    expression: "omega_log * exp(-1.04 * (1 + lambda) / (lambda - mu_star * (1 + 0.62 * lambda)))",
    tree: { type: "op", op: "*", children: [
      { type: "var", name: "omega_log" },
      { type: "op", op: "exp", children: [
        { type: "op", op: "*", children: [
          { type: "const", value: -1.04 },
          { type: "op", op: "/", children: [
            { type: "op", op: "+", children: [{ type: "const", value: 1 }, { type: "var", name: "lambda" }] },
            { type: "op", op: "-", children: [
              { type: "var", name: "lambda" },
              { type: "op", op: "*", children: [
                { type: "var", name: "mu_star" },
                { type: "op", op: "+", children: [{ type: "const", value: 1 }, { type: "op", op: "*", children: [{ type: "const", value: 0.62 }, { type: "var", name: "lambda" }] }] },
              ]},
            ]},
          ]},
        ]},
      ]},
    ]},
    variables: ["omega_log", "lambda", "mu_star"],
    category: "physics",
    physicsInspired: true,
  });

  library.push({
    name: "screened_coupling",
    expression: "lambda / (1 + lambda * mu_star)",
    tree: { type: "op", op: "/", children: [
      { type: "var", name: "lambda" },
      { type: "op", op: "+", children: [
        { type: "const", value: 1 },
        { type: "op", op: "*", children: [{ type: "var", name: "lambda" }, { type: "var", name: "mu_star" }] },
      ]},
    ]},
    variables: ["lambda", "mu_star"],
    category: "physics",
    physicsInspired: true,
  });

  library.push({
    name: "dos_phonon_product",
    expression: "DOS_EF * lambda * omega_log",
    tree: { type: "op", op: "*", children: [
      { type: "op", op: "*", children: [{ type: "var", name: "DOS_EF" }, { type: "var", name: "lambda" }] },
      { type: "var", name: "omega_log" },
    ]},
    variables: ["DOS_EF", "lambda", "omega_log"],
    category: "physics",
    physicsInspired: true,
  });

  library.push({
    name: "nesting_enhanced_coupling",
    expression: "lambda * (1 + nesting_score) * DOS_EF",
    tree: { type: "op", op: "*", children: [
      { type: "op", op: "*", children: [
        { type: "var", name: "lambda" },
        { type: "op", op: "+", children: [{ type: "const", value: 1 }, { type: "var", name: "nesting_score" }] },
      ]},
      { type: "var", name: "DOS_EF" },
    ]},
    variables: ["lambda", "nesting_score", "DOS_EF"],
    category: "physics",
    physicsInspired: true,
  });

  library.push({
    name: "correlation_suppression",
    expression: "lambda * (1 - mott_proximity) * sqrt(DOS_EF)",
    tree: { type: "op", op: "*", children: [
      { type: "op", op: "*", children: [
        { type: "var", name: "lambda" },
        { type: "op", op: "-", children: [{ type: "const", value: 1 }, { type: "var", name: "mott_proximity" }] },
      ]},
      { type: "op", op: "sqrt", children: [{ type: "var", name: "DOS_EF" }] },
    ]},
    variables: ["lambda", "mott_proximity", "DOS_EF"],
    category: "physics",
    physicsInspired: true,
  });

  return library;
}

export interface PhysicsDiscoveryRecord {
  formula: string;
  lambda: number;
  DOS_EF: number;
  omega_log: number;
  Tc: number;
  bandwidth: number;
  pressure: number;
  mu_star: number;
  nesting_score: number;
  band_flatness: number;
  charge_transfer: number;
  atomic_mass_avg: number;
  electronegativity_spread: number;
  mott_proximity: number;
  layeredness: number;
  strain_sensitivity: number;
  dimensionality: number;
  coupling_strength: number;
  U_over_W: number;
  DOS_lambda: number;
  phonon_over_bandwidth: number;
  pressure_over_bulk: number;
  material_family: string;
  pairing_mechanism: string;
  pressure_regime: string;
}

export function buildPhysicsDiscoveryRecord(formula: string): PhysicsDiscoveryRecord {
  let multiScale: MultiScaleFeatures | null = null;
  let crossScale: CrossScaleCoupling | null = null;
  try {
    multiScale = computeMultiScaleFeatures(formula);
    crossScale = computeCrossScaleCoupling(multiScale);
  } catch {}

  let features: Record<string, number> = {};
  try {
    features = extractFeatures(formula);
  } catch {}

  let gb: { tcPredicted: number; score: number } = { tcPredicted: 0, score: 0 };
  try {
    gb = gbPredict(features);
  } catch {}

  const lambda = features.electronPhononLambda ?? multiScale?.electronic?.nesting_score ?? 0.5;
  const DOS_EF = multiScale?.electronic?.DOS_EF ?? features.dosAtEF ?? 1.0;
  const omega_log = features.logPhononFreq ?? 400;
  const bandwidth = multiScale?.electronic?.bandwidth ?? features.avgBulkModulus ?? 2.0;
  const mu_star = features.muStarEstimate ?? 0.10;
  const nesting_score = multiScale?.electronic?.nesting_score ?? features.fermiSurfaceNestingScore ?? 0.1;
  const band_flatness = multiScale?.electronic?.band_flatness ?? features.bandFlatness ?? 0;
  const charge_transfer = multiScale?.atomic?.charge_transfer ?? features.chargeTransferMagnitude ?? 0.3;
  const atomic_mass_avg = multiScale?.atomic?.atomic_mass_avg ?? features.maxAtomicMass ?? 50;
  const electronegativity_spread = multiScale?.atomic?.electronegativity_spread ?? features.avgElectronegativity ?? 1.5;
  const mott_proximity = multiScale?.electronic?.mott_proximity ?? features.mottProximityScore ?? 0;
  const layeredness = multiScale?.mesoscopic?.layeredness ?? (features.layeredStructure ? 0.8 : 0.1);
  const strain_sensitivity = multiScale?.mesoscopic?.strain_sensitivity ?? 0.3;
  const dimensionality = multiScale?.mesoscopic?.dimensionality ?? features.dimensionalityScore ?? 0.5;
  const pressure = features.avgBulkModulus ? features.avgBulkModulus * 0.1 : 0;
  const bulk_modulus = features.avgBulkModulus ?? 100;
  const coupling_strength = crossScale?.electron_phonon_mass_ratio ?? lambda * DOS_EF;

  const U_over_W = mott_proximity > 0 ? mott_proximity * 1.5 : 0.2;
  const DOS_lambda = DOS_EF * lambda;
  const phonon_over_bandwidth = bandwidth > 0 ? omega_log / bandwidth : 0;
  const pressure_over_bulk = bulk_modulus > 0 ? pressure / bulk_modulus : 0;

  const hasH = formula.includes("H");
  const hasCu = formula.includes("Cu");
  const hasFe = formula.includes("Fe");
  const hasBi = formula.includes("Bi");
  let family = "other";
  if (hasH) family = "hydride";
  else if (hasCu && formula.includes("O")) family = "cuprate";
  else if (hasFe && (formula.includes("As") || formula.includes("Se"))) family = "iron-based";
  else if (hasBi) family = "topological";
  else if (layeredness > 0.5) family = "layered";
  else family = "intermetallic";

  let mechanism = "phonon-mediated";
  if (mott_proximity > 0.5) mechanism = "correlation-enhanced";
  else if (nesting_score > 0.4) mechanism = "nesting-driven";
  else if (lambda > 2) mechanism = "strong-coupling";

  let pressureRegime = "ambient";
  if (pressure > 100) pressureRegime = "high";
  else if (pressure > 20) pressureRegime = "moderate";

  return {
    formula,
    lambda, DOS_EF, omega_log, Tc: gb.tcPredicted, bandwidth, pressure,
    mu_star, nesting_score, band_flatness, charge_transfer,
    atomic_mass_avg, electronegativity_spread, mott_proximity,
    layeredness, strain_sensitivity, dimensionality, coupling_strength,
    U_over_W, DOS_lambda, phonon_over_bandwidth, pressure_over_bulk,
    material_family: family,
    pairing_mechanism: mechanism,
    pressure_regime: pressureRegime,
  };
}

export interface PhysicsConstraintCheck {
  variable: string;
  constraint: string;
  min?: number;
  max?: number;
  satisfied: boolean;
  value: number;
}

function checkPhysicsConstraints(row: Record<string, number>): PhysicsConstraintCheck[] {
  const checks: PhysicsConstraintCheck[] = [];

  const constraints: { variable: string; constraint: string; min?: number; max?: number }[] = [
    { variable: "Tc", constraint: "Tc >= 0", min: 0 },
    { variable: "lambda", constraint: "lambda >= 0", min: 0 },
    { variable: "lambda", constraint: "lambda < 5", max: 5 },
    { variable: "DOS_EF", constraint: "DOS_EF >= 0", min: 0 },
    { variable: "omega_log", constraint: "omega_log > 0", min: 0.01 },
    { variable: "mu_star", constraint: "0 <= mu_star <= 0.3", min: 0, max: 0.3 },
    { variable: "U_over_W", constraint: "U/W < Mott threshold (2.0)", max: 2.0 },
    { variable: "nesting_score", constraint: "0 <= nesting <= 1", min: 0, max: 1 },
    { variable: "band_flatness", constraint: "0 <= flatness <= 1", min: 0, max: 1 },
    { variable: "dimensionality", constraint: "0 <= dim <= 1", min: 0, max: 1 },
  ];

  for (const c of constraints) {
    const value = row[c.variable] ?? 0;
    let satisfied = true;
    if (c.min !== undefined && value < c.min) satisfied = false;
    if (c.max !== undefined && value > c.max) satisfied = false;
    checks.push({ ...c, satisfied, value });
  }

  return checks;
}

function validateEquationPhysics(tree: ExprNode, dataset: Record<string, number>[]): number {
  let violations = 0;
  let total = 0;

  for (const row of dataset) {
    const predicted = evaluateNode(tree, row);
    total++;
    if (!isFinite(predicted)) { violations++; continue; }
    if (predicted < -100) violations++;
    if (predicted > 10000) violations++;
  }

  return total > 0 ? 1 - violations / total : 0;
}

export type MaterialFamily = "hydride" | "cuprate" | "iron-based" | "hea" | "topological" | "intermetallic" | "layered" | "other";

export interface CrossScaleValidation {
  family: MaterialFamily;
  sampleCount: number;
  r2: number;
  mae: number;
  meanPredicted: number;
  meanActual: number;
}

function computeR2(predictions: number[], actuals: number[]): { r2: number; mae: number } {
  if (predictions.length < 2) return { r2: 0, mae: 0 };
  const meanActual = actuals.reduce((a, b) => a + b, 0) / actuals.length;
  const ssTot = actuals.reduce((s, y) => s + (y - meanActual) ** 2, 0);
  const ssRes = predictions.reduce((s, p, i) => s + (actuals[i] - p) ** 2, 0);
  const mae = predictions.reduce((s, p, i) => s + Math.abs(actuals[i] - p), 0) / predictions.length;
  return { r2: ssTot > 0 ? 1 - ssRes / ssTot : 0, mae };
}

function crossScaleValidate(tree: ExprNode, dataset: PhysicsDiscoveryRecord[]): CrossScaleValidation[] {
  const families = new Map<MaterialFamily, PhysicsDiscoveryRecord[]>();
  for (const rec of dataset) {
    const fam = rec.material_family as MaterialFamily;
    if (!families.has(fam)) families.set(fam, []);
    families.get(fam)!.push(rec);
  }

  const K_FOLDS = 3;

  const results: CrossScaleValidation[] = [];
  for (const [family, records] of families) {
    if (records.length < 3) continue;
    const allActuals = records.map(r => r.Tc);

    const foldR2s: number[] = [];
    const foldMAEs: number[] = [];
    const allFoldPreds: number[] = [];
    const allFoldActuals: number[] = [];

    for (let fold = 0; fold < K_FOLDS; fold++) {
      const holdout: PhysicsDiscoveryRecord[] = [];
      const train: PhysicsDiscoveryRecord[] = [];
      for (let i = 0; i < records.length; i++) {
        if (i % K_FOLDS === fold) {
          holdout.push(records[i]);
        } else {
          train.push(records[i]);
        }
      }
      if (holdout.length === 0 || train.length === 0) continue;

      const holdoutPreds = holdout.map(r => evaluateNode(tree, r as any));
      const holdoutActuals = holdout.map(r => r.Tc);
      const validPreds: number[] = [];
      const validActuals: number[] = [];
      for (let i = 0; i < holdoutPreds.length; i++) {
        if (isFinite(holdoutPreds[i]) && Math.abs(holdoutPreds[i]) < 1e4) {
          validPreds.push(holdoutPreds[i]);
          validActuals.push(holdoutActuals[i]);
        }
      }
      if (validPreds.length < 1) {
        foldR2s.push(0);
        foldMAEs.push(999);
        continue;
      }
      const { r2, mae } = computeR2(validPreds, validActuals);
      foldR2s.push(Math.max(0, r2));
      foldMAEs.push(mae);
      allFoldPreds.push(...validPreds);
      allFoldActuals.push(...validActuals);
    }

    if (foldR2s.length === 0) {
      results.push({ family, sampleCount: records.length, r2: 0, mae: 999, meanPredicted: 0, meanActual: allActuals.reduce((a, b) => a + b, 0) / allActuals.length });
      continue;
    }

    const avgR2 = foldR2s.reduce((a, b) => a + b, 0) / foldR2s.length;
    const avgMAE = foldMAEs.reduce((a, b) => a + b, 0) / foldMAEs.length;

    results.push({
      family,
      sampleCount: records.length,
      r2: Math.max(0, avgR2),
      mae: avgMAE,
      meanPredicted: allFoldPreds.length > 0 ? allFoldPreds.reduce((a, b) => a + b, 0) / allFoldPreds.length : 0,
      meanActual: allFoldActuals.length > 0 ? allFoldActuals.reduce((a, b) => a + b, 0) / allFoldActuals.length : allActuals.reduce((a, b) => a + b, 0) / allActuals.length,
    });
  }

  return results;
}

function simplifyTree(node: ExprNode): ExprNode {
  if (node.type !== "op") return cloneTree(node);

  const simplified = node.children.map(simplifyTree);

  if (node.op === "+" || node.op === "-") {
    if (simplified[1].type === "const" && simplified[1].value === 0) return simplified[0];
    if (node.op === "+" && simplified[0].type === "const" && simplified[0].value === 0) return simplified[1];
  }

  if (node.op === "*") {
    if (simplified[0].type === "const" && simplified[0].value === 0) return { type: "const", value: 0 };
    if (simplified[1].type === "const" && simplified[1].value === 0) return { type: "const", value: 0 };
    if (simplified[0].type === "const" && simplified[0].value === 1) return simplified[1];
    if (simplified[1].type === "const" && simplified[1].value === 1) return simplified[0];
  }

  if (node.op === "/" && simplified[1].type === "const" && simplified[1].value === 1) {
    return simplified[0];
  }

  if (node.op === "^") {
    if (simplified[1].type === "const" && simplified[1].value === 0) return { type: "const", value: 1 };
    if (simplified[1].type === "const" && simplified[1].value === 1) return simplified[0];
  }

  if (simplified.every(c => c.type === "const")) {
    const vals = simplified.map(c => (c as ConstNode).value);
    let result: number;
    switch (node.op) {
      case "+": result = vals[0] + vals[1]; break;
      case "-": result = vals[0] - vals[1]; break;
      case "*": result = vals[0] * vals[1]; break;
      case "/": result = Math.abs(vals[1]) < 1e-10 ? 1e4 : vals[0] / vals[1]; break;
      case "sqrt": result = vals[0] >= 0 ? Math.sqrt(vals[0]) : 0; break;
      case "log": result = vals[0] > 0 ? Math.log(vals[0]) : 0; break;
      case "exp": result = Math.abs(vals[0]) > 50 ? 1e4 : Math.exp(vals[0]); break;
      default: result = vals[0]; break;
    }
    if (isFinite(result)) return { type: "const", value: Math.round(result * 1000) / 1000 };
  }

  return { type: "op", op: node.op, children: simplified };
}

export interface DiscoveredTheory {
  id: string;
  equation: string;
  tree: ExprNode;
  target: string;
  accuracy: number;
  r2: number;
  mae: number;
  complexity: number;
  simplicity: number;
  generalization: number;
  physicsCompliance: number;
  novelty: number;
  theoryScore: number;
  crossScaleValidation: CrossScaleValidation[];
  dimensionallyValid: boolean;
  constraintViolations: number;
  discoveredAt: number;
  generation: number;
  variables: string[];
  applicableFamilies: string[];
  simplified: string;
  featureImportance: { variable: string; importance: number }[];
}

export interface SymbolicDiscoveryConfig {
  populationSize: number;
  generations: number;
  maxTreeDepth: number;
  mutationRate: number;
  crossoverRate: number;
  tournamentSize: number;
  complexityPenalty: number;
  physicsConstraintWeight: number;
  useFeatureLibrary: boolean;
  targetVariable: string;
}

const DEFAULT_CONFIG: SymbolicDiscoveryConfig = {
  populationSize: 100,
  generations: 60,
  maxTreeDepth: 5,
  mutationRate: 0.3,
  crossoverRate: 0.5,
  tournamentSize: 4,
  complexityPenalty: 0.015,
  physicsConstraintWeight: 0.15,
  useFeatureLibrary: true,
  targetVariable: "Tc",
};

function generateRandomEquation(vars: string[], maxDepth: number, depth: number = 0): ExprNode {
  if (depth >= maxDepth || (depth > 1 && Math.random() < 0.4)) {
    if (Math.random() < 0.6) return { type: "var", name: randomChoice(vars) };
    return { type: "const", value: Math.round((Math.random() * 10 - 5) * 100) / 100 };
  }

  const allOps: OpName[] = [...BINARY_OPS, ...UNARY_OPS];
  const op = randomChoice(allOps);
  const n = isUnary(op) ? 1 : 2;
  const children: ExprNode[] = [];
  for (let i = 0; i < n; i++) {
    children.push(generateRandomEquation(vars, maxDepth, depth + 1));
  }
  return { type: "op", op, children };
}

function mutateEquation(tree: ExprNode, vars: string[]): ExprNode {
  const mutated = cloneTree(tree);
  const r = Math.random();

  if (r < 0.25 && mutated.type === "op") {
    const allOps: OpName[] = isUnary(mutated.op) ? UNARY_OPS : BINARY_OPS;
    mutated.op = randomChoice(allOps);
  } else if (r < 0.5 && mutated.type === "var") {
    return { type: "var", name: randomChoice(vars) };
  } else if (r < 0.7) {
    return generateRandomEquation(vars, 3);
  } else if (r < 0.85 && mutated.type === "op") {
    const childIdx = Math.floor(Math.random() * mutated.children.length);
    mutated.children[childIdx] = mutateEquation(mutated.children[childIdx], vars);
  } else {
    return { type: "op", op: randomChoice(BINARY_OPS), children: [mutated, generateRandomEquation(vars, 2)] };
  }

  return mutated;
}

function crossoverEquations(a: ExprNode, b: ExprNode, vars: string[]): ExprNode {
  const child = cloneTree(a);
  if (child.type !== "op" || b.type !== "op") return child;

  const childIdx = Math.floor(Math.random() * child.children.length);
  const bIdx = Math.floor(Math.random() * b.children.length);
  child.children[childIdx] = cloneTree(b.children[bIdx]);
  return child;
}

function tournamentSelect(pop: ExprNode[], fitnesses: number[], size: number): number {
  let best = Math.floor(Math.random() * pop.length);
  for (let i = 1; i < size; i++) {
    const idx = Math.floor(Math.random() * pop.length);
    if (fitnesses[idx] > fitnesses[best]) best = idx;
  }
  return best;
}

const TARGET_UNITS: Record<string, UnitSpec> = {
  Tc: { energy: 0, frequency: 0, length: 0, temperature: 1, pressure: 0, mass: 0, charge: 0 },
  lambda: { ...ZERO_UNIT },
  DOS_EF: { energy: -1, frequency: 0, length: 0, temperature: 0, pressure: 0, mass: 0, charge: 0 },
};

function isDimensionallyValidForTarget(tree: ExprNode, target: string): boolean {
  const outputUnit = propagateUnits(tree);
  if (outputUnit === null) return false;
  const expected = TARGET_UNITS[target] ?? ZERO_UNIT;
  return unitsMatch(outputUnit, expected);
}

function computeEquationFitness(
  tree: ExprNode,
  dataset: PhysicsDiscoveryRecord[],
  target: string,
  config: SymbolicDiscoveryConfig,
): { fitness: number; r2: number; mae: number; complexity: number; physicsScore: number; dimValid: boolean } {
  const predictions: number[] = [];
  const actuals: number[] = [];

  for (const row of dataset) {
    const predicted = evaluateNode(tree, row as any);
    if (!isFinite(predicted) || Math.abs(predicted) > 1e4) {
      predictions.push(0);
    } else {
      predictions.push(predicted);
    }
    actuals.push((row as any)[target] ?? 0);
  }

  const { r2, mae } = computeR2(predictions, actuals);
  const complexity = treeSize(tree);
  const complexityPenalty = complexity * config.complexityPenalty;

  const physicsScore = validateEquationPhysics(tree, dataset as any[]);
  const dimValid = isDimensionallyValidForTarget(tree, target);
  const dimPenalty = dimValid ? 0.05 : -0.05;

  let constraintViolations = 0;
  let constraintTotal = 0;
  for (const row of dataset) {
    const pred = evaluateNode(tree, row as any);
    constraintTotal++;
    if (!isFinite(pred)) { constraintViolations++; continue; }
    if (target === "Tc") {
      if (pred < -10) constraintViolations++;
      if (pred > 500) constraintViolations++;
    }
    if (pred < 0 && target === "lambda") constraintViolations++;
  }
  const constraintRate = constraintTotal > 0 ? 1 - constraintViolations / constraintTotal : 0;

  const fitness = Math.max(0,
    r2 * 0.5 +
    (1 - Math.min(1, mae / 100)) * 0.15 +
    physicsScore * config.physicsConstraintWeight +
    constraintRate * 0.1 +
    dimPenalty -
    complexityPenalty
  );

  return { fitness, r2: Math.max(0, r2), mae, complexity, physicsScore, dimValid };
}

function extractVariables(tree: ExprNode): string[] {
  const vars = new Set<string>();
  function walk(node: ExprNode) {
    if (node.type === "var") vars.add(node.name);
    if (node.type === "op") node.children.forEach(walk);
  }
  walk(tree);
  return Array.from(vars);
}

function computeFeatureImportance(tree: ExprNode, dataset: PhysicsDiscoveryRecord[], target: string): { variable: string; importance: number }[] {
  const vars = extractVariables(tree);
  const importances: { variable: string; importance: number }[] = [];

  const basePreds = dataset.map(r => evaluateNode(tree, r as any));
  const actuals = dataset.map(r => (r as any)[target] ?? 0);
  const baseR2 = computeR2(basePreds.filter(isFinite), actuals.slice(0, basePreds.filter(isFinite).length)).r2;

  for (const v of vars) {
    const shuffled = dataset.map(r => {
      const copy = { ...r } as any;
      copy[v] = dataset[Math.floor(Math.random() * dataset.length)][v as keyof PhysicsDiscoveryRecord];
      return copy;
    });
    const shuffledPreds = shuffled.map(r => evaluateNode(tree, r));
    const shuffledR2 = computeR2(shuffledPreds.filter(isFinite), actuals.slice(0, shuffledPreds.filter(isFinite).length)).r2;
    importances.push({ variable: v, importance: Math.max(0, baseR2 - shuffledR2) });
  }

  return importances.sort((a, b) => b.importance - a.importance);
}

function computeNovelty(equation: string, existingTheories: DiscoveredTheory[]): number {
  if (existingTheories.length === 0) return 1.0;

  let minSimilarity = 1.0;
  for (const existing of existingTheories) {
    const overlap = equation.split("").filter((c, i) => existing.equation[i] === c).length;
    const maxLen = Math.max(equation.length, existing.equation.length);
    const similarity = maxLen > 0 ? overlap / maxLen : 1;
    if (similarity < minSimilarity) minSimilarity = similarity;
  }

  return 1 - minSimilarity;
}

const theoryDatabase: DiscoveredTheory[] = [];
let totalDiscoveryRuns = 0;
let totalEquationsEvaluated = 0;

export function runSymbolicPhysicsDiscovery(
  dataset: PhysicsDiscoveryRecord[],
  config: Partial<SymbolicDiscoveryConfig> = {},
): DiscoveredTheory[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const target = cfg.targetVariable;
  totalDiscoveryRuns++;

  const featureLibrary = cfg.useFeatureLibrary ? buildFeatureLibrary() : [];
  const vars = PHYSICS_VARIABLES.filter(v => v !== target);

  let population: ExprNode[] = [];

  if (cfg.useFeatureLibrary) {
    const physicTerms = featureLibrary.filter(t => t.physicsInspired);
    for (const term of physicTerms) {
      population.push(cloneTree(term.tree));
    }
    for (const term of featureLibrary.filter(t => !t.physicsInspired).slice(0, Math.max(0, Math.floor(cfg.populationSize * 0.2) - physicTerms.length))) {
      population.push(cloneTree(term.tree));
    }
  }

  while (population.length < cfg.populationSize) {
    const depth = 2 + Math.floor(Math.random() * (cfg.maxTreeDepth - 1));
    population.push(generateRandomEquation(vars, depth));
  }

  let bestOverall: { tree: ExprNode; fitness: number; r2: number; mae: number; gen: number } | null = null;
  const discovered: DiscoveredTheory[] = [];

  for (let gen = 0; gen < cfg.generations; gen++) {
    const fitnesses = population.map(tree => {
      totalEquationsEvaluated++;
      return computeEquationFitness(tree, dataset, target, cfg);
    });

    const fitnessValues = fitnesses.map(f => f.fitness);

    const indexed = fitnesses.map((f, i) => ({ f, i }));
    indexed.sort((a, b) => b.f.fitness - a.f.fitness);

    const best = indexed[0];
    if (!bestOverall || best.f.fitness > bestOverall.fitness) {
      bestOverall = { tree: cloneTree(population[best.i]), fitness: best.f.fitness, r2: best.f.r2, mae: best.f.mae, gen };
    }

    if ((gen % 10 === 9 || gen === cfg.generations - 1) && indexed.length > 0) {
      for (let k = 0; k < Math.min(5, indexed.length); k++) {
        const idx = indexed[k].i;
        const tree = population[idx];
        const fit = fitnesses[idx];

        if (fit.r2 < 0.05) continue;
        if (treeSize(tree) > 35) continue;

        const equation = nodeToString(tree);
        if (discovered.some(d => d.equation === equation)) continue;

        const simplified = simplifyTree(tree);
        const simplifiedEq = nodeToString(simplified);
        const crossScale = crossScaleValidate(tree, dataset);
        const avgGenR2 = crossScale.length > 0
          ? crossScale.reduce((s, c) => s + c.r2, 0) / crossScale.length
          : 0;

        if (avgGenR2 < 0.3) continue;

        const dimValid = isDimensionallyValidForTarget(tree, target);

        let constraintViolations = 0;
        for (const row of dataset) {
          const pred = evaluateNode(tree, row as any);
          if (!isFinite(pred) || pred < -10 || pred > 500) constraintViolations++;
        }

        const complexity = treeSize(tree);
        const simplicity = Math.max(0, 1 - complexity / 25);
        const novelty = computeNovelty(equation, [...theoryDatabase, ...discovered]);
        const physicsCompliance = fit.physicsScore;

        const dimBonus = dimValid ? 0.10 : 0;
        const theoryScore =
          0.30 * fit.r2 +
          0.15 * simplicity +
          0.15 * avgGenR2 +
          0.15 * physicsCompliance +
          dimBonus +
          0.15 * novelty;

        const variables = extractVariables(tree);
        const featureImportance = computeFeatureImportance(tree, dataset, target);

        const applicableFamilies = crossScale
          .filter(c => c.r2 > 0.1 && c.sampleCount >= 3)
          .map(c => c.family);

        const theory: DiscoveredTheory = {
          id: `theory-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          equation,
          tree: cloneTree(tree),
          target,
          accuracy: fit.r2,
          r2: fit.r2,
          mae: fit.mae,
          complexity,
          simplicity,
          generalization: avgGenR2,
          physicsCompliance,
          novelty,
          theoryScore,
          crossScaleValidation: crossScale,
          dimensionallyValid: dimValid,
          constraintViolations,
          discoveredAt: Date.now(),
          generation: gen,
          variables,
          applicableFamilies,
          simplified: simplifiedEq,
          featureImportance,
        };

        discovered.push(theory);
      }
    }

    const nextPop: ExprNode[] = [];
    const eliteCount = Math.floor(cfg.populationSize * 0.1);
    for (let k = 0; k < eliteCount && k < indexed.length; k++) {
      nextPop.push(cloneTree(population[indexed[k].i]));
    }

    while (nextPop.length < cfg.populationSize) {
      const r = Math.random();
      if (r < cfg.crossoverRate) {
        const p1 = tournamentSelect(population, fitnessValues, cfg.tournamentSize);
        const p2 = tournamentSelect(population, fitnessValues, cfg.tournamentSize);
        nextPop.push(crossoverEquations(population[p1], population[p2], vars));
      } else if (r < cfg.crossoverRate + cfg.mutationRate) {
        const p = tournamentSelect(population, fitnessValues, cfg.tournamentSize);
        nextPop.push(mutateEquation(population[p], vars));
      } else {
        const depth = 2 + Math.floor(Math.random() * (cfg.maxTreeDepth - 1));
        nextPop.push(generateRandomEquation(vars, depth));
      }
    }

    population = nextPop;
  }

  for (const theory of discovered) {
    if (!theory.dimensionallyValid) continue;
    if (!theoryDatabase.some(t => t.equation === theory.equation)) {
      theoryDatabase.push(theory);
    }
  }

  const validTheories = theoryDatabase.filter(t => t.dimensionallyValid);
  theoryDatabase.length = 0;
  theoryDatabase.push(...validTheories);

  if (theoryDatabase.length > 100) {
    theoryDatabase.sort((a, b) => b.theoryScore - a.theoryScore);
    theoryDatabase.splice(100);
  }

  return discovered.sort((a, b) => b.theoryScore - a.theoryScore);
}

export function generateSyntheticDataset(count: number = 50): PhysicsDiscoveryRecord[] {
  const formulas = [
    "MgB2", "NbSn3", "LaH10", "YBa2Cu3O7", "FeSe", "Nb3Ge",
    "CaH6", "SrTiO3", "Bi2Sr2CaCu2O8", "LaFeAsO", "NbN", "VN",
    "TiN", "ZrN", "HfN", "MoC", "NbC", "TaC", "WC", "ScH3",
    "BaFe2As2", "LiFeAs", "FeTeSe", "KFe2Se2", "CeCoIn5",
    "PuCoGa5", "UPt3", "CeRhIn5", "Sr2RuO4", "LaRu2P2",
    "NbTiZrHfV", "TaNbHfZrTi", "MoNbTaVW", "LaBH8", "YH6",
    "CeH9", "ThH10", "PrH9", "NdH9", "EuH6", "CaBe2Ge2",
    "Ba2CuO4", "Tl2Ba2CaCu2O8", "HgBa2Ca2Cu3O8", "La2CuO4",
    "PbMo6S8", "LuNi2B2C", "YNi2B2C", "CaC6", "SrC6",
  ];

  const records: PhysicsDiscoveryRecord[] = [];
  for (let i = 0; i < Math.min(count, formulas.length); i++) {
    try {
      const rec = buildPhysicsDiscoveryRecord(formulas[i]);
      if (rec.Tc <= 0 && rec.lambda > 0.1 && rec.omega_log > 10) {
        const muEff = rec.mu_star * (1 + 0.62 * rec.lambda);
        const denom = rec.lambda - muEff;
        if (denom > 0.05) {
          const lambdaBarR = 2.46 * (1 + 3.8 * rec.mu_star);
          const f1r = Math.pow(1 + Math.pow(rec.lambda / lambdaBarR, 3 / 2), 1 / 3);
          rec.Tc = (rec.omega_log * 1.4388 / 1.2) * f1r * Math.exp(-1.04 * (1 + rec.lambda) / denom);
          rec.Tc = Math.max(0, Math.min(300, rec.Tc)) + (Math.random() - 0.5) * 2;
        }
      }
      records.push(rec);
    } catch {}
  }

  while (records.length < count) {
    const lambda = 0.3 + Math.random() * 2.5;
    const omega_log = 100 + Math.random() * 1500;
    const mu_star = 0.08 + Math.random() * 0.07;
    const DOS_EF = 0.5 + Math.random() * 4;
    const bandwidth = 1 + Math.random() * 8;
    const nesting = Math.random() * 0.8;
    const muEff = mu_star * (1 + 0.62 * lambda);
    const denom = lambda - muEff;
    let Tc = 0;
    if (denom > 0.05) {
      const lambdaBarS = 2.46 * (1 + 3.8 * mu_star);
      const f1s = Math.pow(1 + Math.pow(lambda / lambdaBarS, 3 / 2), 1 / 3);
      Tc = (omega_log * 1.4388 / 1.2) * f1s * Math.exp(-1.04 * (1 + lambda) / denom);
      Tc = Math.max(0, Math.min(300, Tc));
    }
    Tc += (Math.random() - 0.5) * 5;
    Tc = Math.max(0, Tc);

    records.push({
      formula: `Synth${records.length}`,
      lambda, DOS_EF, omega_log, Tc, bandwidth,
      pressure: Math.random() * 200,
      mu_star, nesting_score: nesting,
      band_flatness: Math.random() * 0.5,
      charge_transfer: Math.random() * 1.5,
      atomic_mass_avg: 20 + Math.random() * 200,
      electronegativity_spread: 0.5 + Math.random() * 2,
      mott_proximity: Math.random() * 0.5,
      layeredness: Math.random() * 0.8,
      strain_sensitivity: Math.random() * 0.5,
      dimensionality: 0.2 + Math.random() * 0.6,
      coupling_strength: lambda * DOS_EF,
      U_over_W: Math.random() * 1.5,
      DOS_lambda: DOS_EF * lambda,
      phonon_over_bandwidth: bandwidth > 0 ? omega_log / bandwidth : 0,
      pressure_over_bulk: Math.random() * 0.3,
      material_family: ["hydride", "cuprate", "iron-based", "intermetallic", "layered", "other"][Math.floor(Math.random() * 6)],
      pairing_mechanism: "phonon-mediated",
      pressure_regime: Math.random() > 0.5 ? "ambient" : "moderate",
    });
  }

  return records;
}

export interface DiscoveryFeedback {
  biasedVariables: { variable: string; direction: "increase" | "decrease"; strength: number }[];
  suggestedCompositions: string[];
  theoreticalInsight: string;
}

export function generateDiscoveryFeedback(theories: DiscoveredTheory[]): DiscoveryFeedback {
  const topTheories = theories.filter(t => t.theoryScore > 0.3 && t.dimensionallyValid).slice(0, 5);
  const biased: Map<string, { direction: "increase" | "decrease"; totalImportance: number }> = new Map();

  for (const theory of topTheories) {
    for (const fi of theory.featureImportance.slice(0, 3)) {
      const existing = biased.get(fi.variable);
      const tcCorrelation = theory.r2 > 0.3 ? "increase" : "decrease";
      if (!existing || fi.importance > existing.totalImportance) {
        biased.set(fi.variable, { direction: tcCorrelation as "increase" | "decrease", totalImportance: fi.importance });
      }
    }
  }

  const biasedVariables = Array.from(biased.entries())
    .map(([variable, { direction, totalImportance }]) => ({
      variable, direction, strength: Math.min(1, totalImportance * 2),
    }))
    .sort((a, b) => b.strength - a.strength)
    .slice(0, 5);

  const suggestions: string[] = [];
  for (const theory of topTheories.slice(0, 2)) {
    if (theory.variables.includes("lambda") && theory.variables.includes("DOS_EF")) {
      suggestions.push("High-DOS materials with strong electron-phonon coupling (e.g., hydrides, borides)");
    }
    if (theory.variables.includes("nesting_score")) {
      suggestions.push("Materials with Fermi surface nesting (e.g., iron pnictides, CDW hosts)");
    }
    if (theory.variables.includes("layeredness")) {
      suggestions.push("Layered structures with charge reservoir layers (e.g., cuprate variants)");
    }
  }

  let insight = "No significant theoretical patterns discovered yet.";
  if (topTheories.length > 0) {
    const best = topTheories[0];
    insight = `Best discovered relationship: ${best.simplified} (R2=${best.r2.toFixed(3)}, generalizes to ${best.applicableFamilies.length} families)`;
  }

  return {
    biasedVariables,
    suggestedCompositions: [...new Set(suggestions)],
    theoreticalInsight: insight,
  };
}

export interface SymbolicDiscoveryStats {
  totalRuns: number;
  totalEquationsEvaluated: number;
  theoriesDiscovered: number;
  topTheories: {
    id: string;
    equation: string;
    simplified: string;
    theoryScore: number;
    r2: number;
    mae: number;
    complexity: number;
    generalization: number;
    physicsCompliance: number;
    novelty: number;
    dimensionallyValid: boolean;
    variables: string[];
    applicableFamilies: string[];
  }[];
  featureLibrarySize: number;
  averageTheoryScore: number;
  bestTheoryScore: number;
  physicsVariablesUsed: string[];
  familyCoverage: Record<string, number>;
  unitRegistry: { variable: string; unit: string }[];
}

export function getSymbolicDiscoveryStats(): SymbolicDiscoveryStats {
  const library = buildFeatureLibrary();

  const familyCoverage: Record<string, number> = {};
  for (const theory of theoryDatabase) {
    for (const family of theory.applicableFamilies) {
      familyCoverage[family] = (familyCoverage[family] || 0) + 1;
    }
  }

  const allVars = new Set<string>();
  for (const theory of theoryDatabase) {
    for (const v of theory.variables) allVars.add(v);
  }

  const unitRegistry = Object.entries(VARIABLE_UNITS).map(([variable, spec]) => {
    const parts: string[] = [];
    if (spec.energy !== 0) parts.push(`energy^${spec.energy}`);
    if (spec.frequency !== 0) parts.push(`freq^${spec.frequency}`);
    if (spec.length !== 0) parts.push(`length^${spec.length}`);
    if (spec.temperature !== 0) parts.push(`temp^${spec.temperature}`);
    if (spec.pressure !== 0) parts.push(`pressure^${spec.pressure}`);
    if (spec.mass !== 0) parts.push(`mass^${spec.mass}`);
    if (spec.charge !== 0) parts.push(`charge^${spec.charge}`);
    return { variable, unit: parts.length > 0 ? parts.join(" * ") : "dimensionless" };
  });

  return {
    totalRuns: totalDiscoveryRuns,
    totalEquationsEvaluated,
    theoriesDiscovered: theoryDatabase.length,
    topTheories: theoryDatabase.filter(t => t.dimensionallyValid).slice(0, 10).map(t => ({
      id: t.id,
      equation: t.equation,
      simplified: t.simplified,
      theoryScore: Math.round(t.theoryScore * 1000) / 1000,
      r2: Math.round(t.r2 * 1000) / 1000,
      mae: Math.round(t.mae * 100) / 100,
      complexity: t.complexity,
      generalization: Math.round(t.generalization * 1000) / 1000,
      physicsCompliance: Math.round(t.physicsCompliance * 1000) / 1000,
      novelty: Math.round(t.novelty * 1000) / 1000,
      dimensionallyValid: t.dimensionallyValid,
      variables: t.variables,
      applicableFamilies: t.applicableFamilies,
    })),
    featureLibrarySize: library.length,
    averageTheoryScore: theoryDatabase.length > 0
      ? Math.round(theoryDatabase.reduce((s, t) => s + t.theoryScore, 0) / theoryDatabase.length * 1000) / 1000
      : 0,
    bestTheoryScore: theoryDatabase.length > 0
      ? Math.round(Math.max(...theoryDatabase.map(t => t.theoryScore)) * 1000) / 1000
      : 0,
    physicsVariablesUsed: Array.from(allVars),
    familyCoverage,
    unitRegistry,
  };
}

export function getTheoryDatabase(): DiscoveredTheory[] {
  return [...theoryDatabase];
}

export function getFeatureLibrary(): SymbolicTerm[] {
  return buildFeatureLibrary();
}

export { PHYSICS_VARIABLES, checkPhysicsConstraints as validatePhysicsConstraints };
