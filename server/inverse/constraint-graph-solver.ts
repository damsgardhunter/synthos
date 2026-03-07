import { solveConstraints } from "./constraint-solver";

export interface ConstraintGraphNode {
  id: string;
  parameter: string;
  value: number;
  range: { min: number; max: number };
  unit: string;
  feasibility: number;
  propagated: boolean;
  dependsOn: string[];
  influences: string[];
}

export interface ConstraintGraphEdge {
  from: string;
  to: string;
  relation: string;
  strength: number;
  equation: string;
}

export interface FeasibilityRegion {
  id: string;
  center: Record<string, number>;
  ranges: Record<string, { min: number; max: number }>;
  feasibilityScore: number;
  volume: number;
  elementSuggestions: string[];
  structureSuggestions: string[];
  regime: string;
  note: string;
}

export interface GraphSolution {
  targetTc: number;
  muStar: number;
  nodes: ConstraintGraphNode[];
  edges: ConstraintGraphEdge[];
  feasibilityRegions: FeasibilityRegion[];
  globalFeasibility: number;
  parameterCombinations: ParameterCombination[];
  rareRegions: RareRegion[];
  constraintSatisfactionMap: Record<string, boolean>;
  propagationSteps: PropagationStep[];
}

export interface ParameterCombination {
  rank: number;
  lambda: number;
  omegaLogK: number;
  dosEf: number;
  debyeTemp: number;
  phononSoftness: number;
  hopfieldEta: number;
  nestingScore: number;
  chargeTransfer: number;
  predictedTc: number;
  feasibility: number;
  elements: string[];
  structures: string[];
  regime: string;
}

export interface RareRegion {
  id: string;
  description: string;
  center: Record<string, number>;
  elements: string[];
  structures: string[];
  noveltyScore: number;
  feasibility: number;
  mechanisms: string[];
}

export interface PropagationStep {
  step: number;
  from: string;
  to: string;
  constraintApplied: string;
  beforeRange: { min: number; max: number };
  afterRange: { min: number; max: number };
  narrowingFactor: number;
}

function mcMillanTc(lambda: number, omegaLogK: number, muStar: number): number {
  const denom = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denom) < 1e-6 || denom <= 0) return 0;
  let f1: number;
  if (lambda < 1.5) {
    f1 = Math.pow(1 + lambda / (2.46 * (1 + 3.8 * muStar)), 1 / 3);
  } else {
    f1 = Math.sqrt(1 + lambda / 2.46);
  }
  const exponent = -1.04 * (1 + lambda) / denom;
  const tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
  return Number.isFinite(tc) && tc > 0 ? tc : 0;
}

function buildConstraintGraph(targetTc: number, muStar: number): { nodes: ConstraintGraphNode[]; edges: ConstraintGraphEdge[] } {
  const solution = solveConstraints(targetTc, muStar);

  const nodes: ConstraintGraphNode[] = [
    {
      id: "Tc",
      parameter: "Critical Temperature",
      value: targetTc,
      range: { min: targetTc * 0.9, max: targetTc * 1.1 },
      unit: "K",
      feasibility: 1.0,
      propagated: true,
      dependsOn: [],
      influences: ["lambda", "omegaLog"],
    },
    {
      id: "lambda",
      parameter: "Electron-Phonon Coupling",
      value: solution.requiredLambda.optimal,
      range: { min: solution.requiredLambda.min, max: solution.requiredLambda.max },
      unit: "",
      feasibility: solution.requiredCoupling.feasibility,
      propagated: false,
      dependsOn: ["Tc", "DOS", "phonon_softness"],
      influences: ["Tc", "gap_ratio"],
    },
    {
      id: "omegaLog",
      parameter: "Logarithmic Phonon Frequency",
      value: solution.requiredOmegaLog.optimal,
      range: { min: solution.requiredOmegaLog.min, max: solution.requiredOmegaLog.max },
      unit: "K",
      feasibility: solution.requiredPhonon.feasibility,
      propagated: false,
      dependsOn: ["element_mass", "bond_stiffness"],
      influences: ["Tc", "debye_temp"],
    },
    {
      id: "DOS",
      parameter: "Density of States at Ef",
      value: solution.requiredDOS.optimalDOS,
      range: { min: solution.requiredDOS.minDOS, max: solution.requiredDOS.maxDOS },
      unit: "states/eV",
      feasibility: solution.requiredDOS.feasibility,
      propagated: false,
      dependsOn: ["orbital_character", "structure", "nesting"],
      influences: ["lambda", "hopfield_eta"],
    },
    {
      id: "phonon_softness",
      parameter: "Phonon Softness",
      value: solution.requiredCoupling.requiredPhononSoftness,
      range: { min: solution.requiredCoupling.requiredPhononSoftness * 0.7, max: Math.min(1.0, solution.requiredCoupling.requiredPhononSoftness * 1.5) },
      unit: "",
      feasibility: solution.requiredPhonon.feasibility * 0.9,
      propagated: false,
      dependsOn: ["element_mass", "bond_stiffness", "pressure"],
      influences: ["lambda", "omegaLog"],
    },
    {
      id: "hopfield_eta",
      parameter: "Hopfield Parameter",
      value: solution.requiredCoupling.hopfieldParameter.optimal,
      range: { min: solution.requiredCoupling.hopfieldParameter.min, max: solution.requiredCoupling.hopfieldParameter.optimal * 1.5 },
      unit: "eV/A^2",
      feasibility: solution.requiredCoupling.feasibility,
      propagated: false,
      dependsOn: ["DOS", "orbital_character", "bond_stiffness"],
      influences: ["lambda"],
    },
    {
      id: "nesting",
      parameter: "Fermi Surface Nesting",
      value: targetTc > 100 ? 0.7 : 0.4,
      range: { min: 0.1, max: 1.0 },
      unit: "",
      feasibility: 0.7,
      propagated: false,
      dependsOn: ["structure", "DOS"],
      influences: ["DOS", "charge_transfer"],
    },
    {
      id: "charge_transfer",
      parameter: "Charge Transfer",
      value: solution.chargeTransfer.deltaCharge.optimal,
      range: { min: solution.chargeTransfer.deltaCharge.min, max: solution.chargeTransfer.deltaCharge.optimal * 2 },
      unit: "e",
      feasibility: solution.chargeTransfer.feasibility,
      propagated: false,
      dependsOn: ["structure", "nesting"],
      influences: ["DOS", "lambda"],
    },
    {
      id: "structure",
      parameter: "Crystal Structure",
      value: solution.structuralTargets.length,
      range: { min: 1, max: solution.structuralTargets.length + 2 },
      unit: "types",
      feasibility: solution.structuralTargets.length > 0 ? 0.8 : 0.5,
      propagated: false,
      dependsOn: ["pressure", "element_mass"],
      influences: ["DOS", "nesting", "phonon_softness", "charge_transfer"],
    },
    {
      id: "pressure",
      parameter: "Pressure",
      value: 0,
      range: { min: 0, max: 300 },
      unit: "GPa",
      feasibility: 1.0,
      propagated: true,
      dependsOn: [],
      influences: ["structure", "phonon_softness", "bond_stiffness"],
    },
    {
      id: "element_mass",
      parameter: "Average Atomic Mass",
      value: solution.requiredPhonon.elementMassConstraints.maxAvgMass,
      range: { min: 1, max: solution.requiredPhonon.elementMassConstraints.maxAvgMass },
      unit: "amu",
      feasibility: solution.requiredPhonon.feasibility,
      propagated: false,
      dependsOn: [],
      influences: ["omegaLog", "phonon_softness", "debye_temp"],
    },
    {
      id: "bond_stiffness",
      parameter: "Bond Stiffness",
      value: targetTc > 100 ? 400 : 200,
      range: { min: 50, max: 800 },
      unit: "N/m",
      feasibility: 0.8,
      propagated: false,
      dependsOn: ["element_mass", "pressure"],
      influences: ["omegaLog", "phonon_softness", "hopfield_eta"],
    },
    {
      id: "orbital_character",
      parameter: "Orbital Character",
      value: targetTc > 100 ? 0.8 : 0.5,
      range: { min: 0.3, max: 1.0 },
      unit: "",
      feasibility: 0.75,
      propagated: false,
      dependsOn: ["structure"],
      influences: ["DOS", "hopfield_eta"],
    },
    {
      id: "debye_temp",
      parameter: "Debye Temperature",
      value: (solution.requiredPhonon.debyeTempRange.min + solution.requiredPhonon.debyeTempRange.max) / 2,
      range: solution.requiredPhonon.debyeTempRange,
      unit: "K",
      feasibility: solution.requiredPhonon.feasibility,
      propagated: false,
      dependsOn: ["element_mass", "bond_stiffness", "omegaLog"],
      influences: ["omegaLog"],
    },
    {
      id: "gap_ratio",
      parameter: "Gap Ratio (2Δ/kTc)",
      value: solution.requiredLambda.optimal > 1.5 ? 4.5 : 3.53,
      range: { min: 3.0, max: 6.0 },
      unit: "",
      feasibility: 0.9,
      propagated: false,
      dependsOn: ["lambda"],
      influences: [],
    },
  ];

  const edges: ConstraintGraphEdge[] = [
    { from: "Tc", to: "lambda", relation: "determines", strength: 1.0, equation: "Tc = (ωlog/1.2) * f1(λ) * exp(-1.04(1+λ)/(λ-μ*(1+0.62λ)))" },
    { from: "Tc", to: "omegaLog", relation: "determines", strength: 1.0, equation: "McMillan-Allen-Dynes equation" },
    { from: "lambda", to: "Tc", relation: "determines", strength: 1.0, equation: "Allen-Dynes modified McMillan" },
    { from: "omegaLog", to: "Tc", relation: "determines", strength: 0.9, equation: "Tc ∝ ωlog * exp(-1.04(1+λ)/...)" },
    { from: "DOS", to: "lambda", relation: "contributes", strength: 0.8, equation: "λ = N(Ef) * <I²> / (M * <ω²>)" },
    { from: "phonon_softness", to: "lambda", relation: "enhances", strength: 0.7, equation: "λ ∝ 1/(<ω²>)" },
    { from: "element_mass", to: "omegaLog", relation: "inversely_scales", strength: 0.9, equation: "ω ∝ 1/√M" },
    { from: "bond_stiffness", to: "omegaLog", relation: "scales", strength: 0.8, equation: "ω ∝ √(k/M)" },
    { from: "structure", to: "DOS", relation: "determines", strength: 0.7, equation: "Band structure → DOS(Ef)" },
    { from: "structure", to: "nesting", relation: "determines", strength: 0.6, equation: "FS topology → nesting vectors" },
    { from: "nesting", to: "DOS", relation: "enhances", strength: 0.5, equation: "Nesting → VHS → DOS peak" },
    { from: "nesting", to: "charge_transfer", relation: "enables", strength: 0.4, equation: "Nesting → charge instability" },
    { from: "charge_transfer", to: "DOS", relation: "modulates", strength: 0.5, equation: "Doping → Ef shift → DOS change" },
    { from: "charge_transfer", to: "lambda", relation: "enhances", strength: 0.3, equation: "Charge transfer → coupling enhancement" },
    { from: "orbital_character", to: "DOS", relation: "determines", strength: 0.6, equation: "d/f-orbital → higher DOS(Ef)" },
    { from: "orbital_character", to: "hopfield_eta", relation: "determines", strength: 0.7, equation: "η = N(Ef) * <I²>" },
    { from: "DOS", to: "hopfield_eta", relation: "scales", strength: 0.8, equation: "η ∝ N(Ef)" },
    { from: "hopfield_eta", to: "lambda", relation: "determines", strength: 0.9, equation: "λ = η / (M * <ω²>)" },
    { from: "pressure", to: "structure", relation: "transforms", strength: 0.6, equation: "P → phase transitions" },
    { from: "pressure", to: "phonon_softness", relation: "modulates", strength: 0.5, equation: "P → phonon hardening/softening" },
    { from: "pressure", to: "bond_stiffness", relation: "increases", strength: 0.7, equation: "P → k increases" },
    { from: "element_mass", to: "phonon_softness", relation: "inversely_scales", strength: 0.8, equation: "Light atoms → harder phonons" },
    { from: "element_mass", to: "debye_temp", relation: "inversely_scales", strength: 0.9, equation: "θD ∝ √(k/M)" },
    { from: "bond_stiffness", to: "debye_temp", relation: "scales", strength: 0.8, equation: "θD ∝ √(k/M)" },
    { from: "bond_stiffness", to: "hopfield_eta", relation: "contributes", strength: 0.5, equation: "Stiff bonds → stronger coupling matrix" },
    { from: "lambda", to: "gap_ratio", relation: "determines", strength: 0.7, equation: "2Δ/kTc = 3.53(1 + 12.5(Tc/ωlog)²·ln(ωlog/2Tc))" },
    { from: "debye_temp", to: "omegaLog", relation: "correlates", strength: 0.7, equation: "ωlog ≈ 0.65 * θD (Debye)" },
    { from: "structure", to: "charge_transfer", relation: "enables", strength: 0.5, equation: "Layered → charge reservoir" },
    { from: "structure", to: "phonon_softness", relation: "modulates", strength: 0.4, equation: "Structure → phonon spectrum" },
  ];

  return { nodes, edges };
}

function propagateConstraints(
  nodes: ConstraintGraphNode[],
  edges: ConstraintGraphEdge[],
  targetTc: number,
  muStar: number,
): PropagationStep[] {
  const steps: PropagationStep[] = [];
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const propagationQueue = ["Tc"];
  const visited = new Set<string>();
  let stepCount = 0;

  while (propagationQueue.length > 0 && stepCount < 50) {
    const currentId = propagationQueue.shift()!;
    if (visited.has(currentId)) continue;
    visited.add(currentId);

    const currentNode = nodeMap.get(currentId);
    if (!currentNode) continue;
    currentNode.propagated = true;

    const outEdges = edges.filter(e => e.from === currentId);
    for (const edge of outEdges) {
      const targetNode = nodeMap.get(edge.to);
      if (!targetNode || targetNode.propagated) continue;

      const beforeRange = { ...targetNode.range };
      narrowRange(currentNode, targetNode, edge, targetTc, muStar);
      const afterRange = { ...targetNode.range };

      const rangeWidth = beforeRange.max - beforeRange.min;
      const newWidth = afterRange.max - afterRange.min;
      const narrowing = rangeWidth > 0 ? newWidth / rangeWidth : 1;

      stepCount++;
      steps.push({
        step: stepCount,
        from: currentId,
        to: edge.to,
        constraintApplied: edge.equation,
        beforeRange,
        afterRange,
        narrowingFactor: Math.round(narrowing * 1000) / 1000,
      });

      if (!propagationQueue.includes(edge.to)) {
        propagationQueue.push(edge.to);
      }
    }
  }

  return steps;
}

function narrowRange(
  source: ConstraintGraphNode,
  target: ConstraintGraphNode,
  edge: ConstraintGraphEdge,
  _targetTc: number,
  _muStar: number,
): void {
  const strength = edge.strength;

  switch (edge.relation) {
    case "determines":
    case "contributes":
    case "enhances": {
      const center = target.value;
      const halfWidth = (target.range.max - target.range.min) / 2;
      const narrowedHalf = halfWidth * (1 - strength * 0.3);
      target.range.min = Math.max(target.range.min, center - narrowedHalf);
      target.range.max = Math.min(target.range.max, center + narrowedHalf);
      break;
    }
    case "inversely_scales": {
      if (source.value > 0) {
        const factor = 1 / Math.sqrt(source.value);
        const scale = factor * 100;
        target.range.min = Math.max(target.range.min, target.range.min * (1 + strength * 0.1));
        target.range.max = Math.min(target.range.max, target.range.max * (1 - strength * 0.05));
        if (target.range.min > target.range.max) {
          const mid = (target.range.min + target.range.max) / 2;
          target.range.min = mid * 0.95;
          target.range.max = mid * 1.05;
        }
      }
      break;
    }
    case "scales": {
      const center = target.value;
      const halfWidth = (target.range.max - target.range.min) / 2;
      const narrowedHalf = halfWidth * (1 - strength * 0.2);
      target.range.min = Math.max(target.range.min, center - narrowedHalf);
      target.range.max = Math.min(target.range.max, center + narrowedHalf);
      break;
    }
    case "modulates":
    case "enables":
    case "transforms":
    case "correlates": {
      const center = target.value;
      const halfWidth = (target.range.max - target.range.min) / 2;
      const narrowedHalf = halfWidth * (1 - strength * 0.15);
      target.range.min = Math.max(target.range.min, center - narrowedHalf);
      target.range.max = Math.min(target.range.max, center + narrowedHalf);
      break;
    }
    default: {
      break;
    }
  }

  if (target.range.min > target.range.max) {
    const mid = (target.range.min + target.range.max) / 2;
    target.range.min = mid * 0.98;
    target.range.max = mid * 1.02;
  }
}

const REGIME_CONFIGS: {
  name: string;
  lambdaRange: [number, number];
  omegaRange: [number, number];
  elements: string[];
  structures: string[];
  pressureRange: [number, number];
  mechanisms: string[];
}[] = [
  {
    name: "conventional-BCS",
    lambdaRange: [0.3, 1.2],
    omegaRange: [100, 500],
    elements: ["Nb", "V", "Ta", "Sn", "Pb", "Mo", "W"],
    structures: ["A15", "BCC", "HCP"],
    pressureRange: [0, 30],
    mechanisms: ["s-wave phonon-mediated"],
  },
  {
    name: "strong-coupling-metal",
    lambdaRange: [1.0, 2.0],
    omegaRange: [200, 800],
    elements: ["Nb", "V", "B", "C", "Mg", "Ti", "Zr"],
    structures: ["layered-hexagonal", "A15", "NaCl-rocksalt"],
    pressureRange: [0, 50],
    mechanisms: ["strong s-wave", "multi-band"],
  },
  {
    name: "light-element-compound",
    lambdaRange: [0.8, 2.5],
    omegaRange: [500, 1200],
    elements: ["B", "C", "N", "Mg", "Al", "Si", "Ca"],
    structures: ["layered-hexagonal", "honeycomb", "diamond"],
    pressureRange: [0, 80],
    mechanisms: ["high-frequency phonon", "sigma-band coupling"],
  },
  {
    name: "hydride-moderate-pressure",
    lambdaRange: [1.5, 3.0],
    omegaRange: [800, 1500],
    elements: ["H", "La", "Y", "Ca", "Sr", "Ba", "Th", "Ce"],
    structures: ["clathrate-cage", "sodalite-cage", "layered-hydride"],
    pressureRange: [50, 200],
    mechanisms: ["metallic hydrogen", "cage-phonon coupling"],
  },
  {
    name: "superhydride",
    lambdaRange: [2.0, 4.0],
    omegaRange: [1000, 2500],
    elements: ["H", "La", "Y", "Ca", "Ce", "Th", "Ac"],
    structures: ["clathrate-cage", "sodalite-cage", "fcc-like"],
    pressureRange: [100, 350],
    mechanisms: ["metallic hydrogen network", "quantum lattice"],
  },
  {
    name: "unconventional-layered",
    lambdaRange: [0.5, 1.5],
    omegaRange: [100, 600],
    elements: ["Cu", "Fe", "Ni", "La", "Y", "Ba", "Sr", "As", "Se", "O"],
    structures: ["cuprate-layered", "FeAs-layer", "NiO2-layer"],
    pressureRange: [0, 30],
    mechanisms: ["spin-fluctuation", "charge-transfer", "d-wave"],
  },
  {
    name: "kagome-flat-band",
    lambdaRange: [0.6, 2.0],
    omegaRange: [200, 800],
    elements: ["V", "Nb", "Ti", "Co", "Mn", "Sn", "Sb", "Ge"],
    structures: ["kagome-flat", "breathing-kagome", "pyrochlore"],
    pressureRange: [0, 50],
    mechanisms: ["flat-band enhancement", "VHS proximity", "nesting"],
  },
  {
    name: "topological-SC",
    lambdaRange: [0.5, 1.8],
    omegaRange: [150, 700],
    elements: ["Bi", "Sb", "Te", "Se", "Sn", "Pb", "Pt", "Ir"],
    structures: ["topological-insulator-surface", "Weyl", "Dirac"],
    pressureRange: [0, 50],
    mechanisms: ["topological surface state", "band inversion", "p-wave proxy"],
  },
];

function computeFeasibilityRegions(targetTc: number, muStar: number): FeasibilityRegion[] {
  const regions: FeasibilityRegion[] = [];

  for (const regime of REGIME_CONFIGS) {
    const lambdaSamples = 5;
    const omegaSamples = 5;
    let validCount = 0;
    let totalSampled = 0;
    let sumFeasibility = 0;

    const validLambdas: number[] = [];
    const validOmegas: number[] = [];

    for (let li = 0; li < lambdaSamples; li++) {
      const lambda = regime.lambdaRange[0] + (regime.lambdaRange[1] - regime.lambdaRange[0]) * li / (lambdaSamples - 1);
      for (let oi = 0; oi < omegaSamples; oi++) {
        const omega = regime.omegaRange[0] + (regime.omegaRange[1] - regime.omegaRange[0]) * oi / (omegaSamples - 1);
        totalSampled++;
        const tc = mcMillanTc(lambda, omega, muStar);
        if (tc >= targetTc * 0.7 && tc <= targetTc * 1.5) {
          validCount++;
          validLambdas.push(lambda);
          validOmegas.push(omega);

          let feas = 1.0;
          if (lambda > 3.0) feas *= 0.3;
          else if (lambda > 2.0) feas *= 0.6;
          if (omega > 1500) feas *= 0.4;
          else if (omega > 800) feas *= 0.7;
          if (regime.pressureRange[0] > 100) feas *= 0.5;
          sumFeasibility += feas;
        }
      }
    }

    if (validCount === 0) continue;

    const avgFeasibility = sumFeasibility / validCount;
    const lambdaCenter = validLambdas.reduce((a, b) => a + b, 0) / validLambdas.length;
    const omegaCenter = validOmegas.reduce((a, b) => a + b, 0) / validOmegas.length;
    const lambdaMin = Math.min(...validLambdas);
    const lambdaMax = Math.max(...validLambdas);
    const omegaMin = Math.min(...validOmegas);
    const omegaMax = Math.max(...validOmegas);

    const lambdaWidth = lambdaMax - lambdaMin;
    const omegaWidth = omegaMax - omegaMin;
    const volume = Math.max(0.01, lambdaWidth * omegaWidth);

    regions.push({
      id: `region-${regime.name}`,
      center: {
        lambda: Math.round(lambdaCenter * 1000) / 1000,
        omegaLogK: Math.round(omegaCenter),
        predictedTc: Math.round(mcMillanTc(lambdaCenter, omegaCenter, muStar) * 10) / 10,
      },
      ranges: {
        lambda: { min: Math.round(lambdaMin * 100) / 100, max: Math.round(lambdaMax * 100) / 100 },
        omegaLogK: { min: Math.round(omegaMin), max: Math.round(omegaMax) },
        pressure: { min: regime.pressureRange[0], max: regime.pressureRange[1] },
      },
      feasibilityScore: Math.round(avgFeasibility * 1000) / 1000,
      volume: Math.round(volume * 100) / 100,
      elementSuggestions: regime.elements,
      structureSuggestions: regime.structures,
      regime: regime.name,
      note: `${regime.mechanisms.join(", ")} — ${validCount}/${totalSampled} parameter samples reach Tc≈${targetTc}K`,
    });
  }

  regions.sort((a, b) => b.feasibilityScore - a.feasibilityScore);
  return regions;
}

function generateParameterCombinations(
  targetTc: number,
  muStar: number,
  regions: FeasibilityRegion[],
): ParameterCombination[] {
  const combinations: ParameterCombination[] = [];

  for (const region of regions) {
    const regime = REGIME_CONFIGS.find(r => r.name === region.regime);
    if (!regime) continue;

    const lambdaCenter = region.center.lambda as number;
    const omegaCenter = region.center.omegaLogK as number;

    const offsets = [
      { dl: 0, do_: 0 },
      { dl: -0.15, do_: 50 },
      { dl: 0.15, do_: -50 },
      { dl: -0.3, do_: 100 },
      { dl: 0.3, do_: -100 },
    ];

    for (const offset of offsets) {
      const lambda = lambdaCenter + offset.dl;
      const omega = omegaCenter + offset.do_;

      if (lambda < regime.lambdaRange[0] || lambda > regime.lambdaRange[1]) continue;
      if (omega < regime.omegaRange[0] || omega > regime.omegaRange[1]) continue;

      const tc = mcMillanTc(lambda, omega, muStar);
      if (tc < targetTc * 0.5 || tc > targetTc * 2.0) continue;

      const dosEstimate = lambda > 1.5 ? 3 + lambda * 1.5 : 1.5 + lambda * 2;
      const debyeEstimate = Math.round(omega * 1.5);
      const phononSoftness = Math.min(1.0, lambda / (dosEstimate * 0.5));
      const hopfieldEstimate = lambda * omega * 0.01;
      const nestingEstimate = regime.name.includes("kagome") || regime.name.includes("layered") ? 0.7 : 0.4;
      const chargeTransferEstimate = regime.name.includes("unconventional") ? 0.2 : 0;

      let feasibility = region.feasibilityScore;
      const tcDiff = Math.abs(tc - targetTc) / targetTc;
      feasibility *= Math.max(0.3, 1 - tcDiff);

      combinations.push({
        rank: 0,
        lambda: Math.round(lambda * 1000) / 1000,
        omegaLogK: Math.round(omega),
        dosEf: Math.round(dosEstimate * 100) / 100,
        debyeTemp: debyeEstimate,
        phononSoftness: Math.round(phononSoftness * 1000) / 1000,
        hopfieldEta: Math.round(hopfieldEstimate * 100) / 100,
        nestingScore: nestingEstimate,
        chargeTransfer: chargeTransferEstimate,
        predictedTc: Math.round(tc * 10) / 10,
        feasibility: Math.round(feasibility * 1000) / 1000,
        elements: regime.elements.slice(0, 5),
        structures: regime.structures.slice(0, 3),
        regime: regime.name,
      });
    }
  }

  combinations.sort((a, b) => {
    const scoreA = a.feasibility * 0.5 + (1 - Math.abs(a.predictedTc - targetTc) / targetTc) * 0.5;
    const scoreB = b.feasibility * 0.5 + (1 - Math.abs(b.predictedTc - targetTc) / targetTc) * 0.5;
    return scoreB - scoreA;
  });

  return combinations.slice(0, 30).map((c, i) => ({ ...c, rank: i + 1 }));
}

function searchRareRegions(targetTc: number, muStar: number): RareRegion[] {
  const rareRegions: RareRegion[] = [];

  const rareConfigs: {
    id: string;
    description: string;
    lambda: number;
    omega: number;
    elements: string[];
    structures: string[];
    mechanisms: string[];
    novelty: number;
  }[] = [
    {
      id: "rare-ternary-hydride",
      description: "Ternary hydride with mixed-cation cage stabilization",
      lambda: 2.2,
      omega: 1200,
      elements: ["H", "La", "Y", "Mg", "Ca"],
      structures: ["ternary-clathrate", "mixed-cage"],
      mechanisms: ["multi-cation stabilization", "phonon hybridization"],
      novelty: 0.85,
    },
    {
      id: "rare-borohydride",
      description: "Metal borohydride with coupled B-H and metal phonon modes",
      lambda: 1.8,
      omega: 900,
      elements: ["H", "B", "Mg", "Ca", "Na", "Li"],
      structures: ["borohydride-cage", "layered-borohydride"],
      mechanisms: ["B-H stretching coupling", "cage-metal interaction"],
      novelty: 0.9,
    },
    {
      id: "rare-nickelate",
      description: "Infinite-layer nickelate with cuprate-analog physics",
      lambda: 1.0,
      omega: 400,
      elements: ["Ni", "Nd", "La", "Sr", "O"],
      structures: ["infinite-layer", "NiO2-plane"],
      mechanisms: ["d-wave analog", "charge-transfer", "self-doping"],
      novelty: 0.88,
    },
    {
      id: "rare-kagome-vhs",
      description: "Kagome metal near van Hove filling with flat bands",
      lambda: 1.3,
      omega: 350,
      elements: ["V", "Nb", "Ti", "Sn", "Sb", "Ge"],
      structures: ["kagome-flat", "breathing-kagome"],
      mechanisms: ["VHS-driven coupling", "flat-band SC", "nesting-enhanced"],
      novelty: 0.82,
    },
    {
      id: "rare-heavy-fermion-sc",
      description: "Heavy-fermion compound near quantum critical point",
      lambda: 0.8,
      omega: 150,
      elements: ["Ce", "U", "Yb", "Ir", "Rh", "Co", "Si", "Ge"],
      structures: ["ThCr2Si2", "CeCu2Si2-type"],
      mechanisms: ["spin-fluctuation", "quantum criticality", "Kondo breakdown"],
      novelty: 0.75,
    },
    {
      id: "rare-topological-hetero",
      description: "Topological insulator heterostructure with proximity-induced SC",
      lambda: 0.9,
      omega: 300,
      elements: ["Bi", "Se", "Te", "Nb", "Pb", "Sn"],
      structures: ["heterostructure", "topological-surface"],
      mechanisms: ["proximity effect", "Majorana", "topological SC"],
      novelty: 0.92,
    },
    {
      id: "rare-high-entropy",
      description: "High-entropy alloy with cocktail-effect phonon engineering",
      lambda: 1.4,
      omega: 450,
      elements: ["Nb", "Ti", "Zr", "Hf", "Ta", "Mo", "V"],
      structures: ["BCC-HEA", "FCC-HEA"],
      mechanisms: ["phonon softening", "electron scattering", "cocktail effect"],
      novelty: 0.8,
    },
    {
      id: "rare-carbon-intercalated",
      description: "Intercalated carbon structure with graphene-like bands",
      lambda: 1.1,
      omega: 700,
      elements: ["C", "Ca", "Li", "K", "Rb", "Mg"],
      structures: ["intercalated-graphite", "C60-fulleride"],
      mechanisms: ["sigma-band", "intercalation doping", "phonon hardening"],
      novelty: 0.78,
    },
  ];

  for (const config of rareConfigs) {
    const tc = mcMillanTc(config.lambda, config.omega, muStar);
    if (tc < targetTc * 0.3) continue;

    const tcProximity = 1 - Math.min(1, Math.abs(tc - targetTc) / targetTc);
    let feasibility = 0.5;
    if (config.lambda > 2.5) feasibility *= 0.4;
    else if (config.lambda > 1.5) feasibility *= 0.7;
    if (config.omega > 1200) feasibility *= 0.5;
    feasibility *= (0.5 + 0.5 * tcProximity);

    rareRegions.push({
      id: config.id,
      description: config.description,
      center: {
        lambda: config.lambda,
        omegaLogK: config.omega,
        predictedTc: Math.round(tc * 10) / 10,
      },
      elements: config.elements,
      structures: config.structures,
      noveltyScore: config.novelty,
      feasibility: Math.round(feasibility * 1000) / 1000,
      mechanisms: config.mechanisms,
    });
  }

  rareRegions.sort((a, b) => {
    const scoreA = a.noveltyScore * 0.4 + a.feasibility * 0.6;
    const scoreB = b.noveltyScore * 0.4 + b.feasibility * 0.6;
    return scoreB - scoreA;
  });

  return rareRegions;
}

export function solveConstraintGraph(targetTc: number, muStar: number = 0.10): GraphSolution {
  const muStarClamped = Math.max(0.08, Math.min(0.20, muStar));
  const { nodes, edges } = buildConstraintGraph(targetTc, muStarClamped);
  const propagationSteps = propagateConstraints(nodes, edges, targetTc, muStarClamped);

  const feasibilityRegions = computeFeasibilityRegions(targetTc, muStarClamped);
  const parameterCombinations = generateParameterCombinations(targetTc, muStarClamped, feasibilityRegions);
  const rareRegions = searchRareRegions(targetTc, muStarClamped);

  const constraintSatisfactionMap: Record<string, boolean> = {};
  for (const node of nodes) {
    constraintSatisfactionMap[node.id] = node.feasibility > 0.4;
  }

  const feasibleNodeCount = nodes.filter(n => n.feasibility > 0.4).length;
  const globalFeasibility = Math.round((feasibleNodeCount / nodes.length) * 1000) / 1000;

  return {
    targetTc,
    muStar: muStarClamped,
    nodes,
    edges,
    feasibilityRegions,
    globalFeasibility,
    parameterCombinations,
    rareRegions,
    constraintSatisfactionMap,
    propagationSteps,
  };
}

export function getFeasibleRegions(targetTc: number = 200, muStar: number = 0.10): {
  regions: FeasibilityRegion[];
  rareRegions: RareRegion[];
  totalRegions: number;
  bestRegime: string;
  bestFeasibility: number;
  parameterSpace: {
    lambdaBounds: { min: number; max: number };
    omegaLogBounds: { min: number; max: number };
    pressureBounds: { min: number; max: number };
  };
} {
  const muStarClamped = Math.max(0.08, Math.min(0.20, muStar));
  const regions = computeFeasibilityRegions(targetTc, muStarClamped);
  const rareRegions = searchRareRegions(targetTc, muStarClamped);

  const allLambdaMins = regions.map(r => r.ranges.lambda.min);
  const allLambdaMaxs = regions.map(r => r.ranges.lambda.max);
  const allOmegaMins = regions.map(r => r.ranges.omegaLogK.min);
  const allOmegaMaxs = regions.map(r => r.ranges.omegaLogK.max);
  const allPressureMins = regions.map(r => r.ranges.pressure.min);
  const allPressureMaxs = regions.map(r => r.ranges.pressure.max);

  const best = regions[0];

  return {
    regions,
    rareRegions,
    totalRegions: regions.length,
    bestRegime: best?.regime ?? "none",
    bestFeasibility: best?.feasibilityScore ?? 0,
    parameterSpace: {
      lambdaBounds: {
        min: allLambdaMins.length > 0 ? Math.min(...allLambdaMins) : 0,
        max: allLambdaMaxs.length > 0 ? Math.max(...allLambdaMaxs) : 0,
      },
      omegaLogBounds: {
        min: allOmegaMins.length > 0 ? Math.min(...allOmegaMins) : 0,
        max: allOmegaMaxs.length > 0 ? Math.max(...allOmegaMaxs) : 0,
      },
      pressureBounds: {
        min: allPressureMins.length > 0 ? Math.min(...allPressureMins) : 0,
        max: allPressureMaxs.length > 0 ? Math.max(...allPressureMaxs) : 0,
      },
    },
  };
}

export function getConstraintGraphGuidance(targetTc: number): {
  topRegimes: string[];
  preferredElements: string[];
  preferredStructures: string[];
  feasibility: number;
  rareOpportunities: { id: string; description: string; novelty: number }[];
} {
  const solution = solveConstraintGraph(targetTc);

  const topRegimes = solution.feasibilityRegions.slice(0, 3).map(r => r.regime);
  const allElements = new Set<string>();
  const allStructures = new Set<string>();

  for (const combo of solution.parameterCombinations.slice(0, 10)) {
    for (const el of combo.elements) allElements.add(el);
    for (const st of combo.structures) allStructures.add(st);
  }

  const rareOpportunities = solution.rareRegions.slice(0, 3).map(r => ({
    id: r.id,
    description: r.description,
    novelty: r.noveltyScore,
  }));

  return {
    topRegimes,
    preferredElements: Array.from(allElements).slice(0, 10),
    preferredStructures: Array.from(allStructures).slice(0, 6),
    feasibility: solution.globalFeasibility,
    rareOpportunities,
  };
}
