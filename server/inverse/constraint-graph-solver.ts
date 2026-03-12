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

function mcMillanTc(lambda: number, omegaLogK: number, muStar: number, omega2K?: number): number {
  if (omegaLogK <= 0) return 0;
  const denom = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denom) < 1e-6 || denom <= 0) return 0;
  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);
  const w2 = omega2K && omega2K > 0 ? omega2K : omegaLogK * 1.3;
  const omega2Ratio = Math.max(1.0, w2 / omegaLogK);
  const lambda2 = 2.46 * (1 + 3.8 * muStar) * Math.sqrt(omega2Ratio);
  const f2 = 1 + (lambda * lambda / (lambda * lambda + lambda2 * lambda2)) * (omega2Ratio - 1) * 0.15;
  const exponent = -1.04 * (1 + lambda) / denom;
  if (exponent < -50) return 0;
  const tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);
  return Number.isFinite(tc) && tc > 0 ? tc : 0;
}

function computeCarbotteGapRatio(tcK: number, omegaLogK: number): number {
  if (omegaLogK <= 0 || tcK <= 0) return 3.53;
  if (omegaLogK <= 2 * tcK) return 3.53;
  const ratio = Math.min(0.5, tcK / omegaLogK);
  const logTerm = Math.log(omegaLogK / (2 * tcK));
  const correction = 12.5 * ratio * ratio * logTerm;
  const gapRatio = 3.53 * (1 + correction);
  return Math.max(3.53, Math.min(6.0, gapRatio));
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
      dependsOn: ["element_mass", "bond_stiffness", "pressure", "structural_stability"],
      influences: ["lambda", "omegaLog", "structural_stability"],
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
      feasibility: 1.0 / (1.0 + 0 / 100),
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
      influences: ["omegaLog", "phonon_softness", "debye_temp", "Tc"],
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
      value: computeCarbotteGapRatio(targetTc, solution.requiredOmegaLog.optimal),
      range: { min: 3.0, max: 6.0 },
      unit: "",
      feasibility: 0.9,
      propagated: false,
      dependsOn: ["lambda", "Tc", "omegaLog"],
      influences: [],
    },
    {
      id: "structural_stability",
      parameter: "Mechanical Stability",
      value: 1.0,
      range: { min: 0.3, max: 1.0 },
      unit: "",
      feasibility: 0.85,
      propagated: false,
      dependsOn: ["pressure", "phonon_softness"],
      influences: ["phonon_softness"],
    },
  ];

  const edges: ConstraintGraphEdge[] = [
    { from: "Tc", to: "lambda", relation: "determines", strength: 1.0, equation: "Tc = (ωlog/1.2) * f1(λ) * f2(λ,ω₂/ωlog) * exp(-1.04(1+λ)/(λ-μ*(1+0.62λ)))" },
    { from: "Tc", to: "omegaLog", relation: "determines", strength: 1.0, equation: "Allen-Dynes: Tc = (ωlog/1.2) * f1·f2 * exp(...)" },
    { from: "lambda", to: "Tc", relation: "determines", strength: 1.0, equation: "Allen-Dynes with f1·f2 strong-coupling + shape corrections" },
    { from: "omegaLog", to: "Tc", relation: "determines", strength: 0.9, equation: "Tc ∝ ωlog * f1·f2 * exp(-1.04(1+λ)/...)" },
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
    { from: "pressure", to: "bond_stiffness", relation: "scales", strength: 0.7, equation: "P → k increases" },
    { from: "element_mass", to: "phonon_softness", relation: "inversely_scales", strength: 0.8, equation: "Light atoms → harder phonons" },
    { from: "element_mass", to: "debye_temp", relation: "inversely_scales", strength: 0.9, equation: "θD ∝ √(k/M)" },
    { from: "bond_stiffness", to: "debye_temp", relation: "scales", strength: 0.8, equation: "θD ∝ √(k/M)" },
    { from: "bond_stiffness", to: "hopfield_eta", relation: "contributes", strength: 0.5, equation: "Stiff bonds → stronger coupling matrix" },
    { from: "lambda", to: "gap_ratio", relation: "determines", strength: 0.7, equation: "2Δ/kTc = 3.53(1 + 12.5(Tc/ωlog)²·ln(ωlog/2Tc))" },
    { from: "debye_temp", to: "omegaLog", relation: "correlates", strength: 0.7, equation: "ωlog ≈ 0.65 * θD (Debye)" },
    { from: "structure", to: "charge_transfer", relation: "enables", strength: 0.5, equation: "Layered → charge reservoir" },
    { from: "structure", to: "phonon_softness", relation: "modulates", strength: 0.4, equation: "Structure → phonon spectrum" },
    { from: "element_mass", to: "Tc", relation: "inversely_scales", strength: 0.7, equation: "Tc ∝ M^(-α), isotope effect α ≈ 0.5" },
    { from: "structural_stability", to: "phonon_softness", relation: "limits", strength: 0.9, equation: "Stability caps softness: collapse if softness > stability threshold" },
    { from: "phonon_softness", to: "structural_stability", relation: "inversely_scales", strength: 0.8, equation: "High softness → low stability margin" },
    { from: "pressure", to: "structural_stability", relation: "modulates", strength: 0.6, equation: "P → stability changes near phase boundaries" },
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
  const propagationQueue = ["Tc", "pressure", "element_mass"];
  const visited = new Set<string>();
  let stepCount = 0;
  const feedbackRelations = new Set(["limits", "inversely_scales"]);

  while (propagationQueue.length > 0 && stepCount < 80) {
    const currentId = propagationQueue.shift()!;
    if (visited.has(currentId)) continue;
    visited.add(currentId);

    const currentNode = nodeMap.get(currentId);
    if (!currentNode) continue;
    currentNode.propagated = true;

    const outEdges = edges.filter(e => e.from === currentId);
    for (const edge of outEdges) {
      const targetNode = nodeMap.get(edge.to);
      if (!targetNode) continue;
      if (targetNode.propagated && !feedbackRelations.has(edge.relation)) continue;

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

      if (!propagationQueue.includes(edge.to) && !visited.has(edge.to)) {
        propagationQueue.push(edge.to);
      }
    }
  }

  return steps;
}

function estimateTransitionPressureRatio(pRange: { min: number; max: number }): { peakRatio: number; width: number } {
  const pSpan = pRange.max - pRange.min;
  if (pSpan <= 0) return { peakRatio: 0.5, width: 0.2 };
  if (pRange.max <= 50) return { peakRatio: 0.7, width: 0.25 };
  if (pRange.min >= 100) return { peakRatio: 0.45, width: 0.12 };
  if (pRange.min >= 50) return { peakRatio: 0.55, width: 0.15 };
  return { peakRatio: 0.5, width: 0.2 };
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
        if (source.id === "phonon_softness" && target.id === "structural_stability") {
          const softness = source.value;
          const stability = Math.max(0.1, 1.0 - softness * 0.8);
          target.value = stability;
          const halfW = (target.range.max - target.range.min) / 2 * (1 - strength * 0.2);
          target.range.min = Math.max(target.range.min, stability - halfW);
          target.range.max = Math.min(target.range.max, stability + halfW);
        } else if (source.id === "element_mass" && target.id === "omegaLog") {
          const massVal = source.value;
          const isotopeBoost = 1 / Math.sqrt(Math.max(1, massVal));
          const boostedValue = target.value * isotopeBoost;
          const clampedValue = Math.max(target.range.min, Math.min(target.range.max, boostedValue));
          target.value = clampedValue;
          const distToMin = clampedValue - target.range.min;
          const distToMax = target.range.max - clampedValue;
          target.range.min = target.range.min + distToMin * strength * 0.1;
          target.range.max = target.range.max - distToMax * strength * 0.05;
        } else {
          const distToMin = target.value - target.range.min;
          const distToMax = target.range.max - target.value;
          target.range.min = target.range.min + distToMin * strength * 0.15;
          target.range.max = target.range.max - distToMax * strength * 0.1;
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
    case "modulates": {
      if (source.id === "pressure" && target.id === "phonon_softness") {
        const P = source.value;
        const pMax = source.range.max || 300;
        const pRatio = P / pMax;
        const transition = estimateTransitionPressureRatio(source.range);
        const baseHardening = 1 - 0.4 * pRatio;
        const softeningPeak = Math.exp(-Math.pow((pRatio - transition.peakRatio) / transition.width, 2));
        const transferFactor = baseHardening + 1.2 * softeningPeak;
        const scaledValue = target.value * Math.max(0.1, Math.min(2.5, transferFactor));
        const halfWidth = (target.range.max - target.range.min) / 2;
        const newHalf = halfWidth * (1 - strength * 0.15);
        target.value = scaledValue;
        target.range.min = Math.max(target.range.min, scaledValue - newHalf);
        target.range.max = Math.min(target.range.max, scaledValue + newHalf);
      } else {
        const center = target.value;
        const halfWidth = (target.range.max - target.range.min) / 2;
        const narrowedHalf = halfWidth * (1 - strength * 0.15);
        target.range.min = Math.max(target.range.min, center - narrowedHalf);
        target.range.max = Math.min(target.range.max, center + narrowedHalf);
      }
      break;
    }
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
    case "limits": {
      const stabilityVal = source.value;
      const collapseThreshold = stabilityVal * 0.85;
      if (target.range.max > collapseThreshold) {
        target.range.max = Math.max(target.range.min + 0.01, collapseThreshold);
      }
      if (target.value > collapseThreshold) {
        target.value = collapseThreshold;
      }
      break;
    }
    default: {
      break;
    }
  }

  if (target.range.min > target.range.max) {
    target.feasibility = 0;
    const mid = (target.range.min + target.range.max) / 2;
    target.range.min = mid;
    target.range.max = mid;
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
    const lambdaSamples = 12;
    const omegaSamples = 12;
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
          feas *= 1 / (1 + Math.exp(5 * (lambda - 2.5)));
          feas *= 1 / (1 + Math.exp(4 * (omega - 1200) / 1000));
          const avgPressure = (regime.pressureRange[0] + regime.pressureRange[1]) / 2;
          feas *= 1.0 / (1.0 + avgPressure / 100);
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

function sobolSequence2D(n: number): Array<[number, number]> {
  const points: Array<[number, number]> = [];
  const BITS = 30;
  const SCALE = 1 << BITS;
  const directionX: number[] = [];
  for (let i = 0; i < BITS; i++) {
    directionX.push(1 << (BITS - 1 - i));
  }
  const directionY: number[] = [];
  directionY[0] = 1 << (BITS - 1);
  for (let i = 1; i < BITS; i++) {
    let v = directionY[i - 1];
    v ^= (v >>> 1);
    directionY.push(v);
  }

  let xGray = 0;
  let yGray = 0;
  for (let i = 0; i < n; i++) {
    let c = 0;
    let val = i;
    while ((val & 1) !== 0) { val >>>= 1; c++; }
    if (c >= BITS) c = BITS - 1;
    xGray ^= directionX[c];
    yGray ^= directionY[c];
    points.push([xGray / SCALE, yGray / SCALE]);
  }
  return points;
}

const SOBOL_POINTS_PER_REGION = 24;
const sobolCache = sobolSequence2D(SOBOL_POINTS_PER_REGION);

function generateParameterCombinations(
  targetTc: number,
  muStar: number,
  regions: FeasibilityRegion[],
): ParameterCombination[] {
  const combinations: ParameterCombination[] = [];

  for (const region of regions) {
    const regime = REGIME_CONFIGS.find(r => r.name === region.regime);
    if (!regime) continue;

    const lambdaMin = regime.lambdaRange[0];
    const lambdaMax = regime.lambdaRange[1];
    const omegaMin = regime.omegaRange[0];
    const omegaMax = regime.omegaRange[1];

    for (const [u, v] of sobolCache) {
      const lambda = lambdaMin + u * (lambdaMax - lambdaMin);
      const omega = omegaMin + v * (omegaMax - omegaMin);

      const tc = mcMillanTc(lambda, omega, muStar);
      if (tc < targetTc * 0.5 || tc > targetTc * 2.0) continue;

      let dosBaseline: number;
      let nestingEstimate: number;
      let chargeTransferEstimate: number;
      const rName = regime.name;
      if (rName === "kagome-flat-band") {
        dosBaseline = 4.0 + 2.0 * Math.random();
        nestingEstimate = 0.7;
        chargeTransferEstimate = 0;
      } else if (rName === "unconventional-layered") {
        dosBaseline = 2.5 + 1.5 * Math.random();
        nestingEstimate = 0.6;
        chargeTransferEstimate = 0.2;
      } else if (rName === "superhydride" || rName === "hydride-moderate-pressure") {
        dosBaseline = 1.5 + 1.0 * Math.random();
        nestingEstimate = 0.3;
        chargeTransferEstimate = 0;
      } else if (rName === "topological-SC") {
        dosBaseline = 2.0 + 1.0 * Math.random();
        nestingEstimate = 0.5;
        chargeTransferEstimate = 0.05;
      } else if (rName === "light-element-compound") {
        dosBaseline = 2.0 + 1.5 * Math.random();
        nestingEstimate = 0.4;
        chargeTransferEstimate = 0;
      } else if (rName === "strong-coupling-metal") {
        dosBaseline = 3.0 + 1.5 * Math.random();
        nestingEstimate = 0.4;
        chargeTransferEstimate = 0;
      } else {
        dosBaseline = 2.0 + 1.0 * Math.random();
        nestingEstimate = 0.4;
        chargeTransferEstimate = 0;
      }
      const dosEstimate = dosBaseline;
      const avgMass = rName.includes("hydride") || rName.includes("superhydride") ? 5 : rName.includes("light") ? 15 : 50;
      const phononSoftness = Math.min(1.0, lambda * avgMass * 0.01 / (omega * 0.001 + 0.1));
      const debyeEstimate = Math.round(omega * 1.5);
      const hopfieldEstimate = dosEstimate * omega * 0.003;

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
    const accuracyA = Math.max(0, 1 - Math.abs(a.predictedTc - targetTc) / targetTc);
    const accuracyB = Math.max(0, 1 - Math.abs(b.predictedTc - targetTc) / targetTc);
    const scoreA = a.feasibility * accuracyA;
    const scoreB = b.feasibility * accuracyB;
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
    pressureGpa: number;
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
      pressureGpa: 150,
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
      pressureGpa: 30,
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
      pressureGpa: 0,
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
      pressureGpa: 0,
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
      pressureGpa: 0,
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
      pressureGpa: 0,
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
      pressureGpa: 0,
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
      pressureGpa: 0,
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
    const synthesisDifficulty = 1 / (1 + config.pressureGpa / 100);
    let feasibility = 0.5 * synthesisDifficulty;
    if (config.omega > 1500) feasibility *= 0.7;
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
  const maxNodeFeasibility = Math.max(...nodes.map(n => n.feasibility), 0.01);
  const satisfactionThreshold = Math.max(0.15, maxNodeFeasibility * 0.5);
  for (const node of nodes) {
    constraintSatisfactionMap[node.id] = node.feasibility >= satisfactionThreshold;
  }

  const feasibleNodeCount = nodes.filter(n => n.feasibility >= satisfactionThreshold).length;
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

  const globalLambdaMin = REGIME_CONFIGS.reduce((m, r) => Math.min(m, r.lambdaRange[0]), Infinity);
  const globalLambdaMax = REGIME_CONFIGS.reduce((m, r) => Math.max(m, r.lambdaRange[1]), -Infinity);
  const globalOmegaMin = REGIME_CONFIGS.reduce((m, r) => Math.min(m, r.omegaRange[0]), Infinity);
  const globalOmegaMax = REGIME_CONFIGS.reduce((m, r) => Math.max(m, r.omegaRange[1]), -Infinity);
  const globalPressureMin = REGIME_CONFIGS.reduce((m, r) => Math.min(m, r.pressureRange[0]), Infinity);
  const globalPressureMax = REGIME_CONFIGS.reduce((m, r) => Math.max(m, r.pressureRange[1]), -Infinity);

  let lambdaMin = globalLambdaMin;
  let lambdaMax = globalLambdaMax;
  let omegaMin = globalOmegaMin;
  let omegaMax = globalOmegaMax;
  let pressureMin = globalPressureMin;
  let pressureMax = globalPressureMax;

  if (regions.length > 0) {
    lambdaMin = regions.reduce((m, r) => Math.min(m, r.ranges.lambda.min), Infinity);
    lambdaMax = regions.reduce((m, r) => Math.max(m, r.ranges.lambda.max), -Infinity);
    omegaMin = regions.reduce((m, r) => Math.min(m, r.ranges.omegaLogK.min), Infinity);
    omegaMax = regions.reduce((m, r) => Math.max(m, r.ranges.omegaLogK.max), -Infinity);
    pressureMin = regions.reduce((m, r) => Math.min(m, r.ranges.pressure.min), Infinity);
    pressureMax = regions.reduce((m, r) => Math.max(m, r.ranges.pressure.max), -Infinity);
  }

  const best = regions[0];

  return {
    regions,
    rareRegions,
    totalRegions: regions.length,
    bestRegime: best?.regime ?? "none",
    bestFeasibility: best?.feasibilityScore ?? 0,
    parameterSpace: {
      lambdaBounds: { min: lambdaMin, max: lambdaMax },
      omegaLogBounds: { min: omegaMin, max: omegaMax },
      pressureBounds: { min: pressureMin, max: pressureMax },
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
  const elementFreq = new Map<string, number>();
  const structureFreq = new Map<string, number>();

  for (const combo of solution.parameterCombinations.slice(0, 10)) {
    for (const el of combo.elements) elementFreq.set(el, (elementFreq.get(el) ?? 0) + 1);
    for (const st of combo.structures) structureFreq.set(st, (structureFreq.get(st) ?? 0) + 1);
  }

  const sortedElements = Array.from(elementFreq.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([el]) => el);
  const sortedStructures = Array.from(structureFreq.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([st]) => st);

  const rareOpportunities = solution.rareRegions.slice(0, 3).map(r => ({
    id: r.id,
    description: r.description,
    novelty: r.noveltyScore,
  }));

  return {
    topRegimes,
    preferredElements: sortedElements.slice(0, 10),
    preferredStructures: sortedStructures.slice(0, 6),
    feasibility: solution.globalFeasibility,
    rareOpportunities,
  };
}
