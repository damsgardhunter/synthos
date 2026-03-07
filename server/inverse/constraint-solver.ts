import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
} from "../learning/physics-engine";

export interface ConstraintSolution {
  targetTc: number;
  muStar: number;
  requiredLambda: { min: number; max: number; optimal: number };
  requiredOmegaLog: { min: number; max: number; optimal: number };
  requiredDOS: DOSConstraint;
  requiredPhonon: PhononConstraint;
  requiredCoupling: CouplingConstraint;
  chargeTransfer: ChargeTransferConstraint;
  feasibilityScore: number;
  feasibilityNote: string;
  structuralTargets: StructuralTarget[];
  elementSuggestions: ElementSuggestion[];
  constraintChain: ConstraintStep[];
}

interface DOSConstraint {
  minDOS: number;
  optimalDOS: number;
  maxDOS: number;
  requiredOrbitalCharacter: string;
  bandStructureRequirement: string;
  vanHoveProximity: number;
  feasibility: number;
  note: string;
}

interface PhononConstraint {
  minOmegaLog: number;
  optimalOmegaLog: number;
  maxPhononFreq: number;
  requiredBondStiffness: string;
  lightElementFraction: number;
  debyeTempRange: { min: number; max: number };
  feasibility: number;
  note: string;
  elementMassConstraints: { maxAvgMass: number; preferredElements: string[] };
}

interface CouplingConstraint {
  lambdaRange: { min: number; max: number; optimal: number };
  requiredDOSContribution: number;
  requiredPhononSoftness: number;
  orbitalOverlapRequirement: string;
  bondingNetworkType: string;
  hopfieldParameter: { min: number; optimal: number };
  feasibility: number;
  note: string;
}

interface ChargeTransferConstraint {
  required: boolean;
  deltaCharge: { min: number; optimal: number };
  layerType: string;
  donorCandidates: string[];
  acceptorCandidates: string[];
  interlayerCoupling: string;
  feasibility: number;
  note: string;
}

interface StructuralTarget {
  type: string;
  reason: string;
  priority: number;
  exemplars: string[];
}

interface ElementSuggestion {
  elements: string[];
  role: string;
  reason: string;
  confidence: number;
}

interface ConstraintStep {
  step: number;
  parameter: string;
  constraint: string;
  value: string;
  status: "satisfied" | "feasible" | "challenging" | "unlikely";
}

interface SweepPoint {
  lambda: number;
  omegaLogK: number;
  tc: number;
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

function solveRequiredLambda(
  targetTc: number,
  omegaLogK: number,
  muStar: number,
): number {
  let lo = 0.1, hi = 5.0;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    const tc = mcMillanTc(mid, omegaLogK, muStar);
    if (tc < targetTc) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function solveRequiredOmegaLog(
  targetTc: number,
  lambda: number,
  muStar: number,
): number {
  let lo = 50, hi = 5000;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    const tc = mcMillanTc(lambda, mid, muStar);
    if (tc < targetTc) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function sweepLambdaOmega(
  targetTc: number,
  muStar: number,
): SweepPoint[] {
  const solutions: SweepPoint[] = [];
  for (let lam = 0.5; lam <= 4.0; lam += 0.1) {
    const omega = solveRequiredOmegaLog(targetTc, lam, muStar);
    if (omega > 0 && omega < 5000) {
      const actualTc = mcMillanTc(lam, omega, muStar);
      if (Math.abs(actualTc - targetTc) < targetTc * 0.05) {
        solutions.push({ lambda: Math.round(lam * 100) / 100, omegaLogK: Math.round(omega), tc: Math.round(actualTc * 10) / 10 });
      }
    }
  }
  return solutions;
}

function assessFeasibility(lambda: number, omegaLogK: number): { score: number; note: string } {
  let score = 1.0;
  const notes: string[] = [];

  if (lambda > 3.5) {
    score *= 0.2;
    notes.push("lambda > 3.5 is extremely rare");
  } else if (lambda > 2.5) {
    score *= 0.5;
    notes.push("lambda > 2.5 requires strong e-ph coupling (hydride-like)");
  } else if (lambda > 1.5) {
    score *= 0.8;
    notes.push("lambda 1.5-2.5 is achievable in hydrides and some compounds");
  }

  if (omegaLogK > 2000) {
    score *= 0.3;
    notes.push("omegaLog > 2000K requires very light elements (H-dominant)");
  } else if (omegaLogK > 1200) {
    score *= 0.6;
    notes.push("omegaLog 1200-2000K achievable with H, B, C");
  } else if (omegaLogK > 600) {
    score *= 0.9;
    notes.push("omegaLog 600-1200K achievable with light-element compounds");
  }

  if (lambda > 2.0 && omegaLogK > 1500) {
    score *= 1.2;
    notes.push("high lambda + high omegaLog is the hydride sweet spot");
  }

  return { score: Math.min(1, Math.round(score * 100) / 100), note: notes.join("; ") };
}

function suggestStructures(lambda: number, omegaLogK: number): StructuralTarget[] {
  const targets: StructuralTarget[] = [];

  if (omegaLogK > 1200 && lambda > 1.5) {
    targets.push({
      type: "clathrate-cage",
      reason: "High phonon freq + strong coupling requires hydrogen cage structures",
      priority: 1,
      exemplars: ["LaH10", "YH6", "CaH6", "ThH10"],
    });
    targets.push({
      type: "sodalite-cage",
      reason: "Sodalite cages maximize H-H metallic bonding and phonon hardening",
      priority: 2,
      exemplars: ["LaH10", "CeH9", "YH9"],
    });
  }

  if (lambda > 1.0 && omegaLogK > 600 && omegaLogK < 1500) {
    targets.push({
      type: "layered-hexagonal",
      reason: "Layered boride/carbide structures achieve moderate lambda with high phonon freq",
      priority: omegaLogK > 900 ? 1 : 2,
      exemplars: ["MgB2", "NbB2", "TaB2"],
    });
  }

  if (lambda > 0.8 && omegaLogK < 800) {
    targets.push({
      type: "A15",
      reason: "A15 structures offer high DOS and moderate coupling for transition metals",
      priority: 2,
      exemplars: ["Nb3Sn", "Nb3Ge", "V3Si"],
    });
  }

  if (lambda > 1.2 && omegaLogK < 600) {
    targets.push({
      type: "cuprate-layered",
      reason: "Low phonon freq + high coupling suggests unconventional layered mechanism",
      priority: 3,
      exemplars: ["YBCO", "Bi2212", "HgBaCuO"],
    });
  }

  if (omegaLogK > 800 && lambda > 0.8) {
    targets.push({
      type: "honeycomb",
      reason: "Honeycomb lattices with light atoms combine high phonon freq with good nesting",
      priority: 3,
      exemplars: ["MgB2", "LiBC"],
    });
  }

  return targets.sort((a, b) => a.priority - b.priority);
}

function suggestElements(lambda: number, omegaLogK: number): ElementSuggestion[] {
  const suggestions: ElementSuggestion[] = [];

  if (omegaLogK > 1200) {
    suggestions.push({
      elements: ["H"],
      role: "phonon hardener",
      reason: "Hydrogen provides the highest phonon frequencies due to its low mass",
      confidence: 0.9,
    });
  }

  if (omegaLogK > 600) {
    suggestions.push({
      elements: ["B", "C"],
      role: "phonon hardener",
      reason: "Light covalent elements provide high phonon freq and strong bonding",
      confidence: 0.8,
    });
  }

  if (lambda > 1.5) {
    suggestions.push({
      elements: ["La", "Y", "Ca", "Sr", "Ba"],
      role: "electron donor / cage stabilizer",
      reason: "Alkaline earth and rare earth metals stabilize cage structures and donate electrons",
      confidence: 0.85,
    });
    suggestions.push({
      elements: ["Nb", "V", "Ta", "Mo", "W"],
      role: "strong coupling provider",
      reason: "4d/5d transition metals have high Hopfield parameters for strong e-ph coupling",
      confidence: 0.75,
    });
  }

  if (lambda > 0.8 && lambda < 1.5) {
    suggestions.push({
      elements: ["Nb", "V", "Ti", "Zr", "Hf"],
      role: "transition metal backbone",
      reason: "Moderate coupling from d-electron compounds with good metallicity",
      confidence: 0.8,
    });
  }

  if (omegaLogK < 400 && lambda > 1.2) {
    suggestions.push({
      elements: ["Cu", "Fe", "Ni"],
      role: "correlated electron host",
      reason: "Low phonon scale + high coupling may indicate unconventional pairing",
      confidence: 0.5,
    });
  }

  return suggestions.sort((a, b) => b.confidence - a.confidence);
}

function solveDOSConstraint(targetTc: number, lambda: number): DOSConstraint {
  let minDOS: number, optimalDOS: number, maxDOS: number;
  let orbitalChar: string, bandReq: string, vhp: number;
  const notes: string[] = [];

  if (targetTc > 200) {
    minDOS = 5.0;
    optimalDOS = 8.0;
    maxDOS = 15.0;
    orbitalChar = "d-dominant with s-p hybridization";
    bandReq = "Multiple bands crossing Fermi level with van Hove singularity nearby";
    vhp = 0.05;
    notes.push("Ultra-high Tc demands DOS(Ef) > 5 states/eV with VHS proximity");
  } else if (targetTc > 100) {
    minDOS = 3.0;
    optimalDOS = 6.0;
    maxDOS = 12.0;
    orbitalChar = "d-dominant";
    bandReq = "High band degeneracy at Fermi level";
    vhp = 0.15;
    notes.push("High Tc requires DOS(Ef) > 3 states/eV");
  } else if (targetTc > 40) {
    minDOS = 1.5;
    optimalDOS = 3.5;
    maxDOS = 8.0;
    orbitalChar = "mixed d-p";
    bandReq = "Good metallicity with moderate band crossing";
    vhp = 0.3;
    notes.push("Moderate Tc achievable with DOS(Ef) > 1.5 states/eV");
  } else {
    minDOS = 0.8;
    optimalDOS = 2.0;
    maxDOS = 6.0;
    orbitalChar = "s-p or d";
    bandReq = "Metallic band structure";
    vhp = 0.5;
    notes.push("Low Tc requires only modest DOS(Ef)");
  }

  if (lambda > 2.0) {
    minDOS = Math.max(minDOS, 4.0);
    optimalDOS = Math.max(optimalDOS, 6.0);
    notes.push("Strong coupling (lambda > 2) requires enhanced DOS to sustain pairing");
  }

  let feasibility = 1.0;
  if (optimalDOS > 8) feasibility *= 0.4;
  else if (optimalDOS > 5) feasibility *= 0.7;
  else if (optimalDOS > 3) feasibility *= 0.9;

  return {
    minDOS: Math.round(minDOS * 10) / 10,
    optimalDOS: Math.round(optimalDOS * 10) / 10,
    maxDOS: Math.round(maxDOS * 10) / 10,
    requiredOrbitalCharacter: orbitalChar,
    bandStructureRequirement: bandReq,
    vanHoveProximity: vhp,
    feasibility: Math.round(feasibility * 100) / 100,
    note: notes.join("; "),
  };
}

function solvePhononConstraint(_targetTc: number, optimalOmegaLog: number): PhononConstraint {
  const notes: string[] = [];
  let bondStiffness: string;
  let lightFraction: number;
  let maxAvgMass: number;
  let preferred: string[];
  let maxPhononFreq: number;

  if (optimalOmegaLog > 1500) {
    bondStiffness = "Very stiff covalent/metallic bonds (>500 N/m spring constant)";
    lightFraction = 0.7;
    maxAvgMass = 10;
    preferred = ["H", "B", "C", "N"];
    maxPhononFreq = optimalOmegaLog * 3;
    notes.push("Requires H-dominant composition for phonon freq > 1500K");
  } else if (optimalOmegaLog > 800) {
    bondStiffness = "Stiff covalent bonds (200-500 N/m)";
    lightFraction = 0.4;
    maxAvgMass = 30;
    preferred = ["B", "C", "N", "O", "Si"];
    maxPhononFreq = optimalOmegaLog * 2.5;
    notes.push("Moderate light-element content needed for phonon freq 800-1500K");
  } else if (optimalOmegaLog > 400) {
    bondStiffness = "Moderate bond stiffness (100-200 N/m)";
    lightFraction = 0.15;
    maxAvgMass = 60;
    preferred = ["B", "C", "N", "Nb", "V", "Ti"];
    maxPhononFreq = optimalOmegaLog * 2;
    notes.push("Standard metallic/covalent bonds sufficient");
  } else {
    bondStiffness = "Soft bonds acceptable (<100 N/m)";
    lightFraction = 0.0;
    maxAvgMass = 120;
    preferred = ["Nb", "Sn", "Pb", "Bi"];
    maxPhononFreq = optimalOmegaLog * 2;
    notes.push("Heavy-element conventional superconductors feasible");
  }

  const debyeMin = Math.round(optimalOmegaLog * 0.8);
  const debyeMax = Math.round(optimalOmegaLog * 2.5);

  let feasibility = 1.0;
  if (optimalOmegaLog > 2000) feasibility *= 0.3;
  else if (optimalOmegaLog > 1200) feasibility *= 0.6;
  else if (optimalOmegaLog > 600) feasibility *= 0.9;

  if (maxAvgMass < 15) {
    feasibility *= 0.7;
    notes.push("Very low avg atomic mass limits viable compositions");
  }

  return {
    minOmegaLog: Math.round(optimalOmegaLog * 0.6),
    optimalOmegaLog: Math.round(optimalOmegaLog),
    maxPhononFreq: Math.round(maxPhononFreq),
    requiredBondStiffness: bondStiffness,
    lightElementFraction: Math.round(lightFraction * 100) / 100,
    debyeTempRange: { min: debyeMin, max: debyeMax },
    feasibility: Math.round(feasibility * 100) / 100,
    note: notes.join("; "),
    elementMassConstraints: { maxAvgMass, preferredElements: preferred },
  };
}

function solveCouplingConstraint(
  _targetTc: number,
  optimalLambda: number,
  optimalDOS: number,
  _optimalOmegaLog: number,
): CouplingConstraint {
  const notes: string[] = [];

  const phononSoftness = optimalLambda > 2.0 ? 0.6 : optimalLambda > 1.0 ? 0.4 : 0.2;
  const dosContribution = optimalDOS * 0.15;

  let orbitalReq: string;
  let bondingNet: string;
  let hopfieldMin: number;
  let hopfieldOptimal: number;

  if (optimalLambda > 2.5) {
    orbitalReq = "Strong s-d hybridization with large orbital overlap integrals";
    bondingNet = "3D metallic hydrogen network or cage structure with high connectivity";
    hopfieldMin = 8.0;
    hopfieldOptimal = 15.0;
    notes.push("Lambda > 2.5 requires extraordinary orbital overlap (hydride territory)");
  } else if (optimalLambda > 1.5) {
    orbitalReq = "Moderate d-orbital overlap with p-orbital hybridization";
    bondingNet = "Layered or cage structure with strong intralayer bonding";
    hopfieldMin = 4.0;
    hopfieldOptimal = 8.0;
    notes.push("Lambda 1.5-2.5 achievable with good d-band metals and light anions");
  } else if (optimalLambda > 0.8) {
    orbitalReq = "Standard d-band overlap";
    bondingNet = "Compact metallic bonding";
    hopfieldMin = 1.5;
    hopfieldOptimal = 4.0;
    notes.push("Moderate lambda achievable with conventional BCS materials");
  } else {
    orbitalReq = "Any metallic orbital overlap";
    bondingNet = "Simple metallic structure";
    hopfieldMin = 0.5;
    hopfieldOptimal = 2.0;
    notes.push("Weak coupling — standard metals sufficient");
  }

  let feasibility = 1.0;
  if (optimalLambda > 3.0) feasibility *= 0.25;
  else if (optimalLambda > 2.0) feasibility *= 0.5;
  else if (optimalLambda > 1.5) feasibility *= 0.75;

  if (hopfieldOptimal > 10) {
    feasibility *= 0.6;
    notes.push("Hopfield eta > 10 eV/A^2 is rare outside metallic hydrogen systems");
  }

  return {
    lambdaRange: {
      min: Math.round(optimalLambda * 0.7 * 100) / 100,
      max: Math.round(optimalLambda * 1.4 * 100) / 100,
      optimal: Math.round(optimalLambda * 1000) / 1000,
    },
    requiredDOSContribution: Math.round(dosContribution * 100) / 100,
    requiredPhononSoftness: Math.round(phononSoftness * 100) / 100,
    orbitalOverlapRequirement: orbitalReq,
    bondingNetworkType: bondingNet,
    hopfieldParameter: {
      min: Math.round(hopfieldMin * 10) / 10,
      optimal: Math.round(hopfieldOptimal * 10) / 10,
    },
    feasibility: Math.round(feasibility * 100) / 100,
    note: notes.join("; "),
  };
}

function solveChargeTransferConstraint(
  targetTc: number,
  optimalLambda: number,
  optimalOmegaLog: number,
): ChargeTransferConstraint {
  const notes: string[] = [];

  const isUnconventionalRegime = optimalOmegaLog < 600 && optimalLambda > 1.2;
  const isHighTcLayered = targetTc > 80 && optimalOmegaLog < 800;

  if (isUnconventionalRegime || isHighTcLayered) {
    let deltaMin: number, deltaOptimal: number;
    let layerType: string;
    let donors: string[];
    let acceptors: string[];
    let coupling: string;

    if (targetTc > 100) {
      deltaMin = 0.15;
      deltaOptimal = 0.3;
      layerType = "Conducting-insulating bilayer (CuO2-type or FeAs-type)";
      donors = ["La", "Sr", "Ba", "Ca", "Y", "Bi", "Tl"];
      acceptors = ["CuO2", "FeAs", "FeSe", "NiO2"];
      coupling = "Strong interlayer coupling with charge reservoir";
      notes.push("High-Tc unconventional regime: charge transfer between reservoir and SC layer is essential");
    } else {
      deltaMin = 0.08;
      deltaOptimal = 0.18;
      layerType = "Weakly coupled layered structure";
      donors = ["Sr", "Ba", "K", "Rb", "Cs"];
      acceptors = ["FeSe", "TiS2", "NbSe2", "TaS2"];
      coupling = "Moderate interlayer coupling via van der Waals or ionic bonding";
      notes.push("Moderate charge transfer enhances SC in intercalated compounds");
    }

    let feasibility = isUnconventionalRegime ? 0.7 : 0.6;
    if (targetTc > 150 && isUnconventionalRegime) {
      feasibility *= 0.6;
      notes.push("Very high Tc via charge transfer mechanism is challenging to engineer");
    }

    return {
      required: true,
      deltaCharge: { min: deltaMin, optimal: deltaOptimal },
      layerType,
      donorCandidates: donors,
      acceptorCandidates: acceptors,
      interlayerCoupling: coupling,
      feasibility: Math.round(feasibility * 100) / 100,
      note: notes.join("; "),
    };
  }

  return {
    required: false,
    deltaCharge: { min: 0, optimal: 0 },
    layerType: "Not required for conventional phonon-mediated SC",
    donorCandidates: [],
    acceptorCandidates: [],
    interlayerCoupling: "N/A — conventional BCS mechanism",
    feasibility: 1.0,
    note: "Charge transfer not critical for conventional e-ph mediated superconductivity",
  };
}

export function solveConstraints(
  targetTc: number,
  muStar: number = 0.10,
  pressureGpa: number = 0,
): ConstraintSolution {
  const muStarClamped = Math.max(0.08, Math.min(0.20, muStar));

  const optimalOmegaLog = targetTc > 200 ? 1500 : targetTc > 100 ? 800 : targetTc > 50 ? 500 : 300;
  const optimalLambda = solveRequiredLambda(targetTc, optimalOmegaLog, muStarClamped);

  const sweepSolutions = sweepLambdaOmega(targetTc, muStarClamped);

  let lambdaMin = Infinity, lambdaMax = 0;
  let omegaMin = Infinity, omegaMax = 0;
  for (const sol of sweepSolutions) {
    if (sol.lambda < lambdaMin) lambdaMin = sol.lambda;
    if (sol.lambda > lambdaMax) lambdaMax = sol.lambda;
    if (sol.omegaLogK < omegaMin) omegaMin = sol.omegaLogK;
    if (sol.omegaLogK > omegaMax) omegaMax = sol.omegaLogK;
  }

  if (lambdaMin === Infinity) {
    lambdaMin = optimalLambda * 0.7;
    lambdaMax = optimalLambda * 1.5;
  }
  if (omegaMin === Infinity) {
    omegaMin = optimalOmegaLog * 0.5;
    omegaMax = optimalOmegaLog * 2.0;
  }

  const feasibility = assessFeasibility(optimalLambda, optimalOmegaLog);
  const structuralTargets = suggestStructures(optimalLambda, optimalOmegaLog);
  const elementSuggestions = suggestElements(optimalLambda, optimalOmegaLog);

  if (pressureGpa > 50) {
    feasibility.score = Math.min(feasibility.score * 1.3, 1.0);
  }

  const dosConstraint = solveDOSConstraint(targetTc, optimalLambda);
  const phononConstraint = solvePhononConstraint(targetTc, optimalOmegaLog);
  const couplingConstraint = solveCouplingConstraint(targetTc, optimalLambda, dosConstraint.optimalDOS, optimalOmegaLog);
  const chargeTransfer = solveChargeTransferConstraint(targetTc, optimalLambda, optimalOmegaLog);

  const compositeFeasibility = Math.round(
    (feasibility.score * 0.3 +
      dosConstraint.feasibility * 0.2 +
      phononConstraint.feasibility * 0.2 +
      couplingConstraint.feasibility * 0.2 +
      chargeTransfer.feasibility * 0.1) * 100
  ) / 100;

  const allNotes = [feasibility.note];
  if (dosConstraint.note) allNotes.push(`DOS: ${dosConstraint.note}`);
  if (phononConstraint.note) allNotes.push(`Phonon: ${phononConstraint.note}`);
  if (couplingConstraint.note) allNotes.push(`Coupling: ${couplingConstraint.note}`);
  if (chargeTransfer.note) allNotes.push(`ChargeTransfer: ${chargeTransfer.note}`);
  const compositeNote = allNotes.filter(Boolean).join(". ");

  const chain: ConstraintStep[] = [
    {
      step: 1,
      parameter: "Tc",
      constraint: `Target Tc = ${targetTc}K`,
      value: `${targetTc}K`,
      status: "satisfied",
    },
    {
      step: 2,
      parameter: "mu*",
      constraint: `mu* in [0.08, 0.20]`,
      value: muStarClamped.toFixed(3),
      status: "satisfied",
    },
    {
      step: 3,
      parameter: "DOS(Ef)",
      constraint: `DOS(Ef) > ${dosConstraint.minDOS} states/eV (optimal ${dosConstraint.optimalDOS})`,
      value: `${dosConstraint.optimalDOS} states/eV, ${dosConstraint.requiredOrbitalCharacter}`,
      status: dosConstraint.feasibility > 0.7 ? "feasible" : dosConstraint.feasibility > 0.4 ? "challenging" : "unlikely",
    },
    {
      step: 4,
      parameter: "lambda",
      constraint: `lambda in [${lambdaMin.toFixed(2)}, ${lambdaMax.toFixed(2)}]`,
      value: `optimal ${optimalLambda.toFixed(3)}`,
      status: optimalLambda < 3.5 ? (optimalLambda < 2.5 ? "feasible" : "challenging") : "unlikely",
    },
    {
      step: 5,
      parameter: "omegaLog",
      constraint: `omegaLog in [${Math.round(omegaMin)}, ${Math.round(omegaMax)}]K`,
      value: `optimal ${Math.round(optimalOmegaLog)}K`,
      status: optimalOmegaLog < 2000 ? (optimalOmegaLog < 1200 ? "feasible" : "challenging") : "unlikely",
    },
    {
      step: 6,
      parameter: "phonon",
      constraint: `Bond stiffness: ${phononConstraint.requiredBondStiffness.split("(")[0].trim()}; light-element fraction >= ${phononConstraint.lightElementFraction}`,
      value: `Debye ${phononConstraint.debyeTempRange.min}-${phononConstraint.debyeTempRange.max}K; avg mass < ${phononConstraint.elementMassConstraints.maxAvgMass} amu`,
      status: phononConstraint.feasibility > 0.7 ? "feasible" : phononConstraint.feasibility > 0.4 ? "challenging" : "unlikely",
    },
    {
      step: 7,
      parameter: "e-ph coupling",
      constraint: `Hopfield eta > ${couplingConstraint.hopfieldParameter.min} eV/A^2; ${couplingConstraint.orbitalOverlapRequirement.split(" with")[0]}`,
      value: `eta optimal ${couplingConstraint.hopfieldParameter.optimal} eV/A^2; ${couplingConstraint.bondingNetworkType.split(" with")[0]}`,
      status: couplingConstraint.feasibility > 0.7 ? "feasible" : couplingConstraint.feasibility > 0.4 ? "challenging" : "unlikely",
    },
    {
      step: 8,
      parameter: "charge transfer",
      constraint: chargeTransfer.required ? `Delta-charge > ${chargeTransfer.deltaCharge.min}e; ${chargeTransfer.layerType.split("(")[0].trim()}` : "Not required",
      value: chargeTransfer.required ? `optimal ${chargeTransfer.deltaCharge.optimal}e; donors: ${chargeTransfer.donorCandidates.slice(0, 3).join(",")}` : "N/A",
      status: chargeTransfer.required ? (chargeTransfer.feasibility > 0.5 ? "feasible" : "challenging") : "satisfied",
    },
    {
      step: 9,
      parameter: "structure",
      constraint: structuralTargets.length > 0 ? `Prefer ${structuralTargets[0].type}` : "No strong structural preference",
      value: structuralTargets.map(s => s.type).join(", ") || "open",
      status: structuralTargets.length > 0 ? "feasible" : "challenging",
    },
    {
      step: 10,
      parameter: "elements",
      constraint: elementSuggestions.length > 0 ? `Core elements: ${elementSuggestions[0].elements.join(",")}` : "No specific element preference",
      value: elementSuggestions.flatMap(s => s.elements).slice(0, 5).join(", ") || "open",
      status: elementSuggestions.length > 0 ? "feasible" : "challenging",
    },
  ];

  return {
    targetTc,
    muStar: muStarClamped,
    requiredLambda: {
      min: Math.round(lambdaMin * 100) / 100,
      max: Math.round(lambdaMax * 100) / 100,
      optimal: Math.round(optimalLambda * 1000) / 1000,
    },
    requiredOmegaLog: {
      min: Math.round(omegaMin),
      max: Math.round(omegaMax),
      optimal: Math.round(optimalOmegaLog),
    },
    requiredDOS: dosConstraint,
    requiredPhonon: phononConstraint,
    requiredCoupling: couplingConstraint,
    chargeTransfer,
    feasibilityScore: compositeFeasibility,
    feasibilityNote: compositeNote,
    structuralTargets,
    elementSuggestions,
    constraintChain: chain,
  };
}

export interface FormulaEvaluation {
  formula: string;
  satisfiesLambda: boolean;
  satisfiesOmega: boolean;
  satisfiesDOS: boolean;
  satisfiesPhonon: boolean;
  satisfiesCoupling: boolean;
  lambdaValue: number;
  omegaLogValue: number;
  dosValue: number;
  debyeTemp: number;
  orbitalCharacter: string;
  metallicity: number;
  predictedTc: number;
  gapToTarget: number;
  matchScore: number;
  constraintsSatisfied: number;
  constraintsTotal: number;
  constraintSatisfaction: {
    dos: { met: boolean; value: number; required: number; gap: number };
    lambda: { met: boolean; value: number; required: number; gap: number };
    omegaLog: { met: boolean; value: number; required: number; gap: number };
    phonon: { met: boolean; debyeTemp: number; requiredRange: string };
    coupling: { met: boolean; hopfieldEstimate: string; bondingType: string };
    chargeTransfer: { relevant: boolean; note: string };
  };
}

export function evaluateFormulaAgainstConstraints(
  formula: string,
  solution: ConstraintSolution,
): FormulaEvaluation {
  try {
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);
    const omegaLogK = coupling.omegaLog * 1.44;

    const satisfiesLambda = coupling.lambda >= solution.requiredLambda.min * 0.8 &&
      coupling.lambda <= solution.requiredLambda.max * 1.2;
    const satisfiesOmega = omegaLogK >= solution.requiredOmegaLog.min * 0.8 &&
      omegaLogK <= solution.requiredOmegaLog.max * 1.2;
    const satisfiesDOS = electronic.densityOfStatesAtFermi >= solution.requiredDOS.minDOS * 0.8;
    const satisfiesPhonon = phonon.debyeTemperature >= solution.requiredPhonon.debyeTempRange.min * 0.8;
    const satisfiesCoupling = coupling.lambda >= solution.requiredCoupling.lambdaRange.min * 0.8;

    const tc = mcMillanTc(coupling.lambda, omegaLogK, solution.muStar);
    const gapToTarget = Math.abs(tc - solution.targetTc);

    const lambdaDist = Math.abs(coupling.lambda - solution.requiredLambda.optimal) / Math.max(0.1, solution.requiredLambda.optimal);
    const omegaDist = Math.abs(omegaLogK - solution.requiredOmegaLog.optimal) / Math.max(1, solution.requiredOmegaLog.optimal);
    const dosDist = Math.abs(electronic.densityOfStatesAtFermi - solution.requiredDOS.optimalDOS) / Math.max(0.1, solution.requiredDOS.optimalDOS);
    const couplingPenalty = satisfiesCoupling ? 0 : 0.4;
    const phononPenalty = satisfiesPhonon ? 0 : 0.3;

    const matchScore = Math.max(0, 1 - 0.25 * lambdaDist - 0.25 * omegaDist - 0.2 * dosDist - 0.15 * couplingPenalty - 0.15 * phononPenalty);

    const orbChar = electronic.orbitalFractions;
    const dominant = Object.entries(orbChar).sort((a, b) => b[1] - a[1])[0];

    const checks = [satisfiesLambda, satisfiesOmega, satisfiesDOS, satisfiesPhonon, satisfiesCoupling];
    const constraintsSatisfied = checks.filter(Boolean).length;

    const orbDominant = dominant[0];
    const hopfieldEstimate = orbDominant === "d" ? "Moderate-high (d-band)" : orbDominant === "p" ? "Moderate (p-band)" : "Low (s-band)";

    return {
      formula,
      satisfiesLambda,
      satisfiesOmega,
      satisfiesDOS,
      satisfiesPhonon,
      satisfiesCoupling,
      lambdaValue: Math.round(coupling.lambda * 1000) / 1000,
      omegaLogValue: Math.round(omegaLogK),
      dosValue: Math.round(electronic.densityOfStatesAtFermi * 100) / 100,
      debyeTemp: Math.round(phonon.debyeTemperature),
      orbitalCharacter: `${dominant[0]}-dominant (${Math.round(dominant[1] * 100)}%)`,
      metallicity: Math.round(electronic.metallicity * 1000) / 1000,
      predictedTc: Math.round(tc * 10) / 10,
      gapToTarget: Math.round(gapToTarget * 10) / 10,
      matchScore: Math.round(matchScore * 1000) / 1000,
      constraintsSatisfied,
      constraintsTotal: checks.length,
      constraintSatisfaction: {
        dos: {
          met: satisfiesDOS,
          value: Math.round(electronic.densityOfStatesAtFermi * 100) / 100,
          required: solution.requiredDOS.minDOS,
          gap: Math.round((solution.requiredDOS.minDOS - electronic.densityOfStatesAtFermi) * 100) / 100,
        },
        lambda: {
          met: satisfiesLambda,
          value: Math.round(coupling.lambda * 1000) / 1000,
          required: solution.requiredLambda.optimal,
          gap: Math.round((solution.requiredLambda.optimal - coupling.lambda) * 1000) / 1000,
        },
        omegaLog: {
          met: satisfiesOmega,
          value: Math.round(omegaLogK),
          required: solution.requiredOmegaLog.optimal,
          gap: Math.round(solution.requiredOmegaLog.optimal - omegaLogK),
        },
        phonon: {
          met: satisfiesPhonon,
          debyeTemp: Math.round(phonon.debyeTemperature),
          requiredRange: `${solution.requiredPhonon.debyeTempRange.min}-${solution.requiredPhonon.debyeTempRange.max}K`,
        },
        coupling: {
          met: satisfiesCoupling,
          hopfieldEstimate,
          bondingType: electronic.bandStructureType,
        },
        chargeTransfer: {
          relevant: solution.chargeTransfer.required,
          note: solution.chargeTransfer.required
            ? `Charge transfer needed: delta > ${solution.chargeTransfer.deltaCharge.min}e`
            : "Not required for this regime",
        },
      },
    };
  } catch {
    return {
      formula,
      satisfiesLambda: false,
      satisfiesOmega: false,
      satisfiesDOS: false,
      satisfiesPhonon: false,
      satisfiesCoupling: false,
      lambdaValue: 0,
      omegaLogValue: 0,
      dosValue: 0,
      debyeTemp: 0,
      orbitalCharacter: "unknown",
      metallicity: 0,
      predictedTc: 0,
      gapToTarget: solution.targetTc,
      matchScore: 0,
      constraintsSatisfied: 0,
      constraintsTotal: 5,
      constraintSatisfaction: {
        dos: { met: false, value: 0, required: solution.requiredDOS.minDOS, gap: solution.requiredDOS.minDOS },
        lambda: { met: false, value: 0, required: solution.requiredLambda.optimal, gap: solution.requiredLambda.optimal },
        omegaLog: { met: false, value: 0, required: solution.requiredOmegaLog.optimal, gap: solution.requiredOmegaLog.optimal },
        phonon: { met: false, debyeTemp: 0, requiredRange: `${solution.requiredPhonon.debyeTempRange.min}-${solution.requiredPhonon.debyeTempRange.max}K` },
        coupling: { met: false, hopfieldEstimate: "unknown", bondingType: "unknown" },
        chargeTransfer: { relevant: solution.chargeTransfer.required, note: "Evaluation failed" },
      },
    };
  }
}

const constraintCache = new Map<string, ConstraintSolution>();

export function getCachedConstraints(targetTc: number, muStar: number = 0.10): ConstraintSolution {
  const key = `${targetTc}-${muStar.toFixed(3)}`;
  if (constraintCache.has(key)) return constraintCache.get(key)!;
  const solution = solveConstraints(targetTc, muStar);
  constraintCache.set(key, solution);
  if (constraintCache.size > 50) {
    const firstKey = constraintCache.keys().next().value;
    if (firstKey) constraintCache.delete(firstKey);
  }
  return solution;
}

export function getConstraintGuidanceForGenerator(targetTc: number): {
  lambdaRange: [number, number];
  omegaLogRange: [number, number];
  dosRange: [number, number];
  preferredElements: string[];
  preferredStructures: string[];
  phononElements: string[];
  chargeTransferRequired: boolean;
  hopfieldRange: [number, number];
  feasibility: number;
} {
  const solution = getCachedConstraints(targetTc);
  return {
    lambdaRange: [solution.requiredLambda.min, solution.requiredLambda.max],
    omegaLogRange: [solution.requiredOmegaLog.min, solution.requiredOmegaLog.max],
    dosRange: [solution.requiredDOS.minDOS, solution.requiredDOS.maxDOS],
    preferredElements: solution.elementSuggestions.flatMap(s => s.elements).slice(0, 8),
    preferredStructures: solution.structuralTargets.map(s => s.type).slice(0, 4),
    phononElements: solution.requiredPhonon.elementMassConstraints.preferredElements,
    chargeTransferRequired: solution.chargeTransfer.required,
    hopfieldRange: [solution.requiredCoupling.hopfieldParameter.min, solution.requiredCoupling.hopfieldParameter.optimal],
    feasibility: solution.feasibilityScore,
  };
}
