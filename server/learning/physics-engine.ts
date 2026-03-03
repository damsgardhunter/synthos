import OpenAI from "openai";
import type { EventEmitter } from "./engine";
import type { SuperconductorCandidate } from "@shared/schema";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export interface ElectronicStructure {
  bandStructureType: string;
  fermiSurfaceTopology: string;
  densityOfStatesAtFermi: number;
  correlationStrength: number;
  metallicity: number;
  orbitalCharacter: string;
  nestingFeatures: string;
}

export interface PhononSpectrum {
  maxPhononFrequency: number;
  logAverageFrequency: number;
  hasImaginaryModes: boolean;
  anharmonicityIndex: number;
  softModePresent: boolean;
  debyeTemperature: number;
}

export interface ElectronPhononCoupling {
  lambda: number;
  omegaLog: number;
  muStar: number;
  isStrongCoupling: boolean;
  dominantPhononBranch: string;
}

export interface EliashbergResult {
  predictedTc: number;
  gapRatio: number;
  isotropicGap: boolean;
  strongCouplingCorrection: number;
  confidenceBand: [number, number];
}

export interface CompetingPhase {
  phaseName: string;
  type: string;
  transitionTemp: number | null;
  strength: number;
  suppressesSC: boolean;
}

export interface CriticalFieldResult {
  upperCriticalField: number;
  coherenceLength: number;
  londonPenetrationDepth: number;
  anisotropyRatio: number;
  criticalCurrentDensity: number;
  typeIorII: string;
}

const ELEMENT_VALENCE_ELECTRONS: Record<string, number> = {
  H:1,Li:1,Na:1,K:1,Rb:1,Cs:1,Be:2,Mg:2,Ca:2,Sr:2,Ba:2,
  Sc:3,Ti:4,V:5,Cr:6,Mn:7,Fe:8,Co:9,Ni:10,Cu:11,Zn:12,
  Y:3,Zr:4,Nb:5,Mo:6,Tc:7,Ru:8,Rh:9,Pd:10,Ag:11,Cd:12,
  La:3,Hf:4,Ta:5,W:6,Re:7,Os:8,Ir:9,Pt:10,Au:11,Hg:12,
  B:3,Al:3,Ga:3,In:3,Tl:3,C:4,Si:4,Ge:4,Sn:4,Pb:4,
  N:5,P:5,As:5,Sb:5,Bi:5,O:6,S:6,Se:6,Te:6,F:7,Cl:7,Br:7,I:7,
};

function parseFormulaElements(formula: string): string[] {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? [...new Set(matches)] : [];
}

function estimateValenceElectrons(elements: string[]): number {
  return elements.reduce((sum, el) => sum + (ELEMENT_VALENCE_ELECTRONS[el] ?? 3), 0) / elements.length;
}

export function computeElectronicStructure(formula: string, spacegroup?: string | null): ElectronicStructure {
  const elements = parseFormulaElements(formula);
  const avgValence = estimateValenceElectrons(elements);

  const transitionMetals = ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Hf","Ta","W","Re","Os","Ir","Pt","Au"];
  const rareEarths = ["La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"];

  const hasTM = elements.some(e => transitionMetals.includes(e));
  const hasRE = elements.some(e => rareEarths.includes(e));
  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const hasH = elements.includes("H");

  let correlationStrength = 0.2;
  if (hasCu && hasO) correlationStrength = 0.8;
  else if (hasRE) correlationStrength = 0.7;
  else if (hasTM) correlationStrength = 0.5;
  if (elements.includes("Fe") || elements.includes("Mn")) correlationStrength = Math.min(1, correlationStrength + 0.2);

  const densityOfStatesAtFermi = hasTM ? 2.5 + avgValence * 0.3 : hasH ? 1.8 : 1.2;
  const metallicity = hasTM || hasH ? 0.85 : (avgValence > 4 ? 0.3 : 0.6);

  let fermiSurfaceTopology = "simple spherical";
  if (hasCu && hasO) fermiSurfaceTopology = "quasi-2D cylindrical with nesting features at (pi,pi)";
  else if (hasH) fermiSurfaceTopology = "nested multi-sheet with strong e-ph coupling pockets";
  else if (hasTM && elements.length >= 3) fermiSurfaceTopology = "multi-band with electron and hole pockets";
  else if (hasTM) fermiSurfaceTopology = "complex multi-sheet d-band dominated";

  let orbitalCharacter = "sp-hybridized";
  if (hasCu && hasO) orbitalCharacter = "Cu-3d(x2-y2) / O-2p hybridized";
  else if (elements.includes("Fe")) orbitalCharacter = "Fe-3d multi-orbital (t2g/eg)";
  else if (hasTM) orbitalCharacter = "d-band dominated with p-hybridization";
  else if (hasH) orbitalCharacter = "H-1s sigma-bonding network";

  let bandStructureType = "metallic";
  if (!hasTM && !hasH && avgValence > 5) bandStructureType = "insulating";
  else if (correlationStrength > 0.6) bandStructureType = "strongly correlated metal";

  const nestingFeatures = correlationStrength > 0.5
    ? "Significant Fermi surface nesting promoting spin/charge instabilities"
    : "Weak nesting, conventional metallic behavior";

  return {
    bandStructureType,
    fermiSurfaceTopology,
    densityOfStatesAtFermi,
    correlationStrength,
    metallicity,
    orbitalCharacter,
    nestingFeatures,
  };
}

export function computePhononSpectrum(formula: string, electronicStructure: ElectronicStructure): PhononSpectrum {
  const elements = parseFormulaElements(formula);
  const hasH = elements.includes("H");
  const hasO = elements.includes("O");
  const masses = elements.map(e => ({
    H:1,Li:7,Be:9,B:11,C:12,N:14,O:16,F:19,Na:23,Mg:24,Al:27,Si:28,
    S:32,Ca:40,Ti:48,V:51,Cr:52,Fe:56,Cu:64,Zn:65,Sr:88,Y:89,Zr:91,
    Nb:93,Ba:137,La:139,Hf:178,Ta:181,W:184,Pb:207,Bi:209,
  }[e] ?? 50));

  const lightestMass = Math.min(...masses);
  const maxPhononFreq = hasH ? 4000 + Math.random() * 500 : 800 / Math.sqrt(lightestMass / 12) * (1 + Math.random() * 0.2);

  const logAvgFreq = maxPhononFreq * (hasH ? 0.35 : 0.45);

  const hasImaginaryModes = electronicStructure.correlationStrength > 0.7 && Math.random() < 0.3;
  const anharmonicityIndex = hasH ? 0.4 + Math.random() * 0.3 : 0.1 + Math.random() * 0.2;
  const softModePresent = electronicStructure.correlationStrength > 0.5 && Math.random() < 0.4;

  const avgMass = masses.reduce((a, b) => a + b, 0) / masses.length;
  const debyeTemperature = hasH ? 1500 + Math.random() * 500 : 300 * Math.sqrt(30 / avgMass) * (1 + Math.random() * 0.3);

  return {
    maxPhononFrequency: Math.round(maxPhononFreq),
    logAverageFrequency: Math.round(logAvgFreq),
    hasImaginaryModes,
    anharmonicityIndex: Number(anharmonicityIndex.toFixed(3)),
    softModePresent,
    debyeTemperature: Math.round(debyeTemperature),
  };
}

export function computeElectronPhononCoupling(
  electronicStructure: ElectronicStructure,
  phononSpectrum: PhononSpectrum
): ElectronPhononCoupling {
  const N_EF = electronicStructure.densityOfStatesAtFermi;
  const omega_log = phononSpectrum.logAverageFrequency;
  const corr = electronicStructure.correlationStrength;

  let lambda = N_EF * 0.3 * (1 + phononSpectrum.anharmonicityIndex);
  if (electronicStructure.fermiSurfaceTopology.includes("nested")) lambda *= 1.4;
  if (electronicStructure.fermiSurfaceTopology.includes("multi-sheet")) lambda *= 1.2;
  if (corr > 0.6) lambda *= 0.7;

  lambda = Math.max(0.1, Math.min(3.5, lambda + (Math.random() - 0.5) * 0.3));

  const muStar = corr > 0.5 ? 0.13 + corr * 0.05 : 0.1 + Math.random() * 0.04;

  const isStrongCoupling = lambda > 1.5;

  let dominantPhononBranch = "acoustic";
  if (phononSpectrum.maxPhononFrequency > 2000) dominantPhononBranch = "high-frequency optical (H vibrations)";
  else if (phononSpectrum.softModePresent) dominantPhononBranch = "soft optical mode";
  else if (lambda > 1.0) dominantPhononBranch = "low-energy optical";

  return {
    lambda: Number(lambda.toFixed(3)),
    omegaLog: omega_log,
    muStar: Number(muStar.toFixed(4)),
    isStrongCoupling,
    dominantPhononBranch,
  };
}

export function predictTcEliashberg(coupling: ElectronPhononCoupling): EliashbergResult {
  const { lambda, omegaLog, muStar } = coupling;

  const KB = 0.0862;
  const omegaLogK = omegaLog * 1.44;

  let tc: number;
  if (lambda < 1.5) {
    const f1 = Math.pow(1 + (lambda / 2.46 / (1 + 3.8 * muStar)), 1/3);
    const exponent = -1.04 * (1 + lambda) / (lambda - muStar * (1 + 0.62 * lambda));
    tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
  } else {
    const f1 = Math.sqrt(1 + (lambda / 2.46));
    const exponent = -1.04 * (1 + lambda) / (lambda - muStar * (1 + 0.62 * lambda));
    tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
  }

  tc = Math.max(0, tc);

  const gapRatio = lambda > 1.5 ? 2 * 1.764 * (1 + 12.5 * (lambda / (lambda + 5)) * (lambda / (lambda + 5))) : 2 * 1.764;
  const isotropicGap = lambda < 1.0;
  const strongCouplingCorrection = lambda > 1.5 ? 1 + 5.3 * (lambda / (lambda + 6)) * (lambda / (lambda + 6)) : 1.0;

  const uncertainty = tc * 0.15;
  const confidenceBand: [number, number] = [
    Math.max(0, Math.round(tc - uncertainty)),
    Math.round(tc + uncertainty),
  ];

  return {
    predictedTc: Math.round(tc * 10) / 10,
    gapRatio: Number(gapRatio.toFixed(3)),
    isotropicGap,
    strongCouplingCorrection: Number(strongCouplingCorrection.toFixed(3)),
    confidenceBand,
  };
}

export function evaluateCompetingPhases(
  formula: string,
  electronicStructure: ElectronicStructure
): CompetingPhase[] {
  const elements = parseFormulaElements(formula);
  const phases: CompetingPhase[] = [];

  if (elements.includes("Fe") || elements.includes("Mn") || elements.includes("Cr")) {
    const strength = electronicStructure.correlationStrength * 0.8;
    phases.push({
      phaseName: "Antiferromagnetic order",
      type: "magnetism",
      transitionTemp: Math.round(100 + strength * 400),
      strength,
      suppressesSC: strength > 0.6,
    });
  }

  if (electronicStructure.fermiSurfaceTopology.includes("nesting")) {
    phases.push({
      phaseName: "Charge density wave",
      type: "CDW",
      transitionTemp: Math.round(50 + Math.random() * 150),
      strength: 0.4 + Math.random() * 0.3,
      suppressesSC: Math.random() < 0.4,
    });
  }

  if (electronicStructure.correlationStrength > 0.7) {
    phases.push({
      phaseName: "Mott insulating phase",
      type: "Mott",
      transitionTemp: null,
      strength: electronicStructure.correlationStrength,
      suppressesSC: electronicStructure.correlationStrength > 0.9,
    });
  }

  if (elements.includes("Cu") && elements.includes("O")) {
    phases.push({
      phaseName: "Pseudogap phase",
      type: "pseudogap",
      transitionTemp: Math.round(200 + Math.random() * 100),
      strength: 0.5,
      suppressesSC: false,
    });
    phases.push({
      phaseName: "Spin-density wave",
      type: "SDW",
      transitionTemp: Math.round(100 + Math.random() * 200),
      strength: 0.6,
      suppressesSC: true,
    });
  }

  if (elements.length >= 3 && Math.random() < 0.3) {
    phases.push({
      phaseName: "Structural phase transition",
      type: "structural",
      transitionTemp: Math.round(150 + Math.random() * 300),
      strength: 0.3 + Math.random() * 0.3,
      suppressesSC: Math.random() < 0.3,
    });
  }

  return phases;
}

export function computeCriticalFields(
  tc: number,
  coupling: ElectronPhononCoupling,
  dimensionality: string
): CriticalFieldResult {
  if (tc <= 0) {
    return {
      upperCriticalField: 0,
      coherenceLength: 0,
      londonPenetrationDepth: 0,
      anisotropyRatio: 1,
      criticalCurrentDensity: 0,
      typeIorII: "N/A",
    };
  }

  const xi0 = 1000 / (tc * coupling.lambda * 0.5);
  const coherenceLength = Math.max(0.5, Math.min(500, xi0));

  const PHI0 = 2.07e-15;
  const Hc2 = PHI0 / (2 * Math.PI * (coherenceLength * 1e-9) * (coherenceLength * 1e-9));
  const upperCriticalField = Math.round(Hc2 / 10000) / 100;

  const lambdaL = 50 + 200 * coupling.lambda * (1 + coupling.muStar);
  const londonPenetrationDepth = Math.max(30, Math.min(2000, lambdaL));

  let anisotropyRatio = 1.0;
  if (dimensionality === "2D" || dimensionality === "quasi-2D") anisotropyRatio = 5 + Math.random() * 15;
  else if (dimensionality === "layered") anisotropyRatio = 3 + Math.random() * 7;

  const Jc = tc * 1e4 * coupling.lambda / (1 + anisotropyRatio * 0.1);
  const criticalCurrentDensity = Math.round(Jc);

  const kappa = londonPenetrationDepth / coherenceLength;
  const typeIorII = kappa > 0.707 ? "Type-II" : "Type-I";

  return {
    upperCriticalField: Number(upperCriticalField.toFixed(2)),
    coherenceLength: Number(coherenceLength.toFixed(1)),
    londonPenetrationDepth: Number(londonPenetrationDepth.toFixed(1)),
    anisotropyRatio: Number(anisotropyRatio.toFixed(2)),
    criticalCurrentDensity,
    typeIorII,
  };
}

export function assessCorrelationStrength(formula: string): {
  ratio: number;
  regime: string;
  treatmentRequired: string;
} {
  const elements = parseFormulaElements(formula);

  const stronglyCorrelated = ["Cu", "Fe", "Mn", "Co", "Ni", "Ce", "U", "Pu", "Sm", "Yb"];
  const moderatelyCorrelated = ["V", "Cr", "Ti", "Nb", "Mo", "W", "La", "Y"];

  let maxCorrelation = 0;
  for (const el of elements) {
    if (stronglyCorrelated.includes(el)) maxCorrelation = Math.max(maxCorrelation, 0.7 + Math.random() * 0.3);
    else if (moderatelyCorrelated.includes(el)) maxCorrelation = Math.max(maxCorrelation, 0.3 + Math.random() * 0.3);
  }

  if (elements.includes("O") && elements.some(e => stronglyCorrelated.includes(e))) {
    maxCorrelation = Math.min(1.0, maxCorrelation * 1.2);
  }

  if (maxCorrelation === 0 && elements.includes("H")) maxCorrelation = 0.1 + Math.random() * 0.1;
  if (maxCorrelation === 0) maxCorrelation = 0.15 + Math.random() * 0.15;

  let regime = "weakly correlated";
  let treatmentRequired = "DFT + DFPT + Migdal-Eliashberg";

  if (maxCorrelation > 0.7) {
    regime = "strongly correlated";
    treatmentRequired = "DMFT + beyond-DFT (GW/QMC) + unconventional pairing analysis";
  } else if (maxCorrelation > 0.4) {
    regime = "moderately correlated";
    treatmentRequired = "DFT+U or hybrid functionals + extended Eliashberg";
  }

  return {
    ratio: Number(maxCorrelation.toFixed(3)),
    regime,
    treatmentRequired,
  };
}

export async function runFullPhysicsAnalysis(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{
  electronicStructure: ElectronicStructure;
  phononSpectrum: PhononSpectrum;
  coupling: ElectronPhononCoupling;
  eliashberg: EliashbergResult;
  competingPhases: CompetingPhase[];
  criticalFields: CriticalFieldResult;
  correlation: ReturnType<typeof assessCorrelationStrength>;
  dimensionality: string;
  uncertaintyEstimate: number;
}> {
  const formula = candidate.formula;

  emit("log", {
    phase: "phase-10",
    event: "Physics analysis started",
    detail: `Computing electronic structure, phonons, e-ph coupling for ${formula}`,
    dataSource: "Physics Engine",
  });

  const correlation = assessCorrelationStrength(formula);
  const electronicStructure = computeElectronicStructure(formula, candidate.crystalStructure);

  const phononSpectrum = computePhononSpectrum(formula, electronicStructure);
  const coupling = computeElectronPhononCoupling(electronicStructure, phononSpectrum);

  let eliashberg: EliashbergResult;
  if (correlation.ratio < 0.6) {
    eliashberg = predictTcEliashberg(coupling);
  } else {
    eliashberg = predictTcEliashberg(coupling);
    eliashberg.predictedTc = eliashberg.predictedTc * (1 + correlation.ratio * 0.5);
    eliashberg.confidenceBand = [
      Math.round(eliashberg.predictedTc * 0.5),
      Math.round(eliashberg.predictedTc * 1.8),
    ];
  }

  const competingPhases = evaluateCompetingPhases(formula, electronicStructure);

  let dimensionality = candidate.dimensionality || "3D";
  if (!candidate.dimensionality) {
    if (electronicStructure.fermiSurfaceTopology.includes("2D")) dimensionality = "quasi-2D";
    else if (electronicStructure.orbitalCharacter.includes("hybridized") && formula.includes("O")) dimensionality = "layered";
    else dimensionality = "3D";
  }

  const criticalFields = computeCriticalFields(eliashberg.predictedTc, coupling, dimensionality);

  const suppressingPhases = competingPhases.filter(p => p.suppressesSC);
  let uncertaintyEstimate = 0.3;
  if (correlation.ratio > 0.7) uncertaintyEstimate += 0.2;
  if (phononSpectrum.hasImaginaryModes) uncertaintyEstimate += 0.15;
  if (suppressingPhases.length > 0) uncertaintyEstimate += 0.1;
  if (phononSpectrum.anharmonicityIndex > 0.5) uncertaintyEstimate += 0.1;
  uncertaintyEstimate = Math.min(0.95, uncertaintyEstimate);

  emit("log", {
    phase: "phase-10",
    event: "Physics analysis complete",
    detail: `${formula}: Tc=${eliashberg.predictedTc}K (${eliashberg.confidenceBand[0]}-${eliashberg.confidenceBand[1]}K), lambda=${coupling.lambda}, Hc2=${criticalFields.upperCriticalField}T, ${correlation.regime}, ${competingPhases.length} competing phases`,
    dataSource: "Physics Engine",
  });

  return {
    electronicStructure,
    phononSpectrum,
    coupling,
    eliashberg,
    competingPhases,
    criticalFields,
    correlation,
    dimensionality,
    uncertaintyEstimate,
  };
}
