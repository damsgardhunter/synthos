import {
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
  getDebyeTemperature,
} from "../learning/elemental-data";

export interface ChargeTransferAnalysis {
  electronegativityMismatch: number;
  workFunctionMismatch: number;
  chargeTransferDirection: string;
  transferredChargeDensity: number;
  dopingType: "electron" | "hole" | "neutral";
  chargeTransferScore: number;
}

export interface InterfacePhononAnalysis {
  acousticMismatchRatio: number;
  acousticMismatchScore: number;
  softModeCouplingScore: number;
  interfacePhononEnhancement: number;
  dominantPhononMode: string;
  phononScore: number;
}

export interface EpitaxialStrainAnalysis {
  latticeMismatch: number;
  criticalThickness: number;
  strainType: "compressive" | "tensile" | "matched";
  strainMagnitude: number;
  strainEnhancedCoupling: number;
  strainScore: number;
}

export interface DimensionalConfinement {
  confinementDimension: number;
  quantumWellWidth: number;
  twoDEnhancementFactor: number;
  confinementScore: number;
}

export interface InterfaceAnalysis {
  layerA: string;
  layerB: string;
  chargeTransfer: ChargeTransferAnalysis;
  phononCoupling: InterfacePhononAnalysis;
  epitaxialStrain: EpitaxialStrainAnalysis;
  dimensionalConfinement: DimensionalConfinement;
  interfaceScScore: number;
  estimatedInterfaceTc: number;
  mechanism: string;
  confidence: number;
  knownSystemMatch: string | null;
}

export interface HeterostructureCandidate {
  layerA: string;
  layerB: string;
  stackingPattern: string;
  interfaceAnalysis: InterfaceAnalysis;
  rank: number;
}

const SUBSCRIPT_MAP = "₀₁₂₃₄₅₆₇₈₉";
function cleanFormula(formula: string): string {
  if (typeof formula !== "string") formula = String(formula ?? "");
  let cleaned = "";
  for (let i = 0; i < formula.length; i++) {
    const idx = SUBSCRIPT_MAP.indexOf(formula[i]);
    cleaned += idx >= 0 ? String(idx) : formula[i];
  }
  return cleaned.replace(/[\u2080-\u2089]/g, c => String(c.charCodeAt(0) - 0x2080));
}

function parseFormulaElements(formula: string): string[] {
  const cleaned = cleanFormula(formula);
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = cleanFormula(formula);
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  return total > 0 ? total : 1;
}

function getAverageElectronegativity(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  let sum = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const en = data?.paulingElectronegativity ?? 1.5;
    sum += en * (counts[el] || 1) / totalAtoms;
  }
  return sum;
}

function getAverageMassForFormula(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  let sum = 0;
  for (const el of elements) {
    const data = getElementData(el);
    sum += (data?.atomicMass ?? 50) * (counts[el] || 1) / totalAtoms;
  }
  return sum;
}

function getAverageDebyeTemp(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  let sum = 0;
  let count = 0;
  for (const el of elements) {
    const td = getDebyeTemperature(el);
    if (td !== null) {
      sum += td * (counts[el] || 1) / totalAtoms;
      count++;
    }
  }
  return count > 0 ? sum : 300;
}

function getAverageLatticeConstant(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  let sum = 0;
  let count = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data?.latticeConstant) {
      sum += data.latticeConstant * (counts[el] || 1) / totalAtoms;
      count++;
    }
  }
  return count > 0 ? sum : 4.0;
}

function getAverageBulkModulus(formula: string, pressureGpa: number = 0): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  let sum = 0;
  let count = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data?.bulkModulus) {
      sum += data.bulkModulus * (counts[el] || 1) / totalAtoms;
      count++;
    }
  }
  const B0 = count > 0 ? sum : 30;
  const Bprime = 4.0;
  return B0 + Bprime * pressureGpa;
}

const BULK_WORK_FUNCTIONS: Record<string, number> = {
  Al: 4.28, Ag: 4.26, Au: 5.10, Ba: 2.52, Be: 4.98, Bi: 4.34, Ca: 2.87,
  Cd: 4.22, Ce: 2.90, Co: 5.00, Cr: 4.50, Cs: 2.14, Cu: 4.65, Fe: 4.50,
  Ga: 4.20, Ge: 5.00, Hf: 3.90, Hg: 4.49, In: 4.12, Ir: 5.27, K: 2.30,
  La: 3.50, Li: 2.90, Mg: 3.66, Mn: 4.10, Mo: 4.60, Na: 2.75, Nb: 4.30,
  Nd: 3.20, Ni: 5.15, Os: 5.93, Pb: 4.25, Pd: 5.12, Pt: 5.65, Rb: 2.16,
  Re: 4.72, Rh: 4.98, Ru: 4.71, Sc: 3.50, Si: 4.85, Sn: 4.42, Sr: 2.59,
  Ta: 4.25, Te: 4.95, Th: 3.40, Ti: 4.33, Tl: 3.84, U: 3.63, V: 4.30,
  W: 4.55, Y: 3.10, Zn: 4.33, Zr: 4.05,
};

const PACKING_DENSITY: Record<string, number> = {
  fcc: 0.74, hcp: 0.74, bcc: 0.68, diamond: 0.34, sc: 0.52,
};

function estimateWorkFunction(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  let sum = 0;
  let hasBulkData = false;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    const bulkWF = BULK_WORK_FUNCTIONS[el];
    if (bulkWF !== undefined) {
      sum += bulkWF * frac;
      hasBulkData = true;
    } else {
      const data = getElementData(el);
      const ie = data?.firstIonizationEnergy ?? 7;
      const ea = data?.electronAffinity ?? 0;
      const wf = (ie + Math.max(0, ea)) / 2;
      sum += wf * frac;
    }
  }

  if (hasBulkData) {
    const avgMass = elements.reduce((s, el) => {
      const d = getElementData(el);
      return s + (d?.atomicMass ?? 50) * (counts[el] || 1) / totalAtoms;
    }, 0);
    const avgRadius = elements.reduce((s, el) => {
      const d = getElementData(el);
      return s + (d?.atomicRadius ?? 150) * (counts[el] || 1) / totalAtoms;
    }, 0);
    const packingProxy = avgMass > 100 ? 0.74 : avgMass > 50 ? 0.70 : 0.68;
    const packingCorrection = (packingProxy - 0.68) * 0.5;
    sum += packingCorrection;
  }

  return sum;
}

function estimateMetallicity(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  let metalFrac = 0;
  for (const el of elements) {
    if (isTransitionMetal(el) || isRareEarth(el) || isActinide(el) ||
        ["Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba","Al","Ga","In","Tl","Sn","Pb","Bi"].includes(el)) {
      metalFrac += (counts[el] || 1) / totalAtoms;
    }
  }
  return metalFrac;
}

function computeChargeTransfer(layerA: string, layerB: string): ChargeTransferAnalysis {
  const enA = getAverageElectronegativity(layerA);
  const enB = getAverageElectronegativity(layerB);
  const wfA = estimateWorkFunction(layerA);
  const wfB = estimateWorkFunction(layerB);

  const enMismatch = Math.abs(enA - enB);
  const wfMismatch = Math.abs(wfA - wfB);

  const metalA = estimateMetallicity(layerA);
  const metalB = estimateMetallicity(layerB);
  const bothMetallic = metalA > 0.6 && metalB > 0.6;

  let chargeTransferDirection: string;
  let dopingType: "electron" | "hole" | "neutral";

  if (bothMetallic) {
    if (wfA > wfB + 0.1) {
      chargeTransferDirection = `${layerB} → ${layerA}`;
      dopingType = "electron";
    } else if (wfB > wfA + 0.1) {
      chargeTransferDirection = `${layerA} → ${layerB}`;
      dopingType = "hole";
    } else {
      chargeTransferDirection = "minimal";
      dopingType = "neutral";
    }
  } else {
    if (enA > enB + 0.1) {
      chargeTransferDirection = `${layerB} → ${layerA}`;
      dopingType = "electron";
    } else if (enB > enA + 0.1) {
      chargeTransferDirection = `${layerA} → ${layerB}`;
      dopingType = "hole";
    } else {
      chargeTransferDirection = "minimal";
      dopingType = "neutral";
    }
  }

  const enWeight = bothMetallic ? 0.15 : 0.3;
  const wfWeight = bothMetallic ? 0.35 : 0.15;
  const transferredChargeDensity = Math.min(1.0, (enMismatch * enWeight + wfMismatch * wfWeight));

  const enScoreWeight = bothMetallic ? 0.10 : 0.25;
  const wfScoreWeight = bothMetallic ? 0.25 : 0.10;
  let chargeTransferScore = Math.min(1.0,
    enMismatch * enScoreWeight +
    wfMismatch * wfScoreWeight +
    transferredChargeDensity * 0.4 +
    (enMismatch > 0.5 ? 0.15 : 0) +
    (wfMismatch > 1.0 ? 0.1 : 0)
  );

  if (chargeTransferScore > 0.8) {
    const excess = chargeTransferScore - 0.8;
    const stabilityPenalty = excess * 1.5;
    chargeTransferScore = chargeTransferScore - stabilityPenalty;
  }

  return {
    electronegativityMismatch: Number(enMismatch.toFixed(4)),
    workFunctionMismatch: Number(wfMismatch.toFixed(4)),
    chargeTransferDirection,
    transferredChargeDensity: Number(transferredChargeDensity.toFixed(4)),
    dopingType,
    chargeTransferScore: Number(chargeTransferScore.toFixed(4)),
  };
}

function computeInterfacePhonons(layerA: string, layerB: string, pressureGpa: number = 0): InterfacePhononAnalysis {
  const tdA = getAverageDebyeTemp(layerA);
  const tdB = getAverageDebyeTemp(layerB);
  const massA = getAverageMassForFormula(layerA);
  const massB = getAverageMassForFormula(layerB);
  const bulkA = getAverageBulkModulus(layerA, pressureGpa);
  const bulkB = getAverageBulkModulus(layerB, pressureGpa);

  const soundVelocityA = Math.sqrt(bulkA * 1e9 / (massA * 1.66e-27)) * 1e-3;
  const soundVelocityB = Math.sqrt(bulkB * 1e9 / (massB * 1.66e-27)) * 1e-3;
  const impedanceA = massA * soundVelocityA;
  const impedanceB = massB * soundVelocityB;

  const acousticMismatchRatio = impedanceA > 0 && impedanceB > 0
    ? Math.min(impedanceA, impedanceB) / Math.max(impedanceA, impedanceB)
    : 0.5;

  const acousticMismatchScore = 1.0 - acousticMismatchRatio;

  const tdRatio = Math.min(tdA, tdB) / Math.max(tdA, tdB, 1);
  const softModeCouplingScore = Math.min(1.0, (1.0 - tdRatio) * 0.6 + (tdA < 200 || tdB < 200 ? 0.3 : 0));

  const interfacePhononEnhancement = Math.min(2.0,
    1.0 + acousticMismatchScore * 0.4 + softModeCouplingScore * 0.3
  );

  let dominantPhononMode: string;
  if (softModeCouplingScore > 0.5) dominantPhononMode = "soft-optical";
  else if (acousticMismatchScore > 0.5) dominantPhononMode = "interface-acoustic";
  else dominantPhononMode = "bulk-like";

  const phononScore = Math.min(1.0,
    acousticMismatchScore * 0.35 +
    softModeCouplingScore * 0.35 +
    (interfacePhononEnhancement > 1.3 ? 0.2 : interfacePhononEnhancement > 1.1 ? 0.1 : 0)
  );

  return {
    acousticMismatchRatio: Number(acousticMismatchRatio.toFixed(4)),
    acousticMismatchScore: Number(acousticMismatchScore.toFixed(4)),
    softModeCouplingScore: Number(softModeCouplingScore.toFixed(4)),
    interfacePhononEnhancement: Number(interfacePhononEnhancement.toFixed(4)),
    dominantPhononMode,
    phononScore: Number(phononScore.toFixed(4)),
  };
}

function computeEpitaxialStrain(layerA: string, layerB: string, pressureGpa: number = 0): EpitaxialStrainAnalysis {
  const latticeA = getAverageLatticeConstant(layerA);
  const latticeB = getAverageLatticeConstant(layerB);

  const latticeMismatch = Math.abs(latticeA - latticeB) / Math.max(latticeA, latticeB, 0.01);

  const bulkA = getAverageBulkModulus(layerA, pressureGpa);
  const bulkB = getAverageBulkModulus(layerB, pressureGpa);
  const avgBulk = (bulkA + bulkB) / 2;

  const criticalThickness = latticeMismatch > 0.001
    ? Math.min(100, 0.5 / (latticeMismatch * latticeMismatch))
    : 100;

  let strainType: "compressive" | "tensile" | "matched";
  if (latticeMismatch < 0.01) strainType = "matched";
  else if (latticeA > latticeB) strainType = "compressive";
  else strainType = "tensile";

  const strainMagnitude = Math.min(0.1, latticeMismatch);

  let strainEnhancedCoupling = 0;
  if (strainMagnitude > 0.001 && strainMagnitude < 0.05) {
    strainEnhancedCoupling = Math.min(1.0, strainMagnitude * 15 * (1.0 + avgBulk / 200));
  } else if (strainMagnitude >= 0.05) {
    strainEnhancedCoupling = Math.max(0, 0.8 - (strainMagnitude - 0.05) * 10);
  }

  const strainScore = Math.min(1.0,
    strainEnhancedCoupling * 0.5 +
    (strainMagnitude > 0.005 && strainMagnitude < 0.04 ? 0.3 : 0) +
    (criticalThickness > 5 ? 0.15 : 0) +
    (avgBulk > 100 ? 0.05 : 0)
  );

  return {
    latticeMismatch: Number(latticeMismatch.toFixed(6)),
    criticalThickness: Number(criticalThickness.toFixed(2)),
    strainType,
    strainMagnitude: Number(strainMagnitude.toFixed(6)),
    strainEnhancedCoupling: Number(strainEnhancedCoupling.toFixed(4)),
    strainScore: Number(strainScore.toFixed(4)),
  };
}

function computeDimensionalConfinement(layerA: string, layerB: string): DimensionalConfinement {
  const elementsA = parseFormulaElements(layerA);
  const elementsB = parseFormulaElements(layerB);

  let confinementDimension = 2;

  const hasLayered = (els: string[]) =>
    (els.includes("Se") || els.includes("S") || els.includes("Te")) &&
    els.some(e => isTransitionMetal(e));
  const hasPerovskite = (els: string[]) =>
    els.includes("O") && els.some(e => ["Sr", "Ba", "Ca", "La"].includes(e)) &&
    els.some(e => ["Ti", "Al", "Mn", "Fe", "Ir", "Ru"].includes(e));

  if (hasLayered(elementsA) || hasLayered(elementsB)) confinementDimension = 2;
  if (hasPerovskite(elementsA) && hasPerovskite(elementsB)) confinementDimension = 2;

  const quantumWellWidth = 1.0;

  const twoDEnhancementFactor = confinementDimension <= 2
    ? Math.min(2.0, 1.0 + 0.5 / Math.max(0.5, quantumWellWidth))
    : 1.0;

  const confinementScore = Math.min(1.0,
    (confinementDimension <= 2 ? 0.4 : 0.1) +
    (twoDEnhancementFactor > 1.3 ? 0.3 : twoDEnhancementFactor > 1.1 ? 0.15 : 0) +
    (hasLayered(elementsA) || hasLayered(elementsB) ? 0.2 : 0) +
    (hasPerovskite(elementsA) || hasPerovskite(elementsB) ? 0.1 : 0)
  );

  return {
    confinementDimension,
    quantumWellWidth,
    twoDEnhancementFactor: Number(twoDEnhancementFactor.toFixed(4)),
    confinementScore: Number(confinementScore.toFixed(4)),
  };
}

interface KnownInterfaceSystem {
  layerA: string;
  layerB: string;
  knownTc: number;
  mechanism: string;
  label: string;
}

const KNOWN_INTERFACE_SYSTEMS: KnownInterfaceSystem[] = [
  { layerA: "FeSe", layerB: "SrTiO3", knownTc: 65, mechanism: "charge-transfer + interface-phonon", label: "FeSe/SrTiO3" },
  { layerA: "LaAlO3", layerB: "SrTiO3", knownTc: 0.3, mechanism: "interface-2DEG", label: "LAO/STO" },
  { layerA: "C", layerB: "C", knownTc: 1.7, mechanism: "flat-band (twist)", label: "Twisted bilayer graphene" },
];

function matchKnownSystem(layerA: string, layerB: string): KnownInterfaceSystem | null {
  const normA = layerA.replace(/\s/g, "");
  const normB = layerB.replace(/\s/g, "");
  for (const known of KNOWN_INTERFACE_SYSTEMS) {
    const kA = known.layerA.replace(/\s/g, "");
    const kB = known.layerB.replace(/\s/g, "");
    if ((normA === kA && normB === kB) || (normA === kB && normB === kA)) {
      return known;
    }
  }

  const elsA = parseFormulaElements(layerA);
  const elsB = parseFormulaElements(layerB);

  if (elsA.includes("Fe") && elsA.includes("Se") && elsB.includes("Sr") && elsB.includes("Ti") && elsB.includes("O")) {
    return KNOWN_INTERFACE_SYSTEMS[0];
  }
  if (elsB.includes("Fe") && elsB.includes("Se") && elsA.includes("Sr") && elsA.includes("Ti") && elsA.includes("O")) {
    return KNOWN_INTERFACE_SYSTEMS[0];
  }

  if (elsA.includes("La") && elsA.includes("Al") && elsA.includes("O") && elsB.includes("Sr") && elsB.includes("Ti") && elsB.includes("O")) {
    return KNOWN_INTERFACE_SYSTEMS[1];
  }
  if (elsB.includes("La") && elsB.includes("Al") && elsB.includes("O") && elsA.includes("Sr") && elsA.includes("Ti") && elsA.includes("O")) {
    return KNOWN_INTERFACE_SYSTEMS[1];
  }

  return null;
}

export function analyzeInterface(layerA: string, layerB: string, pressureGpa: number = 0): InterfaceAnalysis {
  const chargeTransfer = computeChargeTransfer(layerA, layerB);
  const phononCoupling = computeInterfacePhonons(layerA, layerB, pressureGpa);
  const epitaxialStrain = computeEpitaxialStrain(layerA, layerB, pressureGpa);
  const dimensionalConfinement = computeDimensionalConfinement(layerA, layerB);

  const knownMatch = matchKnownSystem(layerA, layerB);

  const weights = {
    chargeTransfer: 0.30,
    phonon: 0.25,
    strain: 0.20,
    confinement: 0.25,
  };

  let interfaceScScore = 
    chargeTransfer.chargeTransferScore * weights.chargeTransfer +
    phononCoupling.phononScore * weights.phonon +
    epitaxialStrain.strainScore * weights.strain +
    dimensionalConfinement.confinementScore * weights.confinement;

  interfaceScScore = Math.min(1.0, interfaceScScore);

  let mechanisms: string[] = [];
  if (chargeTransfer.chargeTransferScore > 0.3) mechanisms.push("charge-transfer");
  if (phononCoupling.phononScore > 0.3) mechanisms.push("interface-phonon");
  if (epitaxialStrain.strainScore > 0.3) mechanisms.push("strain-enhanced");
  if (dimensionalConfinement.confinementScore > 0.4) mechanisms.push("2D-confinement");
  if (mechanisms.length === 0) mechanisms.push("weak-coupling");

  let estimatedInterfaceTc = 0;
  if (knownMatch) {
    estimatedInterfaceTc = knownMatch.knownTc;
    interfaceScScore = Math.max(interfaceScScore, knownMatch.knownTc > 10 ? 0.7 : 0.3);
  } else {
    estimatedInterfaceTc = Number((interfaceScScore * 80 * phononCoupling.interfacePhononEnhancement * dimensionalConfinement.twoDEnhancementFactor).toFixed(1));
    estimatedInterfaceTc = Math.min(200, estimatedInterfaceTc);
  }

  let confidence = 0.4;
  if (knownMatch) confidence = 0.85;
  else {
    if (chargeTransfer.electronegativityMismatch > 0.3) confidence += 0.1;
    if (epitaxialStrain.latticeMismatch < 0.05) confidence += 0.1;
    if (phononCoupling.acousticMismatchScore > 0.3) confidence += 0.05;
    if (dimensionalConfinement.confinementDimension <= 2) confidence += 0.1;
    confidence = Math.min(0.75, confidence);
  }

  return {
    layerA,
    layerB,
    chargeTransfer,
    phononCoupling,
    epitaxialStrain,
    dimensionalConfinement,
    interfaceScScore: Number(interfaceScScore.toFixed(4)),
    estimatedInterfaceTc,
    mechanism: knownMatch ? knownMatch.mechanism : mechanisms.join(" + "),
    confidence: Number(confidence.toFixed(2)),
    knownSystemMatch: knownMatch ? knownMatch.label : null,
  };
}

const SUBSTRATE_POOL = [
  "SrTiO3", "LaAlO3", "MgO", "Al2O3", "BaTiO3", "KTaO3",
  "NdGaO3", "DyScO3", "TiO2", "SrVO3",
];

const FILM_POOL = [
  "FeSe", "FeTe", "NbSe2", "TaS2", "MoS2", "WTe2",
  "YBa2Cu3O7", "La2CuO4", "BiS2", "NbN",
  "LaNiO3", "SrRuO3", "NdNiO2", "PrNiO2",
  "Bi2Sr2CaCu2O8", "TlBaCaCuO",
];

export function generateHeterostructureCandidates(): HeterostructureCandidate[] {
  const candidates: HeterostructureCandidate[] = [];

  for (const film of FILM_POOL) {
    for (const substrate of SUBSTRATE_POOL) {
      if (film === substrate) continue;

      const analysis = analyzeInterface(film, substrate);

      if (analysis.interfaceScScore > 0.25) {
        candidates.push({
          layerA: film,
          layerB: substrate,
          stackingPattern: `${film}/${substrate}/${film}/${substrate}`,
          interfaceAnalysis: analysis,
          rank: 0,
        });
      }
    }
  }

  candidates.sort((a, b) => b.interfaceAnalysis.interfaceScScore - a.interfaceAnalysis.interfaceScScore);

  for (let i = 0; i < candidates.length; i++) {
    candidates[i].rank = i + 1;
  }

  return candidates.slice(0, 50);
}
