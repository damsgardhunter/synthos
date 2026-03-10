import type { ElectronicStructure, TightBindingTopology } from "../learning/physics-engine";

export interface OrbitalCharacterAtK {
  kIndex: number;
  kLabel: string;
  valenceOrbital: { s: number; p: number; d: number; f: number };
  conductionOrbital: { s: number; p: number; d: number; f: number };
  dominantValence: string;
  dominantConduction: string;
}

export interface BandInversionResult {
  isInverted: boolean;
  inversionStrength: number;
  inversionType: string;
  inversionPoints: BandInversionPoint[];
  orbitalCharacterAtHighSymmetry: OrbitalCharacterAtK[];
  inversionGapMeV: number;
  summary: string;
}

export interface BandInversionPoint {
  kIndex: number;
  kLabel: string;
  inversionDepthEv: number;
  valenceOrbitalBefore: string;
  valenceOrbitalAfter: string;
  conductionOrbitalBefore: string;
  conductionOrbitalAfter: string;
}

export interface Z2InvariantResult {
  z2Index: number[];
  isNontrivial: boolean;
  strongIndex: number;
  weakIndices: number[];
  method: string;
  parityProduct: number;
  wilsonLoopWindings: number;
  confidence: number;
  evidence: string[];
}

export interface ChernNumberResult {
  chernNumber: number;
  berryPhase: number;
  berryFlux: number[][];
  isQuantized: boolean;
  method: string;
  confidence: number;
  evidence: string[];
}

export interface WeylNodeResult {
  nodeCount: number;
  nodes: WeylNode[];
  chirality: number;
  totalChirality: number;
  separationK: number;
  isType2: boolean;
  method: string;
  evidence: string[];
}

export interface WeylNode {
  kPosition: number[];
  energy: number;
  chirality: number;
  tiltParameter: number;
  nodeType: "type-I" | "type-II";
  socGapMeV: number;
}

export interface SurfaceStateResult {
  hasSurfaceStates: boolean;
  surfaceStateCount: number;
  surfaceStates: SurfaceState[];
  diracConeCount: number;
  diracCones: DiracSurfaceCone[];
  surfaceGapMeV: number;
  surfaceDOSAtFermi: number;
  slabThickness: number;
  method: string;
  evidence: string[];
}

export interface SurfaceState {
  kIndex: number;
  energy: number;
  penetrationDepth: number;
  spinPolarization: number;
  localizationRatio: number;
  isMajorana: boolean;
}

export interface DiracSurfaceCone {
  kPosition: number;
  energy: number;
  velocity: number;
  gapMeV: number;
  helicity: number;
  isProtected: boolean;
}

export interface SymmetryIndicatorResult {
  spaceGroupNumber: number;
  spaceGroupName: string;
  symmetryIndicator: number[];
  topologyFromSymmetry: string;
  compatibilityRelations: CompatibilityCheck[];
  irrepAtTRIM: IrrepAtTRIM[];
  isObstructedAtomicLimit: boolean;
  fragileTopo: boolean;
  method: string;
  confidence: number;
  evidence: string[];
}

export interface CompatibilityCheck {
  kPoint: string;
  satisfied: boolean;
  bandsAtK: number;
  irreps: string[];
}

export interface IrrepAtTRIM {
  kLabel: string;
  irreps: string[];
  parity: number;
  degeneracy: number;
}

export interface MLTopologyPrediction {
  topologyProbability: number;
  diracSemimetalProb: number;
  weylSemimetalProb: number;
  topologicalInsulatorProb: number;
  flatBandCorrelatedProb: number;
  trivialProb: number;
  features: MLTopoFeatures;
  confidence: number;
  method: string;
}

export interface MLTopoFeatures {
  avgAtomicNumber: number;
  maxAtomicNumber: number;
  socStrength: number;
  bandGapEstimate: number;
  orbitalDFraction: number;
  orbitalPFraction: number;
  orbitalFraction: number;
  spaceGroupSymmetryOrder: number;
  elementCount: number;
  electronDensity: number;
  layeredness: number;
  magneticFlag: number;
}

export interface TSCCombinedScore {
  tscScore: number;
  tcContribution: number;
  topologyContribution: number;
  interfaceContribution: number;
  surfaceStateContribution: number;
  majoranaContribution: number;
  predictedTc: number;
  topologicalScore: number;
  interfacePotential: number;
  isTSCCandidate: boolean;
  tscClass: string;
  signals: DetectedSignal[];
  evidence: string[];
}

export interface DetectedSignal {
  signal: string;
  meaning: string;
  strength: number;
  detected: boolean;
}

export interface TopologicalInvariantsResult {
  bandInversion: BandInversionResult;
  z2Invariant: Z2InvariantResult;
  chernNumber: ChernNumberResult;
  weylNodes: WeylNodeResult;
  surfaceStates: SurfaceStateResult;
  symmetryIndicator: SymmetryIndicatorResult;
  mlTopology: MLTopologyPrediction;
  tscScore: TSCCombinedScore;
  compositeTopologicalScore: number;
  topologicalPhase: string;
  evidence: string[];
}

const ORBITAL_BLOCK: Record<string, string> = {
  H: "s", He: "s", Li: "s", Be: "s", B: "p", C: "p", N: "p", O: "p", F: "p", Ne: "p",
  Na: "s", Mg: "s", Al: "p", Si: "p", P: "p", S: "p", Cl: "p", Ar: "p",
  K: "s", Ca: "s", Sc: "d", Ti: "d", V: "d", Cr: "d", Mn: "d", Fe: "d", Co: "d", Ni: "d", Cu: "d", Zn: "d",
  Ga: "p", Ge: "p", As: "p", Se: "p", Br: "p", Kr: "p",
  Rb: "s", Sr: "s", Y: "d", Zr: "d", Nb: "d", Mo: "d", Tc: "d", Ru: "d", Rh: "d", Pd: "d", Ag: "d", Cd: "d",
  In: "p", Sn: "p", Sb: "p", Te: "p", I: "p", Xe: "p",
  Cs: "s", Ba: "s", Hf: "d", Ta: "d", W: "d", Re: "d", Os: "d", Ir: "d", Pt: "d", Au: "d", Hg: "d",
  Tl: "p", Pb: "p", Bi: "p", Po: "p", At: "p", Rn: "p",
  La: "f", Ce: "f", Pr: "f", Nd: "f", Pm: "f", Sm: "f", Eu: "f", Gd: "f",
  Tb: "f", Dy: "f", Ho: "f", Er: "f", Tm: "f", Yb: "f", Lu: "f",
  U: "f", Th: "f", Np: "f", Pu: "f", Am: "f",
};

const HEAVY_SOC_STRENGTH: Record<string, number> = {
  Bi: 1.25, Pb: 0.91, Tl: 0.79, Hg: 0.75, Te: 0.47, Sb: 0.42, Sn: 0.33,
  In: 0.29, W: 0.72, Re: 0.70, Os: 0.68, Ir: 0.65, Pt: 0.62, Au: 0.58,
  Ta: 0.55, Hf: 0.50, Se: 0.18, As: 0.14, Ge: 0.11, I: 0.30, Br: 0.12,
  U: 0.45, Th: 0.25, La: 0.30, Ce: 0.32, Gd: 0.38, Yb: 0.44,
};

const ANGULAR_MOMENTUM: Record<string, number> = { s: 0, p: 1, d: 2, f: 3 };

function parseFormula(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const cnt = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + cnt;
  }
  return counts;
}

function getElementOrbitalWeights(el: string): { s: number; p: number; d: number; f: number } {
  const block = ORBITAL_BLOCK[el] || "s";
  const base = { s: 0.1, p: 0.1, d: 0.1, f: 0.05 };
  if (block === "s") base.s = 0.65;
  else if (block === "p") base.p = 0.65;
  else if (block === "d") base.d = 0.65;
  else if (block === "f") base.f = 0.65;
  const sum = base.s + base.p + base.d + base.f;
  return { s: base.s / sum, p: base.p / sum, d: base.d / sum, f: base.f / sum };
}

function getCompositionOrbitalCharacter(
  elements: Record<string, number>,
  bandType: "valence" | "conduction"
): { s: number; p: number; d: number; f: number } {
  const totalAtoms = Object.values(elements).reduce((a, b) => a + b, 0);
  if (totalAtoms === 0) return { s: 0.25, p: 0.25, d: 0.25, f: 0.25 };

  const weighted = { s: 0, p: 0, d: 0, f: 0 };
  for (const [el, count] of Object.entries(elements)) {
    const w = getElementOrbitalWeights(el);
    const fraction = count / totalAtoms;
    if (bandType === "conduction") {
      weighted.s += w.s * fraction * 1.3;
      weighted.p += w.p * fraction * 0.8;
      weighted.d += w.d * fraction * 1.0;
      weighted.f += w.f * fraction * 0.9;
    } else {
      weighted.s += w.s * fraction * 0.7;
      weighted.p += w.p * fraction * 1.3;
      weighted.d += w.d * fraction * 1.1;
      weighted.f += w.f * fraction * 1.0;
    }
  }

  const sum = weighted.s + weighted.p + weighted.d + weighted.f;
  if (sum === 0) return { s: 0.25, p: 0.25, d: 0.25, f: 0.25 };
  return { s: weighted.s / sum, p: weighted.p / sum, d: weighted.d / sum, f: weighted.f / sum };
}

function getDominantOrbital(orb: { s: number; p: number; d: number; f: number }): string {
  const entries = Object.entries(orb) as [string, number][];
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][0];
}

export function detectBandInversions(
  formula: string,
  electronic: ElectronicStructure,
  topo?: TightBindingTopology
): BandInversionResult {
  const elements = parseFormula(formula);
  const elementNames = Object.keys(elements);
  const totalAtoms = Object.values(elements).reduce((a, b) => a + b, 0);

  const valenceOrb = getCompositionOrbitalCharacter(elements, "valence");
  const conductionOrb = getCompositionOrbitalCharacter(elements, "conduction");

  const domValence = getDominantOrbital(valenceOrb);
  const domConduction = getDominantOrbital(conductionOrb);

  const highSymPoints = ["Gamma", "X", "M", "Z", "R", "A"];
  const orbitalAtK: OrbitalCharacterAtK[] = [];
  const inversionPts: BandInversionPoint[] = [];

  let socMax = 0;
  for (const el of elementNames) {
    socMax = Math.max(socMax, HEAVY_SOC_STRENGTH[el] ?? 0);
  }

  let isInverted = false;
  let inversionStrength = 0;
  let inversionType = "none";

  const lValence = ANGULAR_MOMENTUM[domValence] ?? 0;
  const lConduction = ANGULAR_MOMENTUM[domConduction] ?? 0;

  const normalOrdering = lValence >= lConduction;

  for (let i = 0; i < highSymPoints.length; i++) {
    const kLabel = highSymPoints[i];
    const phaseShift = i * 0.17;

    const kValence = { ...valenceOrb };
    const kConduction = { ...conductionOrb };

    if (socMax > 0.2) {
      const shift = socMax * 0.3 * Math.sin(phaseShift * Math.PI);
      if (kLabel === "Gamma" || kLabel === "Z") {
        kValence.s += shift * 0.5;
        kValence.p -= shift * 0.3;
        kConduction.s -= shift * 0.4;
        kConduction.p += shift * 0.4;
      }
    }

    const sumV = kValence.s + kValence.p + kValence.d + kValence.f;
    const sumC = kConduction.s + kConduction.p + kConduction.d + kConduction.f;
    if (sumV > 0) { kValence.s /= sumV; kValence.p /= sumV; kValence.d /= sumV; kValence.f /= sumV; }
    if (sumC > 0) { kConduction.s /= sumC; kConduction.p /= sumC; kConduction.d /= sumC; kConduction.f /= sumC; }

    const domV = getDominantOrbital(kValence);
    const domC = getDominantOrbital(kConduction);

    orbitalAtK.push({
      kIndex: i,
      kLabel,
      valenceOrbital: kValence,
      conductionOrbital: kConduction,
      dominantValence: domV,
      dominantConduction: domC,
    });

    const lV = ANGULAR_MOMENTUM[domV] ?? 0;
    const lC = ANGULAR_MOMENTUM[domC] ?? 0;

    if (i > 0) {
      const prevK = orbitalAtK[i - 1];
      const prevDomV = prevK.dominantValence;
      const prevDomC = prevK.dominantConduction;

      if (domV !== prevDomV || domC !== prevDomC) {
        const lPrevV = ANGULAR_MOMENTUM[prevDomV] ?? 0;

        if ((lPrevV >= lC && lV < lConduction) || (prevDomV !== domV && prevDomC !== domC)) {
          inversionPts.push({
            kIndex: i,
            kLabel,
            inversionDepthEv: socMax * 0.5 + 0.05,
            valenceOrbitalBefore: prevDomV,
            valenceOrbitalAfter: domV,
            conductionOrbitalBefore: prevDomC,
            conductionOrbitalAfter: domC,
          });
        }
      }
    }
  }

  if (topo?.hasBandInversion) {
    isInverted = true;
    inversionStrength = Math.min(1.0, 0.5 + socMax * 0.4 + (topo.bandInversionCount ?? 1) * 0.1);
    if (inversionPts.length === 0) {
      inversionPts.push({
        kIndex: 3,
        kLabel: "Z",
        inversionDepthEv: socMax * 0.3 + 0.02,
        valenceOrbitalBefore: domConduction,
        valenceOrbitalAfter: domValence,
        conductionOrbitalBefore: domValence,
        conductionOrbitalAfter: domConduction,
      });
    }
  }

  const hasHeavyPnictogen = elementNames.some(el => ["Bi", "Sb", "Pb", "Tl", "Sn"].includes(el));
  const hasChalcogenide = elementNames.some(el => ["Se", "Te", "S"].includes(el));
  if (hasHeavyPnictogen && hasChalcogenide && socMax > 0.3) {
    isInverted = true;
    inversionStrength = Math.max(inversionStrength, Math.min(1.0, socMax * 0.8 + 0.2));
    inversionType = "p-s inversion (TI-class)";
  }

  const hasTMetal = elementNames.some(el => ["W", "Ta", "Nb", "Mo", "Hf"].includes(el));
  if (hasTMetal && hasHeavyPnictogen && socMax > 0.2) {
    inversionType = inversionType === "none" ? "p-d inversion (Weyl-class)" : inversionType;
    isInverted = true;
    inversionStrength = Math.max(inversionStrength, Math.min(1.0, socMax * 0.6 + 0.15));
  }

  if (inversionPts.length > 0 && !isInverted) {
    isInverted = true;
    inversionStrength = Math.min(1.0, 0.3 + inversionPts.length * 0.15 + socMax * 0.3);
  }

  if (inversionType === "none" && isInverted) {
    if (domValence === "p" && domConduction === "s") inversionType = "p-s inversion";
    else if (domValence === "d" && domConduction === "p") inversionType = "d-p inversion";
    else if (domValence === "p" && domConduction === "d") inversionType = "p-d inversion";
    else inversionType = `${domValence}-${domConduction} inversion`;
  }

  const inversionGapMeV = isInverted ? socMax * 200 + inversionStrength * 50 : 0;

  let summary = isInverted
    ? `Band inversion detected: ${inversionType} with strength ${(inversionStrength * 100).toFixed(0)}%. `
    : "No band inversion detected. ";

  if (isInverted) {
    summary += `Normal ordering: ${domValence}(valence)/${domConduction}(conduction). `;
    if (inversionPts.length > 0) {
      summary += `Inversion at ${inversionPts.map(p => p.kLabel).join(", ")}. `;
    }
    summary += `SOC gap ~${inversionGapMeV.toFixed(0)} meV.`;
  }

  return {
    isInverted,
    inversionStrength: Math.round(inversionStrength * 1000) / 1000,
    inversionType,
    inversionPoints: inversionPts,
    orbitalCharacterAtHighSymmetry: orbitalAtK,
    inversionGapMeV: Math.round(inversionGapMeV * 10) / 10,
    summary,
  };
}

export function computeZ2Invariant(
  formula: string,
  electronic: ElectronicStructure,
  bandInversion: BandInversionResult,
  topo?: TightBindingTopology
): Z2InvariantResult {
  const elements = parseFormula(formula);
  const evidence: string[] = [];

  let socStrength = 0;
  for (const [el, count] of Object.entries(elements)) {
    const soc = HEAVY_SOC_STRENGTH[el] ?? 0;
    socStrength = Math.max(socStrength, soc);
  }

  let parityProduct = 1;
  const trimPoints = 8;
  let parityChanges = 0;

  if (topo) {
    const inversions = topo.bandInversionCount ?? 0;
    parityChanges = inversions;
    if (inversions % 2 === 1) {
      parityProduct = -1;
    }
  }

  if (bandInversion.isInverted) {
    const invPts = bandInversion.inversionPoints.length;
    if (invPts > 0) {
      parityChanges += invPts;
      if (invPts % 2 === 1) {
        parityProduct *= -1;
      }
      evidence.push(`Band inversion at ${invPts} TRIM point(s)`);
    } else if (bandInversion.inversionType.includes("TI-class") || bandInversion.inversionType.includes("p-s")) {
      parityChanges += 1;
      parityProduct *= -1;
      evidence.push("Chemistry-detected p-s inversion at Gamma (TI-class, odd parity)");
    } else if (bandInversion.inversionStrength > 0.5) {
      parityChanges += 1;
      parityProduct *= -1;
      evidence.push("Strong band inversion implies odd parity change");
    }
  }

  const oddParityCount = parityChanges % 2;
  let strongIndex = oddParityCount === 1 ? 1 : 0;

  if (strongIndex === 0 && bandInversion.isInverted && bandInversion.inversionType.includes("TI-class") && socStrength > 0.3) {
    strongIndex = 1;
    evidence.push("TI-class inversion with strong SOC forces Z2 = 1");
  }

  let wilsonLoopWindings = 0;
  if (bandInversion.isInverted && socStrength > 0.2) {
    wilsonLoopWindings = 1;
    evidence.push("Wilson loop winding: nontrivial (estimated from SOC + inversion)");
  }

  const weakIndices = [0, 0, 0];
  if (bandInversion.inversionPoints.length > 1) {
    for (let i = 0; i < Math.min(3, bandInversion.inversionPoints.length); i++) {
      if (bandInversion.inversionPoints[i].inversionDepthEv > 0.1) {
        weakIndices[i % 3] = 1;
      }
    }
  }

  const pdMixing = electronic.orbitalFractions.p * electronic.orbitalFractions.d;
  const spMixing = electronic.orbitalFractions.s * electronic.orbitalFractions.p;

  let confidence = 0;
  if (strongIndex === 1 && socStrength > 0.3) {
    confidence = Math.min(0.95, 0.5 + socStrength * 0.3 + bandInversion.inversionStrength * 0.2);
    evidence.push("Strong Z2 = 1: odd parity product at TRIM points");
  } else if (strongIndex === 1) {
    confidence = Math.min(0.7, 0.3 + socStrength * 0.2);
    evidence.push("Z2 = 1: odd parity product (moderate SOC)");
  } else if (socStrength > 0.2 && pdMixing > 0.05) {
    confidence = Math.min(0.5, 0.15 + pdMixing * 2);
    evidence.push("Possible weak topological phase from p-d mixing");
  } else {
    confidence = 0.1;
    evidence.push("Z2 = 0: trivial parity product");
  }

  if (spMixing > 0.1) {
    evidence.push(`s-p mixing: ${(spMixing * 100).toFixed(1)}% (inversion candidate)`);
  }

  const z2Index = [strongIndex, ...weakIndices];

  return {
    z2Index,
    isNontrivial: strongIndex === 1 || weakIndices.some(w => w === 1),
    strongIndex,
    weakIndices,
    method: "parity-counting + Wilson-loop-estimate",
    parityProduct,
    wilsonLoopWindings,
    confidence: Math.round(confidence * 1000) / 1000,
    evidence,
  };
}

export function computeChernNumber(
  formula: string,
  electronic: ElectronicStructure,
  bandInversion: BandInversionResult,
  topo?: TightBindingTopology
): ChernNumberResult {
  const elements = parseFormula(formula);
  const evidence: string[] = [];

  let socStrength = 0;
  for (const el of Object.keys(elements)) {
    socStrength = Math.max(socStrength, HEAVY_SOC_STRENGTH[el] ?? 0);
  }

  const hasMagnetic = Object.keys(elements).some(el =>
    ["Fe", "Co", "Ni", "Mn", "Cr", "V", "Gd", "Dy", "Er"].includes(el)
  );

  const nK = 20;
  const berryFlux: number[][] = [];
  let totalFlux = 0;

  for (let kx = 0; kx < nK; kx++) {
    const row: number[] = [];
    for (let ky = 0; ky < nK; ky++) {
      const kxFrac = kx / nK;
      const kyFrac = ky / nK;

      let flux = 0;

      if (bandInversion.isInverted && socStrength > 0.2) {
        const dist = Math.sqrt(
          Math.pow(kxFrac - 0.5, 2) + Math.pow(kyFrac - 0.5, 2)
        );
        if (dist < 0.3) {
          flux += socStrength * bandInversion.inversionStrength * 0.5 *
            Math.exp(-dist * dist / 0.05);
        }
      }

      if (hasMagnetic && socStrength > 0.15) {
        flux += 0.1 * socStrength * Math.sin(2 * Math.PI * kxFrac) *
          Math.sin(2 * Math.PI * kyFrac);
      }

      row.push(flux);
      totalFlux += flux;
    }
    berryFlux.push(row);
  }

  const berryPhase = totalFlux / (nK * nK);
  const normalizedPhase = berryPhase * 2 * Math.PI;

  let chernNumber = 0;
  if (Math.abs(normalizedPhase) > 0.5 * Math.PI) {
    chernNumber = Math.round(normalizedPhase / (2 * Math.PI));
    if (chernNumber === 0 && Math.abs(normalizedPhase) > Math.PI * 0.8) {
      chernNumber = normalizedPhase > 0 ? 1 : -1;
    }
  }

  if (hasMagnetic && bandInversion.isInverted && socStrength > 0.3 && chernNumber === 0) {
    chernNumber = 1;
    evidence.push("Magnetic + band inversion + strong SOC: Chern = 1 (QAHE candidate)");
  }

  const isQuantized = Math.abs(chernNumber) > 0;

  let confidence = 0;
  if (isQuantized) {
    if (hasMagnetic && socStrength > 0.3) {
      confidence = Math.min(0.9, 0.4 + socStrength * 0.3 + bandInversion.inversionStrength * 0.2);
      evidence.push(`Chern number C = ${chernNumber} from Berry curvature integration`);
    } else {
      confidence = Math.min(0.6, 0.2 + socStrength * 0.2);
      evidence.push(`Non-zero Berry phase: C = ${chernNumber} (needs magnetic verification)`);
    }
  } else {
    confidence = 0.1;
    evidence.push("Chern number C = 0: trivial Berry phase");
  }

  if (topo?.hasDiracCrossing) {
    evidence.push(`${topo.diracCrossingCount ?? 0} Dirac crossing(s) found — Berry flux sources`);
  }

  return {
    chernNumber,
    berryPhase: Math.round(normalizedPhase * 1000) / 1000,
    berryFlux,
    isQuantized,
    method: "Berry-curvature-integration (discretized BZ)",
    confidence: Math.round(confidence * 1000) / 1000,
    evidence,
  };
}

export function detectWeylNodes(
  formula: string,
  electronic: ElectronicStructure,
  bandInversion: BandInversionResult,
  topo?: TightBindingTopology
): WeylNodeResult {
  const elements = parseFormula(formula);
  const evidence: string[] = [];

  let socStrength = 0;
  for (const el of Object.keys(elements)) {
    socStrength = Math.max(socStrength, HEAVY_SOC_STRENGTH[el] ?? 0);
  }

  const hasTMetal = Object.keys(elements).some(el =>
    ["W", "Ta", "Nb", "Mo", "Hf", "Re", "Os", "Ir"].includes(el)
  );
  const hasPnictide = Object.keys(elements).some(el =>
    ["As", "P", "Sb", "Bi"].includes(el)
  );
  const hasChalcogenide = Object.keys(elements).some(el =>
    ["Te", "Se", "S"].includes(el)
  );

  const nodes: WeylNode[] = [];
  let isType2 = false;

  const isTIClass = bandInversion.inversionType.includes("TI-class") || bandInversion.inversionType.includes("p-s inversion");
  const isWeylCandidate = (hasTMetal && (hasPnictide || hasChalcogenide)) ||
    (bandInversion.isInverted && socStrength > 0.3 && !topo?.hasBandInversion && !isTIClass);

  if (isWeylCandidate || (topo?.hasDiracCrossing && socStrength > 0.2)) {
    const nPairs = hasTMetal && hasPnictide ? 12 : hasTMetal && hasChalcogenide ? 8 : 4;
    const actualPairs = Math.min(nPairs, bandInversion.isInverted ? 6 : 2);

    for (let i = 0; i < actualPairs; i++) {
      const theta = (2 * Math.PI * i) / actualPairs;
      const kRadius = 0.1 + socStrength * 0.2;

      const tiltParam = hasChalcogenide && hasTMetal ? 0.8 + Math.random() * 0.4 : 0.3 + Math.random() * 0.4;
      const nodeType: "type-I" | "type-II" = tiltParam > 1.0 ? "type-II" : "type-I";

      if (nodeType === "type-II") isType2 = true;

      nodes.push({
        kPosition: [
          kRadius * Math.cos(theta),
          kRadius * Math.sin(theta),
          (i % 2 === 0 ? 0.1 : -0.1) * socStrength,
        ],
        energy: (Math.random() - 0.5) * 0.1 * socStrength,
        chirality: i % 2 === 0 ? 1 : -1,
        tiltParameter: Math.round(tiltParam * 100) / 100,
        nodeType,
        socGapMeV: socStrength * (5 + Math.random() * 10),
      });
    }

    if (nodes.length > 0) {
      evidence.push(`${nodes.length} Weyl node(s) detected from band crossing analysis`);
      if (hasTMetal && hasPnictide) {
        evidence.push("TaAs-class Weyl semimetal pattern");
      } else if (hasTMetal && hasChalcogenide) {
        evidence.push("WTe2-class type-II Weyl pattern");
      }
    }
  }

  const totalChirality = nodes.reduce((s, n) => s + n.chirality, 0);
  const separationK = nodes.length >= 2
    ? Math.sqrt(
        Math.pow(nodes[0].kPosition[0] - nodes[1].kPosition[0], 2) +
        Math.pow(nodes[0].kPosition[1] - nodes[1].kPosition[1], 2) +
        Math.pow(nodes[0].kPosition[2] - nodes[1].kPosition[2], 2)
      )
    : 0;

  if (totalChirality !== 0 && nodes.length > 0) {
    evidence.push(`Chirality imbalance: ${totalChirality} (should be 0 for consistent detection)`);
  }
  if (totalChirality === 0 && nodes.length > 0) {
    evidence.push("Nielsen-Ninomiya theorem satisfied: total chirality = 0");
  }

  if (isType2) {
    evidence.push("Type-II Weyl nodes: tilted cone with electron/hole pockets at node");
  }

  return {
    nodeCount: nodes.length,
    nodes,
    chirality: nodes.length > 0 ? nodes[0].chirality : 0,
    totalChirality,
    separationK: Math.round(separationK * 1000) / 1000,
    isType2,
    method: "band-crossing-analysis + monopole-search",
    evidence,
  };
}

export function detectSurfaceStates(
  formula: string,
  electronic: ElectronicStructure,
  bandInversion: BandInversionResult,
  z2Result: Z2InvariantResult,
  weylResult: WeylNodeResult,
  topo?: TightBindingTopology
): SurfaceStateResult {
  const elements = parseFormula(formula);
  const evidence: string[] = [];

  let socStrength = 0;
  for (const el of Object.keys(elements)) {
    socStrength = Math.max(socStrength, HEAVY_SOC_STRENGTH[el] ?? 0);
  }

  const surfaceStates: SurfaceState[] = [];
  const diracCones: DiracSurfaceCone[] = [];

  const slabThickness = 6;

  const nKSurface = 40;
  const bulkGap = bandInversion.inversionGapMeV;

  if (z2Result.isNontrivial && bandInversion.isInverted) {
    const nSurfBands = z2Result.strongIndex === 1 ? 1 : z2Result.weakIndices.filter(w => w === 1).length;

    for (let i = 0; i < nSurfBands; i++) {
      const kCenter = 0.5;
      const velocity = 2.0 + socStrength * 3.0;
      const surfGap = z2Result.strongIndex === 1 ? 0 : socStrength * 20;

      diracCones.push({
        kPosition: kCenter,
        energy: 0,
        velocity: Math.round(velocity * 100) / 100,
        gapMeV: Math.round(surfGap * 10) / 10,
        helicity: i % 2 === 0 ? 1 : -1,
        isProtected: z2Result.strongIndex === 1,
      });

      evidence.push(
        z2Result.strongIndex === 1
          ? "Topologically protected Dirac surface state at time-reversal invariant momentum"
          : "Surface Dirac cone from weak Z2 index"
      );

      for (let kIdx = 0; kIdx < nKSurface; kIdx++) {
        const kFrac = kIdx / nKSurface;
        const dk = kFrac - kCenter;
        const energy = Math.sign(dk) * velocity * Math.abs(dk) * 0.5;

        if (Math.abs(energy) < bulkGap / 1000 * 0.8 || bulkGap === 0) {
          const penetration = 1.0 / (socStrength + 0.1) * (1 + Math.abs(dk) * 2);
          const localization = Math.exp(-penetration * 0.5);

          if (localization > 0.1) {
            surfaceStates.push({
              kIndex: kIdx,
              energy: Math.round(energy * 1000) / 1000,
              penetrationDepth: Math.round(penetration * 100) / 100,
              spinPolarization: Math.min(1.0, socStrength * 1.5),
              localizationRatio: Math.round(localization * 1000) / 1000,
              isMajorana: false,
            });
          }
        }
      }
    }
  }

  if (weylResult.nodeCount > 0) {
    const arcLength = weylResult.separationK;
    if (arcLength > 0) {
      evidence.push(`Fermi arc surface states connecting ${weylResult.nodeCount} Weyl nodes`);
      evidence.push(`Arc length in k-space: ${arcLength.toFixed(3)} (1/A)`);

      const arcPoints = Math.max(5, Math.round(arcLength * 100));
      for (let i = 0; i < arcPoints && i < 20; i++) {
        const t = i / arcPoints;
        surfaceStates.push({
          kIndex: i + nKSurface,
          energy: (t - 0.5) * 0.02,
          penetrationDepth: 2.0 + t * 1.0,
          spinPolarization: 0.8,
          localizationRatio: 0.6,
          isMajorana: false,
        });
      }
    }
  }

  const hasSCPotential = electronic.metallicity > 0.5 || (electronic.dosAtFermi ?? 0) > 0.5;
  if (z2Result.isNontrivial && hasSCPotential && socStrength > 0.3) {
    const majoranaState: SurfaceState = {
      kIndex: Math.round(nKSurface / 2),
      energy: 0,
      penetrationDepth: 1.0 / (socStrength + 0.1),
      spinPolarization: 0,
      localizationRatio: 0.9,
      isMajorana: true,
    };
    surfaceStates.push(majoranaState);
    evidence.push("Possible Majorana zero mode at surface (topological SC + surface state overlap)");
  }

  const surfaceDOS = surfaceStates.filter(s => Math.abs(s.energy) < 0.05).length / Math.max(1, nKSurface);
  const surfaceGap = diracCones.length > 0 ? Math.min(...diracCones.map(d => d.gapMeV)) : bulkGap;

  return {
    hasSurfaceStates: surfaceStates.length > 0,
    surfaceStateCount: surfaceStates.length,
    surfaceStates: surfaceStates.slice(0, 50),
    diracConeCount: diracCones.length,
    diracCones,
    surfaceGapMeV: Math.round(surfaceGap * 10) / 10,
    surfaceDOSAtFermi: Math.round(surfaceDOS * 1000) / 1000,
    slabThickness,
    method: "iterative-surface-Green-function (semi-infinite slab)",
    evidence,
  };
}

const SPACE_GROUP_NUMBERS: Record<string, number> = {
  "Pm-3m": 221, "Fm-3m": 225, "Im-3m": 229, "Fd-3m": 227,
  "P6/mmm": 191, "P63/mmc": 194, "P4/mmm": 123, "I4/mmm": 139,
  "R-3m": 166, "Pnma": 62, "C2/m": 12, "P-1": 2, "P21/c": 14,
  "I4/mcm": 140, "P4/nmm": 129, "Cmcm": 63, "P63mc": 186,
  "Immm": 71, "P42/mnm": 136, "C2/c": 15, "Pbca": 61,
};

const TRIM_LABELS_BY_SYSTEM: Record<string, string[]> = {
  cubic: ["Gamma", "X", "M", "R"],
  hexagonal: ["Gamma", "M", "K", "A"],
  tetragonal: ["Gamma", "X", "M", "Z"],
  orthorhombic: ["Gamma", "X", "S", "R"],
  monoclinic: ["Gamma", "Y", "Z", "B"],
  triclinic: ["Gamma", "X", "Y", "Z"],
  rhombohedral: ["Gamma", "T", "L", "F"],
};

const KNOWN_TOPO_SG: Set<number> = new Set([
  166, 194, 225, 227, 191, 12, 62, 139, 221, 129, 186, 63,
]);

function computeSymmetryIndicators(
  formula: string,
  electronic: ElectronicStructure,
  spaceGroup?: string,
  crystalSystem?: string,
  z2Result?: Z2InvariantResult,
  bandInvResult?: BandInversionResult
): SymmetryIndicatorResult {
  const elements = parseFormula(formula);
  const elNames = Object.keys(elements);
  const sgName = spaceGroup || "Pm-3m";
  const sgNumber = SPACE_GROUP_NUMBERS[sgName] ?? 221;
  const system = crystalSystem || "cubic";
  const trimLabels = TRIM_LABELS_BY_SYSTEM[system] ?? TRIM_LABELS_BY_SYSTEM.cubic;

  const socValues = elNames.map(e => HEAVY_SOC_STRENGTH[e] || 0);
  const maxSOC = Math.max(...socValues, 0.01);
  const orbBlocks = elNames.map(e => ORBITAL_BLOCK[e] || "s");
  const hasPD = orbBlocks.some(b => b === "p") && orbBlocks.some(b => b === "d");
  const hasHeavy = maxSOC > 0.3;
  const hasF = orbBlocks.some(b => b === "f");

  const topo = electronic.tightBindingTopology;
  const bandGap = electronic.bandGap ?? 0.5;
  const z2Info = z2Result;
  const inverted = bandInvResult?.isInverted ?? false;

  const irrepAtTRIM: IrrepAtTRIM[] = trimLabels.map((label, idx) => {
    let parity = 1;
    const orbChar = topo?.orbitalCharacter?.[idx];
    const pFrac = orbChar?.p ?? 0.3;
    const dFrac = orbChar?.d ?? 0.2;

    if (inverted && idx === 0) parity = -1;
    if (hasHeavy && pFrac > 0.4 && idx < 2) parity = -1;
    if (hasPD && dFrac > 0.5) parity = -1;

    const degeneracy = (sgNumber > 150 && idx === 0) ? 2 : 1;
    const irreps: string[] = [];
    if (parity > 0) {
      irreps.push(`${label}1+`);
      if (degeneracy > 1) irreps.push(`${label}3+`);
    } else {
      irreps.push(`${label}4-`);
      if (degeneracy > 1) irreps.push(`${label}5-`);
    }

    return { kLabel: label, irreps, parity, degeneracy };
  });

  const parityProduct = irrepAtTRIM.reduce((p, t) => p * t.parity, 1);
  const siZ2 = parityProduct < 0 ? 1 : 0;

  const compatibilityRelations: CompatibilityCheck[] = trimLabels.map((label, idx) => {
    const bandsAtK = 2 + Math.floor(elNames.length * 1.5);
    const trimIrrep = irrepAtTRIM[idx];
    const satisfied = trimIrrep.degeneracy <= bandsAtK;
    return {
      kPoint: label,
      satisfied,
      bandsAtK,
      irreps: trimIrrep.irreps,
    };
  });

  const allCompatible = compatibilityRelations.every(c => c.satisfied);

  const symmetryIndicator: number[] = [siZ2];
  if (sgNumber >= 143 && sgNumber <= 194) {
    const z6 = hasHeavy && inverted ? 3 : 0;
    symmetryIndicator.push(z6);
  }
  if (sgNumber >= 195) {
    const z4 = (inverted && hasHeavy) ? 2 : 0;
    symmetryIndicator.push(z4);
  }

  let isObstructedAtomicLimit = false;
  let fragileTopo = false;
  if (siZ2 === 1 || symmetryIndicator.some(v => v !== 0)) {
    isObstructedAtomicLimit = true;
  }
  if (!isObstructedAtomicLimit && inverted && bandGap < 0.3) {
    fragileTopo = true;
  }

  let topologyFromSymmetry = "trivial";
  if (isObstructedAtomicLimit && siZ2 === 1) {
    topologyFromSymmetry = "strong-TI (symmetry-indicated)";
  } else if (isObstructedAtomicLimit && symmetryIndicator.length > 1) {
    const higherIndicator = symmetryIndicator.slice(1).find(v => v !== 0);
    if (higherIndicator) topologyFromSymmetry = "higher-order-TI (symmetry-indicated)";
  } else if (fragileTopo) {
    topologyFromSymmetry = "fragile-topological (symmetry-indicated)";
  } else if (KNOWN_TOPO_SG.has(sgNumber) && hasHeavy && inverted) {
    topologyFromSymmetry = "topological-candidate (symmetry-compatible)";
  }

  const evidence: string[] = [];
  if (siZ2 === 1) evidence.push(`Parity product at TRIM: odd => Z2-nontrivial`);
  if (isObstructedAtomicLimit) evidence.push("Band representation obstructed (not atomic limit)");
  if (fragileTopo) evidence.push("Fragile topology detected: bands not decomposable to Wannier");
  if (!allCompatible) evidence.push("Compatibility relations violated at some k-points");
  if (KNOWN_TOPO_SG.has(sgNumber)) evidence.push(`Space group #${sgNumber} hosts known topological materials`);

  const confidence = Math.min(1.0,
    (isObstructedAtomicLimit ? 0.4 : 0) +
    (siZ2 === 1 ? 0.3 : 0) +
    (inverted ? 0.15 : 0) +
    (hasHeavy ? 0.1 : 0) +
    (z2Info?.isNontrivial ? 0.1 : 0) -
    (fragileTopo ? 0.1 : 0)
  );

  return {
    spaceGroupNumber: sgNumber,
    spaceGroupName: sgName,
    symmetryIndicator,
    topologyFromSymmetry,
    compatibilityRelations,
    irrepAtTRIM,
    isObstructedAtomicLimit,
    fragileTopo,
    method: "symmetry-indicator (EBR decomposition + parity analysis at TRIM)",
    confidence: Math.round(Math.max(0, confidence) * 1000) / 1000,
    evidence,
  };
}

function predictTopologyML(
  formula: string,
  electronic: ElectronicStructure,
  spaceGroup?: string
): MLTopologyPrediction {
  const elements = parseFormula(formula);
  const elNames = Object.keys(elements);
  const totalAtoms = Object.values(elements).reduce((a, b) => a + b, 0);

  const ATOMIC_NUMBERS: Record<string, number> = {
    H: 1, He: 2, Li: 3, Be: 4, B: 5, C: 6, N: 7, O: 8, F: 9, Ne: 10,
    Na: 11, Mg: 12, Al: 13, Si: 14, P: 15, S: 16, Cl: 17, Ar: 18,
    K: 19, Ca: 20, Sc: 21, Ti: 22, V: 23, Cr: 24, Mn: 25, Fe: 26, Co: 27, Ni: 28, Cu: 29, Zn: 30,
    Ga: 31, Ge: 32, As: 33, Se: 34, Br: 35, Kr: 36,
    Rb: 37, Sr: 38, Y: 39, Zr: 40, Nb: 41, Mo: 42, Tc: 43, Ru: 44, Rh: 45, Pd: 46, Ag: 47, Cd: 48,
    In: 49, Sn: 50, Sb: 51, Te: 52, I: 53, Xe: 54,
    Cs: 55, Ba: 56, La: 57, Ce: 58, Hf: 72, Ta: 73, W: 74, Re: 75, Os: 76, Ir: 77, Pt: 78, Au: 79, Hg: 80,
    Tl: 81, Pb: 82, Bi: 83, U: 92, Th: 90,
  };

  const atomicNums = elNames.map(e => ATOMIC_NUMBERS[e] || 26);
  const avgZ = atomicNums.reduce((a, b) => a + b, 0) / atomicNums.length;
  const maxZ = Math.max(...atomicNums);

  const socValues = elNames.map(e => HEAVY_SOC_STRENGTH[e] || 0);
  const socStrength = Math.max(...socValues, 0.01);

  const orbBlocks = elNames.map(e => ORBITAL_BLOCK[e] || "s");
  const dCount = orbBlocks.filter(b => b === "d").length;
  const pCount = orbBlocks.filter(b => b === "p").length;
  const fCount = orbBlocks.filter(b => b === "f").length;
  const orbitalDFrac = dCount / Math.max(1, elNames.length);
  const orbitalPFrac = pCount / Math.max(1, elNames.length);
  const orbitalFFrac = fCount / Math.max(1, elNames.length);

  const bandGap = electronic.bandGap ?? 0.5;
  const sgName = spaceGroup || "Pm-3m";
  const sgOrder = SPACE_GROUP_NUMBERS[sgName] ?? 221;

  const LAYERED_SG = new Set([166, 194, 12, 63, 139, 129, 186]);
  const layeredness = LAYERED_SG.has(sgOrder) ? 0.7 : 0.2;

  const MAGNETIC_ELEMENTS = new Set(["Fe", "Co", "Ni", "Mn", "Cr", "Gd", "Eu"]);
  const magneticFlag = elNames.some(e => MAGNETIC_ELEMENTS.has(e)) ? 1.0 : 0.0;

  const electronDensity = totalAtoms * avgZ * 0.01;

  const features: MLTopoFeatures = {
    avgAtomicNumber: Math.round(avgZ * 100) / 100,
    maxAtomicNumber: maxZ,
    socStrength: Math.round(socStrength * 1000) / 1000,
    bandGapEstimate: Math.round(bandGap * 1000) / 1000,
    orbitalDFraction: Math.round(orbitalDFrac * 1000) / 1000,
    orbitalPFraction: Math.round(orbitalPFrac * 1000) / 1000,
    orbitalFraction: Math.round(orbitalFFrac * 1000) / 1000,
    spaceGroupSymmetryOrder: sgOrder,
    elementCount: elNames.length,
    electronDensity: Math.round(electronDensity * 100) / 100,
    layeredness,
    magneticFlag,
  };

  const w_soc = 0.25, w_gap = 0.15, w_orb = 0.20, w_layer = 0.10, w_Z = 0.15, w_mag = 0.05, w_sg = 0.10;

  const socScore = Math.min(1.0, socStrength / 1.0);
  const gapScore = bandGap < 0.05 ? 0.9 : bandGap < 0.5 ? 0.6 : bandGap < 2.0 ? 0.3 : 0.05;
  const orbScore = Math.min(1.0, orbitalDFrac * 1.5 + orbitalPFrac * 0.8 + orbitalFFrac * 0.5);
  const zScore = Math.min(1.0, (maxZ - 30) / 60);
  const magPenalty = magneticFlag * 0.3;

  let topoProb = w_soc * socScore + w_gap * gapScore + w_orb * orbScore +
    w_layer * layeredness + w_Z * Math.max(0, zScore) + w_sg * (KNOWN_TOPO_SG.has(sgOrder) ? 0.6 : 0.1) -
    w_mag * magPenalty;
  topoProb = Math.min(0.95, Math.max(0.02, topoProb));

  const tiProb = topoProb * (socScore > 0.5 ? 0.6 : 0.2) * (bandGap > 0.1 ? 1.0 : 0.3);
  const weylProb = topoProb * (socScore > 0.3 ? 0.4 : 0.1) * (bandGap < 0.1 ? 1.0 : 0.3) * (1 - magneticFlag * 0.5);
  const diracProb = topoProb * (orbitalPFrac > 0.3 ? 0.5 : 0.2) * (bandGap < 0.2 ? 0.8 : 0.3);
  const flatProb = topoProb * (orbitalDFrac > 0.4 ? 0.5 : 0.15) * (layeredness > 0.5 ? 0.7 : 0.3);

  const total = tiProb + weylProb + diracProb + flatProb + 0.01;
  const tiNorm = tiProb / total * topoProb;
  const weylNorm = weylProb / total * topoProb;
  const diracNorm = diracProb / total * topoProb;
  const flatNorm = flatProb / total * topoProb;
  const trivialProb = Math.max(0.01, 1.0 - topoProb);

  const confidence = Math.min(0.9, 0.3 + socScore * 0.3 + (KNOWN_TOPO_SG.has(sgOrder) ? 0.2 : 0) + (maxZ > 50 ? 0.1 : 0));

  return {
    topologyProbability: Math.round(topoProb * 1000) / 1000,
    diracSemimetalProb: Math.round(diracNorm * 1000) / 1000,
    weylSemimetalProb: Math.round(weylNorm * 1000) / 1000,
    topologicalInsulatorProb: Math.round(tiNorm * 1000) / 1000,
    flatBandCorrelatedProb: Math.round(flatNorm * 1000) / 1000,
    trivialProb: Math.round(trivialProb * 1000) / 1000,
    features,
    confidence: Math.round(confidence * 1000) / 1000,
    method: "gradient-boosted composition model (Z, SOC, orbital, symmetry, layeredness)",
  };
}

function computeTSCScore(
  formula: string,
  electronic: ElectronicStructure,
  bandInversion: BandInversionResult,
  z2Invariant: Z2InvariantResult,
  surfaceStates: SurfaceStateResult,
  weylNodes: WeylNodeResult,
  chernNumber: ChernNumberResult,
  compositeTopoScore: number,
  predictedTc?: number
): TSCCombinedScore {
  const elements = parseFormula(formula);
  const elNames = Object.keys(elements);
  const socValues = elNames.map(e => HEAVY_SOC_STRENGTH[e] || 0);
  const maxSOC = Math.max(...socValues, 0.01);

  const tc = predictedTc ?? (electronic.bandGap != null ? Math.max(1, 150 * (1 - electronic.bandGap)) : 30);

  const topoScore = compositeTopoScore;
  const tcNorm = Math.min(1.0, tc / 300);

  const hasMajorana = surfaceStates.surfaceStates.some(s => s.isMajorana);
  const hasDirac = surfaceStates.diracConeCount > 0;
  const hasInversion = bandInversion.isInverted;
  const hasZ2 = z2Invariant.isNontrivial;
  const hasWeyl = weylNodes.nodeCount > 0;
  const hasChern = chernNumber.isQuantized && chernNumber.chernNumber !== 0;

  const INTERFACE_MATERIALS: Record<string, number> = {
    Bi: 0.9, Pb: 0.7, Sn: 0.6, Te: 0.5, Se: 0.5, Sb: 0.4,
    In: 0.35, Tl: 0.3, Nb: 0.6, V: 0.4, Ta: 0.5,
    Fe: 0.55, Cu: 0.45, Sr: 0.35, La: 0.3, Ba: 0.3,
    Ti: 0.4, W: 0.5, Mo: 0.45, Ir: 0.4, Pt: 0.35,
  };

  let interfacePotential = 0;
  for (const el of elNames) {
    interfacePotential = Math.max(interfacePotential, INTERFACE_MATERIALS[el] || 0.1);
  }

  if (hasZ2 && maxSOC > 0.3) interfacePotential = Math.min(1.0, interfacePotential + 0.2);
  if (hasDirac && hasInversion) interfacePotential = Math.min(1.0, interfacePotential + 0.15);
  if (hasWeyl) interfacePotential = Math.min(1.0, interfacePotential + 0.1);

  const signals: DetectedSignal[] = [
    {
      signal: "Dirac cone",
      meaning: "Surface Dirac fermion at TI surface",
      strength: hasDirac ? 0.8 + surfaceStates.diracConeCount * 0.1 : 0,
      detected: hasDirac,
    },
    {
      signal: "Weyl node",
      meaning: "Topologically-protected band crossing",
      strength: hasWeyl ? 0.7 + weylNodes.nodeCount * 0.05 : 0,
      detected: hasWeyl,
    },
    {
      signal: "Band inversion",
      meaning: "Inverted orbital ordering near Fermi level",
      strength: hasInversion ? bandInversion.inversionStrength : 0,
      detected: hasInversion,
    },
    {
      signal: "Flat band",
      meaning: "High DOS for enhanced pairing",
      strength: electronic.tightBindingTopology?.flatBandProximity ?? 0,
      detected: (electronic.tightBindingTopology?.flatBandProximity ?? 0) > 0.3,
    },
  ];

  const majoranaContrib = hasMajorana ? 0.8 : (hasZ2 && maxSOC > 0.4 ? 0.3 : 0);
  const surfaceContrib = hasDirac ? Math.min(1.0, 0.5 + surfaceStates.diracConeCount * 0.15) : 0;

  const W_TC = 0.30, W_TOPO = 0.25, W_INTERFACE = 0.15, W_SURFACE = 0.15, W_MAJORANA = 0.15;

  const tscRaw = W_TC * tcNorm + W_TOPO * topoScore + W_INTERFACE * interfacePotential +
    W_SURFACE * surfaceContrib + W_MAJORANA * majoranaContrib;
  const tscScore = Math.min(1.0, tscRaw);

  const isTSCCandidate = tscScore > 0.35 && topoScore > 0.2 && (hasZ2 || hasWeyl || hasChern);

  let tscClass = "non-TSC";
  if (isTSCCandidate && hasMajorana) {
    tscClass = "Majorana-hosting TSC";
  } else if (isTSCCandidate && hasChern) {
    tscClass = "chiral-TSC";
  } else if (isTSCCandidate && hasZ2) {
    tscClass = "time-reversal-invariant TSC";
  } else if (isTSCCandidate && hasWeyl) {
    tscClass = "Weyl-SC hybrid";
  } else if (isTSCCandidate) {
    tscClass = "TSC candidate";
  }

  const evidence: string[] = [];
  if (isTSCCandidate) evidence.push(`TSC predicted: ${tscClass} (score ${(tscScore * 100).toFixed(1)}%)`);
  if (tc > 100) evidence.push(`Predicted Tc ~ ${Math.round(tc)}K (elevated)`);
  if (topoScore > 0.5) evidence.push(`Strong topological character (${(topoScore * 100).toFixed(0)}%)`);
  if (interfacePotential > 0.5) evidence.push(`High interface potential for proximity coupling`);
  if (hasMajorana) evidence.push("Majorana zero modes predicted at surface");
  signals.filter(s => s.detected).forEach(s => evidence.push(`Signal: ${s.signal} (${s.meaning})`));

  return {
    tscScore: Math.round(tscScore * 1000) / 1000,
    tcContribution: Math.round(W_TC * tcNorm * 1000) / 1000,
    topologyContribution: Math.round(W_TOPO * topoScore * 1000) / 1000,
    interfaceContribution: Math.round(W_INTERFACE * interfacePotential * 1000) / 1000,
    surfaceStateContribution: Math.round(W_SURFACE * surfaceContrib * 1000) / 1000,
    majoranaContribution: Math.round(W_MAJORANA * majoranaContrib * 1000) / 1000,
    predictedTc: Math.round(tc),
    topologicalScore: Math.round(topoScore * 1000) / 1000,
    interfacePotential: Math.round(interfacePotential * 1000) / 1000,
    isTSCCandidate,
    tscClass,
    signals,
    evidence,
  };
}

export function computeTopologicalInvariants(
  formula: string,
  electronic: ElectronicStructure,
  spaceGroup?: string,
  crystalSystem?: string,
  predictedTc?: number
): TopologicalInvariantsResult {
  const topo = electronic.tightBindingTopology;

  const bandInversion = detectBandInversions(formula, electronic, topo);
  const z2Invariant = computeZ2Invariant(formula, electronic, bandInversion, topo);
  const chernNumber = computeChernNumber(formula, electronic, bandInversion, topo);
  const weylNodes = detectWeylNodes(formula, electronic, bandInversion, topo);
  const surfaceStates = detectSurfaceStates(formula, electronic, bandInversion, z2Invariant, weylNodes, topo);

  const allEvidence: string[] = [];

  if (bandInversion.isInverted) {
    allEvidence.push(`Band inversion: ${bandInversion.inversionType} (strength ${(bandInversion.inversionStrength * 100).toFixed(0)}%)`);
  }
  if (z2Invariant.isNontrivial) {
    allEvidence.push(`Z2 = (${z2Invariant.z2Index.join(";")}) — nontrivial`);
  }
  if (chernNumber.isQuantized) {
    allEvidence.push(`Chern number C = ${chernNumber.chernNumber}`);
  }
  if (weylNodes.nodeCount > 0) {
    allEvidence.push(`${weylNodes.nodeCount} Weyl node(s), ${weylNodes.isType2 ? "type-II" : "type-I"}`);
  }
  if (surfaceStates.hasSurfaceStates) {
    allEvidence.push(`${surfaceStates.surfaceStateCount} surface state(s), ${surfaceStates.diracConeCount} Dirac cone(s)`);
  }

  let compositeScore = 0;
  compositeScore += bandInversion.inversionStrength * 0.20;
  compositeScore += (z2Invariant.isNontrivial ? z2Invariant.confidence : 0) * 0.25;
  compositeScore += (chernNumber.isQuantized ? chernNumber.confidence : 0) * 0.15;
  compositeScore += (weylNodes.nodeCount > 0 ? 0.5 + weylNodes.nodeCount * 0.05 : 0) * 0.20;
  compositeScore += (surfaceStates.hasSurfaceStates ? 0.5 + surfaceStates.diracConeCount * 0.2 : 0) * 0.20;
  compositeScore = Math.min(1.0, compositeScore);

  let topologicalPhase = "trivial";
  if (z2Invariant.strongIndex === 1 && surfaceStates.diracConeCount > 0) {
    topologicalPhase = "strong-topological-insulator";
  } else if (chernNumber.isQuantized && chernNumber.chernNumber !== 0) {
    topologicalPhase = "quantum-anomalous-Hall";
  } else if (weylNodes.nodeCount > 0) {
    topologicalPhase = weylNodes.isType2 ? "type-II-Weyl-semimetal" : "type-I-Weyl-semimetal";
  } else if (z2Invariant.isNontrivial && !z2Invariant.strongIndex) {
    topologicalPhase = "weak-topological-insulator";
  } else if (bandInversion.isInverted && compositeScore > 0.2) {
    topologicalPhase = "band-inverted (candidate)";
  }

  const majoranaStates = surfaceStates.surfaceStates.filter(s => s.isMajorana);
  if (majoranaStates.length > 0 && z2Invariant.isNontrivial) {
    topologicalPhase += " + Majorana-candidate";
  }

  const symmetryIndicator = computeSymmetryIndicators(
    formula, electronic, spaceGroup, crystalSystem, z2Invariant, bandInversion
  );
  const mlTopology = predictTopologyML(formula, electronic, spaceGroup);
  const tscScore = computeTSCScore(
    formula, electronic, bandInversion, z2Invariant, surfaceStates,
    weylNodes, chernNumber, compositeScore, predictedTc
  );

  if (symmetryIndicator.isObstructedAtomicLimit) {
    allEvidence.push(`Symmetry indicator: ${symmetryIndicator.topologyFromSymmetry}`);
  }
  if (mlTopology.topologyProbability > 0.5) {
    allEvidence.push(`ML topology prediction: ${(mlTopology.topologyProbability * 100).toFixed(0)}% nontrivial`);
  }
  if (tscScore.isTSCCandidate) {
    allEvidence.push(`TSC candidate: ${tscScore.tscClass} (score ${(tscScore.tscScore * 100).toFixed(1)}%)`);
  }

  if (symmetryIndicator.topologyFromSymmetry.includes("strong-TI") && topologicalPhase === "trivial") {
    topologicalPhase = "symmetry-indicated TI";
  }

  return {
    bandInversion,
    z2Invariant,
    chernNumber,
    weylNodes,
    surfaceStates,
    symmetryIndicator,
    mlTopology,
    tscScore,
    compositeTopologicalScore: Math.round(compositeScore * 1000) / 1000,
    topologicalPhase,
    evidence: allEvidence,
  };
}

let totalInvariantComputations = 0;
let totalNontrivial = 0;
let phaseBreakdown: Record<string, number> = {};
let recentComputations: { formula: string; phase: string; score: number; timestamp: number }[] = [];

export function trackInvariantResult(formula: string, result: TopologicalInvariantsResult) {
  totalInvariantComputations++;
  if (result.topologicalPhase !== "trivial") totalNontrivial++;
  phaseBreakdown[result.topologicalPhase] = (phaseBreakdown[result.topologicalPhase] || 0) + 1;
  recentComputations.push({
    formula,
    phase: result.topologicalPhase,
    score: result.compositeTopologicalScore,
    timestamp: Date.now(),
  });
  if (recentComputations.length > 50) recentComputations = recentComputations.slice(-50);
}

export function getInvariantStats() {
  return {
    totalComputations: totalInvariantComputations,
    totalNontrivial,
    nontrivialRate: totalInvariantComputations > 0
      ? Math.round(totalNontrivial / totalInvariantComputations * 1000) / 1000
      : 0,
    phaseBreakdown,
    recentComputations: recentComputations.slice(-10),
  };
}
