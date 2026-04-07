import type { LatticeParams, AtomPosition, RelaxationEntry } from "./relaxation-tracker";
import { getElementData } from "../learning/elemental-data";

const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71,
  O: 0.66, F: 0.57, Ne: 0.58, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11,
  P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06, K: 2.03, Ca: 1.76, Sc: 1.70,
  Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24,
  Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20,
  Kr: 1.16, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54,
  Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39,
  Sb: 1.39, Te: 1.38, I: 1.39, Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04,
  Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36,
  Au: 1.36, Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48, Tc: 1.47,
  Pr: 2.03, Nd: 2.01, Sm: 1.98, Eu: 1.98, Gd: 1.96, Tb: 1.94,
  Dy: 1.92, Ho: 1.92, Er: 1.89, Tm: 1.90, Yb: 1.87, Lu: 1.87,
  Th: 2.06, U: 1.96, Pa: 2.00, Np: 1.90, Pu: 1.87,
};

function getCovalentRadius(el: string): number {
  return COVALENT_RADII[el] || (getElementData(el)?.atomicRadius ?? 150) / 100;
}

export interface AtomicDisplacement {
  element: string;
  index: number;
  dx: number;
  dy: number;
  dz: number;
  displacement: number;
}

export interface DistortionMetrics {
  meanDisplacement: number;
  maxDisplacement: number;
  rmsDisplacement: number;
  displacedAtomFraction: number;
  perAtomDisplacements: AtomicDisplacement[];
}

export interface LatticeDistortion {
  strainA: number;
  strainB: number;
  strainC: number;
  strainMagnitude: number;
  deltaAlpha: number;
  deltaBeta: number;
  deltaGamma: number;
  volumeChangePct: number;
  tetragonalRatio: number;
  orthorhombicSplit: number;
}

export interface BondInfo {
  atom1Element: string;
  atom1Index: number;
  atom2Element: string;
  atom2Index: number;
  bondLength: number;
  idealBondLength: number;
  deviation: number;
  deviationPct: number;
}

export interface BondLengthDistortion {
  totalBonds: number;
  bondLengthVariance: number;
  bondLengthStd: number;
  meanBondLength: number;
  meanIdealBondLength: number;
  meanDeviationPct: number;
  maxDeviationPct: number;
  distortedBondCount: number;
  distortedBondFraction: number;
  bondsByType: Record<string, {
    count: number;
    meanLength: number;
    variance: number;
    idealLength: number;
    meanDeviationPct: number;
  }>;
  worstBonds: BondInfo[];
}

export interface OctahedralSite {
  centerElement: string;
  centerIndex: number;
  ligandElements: string[];
  bondLengths: number[];
  bondAngles: number[];
  bondLengthVariance: number;
  bondAngleVariance: number;
  elongationIndex: number;
  quadraticElongation: number;
  isDistorted: boolean;
  distortionType: "regular" | "elongated" | "compressed" | "tilted";
}

export interface OctahedralDistortion {
  siteCount: number;
  sites: OctahedralSite[];
  avgBondAngleVariance: number;
  avgBondLengthVariance: number;
  avgQuadraticElongation: number;
  distortedSiteCount: number;
  distortedSiteFraction: number;
  overallOctahedralDistortion: "none" | "mild" | "moderate" | "severe";
}

export interface PhononInstabilityInfo {
  hasImaginaryModes: boolean;
  imaginaryModeCount: number;
  lowestFrequency: number;
  instabilitySeverity: "none" | "soft-mode" | "moderate" | "severe";
  structureWantsToDistort: boolean;
}

export type DistortionLevel = "none" | "small" | "moderate" | "large" | "severe";

export type SymmetryChangeType =
  | "preserved"
  | "cubic-to-tetragonal"
  | "cubic-to-orthorhombic"
  | "tetragonal-to-orthorhombic"
  | "tetragonal-to-monoclinic"
  | "hexagonal-to-orthorhombic"
  | "other-reduction"
  | "unknown";

export interface SymmetryReduction {
  sgBefore: string;
  sgAfter: string;
  systemBefore: string;
  systemAfter: string;
  symmetryBroken: boolean;
  changeType: SymmetryChangeType;
  groupOrderBefore: number;
  groupOrderAfter: number;
  reductionFactor: number;
}

export interface DistortionAnalysis {
  formula: string;
  atomicDistortion: DistortionMetrics | null;
  latticeDistortion: LatticeDistortion;
  symmetryReduction: SymmetryReduction | null;
  bondLengthDistortion: BondLengthDistortion | null;
  octahedralDistortion: OctahedralDistortion | null;
  phononInstability: PhononInstabilityInfo | null;
  overallLevel: DistortionLevel;
  overallScore: number;
  jahn_teller_likely: boolean;
  peierls_likely: boolean;
  cdw_susceptible: boolean;
  scRelevance: string;
  analyzedAt: number;
}

const DISPLACEMENT_THRESHOLD_SMALL = 0.05;
const DISPLACEMENT_THRESHOLD_MODERATE = 0.15;
const DISPLACEMENT_THRESHOLD_LARGE = 0.30;

const VOLUME_CHANGE_THRESHOLD = 3.0;
const STRAIN_THRESHOLD_MODERATE = 0.03;
const STRAIN_THRESHOLD_LARGE = 0.08;

const CRYSTAL_SYSTEM_ORDER: Record<string, number> = {
  cubic: 48,
  hexagonal: 24,
  trigonal: 12,
  tetragonal: 16,
  orthorhombic: 8,
  monoclinic: 4,
  triclinic: 2,
};

const SG_TO_SYSTEM: Record<string, string> = {
  "Pm-3m": "cubic", "Fm-3m": "cubic", "Im-3m": "cubic", "Fd-3m": "cubic",
  "Pa-3": "cubic", "Ia-3d": "cubic",
  "P4/mmm": "tetragonal", "I4/mmm": "tetragonal", "P4mm": "tetragonal",
  "I4/mcm": "tetragonal", "P42/mnm": "tetragonal",
  "P6/mmm": "hexagonal", "P63/mmc": "hexagonal", "P63mc": "hexagonal",
  "R-3m": "trigonal", "R3m": "trigonal", "R-3c": "trigonal",
  "Pnma": "orthorhombic", "Cmcm": "orthorhombic", "Amm2": "orthorhombic",
  "Pmmm": "orthorhombic", "Immm": "orthorhombic", "Fmmm": "orthorhombic",
  "P21/c": "monoclinic", "C2/m": "monoclinic", "P2/m": "monoclinic",
  "P-1": "triclinic",
};

function latticeVolume(l: LatticeParams): number {
  const toRad = Math.PI / 180;
  const cosA = Math.cos(l.alpha * toRad);
  const cosB = Math.cos(l.beta * toRad);
  const cosG = Math.cos(l.gamma * toRad);
  const factor = Math.sqrt(Math.max(0, 1 - cosA * cosA - cosB * cosB - cosG * cosG + 2 * cosA * cosB * cosG));
  return l.a * l.b * l.c * factor;
}

export function computeAtomicDistortion(
  initial: AtomPosition[],
  relaxed: AtomPosition[]
): DistortionMetrics | null {
  if (!initial || !relaxed || initial.length === 0 || relaxed.length === 0) return null;
  const n = Math.min(initial.length, relaxed.length);

  const displacements: AtomicDisplacement[] = [];
  let sumDisp = 0;
  let sumDispSq = 0;
  let maxDisp = 0;
  let displacedCount = 0;

  for (let i = 0; i < n; i++) {
    const dx = relaxed[i].x - initial[i].x;
    const dy = relaxed[i].y - initial[i].y;
    const dz = relaxed[i].z - initial[i].z;
    const disp = Math.sqrt(dx * dx + dy * dy + dz * dz);

    displacements.push({
      element: initial[i].element || initial[i].symbol || "X",
      index: i,
      dx: Math.round(dx * 10000) / 10000,
      dy: Math.round(dy * 10000) / 10000,
      dz: Math.round(dz * 10000) / 10000,
      displacement: Math.round(disp * 10000) / 10000,
    });

    sumDisp += disp;
    sumDispSq += disp * disp;
    if (disp > maxDisp) maxDisp = disp;
    if (disp > 0.01) displacedCount++;
  }

  return {
    meanDisplacement: Math.round((sumDisp / n) * 10000) / 10000,
    maxDisplacement: Math.round(maxDisp * 10000) / 10000,
    rmsDisplacement: Math.round(Math.sqrt(sumDispSq / n) * 10000) / 10000,
    displacedAtomFraction: Math.round((displacedCount / n) * 1000) / 1000,
    perAtomDisplacements: displacements,
  };
}

export function computeLatticeDistortion(
  before: LatticeParams,
  after: LatticeParams
): LatticeDistortion {
  const strainA = (after.a - before.a) / before.a;
  const strainB = (after.b - before.b) / before.b;
  const strainC = (after.c - before.c) / before.c;
  const strainMagnitude = Math.sqrt(strainA * strainA + strainB * strainB + strainC * strainC);

  const deltaAlpha = after.alpha - before.alpha;
  const deltaBeta = after.beta - before.beta;
  const deltaGamma = after.gamma - before.gamma;

  const volBefore = latticeVolume(before);
  const volAfter = latticeVolume(after);
  const volumeChangePct = volBefore > 0 ? ((volAfter - volBefore) / volBefore) * 100 : 0;

  const avgAB = (after.a + after.b) / 2;
  const tetragonalRatio = avgAB > 0 ? after.c / avgAB : 1;
  const orthorhombicSplit = after.a > 0 ? Math.abs(after.a - after.b) / after.a : 0;

  return {
    strainA: Math.round(strainA * 100000) / 100000,
    strainB: Math.round(strainB * 100000) / 100000,
    strainC: Math.round(strainC * 100000) / 100000,
    strainMagnitude: Math.round(strainMagnitude * 100000) / 100000,
    deltaAlpha: Math.round(deltaAlpha * 1000) / 1000,
    deltaBeta: Math.round(deltaBeta * 1000) / 1000,
    deltaGamma: Math.round(deltaGamma * 1000) / 1000,
    volumeChangePct: Math.round(volumeChangePct * 1000) / 1000,
    tetragonalRatio: Math.round(tetragonalRatio * 10000) / 10000,
    orthorhombicSplit: Math.round(orthorhombicSplit * 10000) / 10000,
  };
}

const BOND_CUTOFF_FACTOR = 1.3;
const DISTORTED_BOND_THRESHOLD_PCT = 10;
const OCTAHEDRAL_ANGLE_VARIANCE_MILD = 10;
const OCTAHEDRAL_ANGLE_VARIANCE_MODERATE = 50;
const OCTAHEDRAL_ANGLE_VARIANCE_SEVERE = 150;

const TRANSITION_METALS = new Set([
  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
  "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
]);

function atomDistance(a1: AtomPosition, a2: AtomPosition): number {
  const dx = a1.x - a2.x;
  const dy = a1.y - a2.y;
  const dz = a1.z - a2.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function bondAngle(center: AtomPosition, a1: AtomPosition, a2: AtomPosition): number {
  const v1x = a1.x - center.x, v1y = a1.y - center.y, v1z = a1.z - center.z;
  const v2x = a2.x - center.x, v2y = a2.y - center.y, v2z = a2.z - center.z;
  const dot = v1x * v2x + v1y * v2y + v1z * v2z;
  const m1 = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
  const m2 = Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
  if (m1 < 1e-10 || m2 < 1e-10) return 0;
  const cosAngle = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

export function computeBondLengthDistortion(atoms: AtomPosition[]): BondLengthDistortion | null {
  if (!atoms || atoms.length < 2) return null;

  const bonds: BondInfo[] = [];
  for (let i = 0; i < atoms.length; i++) {
    const r1 = getCovalentRadius(atoms[i].element || (atoms[i] as any).symbol || "X");
    for (let j = i + 1; j < atoms.length; j++) {
      const r2 = getCovalentRadius(atoms[j].element || (atoms[j] as any).symbol || "X");
      const idealLen = r1 + r2;
      const dist = atomDistance(atoms[i], atoms[j]);
      if (dist < idealLen * BOND_CUTOFF_FACTOR && dist > 0.3) {
        const deviation = dist - idealLen;
        const deviationPct = idealLen > 0 ? (Math.abs(deviation) / idealLen) * 100 : 0;
        bonds.push({
          atom1Element: atoms[i].element || (atoms[i] as any).symbol || "X",
          atom1Index: i,
          atom2Element: atoms[j].element || (atoms[j] as any).symbol || "X",
          atom2Index: j,
          bondLength: Math.round(dist * 10000) / 10000,
          idealBondLength: Math.round(idealLen * 10000) / 10000,
          deviation: Math.round(deviation * 10000) / 10000,
          deviationPct: Math.round(deviationPct * 100) / 100,
        });
      }
    }
  }

  if (bonds.length === 0) return null;

  const lengths = bonds.map(b => b.bondLength);
  const meanLen = lengths.reduce((s, v) => s + v, 0) / lengths.length;
  const variance = lengths.reduce((s, v) => s + (v - meanLen) ** 2, 0) / lengths.length;
  const meanIdeal = bonds.reduce((s, b) => s + b.idealBondLength, 0) / bonds.length;
  const devPcts = bonds.map(b => b.deviationPct);
  const meanDevPct = devPcts.reduce((s, v) => s + v, 0) / devPcts.length;
  const maxDevPct = Math.max(...devPcts);
  const distortedCount = bonds.filter(b => b.deviationPct > DISTORTED_BOND_THRESHOLD_PCT).length;

  const bondsByType: Record<string, { lengths: number[]; ideal: number; devs: number[] }> = {};
  for (const b of bonds) {
    const key = [b.atom1Element, b.atom2Element].sort().join("-");
    if (!bondsByType[key]) bondsByType[key] = { lengths: [], ideal: b.idealBondLength, devs: [] };
    bondsByType[key].lengths.push(b.bondLength);
    bondsByType[key].devs.push(b.deviationPct);
  }

  const bondsByTypeOut: Record<string, { count: number; meanLength: number; variance: number; idealLength: number; meanDeviationPct: number }> = {};
  for (const [key, data] of Object.entries(bondsByType)) {
    const m = data.lengths.reduce((s, v) => s + v, 0) / data.lengths.length;
    const v = data.lengths.reduce((s, val) => s + (val - m) ** 2, 0) / data.lengths.length;
    const md = data.devs.reduce((s, val) => s + val, 0) / data.devs.length;
    bondsByTypeOut[key] = {
      count: data.lengths.length,
      meanLength: Math.round(m * 10000) / 10000,
      variance: Math.round(v * 100000) / 100000,
      idealLength: Math.round(data.ideal * 10000) / 10000,
      meanDeviationPct: Math.round(md * 100) / 100,
    };
  }

  const worst = [...bonds].sort((a, b) => b.deviationPct - a.deviationPct).slice(0, 5);

  return {
    totalBonds: bonds.length,
    bondLengthVariance: Math.round(variance * 100000) / 100000,
    bondLengthStd: Math.round(Math.sqrt(variance) * 10000) / 10000,
    meanBondLength: Math.round(meanLen * 10000) / 10000,
    meanIdealBondLength: Math.round(meanIdeal * 10000) / 10000,
    meanDeviationPct: Math.round(meanDevPct * 100) / 100,
    maxDeviationPct: Math.round(maxDevPct * 100) / 100,
    distortedBondCount: distortedCount,
    distortedBondFraction: Math.round((distortedCount / bonds.length) * 1000) / 1000,
    bondsByType: bondsByTypeOut,
    worstBonds: worst,
  };
}

export function computeOctahedralDistortion(atoms: AtomPosition[]): OctahedralDistortion | null {
  if (!atoms || atoms.length < 3) return null;

  const sites: OctahedralSite[] = [];

  for (let i = 0; i < atoms.length; i++) {
    const el = atoms[i].element || (atoms[i] as any).symbol || "X";
    if (!TRANSITION_METALS.has(el)) continue;

    const neighbors: { index: number; dist: number; element: string }[] = [];
    const rCenter = getCovalentRadius(el);

    for (let j = 0; j < atoms.length; j++) {
      if (i === j) continue;
      const rLigand = getCovalentRadius(atoms[j].element || (atoms[j] as any).symbol || "X");
      const dist = atomDistance(atoms[i], atoms[j]);
      const idealBond = rCenter + rLigand;
      if (dist < idealBond * BOND_CUTOFF_FACTOR && dist > 0.3) {
        neighbors.push({ index: j, dist, element: atoms[j].element || (atoms[j] as any).symbol || "X" });
      }
    }

    if (neighbors.length < 4 || neighbors.length > 8) continue;

    neighbors.sort((a, b) => a.dist - b.dist);
    const coordNeighbors = neighbors.slice(0, Math.min(6, neighbors.length));

    const bondLengths = coordNeighbors.map(n => n.dist);
    const ligandElements = coordNeighbors.map(n => n.element);

    const angles: number[] = [];
    for (let a = 0; a < coordNeighbors.length; a++) {
      for (let b = a + 1; b < coordNeighbors.length; b++) {
        const angle = bondAngle(atoms[i], atoms[coordNeighbors[a].index], atoms[coordNeighbors[b].index]);
        if (angle > 10) angles.push(angle);
      }
    }

    const meanBondLen = bondLengths.reduce((s, v) => s + v, 0) / bondLengths.length;
    const blVariance = bondLengths.reduce((s, v) => s + (v - meanBondLen) ** 2, 0) / bondLengths.length;

    const baVariance = angles.length > 0
      ? angles.reduce((s, a) => s + (a - 90) ** 2, 0) / angles.length
      : 0;

    const qe = bondLengths.length > 0
      ? bondLengths.reduce((s, d) => s + (d / meanBondLen) ** 2, 0) / bondLengths.length
      : 1;

    const maxLen = Math.max(...bondLengths);
    const minLen = Math.min(...bondLengths);
    const elongationIndex = minLen > 0 ? maxLen / minLen : 1;

    let distortionType: "regular" | "elongated" | "compressed" | "tilted" = "regular";
    if (baVariance > OCTAHEDRAL_ANGLE_VARIANCE_MILD) {
      const sortedLens = [...bondLengths].sort((a, b) => a - b);
      if (sortedLens.length >= 4) {
        const shortMean = (sortedLens[0] + sortedLens[1]) / 2;
        const longMean = (sortedLens[sortedLens.length - 1] + sortedLens[sortedLens.length - 2]) / 2;
        const ratio = longMean / shortMean;
        if (ratio > 1.08) distortionType = "elongated";
        else if (ratio < 0.95) distortionType = "compressed";
        else distortionType = "tilted";
      }
    }

    const isDistorted = baVariance > OCTAHEDRAL_ANGLE_VARIANCE_MILD || blVariance > 0.005;

    sites.push({
      centerElement: el,
      centerIndex: i,
      ligandElements,
      bondLengths: bondLengths.map(v => Math.round(v * 10000) / 10000),
      bondAngles: angles.map(v => Math.round(v * 100) / 100),
      bondLengthVariance: Math.round(blVariance * 100000) / 100000,
      bondAngleVariance: Math.round(baVariance * 100) / 100,
      elongationIndex: Math.round(elongationIndex * 10000) / 10000,
      quadraticElongation: Math.round(qe * 100000) / 100000,
      isDistorted,
      distortionType,
    });
  }

  if (sites.length === 0) return null;

  const avgBAV = sites.reduce((s, si) => s + si.bondAngleVariance, 0) / sites.length;
  const avgBLV = sites.reduce((s, si) => s + si.bondLengthVariance, 0) / sites.length;
  const avgQE = sites.reduce((s, si) => s + si.quadraticElongation, 0) / sites.length;
  const distortedCount = sites.filter(s => s.isDistorted).length;

  let overallOD: "none" | "mild" | "moderate" | "severe" = "none";
  if (avgBAV >= OCTAHEDRAL_ANGLE_VARIANCE_SEVERE) overallOD = "severe";
  else if (avgBAV >= OCTAHEDRAL_ANGLE_VARIANCE_MODERATE) overallOD = "moderate";
  else if (avgBAV >= OCTAHEDRAL_ANGLE_VARIANCE_MILD) overallOD = "mild";

  return {
    siteCount: sites.length,
    sites: sites.slice(0, 10),
    avgBondAngleVariance: Math.round(avgBAV * 100) / 100,
    avgBondLengthVariance: Math.round(avgBLV * 100000) / 100000,
    avgQuadraticElongation: Math.round(avgQE * 100000) / 100000,
    distortedSiteCount: distortedCount,
    distortedSiteFraction: Math.round((distortedCount / sites.length) * 1000) / 1000,
    overallOctahedralDistortion: overallOD,
  };
}

export function detectPhononInstability(
  frequencies?: number[],
  hasImaginaryModes?: boolean,
  imaginaryModeCount?: number,
  lowestFrequency?: number,
): PhononInstabilityInfo | null {
  if (!frequencies && hasImaginaryModes === undefined && lowestFrequency === undefined) return null;

  const freqs = frequencies ?? [];
  const lowest = lowestFrequency ?? (freqs.length > 0 ? Math.min(...freqs) : 0);
  const imagCount = imaginaryModeCount ?? freqs.filter(f => f < -5).length;
  const hasImag = hasImaginaryModes ?? imagCount > 0;

  let severity: "none" | "soft-mode" | "moderate" | "severe" = "none";
  if (hasImag) {
    if (imagCount >= 5 || lowest < -500) severity = "severe";
    else if (imagCount >= 2 || lowest < -100) severity = "moderate";
    else severity = "soft-mode";
  }

  return {
    hasImaginaryModes: hasImag,
    imaginaryModeCount: imagCount,
    lowestFrequency: Math.round(lowest * 100) / 100,
    instabilitySeverity: severity,
    structureWantsToDistort: hasImag && lowest < -20,
  };
}

function inferCrystalSystem(sg: string): string {
  if (SG_TO_SYSTEM[sg]) return SG_TO_SYSTEM[sg];
  if (sg.includes("m-3") || sg.includes("d-3") || sg.includes("a-3")) return "cubic";
  if (sg.startsWith("P6") || sg.startsWith("P-6")) return "hexagonal";
  if (sg.startsWith("R")) return "trigonal";
  if (sg.startsWith("P4") || sg.startsWith("I4")) return "tetragonal";
  if (sg.includes("mmm") || sg.includes("mm2")) return "orthorhombic";
  if (sg.includes("2/") || sg.includes("P2") || sg.includes("C2")) return "monoclinic";
  if (sg.includes("-1")) return "triclinic";
  return "unknown";
}

function classifySymmetryChange(sysBefore: string, sysAfter: string): SymmetryChangeType {
  if (sysBefore === sysAfter) return "preserved";
  const key = `${sysBefore}-to-${sysAfter}`;
  const knownChanges: Record<string, SymmetryChangeType> = {
    "cubic-to-tetragonal": "cubic-to-tetragonal",
    "cubic-to-orthorhombic": "cubic-to-orthorhombic",
    "tetragonal-to-orthorhombic": "tetragonal-to-orthorhombic",
    "tetragonal-to-monoclinic": "tetragonal-to-monoclinic",
    "hexagonal-to-orthorhombic": "hexagonal-to-orthorhombic",
  };
  return knownChanges[key] || "other-reduction";
}

export function detectSymmetryReduction(
  sgBefore: string,
  sgAfter: string
): SymmetryReduction {
  const systemBefore = inferCrystalSystem(sgBefore);
  const systemAfter = inferCrystalSystem(sgAfter);
  const orderBefore = CRYSTAL_SYSTEM_ORDER[systemBefore] ?? 2;
  const orderAfter = CRYSTAL_SYSTEM_ORDER[systemAfter] ?? 2;
  const symmetryBroken = sgBefore !== sgAfter && orderAfter < orderBefore;
  const changeType = symmetryBroken ? classifySymmetryChange(systemBefore, systemAfter) : "preserved";
  const reductionFactor = orderBefore > 0 ? orderAfter / orderBefore : 1;

  return {
    sgBefore,
    sgAfter,
    systemBefore,
    systemAfter,
    symmetryBroken,
    changeType,
    groupOrderBefore: orderBefore,
    groupOrderAfter: orderAfter,
    reductionFactor: Math.round(reductionFactor * 1000) / 1000,
  };
}

export function inferSpaceGroupFromLattice(
  before: LatticeParams,
  after: LatticeParams,
  sgBefore: string
): string {
  const tol = 0.005;
  const abEq = Math.abs(after.a - after.b) / after.a < tol;
  const bcEq = Math.abs(after.b - after.c) / after.b < tol;
  const allEq = abEq && bcEq;
  const anglesRight = Math.abs(after.alpha - 90) < 0.5 && Math.abs(after.beta - 90) < 0.5 && Math.abs(after.gamma - 90) < 0.5;
  const gamma120 = Math.abs(after.gamma - 120) < 0.5;

  if (allEq && anglesRight) return sgBefore;
  if (abEq && anglesRight && !bcEq) {
    if (sgBefore.includes("m-3")) return "P4/mmm";
    return "P4/mmm";
  }
  if (anglesRight && !abEq && !bcEq) {
    if (sgBefore.includes("m-3") || sgBefore.includes("4/")) return "Pnma";
    return "Pnma";
  }
  if (abEq && gamma120) return sgBefore;
  if (!anglesRight) return "P21/c";

  return sgBefore;
}

function classifyLevel(score: number): DistortionLevel {
  if (score < 0.02) return "none";
  if (score < 0.08) return "small";
  if (score < 0.20) return "moderate";
  if (score < 0.40) return "large";
  return "severe";
}

function assessSCRelevance(
  level: DistortionLevel,
  symmetryReduction: SymmetryReduction | null,
  jahnTeller: boolean,
  peierls: boolean,
  cdw: boolean,
  bondDist?: BondLengthDistortion | null,
  octDist?: OctahedralDistortion | null,
  phononInst?: PhononInstabilityInfo | null,
): string {
  const parts: string[] = [];

  if (level === "none" || level === "small") {
    parts.push("Minimal structural distortion — stable parent phase likely preserved");
  } else if (level === "moderate") {
    parts.push("Moderate distortion detected — possible phonon softening or structural instability");
  } else {
    parts.push("Significant distortion — structure may be near a phase boundary or instability");
  }

  if (symmetryReduction?.symmetryBroken) {
    parts.push(`Symmetry reduced from ${symmetryReduction.systemBefore} to ${symmetryReduction.systemAfter}`);
    if (symmetryReduction.changeType === "cubic-to-tetragonal") {
      parts.push("Cubic-to-tetragonal transition may indicate Jahn-Teller or charge-ordering");
    }
  }

  if (jahnTeller) parts.push("Jahn-Teller distortion signatures present (Cu/Mn d-orbital splitting)");
  if (peierls) parts.push("Possible Peierls instability (1D chain dimerization)");
  if (cdw) parts.push("CDW susceptibility (quasi-2D with nesting)");

  if (bondDist && bondDist.bondLengthVariance > 0.02) {
    parts.push(`Large bond length spread detected (variance=${bondDist.bondLengthVariance.toFixed(4)} A^2, ${bondDist.distortedBondCount}/${bondDist.totalBonds} bonds deviate >10% from ideal)`);
  }

  if (octDist && octDist.overallOctahedralDistortion !== "none") {
    const distTypes = octDist.sites.filter(s => s.isDistorted).map(s => s.distortionType);
    const uniqueTypes = [...new Set(distTypes)];
    parts.push(`Octahedral distortion (${octDist.overallOctahedralDistortion}): avg bond angle variance=${octDist.avgBondAngleVariance.toFixed(1)} deg^2, ${octDist.distortedSiteCount}/${octDist.siteCount} sites distorted (${uniqueTypes.join(", ")})`);
  }

  if (phononInst && phononInst.hasImaginaryModes) {
    parts.push(`Phonon instability: ${phononInst.imaginaryModeCount} imaginary mode(s), lowest freq=${phononInst.lowestFrequency.toFixed(1)} cm-1 — structure wants to distort`);
  }

  return parts.join(". ") + ".";
}

function detectJahnTeller(
  formula: string,
  atomicDist: DistortionMetrics | null,
  latticeDist: LatticeDistortion
): boolean {
  const jtElements = ["Cu", "Mn", "Cr", "Fe", "Co", "Ni", "Ti", "V"];
  const hasJTElement = jtElements.some(el => formula.includes(el));
  if (!hasJTElement) return false;
  const hasAnisotropicStrain = Math.abs(latticeDist.strainA - latticeDist.strainC) > 0.01
    || latticeDist.tetragonalRatio < 0.95 || latticeDist.tetragonalRatio > 1.05;
  if (!hasAnisotropicStrain) return false;
  if (atomicDist && atomicDist.meanDisplacement > 0.03) return true;
  if (Math.abs(latticeDist.volumeChangePct) > 1.5) return true;
  return false;
}

function detectPeierls(
  formula: string,
  atomicDist: DistortionMetrics | null,
  latticeDist: LatticeDistortion
): boolean {
  const quasi1DElements = ["Nb", "Ta", "Mo", "W"];
  const hasQ1D = quasi1DElements.some(el => formula.includes(el));
  if (!hasQ1D) return false;
  const anisotropicAxes = [
    Math.abs(latticeDist.strainA),
    Math.abs(latticeDist.strainB),
    Math.abs(latticeDist.strainC),
  ].sort((a, b) => b - a);
  if (anisotropicAxes.length >= 2 && anisotropicAxes[0] > 3 * (anisotropicAxes[1] + 0.001)) return true;
  if (atomicDist) {
    const sortedDisp = [...atomicDist.perAtomDisplacements].sort((a, b) => b.displacement - a.displacement);
    if (sortedDisp.length >= 2) {
      const top = sortedDisp[0].displacement;
      const second = sortedDisp[1].displacement;
      if (top > 0.05 && second < top * 0.3) return true;
    }
  }
  return false;
}

function detectCDWSusceptibility(
  formula: string,
  latticeDist: LatticeDistortion
): boolean {
  const layered = formula.includes("Se") || formula.includes("Te") || formula.includes("S");
  const quasi2D = latticeDist.tetragonalRatio > 1.5 || latticeDist.tetragonalRatio < 0.67;
  if (!layered && !quasi2D) return false;
  if (latticeDist.orthorhombicSplit > 0.005) return true;
  if (Math.abs(latticeDist.volumeChangePct) > 2 && quasi2D) return true;
  return false;
}

export function analyzeDistortion(
  formula: string,
  beforeLattice: LatticeParams,
  afterLattice: LatticeParams,
  beforePositions?: AtomPosition[],
  afterPositions?: AtomPosition[],
  sgBefore?: string,
  sgAfter?: string,
  phononFrequencies?: number[],
  phononHasImaginary?: boolean,
  phononImaginaryCount?: number,
  phononLowestFreq?: number,
): DistortionAnalysis {
  const atomicDistortion = computeAtomicDistortion(beforePositions ?? [], afterPositions ?? []);
  const latticeDistortion = computeLatticeDistortion(beforeLattice, afterLattice);

  let symmetryReduction: SymmetryReduction | null = null;
  if (sgBefore) {
    const inferredAfter = sgAfter || inferSpaceGroupFromLattice(beforeLattice, afterLattice, sgBefore);
    symmetryReduction = detectSymmetryReduction(sgBefore, inferredAfter);
  }

  const relaxedAtoms = afterPositions && afterPositions.length >= 2 ? afterPositions : (beforePositions && beforePositions.length >= 2 ? beforePositions : null);
  const bondLengthDistortion = relaxedAtoms ? computeBondLengthDistortion(relaxedAtoms) : null;
  const octahedralDistortion = relaxedAtoms ? computeOctahedralDistortion(relaxedAtoms) : null;
  const phononInstability = detectPhononInstability(phononFrequencies, phononHasImaginary, phononImaginaryCount, phononLowestFreq);

  let atomicScore = 0;
  if (atomicDistortion) {
    if (atomicDistortion.meanDisplacement >= DISPLACEMENT_THRESHOLD_LARGE) atomicScore = 0.5;
    else if (atomicDistortion.meanDisplacement >= DISPLACEMENT_THRESHOLD_MODERATE) atomicScore = 0.3;
    else if (atomicDistortion.meanDisplacement >= DISPLACEMENT_THRESHOLD_SMALL) atomicScore = 0.1;
  }

  let latticeScore = 0;
  if (latticeDistortion.strainMagnitude >= STRAIN_THRESHOLD_LARGE) latticeScore = 0.4;
  else if (latticeDistortion.strainMagnitude >= STRAIN_THRESHOLD_MODERATE) latticeScore = 0.2;
  if (Math.abs(latticeDistortion.volumeChangePct) >= VOLUME_CHANGE_THRESHOLD) latticeScore += 0.15;

  let symmetryScore = 0;
  if (symmetryReduction?.symmetryBroken) {
    symmetryScore = 0.3 * (1 - symmetryReduction.reductionFactor);
  }

  const angleDeviation = (
    Math.abs(latticeDistortion.deltaAlpha) +
    Math.abs(latticeDistortion.deltaBeta) +
    Math.abs(latticeDistortion.deltaGamma)
  ) / 90;
  latticeScore += Math.min(0.1, angleDeviation);

  let bondScore = 0;
  if (bondLengthDistortion) {
    if (bondLengthDistortion.bondLengthVariance > 0.05) bondScore = 0.4;
    else if (bondLengthDistortion.bondLengthVariance > 0.02) bondScore = 0.25;
    else if (bondLengthDistortion.bondLengthVariance > 0.005) bondScore = 0.1;
    if (bondLengthDistortion.distortedBondFraction > 0.3) bondScore += 0.1;
  }

  let octahedralScore = 0;
  if (octahedralDistortion) {
    if (octahedralDistortion.avgBondAngleVariance >= OCTAHEDRAL_ANGLE_VARIANCE_SEVERE) octahedralScore = 0.5;
    else if (octahedralDistortion.avgBondAngleVariance >= OCTAHEDRAL_ANGLE_VARIANCE_MODERATE) octahedralScore = 0.3;
    else if (octahedralDistortion.avgBondAngleVariance >= OCTAHEDRAL_ANGLE_VARIANCE_MILD) octahedralScore = 0.15;
  }

  let phononScore = 0;
  if (phononInstability) {
    if (phononInstability.instabilitySeverity === "severe") phononScore = 0.5;
    else if (phononInstability.instabilitySeverity === "moderate") phononScore = 0.3;
    else if (phononInstability.instabilitySeverity === "soft-mode") phononScore = 0.15;
  }

  const overallScore = Math.min(1.0,
    atomicScore * 0.25 +
    latticeScore * 0.20 +
    symmetryScore * 0.15 +
    bondScore * 0.15 +
    octahedralScore * 0.15 +
    phononScore * 0.10
  );
  const overallLevel = classifyLevel(overallScore);

  const jahnTeller = detectJahnTeller(formula, atomicDistortion, latticeDistortion) ||
    (octahedralDistortion !== null && octahedralDistortion.distortedSiteFraction > 0.5 &&
     octahedralDistortion.sites.some(s => s.distortionType === "elongated"));
  const peierls = detectPeierls(formula, atomicDistortion, latticeDistortion);
  const cdw = detectCDWSusceptibility(formula, latticeDistortion);

  const scRelevance = assessSCRelevance(overallLevel, symmetryReduction, jahnTeller, peierls, cdw,
    bondLengthDistortion, octahedralDistortion, phononInstability);

  return {
    formula,
    atomicDistortion,
    latticeDistortion,
    symmetryReduction,
    bondLengthDistortion,
    octahedralDistortion,
    phononInstability,
    overallLevel,
    overallScore: Math.round(overallScore * 10000) / 10000,
    jahn_teller_likely: jahnTeller,
    peierls_likely: peierls,
    cdw_susceptible: cdw,
    scRelevance,
    analyzedAt: Date.now(),
  };
}

export function analyzeRelaxationEntry(entry: RelaxationEntry): DistortionAnalysis {
  const sgBefore = guessSGFromPrototype(entry.prototype);
  return analyzeDistortion(
    entry.formula,
    entry.beforeLattice,
    entry.afterLattice,
    entry.beforePositions,
    entry.afterPositions,
    sgBefore,
    undefined,
  );
}

function guessSGFromPrototype(proto?: string): string | undefined {
  if (!proto) return undefined;
  const map: Record<string, string> = {
    Perovskite: "Pm-3m",
    A15: "Pm-3m",
    NaCl: "Fm-3m",
    "rock-salt": "Fm-3m",
    AlB2: "P6/mmm",
    Clathrate: "Pm-3m",
    ThCr2Si2: "I4/mmm",
    Spinel: "Fd-3m",
    Heusler: "Fm-3m",
    Laves: "Fd-3m",
    MAX: "P63/mmc",
    Fluorite: "Fm-3m",
    Pyrite: "Pa-3",
    Rutile: "P42/mnm",
    Wurtzite: "P63mc",
    Zincblende: "F-43m",
  };
  return map[proto];
}

export interface DistortionClassifierFeatures {
  meanDisplacement: number;
  maxDisplacement: number;
  bondVariance: number;
  distortedBondFraction: number;
  octahedralAngleVariance: number;
  volumeChangePct: number;
  strainMagnitude: number;
  spaceGroupChanged: boolean;
  coordinationDistorted: boolean;
  phononUnstable: boolean;
}

export interface DistortionClassifierResult {
  distortionProbability: number;
  prediction: "distorted" | "non-distorted";
  confidence: number;
  featureImportance: Record<string, number>;
  features: DistortionClassifierFeatures;
}

interface ClassifierWeights {
  meanDisplacement: number;
  maxDisplacement: number;
  bondVariance: number;
  distortedBondFraction: number;
  octahedralAngleVariance: number;
  volumeChangePct: number;
  strainMagnitude: number;
  spaceGroupChanged: number;
  coordinationDistorted: number;
  phononUnstable: number;
  bias: number;
}

const DEFAULT_WEIGHTS: ClassifierWeights = {
  meanDisplacement: 3.5,
  maxDisplacement: 1.2,
  bondVariance: 8.0,
  distortedBondFraction: 2.5,
  octahedralAngleVariance: 0.04,
  volumeChangePct: 0.15,
  strainMagnitude: 12.0,
  spaceGroupChanged: 1.8,
  coordinationDistorted: 1.5,
  phononUnstable: 2.0,
  bias: -2.0,
};

let trainedWeights: ClassifierWeights = { ...DEFAULT_WEIGHTS };
// Default weights are physically grounded (bond variance, strain, displacement)
// and are used for predictions immediately. classifierTrained is set to false
// until real DFT-derived examples reach CLASSIFIER_RETRAIN_THRESHOLD, at which
// point trainClassifier() sets it to true and updates trainedWeights.
let classifierTrained = false;
let classifierTrainCount = 0;
let classifierAccuracy = 0;
let classifierLastTrainedAt = 0;
const CLASSIFIER_RETRAIN_THRESHOLD = 20;

function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function extractFeatures(analysis: DistortionAnalysis): DistortionClassifierFeatures {
  return {
    meanDisplacement: analysis.atomicDistortion?.meanDisplacement ?? 0,
    maxDisplacement: analysis.atomicDistortion?.maxDisplacement ?? 0,
    bondVariance: analysis.bondLengthDistortion?.bondLengthVariance ?? 0,
    distortedBondFraction: analysis.bondLengthDistortion?.distortedBondFraction ?? 0,
    octahedralAngleVariance: analysis.octahedralDistortion?.avgBondAngleVariance ?? 0,
    volumeChangePct: Math.abs(analysis.latticeDistortion.volumeChangePct),
    strainMagnitude: analysis.latticeDistortion.strainMagnitude,
    spaceGroupChanged: analysis.symmetryReduction?.symmetryBroken ?? false,
    coordinationDistorted: (analysis.octahedralDistortion?.distortedSiteFraction ?? 0) > 0.3,
    phononUnstable: analysis.phononInstability?.hasImaginaryModes ?? false,
  };
}

function predictDistortion(features: DistortionClassifierFeatures, weights: ClassifierWeights): number {
  const z =
    features.meanDisplacement * weights.meanDisplacement +
    features.maxDisplacement * weights.maxDisplacement +
    features.bondVariance * weights.bondVariance +
    features.distortedBondFraction * weights.distortedBondFraction +
    features.octahedralAngleVariance * weights.octahedralAngleVariance +
    features.volumeChangePct * weights.volumeChangePct +
    features.strainMagnitude * weights.strainMagnitude +
    (features.spaceGroupChanged ? 1 : 0) * weights.spaceGroupChanged +
    (features.coordinationDistorted ? 1 : 0) * weights.coordinationDistorted +
    (features.phononUnstable ? 1 : 0) * weights.phononUnstable +
    weights.bias;
  return sigmoid(z);
}

function computeFeatureImportance(features: DistortionClassifierFeatures, weights: ClassifierWeights): Record<string, number> {
  const contributions: Record<string, number> = {
    meanDisplacement: features.meanDisplacement * weights.meanDisplacement,
    maxDisplacement: features.maxDisplacement * weights.maxDisplacement,
    bondVariance: features.bondVariance * weights.bondVariance,
    distortedBondFraction: features.distortedBondFraction * weights.distortedBondFraction,
    octahedralAngleVariance: features.octahedralAngleVariance * weights.octahedralAngleVariance,
    volumeChangePct: features.volumeChangePct * weights.volumeChangePct,
    strainMagnitude: features.strainMagnitude * weights.strainMagnitude,
    spaceGroupChanged: (features.spaceGroupChanged ? 1 : 0) * weights.spaceGroupChanged,
    coordinationDistorted: (features.coordinationDistorted ? 1 : 0) * weights.coordinationDistorted,
    phononUnstable: (features.phononUnstable ? 1 : 0) * weights.phononUnstable,
  };
  const total = Object.values(contributions).reduce((s, v) => s + Math.abs(v), 0) || 1;
  const importance: Record<string, number> = {};
  for (const [k, v] of Object.entries(contributions)) {
    importance[k] = Math.round((Math.abs(v) / total) * 10000) / 10000;
  }
  return importance;
}

function trainClassifier(history: DistortionAnalysis[]): void {
  if (history.length < CLASSIFIER_RETRAIN_THRESHOLD) return;

  const samples = history.map(a => ({
    features: extractFeatures(a),
    label: a.overallLevel !== "none" ? 1 : 0,
  }));

  const posCount = samples.filter(s => s.label === 1).length;
  const negCount = samples.length - posCount;
  if (posCount < 3 || negCount < 3) return;

  const lr = 0.01;
  const epochs = 50;
  const w = { ...trainedWeights };

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const sample of samples) {
      const f = sample.features;
      const p = predictDistortion(f, w);
      const error = p - sample.label;

      w.meanDisplacement -= lr * error * f.meanDisplacement;
      w.maxDisplacement -= lr * error * f.maxDisplacement;
      w.bondVariance -= lr * error * f.bondVariance;
      w.distortedBondFraction -= lr * error * f.distortedBondFraction;
      w.octahedralAngleVariance -= lr * error * f.octahedralAngleVariance;
      w.volumeChangePct -= lr * error * f.volumeChangePct;
      w.strainMagnitude -= lr * error * f.strainMagnitude;
      w.spaceGroupChanged -= lr * error * (f.spaceGroupChanged ? 1 : 0);
      w.coordinationDistorted -= lr * error * (f.coordinationDistorted ? 1 : 0);
      w.phononUnstable -= lr * error * (f.phononUnstable ? 1 : 0);
      w.bias -= lr * error;
    }
  }

  let correct = 0;
  for (const sample of samples) {
    const p = predictDistortion(sample.features, w);
    const pred = p >= 0.5 ? 1 : 0;
    if (pred === sample.label) correct++;
  }

  trainedWeights = w;
  classifierTrained = true;
  classifierTrainCount = samples.length;
  classifierAccuracy = Math.round((correct / samples.length) * 10000) / 10000;
  classifierLastTrainedAt = Date.now();
  console.log(`[DistortionML] Classifier retrained on ${samples.length} samples, accuracy=${classifierAccuracy}, pos=${posCount}, neg=${negCount}`);
}

export function classifyDistortion(analysis: DistortionAnalysis): DistortionClassifierResult {
  const features = extractFeatures(analysis);
  const prob = predictDistortion(features, trainedWeights);
  const prediction = prob >= 0.5 ? "distorted" : "non-distorted";
  const confidence = Math.abs(prob - 0.5) * 2;
  const featureImportance = computeFeatureImportance(features, trainedWeights);
  return {
    distortionProbability: Math.round(prob * 10000) / 10000,
    prediction,
    confidence: Math.round(confidence * 10000) / 10000,
    featureImportance,
    features,
  };
}

export function classifyFormulaDistortion(formula: string): DistortionClassifierResult | null {
  for (let i = distortionHistory.length - 1; i >= 0; i--) {
    if (distortionHistory[i].formula === formula) {
      return classifyDistortion(distortionHistory[i]);
    }
  }
  return null;
}

export function getClassifierStats(): {
  trained: boolean;
  trainCount: number;
  accuracy: number;
  lastTrainedAt: number;
  weights: Record<string, number>;
  recentPredictions: Array<{
    formula: string;
    prediction: string;
    probability: number;
    confidence: number;
    topFeature: string;
  }>;
} {
  const recentPredictions: Array<{
    formula: string;
    prediction: string;
    probability: number;
    confidence: number;
    topFeature: string;
  }> = [];

  const recentSlice = distortionHistory.slice(-15).reverse();
  for (const a of recentSlice) {
    const result = classifyDistortion(a);
    const entries = Object.entries(result.featureImportance);
    entries.sort((a, b) => b[1] - a[1]);
    recentPredictions.push({
      formula: a.formula,
      prediction: result.prediction,
      probability: result.distortionProbability,
      confidence: result.confidence,
      topFeature: entries[0]?.[0] ?? "unknown",
    });
  }

  const { bias, ...displayWeights } = trainedWeights;
  return {
    trained: classifierTrained,
    trainCount: classifierTrainCount,
    accuracy: classifierAccuracy,
    lastTrainedAt: classifierLastTrainedAt,
    weights: displayWeights as unknown as Record<string, number>,
    recentPredictions,
  };
}

const MAX_HISTORY = 2000;
const distortionHistory: DistortionAnalysis[] = [];

export function recordDistortionAnalysis(analysis: DistortionAnalysis): void {
  distortionHistory.push(analysis);
  if (distortionHistory.length > MAX_HISTORY) {
    distortionHistory.splice(0, Math.floor(MAX_HISTORY * 0.1));
  }
  if (distortionHistory.length >= CLASSIFIER_RETRAIN_THRESHOLD &&
      distortionHistory.length % CLASSIFIER_RETRAIN_THRESHOLD === 0) {
    trainClassifier(distortionHistory);
  }
}

export function getDistortionStats(): {
  totalAnalyzed: number;
  levelCounts: Record<DistortionLevel, number>;
  avgOverallScore: number;
  avgMeanDisplacement: number;
  avgStrainMagnitude: number;
  avgVolumeChangePct: number;
  symmetryBrokenCount: number;
  symmetryBrokenRate: number;
  jahnTellerCount: number;
  peierlsCount: number;
  cdwCount: number;
  changeTypeCounts: Record<string, number>;
  bondStats: {
    analyzedCount: number;
    avgBondVariance: number;
    avgDistortedFraction: number;
  };
  octahedralStats: {
    analyzedCount: number;
    avgAngleVariance: number;
    distortedSiteRate: number;
    octDistLevels: Record<string, number>;
  };
  phononStats: {
    analyzedCount: number;
    imaginaryCount: number;
    severityDistribution: Record<string, number>;
  };
  recentDistortions: Array<{
    formula: string;
    level: DistortionLevel;
    score: number;
    meanDisp: number;
    strain: number;
    volChange: number;
    symmetryBroken: boolean;
    bondVariance: number | null;
    octAngleVar: number | null;
    phononUnstable: boolean;
  }>;
} {
  const n = distortionHistory.length;
  if (n === 0) {
    return {
      totalAnalyzed: 0,
      levelCounts: { none: 0, small: 0, moderate: 0, large: 0, severe: 0 },
      avgOverallScore: 0,
      avgMeanDisplacement: 0,
      avgStrainMagnitude: 0,
      avgVolumeChangePct: 0,
      symmetryBrokenCount: 0,
      symmetryBrokenRate: 0,
      jahnTellerCount: 0,
      peierlsCount: 0,
      cdwCount: 0,
      changeTypeCounts: {},
      bondStats: { analyzedCount: 0, avgBondVariance: 0, avgDistortedFraction: 0 },
      octahedralStats: { analyzedCount: 0, avgAngleVariance: 0, distortedSiteRate: 0, octDistLevels: {} },
      phononStats: { analyzedCount: 0, imaginaryCount: 0, severityDistribution: {} },
      recentDistortions: [],
    };
  }

  const levelCounts: Record<DistortionLevel, number> = { none: 0, small: 0, moderate: 0, large: 0, severe: 0 };
  let scoreSum = 0;
  let dispSum = 0;
  let strainSum = 0;
  let volSum = 0;
  let symBroken = 0;
  let jtCount = 0;
  let peierlsCount = 0;
  let cdwCount = 0;
  const changeTypeCounts: Record<string, number> = {};
  let dispCount = 0;

  let bondAnalyzed = 0, bondVarSum = 0, bondDistFracSum = 0;
  let octAnalyzed = 0, octAngleVarSum = 0, octDistSiteSum = 0, octTotalSites = 0;
  const octDistLevels: Record<string, number> = {};
  let phononAnalyzed = 0, phononImagCount = 0;
  const phononSeverity: Record<string, number> = {};

  for (const d of distortionHistory) {
    levelCounts[d.overallLevel]++;
    scoreSum += d.overallScore;
    strainSum += d.latticeDistortion.strainMagnitude;
    volSum += Math.abs(d.latticeDistortion.volumeChangePct);
    if (d.atomicDistortion) {
      dispSum += d.atomicDistortion.meanDisplacement;
      dispCount++;
    }
    if (d.symmetryReduction?.symmetryBroken) {
      symBroken++;
      const ct = d.symmetryReduction.changeType;
      changeTypeCounts[ct] = (changeTypeCounts[ct] || 0) + 1;
    }
    if (d.jahn_teller_likely) jtCount++;
    if (d.peierls_likely) peierlsCount++;
    if (d.cdw_susceptible) cdwCount++;

    if (d.bondLengthDistortion) {
      bondAnalyzed++;
      bondVarSum += d.bondLengthDistortion.bondLengthVariance;
      bondDistFracSum += d.bondLengthDistortion.distortedBondFraction;
    }
    if (d.octahedralDistortion) {
      octAnalyzed++;
      octAngleVarSum += d.octahedralDistortion.avgBondAngleVariance;
      octDistSiteSum += d.octahedralDistortion.distortedSiteCount;
      octTotalSites += d.octahedralDistortion.siteCount;
      const lvl = d.octahedralDistortion.overallOctahedralDistortion;
      octDistLevels[lvl] = (octDistLevels[lvl] || 0) + 1;
    }
    if (d.phononInstability) {
      phononAnalyzed++;
      if (d.phononInstability.hasImaginaryModes) phononImagCount++;
      const sev = d.phononInstability.instabilitySeverity;
      phononSeverity[sev] = (phononSeverity[sev] || 0) + 1;
    }
  }

  const recent = distortionHistory.slice(-15).reverse().map(d => ({
    formula: d.formula,
    level: d.overallLevel,
    score: d.overallScore,
    meanDisp: d.atomicDistortion?.meanDisplacement ?? 0,
    strain: d.latticeDistortion.strainMagnitude,
    volChange: d.latticeDistortion.volumeChangePct,
    symmetryBroken: d.symmetryReduction?.symmetryBroken ?? false,
    bondVariance: d.bondLengthDistortion?.bondLengthVariance ?? null,
    octAngleVar: d.octahedralDistortion?.avgBondAngleVariance ?? null,
    phononUnstable: d.phononInstability?.hasImaginaryModes ?? false,
  }));

  return {
    totalAnalyzed: n,
    levelCounts,
    avgOverallScore: Math.round((scoreSum / n) * 10000) / 10000,
    avgMeanDisplacement: dispCount > 0 ? Math.round((dispSum / dispCount) * 10000) / 10000 : 0,
    avgStrainMagnitude: Math.round((strainSum / n) * 100000) / 100000,
    avgVolumeChangePct: Math.round((volSum / n) * 1000) / 1000,
    symmetryBrokenCount: symBroken,
    symmetryBrokenRate: Math.round((symBroken / n) * 1000) / 1000,
    jahnTellerCount: jtCount,
    peierlsCount,
    cdwCount,
    changeTypeCounts,
    bondStats: {
      analyzedCount: bondAnalyzed,
      avgBondVariance: bondAnalyzed > 0 ? Math.round((bondVarSum / bondAnalyzed) * 100000) / 100000 : 0,
      avgDistortedFraction: bondAnalyzed > 0 ? Math.round((bondDistFracSum / bondAnalyzed) * 1000) / 1000 : 0,
    },
    octahedralStats: {
      analyzedCount: octAnalyzed,
      avgAngleVariance: octAnalyzed > 0 ? Math.round((octAngleVarSum / octAnalyzed) * 100) / 100 : 0,
      distortedSiteRate: octTotalSites > 0 ? Math.round((octDistSiteSum / octTotalSites) * 1000) / 1000 : 0,
      octDistLevels,
    },
    phononStats: {
      analyzedCount: phononAnalyzed,
      imaginaryCount: phononImagCount,
      severityDistribution: phononSeverity,
    },
    recentDistortions: recent,
  };
}

export function getDistortionForFormula(formula: string): DistortionAnalysis | null {
  for (let i = distortionHistory.length - 1; i >= 0; i--) {
    if (distortionHistory[i].formula === formula) return distortionHistory[i];
  }
  return null;
}
