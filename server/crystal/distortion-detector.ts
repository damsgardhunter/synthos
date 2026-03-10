import type { LatticeParams, AtomPosition, RelaxationEntry } from "./relaxation-tracker";

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
  cdw: boolean
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
  sgAfter?: string
): DistortionAnalysis {
  const atomicDistortion = computeAtomicDistortion(beforePositions ?? [], afterPositions ?? []);
  const latticeDistortion = computeLatticeDistortion(beforeLattice, afterLattice);

  let symmetryReduction: SymmetryReduction | null = null;
  if (sgBefore) {
    const inferredAfter = sgAfter || inferSpaceGroupFromLattice(beforeLattice, afterLattice, sgBefore);
    symmetryReduction = detectSymmetryReduction(sgBefore, inferredAfter);
  }

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

  const overallScore = Math.min(1.0,
    atomicScore * 0.40 +
    latticeScore * 0.35 +
    symmetryScore * 0.25
  );
  const overallLevel = classifyLevel(overallScore);

  const jahnTeller = detectJahnTeller(formula, atomicDistortion, latticeDistortion);
  const peierls = detectPeierls(formula, atomicDistortion, latticeDistortion);
  const cdw = detectCDWSusceptibility(formula, latticeDistortion);

  const scRelevance = assessSCRelevance(overallLevel, symmetryReduction, jahnTeller, peierls, cdw);

  return {
    formula,
    atomicDistortion,
    latticeDistortion,
    symmetryReduction,
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
    undefined
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

const MAX_HISTORY = 2000;
const distortionHistory: DistortionAnalysis[] = [];

export function recordDistortionAnalysis(analysis: DistortionAnalysis): void {
  distortionHistory.push(analysis);
  if (distortionHistory.length > MAX_HISTORY) {
    distortionHistory.splice(0, Math.floor(MAX_HISTORY * 0.1));
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
  recentDistortions: Array<{
    formula: string;
    level: DistortionLevel;
    score: number;
    meanDisp: number;
    strain: number;
    volChange: number;
    symmetryBroken: boolean;
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
  }

  const recent = distortionHistory.slice(-15).reverse().map(d => ({
    formula: d.formula,
    level: d.overallLevel,
    score: d.overallScore,
    meanDisp: d.atomicDistortion?.meanDisplacement ?? 0,
    strain: d.latticeDistortion.strainMagnitude,
    volChange: d.latticeDistortion.volumeChangePct,
    symmetryBroken: d.symmetryReduction?.symmetryBroken ?? false,
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
    recentDistortions: recent,
  };
}

export function getDistortionForFormula(formula: string): DistortionAnalysis | null {
  for (let i = distortionHistory.length - 1; i >= 0; i--) {
    if (distortionHistory[i].formula === formula) return distortionHistory[i];
  }
  return null;
}
