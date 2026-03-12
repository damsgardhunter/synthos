import { isPathBreak, type DFTBandStructureResult, type BandEigenvalue, type BandCrossing, type BandInversion, type VanHoveSingularity, type EffectiveMass } from "./band-structure-calculator";
import type { FermiSurfaceResult, FermiPocket, NestingVector, FermiSurfaceMLFeatures } from "../physics/fermi-surface-engine";
import { getElementData, isTransitionMetal, isRareEarth, isActinide } from "../learning/elemental-data";

export interface DFTFermiPocket {
  index: number;
  type: "path-electron" | "path-hole";
  bandIndex: number;
  crossingCount: number;
  kRangeMin: number;
  kRangeMax: number;
  volume: number;
  cylindricalCharacter: number;
  avgVelocity: number;
  avgEnergy: number;
  orbitalCharacter: { s: number; p: number; d: number };
}

export interface DFTNestingAnalysis {
  vectors: NestingVector[];
  nestingScore: number;
  dominantQ: number[] | null;
  connectedPocketPairs: number;
}

export interface DFTTopologyFromBands {
  hasFlatBand: boolean;
  hasVHS: boolean;
  hasDiracCrossing: boolean;
  hasBandInversion: boolean;
  topologyScore: number;
  flatBandCount: number;
  vhsCount: number;
  diracCrossingCount: number;
  dosAtFermi: number;
  bandInversionCount: number;
  nodalLineIndicator: number;
  parityChanges: number;
  diracPointCount: number;
}

export interface DFTElectronicEnhancement {
  bandStructureType: string;
  fermiSurfaceTopology: string;
  densityOfStatesAtFermi: number;
  metallicity: number;
  nestingScore: number;
  vanHoveProximity: number;
  bandFlatness: number;
  flatBandIndicator: number;
}

export interface CrossingDispersion {
  bandIndex: number;
  kIndex: number;
  energy: number;
  type: "linear" | "quadratic" | "flat";
  curvature: number;
  velocity: number;
  velocityMS: number;
  socGap: number;
}

export interface DFTTopologicalClassification {
  topologicalClass: string;
  confidence: number;
  evidence: string[];
  socGapMeV: number;
  bandInversionDepthEv: number;
  diracPointCount: number;
  nodalLineCount: number;
  weylPointCount: number;
  flatBandNearFermi: boolean;
  z2Indicator: number;
  chiralWindingIndicator: number;
  classificationChain: string[];
  socTopologyBoost: number;
}

export interface FermiIsosurfacePoint {
  kx: number;
  ky: number;
  kz: number;
  bandIndex: number;
  velocity: [number, number, number];
  interpolatedEnergy: number;
}

export interface FermiIsosurface {
  points: FermiIsosurfacePoint[];
  totalPoints: number;
  sheetCount: number;
  enclosedVolumeFraction: number;
  avgVelocity: number;
  anisotropy: number;
}

export function classifyCrossingDispersion(bandResult: DFTBandStructureResult): CrossingDispersion[] {
  const crossings: CrossingDispersion[] = [];
  if (bandResult.eigenvalues.length < 5) return crossings;

  const nK = bandResult.eigenvalues.length;
  const kDist = new Float64Array(nK);
  for (let i = 0; i < nK; i++) {
    kDist[i] = bandResult.eigenvalues[i].kDistance;
  }

  for (const crossing of bandResult.bandCrossings) {
    const bIdx = crossing.bandIndex;
    const kFrac = crossing.kFraction;
    const kIdx = Math.round(kFrac * (nK - 1));

    const windowSize = 3;
    const lo = Math.max(0, kIdx - windowSize);
    const hi = Math.min(nK - 1, kIdx + windowSize);

    const localE: { dk: number; e: number }[] = [];
    const kRef = kDist[kIdx];
    for (let i = lo; i <= hi; i++) {
      const e = bandResult.eigenvalues[i]?.energies[bIdx];
      if (e === undefined) continue;
      localE.push({ dk: kDist[i] - kRef, e });
    }

    if (localE.length < 3) continue;

    let velocity = 0;
    let curvature = 0;

    if (localE.length >= 3) {
      let sumDk = 0, sumDk2 = 0, sumDk3 = 0, sumDk4 = 0;
      let sumE = 0, sumDkE = 0, sumDk2E = 0;
      for (const pt of localE) {
        const d = pt.dk;
        const d2 = d * d;
        sumDk += d; sumDk2 += d2; sumDk3 += d2 * d; sumDk4 += d2 * d2;
        sumE += pt.e; sumDkE += d * pt.e; sumDk2E += d2 * pt.e;
      }
      const n = localE.length;
      const S = [
        [n, sumDk, sumDk2],
        [sumDk, sumDk2, sumDk3],
        [sumDk2, sumDk3, sumDk4],
      ];
      const rhs = [sumE, sumDkE, sumDk2E];
      const det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1])
                - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0])
                + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
      if (Math.abs(det) > 1e-30) {
        const bCoeff = (rhs[0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1])
                      - S[0][1] * (rhs[1] * S[2][2] - S[1][2] * rhs[2])
                      + S[0][2] * (rhs[1] * S[2][1] - S[1][1] * rhs[2])) / det;
        const aCoeff = (S[0][0] * (rhs[1] * S[2][2] - S[1][2] * rhs[2])
                      - rhs[0] * (S[1][0] * S[2][2] - S[1][2] * S[2][0])
                      + S[0][2] * (S[1][0] * rhs[2] - rhs[1] * S[2][0])) / det;
        const cCoeff = (S[0][0] * (S[1][1] * rhs[2] - rhs[1] * S[2][1])
                      - S[0][1] * (S[1][0] * rhs[2] - rhs[1] * S[2][0])
                      + rhs[0] * (S[1][0] * S[2][1] - S[1][1] * S[2][0])) / det;
        void cCoeff;
        velocity = Math.abs(bCoeff);
        curvature = 2 * Math.abs(aCoeff);
      }
    }

    let socGap = 0;
    if (bIdx + 1 < bandResult.nBands) {
      const e1 = bandResult.eigenvalues[kIdx]?.energies[bIdx];
      const e2 = bandResult.eigenvalues[kIdx]?.energies[bIdx + 1];
      if (e1 !== undefined && e2 !== undefined) {
        const gap = Math.abs(e2 - e1);
        if (gap < 0.3 && gap > 0.001) {
          socGap = gap * 1000;
        }
      }
    }

    let type: "linear" | "quadratic" | "flat";
    if (velocity < 0.1) {
      type = "flat";
    } else if (curvature < velocity * 0.3) {
      type = "linear";
    } else {
      type = "quadratic";
    }

    const EV_ANG_TO_MS = 1.602176634e-19 / (1.054571817e-34 * 1e10);
    crossings.push({
      bandIndex: bIdx,
      kIndex: kIdx,
      energy: crossing.energy,
      type,
      curvature,
      velocity,
      velocityMS: velocity * EV_ANG_TO_MS,
      socGap,
    });
  }

  return crossings;
}

export function detectSOCGaps(bandResult: DFTBandStructureResult): { kIndex: number; bandPair: [number, number]; gapMeV: number; atFermi: boolean }[] {
  const gaps: { kIndex: number; bandPair: [number, number]; gapMeV: number; atFermi: boolean }[] = [];
  const MAX_BAND_OFFSET = 3;

  for (let ki = 0; ki < bandResult.eigenvalues.length; ki++) {
    const kpt = bandResult.eigenvalues[ki];
    for (let b = 0; b < kpt.energies.length - 1; b++) {
      const e1 = kpt.energies[b];
      if (e1 === undefined) continue;

      const upperLimit = Math.min(b + MAX_BAND_OFFSET, kpt.energies.length - 1);
      for (let b2 = b + 1; b2 <= upperLimit; b2++) {
        const e2 = kpt.energies[b2];
        if (e2 === undefined) continue;
        const gap = e2 - e1;
        if (gap > 0.001 && gap < 0.15) {
          const midE = (e1 + e2) / 2;
          const atFermi = Math.abs(midE) < 0.3;
          gaps.push({
            kIndex: ki,
            bandPair: [b, b2],
            gapMeV: gap * 1000,
            atFermi,
          });
        }
      }
    }
  }

  if (gaps.length > 1000) {
    gaps.length = 1000;
  }
  gaps.sort((a, b) => a.gapMeV - b.gapMeV);
  return gaps.slice(0, 20);
}

export function computeFermiIsosurface(bandResult: DFTBandStructureResult): FermiIsosurface {
  const isoPts: FermiIsosurfacePoint[] = [];

  if (bandResult.eigenvalues.length < 3 || bandResult.nBands === 0) {
    return { points: [], totalPoints: 0, sheetCount: 0, enclosedVolumeFraction: 0, avgVelocity: 0, anisotropy: 1 };
  }

  const bandSheets = new Set<number>();

  for (let b = 0; b < bandResult.nBands; b++) {
    for (let ki = 0; ki < bandResult.eigenvalues.length - 1; ki++) {
      if (isPathBreak(bandResult.eigenvalues, ki + 1)) {
        const c0 = bandResult.eigenvalues[ki].kCoords;
        const c1 = bandResult.eigenvalues[ki + 1].kCoords;
        const coordDist = Math.sqrt(
          (c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2 + (c1[2] - c0[2]) ** 2
        );
        if (coordDist > 0.01) continue;
      }
      const e0 = bandResult.eigenvalues[ki].energies[b];
      const e1 = bandResult.eigenvalues[ki + 1].energies[b];
      if (e0 === undefined || e1 === undefined) continue;

      if ((e0 <= 0 && e1 > 0) || (e0 > 0 && e1 <= 0)) {
        const frac = Math.abs(e0) / (Math.abs(e0) + Math.abs(e1) + 1e-12);
        const k0 = bandResult.eigenvalues[ki].kCoords;
        const k1 = bandResult.eigenvalues[ki + 1].kCoords;
        const kx = k0[0] + frac * (k1[0] - k0[0]);
        const ky = k0[1] + frac * (k1[1] - k0[1]);
        const kz = k0[2] + frac * (k1[2] - k0[2]);

        const dk = bandResult.eigenvalues[ki + 1].kDistance - bandResult.eigenvalues[ki].kDistance;
        const dE = e1 - e0;
        const vMag = dk > 1e-6 ? Math.abs(dE / dk) : 0;

        const dkx = k1[0] - k0[0];
        const dky = k1[1] - k0[1];
        const dkz = k1[2] - k0[2];
        const kNorm = Math.sqrt(dkx * dkx + dky * dky + dkz * dkz) + 1e-12;
        const vel: [number, number, number] = [
          vMag * dkx / kNorm,
          vMag * dky / kNorm,
          vMag * dkz / kNorm,
        ];

        isoPts.push({
          kx, ky, kz,
          bandIndex: b,
          velocity: vel,
          interpolatedEnergy: 0,
        });
        bandSheets.add(b);
      }
    }
  }

  let avgVel = 0;
  let vxSum = 0, vySum = 0, vzSum = 0;
  for (const pt of isoPts) {
    const vm = Math.sqrt(pt.velocity[0] ** 2 + pt.velocity[1] ** 2 + pt.velocity[2] ** 2);
    avgVel += vm;
    vxSum += Math.abs(pt.velocity[0]);
    vySum += Math.abs(pt.velocity[1]);
    vzSum += Math.abs(pt.velocity[2]);
  }
  avgVel = isoPts.length > 0 ? avgVel / isoPts.length : 0;
  const vMax = Math.max(vxSum, vySum, vzSum, 1e-6);
  const vMin = Math.min(vxSum, vySum, vzSum);
  const anisotropy = vMax > 1e-6 ? vMax / (vMin + 1e-6) : 1;

  const enclosedVol = isoPts.length > 0 ? isoPts.length / (bandResult.nKPoints * bandResult.nBands) : 0;

  return {
    points: isoPts,
    totalPoints: isoPts.length,
    sheetCount: bandSheets.size,
    enclosedVolumeFraction: enclosedVol,
    avgVelocity: avgVel,
    anisotropy: Math.min(20, anisotropy),
  };
}

const CENTROSYMMETRIC_SG = new Set([
  "p-1","p2/m","p21/m","c2/m","pmmm","pnma","cmcm","cmmm","fmmm","immm",
  "p4/m","p42/m","p4/mmm","p4/nmm","p42/mnm","i4/m","i4/mmm","i41/amd",
  "p-3","r-3","p-3m1","p-31m","r-3m","p6/m","p6/mmm","p63/mmc",
  "pm-3","pn-3","fm-3","im-3","pm-3m","pn-3m","pm-3n","fm-3m","fm-3c",
  "fd-3m","im-3m","ia-3","ia-3d","pa-3",
  "p121/c1","p21/c","c2/c","pbca","pbcn","p42/nmc","r-3c","i41/a",
  "cmce","cmca","fddd","p-1","i4/mcm","p63/m",
]);

function isCentrosymmetric(spaceGroup: string | undefined): boolean | null {
  if (!spaceGroup) return null;
  const sg = spaceGroup.toLowerCase().replace(/\s+/g, "");
  return CENTROSYMMETRIC_SG.has(sg);
}

interface TopologyRuleContext {
  topo: DFTTopologyFromBands;
  maxSOCGap: number;
  socStrength: number;
  isMetallic: boolean;
  linearCrossings: CrossingDispersion[];
  weylPointCount: number;
  nodalLineCandidates: number;
  flatBandScore: number;
  spaceGroup?: string;
  crossingCount: number;
  parityChanges: number;
}

interface TopologyRuleResult {
  topologicalClass: string;
  confidence: number;
  evidence: string[];
  chainStep: string;
}

interface TopologyRule {
  name: string;
  match: (ctx: TopologyRuleContext) => boolean;
  apply: (ctx: TopologyRuleContext) => TopologyRuleResult;
}

const TOPOLOGY_RULES: TopologyRule[] = [
  {
    name: "strong-TI",
    match: (ctx) => ctx.topo.hasBandInversion && ctx.maxSOCGap > 10 && ctx.socStrength > 0.2 && !ctx.isMetallic,
    apply: (ctx) => {
      let confidence = Math.min(0.95, 0.5 + ctx.topo.bandInversionCount * 0.1 + ctx.maxSOCGap / 200);
      const evidence = [
        `Band inversion with SOC gap ${ctx.maxSOCGap.toFixed(0)} meV`,
        `${ctx.topo.bandInversionCount} band inversion(s) detected`,
      ];
      if (ctx.parityChanges > 3) {
        evidence.push(`${ctx.parityChanges} parity changes (Z2 indicator)`);
        confidence = Math.min(0.95, confidence + 0.1);
      }
      return { topologicalClass: "strong-topological-insulator", confidence, evidence, chainStep: "band-inversion+SOC-gap->TI" };
    },
  },
  {
    name: "topological-metal",
    match: (ctx) => ctx.topo.hasBandInversion && ctx.maxSOCGap > 10 && ctx.socStrength > 0.15 && ctx.isMetallic,
    apply: (ctx) => ({
      topologicalClass: "topological-metal",
      confidence: Math.min(0.9, 0.4 + ctx.topo.bandInversionCount * 0.1 + ctx.maxSOCGap / 300),
      evidence: [`Metallic with band inversion and SOC gap ${ctx.maxSOCGap.toFixed(0)} meV`],
      chainStep: "band-inversion+SOC-gap+metallic->TM",
    }),
  },
  {
    name: "Weyl-semimetal",
    match: (ctx) => {
      if (ctx.linearCrossings.length === 0 || !ctx.linearCrossings.some(c => Math.abs(c.energy) < 0.2)) return false;
      return ctx.weylPointCount > 0 && ctx.socStrength > 0.1 && isCentrosymmetric(ctx.spaceGroup) === false;
    },
    apply: (ctx) => {
      const evidence = [
        `${ctx.weylPointCount} Weyl-like crossing(s) with linear dispersion`,
        `Non-centrosymmetric space group (${ctx.spaceGroup}) allows Weyl nodes`,
      ];
      if (ctx.socStrength > 0.3) evidence.push("Strong SOC lifts degeneracy");
      return {
        topologicalClass: "Weyl-semimetal",
        confidence: Math.min(0.9, 0.4 + ctx.weylPointCount * 0.15 + ctx.socStrength * 0.3),
        evidence,
        chainStep: "linear-crossing+SOC+broken-inversion->Weyl",
      };
    },
  },
  {
    name: "Dirac-conservative",
    match: (ctx) => {
      if (ctx.linearCrossings.length === 0 || !ctx.linearCrossings.some(c => Math.abs(c.energy) < 0.2)) return false;
      return ctx.weylPointCount > 0 && ctx.socStrength > 0.1 && isCentrosymmetric(ctx.spaceGroup) === null;
    },
    apply: (ctx) => ({
      topologicalClass: "Dirac-semimetal",
      confidence: Math.min(0.75, 0.3 + ctx.weylPointCount * 0.12 + ctx.socStrength * 0.2),
      evidence: [`${ctx.weylPointCount} linear crossing(s) with SOC — classified as Dirac (space group unknown, cannot confirm broken inversion symmetry for Weyl)`],
      chainStep: "linear-crossing+SOC+unknown-SG->Dirac-conservative",
    }),
  },
  {
    name: "Dirac-semimetal",
    match: (ctx) => ctx.linearCrossings.length > 0 && ctx.linearCrossings.some(c => Math.abs(c.energy) < 0.2),
    apply: (ctx) => {
      const nearFermiLinear = ctx.linearCrossings.filter(c => Math.abs(c.energy) < 0.2);
      const avgVel = nearFermiLinear.reduce((s, c) => s + c.velocity, 0) / nearFermiLinear.length;
      return {
        topologicalClass: "Dirac-semimetal",
        confidence: Math.min(0.85, 0.35 + nearFermiLinear.length * 0.12),
        evidence: [
          `${nearFermiLinear.length} Dirac crossing(s) with linear dispersion near Fermi level`,
          `Average Fermi velocity: ${avgVel.toFixed(2)} eV*A`,
        ],
        chainStep: "linear-crossing->Dirac",
      };
    },
  },
  {
    name: "nodal-line-semimetal",
    match: (ctx) => ctx.nodalLineCandidates > 0.5,
    apply: (ctx) => {
      const evidence = [`Nodal line indicator: ${ctx.nodalLineCandidates.toFixed(2)}`];
      if (ctx.crossingCount > 4) evidence.push(`${ctx.crossingCount} crossings along high-symmetry path`);
      return {
        topologicalClass: "nodal-line-semimetal",
        confidence: Math.min(0.85, 0.3 + ctx.nodalLineCandidates * 0.3),
        evidence,
        chainStep: "nodal-line-indicator->NLS",
      };
    },
  },
  {
    name: "topological-flat-band",
    match: (ctx) => ctx.topo.hasFlatBand && ctx.topo.flatBandCount > 0 && ctx.flatBandScore > 0.5 && ctx.topo.hasBandInversion,
    apply: (ctx) => ({
      topologicalClass: "topological-flat-band",
      confidence: Math.min(0.8, 0.3 + ctx.topo.flatBandCount * 0.1 + ctx.flatBandScore * 0.3),
      evidence: [`${ctx.topo.flatBandCount} flat band(s) with band inversion`],
      chainStep: "flat-band+inversion->TFB",
    }),
  },
  {
    name: "flat-band-system",
    match: (ctx) => ctx.topo.hasFlatBand && ctx.topo.flatBandCount > 0 && ctx.flatBandScore > 0.5,
    apply: (ctx) => ({
      topologicalClass: "flat-band-system",
      confidence: Math.min(0.75, 0.25 + ctx.flatBandScore * 0.3),
      evidence: [`${ctx.topo.flatBandCount} flat band(s) near Fermi level (score: ${ctx.flatBandScore.toFixed(2)})`],
      chainStep: "flat-band->FBS",
    }),
  },
  {
    name: "weak-topological",
    match: (ctx) => ctx.topo.hasBandInversion && ctx.socStrength > 0.1,
    apply: (ctx) => ({
      topologicalClass: "weak-topological",
      confidence: Math.min(0.7, 0.2 + ctx.topo.bandInversionCount * 0.1),
      evidence: [`${ctx.topo.bandInversionCount} band inversion(s) with moderate SOC`],
      chainStep: "band-inversion+weak-SOC->WTI",
    }),
  },
  {
    name: "SOC-enhanced",
    match: (ctx) => ctx.socStrength > 0.3 && ctx.topo.topologyScore > 0.2,
    apply: (ctx) => ({
      topologicalClass: "SOC-enhanced",
      confidence: Math.min(0.6, 0.15 + ctx.socStrength * 0.3),
      evidence: [`Strong SOC (${ctx.socStrength.toFixed(2)}) with topology score ${ctx.topo.topologyScore.toFixed(2)}`],
      chainStep: "strong-SOC->SOC-enhanced",
    }),
  },
];

export function estimateOrbitalFractions(elements: string[]): { s: number; p: number; d: number } {
  let sSum = 0, pSum = 0, dSum = 0;
  let totalWeight = 0;

  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const ve = data.valenceElectrons || 1;
    totalWeight += ve;
    if (el === "H") {
      sSum += ve;
    } else if (["B", "C", "N", "O", "Si", "P", "S", "Se", "Te", "F", "Cl", "Br", "I"].includes(el)) {
      pSum += ve;
    } else if (isTransitionMetal(el)) {
      dSum += ve;
    } else if (isRareEarth(el) || isActinide(el)) {
      dSum += ve;
    } else {
      const period = data.atomicNumber <= 2 ? 1 : data.atomicNumber <= 10 ? 2 : data.atomicNumber <= 18 ? 3 : 4;
      if (period <= 2) sSum += ve;
      else if (period === 3) pSum += ve;
      else {
        sSum += ve * 0.3;
        pSum += ve * 0.7;
      }
    }
  }

  if (totalWeight > 0) {
    sSum /= totalWeight;
    pSum /= totalWeight;
    dSum /= totalWeight;
  } else {
    sSum = 0.33;
    pSum = 0.34;
    dSum = 0.33;
  }

  return { s: sSum, p: pSum, d: dSum };
}

export function classifyDFTTopology(bandResult: DFTBandStructureResult, socStrength: number, spaceGroup?: string): DFTTopologicalClassification {
  const chain: string[] = ["structure", "band-structure", "detect-crossings"];

  const crossings = classifyCrossingDispersion(bandResult);
  const socGaps = detectSOCGaps(bandResult);
  const topo = extractTopologyFromDFT(bandResult);
  const isosurface = computeFermiIsosurface(bandResult);

  const linearCrossings = crossings.filter(c => c.type === "linear");

  const socGapsAtFermi = socGaps.filter(g => g.atFermi);
  const maxSOCGap = socGapsAtFermi.length > 0 ? Math.max(...socGapsAtFermi.map(g => g.gapMeV)) : 0;

  const inversionDepths = bandResult.bandInversions.map(inv => inv.energyGap);
  const avgInversionDepth = inversionDepths.length > 0 ? inversionDepths.reduce((a, b) => a + b, 0) / inversionDepths.length : 0;

  const nodalLineCandidates = topo.nodalLineIndicator;

  let weylPointCount = 0;
  for (const c of linearCrossings) {
    if (c.socGap < 5 && c.velocity > 1.0) {
      weylPointCount++;
    }
  }

  const z2Indicator = computeZ2FromBands(bandResult, topo);
  const chiralWinding = computeChiralWinding(crossings, bandResult);

  chain.push("classify-topology");

  const ctx: TopologyRuleContext = {
    topo,
    maxSOCGap,
    socStrength,
    isMetallic: bandResult.isMetallicAlongPath,
    linearCrossings,
    weylPointCount,
    nodalLineCandidates,
    flatBandScore: bandResult.flatBandScore,
    spaceGroup,
    crossingCount: crossings.length,
    parityChanges: topo.parityChanges,
  };

  let topologicalClass = "trivial";
  let confidence = 0;
  const evidence: string[] = [];

  for (const rule of TOPOLOGY_RULES) {
    if (rule.match(ctx)) {
      const result = rule.apply(ctx);
      topologicalClass = result.topologicalClass;
      confidence = result.confidence;
      evidence.push(...result.evidence);
      chain.push(result.chainStep);
      break;
    }
  }

  if (topologicalClass === "trivial" && topo.topologyScore > 0.15) {
    confidence = 0.1;
    evidence.push(`Low topology score (${topo.topologyScore.toFixed(2)}), no definitive classification`);
  }

  if (isosurface.sheetCount > 3 && topologicalClass !== "trivial") {
    evidence.push(`Fermi isosurface: ${isosurface.sheetCount} sheets, ${isosurface.totalPoints} crossing points`);
  }

  let socTopologyBoost = 0;
  if (socGapsAtFermi.length > 0 && maxSOCGap > 5) {
    socTopologyBoost = Math.min(0.5, maxSOCGap / 100 + socGapsAtFermi.length * 0.05);
    if (topologicalClass !== "trivial") {
      socTopologyBoost += 0.2;
    }
    socTopologyBoost = Math.min(1.0, socTopologyBoost);
  }

  return {
    topologicalClass,
    confidence,
    evidence,
    socGapMeV: maxSOCGap,
    bandInversionDepthEv: avgInversionDepth,
    diracPointCount: topo.diracPointCount,
    nodalLineCount: Math.round(nodalLineCandidates),
    weylPointCount,
    flatBandNearFermi: topo.hasFlatBand && bandResult.flatBandScore > 0.3,
    z2Indicator,
    chiralWindingIndicator: chiralWinding,
    classificationChain: chain,
    socTopologyBoost,
  };
}

function computeZ2FromBands(bandResult: DFTBandStructureResult, topo: DFTTopologyFromBands): number {
  let z2 = 0;

  if (topo.parityChanges > 0) {
    const oddParity = topo.parityChanges % 2 === 1;
    z2 = oddParity ? 0.8 : 0.3;
  }

  if (topo.hasBandInversion) {
    z2 = Math.max(z2, 0.5 + topo.bandInversionCount * 0.1);
  }

  if (topo.nodalLineIndicator > 0.5 && topo.hasBandInversion) {
    z2 = Math.max(z2, 0.7);
  }

  return Math.min(1.0, z2);
}

function computeChiralWinding(crossings: CrossingDispersion[], bandResult: DFTBandStructureResult): number {
  if (crossings.length < 2) return 0;

  const sorted = [...crossings].sort((a, b) => a.kIndex - b.kIndex);
  let signedSlopeSum = 0;
  let slopeCount = 0;

  for (const c of sorted) {
    const bIdx = c.bandIndex;
    const kIdx = c.kIndex;
    const lo = Math.max(0, kIdx - 1);
    const hi = Math.min(bandResult.eigenvalues.length - 1, kIdx + 1);
    const eLo = bandResult.eigenvalues[lo]?.energies[bIdx];
    const eHi = bandResult.eigenvalues[hi]?.energies[bIdx];
    const kLo = bandResult.eigenvalues[lo]?.kDistance;
    const kHi = bandResult.eigenvalues[hi]?.kDistance;

    if (eLo !== undefined && eHi !== undefined && kLo !== undefined && kHi !== undefined) {
      const dk = kHi - kLo;
      if (Math.abs(dk) > 1e-8) {
        const signedSlope = (eHi - eLo) / dk;
        signedSlopeSum += Math.sign(signedSlope);
        slopeCount++;
      }
    }
  }

  if (slopeCount < 2) return 0;
  return Math.abs(signedSlopeSum) / slopeCount;
}

export function extractFermiPockets(bandResult: DFTBandStructureResult, elements?: string[]): DFTFermiPocket[] {
  const pockets: DFTFermiPocket[] = [];
  if (!bandResult.eigenvalues.length || bandResult.nBands === 0) return pockets;

  const bandCrossingMap = new Map<number, BandCrossing[]>();
  for (const crossing of bandResult.bandCrossings) {
    if (!bandCrossingMap.has(crossing.bandIndex)) {
      bandCrossingMap.set(crossing.bandIndex, []);
    }
    bandCrossingMap.get(crossing.bandIndex)!.push(crossing);
  }

  let pocketIdx = 0;
  for (let b = 0; b < bandResult.nBands; b++) {
    const bandEnergies = bandResult.eigenvalues
      .map(kpt => ({ energy: kpt.energies[b], kDist: kpt.kDistance, kIdx: kpt.kIndex }))
      .filter(e => e.energy !== undefined);

    if (bandEnergies.length < 3) continue;

    const aboveCount = bandEnergies.filter(e => e.energy > 0).length;
    const belowCount = bandEnergies.filter(e => e.energy <= 0).length;
    const crossings = bandCrossingMap.get(b) || [];

    if (crossings.length < 1) continue;

    const type: "path-electron" | "path-hole" = aboveCount > belowCount ? "path-electron" : "path-hole";

    const crossingFracs = bandEnergies.filter(e => Math.abs(e.energy) < 0.3);
    const volume = crossingFracs.length / bandEnergies.length;

    let velocitySum = 0;
    let velCount = 0;
    for (let i = 1; i < bandEnergies.length; i++) {
      const dE = bandEnergies[i].energy - bandEnergies[i - 1].energy;
      const dk = bandEnergies[i].kDist - bandEnergies[i - 1].kDist;
      if (dk > 0 && Math.abs(bandEnergies[i].energy) < 1.0) {
        velocitySum += Math.abs(dE / dk);
        velCount++;
      }
    }

    const bMin = Math.min(...bandEnergies.map(e => e.energy));
    const bMax = Math.max(...bandEnergies.map(e => e.energy));
    const bRange = bMax - bMin;
    const cylindrical = bRange < 0.3 ? 0.8 : bRange < 1.0 ? 0.5 : 0.2;

    const stoichFallback = elements && elements.length > 0
      ? estimateOrbitalFractions(elements)
      : { s: 0.33, p: 0.34, d: 0.33 };

    let orbChar = { ...stoichFallback };

    const fermiKpts = bandResult.eigenvalues.filter(kpt =>
      kpt.weights?.[b] && Math.abs(kpt.energies[b]) < 1.0
    );
    if (fermiKpts.length > 0) {
      let sSum = 0, pSum = 0, dSum = 0;
      for (const kpt of fermiKpts) {
        const w = kpt.weights![b];
        sSum += w.s;
        pSum += w.p;
        dSum += w.d + w.f;
      }
      const wTotal = sSum + pSum + dSum;
      if (wTotal > 0.01) {
        orbChar = { s: sSum / wTotal, p: pSum / wTotal, d: dSum / wTotal };
      }
    }

    pockets.push({
      index: pocketIdx++,
      type,
      bandIndex: b,
      crossingCount: crossings.length,
      kRangeMin: crossings.length > 0 ? Math.min(...crossings.map(c => c.kFraction)) : 0,
      kRangeMax: crossings.length > 0 ? Math.max(...crossings.map(c => c.kFraction)) : 1,
      volume,
      cylindricalCharacter: cylindrical,
      avgVelocity: velCount > 0 ? velocitySum / velCount : 0,
      avgEnergy: bandEnergies.reduce((s, e) => s + e.energy, 0) / bandEnergies.length,
      orbitalCharacter: orbChar,
    });
  }

  return pockets;
}

function stridedSample(arr: number[], maxSamples: number): number[] {
  if (arr.length <= maxSamples) return arr;
  const stride = Math.ceil(arr.length / maxSamples);
  const result: number[] = [];
  for (let i = 0; i < arr.length; i += stride) {
    result.push(arr[i]);
  }
  return result;
}

export function computeDFTNesting(pockets: DFTFermiPocket[], bandResult: DFTBandStructureResult): DFTNestingAnalysis {
  if (pockets.length < 2) {
    return { vectors: [], nestingScore: 0, dominantQ: null, connectedPocketPairs: 0 };
  }

  const crossingKPoints = new Map<number, number[]>();
  for (const pocket of pockets) {
    const kIndices: number[] = [];
    for (const kpt of bandResult.eigenvalues) {
      const e = kpt.energies[pocket.bandIndex];
      if (e !== undefined && Math.abs(e) < 0.2) {
        kIndices.push(kpt.kIndex);
      }
    }
    crossingKPoints.set(pocket.index, kIndices);
  }

  const qBins = new Map<string, { q: number[]; count: number; pocketPair: [number, number] }>();
  const binRes = 0.05;

  for (let i = 0; i < pockets.length; i++) {
    for (let j = i + 1; j < pockets.length; j++) {
      const ki = crossingKPoints.get(pockets[i].index) || [];
      const kj = crossingKPoints.get(pockets[j].index) || [];

      const sampleI = stridedSample(ki, 20);
      const sampleJ = stridedSample(kj, 20);

      for (const kIdxI of sampleI) {
        for (const kIdxJ of sampleJ) {
          const kptI = bandResult.eigenvalues[kIdxI];
          const kptJ = bandResult.eigenvalues[kIdxJ];
          if (!kptI || !kptJ) continue;

          const q = [
            kptJ.kCoords[0] - kptI.kCoords[0],
            kptJ.kCoords[1] - kptI.kCoords[1],
            kptJ.kCoords[2] - kptI.kCoords[2],
          ];
          const binKey = q.map(v => Math.round(v / binRes) * binRes).map(v => v.toFixed(3)).join(",");

          if (qBins.has(binKey)) {
            qBins.get(binKey)!.count++;
          } else {
            qBins.set(binKey, { q: q.map(v => Math.round(v / binRes) * binRes), count: 1, pocketPair: [i, j] });
          }
        }
      }
    }
  }

  const sortedBins = Array.from(qBins.values()).sort((a, b) => b.count - a.count);
  const maxCount = sortedBins.length > 0 ? sortedBins[0].count : 1;

  const nestingVectors: NestingVector[] = sortedBins.slice(0, 5).map(bin => ({
    q: bin.q,
    strength: bin.count / maxCount,
    connectedPockets: bin.pocketPair,
  }));

  const nestingScore = sortedBins.length > 0
    ? Math.min(1.0, (sortedBins[0].count / Math.max(1, maxCount)) * (pockets.length > 2 ? 1.2 : 1.0))
    : 0;

  const connectedPairs = new Set(sortedBins.filter(b => b.count > 1).map(b => `${b.pocketPair[0]}-${b.pocketPair[1]}`));

  return {
    vectors: nestingVectors,
    nestingScore,
    dominantQ: sortedBins.length > 0 ? sortedBins[0].q : null,
    connectedPocketPairs: connectedPairs.size,
  };
}

export function buildFermiSurfaceFromDFT(bandResult: DFTBandStructureResult): FermiSurfaceResult {
  const pockets = extractFermiPockets(bandResult);
  const nesting = computeDFTNesting(pockets, bandResult);

  const electronPockets = pockets.filter(p => p.type === "path-electron");
  const holePockets = pockets.filter(p => p.type === "path-hole");

  const totalElectronVol = electronPockets.reduce((s, p) => s + p.volume, 0);
  const totalHoleVol = holePockets.reduce((s, p) => s + p.volume, 0);
  const totalVol = totalElectronVol + totalHoleVol;
  const ehBalance = totalVol > 0 ? 1.0 - Math.abs(totalElectronVol - totalHoleVol) / totalVol : 0;

  const avgCylindrical = pockets.length > 0
    ? pockets.reduce((s, p) => s + p.cylindricalCharacter, 0) / pockets.length : 0;

  let fsDim = 3.0;
  if (avgCylindrical > 0.7) fsDim = 2.0;
  else if (avgCylindrical > 0.5) fsDim = 2.5;
  const pathDimensionality = fsDim;

  const SIGMA_BAND_MIN_S_CHAR = 0.3;
  const SIGMA_BAND_MIN_VEL_EV_ANG = 2.0;
  const SIGMA_BAND_VEL_SCALE_EV_ANG = 8.0;

  let sigmaBand = 0;
  for (const pocket of pockets) {
    if (pocket.orbitalCharacter.s > SIGMA_BAND_MIN_S_CHAR && pocket.avgVelocity > SIGMA_BAND_MIN_VEL_EV_ANG) {
      const velScore = Math.min(0.5, (pocket.avgVelocity - SIGMA_BAND_MIN_VEL_EV_ANG) / SIGMA_BAND_VEL_SCALE_EV_ANG);
      sigmaBand = Math.max(sigmaBand, pocket.orbitalCharacter.s * 0.6 + velScore + 0.1);
    }
  }
  sigmaBand = Math.min(1.0, sigmaBand);

  const multiBand = Math.min(1.0, pockets.length / 6);

  const fermiPockets: FermiPocket[] = pockets.map(p => ({
    index: p.index,
    type: (p.type === "path-electron" ? "electron" : "hole") as "electron" | "hole",
    volume: p.volume,
    cylindricalCharacter: p.cylindricalCharacter,
    orbitalCharacter: { ...p.orbitalCharacter, f: 0 },
    bandIndex: p.bandIndex,
    avgVelocity: p.avgVelocity,
  }));

  const mlFeatures: FermiSurfaceMLFeatures = {
    fermiPocketCount: pockets.length,
    electronHoleBalance: ehBalance,
    fsDimensionality: pathDimensionality,
    sigmaBandPresence: sigmaBand,
    multiBandScore: multiBand,
  };

  return {
    formula: bandResult.formula,
    fermiEnergy: bandResult.fermiEnergy,
    pocketCount: pockets.length,
    pockets: fermiPockets,
    electronPocketCount: electronPockets.length,
    holePocketCount: holePockets.length,
    totalElectronVolume: totalElectronVol,
    totalHoleVolume: totalHoleVol,
    electronHoleBalance: ehBalance,
    cylindricalCharacter: avgCylindrical,
    nestingVectors: nesting.vectors,
    nestingScore: nesting.nestingScore,
    fsDimensionality: pathDimensionality,
    sigmaBandPresence: sigmaBand,
    multiBandScore: multiBand,
    mlFeatures,
  };
}

export function extractTopologyFromDFT(bandResult: DFTBandStructureResult): DFTTopologyFromBands {
  const { bandCrossings, bandInversions, vanHoveSingularities, topologicalIndicators } = bandResult;

  const hasFlatBand = bandResult.flatBandScore > 0.3;
  const hasVHS = vanHoveSingularities.length > 0 && vanHoveSingularities.some(v => Math.abs(v.energy) < 0.5);
  const hasDiracCrossing = bandResult.diracCrossingScore > 0.2;
  const hasBandInversion = bandInversions.length > 0;

  let flatBandCount = 0;
  if (bandResult.eigenvalues.length > 0 && bandResult.nBands > 0) {
    for (let b = 0; b < bandResult.nBands; b++) {
      const energies = bandResult.eigenvalues.map(kpt => kpt.energies[b]).filter(e => e !== undefined) as number[];
      if (energies.length < 3) continue;
      const range = Math.max(...energies) - Math.min(...energies);
      if (range < 0.15 && Math.abs(energies.reduce((s, e) => s + e, 0) / energies.length) < 2.0) {
        flatBandCount++;
      }
    }
  }

  const dosAtFermi = estimateDOSAtFermi(bandResult);

  let topologyScore = 0;
  if (hasBandInversion) topologyScore += 0.3 * Math.min(1.0, bandInversions.length / 3);
  if (hasDiracCrossing) topologyScore += 0.25 * bandResult.diracCrossingScore;
  if (hasFlatBand) topologyScore += 0.15 * bandResult.flatBandScore;
  if (topologicalIndicators.nodalLineIndicator > 0) topologyScore += 0.15 * topologicalIndicators.nodalLineIndicator;
  if (topologicalIndicators.parityChanges > 0) topologyScore += 0.15 * Math.min(1.0, topologicalIndicators.parityChanges / 8);
  topologyScore = Math.min(1.0, topologyScore);

  return {
    hasFlatBand,
    hasVHS,
    hasDiracCrossing,
    hasBandInversion,
    topologyScore,
    flatBandCount,
    vhsCount: vanHoveSingularities.length,
    diracCrossingCount: topologicalIndicators.diracPointCount,
    dosAtFermi,
    bandInversionCount: bandInversions.length,
    nodalLineIndicator: topologicalIndicators.nodalLineIndicator,
    parityChanges: topologicalIndicators.parityChanges,
    diracPointCount: topologicalIndicators.diracPointCount,
  };
}

function estimateDOSAtFermi(bandResult: DFTBandStructureResult): number {
  if (bandResult.eigenvalues.length === 0 || bandResult.nBands === 0) return 0;

  let statesNearFermi = 0;
  const window = 0.1;

  for (const kpt of bandResult.eigenvalues) {
    for (const e of kpt.energies) {
      if (Math.abs(e) < window) {
        statesNearFermi++;
      }
    }
  }

  const totalStates = bandResult.eigenvalues.length * bandResult.nBands;
  return totalStates > 0 ? (statesNearFermi / totalStates) * bandResult.nBands : 0;
}

export function enhanceElectronicStructure(bandResult: DFTBandStructureResult): DFTElectronicEnhancement {
  const pockets = extractFermiPockets(bandResult);
  const nesting = computeDFTNesting(pockets, bandResult);
  const dos = estimateDOSAtFermi(bandResult);

  const metallicity = bandResult.isMetallicAlongPath ? 1.0 : bandResult.bandGapAlongPath < 0.5 ? 0.5 : 0;

  let bsType = "metallic";
  if (bandResult.bandGapAlongPath > 0.5) bsType = "insulating";
  else if (bandResult.bandGapAlongPath > 0.1) bsType = "semimetallic";
  else if (bandResult.flatBandScore > 0.5) bsType = "flat-band-metal";
  else if (pockets.length > 3) bsType = "multiband-metal";

  let fsTopology = "3D-conventional";
  const avgCyl = pockets.length > 0 ? pockets.reduce((s, p) => s + p.cylindricalCharacter, 0) / pockets.length : 0;
  if (avgCyl > 0.7) fsTopology = "cylindrical-2D";
  else if (avgCyl > 0.5) fsTopology = "quasi-2D";
  else if (pockets.length > 4) fsTopology = "complex-multisheet";

  const closestVHS = bandResult.vanHoveSingularities.length > 0
    ? Math.min(...bandResult.vanHoveSingularities.map(v => Math.abs(v.energy)))
    : 1.0;
  const vhsProx = Math.max(0, 1.0 - closestVHS / 0.5);

  return {
    bandStructureType: bsType,
    fermiSurfaceTopology: fsTopology,
    densityOfStatesAtFermi: dos,
    metallicity,
    nestingScore: nesting.nestingScore,
    vanHoveProximity: vhsProx,
    bandFlatness: bandResult.flatBandScore,
    flatBandIndicator: bandResult.flatBandScore,
  };
}

let totalAnalyses = 0;
let totalWithPockets = 0;
let totalWithInversions = 0;
let totalWithVHS = 0;
let totalWithDirac = 0;
let avgPocketCount = 0;

export function recordDFTBandAnalysis(bandResult: DFTBandStructureResult): void {
  totalAnalyses++;
  const pockets = extractFermiPockets(bandResult);
  if (pockets.length > 0) totalWithPockets++;
  if (bandResult.bandInversions.length > 0) totalWithInversions++;
  if (bandResult.vanHoveSingularities.length > 0) totalWithVHS++;
  if (bandResult.diracCrossingScore > 0.2) totalWithDirac++;
  avgPocketCount = (avgPocketCount * (totalAnalyses - 1) + pockets.length) / totalAnalyses;
}

export function getDFTBandAnalysisStats(): {
  totalAnalyses: number;
  withPockets: number;
  withBandInversions: number;
  withVHS: number;
  withDiracCrossings: number;
  avgPocketCount: number;
} {
  return {
    totalAnalyses,
    withPockets: totalWithPockets,
    withBandInversions: totalWithInversions,
    withVHS: totalWithVHS,
    withDiracCrossings: totalWithDirac,
    avgPocketCount: Math.round(avgPocketCount * 100) / 100,
  };
}

export interface LindhardSusceptibility {
  qVectors: { q: number[]; chi: number }[];
  peakQ: number[] | null;
  peakChi: number;
  avgChi: number;
  nestingStrength: number;
  susceptibilityProfile: number[];
}

export function computeLindhardNesting(bandResult: DFTBandStructureResult): LindhardSusceptibility {
  if (bandResult.eigenvalues.length < 5 || bandResult.nBands === 0) {
    return { qVectors: [], peakQ: null, peakChi: 0, avgChi: 0, nestingStrength: 0, susceptibilityProfile: [] };
  }

  const fermiWindow = 0.5;
  const eta = 0.05;
  const qBinRes = 0.1;

  const nearFermiStates: { kCoords: number[]; energy: number; bandIdx: number }[] = [];
  for (const kpt of bandResult.eigenvalues) {
    for (let b = 0; b < kpt.energies.length; b++) {
      const e = kpt.energies[b];
      if (e !== undefined && Math.abs(e) < fermiWindow) {
        nearFermiStates.push({ kCoords: [...kpt.kCoords], energy: e, bandIdx: b });
      }
    }
  }

  if (nearFermiStates.length < 4) {
    return { qVectors: [], peakQ: null, peakChi: 0, avgChi: 0, nestingStrength: 0, susceptibilityProfile: [] };
  }

  const maxStates = 200;
  const sampledStates = nearFermiStates.length > maxStates
    ? nearFermiStates.filter((_, i) => i % Math.ceil(nearFermiStates.length / maxStates) === 0)
    : nearFermiStates;

  const chiMap = new Map<string, { q: number[]; chi: number; count: number }>();

  for (let i = 0; i < sampledStates.length; i++) {
    for (let j = i + 1; j < sampledStates.length; j++) {
      const si = sampledStates[i];
      const sj = sampledStates[j];
      const dE = sj.energy - si.energy;

      const fi = si.energy < 0 ? 1.0 : 0.0;
      const fj = sj.energy < 0 ? 1.0 : 0.0;
      const df = fi - fj;

      if (Math.abs(df) < 0.01) continue;

      const chiContrib = df / (dE + eta * Math.sign(dE));

      const q = [
        sj.kCoords[0] - si.kCoords[0],
        sj.kCoords[1] - si.kCoords[1],
        sj.kCoords[2] - si.kCoords[2],
      ];

      const binKey = q.map(v => (Math.round(v / qBinRes) * qBinRes).toFixed(2)).join(",");

      if (chiMap.has(binKey)) {
        const entry = chiMap.get(binKey)!;
        entry.chi += Math.abs(chiContrib);
        entry.count++;
      } else {
        chiMap.set(binKey, { q: q.map(v => Math.round(v / qBinRes) * qBinRes), chi: Math.abs(chiContrib), count: 1 });
      }
    }
  }

  const allQ = Array.from(chiMap.values()).sort((a, b) => b.chi - a.chi);
  const topQ = allQ.slice(0, 10);

  const peakEntry = topQ[0];
  const peakChi = peakEntry?.chi ?? 0;
  const avgChi = allQ.length > 0 ? allQ.reduce((s, e) => s + e.chi, 0) / allQ.length : 0;

  const nestingStrength = peakChi > 0
    ? Math.min(1.0, (peakChi / (avgChi + 1e-6)) * 0.2 * Math.min(1, allQ.length / 20))
    : 0;

  const profile = topQ.map(e => e.chi);

  return {
    qVectors: topQ.map(e => ({ q: e.q, chi: e.chi })),
    peakQ: peakEntry?.q ?? null,
    peakChi,
    avgChi,
    nestingStrength,
    susceptibilityProfile: profile,
  };
}

export interface BandFeatureScore {
  tcComponent: number;
  stabilityComponent: number;
  nestingComponent: number;
  topologyComponent: number;
  compositeScore: number;
  weights: { tc: number; stability: number; nesting: number; topology: number };
}

export function computeBandFeatureScore(
  predictedTc: number,
  stabilityScore: number,
  nestingScore: number,
  topologyScore: number
): BandFeatureScore {
  const W_TC = 0.4;
  const W_STABILITY = 0.2;
  const W_NESTING = 0.2;
  const W_TOPOLOGY = 0.2;

  const tcNorm = Math.min(1.0, Math.max(0, predictedTc / 300));
  const stabNorm = Math.min(1.0, Math.max(0, stabilityScore));
  const nestNorm = Math.min(1.0, Math.max(0, nestingScore));
  const topoNorm = Math.min(1.0, Math.max(0, topologyScore));

  const composite = W_TC * tcNorm + W_STABILITY * stabNorm + W_NESTING * nestNorm + W_TOPOLOGY * topoNorm;

  return {
    tcComponent: Math.round(tcNorm * 1000) / 1000,
    stabilityComponent: Math.round(stabNorm * 1000) / 1000,
    nestingComponent: Math.round(nestNorm * 1000) / 1000,
    topologyComponent: Math.round(topoNorm * 1000) / 1000,
    compositeScore: Math.round(composite * 1000) / 1000,
    weights: { tc: W_TC, stability: W_STABILITY, nesting: W_NESTING, topology: W_TOPOLOGY },
  };
}

export interface AutomatedTopologyResult {
  crossingDispersion: CrossingDispersion[];
  socGaps: { kIndex: number; bandPair: [number, number]; gapMeV: number; atFermi: boolean }[];
  fermiIsosurface: FermiIsosurface;
  topologyFromBands: DFTTopologyFromBands;
  fermiSurface: FermiSurfaceResult;
  lindhardNesting: LindhardSusceptibility;
  classification: DFTTopologicalClassification;
  bandFeatureScore: BandFeatureScore;
}

export function runAutomatedTopologyPipeline(
  bandResult: DFTBandStructureResult,
  socStrength: number,
  predictedTc: number,
  stabilityScore: number
): AutomatedTopologyResult {
  const crossingDispersion = classifyCrossingDispersion(bandResult);
  const socGaps = detectSOCGaps(bandResult);
  const fermiIsosurface = computeFermiIsosurface(bandResult);
  const topologyFromBands = extractTopologyFromDFT(bandResult);
  const fermiSurface = buildFermiSurfaceFromDFT(bandResult);
  const lindhardNesting = computeLindhardNesting(bandResult);
  const classification = classifyDFTTopology(bandResult, socStrength);

  const effectiveNesting = Math.max(
    fermiSurface.nestingScore,
    lindhardNesting.nestingStrength
  );
  const effectiveTopo = Math.max(
    topologyFromBands.topologyScore,
    classification.confidence
  );

  const bandFeatureScore = computeBandFeatureScore(
    predictedTc,
    stabilityScore,
    effectiveNesting,
    effectiveTopo
  );

  return {
    crossingDispersion,
    socGaps,
    fermiIsosurface,
    topologyFromBands,
    fermiSurface,
    lindhardNesting,
    classification,
    bandFeatureScore,
  };
}
