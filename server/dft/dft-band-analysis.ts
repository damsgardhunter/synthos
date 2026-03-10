import { isPathBreak, type DFTBandStructureResult, type BandEigenvalue, type BandCrossing, type BandInversion, type VanHoveSingularity, type EffectiveMass } from "./band-structure-calculator";
import type { FermiSurfaceResult, FermiPocket, NestingVector, FermiSurfaceMLFeatures } from "../physics/fermi-surface-engine";

export interface DFTFermiPocket {
  index: number;
  type: "electron" | "hole";
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

  for (const crossing of bandResult.bandCrossings) {
    const bIdx = crossing.bandIndex;
    const kFrac = crossing.kFraction;
    const kIdx = Math.round(kFrac * (bandResult.eigenvalues.length - 1));

    const windowSize = 3;
    const lo = Math.max(0, kIdx - windowSize);
    const hi = Math.min(bandResult.eigenvalues.length - 1, kIdx + windowSize);

    const localE: { dk: number; e: number }[] = [];
    for (let i = lo; i <= hi; i++) {
      const e = bandResult.eigenvalues[i]?.energies[bIdx];
      if (e === undefined) continue;
      const dk = bandResult.eigenvalues[i].kDistance - bandResult.eigenvalues[kIdx].kDistance;
      localE.push({ dk, e });
    }

    if (localE.length < 3) continue;

    let velocity = 0;
    let curvature = 0;

    if (localE.length >= 3) {
      const midIdx = Math.floor(localE.length / 2);
      const dE_left = localE[midIdx].e - localE[midIdx - 1].e;
      const dk_left = localE[midIdx].dk - localE[midIdx - 1].dk;
      const dE_right = localE[midIdx + 1 < localE.length ? midIdx + 1 : midIdx].e - localE[midIdx].e;
      const dk_right = (localE[midIdx + 1 < localE.length ? midIdx + 1 : midIdx].dk - localE[midIdx].dk);

      if (Math.abs(dk_left) > 1e-6 && Math.abs(dk_right) > 1e-6) {
        const v_left = dE_left / dk_left;
        const v_right = dE_right / dk_right;
        velocity = (Math.abs(v_left) + Math.abs(v_right)) / 2;
        curvature = Math.abs(v_right - v_left) / (Math.abs(dk_left + dk_right) / 2 + 1e-6);
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

    crossings.push({
      bandIndex: bIdx,
      kIndex: kIdx,
      energy: crossing.energy,
      type,
      curvature,
      velocity,
      socGap,
    });
  }

  return crossings;
}

export function detectSOCGaps(bandResult: DFTBandStructureResult): { kIndex: number; bandPair: [number, number]; gapMeV: number; atFermi: boolean }[] {
  const gaps: { kIndex: number; bandPair: [number, number]; gapMeV: number; atFermi: boolean }[] = [];

  for (let ki = 0; ki < bandResult.eigenvalues.length; ki++) {
    const kpt = bandResult.eigenvalues[ki];
    for (let b = 0; b < kpt.energies.length - 1; b++) {
      const e1 = kpt.energies[b];
      const e2 = kpt.energies[b + 1];
      if (e1 === undefined || e2 === undefined) continue;
      const gap = e2 - e1;
      if (gap > 0.001 && gap < 0.15) {
        const midE = (e1 + e2) / 2;
        const atFermi = Math.abs(midE) < 0.3;
        gaps.push({
          kIndex: ki,
          bandPair: [b, b + 1],
          gapMeV: gap * 1000,
          atFermi,
        });
      }
    }
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
      if (isPathBreak(bandResult.eigenvalues, ki + 1)) continue;
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

export function classifyDFTTopology(bandResult: DFTBandStructureResult, socStrength: number): DFTTopologicalClassification {
  const chain: string[] = ["structure", "band-structure", "detect-crossings"];
  const evidence: string[] = [];

  const crossings = classifyCrossingDispersion(bandResult);
  const socGaps = detectSOCGaps(bandResult);
  const topo = extractTopologyFromDFT(bandResult);
  const isosurface = computeFermiIsosurface(bandResult);

  const linearCrossings = crossings.filter(c => c.type === "linear");
  const quadraticCrossings = crossings.filter(c => c.type === "quadratic");

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

  let topologicalClass = "trivial";
  let confidence = 0;

  if (topo.hasBandInversion && maxSOCGap > 10 && socStrength > 0.2 && !bandResult.isMetallicAlongPath) {
    topologicalClass = "strong-topological-insulator";
    confidence = Math.min(0.95, 0.5 + topo.bandInversionCount * 0.1 + maxSOCGap / 200);
    evidence.push(`Band inversion with SOC gap ${maxSOCGap.toFixed(0)} meV`);
    evidence.push(`${topo.bandInversionCount} band inversion(s) detected`);
    if (topo.parityChanges > 3) {
      evidence.push(`${topo.parityChanges} parity changes (Z2 indicator)`);
      confidence = Math.min(0.95, confidence + 0.1);
    }
    chain.push("band-inversion+SOC-gap->TI");
  } else if (topo.hasBandInversion && maxSOCGap > 10 && socStrength > 0.15 && bandResult.isMetallicAlongPath) {
    topologicalClass = "topological-metal";
    confidence = Math.min(0.9, 0.4 + topo.bandInversionCount * 0.1 + maxSOCGap / 300);
    evidence.push(`Metallic with band inversion and SOC gap ${maxSOCGap.toFixed(0)} meV`);
    chain.push("band-inversion+SOC-gap+metallic->TM");
  } else if (linearCrossings.length > 0 && linearCrossings.some(c => Math.abs(c.energy) < 0.2)) {
    const nearFermiLinear = linearCrossings.filter(c => Math.abs(c.energy) < 0.2);

    if (weylPointCount > 0 && socStrength > 0.1) {
      topologicalClass = "Weyl-semimetal";
      confidence = Math.min(0.9, 0.4 + weylPointCount * 0.15 + socStrength * 0.3);
      evidence.push(`${weylPointCount} Weyl-like crossing(s) with linear dispersion`);
      if (socStrength > 0.3) evidence.push("Strong SOC lifts degeneracy");
      chain.push("linear-crossing+SOC->Weyl");
    } else {
      topologicalClass = "Dirac-semimetal";
      confidence = Math.min(0.85, 0.35 + nearFermiLinear.length * 0.12);
      evidence.push(`${nearFermiLinear.length} Dirac crossing(s) with linear dispersion near Fermi level`);
      const avgVel = nearFermiLinear.reduce((s, c) => s + c.velocity, 0) / nearFermiLinear.length;
      evidence.push(`Average Fermi velocity: ${avgVel.toFixed(2)} eV*A`);
      chain.push("linear-crossing->Dirac");
    }
  } else if (nodalLineCandidates > 0.5) {
    topologicalClass = "nodal-line-semimetal";
    confidence = Math.min(0.85, 0.3 + nodalLineCandidates * 0.3);
    evidence.push(`Nodal line indicator: ${nodalLineCandidates.toFixed(2)}`);
    if (crossings.length > 4) evidence.push(`${crossings.length} crossings along high-symmetry path`);
    chain.push("nodal-line-indicator->NLS");
  } else if (topo.hasFlatBand && topo.flatBandCount > 0) {
    const hasFBNearFermi = bandResult.flatBandScore > 0.5;
    if (hasFBNearFermi && topo.hasBandInversion) {
      topologicalClass = "topological-flat-band";
      confidence = Math.min(0.8, 0.3 + topo.flatBandCount * 0.1 + bandResult.flatBandScore * 0.3);
      evidence.push(`${topo.flatBandCount} flat band(s) with band inversion`);
      chain.push("flat-band+inversion->TFB");
    } else if (hasFBNearFermi) {
      topologicalClass = "flat-band-system";
      confidence = Math.min(0.75, 0.25 + bandResult.flatBandScore * 0.3);
      evidence.push(`${topo.flatBandCount} flat band(s) near Fermi level (score: ${bandResult.flatBandScore.toFixed(2)})`);
      chain.push("flat-band->FBS");
    }
  } else if (topo.hasBandInversion && socStrength > 0.1) {
    topologicalClass = "weak-topological";
    confidence = Math.min(0.7, 0.2 + topo.bandInversionCount * 0.1);
    evidence.push(`${topo.bandInversionCount} band inversion(s) with moderate SOC`);
    chain.push("band-inversion+weak-SOC->WTI");
  } else if (socStrength > 0.3 && topo.topologyScore > 0.2) {
    topologicalClass = "SOC-enhanced";
    confidence = Math.min(0.6, 0.15 + socStrength * 0.3);
    evidence.push(`Strong SOC (${socStrength.toFixed(2)}) with topology score ${topo.topologyScore.toFixed(2)}`);
    chain.push("strong-SOC->SOC-enhanced");
  }

  if (topologicalClass === "trivial" && topo.topologyScore > 0.15) {
    confidence = 0.1;
    evidence.push(`Low topology score (${topo.topologyScore.toFixed(2)}), no definitive classification`);
  }

  if (isosurface.sheetCount > 3 && topologicalClass !== "trivial") {
    evidence.push(`Fermi isosurface: ${isosurface.sheetCount} sheets, ${isosurface.totalPoints} crossing points`);
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

  let winding = 0;
  const sorted = [...crossings].sort((a, b) => a.kIndex - b.kIndex);

  for (let i = 1; i < sorted.length; i++) {
    const slopeChange = sorted[i].velocity - sorted[i - 1].velocity;
    if (slopeChange !== 0) {
      winding += Math.sign(slopeChange);
    }
  }

  return Math.abs(winding) / Math.max(1, crossings.length);
}

export function extractFermiPockets(bandResult: DFTBandStructureResult): DFTFermiPocket[] {
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

    const type: "electron" | "hole" = aboveCount > belowCount ? "electron" : "hole";

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

    let orbChar = { s: 0.15, p: 0.25, d: 0.6 };

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
    } else {
      const dFrac = bandEnergies.filter(e => Math.abs(e.energy) < 2.0).length / bandEnergies.length;
      orbChar = { s: 0.15, p: 0.25, d: Math.min(0.6, dFrac) };
      const orbSum = orbChar.s + orbChar.p + orbChar.d;
      orbChar.s /= orbSum;
      orbChar.p /= orbSum;
      orbChar.d /= orbSum;
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

      const sampleI = ki.length > 20 ? ki.slice(0, 20) : ki;
      const sampleJ = kj.length > 20 ? kj.slice(0, 20) : kj;

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

  const electronPockets = pockets.filter(p => p.type === "electron");
  const holePockets = pockets.filter(p => p.type === "hole");

  const totalElectronVol = electronPockets.reduce((s, p) => s + p.volume, 0);
  const totalHoleVol = holePockets.reduce((s, p) => s + p.volume, 0);
  const totalVol = totalElectronVol + totalHoleVol;
  const ehBalance = totalVol > 0 ? 1.0 - Math.abs(totalElectronVol - totalHoleVol) / totalVol : 0;

  const avgCylindrical = pockets.length > 0
    ? pockets.reduce((s, p) => s + p.cylindricalCharacter, 0) / pockets.length : 0;

  let fsDim = 3.0;
  if (avgCylindrical > 0.7) fsDim = 2.0;
  else if (avgCylindrical > 0.5) fsDim = 2.5;

  let sigmaBand = 0;
  for (const pocket of pockets) {
    if (pocket.orbitalCharacter.s > 0.3 && pocket.avgVelocity > 3.0) {
      sigmaBand = Math.max(sigmaBand, pocket.orbitalCharacter.s + pocket.avgVelocity / 10);
    }
  }
  sigmaBand = Math.min(1.0, sigmaBand);

  const multiBand = Math.min(1.0, pockets.length / 6);

  const fermiPockets: FermiPocket[] = pockets.map(p => ({
    index: p.index,
    type: p.type,
    volume: p.volume,
    cylindricalCharacter: p.cylindricalCharacter,
    orbitalCharacter: p.orbitalCharacter,
    bandIndex: p.bandIndex,
    avgVelocity: p.avgVelocity,
  }));

  const mlFeatures: FermiSurfaceMLFeatures = {
    fermiPocketCount: pockets.length,
    electronHoleBalance: ehBalance,
    fsDimensionality: fsDim,
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
    fsDimensionality: fsDim,
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
  if (topologicalIndicators.parityChanges > 5) topologyScore += 0.15 * Math.min(1.0, topologicalIndicators.parityChanges / 20);
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
