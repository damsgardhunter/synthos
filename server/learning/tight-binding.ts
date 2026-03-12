import {
  ELEMENTAL_DATA,
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "./elemental-data";
import { predictStructure } from "../crystal/structure-predictor-ml";

export interface SlaterKosterParams {
  ssSigma: number;
  spSigma: number;
  ppSigma: number;
  ppPi: number;
  sdSigma: number;
  pdSigma: number;
  pdPi: number;
  ddSigma: number;
  ddPi: number;
  ddDelta: number;
}

export interface OnsiteEnergies {
  es: number;
  ep: number;
  ed: number;
}

export interface TBBandPoint {
  k: number[];
  kLabel?: string;
  eigenvalues: number[];
  orbitalCharacters: { s: number; p: number; d: number }[];
}

export interface TBBandStructure {
  kPoints: number[][];
  kLabels: { index: number; label: string }[];
  bands: number[][];
  orbitalCharacters: { s: number; p: number; d: number }[][];
  fermiEnergy: number;
  nOrbitals: number;
  formula: string;
  tbConfidence: number;
  latticeType: string;
}

export interface WannierProjection {
  effectiveHoppings: { orbital: string; value: number }[];
  orbitalBandCharacter: { orbital: string; bandwidth: number; center: number }[];
  wannierSpread: number;
  maxLocalization: number;
}

export interface TopologyFeatures {
  flatBands: { bandIndex: number; bandwidth: number; center: number }[];
  vanHoveSingularities: { bandIndex: number; kIndex: number; energy: number; type: string }[];
  diracCrossings: { kIndex: number; energy: number; velocity: number }[];
  bandInversions: { kIndex: number; bands: [number, number]; energyGap: number }[];
  hasFlatBand: boolean;
  hasVHS: boolean;
  hasDiracCrossing: boolean;
  hasBandInversion: boolean;
  topologyScore: number;
}

export interface TBDOS {
  energies: number[];
  dos: number[];
  dosAtFermi: number;
  totalStates: number;
  uncertaintyMultiplier: number;
}

function parseComposition(formula: string): { elements: string[]; counts: Record<string, number> } {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return { elements: Object.keys(counts), counts };
}

function getOnsiteEnergies(el: string): OnsiteEnergies {
  const data = getElementData(el);
  if (!data) return { es: -8.0, ep: -4.0, ed: 0.0 };

  const ie = data.firstIonizationEnergy;
  const ea = data.electronAffinity ?? 0;
  const en = data.paulingElectronegativity ?? 2.0;

  const es = -(ie * 0.5 + ea * 0.5) * 0.8;
  const ep = es + 3.0 + en * 0.5;
  const ed = isTransitionMetal(el) || isRareEarth(el) || isActinide(el)
    ? es + 1.5 + (data.valenceElectrons - 2) * 0.3
    : es + 8.0;

  return { es, ep, ed };
}

function getSlaterKosterParams(el1: string, el2: string, distance: number): SlaterKosterParams {
  const d1 = getElementData(el1);
  const d2 = getElementData(el2);

  const r1 = d1 ? d1.atomicRadius / 100 : 1.5;
  const r2 = d2 ? d2.atomicRadius / 100 : 1.5;
  const r0 = (r1 + r2) * 0.9;

  const distRatio = distance / r0 - 1.0;
  const ssDecay = Math.exp(-1.2 * distRatio);
  const spDecay = Math.exp(-1.4 * distRatio);
  const ppDecay = Math.exp(-1.6 * distRatio);
  const sdDecay = Math.exp(-1.6 * distRatio);
  const pdDecay = Math.exp(-1.8 * distRatio);
  const ddDecay = Math.exp(-2.0 * distRatio);

  const ve1 = d1 ? d1.valenceElectrons : 4;
  const ve2 = d2 ? d2.valenceElectrons : 4;
  const veAvg = (ve1 + ve2) / 2;

  const hasD1 = isTransitionMetal(el1) || isRareEarth(el1) || isActinide(el1);
  const hasD2 = isTransitionMetal(el2) || isRareEarth(el2) || isActinide(el2);
  const eitherHasD = hasD1 || hasD2;

  return {
    ssSigma: -1.5 * ssDecay,
    spSigma: 1.8 * spDecay,
    ppSigma: 2.5 * ppDecay,
    ppPi: -0.8 * ppDecay,
    sdSigma: eitherHasD ? -1.2 * sdDecay : 0,
    pdSigma: eitherHasD ? -1.5 * pdDecay : 0,
    pdPi: eitherHasD ? 0.7 * pdDecay : 0,
    ddSigma: (hasD1 && hasD2) ? -2.0 * ddDecay * (veAvg > 4 ? 1.2 : 0.8) : 0,
    ddPi: (hasD1 && hasD2) ? 1.2 * ddDecay : 0,
    ddDelta: (hasD1 && hasD2) ? -0.4 * ddDecay : 0,
  };
}

function getHighSymmetryPath(latticeType: string): { points: number[][]; labels: string[] } {
  switch (latticeType) {
    case "hexagonal":
      return {
        points: [
          [0, 0, 0],
          [1/3, 1/3, 0],
          [0.5, 0, 0],
          [0, 0, 0],
          [0, 0, 0.5],
        ],
        labels: ["Γ", "K", "M", "Γ", "A"],
      };
    case "bcc":
      return {
        points: [
          [0, 0, 0],
          [0.5, -0.5, 0.5],
          [0.25, 0.25, 0.25],
          [0, 0, 0],
          [0, 0.5, 0],
        ],
        labels: ["Γ", "H", "P", "Γ", "N"],
      };
    case "fcc":
      return {
        points: [
          [0, 0, 0],
          [0.5, 0.5, 0],
          [0.5, 0.25, 0.75],
          [0.375, 0.375, 0.75],
          [0, 0, 0],
          [0.5, 0.5, 0.5],
        ],
        labels: ["Γ", "X", "W", "K", "Γ", "L"],
      };
    case "tetragonal":
      return {
        points: [
          [0, 0, 0],
          [0.5, 0, 0],
          [0.5, 0.5, 0],
          [0, 0, 0],
          [0, 0, 0.5],
          [0.5, 0, 0.5],
          [0.5, 0.5, 0.5],
        ],
        labels: ["Γ", "X", "M", "Γ", "Z", "R", "A"],
      };
    default:
      return {
        points: [
          [0, 0, 0],
          [0.5, 0, 0],
          [0.5, 0.5, 0],
          [0, 0, 0],
          [0.5, 0.5, 0.5],
        ],
        labels: ["Γ", "X", "M", "Γ", "R"],
      };
  }
}

const CRYSTAL_SYSTEM_TO_LATTICE: Record<string, string> = {
  cubic: "cubic", tetragonal: "tetragonal", hexagonal: "hexagonal",
  trigonal: "hexagonal", orthorhombic: "cubic", monoclinic: "cubic", triclinic: "cubic",
};

function guessLatticeType(elements: string[], formula?: string): string {
  if (formula) {
    try {
      const pred = predictStructure(formula);
      if (pred && pred.confidence > 0.3 && pred.crystalSystem?.predicted) {
        const mapped = CRYSTAL_SYSTEM_TO_LATTICE[pred.crystalSystem.predicted];
        if (mapped) return mapped;
      }
    } catch {}
  }

  if (elements.length === 1) {
    const el = elements[0];
    if (["Fe", "Cr", "V", "Nb", "Mo", "W", "Ta", "Na", "K", "Li", "Ba"].includes(el)) return "bcc";
    if (["Cu", "Ag", "Au", "Al", "Ni", "Pd", "Pt", "Pb", "Ca", "Sr"].includes(el)) return "fcc";
    if (["Ti", "Zr", "Hf", "Co", "Zn", "Mg", "Be", "Y", "Sc"].includes(el)) return "hexagonal";
    if (["Sn", "In"].includes(el)) return "tetragonal";
  }

  const hasCu = elements.includes("Cu");
  const hasNi = elements.includes("Ni");
  const hasO = elements.includes("O");
  const hasFe = elements.includes("Fe");
  const hasAsPnictide = elements.includes("As") || elements.includes("Se") || elements.includes("P");

  if (hasCu && hasO && elements.length >= 3) return "tetragonal";
  if (hasNi && hasO && elements.length >= 3) return "tetragonal";
  if (hasFe && hasAsPnictide) return "tetragonal";

  if (elements.includes("B") && elements.some(e => isTransitionMetal(e) || isRareEarth(e))) return "hexagonal";
  if (elements.length >= 3 && hasO &&
    elements.some(e => ["La", "Sr", "Ba", "Ca", "Y", "Nd", "Pr", "Sm", "Gd"].includes(e))) return "tetragonal";
  if (elements.length >= 3 && hasO) return "cubic";
  return "cubic";
}

function interpolateKPoints(path: { points: number[][]; labels: string[] }, nPerSegment: number): { kPoints: number[][]; kLabels: { index: number; label: string }[] } {
  const kPoints: number[][] = [];
  const kLabels: { index: number; label: string }[] = [];

  for (let seg = 0; seg < path.points.length - 1; seg++) {
    const start = path.points[seg];
    const end = path.points[seg + 1];

    if (seg === 0) {
      kLabels.push({ index: 0, label: path.labels[seg] });
    }

    for (let i = 0; i < nPerSegment; i++) {
      if (i === 0) {
        kPoints.push([start[0], start[1], start[2]]);
      } else {
        const t = i / nPerSegment;
        kPoints.push([
          start[0] + (end[0] - start[0]) * t,
          start[1] + (end[1] - start[1]) * t,
          start[2] + (end[2] - start[2]) * t,
        ]);
      }
    }

    kLabels.push({ index: kPoints.length, label: path.labels[seg + 1] });
  }

  const last = path.points[path.points.length - 1];
  kPoints.push([last[0], last[1], last[2]]);

  return { kPoints, kLabels };
}

function getNeighborVectors(latticeType: string): number[][] {
  switch (latticeType) {
    case "bcc":
      return [
        [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
        [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5],
      ];
    case "fcc":
      return [
        [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0],
        [0.5, 0, 0.5], [0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, 0, -0.5],
        [0, 0.5, 0.5], [0, 0.5, -0.5], [0, -0.5, 0.5], [0, -0.5, -0.5],
      ];
    case "hexagonal":
      return [
        [1, 0, 0], [-1, 0, 0],
        [0.5, Math.sqrt(3) / 2, 0], [-0.5, -Math.sqrt(3) / 2, 0],
        [-0.5, Math.sqrt(3) / 2, 0], [0.5, -Math.sqrt(3) / 2, 0],
        [0, 0, 0.5], [0, 0, -0.5],
        [0.5, Math.sqrt(3) / 6, 0.5], [-0.5, -Math.sqrt(3) / 6, -0.5],
        [-0.5, Math.sqrt(3) / 6, 0.5], [0.5, -Math.sqrt(3) / 6, -0.5],
      ];
    default:
      return [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
      ];
  }
}

function buildHamiltonianAtK(
  k: number[],
  elements: string[],
  counts: Record<string, number>,
  latticeConstant: number,
  latticeType: string,
): { eigenvalues: number[]; orbChars: { s: number; p: number; d: number }[] } {
  const atomList: { el: string; orbitalStart: number }[] = [];
  let nOrbitals = 0;

  for (const el of elements) {
    const count = Math.round(counts[el] || 1);
    const hasDOrbs = isTransitionMetal(el) || isRareEarth(el) || isActinide(el);
    const orbsPerAtom = hasDOrbs ? 9 : 4;

    for (let i = 0; i < count; i++) {
      atomList.push({ el, orbitalStart: nOrbitals });
      nOrbitals += orbsPerAtom;
    }
  }

  const MAX_TB_ORBITALS = 128;
  if (nOrbitals > MAX_TB_ORBITALS) {
    nOrbitals = Math.min(nOrbitals, MAX_TB_ORBITALS);
    atomList.length = 0;
    let currentOrb = 0;
    for (const el of elements) {
      const count = Math.round(counts[el] || 1);
      const hasDOrbs = isTransitionMetal(el) || isRareEarth(el) || isActinide(el);
      const orbsPerAtom = hasDOrbs ? 9 : 4;
      for (let i = 0; i < count; i++) {
        if (currentOrb + orbsPerAtom <= MAX_TB_ORBITALS) {
          atomList.push({ el, orbitalStart: currentOrb });
          currentOrb += orbsPerAtom;
        }
      }
    }
    nOrbitals = currentOrb;
  }

  const H: number[][] = [];
  for (let i = 0; i < nOrbitals; i++) {
    H[i] = new Array(nOrbitals).fill(0);
  }

  for (const atom of atomList) {
    const onsite = getOnsiteEnergies(atom.el);
    const hasDOrbs = isTransitionMetal(atom.el) || isRareEarth(atom.el) || isActinide(atom.el);
    const o = atom.orbitalStart;

    H[o][o] = onsite.es;
    if (o + 1 < nOrbitals) H[o + 1][o + 1] = onsite.ep;
    if (o + 2 < nOrbitals) H[o + 2][o + 2] = onsite.ep;
    if (o + 3 < nOrbitals) H[o + 3][o + 3] = onsite.ep;

    if (hasDOrbs) {
      for (let d = 0; d < 5 && o + 4 + d < nOrbitals; d++) {
        H[o + 4 + d][o + 4 + d] = onsite.ed;
      }
    }
  }

  const kDotR = (dx: number, dy: number, dz: number) => {
    return 2 * Math.PI * (k[0] * dx + k[1] * dy + k[2] * dz);
  };

  for (let i = 0; i < atomList.length; i++) {
    for (let j = i + 1; j < atomList.length; j++) {
      const a1 = atomList[i];
      const a2 = atomList[j];

      const d1 = getElementData(a1.el);
      const d2 = getElementData(a2.el);
      const r1 = d1 ? d1.atomicRadius / 100 : 1.5;
      const r2 = d2 ? d2.atomicRadius / 100 : 1.5;
      const bondDist = (r1 + r2) * 0.85;

      const sk = getSlaterKosterParams(a1.el, a2.el, bondDist);

      const neighbors = getNeighborVectors(latticeType);

      for (const [dx, dy, dz] of neighbors) {
        const phase = Math.cos(kDotR(dx, dy, dz));
        const rMag = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-10;
        const l = dx / rMag;
        const m = dy / rMag;
        const n = dz / rMag;
        const dc = [l, m, n];

        const hasDI = isTransitionMetal(a1.el) || isRareEarth(a1.el) || isActinide(a1.el);
        const hasDJ = isTransitionMetal(a2.el) || isRareEarth(a2.el) || isActinide(a2.el);
        const oi = a1.orbitalStart;
        const oj = a2.orbitalStart;
        const nf = 1 / neighbors.length;

        if (oi < nOrbitals && oj < nOrbitals) {
          const v = sk.ssSigma * phase * nf;
          H[oi][oj] += v;
          H[oj][oi] += v;
        }

        for (let p = 0; p < 3; p++) {
          if (oi < nOrbitals && oj + 1 + p < nOrbitals) {
            const v = sk.spSigma * dc[p] * phase * nf;
            H[oi][oj + 1 + p] += v;
            H[oj + 1 + p][oi] += v;
          }
          if (oj < nOrbitals && oi + 1 + p < nOrbitals) {
            const v = -sk.spSigma * dc[p] * phase * nf;
            H[oj][oi + 1 + p] += v;
            H[oi + 1 + p][oj] += v;
          }
        }

        for (let p1 = 0; p1 < 3; p1++) {
          for (let p2 = 0; p2 < 3; p2++) {
            if (oi + 1 + p1 < nOrbitals && oj + 1 + p2 < nOrbitals) {
              const sigmaW = dc[p1] * dc[p2];
              const piW = (p1 === p2 ? 1 : 0) - dc[p1] * dc[p2];
              const v = (sk.ppSigma * sigmaW + sk.ppPi * piW) * phase * nf;
              H[oi + 1 + p1][oj + 1 + p2] += v;
              H[oj + 1 + p2][oi + 1 + p1] += v;
            }
          }
        }

        const l2 = l * l, m2 = m * m, n2 = n * n;

        if (hasDI && hasDJ) {
          for (let d1 = 0; d1 < 5; d1++) {
            for (let d2 = 0; d2 < 5; d2++) {
              if (oi + 4 + d1 < nOrbitals && oj + 4 + d2 < nOrbitals) {
                let v = 0;
                if (d1 === d2) {
                  const diag = d1 < 3 ? [l2, m2, n2][d1] : (d1 === 3 ? l * m : (l2 + m2) * 0.5);
                  v = (sk.ddSigma * diag * diag + sk.ddPi * diag * (1 - diag) + sk.ddDelta * (1 - diag) * (1 - diag)) * phase * nf;
                } else {
                  v = (sk.ddPi * 0.3 + sk.ddDelta * 0.1) * phase * nf * 0.5;
                }
                H[oi + 4 + d1][oj + 4 + d2] += v;
                H[oj + 4 + d2][oi + 4 + d1] += v;
              }
            }
          }
        }

        if (hasDI) {
          for (let d = 0; d < 5; d++) {
            if (oi + 4 + d < nOrbitals && oj < nOrbitals) {
              const sdWeight = d < 3 ? dc[d] * dc[d] : (d === 3 ? l * m : Math.sqrt(3) * 0.5 * (l2 - m * m));
              const v = sk.sdSigma * Math.sqrt(Math.abs(sdWeight)) * Math.sign(sdWeight || 1) * phase * nf;
              H[oi + 4 + d][oj] += v;
              H[oj][oi + 4 + d] += v;
            }
          }
        }
        if (hasDJ) {
          for (let d = 0; d < 5; d++) {
            if (oj + 4 + d < nOrbitals && oi < nOrbitals) {
              const sdWeight = d < 3 ? dc[d] * dc[d] : (d === 3 ? l * m : Math.sqrt(3) * 0.5 * (l * l - m * m));
              const v = sk.sdSigma * Math.sqrt(Math.abs(sdWeight)) * Math.sign(sdWeight || 1) * phase * nf;
              H[oj + 4 + d][oi] += v;
              H[oi][oj + 4 + d] += v;
            }
          }
        }
      }
    }
  }

  const { eigenvalues, eigenvectors } = solveEigenvaluesSymmetric(H, nOrbitals);

  const orbChars: { s: number; p: number; d: number }[] = [];
  const hasEvecs = eigenvectors.length === nOrbitals;

  let sCount = 0, pCount = 0, dCount = 0;
  if (!hasEvecs) {
    for (const atom of atomList) {
      const hasDOrbs = isTransitionMetal(atom.el) || isRareEarth(atom.el) || isActinide(atom.el);
      sCount += 1;
      pCount += 3;
      if (hasDOrbs) dCount += 5;
    }
  }

  for (let band = 0; band < nOrbitals; band++) {
    if (!hasEvecs) {
      const total = sCount + pCount + dCount || 1;
      orbChars.push({ s: sCount / total, p: pCount / total, d: dCount / total });
      continue;
    }
    let sWeight = 0, pWeight = 0, dWeight = 0;
    const psi = eigenvectors[band];
    for (const atom of atomList) {
      const hasDOrbs = isTransitionMetal(atom.el) || isRareEarth(atom.el) || isActinide(atom.el);
      const o = atom.orbitalStart;
      if (o < nOrbitals) sWeight += psi[o] * psi[o];
      for (let p = 0; p < 3; p++) {
        if (o + 1 + p < nOrbitals) pWeight += psi[o + 1 + p] * psi[o + 1 + p];
      }
      if (hasDOrbs) {
        for (let d = 0; d < 5; d++) {
          if (o + 4 + d < nOrbitals) dWeight += psi[o + 4 + d] * psi[o + 4 + d];
        }
      }
    }
    const total = sWeight + pWeight + dWeight || 1;
    orbChars.push({
      s: sWeight / total,
      p: pWeight / total,
      d: dWeight / total,
    });
  }

  return { eigenvalues, orbChars };
}

function householderTridiagonalize(H: number[][], n: number): {
  diag: number[]; offDiag: number[]; Q: number[][];
} {
  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    A[i] = new Array(n);
    for (let j = 0; j < n; j++) A[i][j] = H[i][j];
  }

  const Q: number[][] = [];
  for (let i = 0; i < n; i++) {
    Q[i] = new Array(n).fill(0);
    Q[i][i] = 1;
  }

  for (let k = 0; k < n - 2; k++) {
    const x: number[] = new Array(n - k - 1);
    let sigma = 0;
    for (let i = k + 1; i < n; i++) {
      x[i - k - 1] = A[i][k];
      sigma += A[i][k] * A[i][k];
    }
    if (sigma < 1e-30) continue;

    const alpha = -Math.sign(x[0] || 1) * Math.sqrt(sigma);
    const u0 = x[0] - alpha;
    const uNormSq = sigma - x[0] * x[0] + u0 * u0;
    if (Math.abs(uNormSq) < 1e-30) continue;

    const v: number[] = new Array(n - k - 1);
    v[0] = u0;
    for (let i = 1; i < v.length; i++) v[i] = x[i];
    const invUNormSq = 2.0 / uNormSq;

    const p: number[] = new Array(n - k - 1).fill(0);
    for (let i = 0; i < v.length; i++) {
      for (let j = 0; j < v.length; j++) {
        p[i] += A[i + k + 1][j + k + 1] * v[j];
      }
      p[i] *= invUNormSq;
    }

    let vTp = 0;
    for (let i = 0; i < v.length; i++) vTp += v[i] * p[i];
    const K = invUNormSq * 0.5 * vTp;

    const q: number[] = new Array(v.length);
    for (let i = 0; i < v.length; i++) q[i] = p[i] - K * v[i];

    for (let i = 0; i < v.length; i++) {
      for (let j = 0; j < v.length; j++) {
        A[i + k + 1][j + k + 1] -= v[i] * q[j] + q[i] * v[j];
      }
    }
    A[k + 1][k] = alpha;
    A[k][k + 1] = alpha;
    for (let i = k + 2; i < n; i++) {
      A[i][k] = 0;
      A[k][i] = 0;
    }

    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let i = 0; i < v.length; i++) dot += v[i] * Q[i + k + 1][j];
      dot *= invUNormSq;
      for (let i = 0; i < v.length; i++) Q[i + k + 1][j] -= dot * v[i];
    }
  }

  const diag = new Array(n);
  const offDiag = new Array(n).fill(0);
  for (let i = 0; i < n; i++) diag[i] = A[i][i];
  for (let i = 0; i < n - 1; i++) offDiag[i] = A[i + 1][i];

  return { diag, offDiag, Q };
}

function solveEigenvaluesSymmetric(H: number[][], n: number): { eigenvalues: number[]; eigenvectors: number[][] } {
  if (n <= 0) return { eigenvalues: [], eigenvectors: [] };
  if (n === 1) return { eigenvalues: [H[0][0]], eigenvectors: [[1]] };

  const { diag, offDiag, Q } = householderTridiagonalize(H, n);
  const eigenvalues = solveTridiagonalEigenvalues(diag, offDiag, n);

  const eigenvectors: number[][] = [];
  const MAX_EVEC_SIZE = 80;
  const computeEvecs = n <= MAX_EVEC_SIZE;

  if (computeEvecs) {
    const lm = new Array(n).fill(0);
    const lu = new Array(n).fill(0);

    for (let idx = 0; idx < n; idx++) {
      const lambda = eigenvalues[idx];
      const z = new Array(n);
      for (let i = 0; i < n; i++) z[i] = (i === idx % n) ? 1 : 0.01 / (n + 1);

      const shift = lambda + 1e-10 * (1 + idx * 0.1);
      lm[0] = diag[0] - shift;
      for (let i = 1; i < n; i++) {
        const safe = Math.abs(lm[i - 1]) < 1e-15 ? 1e-15 * (lm[i - 1] >= 0 ? 1 : -1) : lm[i - 1];
        lu[i] = offDiag[i - 1] / safe;
        lm[i] = (diag[i] - shift) - lu[i] * offDiag[i - 1];
      }

      for (let iter = 0; iter < 8; iter++) {
        const w = new Array(n);
        w[0] = z[0];
        for (let i = 1; i < n; i++) w[i] = z[i] - lu[i] * w[i - 1];
        const safe_last = Math.abs(lm[n - 1]) < 1e-15 ? 1e-15 : lm[n - 1];
        z[n - 1] = w[n - 1] / safe_last;
        for (let i = n - 2; i >= 0; i--) {
          const safe = Math.abs(lm[i]) < 1e-15 ? 1e-15 : lm[i];
          z[i] = (w[i] - offDiag[i] * z[i + 1]) / safe;
        }

        let norm = 0;
        for (let i = 0; i < n; i++) norm += z[i] * z[i];
        norm = Math.sqrt(norm) || 1;
        for (let i = 0; i < n; i++) z[i] /= norm;
      }

      const fullVec = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          fullVec[i] += Q[j][i] * z[j];
        }
      }
      let norm2 = 0;
      for (let i = 0; i < n; i++) norm2 += fullVec[i] * fullVec[i];
      norm2 = Math.sqrt(norm2) || 1;
      for (let i = 0; i < n; i++) fullVec[i] /= norm2;
      eigenvectors.push(fullVec);
    }
  }

  return { eigenvalues, eigenvectors };
}

function solveTridiagonalEigenvalues(diag: number[], offDiag: number[], n: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [diag[0]];

  let minVal = diag[0] - Math.abs(offDiag[0] || 0);
  let maxVal = diag[0] + Math.abs(offDiag[0] || 0);
  for (let i = 1; i < n; i++) {
    const lower = diag[i] - Math.abs(offDiag[i] || 0) - Math.abs(offDiag[i - 1] || 0);
    const upper = diag[i] + Math.abs(offDiag[i] || 0) + Math.abs(offDiag[i - 1] || 0);
    if (lower < minVal) minVal = lower;
    if (upper > maxVal) maxVal = upper;
  }

  const eigenvalues: number[] = [];
  const margin = (maxVal - minVal) * 0.01;
  minVal -= margin;
  maxVal += margin;

  function countEigenvaluesBelow(x: number): number {
    let count = 0;
    let d = diag[0] - x;
    if (Math.abs(d) < 1e-12) d = d < 0 ? -1e-12 : 1e-12;
    if (d < 0) count++;
    for (let i = 1; i < n; i++) {
      const safeD = Math.abs(d) < 1e-12 ? (d < 0 ? -1e-12 : 1e-12) : d;
      d = (diag[i] - x) - (offDiag[i - 1] * offDiag[i - 1]) / safeD;
      if (Math.abs(d) < 1e-12) d = d < 0 ? -1e-12 : 1e-12;
      if (d < 0) count++;
    }
    return count;
  }

  for (let eigenIdx = 0; eigenIdx < n; eigenIdx++) {
    let lo = minVal;
    let hi = maxVal;

    for (let iter = 0; iter < 60; iter++) {
      const mid = (lo + hi) / 2;
      if (countEigenvaluesBelow(mid) <= eigenIdx) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    eigenvalues.push((lo + hi) / 2);
  }

  return eigenvalues;
}

const KNOWN_SK_ELEMENTS = new Set(Object.keys(ELEMENTAL_DATA));

const LATTICE_COORD_NUMBER: Record<string, number> = {
  bcc: 8, fcc: 12, hexagonal: 12, tetragonal: 8, cubic: 6,
};

function computeTbConfidence(
  elements: string[],
  latticeType: string,
  formula: string,
): number {
  let structurePrototypeScore = 0.5;
  const highConfLattices = ["bcc", "fcc", "hexagonal"];
  if (highConfLattices.includes(latticeType)) {
    structurePrototypeScore = 0.85;
  } else if (latticeType === "tetragonal") {
    structurePrototypeScore = 0.75;
  }

  try {
    const pred = predictStructure(formula);
    if (pred && pred.confidence > 0.3) {
      const proto = pred.prototype?.predicted;
      const protoProb = proto ? (pred.prototype.probabilities[proto] ?? 0) : 0;
      if (protoProb > 0.5) {
        structurePrototypeScore = Math.min(1.0, structurePrototypeScore + 0.15);
      } else if (protoProb > 0.3) {
        structurePrototypeScore = Math.min(1.0, structurePrototypeScore + 0.08);
      }
    }
  } catch {}

  const elementsWithParams = elements.filter(el => KNOWN_SK_ELEMENTS.has(el));
  const elementsWithData = elements.filter(el => getElementData(el) !== undefined);
  const fullCoverage = elements.length > 0
    ? elementsWithParams.length / elements.length
    : 0;
  const dataCoverage = elements.length > 0
    ? elementsWithData.length / elements.length
    : 0;
  const elementCoverage = Math.max(fullCoverage, dataCoverage * 0.8);

  let orbitalCompleteness = 1.0;
  const tmElements = elements.filter(el => isTransitionMetal(el) || isRareEarth(el) || isActinide(el));
  if (tmElements.length > 0) {
    const tmWithData = tmElements.filter(el => getElementData(el) !== undefined);
    const tmCoverage = tmWithData.length / tmElements.length;
    orbitalCompleteness = tmCoverage;

    if (elements.some(el => isActinide(el))) {
      orbitalCompleteness *= 0.7;
    } else if (elements.some(el => isRareEarth(el))) {
      orbitalCompleteness *= 0.8;
    }
  }

  if (elements.includes("O") && tmElements.length > 0) {
    orbitalCompleteness *= 0.85;
  }

  const confidence = structurePrototypeScore * elementCoverage * orbitalCompleteness;
  return Number(Math.min(1.0, Math.max(0.0, confidence)).toFixed(4));
}

function estimateSynthesisPressureGPa(elements: string[], counts: Record<string, number>): number {
  const hCount = counts["H"] || 0;
  if (hCount === 0) return 0;
  const nonH = elements.filter(e => e !== "H");
  if (nonH.length === 0) return 0;
  let nonHTotal = 0;
  for (const el of nonH) nonHTotal += counts[el] || 1;
  const hRatio = hCount / Math.max(1, nonHTotal);
  if (hRatio >= 8) return 250;
  if (hRatio >= 6) return 200;
  if (hRatio >= 4) return 150;
  if (hRatio >= 2) return 50;
  return 0;
}

function applyMurnaghanCompression(a0: number, pressureGPa: number): number {
  if (pressureGPa <= 0) return a0;
  const B0 = 100;
  const Bp = 4.0;
  const volumeRatio = Math.pow(1 + (Bp * pressureGPa) / B0, -1.0 / Bp);
  return a0 * Math.pow(volumeRatio, 1.0 / 3.0);
}

export function computeTightBindingBands(
  formula: string,
  structure?: string | null,
  pressureGPa?: number,
): TBBandStructure {
  const { elements, counts } = parseComposition(formula);

  const latticeType = guessLatticeType(elements, formula);
  const path = getHighSymmetryPath(latticeType);

  let latticeConstant = 0;
  let totalAtoms = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const count = counts[el] || 1;
    if (data && data.latticeConstant) {
      latticeConstant += data.latticeConstant * count;
      totalAtoms += count;
    }
  }
  if (totalAtoms > 0) {
    latticeConstant /= totalAtoms;
  } else {
    const nElements = elements.length;
    latticeConstant = nElements >= 3 ? 5.5 : nElements === 2 ? 4.5 : 3.5;
  }

  const effectivePressure = pressureGPa ?? estimateSynthesisPressureGPa(elements, counts);
  if (effectivePressure > 0) {
    latticeConstant = applyMurnaghanCompression(latticeConstant, effectivePressure);
  }

  const hasTM = elements.some(el => isTransitionMetal(el) || isRareEarth(el));
  const hasCorrelatedOxide = hasTM && elements.includes("O");
  const nPerSegment = hasCorrelatedOxide ? 50 : (hasTM ? 40 : 30);
  const { kPoints, kLabels } = interpolateKPoints(path, nPerSegment);

  const bands: number[][] = [];
  const orbitalChars: { s: number; p: number; d: number }[][] = [];
  let nOrbitals = 0;
  let kFailures = 0;

  for (let ki = 0; ki < kPoints.length; ki++) {
    try {
      const result = buildHamiltonianAtK(kPoints[ki], elements, counts, latticeConstant, latticeType);
      bands.push(result.eigenvalues);
      orbitalChars.push(result.orbChars);
      if (ki === 0) nOrbitals = result.eigenvalues.length;
    } catch {
      kFailures++;
      if (bands.length > 0) {
        bands.push(bands[bands.length - 1]);
        orbitalChars.push(orbitalChars[orbitalChars.length - 1]);
      }
    }
  }

  let fermiEnergy = 0;
  if (bands.length > 0 && bands[0].length > 0) {
    let totalElectrons = 0;
    for (const el of elements) {
      const data = getElementData(el);
      if (data) totalElectrons += data.valenceElectrons * (counts[el] || 1);
    }
    const occupiedStates = Math.floor(totalElectrons / 2);

    const nK = kPoints.length;
    const segmentDistances: number[] = new Array(nK - 1);
    for (let ki = 0; ki < nK - 1; ki++) {
      const a = kPoints[ki], b = kPoints[ki + 1];
      segmentDistances[ki] = Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) || 1e-10;
    }

    const kWeights = new Array(nK).fill(0);
    for (let ki = 0; ki < nK; ki++) {
      if (ki > 0) kWeights[ki] += segmentDistances[ki - 1] * 0.5;
      if (ki < nK - 1) kWeights[ki] += segmentDistances[ki] * 0.5;
    }
    let totalKWeight = 0;
    for (let ki = 0; ki < nK; ki++) totalKWeight += kWeights[ki];
    if (totalKWeight > 0) {
      for (let ki = 0; ki < nK; ki++) kWeights[ki] /= totalKWeight;
    } else {
      const uniformW = 1.0 / nK;
      for (let ki = 0; ki < nK; ki++) kWeights[ki] = uniformW;
    }

    const nBandsK = bands[0]?.length || 0;
    const weightedEigenvalues: { e: number; w: number }[] = [];
    for (let ki = 0; ki < bands.length; ki++) {
      const w = kWeights[ki];
      for (const e of bands[ki]) {
        if (Number.isFinite(e)) weightedEigenvalues.push({ e, w });
      }
    }
    weightedEigenvalues.sort((a, b) => a.e - b.e);

    const targetFraction = occupiedStates / (nBandsK || 1);
    let cumulativeWeight = 0;
    for (const { e, w } of weightedEigenvalues) {
      cumulativeWeight += w;
      if (cumulativeWeight >= targetFraction) {
        fermiEnergy = e;
        break;
      }
    }
    if (fermiEnergy === 0 && weightedEigenvalues.length > 0) {
      const flatIdx = Math.min(
        occupiedStates * bands.length - 1,
        weightedEigenvalues.length - 1
      );
      if (flatIdx >= 0) fermiEnergy = weightedEigenvalues[Math.max(0, flatIdx)].e;
    }
  }

  let tbConfidence = computeTbConfidence(elements, latticeType, formula);
  if (kFailures > 0) {
    const failRatio = kFailures / kPoints.length;
    tbConfidence = Number((tbConfidence * (1 - failRatio)).toFixed(4));
  }
  if (bands.length === 0) {
    tbConfidence = 0;
  }

  return {
    kPoints,
    kLabels,
    bands,
    orbitalCharacters: orbitalChars,
    fermiEnergy,
    nOrbitals,
    formula,
    tbConfidence,
    latticeType,
  };
}

export function computeWannierProjection(bands: TBBandStructure): WannierProjection {
  const nBands = bands.nOrbitals;
  const nK = bands.bands.length;

  const orbitalBandCharacter: { orbital: string; bandwidth: number; center: number }[] = [];
  const effectiveHoppings: { orbital: string; value: number }[] = [];

  for (let b = 0; b < nBands; b++) {
    const energies = bands.bands.map(bk => bk[b] ?? 0);
    const min = Math.min(...energies);
    const max = Math.max(...energies);
    const bandwidth = max - min;
    const center = (max + min) / 2;

    let avgS = 0, avgP = 0, avgD = 0;
    for (const oc of bands.orbitalCharacters) {
      if (oc[b]) {
        avgS += oc[b].s;
        avgP += oc[b].p;
        avgD += oc[b].d;
      }
    }
    const nChars = bands.orbitalCharacters.length || 1;
    avgS /= nChars;
    avgP /= nChars;
    avgD /= nChars;

    const chars = [
      { label: "s", val: avgS },
      { label: "p", val: avgP },
      { label: "d", val: avgD },
    ].sort((a, b) => b.val - a.val);

    let orbType: string;
    const total = avgS + avgP + avgD || 1;
    const first = chars[0].val / total;
    const second = chars[1].val / total;
    if (first - second < 0.10 && second > 0.15) {
      orbType = `${chars[0].label}${chars[1].label}-hybrid`;
    } else {
      orbType = chars[0].label;
    }

    orbitalBandCharacter.push({
      orbital: `band_${b}_${orbType}`,
      bandwidth: Number(bandwidth.toFixed(4)),
      center: Number(center.toFixed(4)),
    });

    const coordNumber = LATTICE_COORD_NUMBER[bands.latticeType] ?? 6;
    const hopping = bandwidth / (2 * Math.max(1, coordNumber));
    effectiveHoppings.push({
      orbital: `band_${b}_${orbType}`,
      value: Number(hopping.toFixed(4)),
    });
  }

  let wannierSpread = 0;
  for (const obc of orbitalBandCharacter) {
    wannierSpread += 1.0 / Math.max(0.01, obc.bandwidth);
  }
  wannierSpread = nBands > 0 ? Number((wannierSpread / nBands).toFixed(4)) : 1.0;

  const maxLocalization = Number((1.0 / (1.0 + wannierSpread * 0.1)).toFixed(4));

  return {
    effectiveHoppings,
    orbitalBandCharacter,
    wannierSpread,
    maxLocalization,
  };
}

export function detectBandTopology(bands: TBBandStructure): TopologyFeatures {
  const nBands = bands.nOrbitals;
  const nK = bands.bands.length;
  const flatBands: TopologyFeatures["flatBands"] = [];
  const vanHoveSingularities: TopologyFeatures["vanHoveSingularities"] = [];
  const diracCrossings: TopologyFeatures["diracCrossings"] = [];
  const bandInversions: TopologyFeatures["bandInversions"] = [];

  for (let b = 0; b < nBands; b++) {
    const energies = bands.bands.map(bk => bk[b] ?? 0);
    const min = Math.min(...energies);
    const max = Math.max(...energies);
    const bandwidth = max - min;
    const center = (max + min) / 2;

    if (bandwidth < 0.1) {
      flatBands.push({ bandIndex: b, bandwidth: Number(bandwidth.toFixed(4)), center: Number(center.toFixed(4)) });
    }

    for (let ki = 1; ki < nK - 1; ki++) {
      const ePrev = energies[ki - 1];
      const eCurr = energies[ki];
      const eNext = energies[ki + 1];

      const d2E = eNext - 2 * eCurr + ePrev;
      if (Math.abs(d2E) < 0.005 && bandwidth > 0.05) {
        const type = eCurr > bands.fermiEnergy ? "saddle-above" : "saddle-below";
        vanHoveSingularities.push({
          bandIndex: b,
          kIndex: ki,
          energy: Number(eCurr.toFixed(4)),
          type,
        });
      }
    }
  }

  for (let b = 0; b < nBands - 1; b++) {
    for (let ki = 1; ki < nK - 1; ki++) {
      const e1 = bands.bands[ki]?.[b] ?? 0;
      const e2 = bands.bands[ki]?.[b + 1] ?? 0;
      const gap = Math.abs(e2 - e1);

      if (gap < 0.02) {
        const e1Prev = bands.bands[ki - 1]?.[b] ?? 0;
        const e1Next = bands.bands[ki + 1]?.[b] ?? 0;
        const slope1 = (e1Next - e1Prev) / 2;
        const e2Prev = bands.bands[ki - 1]?.[b + 1] ?? 0;
        const e2Next = bands.bands[ki + 1]?.[b + 1] ?? 0;
        const slope2 = (e2Next - e2Prev) / 2;

        if (Math.abs(slope1) > 0.01 && Math.abs(slope2) > 0.01 && slope1 * slope2 < 0) {
          const velocity = Math.abs(slope1 - slope2) / 2;
          diracCrossings.push({
            kIndex: ki,
            energy: Number(((e1 + e2) / 2).toFixed(4)),
            velocity: Number(velocity.toFixed(4)),
          });
        }
      }
    }
  }

  for (let b = 0; b < nBands - 1; b++) {
    const gammaIdx = 0;
    const zoneEdgeIdx = Math.floor(nK / 2);

    if (gammaIdx < nK && zoneEdgeIdx < nK) {
      const gapAtGamma = (bands.bands[gammaIdx]?.[b + 1] ?? 0) - (bands.bands[gammaIdx]?.[b] ?? 0);
      const gapAtEdge = (bands.bands[zoneEdgeIdx]?.[b + 1] ?? 0) - (bands.bands[zoneEdgeIdx]?.[b] ?? 0);

      if (gapAtGamma * gapAtEdge < 0 && Math.abs(gapAtGamma) > 0.01 && Math.abs(gapAtEdge) > 0.01) {
        bandInversions.push({
          kIndex: zoneEdgeIdx,
          bands: [b, b + 1],
          energyGap: Number(Math.abs(gapAtEdge).toFixed(4)),
        });
      }
    }
  }

  const hasFlatBand = flatBands.length > 0;
  const hasVHS = vanHoveSingularities.length > 0;
  const hasDiracCrossing = diracCrossings.length > 0;
  const hasBandInversion = bandInversions.length > 0;

  let topologyScore = 0;
  if (hasFlatBand) topologyScore += 0.3;
  if (hasVHS) topologyScore += 0.2;
  if (hasDiracCrossing) topologyScore += 0.25;
  if (hasBandInversion) topologyScore += 0.25;

  const vhsNearFermi = vanHoveSingularities.filter(
    v => Math.abs(v.energy - bands.fermiEnergy) < 0.5
  );
  if (vhsNearFermi.length > 0) topologyScore += 0.1;

  topologyScore = Math.min(1.0, topologyScore);

  return {
    flatBands,
    vanHoveSingularities,
    diracCrossings,
    bandInversions,
    hasFlatBand,
    hasVHS,
    hasDiracCrossing,
    hasBandInversion,
    topologyScore: Number(topologyScore.toFixed(3)),
  };
}

export function computeTightBindingDOS(bands: TBBandStructure): TBDOS {
  const nBands = bands.nOrbitals;
  const nK = bands.bands.length;

  let allEnergies: number[] = [];
  for (let ki = 0; ki < nK; ki++) {
    for (let b = 0; b < nBands; b++) {
      const e = bands.bands[ki]?.[b];
      if (e !== undefined) allEnergies.push(e);
    }
  }

  if (allEnergies.length === 0) {
    return { energies: [], dos: [], dosAtFermi: 0, totalStates: 0, uncertaintyMultiplier: 1.0 };
  }

  const eMin = Math.min(...allEnergies);
  const eMax = Math.max(...allEnergies);
  const nBins = 200;
  const rawDE = (eMax - eMin) / nBins;
  const dE = rawDE > 1e-6 ? rawDE : 0.01;
  const broadening = Math.max(dE * 2, 0.005);

  const energies: number[] = [];
  const dos: number[] = [];

  for (let i = 0; i <= nBins; i++) {
    const e = eMin + i * dE;
    energies.push(Number(e.toFixed(4)));

    let density = 0;
    for (const eVal of allEnergies) {
      const x = (e - eVal) / broadening;
      density += Math.exp(-0.5 * x * x) / (broadening * Math.sqrt(2 * Math.PI));
    }
    density /= (nK || 1);
    dos.push(Number(density.toFixed(6)));
  }

  let dosAtFermi = 0;
  let minDist = Infinity;
  for (let i = 0; i < energies.length; i++) {
    const dist = Math.abs(energies[i] - bands.fermiEnergy);
    if (dist < minDist) {
      minDist = dist;
      dosAtFermi = dos[i];
    }
  }

  const totalStates = dos.reduce((s, d) => s + d * dE, 0);

  const tbConf = bands.tbConfidence;
  const uncertaintyMultiplier = tbConf >= 0.7 ? 1.0
    : tbConf >= 0.4 ? 1.0 + (0.7 - tbConf)
    : 2.0;

  return {
    energies,
    dos,
    dosAtFermi: Number(dosAtFermi.toFixed(6)),
    totalStates: Number(totalStates.toFixed(4)),
    uncertaintyMultiplier,
  };
}

const tbCache = new Map<string, { bands: TBBandStructure; wannier: WannierProjection; topology: TopologyFeatures; dos: TBDOS }>();
const TB_CACHE_MAX = 500;

export function computeFullTightBinding(formula: string, structure?: string | null): {
  bands: TBBandStructure;
  wannier: WannierProjection;
  topology: TopologyFeatures;
  dos: TBDOS;
} {
  const cacheKey = `${formula}|${structure ?? ""}`;
  const cached = tbCache.get(cacheKey);
  if (cached) return cached;

  const bands = computeTightBindingBands(formula, structure);
  const wannier = computeWannierProjection(bands);
  const topology = detectBandTopology(bands);
  const dos = computeTightBindingDOS(bands);

  const result = { bands, wannier, topology, dos };
  if (tbCache.size >= TB_CACHE_MAX) {
    const firstKey = tbCache.keys().next().value;
    if (firstKey !== undefined) tbCache.delete(firstKey);
  }
  tbCache.set(cacheKey, result);
  return result;
}
