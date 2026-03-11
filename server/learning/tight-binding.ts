import {
  ELEMENTAL_DATA,
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "./elemental-data";

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
}

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
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
  return counts;
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

  const decay = Math.exp(-1.5 * (distance / r0 - 1.0));

  const ve1 = d1 ? d1.valenceElectrons : 4;
  const ve2 = d2 ? d2.valenceElectrons : 4;
  const veAvg = (ve1 + ve2) / 2;

  const baseSsSigma = -1.5 * decay;
  const baseSpSigma = 1.8 * decay;
  const basePpSigma = 2.5 * decay;
  const basePpPi = -0.8 * decay;

  const hasDElectrons = (isTransitionMetal(el1) || isRareEarth(el1) || isActinide(el1))
    || (isTransitionMetal(el2) || isRareEarth(el2) || isActinide(el2));
  const dScale = hasDElectrons ? 1.0 : 0.1;

  return {
    ssSigma: baseSsSigma,
    spSigma: baseSpSigma,
    ppSigma: basePpSigma,
    ppPi: basePpPi,
    sdSigma: -1.2 * decay * dScale,
    pdSigma: -1.5 * decay * dScale,
    pdPi: 0.7 * decay * dScale,
    ddSigma: -2.0 * decay * dScale * (veAvg > 4 ? 1.2 : 0.8),
    ddPi: 1.2 * decay * dScale,
    ddDelta: -0.4 * decay * dScale,
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
        ],
        labels: ["Γ", "K", "M", "Γ"],
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

function guessLatticeType(elements: string[]): string {
  if (elements.length === 1) {
    const el = elements[0];
    if (["Fe", "Cr", "V", "Nb", "Mo", "W", "Ta", "Na", "K", "Li", "Ba"].includes(el)) return "bcc";
    if (["Cu", "Ag", "Au", "Al", "Ni", "Pd", "Pt", "Pb", "Ca", "Sr"].includes(el)) return "fcc";
    if (["Ti", "Zr", "Hf", "Co", "Zn", "Mg", "Be", "Y", "Sc"].includes(el)) return "hexagonal";
  }
  if (elements.includes("B") && elements.some(e => isTransitionMetal(e) || isRareEarth(e))) return "hexagonal";
  if (elements.length >= 3 && elements.includes("O")) return "cubic";
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
      const t = i / nPerSegment;
      kPoints.push([
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t,
      ]);
    }

    kLabels.push({ index: kPoints.length, label: path.labels[seg + 1] });
  }

  kPoints.push(path.points[path.points.length - 1]);

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

  const eigenvalues = solveEigenvaluesSymmetric(H, nOrbitals);

  const orbChars: { s: number; p: number; d: number }[] = [];
  for (let band = 0; band < nOrbitals; band++) {
    let sWeight = 0, pWeight = 0, dWeight = 0;
    for (const atom of atomList) {
      const hasDOrbs = isTransitionMetal(atom.el) || isRareEarth(atom.el) || isActinide(atom.el);
      const o = atom.orbitalStart;
      sWeight += o < nOrbitals ? 1.0 / nOrbitals : 0;
      for (let p = 0; p < 3; p++) {
        pWeight += (o + 1 + p < nOrbitals) ? 1.0 / nOrbitals : 0;
      }
      if (hasDOrbs) {
        for (let d = 0; d < 5; d++) {
          dWeight += (o + 4 + d < nOrbitals) ? 1.0 / nOrbitals : 0;
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

function solveEigenvaluesSymmetric(H: number[][], n: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [H[0][0]];

  const eigenvalues: number[] = [];

  const diag = H.map((row, i) => row[i]);
  const offDiag: number[] = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        offDiag[i] += H[i][j] * H[i][j];
      }
    }
    offDiag[i] = Math.sqrt(offDiag[i]);
  }

  for (let i = 0; i < n; i++) {
    const gershgorinLow = diag[i] - offDiag[i];
    const gershgorinHigh = diag[i] + offDiag[i];
    eigenvalues.push((gershgorinLow + gershgorinHigh) / 2);
  }

  const tridiag = new Array(n).fill(0);
  const offTridiag = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    tridiag[i] = H[i][i];
  }
  for (let i = 0; i < n - 1; i++) {
    let sumSq = 0;
    for (let j = i + 1; j < n; j++) {
      sumSq += H[j][i] * H[j][i];
    }
    offTridiag[i] = Math.sqrt(sumSq);
  }

  const stEigenvalues = solveTridiagonalEigenvalues(tridiag, offTridiag, n);

  return stEigenvalues.sort((a, b) => a - b);
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
    let d = 1.0;
    for (let i = 0; i < n; i++) {
      d = (diag[i] - x) - (i > 0 && d !== 0 ? (offDiag[i - 1] * offDiag[i - 1]) / d : 0);
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

const KNOWN_SK_ELEMENTS = new Set([
  "H", "Li", "Be", "B", "C", "N", "O", "F",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl",
  "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br",
  "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
  "In", "Sn", "Sb", "Te", "I",
  "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
  "Tl", "Pb", "Bi",
]);

function computeTbConfidence(
  elements: string[],
  latticeType: string,
  formula: string,
): number {
  const knownPrototypes = ["bcc", "fcc", "hexagonal"];
  const structurePrototypeScore = knownPrototypes.includes(latticeType) ? 1.0 : 0.5;

  const elementsWithParams = elements.filter(el => KNOWN_SK_ELEMENTS.has(el));
  const elementCoverage = elements.length > 0
    ? elementsWithParams.length / elements.length
    : 0;

  let orbitalCompleteness = 1.0;
  const tmElements = elements.filter(el => isTransitionMetal(el) || isRareEarth(el) || isActinide(el));
  if (tmElements.length > 0) {
    const tmWithParams = tmElements.filter(el => KNOWN_SK_ELEMENTS.has(el));
    const tmCoverage = tmWithParams.length / tmElements.length;
    orbitalCompleteness = tmCoverage;

    if (isActinide(elements[0]) || elements.some(el => isActinide(el))) {
      orbitalCompleteness *= 0.6;
    } else if (elements.some(el => isRareEarth(el))) {
      orbitalCompleteness *= 0.75;
    }
  }

  if (elements.includes("O") && tmElements.length > 0) {
    orbitalCompleteness *= 0.85;
  }

  const confidence = structurePrototypeScore * elementCoverage * orbitalCompleteness;
  return Number(Math.min(1.0, Math.max(0.0, confidence)).toFixed(4));
}

export function computeTightBindingBands(
  formula: string,
  structure?: string | null,
): TBBandStructure {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);

  const latticeType = guessLatticeType(elements);
  const path = getHighSymmetryPath(latticeType);
  const nPerSegment = 30;
  const { kPoints, kLabels } = interpolateKPoints(path, nPerSegment);

  let latticeConstant = 3.5;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.latticeConstant) {
      latticeConstant = data.latticeConstant / 100;
      break;
    }
  }

  const bands: number[][] = [];
  const orbitalChars: { s: number; p: number; d: number }[][] = [];
  let nOrbitals = 0;

  for (let ki = 0; ki < kPoints.length; ki++) {
    const result = buildHamiltonianAtK(kPoints[ki], elements, counts, latticeConstant, latticeType);
    bands.push(result.eigenvalues);
    orbitalChars.push(result.orbChars);
    if (ki === 0) nOrbitals = result.eigenvalues.length;
  }

  let fermiEnergy = 0;
  if (bands.length > 0 && bands[0].length > 0) {
    let totalElectrons = 0;
    for (const el of elements) {
      const data = getElementData(el);
      if (data) totalElectrons += data.valenceElectrons * (counts[el] || 1);
    }
    const occupiedStates = Math.floor(totalElectrons / 2);
    const allEigenvalues: number[] = [];
    for (const bandAtK of bands) {
      for (const e of bandAtK) {
        if (Number.isFinite(e)) allEigenvalues.push(e);
      }
    }
    allEigenvalues.sort((a, b) => a - b);
    const targetIdx = Math.min(
      occupiedStates * bands.length - 1,
      allEigenvalues.length - 1
    );
    if (targetIdx >= 0 && allEigenvalues.length > 0) {
      fermiEnergy = allEigenvalues[Math.max(0, targetIdx)];
    }
  }

  const tbConfidence = computeTbConfidence(elements, latticeType, formula);

  return {
    kPoints,
    kLabels,
    bands,
    orbitalCharacters: orbitalChars,
    fermiEnergy,
    nOrbitals,
    formula,
    tbConfidence,
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

    let orbType = "s";
    if (avgD > avgS && avgD > avgP) orbType = "d";
    else if (avgP > avgS) orbType = "p";

    orbitalBandCharacter.push({
      orbital: `band_${b}_${orbType}`,
      bandwidth: Number(bandwidth.toFixed(4)),
      center: Number(center.toFixed(4)),
    });

    const hopping = bandwidth / (2 * Math.max(1, 6));
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
    return { energies: [], dos: [], dosAtFermi: 0, totalStates: 0 };
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
    density /= (nK * nBands || 1);
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

  return {
    energies,
    dos,
    dosAtFermi: Number(dosAtFermi.toFixed(6)),
    totalStates: Number(totalStates.toFixed(4)),
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
