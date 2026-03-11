import {
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "../learning/elemental-data";

export interface FermiPocket {
  index: number;
  type: "electron" | "hole";
  volume: number;
  cylindricalCharacter: number;
  orbitalCharacter: { s: number; p: number; d: number };
  bandIndex: number;
  avgVelocity: number;
}

export interface NestingVector {
  q: number[];
  strength: number;
  connectedPockets: [number, number];
}

export interface LindhardSusceptibility {
  chi0Peak: number;
  chi0PeakQ: number[];
  chi0Average: number;
  divergenceProximity: number;
  sdwSusceptibility: number;
  cdwSusceptibility: number;
}

export interface FermiSurfaceResult {
  formula: string;
  fermiEnergy: number;
  pocketCount: number;
  pockets: FermiPocket[];
  electronPocketCount: number;
  holePocketCount: number;
  totalElectronVolume: number;
  totalHoleVolume: number;
  electronHoleBalance: number;
  cylindricalCharacter: number;
  nestingVectors: NestingVector[];
  nestingScore: number;
  lindhardSusceptibility: LindhardSusceptibility;
  fsDimensionality: number;
  sigmaBandPresence: number;
  multiBandScore: number;
  mlFeatures: FermiSurfaceMLFeatures;
}

export interface FermiSurfaceMLFeatures {
  fermiPocketCount: number;
  electronHoleBalance: number;
  fsDimensionality: number;
  sigmaBandPresence: number;
  multiBandScore: number;
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

function getSymmetryOps(latticeType: string): number[][][] {
  const identity = [[1,0,0],[0,1,0],[0,0,1]];
  const mx = [[-1,0,0],[0,1,0],[0,0,1]];
  const my = [[1,0,0],[0,-1,0],[0,0,1]];
  const mz = [[1,0,0],[0,1,0],[0,0,-1]];
  const mxy = [[0,1,0],[1,0,0],[0,0,1]];

  switch (latticeType) {
    case "cubic":
      return [identity, mx, my, mz, mxy,
        [[-1,0,0],[0,-1,0],[0,0,1]],
        [[-1,0,0],[0,1,0],[0,0,-1]],
        [[1,0,0],[0,-1,0],[0,0,-1]],
        [[0,0,1],[0,1,0],[1,0,0]],
        [[0,1,0],[0,0,1],[1,0,0]],
      ];
    case "fcc":
    case "bcc":
      return [identity, mx, my, mz, mxy,
        [[-1,0,0],[0,-1,0],[0,0,1]],
        [[-1,0,0],[0,1,0],[0,0,-1]],
        [[1,0,0],[0,-1,0],[0,0,-1]],
      ];
    case "hexagonal":
      return [identity, mx, my, mz,
        [[-1,0,0],[0,-1,0],[0,0,1]],
        [[-1,0,0],[0,1,0],[0,0,-1]],
      ];
    default:
      return [identity, mx, my, mz];
  }
}

function applySymOp(op: number[][], k: number[]): number[] {
  return [
    op[0][0]*k[0] + op[0][1]*k[1] + op[0][2]*k[2],
    op[1][0]*k[0] + op[1][1]*k[1] + op[1][2]*k[2],
    op[2][0]*k[0] + op[2][1]*k[1] + op[2][2]*k[2],
  ];
}

function isInIBZ(kx: number, ky: number, kz: number, latticeType: string): boolean {
  switch (latticeType) {
    case "cubic":
      return kx >= 0 && ky >= 0 && kz >= 0 && ky <= kx;
    case "fcc":
      return kx >= 0 && ky >= 0 && kz >= 0 && ky <= kx;
    case "bcc":
      return kx >= 0 && ky >= 0 && kz >= 0 && ky <= kx;
    case "hexagonal":
      return kx >= 0 && ky >= 0 && kz >= 0;
    default:
      return kx >= 0 && ky >= 0 && kz >= 0;
  }
}

function generateBZGrid(latticeType: string, gridSize: number, caRatio: number = 1.633): number[][] {
  const ibzPoints: number[][] = [];
  const step = 1.0 / gridSize;

  switch (latticeType) {
    case "hexagonal": {
      const hexA = 2.0 / Math.sqrt(3.0);
      const kzMax = 0.5 / caRatio;
      for (let i = 0; i <= gridSize; i++) {
        for (let j = 0; j <= gridSize; j++) {
          for (let k = 0; k <= gridSize; k++) {
            const kx = -0.5 + i * step;
            const ky = -0.5 + j * step;
            const kz = -0.5 + k * step;
            if (Math.abs(kz) <= kzMax) {
              const absY = Math.abs(ky);
              const absX = Math.abs(kx);
              const inHex = absY <= (0.5 * hexA) && (absY + absX * Math.sqrt(3.0)) <= hexA * Math.sqrt(3.0) * 0.5;
              if (inHex && isInIBZ(kx, ky, kz, latticeType)) {
                ibzPoints.push([kx, ky, kz]);
              }
            }
          }
        }
      }
      break;
    }
    case "bcc": {
      for (let i = 0; i <= gridSize; i++) {
        for (let j = 0; j <= gridSize; j++) {
          for (let k = 0; k <= gridSize; k++) {
            const kx = -0.5 + i * step;
            const ky = -0.5 + j * step;
            const kz = -0.5 + k * step;
            const truncOct = Math.abs(kx) + Math.abs(ky) + Math.abs(kz);
            if (truncOct <= 0.75 && Math.abs(kx) <= 0.5 && Math.abs(ky) <= 0.5 && Math.abs(kz) <= 0.5) {
              if (isInIBZ(kx, ky, kz, latticeType)) {
                ibzPoints.push([kx, ky, kz]);
              }
            }
          }
        }
      }
      break;
    }
    case "fcc": {
      for (let i = 0; i <= gridSize; i++) {
        for (let j = 0; j <= gridSize; j++) {
          for (let k = 0; k <= gridSize; k++) {
            const kx = -0.5 + i * step;
            const ky = -0.5 + j * step;
            const kz = -0.5 + k * step;
            const maxPairSum = Math.max(
              Math.abs(kx) + Math.abs(ky),
              Math.abs(ky) + Math.abs(kz),
              Math.abs(kx) + Math.abs(kz),
            );
            if (maxPairSum <= 0.75 && Math.abs(kx) <= 0.5 && Math.abs(ky) <= 0.5 && Math.abs(kz) <= 0.5) {
              if (isInIBZ(kx, ky, kz, latticeType)) {
                ibzPoints.push([kx, ky, kz]);
              }
            }
          }
        }
      }
      break;
    }
    case "cubic":
    default: {
      for (let i = 0; i <= gridSize; i++) {
        for (let j = 0; j <= gridSize; j++) {
          for (let k = 0; k <= gridSize; k++) {
            const kx = -0.5 + i * step;
            const ky = -0.5 + j * step;
            const kz = -0.5 + k * step;
            if (isInIBZ(kx, ky, kz, latticeType)) {
              ibzPoints.push([kx, ky, kz]);
            }
          }
        }
      }
      break;
    }
  }

  const symOps = getSymmetryOps(latticeType);
  const fullGrid: number[][] = [];
  const seen = new Set<string>();
  for (const kIbz of ibzPoints) {
    for (const op of symOps) {
      const kFull = applySymOp(op, kIbz);
      const key = kFull.map(v => v.toFixed(6)).join(",");
      if (!seen.has(key)) {
        seen.add(key);
        fullGrid.push(kFull);
      }
    }
  }

  return fullGrid;
}

interface BZEvaluation {
  k: number[];
  eigenvalues: number[];
  orbChars: { s: number; p: number; d: number }[];
}

function evaluateBZGrid(
  formula: string,
  gridPoints: number[][],
): { evaluations: BZEvaluation[]; fermiEnergy: number; nOrbitals: number } {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  let latticeConstant = 3.5;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.latticeConstant) {
      latticeConstant = data.latticeConstant / 100;
      break;
    }
  }

  let totalVE = 0;
  let nOrbitals = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const count = Math.round(counts[el] || 1);
    if (data) totalVE += data.valenceElectrons * count;
    const hasDOrbs = isTransitionMetal(el) || isRareEarth(el) || isActinide(el);
    nOrbitals += count * (hasDOrbs ? 9 : 4);
  }
  nOrbitals = Math.min(nOrbitals, 50);

  const evaluations: BZEvaluation[] = [];

  for (const kpt of gridPoints) {
    const result = buildHamiltonianAtKForFS(kpt, elements, counts, latticeConstant, nOrbitals);
    evaluations.push({
      k: kpt,
      eigenvalues: result.eigenvalues,
      orbChars: result.orbChars,
    });
  }

  const allEigens: number[] = [];
  for (const ev of evaluations) {
    for (const e of ev.eigenvalues) allEigens.push(e);
  }
  allEigens.sort((a, b) => a - b);

  const nElectrons = Math.round(totalVE);
  const statesPerK = nOrbitals;
  const totalStates = statesPerK * gridPoints.length;
  const filledStates = Math.min(Math.round(nElectrons * gridPoints.length / 2), allEigens.length - 1);
  const fermiEnergy = filledStates > 0 && filledStates < allEigens.length
    ? (allEigens[filledStates - 1] + allEigens[filledStates]) / 2
    : allEigens[Math.floor(allEigens.length / 2)] || 0;

  return { evaluations, fermiEnergy, nOrbitals };
}

function getNeighborVectorsFS(latticeType: string): number[][] {
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

function isCentrosymmetric(elements: string[]): boolean {
  if (elements.length === 1) return true;
  if (elements.length === 2) {
    const el1 = getElementData(elements[0]);
    const el2 = getElementData(elements[1]);
    if (el1 && el2) {
      const enDiff = Math.abs(
        (el1.paulingElectronegativity ?? 2.0) - (el2.paulingElectronegativity ?? 2.0)
      );
      if (enDiff > 1.0) return false;
    }
  }
  if (elements.length >= 3) return false;
  return true;
}

function buildHamiltonianAtKForFS(
  k: number[],
  elements: string[],
  counts: Record<string, number>,
  latticeConstant: number,
  maxOrbitals: number,
): { eigenvalues: number[]; orbChars: { s: number; p: number; d: number }[] } {
  const atomList: { el: string; orbitalStart: number }[] = [];
  let nOrbitals = 0;

  for (const el of elements) {
    const count = Math.round(counts[el] || 1);
    const hasDOrbs = isTransitionMetal(el) || isRareEarth(el) || isActinide(el);
    const orbsPerAtom = hasDOrbs ? 9 : 4;

    for (let i = 0; i < count; i++) {
      if (nOrbitals + orbsPerAtom > maxOrbitals) break;
      atomList.push({ el, orbitalStart: nOrbitals });
      nOrbitals += orbsPerAtom;
    }
  }

  if (nOrbitals > 50) nOrbitals = 50;

  const H: number[][] = [];
  const H_im: number[][] = [];
  for (let i = 0; i < nOrbitals; i++) {
    H[i] = new Array(nOrbitals).fill(0);
    H_im[i] = new Array(nOrbitals).fill(0);
  }

  const centrosymmetric = isCentrosymmetric(elements);

  for (const atom of atomList) {
    const data = getElementData(atom.el);
    if (!data) continue;
    const ie = data.firstIonizationEnergy;
    const ea = data.electronAffinity ?? 0;
    const en = data.paulingElectronegativity ?? 2.0;
    const es = -(ie * 0.5 + ea * 0.5) * 0.8;
    const ep = es + 3.0 + en * 0.5;
    const hasDOrbs = isTransitionMetal(atom.el) || isRareEarth(atom.el) || isActinide(atom.el);
    const ed = hasDOrbs
      ? es + 1.5 + (data.valenceElectrons - 2) * 0.3
      : es + 8.0;

    const o = atom.orbitalStart;
    if (o < nOrbitals) H[o][o] = es;
    if (o + 1 < nOrbitals) H[o + 1][o + 1] = ep;
    if (o + 2 < nOrbitals) H[o + 2][o + 2] = ep;
    if (o + 3 < nOrbitals) H[o + 3][o + 3] = ep;
    if (hasDOrbs) {
      for (let d = 0; d < 5 && o + 4 + d < nOrbitals; d++) {
        H[o + 4 + d][o + 4 + d] = ed;
      }
    }
  }

  const latticeType = guessLatticeType(elements);

  const kDotR = (dx: number, dy: number, dz: number) =>
    2 * Math.PI * (k[0] * dx + k[1] * dy + k[2] * dz);

  const neighbors = getNeighborVectorsFS(latticeType);

  for (let i = 0; i < atomList.length; i++) {
    for (let j = i + 1; j < atomList.length; j++) {
      const a1 = atomList[i];
      const a2 = atomList[j];
      const d1 = getElementData(a1.el);
      const d2 = getElementData(a2.el);
      const r1 = d1 ? d1.atomicRadius / 100 : 1.5;
      const r2 = d2 ? d2.atomicRadius / 100 : 1.5;
      const bondDist = (r1 + r2) * 0.85;
      const r0 = (r1 + r2) * 0.9;
      const decay = Math.exp(-1.5 * (bondDist / r0 - 1.0));

      const ve1 = d1 ? d1.valenceElectrons : 4;
      const ve2 = d2 ? d2.valenceElectrons : 4;
      const veAvg = (ve1 + ve2) / 2;

      const hasDI = isTransitionMetal(a1.el) || isRareEarth(a1.el) || isActinide(a1.el);
      const hasDJ = isTransitionMetal(a2.el) || isRareEarth(a2.el) || isActinide(a2.el);
      const dScale = (hasDI || hasDJ) ? 1.0 : 0.1;

      const ssSigma = -1.5 * decay;
      const spSigma = 1.8 * decay;
      const ppSigma = 2.5 * decay;
      const ppPi = -0.8 * decay;
      const sdSigma = -1.2 * decay * dScale;
      const ddSigma = -2.0 * decay * dScale * (veAvg > 4 ? 1.2 : 0.8);
      const ddPi = 1.2 * decay * dScale;
      const ddDelta = -0.4 * decay * dScale;

      for (const [dx, dy, dz] of neighbors) {
        const theta = kDotR(dx, dy, dz);
        const phaseRe = Math.cos(theta);
        const phaseIm = centrosymmetric ? 0 : -Math.sin(theta);
        const oi = a1.orbitalStart;
        const oj = a2.orbitalStart;

        if (oi < nOrbitals && oj < nOrbitals) {
          const vRe = ssSigma * phaseRe / neighbors.length;
          H[oi][oj] += vRe;
          H[oj][oi] += vRe;
          if (!centrosymmetric) {
            const vIm = ssSigma * phaseIm / neighbors.length;
            H_im[oi][oj] += vIm;
            H_im[oj][oi] -= vIm;
          }
        }

        for (let p = 0; p < 3; p++) {
          if (oi < nOrbitals && oj + 1 + p < nOrbitals) {
            const vRe = spSigma * phaseRe / neighbors.length * 0.577;
            H[oi][oj + 1 + p] += vRe;
            H[oj + 1 + p][oi] += vRe;
            if (!centrosymmetric) {
              const vIm = spSigma * phaseIm / neighbors.length * 0.577;
              H_im[oi][oj + 1 + p] += vIm;
              H_im[oj + 1 + p][oi] -= vIm;
            }
          }
          if (oj < nOrbitals && oi + 1 + p < nOrbitals) {
            const vRe = spSigma * phaseRe / neighbors.length * 0.577;
            H[oj][oi + 1 + p] += vRe;
            H[oi + 1 + p][oj] += vRe;
            if (!centrosymmetric) {
              const vIm = spSigma * phaseIm / neighbors.length * 0.577;
              H_im[oj][oi + 1 + p] -= vIm;
              H_im[oi + 1 + p][oj] += vIm;
            }
          }
        }

        for (let p1 = 0; p1 < 3; p1++) {
          for (let p2 = 0; p2 < 3; p2++) {
            if (oi + 1 + p1 < nOrbitals && oj + 1 + p2 < nOrbitals) {
              const sigmaW = p1 === p2 ? 1.0 / 3.0 : 0;
              const piW = p1 === p2 ? 2.0 / 3.0 : (p1 !== p2 ? -1.0 / 3.0 : 0);
              const hop = ppSigma * sigmaW + ppPi * piW;
              const vRe = hop * phaseRe / neighbors.length;
              H[oi + 1 + p1][oj + 1 + p2] += vRe;
              H[oj + 1 + p2][oi + 1 + p1] += vRe;
              if (!centrosymmetric) {
                const vIm = hop * phaseIm / neighbors.length;
                H_im[oi + 1 + p1][oj + 1 + p2] += vIm;
                H_im[oj + 1 + p2][oi + 1 + p1] -= vIm;
              }
            }
          }
        }

        if (hasDI && hasDJ) {
          for (let d1 = 0; d1 < 5; d1++) {
            for (let d2 = 0; d2 < 5; d2++) {
              if (oi + 4 + d1 < nOrbitals && oj + 4 + d2 < nOrbitals) {
                let hop = 0;
                if (d1 === d2) {
                  hop = ddSigma * 0.2 + ddPi * 0.5 + ddDelta * 0.3;
                } else {
                  hop = (ddPi * 0.3 + ddDelta * 0.1) * 0.5;
                }
                const vRe = hop * phaseRe / neighbors.length;
                H[oi + 4 + d1][oj + 4 + d2] += vRe;
                H[oj + 4 + d2][oi + 4 + d1] += vRe;
                if (!centrosymmetric) {
                  const vIm = hop * phaseIm / neighbors.length;
                  H_im[oi + 4 + d1][oj + 4 + d2] += vIm;
                  H_im[oj + 4 + d2][oi + 4 + d1] -= vIm;
                }
              }
            }
          }
        }

        if (hasDI) {
          for (let d = 0; d < 5; d++) {
            if (oi + 4 + d < nOrbitals && oj < nOrbitals) {
              const vRe = sdSigma * phaseRe / neighbors.length * 0.447;
              H[oi + 4 + d][oj] += vRe;
              H[oj][oi + 4 + d] += vRe;
              if (!centrosymmetric) {
                const vIm = sdSigma * phaseIm / neighbors.length * 0.447;
                H_im[oi + 4 + d][oj] += vIm;
                H_im[oj][oi + 4 + d] -= vIm;
              }
            }
          }
        }
        if (hasDJ) {
          for (let d = 0; d < 5; d++) {
            if (oj + 4 + d < nOrbitals && oi < nOrbitals) {
              const vRe = sdSigma * phaseRe / neighbors.length * 0.447;
              H[oj + 4 + d][oi] += vRe;
              H[oi][oj + 4 + d] += vRe;
              if (!centrosymmetric) {
                const vIm = sdSigma * phaseIm / neighbors.length * 0.447;
                H_im[oj + 4 + d][oi] += vIm;
                H_im[oi][oj + 4 + d] -= vIm;
              }
            }
          }
        }
      }
    }
  }

  let eigenvalues: number[];
  if (centrosymmetric) {
    eigenvalues = solveEigenvaluesSymmetricFS(H, nOrbitals);
  } else {
    const H_eff: number[][] = [];
    for (let i = 0; i < nOrbitals; i++) {
      H_eff[i] = new Array(nOrbitals).fill(0);
      for (let j = 0; j < nOrbitals; j++) {
        H_eff[i][j] = H[i][j];
      }
    }
    for (let i = 0; i < nOrbitals; i++) {
      for (let j = 0; j < nOrbitals; j++) {
        if (i !== j) {
          const re = H[i][j];
          const im = H_im[i][j];
          const mag = Math.sqrt(re * re + im * im);
          const sign = re >= 0 ? 1 : -1;
          H_eff[i][j] = sign * mag;
        }
      }
    }
    eigenvalues = solveEigenvaluesSymmetricFS(H_eff, nOrbitals);
  }

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
    orbChars.push({ s: sWeight / total, p: pWeight / total, d: dWeight / total });
  }

  return { eigenvalues, orbChars };
}

function solveEigenvaluesSymmetricFS(H: number[][], n: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [H[0][0]];

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

  return solveTridiagonalFS(tridiag, offTridiag, n).sort((a, b) => a - b);
}

function solveTridiagonalFS(diag: number[], offDiag: number[], n: number): number[] {
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

  function countBelow(x: number): number {
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
      if (countBelow(mid) <= eigenIdx) lo = mid;
      else hi = mid;
    }
    eigenvalues.push((lo + hi) / 2);
  }

  return eigenvalues;
}

function detectFermiPockets(
  evaluations: BZEvaluation[],
  fermiEnergy: number,
  nOrbitals: number,
): FermiPocket[] {
  const pockets: FermiPocket[] = [];
  const fermiTolerance = 0.15;

  for (let b = 0; b < nOrbitals; b++) {
    let crossingCount = 0;
    let aboveCount = 0;
    let belowCount = 0;
    let totalVelocity = 0;
    let totalS = 0, totalP = 0, totalD = 0;
    let charCount = 0;

    for (const ev of evaluations) {
      if (b >= ev.eigenvalues.length) continue;
      const energy = ev.eigenvalues[b];
      const diff = energy - fermiEnergy;

      if (Math.abs(diff) < fermiTolerance) {
        crossingCount++;
      }
      if (diff > 0) aboveCount++;
      else belowCount++;

      if (ev.orbChars[b]) {
        totalS += ev.orbChars[b].s;
        totalP += ev.orbChars[b].p;
        totalD += ev.orbChars[b].d;
        charCount++;
      }
    }

    if (crossingCount < 2) continue;

    const totalPoints = evaluations.length;
    const volumeFraction = crossingCount / totalPoints;

    const type: "electron" | "hole" = aboveCount > belowCount ? "electron" : "hole";

    for (let i = 1; i < evaluations.length; i++) {
      if (b < evaluations[i].eigenvalues.length && b < evaluations[i - 1].eigenvalues.length) {
        const dE = Math.abs(evaluations[i].eigenvalues[b] - evaluations[i - 1].eigenvalues[b]);
        const dk = Math.sqrt(
          (evaluations[i].k[0] - evaluations[i - 1].k[0]) ** 2 +
          (evaluations[i].k[1] - evaluations[i - 1].k[1]) ** 2 +
          (evaluations[i].k[2] - evaluations[i - 1].k[2]) ** 2
        );
        if (dk > 0.001) totalVelocity += dE / dk;
      }
    }
    const avgVelocity = totalVelocity / Math.max(1, evaluations.length - 1);

    let kzVariation = 0;
    let kxyVariation = 0;
    const crossingKPoints = evaluations.filter((ev, _) =>
      b < ev.eigenvalues.length && Math.abs(ev.eigenvalues[b] - fermiEnergy) < fermiTolerance
    );

    if (crossingKPoints.length > 2) {
      const kzValues = crossingKPoints.map(ev => ev.k[2]);
      const kxyValues = crossingKPoints.map(ev => Math.sqrt(ev.k[0] ** 2 + ev.k[1] ** 2));
      kzVariation = Math.max(...kzValues) - Math.min(...kzValues);
      kxyVariation = Math.max(...kxyValues) - Math.min(...kxyValues);
    }

    const kxyPlusKz = kxyVariation + kzVariation;
    const cylindricalCharacter = kxyVariation > 0.01 && kxyPlusKz > 0.001
      ? Math.min(1.0, kzVariation > 0.01 ? kxyVariation / kxyPlusKz : 1.0)
      : 0.0;

    const orbChar = charCount > 0
      ? { s: totalS / charCount, p: totalP / charCount, d: totalD / charCount }
      : { s: 0.33, p: 0.33, d: 0.34 };

    pockets.push({
      index: pockets.length,
      type,
      volume: Number(volumeFraction.toFixed(4)),
      cylindricalCharacter: Number(cylindricalCharacter.toFixed(4)),
      orbitalCharacter: {
        s: Number(orbChar.s.toFixed(4)),
        p: Number(orbChar.p.toFixed(4)),
        d: Number(orbChar.d.toFixed(4)),
      },
      bandIndex: b,
      avgVelocity: Number(avgVelocity.toFixed(4)),
    });
  }

  return pockets;
}

function computeLindhardSusceptibility(
  pockets: FermiPocket[],
  evaluations: BZEvaluation[],
  fermiEnergy: number,
  nestingVectors: NestingVector[],
): LindhardSusceptibility {
  if (pockets.length < 2 || evaluations.length === 0) {
    return {
      chi0Peak: 0, chi0PeakQ: [0, 0, 0], chi0Average: 0,
      divergenceProximity: 0, sdwSusceptibility: 0, cdwSusceptibility: 0,
    };
  }

  const fermiTolerance = 0.15;
  const kBT = 0.025;

  const fermiPoints: { k: number[]; energy: number; velocity: number; bandIndex: number; pocketIndex: number }[] = [];
  for (const pocket of pockets) {
    for (const ev of evaluations) {
      if (pocket.bandIndex < ev.eigenvalues.length) {
        const e = ev.eigenvalues[pocket.bandIndex];
        if (Math.abs(e - fermiEnergy) < fermiTolerance * 2) {
          fermiPoints.push({
            k: ev.k,
            energy: e,
            velocity: pocket.avgVelocity || 0.5,
            bandIndex: pocket.bandIndex,
            pocketIndex: pocket.index,
          });
        }
      }
    }
  }

  if (fermiPoints.length < 4) {
    return {
      chi0Peak: 0, chi0PeakQ: [0, 0, 0], chi0Average: 0,
      divergenceProximity: 0, sdwSusceptibility: 0, cdwSusceptibility: 0,
    };
  }

  const qVectors: { q: number[]; chi0: number }[] = [];

  if (nestingVectors.length > 0) {
    for (const nv of nestingVectors) {
      let chi0_q = 0;
      let pairCount = 0;

      for (let i = 0; i < Math.min(fermiPoints.length, 60); i++) {
        for (let j = 0; j < Math.min(fermiPoints.length, 60); j++) {
          if (i === j) continue;
          if (fermiPoints[i].pocketIndex === fermiPoints[j].pocketIndex) continue;

          const dq = [
            fermiPoints[j].k[0] - fermiPoints[i].k[0] - nv.q[0],
            fermiPoints[j].k[1] - fermiPoints[i].k[1] - nv.q[1],
            fermiPoints[j].k[2] - fermiPoints[i].k[2] - nv.q[2],
          ];
          const dqMag = Math.sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2]);

          if (dqMag < 0.1) {
            const eDiff = fermiPoints[j].energy - fermiPoints[i].energy;
            const denominator = Math.abs(eDiff) + kBT;
            const fDiff = 1 / (1 + Math.exp(fermiPoints[i].energy / kBT))
                        - 1 / (1 + Math.exp(fermiPoints[j].energy / kBT));
            chi0_q += Math.abs(fDiff) / denominator;
            pairCount++;
          }
        }
      }

      if (pairCount > 0) {
        chi0_q /= pairCount;
        chi0_q *= pairCount;
      }

      qVectors.push({ q: nv.q, chi0: chi0_q });
    }
  }

  if (qVectors.length === 0) {
    const dosEF = fermiPoints.filter(fp => Math.abs(fp.energy - fermiEnergy) < fermiTolerance).length;
    const baseChi0 = dosEF * 0.01;
    return {
      chi0Peak: baseChi0,
      chi0PeakQ: [0, 0, 0],
      chi0Average: baseChi0,
      divergenceProximity: Math.min(1.0, baseChi0 * 0.1),
      sdwSusceptibility: baseChi0 * 0.5,
      cdwSusceptibility: baseChi0 * 0.3,
    };
  }

  qVectors.sort((a, b) => b.chi0 - a.chi0);
  const chi0Peak = qVectors[0].chi0;
  const chi0PeakQ = qVectors[0].q;
  const chi0Average = qVectors.reduce((s, v) => s + v.chi0, 0) / qVectors.length;

  const divergenceProximity = Math.min(1.0, chi0Peak / (chi0Peak + 5.0));

  const electronPockets = pockets.filter(p => p.type === "electron");
  const holePockets = pockets.filter(p => p.type === "hole");
  const hasEHNesting = electronPockets.length > 0 && holePockets.length > 0;

  const sdwSusceptibility = chi0Peak * (hasEHNesting ? 1.2 : 0.6) * (1 + divergenceProximity);
  const cdwSusceptibility = chi0Peak * 0.8 * (1 + divergenceProximity * 0.5);

  return {
    chi0Peak: Number(chi0Peak.toFixed(4)),
    chi0PeakQ: chi0PeakQ.map(v => Number(v.toFixed(4))),
    chi0Average: Number(chi0Average.toFixed(4)),
    divergenceProximity: Number(divergenceProximity.toFixed(4)),
    sdwSusceptibility: Number(sdwSusceptibility.toFixed(4)),
    cdwSusceptibility: Number(cdwSusceptibility.toFixed(4)),
  };
}

function computeNestingVectors(
  pockets: FermiPocket[],
  evaluations: BZEvaluation[],
  fermiEnergy: number,
): NestingVector[] {
  if (pockets.length < 2) return [];

  const nestingVectors: NestingVector[] = [];
  const fermiTolerance = 0.15;

  const fermiKPoints: { k: number[]; bandIndex: number; pocketIndex: number }[] = [];
  for (const pocket of pockets) {
    for (const ev of evaluations) {
      if (pocket.bandIndex < ev.eigenvalues.length &&
          Math.abs(ev.eigenvalues[pocket.bandIndex] - fermiEnergy) < fermiTolerance) {
        fermiKPoints.push({
          k: ev.k,
          bandIndex: pocket.bandIndex,
          pocketIndex: pocket.index,
        });
      }
    }
  }

  const sampleSize = Math.min(fermiKPoints.length, 100);
  const sampled = fermiKPoints.length > sampleSize
    ? fermiKPoints.filter((_, i) => i % Math.ceil(fermiKPoints.length / sampleSize) === 0)
    : fermiKPoints;

  const qBins: Map<string, { q: number[]; count: number; pockets: Set<string> }> = new Map();
  const qResolution = 0.05;

  for (let i = 0; i < sampled.length; i++) {
    for (let j = i + 1; j < sampled.length; j++) {
      if (sampled[i].pocketIndex === sampled[j].pocketIndex) continue;

      const q = [
        sampled[j].k[0] - sampled[i].k[0],
        sampled[j].k[1] - sampled[i].k[1],
        sampled[j].k[2] - sampled[i].k[2],
      ];

      const qKey = q.map(v => Math.round(v / qResolution) * qResolution).join(",");

      if (!qBins.has(qKey)) {
        qBins.set(qKey, {
          q: q.map(v => Number(v.toFixed(3))),
          count: 0,
          pockets: new Set(),
        });
      }
      const bin = qBins.get(qKey)!;
      bin.count++;
      bin.pockets.add(`${sampled[i].pocketIndex}-${sampled[j].pocketIndex}`);
    }
  }

  const sortedBins = Array.from(qBins.values())
    .filter(b => b.count > 1)
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);

  const maxCount = sortedBins.length > 0 ? sortedBins[0].count : 1;

  for (const bin of sortedBins) {
    const pocketPairs = Array.from(bin.pockets);
    const firstPair = pocketPairs[0]?.split("-").map(Number) ?? [0, 1];
    nestingVectors.push({
      q: bin.q,
      strength: Number((bin.count / maxCount).toFixed(4)),
      connectedPockets: [firstPair[0], firstPair[1]],
    });
  }

  return nestingVectors;
}

function computeFSDimensionality(
  pockets: FermiPocket[],
  evaluations: BZEvaluation[],
  fermiEnergy: number,
): number {
  if (pockets.length === 0) return 3;

  const fermiTolerance = 0.15;
  let totalCylindrical = 0;
  let totalWeight = 0;

  for (const pocket of pockets) {
    const crossingPoints = evaluations.filter(ev =>
      pocket.bandIndex < ev.eigenvalues.length &&
      Math.abs(ev.eigenvalues[pocket.bandIndex] - fermiEnergy) < fermiTolerance
    );

    if (crossingPoints.length < 2) continue;

    const kzValues = crossingPoints.map(ev => ev.k[2]);
    const kxyValues = crossingPoints.map(ev => Math.sqrt(ev.k[0] ** 2 + ev.k[1] ** 2));

    const kzSpread = Math.max(...kzValues) - Math.min(...kzValues);
    const kxySpread = Math.max(...kxyValues) - Math.min(...kxyValues);

    const kzDispersion = kzSpread / (kxySpread + kzSpread + 0.001);

    totalCylindrical += (1 - kzDispersion) * pocket.volume;
    totalWeight += pocket.volume;
  }

  if (totalWeight < 0.001) return 3;

  let totalKzVar = 0;
  let totalKxyVar = 0;
  let varWeight = 0;

  for (const pocket of pockets) {
    const crossingPoints = evaluations.filter(ev =>
      pocket.bandIndex < ev.eigenvalues.length &&
      Math.abs(ev.eigenvalues[pocket.bandIndex] - fermiEnergy) < fermiTolerance
    );

    if (crossingPoints.length < 2) continue;

    const kzValues = crossingPoints.map(ev => ev.k[2]);
    const kxyValues = crossingPoints.map(ev => Math.sqrt(ev.k[0] ** 2 + ev.k[1] ** 2));

    const kzSpread = Math.max(...kzValues) - Math.min(...kzValues);
    const kxySpread = Math.max(...kxyValues) - Math.min(...kxyValues);

    totalKzVar += kzSpread * pocket.volume;
    totalKxyVar += kxySpread * pocket.volume;
    varWeight += pocket.volume;
  }

  if (varWeight > 0.001) {
    const avgKzVar = totalKzVar / varWeight;
    const avgKxyVar = totalKxyVar / varWeight;

    if (avgKzVar > 3 * avgKxyVar) {
      return 1.0 + avgKxyVar / (avgKzVar + 1e-6);
    }
  }

  const avgCylindrical = totalCylindrical / totalWeight;

  if (avgCylindrical > 0.8) return 2;
  if (avgCylindrical > 0.5) return 2.5;
  return 3;
}

function computeSigmaBandPresence(
  pockets: FermiPocket[],
  elements: string[],
): number {
  let sigmaScore = 0;

  for (const pocket of pockets) {
    const isStronglyBonding =
      pocket.orbitalCharacter.p > 0.4 ||
      (pocket.orbitalCharacter.s > 0.3 && pocket.orbitalCharacter.p > 0.2);

    if (isStronglyBonding) {
      sigmaScore += pocket.volume * 2;
    }
  }

  const hasLightElements = elements.some(el => {
    const data = getElementData(el);
    return data && data.atomicMass < 15;
  });

  if (hasLightElements) sigmaScore *= 1.5;

  const hasBoron = elements.includes("B");
  if (hasBoron) sigmaScore *= 1.3;

  return Number(Math.min(1.0, sigmaScore).toFixed(4));
}

function computeMultiBandScore(pockets: FermiPocket[]): number {
  if (pockets.length <= 1) return 0;

  const electronPockets = pockets.filter(p => p.type === "electron");
  const holePockets = pockets.filter(p => p.type === "hole");

  let score = 0;

  score += Math.min(0.4, pockets.length * 0.1);

  if (electronPockets.length > 0 && holePockets.length > 0) {
    score += 0.3;
  }

  const orbitalTypes = new Set<string>();
  for (const pocket of pockets) {
    if (pocket.orbitalCharacter.d > 0.5) orbitalTypes.add("d");
    else if (pocket.orbitalCharacter.p > 0.5) orbitalTypes.add("p");
    else if (pocket.orbitalCharacter.s > 0.5) orbitalTypes.add("s");
    else orbitalTypes.add("mixed");
  }
  score += Math.min(0.3, (orbitalTypes.size - 1) * 0.15);

  return Number(Math.min(1.0, score).toFixed(4));
}

const fsCache = new Map<string, FermiSurfaceResult>();
const FS_CACHE_MAX = 200;

export function computeFermiSurface(formula: string): FermiSurfaceResult {
  const cached = fsCache.get(formula);
  if (cached) return cached;

  const elements = parseFormulaElements(formula);
  const latticeType = guessLatticeType(elements);

  const gridSize = 8;
  let caRatio = 1.633;
  if (latticeType === "hexagonal") {
    const counts = parseFormulaCounts(formula);
    const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
    const hCount = counts["H"] || 0;
    const hFraction = totalAtoms > 0 ? hCount / totalAtoms : 0;
    if (hFraction > 0.5) {
      caRatio = 1.2 + 0.3 * (1 - hFraction);
    } else if (elements.some(e => ["Ti", "Zr", "Hf"].includes(e))) {
      caRatio = 1.58;
    } else if (elements.some(e => ["Co", "Zn"].includes(e))) {
      caRatio = 1.62;
    } else if (elements.some(e => ["Mg", "Be"].includes(e))) {
      caRatio = 1.623;
    }
  }
  const gridPoints = generateBZGrid(latticeType, gridSize, caRatio);

  const { evaluations, fermiEnergy, nOrbitals } = evaluateBZGrid(formula, gridPoints);

  const pockets = detectFermiPockets(evaluations, fermiEnergy, nOrbitals);

  const nestingVectors = computeNestingVectors(pockets, evaluations, fermiEnergy);

  const electronPockets = pockets.filter(p => p.type === "electron");
  const holePockets = pockets.filter(p => p.type === "hole");
  const totalElectronVolume = electronPockets.reduce((s, p) => s + p.volume, 0);
  const totalHoleVolume = holePockets.reduce((s, p) => s + p.volume, 0);
  const totalVolume = totalElectronVolume + totalHoleVolume;
  const electronHoleBalance = totalVolume > 0
    ? Number((1 - Math.abs(totalElectronVolume - totalHoleVolume) / totalVolume).toFixed(4))
    : 0;

  const avgCylindrical = pockets.length > 0
    ? pockets.reduce((s, p) => s + p.cylindricalCharacter * p.volume, 0) / Math.max(0.001, totalVolume)
    : 0;

  const nestingScore = nestingVectors.length > 0
    ? Number(Math.min(1.0, nestingVectors.reduce((s, nv) => s + nv.strength, 0) / Math.max(1, nestingVectors.length)).toFixed(4))
    : 0;

  const lindhardSusceptibility = computeLindhardSusceptibility(pockets, evaluations, fermiEnergy, nestingVectors);

  const fsDimensionality = computeFSDimensionality(pockets, evaluations, fermiEnergy);
  const sigmaBandPresence = computeSigmaBandPresence(pockets, elements);
  const multiBandScore = computeMultiBandScore(pockets);

  const result: FermiSurfaceResult = {
    formula,
    fermiEnergy: Number(fermiEnergy.toFixed(4)),
    pocketCount: pockets.length,
    pockets,
    electronPocketCount: electronPockets.length,
    holePocketCount: holePockets.length,
    totalElectronVolume: Number(totalElectronVolume.toFixed(4)),
    totalHoleVolume: Number(totalHoleVolume.toFixed(4)),
    electronHoleBalance,
    cylindricalCharacter: Number(avgCylindrical.toFixed(4)),
    nestingVectors,
    nestingScore,
    lindhardSusceptibility,
    fsDimensionality,
    sigmaBandPresence,
    multiBandScore,
    mlFeatures: {
      fermiPocketCount: pockets.length,
      electronHoleBalance,
      fsDimensionality,
      sigmaBandPresence,
      multiBandScore,
    },
  };

  if (fsCache.size >= FS_CACHE_MAX) {
    const firstKey = fsCache.keys().next().value;
    if (firstKey !== undefined) fsCache.delete(firstKey);
  }
  fsCache.set(formula, result);

  return result;
}
