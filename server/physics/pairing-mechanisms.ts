import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  computePhononDispersion,
  computeDynamicSpinSusceptibility,
  estimateBandwidthW,
  parseFormulaElements,
  type ElectronicStructure,
  type PhononSpectrum,
  type ElectronPhononCoupling,
} from "../learning/physics-engine";
import { analyzeTopology } from "./topology-engine";
import {
  getElementData,
  getHubbardU,
  getStonerParameter,
  isTransitionMetal,
  isRareEarth,
} from "../learning/elemental-data";
import { computeFullTightBinding } from "../learning/tight-binding";

export interface ModeResolvedCoupling {
  acousticLambda: number;
  opticalLambda: number;
  highFreqLambda: number;
  softModeLambda: number;
  dominantBranch: string;
  branchContributions: { branch: string; lambda: number; freqRange: string }[];
  hasSoftModeInstability: boolean;
}

export interface PhononPairingResult {
  lambda: number;
  omegaLogK: number;
  muStar: number;
  tcAllenDynes: number;
  modeResolved: ModeResolvedCoupling;
  phononPairingStrength: number;
}

export interface SpinPairingResult {
  chiQ: number;
  stonerEnhancement: number;
  spinFluctuationEnergy: number;
  isNearQCP: boolean;
  correlationFactor: number;
  nestingAmplification: number;
  spinCoupling: number;
  spinPairingStrength: number;
  pairingSymmetry: string;
}

export interface OrbitalPairingResult {
  orbitalDegeneracy: number;
  partiallyFilledCount: number;
  orbitalFluctuation: number;
  interOrbitalHopping: number;
  hundsCoupling: number;
  orbitalPairingStrength: number;
  activeOrbitals: string[];
}

export interface ExcitonicPairingResult {
  effectiveBandGap: number;
  excitonBindingProxy: number;
  electronHoleAsymmetry: number;
  dielectricScreening: number;
  excitonCoupling: number;
  excitonicPairingStrength: number;
  isExcitonicCandidate: boolean;
}

export interface CDWPairingResult {
  nestingDrivenCDW: number;
  vanHoveAmplification: number;
  anharmonicSuppression: number;
  cdwOrderParameter: number;
  cdwScCompetition: number;
  cdwPairingStrength: number;
  isCDWCandidate: boolean;
}

export interface PolaronicPairingResult {
  polaronCoupling: number;
  bcsBecCrossover: number;
  massEnhancement: number;
  bipolaronBindingProxy: number;
  latticeDistortion: number;
  polaronicPairingStrength: number;
  isPolaronicCandidate: boolean;
}

export interface PlasmonPairingResult {
  plasmaFrequency: number;
  plasmonCoupling: number;
  dimensionalityFactor: number;
  metallicScreening: number;
  collectiveOscillationStrength: number;
  plasmonPairingStrength: number;
  isPlasmonCandidate: boolean;
}

export interface PairingProfile {
  formula: string;
  phonon: PhononPairingResult;
  spin: SpinPairingResult;
  orbital: OrbitalPairingResult;
  excitonic: ExcitonicPairingResult;
  cdw: CDWPairingResult;
  polaronic: PolaronicPairingResult;
  plasmon: PlasmonPairingResult;
  compositePairingStrength: number;
  mechanismWeights: { phonon: number; spin: number; orbital: number; excitonic: number; cdw: number; polaronic: number; plasmon: number };
  dominantMechanism: string;
  secondaryMechanism: string;
  pairingSymmetry: string;
  estimatedTcFromPairing: number;
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    counts[m[1]] = (counts[m[1]] || 0) + (m[2] ? parseFloat(m[2]) : 1);
  }
  return counts;
}

export function computePhononPairing(
  formula: string,
  electronic: ElectronicStructure,
  phonon: PhononSpectrum,
  coupling: ElectronPhononCoupling,
): PhononPairingResult {
  const lambda = coupling.lambda;
  const omegaLogK = coupling.omegaLog * 1.44;
  const muStar = coupling.muStar;

  let tcAllenDynes = 0;
  const allenDynesDenom = lambda - muStar * (1 + 0.62 * lambda);
  if (allenDynesDenom > 1e-6) {
    const prefactor = omegaLogK / 1.2;
    const exponent = -1.04 * (1 + lambda) / allenDynesDenom;
    tcAllenDynes = prefactor * Math.exp(exponent);
    if (!Number.isFinite(tcAllenDynes)) tcAllenDynes = 0;

    if (lambda > 1.5 && coupling.omegaLog > 0) {
      const omega2Avg = coupling.omega2Avg ?? (coupling.omegaLog * coupling.omegaLog * 1.2);
      const sqrtLambda = Math.sqrt(lambda);
      const f1 = Math.pow(1 + Math.pow(lambda / (2.46 * (1 + 3.8 * muStar)), 3/2), 1/3);
      const omegaRatio = omega2Avg > 0 ? Math.sqrt(omega2Avg) / coupling.omegaLog : 1.0;
      const f2Base = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + 1.6 * (1 + muStar));
      const f2Exponent = (1 - lambda * lambda) / (1 + lambda * lambda);
      const f2 = Math.pow(Math.max(0.5, f2Base), f2Exponent);
      tcAllenDynes *= f1 * Math.max(1, f2);
    }

    tcAllenDynes = Math.max(0, Math.min(400, tcAllenDynes));
  }

  let modeResolved: ModeResolvedCoupling;
  try {
    const dispersion = computePhononDispersion(formula, electronic, phonon);
    const branches = dispersion.branches;

    let acousticLambda = 0;
    let opticalLambda = 0;
    let highFreqLambda = 0;
    let softModeLambda = 0;
    let maxBranchLambda = 0;
    let dominantBranch = "acoustic";
    const branchContributions: { branch: string; lambda: number; freqRange: string }[] = [];

    const maxFreq = phonon.maxPhononFrequency;
    const nBranches = branches.length;

    for (let b = 0; b < nBranches; b++) {
      const br = branches[b];
      const freqs = br.frequencies;
      const avgFreq = freqs.reduce((s, f) => s + Math.abs(f), 0) / Math.max(1, freqs.length);
      const minFreq = Math.min(...freqs.map(Math.abs));
      const maxBrFreq = Math.max(...freqs.map(Math.abs));

      const isAcoustic = br.label.startsWith("L") || br.label.startsWith("T");
      const isHighFreq = avgFreq > maxFreq * 0.7;

      let branchWeight = avgFreq > 0 ? (1.0 / avgFreq) : 0;
      if (br.isSoft) branchWeight *= 2.0;

      const normSum = branches.reduce((s, bb) => {
        const af = bb.frequencies.reduce((ss, f) => ss + Math.abs(f), 0) / Math.max(1, bb.frequencies.length);
        return s + (af > 0 ? (1.0 / af) * (bb.isSoft ? 2.0 : 1.0) : 0);
      }, 0);
      const branchLambda = normSum > 0
        ? lambda * branchWeight / normSum
        : lambda / Math.max(1, nBranches);

      if (isAcoustic) acousticLambda += branchLambda;
      else if (isHighFreq) highFreqLambda += branchLambda;
      else opticalLambda += branchLambda;

      if (br.isSoft) softModeLambda += branchLambda;

      if (branchLambda > maxBranchLambda) {
        maxBranchLambda = branchLambda;
        dominantBranch = br.label;
      }

      branchContributions.push({
        branch: br.label,
        lambda: Number(branchLambda.toFixed(4)),
        freqRange: `${minFreq.toFixed(0)}-${maxBrFreq.toFixed(0)} cm⁻¹`,
      });
    }

    modeResolved = {
      acousticLambda: Number(acousticLambda.toFixed(4)),
      opticalLambda: Number(opticalLambda.toFixed(4)),
      highFreqLambda: Number(highFreqLambda.toFixed(4)),
      softModeLambda: Number(softModeLambda.toFixed(4)),
      dominantBranch,
      branchContributions: branchContributions.sort((a, b) => b.lambda - a.lambda).slice(0, 8),
      hasSoftModeInstability: softModeLambda > lambda * 0.3,
    };
  } catch {
    modeResolved = {
      acousticLambda: lambda * 0.4,
      opticalLambda: lambda * 0.45,
      highFreqLambda: lambda * 0.15,
      softModeLambda: 0,
      dominantBranch: coupling.dominantPhononBranch,
      branchContributions: [],
      hasSoftModeInstability: false,
    };
  }

  const phononPairingStrength = Math.min(1.0, lambda / 3.0);

  return {
    lambda,
    omegaLogK,
    muStar,
    tcAllenDynes: Number(tcAllenDynes.toFixed(2)),
    modeResolved,
    phononPairingStrength: Number(phononPairingStrength.toFixed(4)),
  };
}

export function computeSpinPairing(
  formula: string,
  electronic: ElectronicStructure,
): SpinPairingResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  const spin = computeDynamicSpinSusceptibility(formula, electronic);
  const dosEF = electronic.densityOfStatesAtFermi;
  const nestingScore = electronic.nestingScore ?? 0;
  const nestingFloor = Math.max(nestingScore, 0.1);
  const correlation = electronic.correlationStrength ?? 0;

  if (totalAtoms <= 0 || elements.length === 0) {
    return {
      chiQ: 0, stonerEnhancement: 1, spinFluctuationEnergy: 0,
      isNearQCP: false, correlationFactor: 0, nestingAmplification: 1,
      spinCoupling: 0, spinPairingStrength: 0, pairingSymmetry: "s-wave",
    };
  }

  let maxUoverW = 0;
  for (const el of elements) {
    const U = getHubbardU(el);
    if (U === null) continue;
    const W = Math.max(0.5, estimateBandwidthW(el));
    const frac = (counts[el] || 1) / totalAtoms;
    const ratio = (U / W) * Math.sqrt(frac);
    if (ratio > maxUoverW) maxUoverW = ratio;
  }
  const correlationFactor = Math.max(0.05, Math.min(2.0, maxUoverW || correlation * 0.5));

  const chiQ = dosEF * nestingFloor * Math.max(1, spin.stonerEnhancement * 0.5);

  const nestingAmplification = nestingScore > 0.5 ? 1.0 + (nestingScore - 0.5) * 2.0 : 1.0;

  const spinCoupling = dosEF * nestingFloor * correlationFactor;

  let spinPairingStrength = Math.min(1.0,
    spinCoupling * 0.3 +
    (spin.isNearQCP ? 0.3 : 0) +
    (nestingScore > 0.7 ? 0.15 : 0) +
    (correlation > 0.5 ? 0.1 : 0) +
    (spin.stonerEnhancement > 5 ? 0.1 : 0)
  );

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.some(e => ["Fe", "Co"].includes(e)) &&
    elements.some(e => ["As", "P", "Se", "S"].includes(e));
  const isHeavyFermion = elements.some(e => isRareEarth(e)) &&
    elements.some(e => ["Si", "Ge", "In", "Ga", "Sn"].includes(e)) &&
    elements.some(e => isTransitionMetal(e));

  if (isCuprate) spinPairingStrength = Math.max(spinPairingStrength, 0.85);
  else if (isPnictide) spinPairingStrength = Math.max(spinPairingStrength, 0.70);
  else if (isHeavyFermion) spinPairingStrength = Math.max(spinPairingStrength, 0.50);

  let pairingSymmetry = "s-wave";
  if (isCuprate) pairingSymmetry = "d-wave (dx2-y2)";
  else if (isPnictide) pairingSymmetry = "s+/-";
  else if (isHeavyFermion) pairingSymmetry = "d-wave";
  else if (spinPairingStrength > 0.5 && nestingScore > 0.5) pairingSymmetry = "d-wave";

  return {
    chiQ: Number(chiQ.toFixed(4)),
    stonerEnhancement: spin.stonerEnhancement,
    spinFluctuationEnergy: spin.spinFluctuationEnergy,
    isNearQCP: spin.isNearQCP,
    correlationFactor: Number(correlationFactor.toFixed(4)),
    nestingAmplification: Number(nestingAmplification.toFixed(4)),
    spinCoupling: Number(spinCoupling.toFixed(4)),
    spinPairingStrength: Number(spinPairingStrength.toFixed(4)),
    pairingSymmetry,
  };
}

export function computeOrbitalPairing(
  formula: string,
  electronic: ElectronicStructure,
): OrbitalPairingResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  const dFraction = electronic.orbitalFractions?.d ?? 0;
  const fFraction = electronic.orbitalFractions?.f ?? 0;

  let partiallyFilledCount = 0;
  const activeOrbitals: string[] = [];

  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;

    if (isTransitionMetal(el)) {
      const ve = data.valenceElectrons;
      if (ve >= 2 && ve <= 8) {
        partiallyFilledCount++;

        if (ve >= 5 && ve <= 7) {
          activeOrbitals.push(`${el}-t2g`);
        }
        if (ve >= 3 && ve <= 5) {
          activeOrbitals.push(`${el}-eg`);
        }
        if (ve >= 4 && ve <= 7) {
          activeOrbitals.push(`${el}-dxy/dxz/dyz`);
        }
      }
    } else if (isRareEarth(el)) {
      partiallyFilledCount++;
      activeOrbitals.push(`${el}-4f`);
    }
  }

  const orbitalDegeneracy = Math.min(5, partiallyFilledCount * (dFraction > 0.3 ? 3 : 1));

  let interOrbitalHopping = 0;
  try {
    const tb = computeFullTightBinding(formula, null);
    if (tb.bands.tbConfidence > 0.3) {
      const nOrb = tb.bands.nOrbitals;
      const flatBands = tb.topology.flatBands.length;
      interOrbitalHopping = Math.min(1.0, (nOrb / 10) * 0.3 + flatBands * 0.1);
    }
  } catch (err) {
    console.error(`[OrbitalPairing] tight-binding computation failed for ${formula}:`, err);
  }

  if (interOrbitalHopping === 0 && dFraction > 0.3) {
    interOrbitalHopping = dFraction * 0.5;
  }

  let hundsCoupling = 0;
  for (const el of elements) {
    if (isTransitionMetal(el)) {
      const data = getElementData(el);
      if (data && data.valenceElectrons >= 4 && data.valenceElectrons <= 7) {
        const frac = (counts[el] || 1) / totalAtoms;
        hundsCoupling = Math.max(hundsCoupling, 0.3 + frac * 0.5);
      }
    }
  }

  const dosEF = electronic.densityOfStatesAtFermi;
  const orbitalFluctuation = orbitalDegeneracy * dosEF * 0.1;

  let orbitalPairingStrength = Math.min(1.0,
    orbitalDegeneracy * 0.08 +
    orbitalFluctuation * 0.15 +
    interOrbitalHopping * 0.3 +
    hundsCoupling * 0.2 +
    (dFraction > 0.4 ? 0.15 : 0)
  );

  const isPnictide = elements.some(e => ["Fe", "Co", "Ni"].includes(e)) &&
    elements.some(e => ["As", "P", "Se"].includes(e));
  const isRuthenate = elements.includes("Ru") && elements.includes("O");
  const isNickelate = elements.includes("Ni") && elements.includes("O") && elements.length >= 3;

  if (isPnictide) orbitalPairingStrength = Math.max(orbitalPairingStrength, 0.65);
  if (isRuthenate) orbitalPairingStrength = Math.max(orbitalPairingStrength, 0.70);
  if (isNickelate) orbitalPairingStrength = Math.max(orbitalPairingStrength, 0.55);

  return {
    orbitalDegeneracy: Number(orbitalDegeneracy.toFixed(2)),
    partiallyFilledCount,
    orbitalFluctuation: Number(orbitalFluctuation.toFixed(4)),
    interOrbitalHopping: Number(interOrbitalHopping.toFixed(4)),
    hundsCoupling: Number(hundsCoupling.toFixed(4)),
    orbitalPairingStrength: Number(orbitalPairingStrength.toFixed(4)),
    activeOrbitals: activeOrbitals.slice(0, 8),
  };
}

export function computeExcitonicPairing(
  formula: string,
  electronic: ElectronicStructure,
): ExcitonicPairingResult {
  const elements = parseFormulaElements(formula);
  const dosEF = electronic.densityOfStatesAtFermi;
  const metallicity = electronic.metallicity;

  let effectiveBandGap = 0;
  if (metallicity >= 0.7) {
    effectiveBandGap = 0;
  } else if (metallicity >= 0.3) {
    effectiveBandGap = (1.0 - metallicity) * 2.0;
  } else {
    effectiveBandGap = (1.0 - metallicity) * 5.0;
  }

  let excitonBindingProxy = 0;
  if (effectiveBandGap > 0 && effectiveBandGap < 2.0) {
    excitonBindingProxy = Math.min(1.0, dosEF / Math.max(0.1, effectiveBandGap));
  } else if (effectiveBandGap <= 0 && metallicity > 0.3 && metallicity < 0.7) {
    excitonBindingProxy = Math.min(1.0, dosEF * 0.3);
  }

  let electronHoleAsymmetry = 0;
  const hasTM = elements.some(e => isTransitionMetal(e));
  const hasChalcogen = elements.some(e => ["Se", "S", "Te"].includes(e));
  if (hasTM && hasChalcogen && metallicity < 0.7) {
    electronHoleAsymmetry = Math.min(1.0, (1.0 - metallicity) * 0.8);
  }
  if (electronic.correlationStrength > 0.5 && metallicity < 0.5) {
    electronHoleAsymmetry = Math.max(electronHoleAsymmetry, electronic.correlationStrength * 0.6);
  }

  let dielectricScreening = metallicity > 0.7 ? 1.0 : Math.max(0.1, metallicity * 1.2);

  const excitonCoupling = excitonBindingProxy * (1.0 - dielectricScreening * 0.5) * (1.0 + electronHoleAsymmetry * 0.3);

  let excitonicPairingStrength = Math.min(1.0,
    excitonCoupling * 0.4 +
    excitonBindingProxy * 0.2 +
    electronHoleAsymmetry * 0.15 +
    (effectiveBandGap > 0 && effectiveBandGap < 0.5 ? 0.2 : 0) +
    (metallicity > 0.3 && metallicity < 0.6 ? 0.1 : 0)
  );

  const isTa2NiSe5Type = elements.includes("Ta") && elements.includes("Ni") && elements.includes("Se");
  const isTMDHetero = hasTM && hasChalcogen && elements.length === 2 && metallicity < 0.5;

  if (isTa2NiSe5Type) excitonicPairingStrength = Math.max(excitonicPairingStrength, 0.80);
  if (isTMDHetero) excitonicPairingStrength = Math.max(excitonicPairingStrength, 0.40);

  const isExcitonicCandidate = excitonicPairingStrength > 0.3 &&
    effectiveBandGap < 2.0 && electronHoleAsymmetry > 0.1;

  return {
    effectiveBandGap: Number(effectiveBandGap.toFixed(4)),
    excitonBindingProxy: Number(excitonBindingProxy.toFixed(4)),
    electronHoleAsymmetry: Number(electronHoleAsymmetry.toFixed(4)),
    dielectricScreening: Number(dielectricScreening.toFixed(4)),
    excitonCoupling: Number(excitonCoupling.toFixed(4)),
    excitonicPairingStrength: Number(excitonicPairingStrength.toFixed(4)),
    isExcitonicCandidate,
  };
}

export function computeCDWPairing(
  formula: string,
  electronic: ElectronicStructure,
  phonon: PhononSpectrum,
  coupling: ElectronPhononCoupling,
): CDWPairingResult {
  const nestingScore = electronic.nestingScore ?? 0;
  const vanHoveProximity = electronic.vanHoveProximity ?? 0;
  const anharmonicityIndex = phonon.anharmonicityIndex ?? 0;
  const dosEF = electronic.densityOfStatesAtFermi;
  const lambda = coupling.lambda;

  const nestingDrivenCDW = Math.min(1.0, nestingScore * dosEF * 0.5);

  const vanHoveAmplification = vanHoveProximity > 0.3
    ? 1.0 + (vanHoveProximity - 0.3) * 2.5
    : 1.0;

  const anharmonicSuppression = Math.max(0.3, 1.0 - anharmonicityIndex * 0.6);

  const cdwOrderParameter = Math.min(1.0,
    nestingDrivenCDW * vanHoveAmplification * anharmonicSuppression
  );

  const cdwScCompetition = cdwOrderParameter > 0.5
    ? Math.max(0.0, 1.0 - (cdwOrderParameter - 0.5) * 1.5)
    : 1.0;

  let cdwPairingStrength = Math.min(1.0,
    cdwOrderParameter * 0.3 * cdwScCompetition +
    nestingScore * 0.2 +
    (vanHoveProximity > 0.5 ? 0.15 : 0) +
    (lambda > 0.5 && nestingScore > 0.4 ? 0.1 : 0) +
    (phonon.softModePresent ? 0.1 : 0)
  );

  const elements = parseFormulaElements(formula);
  const isNbSe2Type = elements.includes("Nb") && elements.includes("Se");
  const isTaSe2Type = elements.includes("Ta") && elements.includes("Se");
  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;

  if (isNbSe2Type) cdwPairingStrength = Math.max(cdwPairingStrength, 0.70);
  if (isTaSe2Type) cdwPairingStrength = Math.max(cdwPairingStrength, 0.65);
  if (isCuprate && nestingScore > 0.4) cdwPairingStrength = Math.max(cdwPairingStrength, 0.50);

  const isCDWCandidate = cdwOrderParameter > 0.3 && nestingScore > 0.3;

  return {
    nestingDrivenCDW: Number(nestingDrivenCDW.toFixed(4)),
    vanHoveAmplification: Number(vanHoveAmplification.toFixed(4)),
    anharmonicSuppression: Number(anharmonicSuppression.toFixed(4)),
    cdwOrderParameter: Number(cdwOrderParameter.toFixed(4)),
    cdwScCompetition: Number(cdwScCompetition.toFixed(4)),
    cdwPairingStrength: Number(cdwPairingStrength.toFixed(4)),
    isCDWCandidate,
  };
}

export function computePolaronicPairing(
  formula: string,
  electronic: ElectronicStructure,
  coupling: ElectronPhononCoupling,
): PolaronicPairingResult {
  const lambda = coupling.lambda;
  const metallicity = electronic.metallicity;
  const dosEF = electronic.densityOfStatesAtFermi;

  const polaronCoupling = lambda > 1.5
    ? Math.min(1.0, (lambda - 1.5) * 0.5 + 0.3)
    : lambda > 0.8
    ? Math.min(0.3, (lambda - 0.8) * 0.4)
    : 0;

  const bcsBecCrossover = lambda > 2.0
    ? Math.min(1.0, (lambda - 2.0) * 0.3 + 0.5)
    : lambda > 1.0
    ? Math.min(0.5, (lambda - 1.0) * 0.5)
    : 0;

  const massEnhancement = 1.0 + lambda;

  const bipolaronBindingProxy = lambda > 1.5 && metallicity < 0.5
    ? Math.min(1.0, (lambda - 1.5) * 0.6 * (1.0 - metallicity))
    : 0;

  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  let latticeDistortion = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) {
      const frac = (counts[el] || 1) / totalAtoms;
      if (data.atomicMass > 80) {
        latticeDistortion += frac * 0.4;
      }
    }
  }
  latticeDistortion = Math.min(1.0, latticeDistortion + lambda * 0.15);

  let polaronicPairingStrength = Math.min(1.0,
    polaronCoupling * 0.35 +
    bcsBecCrossover * 0.2 +
    bipolaronBindingProxy * 0.15 +
    latticeDistortion * 0.1 +
    (lambda > 2.0 && metallicity < 0.4 ? 0.2 : 0) +
    (metallicity < 0.3 && lambda > 1.0 ? 0.1 : 0)
  );

  const isBismuthate = elements.includes("Bi") && elements.includes("O") &&
    elements.some(e => ["Ba", "K", "Rb"].includes(e));
  const isTitanate = elements.includes("Ti") && elements.includes("O") &&
    elements.some(e => ["Sr", "Ba", "Ca"].includes(e));

  if (isBismuthate) polaronicPairingStrength = Math.max(polaronicPairingStrength, 0.65);
  if (isTitanate) polaronicPairingStrength = Math.max(polaronicPairingStrength, 0.45);

  const isPolaronicCandidate = lambda > 1.0 && (metallicity < 0.5 || bipolaronBindingProxy > 0.2);

  return {
    polaronCoupling: Number(polaronCoupling.toFixed(4)),
    bcsBecCrossover: Number(bcsBecCrossover.toFixed(4)),
    massEnhancement: Number(massEnhancement.toFixed(4)),
    bipolaronBindingProxy: Number(bipolaronBindingProxy.toFixed(4)),
    latticeDistortion: Number(latticeDistortion.toFixed(4)),
    polaronicPairingStrength: Number(polaronicPairingStrength.toFixed(4)),
    isPolaronicCandidate,
  };
}

export function computePlasmonPairing(
  formula: string,
  electronic: ElectronicStructure,
): PlasmonPairingResult {
  const metallicity = electronic.metallicity;
  const dosEF = electronic.densityOfStatesAtFermi;
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  let totalVE = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
  }
  const electronDensity = totalVE / Math.max(1, totalAtoms);

  const plasmaFrequency = Math.min(1.0, Math.sqrt(electronDensity * metallicity) * 0.3);

  const dimensionalityFactor = electronic.bandFlatness > 0.5
    ? 0.7
    : electronic.bandFlatness > 0.3
    ? 0.85
    : 1.0;

  const metallicScreening = metallicity > 0.7
    ? Math.min(1.0, (metallicity - 0.7) * 3.0 + 0.5)
    : metallicity * 0.7;

  const plasmonCoupling = plasmaFrequency * dimensionalityFactor * (1.0 - metallicScreening * 0.3);

  const collectiveOscillationStrength = Math.min(1.0,
    plasmonCoupling * dosEF * 0.3
  );

  let plasmonPairingStrength = Math.min(1.0,
    plasmonCoupling * 0.3 +
    collectiveOscillationStrength * 0.2 +
    (metallicity > 0.3 && metallicity < 0.7 ? 0.15 : 0) +
    (dosEF > 2.0 ? 0.1 : 0) +
    (electronDensity > 4 ? 0.1 : 0) +
    (dimensionalityFactor < 0.9 ? 0.1 : 0)
  );

  const isSrTiO3Type = elements.includes("Sr") && elements.includes("Ti") && elements.includes("O");
  const isLowCarrier = metallicity > 0.2 && metallicity < 0.5 &&
    elements.some(e => ["Sr", "Ba", "K"].includes(e)) && elements.includes("O");

  if (isSrTiO3Type) plasmonPairingStrength = Math.max(plasmonPairingStrength, 0.60);
  if (isLowCarrier) plasmonPairingStrength = Math.max(plasmonPairingStrength, 0.40);

  const isPlasmonCandidate = plasmonCoupling > 0.2 && metallicity > 0.2 && metallicity < 0.8;

  return {
    plasmaFrequency: Number(plasmaFrequency.toFixed(4)),
    plasmonCoupling: Number(plasmonCoupling.toFixed(4)),
    dimensionalityFactor: Number(dimensionalityFactor.toFixed(4)),
    metallicScreening: Number(metallicScreening.toFixed(4)),
    collectiveOscillationStrength: Number(collectiveOscillationStrength.toFixed(4)),
    plasmonPairingStrength: Number(plasmonPairingStrength.toFixed(4)),
    isPlasmonCandidate,
  };
}

export function computePairingProfile(formula: string): PairingProfile {
  const electronic = computeElectronicStructure(formula, null);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

  const phononResult = computePhononPairing(formula, electronic, phonon, coupling);
  const spinResult = computeSpinPairing(formula, electronic);
  const orbitalResult = computeOrbitalPairing(formula, electronic);
  const excitonicResult = computeExcitonicPairing(formula, electronic);
  const cdwResult = computeCDWPairing(formula, electronic, phonon, coupling);
  const polaronicResult = computePolaronicPairing(formula, electronic, coupling);

  let topoStrength = 0;
  let topoPairingSymmetry = "p-wave";
  try {
    const topoAnalysis = analyzeTopology(formula, electronic);
    topoStrength = Math.min(1, topoAnalysis.majoranaFeasibility * 0.6 + topoAnalysis.z2Score * 0.3 + topoAnalysis.socStrength * 0.1);
    if (topoAnalysis.topologicalClass === "strong-TI" || topoAnalysis.z2Score > 0.7) {
      topoPairingSymmetry = "p+ip (topological)";
    }
  } catch (_) {
    topoStrength = 0;
  }
  const plasmonResult = computePlasmonPairing(formula, electronic);

  const elements = parseFormulaElements(formula);
  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.some(e => ["Fe", "Co"].includes(e)) &&
    elements.some(e => ["As", "P", "Se", "S"].includes(e));
  const isHydride = elements.includes("H") &&
    elements.some(e => isTransitionMetal(e) || isRareEarth(e));
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const hCount = counts["H"] || 0;
  const hRatio = totalAtoms > 0 ? hCount / totalAtoms : 0;
  const isSuperhydride = isHydride && hRatio > 0.6;
  const isNickelate = elements.includes("Ni") && elements.includes("O") && elements.length >= 3;
  const isCDWMaterial = cdwResult.isCDWCandidate && cdwResult.cdwPairingStrength > 0.4;
  const isBismuthate = elements.includes("Bi") && elements.includes("O") &&
    elements.some(e => ["Ba", "K", "Rb"].includes(e));
  const isTitanate = elements.includes("Sr") && elements.includes("Ti") && elements.includes("O");

  let wPhonon = 0.28, wSpin = 0.21, wOrbital = 0.14, wExcitonic = 0.07, wCDW = 0.09, wPolaronic = 0.07, wPlasmon = 0.07, wTopo = 0.07;

  if (isCuprate) {
    wPhonon = 0.08; wSpin = 0.38; wOrbital = 0.17; wExcitonic = 0.05; wCDW = 0.14; wPolaronic = 0.06; wPlasmon = 0.06; wTopo = 0.06;
  } else if (isPnictide) {
    wPhonon = 0.11; wSpin = 0.28; wOrbital = 0.23; wExcitonic = 0.05; wCDW = 0.12; wPolaronic = 0.07; wPlasmon = 0.06; wTopo = 0.08;
  } else if (isSuperhydride) {
    wPhonon = 0.85; wSpin = 0.02; wOrbital = 0.02; wExcitonic = 0.01; wCDW = 0.02; wPolaronic = 0.04; wPlasmon = 0.02; wTopo = 0.02;
  } else if (isHydride) {
    wPhonon = 0.52; wSpin = 0.07; wOrbital = 0.07; wExcitonic = 0.03; wCDW = 0.05; wPolaronic = 0.14; wPlasmon = 0.07; wTopo = 0.05;
  } else if (isNickelate) {
    wPhonon = 0.09; wSpin = 0.33; wOrbital = 0.18; wExcitonic = 0.05; wCDW = 0.11; wPolaronic = 0.09; wPlasmon = 0.07; wTopo = 0.08;
  } else if (isBismuthate) {
    wPhonon = 0.18; wSpin = 0.09; wOrbital = 0.09; wExcitonic = 0.05; wCDW = 0.09; wPolaronic = 0.33; wPlasmon = 0.09; wTopo = 0.08;
  } else if (isTitanate) {
    wPhonon = 0.14; wSpin = 0.07; wOrbital = 0.06; wExcitonic = 0.05; wCDW = 0.05; wPolaronic = 0.18; wPlasmon = 0.38; wTopo = 0.07;
  } else if (isCDWMaterial) {
    wPhonon = 0.18; wSpin = 0.14; wOrbital = 0.09; wExcitonic = 0.05; wCDW = 0.28; wPolaronic = 0.09; wPlasmon = 0.09; wTopo = 0.08;
  } else if (excitonicResult.isExcitonicCandidate) {
    wPhonon = 0.16; wSpin = 0.14; wOrbital = 0.09; wExcitonic = 0.28; wCDW = 0.09; wPolaronic = 0.09; wPlasmon = 0.06; wTopo = 0.09;
  } else if (plasmonResult.isPlasmonCandidate) {
    wPhonon = 0.18; wSpin = 0.14; wOrbital = 0.09; wExcitonic = 0.05; wCDW = 0.09; wPolaronic = 0.09; wPlasmon = 0.28; wTopo = 0.08;
  }

  if (topoStrength > 0.5) {
    const topoBoost = Math.min(0.08, topoStrength * 0.1);
    wTopo += topoBoost;
    wPhonon -= topoBoost * 0.5;
    wSpin -= topoBoost * 0.5;
  }

  const compositePairingStrength =
    wPhonon * phononResult.phononPairingStrength +
    wSpin * spinResult.spinPairingStrength +
    wOrbital * orbitalResult.orbitalPairingStrength +
    wExcitonic * excitonicResult.excitonicPairingStrength +
    wCDW * cdwResult.cdwPairingStrength +
    wPolaronic * polaronicResult.polaronicPairingStrength +
    wPlasmon * plasmonResult.plasmonPairingStrength +
    wTopo * topoStrength;

  const mechanisms = [
    { name: "phonon", strength: phononResult.phononPairingStrength * wPhonon },
    { name: "spin-fluctuation", strength: spinResult.spinPairingStrength * wSpin },
    { name: "orbital-fluctuation", strength: orbitalResult.orbitalPairingStrength * wOrbital },
    { name: "excitonic", strength: excitonicResult.excitonicPairingStrength * wExcitonic },
    { name: "cdw-mediated", strength: cdwResult.cdwPairingStrength * wCDW },
    { name: "polaronic", strength: polaronicResult.polaronicPairingStrength * wPolaronic },
    { name: "plasmon-mediated", strength: plasmonResult.plasmonPairingStrength * wPlasmon },
    { name: "topological", strength: topoStrength * wTopo },
  ];
  mechanisms.sort((a, b) => b.strength - a.strength);

  let pairingSymmetry = "s-wave";
  if (spinResult.spinPairingStrength > phononResult.phononPairingStrength) {
    pairingSymmetry = spinResult.pairingSymmetry;
  } else if (phononResult.phononPairingStrength > 0.3) {
    pairingSymmetry = "s-wave";
    if (phononResult.modeResolved.hasSoftModeInstability) {
      pairingSymmetry = "anisotropic s-wave";
    }
  }
  if (mechanisms[0].name === "cdw-mediated" && cdwResult.cdwPairingStrength > 0.5) {
    pairingSymmetry = "CDW-modulated s-wave";
  }
  if (mechanisms[0].name === "polaronic" && polaronicResult.polaronicPairingStrength > 0.5) {
    pairingSymmetry = polaronicResult.bcsBecCrossover > 0.5 ? "BEC-like s-wave" : "polaronic s-wave";
  }
  if (mechanisms[0].name === "topological" && topoStrength > 0.5) {
    pairingSymmetry = topoPairingSymmetry;
  }

  let estimatedTcFromPairing = phononResult.tcAllenDynes;
  if (mechanisms[0].name === "spin-fluctuation" && spinResult.spinPairingStrength > 0.5) {
    const spinTc = spinResult.spinFluctuationEnergy * spinResult.spinPairingStrength * 0.5;
    estimatedTcFromPairing = Math.max(estimatedTcFromPairing, Math.min(300, spinTc));
  }
  if (mechanisms[0].name === "orbital-fluctuation" && orbitalResult.orbitalPairingStrength > 0.5) {
    const orbTc = orbitalResult.orbitalFluctuation * 20;
    estimatedTcFromPairing = Math.max(estimatedTcFromPairing, Math.min(200, orbTc));
  }
  if (mechanisms[0].name === "cdw-mediated" && cdwResult.cdwPairingStrength > 0.5) {
    const cdwTc = cdwResult.cdwPairingStrength * cdwResult.cdwScCompetition * 30;
    estimatedTcFromPairing = Math.max(estimatedTcFromPairing, Math.min(100, cdwTc));
  }
  if (mechanisms[0].name === "polaronic" && polaronicResult.polaronicPairingStrength > 0.4) {
    const polTc = polaronicResult.polaronicPairingStrength * 40;
    estimatedTcFromPairing = Math.max(estimatedTcFromPairing, Math.min(150, polTc));
  }
  if (mechanisms[0].name === "plasmon-mediated" && plasmonResult.plasmonPairingStrength > 0.4) {
    const plasTc = plasmonResult.plasmonPairingStrength * plasmonResult.collectiveOscillationStrength * 20;
    estimatedTcFromPairing = Math.max(estimatedTcFromPairing, Math.min(50, plasTc));
  }
  if (mechanisms[0].name === "topological" && topoStrength > 0.4) {
    const topoTc = topoStrength * 30;
    estimatedTcFromPairing = Math.max(estimatedTcFromPairing, Math.min(30, topoTc));
  }

  return {
    formula,
    phonon: phononResult,
    spin: spinResult,
    orbital: orbitalResult,
    excitonic: excitonicResult,
    cdw: cdwResult,
    polaronic: polaronicResult,
    plasmon: plasmonResult,
    compositePairingStrength: Number(compositePairingStrength.toFixed(4)),
    mechanismWeights: {
      phonon: Number(wPhonon.toFixed(2)),
      spin: Number(wSpin.toFixed(2)),
      orbital: Number(wOrbital.toFixed(2)),
      excitonic: Number(wExcitonic.toFixed(2)),
      cdw: Number(wCDW.toFixed(2)),
      polaronic: Number(wPolaronic.toFixed(2)),
      plasmon: Number(wPlasmon.toFixed(2)),
      topological: Number(wTopo.toFixed(2)),
    },
    dominantMechanism: mechanisms[0].name,
    secondaryMechanism: mechanisms[1]?.name ?? "none",
    pairingSymmetry,
    estimatedTcFromPairing: Number(estimatedTcFromPairing.toFixed(2)),
  };
}

export function computePairingFeatureVector(formula: string): {
  phononPairingStrength: number;
  spinPairingStrength: number;
  orbitalPairingStrength: number;
  excitonPairingStrength: number;
  cdwPairingStrength: number;
  polaronicPairingStrength: number;
  plasmonPairingStrength: number;
  dominantPairingType: number;
  compositePairing: number;
} {
  const profile = computePairingProfile(formula);

  const typeMap: Record<string, number> = {
    "phonon": 1.0,
    "spin-fluctuation": 2.0,
    "orbital-fluctuation": 3.0,
    "excitonic": 4.0,
    "cdw-mediated": 5.0,
    "polaronic": 6.0,
    "plasmon-mediated": 7.0,
  };

  return {
    phononPairingStrength: profile.phonon.phononPairingStrength,
    spinPairingStrength: profile.spin.spinPairingStrength,
    orbitalPairingStrength: profile.orbital.orbitalPairingStrength,
    excitonPairingStrength: profile.excitonic.excitonicPairingStrength,
    cdwPairingStrength: profile.cdw.cdwPairingStrength,
    polaronicPairingStrength: profile.polaronic.polaronicPairingStrength,
    plasmonPairingStrength: profile.plasmon.plasmonPairingStrength,
    dominantPairingType: typeMap[profile.dominantMechanism] ?? 0,
    compositePairing: profile.compositePairingStrength,
  };
}
