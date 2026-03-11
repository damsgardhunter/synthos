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
import { analyzeTopology, type TopologicalAnalysis } from "./topology-engine";
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
  pairingSuppressed?: boolean;
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
  mechanismWeights: { phonon: number; spin: number; orbital: number; excitonic: number; cdw: number; polaronic: number; plasmon: number; topological?: number };
  dominantMechanism: string;
  secondaryMechanism: string;
  pairingSymmetry: string;
  estimatedTcFromPairing: number;
  topologicalBoostApplied?: boolean;
  tripletOddParityWeight?: number;
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
  const omegaLogK = coupling.omegaLog * 1.4388;
  const muStar = coupling.muStar;

  let tcAllenDynes = 0;
  let pairingSuppressed = false;
  const allenDynesDenom = lambda - muStar * (1 + 0.62 * lambda);
  if (allenDynesDenom > 1e-6) {
    const lambdaBar = 2.46 * (1 + 3.8 * muStar);
    const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

    let f2 = 1.0;
    if (coupling.omegaLog > 0) {
      const omega2Avg = coupling.omega2Avg;
      if (omega2Avg > 0) {
        const sqrtOmega2 = Math.sqrt(omega2Avg);
        const omegaRatio = sqrtOmega2 / coupling.omegaLog;
        const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
        f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
        f2 = Math.max(0.8, f2);
      }
    }

    const exponent = -1.04 * (1 + lambda) / allenDynesDenom;
    tcAllenDynes = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);
    if (!Number.isFinite(tcAllenDynes)) tcAllenDynes = 0;

    const elements = formula.match(/[A-Z][a-z]*/g) || [];
    const isHydride = elements.includes("H") && lambda > 1.5;
    if (isHydride) {
      const lambdaEff = lambda - muStar * (1 + lambda);
      if (lambdaEff > 0) {
        const tcSisso = 0.182 * omegaLogK * Math.pow(lambdaEff, 0.572) *
          Math.pow(1 + 6.5 * muStar * Math.log(Math.max(1.1, lambda)), -0.278);
        if (Number.isFinite(tcSisso) && tcSisso > tcAllenDynes) {
          tcAllenDynes = tcSisso;
        }
      } else {
        pairingSuppressed = true;
        console.warn(`[PairingMech] μ*-suppressed hydride pairing: λ=${lambda.toFixed(3)}, μ*=${muStar.toFixed(3)}, λ_eff=${lambdaEff.toFixed(3)} for ${formula}`);
      }
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

    const normSum = branches.reduce((s, bb) => {
      const af = bb.frequencies.reduce((ss, f) => ss + Math.abs(f), 0) / Math.max(1, bb.frequencies.length);
      return s + (af > 0 ? (1.0 / af) * (bb.isSoft ? 2.0 : 1.0) : 0);
    }, 0);

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
    const elems = formula.match(/[A-Z][a-z]*/g) || [];
    const hasH = elems.includes("H");
    const HEAVY_METALS = ["Pb", "Hg", "Tl", "Bi", "U", "Th", "La", "Ce", "Pr", "Nd"];
    const hasHeavy = elems.some(e => HEAVY_METALS.includes(e));

    let acRatio = 0.4, opRatio = 0.45, hfRatio = 0.15;
    if (hasH) {
      hfRatio = 0.6;
      opRatio = 0.25;
      acRatio = 0.15;
    } else if (hasHeavy) {
      acRatio = 0.55;
      opRatio = 0.35;
      hfRatio = 0.10;
    }

    modeResolved = {
      acousticLambda: lambda * acRatio,
      opticalLambda: lambda * opRatio,
      highFreqLambda: lambda * hfRatio,
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
    ...(pairingSuppressed ? { pairingSuppressed: true } : {}),
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

  let weightedUoverW = 0;
  let totalCorrelatedFrac = 0;
  for (const el of elements) {
    const U = getHubbardU(el);
    if (U === null) continue;
    const W = Math.max(0.5, estimateBandwidthW(el));
    const frac = (counts[el] || 1) / totalAtoms;
    const ratio = (U / W) * Math.sqrt(frac);
    weightedUoverW += ratio * frac;
    totalCorrelatedFrac += frac;
  }
  const avgUoverW = totalCorrelatedFrac > 0 ? weightedUoverW / totalCorrelatedFrac : 0;
  const correlationFactor = Math.max(0.05, Math.min(2.0, avgUoverW || correlation * 0.5));

  const clampedStoner = Math.min(20.0, spin.stonerEnhancement);
  const chiQ = dosEF * nestingFloor * Math.max(1, clampedStoner * 0.5);

  const nestingAmplification = nestingScore > 0.5 ? 1.0 + (nestingScore - 0.5) * 2.0 : 1.0;

  const spinCoupling = dosEF * nestingFloor * correlationFactor;

  let spinPairingStrength = Math.min(1.0,
    spinCoupling * 0.3 +
    (spin.isNearQCP ? 0.3 : 0) +
    (nestingScore > 0.7 ? 0.15 : 0) +
    (correlation > 0.5 ? 0.1 : 0) +
    (clampedStoner > 5 ? 0.1 : 0)
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

  const hCount = counts["H"] || 0;
  const hFraction = hCount / totalAtoms;
  const isSuperhydride = hFraction > 0.7;

  let pairingSymmetry = "s-wave";
  if (isSuperhydride && spinPairingStrength < 0.7) {
    pairingSymmetry = "s-wave";
  } else if (isCuprate) {
    pairingSymmetry = "d-wave (dx2-y2)";
  } else if (isPnictide) {
    pairingSymmetry = "s+/-";
  } else if (isHeavyFermion) {
    pairingSymmetry = "d-wave";
  } else if (spinPairingStrength > 0.5 && nestingScore > 0.5) {
    pairingSymmetry = "d-wave";
  }

  return {
    chiQ: Number(chiQ.toFixed(4)),
    stonerEnhancement: clampedStoner,
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
  tbOverride?: { bands: { tbConfidence: number; nOrbitals: number }; topology: { flatBands: any[] } } | null,
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

  const orbitalDegeneracy = Math.min(5, activeOrbitals.length / Math.max(1, elements.length) + (fFraction > 0.1 ? 1.5 : 0));

  let interOrbitalHopping = 0;
  if (tbOverride) {
    if (tbOverride.bands.tbConfidence > 0.3) {
      const nOrb = tbOverride.bands.nOrbitals;
      const flatBands = tbOverride.topology.flatBands.length;
      interOrbitalHopping = Math.min(1.0, (nOrb / 10) * 0.3 + flatBands * 0.1);
    }
  } else {
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
  }

  if (interOrbitalHopping === 0 && dFraction > 0.3) {
    interOrbitalHopping = dFraction * 0.5;
  }

  const STRONG_FIELD_LIGANDS = ["C", "N", "F"];
  const WEAK_FIELD_LIGANDS = ["O", "S", "Se", "Te", "Cl", "Br", "I"];
  const hasStrongFieldLigand = elements.some(e => STRONG_FIELD_LIGANDS.includes(e));
  const hasWeakFieldLigand = elements.some(e => WEAK_FIELD_LIGANDS.includes(e));

  let hundsCoupling = 0;
  for (const el of elements) {
    if (isTransitionMetal(el)) {
      const data = getElementData(el);
      if (data && data.valenceElectrons >= 4 && data.valenceElectrons <= 7) {
        const frac = (counts[el] || 1) / totalAtoms;
        let hundsBase = 0.3 + frac * 0.5;

        const ve = data.valenceElectrons;
        const isLowSpinCandidate = (ve === 5 || ve === 6 || ve === 7) && hasStrongFieldLigand && !hasWeakFieldLigand;
        if (isLowSpinCandidate) {
          hundsBase *= 0.2;
        } else if (hasStrongFieldLigand && hasWeakFieldLigand) {
          hundsBase *= 0.6;
        }

        hundsCoupling = Math.max(hundsCoupling, hundsBase);
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

  const POLARIZABILITY: Record<string, number> = {
    Te: 5.5, Bi: 7.4, Pb: 6.8, Tl: 7.6, Se: 3.8, Sb: 6.6, Sn: 7.7,
    In: 5.4, I: 5.35, Hg: 5.0, As: 4.3, Ge: 6.07, Ga: 8.12, Po: 6.8,
    Ba: 39.7, Sr: 27.6, La: 31.1, Ce: 29.6, Nd: 31.4, Y: 22.7,
  };
  let avgPolarizability = 0;
  let polCount = 0;
  for (const el of elements) {
    if (POLARIZABILITY[el]) {
      avgPolarizability += POLARIZABILITY[el];
      polCount++;
    }
  }
  avgPolarizability = polCount > 0 ? avgPolarizability / polCount : 0;
  const polarizabilityScreening = Math.min(0.4, avgPolarizability / 20.0);

  let dielectricScreening = metallicity > 0.7
    ? 1.0
    : Math.max(0.1, metallicity * 1.2 + polarizabilityScreening);

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
  const IONIC_POLARON_ELEMENTS: Record<string, number> = {
    O: 0.7, F: 0.65, N: 0.5, Cl: 0.45, Li: 0.55, S: 0.35, Se: 0.25,
  };
  for (const el of elements) {
    const data = getElementData(el);
    if (data) {
      const frac = (counts[el] || 1) / totalAtoms;
      const ionicScore = IONIC_POLARON_ELEMENTS[el];
      if (ionicScore !== undefined) {
        latticeDistortion += frac * ionicScore;
      } else if (data.atomicMass < 30 && (data.paulingElectronegativity ?? 0) > 2.5) {
        latticeDistortion += frac * 0.4;
      } else if (data.atomicMass > 80) {
        latticeDistortion += frac * 0.05;
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

  const effectiveMass = 1.0 + (electronic.bandFlatness > 0 ? electronic.bandFlatness * 4.0 : 0);
  const plasmaFrequency = Math.min(1.0, Math.sqrt(electronDensity * metallicity / effectiveMass) * 0.3);

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

export function computePairingProfile(formula: string, externalTopo?: TopologicalAnalysis): PairingProfile {
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
  let hasNontrivialZ2 = false;
  let hasWeylNodes = false;
  let topoAnalysisUsed: TopologicalAnalysis | null = null;
  try {
    const topoAnalysis = externalTopo ?? analyzeTopology(formula, electronic);
    topoAnalysisUsed = topoAnalysis;
    topoStrength = Math.min(1, topoAnalysis.majoranaFeasibility * 0.6 + topoAnalysis.z2Score * 0.3 + topoAnalysis.socStrength * 0.1);
    hasNontrivialZ2 = topoAnalysis.z2Score > 0.5;
    hasWeylNodes = topoAnalysis.diracNodeProbability > 0.4 ||
      topoAnalysis.topologicalClass === "Weyl-semimetal" ||
      topoAnalysis.indicators.some(ind => ind.toLowerCase().includes("weyl") || ind.toLowerCase().includes("dirac"));
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

  const WEIGHT_SETS: Record<string, number[]> = {
    default:      [0.28, 0.21, 0.14, 0.07, 0.09, 0.07, 0.07, 0.07],
    cuprate:      [0.08, 0.38, 0.17, 0.05, 0.14, 0.06, 0.06, 0.06],
    pnictide:     [0.11, 0.28, 0.23, 0.05, 0.12, 0.07, 0.06, 0.08],
    superhydride: [0.85, 0.02, 0.02, 0.01, 0.02, 0.04, 0.02, 0.02],
    hydride:      [0.52, 0.07, 0.07, 0.03, 0.05, 0.14, 0.07, 0.05],
    nickelate:    [0.09, 0.33, 0.18, 0.05, 0.11, 0.09, 0.07, 0.08],
    bismuthate:   [0.18, 0.09, 0.09, 0.05, 0.09, 0.33, 0.09, 0.08],
    titanate:     [0.14, 0.07, 0.06, 0.05, 0.05, 0.18, 0.38, 0.07],
    cdw:          [0.18, 0.14, 0.09, 0.05, 0.28, 0.09, 0.09, 0.08],
    excitonic:    [0.16, 0.14, 0.09, 0.28, 0.09, 0.09, 0.06, 0.09],
    plasmon:      [0.18, 0.14, 0.09, 0.05, 0.09, 0.09, 0.28, 0.08],
  };

  const matchedSets: number[][] = [];
  if (isCuprate) matchedSets.push(WEIGHT_SETS.cuprate);
  if (isPnictide) matchedSets.push(WEIGHT_SETS.pnictide);
  if (isSuperhydride) matchedSets.push(WEIGHT_SETS.superhydride);
  else if (isHydride) matchedSets.push(WEIGHT_SETS.hydride);
  if (isNickelate) matchedSets.push(WEIGHT_SETS.nickelate);
  if (isBismuthate) matchedSets.push(WEIGHT_SETS.bismuthate);
  if (isTitanate) matchedSets.push(WEIGHT_SETS.titanate);
  if (isCDWMaterial) matchedSets.push(WEIGHT_SETS.cdw);
  if (excitonicResult.isExcitonicCandidate) matchedSets.push(WEIGHT_SETS.excitonic);
  if (plasmonResult.isPlasmonCandidate) matchedSets.push(WEIGHT_SETS.plasmon);

  const blended = matchedSets.length > 0
    ? matchedSets[0].map((_, i) => matchedSets.reduce((s, ws) => s + ws[i], 0) / matchedSets.length)
    : WEIGHT_SETS.default;

  let [wPhonon, wSpin, wOrbital, wExcitonic, wCDW, wPolaronic, wPlasmon, wTopo] = blended;

  let topologicalBoostApplied = false;
  let tripletOddParityWeight = 0;

  if (hasNontrivialZ2 || hasWeylNodes) {
    topologicalBoostApplied = true;

    const z2Strength = topoAnalysisUsed ? topoAnalysisUsed.z2Score : 0;
    const weylStrength = topoAnalysisUsed ? topoAnalysisUsed.diracNodeProbability : 0;
    const socStrength = topoAnalysisUsed ? topoAnalysisUsed.socStrength : 0;
    const topoSignal = Math.max(z2Strength, weylStrength);

    const tripletBoost = Math.min(0.20, topoSignal * 0.25);
    const dampFactor = Math.max(0.3, 1.0 - tripletBoost);

    wPhonon *= dampFactor;
    wOrbital *= dampFactor;
    wCDW *= dampFactor;
    wPolaronic *= dampFactor;
    wPlasmon *= dampFactor;
    wExcitonic *= dampFactor;

    wSpin += tripletBoost * 0.6;
    wTopo += tripletBoost * 0.4;

    const weightSum = wPhonon + wSpin + wOrbital + wExcitonic + wCDW + wPolaronic + wPlasmon + wTopo;
    if (weightSum > 0) {
      wPhonon /= weightSum;
      wSpin /= weightSum;
      wOrbital /= weightSum;
      wExcitonic /= weightSum;
      wCDW /= weightSum;
      wPolaronic /= weightSum;
      wPlasmon /= weightSum;
      wTopo /= weightSum;
    }

    const socGate = Math.min(1.0, socStrength / 0.3);
    tripletOddParityWeight = (wSpin + wTopo) * socGate;

    if (hasNontrivialZ2 && z2Strength > 0.6) {
      topoPairingSymmetry = "p+ip (topological)";
    } else if (hasWeylNodes && weylStrength > 0.5) {
      topoPairingSymmetry = "odd-parity (Weyl-node driven)";
    }
  } else if (topoStrength > 0.5) {
    const topoBoost = Math.min(0.08, topoStrength * 0.1);
    const dampFactor = Math.max(0.5, 1.0 - topoBoost);

    wPhonon *= dampFactor;
    wSpin *= dampFactor;
    wOrbital *= dampFactor;
    wExcitonic *= dampFactor;
    wCDW *= dampFactor;
    wPolaronic *= dampFactor;
    wPlasmon *= dampFactor;

    wTopo += topoBoost;

    const weightSum = wPhonon + wSpin + wOrbital + wExcitonic + wCDW + wPolaronic + wPlasmon + wTopo;
    if (weightSum > 0) {
      wPhonon /= weightSum;
      wSpin /= weightSum;
      wOrbital /= weightSum;
      wExcitonic /= weightSum;
      wCDW /= weightSum;
      wPolaronic /= weightSum;
      wPlasmon /= weightSum;
      wTopo /= weightSum;
    }
  }

  const cdwPhononPenalty = cdwResult.cdwOrderParameter > 0.5
    ? Math.max(0.4, 1.0 - (cdwResult.cdwOrderParameter - 0.5) * 0.8)
    : 1.0;
  const effectivePhononStrength = phononResult.phononPairingStrength * cdwPhononPenalty;

  const compositePairingStrength =
    wPhonon * effectivePhononStrength +
    wSpin * spinResult.spinPairingStrength +
    wOrbital * orbitalResult.orbitalPairingStrength +
    wExcitonic * excitonicResult.excitonicPairingStrength +
    wCDW * cdwResult.cdwPairingStrength +
    wPolaronic * polaronicResult.polaronicPairingStrength +
    wPlasmon * plasmonResult.plasmonPairingStrength +
    wTopo * topoStrength;

  const mechanisms = [
    { name: "phonon", strength: effectivePhononStrength * wPhonon },
    { name: "spin-fluctuation", strength: spinResult.spinPairingStrength * wSpin },
    { name: "orbital-fluctuation", strength: orbitalResult.orbitalPairingStrength * wOrbital },
    { name: "excitonic", strength: excitonicResult.excitonicPairingStrength * wExcitonic },
    { name: "cdw-mediated", strength: cdwResult.cdwPairingStrength * wCDW },
    { name: "polaronic", strength: polaronicResult.polaronicPairingStrength * wPolaronic },
    { name: "plasmon-mediated", strength: plasmonResult.plasmonPairingStrength * wPlasmon },
    { name: "topological", strength: topoStrength * wTopo },
  ];
  mechanisms.sort((a, b) => b.strength - a.strength);

  const dominant = mechanisms[0].name;
  let pairingSymmetry = "s-wave";

  if (dominant === "phonon-mediated" || dominant === "phonon") {
    pairingSymmetry = "s-wave";
    if (phononResult.modeResolved.hasSoftModeInstability) {
      pairingSymmetry = "anisotropic s-wave";
    }
  } else if (dominant === "spin-fluctuation") {
    if (topologicalBoostApplied) {
      pairingSymmetry = "triplet (topology-driven)";
    } else {
      pairingSymmetry = spinResult.pairingSymmetry;
      const fsTopo = electronic.fermiSurfaceTopology?.toLowerCase() ?? "";
      const isCylindrical = fsTopo.includes("cylindrical") || fsTopo.includes("2d");
      if (pairingSymmetry === "s-wave" && isCylindrical) {
        pairingSymmetry = "d-wave";
      } else if (pairingSymmetry === "s-wave" && !isCylindrical) {
        pairingSymmetry = spinResult.nestingAmplification > 1.5 ? "d-wave" : "s+/-";
      }
    }
  } else if (dominant === "orbital-fluctuation") {
    pairingSymmetry = topologicalBoostApplied ? "odd-parity s+/-" : "s+/-";
  } else if (dominant === "topological") {
    pairingSymmetry = topoPairingSymmetry;
  } else if (dominant === "cdw-mediated" && cdwResult.cdwPairingStrength > 0.5) {
    pairingSymmetry = "CDW-modulated s-wave";
  } else if (dominant === "polaronic" && polaronicResult.polaronicPairingStrength > 0.5) {
    pairingSymmetry = polaronicResult.bcsBecCrossover > 0.5 ? "BEC-like s-wave" : "polaronic s-wave";
  } else if (dominant === "excitonic") {
    pairingSymmetry = "d-wave";
  } else if (dominant === "plasmon-mediated") {
    pairingSymmetry = "s-wave";
  }

  const clampedSpinFlucE = Math.min(500, spinResult.spinFluctuationEnergy);

  let estimatedTcFromPairing = phononResult.tcAllenDynes;
  if (mechanisms[0].name === "spin-fluctuation" && spinResult.spinPairingStrength > 0.5) {
    const spinTc = clampedSpinFlucE * spinResult.spinPairingStrength * 0.5;
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

  const rawStrengths = [
    phononResult.phononPairingStrength,
    spinResult.spinPairingStrength,
    orbitalResult.orbitalPairingStrength,
    cdwResult.cdwPairingStrength,
    polaronicResult.polaronicPairingStrength,
    plasmonResult.plasmonPairingStrength,
  ];
  const cooperatingCount = rawStrengths.filter(s => s > 0.4).length;
  if (cooperatingCount >= 2 && compositePairingStrength > 0.4) {
    const cooperationBonus = 1.0 + 0.08 * (cooperatingCount - 1);
    estimatedTcFromPairing *= cooperationBonus;
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
    topologicalBoostApplied,
    tripletOddParityWeight: topologicalBoostApplied ? Number(tripletOddParityWeight.toFixed(3)) : undefined,
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
