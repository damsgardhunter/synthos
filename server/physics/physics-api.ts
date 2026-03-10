import { runFullDFT, isQEAvailable } from "../dft/qe-worker";
import { runXTBEnrichment, isDFTAvailable } from "../dft/qe-dft-engine";
import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  type ElectronicStructure,
  type PhononSpectrum,
  type ElectronPhononCoupling,
} from "../learning/physics-engine";
import { runEliashbergPipeline } from "./eliashberg-pipeline";

export type PhysicsTier = "full-dft" | "xtb" | "surrogate";

export interface RelaxationResult {
  formula: string;
  tier: PhysicsTier;
  converged: boolean;
  totalEnergy: number | null;
  totalEnergyPerAtom: number | null;
  formationEnergy: number | null;
  bandGap: number | null;
  isMetallic: boolean;
  fermiEnergy: number | null;
  wallTimeMs: number;
  error: string | null;
}

export interface PhononResult {
  formula: string;
  tier: PhysicsTier;
  frequencies: number[];
  isStable: boolean;
  imaginaryCount: number;
  lowestFrequency: number;
  highestFrequency: number;
  debyeTemperature: number;
  maxPhononFrequency: number;
  wallTimeMs: number;
  error: string | null;
}

export interface EphResult {
  formula: string;
  pressureGpa: number;
  tier: PhysicsTier;
  lambda: number;
  lambdaUncorrected: number;
  omegaLog: number;
  muStar: number;
  isStrongCoupling: boolean;
  dominantPhononBranch: string;
  bandwidth: number;
  wallTimeMs: number;
  error: string | null;
}

export interface TcResult {
  formula: string;
  pressureGpa: number;
  tier: PhysicsTier;
  tcAllenDynes: number;
  tcEliashberg: number;
  tcBest: number;
  lambda: number;
  omegaLog: number;
  muStar: number;
  gapRatio: number;
  isStrongCoupling: boolean;
  confidence: "low" | "medium" | "high";
  confidenceBand: [number, number];
  wallTimeMs: number;
  error: string | null;
}

const apiStats = {
  relaxationCalls: 0,
  phononCalls: 0,
  ephCalls: 0,
  tcCalls: 0,
  tierBreakdown: { "full-dft": 0, xtb: 0, surrogate: 0 } as Record<PhysicsTier, number>,
};

function determineTier(): PhysicsTier {
  if (isQEAvailable()) return "full-dft";
  if (isDFTAvailable()) return "xtb";
  return "surrogate";
}

export async function runRelaxation(formula: string): Promise<RelaxationResult> {
  apiStats.relaxationCalls++;
  const startTime = Date.now();
  const topTier = determineTier();

  if (topTier === "full-dft") {
    try {
      const dft = await runFullDFT(formula);
      if (dft.scf?.converged) {
        apiStats.tierBreakdown["full-dft"]++;
        return {
          formula,
          tier: "full-dft",
          converged: true,
          totalEnergy: dft.scf.totalEnergy,
          totalEnergyPerAtom: dft.scf.totalEnergyPerAtom,
          formationEnergy: dft.scf.totalEnergyPerAtom,
          bandGap: dft.scf.bandGap,
          isMetallic: dft.scf.isMetallic,
          fermiEnergy: dft.scf.fermiEnergy,
          wallTimeMs: Date.now() - startTime,
          error: null,
        };
      }
    } catch {}
  }

  if (topTier === "full-dft" || topTier === "xtb") {
    try {
      const xtb = await runXTBEnrichment(formula);
      if (xtb && xtb.converged) {
        apiStats.tierBreakdown.xtb++;
        return {
          formula,
          tier: "xtb",
          converged: true,
          totalEnergy: xtb.totalEnergy,
          totalEnergyPerAtom: xtb.totalEnergyPerAtom,
          formationEnergy: xtb.formationEnergyPerAtom,
          bandGap: xtb.bandGap,
          isMetallic: xtb.isMetallic,
          fermiEnergy: xtb.fermiLevel,
          wallTimeMs: Date.now() - startTime,
          error: null,
        };
      }
    } catch {}
  }

  try {
    const elec = computeElectronicStructure(formula);
    apiStats.tierBreakdown.surrogate++;
    return {
      formula,
      tier: "surrogate",
      converged: true,
      totalEnergy: null,
      totalEnergyPerAtom: null,
      formationEnergy: null,
      bandGap: (elec as any).bandGap ?? null,
      isMetallic: elec.metallicity > 0.5,
      fermiEnergy: null,
      wallTimeMs: Date.now() - startTime,
      error: null,
    };
  } catch (e: any) {
    apiStats.tierBreakdown.surrogate++;
    return {
      formula,
      tier: "surrogate",
      converged: false,
      totalEnergy: null,
      totalEnergyPerAtom: null,
      formationEnergy: null,
      bandGap: null,
      isMetallic: false,
      fermiEnergy: null,
      wallTimeMs: Date.now() - startTime,
      error: e.message ?? "Surrogate relaxation failed",
    };
  }
}

export async function computePhonons(formula: string): Promise<PhononResult> {
  apiStats.phononCalls++;
  const startTime = Date.now();
  const topTier = determineTier();

  if (topTier === "full-dft") {
    try {
      const dft = await runFullDFT(formula);
      if (dft.phonon) {
        apiStats.tierBreakdown["full-dft"]++;
        return {
          formula,
          tier: "full-dft",
          frequencies: dft.phonon.frequencies,
          isStable: !dft.phonon.hasImaginary,
          imaginaryCount: dft.phonon.imaginaryCount,
          lowestFrequency: dft.phonon.lowestFrequency,
          highestFrequency: dft.phonon.highestFrequency,
          debyeTemperature: dft.phonon.highestFrequency * 1.44,
          maxPhononFrequency: dft.phonon.highestFrequency,
          wallTimeMs: Date.now() - startTime,
          error: null,
        };
      }
    } catch {}
  }

  if (topTier === "full-dft" || topTier === "xtb") {
    try {
      const xtb = await runXTBEnrichment(formula);
      if (xtb?.phononStability) {
        const freqs = xtb.phononStability.frequencies;
        apiStats.tierBreakdown.xtb++;
        return {
          formula,
          tier: "xtb",
          frequencies: freqs,
          isStable: !xtb.phononStability.hasImaginaryModes,
          imaginaryCount: xtb.phononStability.imaginaryModeCount,
          lowestFrequency: xtb.phononStability.lowestFrequency,
          highestFrequency: freqs.length > 0 ? Math.max(...freqs) : 0,
          debyeTemperature: freqs.length > 0 ? Math.max(...freqs) * 1.44 : 0,
          maxPhononFrequency: freqs.length > 0 ? Math.max(...freqs) : 0,
          wallTimeMs: Date.now() - startTime,
          error: null,
        };
      }
    } catch {}
  }

  try {
    const elec = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, elec);
    apiStats.tierBreakdown.surrogate++;
    return {
      formula,
      tier: "surrogate",
      frequencies: [],
      isStable: !phonon.hasImaginaryModes,
      imaginaryCount: phonon.hasImaginaryModes ? 1 : 0,
      lowestFrequency: phonon.softModePresent ? -phonon.softModeScore * 10 : phonon.logAverageFrequency * 0.1,
      highestFrequency: phonon.maxPhononFrequency,
      debyeTemperature: phonon.debyeTemperature,
      maxPhononFrequency: phonon.maxPhononFrequency,
      wallTimeMs: Date.now() - startTime,
      error: null,
    };
  } catch (e: any) {
    apiStats.tierBreakdown.surrogate++;
    return {
      formula,
      tier: "surrogate",
      frequencies: [],
      isStable: false,
      imaginaryCount: 0,
      lowestFrequency: 0,
      highestFrequency: 0,
      debyeTemperature: 0,
      maxPhononFrequency: 0,
      wallTimeMs: Date.now() - startTime,
      error: e.message ?? "Phonon computation failed",
    };
  }
}

export async function computeEph(formula: string, pressureGpa: number = 0): Promise<EphResult> {
  apiStats.ephCalls++;
  const startTime = Date.now();
  const topTier = determineTier();

  let tier: PhysicsTier = "surrogate";
  let electronicOverride: ElectronicStructure | undefined;
  let phononOverride: PhononSpectrum | undefined;
  let couplingOverride: ElectronPhononCoupling | undefined;

  if (topTier === "full-dft") {
    try {
      const dft = await runFullDFT(formula);
      if (dft.scf?.converged) {
        tier = "full-dft";
        electronicOverride = computeElectronicStructure(formula);
        if (dft.scf.fermiEnergy !== null) {
          electronicOverride.densityOfStatesAtFermi = dft.scf.isMetallic ? 2.5 : 0.5;
        }
        if (dft.scf.bandGap !== null) {
          (electronicOverride as any).bandGap = dft.scf.bandGap;
        }
        electronicOverride.metallicity = dft.scf.isMetallic ? 0.95 : 0.3;
        if (dft.phonon && dft.phonon.frequencies.length > 0) {
          phononOverride = computePhononSpectrum(formula, electronicOverride);
          (phononOverride as any).frequencies = dft.phonon.frequencies;
          phononOverride.hasImaginaryModes = dft.phonon.hasImaginary;
          phononOverride.debyeTemperature = dft.phonon.highestFrequency * 1.44;
          couplingOverride = computeElectronPhononCoupling(electronicOverride, phononOverride, formula, pressureGpa);
        }
      }
    } catch {}
  }

  if (tier === "surrogate" && (topTier === "full-dft" || topTier === "xtb")) {
    try {
      const xtb = await runXTBEnrichment(formula);
      if (xtb && xtb.converged) {
        tier = "xtb";
        electronicOverride = computeElectronicStructure(formula);
        electronicOverride.metallicity = xtb.isMetallic ? 0.9 : 0.3;
        (electronicOverride as any).bandGap = xtb.bandGap;
      }
    } catch {}
  }

  try {
    const elec = electronicOverride ?? computeElectronicStructure(formula);
    const phonon = phononOverride ?? computePhononSpectrum(formula, elec);
    const coupling = couplingOverride ?? computeElectronPhononCoupling(elec, phonon, formula, pressureGpa);

    apiStats.tierBreakdown[tier]++;
    return {
      formula,
      pressureGpa,
      tier,
      lambda: coupling.lambda,
      lambdaUncorrected: coupling.lambdaUncorrected,
      omegaLog: coupling.omegaLog,
      muStar: coupling.muStar,
      isStrongCoupling: coupling.isStrongCoupling,
      dominantPhononBranch: coupling.dominantPhononBranch,
      bandwidth: coupling.bandwidth,
      wallTimeMs: Date.now() - startTime,
      error: null,
    };
  } catch (e: any) {
    apiStats.tierBreakdown.surrogate++;
    return {
      formula,
      pressureGpa,
      tier: "surrogate",
      lambda: 0,
      lambdaUncorrected: 0,
      omegaLog: 0,
      muStar: 0.10,
      isStrongCoupling: false,
      dominantPhononBranch: "unknown",
      bandwidth: 0,
      wallTimeMs: Date.now() - startTime,
      error: e.message ?? "Electron-phonon coupling computation failed",
    };
  }
}

export async function computeTc(formula: string, pressureGpa: number = 0): Promise<TcResult> {
  apiStats.tcCalls++;
  const startTime = Date.now();
  const topTier = determineTier();

  let tier: PhysicsTier = "surrogate";
  let electronicOverride: ElectronicStructure | undefined;
  let phononOverride: PhononSpectrum | undefined;
  let couplingOverride: ElectronPhononCoupling | undefined;

  if (topTier === "full-dft") {
    try {
      const dft = await runFullDFT(formula);
      if (dft.scf?.converged) {
        tier = "full-dft";
        electronicOverride = computeElectronicStructure(formula);
        if (dft.scf.fermiEnergy !== null) {
          electronicOverride.densityOfStatesAtFermi = dft.scf.isMetallic ? 2.5 : 0.5;
        }
        if (dft.scf.bandGap !== null) {
          (electronicOverride as any).bandGap = dft.scf.bandGap;
        }
        electronicOverride.metallicity = dft.scf.isMetallic ? 0.95 : 0.3;
        if (dft.phonon && dft.phonon.frequencies.length > 0) {
          phononOverride = computePhononSpectrum(formula, electronicOverride);
          (phononOverride as any).frequencies = dft.phonon.frequencies;
          phononOverride.hasImaginaryModes = dft.phonon.hasImaginary;
          phononOverride.debyeTemperature = dft.phonon.highestFrequency * 1.44;
          couplingOverride = computeElectronPhononCoupling(electronicOverride, phononOverride, formula, pressureGpa);
        }
      }
    } catch {}
  }

  if (tier === "surrogate" && (topTier === "full-dft" || topTier === "xtb")) {
    try {
      const xtb = await runXTBEnrichment(formula);
      if (xtb && xtb.converged) {
        tier = "xtb";
        electronicOverride = computeElectronicStructure(formula);
        electronicOverride.metallicity = xtb.isMetallic ? 0.9 : 0.3;
        (electronicOverride as any).bandGap = xtb.bandGap;
      }
    } catch {}
  }

  try {
    const eliashberg = runEliashbergPipeline(
      formula,
      pressureGpa,
      electronicOverride,
      phononOverride,
      couplingOverride
    );

    apiStats.tierBreakdown[tier]++;
    return {
      formula,
      pressureGpa,
      tier,
      tcAllenDynes: eliashberg.tcAllenDynes.tc,
      tcEliashberg: eliashberg.tcEliashbergGap.tc,
      tcBest: eliashberg.tcBest,
      lambda: eliashberg.lambda,
      omegaLog: eliashberg.omegaLog,
      muStar: eliashberg.muStar,
      gapRatio: eliashberg.gapRatio,
      isStrongCoupling: eliashberg.isStrongCoupling,
      confidence: eliashberg.confidence,
      confidenceBand: eliashberg.confidenceBand,
      wallTimeMs: Date.now() - startTime,
      error: null,
    };
  } catch (e: any) {
    apiStats.tierBreakdown.surrogate++;
    return {
      formula,
      pressureGpa,
      tier: "surrogate",
      tcAllenDynes: 0,
      tcEliashberg: 0,
      tcBest: 0,
      lambda: 0,
      omegaLog: 0,
      muStar: 0.10,
      gapRatio: 3.53,
      isStrongCoupling: false,
      confidence: "low",
      confidenceBand: [0, 0],
      wallTimeMs: Date.now() - startTime,
      error: e.message ?? "Tc computation failed",
    };
  }
}

export function getPhysicsApiStats() {
  return {
    relaxationCalls: apiStats.relaxationCalls,
    phononCalls: apiStats.phononCalls,
    ephCalls: apiStats.ephCalls,
    tcCalls: apiStats.tcCalls,
    totalCalls: apiStats.relaxationCalls + apiStats.phononCalls + apiStats.ephCalls + apiStats.tcCalls,
    tierBreakdown: { ...apiStats.tierBreakdown },
  };
}
