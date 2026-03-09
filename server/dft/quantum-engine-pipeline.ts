import { runFullDFT, isQEAvailable, type QEFullResult } from "./qe-worker";
import { runXTBEnrichment } from "./qe-dft-engine";
import { runEliashbergPipeline, type EliashbergPipelineResult } from "../physics/eliashberg-pipeline";
import { computeElectronicStructure, computePhononSpectrum, computeElectronPhononCoupling } from "../learning/physics-engine";

export interface QuantumEngineDatasetEntry {
  material: string;
  pressure: number;
  lambda: number;
  omegaLog: number;
  tc: number;
  dosAtEF: number;
  phononSpectrum: number[];
  alpha2FSummary: {
    peakFrequency: number;
    peakHeight: number;
    nBins: number;
    lambdaByRange: Record<string, number>;
  };
  formationEnergy: number | null;
  bandGap: number | null;
  isMetallic: boolean;
  isPhononStable: boolean;
  scfConverged: boolean;
  gapRatio: number;
  muStar: number;
  omega2: number;
  isStrongCoupling: boolean;
  isotopeAlpha: number;
  tcAllenDynes: number;
  tcEliashberg: number;
  confidence: "low" | "medium" | "high";
  tier: "full-dft" | "xtb" | "surrogate";
  wallTimeMs: number;
  timestamp: number;
}

export interface QuantumEngineResult {
  entry: QuantumEngineDatasetEntry;
  eliashberg: EliashbergPipelineResult | null;
  dftResult: QEFullResult | null;
  steps: QuantumEngineStep[];
}

export interface QuantumEngineStep {
  name: string;
  status: "success" | "failed" | "skipped";
  wallTimeMs: number;
  detail: string;
}

const datasetStore: QuantumEngineDatasetEntry[] = [];
const MAX_DATASET_SIZE = 2000;

const pipelineStats = {
  totalRuns: 0,
  fullDftRuns: 0,
  xtbRuns: 0,
  surrogateRuns: 0,
  successCount: 0,
  failCount: 0,
  lambdaSum: 0,
  tcSum: 0,
  dosSum: 0,
  avgWallTimeMs: 0,
  wallTimeTotalMs: 0,
  bestTc: 0,
  bestTcMaterial: "",
  strongCouplingCount: 0,
  highTcCount: 0,
};

export async function runQuantumEnginePipeline(
  formula: string,
  pressureGpa: number = 0
): Promise<QuantumEngineResult> {
  const startTime = Date.now();
  const steps: QuantumEngineStep[] = [];

  let dftResult: QEFullResult | null = null;
  let eliashbergResult: EliashbergPipelineResult | null = null;

  let scfConverged = false;
  let phononStable = false;
  let isMetallic = false;
  let fermiDos = 0;
  let formationEnergy: number | null = null;
  let bandGap: number | null = null;
  let phononFreqs: number[] = [];

  let tier: "full-dft" | "xtb" | "surrogate" = "surrogate";

  const qeAvailable = isQEAvailable();

  if (qeAvailable) {
    const dftStart = Date.now();
    try {
      dftResult = await runFullDFT(formula);
      const dftTime = Date.now() - dftStart;

      if (dftResult.scf?.converged) {
        scfConverged = true;
        isMetallic = dftResult.scf.isMetallic;
        formationEnergy = dftResult.scf.totalEnergyPerAtom;
        bandGap = dftResult.scf.bandGap;
        tier = "full-dft";

        if (dftResult.scf.fermiEnergy !== null) {
          try {
            const elec = computeElectronicStructure(formula);
            fermiDos = elec.densityOfStatesAtFermi;
          } catch {
            fermiDos = isMetallic ? 2.5 : 0.5;
          }
        }

        steps.push({
          name: "4.1 DFT Structure Optimization",
          status: "success",
          wallTimeMs: dftTime,
          detail: `SCF converged: E=${dftResult.scf.totalEnergy.toFixed(4)} eV, Ef=${dftResult.scf.fermiEnergy?.toFixed(3) ?? "N/A"} eV, metallic=${isMetallic}`,
        });

        if (dftResult.bandStructure) {
          steps.push({
            name: "4.1b Band Structure",
            status: dftResult.bandStructure.converged ? "success" : "failed",
            wallTimeMs: dftResult.bandStructure.wallTimeSeconds * 1000,
            detail: `${dftResult.bandStructure.nBands} bands, flat=${dftResult.bandStructure.flatBandScore.toFixed(3)}, crossings=${dftResult.bandStructure.bandCrossings.length}`,
          });
        }
      } else {
        steps.push({
          name: "4.1 DFT Structure Optimization",
          status: "failed",
          wallTimeMs: dftTime,
          detail: `SCF failed: ${dftResult.error ?? dftResult.scf?.error ?? "unknown"}`.slice(0, 200),
        });
      }

      if (dftResult.phonon) {
        phononStable = !dftResult.phonon.hasImaginary;
        phononFreqs = dftResult.phonon.frequencies;
        steps.push({
          name: "4.2 Phonon Calculation",
          status: phononStable ? "success" : "failed",
          wallTimeMs: dftResult.phonon.wallTimeSeconds * 1000,
          detail: `${phononFreqs.length} modes, lowest=${dftResult.phonon.lowestFrequency.toFixed(1)} cm-1, imaginary=${dftResult.phonon.imaginaryCount}`,
        });
      } else if (scfConverged) {
        steps.push({
          name: "4.2 Phonon Calculation",
          status: "skipped",
          wallTimeMs: 0,
          detail: "No phonon result available from DFT",
        });
      }
    } catch (e: any) {
      steps.push({
        name: "4.1 DFT Structure Optimization",
        status: "failed",
        wallTimeMs: Date.now() - dftStart,
        detail: `Exception: ${e.message?.slice(0, 200) ?? "unknown"}`,
      });
    }
  }

  if (!scfConverged) {
    const xtbStart = Date.now();
    try {
      const xtbResult = await runXTBEnrichment(formula);
      const xtbTime = Date.now() - xtbStart;

      if (xtbResult) {
        tier = "xtb";
        formationEnergy = xtbResult.formationEnergy;
        bandGap = xtbResult.bandGap;
        isMetallic = (bandGap ?? 999) < 0.1;
        phononStable = xtbResult.phononStable;

        try {
          const elec = computeElectronicStructure(formula);
          fermiDos = elec.densityOfStatesAtFermi;
        } catch {
          fermiDos = isMetallic ? 2.0 : 0.5;
        }

        if (xtbResult.phononFrequencies && xtbResult.phononFrequencies.length > 0) {
          phononFreqs = xtbResult.phononFrequencies;
        }

        steps.push({
          name: "4.1 xTB Optimization (fallback)",
          status: "success",
          wallTimeMs: xtbTime,
          detail: `formE=${formationEnergy?.toFixed(3)}, gap=${bandGap?.toFixed(3)}, phonon_stable=${phononStable}`,
        });
      } else {
        steps.push({
          name: "4.1 xTB Optimization (fallback)",
          status: "failed",
          wallTimeMs: xtbTime,
          detail: "xTB enrichment returned null",
        });
      }
    } catch (e: any) {
      steps.push({
        name: "4.1 xTB Optimization (fallback)",
        status: "failed",
        wallTimeMs: Date.now() - xtbStart,
        detail: `xTB exception: ${e.message?.slice(0, 200) ?? "unknown"}`,
      });
    }
  }

  if (tier === "surrogate") {
    const surrogateStart = Date.now();
    try {
      const elec = computeElectronicStructure(formula);
      fermiDos = elec.densityOfStatesAtFermi;
      isMetallic = elec.metallicity > 0.5;
      bandGap = elec.bandGap;

      const phonon = computePhononSpectrum(formula);
      phononStable = !phonon.hasImaginaryModes;
      phononFreqs = phonon.frequencies;

      steps.push({
        name: "4.1 Surrogate Electronic Structure (fallback)",
        status: "success",
        wallTimeMs: Date.now() - surrogateStart,
        detail: `DOS(EF)=${fermiDos.toFixed(2)}, metallic=${isMetallic}, gap=${bandGap.toFixed(3)}, ${phononFreqs.length} phonon modes`,
      });
    } catch (e: any) {
      steps.push({
        name: "4.1 Surrogate Electronic Structure (fallback)",
        status: "failed",
        wallTimeMs: Date.now() - surrogateStart,
        detail: `Surrogate failed: ${e.message?.slice(0, 200) ?? "unknown"}`,
      });
    }
  }

  const eliashbergStart = Date.now();
  try {
    let electronicOverride;
    let phononOverride;
    let couplingOverride;

    if (tier === "full-dft" && dftResult?.scf?.converged) {
      electronicOverride = computeElectronicStructure(formula);
      if (dftResult.scf.fermiEnergy !== null) {
        electronicOverride.densityOfStatesAtFermi = fermiDos;
      }
      if (dftResult.scf.bandGap !== null) {
        electronicOverride.bandGap = dftResult.scf.bandGap;
      }
      electronicOverride.metallicity = dftResult.scf.isMetallic ? 0.95 : 0.3;

      if (phononFreqs.length > 0) {
        phononOverride = computePhononSpectrum(formula);
        phononOverride.frequencies = phononFreqs;
        phononOverride.hasImaginaryModes = !phononStable;
        if (phononFreqs.length > 0) {
          phononOverride.debyeTemperature = phononFreqs[phononFreqs.length - 1] * 1.44;
        }
      }

      if (phononFreqs.length > 0) {
        couplingOverride = computeElectronPhononCoupling(formula);
      }
    }

    eliashbergResult = runEliashbergPipeline(
      formula,
      pressureGpa,
      electronicOverride,
      phononOverride,
      couplingOverride
    );

    steps.push({
      name: "4.3 Electron-Phonon Coupling (Eliashberg)",
      status: "success",
      wallTimeMs: Date.now() - eliashbergStart,
      detail: `lambda=${eliashbergResult.lambda.toFixed(3)}, omegaLog=${eliashbergResult.omegaLog.toFixed(1)} cm-1, strong=${eliashbergResult.isStrongCoupling}`,
    });

    steps.push({
      name: "4.4 Tc Computation (Allen-Dynes)",
      status: "success",
      wallTimeMs: eliashbergResult.wallTimeMs,
      detail: `Tc(AD)=${eliashbergResult.tcAllenDynes.tc.toFixed(1)}K, Tc(Eliashberg)=${eliashbergResult.tcEliashbergGap.tc.toFixed(1)}K, Tc(best)=${eliashbergResult.tcBest.toFixed(1)}K, regime=${eliashbergResult.tcAllenDynes.regime}`,
    });
  } catch (e: any) {
    steps.push({
      name: "4.3 Electron-Phonon Coupling",
      status: "failed",
      wallTimeMs: Date.now() - eliashbergStart,
      detail: `Eliashberg pipeline failed: ${e.message?.slice(0, 200) ?? "unknown"}`,
    });
  }

  const totalWallTime = Date.now() - startTime;

  const lambda = eliashbergResult?.lambda ?? 0;
  const omegaLog = eliashbergResult?.omegaLog ?? 0;
  const tc = eliashbergResult?.tcBest ?? 0;

  let alpha2FSummary = {
    peakFrequency: 0,
    peakHeight: 0,
    nBins: 0,
    lambdaByRange: {} as Record<string, number>,
  };

  if (eliashbergResult) {
    const a2f = eliashbergResult.alpha2F;
    let peakIdx = 0;
    let peakVal = 0;
    for (let i = 0; i < a2f.alpha2F.length; i++) {
      if (a2f.alpha2F[i] > peakVal) {
        peakVal = a2f.alpha2F[i];
        peakIdx = i;
      }
    }
    alpha2FSummary = {
      peakFrequency: a2f.frequencies[peakIdx] ?? 0,
      peakHeight: peakVal,
      nBins: a2f.frequencies.length,
      lambdaByRange: { ...eliashbergResult.modeResolved },
    };
  }

  const entry: QuantumEngineDatasetEntry = {
    material: formula,
    pressure: pressureGpa,
    lambda,
    omegaLog,
    tc,
    dosAtEF: fermiDos,
    phononSpectrum: phononFreqs.slice(0, 50),
    alpha2FSummary,
    formationEnergy,
    bandGap,
    isMetallic,
    isPhononStable: phononStable,
    scfConverged,
    gapRatio: eliashbergResult?.gapRatio ?? 3.53,
    muStar: eliashbergResult?.muStar ?? 0.10,
    omega2: eliashbergResult?.omega2 ?? 0,
    isStrongCoupling: eliashbergResult?.isStrongCoupling ?? false,
    isotopeAlpha: eliashbergResult?.isotopeEffect.alpha ?? 0.5,
    tcAllenDynes: eliashbergResult?.tcAllenDynes.tc ?? 0,
    tcEliashberg: eliashbergResult?.tcEliashbergGap.tc ?? 0,
    confidence: eliashbergResult?.confidence ?? "low",
    tier,
    wallTimeMs: totalWallTime,
    timestamp: Date.now(),
  };

  if (datasetStore.length >= MAX_DATASET_SIZE) {
    datasetStore.splice(0, Math.floor(MAX_DATASET_SIZE * 0.1));
  }
  datasetStore.push(entry);

  pipelineStats.totalRuns++;
  if (tier === "full-dft") pipelineStats.fullDftRuns++;
  else if (tier === "xtb") pipelineStats.xtbRuns++;
  else pipelineStats.surrogateRuns++;

  if (tc > 0) {
    pipelineStats.successCount++;
    pipelineStats.lambdaSum += lambda;
    pipelineStats.tcSum += tc;
    pipelineStats.dosSum += fermiDos;
    if (eliashbergResult?.isStrongCoupling) pipelineStats.strongCouplingCount++;
    if (tc > 100) pipelineStats.highTcCount++;
    if (tc > pipelineStats.bestTc) {
      pipelineStats.bestTc = tc;
      pipelineStats.bestTcMaterial = formula;
    }
  } else {
    pipelineStats.failCount++;
  }
  pipelineStats.wallTimeTotalMs += totalWallTime;
  pipelineStats.avgWallTimeMs = pipelineStats.wallTimeTotalMs / pipelineStats.totalRuns;

  return { entry, eliashberg: eliashbergResult, dftResult, steps };
}

export function getQuantumEngineStats() {
  const n = pipelineStats.successCount || 1;
  return {
    totalRuns: pipelineStats.totalRuns,
    successCount: pipelineStats.successCount,
    failCount: pipelineStats.failCount,
    tierBreakdown: {
      fullDft: pipelineStats.fullDftRuns,
      xtb: pipelineStats.xtbRuns,
      surrogate: pipelineStats.surrogateRuns,
    },
    avgLambda: Number((pipelineStats.lambdaSum / n).toFixed(4)),
    avgTc: Number((pipelineStats.tcSum / n).toFixed(2)),
    avgDosAtEF: Number((pipelineStats.dosSum / n).toFixed(3)),
    avgWallTimeMs: Math.round(pipelineStats.avgWallTimeMs),
    bestTc: pipelineStats.bestTc,
    bestTcMaterial: pipelineStats.bestTcMaterial,
    strongCouplingCount: pipelineStats.strongCouplingCount,
    highTcCount: pipelineStats.highTcCount,
    datasetSize: datasetStore.length,
    datasetMaxSize: MAX_DATASET_SIZE,
  };
}

export function getQuantumEngineDataset(): QuantumEngineDatasetEntry[] {
  return [...datasetStore];
}

export function getRecentQuantumEngineResults(limit: number = 20): QuantumEngineDatasetEntry[] {
  return datasetStore.slice(-limit).reverse();
}

export function addExternalDatasetEntry(entry: QuantumEngineDatasetEntry): void {
  if (datasetStore.length >= MAX_DATASET_SIZE) {
    datasetStore.splice(0, Math.floor(MAX_DATASET_SIZE * 0.1));
  }
  datasetStore.push(entry);
}
