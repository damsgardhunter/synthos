import { runFullDFT, isQEAvailable, type QEFullResult } from "./qe-worker";
import { runXTBEnrichment } from "./qe-dft-engine";
import { runEliashbergPipeline, type EliashbergPipelineResult } from "../physics/eliashberg-pipeline";
import { computeElectronicStructure, computePhononSpectrum, computeElectronPhononCoupling, type ElectronicStructure } from "../learning/physics-engine";
import { recordPhysicsResult } from "../learning/physics-results-store";
import { predictLambda, recordLambdaValidation } from "../learning/lambda-regressor";
import { recordStructureFailure } from "../crystal/structure-failure-db";
import { recordRelaxation } from "../crystal/relaxation-tracker";
import { analyzeDistortion, recordDistortionAnalysis } from "../crystal/distortion-detector";
import { getEntryByFormula } from "../crystal/crystal-structure-dataset";
import { dosPrefilter, predictDOS, type DOSSurrogateResult } from "../physics/dos-surrogate";
import type { DFTBandStructureResult } from "./band-structure-calculator";
import { db } from "../db";
import { quantumEngineDataset } from "@shared/schema";
import { desc, count } from "drizzle-orm";

function estimateDOSFromBands(bandResult: DFTBandStructureResult, fermiEnergy: number): number {
  const smearingWidth = 0.15;
  let dos = 0;
  let totalWeight = 0;
  for (const kPt of bandResult.eigenvalues) {
    const kWeight = 1.0;
    for (const energy of kPt.energies) {
      const x = (energy - fermiEnergy) / smearingWidth;
      dos += kWeight * Math.exp(-0.5 * x * x) / (smearingWidth * Math.sqrt(2 * Math.PI));
      totalWeight += kWeight;
    }
  }
  if (totalWeight > 0) {
    dos *= bandResult.nBands / totalWeight;
  }
  return Math.max(0.1, dos);
}

function metallicityFromDOS(dosAtFermi: number, isMetallic: boolean): number {
  if (!isMetallic) return Math.min(0.3, dosAtFermi * 0.1);
  return Math.min(1.0, 0.5 + 0.5 * Math.tanh((dosAtFermi - 1.0) / 2.0));
}

function computeOmegaLog(freqs: number[]): number {
  const positive = freqs.filter(f => f > 1.0);
  if (positive.length === 0) return 0;
  let sumLog = 0;
  for (const f of positive) {
    sumLog += Math.log(f);
  }
  return Math.exp(sumLog / positive.length);
}

function computeDebyeTemperature(freqs: number[]): number {
  const positive = freqs.filter(f => f > 1.0);
  if (positive.length === 0) return 0;
  let sumSq = 0;
  for (const f of positive) {
    sumSq += f * f;
  }
  const omega2 = Math.sqrt(sumSq / positive.length);
  return omega2 * 1.44;
}

function buildDFTElectronicOverride(
  formula: string,
  dftResult: QEFullResult,
  dftDos: number,
  cachedSurrogate?: ElectronicStructure | null,
): ElectronicStructure {
  const surrogate = cachedSurrogate ?? computeElectronicStructure(formula);
  return {
    ...surrogate,
    densityOfStatesAtFermi: dftDos,
    metallicity: metallicityFromDOS(dftDos, dftResult.scf!.isMetallic),
    bandFlatness: dftResult.bandStructure?.flatBandScore ?? surrogate.bandFlatness,
    vanHoveProximity: dftResult.bandStructure?.vanHoveSingularities?.length
      ? Math.min(1.0, dftResult.bandStructure.vanHoveSingularities.length * 0.3)
      : surrogate.vanHoveProximity,
  };
}

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
  dosPrefilter?: {
    pass: boolean;
    vhsScore: number;
    scFavorability: number;
    flatBandIndicator: number;
    nestingScore: number;
    vhsCount: number;
  };
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

const MAX_DATASET_SIZE = 2000;
let datasetCache: QuantumEngineDatasetEntry[] = [];
let cacheLoadPromise: Promise<void> | null = null;

function entryToRow(entry: QuantumEngineDatasetEntry) {
  return {
    material: entry.material,
    pressure: entry.pressure,
    lambda: entry.lambda,
    omegaLog: entry.omegaLog,
    tc: entry.tc,
    dosAtEF: entry.dosAtEF,
    phononSpectrum: entry.phononSpectrum as any,
    alpha2FSummary: entry.alpha2FSummary as any,
    formationEnergy: entry.formationEnergy,
    bandGap: entry.bandGap,
    isMetallic: entry.isMetallic,
    isPhononStable: entry.isPhononStable,
    scfConverged: entry.scfConverged,
    gapRatio: entry.gapRatio,
    muStar: entry.muStar,
    omega2: entry.omega2,
    isStrongCoupling: entry.isStrongCoupling,
    isotopeAlpha: entry.isotopeAlpha,
    tcAllenDynes: entry.tcAllenDynes,
    tcEliashberg: entry.tcEliashberg,
    confidence: entry.confidence,
    tier: entry.tier,
    wallTimeMs: entry.wallTimeMs,
    dosPrefilter: entry.dosPrefilter as any ?? null,
  };
}

function rowToEntry(row: any): QuantumEngineDatasetEntry {
  return {
    material: row.material,
    pressure: row.pressure,
    lambda: row.lambda,
    omegaLog: row.omegaLog ?? row.omega_log ?? 0,
    tc: row.tc,
    dosAtEF: row.dosAtEF ?? row.dos_at_ef ?? 0,
    phononSpectrum: (row.phononSpectrum ?? row.phonon_spectrum ?? []) as number[],
    alpha2FSummary: (row.alpha2FSummary ?? row.alpha2f_summary ?? { peakFrequency: 0, peakHeight: 0, nBins: 0, lambdaByRange: {} }) as any,
    formationEnergy: row.formationEnergy ?? row.formation_energy ?? null,
    bandGap: row.bandGap ?? row.band_gap ?? null,
    isMetallic: row.isMetallic ?? row.is_metallic ?? false,
    isPhononStable: row.isPhononStable ?? row.is_phonon_stable ?? false,
    scfConverged: row.scfConverged ?? row.scf_converged ?? false,
    gapRatio: row.gapRatio ?? row.gap_ratio ?? 3.53,
    muStar: row.muStar ?? row.mu_star ?? 0.1,
    omega2: row.omega2 ?? 0,
    isStrongCoupling: row.isStrongCoupling ?? row.is_strong_coupling ?? false,
    isotopeAlpha: row.isotopeAlpha ?? row.isotope_alpha ?? 0.5,
    tcAllenDynes: row.tcAllenDynes ?? row.tc_allen_dynes ?? 0,
    tcEliashberg: row.tcEliashberg ?? row.tc_eliashberg ?? 0,
    confidence: row.confidence ?? "low",
    tier: row.tier ?? "surrogate",
    wallTimeMs: row.wallTimeMs ?? row.wall_time_ms ?? 0,
    timestamp: row.createdAt ? new Date(row.createdAt).getTime() : (row.timestamp ?? Date.now()),
    dosPrefilter: (row.dosPrefilter ?? row.dos_prefilter ?? undefined) as any,
  };
}

function ensureCacheLoaded(): Promise<void> {
  if (cacheLoadPromise) return cacheLoadPromise;
  cacheLoadPromise = (async () => {
    try {
      const rows = await db.select().from(quantumEngineDataset)
        .orderBy(desc(quantumEngineDataset.createdAt))
        .limit(MAX_DATASET_SIZE);
      const dbEntries = rows.reverse().map(rowToEntry);
      const existingTimestamps = new Set(datasetCache.map(e => `${e.material}_${e.timestamp}`));
      for (const entry of dbEntries) {
        if (!existingTimestamps.has(`${entry.material}_${entry.timestamp}`)) {
          datasetCache.unshift(entry);
        }
      }
      if (datasetCache.length > MAX_DATASET_SIZE) {
        datasetCache = datasetCache.slice(-MAX_DATASET_SIZE);
      }
      console.log(`[QE-Dataset] Loaded ${datasetCache.length} entries from database`);
    } catch (err) {
      console.error(`[QE-Dataset] Failed to load from DB, will retry:`, err);
      cacheLoadPromise = null;
      throw err;
    }
  })();
  return cacheLoadPromise;
}

async function appendToDataset(entry: QuantumEngineDatasetEntry): Promise<void> {
  await ensureCacheLoaded().catch(() => {});
  datasetCache.push(entry);
  if (datasetCache.length > MAX_DATASET_SIZE) {
    datasetCache = datasetCache.slice(-MAX_DATASET_SIZE);
  }

  try {
    const { inArray, sql } = await import("drizzle-orm");
    await db.transaction(async (tx) => {
      await tx.insert(quantumEngineDataset).values(entryToRow(entry));
      const [{ total }] = await tx.select({ total: count() }).from(quantumEngineDataset);
      if (total > MAX_DATASET_SIZE) {
        const excess = total - MAX_DATASET_SIZE;
        await tx.execute(sql`
          DELETE FROM quantum_engine_dataset
          WHERE id IN (
            SELECT id FROM quantum_engine_dataset
            ORDER BY created_at ASC, id ASC
            LIMIT ${excess}
          )
        `);
      }
    });
  } catch (err) {
    console.error(`[QE-Dataset] DB write failed (cache still updated):`, err);
  }
}

ensureCacheLoaded().catch(() => {});

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
  dosPrefilterRuns: 0,
  dosPrefilterPassed: 0,
  dosPrefilterRejected: 0,
};

export async function runQuantumEnginePipeline(
  formula: string,
  pressureGpa: number = 0,
  skipXTB: boolean = false
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

  // Yield before any synchronous physics work so heartbeat timers (and DB keepalives)
  // can fire between sequential pipeline calls in the active-learning loop.
  await new Promise<void>(r => setTimeout(r, 0));

  let dosFilterResult: ReturnType<typeof dosPrefilter> | null = null;
  let dosAnalysis: DOSSurrogateResult | null = null;
  try {
    const dosStart = Date.now();
    dosAnalysis = predictDOS(formula);
    dosFilterResult = dosPrefilter(formula);

    steps.push({
      name: "4.0 DOS Surrogate Pre-filter",
      status: dosFilterResult.pass ? "success" : "failed",
      wallTimeMs: Date.now() - dosStart,
      detail: `DOS(EF)=${dosFilterResult.dosAtFermi.toFixed(3)}, VHS=${dosFilterResult.vhsCount}, scFav=${dosFilterResult.scFavorability.toFixed(3)}, flat=${dosFilterResult.flatBandIndicator.toFixed(3)} — ${dosFilterResult.reason}`,
    });

    if (!dosFilterResult.pass) {
      const passthroughRoll = Math.random();
      if (passthroughRoll < 0.05) {
        console.log(`[DOS-Prefilter] ${formula} rejected but passed through for surrogate validation (5% calibration sample): ${dosFilterResult.reason}`);
        dosFilterResult = { ...dosFilterResult, pass: true };
      } else {
        console.log(`[DOS-Prefilter] ${formula} rejected: ${dosFilterResult.reason}`);
      }
    }
  } catch (e: any) {
    steps.push({
      name: "4.0 DOS Surrogate Pre-filter",
      status: "skipped",
      wallTimeMs: 0,
      detail: `DOS pre-filter error: ${e.message?.slice(-200) ?? "unknown"}`,
    });
  }

  let surrogateElec: ReturnType<typeof computeElectronicStructure> | null = null;
  try {
    surrogateElec = computeElectronicStructure(formula);
  } catch {}

  // When DFT is offloaded to GCP, skip local QE entirely — GCP handles all DFT.
  // Running QE locally would block the event loop for 5-30 min per formula.
  const offloadedToGCP = process.env.OFFLOAD_DFT_TO_GCP === "true";
  const qeAvailable = !offloadedToGCP && isQEAvailable();

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
          if (dftResult.bandStructure?.converged && dftResult.bandStructure.eigenvalues.length > 0) {
            fermiDos = estimateDOSFromBands(dftResult.bandStructure, dftResult.scf.fermiEnergy);
            console.log(`[Pipeline] DOS(EF) from DFT bands for ${formula}: ${fermiDos.toFixed(3)} states/eV`);
          } else {
            fermiDos = isMetallic ? 3.0 : 0.5;
            console.log(`[Pipeline] DOS(EF) fallback for ${formula}: ${fermiDos.toFixed(3)} (no converged bands)`);
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
          detail: `SCF failed: ${dftResult.error ?? dftResult.scf?.error ?? "unknown"}`.slice(-200),
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

        if (dftResult.phonon.hasImaginary) {
          console.log(`[Pipeline] ${formula} structurally unstable: ${dftResult.phonon.imaginaryCount} imaginary phonon modes (lowest=${dftResult.phonon.lowestFrequency.toFixed(1)} cm-1) — DFT instability is authoritative, surrogates cannot override`);
        }
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
        detail: `Exception: ${e.message?.slice(-200) ?? "unknown"}`,
      });
    }
  }

  if (!scfConverged && !skipXTB) {
    // skipXTB=true: inline xTB takes 30-90s per formula; DFT queue handles it asynchronously.
    const xtbStart = Date.now();
    try {
      const xtbResult = await runXTBEnrichment(formula, pressureGpa);
      const xtbTime = Date.now() - xtbStart;

      if (xtbResult) {
        tier = "xtb";
        formationEnergy = xtbResult.formationEnergy;
        bandGap = xtbResult.bandGap;
        isMetallic = (bandGap ?? 999) < 0.1;
        phononStable = xtbResult.phononStable;

        if (surrogateElec) {
          fermiDos = surrogateElec.densityOfStatesAtFermi;
        } else {
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
        detail: `xTB exception: ${e.message?.slice(-200) ?? "unknown"}`,
      });
    }
  }

  if (tier === "surrogate") {
    const surrogateStart = Date.now();
    try {
      const elec = surrogateElec ?? computeElectronicStructure(formula);
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
        detail: `Surrogate failed: ${e.message?.slice(-200) ?? "unknown"}`,
      });
    }
  }

  const eliashbergStart = Date.now();
  try {
    let electronicOverride;
    let phononOverride;
    let couplingOverride;

    const dftPhononUnstable = tier === "full-dft" && dftResult?.phonon?.hasImaginary === true;

    if (tier === "full-dft" && dftResult?.scf?.converged) {
      electronicOverride = buildDFTElectronicOverride(formula, dftResult, fermiDos, surrogateElec);

      if (phononFreqs.length > 0) {
        phononOverride = computePhononSpectrum(formula);
        phononOverride.frequencies = phononFreqs;
        phononOverride.hasImaginaryModes = !phononStable;
        phononOverride.debyeTemperature = computeDebyeTemperature(phononFreqs);
        const omegaLog = computeOmegaLog(phononFreqs);
        if (omegaLog > 0) {
          phononOverride.logAverageFrequency = omegaLog;
        }
      }

      if (phononFreqs.length > 0 && !dftPhononUnstable) {
        couplingOverride = computeElectronPhononCoupling(formula);
      }
    }

    if (dftPhononUnstable) {
      console.log(`[Pipeline] ${formula}: DFT phonon instability — skipping Eliashberg (structure is not a valid superconductor candidate)`);
      steps.push({
        name: "4.3 Electron-Phonon Coupling (Eliashberg)",
        status: "failed",
        wallTimeMs: Date.now() - eliashbergStart,
        detail: `Skipped: DFT phonons show ${dftResult!.phonon!.imaginaryCount} imaginary modes — structure is dynamically unstable`,
      });
    } else {
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
    }
  } catch (e: any) {
    steps.push({
      name: "4.3 Electron-Phonon Coupling",
      status: "failed",
      wallTimeMs: Date.now() - eliashbergStart,
      detail: `Eliashberg pipeline failed: ${e.message?.slice(-200) ?? "unknown"}`,
    });
  }

  const totalWallTime = Date.now() - startTime;

  const lambda = eliashbergResult?.lambda ?? 0;
  const omegaLog = eliashbergResult?.omegaLog ?? 0;
  const tc = eliashbergResult?.tcBest ?? 0;

  const STANDARD_A2F_BINS = 100;
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
      nBins: STANDARD_A2F_BINS,
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
    dosPrefilter: dosFilterResult && dosAnalysis ? {
      pass: dosFilterResult.pass,
      vhsScore: dosAnalysis.vhsScore,
      scFavorability: dosAnalysis.scFavorability,
      flatBandIndicator: dosAnalysis.flatBandIndicator,
      nestingScore: dosAnalysis.nestingScore,
      vhsCount: dosAnalysis.vanHoveSingularities.length,
    } : undefined,
  };

  await appendToDataset(entry);

  try {
    recordPhysicsResult({
      formula,
      pressure: pressureGpa,
      lambda,
      omegaLog,
      tc,
      dosAtEF: fermiDos,
      phononStable,
      muStar: eliashbergResult?.muStar ?? 0.10,
      omega2: eliashbergResult?.omega2 ?? 0,
      gapRatio: eliashbergResult?.gapRatio ?? 3.53,
      isStrongCoupling: eliashbergResult?.isStrongCoupling ?? false,
      isotopeAlpha: eliashbergResult?.isotopeEffect.alpha ?? 0.5,
      formationEnergy,
      bandGap,
      isMetallic,
      tier,
      alpha2FPeak: alpha2FSummary.peakHeight,
      alpha2FPeakFreq: alpha2FSummary.peakFrequency,
      modeResolvedLambda: alpha2FSummary.lambdaByRange,
      timestamp: Date.now(),
    });
  } catch {}

  if (lambda > 0) {
    try {
      const mlPred = predictLambda(formula, pressureGpa);
      if (mlPred.tier === "ml-regression") {
        recordLambdaValidation(formula, mlPred.lambda, lambda);
      }
    } catch {}
  }

  try {
    const crystalEntry = getEntryByFormula(formula);
    if (crystalEntry) {
      const beforeLattice = { ...crystalEntry.lattice };
      let afterLattice: typeof beforeLattice;
      if (dftResult?.relaxedLatticeA && dftResult.relaxedLatticeA > 0.5) {
        const scale = dftResult.relaxedLatticeA / (dftResult.initialLatticeA ?? beforeLattice.a);
        afterLattice = {
          a: beforeLattice.a * scale,
          b: beforeLattice.b * scale,
          c: beforeLattice.c * scale,
          alpha: beforeLattice.alpha,
          beta: beforeLattice.beta,
          gamma: beforeLattice.gamma,
        };
      } else {
        afterLattice = { ...beforeLattice };
      }
      const beforePositions = dftResult?.initialPositions
        ?? crystalEntry.atomicPositions?.map(p => ({ element: p.element, x: p.x, y: p.y, z: p.z }));
      recordRelaxation({
        formula,
        beforeLattice,
        afterLattice,
        beforePositions,
        energyBefore: undefined,
        energyAfter: dftResult?.scf?.totalEnergyPerAtom ?? formationEnergy ?? undefined,
        pressureBefore: pressureGpa,
        pressureAfter: dftResult?.scf?.pressure ?? undefined,
        forcesConverged: scfConverged,
        relaxedAt: Date.now(),
        tier,
        crystalSystem: crystalEntry.crystalSystem,
        prototype: crystalEntry.prototype,
      });

      try {
        const sgMap: Record<string, string> = {
          Perovskite: "Pm-3m", A15: "Pm-3m", NaCl: "Fm-3m", AlB2: "P6/mmm",
          ThCr2Si2: "I4/mmm", Spinel: "Fd-3m", Heusler: "Fm-3m", Laves: "Fd-3m",
          MAX: "P63/mmc", Fluorite: "Fm-3m",
        };
        const sgBefore = sgMap[crystalEntry.prototype] || undefined;
        const beforePos = beforePositions;
        const distortionResult = analyzeDistortion(
          formula,
          beforeLattice,
          afterLattice,
          beforePos,
          undefined,
          sgBefore,
        );
        recordDistortionAnalysis(distortionResult);
      } catch {}
    }
  } catch {}


  pipelineStats.totalRuns++;
  if (tier === "full-dft") pipelineStats.fullDftRuns++;
  else if (tier === "xtb") pipelineStats.xtbRuns++;
  else pipelineStats.surrogateRuns++;
  if (dosFilterResult) {
    pipelineStats.dosPrefilterRuns++;
    if (dosFilterResult.pass) pipelineStats.dosPrefilterPassed++;
    else pipelineStats.dosPrefilterRejected++;
  }

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
    try {
      let failureReason: "unstable_phonons" | "structure_collapse" | "high_formation_energy" | "non_metallic" | "scf_divergence" | "geometry_rejected" = "scf_divergence";
      const dftStage = dftResult?.failureStage;
      if (dftStage === "geometry") {
        failureReason = "geometry_rejected";
      } else if (dftStage === "xtb_prefilter") {
        failureReason = "structure_collapse";
      } else if (!phononStable && phononFreqs.length > 0 && Math.min(...phononFreqs) < -100) {
        failureReason = "structure_collapse";
      } else if (!phononStable) {
        failureReason = "unstable_phonons";
      } else if (!isMetallic) {
        failureReason = "non_metallic";
      } else if (formationEnergy !== null && formationEnergy > 0.5) {
        failureReason = "high_formation_energy";
      } else if (!scfConverged) {
        failureReason = "scf_divergence";
      }
      recordStructureFailure({
        formula,
        failureReason,
        failedAt: Date.now(),
        source: tier === "full-dft" ? "dft" : tier === "xtb" ? "xtb" : "pipeline",
        formationEnergy: formationEnergy ?? undefined,
        bandGap: bandGap ?? undefined,
        lowestPhononFreq: phononFreqs.length > 0 ? Math.min(...phononFreqs) : undefined,
        imaginaryModeCount: !phononStable && phononFreqs.length > 0 ? phononFreqs.filter(f => f < 0).length : undefined,
        details: `QE pipeline: tier=${tier}, scf=${scfConverged}, metallic=${isMetallic}, phonon_stable=${phononStable}, dft_stage=${dftStage ?? "N/A"}`,
      });
    } catch {}
  }
  pipelineStats.wallTimeTotalMs += totalWallTime;
  pipelineStats.avgWallTimeMs = pipelineStats.wallTimeTotalMs / pipelineStats.totalRuns;

  return { entry, eliashberg: eliashbergResult, dftResult, steps };
}

export function getQuantumEngineStats() {
  const n = pipelineStats.successCount;
  return {
    totalRuns: pipelineStats.totalRuns,
    successCount: n,
    failCount: pipelineStats.failCount,
    tierBreakdown: {
      fullDft: pipelineStats.fullDftRuns,
      xtb: pipelineStats.xtbRuns,
      surrogate: pipelineStats.surrogateRuns,
    },
    avgLambda: n > 0 ? Number((pipelineStats.lambdaSum / n).toFixed(4)) : null,
    avgTc: n > 0 ? Number((pipelineStats.tcSum / n).toFixed(2)) : null,
    avgDosAtEF: n > 0 ? Number((pipelineStats.dosSum / n).toFixed(3)) : null,
    avgWallTimeMs: Math.round(pipelineStats.avgWallTimeMs),
    bestTc: pipelineStats.bestTc,
    bestTcMaterial: pipelineStats.bestTcMaterial,
    strongCouplingCount: pipelineStats.strongCouplingCount,
    highTcCount: pipelineStats.highTcCount,
    datasetSize: datasetCache.length,
    datasetMaxSize: MAX_DATASET_SIZE,
    dosPrefilter: {
      runs: pipelineStats.dosPrefilterRuns,
      passed: pipelineStats.dosPrefilterPassed,
      rejected: pipelineStats.dosPrefilterRejected,
      passRate: pipelineStats.dosPrefilterRuns > 0
        ? Number((pipelineStats.dosPrefilterPassed / pipelineStats.dosPrefilterRuns * 100).toFixed(1))
        : 0,
    },
  };
}

export async function getQuantumEngineDataset(): Promise<QuantumEngineDatasetEntry[]> {
  await ensureCacheLoaded().catch(() => {});
  return [...datasetCache];
}

export async function getRecentQuantumEngineResults(limit: number = 20): Promise<QuantumEngineDatasetEntry[]> {
  await ensureCacheLoaded().catch(() => {});
  return datasetCache.slice(-limit).reverse();
}

export async function addExternalDatasetEntry(entry: QuantumEngineDatasetEntry): Promise<void> {
  await appendToDataset(entry);
}
