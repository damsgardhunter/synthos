import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { extractFeatures, runMLPrediction } from "./ml-predictor";
import { classifyFamily } from "./utils";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

const VERIFICATION_CHECKLIST = [
  "zero_resistance_confirmed",
  "room_temperature_confirmed",
  "meissner_effect_verified",
  "crystal_structure_validated",
  "synthesis_path_feasible",
  "thermodynamic_stability_checked",
  "reproducibility_assessed",
  "pressure_requirements_acceptable",
] as const;

function determineStatus(candidate: any): string {
  const tc = candidate.predictedTc ?? 0;
  const pressure = candidate.pressureGpa ?? 999;
  const hasZeroResistance = candidate.zeroResistance === true;
  const hasMeissner = candidate.meissnerEffect === true;
  const ensembleScore = candidate.ensembleScore ?? 0;

  const isRoomTemp = tc >= 293;
  const isLowPressure = pressure <= 50;
  const isAmbientPressure = pressure <= 1;

  if (!hasZeroResistance || !hasMeissner) {
    return "theoretical";
  }

  if (isRoomTemp && isAmbientPressure && hasZeroResistance && hasMeissner && ensembleScore > 0.8) {
    return "requires-verification";
  }

  if (isRoomTemp && isLowPressure && hasZeroResistance && hasMeissner && ensembleScore > 0.7) {
    return "under-review";
  }

  if (isRoomTemp && hasZeroResistance && hasMeissner) {
    return "promising";
  }

  if (tc > 100 && hasZeroResistance) {
    return "high-tc-candidate";
  }

  if (ensembleScore > 0.5) {
    return "promising";
  }

  return "theoretical";
}

function buildVerificationNotes(candidate: any): string {
  const checks: string[] = [];
  const failures: string[] = [];
  const tc = candidate.predictedTc ?? 0;
  const pressure = candidate.pressureGpa ?? null;

  if (candidate.zeroResistance === true) {
    checks.push("Zero electrical resistance: PREDICTED");
  } else {
    failures.push("MISSING: Zero electrical resistance not confirmed - fundamental SC requirement");
  }

  if (tc >= 293) {
    checks.push(`Room temperature: Tc=${tc}K >= 293K (20C)`);
  } else if (tc > 0) {
    failures.push(`NOT room temperature: Tc=${tc}K < 293K required`);
  } else {
    failures.push("MISSING: No Tc prediction available");
  }

  if (candidate.meissnerEffect === true) {
    checks.push("Meissner effect (magnetic flux expulsion): PREDICTED");
  } else {
    failures.push("MISSING: Meissner effect not confirmed - must expel magnetic flux");
  }

  if (pressure != null && pressure <= 1) {
    checks.push(`Ambient pressure: ${pressure} GPa`);
  } else if (pressure != null && pressure <= 50) {
    checks.push(`Moderate pressure: ${pressure} GPa (lab achievable)`);
  } else if (pressure != null) {
    failures.push(`High pressure required: ${pressure} GPa - difficult to maintain`);
  }

  const status = determineStatus(candidate);
  let prefix = "";
  if (status === "requires-verification") {
    prefix = "AWAITING VERIFICATION - All criteria appear met but requires: independent synthesis confirmation, four-probe resistance measurement to absolute zero, SQUID magnetometry for Meissner verification, and reproducibility by at least two independent labs. ";
  } else if (status === "under-review") {
    prefix = "UNDER REVIEW - Promising but needs: detailed synthesis attempt, resistance vs temperature measurement, and magnetic susceptibility testing. ";
  }

  return prefix + "Checks passed: " + checks.join("; ") + (failures.length > 0 ? ". Remaining issues: " + failures.join("; ") : "");
}

interface StrategyContext {
  strategyFocusAreas?: { area: string; priority: number }[];
  familyCounts?: Record<string, number>;
  stagnationInfo?: { cyclesSinceImproved: number; currentBestTc: number };
}

export async function runSuperconductorResearch(
  emit: EventEmitter,
  allInsights: string[],
  materialOffset: number = 0,
  strategyCtx?: StrategyContext
): Promise<{ generated: number; insights: string[]; duplicatesSkipped: number }> {
  let generated = 0;
  const newInsights: string[] = [];

  emit("log", {
    phase: "phase-7",
    event: "Superconductor research cycle started",
    detail: "XGBoost+NN ensemble with strict verification: both zero resistance AND room temperature required",
    dataSource: "SC Research",
  });

  const materials = await storage.getMaterials(50, materialOffset);

  const synthesisProcesses = await storage.getSynthesisProcesses(20);
  const chemicalReactions = await storage.getChemicalReactions(20);

  const knowledgeDepth = new Map<string, { hasSynthesis: boolean; hasCrystal: boolean; pipelineStagesPassed: number; hasRelatedInsights: boolean }>();
  try {
    const allSynth = await storage.getSynthesisProcesses(200);
    const synthFormulas = new Set(allSynth.map(s => s.formula));
    const allCrystals = await storage.getCrystalStructures(200);
    const crystalFormulas = new Set(allCrystals.map(c => c.formula));
    const allResults = await storage.getComputationalResults(200);
    const pipelineByFormula = new Map<string, number>();
    for (const r of allResults) {
      if (r.passed) {
        pipelineByFormula.set(r.formula, (pipelineByFormula.get(r.formula) ?? 0) + 1);
      }
    }
    const recentInsights = await storage.getNovelInsightsOnly(20);
    const insightTexts = recentInsights.map(i => i.insightText.toLowerCase());

    for (const mat of materials) {
      const f = mat.formula;
      const family = classifyFamily(f).toLowerCase();
      knowledgeDepth.set(f, {
        hasSynthesis: synthFormulas.has(f),
        hasCrystal: crystalFormulas.has(f),
        pipelineStagesPassed: pipelineByFormula.get(f) ?? 0,
        hasRelatedInsights: insightTexts.some(t => t.includes(f.toLowerCase()) || t.includes(family)),
      });
    }
  } catch {}

  const mlResult = await runMLPrediction(emit, materials, {
    synthesisCount: synthesisProcesses.length,
    reactionCount: chemicalReactions.length,
    hasSynthesisKnowledge: synthesisProcesses.length > 0,
    hasReactionKnowledge: chemicalReactions.length > 0,
    strategyFocusAreas: strategyCtx?.strategyFocusAreas,
    familyCounts: strategyCtx?.familyCounts,
    knowledgeDepth,
  });
  newInsights.push(...mlResult.insights);

  let duplicatesSkipped = 0;
  for (const candidate of mlResult.candidates) {
    const formula = candidate.formula || "Unknown";
    const newScore = candidate.ensembleScore ?? 0;

    const existing = await storage.getSuperconductorByFormula(formula);
    if (existing) {
      const newTc = candidate.predictedTc ?? existing.predictedTc;
      const existingTc = existing.predictedTc ?? 0;
      const tcImproved = (newTc ?? 0) > existingTc;
      if (newScore > (existing.ensembleScore ?? 0) || tcImproved) {
        const updates: any = {
          ensembleScore: Math.max(newScore, existing.ensembleScore ?? 0),
          xgboostScore: candidate.xgboostScore ?? existing.xgboostScore,
          neuralNetScore: candidate.neuralNetScore ?? existing.neuralNetScore,
          mlFeatures: candidate.mlFeatures ?? existing.mlFeatures,
          notes: buildVerificationNotes(candidate),
        };
        if ((newTc ?? 0) > existingTc) {
          updates.predictedTc = newTc;
        }
        await storage.updateSuperconductorCandidate(existing.id, updates);
        const tcDetail = tcImproved ? `, Tc ${existingTc}K -> ${newTc}K` : "";
        emit("log", { phase: "phase-7", event: "SC candidate upgraded", detail: `${formula}: score ${(existing.ensembleScore ?? 0).toFixed(3)} -> ${Math.max(newScore, existing.ensembleScore ?? 0).toFixed(3)}${tcDetail}`, dataSource: "SC Research" });
      } else {
        duplicatesSkipped++;
      }
      continue;
    }

    const id = `sc-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const status = determineStatus(candidate);
    const verificationNotes = buildVerificationNotes(candidate);

    const isActuallyRoomTemp = (candidate.predictedTc ?? 0) >= 293 &&
      candidate.zeroResistance === true &&
      candidate.meissnerEffect === true;

    try {
      await storage.insertSuperconductorCandidate({
        id,
        name: candidate.name || "Unknown",
        formula,
        predictedTc: candidate.predictedTc ?? null,
        pressureGpa: candidate.pressureGpa ?? null,
        meissnerEffect: candidate.meissnerEffect ?? false,
        zeroResistance: candidate.zeroResistance ?? false,
        cooperPairMechanism: candidate.cooperPairMechanism ?? null,
        crystalStructure: candidate.crystalStructure ?? null,
        quantumCoherence: candidate.quantumCoherence ?? null,
        stabilityScore: candidate.stabilityScore ?? null,
        synthesisPath: candidate.synthesisPath ?? null,
        mlFeatures: candidate.mlFeatures ?? null,
        xgboostScore: candidate.xgboostScore ?? null,
        neuralNetScore: candidate.neuralNetScore ?? null,
        ensembleScore: candidate.ensembleScore ?? null,
        roomTempViable: isActuallyRoomTemp,
        status,
        notes: verificationNotes,
        electronPhononCoupling: candidate.electronPhononCoupling ?? null,
        logPhononFrequency: candidate.logPhononFrequency ?? null,
        coulombPseudopotential: candidate.coulombPseudopotential ?? null,
        pairingSymmetry: candidate.pairingSymmetry ?? null,
        pairingMechanism: candidate.pairingMechanism ?? null,
        correlationStrength: candidate.correlationStrength ?? null,
        dimensionality: candidate.dimensionality ?? null,
        fermiSurfaceTopology: candidate.fermiSurfaceTopology ?? null,
        uncertaintyEstimate: candidate.uncertaintyEstimate ?? null,
        verificationStage: candidate.verificationStage ?? 0,
      });
      generated++;

      emit("prediction", {
        type: "superconductor",
        id,
        name: candidate.name,
        formula,
        predictedTc: candidate.predictedTc,
        ensembleScore: candidate.ensembleScore,
        roomTempViable: isActuallyRoomTemp,
        meissnerEffect: candidate.meissnerEffect,
        zeroResistance: candidate.zeroResistance,
        status,
      });
    } catch (e: any) {
      emit("log", { phase: "phase-7", event: "SC candidate insert error", detail: `${formula}: ${e.message?.slice(0, 100)}`, dataSource: "SC Research" });
    }
  }

  if (duplicatesSkipped > 0) {
    emit("log", { phase: "phase-7", event: "Duplicates skipped", detail: `${duplicatesSkipped} duplicate formulas already in database`, dataSource: "SC Research" });
  }

  if (generated > 0) {
    const roomTempCount = mlResult.candidates.filter(c =>
      (c.predictedTc ?? 0) >= 293 && c.zeroResistance === true && c.meissnerEffect === true
    ).length;
    emit("log", {
      phase: "phase-7",
      event: "Superconductor candidates evaluated",
      detail: `${generated} candidates scored. ${roomTempCount} meet both zero-resistance AND room-temperature criteria (pending verification). ${mlResult.candidates.filter(c => c.meissnerEffect).length} with Meissner effect predicted.`,
      dataSource: "SC Research",
    });
  }

  try {
    const existingCandidates = await storage.getSuperconductorCandidates(10);
    if (existingCandidates.length > 0) {
      const novelResult = await generateNovelSuperconductors(emit, existingCandidates, allInsights, strategyCtx?.stagnationInfo);
      generated += novelResult;
    }
  } catch (err: any) {
    emit("log", { phase: "phase-7", event: "Novel SC generation error", detail: err.message?.slice(0, 200), dataSource: "SC Research" });
  }

  return { generated, insights: newInsights, duplicatesSkipped };
}

async function generateNovelSuperconductors(
  emit: EventEmitter,
  existingCandidates: any[],
  allInsights: string[],
  stagnationInfo?: { cyclesSinceImproved: number; currentBestTc: number }
): Promise<number> {
  let generated = 0;

  const sorted = existingCandidates
    .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));

  const byTc = [...existingCandidates]
    .sort((a, b) => (b.predictedTc ?? 0) - (a.predictedTc ?? 0));

  const seenFamilies = new Set<string>();
  const diverseExamples: any[] = [];
  for (const c of byTc) {
    const family = classifyFamily(c.formula);
    if (!seenFamilies.has(family) && diverseExamples.length < 5) {
      seenFamilies.add(family);
      diverseExamples.push(c);
    }
  }
  if (diverseExamples.length < 3) {
    for (const c of sorted) {
      if (!diverseExamples.includes(c) && diverseExamples.length < 5) {
        diverseExamples.push(c);
      }
    }
  }

  const bestCandidates = diverseExamples.map(c => ({
    formula: c.formula,
    family: classifyFamily(c.formula),
    predictedTc: c.predictedTc,
    electronPhononCoupling: c.electronPhononCoupling,
    ensembleScore: c.ensembleScore,
    meissnerEffect: c.meissnerEffect,
    zeroResistance: c.zeroResistance,
    cooperPairMechanism: c.cooperPairMechanism,
    status: c.status,
  }));

  const existingFormulas = sorted.slice(0, 20).map(c => c.formula);
  const exclusionList = [...new Set(existingFormulas)].slice(0, 15).join(", ");

  const stagnationCycles = stagnationInfo?.cyclesSinceImproved ?? 0;
  const currentCeiling = stagnationInfo?.currentBestTc ?? 0;
  const isStagnating = stagnationCycles > 5;
  const stagnationContext = isStagnating
    ? `\n\nCRITICAL CONTEXT: Our best Tc has been stuck at ${Math.round(currentCeiling)}K for ${stagnationCycles} cycles. We need candidates that can EXCEED this ceiling. Focus on:
- Ultra-high electron-phonon coupling (lambda > 2.5) via light elements (H, B, C, N) in cage/clathrate structures
- Multi-component hydrides with synergistic coupling (e.g., ternary/quaternary hydrides combining Ti, Sc, Y, La with H-rich sublattices)
- Novel pairing mechanisms: combined phonon + spin-fluctuation, charge-density wave enhanced, or topological surface states
- Predict Tc values ABOVE ${Math.round(currentCeiling)}K - push boundaries based on strong-coupling Eliashberg theory`
    : "";

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are the discovery module of a superconductor AI. Based on the best candidates found so far and known patterns, propose 2-3 NEW chemical formulas that could be room-temperature superconductors.

STRICT REQUIREMENTS - A true room-temperature superconductor MUST have ALL of these:
1. ZERO electrical resistance (not just low resistance - absolute zero ohms below Tc). This means electrons flow with literally no energy loss.
2. Critical temperature (Tc) >= 293K (20C, room temperature). The material must superconduct at normal room conditions.
3. Meissner effect - complete expulsion of ALL magnetic flux from the material interior when cooled below Tc.
4. Achievable at low or ambient pressure (< 10 GPa ideal, < 50 GPa maximum acceptable).
5. Must support Cooper pair formation through a known mechanism (phonon-mediated BCS, spin-fluctuation, charge-density wave, or unconventional).
6. Must be thermodynamically stable or metastable - it cannot decompose at room temperature.

IMPORTANT: Do NOT claim any candidate is a "confirmed breakthrough." All candidates are THEORETICAL PREDICTIONS that would require:
- Independent laboratory synthesis
- Four-probe resistance measurement from 300K down to 2K showing zero resistance below Tc
- SQUID magnetometry confirming complete Meissner effect
- Reproduction by at least 2 independent research groups
- Crystal structure verification by X-ray diffraction

For each candidate, describe the synthesis pathway with exact temperatures, durations, and equipment.

Return JSON with 'candidates' array:
- 'name' (descriptive)
- 'formula' (chemical formula)
- 'predictedTc' (Kelvin - be realistic based on known physics)
- 'pressureGpa' (required pressure, 0 for ambient)
- 'meissnerEffect' (boolean - predicted based on theory)
- 'zeroResistance' (boolean - predicted based on Cooper pair mechanism)
- 'cooperPairMechanism' (detailed description of how Cooper pairs would form)
- 'crystalStructure' (predicted with space group if possible)
- 'quantumCoherence' (0-1, realistic estimate)
- 'roomTempViable' (boolean - ONLY true if Tc >= 293K AND zero resistance AND Meissner)
- 'synthesisPath' (object with 'method', 'steps' array with exact temperatures/times, 'precursors' array, 'conditions' object)
- 'reasoning' (string under 200 chars explaining the physics of why this could work)`,
        },
        {
          role: "user",
          content: `Best candidates so far (diverse examples from different material families):\n${JSON.stringify(bestCandidates, null, 2)}\n\nPatterns discovered:\n${allInsights.slice(-8).join("\n")}${stagnationContext}\n\nIMPORTANT CONSTRAINTS:\n- Do NOT generate any of these existing formulas: ${exclusionList}\n- Generate candidates from DIFFERENT chemical families than the examples (explore pnictides, borides, nitrides, clathrate hydrides, kagome metals, heavy fermion compounds)\n- Each proposed candidate must have a genuinely novel composition not yet in our database\n- Prioritize candidates with very high electron-phonon coupling (lambda > 2.0) and light-element sublattices\n\nPropose novel candidates. Remember: both ZERO RESISTANCE and ROOM TEMPERATURE are required. Do not overstate confidence - these are theoretical predictions requiring experimental verification.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return 0;

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      return 0;
    }

    const newCandidates = parsed.candidates ?? [];

    for (const c of newCandidates) {
      if (!c.formula) continue;

      const existingNovel = await storage.getSuperconductorByFormula(c.formula);
      if (existingNovel) {
        continue;
      }

      const id = `sc-novel-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const features = extractFeatures(c.formula);

      let cappedTc = c.predictedTc ?? null;
      if (cappedTc != null && cappedTc > 0) {
        const featureLambda = features.electronPhononLambda ?? 0;
        const omegaLogK = (features.logPhononFreq ?? 300) * 1.44;
        const muStar = 0.12;
        let mcMillanMax = 0;
        const denom = featureLambda - muStar * (1 + 0.62 * featureLambda);
        if (featureLambda > 0.2 && Math.abs(denom) > 1e-6) {
          const exponent = -1.04 * (1 + featureLambda) / denom;
          mcMillanMax = (omegaLogK / 1.2) * Math.exp(exponent);
          if (!Number.isFinite(mcMillanMax) || mcMillanMax < 0) mcMillanMax = 0;
        }

        const corrStr = features.correlationStrength ?? 0;
        const metalScore = features.metallicity ?? 0.5;
        let tcCap: number;
        if (metalScore < 0.3) {
          tcCap = Math.min(20, mcMillanMax * 0.1 || 10);
        } else if (metalScore < 0.5) {
          tcCap = Math.min(80, mcMillanMax * 0.3 || 40);
        } else if (corrStr > 0.85) {
          tcCap = Math.min(80, mcMillanMax * 0.3 || 30);
        } else if (corrStr > 0.7) {
          tcCap = Math.min(200, mcMillanMax * 0.5 || 80);
        } else if (featureLambda < 0.3) {
          tcCap = Math.min(150, mcMillanMax > 0 ? mcMillanMax * 3.0 : 150);
        } else if (featureLambda < 0.5) {
          tcCap = Math.min(200, mcMillanMax > 0 ? mcMillanMax * 2.5 : 200);
        } else if (featureLambda < 1.0) {
          tcCap = Math.min(300, mcMillanMax > 0 ? mcMillanMax * 2.0 : 300);
        } else if (featureLambda < 1.5) {
          tcCap = mcMillanMax > 0 ? Math.min(400, mcMillanMax * 1.8) : 350;
        } else if (featureLambda < 2.5) {
          tcCap = mcMillanMax > 0 ? Math.min(450, mcMillanMax * 1.5) : 400;
        } else {
          tcCap = mcMillanMax > 0 ? Math.min(500, mcMillanMax * 1.3) : 450;
        }
        tcCap = Math.round(tcCap);

        if (cappedTc > tcCap) {
          cappedTc = tcCap;
        }
      }

      const isActuallyRoomTemp = (cappedTc ?? 0) >= 293 &&
        c.zeroResistance === true &&
        c.meissnerEffect === true;

      const status = determineStatus({
        ...c,
        ensembleScore: c.quantumCoherence ?? 0.5,
        roomTempViable: isActuallyRoomTemp,
      });

      const verificationNotes = buildVerificationNotes({
        ...c,
        roomTempViable: isActuallyRoomTemp,
      });

      try {
        await storage.insertSuperconductorCandidate({
          id,
          name: c.name || c.formula,
          formula: c.formula,
          predictedTc: cappedTc,
          pressureGpa: c.pressureGpa ?? null,
          meissnerEffect: c.meissnerEffect ?? false,
          zeroResistance: c.zeroResistance ?? false,
          cooperPairMechanism: c.cooperPairMechanism ?? null,
          crystalStructure: c.crystalStructure ?? null,
          quantumCoherence: c.quantumCoherence ?? null,
          stabilityScore: features.cooperPairStrength,
          synthesisPath: c.synthesisPath ?? null,
          mlFeatures: features as any,
          xgboostScore: null,
          neuralNetScore: null,
          ensembleScore: c.quantumCoherence ?? 0.5,
          roomTempViable: isActuallyRoomTemp,
          status,
          notes: (cappedTc !== (c.predictedTc ?? null) ? `[LLM proposed Tc=${c.predictedTc}K, capped to ${cappedTc}K] ` : '') + verificationNotes,
          electronPhononCoupling: features.electronPhononLambda ?? null,
          logPhononFrequency: features.logPhononFreq ?? null,
          coulombPseudopotential: 0.12,
          pairingSymmetry: features.dWaveSymmetry ? "d-wave" : "s-wave",
          pairingMechanism: features.correlationStrength > 0.6 ? "spin-fluctuation" : "phonon-mediated",
          correlationStrength: features.correlationStrength ?? null,
          dimensionality: features.layeredStructure ? "quasi-2D" : "3D",
          fermiSurfaceTopology: features.fermiSurfaceType ?? null,
          uncertaintyEstimate: 0.6,
          verificationStage: 0,
        });
        generated++;

        emit("prediction", {
          type: "novel-superconductor",
          id,
          name: c.name,
          formula: c.formula,
          predictedTc: c.predictedTc,
          roomTempViable: isActuallyRoomTemp,
          meissnerEffect: c.meissnerEffect,
          zeroResistance: c.zeroResistance,
          status,
          hasSynthesisPath: !!c.synthesisPath,
        });
      } catch (e: any) {
        emit("log", { phase: "phase-7", event: "Novel SC insert error", detail: `${c.formula}: ${e.message?.slice(0, 100)}`, dataSource: "SC Research" });
      }
    }

    if (generated > 0) {
      emit("log", {
        phase: "phase-7",
        event: "Novel superconductor designs proposed",
        detail: `${generated} new theoretical designs with synthesis pathways - all require experimental verification before any claims`,
        dataSource: "SC Research",
      });
    }
  } catch (err: any) {
    emit("log", { phase: "phase-7", event: "Novel SC generation error", detail: err.message?.slice(0, 200), dataSource: "SC Research" });
  }

  return generated;
}
