import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { extractFeatures, runMLPrediction } from "./ml-predictor";
import { gbPredict } from "./gradient-boost";
import { classifyFamily, getPrototypeHash, normalizeFormula, isValidFormula } from "./utils";
import { applyAmbientTcCap, computeElectronicStructure, computePhononSpectrum, computeElectronPhononCoupling, parseFormulaElements, computeDimensionalityScore, detectStructuralMotifs, evaluateCompetingPhases } from "./physics-engine";
import type { CapExtensionEvidence } from "./physics-engine";
import { passesStabilityGate } from "./phase-diagram-engine";
import { passesElementCountCap, estimateFamilyPressure } from "./candidate-generator";
import { getMinedRules } from "./pattern-miner";
import type { PatternRule } from "./pattern-miner";
import { allenDynesTcRaw } from "./physics-engine";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export const ROOM_TEMP_K = 293;

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

  const isRoomTemp = tc >= ROOM_TEMP_K;
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

  if (tc >= ROOM_TEMP_K) {
    checks.push(`Room temperature: Tc=${tc}K >= ${ROOM_TEMP_K}K (20C)`);
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

  const materials = await storage.getMaterials(200, materialOffset);

  const synthesisProcesses = await storage.getSynthesisProcesses(20);
  const chemicalReactions = await storage.getChemicalReactions(20);

  const mlResult = await runMLPrediction(emit, materials, {
    synthesisCount: synthesisProcesses.length,
    reactionCount: chemicalReactions.length,
    hasSynthesisKnowledge: synthesisProcesses.length > 0,
    hasReactionKnowledge: chemicalReactions.length > 0,
    strategyFocusAreas: strategyCtx?.strategyFocusAreas,
    familyCounts: strategyCtx?.familyCounts,
  });
  newInsights.push(...mlResult.insights);

  let duplicatesSkipped = 0;
  const existingCandidates = await storage.getSuperconductorCandidates(200);
  const existingPrototypeMap = new Map<string, string>();
  for (const ec of existingCandidates) {
    const proto = getPrototypeHash(ec.formula);
    if (!existingPrototypeMap.has(proto)) {
      existingPrototypeMap.set(proto, ec.formula);
    }
  }

  const pendingUpdates: Array<{ id: string; updates: any; logDetail: string }> = [];
  const pendingInserts: Array<{ payload: any; logEvent: any }> = [];

  for (const candidate of mlResult.candidates) {
    const rawFormula = candidate.formula || "Unknown";
    if (!isValidFormula(rawFormula)) {
      emit("log", { phase: "phase-7", event: "SC candidate rejected (invalid elements)", detail: `${rawFormula}: contains elements not in the periodic table`, dataSource: "SC Research" });
      duplicatesSkipped++;
      continue;
    }
    const formula = normalizeFormula(rawFormula);
    candidate.formula = formula;
    const newScore = candidate.ensembleScore ?? 0;

    if (!passesElementCountCap(formula)) {
      duplicatesSkipped++;
      continue;
    }
    const stabilityCheck = passesStabilityGate(formula);
    if (!stabilityCheck.pass) {
      duplicatesSkipped++;
      continue;
    }

    const mlFeatures = extractFeatures(formula);
    const lambdaML = candidate.electronPhononCoupling ?? mlFeatures.electronPhononLambda ?? 0;
    const pressureML = candidate.pressureGpa ?? 0;
    const metallicityML = mlFeatures.metallicity ?? 0.5;

    const hasMottDopingPotential = (mlFeatures.mottProximityScore ?? 0) > 0.4;
    const bandGapThreshold = hasMottDopingPotential ? 1.5 : 0.5;
    if (mlFeatures.bandGap !== null && mlFeatures.bandGap > bandGapThreshold) {
      emit("log", { phase: "phase-7", event: "SC candidate rejected (insulator)", detail: `${formula}: bandGap=${mlFeatures.bandGap.toFixed(2)} eV > ${bandGapThreshold} eV${hasMottDopingPotential ? " (Mott parent, relaxed threshold)" : ""} — superconductors must be metallic`, dataSource: "SC Research" });
      duplicatesSkipped++;
      continue;
    }

    const metallicityFloor = hasMottDopingPotential ? 0.08 : 0.2;
    if (mlFeatures.metallicity < metallicityFloor) {
      emit("log", { phase: "phase-7", event: "SC candidate rejected (non-metallic)", detail: `${formula}: metallicity=${mlFeatures.metallicity.toFixed(2)} < ${metallicityFloor}${hasMottDopingPotential ? " (Mott parent, relaxed)" : ""} — insufficient metallicity for superconductivity`, dataSource: "SC Research" });
      duplicatesSkipped++;
      continue;
    }

    const protoHash = getPrototypeHash(formula);
    const isostructuralMatch = existingPrototypeMap.get(protoHash);
    if (isostructuralMatch && isostructuralMatch !== formula) {
      const existingIso = await storage.getSuperconductorByFormula(isostructuralMatch);
      if (existingIso) {
        const isoScoreBetter = newScore > (existingIso.ensembleScore ?? 0);
        const isoTcBetter = (candidate.predictedTc ?? 0) > (existingIso.predictedTc ?? 0) * 1.1;
        if (!isoScoreBetter && !isoTcBetter) {
          emit("log", { phase: "phase-7", event: "Isostructural duplicate skipped", detail: `${formula} matches prototype of ${isostructuralMatch} — no score or Tc improvement`, dataSource: "SC Research" });
          duplicatesSkipped++;
          continue;
        }
      }
    }
    existingPrototypeMap.set(protoHash, formula);

    const capEvidence: CapExtensionEvidence = {
      eliashbergLambda: lambdaML > 0 ? lambdaML : undefined,
      eliashbergTc: candidate.predictedTc != null && candidate.predictedTc > 0 ? candidate.predictedTc : undefined,
      gnnEnsembleStd: (candidate.uncertaintyEstimate != null && candidate.uncertaintyEstimate > 0)
        ? candidate.uncertaintyEstimate * (candidate.predictedTc ?? 50)
        : undefined,
    };
    const effectiveTcCapML = applyAmbientTcCap(9999, lambdaML, pressureML, metallicityML, formula, capEvidence);
    if (candidate.predictedTc != null) {
      candidate.predictedTc = Math.min(
        applyAmbientTcCap(candidate.predictedTc, lambdaML, pressureML, metallicityML, formula, capEvidence),
        effectiveTcCapML
      );
    }

    const existing = await storage.getSuperconductorByFormula(formula);
    if (existing) {
      const newTc = candidate.predictedTc ?? existing.predictedTc;
      const existingTc = existing.predictedTc ?? 0;
      const existingLambda = existing.electronPhononCoupling ?? candidate.electronPhononCoupling ?? 0;
      const tcImproved = (newTc ?? 0) > existingTc;
      const scoreMuchHigher = newScore > (existing.ensembleScore ?? 0) * 1.15;
      const tcDowngradeNeeded = scoreMuchHigher && (newTc ?? 0) < existingTc;
      if (newScore > (existing.ensembleScore ?? 0) || tcImproved) {
        const updates: any = {
          ensembleScore: Math.max(newScore, existing.ensembleScore ?? 0),
          xgboostScore: candidate.xgboostScore ?? existing.xgboostScore,
          neuralNetScore: candidate.neuralNetScore ?? existing.neuralNetScore,
          mlFeatures: candidate.mlFeatures ?? existing.mlFeatures,
          notes: buildVerificationNotes(candidate),
        };
        if ((newTc ?? 0) > existingTc) {
          const tcUpCap = existingLambda > 2.5 ? 150 : existingLambda > 2.0 ? 120 : existingLambda > 1.5 ? 90 : existingLambda > 1.0 ? 70 : 50;
          let cappedUpTc = Math.min(newTc ?? 0, existingTc + tcUpCap);
          cappedUpTc = Math.min(cappedUpTc, effectiveTcCapML);
          updates.predictedTc = cappedUpTc;
        } else if (tcDowngradeNeeded) {
          updates.predictedTc = Math.max(newTc ?? 0, Math.round(existingTc * 0.5));
          updates.ensembleScore = newScore;
        }
        const tcDetail = tcImproved
          ? `, Tc ${existingTc}K -> ${updates.predictedTc ?? newTc}K`
          : tcDowngradeNeeded
            ? `, Tc corrected ${existingTc}K -> ${updates.predictedTc}K (higher-confidence model)`
            : "";
        pendingUpdates.push({
          id: existing.id,
          updates,
          logDetail: `${formula}: score ${(existing.ensembleScore ?? 0).toFixed(3)} -> ${Math.max(newScore, existing.ensembleScore ?? 0).toFixed(3)}${tcDetail}`,
        });
      } else {
        duplicatesSkipped++;
      }
      continue;
    }

    const id = `sc-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const status = determineStatus(candidate);
    const verificationNotes = buildVerificationNotes(candidate);

    const isActuallyRoomTemp = (candidate.predictedTc ?? 0) >= ROOM_TEMP_K &&
      candidate.zeroResistance === true &&
      candidate.meissnerEffect === true;

    pendingInserts.push({
      payload: {
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
      },
      logEvent: {
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
      },
    });
  }

  if (pendingUpdates.length > 0) {
    try {
      await storage.bulkUpdateSuperconductorCandidates(
        pendingUpdates.map(u => ({ id: u.id, updates: u.updates }))
      );
      for (const u of pendingUpdates) {
        emit("log", { phase: "phase-7", event: "SC candidate upgraded", detail: u.logDetail, dataSource: "SC Research" });
      }
    } catch (e: any) {
      emit("log", { phase: "phase-7", event: "SC batch update error", detail: `${pendingUpdates.length} updates failed: ${e.message?.slice(0, 100)}`, dataSource: "SC Research" });
    }
  }

  if (pendingInserts.length > 0) {
    try {
      const insertCount = await storage.bulkInsertSuperconductorCandidates(
        pendingInserts.map(i => i.payload)
      );
      generated += insertCount;
      for (const i of pendingInserts) {
        emit("prediction", i.logEvent);
      }
    } catch (e: any) {
      emit("log", { phase: "phase-7", event: "SC batch insert error", detail: `${pendingInserts.length} inserts failed: ${e.message?.slice(0, 100)}`, dataSource: "SC Research" });
    }
  }

  if (duplicatesSkipped > 0) {
    emit("log", { phase: "phase-7", event: "Duplicates skipped", detail: `${duplicatesSkipped} duplicate formulas already in database`, dataSource: "SC Research" });
  }

  if (generated > 0) {
    const roomTempCount = mlResult.candidates.filter(c =>
      (c.predictedTc ?? 0) >= ROOM_TEMP_K && c.zeroResistance === true && c.meissnerEffect === true
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

function condenseInsightsWithRules(allInsights: string[]): string {
  const rules = getMinedRules();

  const negativeRules = rules
    .filter(r => r.consequent === "low-tc" && r.confidence >= 0.6 && r.support >= 3)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);

  const positiveRules = rules
    .filter(r => r.consequent === "high-tc" && r.confidence >= 0.6 && r.support >= 3)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);

  function ruleToText(r: PatternRule): string {
    const conds = r.conditions.map(c => {
      if (c.operator === "between") return `${c.property} in [${c.threshold.toFixed(2)}, ${c.upperThreshold?.toFixed(2)}]`;
      return `${c.property} ${c.operator} ${c.threshold.toFixed(2)}`;
    }).join(" AND ");
    const family = r.family ? `[${r.family}] ` : "";
    return `${family}${conds} -> ${r.consequent} (conf=${r.confidence.toFixed(2)}, n=${r.support})`;
  }

  const parts: string[] = [];

  if (negativeRules.length > 0) {
    parts.push("AVOID (mined negative patterns):\n" + negativeRules.map(r => "- " + ruleToText(r)).join("\n"));
  }
  if (positiveRules.length > 0) {
    parts.push("FAVORABLE (mined positive patterns):\n" + positiveRules.map(r => "- " + ruleToText(r)).join("\n"));
  }

  const recentInsights = allInsights.slice(-5);
  const earlyInsights = allInsights.length > 10
    ? allInsights.slice(0, 3).map(i => `[early] ${i}`)
    : [];
  const insightLines = [...earlyInsights, ...recentInsights];
  if (insightLines.length > 0) {
    parts.push("Key insights:\n" + insightLines.join("\n"));
  }

  return parts.join("\n\n");
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

  const allFormulas = existingCandidates.map(c => c.formula);
  const formulaSet = new Set(allFormulas);

  const familyCounts: Record<string, { count: number; bestTc: number; examples: string[] }> = {};
  for (const c of existingCandidates) {
    const fam = classifyFamily(c.formula);
    if (!familyCounts[fam]) familyCounts[fam] = { count: 0, bestTc: 0, examples: [] };
    familyCounts[fam].count++;
    familyCounts[fam].bestTc = Math.max(familyCounts[fam].bestTc, c.predictedTc ?? 0);
    if (familyCounts[fam].examples.length < 3) familyCounts[fam].examples.push(c.formula);
  }

  const exhaustedFamilies = Object.entries(familyCounts)
    .filter(([, v]) => v.count >= 10)
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 8);

  const familyExclusionLines = exhaustedFamilies.map(([fam, v]) =>
    `- ${fam}: ${v.count} candidates explored, best Tc=${Math.round(v.bestTc)}K (e.g. ${v.examples.join(", ")})`
  );

  const topFormulaSample = [...new Set(sorted.slice(0, 8).map(c => c.formula))].join(", ");
  const exclusionContext = familyExclusionLines.length > 0
    ? `EXHAUSTED FAMILIES (avoid these stoichiometric regimes):\n${familyExclusionLines.join("\n")}\n\nAlso avoid these specific top formulas: ${topFormulaSample}`
    : `Do NOT generate any of these existing formulas: ${topFormulaSample}`;

  const stagnationCycles = stagnationInfo?.cyclesSinceImproved ?? 0;
  const currentCeiling = stagnationInfo?.currentBestTc ?? 0;
  const isStagnating = stagnationCycles > 3;
  const isDeepStagnation = stagnationCycles > 8;
  const stagnationContext = isDeepStagnation
    ? `\n\nCRITICAL CONTEXT: Deep stagnation — no meaningful improvement in ${stagnationCycles} cycles. Current best Tc: ${Math.round(currentCeiling)}K. ABANDON current chemical regime entirely:
- Explore fundamentally different bonding environments and crystal chemistries
- Consider mixed-anion systems, kagome lattices, geometric frustration motifs
- Try unconventional pairing: spin-fluctuation near AFM QCP, excitonic in mixed-dimensional systems, topological pairing
- Seek mixed stiff-soft bonding networks with high phonon spectral weight at moderate frequencies
Let Tc emerge from strong pairing conditions rather than targeting a specific Tc value.`
    : isStagnating
      ? `\n\nCONTEXT: Marginal progress in recent cycles (${stagnationCycles} cycles since meaningful improvement). Current best Tc: ${Math.round(currentCeiling)}K. Shift focus from Tc maximization to PAIRING CONDITIONS:
- Maximize DOS at Fermi level (>3 states/eV/atom) via flat bands, van Hove singularities, or geometric frustration
- Optimize Fermi surface nesting for strong electron-phonon or spin-fluctuation coupling channels
- Target materials near quantum critical points (magnetic, structural, or charge instabilities)
Let Tc emerge from strong pairing conditions rather than targeting a specific Tc value.`
      : "";

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a superconductor discovery AI. Propose 2-3 NEW formulas with potential for high-Tc superconductivity. All candidates are THEORETICAL PREDICTIONS — never say "confirmed" or "breakthrough."

Tc BENCHMARKS (anchor predictions here): Cuprates ~135K ambient; Pnictides ~55K; Nickelates ~80K@pressure; BCS metals ~39K (MgB2); Hydrides ~250K@150GPa (NOT ambient); Heavy fermion ~2K; Bismuthates ~30K; Borocarbides ~23K. Exceeding family records needs extraordinary justification.

Be HONEST about uncertainty. A well-supported 50K prediction beats an unsupported 300K hallucination. Describe plausible synthesis strategies with approximate conditions — do not hallucinate exact lab parameters.

Return JSON 'candidates' array: 'name', 'formula', 'predictedTc' (K, realistic), 'pressureGpa', 'meissnerEffect' (bool), 'meissnerConfidence' (0-1), 'zeroResistance' (bool), 'zeroResistanceConfidence' (0-1), 'cooperPairMechanism', 'crystalStructure', 'quantumCoherence' (0-1), 'theoreticalConfidence' (0-1), 'roomTempViable' (bool, ONLY if Tc>=293K AND zeroRes AND Meissner AND conf>0.3), 'synthesisPath' (object: method, steps, precursors, conditions), 'reasoning' (<200 chars).`,
        },
        {
          role: "user",
          content: `Top candidates:\n${JSON.stringify(bestCandidates, null, 2)}\n\n${condenseInsightsWithRules(allInsights)}${stagnationContext}\n\n${exclusionContext}\n\nCONSTRAINTS:\n- Propose genuinely novel compositions from UNDER-EXPLORED families\n- PRIORITIZE PAIRING SUSCEPTIBILITY over raw Tc: high DOS(Ef), strong nesting, favorable phonon spectral weight, proximity to quantum critical points\n- Set theoreticalConfidence HONESTLY: most materials 0.1-0.4, only exceptionally well-supported >0.5`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 900,
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
      if (!isValidFormula(c.formula)) {
        emit("log", { phase: "phase-7", event: "Novel SC rejected (invalid elements)", detail: `${c.formula}: contains elements not in the periodic table`, dataSource: "SC Research" });
        continue;
      }
      c.formula = normalizeFormula(c.formula);

      if (formulaSet.has(c.formula)) {
        continue;
      }

      const existingNovel = await storage.getSuperconductorByFormula(c.formula);
      if (existingNovel) {
        formulaSet.add(c.formula);
        continue;
      }

      const id = `sc-novel-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const features = extractFeatures(c.formula);
      const gbResult = gbPredict(features);

      const featureLambda = features.electronPhononLambda ?? 0;
      const srHCount = (features as any).hCount ?? 0;
      const srTotalAtoms = (features as any).totalAtoms ?? 1;
      const srIsHydride = srHCount >= 4 && srHCount / srTotalAtoms >= 0.5;
      const physicsTc = featureLambda > 0
        ? Math.round(allenDynesTcRaw(featureLambda, features.logPhononFreq ?? 300, 0.12, undefined, srIsHydride))
        : 0;
      const enforcedPressure = estimateFamilyPressure(c.formula);
      const effectivePressure = Math.max(c.pressureGpa ?? 0, enforcedPressure);
      let cappedTc: number | null = physicsTc > 0 ? physicsTc : null;
      if (cappedTc != null) {
        cappedTc = Math.round(applyAmbientTcCap(cappedTc, featureLambda, effectivePressure, features.metallicity ?? 0.5, c.formula));
      }
      const llmProposedTc = c.predictedTc ?? null;

      const llmTheoreticalConf = Math.min(1, Math.max(0, c.theoreticalConfidence ?? 0.3));
      const meissnerConf = Math.min(1, Math.max(0, c.meissnerConfidence ?? (c.meissnerEffect ? 0.5 : 0.1)));
      const zeroResConf = Math.min(1, Math.max(0, c.zeroResistanceConfidence ?? (c.zeroResistance ? 0.5 : 0.1)));

      if (llmTheoreticalConf < 0.1 && meissnerConf < 0.15 && zeroResConf < 0.15) {
        emit("log", { phase: "phase-7", event: "Novel SC rejected (low LLM confidence)", detail: `${c.formula}: theoreticalConf=${llmTheoreticalConf.toFixed(2)}, meissnerConf=${meissnerConf.toFixed(2)}, zeroResConf=${zeroResConf.toFixed(2)}`, dataSource: "SC Research" });
        continue;
      }

      const confDamping = 0.5 + 0.5 * llmTheoreticalConf;
      const novelNNScore = (c.quantumCoherence ?? 0.3) * confDamping;
      const novelXGBScore = gbResult.score;
      const novelEnsembleScore = Math.min(0.95, novelXGBScore * 0.4 + novelNNScore * 0.6);

      const isActuallyRoomTemp = (cappedTc ?? 0) >= ROOM_TEMP_K &&
        c.zeroResistance === true &&
        c.meissnerEffect === true &&
        effectivePressure <= 50;

      const status = determineStatus({
        ...c,
        ensembleScore: novelEnsembleScore,
        roomTempViable: isActuallyRoomTemp,
      });

      const verificationNotes = buildVerificationNotes({
        ...c,
        roomTempViable: isActuallyRoomTemp,
      });

      try {
        if (!passesElementCountCap(c.formula)) continue;
        const stabilityCheck = passesStabilityGate(c.formula);
        if (!stabilityCheck.pass) {
          continue;
        }
        await storage.insertSuperconductorCandidate({
          id,
          name: c.name || c.formula,
          formula: c.formula,
          predictedTc: cappedTc,
          pressureGpa: effectivePressure,
          meissnerEffect: c.meissnerEffect ?? false,
          zeroResistance: c.zeroResistance ?? false,
          cooperPairMechanism: c.cooperPairMechanism ?? (features.correlationStrength > 0.6
            ? "Unconventional pairing via spin-fluctuation exchange"
            : "Phonon-mediated BCS pairing"),
          crystalStructure: c.crystalStructure ?? null,
          quantumCoherence: c.quantumCoherence ?? null,
          stabilityScore: features.cooperPairStrength,
          synthesisPath: c.synthesisPath ?? null,
          mlFeatures: features as any,
          xgboostScore: novelXGBScore,
          neuralNetScore: novelNNScore,
          ensembleScore: novelEnsembleScore,
          roomTempViable: isActuallyRoomTemp && (c.pressureGpa ?? 999) <= 50,
          status,
          notes: `[synthesis_origin: llm_speculative] ` + (llmProposedTc != null ? `[LLM suggested Tc=${llmProposedTc}K, physics-only Tc=${cappedTc}K (Allen-Dynes, lambda=${featureLambda.toFixed(2)})] ` : '') + verificationNotes,
          electronPhononCoupling: features.electronPhononLambda ?? null,
          logPhononFrequency: features.logPhononFreq ?? null,
          coulombPseudopotential: 0.12,
          pairingMechanism: features.correlationStrength > 0.6 ? "spin-fluctuation" : "phonon-mediated",
          pairingSymmetry: features.correlationStrength > 0.6 ? "d-wave" : (features.dWaveSymmetry ? "d-wave" : "s-wave"),
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

export function computePairingSusceptibility(formula: string): {
  score: number;
  lambda: number;
  nestingFactor: number;
  dosAtEf: number;
  phononSoftness: number;
} {
  const electronic = computeElectronicStructure(formula);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula);

  const lambda = coupling.lambda;
  const dosAtEf = electronic.densityOfStatesAtFermi;
  const nestingScore = electronic.nestingScore ?? 0;
  const vanHoveProx = electronic.vanHoveProximity ?? 0;
  const bandFlat = electronic.bandFlatness ?? 0;
  const softModeScore = phonon.softModeScore ?? (phonon.softModePresent ? 0.6 : 0.2);
  const mottProx = electronic.mottProximityScore ?? 0;
  const topoScore = electronic.topologicalBandScore ?? 0;
  const dimScore = computeDimensionalityScore(formula);
  const motifResult = detectStructuralMotifs(formula);

  const locPenalty = Math.max(0, (electronic.correlationStrength - 0.5) * (1 - electronic.metallicity));
  const dosWithPenalty = Math.min(1.0, dosAtEf / 5.0) * (1 - locPenalty * 0.3);

  const competingPhases = evaluateCompetingPhases(formula, electronic);
  let competingOrderProx = 0;
  if (electronic.metallicity > 0.4) {
    const bestProx = competingPhases.reduce((mx, p) => Math.max(mx, p.strength * (p.suppressesSC ? 0.5 : 1.0)), 0);
    competingOrderProx = Math.min(1.0, bestProx);
  }

  const orbD = electronic.orbitalFractions?.d ?? 0;
  const orbP = electronic.orbitalFractions?.p ?? 0;
  const orbitalCharScore = Math.min(1.0, orbD * 0.7 + orbP * 0.5);

  let score = (
    Math.min(1.0, lambda / 2.5) * 0.22 +
    dosWithPenalty * 0.15 +
    nestingScore * 0.14 +
    competingOrderProx * 0.12 +
    softModeScore * 0.10 +
    vanHoveProx * 0.08 +
    bandFlat * 0.05 +
    dimScore * 0.05 +
    orbitalCharScore * 0.05 +
    (electronic.metallicity > 0.7 ? 0.04 : electronic.metallicity * 0.04)
  );

  if (mottProx > 0.5 && mottProx < 0.8) {
    score *= (1 + (mottProx - 0.5) * 0.15);
  }

  if (topoScore > 0.6) {
    score *= 1.05;
  }

  const novelty = computeCompositionNovelty(formula);
  score = 0.85 * score + 0.15 * novelty;

  const features = extractFeatures(formula);
  const fe = features.formationEnergy ?? 0;
  const stabilityFactor = fe < 0 ? 1.0 : Math.max(0.5, 1.0 - fe * 0.2);
  score *= stabilityFactor;

  const nestingFactor = nestingScore;
  const phononSoftness = softModeScore;

  return { score: Math.min(1.0, score), lambda, nestingFactor, dosAtEf, phononSoftness };
}

export function computeCompositionNovelty(formula: string): number {
  const targetElements = new Set(parseFormulaElements(formula));
  if (targetElements.size === 0) return 1;

  let minDistance = 1;

  for (const entry of SUPERCON_TRAINING_DATA) {
    const entryElements = new Set(parseFormulaElements(entry.formula));
    if (entryElements.size === 0) continue;

    const intersectionSize = Array.from(targetElements).filter(e => entryElements.has(e)).length;
    const unionSize = new Set(Array.from(targetElements).concat(Array.from(entryElements))).size;

    const jaccard = unionSize > 0 ? 1 - intersectionSize / unionSize : 1;

    if (jaccard < minDistance) {
      minDistance = jaccard;
    }

    if (minDistance === 0) break;
  }

  return Math.max(0, Math.min(1, minDistance));
}

let totalInverseDesignGenerated = 0;

export function getInverseDesignCount(): number {
  return totalInverseDesignGenerated;
}

export async function generateInverseDesignCandidates(
  emit: EventEmitter,
  allInsights: string[]
): Promise<number> {
  let generated = 0;

  emit("log", {
    phase: "phase-7",
    event: "Inverse design cycle started",
    detail: "Generating candidates optimized for pairing susceptibility rather than Tc prediction",
    dataSource: "Inverse Design",
  });

  const existingTop = await storage.getSuperconductorCandidates(20);
  const topByPairing = existingTop
    .map(c => {
      const ps = computePairingSusceptibility(c.formula);
      return { formula: c.formula, pairingScore: ps.score, lambda: ps.lambda, dos: ps.dosAtEf };
    })
    .sort((a, b) => b.pairingScore - a.pairingScore)
    .slice(0, 5);

  const existingFormulas = existingTop.map(c => c.formula).slice(0, 20);

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are an inverse materials designer. Instead of predicting Tc for known compositions, you DESIGN materials that maximize specific physical properties favorable for superconductivity.

ESTABLISHED Tc BENCHMARKS (anchor predictions realistically):
- Cuprates: max ~135K ambient, ~165K at 30 GPa
- Iron pnictides: max ~55K
- Nickelates: max ~80K under pressure
- Conventional BCS: max ~39K (MgB2)
- Hydrides: ~250K at 150-200 GPa (NOT ambient)
Exceeding these records requires extraordinary theoretical justification.

TARGET PHYSICS PROPERTIES to optimize:
1. Electron-phonon coupling lambda > 2.0 (essential for high Tc)
2. Density of States at Fermi level > 5 states/eV/atom (high pairing susceptibility)
3. Log-average phonon frequency omega_log between 500-1500K (optimal for BCS)
4. Quasi-2D Fermi surface with nesting features (enhances Cooper pairing)
5. Flat bands near Fermi level (enhances DOS without requiring heavy atoms)
6. Mixed stiff-soft bonding (stiff framework + soft rattler modes boost coupling)

DESIGN STRATEGIES:
- Clathrate/cage structures with light atoms (H, B, C, N) inside heavy-atom frameworks
- Layered materials with electronically active planes
- Materials at the edge of structural or magnetic instabilities
- High-entropy combinations that break symmetry and create flat bands
- Intercalated structures with enhanced phonon coupling

Return JSON with 'candidates' array: 'formula', 'name', 'predictedTc' (Kelvin), 'pressureGpa', 'meissnerEffect' (boolean), 'zeroResistance' (boolean), 'cooperPairMechanism', 'crystalStructure', 'roomTempViable' (boolean), 'inverseDesignTarget' (which property was optimized), 'reasoning' (under 150 chars)`,
        },
        {
          role: "user",
          content: `Materials with highest pairing susceptibility so far:\n${JSON.stringify(topByPairing, null, 2)}\n\n${condenseInsightsWithRules(allInsights)}\n\nDo NOT generate: ${existingFormulas.join(", ")}\n\nDesign 2-3 NEW compositions optimizing for pairing susceptibility. Focus on maximizing lambda and DOS(Ef) simultaneously. All predictions are theoretical.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1200,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return 0;

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      return 0;
    }

    const candidates = parsed.candidates ?? [];

    for (const c of candidates) {
      if (!c.formula) continue;
      if (!isValidFormula(c.formula)) {
        emit("log", { phase: "phase-7", event: "Inverse design rejected (invalid elements)", detail: `${c.formula}: contains elements not in the periodic table`, dataSource: "SC Research" });
        continue;
      }
      c.formula = normalizeFormula(c.formula);

      const existing = await storage.getSuperconductorByFormula(c.formula);
      if (existing) continue;

      const features = extractFeatures(c.formula);
      const gbResult = gbPredict(features);
      const pairingSusc = computePairingSusceptibility(c.formula);

      const lambdaML = features.electronPhononLambda ?? 0;
      const effectiveLambda = pairingSusc.lambda > 0 ? pairingSusc.lambda : lambdaML;
      const invHCount = (features as any).hCount ?? 0;
      const invTotalAtoms = (features as any).totalAtoms ?? 1;
      const invIsHydride = invHCount >= 4 && invHCount / invTotalAtoms >= 0.5;
      const invPhysicsTc = effectiveLambda > 0
        ? Math.round(allenDynesTcRaw(effectiveLambda, features.logPhononFreq ?? 300, 0.12, undefined, invIsHydride))
        : 0;
      const invEnforcedPressure = Math.max(c.pressureGpa ?? 0, estimateFamilyPressure(c.formula));
      const metallicityML = features.metallicity ?? 0.5;
      let cappedTc = invPhysicsTc > 0 ? invPhysicsTc : 0;
      const invDesEvidence: CapExtensionEvidence = {
        eliashbergLambda: effectiveLambda > 0 ? effectiveLambda : undefined,
        eliashbergTc: cappedTc > 0 ? cappedTc : undefined,
      };
      cappedTc = Math.round(applyAmbientTcCap(cappedTc, lambdaML, invEnforcedPressure, metallicityML, c.formula, invDesEvidence));

      const inverseDesignScore = Math.min(0.95, pairingSusc.score * 0.6 + gbResult.score * 0.4);

      const id = `sc-invdes-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

      const isActuallyRoomTemp = (cappedTc ?? 0) >= ROOM_TEMP_K &&
        c.zeroResistance === true &&
        c.meissnerEffect === true &&
        invEnforcedPressure <= 50;

      try {
        if (!passesElementCountCap(c.formula)) continue;
        const stabilityCheck = passesStabilityGate(c.formula);
        if (!stabilityCheck.pass) {
          continue;
        }
        await storage.insertSuperconductorCandidate({
          id,
          name: c.name || c.formula,
          formula: c.formula,
          predictedTc: cappedTc,
          pressureGpa: invEnforcedPressure,
          meissnerEffect: c.meissnerEffect ?? false,
          zeroResistance: c.zeroResistance ?? false,
          cooperPairMechanism: c.cooperPairMechanism ?? "Optimized for pairing susceptibility",
          crystalStructure: c.crystalStructure ?? null,
          quantumCoherence: pairingSusc.score,
          stabilityScore: features.cooperPairStrength,
          synthesisPath: null,
          mlFeatures: features as any,
          xgboostScore: gbResult.score,
          neuralNetScore: pairingSusc.score,
          ensembleScore: inverseDesignScore,
          roomTempViable: isActuallyRoomTemp,
          status: determineStatus({ ...c, predictedTc: cappedTc, ensembleScore: inverseDesignScore, roomTempViable: isActuallyRoomTemp }),
          notes: `[synthesis_origin: llm_speculative] [Inverse design: target=${c.inverseDesignTarget ?? "pairing susceptibility"}, PS=${pairingSusc.score.toFixed(3)}, lambda=${pairingSusc.lambda.toFixed(2)}, DOS=${pairingSusc.dosAtEf.toFixed(2)}] ${c.reasoning ?? ""}`,
          electronPhononCoupling: features.electronPhononLambda ?? null,
          logPhononFrequency: features.logPhononFreq ?? null,
          coulombPseudopotential: 0.12,
          pairingMechanism: features.correlationStrength > 0.6 ? "spin-fluctuation" : "phonon-mediated",
          pairingSymmetry: features.correlationStrength > 0.6 ? "d-wave" : (features.dWaveSymmetry ? "d-wave" : "s-wave"),
          correlationStrength: features.correlationStrength ?? null,
          dimensionality: features.layeredStructure ? "quasi-2D" : "3D",
          fermiSurfaceTopology: features.fermiSurfaceType ?? null,
          uncertaintyEstimate: 0.5,
          verificationStage: 0,
          dataConfidence: "medium",
        });
        generated++;
        totalInverseDesignGenerated++;

        emit("prediction", {
          type: "inverse-design-superconductor",
          id,
          name: c.name,
          formula: c.formula,
          predictedTc: cappedTc,
          pairingSusceptibility: pairingSusc.score,
          inverseDesignTarget: c.inverseDesignTarget,
        });
      } catch (e: any) {
        emit("log", { phase: "phase-7", event: "Inverse design insert error", detail: `${c.formula}: ${e.message?.slice(0, 100)}`, dataSource: "Inverse Design" });
      }
    }

    if (generated > 0) {
      emit("log", {
        phase: "phase-7",
        event: "Inverse design candidates created",
        detail: `${generated} candidates designed for optimal pairing susceptibility`,
        dataSource: "Inverse Design",
      });
    }
  } catch (err: any) {
    emit("log", { phase: "phase-7", event: "Inverse design error", detail: err.message?.slice(0, 200), dataSource: "Inverse Design" });
  }

  return generated;
}
