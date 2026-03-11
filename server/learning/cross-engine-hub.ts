import type { TopologicalAnalysis } from "../physics/topology-engine";
import type { FermiSurfaceResult } from "../physics/fermi-surface-engine";
import type { PairingProfile } from "../physics/pairing-mechanisms";
import type { DefectStructure, ElectronicAdjustment } from "../physics/defect-engine";
import type { PressureTcPoint, HydrideFormationResult } from "./pressure-engine";
import { normalizeFormula } from "./utils";

export interface TopologyInsight {
  topologicalScore: number;
  z2Score: number;
  chernScore: number;
  socStrength: number;
  bandInversionProbability: number;
  flatBandIndicator: number;
  majoranaFeasibility: number;
  topologicalClass: string;
  indicators: string[];
}

export interface FermiInsight {
  nestingScore: number;
  pocketCount: number;
  electronPocketCount: number;
  holePocketCount: number;
  electronHoleBalance: number;
  cylindricalCharacter: number;
  fsDimensionality: number;
  sigmaBandPresence: number;
  multiBandScore: number;
}

export interface PairingInsight {
  dominantMechanism: string;
  secondaryMechanism: string;
  pairingSymmetry: string;
  compositePairingStrength: number;
  phononStrength: number;
  spinStrength: number;
  orbitalStrength: number;
  estimatedTcFromPairing: number;
}

export interface PressureInsight {
  optimalPressure: number;
  maxTc: number;
  retentionFraction: number;
  stabilizationStrategy: string;
  hydridePhaseCount: number;
}

export interface DefectInsight {
  optimalDopants: string[];
  bestTcModifier: number;
  bestDefectType: string;
  variantCount: number;
  bestMutatedFormula: string;
}

export interface PhysicsInsight {
  lambda: number;
  dosAtFermi: number;
  omegaLog: number;
  metallicity: number;
  correlationStrength: number;
  instabilityFlags: string[];
}

export interface TheoryInsight {
  discoveredEquations: string[];
  biases: string[];
  symbolicScore: number;
}

export interface MLInsight {
  predictedTc: number;
  uncertainty: number;
  featureImportance: Record<string, number>;
  confidence: string;
}

export interface SynthesisInsight {
  feasibilityScore: number;
  complexity: number;
  bestMethod: string;
  pathStepCount: number;
}

export interface EngineInsight {
  formula: string;
  timestamp: number;
  topology?: TopologyInsight;
  fermi?: FermiInsight;
  pairing?: PairingInsight;
  pressure?: PressureInsight;
  defect?: DefectInsight;
  physics?: PhysicsInsight;
  theory?: TheoryInsight;
  ml?: MLInsight;
  synthesis?: SynthesisInsight;
}

export type EngineType =
  | "topology"
  | "fermi"
  | "pairing"
  | "pressure"
  | "defect"
  | "physics"
  | "theory"
  | "ml"
  | "synthesis";

export interface CrossEnginePattern {
  name: string;
  description: string;
  frequency: number;
  avgTcBoost: number;
  involvedEngines: EngineType[];
  formulaExamples: string[];
}

export interface TopologicalGuidance {
  prioritizeSOC: boolean;
  suggestLayered: boolean;
  topologicalClass: string;
  majoranaViable: boolean;
  bandInversionLikely: boolean;
  recommendedElements: string[];
}

export interface SynthesisGuidanceResult {
  recommendedMethod: string;
  recommendedAtmosphere: string;
  optimalTemperatureRange: [number, number];
  pressureRequired: boolean;
  dopingRecommended: boolean;
  dopantSuggestions: string[];
  annealingGuidance: string;
}

export interface PhysicsGuidanceResult {
  lambdaTarget: number;
  nestingImportant: boolean;
  correlationRegime: string;
  pairingMechanism: string;
  dimensionalityPreference: number;
  phononFrequencyTarget: number;
}

const MAX_INSIGHTS_PER_FORMULA = 50;
const MAX_FORMULAS = 500;
const PATTERN_MIN_OCCURRENCES = 3;

class CrossEngineHub {
  private insightStore: Map<string, EngineInsight> = new Map();
  private insightHistory: Map<string, EngineInsight[]> = new Map();
  private patternCache: CrossEnginePattern[] = [];
  private patternCacheTime = 0;
  private readonly PATTERN_CACHE_TTL = 60000;
  private totalInsightsRecorded = 0;

  recordInsight(engine: EngineType, formula: string, data: any): void {
    if (!formula || typeof formula !== "string") return;
    const normalized = normalizeFormula(formula);

    let insight = this.insightStore.get(normalized);
    if (!insight) {
      insight = { formula: normalized, timestamp: Date.now() };
      this.insightStore.set(normalized, insight);
    }
    insight.timestamp = Date.now();

    switch (engine) {
      case "topology":
        insight.topology = this.extractTopologyInsight(data);
        break;
      case "fermi":
        insight.fermi = this.extractFermiInsight(data);
        break;
      case "pairing":
        insight.pairing = this.extractPairingInsight(data);
        break;
      case "pressure":
        insight.pressure = this.extractPressureInsight(data);
        break;
      case "defect":
        insight.defect = this.extractDefectInsight(data);
        break;
      case "physics":
        insight.physics = this.extractPhysicsInsight(data);
        break;
      case "theory":
        insight.theory = this.extractTheoryInsight(data);
        break;
      case "ml":
        insight.ml = this.extractMLInsight(data);
        break;
      case "synthesis":
        insight.synthesis = this.extractSynthesisInsight(data);
        break;
    }

    let history = this.insightHistory.get(normalized);
    if (!history) {
      history = [];
      this.insightHistory.set(normalized, history);
    }
    history.push({ ...insight });
    if (history.length > MAX_INSIGHTS_PER_FORMULA) {
      history.splice(0, history.length - MAX_INSIGHTS_PER_FORMULA);
    }

    if (this.insightStore.size > MAX_FORMULAS) {
      this.evictOldest();
    }

    this.totalInsightsRecorded++;
    this.patternCacheTime = 0;
  }

  getInsightsFor(formula: string): EngineInsight | null {
    return this.insightStore.get(normalizeFormula(formula)) ?? null;
  }

  getInsightHistoryFor(formula: string): EngineInsight[] {
    return this.insightHistory.get(normalizeFormula(formula)) ?? [];
  }

  getAllFormulas(): string[] {
    return Array.from(this.insightStore.keys());
  }

  getGlobalPatterns(): CrossEnginePattern[] {
    if (Date.now() - this.patternCacheTime < this.PATTERN_CACHE_TTL && this.patternCache.length > 0) {
      return this.patternCache;
    }

    const patterns: CrossEnginePattern[] = [];
    const insights = Array.from(this.insightStore.values());

    patterns.push(...this.detectTopologyNestingPattern(insights));
    patterns.push(...this.detectDefectPressurePattern(insights));
    patterns.push(...this.detectPairingTopologyPattern(insights));
    patterns.push(...this.detectFermiPhysicsPattern(insights));
    patterns.push(...this.detectMLPhysicsConsistencyPattern(insights));
    patterns.push(...this.detectMultiEngineHighTcPattern(insights));

    this.patternCache = patterns.filter(p => p.frequency >= PATTERN_MIN_OCCURRENCES);
    this.patternCacheTime = Date.now();
    return this.patternCache;
  }

  getTopologicalGuidance(): TopologicalGuidance {
    const insights = Array.from(this.insightStore.values());
    const topoInsights = insights.filter(i => i.topology);

    if (topoInsights.length === 0) {
      return {
        prioritizeSOC: false,
        suggestLayered: false,
        topologicalClass: "trivial",
        majoranaViable: false,
        bandInversionLikely: false,
        recommendedElements: [],
      };
    }

    const avgSOC = topoInsights.reduce((s, i) => s + (i.topology?.socStrength ?? 0), 0) / topoInsights.length;
    const avgBandInversion = topoInsights.reduce((s, i) => s + (i.topology?.bandInversionProbability ?? 0), 0) / topoInsights.length;
    const avgMajorana = topoInsights.reduce((s, i) => s + (i.topology?.majoranaFeasibility ?? 0), 0) / topoInsights.length;

    const highTcTopo = topoInsights
      .filter(i => i.ml && i.ml.predictedTc > 50 && i.topology && i.topology.topologicalScore > 0.3)
      .sort((a, b) => (b.ml?.predictedTc ?? 0) - (a.ml?.predictedTc ?? 0));

    const bestClass = highTcTopo.length > 0
      ? highTcTopo[0].topology?.topologicalClass ?? "trivial"
      : topoInsights.sort((a, b) => (b.topology?.topologicalScore ?? 0) - (a.topology?.topologicalScore ?? 0))[0]?.topology?.topologicalClass ?? "trivial";

    const layeredCount = topoInsights.filter(i =>
      i.topology?.indicators.some(ind => ind.includes("layered"))
    ).length;

    const recommendedElements: string[] = [];

    const CLASS_ELEMENTS: Record<string, string[]> = {
      "topological_insulator": ["Bi", "Sb", "Te", "Se", "Sn"],
      "weyl_semimetal": ["Ta", "Nb", "As", "P", "W"],
      "dirac_semimetal": ["Cd", "As", "Na", "Bi", "Zn"],
      "nodal_line": ["Ca", "Sr", "Si", "Ge", "Pb"],
      "topological_superconductor": ["Cu", "Bi", "Sr", "Ir", "Pt"],
    };

    const classEls = CLASS_ELEMENTS[bestClass];
    if (classEls) {
      recommendedElements.push(...classEls);
    } else {
      if (avgSOC > 0.3) recommendedElements.push("Bi", "Pb", "Ir", "Pt");
      if (avgBandInversion > 0.3) recommendedElements.push("Sn", "Te", "Se");
    }

    if (avgMajorana > 0.3 && !recommendedElements.includes("Ir")) recommendedElements.push("Ir");

    const uniqueElements = Array.from(new Set(recommendedElements));

    return {
      prioritizeSOC: avgSOC > 0.3,
      suggestLayered: layeredCount > topoInsights.length * 0.3,
      topologicalClass: bestClass,
      majoranaViable: avgMajorana > 0.3,
      bandInversionLikely: avgBandInversion > 0.3,
      recommendedElements: uniqueElements,
    };
  }

  getSynthesisGuidance(): SynthesisGuidanceResult {
    const insights = Array.from(this.insightStore.values());

    const withPairing = insights.filter(i => i.pairing);
    const withPressure = insights.filter(i => i.pressure);
    const withDefect = insights.filter(i => i.defect);
    const withFermi = insights.filter(i => i.fermi);

    let recommendedMethod = "solid-state";
    let recommendedAtmosphere = "argon";
    let optimalTempRange: [number, number] = [800, 1200];
    let pressureRequired = false;
    let dopingRecommended = false;
    const dopantSuggestions: string[] = [];
    let annealingGuidance = "Standard anneal at 600-800K for 12-24 hours";

    if (withPairing.length > 0) {
      const dominant = this.getMostCommonValue(
        withPairing.map(i => i.pairing?.dominantMechanism ?? "phonon")
      );

      if (dominant === "phonon") {
        recommendedAtmosphere = "argon";
        recommendedMethod = "arc-melting";
      } else if (dominant === "spin") {
        recommendedAtmosphere = "oxygen";
        recommendedMethod = "heat-treatment";
        optimalTempRange = [850, 1000];
      }
    }

    if (withPressure.length > 0) {
      const avgOptP = withPressure.reduce((s, i) => s + (i.pressure?.optimalPressure ?? 0), 0) / withPressure.length;
      if (avgOptP > 50) {
        pressureRequired = true;
        recommendedMethod = "high-pressure";
      }
    }

    if (withDefect.length > 0) {
      const bestDopants = withDefect
        .filter(i => i.defect && i.defect.bestTcModifier > 1.0)
        .flatMap(i => i.defect?.optimalDopants ?? []);
      if (bestDopants.length > 0) {
        dopingRecommended = true;
        const uniqueDopants = Array.from(new Set(bestDopants));
        dopantSuggestions.push(...uniqueDopants.slice(0, 5));
      }
    }

    if (withFermi.length > 0) {
      const avgNesting = withFermi.reduce((s, i) => s + (i.fermi?.nestingScore ?? 0), 0) / withFermi.length;
      if (avgNesting > 0.5) {
        annealingGuidance = "Extended anneal near nesting-driven instability temperature (400-600K) for 24-48 hours to optimize Fermi surface topology";
      }
    }

    return {
      recommendedMethod,
      recommendedAtmosphere,
      optimalTemperatureRange: optimalTempRange,
      pressureRequired,
      dopingRecommended,
      dopantSuggestions,
      annealingGuidance,
    };
  }

  getPhysicsGuidance(): PhysicsGuidanceResult {
    const insights = Array.from(this.insightStore.values());
    const withPhysics = insights.filter(i => i.physics);
    const withPairing = insights.filter(i => i.pairing);
    const withFermi = insights.filter(i => i.fermi);

    if (withPhysics.length === 0) {
      return {
        lambdaTarget: 1.0,
        nestingImportant: false,
        correlationRegime: "weak",
        pairingMechanism: "phonon",
        dimensionalityPreference: 3,
        phononFrequencyTarget: 300,
      };
    }

    const highTcInsights = insights
      .filter(i => i.ml && i.ml.predictedTc > 30 && i.physics)
      .sort((a, b) => (b.ml?.predictedTc ?? 0) - (a.ml?.predictedTc ?? 0));

    const targetInsights = highTcInsights.length > 3 ? highTcInsights.slice(0, 10) : withPhysics;

    const avgLambda = targetInsights.reduce((s, i) => s + (i.physics?.lambda ?? 0), 0) / targetInsights.length;
    const avgOmega = targetInsights.reduce((s, i) => s + (i.physics?.omegaLog ?? 0), 0) / targetInsights.length;
    const avgCorr = targetInsights.reduce((s, i) => s + (i.physics?.correlationStrength ?? 0), 0) / targetInsights.length;

    let nestingImportant = false;
    if (withFermi.length > 0) {
      const avgNesting = withFermi.reduce((s, i) => s + (i.fermi?.nestingScore ?? 0), 0) / withFermi.length;
      nestingImportant = avgNesting > 0.4;
    }

    let correlationRegime = "weak";
    if (avgCorr > 0.7) correlationRegime = "strong";
    else if (avgCorr > 0.4) correlationRegime = "intermediate";

    let pairingMechanism = "phonon";
    if (withPairing.length > 0) {
      pairingMechanism = this.getMostCommonValue(
        withPairing.map(i => i.pairing?.dominantMechanism ?? "phonon")
      );
    }

    let dimensionalityPreference = 3;
    if (withFermi.length > 0) {
      const avgDim = withFermi.reduce((s, i) => s + (i.fermi?.fsDimensionality ?? 3), 0) / withFermi.length;
      dimensionalityPreference = Math.round(avgDim * 10) / 10;
    }

    return {
      lambdaTarget: Math.round(avgLambda * 100) / 100,
      nestingImportant,
      correlationRegime,
      pairingMechanism,
      dimensionalityPreference,
      phononFrequencyTarget: Math.round(avgOmega),
    };
  }

  private extractTopologyInsight(data: any): TopologyInsight {
    return {
      topologicalScore: data.topologicalScore ?? 0,
      z2Score: data.z2Score ?? 0,
      chernScore: data.chernScore ?? 0,
      socStrength: data.socStrength ?? 0,
      bandInversionProbability: data.bandInversionProbability ?? 0,
      flatBandIndicator: data.flatBandIndicator ?? 0,
      majoranaFeasibility: data.majoranaFeasibility ?? 0,
      topologicalClass: data.topologicalClass ?? "trivial",
      indicators: data.indicators ?? [],
    };
  }

  private extractFermiInsight(data: any): FermiInsight {
    return {
      nestingScore: data.nestingScore ?? 0,
      pocketCount: data.pocketCount ?? 0,
      electronPocketCount: data.electronPocketCount ?? 0,
      holePocketCount: data.holePocketCount ?? 0,
      electronHoleBalance: data.electronHoleBalance ?? 0,
      cylindricalCharacter: data.cylindricalCharacter ?? 0,
      fsDimensionality: data.fsDimensionality ?? 3,
      sigmaBandPresence: data.sigmaBandPresence ?? 0,
      multiBandScore: data.multiBandScore ?? 0,
    };
  }

  private extractPairingInsight(data: any): PairingInsight {
    return {
      dominantMechanism: data.dominantMechanism ?? "phonon",
      secondaryMechanism: data.secondaryMechanism ?? "none",
      pairingSymmetry: data.pairingSymmetry ?? "s-wave",
      compositePairingStrength: data.compositePairingStrength ?? 0,
      phononStrength: data.phonon?.phononPairingStrength ?? data.phononStrength ?? 0,
      spinStrength: data.spin?.spinPairingStrength ?? data.spinStrength ?? 0,
      orbitalStrength: data.orbital?.orbitalPairingStrength ?? data.orbitalStrength ?? 0,
      estimatedTcFromPairing: data.estimatedTcFromPairing ?? 0,
    };
  }

  private extractPressureInsight(data: any): PressureInsight {
    const curve: PressureTcPoint[] = data.pressureTcCurve ?? [];
    const stablePoints = curve.filter((p: PressureTcPoint) => p.stable);
    const ambientTc = curve.find((p: PressureTcPoint) => p.pressure === 0)?.Tc ?? 0;
    const maxTc = data.maxTc ?? 0;
    const retentionFraction = maxTc > 0 && ambientTc > 0 ? ambientTc / maxTc : 0;

    let stabilizationStrategy = "ambient-stable";
    if (data.optimalPressure > 100) stabilizationStrategy = "high-pressure-required";
    else if (data.optimalPressure > 20) stabilizationStrategy = "moderate-pressure";
    else if (data.optimalPressure > 0) stabilizationStrategy = "low-pressure-enhancement";

    return {
      optimalPressure: data.optimalPressure ?? 0,
      maxTc,
      retentionFraction: Math.round(retentionFraction * 1000) / 1000,
      stabilizationStrategy,
      hydridePhaseCount: data.hydrideFormation?.stableHydrides?.length ?? 0,
    };
  }

  private extractDefectInsight(data: any): DefectInsight {
    const variants: DefectStructure[] = Array.isArray(data) ? data : data.variants ?? [];
    const adjustments: { variant: DefectStructure; adj: ElectronicAdjustment }[] = data.adjustments ?? [];

    let bestTcModifier = 1.0;
    let bestDefectType = "none";
    let bestMutatedFormula = "";
    const optimalDopants: string[] = [];

    if (adjustments.length > 0) {
      for (const { variant, adj } of adjustments) {
        if (adj.tcModifier > bestTcModifier) {
          bestTcModifier = adj.tcModifier;
          bestDefectType = variant.type;
          bestMutatedFormula = variant.mutatedFormula;
        }
        if (variant.type === "dopant" && adj.tcModifier > 1.0) {
          optimalDopants.push(variant.element);
        }
      }
    } else if (variants.length > 0) {
      for (const v of variants) {
        if (v.type === "dopant") {
          optimalDopants.push(v.element);
        }
      }
      bestMutatedFormula = variants[0]?.mutatedFormula ?? "";
      bestDefectType = variants[0]?.type ?? "none";
    }

    return {
      optimalDopants: Array.from(new Set(optimalDopants)).slice(0, 5),
      bestTcModifier: Math.round(bestTcModifier * 1000) / 1000,
      bestDefectType,
      variantCount: variants.length,
      bestMutatedFormula,
    };
  }

  private extractPhysicsInsight(data: any): PhysicsInsight {
    const instabilityFlags: string[] = [];
    if (data.hasImaginaryModes || data.phonon?.hasImaginaryModes) instabilityFlags.push("imaginary-phonon-modes");
    if (data.softModePresent || data.phonon?.softModePresent) instabilityFlags.push("soft-mode");
    if ((data.correlationStrength ?? data.correlation?.ratio ?? 0) > 0.8) instabilityFlags.push("strong-correlation");
    if ((data.metallicity ?? 0) < 0.2) instabilityFlags.push("low-metallicity");

    return {
      lambda: data.lambda ?? data.coupling?.lambda ?? 0,
      dosAtFermi: data.densityOfStatesAtFermi ?? data.electronic?.densityOfStatesAtFermi ?? 0,
      omegaLog: data.omegaLog ?? data.coupling?.omegaLog ?? 0,
      metallicity: data.metallicity ?? data.electronic?.metallicity ?? 0,
      correlationStrength: data.correlationStrength ?? data.correlation?.ratio ?? 0,
      instabilityFlags,
    };
  }

  private extractTheoryInsight(data: any): TheoryInsight {
    return {
      discoveredEquations: data.discoveredEquations ?? data.equations ?? [],
      biases: data.biases ?? [],
      symbolicScore: data.symbolicScore ?? data.score ?? 0,
    };
  }

  private extractMLInsight(data: any): MLInsight {
    return {
      predictedTc: data.predictedTc ?? data.tcEstimate ?? 0,
      uncertainty: data.uncertainty ?? data.predictedStd ?? 0,
      featureImportance: data.featureImportance ?? {},
      confidence: data.confidence ?? "low",
    };
  }

  private extractSynthesisInsight(data: any): SynthesisInsight {
    return {
      feasibilityScore: data.feasibilityScore ?? 0,
      complexity: data.overallComplexity ?? data.complexity ?? 0,
      bestMethod: data.steps?.[0]?.method ?? data.bestMethod ?? "unknown",
      pathStepCount: data.steps?.length ?? data.pathStepCount ?? 0,
    };
  }

  private detectTopologyNestingPattern(insights: EngineInsight[]): CrossEnginePattern[] {
    const patterns: CrossEnginePattern[] = [];
    const matching = insights.filter(i =>
      i.topology && i.fermi &&
      i.topology.topologicalScore > 0.3 &&
      i.fermi.nestingScore > 0.4
    );

    if (matching.length >= PATTERN_MIN_OCCURRENCES) {
      const avgTcBoost = matching
        .filter(i => i.ml)
        .reduce((s, i) => s + (i.ml?.predictedTc ?? 0), 0) / Math.max(1, matching.filter(i => i.ml).length);

      patterns.push({
        name: "topological-nesting-synergy",
        description: "Topological band features combined with strong Fermi surface nesting correlate with enhanced Tc",
        frequency: matching.length,
        avgTcBoost: Math.round(avgTcBoost * 10) / 10,
        involvedEngines: ["topology", "fermi"],
        formulaExamples: matching.slice(0, 5).map(i => i.formula),
      });
    }

    return patterns;
  }

  private detectDefectPressurePattern(insights: EngineInsight[]): CrossEnginePattern[] {
    const patterns: CrossEnginePattern[] = [];
    const matching = insights.filter(i =>
      i.defect && i.pressure &&
      i.defect.bestTcModifier > 1.0 &&
      i.pressure.stabilizationStrategy !== "high-pressure-required"
    );

    if (matching.length >= PATTERN_MIN_OCCURRENCES) {
      patterns.push({
        name: "defect-ambient-stabilization",
        description: "Doping-enhanced materials that maintain stability at ambient or low pressure show improved practical viability",
        frequency: matching.length,
        avgTcBoost: matching.reduce((s, i) => s + ((i.defect?.bestTcModifier ?? 1) - 1) * 100, 0) / matching.length,
        involvedEngines: ["defect", "pressure"],
        formulaExamples: matching.slice(0, 5).map(i => i.formula),
      });
    }

    return patterns;
  }

  private detectPairingTopologyPattern(insights: EngineInsight[]): CrossEnginePattern[] {
    const patterns: CrossEnginePattern[] = [];
    const matching = insights.filter(i =>
      i.pairing && i.topology &&
      i.pairing.compositePairingStrength > 0.5 &&
      i.topology.topologicalScore > 0.3
    );

    if (matching.length >= PATTERN_MIN_OCCURRENCES) {
      const avgTcFromPairing = matching.reduce((s, i) => s + (i.pairing?.estimatedTcFromPairing ?? 0), 0) / matching.length;

      patterns.push({
        name: "pairing-topology-enhancement",
        description: "Strong pairing mechanisms in topologically non-trivial materials suggest enhanced or unconventional superconductivity",
        frequency: matching.length,
        avgTcBoost: Math.round(avgTcFromPairing * 10) / 10,
        involvedEngines: ["pairing", "topology"],
        formulaExamples: matching.slice(0, 5).map(i => i.formula),
      });
    }

    return patterns;
  }

  private detectFermiPhysicsPattern(insights: EngineInsight[]): CrossEnginePattern[] {
    const patterns: CrossEnginePattern[] = [];
    const matching = insights.filter(i =>
      i.fermi && i.physics &&
      i.fermi.multiBandScore > 0.4 &&
      i.physics.lambda > 0.8
    );

    if (matching.length >= PATTERN_MIN_OCCURRENCES) {
      const avgLambda = matching.reduce((s, i) => s + (i.physics?.lambda ?? 0), 0) / matching.length;

      patterns.push({
        name: "multiband-strong-coupling",
        description: "Multi-band Fermi surfaces with strong electron-phonon coupling indicate robust conventional superconductivity",
        frequency: matching.length,
        avgTcBoost: Math.round(avgLambda * 30),
        involvedEngines: ["fermi", "physics"],
        formulaExamples: matching.slice(0, 5).map(i => i.formula),
      });
    }

    return patterns;
  }

  private detectMLPhysicsConsistencyPattern(insights: EngineInsight[]): CrossEnginePattern[] {
    const patterns: CrossEnginePattern[] = [];
    const matching = insights.filter(i =>
      i.ml && i.physics &&
      i.ml.predictedTc > 20 &&
      i.physics.lambda > 0.5 &&
      i.physics.metallicity > 0.5
    );

    if (matching.length >= PATTERN_MIN_OCCURRENCES) {
      const avgTc = matching.reduce((s, i) => s + (i.ml?.predictedTc ?? 0), 0) / matching.length;

      patterns.push({
        name: "ml-physics-consistent-candidates",
        description: "Candidates where ML predictions align with physics engine outputs (metallic, significant lambda) are high-confidence",
        frequency: matching.length,
        avgTcBoost: Math.round(avgTc * 10) / 10,
        involvedEngines: ["ml", "physics"],
        formulaExamples: matching.slice(0, 5).map(i => i.formula),
      });
    }

    return patterns;
  }

  private detectMultiEngineHighTcPattern(insights: EngineInsight[]): CrossEnginePattern[] {
    const patterns: CrossEnginePattern[] = [];

    const matching = insights.filter(i => {
      let engineCount = 0;
      if (i.topology && i.topology.topologicalScore > 0.2) engineCount++;
      if (i.fermi && i.fermi.nestingScore > 0.3) engineCount++;
      if (i.pairing && i.pairing.compositePairingStrength > 0.4) engineCount++;
      if (i.physics && i.physics.lambda > 0.5) engineCount++;
      if (i.ml && i.ml.predictedTc > 30) engineCount++;
      return engineCount >= 4;
    });

    if (matching.length >= PATTERN_MIN_OCCURRENCES) {
      const avgTc = matching
        .filter(i => i.ml)
        .reduce((s, i) => s + (i.ml?.predictedTc ?? 0), 0) / Math.max(1, matching.filter(i => i.ml).length);

      const enginesInvolved: EngineType[] = ["topology", "fermi", "pairing", "physics", "ml"];

      patterns.push({
        name: "multi-engine-convergence",
        description: "Candidates scoring highly across 4+ engines represent the most promising discoveries with cross-validated properties",
        frequency: matching.length,
        avgTcBoost: Math.round(avgTc * 10) / 10,
        involvedEngines: enginesInvolved,
        formulaExamples: matching.slice(0, 5).map(i => i.formula),
      });
    }

    return patterns;
  }

  private getMostCommonValue(values: string[]): string {
    const counts: Record<string, number> = {};
    for (const v of values) {
      counts[v] = (counts[v] || 0) + 1;
    }
    let best = values[0] ?? "";
    let bestCount = 0;
    for (const [v, c] of Object.entries(counts)) {
      if (c > bestCount) {
        bestCount = c;
        best = v;
      }
    }
    return best;
  }

  getStats() {
    const insights = Array.from(this.insightStore.values());
    const withTopology = insights.filter(i => i.topology).length;
    const withFermi = insights.filter(i => i.fermi).length;
    const withPairing = insights.filter(i => i.pairing).length;
    const withPressure = insights.filter(i => i.pressure).length;
    const withDefect = insights.filter(i => i.defect).length;
    const withPhysics = insights.filter(i => i.physics).length;
    const withML = insights.filter(i => i.ml).length;
    const withSynthesis = insights.filter(i => i.synthesis).length;
    const withTheory = insights.filter(i => i.theory).length;

    const multiEngine = insights.filter(i => {
      let count = 0;
      if (i.topology) count++;
      if (i.fermi) count++;
      if (i.pairing) count++;
      if (i.physics) count++;
      if (i.ml) count++;
      if (i.defect) count++;
      if (i.pressure) count++;
      if (i.synthesis) count++;
      return count >= 3;
    }).length;

    const patterns = this.getGlobalPatterns();

    return {
      totalFormulas: this.insightStore.size,
      totalInsightsRecorded: this.totalInsightsRecorded,
      currentActiveInsights: this.insightStore.size,
      totalHistoryEntries: Array.from(this.insightHistory.values()).reduce((s, h) => s + h.length, 0),
      engineCoverage: {
        topology: withTopology,
        fermi: withFermi,
        pairing: withPairing,
        pressure: withPressure,
        defect: withDefect,
        physics: withPhysics,
        ml: withML,
        synthesis: withSynthesis,
        theory: withTheory,
      },
      multiEngineFormulas: multiEngine,
      activePatterns: patterns.length,
      patternNames: patterns.map(p => p.name),
    };
  }

  private evictOldest(): void {
    const entries = Array.from(this.insightStore.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
    const toRemove = Math.floor(entries.length * 0.2);
    for (let i = 0; i < toRemove; i++) {
      this.insightStore.delete(entries[i][0]);
      this.insightHistory.delete(entries[i][0]);
    }
  }
}

export const crossEngineHub = new CrossEngineHub();
