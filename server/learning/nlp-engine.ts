import OpenAI from "openai";
import { storage } from "../storage";
import { batchProcess } from "../replit_integrations/batch/utils";
import type { Material, SuperconductorCandidate } from "@shared/schema";
import type { EventEmitter } from "./engine";
import { sanitizeForbiddenWords, classifyFamily } from "./utils";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

const PHYSICS_RULES: { pattern: RegExp; valid: boolean; reason: string }[] = [
  { pattern: /higher\s+formation\s+energy.*(?:correlat|indicat|lead|suggest).*(?:stab|more stable)/i, valid: false, reason: "Higher formation energy does NOT correlate with stability" },
  { pattern: /(?:increas|higher|more)\s+formation\s+energy.*(?:more|higher|greater)\s+stabil/i, valid: false, reason: "Higher formation energy means LESS stability" },
  { pattern: /(?:negative|lower)\s+band\s*gap.*(?:metal|conduct)/i, valid: false, reason: "Band gap cannot be negative" },
  { pattern: /(?:superconducti|superconduct).*(?:increas|higher).*(?:temperatur|heat)/i, valid: false, reason: "Superconductivity is suppressed, not enhanced, by higher temperature" },
  { pattern: /(?:insulator|semiconductor).*(?:zero|no)\s+(?:resistance|resistiv)/i, valid: false, reason: "Insulators cannot have zero resistance" },
];

function validatePhysicsRules(insight: string): { valid: boolean; reason?: string } {
  for (const rule of PHYSICS_RULES) {
    if (rule.pattern.test(insight) && !rule.valid) {
      return { valid: false, reason: rule.reason };
    }
  }
  return { valid: true };
}

const VAGUE_PATTERNS: RegExp[] = [
  /show\s+varied/i,
  /can\s+have\s+different/i,
  /display\s+varied/i,
  /exhibit\s+(?:varied|various|diverse|different)/i,
  /behave\s+differently/i,
  /show\s+(?:a\s+)?(?:range|variety|mix)/i,
  /materials?\s+(?:can|may|might)\s+(?:be|have|show)/i,
  /(?:some|certain|many|several)\s+materials?\s+(?:have|show|display|exhibit)/i,
  /tend\s+to\s+(?:be|have|show)/i,
];

const STATISTICAL_SUMMARY_PATTERNS: RegExp[] = [
  /\d+\s*%\s+(?:of\s+)?materials?\s+have/i,
  /\d+\s*%\s+(?:of\s+)?materials?\s+(?:are|show|exhibit|display|fall)/i,
  /average\s+band\s*gap/i,
  /average\s+formation\s+energy/i,
  /majority\s+of\s+materials/i,
  /most\s+(?:of\s+the\s+)?materials/i,
  /the\s+dataset\s+(?:contains|shows|has|includes)/i,
  /out\s+of\s+\d+\s+materials/i,
  /\d+\s+out\s+of\s+\d+/i,
  /distribution\s+of\s+(?:band\s*gap|formation|stability)/i,
  /(?:mean|median|mode)\s+(?:value|band\s*gap|formation|stability)/i,
];

const QUANTITATIVE_PATTERN = /\d+\.?\d*\s*(?:eV|K|GPa|%|nm|cm|T\b|meV|A\b|Å)/i;
const SPECIFIC_MATERIAL_PATTERN = /[A-Z][a-z]?\d*[A-Z][a-z]?\d*/;

function isInsightSpecificEnough(insight: string): { valid: boolean; reason?: string } {
  for (const pattern of VAGUE_PATTERNS) {
    if (pattern.test(insight)) {
      return { valid: false, reason: `Vague language: "${insight.slice(0, 60)}..."` };
    }
  }

  for (const pattern of STATISTICAL_SUMMARY_PATTERNS) {
    if (pattern.test(insight)) {
      return { valid: false, reason: `Statistical summary, not a correlation insight: "${insight.slice(0, 60)}..."` };
    }
  }

  const hasNumber = QUANTITATIVE_PATTERN.test(insight);
  const hasMaterial = SPECIFIC_MATERIAL_PATTERN.test(insight);
  const hasCorrelation = /correlat|predict|increas|decreas|higher|lower|stronger|weaker/i.test(insight);

  if (!hasNumber && !hasMaterial && !hasCorrelation) {
    return { valid: false, reason: `Lacks quantitative data, specific materials, or clear correlation` };
  }

  if (insight.length < 30) {
    return { valid: false, reason: `Too short to be meaningful` };
  }

  return { valid: true };
}

const MIN_DATASET_FOR_INSIGHTS = 100;

function pearsonCorrelation(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 5) return 0;
  const avgX = xs.reduce((s, v) => s + v, 0) / n;
  const avgY = ys.reduce((s, v) => s + v, 0) / n;
  let cov = 0, varX = 0, varY = 0;
  for (let i = 0; i < n; i++) {
    cov += (xs[i] - avgX) * (ys[i] - avgY);
    varX += (xs[i] - avgX) ** 2;
    varY += (ys[i] - avgY) ** 2;
  }
  return varX > 0 && varY > 0 ? cov / Math.sqrt(varX * varY) : 0;
}

function computeSuperconductorCorrelations(candidates: SuperconductorCandidate[]): string {
  const stats: string[] = [];
  if (candidates.length < 5) return "";

  stats.push(`\n--- SUPERCONDUCTOR CROSS-PROPERTY CORRELATIONS (${candidates.length} candidates) ---`);

  const withLambdaAndTc = candidates.filter(c => c.electronPhononCoupling != null && c.predictedTc != null);
  if (withLambdaAndTc.length >= 5) {
    const lambdas = withLambdaAndTc.map(c => c.electronPhononCoupling as number);
    const tcs = withLambdaAndTc.map(c => c.predictedTc as number);
    const r = pearsonCorrelation(lambdas, tcs);
    const avgLambdaHighTc = withLambdaAndTc.filter(c => (c.predictedTc ?? 0) > 30).map(c => c.electronPhononCoupling as number);
    const avgLambdaLowTc = withLambdaAndTc.filter(c => (c.predictedTc ?? 0) <= 30).map(c => c.electronPhononCoupling as number);
    stats.push(`Correlation(electron_phonon_coupling λ, predicted_Tc): r=${r.toFixed(3)} (n=${withLambdaAndTc.length})`);
    if (avgLambdaHighTc.length > 0) stats.push(`  Mean λ for Tc>30K: ${(avgLambdaHighTc.reduce((s,v) => s+v, 0) / avgLambdaHighTc.length).toFixed(3)}`);
    if (avgLambdaLowTc.length > 0) stats.push(`  Mean λ for Tc≤30K: ${(avgLambdaLowTc.reduce((s,v) => s+v, 0) / avgLambdaLowTc.length).toFixed(3)}`);
  }

  const withStabAndTc = candidates.filter(c => c.stabilityScore != null && c.predictedTc != null);
  if (withStabAndTc.length >= 5) {
    const stabs = withStabAndTc.map(c => c.stabilityScore as number);
    const tcs = withStabAndTc.map(c => c.predictedTc as number);
    const r = pearsonCorrelation(stabs, tcs);
    stats.push(`Correlation(stability_score, predicted_Tc): r=${r.toFixed(3)} (n=${withStabAndTc.length})`);
  }

  const withCorrAndTc = candidates.filter(c => c.correlationStrength != null && c.predictedTc != null);
  if (withCorrAndTc.length >= 5) {
    const corrs = withCorrAndTc.map(c => c.correlationStrength as number);
    const tcs = withCorrAndTc.map(c => c.predictedTc as number);
    const r = pearsonCorrelation(corrs, tcs);
    stats.push(`Correlation(correlation_strength, predicted_Tc): r=${r.toFixed(3)} (n=${withCorrAndTc.length})`);
  }

  const dimensionGroups: Record<string, number[]> = {};
  for (const c of candidates) {
    if (c.dimensionality && c.predictedTc != null) {
      if (!dimensionGroups[c.dimensionality]) dimensionGroups[c.dimensionality] = [];
      dimensionGroups[c.dimensionality].push(c.predictedTc);
    }
  }
  const dimEntries = Object.entries(dimensionGroups).filter(([, tcs]) => tcs.length >= 2);
  if (dimEntries.length >= 2) {
    const dimStats = dimEntries.map(([dim, tcs]) => {
      const avg = tcs.reduce((s, v) => s + v, 0) / tcs.length;
      const maxTc = Math.max(...tcs);
      return `${dim}: avgTc=${avg.toFixed(1)}K, maxTc=${maxTc.toFixed(1)}K, n=${tcs.length}`;
    });
    stats.push(`Dimensionality vs Tc breakdown: ${dimStats.join("; ")}`);
  }

  const dimLambdaGroups: Record<string, number[]> = {};
  for (const c of candidates) {
    if (c.dimensionality && c.electronPhononCoupling != null) {
      if (!dimLambdaGroups[c.dimensionality]) dimLambdaGroups[c.dimensionality] = [];
      dimLambdaGroups[c.dimensionality].push(c.electronPhononCoupling);
    }
  }
  const dimLambdaEntries = Object.entries(dimLambdaGroups).filter(([, ls]) => ls.length >= 2);
  if (dimLambdaEntries.length >= 2) {
    const dlStats = dimLambdaEntries.map(([dim, ls]) => {
      const avg = ls.reduce((s, v) => s + v, 0) / ls.length;
      return `${dim}: avgλ=${avg.toFixed(3)}, n=${ls.length}`;
    });
    stats.push(`Dimensionality vs λ breakdown: ${dlStats.join("; ")}`);
  }

  const familyGroups: Record<string, { tcs: number[]; lambdas: number[] }> = {};
  for (const c of candidates) {
    const family = classifyFamily(c.formula);
    if (!familyGroups[family]) familyGroups[family] = { tcs: [], lambdas: [] };
    if (c.predictedTc != null) familyGroups[family].tcs.push(c.predictedTc);
    if (c.electronPhononCoupling != null) familyGroups[family].lambdas.push(c.electronPhononCoupling);
  }
  const familyEntries = Object.entries(familyGroups).filter(([, g]) => g.tcs.length >= 2);
  if (familyEntries.length >= 2) {
    const famStats = familyEntries.map(([fam, g]) => {
      const avgTc = g.tcs.length > 0 ? g.tcs.reduce((s, v) => s + v, 0) / g.tcs.length : 0;
      const maxTc = g.tcs.length > 0 ? Math.max(...g.tcs) : 0;
      const avgL = g.lambdas.length > 0 ? g.lambdas.reduce((s, v) => s + v, 0) / g.lambdas.length : 0;
      return `${fam}: avgTc=${avgTc.toFixed(1)}K, maxTc=${maxTc.toFixed(1)}K, avgλ=${avgL.toFixed(3)}, n=${g.tcs.length}`;
    });
    stats.push(`Per-family breakdown: ${famStats.join("; ")}`);
  }

  const mechGroups: Record<string, number[]> = {};
  for (const c of candidates) {
    if (c.pairingMechanism && c.predictedTc != null) {
      if (!mechGroups[c.pairingMechanism]) mechGroups[c.pairingMechanism] = [];
      mechGroups[c.pairingMechanism].push(c.predictedTc);
    }
  }
  const mechEntries = Object.entries(mechGroups).filter(([, tcs]) => tcs.length >= 2);
  if (mechEntries.length >= 2) {
    const mechStats = mechEntries.map(([mech, tcs]) => {
      const avg = tcs.reduce((s, v) => s + v, 0) / tcs.length;
      return `${mech}: avgTc=${avg.toFixed(1)}K, n=${tcs.length}`;
    });
    stats.push(`Pairing mechanism vs Tc: ${mechStats.join("; ")}`);
  }

  return stats.join("\n");
}

function computeDatasetStatistics(materials: Material[]): string {
  const withBG = materials.filter(m => m.bandGap !== null && m.bandGap !== undefined);
  const withFE = materials.filter(m => m.formationEnergy !== null && m.formationEnergy !== undefined);
  const withStab = materials.filter(m => m.stability !== null && m.stability !== undefined);

  const stats: string[] = [];
  stats.push(`Sample size: n=${materials.length}`);

  if (withBG.length >= 10 && withFE.length >= 10) {
    const paired = materials.filter(m => m.bandGap != null && m.formationEnergy != null);
    if (paired.length >= 10) {
      const bgs = paired.map(m => m.bandGap as number);
      const fes = paired.map(m => m.formationEnergy as number);
      const r = pearsonCorrelation(bgs, fes);
      stats.push(`Correlation(band_gap, formation_energy): r=${r.toFixed(3)} (n=${paired.length})`);
    }
  }

  if (withBG.length >= 10 && withStab.length >= 10) {
    const paired = materials.filter(m => m.bandGap != null && m.stability != null);
    if (paired.length >= 10) {
      const bgs = paired.map(m => m.bandGap as number);
      const stabs = paired.map(m => m.stability as number);
      const r = pearsonCorrelation(bgs, stabs);
      stats.push(`Correlation(band_gap, stability): r=${r.toFixed(3)} (n=${paired.length})`);
    }
  }

  if (withFE.length >= 10 && withStab.length >= 10) {
    const paired = materials.filter(m => m.formationEnergy != null && m.stability != null);
    if (paired.length >= 10) {
      const fes = paired.map(m => m.formationEnergy as number);
      const stabs = paired.map(m => m.stability as number);
      const r = pearsonCorrelation(fes, stabs);
      stats.push(`Correlation(formation_energy, stability): r=${r.toFixed(3)} (n=${paired.length})`);
    }
  }

  const elementFreq: Record<string, number> = {};
  for (const m of materials) {
    const els = (m.formula || "").match(/[A-Z][a-z]?/g) || [];
    for (const el of els) elementFreq[el] = (elementFreq[el] || 0) + 1;
  }
  const topElements = Object.entries(elementFreq).sort((a, b) => b[1] - a[1]).slice(0, 10);
  if (topElements.length > 0) {
    stats.push(`Most common elements: ${topElements.map(([el, n]) => `${el}(${n})`).join(", ")}`);
  }

  return stats.join("\n");
}

export async function analyzeBondingPatterns(
  emit: EventEmitter,
  materials: Material[]
): Promise<string[]> {
  if (materials.length === 0) return [];

  const allMats = await storage.getMaterials(2000, 0);
  const dataset = allMats.length >= MIN_DATASET_FOR_INSIGHTS ? allMats : materials;

  if (dataset.length < 30) {
    emit("log", {
      phase: "phase-3",
      event: "Bonding analysis deferred",
      detail: `Dataset too small (${dataset.length} materials, need 30+). Skipping insight generation to avoid hallucinations.`,
      dataSource: "Statistical Analysis",
    });
    return [];
  }

  const dataStats = computeDatasetStatistics(dataset);

  let scCorrelations = "";
  try {
    const candidates = await storage.getSuperconductorCandidates(500);
    scCorrelations = computeSuperconductorCorrelations(candidates);
  } catch (_) {}

  emit("log", {
    phase: "phase-3",
    event: "Bonding statistical analysis started",
    detail: `Analyzing cross-property correlations across ${dataset.length} materials. ${dataStats.split("\n")[0] || ""}`,
    dataSource: "Statistical Analysis",
  });

  const materialSummaries = dataset.slice(0, 100).map((m) => ({
    name: m.name,
    formula: m.formula,
    spacegroup: m.spacegroup,
    bandGap: m.bandGap,
    formationEnergy: m.formationEnergy,
    stability: m.stability,
  }));

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            `You are a condensed matter physics AI specializing in superconductor discovery. Your task is to identify CROSS-PROPERTY CORRELATIONS and PHYSICS RELATIONSHIPS from the provided correlation data.

CRITICAL INSTRUCTIONS:
- Do NOT produce dataset statistics or summaries (e.g., "85% of materials have...", "the average band gap is...").
- Only produce insights about RELATIONSHIPS BETWEEN PROPERTIES (e.g., "Higher electron-phonon coupling λ correlates with elevated Tc in hydrides").
- Each insight must describe a correlation, trend, or causal relationship between two or more physical properties.
- Reference specific material families, dimensionalities, or pairing mechanisms when the data supports it.
- Include quantitative evidence (correlation coefficients, Tc values) from the provided statistics.

PHYSICS RULES:
- Lower (more negative) formation energy = MORE stable
- Band gap cannot be negative
- Superconductivity occurs at LOW temperatures
- Higher λ (electron-phonon coupling) generally predicts higher Tc in conventional superconductors

Return a JSON object with a single key 'insights' containing an array of 3-5 concise cross-property correlation statements (each under 120 characters).`,
        },
        {
          role: "user",
          content: `Cross-property correlation data:\n${dataStats}\n${scCorrelations}\n\nSample materials:\n${JSON.stringify(materialSummaries.slice(0, 50), null, 2)}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      emit("log", { phase: "phase-3", event: "NLP returned empty response", detail: "No content in OpenAI response", dataSource: "OpenAI NLP" });
      return [];
    }

    let parsed: { insights: string[] };
    try {
      parsed = JSON.parse(content);
    } catch (parseErr) {
      emit("log", { phase: "phase-3", event: "NLP JSON parse error", detail: content.slice(0, 200), dataSource: "OpenAI NLP" });
      return [];
    }

    const rawInsights = (parsed.insights ?? []).map(s => sanitizeForbiddenWords(s));
    const insights = rawInsights.filter(insight => {
      const physCheck = validatePhysicsRules(insight);
      if (!physCheck.valid) {
        emit("log", {
          phase: "phase-3",
          event: "Insight rejected (physics violation)",
          detail: `"${insight}" — ${physCheck.reason}`,
          dataSource: "Physics Validator",
        });
        return false;
      }
      const qualCheck = isInsightSpecificEnough(insight);
      if (!qualCheck.valid) {
        emit("log", {
          phase: "phase-3",
          event: "Insight rejected (low quality)",
          detail: `"${insight}" — ${qualCheck.reason}`,
          dataSource: "Quality Filter",
        });
        return false;
      }
      return true;
    });

    if (insights.length > 0) {
      emit("log", {
        phase: "phase-3",
        event: "Bonding patterns discovered",
        detail: insights[0],
        dataSource: "Statistical Analysis",
      });
      emit("insight", { phase: 3, insights });
    }

    return insights;
  } catch (err: any) {
    emit("log", {
      phase: "phase-3",
      event: "NLP analysis error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "Statistical Analysis",
    });
    return [];
  }
}

export async function analyzePropertyPredictionPatterns(
  emit: EventEmitter,
  materials: Material[]
): Promise<string[]> {
  if (materials.length === 0) return [];

  const allMats = await storage.getMaterials(2000, 0);
  const dataset = allMats.length >= MIN_DATASET_FOR_INSIGHTS ? allMats : materials;

  if (dataset.length < 30) {
    emit("log", {
      phase: "phase-5",
      event: "Prediction analysis deferred",
      detail: `Dataset too small (${dataset.length} materials, need 30+). Skipping to avoid unreliable patterns.`,
      dataSource: "Statistical Analysis",
    });
    return [];
  }

  const dataStats = computeDatasetStatistics(dataset);

  let scCorrelations = "";
  try {
    const candidates = await storage.getSuperconductorCandidates(500);
    scCorrelations = computeSuperconductorCorrelations(candidates);
  } catch (_) {}

  emit("log", {
    phase: "phase-5",
    event: "Property prediction statistical analysis started",
    detail: `Analyzing cross-property correlations across ${dataset.length} materials for predictive rules. ${dataStats.split("\n")[0] || ""}`,
    dataSource: "Statistical Analysis",
  });

  const materialSummaries = dataset.slice(0, 100).map((m) => ({
    formula: m.formula,
    bandGap: m.bandGap,
    formationEnergy: m.formationEnergy,
    stability: m.stability,
    source: m.source,
  }));

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            `You are a condensed matter physics AI specializing in superconductor property prediction. Your task is to derive PREDICTIVE RULES from cross-property correlations.

CRITICAL INSTRUCTIONS:
- Do NOT produce dataset statistics or summaries (e.g., "X% of materials have...", "average band gap is...").
- Only produce insights about RELATIONSHIPS BETWEEN PROPERTIES that can predict unknown material behavior.
- Each insight must be a predictive rule linking two or more physical properties (e.g., "Low band gap metals with λ>1.5 predict Tc above 40K in boride families").
- Reference specific material families, element groups, or structural features when the data supports it.
- Include quantitative thresholds from the correlation data.

PHYSICS RULES:
- Lower (more negative) formation energy = MORE stable
- Band gap is always >= 0; metals have near-zero band gap
- Higher λ (electron-phonon coupling) generally predicts higher Tc in conventional superconductors
- Dimensionality affects pairing: 2D materials can have enhanced Tc via nesting effects

Return a JSON object with 'insights' (array of 3-5 concise predictive correlation rules, each under 120 chars) and 'applications' (array of objects with 'pattern' and 'targetProperty' keys).`,
        },
        {
          role: "user",
          content: `Cross-property correlation data:\n${dataStats}\n${scCorrelations}\n\nSample materials:\n${JSON.stringify(materialSummaries.slice(0, 50), null, 2)}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 600,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      emit("log", { phase: "phase-5", event: "NLP returned empty response", detail: "No content in prediction response", dataSource: "Statistical Analysis" });
      return [];
    }

    let parsed: { insights: string[]; applications?: { pattern: string; targetProperty: string }[] };
    try {
      parsed = JSON.parse(content);
    } catch (parseErr) {
      emit("log", { phase: "phase-5", event: "NLP JSON parse error", detail: content.slice(0, 200), dataSource: "Statistical Analysis" });
      return [];
    }

    const rawInsights = (parsed.insights ?? []).map(s => sanitizeForbiddenWords(s));
    const insights = rawInsights.filter(insight => {
      const physCheck = validatePhysicsRules(insight);
      if (!physCheck.valid) {
        emit("log", {
          phase: "phase-5",
          event: "Insight rejected (physics violation)",
          detail: `"${insight}" — ${physCheck.reason}`,
          dataSource: "Physics Validator",
        });
        return false;
      }
      const qualCheck = isInsightSpecificEnough(insight);
      if (!qualCheck.valid) {
        emit("log", {
          phase: "phase-5",
          event: "Insight rejected (low quality)",
          detail: `"${insight}" — ${qualCheck.reason}`,
          dataSource: "Quality Filter",
        });
        return false;
      }
      return true;
    });

    if (insights.length > 0) {
      emit("log", {
        phase: "phase-5",
        event: "Prediction patterns discovered",
        detail: insights[0],
        dataSource: "Statistical Analysis",
      });
      emit("insight", { phase: 5, insights });
    }

    return insights;
  } catch (err: any) {
    emit("log", {
      phase: "phase-5",
      event: "Property prediction error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "Statistical Analysis",
    });
    return [];
  }
}

export async function classifyMaterialApplications(
  emit: EventEmitter,
  materials: Material[]
): Promise<Map<string, string>> {
  const results = new Map<string, string>();
  if (materials.length === 0) return results;

  const batch = materials.slice(0, 10);
  try {
    const classified = await batchProcess(
      batch,
      async (mat) => {
        const response = await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [
            {
              role: "system",
              content:
                'Classify this material into one application category: "energy", "aerospace", "electronics", "biomedical", "construction", or "catalysis". Return JSON with "category" key only.',
            },
            {
              role: "user",
              content: `Material: ${mat.name} (${mat.formula}), band gap: ${mat.bandGap ?? "unknown"} eV, formation energy: ${mat.formationEnergy ?? "unknown"} eV/atom`,
            },
          ],
          response_format: { type: "json_object" },
          max_completion_tokens: 50,
        });
        const content = response.choices[0]?.message?.content;
        if (!content) return { id: mat.id, category: "unknown" };
        const parsed = JSON.parse(content) as { category: string };
        return { id: mat.id, category: parsed.category || "unknown" };
      },
      { concurrency: 2, retries: 3 }
    );

    for (const c of classified) {
      if (c) results.set(c.id, c.category);
    }
  } catch (err: any) {
    emit("log", {
      phase: "phase-5",
      event: "Classification error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "OpenAI NLP",
    });
  }

  return results;
}
