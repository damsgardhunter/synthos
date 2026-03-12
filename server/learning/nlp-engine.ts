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
  { pattern: /formation\s+energy.*(?:enhance|increas|boost|improv|correlat|predict|indicat).*(?:superconduct|tc|critical\s+temp|pairing)/i, valid: false, reason: "Formation energy measures thermodynamic stability, NOT superconducting tendency" },
  { pattern: /(?:low|negative|lower)\s+formation\s+energy.*(?:enhance|favor|promot|lead).*(?:superconduct|tc|higher\s+tc)/i, valid: false, reason: "Formation energy does not predict superconductivity" },
];

const HC2_PATTERN = /Hc2\s*[=:>]\s*(\d+)/i;
const HC2_FAMILY_THRESHOLDS: Record<string, number> = {
  hydride: 200,
  cuprate: 200,
  "iron-based": 150,
};
const HC2_DEFAULT_THRESHOLD = 100;

function detectFamilyFromText(text: string): string {
  const lower = text.toLowerCase();
  if (/hydride|hydrogen.rich|h\d{2,}|lah|yh|csh/i.test(lower)) return "hydride";
  if (/cuprate|ybco|bscco|cuo|cu.o.*layer/i.test(lower)) return "cuprate";
  if (/iron.based|pnictide|feas|fese|iron.se/i.test(lower)) return "iron-based";
  return "";
}

function validatePhysicsRules(insight: string, family?: string): { valid: boolean; reason?: string } {
  for (const rule of PHYSICS_RULES) {
    if (rule.pattern.test(insight) && !rule.valid) {
      return { valid: false, reason: rule.reason };
    }
  }

  const hc2Match = insight.match(HC2_PATTERN);
  if (hc2Match) {
    const hc2Val = parseInt(hc2Match[1], 10);
    const resolvedFamily = family?.toLowerCase() || detectFamilyFromText(insight);
    const threshold = HC2_FAMILY_THRESHOLDS[resolvedFamily] ?? HC2_DEFAULT_THRESHOLD;
    if (hc2Val > threshold) {
      return { valid: false, reason: `Hc2=${hc2Val}T exceeds ${threshold}T plausibility limit for ${resolvedFamily || "conventional"} superconductors` };
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
const KNOWN_ELEMENTS = new Set([
  "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
  "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
  "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
  "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am",
]);

function isValidChemicalFormula(text: string): boolean {
  const elMatches = text.match(/[A-Z][a-z]?/g);
  if (!elMatches || elMatches.length < 2) return false;
  let validCount = 0;
  for (const el of elMatches) {
    if (KNOWN_ELEMENTS.has(el)) validCount++;
  }
  return validCount >= 2 && validCount / elMatches.length >= 0.5;
}

const SPECIFIC_MATERIAL_PATTERN = /[A-Z][a-z]?\d*[A-Z][a-z]?\d*/;

function isInsightSpecificEnough(insight: string): { valid: boolean; reason?: string } {
  for (const pattern of VAGUE_PATTERNS) {
    if (pattern.test(insight)) {
      return { valid: false, reason: `Vague language: "${insight.slice(0, 60)}..."` };
    }
  }

  for (const pattern of STATISTICAL_SUMMARY_PATTERNS) {
    if (pattern.test(insight)) {
      return { valid: true, reason: `[INFO:batch-summary] ${insight.slice(0, 80)}` };
    }
  }

  const hasNumber = QUANTITATIVE_PATTERN.test(insight);
  const rawMatch = insight.match(SPECIFIC_MATERIAL_PATTERN);
  const hasMaterial = rawMatch ? isValidChemicalFormula(rawMatch[0]) : false;
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
  if (n < 5) return NaN;
  const avgX = xs.reduce((s, v) => s + v, 0) / n;
  const avgY = ys.reduce((s, v) => s + v, 0) / n;
  let cov = 0, varX = 0, varY = 0;
  for (let i = 0; i < n; i++) {
    cov += (xs[i] - avgX) * (ys[i] - avgY);
    varX += (xs[i] - avgX) ** 2;
    varY += (ys[i] - avgY) ** 2;
  }
  if (varX < 1e-12 || varY < 1e-12) return NaN;
  return cov / Math.sqrt(varX * varY);
}

function approxPValue(r: number, n: number): number {
  if (n <= 2 || Math.abs(r) >= 1) return 1;
  const df = n - 2;
  const t = r * Math.sqrt(df / (1 - r * r));
  const x = df / (df + t * t);
  let p = incompleteBetaApprox(df / 2, 0.5, x);
  return Math.min(1, Math.max(0, p));
}

function incompleteBetaApprox(a: number, b: number, x: number): number {
  if (x <= 0) return 1;
  if (x >= 1) return 0;
  const lnBeta = lnGammaApprox(a) + lnGammaApprox(b) - lnGammaApprox(a + b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lnBeta) / a;
  let sum = 0, term = 1;
  for (let k = 0; k < 200; k++) {
    sum += term;
    term *= (k + 1 - b) * x / (k + 1 + a);
    if (Math.abs(term) < 1e-12) break;
  }
  return Math.min(1, front * sum * a);
}

function lnGammaApprox(z: number): number {
  if (z <= 0) return 0;
  const c = [76.18009172947146, -86.50532032941678, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function significanceLabel(p: number): string {
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "n.s.";
}

function formatCorrelation(label: string, r: number, n: number): string | null {
  if (!Number.isFinite(r)) return `${label}: INVARIANT (zero variance in data, n=${n})`;
  const p = approxPValue(r, n);
  const sig = significanceLabel(p);
  return `${label}: r=${r.toFixed(3)}, p=${p < 0.001 ? "<0.001" : p.toFixed(3)} ${sig} (n=${n})`;
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
    stats.push(formatCorrelation("Correlation(electron_phonon_coupling λ, predicted_Tc)", r, withLambdaAndTc.length)!);
    if (avgLambdaHighTc.length > 0) stats.push(`  Mean λ for Tc>30K: ${(avgLambdaHighTc.reduce((s,v) => s+v, 0) / avgLambdaHighTc.length).toFixed(3)}`);
    if (avgLambdaLowTc.length > 0) stats.push(`  Mean λ for Tc≤30K: ${(avgLambdaLowTc.reduce((s,v) => s+v, 0) / avgLambdaLowTc.length).toFixed(3)}`);
  }

  const withStabAndTc = candidates.filter(c => c.stabilityScore != null && c.predictedTc != null);
  if (withStabAndTc.length >= 5) {
    const stabs = withStabAndTc.map(c => c.stabilityScore as number);
    const tcs = withStabAndTc.map(c => c.predictedTc as number);
    const r = pearsonCorrelation(stabs, tcs);
    stats.push(formatCorrelation("Correlation(stability_score, predicted_Tc)", r, withStabAndTc.length)!);
  }

  const withCorrAndTc = candidates.filter(c => c.correlationStrength != null && c.predictedTc != null);
  if (withCorrAndTc.length >= 5) {
    const corrs = withCorrAndTc.map(c => c.correlationStrength as number);
    const tcs = withCorrAndTc.map(c => c.predictedTc as number);
    const r = pearsonCorrelation(corrs, tcs);
    stats.push(formatCorrelation("Correlation(correlation_strength, predicted_Tc)", r, withCorrAndTc.length)!);
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
  const bgs: number[] = [];
  const fes: number[] = [];
  const stabs: number[] = [];
  const bgFePairs: { bg: number; fe: number }[] = [];
  const bgStabPairs: { bg: number; stab: number }[] = [];
  const feStabPairs: { fe: number; stab: number }[] = [];
  const uniqueFormulas = new Set<string>();
  const elementFreq: Record<string, number> = {};

  for (const m of materials) {
    const hasBG = m.bandGap != null;
    const hasFE = m.formationEnergy != null;
    const hasStab = m.stability != null;

    if (hasBG) bgs.push(m.bandGap as number);
    if (hasFE) fes.push(m.formationEnergy as number);
    if (hasStab) stabs.push(m.stability as number);
    if (hasBG && hasFE) bgFePairs.push({ bg: m.bandGap as number, fe: m.formationEnergy as number });
    if (hasBG && hasStab) bgStabPairs.push({ bg: m.bandGap as number, stab: m.stability as number });
    if (hasFE && hasStab) feStabPairs.push({ fe: m.formationEnergy as number, stab: m.stability as number });

    const f = m.formula || "";
    if (f && !uniqueFormulas.has(f)) {
      uniqueFormulas.add(f);
      const els = f.match(/[A-Z][a-z]?/g) || [];
      for (const el of els) elementFreq[el] = (elementFreq[el] || 0) + 1;
    }
  }

  const stats: string[] = [];
  stats.push(`Sample size: n=${materials.length}`);

  if (bgFePairs.length >= 10) {
    const r = pearsonCorrelation(bgFePairs.map(p => p.bg), bgFePairs.map(p => p.fe));
    const line = formatCorrelation("Correlation(band_gap, formation_energy)", r, bgFePairs.length);
    if (line) stats.push(line);
  }

  if (bgStabPairs.length >= 10) {
    const r = pearsonCorrelation(bgStabPairs.map(p => p.bg), bgStabPairs.map(p => p.stab));
    const line = formatCorrelation("Correlation(band_gap, stability)", r, bgStabPairs.length);
    if (line) stats.push(line);
  }

  if (feStabPairs.length >= 10) {
    const r = pearsonCorrelation(feStabPairs.map(p => p.fe), feStabPairs.map(p => p.stab));
    const line = formatCorrelation("Correlation(formation_energy, stability)", r, feStabPairs.length);
    if (line) stats.push(line);
  }

  if (bgs.length >= 10) {
    const mean = bgs.reduce((s, v) => s + v, 0) / bgs.length;
    const std = Math.sqrt(bgs.reduce((s, v) => s + (v - mean) ** 2, 0) / bgs.length);
    stats.push(`Band gap: mean=${mean.toFixed(3)} eV, std=${std.toFixed(3)}, n=${bgs.length}`);
  }
  if (fes.length >= 10) {
    const mean = fes.reduce((s, v) => s + v, 0) / fes.length;
    const std = Math.sqrt(fes.reduce((s, v) => s + (v - mean) ** 2, 0) / fes.length);
    stats.push(`Formation energy: mean=${mean.toFixed(3)} eV/atom, std=${std.toFixed(3)}, n=${fes.length}`);
  }

  const topElements = Object.entries(elementFreq).sort((a, b) => b[1] - a[1]).slice(0, 10);
  if (topElements.length > 0) {
    stats.push(`Most common elements (${uniqueFormulas.size} unique formulas): ${topElements.map(([el, n]) => `${el}(${n})`).join(", ")}`);
  }

  return stats.join("\n");
}

let _datasetStatsCache: { stats: string; timestamp: number; count: number } | null = null;
const STATS_CACHE_TTL = 5 * 60 * 1000;

async function getCachedDatasetStats(emit: EventEmitter, materials: Material[], phase: string): Promise<{ dataset: Material[]; dataStats: string } | null> {
  let dataset: Material[];

  if (_datasetStatsCache && Date.now() - _datasetStatsCache.timestamp < STATS_CACHE_TTL) {
    const newMats = await storage.getMaterials(100, 0);
    if (newMats.length > 0 && newMats.length !== _datasetStatsCache.count) {
      const allMats = await storage.getMaterials(2000, 0);
      dataset = allMats.length >= MIN_DATASET_FOR_INSIGHTS ? allMats : materials;
      _datasetStatsCache = { stats: computeDatasetStatistics(dataset), timestamp: Date.now(), count: allMats.length };
    } else {
      dataset = materials;
    }
  } else {
    const allMats = await storage.getMaterials(2000, 0);
    dataset = allMats.length >= MIN_DATASET_FOR_INSIGHTS ? allMats : materials;
    _datasetStatsCache = { stats: computeDatasetStatistics(dataset), timestamp: Date.now(), count: allMats.length };
  }

  if (dataset.length < 30) {
    emit("log", {
      phase,
      event: "Analysis deferred",
      detail: `Dataset too small (${dataset.length} materials, need 30+). Skipping insight generation to avoid hallucinations.`,
      dataSource: "Statistical Analysis",
    });
    return null;
  }

  return { dataset, dataStats: _datasetStatsCache!.stats };
}

export async function analyzeBondingPatterns(
  emit: EventEmitter,
  materials: Material[]
): Promise<string[]> {
  if (materials.length === 0) return [];

  const cached = await getCachedDatasetStats(emit, materials, "phase-3");
  if (!cached) return [];
  const { dataset, dataStats } = cached;

  let scCorrelations = "";
  try {
    const candidates = await storage.getSuperconductorCandidates(500);
    scCorrelations = computeSuperconductorCorrelations(candidates);
  } catch (err: any) {
    emit("log", {
      phase: "phase-3",
      event: "Superconductor correlation fetch failed",
      detail: `Proceeding without SC correlations: ${err?.message?.slice(0, 150) || "unknown error"}`,
      dataSource: "Statistical Analysis",
    });
  }

  emit("log", {
    phase: "phase-3",
    event: "Bonding statistical analysis started",
    detail: `Analyzing cross-property correlations across ${dataset.length} materials. ${dataStats.split("\n")[0] || ""}`,
    dataSource: "Statistical Analysis",
  });

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
- Include quantitative evidence (correlation coefficients, Tc values, p-values) from the provided statistics.
- Only cite correlations that are statistically significant (p < 0.05). Ignore n.s. (not significant) correlations.
- VARY your insight topics. Rotate through these categories:
  * Phonon softening: How soft modes or anharmonic phonons in specific structures enhance Tc
  * Fermi surface nesting: Nesting vectors, topology effects on pairing strength
  * Charge transfer layers: How charge reservoir layers affect carrier density and Tc
  * Structural motifs: Role of octahedral tilts, layering, cage structures, or rattler atoms
  * Electron density redistribution: Charge localization, orbital hybridization effects on DOS
  * Spin-orbit coupling: SOC effects on band topology and unconventional pairing
  * Pressure-dependent phonon hardening: How lattice stiffening under pressure shifts omega_log

PHYSICS RULES:
- Lower (more negative) formation energy = MORE stable
- Band gap cannot be negative
- Superconductivity occurs at LOW temperatures
- Higher λ (electron-phonon coupling) generally predicts higher Tc in conventional superconductors
- Formation energy measures THERMODYNAMIC STABILITY only. It does NOT predict or enhance superconductivity or Tc. Never claim formation energy correlates with Tc or superconducting properties.
- Hc2 (upper critical field) for conventional superconductors is bounded by ~2*Tc Tesla (WHH limit). For hydrides/cuprates the limit is ~200T.

Return a JSON object with a single key 'insights' containing an array of 3-5 concise cross-property correlation statements (each under 120 characters).`,
        },
        {
          role: "user",
          content: `Cross-property correlation data (with significance levels):\n${dataStats}\n${scCorrelations}`,
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

  const cached = await getCachedDatasetStats(emit, materials, "phase-5");
  if (!cached) return [];
  const { dataset, dataStats } = cached;

  let scCorrelations = "";
  try {
    const candidates = await storage.getSuperconductorCandidates(500);
    scCorrelations = computeSuperconductorCorrelations(candidates);
  } catch (err: any) {
    emit("log", {
      phase: "phase-5",
      event: "Superconductor correlation fetch failed",
      detail: `Proceeding without SC correlations: ${err?.message?.slice(0, 150) || "unknown error"}`,
      dataSource: "Statistical Analysis",
    });
  }

  emit("log", {
    phase: "phase-5",
    event: "Property prediction statistical analysis started",
    detail: `Analyzing cross-property correlations across ${dataset.length} materials for predictive rules. ${dataStats.split("\n")[0] || ""}`,
    dataSource: "Statistical Analysis",
  });

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
- Only cite correlations that are statistically significant (p < 0.05). Ignore n.s. (not significant) correlations.
- VARY your insight topics across these physics categories:
  * Phonon softening effects on Tc predictions
  * Fermi surface nesting strength as a Tc predictor
  * Charge transfer between structural layers and carrier doping
  * Structural motifs (cages, layers, chains) that predict high lambda
  * Electron density at bond critical points as stability predictor
  * Atomic packing fraction effects on phonon spectrum
  * Debye temperature / bulk modulus ratio as Tc screening metric

PHYSICS RULES:
- Lower (more negative) formation energy = MORE stable
- Band gap is always >= 0; metals have near-zero band gap
- Higher λ (electron-phonon coupling) generally predicts higher Tc in conventional superconductors
- Dimensionality affects pairing: 2D materials can have enhanced Tc via nesting effects
- Formation energy measures THERMODYNAMIC STABILITY only. It does NOT predict or correlate with superconductivity or Tc. Never generate rules linking formation energy to Tc or superconducting properties.
- Better Tc predictors: density of states at Fermi level, phonon frequency, electron-phonon coupling λ, dimensionality.

Return a JSON object with 'insights' (array of 3-5 concise predictive correlation rules, each under 120 chars) and 'applications' (array of objects with 'pattern' and 'targetProperty' keys).`,
        },
        {
          role: "user",
          content: `Cross-property correlation data (with significance levels):\n${dataStats}\n${scCorrelations}`,
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

  try {
    const classified = await batchProcess(
      materials,
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
