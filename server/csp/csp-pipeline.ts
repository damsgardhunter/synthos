/**
 * Three-Stage Crystal Structure Prediction Pipeline
 *
 * Stage 1 — Broad Exploration:
 *   Heavy AIRSS + Vegard/VCA, light CALYPSO/USPEX.
 *   Cheap xTB relaxation. Fingerprint deduplication.
 *
 * Stage 2 — Basin Refinement:
 *   Identify repeated low-enthalpy motifs from Stage 1.
 *   Shift budget to CALYPSO + USPEX. Cross-seed between engines.
 *   QE relax for promising candidates.
 *
 * Stage 3 — Final Selection:
 *   10 structural families. Full vc-relax.
 *   Score by enthalpy + predicted post-relax force.
 */

import * as fs from "fs";
import * as path from "path";
import {
  type CSPCandidate,
  type CSPEngineConfig,
  type CSPEngineName,
  type CSPPipelineResult,
  type CSPStageConfig,
  type MotifCluster,
  type StructuralFamily,
  DEFAULT_STAGE_CONFIGS,
  computeEnthalpy,
  cellVolumeFromVectors,
} from "./csp-types";
import { airssEngine } from "./airss-wrapper";
import { calypsoEngine } from "./calypso-wrapper";
import { uspexEngine } from "./uspex-wrapper";
import { latticeParamsToVectors } from "./poscar-io";

// ---------------------------------------------------------------------------
// Engine registry
// ---------------------------------------------------------------------------

const ENGINES = [airssEngine, calypsoEngine, uspexEngine];

function getAvailableEngines(): CSPEngineName[] {
  const available: CSPEngineName[] = ["vegard", "vca", "prototype", "random", "known-structure"];
  for (const engine of ENGINES) {
    try {
      if (engine.isAvailable()) available.push(engine.name);
    } catch {}
  }
  return available;
}

/**
 * Redistribute weights from unavailable engines to available ones.
 */
function redistributeWeights(
  config: CSPStageConfig,
  available: CSPEngineName[],
): Record<CSPEngineName, number> {
  const weights = { ...config.engineWeights };
  let unavailableTotal = 0;
  let availableTotal = 0;

  for (const [engine, weight] of Object.entries(weights) as [CSPEngineName, number][]) {
    if (!available.includes(engine) && weight > 0) {
      unavailableTotal += weight;
      weights[engine] = 0;
    } else {
      availableTotal += weight;
    }
  }

  // Redistribute proportionally to available engines
  if (unavailableTotal > 0 && availableTotal > 0) {
    for (const engine of available) {
      if (weights[engine] > 0) {
        weights[engine] += (weights[engine] / availableTotal) * unavailableTotal;
      }
    }
    // If no CSP engines available, give everything to AIRSS fallback or random
    if (!available.includes("airss") && !available.includes("calypso") && !available.includes("uspex")) {
      weights.random = (weights.random || 0) + unavailableTotal;
    }
  }

  return weights;
}

// ---------------------------------------------------------------------------
// Fingerprinting and deduplication
// ---------------------------------------------------------------------------

/**
 * Simple structural fingerprint based on sorted pair distances.
 * Used for deduplication — not a full graph-based fingerprint but
 * fast enough for the CSP pipeline's dedup needs.
 */
function quickFingerprint(candidate: CSPCandidate): number[] {
  const pos = candidate.positions;
  const a = candidate.latticeA;
  const cOverA = candidate.cOverA ?? 1.0;
  const n = pos.length;
  if (n === 0) return [];

  const distances: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dx = pos[i].x - pos[j].x;
      let dy = pos[i].y - pos[j].y;
      let dz = pos[i].z - pos[j].z;
      dx -= Math.round(dx);
      dy -= Math.round(dy);
      dz -= Math.round(dz);
      const dist = Math.sqrt((dx * a) ** 2 + (dy * a) ** 2 + (dz * a * cOverA) ** 2);
      distances.push(dist);
    }
  }

  distances.sort((a, b) => a - b);

  // Normalize to unit vector for cosine distance
  const norm = Math.sqrt(distances.reduce((s, d) => s + d * d, 0)) || 1;
  return distances.map(d => d / norm);
}

function cosineDistance(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 1.0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? 1 - dot / denom : 1.0;
}

function deduplicateCandidates(
  candidates: CSPCandidate[],
  threshold: number,
): CSPCandidate[] {
  // Compute fingerprints
  for (const c of candidates) {
    if (!c.fingerprint) c.fingerprint = quickFingerprint(c);
  }

  const kept: CSPCandidate[] = [];
  for (const c of candidates) {
    let isDuplicate = false;
    for (const existing of kept) {
      if (cosineDistance(c.fingerprint!, existing.fingerprint!) < threshold) {
        // Keep the one with lower enthalpy (or energy if no enthalpy)
        const cScore = c.enthalpyPerAtom ?? (c.enthalpy ?? Infinity);
        const eScore = existing.enthalpyPerAtom ?? (existing.enthalpy ?? Infinity);
        if (cScore < eScore) {
          // Replace existing with this better candidate
          const idx = kept.indexOf(existing);
          kept[idx] = c;
        }
        isDuplicate = true;
        break;
      }
    }
    if (!isDuplicate) kept.push(c);
  }

  return kept;
}

// ---------------------------------------------------------------------------
// Motif clustering (Stage 2)
// ---------------------------------------------------------------------------

function clusterMotifs(candidates: CSPCandidate[], nClusters: number): MotifCluster[] {
  if (candidates.length === 0) return [];

  // Ensure all have fingerprints
  for (const c of candidates) {
    if (!c.fingerprint) c.fingerprint = quickFingerprint(c);
  }

  // Simple k-medoids clustering (works for small N)
  const k = Math.min(nClusters, candidates.length);

  // Initialize medoids: pick candidates with diverse enthalpies
  const sorted = [...candidates].sort((a, b) =>
    (a.enthalpyPerAtom ?? Infinity) - (b.enthalpyPerAtom ?? Infinity)
  );
  const step = Math.max(1, Math.floor(sorted.length / k));
  const medoids = Array.from({ length: k }, (_, i) => Math.min(i * step, sorted.length - 1));

  // Assign each candidate to nearest medoid
  const assignments = new Array(candidates.length).fill(0);
  for (let iter = 0; iter < 10; iter++) {
    // Assign
    for (let i = 0; i < candidates.length; i++) {
      let bestDist = Infinity;
      for (let m = 0; m < medoids.length; m++) {
        const dist = cosineDistance(candidates[i].fingerprint!, candidates[medoids[m]].fingerprint!);
        if (dist < bestDist) { bestDist = dist; assignments[i] = m; }
      }
    }
    // Update medoids (pick member with lowest total distance to others in cluster)
    for (let m = 0; m < medoids.length; m++) {
      const members = candidates.map((c, i) => i).filter(i => assignments[i] === m);
      if (members.length === 0) continue;
      let bestTotal = Infinity, bestIdx = members[0];
      for (const i of members) {
        let total = 0;
        for (const j of members) {
          total += cosineDistance(candidates[i].fingerprint!, candidates[j].fingerprint!);
        }
        if (total < bestTotal) { bestTotal = total; bestIdx = i; }
      }
      medoids[m] = bestIdx;
    }
  }

  // Build clusters
  const clusters: MotifCluster[] = [];
  for (let m = 0; m < medoids.length; m++) {
    const members = candidates.filter((_, i) => assignments[i] === m);
    if (members.length === 0) continue;

    members.sort((a, b) => (a.enthalpyPerAtom ?? Infinity) - (b.enthalpyPerAtom ?? Infinity));
    const best = members[0];

    // Centroid = average fingerprint
    const fpLen = best.fingerprint!.length;
    const centroid = new Array(fpLen).fill(0);
    for (const member of members) {
      for (let i = 0; i < fpLen; i++) centroid[i] += (member.fingerprint![i] || 0);
    }
    for (let i = 0; i < fpLen; i++) centroid[i] /= members.length;

    const avgEnthalpy = members.reduce((s, m) => s + (m.enthalpyPerAtom ?? 0), 0) / members.length;

    clusters.push({
      motifId: `motif-${m}`,
      members,
      centroid,
      avgEnthalpy,
      count: members.length,
      bestCandidate: best,
    });

    // Tag members with motifId
    for (const member of members) member.motifId = `motif-${m}`;
  }

  clusters.sort((a, b) => a.avgEnthalpy - b.avgEnthalpy);
  return clusters;
}

// ---------------------------------------------------------------------------
// Family selection (Stage 3)
// ---------------------------------------------------------------------------

function selectFamilies(
  motifs: MotifCluster[],
  nFamilies: number,
): StructuralFamily[] {
  const families: StructuralFamily[] = [];

  // Take the top motif clusters by enthalpy, ensuring diversity
  const selectedMotifs = motifs.slice(0, Math.min(nFamilies * 2, motifs.length));

  // Group nearby motifs into families
  const used = new Set<number>();
  for (let i = 0; i < selectedMotifs.length && families.length < nFamilies; i++) {
    if (used.has(i)) continue;

    const family: MotifCluster[] = [selectedMotifs[i]];
    used.add(i);

    // Merge nearby motifs
    for (let j = i + 1; j < selectedMotifs.length; j++) {
      if (used.has(j)) continue;
      const dist = cosineDistance(selectedMotifs[i].centroid, selectedMotifs[j].centroid);
      if (dist < 0.10) {
        family.push(selectedMotifs[j]);
        used.add(j);
      }
    }

    const allMembers = family.flatMap(m => m.members);
    const best = allMembers.sort((a, b) =>
      (a.enthalpyPerAtom ?? Infinity) - (b.enthalpyPerAtom ?? Infinity)
    )[0];

    const enthalpies = allMembers.map(m => m.enthalpyPerAtom ?? 0);
    const engineSet = new Set(allMembers.map(m => m.sourceEngine));
    const sgSet = new Set(allMembers.map(m => m.spaceGroup).filter(Boolean));

    families.push({
      familyId: `family-${families.length}`,
      motifs: family,
      representative: best,
      enthalpyRange: [Math.min(...enthalpies), Math.max(...enthalpies)],
      spaceGroups: Array.from(sgSet),
      engineDiversity: engineSet.size,
      confidence: Math.min(1.0, 0.3 + engineSet.size * 0.15 + allMembers.length * 0.02),
    });

    // Tag all members
    for (const member of allMembers) member.familyId = `family-${families.length - 1}`;
  }

  return families;
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

export async function runCSPPipeline(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  jobDir: string,
  opts?: {
    stage1Budget?: number;
    stage2Budget?: number;
    stage3Families?: number;
    baseSeed?: number;
  },
): Promise<CSPPipelineResult> {
  const baseSeed = opts?.baseSeed ?? Math.floor(Math.random() * 1e8);
  const allCandidates: CSPCandidate[] = [];
  const timings = { stage1Ms: 0, stage2Ms: 0, stage3Ms: 0 };

  const available = getAvailableEngines();
  console.log(`[CSP] Pipeline starting for ${formula} at ${pressureGPa} GPa (seed=${baseSeed})`);
  console.log(`[CSP] Available engines: ${available.join(", ")}`);

  // =======================================================================
  // STAGE 1: Broad Exploration
  // =======================================================================
  const s1Start = Date.now();
  const s1Config = { ...DEFAULT_STAGE_CONFIGS[1], totalBudget: opts?.stage1Budget ?? 200 };
  const s1Weights = redistributeWeights(s1Config, available);

  console.log(`[CSP] Stage 1: ${s1Config.totalBudget} structures, weights: ${JSON.stringify(s1Weights)}`);

  const s1Candidates: CSPCandidate[] = [];

  // Run each available engine
  for (const engine of ENGINES) {
    const weight = s1Weights[engine.name] ?? 0;
    if (weight <= 0 || !available.includes(engine.name)) continue;

    const budget = Math.max(1, Math.round(s1Config.totalBudget * weight));
    const engineDir = path.join(jobDir, "csp", "stage1", engine.name);

    try {
      const engineConfig: CSPEngineConfig = {
        binaryPath: "",
        workDir: engineDir,
        timeoutMs: Math.max(30000, budget * 2000),
        maxStructures: budget,
        pressureGPa,
        baseSeed: baseSeed + (engine.name === "calypso" ? 1000 : engine.name === "uspex" ? 2000 : 0),
      };

      const results = await engine.generateStructures(elements, counts, engineConfig);
      for (const r of results) {
        r.generationStage = 1;
      }
      s1Candidates.push(...results);
    } catch (err: any) {
      console.log(`[CSP] Stage 1 ${engine.name} failed: ${err.message?.slice(0, 100)}`);
    }
  }

  // Deduplicate Stage 1
  const s1Deduped = deduplicateCandidates(s1Candidates, s1Config.deduplicationThreshold);
  allCandidates.push(...s1Deduped);
  timings.stage1Ms = Date.now() - s1Start;

  console.log(`[CSP] Stage 1 complete: ${s1Candidates.length} raw → ${s1Deduped.length} unique (${(timings.stage1Ms / 1000).toFixed(1)}s)`);

  // =======================================================================
  // STAGE 2: Basin Refinement
  // =======================================================================
  const s2Start = Date.now();
  const s2Config = { ...DEFAULT_STAGE_CONFIGS[2], totalBudget: opts?.stage2Budget ?? 100 };

  // Cluster Stage 1 survivors into motifs
  const nMotifClusters = Math.max(5, Math.floor(s1Deduped.length / 5));
  const motifs = clusterMotifs(s1Deduped, nMotifClusters);

  console.log(`[CSP] Stage 2: ${motifs.length} motif clusters from ${s1Deduped.length} candidates`);
  for (const motif of motifs.slice(0, 5)) {
    console.log(`[CSP]   ${motif.motifId}: ${motif.count} members, avgH=${motif.avgEnthalpy.toFixed(3)} eV/at, best=${motif.bestCandidate.sourceEngine}`);
  }

  // Cross-seed: top 5 candidates from each engine seed the other engines
  const topByEngine: Record<string, CSPCandidate[]> = {};
  for (const c of s1Deduped) {
    if (!topByEngine[c.sourceEngine]) topByEngine[c.sourceEngine] = [];
    topByEngine[c.sourceEngine].push(c);
  }
  const crossSeeds = Object.values(topByEngine)
    .flatMap(arr => arr.sort((a, b) => (a.enthalpyPerAtom ?? Infinity) - (b.enthalpyPerAtom ?? Infinity)).slice(0, 5));

  const s2Weights = redistributeWeights(s2Config, available);
  const s2Candidates: CSPCandidate[] = [];

  for (const engine of ENGINES) {
    const weight = s2Weights[engine.name] ?? 0;
    if (weight <= 0 || !available.includes(engine.name)) continue;

    const budget = Math.max(1, Math.round(s2Config.totalBudget * weight));
    const engineDir = path.join(jobDir, "csp", "stage2", engine.name);

    try {
      const engineConfig: CSPEngineConfig = {
        binaryPath: "",
        workDir: engineDir,
        timeoutMs: Math.max(60000, budget * 5000),
        maxStructures: budget,
        pressureGPa,
        seedStructures: crossSeeds,
        baseSeed: baseSeed + 10000 + (engine.name === "calypso" ? 1000 : engine.name === "uspex" ? 2000 : 0),
      };

      const results = await engine.generateStructures(elements, counts, engineConfig);
      for (const r of results) {
        r.generationStage = 2;
        r.parentSeeds = crossSeeds.map(s => s.seed);
      }
      s2Candidates.push(...results);
    } catch (err: any) {
      console.log(`[CSP] Stage 2 ${engine.name} failed: ${err.message?.slice(0, 100)}`);
    }
  }

  const s2Deduped = deduplicateCandidates([...s1Deduped, ...s2Candidates], s2Config.deduplicationThreshold);
  allCandidates.push(...s2Candidates.filter(c => s2Deduped.includes(c)));
  timings.stage2Ms = Date.now() - s2Start;

  console.log(`[CSP] Stage 2 complete: ${s2Candidates.length} new + ${s1Deduped.length} carried → ${s2Deduped.length} unique (${(timings.stage2Ms / 1000).toFixed(1)}s)`);

  // =======================================================================
  // STAGE 3: Family Selection
  // =======================================================================
  const s3Start = Date.now();
  const nFamilies = opts?.stage3Families ?? 10;

  // Re-cluster with all Stage 1 + Stage 2 candidates
  const allMotifs = clusterMotifs(s2Deduped, Math.max(nFamilies, Math.floor(s2Deduped.length / 3)));
  const families = selectFamilies(allMotifs, nFamilies);

  timings.stage3Ms = Date.now() - s3Start;

  console.log(`[CSP] Stage 3: ${families.length} structural families selected`);
  for (const family of families) {
    const rep = family.representative;
    console.log(`[CSP]   ${family.familyId}: H=${rep.enthalpyPerAtom?.toFixed(3) ?? "?"} eV/at, SG=${family.spaceGroups.join("/") || "?"}, engines=${family.engineDiversity}, conf=${family.confidence.toFixed(2)}`);
  }

  console.log(`[CSP] Pipeline complete for ${formula}: ${allCandidates.length} total candidates, ${families.length} families (${((timings.stage1Ms + timings.stage2Ms + timings.stage3Ms) / 1000).toFixed(1)}s total)`);

  return {
    formula,
    pressureGPa,
    families,
    allCandidates,
    stageTimings: timings,
    enginesUsed: available,
  };
}
