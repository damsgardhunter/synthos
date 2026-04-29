/**
 * DFT Admission Scoring & Selection
 *
 * Decides which candidates get promoted to expensive DFT relaxation.
 * Uses a weighted multi-criteria score balancing exploitation (likely
 * good candidates) and exploration (weird but plausible candidates).
 *
 * Score = 30% confidence + 25% pre-relax enthalpy + 15% diversity
 *       + 10% hydride chemistry + 10% source diversity + 5% novelty
 *       + 5% exploration bonus
 *
 * Selection split (prevents premature convergence):
 *   50% best predicted enthalpy / confidence
 *   20% most diverse cluster representatives
 *   15% hydride-chemistry promising
 *   10% high-uncertainty / underexplored
 *   5%  random survivors
 */

import type { CSPCandidate } from "./csp-types";
import type { StructureCluster } from "./dedup-cluster";
import { analyzeHydrideChemistry } from "./candidate-metadata";

// ---------------------------------------------------------------------------
// Admission score components
// ---------------------------------------------------------------------------

interface AdmissionScore {
  total: number;
  components: {
    confidence: number;       // 30%
    enthalpyRank: number;     // 25%
    diversity: number;        // 15%
    hydrideChemistry: number; // 10%
    sourceDiversity: number;  // 10%
    novelty: number;          // 5%
    exploration: number;      // 5%
  };
  selectionCategory: "exploitation" | "diverse-rep" | "hydride-promising" | "exploration" | "random";
}

/**
 * Score a candidate for DFT admission.
 */
function scoreCandidate(
  candidate: CSPCandidate,
  allCandidates: CSPCandidate[],
  cluster?: StructureCluster,
): AdmissionScore {
  // --- Confidence (30%) ---
  const confidence = Math.min(1.0, candidate.confidence ?? 0.3);

  // --- Pre-relax proxy (25%) ---
  // Before DFT/MLIP, we don't have real enthalpy. Use a composite proxy
  // from geometry + hydride chemistry + volume reasonableness + cluster support.
  // This replaces the fake enthalpyRank that was pretending to rank by energy.
  const vol = candidate.cellVolume ?? 100;
  const volPerAtom = vol / Math.max(1, candidate.positions.length);
  const volReasonable = (volPerAtom >= 5 && volPerAtom <= 30) ? 1.0 :
                        (volPerAtom >= 3 && volPerAtom <= 50) ? 0.7 : 0.3;
  const clusterSupport = cluster ? Math.min(1.0, cluster.size * 0.05) : 0.3;
  const geomFromConf = candidate.confidence ?? 0.5;

  const hydrideForProxy = analyzeHydrideChemistry(candidate);
  const hydrogenProxy = hydrideForProxy?.hasCageStructure ? 1.0 :
                        (hydrideForProxy?.networkType === "extended-chain" ? 0.8 : 0.5);

  const preRelaxProxy =
    0.35 * geomFromConf +
    0.25 * hydrogenProxy +
    0.20 * volReasonable +
    0.10 * (cluster ? Math.min(1.0, cluster.sources.length * 0.25) : 0.3) +
    0.10 * clusterSupport;

  // --- Diversity (15%) ---
  // How different is this candidate from the rest? Use cluster size as proxy:
  // singleton clusters are more diverse, large clusters are less.
  let diversity = 0.5;
  if (cluster) {
    if (cluster.size === 1) diversity = 1.0;
    else if (cluster.size <= 3) diversity = 0.7;
    else if (cluster.size <= 10) diversity = 0.4;
    else diversity = 0.2;
  }

  // --- Hydride chemistry (10%) ---
  let hydrideChemistry = 0.5;
  const hydrideAnalysis = analyzeHydrideChemistry(candidate);
  if (hydrideAnalysis) {
    // Cage structures and extended H networks score highest
    if (hydrideAnalysis.hasCageStructure) hydrideChemistry = 1.0;
    else if (hydrideAnalysis.networkType === "extended-chain") hydrideChemistry = 0.8;
    else if (hydrideAnalysis.networkType === "extended-sheet") hydrideChemistry = 0.8;
    else if (hydrideAnalysis.networkType === "layered-hydride") hydrideChemistry = 0.7;
    else if (hydrideAnalysis.networkType === "interstitial") hydrideChemistry = 0.6;
    else if (hydrideAnalysis.networkType === "H2-molecular") hydrideChemistry = 0.2;
    else if (hydrideAnalysis.networkType === "isolated-H") hydrideChemistry = 0.3;

    // High M-H coordination is good for superconductivity
    if (hydrideAnalysis.mhCoordinationNumber >= 8) hydrideChemistry = Math.min(1.0, hydrideChemistry + 0.2);
  }

  // --- Source diversity (10%) ---
  // Bonus if this cluster was found by multiple engines
  let sourceDiversity = 0.3;
  if (cluster) {
    sourceDiversity = Math.min(1.0, cluster.sources.length * 0.25);
    // Extra bonus if cluster spans multiple Z values
    if (cluster.zValuesSeen.length >= 2) sourceDiversity = Math.min(1.0, sourceDiversity + 0.2);
  }

  // --- Novelty (5%) ---
  // Is this from a mutation or unusual source?
  let novelty = 0.3;
  const proto = candidate.prototype ?? "";
  if (proto.startsWith("mutant-")) novelty = 0.8;
  if (proto === "PyXtal-random") novelty = 0.6;
  if (proto === "AIRSS-buildcell") novelty = 0.5;

  // --- Exploration bonus (5%) ---
  // Random bonus to prevent deterministic selection
  const exploration = 0.2 + Math.random() * 0.6;

  // --- Weighted total ---
  const total =
    0.30 * confidence +
    0.25 * preRelaxProxy +
    0.15 * diversity +
    0.10 * hydrideChemistry +
    0.10 * sourceDiversity +
    0.05 * novelty +
    0.05 * exploration;

  return {
    total,
    components: {
      confidence, enthalpyRank: preRelaxProxy, diversity,
      hydrideChemistry, sourceDiversity, novelty, exploration,
    },
    selectionCategory: "exploitation", // will be assigned during selection
  };
}

// ---------------------------------------------------------------------------
// DFT Selection
// ---------------------------------------------------------------------------

export interface DFTSelectionResult {
  selected: CSPCandidate[];
  scores: Map<CSPCandidate, AdmissionScore>;
  selectionBreakdown: {
    exploitation: number;
    diverseRep: number;
    hydridePromising: number;
    exploration: number;
    random: number;
  };
}

/**
 * Select candidates for DFT from a clustered candidate pool.
 *
 * @param clusters - Structural clusters from dedup-cluster
 * @param nSelect - How many candidates to send to DFT (default 3)
 * @param allCandidates - Full candidate pool (for ranking context)
 * @returns Selected candidates with scores and category assignments
 */
export function selectForDFT(
  clusters: StructureCluster[],
  nSelect: number = 3,
  allCandidates: CSPCandidate[] = [],
): DFTSelectionResult {
  if (allCandidates.length === 0) {
    allCandidates = clusters.flatMap(c => c.members);
  }

  // Score all cluster representatives
  const scored: Array<{ candidate: CSPCandidate; score: AdmissionScore; cluster: StructureCluster }> = [];
  for (const cluster of clusters) {
    const score = scoreCandidate(cluster.representative, allCandidates, cluster);
    scored.push({ candidate: cluster.representative, score, cluster });
  }

  // Selection split
  const nExploitation = Math.max(1, Math.ceil(nSelect * 0.50));
  const nDiverse = Math.max(0, Math.ceil(nSelect * 0.20));
  const nHydride = Math.max(0, Math.ceil(nSelect * 0.15));
  const nExploration = Math.max(0, Math.ceil(nSelect * 0.10));
  const nRandom = Math.max(0, nSelect - nExploitation - nDiverse - nHydride - nExploration);

  const selected: CSPCandidate[] = [];
  const scores = new Map<CSPCandidate, AdmissionScore>();
  const used = new Set<string>();
  const breakdown = { exploitation: 0, diverseRep: 0, hydridePromising: 0, exploration: 0, random: 0 };

  const pick = (candidates: typeof scored, category: AdmissionScore["selectionCategory"], n: number) => {
    for (const s of candidates) {
      if (selected.length >= nSelect) break;
      if (n <= 0) break;
      const key = s.cluster.clusterId;
      if (used.has(key)) continue;
      used.add(key);
      s.score.selectionCategory = category;
      selected.push(s.candidate);
      scores.set(s.candidate, s.score);
      breakdown[category === "diverse-rep" ? "diverseRep" : category === "hydride-promising" ? "hydridePromising" : category]++;
      n--;
    }
  };

  // 1. Exploitation: best overall score
  const byScore = [...scored].sort((a, b) => b.score.total - a.score.total);
  pick(byScore, "exploitation", nExploitation);

  // 2. Diverse representatives: singleton or small clusters
  const byDiversity = [...scored].sort((a, b) =>
    b.score.components.diversity - a.score.components.diversity
  );
  pick(byDiversity, "diverse-rep", nDiverse);

  // 3. Hydride-promising: best hydride chemistry score
  const byHydride = [...scored].sort((a, b) =>
    b.score.components.hydrideChemistry - a.score.components.hydrideChemistry
  );
  pick(byHydride, "hydride-promising", nHydride);

  // 4. Exploration: high uncertainty or mutants
  const byExploration = [...scored].sort((a, b) =>
    (b.score.components.novelty + b.score.components.exploration) -
    (a.score.components.novelty + a.score.components.exploration)
  );
  pick(byExploration, "exploration", nExploration);

  // 5. Random survivors
  const remaining = scored.filter(s => !used.has(s.cluster.clusterId));
  for (let i = remaining.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
  }
  pick(remaining, "random", nRandom);

  // Log selection
  console.log(`[DFT-Admission] Selected ${selected.length}/${clusters.length} clusters for DFT: exploitation=${breakdown.exploitation}, diverse=${breakdown.diverseRep}, hydride=${breakdown.hydridePromising}, explore=${breakdown.exploration}, random=${breakdown.random}`);
  for (const c of selected) {
    const s = scores.get(c)!;
    console.log(`[DFT-Admission]   ${s.selectionCategory}: score=${s.total.toFixed(3)} (conf=${s.components.confidence.toFixed(2)}, H=${s.components.enthalpyRank.toFixed(2)}, div=${s.components.diversity.toFixed(2)}, chem=${s.components.hydrideChemistry.toFixed(2)}) src=${c.prototype}`);
  }

  return { selected, scores, selectionBreakdown: breakdown };
}
