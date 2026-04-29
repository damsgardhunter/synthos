/**
 * Cage-Aware Structure Seeding
 *
 * Two strategies for generating cage hydride candidates:
 *
 * 1. Parent Seeding: Take known cage structures (LaH10, CaH6, YH9) and
 *    derive new compositions by metal substitution + H count adjustment.
 *    LaH10 → LaH11Li2: substitute 2 cage-H positions with Li, add 1 H
 *    at the next available Wyckoff orbit.
 *
 * 2. Cage-Aware Wyckoff Generation: Instead of random placement, build
 *    structures by explicitly populating cage-forming Wyckoff orbits:
 *    - Sodalite cage: Im-3m, H at 12d (corner-sharing squares)
 *    - Clathrate-I: Fm-3m, H at 32f + 8c (sodalite-like cage)
 *    - Hex-clathrate: P63/mmc, H at 6h + 12k + 2b
 */

import type { CSPCandidate } from "./csp-types";
import { getKnownStructureFormulas, lookupKnownStructure } from "../learning/known-structures";
import { PROTOTYPE_TEMPLATES, type PrototypeTemplate } from "../learning/crystal-prototypes";

// ---------------------------------------------------------------------------
// Cage templates pulled from PROTOTYPE_TEMPLATES
// ---------------------------------------------------------------------------

/**
 * Get all cage-type templates from the existing 130+ PROTOTYPE_TEMPLATES.
 * Uses the cageType tag + wyckoff/orbitPriority fields added to the template interface.
 * No duplicate template database needed.
 */
function getCagePrototypeTemplates(): PrototypeTemplate[] {
  return PROTOTYPE_TEMPLATES.filter(t => t.cageType != null);
}

// ---------------------------------------------------------------------------
// Strategy 1: Parent seeding
// ---------------------------------------------------------------------------

function parentSeed(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  seed: number,
): CSPCandidate[] {
  const candidates: CSPCandidate[] = [];
  const targetH = Math.round(counts["H"] ?? 0);
  const metals = elements.filter(e => e !== "H");
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  // Find parent structures in known-structures.ts
  for (const ksFormula of getKnownStructureFormulas()) {
    const ks = lookupKnownStructure(ksFormula);
    if (!ks) continue;

    const ksH = ks.atoms.filter(a => a.element === "H").length;
    const ksMetals = ks.atoms.filter(a => a.element !== "H");
    const ksMetalEls = Array.from(new Set(ksMetals.map(m => m.element)));

    // Skip if not a hydride or too different in size
    if (ksH < 3) continue;
    if (Math.abs(ks.atoms.length - totalAtoms) > 6) continue;

    // Need at least one shared metal
    const sharedMetals = metals.filter(m => ksMetalEls.includes(m));
    if (sharedMetals.length === 0 && ksMetalEls.length > 0) {
      // Try if metals are chemically similar (same group/period)
      const hasRelated = metals.some(m => ksMetalEls.some(km => areChemicallySimilar(m, km)));
      if (!hasRelated) continue;
    }

    // Derive new composition from parent
    let positions = ks.atoms.map(a => ({ element: a.element, x: a.x, y: a.y, z: a.z }));

    // Step 1: Metal substitution
    for (const targetMetal of metals) {
      const targetCount = Math.round(counts[targetMetal] ?? 0);
      const currentCount = positions.filter(p => p.element === targetMetal).length;

      if (currentCount >= targetCount) continue;
      const needed = targetCount - currentCount;

      // Find H positions to convert to this metal (pick most isolated H)
      const hPositions = positions
        .map((p, i) => ({ ...p, idx: i }))
        .filter(p => p.element === "H");

      // Score by distance to other atoms — most isolated H is best to replace
      const hScored = hPositions.map(h => {
        let avgDist = 0;
        for (const p of positions) {
          if (p === h) continue;
          let dx = h.x - p.x, dy = h.y - p.y, dz = h.z - p.z;
          dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
          avgDist += Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        return { ...h, avgDist: avgDist / positions.length };
      });
      hScored.sort((a, b) => b.avgDist - a.avgDist);

      for (let i = 0; i < Math.min(needed, hScored.length); i++) {
        positions[hScored[i].idx].element = targetMetal;
      }
    }

    // Step 2: Adjust H count
    const currentH = positions.filter(p => p.element === "H").length;
    if (currentH < targetH) {
      // Add H at interstitial Wyckoff sites from prototype templates
      const template = getCagePrototypeTemplates().find(t =>
        t.spaceGroup === ks.spaceGroup || t.name.includes(ks.spaceGroup.replace(/[\s-]/g, ""))
      );
      if (template) {
        const toAdd = targetH - currentH;
        let added = 0;
        // Get H sites sorted by orbit priority from the prototype template
        const hSites = template.sites
          .filter(s => s.label === "H")
          .sort((a, b) => (a.orbitPriority ?? 99) - (b.orbitPriority ?? 99));
        for (const site of hSites) {
          if (added >= toAdd) break;
          const occupied = positions.some(p => {
            let dx = Math.abs(p.x - site.x), dy = Math.abs(p.y - site.y), dz = Math.abs(p.z - site.z);
            dx = Math.min(dx, 1 - dx); dy = Math.min(dy, 1 - dy); dz = Math.min(dz, 1 - dz);
            return Math.sqrt(dx * dx + dy * dy + dz * dz) < 0.05;
          });
          if (!occupied) {
            positions.push({ element: "H", x: site.x, y: site.y, z: site.z });
            added++;
          }
        }
      }
    } else if (currentH > targetH) {
      // Remove excess H (most isolated first)
      const excess = currentH - targetH;
      const hIdxs = positions.map((p, i) => ({ el: p.element, idx: i }))
        .filter(p => p.el === "H").map(p => p.idx);
      // Remove from end (least important positions)
      for (let i = 0; i < excess && hIdxs.length > 0; i++) {
        positions.splice(hIdxs.pop()!, 1);
      }
    }

    // Scale lattice
    const volScale = Math.pow(totalAtoms / positions.length, 1 / 3);
    const newLatticeA = ks.latticeA * Math.max(0.85, Math.min(1.15, volScale));

    candidates.push({
      latticeA: newLatticeA,
      latticeC: ks.latticeC ? ks.latticeC * volScale : undefined,
      cOverA: ks.latticeC ? (ks.latticeC * volScale) / newLatticeA : 1.0,
      positions,
      prototype: `parent-seed-${ksFormula}`,
      crystalSystem: ks.latticeType,
      spaceGroup: ks.spaceGroup,
      source: `Parent seed from ${ksFormula} (${ks.spaceGroup})`,
      confidence: 0.80,
      isMetallic: null,
      sourceEngine: "known-structure",
      generationStage: 1,
      seed: seed++,
      pressureGPa,
      relaxationLevel: "raw",
    });

    if (candidates.length >= 8) break;
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// Strategy 2: Cage-aware Wyckoff generation
// ---------------------------------------------------------------------------

function cageAwareWyckoff(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  seed: number,
): CSPCandidate[] {
  const candidates: CSPCandidate[] = [];
  const targetH = Math.round(counts["H"] ?? 0);
  const metals = elements.filter(e => e !== "H");
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  const totalMetalCount = metals.reduce((s, m) => s + Math.round(counts[m] ?? 0), 0);

  // Pull cage templates from the existing 130+ PROTOTYPE_TEMPLATES
  const cageTemplates = getCagePrototypeTemplates();

  for (const template of cageTemplates) {
    // Count metal and H sites in this template
    const metalSites = template.sites.filter(s => s.label !== "H");
    const hSites = template.sites.filter(s => s.label === "H");

    // Metal slot count must be close to our target
    if (Math.abs(totalMetalCount - metalSites.length) > 2) continue;

    // Build structure by filling sites in orbitPriority order
    const sortedSites = [...template.sites].sort((a, b) =>
      (a.orbitPriority ?? 99) - (b.orbitPriority ?? 99)
    );

    const positions: Array<{ element: string; x: number; y: number; z: number }> = [];

    // Place metals on metal sites
    let metalIdx = 0;
    for (const site of sortedSites) {
      if (site.label === "H") continue;
      const metalEl = metals[metalIdx % metals.length] ?? metals[0];
      positions.push({ element: metalEl, x: site.x, y: site.y, z: site.z });
      metalIdx++;
    }

    // Extra metals if needed
    for (const metal of metals) {
      const placed = positions.filter(p => p.element === metal).length;
      const needed = Math.round(counts[metal] ?? 0) - placed;
      if (needed > 0) {
        const interstitials = [
          { x: 0.25, y: 0.25, z: 0.25 },
          { x: 0.75, y: 0.75, z: 0.75 },
          { x: 0.5, y: 0.5, z: 0.5 },
        ];
        for (let i = 0; i < needed && i < interstitials.length; i++) {
          positions.push({ element: metal, ...interstitials[i] });
        }
      }
    }

    // Fill H sites in priority order up to target H count
    let hPlaced = 0;
    const hSitesSorted = sortedSites.filter(s => s.label === "H");
    for (const site of hSitesSorted) {
      if (hPlaced >= targetH) break;
      // Check not overlapping with already-placed atoms
      let tooClose = false;
      for (const p of positions) {
        let dx = site.x - p.x, dy = site.y - p.y, dz = site.z - p.z;
        dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
        if (Math.sqrt(dx * dx + dy * dy + dz * dz) < 0.03) { tooClose = true; break; }
      }
      if (!tooClose) {
        positions.push({ element: "H", x: site.x, y: site.y, z: site.z });
        hPlaced++;
      }
    }

    // Estimate lattice constant
    const volPerAtom = pressureGPa > 100 ? 5.0 : 8.0;
    const vol = volPerAtom * positions.length;
    const estA = Math.pow(vol / template.cOverA, 1 / 3);

    // Build orbit description from Wyckoff labels
    const orbitsUsed = Array.from(new Set(hSitesSorted.slice(0, hPlaced).map(s => s.wyckoff ?? s.role)));

    candidates.push({
      latticeA: estA,
      latticeC: template.latticeType === "hexagonal" ? estA * template.cOverA : undefined,
      cOverA: template.cOverA,
      positions,
      prototype: `cage-wyckoff-${template.name}`,
      crystalSystem: template.latticeType,
      spaceGroup: template.spaceGroup,
      source: `Cage Wyckoff (${template.name}, ${hPlaced}H on ${orbitsUsed.join("+")})`,
      confidence: 0.85,
      isMetallic: null,
      sourceEngine: "known-structure",
      generationStage: 1,
      seed: seed++,
      pressureGPa,
      relaxationLevel: "raw",
    });
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// Chemical similarity helper
// ---------------------------------------------------------------------------

const SIMILAR_GROUPS: string[][] = [
  ["La", "Ce", "Pr", "Nd", "Y", "Sc", "Gd", "Th"],
  ["Ca", "Sr", "Ba", "Mg"],
  ["Li", "Na", "K"],
  ["Ti", "Zr", "Hf"],
  ["V", "Nb", "Ta"],
  ["Cr", "Mo", "W"],
  ["Fe", "Co", "Ni"],
  ["Cu", "Ag", "Au"],
];

function areChemicallySimilar(el1: string, el2: string): boolean {
  return SIMILAR_GROUPS.some(g => g.includes(el1) && g.includes(el2));
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Generate cage-seeded candidates using parent seeding + Wyckoff generation.
 */
export function generateCageSeededCandidates(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  nCandidates: number = 10,
): CSPCandidate[] {
  const hasH = elements.includes("H");
  const targetH = Math.round(counts["H"] ?? 0);
  if (!hasH || targetH < 3) return [];

  const seed = Date.now() % 1e8;
  const parentCandidates = parentSeed(elements, counts, pressureGPa, seed);
  const wyckoffCandidates = cageAwareWyckoff(elements, counts, pressureGPa, seed + 10000);

  const all = [...parentCandidates, ...wyckoffCandidates].slice(0, nCandidates);

  if (all.length > 0) {
    const parentCount = parentCandidates.length;
    const wyckoffCount = wyckoffCandidates.length;
    console.log(`[Cage-Seeder] ${elements.join("")}: ${all.length} cage candidates (${parentCount} parent-seeded, ${wyckoffCount} Wyckoff-generated)`);
  }

  return all;
}
