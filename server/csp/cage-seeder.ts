/**
 * Cage-Aware Structure Seeding (v2)
 *
 * Two strategies for generating cage hydride candidates:
 *
 * 1. Parent Seeding: Take known cage structures (LaH10, CaH6, YH9) and
 *    derive new compositions by metal substitution + H count adjustment.
 *    Now generates at multiple pressure-compressed volumes.
 *
 * 2. Cage-Aware Wyckoff Generation: Build structures by explicitly
 *    populating cage-forming Wyckoff orbits with proper A/M site
 *    distinction for ternary hydrides. Generates at 3-5 volumes.
 *
 * Improvements over v1:
 * - 20-30 candidates instead of 10
 * - Volume ensemble (3-5 pressure-compressed volumes per template)
 * - Proper ternary guest placement using template A/M site labels
 */

import type { CSPCandidate } from "./csp-types";
import { getKnownStructureFormulas, lookupKnownStructure } from "../learning/known-structures";
import { PROTOTYPE_TEMPLATES, type PrototypeTemplate } from "../learning/crystal-prototypes";

// ---------------------------------------------------------------------------
// Cage templates pulled from PROTOTYPE_TEMPLATES
// ---------------------------------------------------------------------------

function getCagePrototypeTemplates(): PrototypeTemplate[] {
  return PROTOTYPE_TEMPLATES.filter(t => t.cageType != null);
}

// ---------------------------------------------------------------------------
// Volume ensemble for cage seeds
// ---------------------------------------------------------------------------

/**
 * Generate volume scale factors for a given pressure.
 * At high pressure, bias toward compressed volumes but keep some expanded.
 */
function cageVolumeScales(pressureGPa: number): number[] {
  if (pressureGPa >= 150) {
    // Very high pressure: 5 points, heavily compressed
    return [0.75, 0.85, 0.95, 1.00, 1.10];
  } else if (pressureGPa >= 50) {
    // High pressure: 4 points
    return [0.85, 0.95, 1.00, 1.10];
  } else {
    // Ambient/moderate: 3 points
    return [0.95, 1.00, 1.05];
  }
}

// ---------------------------------------------------------------------------
// Strategy 1: Parent seeding (with volume ensemble)
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
  const volScales = cageVolumeScales(pressureGPa);

  for (const ksFormula of getKnownStructureFormulas()) {
    const ks = lookupKnownStructure(ksFormula);
    if (!ks) continue;

    const ksH = ks.atoms.filter(a => a.element === "H").length;
    const ksMetals = ks.atoms.filter(a => a.element !== "H");
    const ksMetalEls = Array.from(new Set(ksMetals.map(m => m.element)));

    if (ksH < 3) continue;
    if (Math.abs(ks.atoms.length - totalAtoms) > 6) continue;

    const sharedMetals = metals.filter(m => ksMetalEls.includes(m));
    if (sharedMetals.length === 0 && ksMetalEls.length > 0) {
      const hasRelated = metals.some(m => ksMetalEls.some(km => areChemicallySimilar(m, km)));
      if (!hasRelated) continue;
    }

    // Derive positions from parent
    let positions = ks.atoms.map(a => ({ element: a.element, x: a.x, y: a.y, z: a.z }));

    // Step 1: Metal substitution
    for (const targetMetal of metals) {
      const targetCount = Math.round(counts[targetMetal] ?? 0);
      const currentCount = positions.filter(p => p.element === targetMetal).length;
      if (currentCount >= targetCount) continue;
      const needed = targetCount - currentCount;

      const hPositions = positions
        .map((p, i) => ({ ...p, idx: i }))
        .filter(p => p.element === "H");

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
      const template = getCagePrototypeTemplates().find(t =>
        t.spaceGroup === ks.spaceGroup || t.name.includes(ks.spaceGroup.replace(/[\s-]/g, ""))
      );
      if (template) {
        const toAdd = targetH - currentH;
        let added = 0;
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
      const excess = currentH - targetH;
      const hIdxs = positions.map((p, i) => ({ el: p.element, idx: i }))
        .filter(p => p.el === "H").map(p => p.idx);
      for (let i = 0; i < excess && hIdxs.length > 0; i++) {
        positions.splice(hIdxs.pop()!, 1);
      }
    }

    // Generate at multiple volumes
    const volScale = Math.pow(totalAtoms / positions.length, 1 / 3);
    const baseLatticeA = ks.latticeA * Math.max(0.85, Math.min(1.15, volScale));

    for (const vs of volScales) {
      const scaledA = baseLatticeA * Math.pow(vs, 1 / 3);
      candidates.push({
        latticeA: scaledA,
        latticeC: ks.latticeC ? ks.latticeC * volScale * Math.pow(vs, 1 / 3) : undefined,
        cOverA: ks.latticeC ? (ks.latticeC * volScale) / baseLatticeA : 1.0,
        positions: positions.map(p => ({ ...p })), // deep copy
        prototype: `parent-seed-${ksFormula}`,
        crystalSystem: ks.latticeType,
        spaceGroup: ks.spaceGroup,
        source: `Parent seed from ${ksFormula} (${ks.spaceGroup}, vol=${vs.toFixed(2)})`,
        confidence: 0.80,
        isMetallic: null,
        sourceEngine: "known-structure",
        generationStage: 1,
        seed: seed++,
        pressureGPa,
        relaxationLevel: "raw",
      });
    }

    if (candidates.length >= 15) break; // More parents than before (was 8)
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// Strategy 2: Cage-aware Wyckoff generation (with ternary A/M placement)
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
  const volScales = cageVolumeScales(pressureGPa);

  // Classify metals for A/M site assignment in ternary templates.
  // "A" sites are interstitial (alkali, small atoms): Li, Na, K, Be, Mg
  // "M" sites are cage-center (rare earth, heavy): La, Y, Sc, Ca, Sr, Ba, Th, etc.
  const alkaliLike = new Set(["Li", "Na", "K", "Rb", "Cs", "Be", "Mg"]);
  const guestMetals = metals.filter(m => alkaliLike.has(m));
  const hostMetals = metals.filter(m => !alkaliLike.has(m));

  const cageTemplates = getCagePrototypeTemplates();

  for (const template of cageTemplates) {
    const metalSites = template.sites.filter(s => s.label !== "H");
    const hSites = template.sites.filter(s => s.label === "H");

    if (Math.abs(totalMetalCount - metalSites.length) > 2) continue;

    // Build positions with proper A/M site distinction
    const sortedSites = [...template.sites].sort((a, b) =>
      (a.orbitPriority ?? 99) - (b.orbitPriority ?? 99)
    );

    const positions: Array<{ element: string; x: number; y: number; z: number }> = [];

    // --- Ternary-aware metal placement ---
    // Use template site labels: "M" = host metal (cage center), "A" = guest metal (interstitial)
    for (const site of sortedSites) {
      if (site.label === "H") continue;

      if (site.label === "A" && guestMetals.length > 0) {
        // Place guest/alkali metal at A sites (interstitial positions)
        const guestEl = guestMetals[0]; // Primary guest
        const guestPlaced = positions.filter(p => p.element === guestEl).length;
        const guestNeeded = Math.round(counts[guestEl] ?? 0);
        if (guestPlaced < guestNeeded) {
          positions.push({ element: guestEl, x: site.x, y: site.y, z: site.z });
          continue;
        }
      }

      if (site.label === "M" && hostMetals.length > 0) {
        // Place host metal at M sites (cage center)
        const hostEl = hostMetals[0]; // Primary host
        const hostPlaced = positions.filter(p => p.element === hostEl).length;
        const hostNeeded = Math.round(counts[hostEl] ?? 0);
        if (hostPlaced < hostNeeded) {
          positions.push({ element: hostEl, x: site.x, y: site.y, z: site.z });
          continue;
        }
      }

      // Fallback: round-robin metal placement
      const unplacedMetals = metals.filter(m => {
        const placed = positions.filter(p => p.element === m).length;
        return placed < Math.round(counts[m] ?? 0);
      });
      if (unplacedMetals.length > 0) {
        positions.push({ element: unplacedMetals[0], x: site.x, y: site.y, z: site.z });
      }
    }

    // Extra metals if template didn't have enough sites
    for (const metal of metals) {
      const placed = positions.filter(p => p.element === metal).length;
      const needed = Math.round(counts[metal] ?? 0) - placed;
      if (needed > 0) {
        // Use Wyckoff interstitial sites from the template
        const interstitialSites = template.sites
          .filter(s => s.role === "interstitial" || s.role === "interstice")
          .filter(s => !positions.some(p => {
            let dx = Math.abs(p.x - s.x), dy = Math.abs(p.y - s.y), dz = Math.abs(p.z - s.z);
            dx = Math.min(dx, 1 - dx); dy = Math.min(dy, 1 - dy); dz = Math.min(dz, 1 - dz);
            return Math.sqrt(dx * dx + dy * dy + dz * dz) < 0.05;
          }));

        for (let i = 0; i < needed; i++) {
          if (i < interstitialSites.length) {
            positions.push({ element: metal, x: interstitialSites[i].x, y: interstitialSites[i].y, z: interstitialSites[i].z });
          } else {
            // Last resort: generic interstitial
            const fallbacks = [
              { x: 0.25, y: 0.25, z: 0.25 },
              { x: 0.75, y: 0.75, z: 0.75 },
              { x: 0.5, y: 0.5, z: 0.5 },
            ];
            if (i - interstitialSites.length < fallbacks.length) {
              positions.push({ element: metal, ...fallbacks[i - interstitialSites.length] });
            }
          }
        }
      }
    }

    // Fill H sites in priority order
    let hPlaced = 0;
    const hSitesSorted = sortedSites.filter(s => s.label === "H");
    for (const site of hSitesSorted) {
      if (hPlaced >= targetH) break;
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

    // Base lattice estimate
    const volPerAtom = pressureGPa > 100 ? 5.0 : 8.0;
    const vol = volPerAtom * positions.length;
    const baseA = Math.pow(vol / template.cOverA, 1 / 3);
    const orbitsUsed = Array.from(new Set(hSitesSorted.slice(0, hPlaced).map(s => s.wyckoff ?? s.role)));

    // Generate at multiple volumes
    for (const vs of volScales) {
      const scaledA = baseA * Math.pow(vs, 1 / 3);
      candidates.push({
        latticeA: scaledA,
        latticeC: template.latticeType === "hexagonal" ? scaledA * template.cOverA : undefined,
        cOverA: template.cOverA,
        positions: positions.map(p => ({ ...p })), // deep copy
        prototype: `cage-wyckoff-${template.name}`,
        crystalSystem: template.latticeType,
        spaceGroup: template.spaceGroup,
        source: `Cage Wyckoff (${template.name}, ${hPlaced}H on ${orbitsUsed.join("+")}, vol=${vs.toFixed(2)})`,
        confidence: 0.85,
        isMetallic: null,
        sourceEngine: "known-structure",
        generationStage: 1,
        seed: seed++,
        pressureGPa,
        relaxationLevel: "raw",
      });
    }
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
 * Now generates 20-30 candidates at multiple volumes with proper ternary placement.
 */
export function generateCageSeededCandidates(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  nCandidates: number = 30,
): CSPCandidate[] {
  const hasH = elements.includes("H");
  const targetH = Math.round(counts["H"] ?? 0);
  if (!hasH || targetH < 3) return [];

  const seed = Date.now() % 1e8;
  const parentCandidates = parentSeed(elements, counts, pressureGPa, seed);
  const wyckoffCandidates = cageAwareWyckoff(elements, counts, pressureGPa, seed + 10000);

  const all = [...parentCandidates, ...wyckoffCandidates].slice(0, nCandidates);

  if (all.length > 0) {
    const parentCount = Math.min(parentCandidates.length, all.filter(c => c.prototype.startsWith("parent-seed")).length);
    const wyckoffCount = all.filter(c => c.prototype.startsWith("cage-wyckoff")).length;
    const volPoints = cageVolumeScales(pressureGPa).length;
    console.log(`[Cage-Seeder] ${elements.join("")}: ${all.length} cage candidates (${parentCount} parent-seeded, ${wyckoffCount} Wyckoff-generated, ${volPoints} volumes each)`);
  }

  return all;
}
