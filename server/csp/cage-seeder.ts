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

// ---------------------------------------------------------------------------
// Cage Wyckoff templates — explicit cage-forming orbits per SG
// ---------------------------------------------------------------------------

interface CageWyckoffTemplate {
  name: string;
  spaceGroup: string;
  sgNumber: number;
  latticeType: "cubic" | "hexagonal";
  defaultCOverA: number;
  /** Metal site(s) — where the heavy atom sits. */
  metalSites: Array<{ x: number; y: number; z: number; wyckoff: string; multiplicity: number }>;
  /** H cage orbits — ordered by priority (fill first = most cage-like). */
  hCageOrbits: Array<{
    x: number; y: number; z: number;
    wyckoff: string;
    multiplicity: number;
    /** All equivalent positions in the orbit (primitive cell). */
    equivalents: Array<{ x: number; y: number; z: number }>;
  }>;
}

const CAGE_WYCKOFF_TEMPLATES: CageWyckoffTemplate[] = [
  {
    // Fm-3m Clathrate (LaH10-type): 1 metal + 10 H in primitive cell
    name: "clathrate-Fm3m",
    spaceGroup: "Fm-3m",
    sgNumber: 225,
    latticeType: "cubic",
    defaultCOverA: 1.0,
    metalSites: [
      { x: 0.0, y: 0.0, z: 0.0, wyckoff: "4a", multiplicity: 1 },
    ],
    hCageOrbits: [
      {
        wyckoff: "32f", multiplicity: 8,
        x: 0.375, y: 0.375, z: 0.375,
        equivalents: [
          { x: 0.375, y: 0.375, z: 0.375 },
          { x: 0.375, y: 0.375, z: 0.875 },
          { x: 0.375, y: 0.875, z: 0.375 },
          { x: 0.875, y: 0.375, z: 0.375 },
          { x: 0.625, y: 0.625, z: 0.625 },
          { x: 0.625, y: 0.625, z: 0.125 },
          { x: 0.625, y: 0.125, z: 0.625 },
          { x: 0.125, y: 0.625, z: 0.625 },
        ],
      },
      {
        wyckoff: "8c", multiplicity: 2,
        x: 0.25, y: 0.25, z: 0.25,
        equivalents: [
          { x: 0.25, y: 0.25, z: 0.25 },
          { x: 0.75, y: 0.75, z: 0.75 },
        ],
      },
      {
        // Extra interstice for higher hydrides (LaH12-type)
        wyckoff: "octahedral", multiplicity: 3,
        x: 0.5, y: 0.0, z: 0.0,
        equivalents: [
          { x: 0.5, y: 0.0, z: 0.0 },
          { x: 0.0, y: 0.5, z: 0.0 },
          { x: 0.0, y: 0.0, z: 0.5 },
        ],
      },
    ],
  },
  {
    // Im-3m Sodalite (CaH6-type): 1 metal + 6 H in primitive cell
    name: "sodalite-Im3m",
    spaceGroup: "Im-3m",
    sgNumber: 229,
    latticeType: "cubic",
    defaultCOverA: 1.0,
    metalSites: [
      { x: 0.0, y: 0.0, z: 0.0, wyckoff: "2a", multiplicity: 1 },
    ],
    hCageOrbits: [
      {
        wyckoff: "12d", multiplicity: 6,
        x: 0.5, y: 0.75, z: 0.25,
        equivalents: [
          { x: 0.5, y: 0.75, z: 0.25 },
          { x: 0.5, y: 0.25, z: 0.75 },
          { x: 0.75, y: 0.5, z: 0.25 },
          { x: 0.25, y: 0.5, z: 0.75 },
          { x: 0.75, y: 0.25, z: 0.5 },
          { x: 0.25, y: 0.75, z: 0.5 },
        ],
      },
      {
        // Extra orbit for MH8+ compositions
        wyckoff: "6b", multiplicity: 3,
        x: 0.5, y: 0.5, z: 0.0,
        equivalents: [
          { x: 0.5, y: 0.5, z: 0.0 },
          { x: 0.5, y: 0.0, z: 0.5 },
          { x: 0.0, y: 0.5, z: 0.5 },
        ],
      },
    ],
  },
  {
    // P63/mmc Hex-clathrate (YH9-type): 1 metal + 9 H
    name: "hex-clathrate-P63mmc",
    spaceGroup: "P63/mmc",
    sgNumber: 194,
    latticeType: "hexagonal",
    defaultCOverA: 1.55,
    metalSites: [
      { x: 0.3333, y: 0.6667, z: 0.25, wyckoff: "2d", multiplicity: 1 },
    ],
    hCageOrbits: [
      {
        wyckoff: "6h", multiplicity: 3,
        x: 0.155, y: 0.31, z: 0.25,
        equivalents: [
          { x: 0.155, y: 0.31, z: 0.25 },
          { x: 0.69, y: 0.845, z: 0.25 },
          { x: 0.845, y: 0.155, z: 0.25 },
        ],
      },
      {
        wyckoff: "2b", multiplicity: 1,
        x: 0.0, y: 0.0, z: 0.25,
        equivalents: [
          { x: 0.0, y: 0.0, z: 0.25 },
        ],
      },
      {
        wyckoff: "12k", multiplicity: 5,
        x: 0.52, y: 0.04, z: 0.08,
        equivalents: [
          { x: 0.52, y: 0.04, z: 0.08 },
          { x: 0.96, y: 0.48, z: 0.08 },
          { x: 0.48, y: 0.52, z: 0.08 },
          { x: 0.04, y: 0.52, z: 0.42 },
          { x: 0.52, y: 0.48, z: 0.42 },
        ],
      },
    ],
  },
];

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
    const ksMetalEls = [...new Set(ksMetals.map(m => m.element))];

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
      // Add H at interstitial Wyckoff sites
      const template = CAGE_WYCKOFF_TEMPLATES.find(t => t.sgNumber === ks.spaceGroupNumber);
      if (template) {
        const toAdd = targetH - currentH;
        let added = 0;
        for (const orbit of template.hCageOrbits) {
          for (const eq of orbit.equivalents) {
            if (added >= toAdd) break;
            // Check not already occupied
            const occupied = positions.some(p => {
              let dx = Math.abs(p.x - eq.x), dy = Math.abs(p.y - eq.y), dz = Math.abs(p.z - eq.z);
              dx = Math.min(dx, 1 - dx); dy = Math.min(dy, 1 - dy); dz = Math.min(dz, 1 - dz);
              return Math.sqrt(dx * dx + dy * dy + dz * dz) < 0.05;
            });
            if (!occupied) {
              positions.push({ element: "H", x: eq.x, y: eq.y, z: eq.z });
              added++;
            }
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

  for (const template of CAGE_WYCKOFF_TEMPLATES) {
    // Check if this cage template can fit the target composition
    const metalSlots = template.metalSites.reduce((s, m) => s + m.multiplicity, 0);
    const totalMetalCount = metals.reduce((s, m) => s + Math.round(counts[m] ?? 0), 0);

    // Metal count must be close to template metal slots
    if (Math.abs(totalMetalCount - metalSlots) > 2) continue;

    // Build structure by filling Wyckoff orbits
    const positions: Array<{ element: string; x: number; y: number; z: number }> = [];

    // Place metals
    let metalIdx = 0;
    for (const site of template.metalSites) {
      const metalEl = metals[metalIdx % metals.length] ?? metals[0];
      positions.push({ element: metalEl, x: site.x, y: site.y, z: site.z });
      metalIdx++;
    }

    // Extra metals if needed (on remaining metal sites or interstitials)
    for (const metal of metals) {
      const placed = positions.filter(p => p.element === metal).length;
      const needed = Math.round(counts[metal] ?? 0) - placed;
      if (needed > 0) {
        // Place at interstitial metal sites
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

    // Fill H cage orbits up to target H count
    let hPlaced = 0;
    for (const orbit of template.hCageOrbits) {
      for (const eq of orbit.equivalents) {
        if (hPlaced >= targetH) break;
        // Check not too close to metals
        let tooClose = false;
        for (const p of positions) {
          let dx = eq.x - p.x, dy = eq.y - p.y, dz = eq.z - p.z;
          dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
          if (Math.sqrt(dx * dx + dy * dy + dz * dz) < 0.03) { tooClose = true; break; }
        }
        if (!tooClose) {
          positions.push({ element: "H", x: eq.x, y: eq.y, z: eq.z });
          hPlaced++;
        }
      }
      if (hPlaced >= targetH) break;
    }

    // Estimate lattice constant
    const volPerAtom = pressureGPa > 100 ? 5.0 : 8.0;
    const vol = volPerAtom * positions.length;
    const estA = Math.pow(vol / template.defaultCOverA, 1 / 3);

    candidates.push({
      latticeA: estA,
      latticeC: template.latticeType === "hexagonal" ? estA * template.defaultCOverA : undefined,
      cOverA: template.defaultCOverA,
      positions,
      prototype: `cage-wyckoff-${template.name}`,
      crystalSystem: template.latticeType,
      spaceGroup: template.spaceGroup,
      source: `Cage Wyckoff (${template.name}, ${hPlaced}H on ${template.hCageOrbits.map(o => o.wyckoff).join("+")})`,
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
