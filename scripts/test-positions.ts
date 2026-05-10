import { lookupKnownStructure } from "../server/learning/known-structures";
import { selectPrototype, fillPrototype } from "../server/learning/crystal-prototypes";

function validateFracPositions(
  formula: string,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
): string[] {
  const issues: string[] = [];
  if (positions.length === 0) { issues.push("No positions"); return issues; }
  for (const p of positions) {
    if (!isFinite(p.x) || !isFinite(p.y) || !isFinite(p.z))
      issues.push(`NaN/Inf: ${p.element} (${p.x}, ${p.y}, ${p.z})`);
  }
  // Min distance check (fractional, with PBC)
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      const dx = Math.min(Math.abs(positions[i].x - positions[j].x), 1 - Math.abs(positions[i].x - positions[j].x));
      const dy = Math.min(Math.abs(positions[i].y - positions[j].y), 1 - Math.abs(positions[i].y - positions[j].y));
      const dz = Math.min(Math.abs(positions[i].z - positions[j].z), 1 - Math.abs(positions[i].z - positions[j].z));
      if (Math.sqrt(dx*dx + dy*dy + dz*dz) < 0.01)
        issues.push(`Overlap: ${positions[i].element}[${i}] ↔ ${positions[j].element}[${j}]`);
    }
  }
  return issues;
}

const formulas = [
  "LaH10", "CaH6", "H3S", "YH9", "MgB2", "Nb3Sn", "NbN",
  "YBa2Cu3O7", "BaFe2As2", "LaFeAsO", "FeSe", "LiFeAs",
  "CsPbI3", "SnSe", "FeSi", "La2O3", "CaWO4", "Fe3C",
  "SmCo5", "Na3Bi", "MnWO4", "ZrO2", "LaPO4", "Cu2O",
  "Bi2Sr2CaCu2O8", "HgBa2CuO4", "SrTaO2N",
  "MoSi2", "YB4", "ZrB12", "Ni3Al", "Cr3C2",
  "Bi", "Sb", "Sn", "Si", "Nb", "Al",
];

console.log("╔══════════════════════════════════════════════════════════════╗");
console.log("║        QAE Atomic Position Validation Suite                 ║");
console.log("╚══════════════════════════════════════════════════════════════╝\n");

// --- Known Structures ---
console.log("── Known Structure Database ──");
let kPass = 0, kFail = 0, kMiss = 0;
for (const f of formulas) {
  const known = lookupKnownStructure(f);
  if (!known) { kMiss++; continue; }
  const issues = validateFracPositions(f, known.atoms);
  if (issues.length === 0) { kPass++; }
  else { kFail++; console.log(`  FAIL ${f}: ${issues.join("; ")}`); }
}
console.log(`  Result: ${kPass} pass, ${kFail} fail, ${kMiss} no entry (${formulas.length} tested)\n`);

// --- Prototype Template Matching ---
console.log("── Prototype Template Matching ──");
let pPass = 0, pFail = 0, pMiss = 0;
for (const f of formulas) {
  const proto = selectPrototype(f);
  if (!proto) { pMiss++; console.log(`  MISS ${f}`); continue; }
  const positions = proto.template.sites.map(s => ({
    element: proto.siteMap[s.label] || "?", x: s.x, y: s.y, z: s.z
  }));
  const issues = validateFracPositions(f, positions);
  if (issues.length === 0) { pPass++; }
  else { pFail++; console.log(`  FAIL ${f} (${proto.template.name}): ${issues.join("; ")}`); }
}
console.log(`  Result: ${pPass} pass, ${pFail} fail, ${pMiss} no match\n`);

// --- fillPrototype Cartesian ---
console.log("── fillPrototype() Cartesian Coords ──");
let fPass = 0, fFail = 0, fMiss = 0;
for (const f of formulas) {
  const filled = fillPrototype(f);
  if (!filled) { fMiss++; continue; }
  const issues: string[] = [];
  for (const a of filled.atoms) {
    if (!isFinite(a.x) || !isFinite(a.y) || !isFinite(a.z))
      issues.push(`NaN: ${a.element} (${a.x}, ${a.y}, ${a.z})`);
  }
  if (filled.latticeParam <= 0 || !isFinite(filled.latticeParam))
    issues.push(`Bad lattice: ${filled.latticeParam}`);
  // Min Cartesian distance
  for (let i = 0; i < filled.atoms.length; i++) {
    for (let j = i + 1; j < filled.atoms.length; j++) {
      const d = Math.sqrt(
        (filled.atoms[i].x - filled.atoms[j].x) ** 2 +
        (filled.atoms[i].y - filled.atoms[j].y) ** 2 +
        (filled.atoms[i].z - filled.atoms[j].z) ** 2
      );
      if (d < 0.3) issues.push(`Too close: ${filled.atoms[i].element}[${i}]↔${filled.atoms[j].element}[${j}] d=${d.toFixed(2)}Å`);
    }
  }
  if (issues.length === 0) { fPass++; }
  else { fFail++; console.log(`  FAIL ${f} (${filled.templateName}, a=${filled.latticeParam.toFixed(2)}Å): ${issues.join("; ")}`); }
}
console.log(`  Result: ${fPass} pass, ${fFail} fail, ${fMiss} no match\n`);

// --- Detailed: LaH10 ---
console.log("── LaH10 Detailed ──");
const laH10 = lookupKnownStructure("LaH10");
if (laH10) {
  const la = laH10.atoms.filter(a => a.element === "La").length;
  const h = laH10.atoms.filter(a => a.element === "H").length;
  console.log(`  ${laH10.spaceGroup} | a=${laH10.latticeA}Å | ${la} La + ${h} H = ${laH10.atoms.length} atoms`);
  for (const a of laH10.atoms)
    console.log(`    ${a.element.padEnd(3)} (${a.x.toFixed(4)}, ${a.y.toFixed(4)}, ${a.z.toFixed(4)}) ${a.wyckoff ?? ""}`);
  console.log(la === 1 && h === 10 ? "  ✓ Stoichiometry correct" : "  ✗ Stoichiometry WRONG");
}

// --- Detailed: MnWO4 monoclinic ---
console.log("\n── MnWO4 Monoclinic ──");
const mnwo4 = lookupKnownStructure("MnWO4");
if (mnwo4) {
  console.log(`  ${mnwo4.spaceGroup} | a=${mnwo4.latticeA} b=${mnwo4.latticeB} c=${mnwo4.latticeC} beta=${mnwo4.beta}°`);
  console.log(`  latticeType=${mnwo4.latticeType} | ${mnwo4.atoms.length} atoms`);
  console.log(mnwo4.latticeType === "monoclinic" ? "  ✓ Monoclinic type correct" : "  ✗ Type WRONG");
}

// --- Detailed: fillPrototype for a monoclinic compound ---
console.log("\n── Wolframite fillPrototype: CoWO4 (novel) ──");
const coWO4 = fillPrototype("CoWO4");
if (coWO4) {
  console.log(`  Template: ${coWO4.templateName} | a=${coWO4.latticeParam.toFixed(3)}Å | ${coWO4.atoms.length} atoms`);
  for (const a of coWO4.atoms)
    console.log(`    ${a.element.padEnd(3)} (${a.x.toFixed(3)}, ${a.y.toFixed(3)}, ${a.z.toFixed(3)}) Å`);
} else {
  console.log("  No match (expected Wolframite template)");
}
