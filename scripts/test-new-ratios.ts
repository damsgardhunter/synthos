import { selectPrototype, fillPrototype } from "../server/learning/crystal-prototypes";

const tests = [
  { formula: "Cs2SnI6", expect: "VacancyPerovskite" },
  { formula: "FeTa2O6", expect: "Columbite" },
  { formula: "CoCr3O4", expect: "NormalSpinel" },
  { formula: "Li3AlN4", expect: "Antibixbyite" },
  { formula: "LaRu3Sb3", expect: "CoSb3" },
  { formula: "Li2TiO3", expect: "InverseSpinel" },
  { formula: "NaLaCoO3", expect: "OrderedPerovskite" },
  { formula: "LiFePO4", expect: "QuaternaryOlivine" },
  { formula: "LiNiMn2O4", expect: "OrderedSpinel" },
  { formula: "CuLaOS2", expect: "QuaternaryChalc" },
];

let pass = 0, fail = 0;
for (const t of tests) {
  const proto = selectPrototype(t.formula);
  if (!proto) {
    console.log(`MISS  ${t.formula.padEnd(15)} — no match (expected ${t.expect})`);
    fail++; continue;
  }
  const filled = fillPrototype(t.formula);
  const ok = filled && filled.atoms.every(a => isFinite(a.x) && isFinite(a.y) && isFinite(a.z));
  const nameMatch = proto.template.name.toLowerCase().includes(t.expect.toLowerCase().split("-")[0].split(/(?=[A-Z])/)[0].toLowerCase());
  
  console.log(`${nameMatch && ok ? "PASS " : "CHECK"} ${t.formula.padEnd(15)} → ${proto.template.name} (${filled?.atoms.length ?? 0} atoms, a=${filled?.latticeParam?.toFixed(2) ?? "?"}Å)`);
  if (nameMatch && ok) pass++; else fail++;
}
console.log(`\n${pass}/${tests.length} pass, ${fail} fail`);
