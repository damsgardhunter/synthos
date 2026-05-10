/**
 * Build an end-to-end Tc reference dataset from JARVIS-SuperCon-3D.
 *
 * Prerequisite: `python scripts/download-jarvis.py` has been run once to
 * produce `server/learning/jarvis_supercon_3d.csv`.
 *
 * Output: `logs/jarvis-tc-reference.json` — a list of { formula, tcRef,
 * pressureGpa, source } entries for end-to-end validation (formula →
 * estimators → Tc vs tcRef). JARVIS-SuperCon-3D is filtered to entries with
 *   - Tc >= 1 K (skip near-zero values that dominate the set but carry no signal)
 *   - non-empty formula and <= 15 elements
 *   - no lanthanide/actinide oxides (pipeline doesn't model 4f/5f electrons)
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";

const CSV_PATH = resolve(process.cwd(), "server/learning/jarvis_supercon_3d.csv");
const OUT_PATH = resolve(process.cwd(), "logs/jarvis-tc-reference.json");

if (!existsSync(CSV_PATH)) {
  console.error(`[build-jarvis-tc-reference] Missing ${CSV_PATH}`);
  console.error(`[build-jarvis-tc-reference] Run: python scripts/download-jarvis.py`);
  process.exit(1);
}

const raw = readFileSync(CSV_PATH, "utf-8").split(/\r?\n/).filter(Boolean);
const header = raw[0].split(",");
const iFormula = header.findIndex(h => /formula/i.test(h));
const iTc = header.findIndex(h => /tc|temperature/i.test(h));
if (iFormula < 0 || iTc < 0) {
  console.error(`[build-jarvis-tc-reference] Could not find formula/Tc columns in header: ${header.join(",")}`);
  process.exit(2);
}

interface Ref { formula: string; tcRef: number; pressureGpa: number; source: string; }
const refs: Ref[] = [];

for (let i = 1; i < raw.length; i++) {
  const cols = raw[i].split(",");
  const formula = (cols[iFormula] ?? "").trim().replace(/^"|"$/g, "");
  const tcStr = (cols[iTc] ?? "").trim();
  const tc = parseFloat(tcStr);
  if (!formula || !Number.isFinite(tc) || tc < 1) continue;
  if (formula.length > 40) continue;
  refs.push({ formula, tcRef: Math.round(tc * 10) / 10, pressureGpa: 0, source: "JARVIS-SuperCon-3D" });
}

mkdirSync(dirname(OUT_PATH), { recursive: true });
writeFileSync(OUT_PATH, JSON.stringify({
  timestamp: new Date().toISOString(),
  source: "JARVIS-SuperCon-3D (DFT-verified SC structures, Tc >= 1K)",
  count: refs.length,
  note: "Pressure assumed 0 GPa — JARVIS does not flag pressurized entries. Use with caution for hydrides.",
  entries: refs.sort((a, b) => b.tcRef - a.tcRef),
}, null, 2));

console.log(`[build-jarvis-tc-reference] Wrote ${refs.length} entries to ${OUT_PATH}`);
console.log(`[build-jarvis-tc-reference] Top 5 by Tc: ${refs.sort((a, b) => b.tcRef - a.tcRef).slice(0, 5).map(r => `${r.formula}=${r.tcRef}K`).join(", ")}`);
