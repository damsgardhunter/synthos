/**
 * End-to-end pipeline benchmark.
 *
 * Runs known superconductors through the FULL DFT pipeline (runFullDFT:
 * structure → relax → phonon → DFPT → Tc) and compares the predicted Tc
 * to the experimental value. This is the calibration test that actually
 * verifies the pipeline produces real data.
 *
 * It is distinct from scripts/validate-physics.ts, which exercises only the
 * Tc *formulas* on literature-stored λ/ω_log/μ* — it never runs DFT.
 *
 * Each material is a multi-hour QE job, so the harness is incremental and
 * persists state. Drive it from a loop / cron on a DFT worker:
 *
 *   tsx scripts/benchmark-pipeline.ts              # run the next undone material
 *   tsx scripts/benchmark-pipeline.ts --material H3S
 *   tsx scripts/benchmark-pipeline.ts --all        # run everything (very long)
 *   tsx scripts/benchmark-pipeline.ts --report     # print current report, run nothing
 *   tsx scripts/benchmark-pipeline.ts --dry-run    # print the dataset, run nothing
 *   tsx scripts/benchmark-pipeline.ts --reset      # clear state + report
 *
 * State:  logs/benchmark-pipeline-state.json   (which materials are done)
 * Report: logs/benchmark-pipeline-report.json  (per-material results + summary)
 */
import { writeFileSync, readFileSync, existsSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";

// runFullDFT is imported lazily inside runOne(): pulling in the qe-worker
// module graph eagerly also loads modules that require server credentials
// (e.g. OPENAI_API_KEY) at import time, which would break --dry-run /
// --report on any machine without the full worker environment.
type RunFullDFT = typeof import("../server/dft/qe-worker")["runFullDFT"];

// ---------------------------------------------------------------------------
// Reference dataset — known superconductors with experimental Tc.
//
// Conventional phonon-mediated BCS, A15 intermetallics, the MgB₂ two-gap
// case, and the high-pressure superhydrides. Cuprates / Fe-pnictides are
// deliberately excluded: they are unconventional, and the pipeline's
// family gate routes them away from Allen-Dynes — a phonon-Tc benchmark
// would not be a fair test there.
// ---------------------------------------------------------------------------

type Mechanism = "phonon-BCS" | "A15" | "two-gap" | "superhydride";

interface BenchmarkMaterial {
  formula: string;
  pressureGpa: number;
  expTcK: number;
  /** Acceptable absolute spread on the experimental value (K). */
  expTcTolK: number;
  mechanism: Mechanism;
  note: string;
  ref: string;
}

const BENCHMARK: BenchmarkMaterial[] = [
  // --- Elemental BCS (weak → intermediate coupling) ---
  { formula: "Al", pressureGpa: 0, expTcK: 1.18, expTcTolK: 0.1, mechanism: "phonon-BCS",
    note: "weak coupling, λ≈0.4 — hard case for Allen-Dynes", ref: "Cochran & Mapother, Phys. Rev. 111, 132 (1958)" },
  { formula: "Sn", pressureGpa: 0, expTcK: 3.72, expTcTolK: 0.1, mechanism: "phonon-BCS",
    note: "white (β) tin", ref: "Roberts, J. Phys. Chem. Ref. Data 5, 581 (1976)" },
  { formula: "Ta", pressureGpa: 0, expTcK: 4.47, expTcTolK: 0.1, mechanism: "phonon-BCS",
    note: "elemental 5d", ref: "Roberts (1976)" },
  { formula: "V", pressureGpa: 0, expTcK: 5.40, expTcTolK: 0.1, mechanism: "phonon-BCS",
    note: "elemental 3d", ref: "Roberts (1976)" },
  { formula: "Pb", pressureGpa: 0, expTcK: 7.20, expTcTolK: 0.1, mechanism: "phonon-BCS",
    note: "strong coupling, λ≈1.55 — classic Eliashberg test", ref: "Roberts (1976)" },
  { formula: "Nb", pressureGpa: 0, expTcK: 9.25, expTcTolK: 0.1, mechanism: "phonon-BCS",
    note: "highest-Tc element at ambient pressure", ref: "Roberts (1976)" },

  // --- A15 intermetallics (intermediate coupling) ---
  { formula: "NbC", pressureGpa: 0, expTcK: 11.1, expTcTolK: 1.0, mechanism: "phonon-BCS",
    note: "rocksalt carbide", ref: "Giorgi et al., Phys. Rev. 125, 837 (1962)" },
  { formula: "NbN", pressureGpa: 0, expTcK: 16.0, expTcTolK: 1.0, mechanism: "phonon-BCS",
    note: "rocksalt nitride; some spin-fluctuation suppression", ref: "Roedhammer et al., Phys. Rev. B 15, 711 (1977)" },
  { formula: "V3Si", pressureGpa: 0, expTcK: 17.0, expTcTolK: 0.5, mechanism: "A15",
    note: "A15 (Pm-3n)", ref: "Hardy & Hulm, Phys. Rev. 93, 1004 (1954)" },
  { formula: "Nb3Sn", pressureGpa: 0, expTcK: 18.3, expTcTolK: 0.5, mechanism: "A15",
    note: "A15 (Pm-3n)", ref: "Matthias et al., Phys. Rev. 95, 1435 (1954)" },
  { formula: "Nb3Ge", pressureGpa: 0, expTcK: 23.2, expTcTolK: 0.8, mechanism: "A15",
    note: "A15; record ambient-pressure Tc pre-cuprates", ref: "Gavaler, Appl. Phys. Lett. 23, 480 (1973)" },

  // --- Two-gap ---
  { formula: "MgB2", pressureGpa: 0, expTcK: 39.0, expTcTolK: 1.0, mechanism: "two-gap",
    note: "σ/π two-gap — pipeline should route through the two-gap formula", ref: "Nagamatsu et al., Nature 410, 63 (2001)" },

  // --- High-pressure superhydrides ---
  { formula: "H3S", pressureGpa: 155, expTcK: 203, expTcTolK: 8, mechanism: "superhydride",
    note: "Im-3m; the cleanest hydride benchmark", ref: "Drozdov et al., Nature 525, 73 (2015), 10.1038/nature14964" },
  { formula: "CaH6", pressureGpa: 172, expTcK: 215, expTcTolK: 10, mechanism: "superhydride",
    note: "sodalite-like clathrate (Im-3m)", ref: "Li et al., Nat. Commun. 13, 2863 (2022), 10.1038/s41467-022-30454-w" },
  { formula: "YH9", pressureGpa: 201, expTcK: 243, expTcTolK: 12, mechanism: "superhydride",
    note: "hexagonal clathrate (P6/mmm)", ref: "Kong et al., Nat. Commun. 12, 5075 (2021), 10.1038/s41467-021-25372-2" },
  { formula: "LaH10", pressureGpa: 170, expTcK: 250, expTcTolK: 15, mechanism: "superhydride",
    note: "fcc clathrate (Fm-3m); ~250-260 K across groups", ref: "Drozdov et al., Nature 569, 528 (2019), 10.1038/s41586-019-1201-8" },
];

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const STATE_PATH = resolve(process.cwd(), "logs/benchmark-pipeline-state.json");
const REPORT_PATH = resolve(process.cwd(), "logs/benchmark-pipeline-report.json");

interface MaterialResult {
  formula: string;
  pressureGpa: number;
  expTcK: number;
  mechanism: Mechanism;
  predTcK: number | null;
  /** Where the predicted Tc came from. */
  predSource: "dfpt-best" | "epw-migdal" | "epw-allen-dynes" | "none";
  lambda: number | null;
  omegaLogK: number | null;
  /** Signed relative error (pred-exp)/exp, null when no prediction. */
  relErr: number | null;
  status: "pass" | "warn" | "fail" | "no-prediction" | "error";
  qualityTier: string | null;
  pipelineError: string | null;
  wallTimeS: number;
  ref: string;
  ranAt: string;
}

interface Report {
  generatedAt: string;
  results: MaterialResult[];
  summary: {
    total: number;
    scored: number;
    pass: number;
    warn: number;
    fail: number;
    noPrediction: number;
    error: number;
    /** Mean absolute relative error over scored materials, percent. */
    maeRelPct: number | null;
  };
}

// ---------------------------------------------------------------------------
// State / report IO
// ---------------------------------------------------------------------------

function loadState(): { done: string[] } {
  if (!existsSync(STATE_PATH)) return { done: [] };
  try {
    return JSON.parse(readFileSync(STATE_PATH, "utf-8"));
  } catch {
    return { done: [] };
  }
}

function saveState(state: { done: string[] }): void {
  mkdirSync(dirname(STATE_PATH), { recursive: true });
  writeFileSync(STATE_PATH, JSON.stringify(state, null, 2));
}

function loadReport(): Report {
  if (existsSync(REPORT_PATH)) {
    try {
      return JSON.parse(readFileSync(REPORT_PATH, "utf-8"));
    } catch { /* fall through */ }
  }
  return {
    generatedAt: new Date().toISOString(),
    results: [],
    summary: { total: 0, scored: 0, pass: 0, warn: 0, fail: 0, noPrediction: 0, error: 0, maeRelPct: null },
  };
}

function recomputeSummary(report: Report): void {
  const r = report.results;
  const scored = r.filter(x => x.relErr != null);
  report.summary = {
    total: BENCHMARK.length,
    scored: scored.length,
    pass: r.filter(x => x.status === "pass").length,
    warn: r.filter(x => x.status === "warn").length,
    fail: r.filter(x => x.status === "fail").length,
    noPrediction: r.filter(x => x.status === "no-prediction").length,
    error: r.filter(x => x.status === "error").length,
    maeRelPct: scored.length > 0
      ? Number((scored.reduce((s, x) => s + Math.abs(x.relErr!), 0) / scored.length * 100).toFixed(1))
      : null,
  };
}

function saveReport(report: Report): void {
  mkdirSync(dirname(REPORT_PATH), { recursive: true });
  report.generatedAt = new Date().toISOString();
  recomputeSummary(report);
  writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2));
}

// ---------------------------------------------------------------------------
// Run one material through the pipeline
// ---------------------------------------------------------------------------

function classify(predTcK: number | null, mat: BenchmarkMaterial): MaterialResult["status"] {
  if (predTcK == null) return "no-prediction";
  const relErr = Math.abs(predTcK - mat.expTcK) / mat.expTcK;
  if (relErr <= 0.25) return "pass";
  if (relErr <= 0.50) return "warn";
  return "fail";
}

async function runOne(mat: BenchmarkMaterial, runFullDFT: RunFullDFT): Promise<MaterialResult> {
  const t0 = Date.now();
  console.log(`\n[Benchmark] ${mat.formula} @ ${mat.pressureGpa} GPa — experimental Tc = ${mat.expTcK} K (${mat.mechanism})`);
  console.log(`[Benchmark]   ${mat.note}`);

  const base: MaterialResult = {
    formula: mat.formula,
    pressureGpa: mat.pressureGpa,
    expTcK: mat.expTcK,
    mechanism: mat.mechanism,
    predTcK: null,
    predSource: "none",
    lambda: null,
    omegaLogK: null,
    relErr: null,
    status: "error",
    qualityTier: null,
    pipelineError: null,
    wallTimeS: 0,
    ref: mat.ref,
    ranAt: new Date().toISOString(),
  };

  try {
    // ensembleScore = 1.0 so the DFPT / e-ph gate (which normally needs a
    // high score) always fires — a benchmark must exercise the real λ path.
    const result = await runFullDFT(mat.formula, {
      pressureGpa: mat.pressureGpa,
      ensembleScore: 1.0,
    });

    const predTcK =
      (result.dfpt && Number.isFinite(result.dfpt.tcBest) && result.dfpt.tcBest > 0)
        ? result.dfpt.tcBest
        : (result.epw && Number.isFinite(result.epw.tcMigdalEliashberg) && result.epw.tcMigdalEliashberg > 0)
          ? result.epw.tcMigdalEliashberg
          : (result.epw && Number.isFinite(result.epw.tcAllenDynes) && result.epw.tcAllenDynes > 0)
            ? result.epw.tcAllenDynes
            : null;
    const predSource: MaterialResult["predSource"] =
      predTcK == null ? "none"
        : (result.dfpt && result.dfpt.tcBest === predTcK) ? "dfpt-best"
          : (result.epw && result.epw.tcMigdalEliashberg === predTcK) ? "epw-migdal"
            : "epw-allen-dynes";

    base.predTcK = predTcK == null ? null : Number(predTcK.toFixed(2));
    base.predSource = predSource;
    base.lambda = result.dfpt?.lambda ?? null;
    base.omegaLogK = result.dfpt?.omegaLog ?? null;
    base.qualityTier = result.qualityTier ?? null;
    base.pipelineError = result.error ?? null;
    base.relErr = predTcK == null ? null : Number(((predTcK - mat.expTcK) / mat.expTcK).toFixed(4));
    base.status = classify(predTcK, mat);
  } catch (err: any) {
    base.pipelineError = err?.message ? String(err.message).slice(0, 300) : String(err);
    base.status = "error";
    console.error(`[Benchmark] ${mat.formula} threw: ${base.pipelineError}`);
  }

  base.wallTimeS = Number(((Date.now() - t0) / 1000).toFixed(1));
  const predStr = base.predTcK == null ? "—" : `${base.predTcK} K`;
  console.log(`[Benchmark] ${mat.formula}: predicted ${predStr} vs experimental ${mat.expTcK} K → ` +
    `${base.relErr == null ? "no prediction" : `${(base.relErr * 100).toFixed(1)}% error`} [${base.status}] (${base.wallTimeS}s)`);
  return base;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

function printReport(report: Report): void {
  console.log(`\n=== Pipeline benchmark report (${report.results.length}/${BENCHMARK.length} run) ===`);
  const pad = (s: string, n: number) => s.padEnd(n);
  console.log(pad("Material", 10) + pad("P/GPa", 8) + pad("exp Tc", 9) + pad("pred Tc", 10) +
    pad("err%", 9) + pad("λ", 8) + pad("status", 14) + "source");
  for (const r of report.results) {
    console.log(
      pad(r.formula, 10) +
      pad(String(r.pressureGpa), 8) +
      pad(`${r.expTcK} K`, 9) +
      pad(r.predTcK == null ? "—" : `${r.predTcK} K`, 10) +
      pad(r.relErr == null ? "—" : `${(r.relErr * 100).toFixed(1)}`, 9) +
      pad(r.lambda == null ? "—" : r.lambda.toFixed(2), 8) +
      pad(r.status, 14) +
      r.predSource,
    );
  }
  const s = report.summary;
  console.log(`\nScored ${s.scored}/${s.total}  |  pass ${s.pass}  warn ${s.warn}  fail ${s.fail}  ` +
    `no-prediction ${s.noPrediction}  error ${s.error}`);
  console.log(`Mean absolute relative error: ${s.maeRelPct == null ? "n/a" : s.maeRelPct + "%"}`);
  console.log(`Report: ${REPORT_PATH}`);
}

function printDataset(): void {
  console.log(`\n=== Benchmark dataset (${BENCHMARK.length} known superconductors) ===`);
  for (const m of BENCHMARK) {
    console.log(`  ${m.formula.padEnd(8)} ${String(m.pressureGpa).padStart(4)} GPa  ` +
      `Tc=${String(m.expTcK).padStart(6)} ±${m.expTcTolK} K  [${m.mechanism}]`);
    console.log(`    ${m.note}`);
    console.log(`    ${m.ref}`);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const flag = (name: string) => args.includes(name);

  if (flag("--reset")) {
    saveState({ done: [] });
    saveReport({
      generatedAt: new Date().toISOString(),
      results: [],
      summary: { total: BENCHMARK.length, scored: 0, pass: 0, warn: 0, fail: 0, noPrediction: 0, error: 0, maeRelPct: null },
    });
    console.log("[Benchmark] state + report reset.");
    return;
  }

  if (flag("--dry-run")) {
    printDataset();
    return;
  }

  if (flag("--report")) {
    printReport(loadReport());
    return;
  }

  const state = loadState();
  const report = loadReport();

  // Decide which materials to run this invocation.
  let queue: BenchmarkMaterial[];
  const matIdx = args.indexOf("--material");
  if (matIdx >= 0 && args[matIdx + 1]) {
    const want = args[matIdx + 1];
    const m = BENCHMARK.find(x => x.formula.toLowerCase() === want.toLowerCase());
    if (!m) {
      console.error(`[Benchmark] unknown material "${want}". Known: ${BENCHMARK.map(x => x.formula).join(", ")}`);
      process.exitCode = 1;
      return;
    }
    queue = [m];
  } else if (flag("--all")) {
    queue = BENCHMARK.filter(m => !state.done.includes(m.formula));
  } else {
    // Default: process the next undone material only (each is a multi-hour
    // QE job — drive the harness from a loop / cron).
    const next = BENCHMARK.find(m => !state.done.includes(m.formula));
    if (!next) {
      console.log("[Benchmark] all materials already run. Use --reset to re-run, or --report to view.");
      printReport(report);
      return;
    }
    queue = [next];
  }

  // Load the worker only now that we know DFT will actually run.
  const { runFullDFT } = await import("../server/dft/qe-worker");

  for (const mat of queue) {
    const res = await runOne(mat, runFullDFT);
    // Replace any prior result for this formula, then persist immediately so
    // a crash mid-run never loses completed materials.
    report.results = report.results.filter(r => r.formula !== mat.formula);
    report.results.push(res);
    if (!state.done.includes(mat.formula)) state.done.push(mat.formula);
    saveReport(report);
    saveState(state);
  }

  printReport(report);
}

main().catch(err => {
  console.error("[Benchmark] fatal:", err);
  process.exitCode = 1;
});
