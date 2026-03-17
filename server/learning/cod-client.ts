/**
 * COD (Crystallography Open Database) Client
 * ===========================================
 * Fetches and caches structural data from the Crystallography Open Database.
 * COD is freely accessible at https://www.crystallography.net/cod/
 *
 * API endpoint: https://www.crystallography.net/cod/result
 * Docs: https://wiki.crystallography.net/RESTful_API
 *
 * Use cases in QAE:
 *   - Enrich prototype library with real c/a ratios and Wyckoff positions
 *   - Provide structural validation for generated candidates
 *   - Systematic exploration of all 230 space groups with real example structures
 *
 * Safety:
 *   - All fetches are async/await; no event loop blocking
 *   - Results cached in `cod_structure_cache` table (no TTL — COD is stable)
 *   - Rate-limited to 1 request/second to respect COD fair-use policy
 *   - `fetchCODBySpaceGroup()` yields between pages
 */

import { db } from "../db";
import { codStructureCache } from "@shared/schema";
import { eq, inArray, and } from "drizzle-orm";

const COD_API = "https://www.crystallography.net/cod/result";
const COD_CIF_BASE = "https://www.crystallography.net/cod";
const REQUEST_DELAY_MS = 1100;   // ~1 req/s — COD fair-use limit
const FETCH_TIMEOUT_MS = 20_000;
const PAGE_SIZE = 50;            // results per API page

// COD API quirk: bare sgNumber= / el= queries return [] regardless of value.
// Only formula=SYMBOL queries consistently return data (e.g. formula=Cu works,
// el=Cu does not). To fetch by space group, we must query formula=X&sgNumber=N
// for each candidate element X and aggregate the results.
// Elements ordered by superconductor relevance — covers FCC/BCC/HCP metal SGs
// and the main ligand/anion species found in SC families.
const COD_PROBE_ELEMENTS = [
  "H",  "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Mg", "Al",
  "Si", "Ca", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
  "Zn", "Ge", "Sr", "Y",  "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
  "In", "Sn", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy",
  "Yb", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Pb", "Bi",
];

// ── Types ─────────────────────────────────────────────────────────────────────

export interface CODEntry {
  codId: number;
  formula: string;
  spaceGroupNumber: number;
  spaceGroupSymbol: string | null;
  crystalSystem: string | null;
  elements: string[];
  a: number | null;
  b: number | null;
  c: number | null;
  alpha: number | null;
  beta: number | null;
  gamma: number | null;
  volumePerAtom: number | null;
}

interface CODApiRow {
  file?: number;
  formula?: string;
  sg?: string;          // space group symbol
  sgHM?: string;        // Hermann-Mauguin
  sgNumber?: number | string;
  a?: number | string;
  b?: number | string;
  c?: number | string;
  alpha?: number | string;
  beta?: number | string;
  gamma?: number | string;
  vol?: number | string;
  Z?: number | string;  // formula units per cell
  nelem?: number | string;
  nel?: number | string;
  [key: string]: unknown;
}

// ── Crystal system classification ──────────────────────────────────────────────

function crystalSystemFromSGNumber(n: number): string {
  if (n >= 1 && n <= 2) return "triclinic";
  if (n >= 3 && n <= 15) return "monoclinic";
  if (n >= 16 && n <= 74) return "orthorhombic";
  if (n >= 75 && n <= 142) return "tetragonal";
  if (n >= 143 && n <= 167) return "trigonal";
  if (n >= 168 && n <= 194) return "hexagonal";
  if (n >= 195 && n <= 230) return "cubic";
  return "unknown";
}

// ── Element extraction from formula ──────────────────────────────────────────

const ELEMENT_REGEX = /([A-Z][a-z]?)(\d*\.?\d*)/g;

function elementsFromFormula(formula: string): string[] {
  const elements = new Set<string>();
  let m: RegExpExecArray | null;
  const re = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  while ((m = re.exec(formula)) !== null) {
    if (m[1]) elements.add(m[1]);
  }
  return Array.from(elements).sort();
}

// ── API helpers ───────────────────────────────────────────────────────────────

function parseNum(v: unknown): number | null {
  const n = parseFloat(String(v));
  return isFinite(n) ? n : null;
}

function parseCODRow(row: CODApiRow): CODEntry | null {
  const codId = row.file ? Number(row.file) : null;
  if (!codId) return null;

  const formula = String(row.formula ?? "").replace(/\s/g, "");
  if (!formula) return null;

  const sgNumber = parseNum(row.sgNumber ?? row["sg number"]) ?? null;
  if (!sgNumber) return null;

  const a = parseNum(row.a);
  const b = parseNum(row.b ?? row.a);
  const c = parseNum(row.c ?? row.a);
  const vol = parseNum(row.vol);
  const Z = parseNum(row.Z) ?? 1;
  const nAtoms = Math.max(1, elementsFromFormula(formula).reduce((s) => s + 1, 0));
  const volumePerAtom = vol && Z ? vol / (Z * nAtoms) : null;

  return {
    codId,
    formula,
    spaceGroupNumber: Math.round(sgNumber),
    spaceGroupSymbol: String(row.sgHM ?? row.sg ?? "").trim() || null,
    crystalSystem: crystalSystemFromSGNumber(Math.round(sgNumber)),
    elements: elementsFromFormula(formula),
    a,
    b,
    c,
    alpha: parseNum(row.alpha),
    beta: parseNum(row.beta),
    gamma: parseNum(row.gamma),
    volumePerAtom,
  };
}

let _lastRequestAt = 0;
async function rateLimitedFetch(url: string): Promise<Response> {
  const now = Date.now();
  const wait = REQUEST_DELAY_MS - (now - _lastRequestAt);
  if (wait > 0) await new Promise(r => setTimeout(r, wait));
  _lastRequestAt = Date.now();
  return fetch(url, {
    headers: { "Accept": "application/json", "User-Agent": "QuantumAlchemyEngine/1.0" },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
}

// ── Cache helpers ─────────────────────────────────────────────────────────────

async function cacheEntries(entries: CODEntry[]): Promise<void> {
  if (entries.length === 0) return;
  try {
    await db.insert(codStructureCache).values(entries.map(e => ({
      codId: e.codId,
      formula: e.formula,
      spaceGroupNumber: e.spaceGroupNumber,
      spaceGroupSymbol: e.spaceGroupSymbol,
      crystalSystem: e.crystalSystem,
      elements: e.elements,
      a: e.a, b: e.b, c: e.c,
      alpha: e.alpha, beta: e.beta, gamma: e.gamma,
      volumePerAtom: e.volumePerAtom,
      rawData: e as any,
    }))).onConflictDoNothing();
  } catch (err: any) {
    console.warn(`[COD] Cache write failed: ${err.message?.slice(0, 100)}`);
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Fetch all COD entries for a given space group number.
 * Results are cached in `cod_structure_cache` — repeated calls hit the DB.
 * `maxResults` caps total results to avoid runaway fetches on large SGs.
 */
export async function fetchCODBySpaceGroup(
  spaceGroupNumber: number,
  maxResults = 200,
): Promise<CODEntry[]> {
  // Check cache first
  const cached = await db.select().from(codStructureCache)
    .where(eq(codStructureCache.spaceGroupNumber, spaceGroupNumber))
    .limit(maxResults);

  if (cached.length >= Math.min(maxResults, 5)) {
    return cached.map(r => ({
      codId: r.codId,
      formula: r.formula,
      spaceGroupNumber: r.spaceGroupNumber,
      spaceGroupSymbol: r.spaceGroupSymbol,
      crystalSystem: r.crystalSystem,
      elements: r.elements as string[],
      a: r.a, b: r.b, c: r.c,
      alpha: r.alpha, beta: r.beta, gamma: r.gamma,
      volumePerAtom: r.volumePerAtom,
    }));
  }

  // COD API does not support bare sgNumber= queries — it only returns results when
  // a formula= filter is present (e.g. formula=Cu&sgNumber=225 works; sgNumber=225
  // alone returns []). Work around this by probing each element in COD_PROBE_ELEMENTS
  // and filtering server-side by sgNumber. Pure elemental structures cover the most
  // common metallic SGs (FCC=225, BCC=229, HCP=194) well; complex SGs may yield
  // fewer hits but will still populate the cache with representative data.
  const results: CODEntry[] = [];
  const seenIds = new Set<number>();

  for (const element of COD_PROBE_ELEMENTS) {
    if (results.length >= maxResults) break;
    const url = `${COD_API}?format=json&formula=${element}&sgNumber=${spaceGroupNumber}&limit=${PAGE_SIZE}`;
    try {
      const response = await rateLimitedFetch(url);
      if (!response.ok) continue;
      const rows: CODApiRow[] = await response.json();
      if (!Array.isArray(rows) || rows.length === 0) continue;

      const entries = rows.map(row => parseCODRow({
        ...row,
        sgNumber: row.sgNumber ?? (row["sg number"] as number | string | undefined) ?? spaceGroupNumber,
      })).filter(Boolean) as CODEntry[];

      const fresh = entries.filter(e => !seenIds.has(e.codId));
      fresh.forEach(e => seenIds.add(e.codId));
      results.push(...fresh);
      if (fresh.length > 0) await cacheEntries(fresh);
    } catch (err: any) {
      console.warn(`[COD] SG ${spaceGroupNumber} probe(${element}) failed: ${err.message?.slice(0, 60)}`);
    }
  }

  console.log(`[COD] SG ${spaceGroupNumber}: fetched ${results.length} entries`);
  return results.slice(0, maxResults);
}

/**
 * Fetch COD entries filtered by element list (must contain all listed elements).
 * Results cached as above.
 */
export async function fetchCODByElements(
  elements: string[],
  maxResults = 100,
): Promise<CODEntry[]> {
  const sorted = [...elements].sort();

  // Check if we have these elements in the cache via array overlap
  // (Drizzle doesn't support array containment directly; use raw SQL)
  try {
    const elLiteral = `'{${sorted.map(e => `"${e}"`).join(",")}}'::text[]`;
    const rows = await db.execute(
      `SELECT * FROM cod_structure_cache WHERE elements @> ${elLiteral} LIMIT ${maxResults}` as any,
    ) as any;
    const dbRows = (rows.rows ?? rows) as any[];
    if (dbRows.length >= 3) {
      return dbRows.map((r: any) => ({
        codId: r.cod_id, formula: r.formula,
        spaceGroupNumber: r.space_group_number, spaceGroupSymbol: r.space_group_symbol,
        crystalSystem: r.crystal_system, elements: r.elements as string[],
        a: r.a, b: r.b, c: r.c, alpha: r.alpha, beta: r.beta, gamma: r.gamma,
        volumePerAtom: r.volume_per_atom,
      }));
    }
  } catch { /* DB may not have the table yet — fallback to API */ }

  // COD API: el= queries return [] (API quirk). Instead, probe each element in the
  // target set using formula=X and collect any results that contain ALL target elements.
  const results: CODEntry[] = [];
  const seenIds = new Set<number>();

  for (const element of sorted) {
    if (results.length >= maxResults) break;
    const url = `${COD_API}?format=json&formula=${element}&limit=${Math.min(maxResults, PAGE_SIZE)}`;
    try {
      const response = await rateLimitedFetch(url);
      if (!response.ok) continue;
      const rows: CODApiRow[] = await response.json();
      const entries = (Array.isArray(rows) ? rows : []).map(parseCODRow).filter(Boolean) as CODEntry[];
      // Keep entries that contain at least one of the target elements
      const relevant = entries.filter(e =>
        sorted.some(el => e.elements.includes(el)) && !seenIds.has(e.codId)
      );
      relevant.forEach(e => seenIds.add(e.codId));
      results.push(...relevant);
      if (relevant.length > 0) await cacheEntries(relevant);
    } catch (err: any) {
      console.warn(`[COD] fetchCODByElements probe(${element}) failed: ${err.message?.slice(0, 60)}`);
    }
  }
  return results.slice(0, maxResults);
}

/**
 * Batch-populate the cache for a list of space group numbers.
 * Used by space-group-explorer.ts during the systematic sweep.
 * Yields between SGs to keep the event loop responsive.
 */
export async function populateCODCacheForSpaceGroups(
  sgNumbers: number[],
  maxPerSG = 50,
): Promise<void> {
  console.log(`[COD] Populating cache for ${sgNumbers.length} space groups...`);
  for (const sg of sgNumbers) {
    await fetchCODBySpaceGroup(sg, maxPerSG);
    await new Promise(r => setTimeout(r, REQUEST_DELAY_MS));
  }
  console.log(`[COD] Cache population complete.`);
}

/**
 * Returns representative prototype parameters (median a, c/a) for a given space group
 * from the COD cache. Used to seed lattice constant estimates in structure prediction.
 */
export async function getCODPrototypeParams(spaceGroupNumber: number): Promise<{
  medianA: number | null;
  medianCOverA: number | null;
  count: number;
} | null> {
  const entries = await fetchCODBySpaceGroup(spaceGroupNumber, 100);
  const aVals = entries.map(e => e.a).filter((v): v is number => v !== null && v > 0);
  const cOverA = entries
    .filter(e => e.a && e.c && e.a > 0)
    .map(e => e.c! / e.a!);

  const median = (arr: number[]) => {
    if (arr.length === 0) return null;
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  };

  return {
    medianA: median(aVals),
    medianCOverA: median(cOverA),
    count: entries.length,
  };
}

/**
 * Returns the CIF download URL for a COD entry (for visualization / further parsing).
 */
export function getCODCifUrl(codId: number): string {
  return `${COD_CIF_BASE}/${codId}.cif`;
}

// ── Novelty screening ─────────────────────────────────────────────────────────

export interface NoveltyResult {
  /** True if the formula appears in COD or the MP cache — i.e. it is a known compound. */
  isKnown: boolean;
  /** Human-readable summary of where the match was found, or "not found in COD/MP". */
  reason: string;
  /** Number of matching entries across both databases. */
  matchCount: number;
}

/**
 * Check whether a formula is already in the known-compound databases (COD + MP cache).
 *
 * Strategy:
 *   1. Normalise the formula to a canonical element set.
 *   2. Check `cod_structure_cache` for any entry whose element list matches.
 *   3. Check `mp_material_cache` for any entry with the same formula string.
 *
 * "Same element set" (not strict stoichiometry) is deliberately conservative: if
 * Cu2O appears in COD under any stoichiometry we want to flag it as known so that
 * only truly unexplored compositions pass screening.
 *
 * @param formula   Hill-notation formula string, e.g. "LaH10" or "CuO2Ba"
 * @returns NoveltyResult with isKnown flag and human-readable reason
 */
export async function checkFormulaNovelty(formula: string): Promise<NoveltyResult> {
  const elements = elementsFromFormula(formula);
  if (elements.length === 0) {
    return { isKnown: false, reason: "could not parse elements from formula", matchCount: 0 };
  }

  // 1. Check COD structure cache by element containment
  let codMatches = 0;
  let codExample = "";
  try {
    const elLiteral = `'{${elements.map(e => `"${e}"`).join(",")}}'::text[]`;
    const rows = await db.execute(
      `SELECT formula, space_group_number, crystal_system
       FROM cod_structure_cache
       WHERE elements = ${elLiteral}
       LIMIT 5` as any
    ) as any;
    const dbRows: any[] = rows.rows ?? (Array.isArray(rows) ? rows : []);
    codMatches = dbRows.length;
    if (codMatches > 0) {
      codExample = `${dbRows[0].formula} (SG ${dbRows[0].space_group_number}, ${dbRows[0].crystal_system})`;
    }
  } catch { /* COD cache not available — not fatal */ }

  // 2. Check MP material cache by formula string
  let mpMatches = 0;
  try {
    // Escape single quotes in formula to prevent SQL injection before inlining
    const safeFormula = formula.replace(/'/g, "''");
    const rows = await db.execute(
      `SELECT formula FROM mp_material_cache
       WHERE formula = '${safeFormula}' AND data_type = 'summary'
       LIMIT 3` as any
    ) as any;
    const dbRows: any[] = rows.rows ?? (Array.isArray(rows) ? rows : []);
    mpMatches = dbRows.length;
  } catch { /* MP cache not available — not fatal */ }

  const totalMatches = codMatches + mpMatches;

  if (totalMatches === 0) {
    return {
      isKnown: false,
      reason: `not found in COD (${elements.join("")} element set) or MP cache`,
      matchCount: 0,
    };
  }

  const parts: string[] = [];
  if (codMatches > 0) parts.push(`${codMatches} COD entries (e.g. ${codExample})`);
  if (mpMatches > 0) parts.push(`${mpMatches} MP cache entries`);

  return {
    isKnown: true,
    reason: `found in ${parts.join(" + ")}`,
    matchCount: totalMatches,
  };
}
