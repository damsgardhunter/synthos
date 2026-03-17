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

  // Fetch from COD API with pagination
  const results: CODEntry[] = [];
  let page = 0;

  while (results.length < maxResults) {
    const url = `${COD_API}?format=json&sg=${spaceGroupNumber}&limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`;
    try {
      const response = await rateLimitedFetch(url);
      if (!response.ok) {
        console.warn(`[COD] SG ${spaceGroupNumber} page ${page}: HTTP ${response.status}`);
        break;
      }
      const rows: CODApiRow[] = await response.json();
      if (!Array.isArray(rows) || rows.length === 0) break;

      const entries = rows.map(parseCODRow).filter(Boolean) as CODEntry[];
      results.push(...entries);
      await cacheEntries(entries);

      if (rows.length < PAGE_SIZE) break;  // last page
      page++;
    } catch (err: any) {
      console.warn(`[COD] SG ${spaceGroupNumber} fetch failed: ${err.message?.slice(0, 80)}`);
      break;
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

  // Build COD query: el=X,Y for "contains these elements"
  const elParam = sorted.join(",");
  const url = `${COD_API}?format=json&el=${elParam}&limit=${Math.min(maxResults, PAGE_SIZE)}`;
  try {
    const response = await rateLimitedFetch(url);
    if (!response.ok) return [];
    const rows: CODApiRow[] = await response.json();
    const entries = (Array.isArray(rows) ? rows : []).map(parseCODRow).filter(Boolean) as CODEntry[];
    await cacheEntries(entries);
    return entries.slice(0, maxResults);
  } catch (err: any) {
    console.warn(`[COD] fetchCODByElements(${elParam}) failed: ${err.message?.slice(0, 80)}`);
    return [];
  }
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
