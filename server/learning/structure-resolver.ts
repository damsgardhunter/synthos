import { fetchMPStructureData, MPStructureData } from "./materials-project-client";

/**
 * Pre-fetch MP structure data for a batch of formulas.
 * Sequential fetches respect the MP API serial rate-limiter fence.
 * Returns a map from formula → structure (only entries where data was found).
 */
export async function prefetchStructures(formulas: string[]): Promise<Map<string, MPStructureData>> {
  const map = new Map<string, MPStructureData>();

  for (const formula of formulas) {
    try {
      const s = await fetchMPStructureData(formula);
      if (s) map.set(formula, s);
    } catch {
      // Non-fatal: structure data unavailable for this formula; heuristic graph used instead
    }
  }

  return map;
}
