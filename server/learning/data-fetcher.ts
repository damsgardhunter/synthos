import { storage } from "../storage";
import type { EventEmitter } from "./engine";

interface OQMDEntry {
  name: string;
  entry_id: number;
  spacegroup: string;
  band_gap: number;
  delta_e: number;
  stability: number;
  composition: string;
}

interface AFLOWEntry {
  compound: string;
  auid: string;
  spacegroup_relax: string;
  Egap: number;
  enthalpy_formation_atom: number;
  species: string[];
}

async function fetchWithTimeout(url: string, timeoutMs = 15000): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { signal: controller.signal });
    return resp;
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchOQMDMaterials(
  emit: EventEmitter,
  limit = 10,
  offset = 0
): Promise<number> {
  let indexed = 0;
  try {
    emit("log", { phase: "phase-4", event: "OQMD fetch started", detail: `Requesting ${limit} entries from OQMD API (offset ${offset})`, dataSource: "OQMD" });

    const url = `http://oqmd.org/oqmdapi/formationenergy?fields=name,entry_id,spacegroup,band_gap,stability,delta_e,composition&limit=${limit}&offset=${offset}&format=json`;
    const resp = await fetchWithTimeout(url, 20000);

    if (!resp.ok) {
      emit("log", { phase: "phase-4", event: "OQMD fetch failed", detail: `HTTP ${resp.status}: ${resp.statusText}`, dataSource: "OQMD" });
      return 0;
    }

    const data = await resp.json() as any;
    const entries: OQMDEntry[] = data?.data ?? [];

    for (const entry of entries) {
      if (!entry.name || !entry.composition) continue;
      const id = `oqmd-live-${entry.entry_id}`;
      try {
        await storage.insertMaterial({
          id,
          name: entry.name,
          formula: entry.composition || entry.name,
          spacegroup: entry.spacegroup || null,
          bandGap: entry.band_gap ?? null,
          formationEnergy: entry.delta_e ?? null,
          stability: entry.stability ?? null,
          source: "OQMD",
          properties: {
            entry_id: entry.entry_id,
            fetchedLive: true,
          },
        });
        indexed++;
      } catch (e) {
      }
    }

    if (indexed > 0) {
      emit("log", { phase: "phase-4", event: "OQMD materials indexed", detail: `Successfully indexed ${indexed} new materials from OQMD`, dataSource: "OQMD" });
      emit("progress", { phase: 4, newItems: indexed });
    }
  } catch (err: any) {
    emit("log", { phase: "phase-4", event: "OQMD fetch error", detail: err.message?.slice(0, 200) || "Unknown error", dataSource: "OQMD" });
  }
  return indexed;
}

export async function fetchAFLOWMaterials(
  emit: EventEmitter,
  species = "Cu",
  limit = 10
): Promise<number> {
  let indexed = 0;
  try {
    emit("log", { phase: "phase-4", event: "AFLOW fetch started", detail: `Requesting materials containing ${species} from AFLOW`, dataSource: "AFLOW" });

    const url = `http://aflow.org/API/aflowlib/?species(${species}),$paging(${limit})&format=json`;
    const resp = await fetchWithTimeout(url, 20000);

    if (!resp.ok) {
      emit("log", { phase: "phase-4", event: "AFLOW fetch failed", detail: `HTTP ${resp.status}`, dataSource: "AFLOW" });
      return 0;
    }

    const entries: AFLOWEntry[] = await resp.json() as any;

    if (!Array.isArray(entries)) {
      emit("log", { phase: "phase-4", event: "AFLOW parse issue", detail: "Response was not an array", dataSource: "AFLOW" });
      return 0;
    }

    for (const entry of entries) {
      if (!entry.compound && !entry.auid) continue;
      const id = `aflow-live-${entry.auid || Math.random().toString(36).slice(2)}`;
      try {
        await storage.insertMaterial({
          id,
          name: entry.compound || `${species} compound`,
          formula: entry.compound || species,
          spacegroup: entry.spacegroup_relax || null,
          bandGap: entry.Egap ?? null,
          formationEnergy: entry.enthalpy_formation_atom ?? null,
          stability: null,
          source: "AFLOW",
          properties: {
            auid: entry.auid,
            species: entry.species,
            fetchedLive: true,
          },
        });
        indexed++;
      } catch (e) {
      }
    }

    if (indexed > 0) {
      emit("log", { phase: "phase-4", event: "AFLOW materials indexed", detail: `Successfully indexed ${indexed} new ${species}-based materials from AFLOW`, dataSource: "AFLOW" });
      emit("progress", { phase: 4, newItems: indexed });
    }
  } catch (err: any) {
    emit("log", { phase: "phase-4", event: "AFLOW fetch error", detail: err.message?.slice(0, 200) || "Unknown error", dataSource: "AFLOW" });
  }
  return indexed;
}

const SPECIES_LIST = ["Cu", "Fe", "Ti", "Al", "Si", "Ni", "Zn", "Co", "Mn", "Cr", "V", "Mo", "W", "Nb", "Ta", "Zr", "Hf", "Sc", "Y", "La"];
let speciesIndex = 0;

export function getNextSpecies(): string {
  const s = SPECIES_LIST[speciesIndex % SPECIES_LIST.length];
  speciesIndex++;
  return s;
}

let oqmdOffset = 0;
export function getNextOQMDOffset(): number {
  const o = oqmdOffset;
  oqmdOffset += 10;
  return o;
}
