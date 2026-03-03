import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

interface OQMDEntry {
  name: string;
  entry_id: number;
  spacegroup: string;
  band_gap: number;
  delta_e: number;
  stability: number;
  composition: string;
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

const ELEMENT_FOCUSED_TOPICS = [
  { element: "Cu", focus: "copper-based compounds and alloys (brass, bronze, cuprates, copper oxides)" },
  { element: "Fe", focus: "iron-based materials (steels, iron oxides, iron pnictides, ferrites)" },
  { element: "Ti", focus: "titanium alloys and compounds (Ti-6Al-4V, TiO2, titanium nitride, titanium carbide)" },
  { element: "Al", focus: "aluminum alloys and ceramics (Al2O3, aluminum bronze, aluminosilicates)" },
  { element: "Si", focus: "silicon-based materials (SiC, Si3N4, silicones, silicates, solar-grade silicon)" },
  { element: "Ni", focus: "nickel superalloys and compounds (Inconel, NiTi shape memory, nickel oxides)" },
  { element: "Zn", focus: "zinc compounds and alloys (ZnO, zinc blende, galvanized steel, zinc phosphate)" },
  { element: "Co", focus: "cobalt-based materials (LiCoO2, cobalt alloys, cobalt oxides, stellite)" },
  { element: "Mn", focus: "manganese compounds (MnO2, LiMn2O4, manganese steel, permanganates)" },
  { element: "W", focus: "tungsten and refractory materials (WC, tungsten heavy alloys, W-Re)" },
  { element: "Nb", focus: "niobium compounds (NbTi, Nb3Sn superconductors, niobium carbide, ferroniobium)" },
  { element: "Mo", focus: "molybdenum materials (MoS2, molybdenum alloys, Mo2C, MoSi2)" },
  { element: "Zr", focus: "zirconium compounds (ZrO2 zirconia, Zircaloy nuclear cladding, zirconium carbide)" },
  { element: "Y", focus: "yttrium compounds (Y2O3, YAG laser crystals, YBCO superconductor, yttrium iron garnet)" },
  { element: "La", focus: "lanthanum and rare earth materials (LaAlO3, LaNiO3, lanthanum manganites, LaH10)" },
  { element: "Bi", focus: "bismuth compounds (BSCCO superconductors, bismuth telluride, BiFeO3 multiferroic)" },
  { element: "Sn", focus: "tin-based materials (tin oxides, tin alloys, organotin compounds, Nb3Sn)" },
  { element: "Pb", focus: "lead compounds (PZT piezoelectrics, lead-acid batteries, lead chalcogenides)" },
  { element: "Cr", focus: "chromium materials (stainless steel chromium, Cr2O3, chromium carbides)" },
  { element: "V", focus: "vanadium compounds (V2O5, vanadium redox batteries, vanadium steel, VO2)" },
];
let elementTopicIndex = 0;

export async function fetchElementFocusedMaterials(
  emit: EventEmitter
): Promise<number> {
  let indexed = 0;
  const topic = ELEMENT_FOCUSED_TOPICS[elementTopicIndex % ELEMENT_FOCUSED_TOPICS.length];
  elementTopicIndex++;

  emit("log", {
    phase: "phase-4",
    event: "Element-focused material fetch started",
    detail: `Researching ${topic.element}-based materials from scientific databases`,
    dataSource: "Materials Science DB",
  });

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a materials science database. Draw from peer-reviewed sources: Materials Project, ICSD, Springer Materials, ASM International, and published literature in Nature Materials, Acta Materialia, Journal of Alloys and Compounds.

Generate data about REAL, EXISTING materials containing ${topic.element} that are well-documented in scientific databases. These must be real compounds, not theoretical.

For each material provide:
- 'id' (unique string like "matdb-{formula-simplified}-{random}")
- 'name' (full name)
- 'formula' (correct chemical formula)
- 'spacegroup' (crystallographic space group)
- 'bandGap' (in eV, null for metals)
- 'formationEnergy' (in eV/atom)
- 'stability' (eV above hull, 0 = on hull)
- 'source' ("Materials Science DB")
- 'properties' (object): 'structure', 'density' (g/cm3), 'meltingPoint' (K), 'application', 'discoveredYear'

Return JSON with 'materials' array containing exactly 5 materials. Only real compounds with published data.`,
        },
        {
          role: "user",
          content: `List 5 real, well-characterized materials: ${topic.focus}. Include accurate crystallographic and thermodynamic data.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1200,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return 0;

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      emit("log", { phase: "phase-4", event: "Material DB parse error", detail: content.slice(0, 200), dataSource: "Materials Science DB" });
      return 0;
    }

    const materials = parsed.materials ?? [];

    for (const mat of materials) {
      if (!mat.formula || !mat.name) continue;
      const id = mat.id || `matdb-${mat.formula.replace(/[^a-zA-Z0-9]/g, "").slice(0, 20)}-${Math.random().toString(36).slice(2, 6)}`;
      try {
        await storage.insertMaterial({
          id,
          name: mat.name,
          formula: mat.formula,
          spacegroup: mat.spacegroup || null,
          bandGap: mat.bandGap ?? null,
          formationEnergy: mat.formationEnergy ?? null,
          stability: mat.stability ?? null,
          source: "Materials Science DB",
          properties: mat.properties || {},
        });
        indexed++;
      } catch (e) {
      }
    }

    if (indexed > 0) {
      emit("log", {
        phase: "phase-4",
        event: "Element-focused materials indexed",
        detail: `Indexed ${indexed} ${topic.element}-based materials from scientific databases`,
        dataSource: "Materials Science DB",
      });
      emit("progress", { phase: 4, newItems: indexed });
    }
  } catch (err: any) {
    emit("log", {
      phase: "phase-4",
      event: "Material DB fetch error",
      detail: err.message?.slice(0, 200) || "Unknown error",
      dataSource: "Materials Science DB",
    });
  }

  return indexed;
}

const KNOWN_MATERIAL_TOPICS = [
  {
    category: "beginner",
    topic: "common everyday materials",
    examples: "table salt (NaCl), water ice, steel (Fe-C alloy), glass (SiO2), concrete (calcium silicate hydrate), rubber (polyisoprene), aluminum foil, copper wire, brass (Cu-Zn), bronze (Cu-Sn)"
  },
  {
    category: "beginner",
    topic: "basic ceramics and oxides",
    examples: "porcelain (Al2O3-SiO2-K2O), alumina (Al2O3), zirconia (ZrO2), magnesia (MgO), lime (CaO), silica glass, soda-lime glass, clay minerals (kaolinite), cement (Ca3SiO5)"
  },
  {
    category: "intermediate",
    topic: "semiconductor materials",
    examples: "silicon (Si), germanium (Ge), gallium arsenide (GaAs), indium phosphide (InP), cadmium telluride (CdTe), silicon carbide (SiC), gallium nitride (GaN), zinc oxide (ZnO)"
  },
  {
    category: "intermediate",
    topic: "battery and energy storage materials",
    examples: "lithium cobalt oxide (LiCoO2), lithium iron phosphate (LiFePO4), lithium manganese oxide (LiMn2O4), sodium-sulfur, vanadium redox flow battery electrolyte, solid electrolyte Li7La3Zr2O12 (LLZO)"
  },
  {
    category: "intermediate",
    topic: "structural alloys and composites",
    examples: "stainless steel 304 (Fe-Cr-Ni), titanium alloy Ti-6Al-4V, Inconel 718 (Ni superalloy), carbon fiber reinforced polymer, Kevlar (poly-paraphenylene terephthalamide), maraging steel"
  },
  {
    category: "advanced",
    topic: "known superconducting materials",
    examples: "YBCO (YBa2Cu3O7), BSCCO (Bi2Sr2CaCu2O8), MgB2, NbTi, Nb3Sn, LaH10 (under pressure), iron pnictide BaFe2As2, HgBa2Ca2Cu3O8 (Hg-1223, Tc=133K)"
  },
  {
    category: "advanced",
    topic: "topological and quantum materials",
    examples: "Bi2Se3 (topological insulator), Bi2Te3, SmB6 (Kondo insulator), Weyl semimetal TaAs, Dirac semimetal Cd3As2, Majorana platform InSb nanowire, quantum spin liquid herbertsmithite ZnCu3(OH)6Cl2"
  },
  {
    category: "advanced",
    topic: "advanced functional ceramics",
    examples: "barium titanate BaTiO3 (piezoelectric), PZT Pb(Zr,Ti)O3, yttria-stabilized zirconia YSZ, silicon nitride Si3N4, boron carbide B4C, aluminum nitride AlN, hydroxyapatite Ca10(PO4)6(OH)2"
  },
  {
    category: "master",
    topic: "cutting-edge recently discovered materials",
    examples: "MXenes (Ti3C2Tx), metal-organic frameworks (MOFs like HKUST-1 Cu3(BTC)2), perovskite solar cell materials (CH3NH3PbI3), covalent organic frameworks (COFs), 2D materials beyond graphene (MoS2, WS2, hBN)"
  },
  {
    category: "master",
    topic: "extreme environment and nuclear materials",
    examples: "tungsten (plasma-facing), RAFM steel (reduced activation ferritic/martensitic), uranium dioxide UO2 (nuclear fuel), boron carbide B4C (neutron absorber), hafnium carbide HfC (highest melting point), tantalum hafnium carbide Ta4HfC5"
  },
  {
    category: "master",
    topic: "metamaterials and advanced composites",
    examples: "negative refractive index metamaterials (split-ring resonators), photonic crystals (inverse opals), auxetic materials (negative Poisson ratio), shape memory alloys NiTi (Nitinol), self-healing polymers, aerogels (silica, carbon, graphene)"
  },
  {
    category: "master",
    topic: "high-entropy alloys and novel compositions",
    examples: "CoCrFeMnNi (Cantor alloy), TiZrHfNbTa (refractory HEA), AlCoCrFeNi, high-entropy oxides (Mg,Co,Ni,Cu,Zn)O, high-entropy carbides (Ti,Zr,Hf,Nb,Ta)C, medium-entropy alloys CoCrNi"
  },
];
let knownMaterialIndex = 0;

export async function fetchKnownMaterials(
  emit: EventEmitter
): Promise<number> {
  let indexed = 0;
  const topic = KNOWN_MATERIAL_TOPICS[knownMaterialIndex % KNOWN_MATERIAL_TOPICS.length];
  knownMaterialIndex++;

  emit("log", {
    phase: "phase-4",
    event: "Known materials import started",
    detail: `Importing ${topic.category}-level materials: ${topic.topic}`,
    dataSource: "Materials Science KB",
  });

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a materials science encyclopedia. Your knowledge comes from reputable scientific sources: peer-reviewed journals (Nature, Science, Acta Materialia, Journal of the American Chemical Society), textbooks (Callister's Materials Science, Askeland's Engineering Materials), and established databases (NIST, Materials Project, ICSD).

Generate accurate data about REAL, EXISTING materials that humans have already created and characterized in laboratories. These are not theoretical - they are well-documented in scientific literature.

For each material provide:
- 'id' (unique string like "kb-{formula-simplified}")
- 'name' (full descriptive name with common name in parentheses if applicable)
- 'formula' (correct chemical formula with proper stoichiometry)
- 'spacegroup' (crystallographic space group if known, null if amorphous)
- 'bandGap' (in eV, null for metals, accurate to published values)
- 'formationEnergy' (in eV/atom, negative means stable)
- 'stability' (0 = stable, higher = less stable, in eV above hull)
- 'source' (always "Materials Science KB")
- 'properties' (object with relevant measured properties from literature):
  - For structural materials: 'youngsModulus' (GPa), 'hardness' (Mohs or Vickers), 'density' (g/cm3), 'meltingPoint' (K), 'tensileStrength' (MPa)
  - For electronic materials: 'electronMobility' (cm2/Vs), 'dielectric', 'conductivity' (S/m)
  - For superconductors: 'criticalTemp' (K), 'criticalField' (T), 'type' (Type-I or Type-II)
  - For energy materials: 'capacity' (mAh/g), 'voltage' (V), 'efficiency' (%)
  - Always include: 'structure' (crystal system), 'discoveredYear' (integer or null), 'application' (primary use), 'category' ("${topic.category}")

Return JSON with 'materials' array containing 5-8 materials. Only include materials with well-established, published properties from reputable sources. Do not fabricate property values.`,
        },
        {
          role: "user",
          content: `List 5-8 real, existing ${topic.category}-level materials in the category: ${topic.topic}. Examples to consider: ${topic.examples}. Provide accurate properties from published scientific literature.`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1500,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return 0;

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      emit("log", { phase: "phase-4", event: "Known materials parse error", detail: content.slice(0, 200), dataSource: "Materials Science KB" });
      return 0;
    }

    const materials = parsed.materials ?? [];

    for (const mat of materials) {
      if (!mat.formula || !mat.name) continue;
      const id = mat.id || `kb-${mat.formula.replace(/[^a-zA-Z0-9]/g, "").slice(0, 20)}-${Math.random().toString(36).slice(2, 6)}`;
      try {
        await storage.insertMaterial({
          id,
          name: mat.name,
          formula: mat.formula,
          spacegroup: mat.spacegroup || null,
          bandGap: mat.bandGap ?? null,
          formationEnergy: mat.formationEnergy ?? null,
          stability: mat.stability ?? null,
          source: "Materials Science KB",
          properties: mat.properties || {},
        });
        indexed++;
      } catch (e) {
      }
    }

    if (indexed > 0) {
      emit("log", {
        phase: "phase-4",
        event: "Known materials imported",
        detail: `Indexed ${indexed} real ${topic.category}-level materials for ${topic.topic}`,
        dataSource: "Materials Science KB",
      });
      emit("progress", { phase: 4, newItems: indexed });
    }
  } catch (err: any) {
    emit("log", {
      phase: "phase-4",
      event: "Known materials import error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "Materials Science KB",
    });
  }

  return indexed;
}

let oqmdOffset = 0;
export function getNextOQMDOffset(): number {
  const o = oqmdOffset;
  oqmdOffset += 10;
  return o;
}
