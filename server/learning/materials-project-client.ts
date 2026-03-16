import { db } from "../db";
import { mpMaterialCache } from "@shared/schema";
import { eq, and } from "drizzle-orm";

// Curated list of metallic and superconductor compounds to seed MP lookups.
// MP API filter queries (is_metal, material_ids) only return material_id with no other fields,
// so we fall back to per-formula fetchSummary using this list.
const METALLIC_SEED_FORMULAS: string[] = [
  // --- Elemental metals ---
  "Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba",
  "Sc","Y","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Ti","Zr","Hf","V","Nb","Ta","Cr","Mo","W","Mn","Tc","Re",
  "Fe","Co","Ni","Cu","Zn","Ru","Rh","Pd","Ag","Cd",
  "Os","Ir","Pt","Au","Hg","Tl","Al","Ga","In","Sn","Pb","Bi","Ge","Sb","As","Se","Te",

  // --- Binary nitrides/carbides (hard superconductors) ---
  "NbN","NbC","TaN","TaC","MoN","MoC","VN","VC","WC","WN",
  "TiN","TiC","ZrN","ZrC","HfN","HfC","ScN","YN","LaN",
  "CrN","CrC","MnN","FeN","CoN","NiN","CuN",
  "CeN","PrN","NdN","GdN","TbN","DyN","HoN","ErN","TmN","YbN","LuN",
  "Mo2C","W2C","V2C","Nb2C","Ta2C","Cr3C2","Fe3C","Mn3C",
  "Si3N4","BN","AlN","GaN","InN",

  // --- Borides ---
  "MgB2","MgB4","AlB2","CaB6","LaB6","CeB6","PrB6","NdB6","SmB6","EuB6",
  "GdB6","TbB6","DyB6","HoB6","ErB6","TmB6","YbB6","LuB6",
  "TiB2","ZrB2","HfB2","VB2","NbB2","TaB2","CrB2","MoB2","WB2",
  "FeB","Fe2B","CoB","Co2B","NiB","Ni3B","NiB3","LiB",

  // --- A15 intermetallics (high-Tc conventional SCs) ---
  "Nb3Sn","Nb3Al","Nb3Ge","Nb3Ga","Nb3Si","Nb3Pt","Nb3Au","Nb3Sb","Nb3Os",
  "V3Si","V3Ga","V3Ge","V3As","V3Sb","V3Sn","V3In",
  "Mo3Re","Mo3Os","Mo3Ir","Mo3Pt","Mo3Al","Mo3Ga",
  "Cr3Si","Cr3Ge","Cr3As","Cr3Os",
  "W3Os","W3Re","W3Ir","Ti3Sn","Ti3Al",

  // --- Laves phases (AB2) ---
  "TiFe2","TiCo2","TiNi2","TiMn2","TiCr2",
  "ZrFe2","ZrCo2","ZrNi2","ZrMn2","ZrCr2",
  "HfFe2","HfCo2","HfNi2",
  "YFe2","YCo2","YNi2","YMn2","YAl2",
  "GdFe2","GdCo2","GdNi2","SmFe2","NdFe2","CeFe2",
  "LaFe2","LaAl2","GdAl2","CeAl2","ThFe2","ThNi2","ThCo2",
  "MgCu2","MgNi2","MgZn2","CaAl2","SrAl2","BaAl2",

  // --- Heusler alloys ---
  "Cu2MnAl","Cu2MnIn","Cu2MnSn","Cu2FeAl","Cu2FeSn",
  "Ni2MnGa","Ni2MnAl","Ni2MnIn","Ni2MnSn","Ni2MnSb",
  "Co2MnSi","Co2MnGe","Co2MnSn","Co2MnGa","Co2MnAl",
  "Co2FeSi","Co2FeGe","Co2FeAl","Co2CrAl","Co2CrGa",
  "PdMnSb","PtMnSb","NiMnSb","CoMnSb","FeMnSb",

  // --- Hydrides ---
  "PdH","PdH2","TiH2","ZrH2","NbH","HfH2","VH","VH2",
  "LaH3","YH3","CeH3","PrH3","NdH3","GdH3","TbH3",
  "DyH3","HoH3","ErH3","TmH3","LuH3","SmH3","EuH2",
  "LaH10","YH10","CeH9","PrH9","ThH10","LaH6","YH6","CaH6","BaH6","SrH6","MgH6",
  "ScH3","ScH6","ThH4","UH3",

  // --- Silicides ---
  "Mg2Si","Mg2Ge","Mg2Sn","FeSi","FeSi2","Fe3Si",
  "CoSi","CoSi2","NiSi","NiSi2","Ni3Si","PtSi","PdSi",
  "CrSi2","MoSi2","WSi2","TiSi2","ZrSi2","HfSi2","NbSi2","TaSi2","VSi2",
  "CaSi2","BaSi2","SrSi2","LaSi","CeSi","NdSi","SmSi",

  // --- Phosphides (topological semimetals + SCs) ---
  "NbP","TaP","WP","MoP","CrP","TiP","ZrP","HfP",
  "FeP","CoP","NiP","RhP","IrP","PdP","PtP",
  "InP","GaP","AlP","BP",

  // --- Dichalcogenides (CDW + SC) ---
  "MoS2","WS2","MoSe2","WSe2","MoTe2","WTe2",
  "NbSe2","TaSe2","NbS2","TaS2","TiSe2","TiS2",
  "ZrSe2","ZrS2","HfSe2","HfS2","VSe2","VTe2",
  "CrSe2","CrTe2","PtSe2","PdSe2","PtS2","PdS2",
  "IrSe2","RhSe2","CoSe2","NiSe2","FeSe2","FeS2","ReS2","ReSe2",

  // --- Tellurides / selenides ---
  "Bi2Te3","Bi2Se3","Sb2Te3","As2Te3",
  "PbTe","PbS","PbSe","SnTe","SnSe","GeTe","GeSe",
  "HgTe","HgSe","HgS","CdTe","CdSe","CdS","ZnTe","ZnSe","ZnS",
  "In2Te3","Tl2Te3","Ag2Te","Ag2Se","Ag2S","Cu2Se","Cu2S",

  // --- Skutterudites ---
  "CoSb3","RhSb3","IrSb3","CoAs3","CoP3","RhAs3","IrAs3","NiP3","PdAs3","PtSb3",

  // --- Iron-based superconductors ---
  "FeSe","FeS","FeTe","FeAs","LiFeAs",
  "LaFeAsO","NdFeAsO","SmFeAsO","PrFeAsO","CeFeAsO","GdFeAsO",
  "BaFe2As2","SrFe2As2","CaFe2As2","EuFe2As2",
  "Fe3GaTe2","Fe3GeTe2","Fe3SnTe2",

  // --- Cuprate superconductors ---
  "CuO","La2CuO4","Nd2CuO4","Sm2CuO4","Eu2CuO4","Gd2CuO4","Pr2CuO4",
  "YBa2Cu3O7","NdBa2Cu3O7","EuBa2Cu3O7","GdBa2Cu3O7",
  "Bi2Sr2CuO6","Bi2Sr2CaCu2O8","Bi2Sr2Ca2Cu3O10",
  "Tl2Ba2CuO6","Tl2Ba2CaCu2O8","Tl2Ba2Ca2Cu3O10",
  "HgBa2CuO4","HgBa2CaCu2O6","HgBa2Ca2Cu3O8",

  // --- Heavy fermion compounds ---
  "UPt3","UPd2Al3","UNi2Al3","URu2Si2","UBe13","UPd3",
  "CeCu2Si2","CeCu2Ge2","CeNi2Ge2","CeRu2Si2","CeRu2Ge2",
  "CeCoIn5","CeRhIn5","CeIrIn5","PuCoGa5","PuRhGa5",
  "YbRh2Si2","YbCu2Si2","YbNi2Ge2",

  // --- Graphite intercalation compounds ---
  "CaC6","KC8","RbC8","CsC8","LiC6","NaC6","SrC6","BaC6",

  // --- Alkali fullerides ---
  "K3C60","Rb3C60","Cs3C60","RbCs2C60","K2RbC60",

  // --- Perovskite oxides (metallic) ---
  "SrTiO3","BaTiO3","KTaO3","SrVO3","CaVO3","BaVO3",
  "BaMoO3","SrMoO3","CaMoO3","SrRuO3","CaRuO3","BaRuO3",
  "LaTiO3","YTiO3","LaCrO3","LaFeO3","LaCoO3","LaNiO3","LaMnO3",
  "BaPbO3","BaBiO3","SrPbO3","Ba2PbO4","Ba2SnO4",
  "Sr2RuO4","Ca2RuO4","Sr3Ru2O7","Ca3Ru2O7",

  // --- Simple metallic oxides ---
  "IrO2","RuO2","OsO2","RhO2","ReO3","WO3","MoO3","VO2","V2O3",
  "TiO","TiO2","NbO","NbO2","Nb2O5","V2O5","Ta2O5","Cr2O3","MnO2","FeO","Fe3O4","CoO","NiO","CuO",

  // --- Antimonides ---
  "NiSb","CoSb","FeSb","MnSb","CrSb","VSb","NbSb","TaSb",
  "PdSb","PtSb","RhSb","IrSb","OsSb","ReSb",
  "InSb","GaSb","AlSb","ZnSb","CdSb",

  // --- Other binary intermetallics ---
  "La3In","La3Tl","YPd2","LaPd2","NbPd3","MoPd3",
  "LaRh3","LaRu2","YRh3","CeRu2","LuRu2","GdRu2",
  "MoRe","NbRe","WRe","OsRe","TcRe",
  "InSn","InPb","GaIn","BiPb","BiTl","SbSn","SbPb",
  "NiAl","Ni3Al","NiAl3","FeAl","Fe3Al","CoAl","Co3Al",
  "PdAl","PtAl","IrAl","RhAl","TiAl","Ti3Al","TiAl3",
  "ZrAl2","HfAl2","NbAl3","TaAl3","MoAl","WAl",
  "Cu3Au","CuAu","CuAu3","Pd3Fe","Pd3Co","Pd3Mn","Pd3Cr",
  "Pt3Fe","Pt3Co","Pt3Mn","Pt3Cr","Pt3Ti","Pt3Zr",
  "FePt","FePd","CoPt","MnPt","MnPd",

  // --- Alkali-metal pnictides (electrides/SCs) ---
  "K3Bi","Rb3Bi","Cs3Bi","K3Sb","Rb3Sb","Cs3Sb",
  "RbCsSb","K2CsSb","Na3Bi","Li3Bi","Li3As","Na3As",

  // --- Topological semimetals ---
  "TaAs","NbAs","TaP","NbP","WTe2","MoTe2",
  "Cd3As2","ZrSiS","ZrSiSe","ZrSiTe","ZrSnSe",
  "TlBiSe2","TlBiTe2","TlBiS2",

  // --- Bismuthates ---
  "BaBiO3","BaBi3","Ba2BiO4","BaBi2O6",

  // --- Rare-earth intermetallics ---
  "SmCo5","SmCo3","Sm2Co17","Sm2Co7",
  "NdFe14B","NdFe11Ti","SmFe11Ti",
  "GdFe2","GdCo5","GdNi5","TbFe2","DyFe2","HoFe2","ErFe2",
  "CeRu4Sn6","CeOs4Sn12","LuRu4Sn6",
  "YbFe4Sb12","LaFe4Sb12","CeOs4Sb12","PrOs4Sb12",

  // --- Organic-adjacent / cage compounds ---
  "CoAs3","NiAs","NiAs2","CoAs2","FeAs2","MnAs",

  // --- Additional chalcogenides + ternaries ---
  "Cu2ZnSnS4","Cu2ZnSnSe4","Cu2ZnGeS4","Cu2CdSnS4",
  "AgGaS2","AgGaSe2","AgInS2","AgInSe2","CuGaS2","CuInS2","CuInSe2",
  "ZnGeAs2","CdGeAs2","ZnSiAs2","CdSiAs2",
];

const MP_API_BASE = "https://api.materialsproject.org";

function getApiKey(): string | null {
  return process.env.MATERIALS_PROJECT_API_KEY || null;
}

export interface MPSummaryData {
  mpId: string;
  formula: string;
  bandGap: number;
  formationEnergyPerAtom: number;
  energyAboveHull: number;
  isStable: boolean;
  spaceGroup: string;
  crystalSystem: string;
  volume: number;
  density: number;
  nsites: number;
  magneticOrdering: string;
  totalMagnetization: number;
  isMetallic: boolean;
  efermi: number | null;
}

export interface MPElasticityData {
  bulkModulus: number;
  shearModulus: number;
  youngsModulus: number;
  poissonRatio: number;
  bulkModulusVoigt: number;
  bulkModulusReuss: number;
  shearModulusVoigt: number;
  shearModulusReuss: number;
}

export interface MPPhononData {
  hasPhononData: boolean;
  lastPhononFreq: number | null;
  phononBandStructure: any | null;
  phononDos: any | null;
}

export interface MPElectronicStructureData {
  dosAtFermi: number | null;
  bandStructureType: string | null;
  fermiEnergy: number | null;
  bandGap: number | null;
  isMetal: boolean;
}

export interface MPThermoData {
  energyAboveHull: number;
  debyeTemperature: number | null;
  formationEnergyPerAtom: number;
  isStable: boolean;
  uncorrectedEnergy: number | null;
}

export interface MPMagnetismData {
  ordering: string;
  totalMagnetization: number;
  totalMagnetizationNormalized: number;
  types: string[];
}

async function getCachedData(formula: string, dataType: string): Promise<any | null> {
  try {
    const normalizedFormula = normalizeFormula(formula);
    const results = await db
      .select()
      .from(mpMaterialCache)
      .where(
        and(
          eq(mpMaterialCache.formula, normalizedFormula),
          eq(mpMaterialCache.dataType, dataType)
        )
      )
      .limit(1);

    if (results.length > 0) {
      const cached = results[0];
      const age = Date.now() - new Date(cached.fetchedAt!).getTime();
      const maxAge = 7 * 24 * 60 * 60 * 1000;
      if (age < maxAge) {
        return cached.data;
      }
    }
    return null;
  } catch {
    return null;
  }
}

async function setCachedData(formula: string, dataType: string, data: any, mpId?: string): Promise<void> {
  try {
    const normalizedFormula = normalizeFormula(formula);
    await db
      .insert(mpMaterialCache)
      .values({
        formula: normalizedFormula,
        mpId: mpId || null,
        dataType,
        data,
      })
      .onConflictDoUpdate({
        target: [mpMaterialCache.formula, mpMaterialCache.dataType],
        set: {
          data,
          mpId: mpId || null,
          fetchedAt: new Date(),
        },
      });
  } catch {
  }
}

function normalizeFormula(formula: string): string {
  if (typeof formula !== "string") formula = String(formula ?? "");
  return formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
}

/**
 * Returns true if the formula is not a plain inorganic compound formula that
 * the MP API can handle. Rejects:
 *  - Chemical-system notation: "Fe-C"
 *  - Polymer / repeat notation: "(C5H8)n", "(AB)x"
 *  - Any formula containing parentheses, brackets, or non-element characters
 */
function isInvalidMPFormula(formula: string): boolean {
  if (/[-()[\]{}]/.test(formula)) return true;      // hyphens, brackets
  if (/[a-z]$/.test(formula)) return true;           // ends with lowercase (polymer 'n', 'x', …)
  if (!/^([A-Z][a-z]?\d*)+$/.test(formula)) return true; // not purely Element+digits
  return false;
}

async function mpFetch(endpoint: string, params: Record<string, string> = {}, attempt = 1): Promise<any | null> {
  const apiKey = getApiKey();
  if (!apiKey) return null;

  const url = new URL(`${MP_API_BASE}${endpoint}`);
  for (const [k, v] of Object.entries(params)) {
    url.searchParams.set(k, v);
  }

  try {
    const response = await fetch(url.toString(), {
      headers: {
        "X-API-KEY": apiKey,
        "Accept": "application/json",
      },
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      let body = "";
      try { body = await response.text(); } catch {}
      if (response.status === 429) {
        console.warn(`[MP API] Rate limited (429): ${body.slice(0, 300)}`);
      } else if (response.status === 403) {
        console.warn(`[MP API] Forbidden (403) — check API key: ${body.slice(0, 300)}`);
      } else {
        console.warn(`[MP API] HTTP ${response.status} for ${url.pathname}?${url.searchParams}: ${body.slice(0, 500)}`);
      }
      return null;
    }

    return await response.json();
  } catch (err: any) {
    if (err?.name === "AbortError" || err?.name === "TimeoutError") {
      console.warn(`[MP API] Timeout (30s) for ${endpoint} (attempt ${attempt})`);
    } else {
      const cause = err?.cause?.message || err?.cause?.code || "";
      console.warn(`[MP API] Request failed for ${endpoint} (attempt ${attempt}): ${err?.message || "unknown"}${cause ? ` (cause: ${cause})` : ""}`);
    }
    if (attempt < 3) {
      await new Promise(r => setTimeout(r, attempt * 3000));
      return mpFetch(endpoint, params, attempt + 1);
    }
    return null;
  }
}

export async function fetchSummary(formula: string): Promise<MPSummaryData | null> {
  const cached = await getCachedData(formula, "summary");
  if (cached) return cached as MPSummaryData;

  const normalizedFormula = normalizeFormula(formula);
  if (isInvalidMPFormula(normalizedFormula)) return null;

  const data = await mpFetch("/materials/summary/", {
    formula: normalizedFormula,
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const result: MPSummaryData = {
    mpId: entry.material_id,
    formula: entry.formula_pretty || normalizedFormula,
    bandGap: entry.band_gap ?? 0,
    formationEnergyPerAtom: entry.formation_energy_per_atom ?? 0,
    energyAboveHull: entry.energy_above_hull ?? 0,
    isStable: entry.is_stable ?? false,
    spaceGroup: entry.symmetry?.symbol ?? "",
    crystalSystem: entry.symmetry?.crystal_system ?? "",
    volume: entry.volume ?? 0,
    density: entry.density ?? 0,
    nsites: entry.nsites ?? 1,
    magneticOrdering: entry.ordering ?? "NM",
    totalMagnetization: entry.total_magnetization ?? 0,
    isMetallic: entry.is_metal ?? false,
    efermi: entry.efermi ?? null,
  };

  await setCachedData(formula, "summary", result, result.mpId);
  return result;
}

export async function fetchElasticity(formula: string): Promise<MPElasticityData | null> {
  const cached = await getCachedData(formula, "elasticity");
  if (cached) return cached as MPElasticityData;

  // Elasticity endpoint rejects both formula and fields; look up material_id first
  const summary = await fetchSummary(formula);
  if (!summary?.mpId) return null;

  const data = await mpFetch("/materials/elasticity/", {
    material_ids: summary.mpId,
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const bm = entry.bulk_modulus || {};
  const sm = entry.shear_modulus || {};

  const result: MPElasticityData = {
    bulkModulus: bm.vrh ?? 0,
    shearModulus: sm.vrh ?? 0,
    youngsModulus: entry.young_modulus ?? 0,
    poissonRatio: entry.poisson_ratio ?? 0.3,
    bulkModulusVoigt: bm.voigt ?? bm.vrh ?? 0,
    bulkModulusReuss: bm.reuss ?? bm.vrh ?? 0,
    shearModulusVoigt: sm.voigt ?? sm.vrh ?? 0,
    shearModulusReuss: sm.reuss ?? sm.vrh ?? 0,
  };

  await setCachedData(formula, "elasticity", result, summary.mpId);
  return result;
}

export async function fetchMagnetism(formula: string): Promise<MPMagnetismData | null> {
  const cached = await getCachedData(formula, "magnetism");
  if (cached) return cached as MPMagnetismData;

  // Magnetism endpoint rejects formula; look up material_id first
  const summary = await fetchSummary(formula);
  if (!summary?.mpId) return null;

  const data = await mpFetch("/materials/magnetism/", {
    material_ids: summary.mpId,
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const result: MPMagnetismData = {
    ordering: entry.ordering ?? "NM",
    totalMagnetization: entry.total_magnetization ?? 0,
    totalMagnetizationNormalized: entry.total_magnetization_normalized_formula_units ?? 0,
    types: entry.types ?? [],
  };

  await setCachedData(formula, "magnetism", result, summary.mpId);
  return result;
}

export async function fetchPhonon(formula: string): Promise<MPPhononData | null> {
  const cached = await getCachedData(formula, "phonon");
  if (cached) return cached as MPPhononData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/phonon/", {
    formula: normalizedFormula,
    _limit: "1",
  });

  if (!data?.data?.length) {
    const noData: MPPhononData = { hasPhononData: false, lastPhononFreq: null, phononBandStructure: null, phononDos: null };
    await setCachedData(formula, "phonon", noData);
    return noData;
  }

  const entry = data.data[0];
  const result: MPPhononData = {
    hasPhononData: true,
    lastPhononFreq: entry.last_phonon_freq ?? null,
    phononBandStructure: entry.ph_bs ? { available: true } : null,
    phononDos: entry.ph_dos ? { available: true } : null,
  };

  await setCachedData(formula, "phonon", result, entry.material_id);
  return result;
}

export async function fetchElectronicStructure(formula: string): Promise<MPElectronicStructureData | null> {
  const cached = await getCachedData(formula, "electronic_structure");
  if (cached) return cached as MPElectronicStructureData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/electronic_structure/", {
    formula: normalizedFormula,
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];

  const bandStructureType: string | null = entry.band_gap != null
    ? (entry.band_gap > 0 ? (entry.is_metal ? "metallic" : "indirect") : "metallic")
    : null;

  const result: MPElectronicStructureData = {
    dosAtFermi: null,
    bandStructureType,
    fermiEnergy: entry.efermi ?? null,
    bandGap: entry.band_gap ?? null,
    isMetal: entry.is_metal ?? false,
  };

  await setCachedData(formula, "electronic_structure", result, entry.material_id);
  return result;
}

export async function fetchThermo(formula: string): Promise<MPThermoData | null> {
  const cached = await getCachedData(formula, "thermo");
  if (cached) return cached as MPThermoData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/thermo/", {
    formula: normalizedFormula,
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const result: MPThermoData = {
    energyAboveHull: entry.energy_above_hull ?? 0,
    debyeTemperature: entry.debye_temperature ?? null,
    formationEnergyPerAtom: entry.formation_energy_per_atom ?? 0,
    isStable: entry.is_stable ?? false,
    uncorrectedEnergy: entry.uncorrected_energy_per_atom ?? null,
  };

  await setCachedData(formula, "thermo", result, entry.material_id);
  return result;
}

export async function fetchThermodynamicStability(formula: string): Promise<{ energyAboveHull: number; isStable: boolean; decompositionProducts: string[] } | null> {
  const summary = await fetchSummary(formula);
  if (!summary) return null;
  return {
    energyAboveHull: summary.energyAboveHull,
    isStable: summary.isStable,
    decompositionProducts: [],
  };
}

export async function fetchAllData(formula: string): Promise<{
  summary: MPSummaryData | null;
  elasticity: MPElasticityData | null;
  magnetism: MPMagnetismData | null;
  phonon: MPPhononData | null;
  electronicStructure: MPElectronicStructureData | null;
  thermo: MPThermoData | null;
}> {
  // Sequential fetches to avoid hammering the MP API with 6 simultaneous requests
  const summary = await fetchSummary(formula);
  const elasticity = await fetchElasticity(formula);
  const magnetism = await fetchMagnetism(formula);
  const phonon = await fetchPhonon(formula);
  const electronicStructure = await fetchElectronicStructure(formula);
  const thermo = await fetchThermo(formula);

  return { summary, elasticity, magnetism, phonon, electronicStructure, thermo };
}

export function isApiAvailable(): boolean {
  return !!getApiKey();
}

export async function fetchCachedFormationEnergies(formulas: string[]): Promise<Map<string, number>> {
  const result = new Map<string, number>();
  for (const formula of formulas) {
    const cached = await getCachedData(formula, "summary");
    if (cached && (cached as MPSummaryData).formationEnergyPerAtom != null) {
      result.set(formula, (cached as MPSummaryData).formationEnergyPerAtom);
    }
  }
  return result;
}

/**
 * Fetches a paginated batch of metallic materials directly from the MP API.
 * Used for progressive GNN augmentation: call with skip=0 for batch 1, skip=500 for batch 2, etc.
 * Caches each result in the DB so future startups don't re-fetch.
 */
export async function fetchMPBatchFromAPI(limit: number, skip: number): Promise<MPGNNSeedRecord[]> {
  const results: MPGNNSeedRecord[] = [];
  if (!getApiKey()) {
    console.log(`[MP-Batch] No API key available, skipping batch fetch (skip=${skip})`);
    return results;
  }
  try {
    const summaries = await fetchSeedFormulaSlice(limit, skip);
    for (const s of summaries) {
      results.push({
        formula: s.formula,
        bandGap: s.bandGap,
        formationEnergy: s.formationEnergyPerAtom,
        spaceGroup: s.spaceGroup || null,
        crystalSystem: s.crystalSystem || null,
        isMetallic: s.isMetallic,
      });
    }
    console.log(`[MP-Batch] Fetched ${results.length} records (skip=${skip}, limit=${limit})`);
  } catch (err: any) {
    console.warn(`[MP-Batch] Fetch failed (skip=${skip}): ${err?.message?.slice(0, 100)}`);
  }
  return results;
}

/**
 * Fetch MP summary data for a slice of the metallic seed formula list.
 * MP API filter queries (is_metal, material_ids) only return material_id,
 * so we use per-formula fetchSummary which returns full data.
 */
async function fetchSeedFormulaSlice(limit: number, skip: number): Promise<MPSummaryData[]> {
  const slice = METALLIC_SEED_FORMULAS.slice(skip, skip + limit);
  const results: MPSummaryData[] = [];
  for (const formula of slice) {
    try {
      const s = await fetchSummary(formula);
      if (s) results.push(s);
    } catch {}
    await new Promise(r => setTimeout(r, 100)); // gentle rate limiting
  }
  return results;
}

export interface MPGNNSeedRecord {
  formula: string;
  bandGap: number | null;
  formationEnergy: number | null;
  spaceGroup: string | null;
  crystalSystem: string | null;
  isMetallic: boolean;
}

/**
 * Loads MP summary data for GNN pre-seeding at startup.
 * Step 1: reads all cached "summary" rows from the local DB (instant, no API call).
 * Step 2: if the API key is set and the cache has fewer than 400 entries, bulk-fetches
 *         up to 500 metallic materials from the MP API and stores them in the cache.
 * Returns de-duplicated records suitable for passing to addDFTTrainingResult.
 */
export async function fetchGNNSeedData(): Promise<MPGNNSeedRecord[]> {
  const seen = new Map<string, MPGNNSeedRecord>();

  // --- Step 1: read all cached summary rows from DB ---
  try {
    const rows = await db
      .select()
      .from(mpMaterialCache)
      .where(eq(mpMaterialCache.dataType, "summary"))
      .limit(2000);

    for (const row of rows) {
      const d = row.data as MPSummaryData;
      if (!d || typeof d !== "object") continue;
      seen.set(row.formula, {
        formula: row.formula,
        bandGap: typeof d.bandGap === "number" ? d.bandGap : null,
        formationEnergy: typeof d.formationEnergyPerAtom === "number" ? d.formationEnergyPerAtom : null,
        spaceGroup: d.spaceGroup ?? null,
        crystalSystem: d.crystalSystem ?? null,
        isMetallic: d.isMetallic ?? false,
      });
    }
    console.log(`[MP-GNNSeed] Loaded ${seen.size} cached MP summaries from DB`);
  } catch (err: any) {
    console.warn(`[MP-GNNSeed] DB cache load failed: ${err?.message?.slice(0, 100)}`);
  }

  // --- Step 2: fetch summaries for metallic seed formulas not already cached ---
  if (seen.size < 400 && getApiKey()) {
    try {
      const toFetch = METALLIC_SEED_FORMULAS.filter(f => !seen.has(f));
      let fetched = 0;
      for (const formula of toFetch) {
        try {
          const s = await fetchSummary(formula);
          if (s && !seen.has(formula)) {
            seen.set(formula, {
              formula: s.formula,
              bandGap: s.bandGap,
              formationEnergy: s.formationEnergyPerAtom,
              spaceGroup: s.spaceGroup || null,
              crystalSystem: s.crystalSystem || null,
              isMetallic: s.isMetallic,
            });
            fetched++;
          }
        } catch {}
        await new Promise(r => setTimeout(r, 100));
      }
      console.log(`[MP-GNNSeed] Fetched ${fetched} new MP materials via formula lookup (total: ${seen.size})`);
    } catch (err: any) {
      console.warn(`[MP-GNNSeed] Formula fetch failed: ${err?.message?.slice(0, 100)}`);
    }
  }

  return Array.from(seen.values());
}
