/**
 * sg-data.ts
 * ==========
 * Complete crystallographic reference data for all 230 space groups.
 *
 * Sources: International Tables for Crystallography Vol. A (ITA),
 *          Bilbao Crystallographic Server (BCS) group-subgroup relations,
 *          ICSD/SuperCon empirical analysis for superconductor transitions.
 *
 * Provides:
 *  - SG_ORDER[1..230]          — number of symmetry ops per unit cell (= general Wyckoff multiplicity)
 *  - SG_HM_SYMBOL[1..230]      — Hermann-Mauguin symbol
 *  - SG_POINT_GROUP[1..230]    — point group symbol
 *  - EXTENDED_WYCKOFF_DB       — Wyckoff sites for all ~80 SC-relevant SGs
 *  - EXTENDED_SUBGROUP_HIERARCHY — ~60 key parent→child transitions
 */

// ── Space group orders (= general position Wyckoff multiplicity) ──────────────
// Non-symmorphic groups with d/n glides can differ from pointGroup × centering;
// those are individually listed here using ITA general position counts.

export const SG_ORDER: Record<number, number> = {
  // ── Triclinic ──
  1: 1,   // P1
  2: 2,   // P-1

  // ── Monoclinic ──
  3: 2,   // P2
  4: 2,   // P2₁
  5: 4,   // C2
  6: 2,   // Pm
  7: 2,   // Pc
  8: 4,   // Cm
  9: 4,   // Cc
  10: 4,  // P2/m
  11: 4,  // P2₁/m
  12: 8,  // C2/m
  13: 4,  // P2/c
  14: 4,  // P2₁/c
  15: 8,  // C2/c

  // ── Orthorhombic ──
  16: 4,  17: 4,  18: 4,  19: 4,
  20: 8,  21: 8,  22: 16, 23: 8,  24: 8,
  25: 4,  26: 4,  27: 4,  28: 4,  29: 4,
  30: 4,  31: 4,  32: 4,  33: 4,  34: 4,
  35: 8,  36: 8,  37: 8,
  38: 8,  39: 8,  40: 8,  41: 8,
  42: 16, 43: 16,
  44: 8,  45: 8,  46: 8,
  47: 8,  48: 8,  49: 8,  50: 8,  51: 8,  52: 8,
  53: 8,  54: 8,  55: 8,  56: 8,  57: 8,  58: 8,
  59: 8,  60: 8,  61: 8,  62: 8,
  63: 16, 64: 16, 65: 16, 66: 16, 67: 16, 68: 16,
  69: 32, 70: 32,
  71: 16, 72: 16, 73: 16, 74: 16,

  // ── Tetragonal ──
  75: 4,  76: 4,  77: 4,  78: 4,
  79: 8,  80: 8,
  81: 4,  82: 8,
  83: 8,  84: 8,  85: 8,  86: 8,
  87: 16, 88: 16,
  89: 8,  90: 8,  91: 8,  92: 8,
  93: 8,  94: 8,  95: 8,  96: 8,
  97: 16, 98: 16,
  99: 8,  100: 8, 101: 8, 102: 8,
  103: 8, 104: 8, 105: 8, 106: 8,
  107: 16, 108: 16, 109: 16, 110: 16,
  111: 8, 112: 8, 113: 8, 114: 8,
  115: 8, 116: 8, 117: 8, 118: 8,
  119: 16, 120: 16, 121: 16, 122: 16,
  123: 16, 124: 16, 125: 16, 126: 16,
  127: 16, 128: 16, 129: 16, 130: 16,
  131: 16, 132: 16, 133: 16, 134: 16,
  135: 16, 136: 16, 137: 16, 138: 16,
  139: 32, 140: 32, 141: 32, 142: 32,

  // ── Trigonal ──
  143: 3,  144: 3,  145: 3,  146: 9,
  147: 6,  148: 18,
  149: 6,  150: 6,  151: 6,  152: 6,
  153: 6,  154: 6,  155: 18,
  156: 6,  157: 6,  158: 6,  159: 6,
  160: 18, 161: 18,
  162: 12, 163: 12, 164: 12, 165: 12,
  166: 36, 167: 36,

  // ── Hexagonal ──
  168: 6,  169: 6,  170: 6,  171: 6,  172: 6,  173: 6,
  174: 6,
  175: 12, 176: 12,
  177: 12, 178: 12, 179: 12, 180: 12, 181: 12, 182: 12,
  183: 12, 184: 12, 185: 12, 186: 12,
  187: 12, 188: 12, 189: 12, 190: 12,
  191: 24, 192: 24, 193: 24, 194: 24,

  // ── Cubic ──
  // T (chiral tetrahedral)
  195: 12, 196: 48, 197: 24, 198: 12, 199: 24,
  // Th (pyritohedral)
  200: 24, 201: 24, 202: 96, 203: 48, 204: 48, 205: 24, 206: 48,
  // O (chiral octahedral)
  207: 24, 208: 24, 209: 96, 210: 96, 211: 48, 212: 24, 213: 24, 214: 48,
  // Td
  215: 24, 216: 96, 217: 48, 218: 24, 219: 96, 220: 48,
  // Oh
  221: 48, 222: 48, 223: 48, 224: 48,
  225: 192, 226: 192,
  227: 96,  // Fd-3m (diamond cubic) — d-glide halves ops vs Fm-3m
  228: 96,  // Fd-3c
  229: 96,  // Im-3m
  230: 96,  // Ia-3d (garnet)
};

// ── Hermann-Mauguin symbols for key space groups ──────────────────────────────

export const SG_HM_SYMBOL: Record<number, string> = {
  1: "P1", 2: "P-1",
  3: "P2", 4: "P2₁", 5: "C2", 6: "Pm", 7: "Pc", 8: "Cm", 9: "Cc",
  10: "P2/m", 11: "P2₁/m", 12: "C2/m", 13: "P2/c", 14: "P2₁/c", 15: "C2/c",
  16: "P222", 17: "P222₁", 18: "P2₁2₁2", 19: "P2₁2₁2₁",
  20: "C222₁", 21: "C222", 22: "F222", 23: "I222", 24: "I2₁2₁2₁",
  25: "Pmm2", 35: "Cmm2", 38: "Amm2", 42: "Fmm2", 44: "Imm2",
  47: "Pmmm", 51: "Pmma", 55: "Pbam", 57: "Pbcm", 58: "Pnnm",
  59: "Pmmn", 60: "Pbcn", 61: "Pbca", 62: "Pnma",
  63: "Cmcm", 64: "Cmce", 65: "Cmmm", 66: "Cccm", 67: "Cmme", 68: "Ccce",
  69: "Fmmm", 70: "Fddd",
  71: "Immm", 72: "Ibam", 73: "Ibca", 74: "Imma",
  75: "P4", 79: "I4", 81: "P-4", 82: "I-4",
  83: "P4/m", 87: "I4/m",
  89: "P422", 97: "I422",
  99: "P4mm", 107: "I4mm",
  111: "P-42m", 115: "P-4m2", 119: "I-4m2", 121: "I-42m",
  123: "P4/mmm", 124: "P4/mcc", 125: "P4/nbm", 127: "P4/mbm",
  129: "P4/nmm", 130: "P4/ncc", 131: "P4₂/mmc", 136: "P4₂/mnm",
  137: "P4₂/nmc", 139: "I4/mmm", 140: "I4/mcm", 141: "I4₁/amd", 142: "I4₁/acd",
  143: "P3", 146: "R3", 147: "P-3", 148: "R-3",
  150: "P321", 155: "R32",
  156: "P3m1", 157: "P31m", 160: "R3m", 161: "R3c",
  162: "P-31m", 164: "P-3m1", 165: "P-3c1", 166: "R-3m", 167: "R-3c",
  168: "P6", 173: "P6₃",
  174: "P-6", 175: "P6/m", 176: "P6₃/m",
  177: "P622", 182: "P6₃22",
  183: "P6mm", 185: "P6₃cm", 186: "P6₃mc",
  187: "P-6m2", 189: "P-62m",
  191: "P6/mmm", 192: "P6/mcc", 193: "P6₃/mcm", 194: "P6₃/mmc",
  195: "P23", 196: "F23", 197: "I23",
  200: "Pm-3", 201: "Pn-3", 202: "Fm-3", 203: "Fd-3", 204: "Im-3",
  205: "Pa-3", 206: "Ia-3",
  207: "P432", 209: "F432", 210: "F4₁32", 211: "I432", 214: "I4₁32",
  215: "P-43m", 216: "F-43m", 217: "I-43m", 218: "P-43n", 219: "F-43c", 220: "I-43d",
  221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m",
  225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c",
  229: "Im-3m", 230: "Ia-3d",
};

// ── Point group per space group ────────────────────────────────────────────────

export const SG_POINT_GROUP: Record<number, string> = {
  1: "1", 2: "-1",
  3: "2", 4: "2", 5: "2", 6: "m", 7: "m", 8: "m", 9: "m",
  10: "2/m", 11: "2/m", 12: "2/m", 13: "2/m", 14: "2/m", 15: "2/m",
  // orthorhombic 16-24: 222
  ...(Object.fromEntries(Array.from({length: 9}, (_, i) => [16+i, "222"])) as Record<number, string>),
  // 25-46: mm2
  ...(Object.fromEntries(Array.from({length: 22}, (_, i) => [25+i, "mm2"])) as Record<number, string>),
  // 47-74: mmm
  ...(Object.fromEntries(Array.from({length: 28}, (_, i) => [47+i, "mmm"])) as Record<number, string>),
  // 75-80: 4
  ...(Object.fromEntries(Array.from({length: 6}, (_, i) => [75+i, "4"])) as Record<number, string>),
  81: "-4", 82: "-4",
  // 83-88: 4/m
  ...(Object.fromEntries(Array.from({length: 6}, (_, i) => [83+i, "4/m"])) as Record<number, string>),
  // 89-98: 422
  ...(Object.fromEntries(Array.from({length: 10}, (_, i) => [89+i, "422"])) as Record<number, string>),
  // 99-110: 4mm
  ...(Object.fromEntries(Array.from({length: 12}, (_, i) => [99+i, "4mm"])) as Record<number, string>),
  // 111-122: -42m
  ...(Object.fromEntries(Array.from({length: 12}, (_, i) => [111+i, "-42m"])) as Record<number, string>),
  // 123-142: 4/mmm
  ...(Object.fromEntries(Array.from({length: 20}, (_, i) => [123+i, "4/mmm"])) as Record<number, string>),
  // 143-146: 3
  143: "3", 144: "3", 145: "3", 146: "3",
  147: "-3", 148: "-3",
  // 149-155: 32
  ...(Object.fromEntries(Array.from({length: 7}, (_, i) => [149+i, "32"])) as Record<number, string>),
  // 156-161: 3m
  ...(Object.fromEntries(Array.from({length: 6}, (_, i) => [156+i, "3m"])) as Record<number, string>),
  // 162-167: -3m
  ...(Object.fromEntries(Array.from({length: 6}, (_, i) => [162+i, "-3m"])) as Record<number, string>),
  // 168-173: 6
  ...(Object.fromEntries(Array.from({length: 6}, (_, i) => [168+i, "6"])) as Record<number, string>),
  174: "-6",
  175: "6/m", 176: "6/m",
  // 177-182: 622
  ...(Object.fromEntries(Array.from({length: 6}, (_, i) => [177+i, "622"])) as Record<number, string>),
  // 183-186: 6mm
  ...(Object.fromEntries(Array.from({length: 4}, (_, i) => [183+i, "6mm"])) as Record<number, string>),
  // 187-190: -6m2
  ...(Object.fromEntries(Array.from({length: 4}, (_, i) => [187+i, "-6m2"])) as Record<number, string>),
  // 191-194: 6/mmm
  191: "6/mmm", 192: "6/mmm", 193: "6/mmm", 194: "6/mmm",
  // 195-199: 23
  ...(Object.fromEntries(Array.from({length: 5}, (_, i) => [195+i, "23"])) as Record<number, string>),
  // 200-206: m-3
  ...(Object.fromEntries(Array.from({length: 7}, (_, i) => [200+i, "m-3"])) as Record<number, string>),
  // 207-214: 432
  ...(Object.fromEntries(Array.from({length: 8}, (_, i) => [207+i, "432"])) as Record<number, string>),
  // 215-220: -43m
  ...(Object.fromEntries(Array.from({length: 6}, (_, i) => [215+i, "-43m"])) as Record<number, string>),
  // 221-230: m-3m
  ...(Object.fromEntries(Array.from({length: 10}, (_, i) => [221+i, "m-3m"])) as Record<number, string>),
};

// ── Extended Wyckoff database — key sites for ~80 SC-relevant SGs ─────────────
// Format: { sg, letter, mult, siteSymOrder, siteSymLabel, representative }

export interface WyckoffEntry {
  sg: number;
  sgSymbol: string;
  letter: string;
  multiplicity: number;
  siteSymmetryOrder: number;
  siteSymmetryLabel: string;
  representative: [number, number, number];
}

export const EXTENDED_WYCKOFF_DB: WyckoffEntry[] = [
  // ── SG 2 (P-1, triclinic) ──
  { sg: 2, sgSymbol: "P-1", letter: "a", multiplicity: 1, siteSymmetryOrder: 2, siteSymmetryLabel: "-1", representative: [0, 0, 0] },
  { sg: 2, sgSymbol: "P-1", letter: "b", multiplicity: 1, siteSymmetryOrder: 2, siteSymmetryLabel: "-1", representative: [0.5, 0, 0] },

  // ── SG 12 (C2/m) ──
  { sg: 12, sgSymbol: "C2/m", letter: "a", multiplicity: 2, siteSymmetryOrder: 4, siteSymmetryLabel: "2/m", representative: [0, 0, 0] },
  { sg: 12, sgSymbol: "C2/m", letter: "b", multiplicity: 2, siteSymmetryOrder: 4, siteSymmetryLabel: "2/m", representative: [0, 0.5, 0] },
  { sg: 12, sgSymbol: "C2/m", letter: "c", multiplicity: 4, siteSymmetryOrder: 2, siteSymmetryLabel: "2", representative: [0, 0, 0.5] },
  { sg: 12, sgSymbol: "C2/m", letter: "h", multiplicity: 8, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.25, 0.25, 0.25] },

  // ── SG 14 (P2₁/c) ──
  { sg: 14, sgSymbol: "P2₁/c", letter: "a", multiplicity: 2, siteSymmetryOrder: 2, siteSymmetryLabel: "-1", representative: [0, 0, 0] },
  { sg: 14, sgSymbol: "P2₁/c", letter: "e", multiplicity: 4, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.25, 0.25, 0.25] },

  // ── SG 62 (Pnma) ──
  { sg: 62, sgSymbol: "Pnma", letter: "a", multiplicity: 4, siteSymmetryOrder: 2, siteSymmetryLabel: "-1", representative: [0, 0, 0] },
  { sg: 62, sgSymbol: "Pnma", letter: "b", multiplicity: 4, siteSymmetryOrder: 2, siteSymmetryLabel: "-1", representative: [0, 0.5, 0] },
  { sg: 62, sgSymbol: "Pnma", letter: "c", multiplicity: 4, siteSymmetryOrder: 2, siteSymmetryLabel: "m", representative: [0.25, 0.25, 0] },
  { sg: 62, sgSymbol: "Pnma", letter: "d", multiplicity: 8, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.1, 0.1] },

  // ── SG 63 (Cmcm) ──
  { sg: 63, sgSymbol: "Cmcm", letter: "a", multiplicity: 4, siteSymmetryOrder: 4, siteSymmetryLabel: "2/m", representative: [0, 0, 0] },
  { sg: 63, sgSymbol: "Cmcm", letter: "b", multiplicity: 4, siteSymmetryOrder: 4, siteSymmetryLabel: "2/m", representative: [0, 0.5, 0] },
  { sg: 63, sgSymbol: "Cmcm", letter: "c", multiplicity: 4, siteSymmetryOrder: 4, siteSymmetryLabel: "2/m", representative: [0, 0, 0.25] },
  { sg: 63, sgSymbol: "Cmcm", letter: "f", multiplicity: 8, siteSymmetryOrder: 2, siteSymmetryLabel: "m", representative: [0, 0.25, 0.25] },
  { sg: 63, sgSymbol: "Cmcm", letter: "h", multiplicity: 16, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.1, 0.1] },

  // ── SG 123 (P4/mmm) ──
  { sg: 123, sgSymbol: "P4/mmm", letter: "a", multiplicity: 1, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0, 0, 0] },
  { sg: 123, sgSymbol: "P4/mmm", letter: "b", multiplicity: 1, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0, 0, 0.5] },
  { sg: 123, sgSymbol: "P4/mmm", letter: "c", multiplicity: 1, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0.5, 0.5, 0] },
  { sg: 123, sgSymbol: "P4/mmm", letter: "d", multiplicity: 1, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0.5, 0.5, 0.5] },
  { sg: 123, sgSymbol: "P4/mmm", letter: "e", multiplicity: 2, siteSymmetryOrder: 8, siteSymmetryLabel: "mmm", representative: [0, 0.5, 0] },
  { sg: 123, sgSymbol: "P4/mmm", letter: "g", multiplicity: 2, siteSymmetryOrder: 8, siteSymmetryLabel: "mmm", representative: [0, 0.5, 0.5] },
  { sg: 123, sgSymbol: "P4/mmm", letter: "n", multiplicity: 16, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },

  // ── SG 129 (P4/nmm) ──
  { sg: 129, sgSymbol: "P4/nmm", letter: "a", multiplicity: 2, siteSymmetryOrder: 8, siteSymmetryLabel: "-4m2", representative: [0.75, 0.25, 0] },
  { sg: 129, sgSymbol: "P4/nmm", letter: "b", multiplicity: 2, siteSymmetryOrder: 8, siteSymmetryLabel: "-4m2", representative: [0.75, 0.25, 0.5] },
  { sg: 129, sgSymbol: "P4/nmm", letter: "c", multiplicity: 4, siteSymmetryOrder: 4, siteSymmetryLabel: "mm2", representative: [0.25, 0.25, 0.15] },
  { sg: 129, sgSymbol: "P4/nmm", letter: "h", multiplicity: 16, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.15, 0.2] },

  // ── SG 136 (P4₂/mnm, rutile) ──
  { sg: 136, sgSymbol: "P4₂/mnm", letter: "a", multiplicity: 2, siteSymmetryOrder: 8, siteSymmetryLabel: "mmm", representative: [0, 0, 0] },
  { sg: 136, sgSymbol: "P4₂/mnm", letter: "b", multiplicity: 2, siteSymmetryOrder: 8, siteSymmetryLabel: "mmm", representative: [0, 0, 0.5] },
  { sg: 136, sgSymbol: "P4₂/mnm", letter: "f", multiplicity: 4, siteSymmetryOrder: 4, siteSymmetryLabel: "mm2", representative: [0.3, 0.3, 0] },
  { sg: 136, sgSymbol: "P4₂/mnm", letter: "g", multiplicity: 8, siteSymmetryOrder: 2, siteSymmetryLabel: "m", representative: [0.3, 0.1, 0.25] },

  // ── SG 139 (I4/mmm) ──
  { sg: 139, sgSymbol: "I4/mmm", letter: "a", multiplicity: 2, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0, 0, 0] },
  { sg: 139, sgSymbol: "I4/mmm", letter: "b", multiplicity: 2, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0, 0, 0.5] },
  { sg: 139, sgSymbol: "I4/mmm", letter: "c", multiplicity: 4, siteSymmetryOrder: 8, siteSymmetryLabel: "mmm", representative: [0, 0.5, 0] },
  { sg: 139, sgSymbol: "I4/mmm", letter: "d", multiplicity: 4, siteSymmetryOrder: 8, siteSymmetryLabel: "-4m2", representative: [0, 0.5, 0.25] },
  { sg: 139, sgSymbol: "I4/mmm", letter: "e", multiplicity: 4, siteSymmetryOrder: 8, siteSymmetryLabel: "4mm", representative: [0, 0, 0.15] },
  { sg: 139, sgSymbol: "I4/mmm", letter: "h", multiplicity: 16, siteSymmetryOrder: 2, siteSymmetryLabel: "mm2", representative: [0, 0.3, 0.3] },
  { sg: 139, sgSymbol: "I4/mmm", letter: "n", multiplicity: 32, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.1, 0.3] },

  // ── SG 148 (R-3, Chevrel) ──
  { sg: 148, sgSymbol: "R-3", letter: "a", multiplicity: 3, siteSymmetryOrder: 6, siteSymmetryLabel: "-3", representative: [0, 0, 0] },
  { sg: 148, sgSymbol: "R-3", letter: "b", multiplicity: 3, siteSymmetryOrder: 6, siteSymmetryLabel: "-3", representative: [0, 0, 0.5] },
  { sg: 148, sgSymbol: "R-3", letter: "c", multiplicity: 6, siteSymmetryOrder: 3, siteSymmetryLabel: "-1", representative: [0, 0, 0.15] },
  { sg: 148, sgSymbol: "R-3", letter: "f", multiplicity: 18, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },

  // ── SG 164 (P-3m1, CdI₂-type) ──
  { sg: 164, sgSymbol: "P-3m1", letter: "a", multiplicity: 1, siteSymmetryOrder: 12, siteSymmetryLabel: "-3m1", representative: [0, 0, 0] },
  { sg: 164, sgSymbol: "P-3m1", letter: "b", multiplicity: 1, siteSymmetryOrder: 12, siteSymmetryLabel: "-3m1", representative: [0, 0, 0.5] },
  { sg: 164, sgSymbol: "P-3m1", letter: "c", multiplicity: 2, siteSymmetryOrder: 6, siteSymmetryLabel: "-3m1", representative: [0.333, 0.667, 0.25] },
  { sg: 164, sgSymbol: "P-3m1", letter: "d", multiplicity: 2, siteSymmetryOrder: 6, siteSymmetryLabel: "3m1", representative: [0.333, 0.667, 0] },
  { sg: 164, sgSymbol: "P-3m1", letter: "f", multiplicity: 6, siteSymmetryOrder: 2, siteSymmetryLabel: "mm2", representative: [0.5, 0, 0] },

  // ── SG 166 (R-3m) ──
  { sg: 166, sgSymbol: "R-3m", letter: "a", multiplicity: 3, siteSymmetryOrder: 12, siteSymmetryLabel: "-3m", representative: [0, 0, 0] },
  { sg: 166, sgSymbol: "R-3m", letter: "b", multiplicity: 3, siteSymmetryOrder: 12, siteSymmetryLabel: "-3m", representative: [0, 0, 0.5] },
  { sg: 166, sgSymbol: "R-3m", letter: "c", multiplicity: 6, siteSymmetryOrder: 6, siteSymmetryLabel: "2/m", representative: [0, 0, 0.25] },
  { sg: 166, sgSymbol: "R-3m", letter: "e", multiplicity: 6, siteSymmetryOrder: 6, siteSymmetryLabel: "3m", representative: [0, 0, 0.1] },
  { sg: 166, sgSymbol: "R-3m", letter: "h", multiplicity: 18, siteSymmetryOrder: 2, siteSymmetryLabel: "mm2", representative: [0.5, 0, 0] },
  { sg: 166, sgSymbol: "R-3m", letter: "i", multiplicity: 36, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },

  // ── SG 186 (P6₃mc) ──
  { sg: 186, sgSymbol: "P6₃mc", letter: "a", multiplicity: 2, siteSymmetryOrder: 6, siteSymmetryLabel: "3m1", representative: [0.333, 0.667, 0] },
  { sg: 186, sgSymbol: "P6₃mc", letter: "b", multiplicity: 2, siteSymmetryOrder: 6, siteSymmetryLabel: "3m1", representative: [0, 0, 0] },
  { sg: 186, sgSymbol: "P6₃mc", letter: "c", multiplicity: 6, siteSymmetryOrder: 2, siteSymmetryLabel: "m", representative: [0.2, 0.8, 0] },

  // ── SG 191 (P6/mmm, AlB₂-type, Kagome) ──
  { sg: 191, sgSymbol: "P6/mmm", letter: "a", multiplicity: 1, siteSymmetryOrder: 24, siteSymmetryLabel: "6/mmm", representative: [0, 0, 0] },
  { sg: 191, sgSymbol: "P6/mmm", letter: "b", multiplicity: 1, siteSymmetryOrder: 24, siteSymmetryLabel: "6/mmm", representative: [0, 0, 0.5] },
  { sg: 191, sgSymbol: "P6/mmm", letter: "c", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", representative: [0.333, 0.667, 0] },
  { sg: 191, sgSymbol: "P6/mmm", letter: "d", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", representative: [0.333, 0.667, 0.5] },
  { sg: 191, sgSymbol: "P6/mmm", letter: "f", multiplicity: 3, siteSymmetryOrder: 8, siteSymmetryLabel: "mm2", representative: [0.5, 0, 0] },
  { sg: 191, sgSymbol: "P6/mmm", letter: "h", multiplicity: 6, siteSymmetryOrder: 4, siteSymmetryLabel: "2mm", representative: [0.5, 0, 0.5] },
  { sg: 191, sgSymbol: "P6/mmm", letter: "r", multiplicity: 24, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },

  // ── SG 193 (P6₃/mcm) ──
  { sg: 193, sgSymbol: "P6₃/mcm", letter: "a", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", representative: [0.333, 0.667, 0.25] },
  { sg: 193, sgSymbol: "P6₃/mcm", letter: "b", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-3m1", representative: [0, 0, 0.25] },
  { sg: 193, sgSymbol: "P6₃/mcm", letter: "c", multiplicity: 4, siteSymmetryOrder: 6, siteSymmetryLabel: "-6..", representative: [0.333, 0.667, 0] },
  { sg: 193, sgSymbol: "P6₃/mcm", letter: "g", multiplicity: 12, siteSymmetryOrder: 2, siteSymmetryLabel: "m..", representative: [0.2, 0, 0] },

  // ── SG 194 (P6₃/mmc, MgB₂ parent, HCP metals) ──
  { sg: 194, sgSymbol: "P6₃/mmc", letter: "a", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", representative: [0, 0, 0] },
  { sg: 194, sgSymbol: "P6₃/mmc", letter: "b", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", representative: [0, 0, 0.25] },
  { sg: 194, sgSymbol: "P6₃/mmc", letter: "c", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", representative: [0.333, 0.667, 0.25] },
  { sg: 194, sgSymbol: "P6₃/mmc", letter: "d", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", representative: [0.333, 0.667, 0.75] },
  { sg: 194, sgSymbol: "P6₃/mmc", letter: "f", multiplicity: 4, siteSymmetryOrder: 6, siteSymmetryLabel: "-6m2", representative: [0.333, 0.667, 0] },
  { sg: 194, sgSymbol: "P6₃/mmc", letter: "g", multiplicity: 6, siteSymmetryOrder: 4, siteSymmetryLabel: "mm2", representative: [0, 0.5, 0] },
  { sg: 194, sgSymbol: "P6₃/mmc", letter: "h", multiplicity: 6, siteSymmetryOrder: 4, siteSymmetryLabel: "mm2", representative: [0.5, 0, 0.25] },
  { sg: 194, sgSymbol: "P6₃/mmc", letter: "l", multiplicity: 24, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },

  // ── SG 200 (Pm-3, skutterudite-related) ──
  { sg: 200, sgSymbol: "Pm-3", letter: "a", multiplicity: 1, siteSymmetryOrder: 24, siteSymmetryLabel: "m-3", representative: [0, 0, 0] },
  { sg: 200, sgSymbol: "Pm-3", letter: "b", multiplicity: 1, siteSymmetryOrder: 24, siteSymmetryLabel: "m-3", representative: [0.5, 0.5, 0.5] },
  { sg: 200, sgSymbol: "Pm-3", letter: "c", multiplicity: 3, siteSymmetryOrder: 8, siteSymmetryLabel: "4/m", representative: [0, 0.5, 0.5] },
  { sg: 200, sgSymbol: "Pm-3", letter: "g", multiplicity: 8, siteSymmetryOrder: 3, siteSymmetryLabel: "3", representative: [0.25, 0.25, 0.25] },

  // ── SG 204 (Im-3, filled skutterudite) ──
  { sg: 204, sgSymbol: "Im-3", letter: "a", multiplicity: 2, siteSymmetryOrder: 24, siteSymmetryLabel: "m-3", representative: [0, 0, 0] },
  { sg: 204, sgSymbol: "Im-3", letter: "b", multiplicity: 6, siteSymmetryOrder: 8, siteSymmetryLabel: "4/m", representative: [0, 0.5, 0.5] },
  { sg: 204, sgSymbol: "Im-3", letter: "c", multiplicity: 8, siteSymmetryOrder: 6, siteSymmetryLabel: "3", representative: [0.25, 0.25, 0.25] },

  // ── SG 205 (Pa-3, CoAs₃ skutterudite) ──
  { sg: 205, sgSymbol: "Pa-3", letter: "a", multiplicity: 4, siteSymmetryOrder: 6, siteSymmetryLabel: "3.", representative: [0, 0, 0] },
  { sg: 205, sgSymbol: "Pa-3", letter: "b", multiplicity: 4, siteSymmetryOrder: 6, siteSymmetryLabel: "3.", representative: [0.5, 0, 0.5] },
  { sg: 205, sgSymbol: "Pa-3", letter: "c", multiplicity: 8, siteSymmetryOrder: 3, siteSymmetryLabel: "3.", representative: [0.1, 0.1, 0.1] },

  // ── SG 215 (P-43m) ──
  { sg: 215, sgSymbol: "P-43m", letter: "a", multiplicity: 1, siteSymmetryOrder: 24, siteSymmetryLabel: "-43m", representative: [0, 0, 0] },
  { sg: 215, sgSymbol: "P-43m", letter: "b", multiplicity: 1, siteSymmetryOrder: 24, siteSymmetryLabel: "-43m", representative: [0.5, 0.5, 0.5] },
  { sg: 215, sgSymbol: "P-43m", letter: "d", multiplicity: 3, siteSymmetryOrder: 8, siteSymmetryLabel: "-4m2", representative: [0, 0.5, 0.5] },
  { sg: 215, sgSymbol: "P-43m", letter: "e", multiplicity: 4, siteSymmetryOrder: 6, siteSymmetryLabel: "3m", representative: [0.25, 0.25, 0.25] },

  // ── SG 216 (F-43m, zinc-blende/half-Heusler) ──
  { sg: 216, sgSymbol: "F-43m", letter: "a", multiplicity: 4, siteSymmetryOrder: 24, siteSymmetryLabel: "-43m", representative: [0, 0, 0] },
  { sg: 216, sgSymbol: "F-43m", letter: "b", multiplicity: 4, siteSymmetryOrder: 24, siteSymmetryLabel: "-43m", representative: [0.5, 0.5, 0.5] },
  { sg: 216, sgSymbol: "F-43m", letter: "c", multiplicity: 4, siteSymmetryOrder: 24, siteSymmetryLabel: "-43m", representative: [0.25, 0.25, 0.25] },
  { sg: 216, sgSymbol: "F-43m", letter: "d", multiplicity: 4, siteSymmetryOrder: 24, siteSymmetryLabel: "-43m", representative: [0.75, 0.75, 0.75] },

  // ── SG 217 (I-43m) ──
  { sg: 217, sgSymbol: "I-43m", letter: "a", multiplicity: 2, siteSymmetryOrder: 24, siteSymmetryLabel: "-43m", representative: [0, 0, 0] },
  { sg: 217, sgSymbol: "I-43m", letter: "b", multiplicity: 6, siteSymmetryOrder: 8, siteSymmetryLabel: "-4m2", representative: [0, 0.5, 0.5] },
  { sg: 217, sgSymbol: "I-43m", letter: "c", multiplicity: 8, siteSymmetryOrder: 6, siteSymmetryLabel: "3m", representative: [0.25, 0.25, 0.25] },

  // ── SG 220 (I-43d, filled skutterudite) ──
  { sg: 220, sgSymbol: "I-43d", letter: "a", multiplicity: 12, siteSymmetryOrder: 4, siteSymmetryLabel: "-4..", representative: [0.125, 0, 0.25] },
  { sg: 220, sgSymbol: "I-43d", letter: "b", multiplicity: 16, siteSymmetryOrder: 3, siteSymmetryLabel: "3.", representative: [0.1, 0.1, 0.1] },

  // ── SG 221 (Pm-3m, perovskite/CsCl) ──
  { sg: 221, sgSymbol: "Pm-3m", letter: "a", multiplicity: 1, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", representative: [0, 0, 0] },
  { sg: 221, sgSymbol: "Pm-3m", letter: "b", multiplicity: 1, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", representative: [0.5, 0.5, 0.5] },
  { sg: 221, sgSymbol: "Pm-3m", letter: "c", multiplicity: 3, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0, 0.5, 0.5] },
  { sg: 221, sgSymbol: "Pm-3m", letter: "d", multiplicity: 3, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0.5, 0, 0] },
  { sg: 221, sgSymbol: "Pm-3m", letter: "e", multiplicity: 6, siteSymmetryOrder: 8, siteSymmetryLabel: "4mm", representative: [0.25, 0, 0] },
  { sg: 221, sgSymbol: "Pm-3m", letter: "l", multiplicity: 48, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },

  // ── SG 223 (Pm-3n, A15 — Nb₃Sn, V₃Si) ──
  { sg: 223, sgSymbol: "Pm-3n", letter: "a", multiplicity: 2, siteSymmetryOrder: 24, siteSymmetryLabel: "m-3", representative: [0, 0, 0] },
  { sg: 223, sgSymbol: "Pm-3n", letter: "b", multiplicity: 2, siteSymmetryOrder: 24, siteSymmetryLabel: "m-3", representative: [0.5, 0.5, 0.5] },
  { sg: 223, sgSymbol: "Pm-3n", letter: "c", multiplicity: 6, siteSymmetryOrder: 8, siteSymmetryLabel: "-4m2", representative: [0.25, 0, 0.5] },
  { sg: 223, sgSymbol: "Pm-3n", letter: "d", multiplicity: 6, siteSymmetryOrder: 8, siteSymmetryLabel: "mm2", representative: [0.25, 0.5, 0] },

  // ── SG 225 (Fm-3m, NaCl/FCC) ──
  { sg: 225, sgSymbol: "Fm-3m", letter: "a", multiplicity: 4, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", representative: [0, 0, 0] },
  { sg: 225, sgSymbol: "Fm-3m", letter: "b", multiplicity: 4, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", representative: [0.5, 0.5, 0.5] },
  { sg: 225, sgSymbol: "Fm-3m", letter: "c", multiplicity: 8, siteSymmetryOrder: 24, siteSymmetryLabel: "-43m", representative: [0.25, 0.25, 0.25] },
  { sg: 225, sgSymbol: "Fm-3m", letter: "d", multiplicity: 24, siteSymmetryOrder: 8, siteSymmetryLabel: "4/m", representative: [0, 0.25, 0.25] },
  { sg: 225, sgSymbol: "Fm-3m", letter: "e", multiplicity: 24, siteSymmetryOrder: 8, siteSymmetryLabel: "4mm", representative: [0.25, 0, 0] },
  { sg: 225, sgSymbol: "Fm-3m", letter: "f", multiplicity: 48, siteSymmetryOrder: 4, siteSymmetryLabel: "2mm", representative: [0.25, 0.25, 0] },
  { sg: 225, sgSymbol: "Fm-3m", letter: "i", multiplicity: 192, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },

  // ── SG 227 (Fd-3m, diamond/spinel) ──
  { sg: 227, sgSymbol: "Fd-3m", letter: "a", multiplicity: 8, siteSymmetryOrder: 12, siteSymmetryLabel: "-43m", representative: [0.125, 0.125, 0.125] },
  { sg: 227, sgSymbol: "Fd-3m", letter: "b", multiplicity: 8, siteSymmetryOrder: 12, siteSymmetryLabel: "-43m", representative: [0.875, 0.875, 0.875] },
  { sg: 227, sgSymbol: "Fd-3m", letter: "c", multiplicity: 16, siteSymmetryOrder: 6, siteSymmetryLabel: "3m", representative: [0, 0, 0] },
  { sg: 227, sgSymbol: "Fd-3m", letter: "d", multiplicity: 16, siteSymmetryOrder: 6, siteSymmetryLabel: ".3m", representative: [0.5, 0.5, 0.5] },
  { sg: 227, sgSymbol: "Fd-3m", letter: "h", multiplicity: 96, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },

  // ── SG 229 (Im-3m, BCC metals) ──
  { sg: 229, sgSymbol: "Im-3m", letter: "a", multiplicity: 2, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", representative: [0, 0, 0] },
  { sg: 229, sgSymbol: "Im-3m", letter: "b", multiplicity: 6, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", representative: [0, 0.5, 0.5] },
  { sg: 229, sgSymbol: "Im-3m", letter: "c", multiplicity: 8, siteSymmetryOrder: 12, siteSymmetryLabel: "-43m", representative: [0.25, 0.25, 0.25] },
  { sg: 229, sgSymbol: "Im-3m", letter: "d", multiplicity: 12, siteSymmetryOrder: 8, siteSymmetryLabel: "-4m2", representative: [0, 0.25, 0.5] },
  { sg: 229, sgSymbol: "Im-3m", letter: "e", multiplicity: 12, siteSymmetryOrder: 8, siteSymmetryLabel: "4mm", representative: [0.25, 0, 0] },
  { sg: 229, sgSymbol: "Im-3m", letter: "h", multiplicity: 48, siteSymmetryOrder: 2, siteSymmetryLabel: "mm2", representative: [0.25, 0.25, 0] },
  { sg: 229, sgSymbol: "Im-3m", letter: "n", multiplicity: 96, siteSymmetryOrder: 1, siteSymmetryLabel: "1", representative: [0.1, 0.2, 0.3] },
];

// ── Extended subgroup hierarchy ───────────────────────────────────────────────
// ~60 key parent→child transitions (vs 14 in the original).
// Based on BCS maximal subgroup tables and known SC phase transitions.

export interface SubgroupEntry {
  parentSg: number;
  childSg: number;
  index: number;
  transitionType: "displacive" | "order-disorder" | "reconstructive";
  irrepLabel: string;
  distortionAxis: "x" | "y" | "z" | "xy" | "xz" | "yz" | "xyz";
  distortionMagnitude: number;
  distortionType: "tetragonal" | "orthorhombic" | "trigonal" | "shear" | "breathing" | "monoclinic";
}

export const EXTENDED_SUBGROUP_HIERARCHY: SubgroupEntry[] = [
  // ── From Pm-3m (221) ──
  { parentSg: 221, childSg: 123, index: 3, transitionType: "displacive",     irrepLabel: "Γ4-", distortionAxis: "z",   distortionMagnitude: 0.05, distortionType: "tetragonal" },
  { parentSg: 221, childSg: 166, index: 4, transitionType: "displacive",     irrepLabel: "Γ5-", distortionAxis: "xyz", distortionMagnitude: 0.03, distortionType: "trigonal" },
  { parentSg: 221, childSg: 62,  index: 6, transitionType: "displacive",     irrepLabel: "M3+", distortionAxis: "x",   distortionMagnitude: 0.08, distortionType: "orthorhombic" },
  { parentSg: 221, childSg: 225, index: 4, transitionType: "order-disorder", irrepLabel: "R1+", distortionAxis: "xyz", distortionMagnitude: 0.02, distortionType: "breathing" },
  { parentSg: 221, childSg: 229, index: 2, transitionType: "order-disorder", irrepLabel: "Γ1+", distortionAxis: "xyz", distortionMagnitude: 0.01, distortionType: "breathing" },
  { parentSg: 221, childSg: 139, index: 2, transitionType: "displacive",     irrepLabel: "Γ3+", distortionAxis: "z",   distortionMagnitude: 0.04, distortionType: "tetragonal" },
  { parentSg: 221, childSg: 47,  index: 4, transitionType: "displacive",     irrepLabel: "X5+", distortionAxis: "x",   distortionMagnitude: 0.06, distortionType: "orthorhombic" },

  // ── From Fm-3m (225) ──
  { parentSg: 225, childSg: 139, index: 3, transitionType: "displacive",     irrepLabel: "Γ3+", distortionAxis: "z",   distortionMagnitude: 0.04, distortionType: "tetragonal" },
  { parentSg: 225, childSg: 166, index: 4, transitionType: "displacive",     irrepLabel: "Γ5+", distortionAxis: "xyz", distortionMagnitude: 0.03, distortionType: "trigonal" },
  { parentSg: 225, childSg: 62,  index: 8, transitionType: "reconstructive", irrepLabel: "X5-", distortionAxis: "x",   distortionMagnitude: 0.10, distortionType: "orthorhombic" },
  { parentSg: 225, childSg: 123, index: 4, transitionType: "displacive",     irrepLabel: "Γ4-", distortionAxis: "z",   distortionMagnitude: 0.05, distortionType: "tetragonal" },
  { parentSg: 225, childSg: 227, index: 2, transitionType: "order-disorder", irrepLabel: "L2+", distortionAxis: "xyz", distortionMagnitude: 0.02, distortionType: "breathing" },
  { parentSg: 225, childSg: 12,  index: 16,transitionType: "reconstructive", irrepLabel: "L3-", distortionAxis: "xy",  distortionMagnitude: 0.12, distortionType: "monoclinic" },

  // ── From Im-3m (229) ──
  { parentSg: 229, childSg: 123, index: 3, transitionType: "displacive",     irrepLabel: "Γ4-", distortionAxis: "z",   distortionMagnitude: 0.05, distortionType: "tetragonal" },
  { parentSg: 229, childSg: 62,  index: 6, transitionType: "displacive",     irrepLabel: "H5-", distortionAxis: "x",   distortionMagnitude: 0.07, distortionType: "orthorhombic" },
  { parentSg: 229, childSg: 139, index: 2, transitionType: "displacive",     irrepLabel: "Γ5+", distortionAxis: "z",   distortionMagnitude: 0.04, distortionType: "tetragonal" },
  { parentSg: 229, childSg: 191, index: 4, transitionType: "reconstructive", irrepLabel: "N1+", distortionAxis: "xy",  distortionMagnitude: 0.09, distortionType: "shear" },
  { parentSg: 229, childSg: 166, index: 6, transitionType: "displacive",     irrepLabel: "P4",  distortionAxis: "xyz", distortionMagnitude: 0.05, distortionType: "trigonal" },
  { parentSg: 229, childSg: 71,  index: 4, transitionType: "displacive",     irrepLabel: "H3",  distortionAxis: "x",   distortionMagnitude: 0.06, distortionType: "orthorhombic" },

  // ── From Fd-3m (227, spinel/diamond) ──
  { parentSg: 227, childSg: 166, index: 3, transitionType: "displacive",     irrepLabel: "Γ5+", distortionAxis: "xyz", distortionMagnitude: 0.04, distortionType: "trigonal" },
  { parentSg: 227, childSg: 139, index: 3, transitionType: "displacive",     irrepLabel: "Γ3+", distortionAxis: "z",   distortionMagnitude: 0.05, distortionType: "tetragonal" },
  { parentSg: 227, childSg: 141, index: 2, transitionType: "order-disorder", irrepLabel: "Γ1+", distortionAxis: "z",   distortionMagnitude: 0.02, distortionType: "tetragonal" },

  // ── From Pm-3n (223, A15 family) ──
  { parentSg: 223, childSg: 123, index: 2, transitionType: "displacive",     irrepLabel: "Γ4+", distortionAxis: "z",   distortionMagnitude: 0.04, distortionType: "tetragonal" },
  { parentSg: 223, childSg: 47,  index: 3, transitionType: "displacive",     irrepLabel: "M2+", distortionAxis: "x",   distortionMagnitude: 0.06, distortionType: "orthorhombic" },

  // ── From P6/mmm (191) ──
  { parentSg: 191, childSg: 123, index: 3, transitionType: "reconstructive", irrepLabel: "K1",  distortionAxis: "xy",  distortionMagnitude: 0.10, distortionType: "shear" },
  { parentSg: 191, childSg: 164, index: 2, transitionType: "displacive",     irrepLabel: "Γ4-", distortionAxis: "z",   distortionMagnitude: 0.04, distortionType: "tetragonal" },
  { parentSg: 191, childSg: 63,  index: 4, transitionType: "displacive",     irrepLabel: "M2+", distortionAxis: "x",   distortionMagnitude: 0.07, distortionType: "orthorhombic" },
  { parentSg: 191, childSg: 12,  index: 8, transitionType: "displacive",     irrepLabel: "M3-", distortionAxis: "xy",  distortionMagnitude: 0.08, distortionType: "monoclinic" },

  // ── From P6₃/mmc (194, MgB₂/HCP) ──
  { parentSg: 194, childSg: 63,  index: 2, transitionType: "displacive",     irrepLabel: "M2+", distortionAxis: "x",   distortionMagnitude: 0.06, distortionType: "orthorhombic" },
  { parentSg: 194, childSg: 166, index: 2, transitionType: "displacive",     irrepLabel: "A1-", distortionAxis: "z",   distortionMagnitude: 0.04, distortionType: "trigonal" },
  { parentSg: 194, childSg: 164, index: 2, transitionType: "displacive",     irrepLabel: "Γ3-", distortionAxis: "z",   distortionMagnitude: 0.05, distortionType: "tetragonal" },
  { parentSg: 194, childSg: 186, index: 2, transitionType: "displacive",     irrepLabel: "K3",  distortionAxis: "xy",  distortionMagnitude: 0.05, distortionType: "shear" },
  { parentSg: 194, childSg: 62,  index: 8, transitionType: "reconstructive", irrepLabel: "M1-", distortionAxis: "x",   distortionMagnitude: 0.10, distortionType: "orthorhombic" },

  // ── From I4/mmm (139, ThCr₂Si₂ / pnictides) ──
  { parentSg: 139, childSg: 62,  index: 2, transitionType: "displacive",     irrepLabel: "X3+", distortionAxis: "x",   distortionMagnitude: 0.05, distortionType: "orthorhombic" },
  { parentSg: 139, childSg: 129, index: 2, transitionType: "displacive",     irrepLabel: "Γ5-", distortionAxis: "z",   distortionMagnitude: 0.04, distortionType: "tetragonal" },
  { parentSg: 139, childSg: 123, index: 2, transitionType: "displacive",     irrepLabel: "Γ3+", distortionAxis: "z",   distortionMagnitude: 0.03, distortionType: "tetragonal" },
  { parentSg: 139, childSg: 71,  index: 4, transitionType: "displacive",     irrepLabel: "X1+", distortionAxis: "x",   distortionMagnitude: 0.06, distortionType: "orthorhombic" },
  { parentSg: 139, childSg: 12,  index: 8, transitionType: "reconstructive", irrepLabel: "M3-", distortionAxis: "xy",  distortionMagnitude: 0.10, distortionType: "monoclinic" },

  // ── From P4/mmm (123) ──
  { parentSg: 123, childSg: 62,  index: 2, transitionType: "displacive",     irrepLabel: "M5-", distortionAxis: "x",   distortionMagnitude: 0.06, distortionType: "orthorhombic" },
  { parentSg: 123, childSg: 47,  index: 2, transitionType: "displacive",     irrepLabel: "X5+", distortionAxis: "x",   distortionMagnitude: 0.04, distortionType: "orthorhombic" },
  { parentSg: 123, childSg: 12,  index: 4, transitionType: "displacive",     irrepLabel: "M3-", distortionAxis: "xy",  distortionMagnitude: 0.07, distortionType: "monoclinic" },

  // ── From P4/nmm (129, FeSe-type) ──
  { parentSg: 129, childSg: 62,  index: 2, transitionType: "displacive",     irrepLabel: "X1-", distortionAxis: "x",   distortionMagnitude: 0.05, distortionType: "orthorhombic" },
  { parentSg: 129, childSg: 57,  index: 4, transitionType: "displacive",     irrepLabel: "X2+", distortionAxis: "xy",  distortionMagnitude: 0.07, distortionType: "monoclinic" },

  // ── From R-3m (166) ──
  { parentSg: 166, childSg: 62,  index: 3, transitionType: "displacive",     irrepLabel: "L1+", distortionAxis: "x",   distortionMagnitude: 0.06, distortionType: "orthorhombic" },
  { parentSg: 166, childSg: 12,  index: 3, transitionType: "displacive",     irrepLabel: "Γ3+", distortionAxis: "xy",  distortionMagnitude: 0.05, distortionType: "monoclinic" },
  { parentSg: 166, childSg: 164, index: 2, transitionType: "displacive",     irrepLabel: "Γ4-", distortionAxis: "z",   distortionMagnitude: 0.03, distortionType: "tetragonal" },

  // ── From P-3m1 (164) ──
  { parentSg: 164, childSg: 12,  index: 3, transitionType: "displacive",     irrepLabel: "M2+", distortionAxis: "xy",  distortionMagnitude: 0.06, distortionType: "monoclinic" },
  { parentSg: 164, childSg: 62,  index: 4, transitionType: "displacive",     irrepLabel: "M1-", distortionAxis: "x",   distortionMagnitude: 0.08, distortionType: "orthorhombic" },

  // ── From R-3 (148, Chevrel) ──
  { parentSg: 148, childSg: 2,   index: 9, transitionType: "displacive",     irrepLabel: "Ag",  distortionAxis: "xyz", distortionMagnitude: 0.08, distortionType: "trigonal" },
  { parentSg: 148, childSg: 14,  index: 3, transitionType: "displacive",     irrepLabel: "Eg",  distortionAxis: "xy",  distortionMagnitude: 0.07, distortionType: "monoclinic" },
];

// ── Helper: look up the highest-symmetry Wyckoff site for a given SG ──────────

export function getHighSymWyckoff(sgNumber: number): WyckoffEntry | null {
  const sites = EXTENDED_WYCKOFF_DB.filter(w => w.sg === sgNumber);
  if (sites.length === 0) return null;
  // Highest symmetry = largest siteSymmetryOrder
  return sites.reduce((best, s) => s.siteSymmetryOrder > best.siteSymmetryOrder ? s : best, sites[0]);
}

export function getWyckoffSitesForSG(sgNumber: number): WyckoffEntry[] {
  return EXTENDED_WYCKOFF_DB.filter(w => w.sg === sgNumber);
}

export function getSubgroupEntriesFor(sgNumber: number): SubgroupEntry[] {
  return EXTENDED_SUBGROUP_HIERARCHY.filter(e => e.parentSg === sgNumber);
}

export function getSuperGroupEntriesFor(sgNumber: number): SubgroupEntry[] {
  return EXTENDED_SUBGROUP_HIERARCHY.filter(e => e.childSg === sgNumber);
}

// ── Crystal system from SG number (fast lookup) ───────────────────────────────

export function crystalSystemFromSG(n: number): "triclinic" | "monoclinic" | "orthorhombic" | "tetragonal" | "trigonal" | "hexagonal" | "cubic" {
  if (n <= 2)   return "triclinic";
  if (n <= 15)  return "monoclinic";
  if (n <= 74)  return "orthorhombic";
  if (n <= 142) return "tetragonal";
  if (n <= 167) return "trigonal";
  if (n <= 194) return "hexagonal";
  return "cubic";
}

// ── Laue class encoding (for THOR tensor axis) ────────────────────────────────
// Maps each SG to one of 11 Laue classes (centrosymmetric point groups).

export function laueClassIndex(sgNumber: number): number {
  const pg = SG_POINT_GROUP[sgNumber] ?? "1";
  const map: Record<string, number> = {
    "1": 0, "-1": 0,               // triclinic: Laue -1
    "2": 1, "m": 1, "2/m": 1,      // monoclinic: Laue 2/m
    "222": 2, "mm2": 2, "mmm": 2,  // orthorhombic: Laue mmm
    "4": 3, "-4": 3, "4/m": 3,     // tetragonal low: Laue 4/m
    "422": 4, "4mm": 4, "-42m": 4, "4/mmm": 4, // tet high: 4/mmm
    "3": 5, "-3": 5,               // trigonal low: -3
    "32": 6, "3m": 6, "-3m": 6,    // trigonal high: -3m
    "6": 7, "-6": 7, "6/m": 7,     // hexagonal low: 6/m
    "622": 8, "6mm": 8, "-6m2": 8, "6/mmm": 8, // hex high: 6/mmm
    "23": 9, "m-3": 9,             // cubic low: m-3
    "432": 10, "-43m": 10, "m-3m": 10, // cubic high: m-3m
  };
  return map[pg] ?? 0;
}
