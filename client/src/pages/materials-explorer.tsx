import { useState, useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import type { Element, Material, SuperconductorCandidate } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PaginationBar } from "@/components/ui/pagination-bar";
import { Search, X, Atom, Database, FlaskConical } from "lucide-react";
import { LoadingCard } from "@/components/loading-animation";

const CATEGORY_CELL_BG: Record<string, string> = {
  "nonmetal": "bg-green-500/80 dark:bg-green-600/70",
  "noble gas": "bg-purple-500/80 dark:bg-purple-600/70",
  "alkali metal": "bg-red-500/80 dark:bg-red-600/70",
  "alkaline earth metal": "bg-orange-500/80 dark:bg-orange-600/70",
  "metalloid": "bg-teal-500/80 dark:bg-teal-600/70",
  "post-transition metal": "bg-blue-500/80 dark:bg-blue-600/70",
  "transition metal": "bg-yellow-500/80 dark:bg-yellow-600/70",
  "lanthanide": "bg-pink-500/80 dark:bg-pink-600/70",
  "actinide": "bg-gray-500/80 dark:bg-gray-600/70",
  "halogen": "bg-cyan-500/80 dark:bg-cyan-600/70",
};

const CATEGORY_COLORS: Record<string, string> = {
  "nonmetal": "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300",
  "noble gas": "bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300",
  "alkali metal": "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  "alkaline earth metal": "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300",
  "metalloid": "bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300",
  "post-transition metal": "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
  "transition metal": "bg-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-300",
  "lanthanide": "bg-pink-100 text-pink-800 dark:bg-pink-950 dark:text-pink-300",
  "actinide": "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300",
  "halogen": "bg-cyan-100 text-cyan-800 dark:bg-cyan-950 dark:text-cyan-300",
};

const PERIODIC_TABLE_LAYOUT: [number, number, number][] = [
  [1,1,1],[2,1,18],
  [3,2,1],[4,2,2],[5,2,13],[6,2,14],[7,2,15],[8,2,16],[9,2,17],[10,2,18],
  [11,3,1],[12,3,2],[13,3,13],[14,3,14],[15,3,15],[16,3,16],[17,3,17],[18,3,18],
  [19,4,1],[20,4,2],[21,4,3],[22,4,4],[23,4,5],[24,4,6],[25,4,7],[26,4,8],[27,4,9],[28,4,10],[29,4,11],[30,4,12],[31,4,13],[32,4,14],[33,4,15],[34,4,16],[35,4,17],[36,4,18],
  [37,5,1],[38,5,2],[39,5,3],[40,5,4],[41,5,5],[42,5,6],[43,5,7],[44,5,8],[45,5,9],[46,5,10],[47,5,11],[48,5,12],[49,5,13],[50,5,14],[51,5,15],[52,5,16],[53,5,17],[54,5,18],
  [55,6,1],[56,6,2],
  [72,6,4],[73,6,5],[74,6,6],[75,6,7],[76,6,8],[77,6,9],[78,6,10],[79,6,11],[80,6,12],[81,6,13],[82,6,14],[83,6,15],[84,6,16],[85,6,17],[86,6,18],
  [87,7,1],[88,7,2],
  [104,7,4],[105,7,5],[106,7,6],[107,7,7],[108,7,8],[109,7,9],[110,7,10],[111,7,11],[112,7,12],[113,7,13],[114,7,14],[115,7,15],[116,7,16],[117,7,17],[118,7,18],
  [57,9,4],[58,9,5],[59,9,6],[60,9,7],[61,9,8],[62,9,9],[63,9,10],[64,9,11],[65,9,12],[66,9,13],[67,9,14],[68,9,15],[69,9,16],[70,9,17],[71,9,18],
  [89,10,4],[90,10,5],[91,10,6],[92,10,7],[93,10,8],[94,10,9],[95,10,10],[96,10,11],[97,10,12],[98,10,13],[99,10,14],[100,10,15],[101,10,16],[102,10,17],[103,10,18],
];

const ITEMS_PER_PAGE = 25;

function parseFormulaElements(formula: string): string[] {
  const matches = formula.match(/[A-Z][a-z]?/g);
  return matches ?? [];
}

function formulaContainsAllSymbols(formula: string, symbols: string[]): boolean {
  if (symbols.length === 0) return true;
  const formulaElements = parseFormulaElements(formula);
  return symbols.every(sym => formulaElements.includes(sym));
}

function countElementAtoms(formula: string, symbol: string): number {
  // Match element symbol followed by optional digits, using proper element boundary
  const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`${escaped}(\\d+)?(?=[A-Z(]|$)`, "g");
  let total = 0;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    total += match[1] ? parseInt(match[1], 10) : 1;
  }
  return total;
}

function totalAtomsInFormula(formula: string): number {
  const regex = /[A-Z][a-z]?(\d+)?/g;
  let total = 0;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    total += match[1] ? parseInt(match[1], 10) : 1;
  }
  return total;
}

function smartSort<T extends { formula: string }>(items: T[], selectedSymbols: string[]): T[] {
  if (selectedSymbols.length === 0) return items;

  return [...items].sort((a, b) => {
    const aFormula = a.formula;
    const bFormula = b.formula;

    // 1. Items that START with a selected element come first
    const aStartsWithSelected = selectedSymbols.some(sym => aFormula.startsWith(sym));
    const bStartsWithSelected = selectedSymbols.some(sym => bFormula.startsWith(sym));
    if (aStartsWithSelected && !bStartsWithSelected) return -1;
    if (!aStartsWithSelected && bStartsWithSelected) return 1;

    // 2. Sort by total count of selected element atoms (more = first)
    const aSelectedAtoms = selectedSymbols.reduce((sum, sym) => sum + countElementAtoms(aFormula, sym), 0);
    const bSelectedAtoms = selectedSymbols.reduce((sum, sym) => sum + countElementAtoms(bFormula, sym), 0);
    if (aSelectedAtoms !== bSelectedAtoms) return bSelectedAtoms - aSelectedAtoms;

    // 3. Sort by total atoms in formula (fewer total = simpler compound first)
    const aTotalAtoms = totalAtomsInFormula(aFormula);
    const bTotalAtoms = totalAtomsInFormula(bFormula);
    if (aTotalAtoms !== bTotalAtoms) return aTotalAtoms - bTotalAtoms;

    // 4. Alphabetical fallback
    return aFormula.localeCompare(bFormula);
  });
}

function getElectronShells(atomicNum: number): number[] {
  const shellCapacity = [2, 8, 18, 32, 32, 18, 8];
  const shells: number[] = [];
  let remaining = atomicNum;
  for (const cap of shellCapacity) {
    if (remaining <= 0) break;
    const fill = Math.min(remaining, cap);
    shells.push(fill);
    remaining -= fill;
  }
  return shells;
}

function AtomDiagram({ element }: { element: Element }) {
  const shells = getElectronShells(element.id);
  const maxShell = shells.length;
  const cx = 100, cy = 100;
  const radii = [22, 42, 60, 76, 88, 98, 108].slice(0, maxShell);

  return (
    <svg width="200" height="200" className="mx-auto" viewBox="0 0 200 200">
      <circle cx={cx} cy={cy} r="15" fill="hsl(var(--gold))" opacity="0.85" />
      <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="9" fontWeight="bold">{element.symbol}</text>

      {radii.map((r, si) => (
        <circle key={si} cx={cx} cy={cy} r={r} fill="none" stroke="hsl(var(--gold)/0.3)" strokeWidth="1" strokeDasharray="3 2" />
      ))}

      {shells.map((count, si) => {
        const r = radii[si];
        return Array.from({ length: count }).map((_, ei) => {
          const angle = (2 * Math.PI * ei) / count - Math.PI / 2;
          const ex = cx + r * Math.cos(angle);
          const ey = cy + r * Math.sin(angle);
          return (
            <circle key={`${si}-${ei}`} cx={ex} cy={ey} r="3" fill="hsl(var(--gold-light))" opacity="0.7" />
          );
        });
      })}
    </svg>
  );
}

export default function MaterialsExplorer() {
  const [selectedElements, setSelectedElements] = useState<string[]>([]);
  const [focusedElement, setFocusedElement] = useState<Element | null>(null);
  const [searchText, setSearchText] = useState("");
  const [materialsPage, setMaterialsPage] = useState(1);
  const [candidatesPage, setCandidatesPage] = useState(1);
  const [confidenceFilter, setConfidenceFilter] = useState<string>("");
  // null = no results shown, "materials" or "candidates" = show that panel
  const [showResults, setShowResults] = useState<"materials" | "candidates" | null>(null);

  const { data: elements, isLoading: elementsLoading } = useQuery<Element[]>({
    queryKey: ["/api/elements"],
  });

  const { data: materialsData, isLoading: materialsLoading } = useQuery<{ materials: Material[]; total: number }>({
    queryKey: ["/api/materials"],
    enabled: showResults === "materials",
  });

  const elementsFilter = selectedElements.length > 0 ? selectedElements.join(",") : "";
  const { data: candidatesData, isLoading: candidatesLoading } = useQuery<{ candidates: SuperconductorCandidate[]; total: number }>({
    queryKey: ["/api/superconductor-candidates", elementsFilter, confidenceFilter],
    queryFn: () => {
      const params = new URLSearchParams({ limit: "5000", offset: "0" });
      if (elementsFilter) params.set("elements", elementsFilter);
      if (confidenceFilter) params.set("confidence", confidenceFilter);
      return fetch(`/api/superconductor-candidates?${params}`).then(r => r.json());
    },
  });

  const { data: relatedCounts } = useQuery<{ materialCount: number; candidateCount: number }>({
    queryKey: ["/api/elements", focusedElement?.symbol, "related-counts"],
    queryFn: () => fetch(`/api/elements/${focusedElement!.symbol}/related-counts`).then(r => r.json()),
    enabled: !!focusedElement,
  });

  const elementMap = useMemo(() => {
    if (!elements) return new Map<number, Element>();
    return new Map(elements.map(e => [e.id, e]));
  }, [elements]);

  const activeSymbols = useMemo(() => {
    const symbols = [...selectedElements];
    if (searchText.trim()) {
      const parts = searchText.trim().split(/[\s,]+/);
      symbols.push(...parts);
    }
    return symbols;
  }, [selectedElements, searchText]);

  // Build a display label for the selected element combination
  const selectionLabel = useMemo(() => {
    if (selectedElements.length === 0) return null;
    if (selectedElements.length === 1) {
      const el = elements?.find(e => e.symbol === selectedElements[0]);
      return el ? el.name : selectedElements[0];
    }
    return selectedElements.join("-") + " compounds";
  }, [selectedElements, elements]);

  const filteredMaterials = useMemo(() => {
    if (!materialsData?.materials) return [];
    const filtered = activeSymbols.length === 0
      ? materialsData.materials
      : materialsData.materials.filter(m => formulaContainsAllSymbols(m.formula, activeSymbols));
    return smartSort(filtered, activeSymbols);
  }, [materialsData, activeSymbols]);

  const filteredCandidates = useMemo(() => {
    if (!candidatesData?.candidates) return [];
    const filtered = activeSymbols.length === 0
      ? candidatesData.candidates
      : candidatesData.candidates.filter(c => formulaContainsAllSymbols(c.formula, activeSymbols));
    return smartSort(filtered, activeSymbols);
  }, [candidatesData, activeSymbols]);

  const materialsTotalPages = Math.max(1, Math.ceil(filteredMaterials.length / ITEMS_PER_PAGE));
  const candidatesTotalPages = Math.max(1, Math.ceil(filteredCandidates.length / ITEMS_PER_PAGE));

  const paginatedMaterials = filteredMaterials.slice((materialsPage - 1) * ITEMS_PER_PAGE, materialsPage * ITEMS_PER_PAGE);
  const paginatedCandidates = filteredCandidates.slice((candidatesPage - 1) * ITEMS_PER_PAGE, candidatesPage * ITEMS_PER_PAGE);

  const toggleElement = useCallback((symbol: string) => {
    setSelectedElements(prev => {
      const next = prev.includes(symbol) ? prev.filter(s => s !== symbol) : [...prev, symbol];
      return next;
    });
    setMaterialsPage(1);
    setCandidatesPage(1);
  }, []);

  const handleElementClick = useCallback((el: Element) => {
    toggleElement(el.symbol);
    setFocusedElement(prev => prev?.id === el.id ? null : el);
  }, [toggleElement]);

  const clearAll = useCallback(() => {
    setSelectedElements([]);
    setSearchText("");
    setFocusedElement(null);
    setMaterialsPage(1);
    setCandidatesPage(1);
    setShowResults(null);
  }, []);

  return (
    <div className="min-h-screen p-4 md:p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="gold-text synthos-heading text-3xl md:text-4xl font-bold">Material Explorer</h1>
        {(selectedElements.length > 0 || searchText) && (
          <button
            onClick={clearAll}
            className="text-sm text-[hsl(var(--gold-muted))] hover:text-[hsl(var(--gold))] transition-colors"
          >
            Clear All
          </button>
        )}
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[hsl(var(--gold-muted))]" />
        <Input
          placeholder="Search by element symbols or formula (e.g. Li H, YBaCuO)..."
          value={searchText}
          onChange={e => { setSearchText(e.target.value); setMaterialsPage(1); setCandidatesPage(1); }}
          className="pl-10 border-[hsl(var(--gold)/0.3)] bg-[hsl(var(--card))] text-foreground placeholder:text-[hsl(var(--gold-muted))]"
        />
      </div>

      {selectedElements.length > 0 && (
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-sm text-[hsl(var(--gold-muted))]">Selected:</span>
          {selectedElements.map(sym => (
            <Badge
              key={sym}
              variant="outline"
              className="border-[hsl(var(--gold)/0.5)] bg-[hsl(var(--gold)/0.1)] text-[hsl(var(--gold-light))] cursor-pointer hover:bg-[hsl(var(--gold)/0.2)] transition-colors"
              onClick={() => toggleElement(sym)}
            >
              {sym}
              <X className="ml-1 h-3 w-3" />
            </Badge>
          ))}
          {selectionLabel && (
            <span className="text-sm text-[hsl(var(--gold-light))] ml-2 font-medium">{selectionLabel}</span>
          )}
        </div>
      )}

      <Card className="gold-card overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="text-[hsl(var(--gold))] flex items-center gap-2 text-lg">
            <Atom className="h-5 w-5" />
            Periodic Table
          </CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto pb-4">
          {elementsLoading ? (
            <LoadingCard height="h-[320px]" size="lg" />
          ) : (
            <div
              className="grid gap-[2px] min-w-[720px]"
              style={{
                gridTemplateColumns: "repeat(18, minmax(0, 1fr))",
                gridTemplateRows: "repeat(10, minmax(0, 1fr))",
              }}
            >
              {PERIODIC_TABLE_LAYOUT.map(([atomicNum, row, col]) => {
                const el = elementMap.get(atomicNum);
                if (!el) return null;
                const isSelected = selectedElements.includes(el.symbol);
                const isFocused = focusedElement?.id === el.id;
                const cellBg = CATEGORY_CELL_BG[el.category ?? ""] ?? "bg-gray-500/80";
                return (
                  <button
                    key={atomicNum}
                    onClick={() => handleElementClick(el)}
                    className={`
                      relative flex flex-col items-center justify-center p-0.5 rounded-sm text-white cursor-pointer
                      transition-all duration-150 hover:scale-110 hover:z-10
                      ${cellBg}
                      ${isSelected ? "ring-2 ring-[hsl(var(--gold))] ring-offset-1 ring-offset-black scale-105 z-10" : ""}
                      ${isFocused ? "ring-2 ring-white" : ""}
                    `}
                    style={{ gridRow: row, gridColumn: col }}
                    title={`${el.name} (${el.symbol}) - ${el.category}`}
                  >
                    <span className="text-[8px] leading-none opacity-70">{el.id}</span>
                    <span className="text-xs font-bold leading-none">{el.symbol}</span>
                  </button>
                );
              })}
            </div>
          )}
          <div className="flex flex-wrap gap-2 mt-3">
            {Object.entries(CATEGORY_CELL_BG).map(([cat, bg]) => (
              <div key={cat} className="flex items-center gap-1">
                <div className={`w-3 h-3 rounded-sm ${bg}`} />
                <span className="text-[10px] text-muted-foreground capitalize">{cat}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {focusedElement && (
        <Card className="gold-card">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-white font-bold text-xl ${CATEGORY_CELL_BG[focusedElement.category ?? ""] ?? "bg-gray-500"}`}>
                  {focusedElement.symbol}
                </div>
                <div>
                  <CardTitle className="text-[hsl(var(--gold-light))] text-xl">
                    {selectionLabel || focusedElement.name}
                  </CardTitle>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-sm text-muted-foreground">#{focusedElement.id}</span>
                    <Badge className={CATEGORY_COLORS[focusedElement.category ?? ""] ?? ""}>
                      {focusedElement.category}
                    </Badge>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setFocusedElement(null)}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-1">
                <h4 className="text-sm font-semibold text-[hsl(var(--gold))] mb-2">Physical Properties</h4>
                <PropertyRow label="Atomic Mass" value={focusedElement.atomicMass} unit="u" />
                <PropertyRow label="Density" value={focusedElement.density} unit="g/cm³" />
                <PropertyRow label="Melting Point" value={focusedElement.meltingPoint} unit="K" />
                <PropertyRow label="Boiling Point" value={focusedElement.boilingPoint} unit="K" />
              </div>
              <div className="space-y-1">
                <h4 className="text-sm font-semibold text-[hsl(var(--gold))] mb-2">Quantum Properties</h4>
                <PropertyRow label="Electronegativity" value={focusedElement.electronegativity} />
                <PropertyRow label="Period" value={focusedElement.period} />
                <PropertyRow label="Group" value={focusedElement.group} />
                <PropertyRow label="Electron Config" value={focusedElement.electronConfiguration} />
              </div>
              <div className="space-y-1">
                <h4 className="text-sm font-semibold text-[hsl(var(--gold))] mb-2">Database Connections</h4>
                {relatedCounts ? (
                  <div className="rounded-lg border border-[hsl(var(--gold)/0.3)] bg-[hsl(var(--gold)/0.05)] p-3 space-y-2">
                    <button
                      className={`flex items-center justify-between w-full py-2 px-3 rounded-md border transition-colors ${
                        showResults === "materials"
                          ? "border-[hsl(var(--gold))] bg-[hsl(var(--gold)/0.15)] text-[hsl(var(--gold))]"
                          : "border-[hsl(var(--gold)/0.2)] hover:border-[hsl(var(--gold)/0.5)] hover:bg-[hsl(var(--gold)/0.08)]"
                      }`}
                      onClick={() => {
                        setShowResults(prev => prev === "materials" ? null : "materials");
                        setMaterialsPage(1);
                      }}
                    >
                      <span className="text-sm flex items-center gap-2">
                        <Database className="h-4 w-4" /> Materials
                      </span>
                      <span className="text-sm font-mono font-medium text-[hsl(var(--gold-light))]">{relatedCounts.materialCount}</span>
                    </button>
                    <button
                      className={`flex items-center justify-between w-full py-2 px-3 rounded-md border transition-colors ${
                        showResults === "candidates"
                          ? "border-[hsl(var(--gold))] bg-[hsl(var(--gold)/0.15)] text-[hsl(var(--gold))]"
                          : "border-[hsl(var(--gold)/0.2)] hover:border-[hsl(var(--gold)/0.5)] hover:bg-[hsl(var(--gold)/0.08)]"
                      }`}
                      onClick={() => {
                        setShowResults(prev => prev === "candidates" ? null : "candidates");
                        setCandidatesPage(1);
                      }}
                    >
                      <span className="text-sm flex items-center gap-2">
                        <FlaskConical className="h-4 w-4" /> SC Candidates
                      </span>
                      <span className="text-sm font-mono font-medium text-[hsl(var(--gold-light))]">{relatedCounts.candidateCount}</span>
                    </button>
                  </div>
                ) : (
                  <Skeleton className="h-24 w-full" />
                )}
              </div>
            </div>

            {/* Bohr Model */}
            <div className="mt-6 border-t border-[hsl(var(--gold)/0.2)] pt-4">
              <h4 className="text-sm font-semibold text-[hsl(var(--gold))] mb-2 flex items-center gap-2">
                <Atom className="h-4 w-4" />
                Bohr Model — Electron Shell Diagram
              </h4>
              <div className="flex flex-col sm:flex-row items-center gap-6">
                <AtomDiagram element={focusedElement} />
                <div className="space-y-2 flex-1">
                  <h4 className="text-sm font-medium text-muted-foreground">Shell Configuration</h4>
                  {getElectronShells(focusedElement.id).map((count, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <div className="text-xs font-mono text-muted-foreground w-12">Shell {i + 1}</div>
                      <div className="flex-1 bg-[hsl(var(--gold)/0.1)] rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-[hsl(var(--gold))] h-full rounded-full transition-all"
                          style={{ width: `${(count / [2, 8, 18, 32, 32, 18, 8][i]) * 100}%` }}
                        />
                      </div>
                      <div className="text-xs font-mono w-16 text-right text-muted-foreground">{count} / {[2, 8, 18, 32, 32, 18, 8][i]}</div>
                    </div>
                  ))}
                  {focusedElement.electronConfiguration && (
                    <div className="mt-3 p-2 bg-[hsl(var(--gold)/0.05)] border border-[hsl(var(--gold)/0.2)] rounded-md">
                      <p className="text-xs text-muted-foreground">Configuration</p>
                      <p className="text-sm font-mono font-medium mt-0.5 text-[hsl(var(--gold-light))]">{focusedElement.electronConfiguration}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {showResults === "materials" && (
        <Card className="gold-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-[hsl(var(--gold))] text-lg flex items-center gap-2">
              <Database className="h-5 w-5" />
              Materials {selectionLabel ? `— ${selectionLabel}` : ""} ({filteredMaterials.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {materialsLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-12 w-full" />)}
              </div>
            ) : filteredMaterials.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                {activeSymbols.length > 0
                  ? "No materials found matching the selected elements."
                  : "Select elements to filter materials."}
              </div>
            ) : (
              <>
                <ScrollArea className="max-h-[500px]">
                  <div className="space-y-1">
                    {paginatedMaterials.map((mat, i) => (
                      <Link
                        key={mat.id ?? i}
                        href={`/candidate/${encodeURIComponent(mat.formula)}`}
                        className="flex items-center justify-between px-3 py-2.5 rounded-md border border-[hsl(var(--gold)/0.15)] hover:border-[hsl(var(--gold)/0.4)] hover:bg-[hsl(var(--gold)/0.05)] transition-colors group"
                      >
                        <div className="flex items-center gap-3">
                          <span className="font-mono font-semibold text-[hsl(var(--gold-light))] group-hover:text-[hsl(var(--gold))]">
                            {mat.formula}
                          </span>
                          {mat.name && (
                            <span className="text-sm text-muted-foreground">{mat.name}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          {mat.spaceGroup && (
                            <Badge variant="outline" className="text-xs border-[hsl(var(--gold)/0.3)] text-muted-foreground">
                              {mat.spaceGroup}
                            </Badge>
                          )}
                        </div>
                      </Link>
                    ))}
                  </div>
                </ScrollArea>
                <PaginationBar page={materialsPage} totalPages={materialsTotalPages} onPageChange={setMaterialsPage} />
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* SC Candidates — always visible with confidence filter */}
      <Card className="gold-card">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <CardTitle className="text-[hsl(var(--gold))] text-lg flex items-center gap-2">
              <FlaskConical className="h-5 w-5" />
              SC Candidates {selectionLabel ? `— ${selectionLabel}` : ""} ({filteredCandidates.length})
            </CardTitle>
            <div className="flex items-center gap-1.5">
              {[
                { value: "", label: "All" },
                { value: "dft-verified", label: "DFT Verified" },
                { value: "high", label: "High Confidence" },
                { value: "medium", label: "Model Est." },
                { value: "low", label: "Heuristic" },
              ].map(opt => (
                <button
                  key={opt.value}
                  onClick={() => { setConfidenceFilter(opt.value); setCandidatesPage(1); }}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                    confidenceFilter === opt.value
                      ? "bg-[hsl(var(--gold)/0.2)] text-[hsl(var(--gold))] border border-[hsl(var(--gold)/0.4)]"
                      : "text-muted-foreground hover:text-[hsl(var(--gold-light))] border border-transparent hover:border-[hsl(var(--gold)/0.2)]"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {candidatesLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-12 w-full" />)}
              </div>
            ) : filteredCandidates.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                {confidenceFilter === "dft-verified"
                  ? "No DFT-verified candidates found. DFT verification requires running Quantum ESPRESSO calculations."
                  : activeSymbols.length > 0
                  ? "No superconductor candidates found matching the selected elements."
                  : "No candidates match the current filters."}
              </div>
            ) : (
              <>
                <ScrollArea className="max-h-[500px]">
                  <div className="space-y-1">
                    {paginatedCandidates.map((cand, i) => (
                      <Link
                        key={cand.id ?? i}
                        href={`/candidate/${encodeURIComponent(cand.formula)}`}
                        className="flex items-center justify-between px-3 py-2.5 rounded-md border border-[hsl(var(--gold)/0.15)] hover:border-[hsl(var(--gold)/0.4)] hover:bg-[hsl(var(--gold)/0.05)] transition-colors group"
                      >
                        <div className="flex items-center gap-3">
                          <span className="font-mono font-semibold text-[hsl(var(--gold-light))] group-hover:text-[hsl(var(--gold))]">
                            {cand.formula}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          {cand.predictedTc != null && (
                            <Badge variant="outline" className="text-xs border-[hsl(var(--gold)/0.3)] text-[hsl(var(--gold-muted))]">
                              Tc: {Number(cand.predictedTc).toFixed(1)} K
                            </Badge>
                          )}
                          {cand.confidence != null && (
                            <Badge variant="outline" className="text-xs border-[hsl(var(--gold)/0.3)] text-[hsl(var(--gold-muted))]">
                              {(Number(cand.confidence) * 100).toFixed(0)}%
                            </Badge>
                          )}
                          {cand.dataConfidence && (() => {
                            // Only show "DFT Verified" if verification stage >= 2 (actual DFT ran)
                            const hasRealDFT = (cand.verificationStage ?? 0) >= 2;
                            const displayConfidence = (cand.dataConfidence === "dft-verified" && !hasRealDFT) ? "high" : cand.dataConfidence;
                            return (
                              <Badge
                                className={
                                  displayConfidence === "dft-verified"
                                    ? "bg-green-500/20 text-green-400 border-green-500/30"
                                    : displayConfidence === "high"
                                    ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                                    : displayConfidence === "medium"
                                    ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
                                    : "bg-gray-500/20 text-gray-400 border-gray-500/30"
                                }
                                variant="outline"
                              >
                                {displayConfidence === "dft-verified" ? "DFT Verified"
                                  : displayConfidence === "high" ? "Physics Est."
                                  : displayConfidence === "medium" ? "Model"
                                  : displayConfidence}
                              </Badge>
                            );
                          })()}
                          {cand.status && (
                            <Badge className="bg-gray-500/20 text-gray-400 border-gray-500/30" variant="outline">
                              {cand.status}
                            </Badge>
                          )}
                        </div>
                      </Link>
                    ))}
                  </div>
                </ScrollArea>
                <PaginationBar page={candidatesPage} totalPages={candidatesTotalPages} onPageChange={setCandidatesPage} />
              </>
            )}
          </CardContent>
        </Card>
    </div>
  );
}

function PropertyRow({ label, value, unit }: { label: string; value?: string | number | null; unit?: string }) {
  if (value == null) return null;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-[hsl(var(--gold)/0.2)] last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-mono font-medium text-foreground">
        {typeof value === "number" ? value.toLocaleString() : value}
        {unit && <span className="text-muted-foreground text-xs ml-1">{unit}</span>}
      </span>
    </div>
  );
}
