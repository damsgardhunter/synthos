import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import type { Element } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Atom, Search, Database, FlaskConical } from "lucide-react";

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

const CATEGORY_BG: Record<string, string> = {
  "nonmetal": "bg-green-500",
  "noble gas": "bg-purple-500",
  "alkali metal": "bg-red-500",
  "alkaline earth metal": "bg-orange-500",
  "metalloid": "bg-teal-500",
  "post-transition metal": "bg-blue-500",
  "transition metal": "bg-yellow-500",
  "lanthanide": "bg-pink-500",
  "actinide": "bg-gray-500",
  "halogen": "bg-cyan-500",
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

function PropertyRow({ label, value, unit }: { label: string; value?: string | number | null; unit?: string }) {
  if (value == null) return null;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-mono font-medium text-foreground">
        {typeof value === "number" ? value.toLocaleString() : value}
        {unit && <span className="text-muted-foreground text-xs ml-1">{unit}</span>}
      </span>
    </div>
  );
}

function AtomDiagram({ element }: { element: Element }) {
  const shells = getElectronShells(element.id);
  const maxShell = shells.length;
  const cx = 120, cy = 120;
  const radii = [28, 52, 76, 96, 112, 126, 138].slice(0, maxShell);

  return (
    <svg width="240" height="240" className="mx-auto" viewBox="0 0 240 240">
      <circle cx={cx} cy={cy} r="18" fill="hsl(var(--primary))" opacity="0.85" />
      <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize="10" fontWeight="bold">{element.symbol}</text>

      {radii.map((r, si) => (
        <circle key={si} cx={cx} cy={cy} r={r} fill="none" stroke="hsl(var(--border))" strokeWidth="1" strokeDasharray="3 2" />
      ))}

      {shells.map((count, si) => {
        const r = radii[si];
        return Array.from({ length: count }).map((_, ei) => {
          const angle = (2 * Math.PI * ei) / count - Math.PI / 2;
          const ex = cx + r * Math.cos(angle);
          const ey = cy + r * Math.sin(angle);
          return (
            <circle key={`${si}-${ei}`} cx={ex} cy={ey} r="4" fill="hsl(var(--primary))" opacity="0.7" />
          );
        });
      })}
    </svg>
  );
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

function PeriodicTableCell({
  element,
  selected,
  highlighted,
  dimmed,
  onClick,
}: {
  element: Element;
  selected: boolean;
  highlighted: boolean;
  dimmed: boolean;
  onClick: () => void;
}) {
  const cellBg = CATEGORY_CELL_BG[element.category ?? ""] ?? "bg-primary/80";
  return (
    <button
      onClick={onClick}
      data-testid={`element-cell-${element.id}`}
      className={`relative flex flex-col items-center justify-center rounded-md p-0.5 text-white transition-all w-full h-full min-w-[2.5rem] min-h-[2.5rem] ${cellBg} ${
        selected
          ? "ring-2 ring-primary ring-offset-1 ring-offset-background scale-110 z-10"
          : ""
      } ${dimmed ? "opacity-25" : ""} ${highlighted && !selected ? "ring-1 ring-white/60" : ""}`}
      style={{ fontSize: "0.6rem" }}
    >
      <span className="text-[0.5rem] leading-none opacity-80">{element.id}</span>
      <span className="font-bold text-xs leading-tight">{element.symbol}</span>
    </button>
  );
}

function CategoryLegend() {
  const categories = [
    "alkali metal", "alkaline earth metal", "transition metal", "post-transition metal",
    "metalloid", "nonmetal", "halogen", "noble gas", "lanthanide", "actinide",
  ];
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {categories.map(cat => (
        <div key={cat} className="flex items-center gap-1.5">
          <div className={`w-3 h-3 rounded-sm ${CATEGORY_BG[cat]}`} />
          <span className="text-xs text-muted-foreground capitalize">{cat}</span>
        </div>
      ))}
    </div>
  );
}

function ElementDetailPanel({ element }: { element: Element }) {
  const { data: relatedCounts } = useQuery<{ materialCount: number; candidateCount: number }>({
    queryKey: ["/api/elements", element.symbol, "related-counts"],
    queryFn: async () => {
      const res = await fetch(`/api/elements/${encodeURIComponent(element.symbol)}/related-counts`);
      if (!res.ok) throw new Error("Failed to fetch");
      return res.json();
    },
  });

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-2 flex-wrap">
            <div>
              <div className="flex items-center gap-3">
                <div className={`flex h-14 w-14 items-center justify-center rounded-md text-white font-bold text-xl ${CATEGORY_BG[element.category ?? ""] ?? "bg-primary"}`}>
                  {element.symbol}
                </div>
                <div>
                  <CardTitle className="text-xl">{element.name}</CardTitle>
                  <p className="text-sm text-muted-foreground font-mono">Atomic Number: {element.id}</p>
                </div>
              </div>
            </div>
            <Badge className={`${CATEGORY_COLORS[element.category ?? ""] ?? ""} border-0`} data-testid="badge-category">
              {element.category}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          {element.description && (
            <p className="text-sm text-muted-foreground mb-4 leading-relaxed">{element.description}</p>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Physical Properties</h3>
              <PropertyRow label="Atomic Mass" value={element.atomicMass} unit="u" />
              <PropertyRow label="Density" value={element.density ? `${element.density} g/cm\u00B3` : null} />
              <PropertyRow label="Melting Point" value={element.meltingPoint ? `${element.meltingPoint} K` : null} />
              <PropertyRow label="Boiling Point" value={element.boilingPoint ? `${element.boilingPoint} K` : null} />
            </div>
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Quantum Properties</h3>
              <PropertyRow label="Electronegativity" value={element.electronegativity} />
              <PropertyRow label="Period" value={element.period} />
              <PropertyRow label="Group" value={element.group} />
              <PropertyRow label="Electron Config" value={element.electronConfig} />
              {element.discoveredYear && (
                <PropertyRow label="Discovered" value={element.discoveredYear} />
              )}
            </div>
          </div>

          {relatedCounts && (
            <div className="mt-4 pt-4 border-t border-border">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">Database Connections</h3>
              <div className="grid grid-cols-2 gap-3">
                <Link href="/materials" data-testid={`link-materials-${element.symbol}`}>
                  <Card className="hover-elevate cursor-pointer">
                    <CardContent className="p-3 flex items-center gap-3">
                      <Database className="h-5 w-5 text-blue-500 flex-shrink-0" />
                      <div>
                        <p className="text-lg font-bold" data-testid={`text-material-count-${element.symbol}`}>{relatedCounts.materialCount}</p>
                        <p className="text-xs text-muted-foreground">Materials</p>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
                <Link href="/superconductor-lab" data-testid={`link-candidates-${element.symbol}`}>
                  <Card className="hover-elevate cursor-pointer">
                    <CardContent className="p-3 flex items-center gap-3">
                      <FlaskConical className="h-5 w-5 text-yellow-500 flex-shrink-0" />
                      <div>
                        <p className="text-lg font-bold" data-testid={`text-candidate-count-${element.symbol}`}>{relatedCounts.candidateCount}</p>
                        <p className="text-xs text-muted-foreground">SC Candidates</p>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Atom className="h-4 w-4 text-primary" />
            Bohr Model — Electron Shell Diagram
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row items-center gap-6">
            <AtomDiagram element={element} />
            <div className="space-y-2 flex-1">
              <h3 className="text-sm font-medium">Shell Configuration</h3>
              {getElectronShells(element.id).map((count, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="text-xs font-mono text-muted-foreground w-12">Shell {i + 1}</div>
                  <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-primary h-full rounded-full transition-all"
                      style={{ width: `${(count / [2, 8, 18, 32, 32, 18, 8][i]) * 100}%` }}
                    />
                  </div>
                  <div className="text-xs font-mono w-16 text-right">{count} / {[2, 8, 18, 32, 32, 18, 8][i]}</div>
                </div>
              ))}
              {element.electronConfig && (
                <div className="mt-3 p-2 bg-muted rounded-md">
                  <p className="text-xs text-muted-foreground">Configuration</p>
                  <p className="text-sm font-mono font-medium mt-0.5">{element.electronConfig}</p>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function AtomicExplorer() {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const { data: elements, isLoading } = useQuery<Element[]>({ queryKey: ["/api/elements"] });

  const elementMap = useMemo(() => {
    const map = new Map<number, Element>();
    elements?.forEach(el => map.set(el.id, el));
    return map;
  }, [elements]);

  const matchingIds = useMemo(() => {
    if (!search.trim() || !elements) return null;
    const q = search.toLowerCase();
    return new Set(
      elements
        .filter(el =>
          el.name.toLowerCase().includes(q) ||
          el.symbol.toLowerCase().includes(q) ||
          el.category?.toLowerCase().includes(q)
        )
        .map(el => el.id)
    );
  }, [search, elements]);

  const selected = selectedId != null ? elementMap.get(selectedId) : undefined;

  return (
    <div className="p-6 max-w-full mx-auto space-y-4">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Atom className="h-6 w-6 text-primary" />
            Atomic Explorer
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            Interactive periodic table — click any element to explore its properties and database connections.
          </p>
        </div>
        <div className="relative w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search elements..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-9"
            data-testid="input-element-search"
          />
        </div>
      </div>

      {isLoading ? (
        <Skeleton className="h-80 w-full" />
      ) : (
        <Card>
          <CardContent className="p-4">
            <div className="overflow-x-auto">
              <div
                className="grid gap-0.5"
                style={{
                  gridTemplateColumns: "repeat(18, minmax(2.5rem, 1fr))",
                  gridTemplateRows: "repeat(10, minmax(2.5rem, auto))",
                  minWidth: "52rem",
                }}
              >
                {PERIODIC_TABLE_LAYOUT.map(([atomicNum, row, col]) => {
                  const el = elementMap.get(atomicNum);
                  if (!el) return null;
                  const isHighlighted = matchingIds ? matchingIds.has(atomicNum) : false;
                  const isDimmed = matchingIds !== null && !matchingIds.has(atomicNum);
                  return (
                    <div
                      key={atomicNum}
                      style={{ gridRow: row, gridColumn: col }}
                    >
                      <PeriodicTableCell
                        element={el}
                        selected={selectedId === atomicNum}
                        highlighted={isHighlighted}
                        dimmed={isDimmed}
                        onClick={() => setSelectedId(atomicNum)}
                      />
                    </div>
                  );
                })}

                <div style={{ gridRow: 6, gridColumn: 3 }} className="flex items-center justify-center">
                  <span className="text-[0.6rem] text-muted-foreground">57-71</span>
                </div>
                <div style={{ gridRow: 7, gridColumn: 3 }} className="flex items-center justify-center">
                  <span className="text-[0.6rem] text-muted-foreground">89-103</span>
                </div>
              </div>
            </div>
            <CategoryLegend />
          </CardContent>
        </Card>
      )}

      {selected ? (
        <ElementDetailPanel element={selected} />
      ) : (
        <div className="flex flex-col items-center justify-center h-40 text-center">
          <Atom className="h-12 w-12 text-muted-foreground/30 mb-4" />
          <p className="text-muted-foreground">Select an element from the periodic table to explore</p>
        </div>
      )}
    </div>
  );
}
