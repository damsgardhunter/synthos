import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Element } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Atom, Search, Thermometer, Zap, Weight, Calendar } from "lucide-react";

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

function ElementCard({ element, selected, onClick }: { element: Element; selected: boolean; onClick: () => void }) {
  const bg = CATEGORY_BG[element.category ?? ""] ?? "bg-primary";
  return (
    <button
      onClick={onClick}
      data-testid={`element-card-${element.id}`}
      className={`relative rounded-md p-2 text-left transition-all hover-elevate w-full border ${
        selected
          ? "border-primary bg-primary/5 dark:bg-primary/10"
          : "border-border bg-card"
      }`}
    >
      <div className="flex items-start gap-2">
        <div className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md ${bg} text-white font-bold text-sm`}>
          {element.symbol}
        </div>
        <div className="min-w-0">
          <p className="text-sm font-medium leading-none">{element.name}</p>
          <p className="text-xs text-muted-foreground font-mono mt-0.5">Z={element.id}</p>
          <p className="text-xs text-muted-foreground mt-0.5 truncate">{element.category}</p>
        </div>
      </div>
    </button>
  );
}

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

export default function AtomicExplorer() {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(6);

  const { data: elements, isLoading } = useQuery<Element[]>({ queryKey: ["/api/elements"] });

  const filtered = elements?.filter(el =>
    el.name.toLowerCase().includes(search.toLowerCase()) ||
    el.symbol.toLowerCase().includes(search.toLowerCase()) ||
    el.category?.toLowerCase().includes(search.toLowerCase())
  );

  const selected = elements?.find(e => e.id === selectedId);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <Atom className="h-6 w-6 text-primary" />
          Atomic Explorer
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Deep dive into all elements learned by the supercomputer — from atomic structure to physical properties.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1 space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search elements..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="pl-9"
              data-testid="input-element-search"
            />
          </div>
          <div className="text-xs text-muted-foreground font-mono">
            {filtered?.length ?? 0} of {elements?.length ?? 0} elements
          </div>
          <ScrollArea className="h-[calc(100vh-280px)]">
            {isLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-16" />)}
              </div>
            ) : (
              <div className="space-y-2 pr-3">
                {filtered?.map(el => (
                  <ElementCard
                    key={el.id}
                    element={el}
                    selected={selectedId === el.id}
                    onClick={() => setSelectedId(el.id)}
                  />
                ))}
              </div>
            )}
          </ScrollArea>
        </div>

        <div className="lg:col-span-2 space-y-4">
          {selected ? (
            <>
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-2 flex-wrap">
                    <div>
                      <div className="flex items-center gap-3">
                        <div className={`flex h-14 w-14 items-center justify-center rounded-md text-white font-bold text-xl ${CATEGORY_BG[selected.category ?? ""] ?? "bg-primary"}`}>
                          {selected.symbol}
                        </div>
                        <div>
                          <CardTitle className="text-xl">{selected.name}</CardTitle>
                          <p className="text-sm text-muted-foreground font-mono">Atomic Number: {selected.id}</p>
                        </div>
                      </div>
                    </div>
                    <Badge className={`${CATEGORY_COLORS[selected.category ?? ""] ?? ""} border-0`} data-testid="badge-category">
                      {selected.category}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  {selected.description && (
                    <p className="text-sm text-muted-foreground mb-4 leading-relaxed">{selected.description}</p>
                  )}
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Physical Properties</h3>
                      <PropertyRow label="Atomic Mass" value={selected.atomicMass} unit="u" />
                      <PropertyRow label="Density" value={selected.density ? `${selected.density} g/cm³` : null} />
                      <PropertyRow label="Melting Point" value={selected.meltingPoint ? `${selected.meltingPoint} K` : null} />
                      <PropertyRow label="Boiling Point" value={selected.boilingPoint ? `${selected.boilingPoint} K` : null} />
                    </div>
                    <div>
                      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Quantum Properties</h3>
                      <PropertyRow label="Electronegativity" value={selected.electronegativity} />
                      <PropertyRow label="Period" value={selected.period} />
                      <PropertyRow label="Group" value={selected.group} />
                      <PropertyRow label="Electron Config" value={selected.electronConfig} />
                      {selected.discoveredYear && (
                        <PropertyRow label="Discovered" value={selected.discoveredYear} />
                      )}
                    </div>
                  </div>
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
                    <AtomDiagram element={selected} />
                    <div className="space-y-2 flex-1">
                      <h3 className="text-sm font-medium">Shell Configuration</h3>
                      {getElectronShells(selected.id).map((count, i) => (
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
                      {selected.electronConfig && (
                        <div className="mt-3 p-2 bg-muted rounded-md">
                          <p className="text-xs text-muted-foreground">Configuration</p>
                          <p className="text-sm font-mono font-medium mt-0.5">{selected.electronConfig}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <Atom className="h-12 w-12 text-muted-foreground/30 mb-4" />
              <p className="text-muted-foreground">Select an element to explore</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
