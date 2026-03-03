import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Material } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Database, Search, Zap, Layers, ChevronRight, Info } from "lucide-react";

const SOURCE_COLORS: Record<string, string> = {
  "NIST": "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  "Materials Project": "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  "OQMD": "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
  "AFLOW": "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
};

function PropertyBadge({ label, value, unit }: { label: string; value: any; unit?: string }) {
  if (value == null || value === undefined) return null;
  return (
    <div className="flex items-center justify-between text-sm py-1.5 border-b border-border last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono font-medium">
        {typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(value)}
        {unit && <span className="text-muted-foreground text-xs ml-1">{unit}</span>}
      </span>
    </div>
  );
}

function MaterialCard({ material, selected, onClick }: { material: Material; selected: boolean; onClick: () => void }) {
  const props = material.properties as Record<string, any> ?? {};
  return (
    <button
      onClick={onClick}
      data-testid={`material-card-${material.id}`}
      className={`w-full text-left rounded-md p-3 border transition-all hover-elevate ${
        selected ? "border-primary bg-primary/5 dark:bg-primary/10" : "border-border bg-card"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold truncate">{material.name}</p>
          <p className="text-xs font-mono text-primary mt-0.5">{material.formula}</p>
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${SOURCE_COLORS[material.source] ?? "bg-muted text-muted-foreground"}`}>
              {material.source}
            </span>
            {material.bandGap !== null && material.bandGap !== undefined && (
              <span className="text-xs text-muted-foreground">
                Eg: <span className="font-mono">{material.bandGap.toFixed(2)} eV</span>
              </span>
            )}
          </div>
        </div>
        <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0 mt-1" />
      </div>
    </button>
  );
}

export default function MaterialsDatabase() {
  const [search, setSearch] = useState("");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { data, isLoading } = useQuery<{ materials: Material[]; total: number }>({
    queryKey: ["/api/materials"],
  });

  const materials = data?.materials ?? [];

  const filtered = materials.filter(m => {
    const matchesSearch = !search ||
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.formula.toLowerCase().includes(search.toLowerCase());
    const matchesSource = sourceFilter === "all" || m.source === sourceFilter;
    return matchesSearch && matchesSource;
  });

  const selected = materials.find(m => m.id === selectedId);
  const selectedProps = selected?.properties as Record<string, any> ?? {};

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <Database className="h-6 w-6 text-primary" />
          Materials Database
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Indexed materials from NIST, Materials Project, OQMD, and AFLOW databases with full property characterization.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { source: "NIST", label: "NIST WebBook", icon: "🔵", count: materials.filter(m => m.source === "NIST").length },
          { source: "Materials Project", label: "Mat. Project", icon: "🟣", count: materials.filter(m => m.source === "Materials Project").length },
          { source: "OQMD", label: "OQMD", icon: "🟠", count: materials.filter(m => m.source === "OQMD").length },
          { source: "AFLOW", label: "AFLOW", icon: "🟢", count: materials.filter(m => m.source === "AFLOW").length },
        ].map(src => (
          <Card key={src.source} className="cursor-pointer hover-elevate" onClick={() => setSourceFilter(sourceFilter === src.source ? "all" : src.source)} data-testid={`filter-${src.source.toLowerCase()}`}>
            <CardContent className="p-3">
              <div className="text-2xl font-bold font-mono text-foreground">{src.count}</div>
              <div className="text-xs text-muted-foreground mt-1">{src.label}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-2 space-y-3">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search materials..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="pl-9"
                data-testid="input-material-search"
              />
            </div>
            <Select value={sourceFilter} onValueChange={setSourceFilter}>
              <SelectTrigger className="w-36" data-testid="select-source-filter">
                <SelectValue placeholder="Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sources</SelectItem>
                <SelectItem value="NIST">NIST</SelectItem>
                <SelectItem value="Materials Project">Mat. Project</SelectItem>
                <SelectItem value="OQMD">OQMD</SelectItem>
                <SelectItem value="AFLOW">AFLOW</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <p className="text-xs text-muted-foreground font-mono">{filtered.length} of {materials.length} materials</p>
          <ScrollArea className="h-[calc(100vh-320px)]">
            {isLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
              </div>
            ) : (
              <div className="space-y-2 pr-2">
                {filtered.map(m => (
                  <MaterialCard
                    key={m.id}
                    material={m}
                    selected={selectedId === m.id}
                    onClick={() => setSelectedId(m.id)}
                  />
                ))}
                {filtered.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground text-sm">No materials match your search</div>
                )}
              </div>
            )}
          </ScrollArea>
        </div>

        <div className="lg:col-span-3">
          {selected ? (
            <div className="space-y-4">
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-2 flex-wrap">
                    <div>
                      <CardTitle className="text-lg">{selected.name}</CardTitle>
                      <p className="text-sm font-mono text-primary mt-1">{selected.formula}</p>
                    </div>
                    <Badge className={`${SOURCE_COLORS[selected.source] ?? ""} border-0`}>
                      {selected.source}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Crystallographic</h3>
                      <PropertyBadge label="Space Group" value={selected.spacegroup} />
                      <PropertyBadge label="Band Gap" value={selected.bandGap} unit="eV" />
                      <PropertyBadge label="Formation Energy" value={selected.formationEnergy} unit="eV/atom" />
                      <PropertyBadge label="Stability (eH)" value={selected.stability} unit="eV" />
                    </div>
                    <div>
                      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Additional Properties</h3>
                      {Object.entries(selectedProps).map(([k, v]) => (
                        <PropertyBadge
                          key={k}
                          label={k.replace(/([A-Z])/g, " $1").replace(/^./, s => s.toUpperCase())}
                          value={v}
                        />
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Layers className="h-4 w-4 text-primary" />
                    Electronic Properties Visualization
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-sm text-muted-foreground">Band Gap Classification</span>
                        <Badge variant="secondary" className={
                          selected.bandGap === 0 ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-300 border-0" :
                          (selected.bandGap ?? 0) < 2 ? "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 border-0" :
                          "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300 border-0"
                        }>
                          {selected.bandGap === 0 ? "Metal / Superconductor" :
                           (selected.bandGap ?? 0) < 2 ? "Semiconductor" :
                           (selected.bandGap ?? 0) < 4 ? "Wide-Bandgap" : "Insulator"}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono w-16 text-muted-foreground">0 eV</span>
                        <div className="flex-1 h-6 rounded-md overflow-hidden bg-gradient-to-r from-yellow-400 via-blue-400 to-gray-300 relative">
                          {selected.bandGap !== null && (
                            <div
                              className="absolute top-0 bottom-0 w-1 bg-foreground/80"
                              style={{ left: `${Math.min(((selected.bandGap ?? 0) / 15) * 100, 100)}%` }}
                            />
                          )}
                        </div>
                        <span className="text-xs font-mono w-16 text-muted-foreground text-right">15+ eV</span>
                      </div>
                      {selected.bandGap !== null && (
                        <p className="text-xs text-center font-mono text-primary mt-1">{selected.bandGap} eV</p>
                      )}
                    </div>

                    {selectedProps.criticalTemp && (
                      <div className="p-3 rounded-md bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                          <span className="text-sm font-medium text-blue-800 dark:text-blue-200">Superconducting Material</span>
                        </div>
                        <p className="text-sm text-blue-700 dark:text-blue-300 mt-1 font-mono">
                          Critical Temperature: {selectedProps.criticalTemp} K ({(selectedProps.criticalTemp - 273.15).toFixed(1)}°C)
                        </p>
                      </div>
                    )}

                    {selectedProps.ZT && (
                      <div className="p-3 rounded-md bg-orange-50 dark:bg-orange-950/30 border border-orange-200 dark:border-orange-900">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                          <span className="text-sm font-medium text-orange-800 dark:text-orange-200">Thermoelectric Material</span>
                        </div>
                        <p className="text-sm text-orange-700 dark:text-orange-300 mt-1 font-mono">
                          Figure of Merit (ZT): {selectedProps.ZT}
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-80 text-center">
              <Database className="h-12 w-12 text-muted-foreground/30 mb-4" />
              <p className="text-muted-foreground">Select a material to explore its properties</p>
              <p className="text-xs text-muted-foreground mt-2">Browse {materials.length} indexed materials from 4 databases</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
