import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Activity, Atom, Zap, Search, Gauge,
  Layers, CheckCircle2, XCircle, TrendingUp,
} from "lucide-react";

interface OrbitalDOSResponse {
  energyGrid: number[];
  totalDOS: number[];
  s: number[];
  p: number[];
  d: number[];
  f: number[];
  dosAtFermi: number;
  orbitalDOSAtFermi: { s: number; p: number; d: number; f: number };
}

interface VHSEntry {
  energyEv: number;
  binIndex: number;
  dosValue: number;
  relativeToFermi: number;
  type: string;
  dominantOrbital: string;
  strength: number;
}

interface DOSPrediction {
  formula: string;
  orbitalDOS: OrbitalDOSResponse;
  vanHoveSingularities: VHSEntry[];
  scores: {
    vhsScore: number;
    scFavorability: number;
    flatBandIndicator: number;
    nestingScore: number;
    orbitalMixingAtFermi: number;
  };
  isMetallic: boolean;
  predictionTier: string;
  wallTimeMs: number;
  gnnTcPrediction: number;
}

const ORBITAL_COLORS = {
  s: "#3b82f6",
  p: "#10b981",
  d: "#f59e0b",
  f: "#ef4444",
};

const VHS_TYPE_COLORS: Record<string, string> = {
  "M0-onset": "#6366f1",
  "M1-saddle": "#f59e0b",
  "M2-peak": "#ef4444",
  "logarithmic": "#ec4899",
};

function DOSPlot({ dos, vhs }: { dos: OrbitalDOSResponse; vhs: VHSEntry[] }) {
  const width = 600;
  const height = 280;
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const energyGrid = dos.energyGrid;
  const maxDOS = Math.max(...dos.totalDOS, 0.001);
  const minE = energyGrid[0] ?? -5;
  const maxE = energyGrid[energyGrid.length - 1] ?? 5;

  const xScale = (e: number) => margin.left + ((e - minE) / (maxE - minE)) * plotW;
  const yScale = (d: number) => margin.top + plotH - (d / maxDOS) * plotH;

  const makePath = (values: number[]) => {
    return values.map((v, i) => {
      const x = xScale(energyGrid[i]);
      const y = yScale(v);
      return `${i === 0 ? "M" : "L"}${x},${y}`;
    }).join(" ");
  };

  const fermiX = xScale(0);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" data-testid="dos-plot">
      <rect x={margin.left} y={margin.top} width={plotW} height={plotH}
        fill="none" stroke="hsl(var(--border))" strokeWidth="1" />

      <line x1={fermiX} y1={margin.top} x2={fermiX} y2={margin.top + plotH}
        stroke="hsl(var(--destructive))" strokeWidth="1.5" strokeDasharray="4,4" />
      <text x={fermiX + 4} y={margin.top + 12}
        fontSize="10" fill="hsl(var(--destructive))" fontFamily="Open Sans">
        EF
      </text>

      {(["s", "p", "d", "f"] as const).map(orb => (
        <path key={orb} d={makePath(dos[orb])}
          fill="none" stroke={ORBITAL_COLORS[orb]} strokeWidth="1.5" opacity="0.7" />
      ))}

      <path d={makePath(dos.totalDOS)}
        fill="none" stroke="hsl(var(--foreground))" strokeWidth="2" />

      {vhs.map((v, i) => (
        <g key={i}>
          <circle cx={xScale(v.energyEv)} cy={yScale(v.dosValue)}
            r={4 + v.strength * 6} fill={VHS_TYPE_COLORS[v.type] ?? "#888"}
            opacity="0.6" stroke="white" strokeWidth="1" />
          <text x={xScale(v.energyEv)} y={yScale(v.dosValue) - 8}
            fontSize="8" fill={VHS_TYPE_COLORS[v.type] ?? "#888"} textAnchor="middle"
            fontFamily="Open Sans">
            {v.type.split("-")[0]}
          </text>
        </g>
      ))}

      {[-4, -2, 0, 2, 4].map(e => (
        <text key={e} x={xScale(e)} y={margin.top + plotH + 15}
          fontSize="10" fill="hsl(var(--muted-foreground))" textAnchor="middle"
          fontFamily="Open Sans">
          {e} eV
        </text>
      ))}

      <text x={width / 2} y={height - 2}
        fontSize="11" fill="hsl(var(--muted-foreground))" textAnchor="middle"
        fontFamily="Open Sans">
        Energy relative to Fermi level (eV)
      </text>

      <text x={12} y={height / 2}
        fontSize="11" fill="hsl(var(--muted-foreground))" textAnchor="middle"
        fontFamily="Open Sans" transform={`rotate(-90, 12, ${height / 2})`}>
        DOS (a.u.)
      </text>

      <g transform={`translate(${width - 120}, ${margin.top + 5})`}>
        {(["total", "s", "p", "d", "f"] as const).map((label, i) => (
          <g key={label} transform={`translate(0, ${i * 14})`}>
            <line x1="0" y1="5" x2="15" y2="5"
              stroke={label === "total" ? "hsl(var(--foreground))" : ORBITAL_COLORS[label as keyof typeof ORBITAL_COLORS]}
              strokeWidth={label === "total" ? 2 : 1.5} />
            <text x="20" y="8" fontSize="9" fill="hsl(var(--muted-foreground))"
              fontFamily="Open Sans">
              {label}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

export default function DOSVisualizer() {
  const [formula, setFormula] = useState("MgB2");
  const [searchFormula, setSearchFormula] = useState("MgB2");

  const { data, isLoading, error } = useQuery<DOSPrediction>({
    queryKey: ["/api/dos-surrogate/predict", searchFormula],
    queryFn: async () => {
      const res = await fetch(`/api/dos-surrogate/predict/${encodeURIComponent(searchFormula)}`);
      if (!res.ok) throw new Error("Failed to fetch DOS prediction");
      return res.json();
    },
    enabled: !!searchFormula,
  });

  const { data: stats } = useQuery({
    queryKey: ["/api/dos-surrogate/stats"],
  });

  const handleSearch = () => {
    if (formula.trim()) {
      setSearchFormula(formula.trim());
    }
  };

  return (
    <div className="space-y-4">
      <Card data-testid="dos-search-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Orbital-Resolved DOS Surrogate
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Graph-to-Physics surrogate predicts electronic density of states with orbital decomposition
            and Van Hove singularity detection -- pre-filtering candidates before DFT
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Input
              data-testid="input-dos-formula"
              value={formula}
              onChange={(e) => setFormula(e.target.value)}
              placeholder="Enter formula (e.g. MgB2, LaH10, YBa2Cu3O7)"
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="font-mono"
            />
            <Button data-testid="button-dos-predict" onClick={handleSearch} disabled={isLoading}>
              <Search className="h-4 w-4 mr-1" />
              Predict DOS
            </Button>
          </div>

          {!!stats && (
            <div className="flex gap-4 mt-3 text-xs text-muted-foreground">
              <span>{(stats as any).dosBins} energy bins</span>
              <span>{(stats as any).orbitalChannels?.join(", ")} channels</span>
              <span>[{(stats as any).energyRangeEv?.[0]}, {(stats as any).energyRangeEv?.[1]}] eV</span>
            </div>
          )}
        </CardContent>
      </Card>

      {isLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Skeleton className="h-80" />
          <Skeleton className="h-80" />
        </div>
      )}

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-4">
            <p className="text-destructive text-sm" data-testid="text-dos-error">
              Failed to predict DOS for {searchFormula}
            </p>
          </CardContent>
        </Card>
      )}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card data-testid="card-dos-spectrum">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Layers className="h-4 w-4" />
                Predicted eDOS: {data.formula}
              </CardTitle>
              <div className="flex gap-2">
                <Badge variant={data.isMetallic ? "default" : "secondary"} data-testid="badge-metallic">
                  {data.isMetallic ? "Metallic" : "Non-metallic"}
                </Badge>
                <Badge variant="outline" data-testid="badge-tier">
                  {data.predictionTier}
                </Badge>
                <Badge variant="outline" data-testid="badge-walltime">
                  {data.wallTimeMs}ms
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <DOSPlot dos={data.orbitalDOS} vhs={data.vanHoveSingularities} />
            </CardContent>
          </Card>

          <div className="space-y-4">
            <Card data-testid="card-sc-scores">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Gauge className="h-4 w-4" />
                  SC Favorability Scores
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <ScoreBar label="SC Favorability (composite)" value={data.scores.scFavorability} color="#3b82f6" />
                <ScoreBar label="VHS Score" value={data.scores.vhsScore} color="#ef4444" />
                <ScoreBar label="Flat Band Indicator" value={data.scores.flatBandIndicator} color="#f59e0b" />
                <ScoreBar label="Nesting Score" value={data.scores.nestingScore} color="#10b981" />
                <ScoreBar label="Orbital Mixing at EF" value={data.scores.orbitalMixingAtFermi} color="#8b5cf6" />

                <div className="pt-2 border-t">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">DOS at Fermi level</span>
                    <span className="font-mono" data-testid="text-dos-at-fermi">
                      {data.orbitalDOS.dosAtFermi.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">GNN Tc prediction</span>
                    <span className="font-mono" data-testid="text-gnn-tc">
                      {data.gnnTcPrediction != null ? `${data.gnnTcPrediction.toFixed(1)} K` : "N/A"}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card data-testid="card-orbital-fermi">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Atom className="h-4 w-4" />
                  Orbital DOS at Fermi Level
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-2">
                  {(["s", "p", "d", "f"] as const).map(orb => {
                    const val = data.orbitalDOS.orbitalDOSAtFermi[orb];
                    const total = data.orbitalDOS.dosAtFermi || 0.001;
                    const pct = (val / total * 100).toFixed(1);
                    return (
                      <div key={orb} className="text-center p-2 rounded-lg bg-muted" data-testid={`orbital-${orb}`}>
                        <div className="text-lg font-bold font-mono" style={{ color: ORBITAL_COLORS[orb] }}>
                          {orb}
                        </div>
                        <div className="text-xs text-muted-foreground">{val.toFixed(3)}</div>
                        <div className="text-xs font-medium">{pct}%</div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>

          {data.vanHoveSingularities.length > 0 && (
            <Card className="lg:col-span-2" data-testid="card-vhs-list">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  Van Hove Singularities ({data.vanHoveSingularities.length})
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  DOS peaks/divergences near the Fermi level that enhance electron-phonon coupling
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                  {data.vanHoveSingularities.map((vhs, i) => (
                    <div key={i}
                      className="flex items-center justify-between p-2 rounded-lg border"
                      data-testid={`vhs-entry-${i}`}>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: VHS_TYPE_COLORS[vhs.type] ?? "#888" }} />
                        <div>
                          <div className="text-sm font-mono">{vhs.energyEv.toFixed(2)} eV</div>
                          <div className="text-xs text-muted-foreground">{vhs.type}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant="outline" className="text-xs font-mono"
                          style={{ color: ORBITAL_COLORS[vhs.dominantOrbital as keyof typeof ORBITAL_COLORS] }}>
                          {vhs.dominantOrbital}-orbital
                        </Badge>
                        <div className="text-xs text-muted-foreground mt-1">
                          str: {(vhs.strength * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
