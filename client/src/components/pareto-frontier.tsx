import { useQuery } from "@tanstack/react-query";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Target } from "lucide-react";

interface ParetoObjectives {
  tc: number;
  stability: number;
  synthesizability: number;
  tcIsProxy?: boolean;
}

interface ParetoResult {
  formula: string;
  rank: number;
  objectives: ParetoObjectives;
  isFront: boolean;
}

interface ParetoFrontierData {
  results: ParetoResult[];
  rank1Count: number;
  totalCandidates: number;
  lastRecomputedAt: number;
}

interface LastCycleStats {
  totalCandidates: number;
  passed: number;
  bestTc: number;
  bestFormula: string;
  familyCounts: Record<string, number>;
  candidates: { formula: string; tc: number; passed: boolean; reason: string; family: string }[];
}

interface PlotPoint {
  formula: string;
  tcK: number;
  stability: number;
  synthesizability: number;
  rank: number;
  isFront: boolean;
  isTcProxy: boolean;
}

function synthColor(s: number): string {
  const hue = Math.round(s * 120);
  return `hsl(${hue}, 70%, 50%)`;
}

const RANK_COLORS: Record<number, string> = {
  1: "#22c55e",
  2: "#f59e0b",
  3: "#6366f1",
};
function rankColor(rank: number): string {
  return RANK_COLORS[rank] ?? "#6b7280";
}

const FAMILY_COLORS: Record<string, string> = {
  "Hydrides": "bg-red-500/20 text-red-400 border-red-500/30",
  "Cuprates": "bg-blue-500/20 text-blue-400 border-blue-500/30",
  "Pnictides": "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
  "Chalcogenides": "bg-amber-500/20 text-amber-400 border-amber-500/30",
  "Borides": "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  "Carbides": "bg-teal-500/20 text-teal-400 border-teal-500/30",
  "Nitrides": "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  "Oxides": "bg-orange-500/20 text-orange-400 border-orange-500/30",
  "Intermetallics": "bg-purple-500/20 text-purple-400 border-purple-500/30",
  "Other": "bg-gray-500/20 text-gray-400 border-gray-500/30",
};

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d: PlotPoint = payload[0].payload;
  return (
    <div className="bg-popover border border-border rounded-md px-3 py-2 text-xs shadow-md space-y-1">
      <p className="font-semibold font-mono text-foreground">{d.formula}</p>
      <p className="text-muted-foreground">
        Tc: <span className="text-foreground font-mono">{(d.tcK).toFixed(0)} K</span>
        {d.isTcProxy && <span className="ml-1 text-amber-500">(ML proxy)</span>}
      </p>
      <p className="text-muted-foreground">Stability: <span className="text-foreground font-mono">{(d.stability * 100).toFixed(0)}%</span></p>
      <p className="text-muted-foreground">Synthesizability: <span className="text-foreground font-mono">{(d.synthesizability * 100).toFixed(0)}%</span></p>
      <p className="text-muted-foreground">Pareto rank: <span className="font-mono" style={{ color: rankColor(d.rank) }}>{d.rank}</span></p>
    </div>
  );
}

export function ParetoFrontierChart() {
  const { data, isLoading } = useQuery<ParetoFrontierData>({
    queryKey: ["/api/pareto-frontier"],
    refetchInterval: 60_000,
  });

  const { data: cycleStats } = useQuery<LastCycleStats>({
    queryKey: ["/api/last-cycle-stats"],
    refetchInterval: 30_000,
  });

  const points: PlotPoint[] = (data?.results ?? [])
    .filter(r => r.rank <= 5)
    .map(r => ({
      formula: r.formula,
      tcK: r.objectives.tc * 400,
      stability: r.objectives.stability,
      synthesizability: r.objectives.synthesizability,
      rank: r.rank,
      isFront: r.isFront,
      isTcProxy: r.objectives.tcIsProxy ?? false,
    }));

  const frontPoints = points.filter(p => p.isFront);
  const otherPoints = points.filter(p => !p.isFront);

  const sortedFamilies = Object.entries(cycleStats?.familyCounts ?? {})
    .sort((a, b) => b[1] - a[1]);

  return (
    <Card data-testid="pareto-frontier-card" className="border-[hsl(var(--gold)/0.2)]">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="h-4 w-4 text-[hsl(var(--gold))]" />
            Knowledge Map
          </CardTitle>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="h-2.5 w-2.5 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs font-medium text-green-400">LIVE</span>
              <span className="text-xs text-muted-foreground">Heatmap</span>
            </div>
            {data?.rank1Count != null && (
              <Badge variant="secondary" className="text-xs border-0 bg-green-500/15 text-green-600 dark:text-green-400">
                {data.rank1Count} rank-1
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0 space-y-4">
        {/* Last Cycle Summary */}
        {cycleStats && cycleStats.totalCandidates > 0 && (
          <div className="rounded-lg border border-[hsl(var(--gold)/0.2)] bg-[hsl(var(--gold)/0.03)] p-3 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-[hsl(var(--gold))] animate-pulse" />
                <span className="text-xs font-semibold text-[hsl(var(--gold))] uppercase tracking-wider">Last Cycle</span>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-muted-foreground">
                  <span className="font-mono font-bold text-foreground">{cycleStats.totalCandidates}</span> screened
                </span>
                <span className="text-muted-foreground">
                  <span className="font-mono font-bold text-green-400">{cycleStats.passed}</span> passed
                </span>
                {cycleStats.bestTc > 0 && (
                  <span className="text-muted-foreground">
                    Best: <span className="font-mono font-bold text-[hsl(var(--gold))]">{cycleStats.bestTc}K</span>
                    {cycleStats.bestFormula && <span className="ml-1 font-mono text-foreground">{cycleStats.bestFormula}</span>}
                  </span>
                )}
              </div>
            </div>

            {sortedFamilies.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {sortedFamilies.map(([family, count]) => (
                  <Badge
                    key={family}
                    variant="outline"
                    className={`text-[10px] ${FAMILY_COLORS[family] ?? FAMILY_COLORS["Other"]}`}
                  >
                    {family}: {count}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        )}

        {cycleStats && cycleStats.totalCandidates === 0 && (
          <div className="rounded-lg border border-[hsl(var(--gold)/0.1)] bg-[hsl(var(--gold)/0.02)] p-3 text-center">
            <span className="text-xs text-muted-foreground">Waiting for engine cycle to complete...</span>
          </div>
        )}

        {/* Chart */}
        {isLoading ? (
          <Skeleton className="h-56 w-full" />
        ) : points.length === 0 ? (
          <div className="h-56 flex items-center justify-center">
            <p className="text-sm text-muted-foreground italic">
              Pareto frontier will appear after DFT cycles complete
            </p>
          </div>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart margin={{ top: 8, right: 8, bottom: 20, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" />
                <XAxis
                  type="number"
                  dataKey="tcK"
                  name="Tc (K)"
                  domain={[0, "auto"]}
                  label={{ value: "Tc (K)", position: "insideBottom", offset: -12, fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                />
                <YAxis
                  type="number"
                  dataKey="stability"
                  name="Stability"
                  domain={[0, 1]}
                  tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  width={36}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine x={293} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 2" strokeOpacity={0.5} label={{ value: "RT", fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                <Scatter name="Other ranks" data={otherPoints} isAnimationActive={false}>
                  {otherPoints.map((p, i) => (
                    <Cell key={i} fill={synthColor(p.synthesizability)} fillOpacity={0.55} stroke={rankColor(p.rank)} strokeWidth={0.8} r={4} />
                  ))}
                </Scatter>
                <Scatter name="Pareto front" data={frontPoints} isAnimationActive={false}>
                  {frontPoints.map((p, i) => (
                    <Cell key={i} fill={synthColor(p.synthesizability)} fillOpacity={0.9} stroke="#22c55e" strokeWidth={1.5} r={6} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            <div className="flex items-center justify-center gap-4 mt-1 flex-wrap">
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <span className="inline-block w-3 h-3 rounded-full border-2 border-green-500 bg-green-500/40" />
                Rank 1 (Pareto front)
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <span className="inline-block w-3 h-3 rounded-full border border-amber-500 bg-amber-500/40" />
                Rank 2–3
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <span className="inline-block w-8 h-2.5 rounded" style={{ background: "linear-gradient(to right, hsl(0,70%,50%), hsl(60,70%,50%), hsl(120,70%,50%))" }} />
                Synthesizability (low→high)
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
