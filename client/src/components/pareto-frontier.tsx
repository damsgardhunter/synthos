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

interface PlotPoint {
  formula: string;
  tcK: number;
  stability: number;
  synthesizability: number;
  rank: number;
  isFront: boolean;
  isTcProxy: boolean;
}

/** Map synthesizability [0,1] → a CSS hsl colour green→yellow→red */
function synthColor(s: number): string {
  const hue = Math.round(s * 120); // 0=red, 120=green
  return `hsl(${hue}, 70%, 50%)`;
}

const RANK_COLORS: Record<number, string> = {
  1: "#22c55e", // green-500
  2: "#f59e0b", // amber-500
  3: "#6366f1", // indigo-400
};
function rankColor(rank: number): string {
  return RANK_COLORS[rank] ?? "#6b7280";
}

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

  // Split into front (rank-1) and others for layering
  const frontPoints = points.filter(p => p.isFront);
  const otherPoints = points.filter(p => !p.isFront);

  const lastUpdated = data?.lastRecomputedAt
    ? new Date(data.lastRecomputedAt).toLocaleTimeString()
    : null;

  return (
    <Card data-testid="pareto-frontier-card">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Pareto Frontier
          </CardTitle>
          <div className="flex items-center gap-2">
            {data?.rank1Count != null && (
              <Badge variant="secondary" className="text-xs border-0 bg-green-500/15 text-green-600 dark:text-green-400">
                {data.rank1Count} rank-1
              </Badge>
            )}
            {data?.totalCandidates != null && (
              <Badge variant="outline" className="text-xs">
                {data.totalCandidates} total
              </Badge>
            )}
          </div>
        </div>
        <p className="text-[11px] text-muted-foreground mt-1">
          Tc vs stability — colour = synthesizability. Pareto front (rank 1) shown in green.
          {lastUpdated && <span className="ml-2">Updated {lastUpdated}</span>}
        </p>
      </CardHeader>
      <CardContent className="pt-0">
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
                {/* Ideal corner reference */}
                <ReferenceLine x={293} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 2" strokeOpacity={0.5} label={{ value: "RT", fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                {/* Rank 2-5 candidates */}
                <Scatter name="Other ranks" data={otherPoints} isAnimationActive={false}>
                  {otherPoints.map((p, i) => (
                    <Cell
                      key={i}
                      fill={synthColor(p.synthesizability)}
                      fillOpacity={0.55}
                      stroke={rankColor(p.rank)}
                      strokeWidth={0.8}
                      r={4}
                    />
                  ))}
                </Scatter>
                {/* Rank-1 Pareto front on top */}
                <Scatter name="Pareto front" data={frontPoints} isAnimationActive={false}>
                  {frontPoints.map((p, i) => (
                    <Cell
                      key={i}
                      fill={synthColor(p.synthesizability)}
                      fillOpacity={0.9}
                      stroke="#22c55e"
                      strokeWidth={1.5}
                      r={6}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            {/* Legend */}
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
