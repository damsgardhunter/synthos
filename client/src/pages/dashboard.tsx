import { useQuery } from "@tanstack/react-query";
import { useEffect, useState, useRef, useMemo } from "react";
import { queryClient } from "@/lib/queryClient";
import type { LearningPhase, ResearchLog, ResearchStrategy } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Atom, Database, FlaskConical, Brain,
  TrendingUp,
  Zap, BookOpen, Microscope, BarChart3, FileText,
  Compass, RefreshCw, Star, ChevronDown, ChevronUp,
  MessageSquare, Lightbulb, AlertTriangle, Trophy, Archive,
  Cpu, Shield, Activity, Network, GitMerge, GitBranch, Layers, Mountain, Shuffle,
} from "lucide-react";
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip, LineChart, Line, AreaChart, Area } from "recharts";
import { useWebSocket, type ThoughtMessage } from "@/hooks/use-websocket";
import { EngineControls } from "@/components/engine-controls";

interface Stats {
  elementsLearned: number;
  materialsIndexed: number;
  predictionsGenerated: number;
  overallProgress: number;
  synthesisProcesses: number;
  chemicalReactions: number;
  superconductorCandidates: number;
}

interface ActiveLearningStats {
  totalDFTRuns: number;
  avgUncertaintyBefore: number;
  avgUncertaintyAfter: number;
  modelRetrains: number;
  bestTcFromLoop: number;
}

interface AutonomousLoopStats {
  totalScreened: number;
  totalPassed: number;
  passRate: number;
  bestTc: number;
  throughputPerHour: number;
  gnnRetrainCount: number;
  activeLearning: ActiveLearningStats;
  inverseOptimizer?: {
    bestTcAcrossAll: number;
    activeCampaigns: number;
    totalCandidatesGenerated: number;
    totalPassed: number;
  };
}

interface EngineMemory {
  currentHypothesis: { family: string; priority: number; reasoning: string } | null;
  familyStats: Record<string, any>;
  topInsights: { text: string; noveltyScore: number; category: string; discoveredAt: string }[];
  abandonedStrategies: string[];
  milestoneCount: number;
  recentMilestones: any[];
  totalCycles: number;
  bestTc: number;
  bestScore: number;
  familyDiversity: number;
  pipelinePassRate: number;
  cycleNarratives: { detail: string; timestamp: string }[];
  autonomousLoopStats?: AutonomousLoopStats;
}

function MiniSparkline({ data }: { data: number[] }) {
  if (data.length < 2) return null;
  const chartData = data.map((v, i) => ({ v, i }));
  return (
    <div className="w-16 h-6" data-testid="sparkline">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <Line type="monotone" dataKey="v" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function StatCard({ title, value, icon: Icon, sub, history }: { title: string; value: string | number; icon: any; sub?: string; history?: number[] }) {
  const delta = history && history.length >= 2 ? (history[history.length - 1] - history[0]) : 0;
  return (
    <Card data-testid={`stat-card-${title.toLowerCase().replace(/ /g, "-")}`}>
      <CardHeader className="flex flex-row items-center justify-between gap-1 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <Icon className="h-4 w-4 text-primary" />
      </CardHeader>
      <CardContent>
        <div className="flex items-end justify-between gap-2">
          <div>
            <div className="text-2xl font-bold font-mono text-foreground">{value}</div>
            {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
          </div>
          <div className="flex flex-col items-end gap-0.5">
            {history && <MiniSparkline data={history} />}
            {delta !== 0 && (
              <span className={`text-[10px] font-mono ${delta > 0 ? "text-green-600 dark:text-green-400" : "text-red-500"}`} data-testid="stat-delta">
                {delta > 0 ? "+" : ""}{typeof delta === "number" && delta % 1 !== 0 ? delta.toFixed(1) : delta} this session
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

const FAMILY_COLORS: Record<string, string> = {
  "Hydrides": "bg-red-500",
  "Cuprates": "bg-blue-500",
  "Pnictides": "bg-indigo-500",
  "Chalcogenides": "bg-amber-500",
  "Borides": "bg-emerald-500",
  "Carbides": "bg-teal-500",
  "Nitrides": "bg-cyan-500",
  "Oxides": "bg-orange-500",
  "Intermetallics": "bg-purple-500",
  "Other": "bg-gray-400",
};

interface FocusArea {
  area: string;
  priority: number;
  reasoning: string;
}

const THOUGHT_STYLES: Record<string, { color: string; icon: any }> = {
  strategy: { color: "text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-950/30", icon: Compass },
  discovery: { color: "text-green-600 dark:text-green-400 border-green-200 dark:border-green-800 bg-green-50/50 dark:bg-green-950/30", icon: Lightbulb },
  stagnation: { color: "text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800 bg-amber-50/50 dark:bg-amber-950/30", icon: AlertTriangle },
  milestone: { color: "text-purple-600 dark:text-purple-400 border-purple-200 dark:border-purple-800 bg-purple-50/50 dark:bg-purple-950/30", icon: Trophy },
};

function ThoughtFeed({ thoughts, tempo }: { thoughts: ThoughtMessage[]; tempo: string }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const displayed = thoughts.slice(-15);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [displayed.length]);

  const tempoLabel = tempo === "excited" ? "Excited" : tempo === "contemplating" ? "Contemplating" : "Exploring";
  const tempoDotColor = tempo === "excited" ? "bg-green-500" : tempo === "contemplating" ? "bg-amber-500" : "bg-blue-500";
  const pulseClass = tempo === "excited" ? "animate-pulse" : tempo === "contemplating" ? "animate-[pulse_3s_ease-in-out_infinite]" : "animate-pulse";

  return (
    <Card data-testid="thought-feed">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            Engine Thoughts
          </CardTitle>
          <div className="flex items-center gap-1.5">
            <div className={`h-2 w-2 rounded-full ${tempoDotColor} ${pulseClass}`} />
            <span className="text-[10px] text-muted-foreground font-medium">{tempoLabel}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div ref={scrollRef} className="h-64 overflow-y-auto px-4 pb-4">
          {displayed.length === 0 ? (
            <p className="text-sm text-muted-foreground italic py-4" data-testid="thought-placeholder">
              Engine thoughts will appear once cycles begin running
            </p>
          ) : (
            <div className="space-y-2">
              {displayed.map((thought, i) => {
                const style = THOUGHT_STYLES[thought.category] ?? THOUGHT_STYLES.strategy;
                const ThIcon = style.icon;
                const opacity = i < displayed.length - 5 ? Math.max(0.4, 0.4 + (i / displayed.length) * 0.6) : 1;
                return (
                  <div
                    key={`${thought.timestamp}-${i}`}
                    className={`rounded-md border px-3 py-2 ${style.color} transition-opacity duration-300`}
                    style={{ opacity, animation: i === displayed.length - 1 ? "fadeIn 0.5s ease-in" : undefined }}
                    data-testid={`thought-${thought.category}-${i}`}
                  >
                    <div className="flex items-start gap-2">
                      <ThIcon className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs leading-relaxed">{thought.text}</p>
                        <span className="text-[9px] opacity-60 font-mono">
                          {new Date(thought.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function StrategyCard() {
  const { data: strategy, isLoading } = useQuery<ResearchStrategy | null>({
    queryKey: ["/api/research-strategy"],
  });
  const { data: history } = useQuery<ResearchStrategy[]>({
    queryKey: ["/api/research-strategy/history"],
  });

  const evolutionCount = history?.length ?? 0;
  const focusAreas = (strategy?.focusAreas as FocusArea[] | undefined) ?? [];

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Compass className="h-4 w-4 text-primary" />
            Research Strategy
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-32" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="strategy-card">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Compass className="h-4 w-4 text-primary" />
            Research Strategy
          </CardTitle>
          {evolutionCount > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground" data-testid="strategy-evolution-count">
              <RefreshCw className="h-3 w-3" />
              Strategy evolved {evolutionCount} time{evolutionCount !== 1 ? "s" : ""}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {!strategy ? (
          <p className="text-sm text-muted-foreground italic" data-testid="strategy-placeholder">
            Strategy will evolve once the engine begins learning
          </p>
        ) : (
          <div className="space-y-4">
            <div className="space-y-2.5" data-testid="strategy-focus-areas">
              {focusAreas.map((fa, i) => (
                <div key={fa.area} className="space-y-1" data-testid={`focus-area-${i}`}>
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold text-muted-foreground w-4">#{i + 1}</span>
                      <span className="text-sm font-medium">{fa.area}</span>
                    </div>
                    <span className="text-xs font-mono text-muted-foreground">
                      {(fa.priority * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${FAMILY_COLORS[fa.area] ?? "bg-primary"}`}
                      style={{ width: `${Math.max(5, fa.priority * 100)}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-muted-foreground pl-6 leading-relaxed">{fa.reasoning}</p>
                </div>
              ))}
            </div>
            {strategy.summary && (
              <p className="text-xs text-muted-foreground leading-relaxed border-t border-border pt-3" data-testid="strategy-summary">
                {strategy.summary}
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ResearchMemoryCard() {
  const [expanded, setExpanded] = useState(false);
  const { data: memory, isLoading } = useQuery<EngineMemory>({
    queryKey: ["/api/engine/memory"],
    refetchInterval: 30000,
  });

  if (isLoading) return <Card><CardContent className="p-6"><Skeleton className="h-20" /></CardContent></Card>;
  if (!memory) return null;

  const familyEntries = Object.entries(memory.familyStats).sort((a: any, b: any) => (b[1]?.count ?? 0) - (a[1]?.count ?? 0));
  const maxCount = familyEntries.length > 0 ? Math.max(...familyEntries.map((e: any) => e[1]?.count ?? 0), 1) : 1;

  return (
    <Card data-testid="research-memory-card">
      <CardHeader className="pb-2">
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setExpanded(!expanded)}
          data-testid="toggle-memory-card"
        >
          <CardTitle className="text-base flex items-center gap-2">
            <Archive className="h-4 w-4 text-primary" />
            Research Memory
          </CardTitle>
          {expanded ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
        </button>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {memory.currentHypothesis && (
            <div data-testid="current-hypothesis">
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Current Hypothesis</p>
              <p className="text-xs text-foreground">
                Focusing on <span className="font-semibold">{memory.currentHypothesis.family}</span> (priority: {(memory.currentHypothesis.priority * 100).toFixed(0)}%)
              </p>
              {memory.currentHypothesis.reasoning && (
                <p className="text-[10px] text-muted-foreground mt-0.5">{memory.currentHypothesis.reasoning}</p>
              )}
            </div>
          )}

          <div className="flex gap-4 text-xs">
            <div data-testid="memory-cycles"><span className="text-muted-foreground">Cycles:</span> <span className="font-mono font-medium">{memory.totalCycles}</span></div>
            <div data-testid="memory-best-tc"><span className="text-muted-foreground">Best Tc:</span> <span className="font-mono font-medium">{Math.round(memory.bestTc)}K</span></div>
            <div data-testid="memory-milestones"><span className="text-muted-foreground">Milestones:</span> <span className="font-mono font-medium">{memory.milestoneCount}</span></div>
          </div>

          {expanded && (
            <div className="space-y-3 pt-2 border-t border-border">
              {memory.topInsights.length > 0 && (
                <div data-testid="top-insights">
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Key Discoveries ({memory.topInsights.length})</p>
                  <div className="space-y-1.5">
                    {memory.topInsights.slice(0, 3).map((ins, i) => (
                      <div key={i} className="text-[11px] text-muted-foreground bg-muted/50 rounded px-2 py-1.5 leading-relaxed" data-testid={`insight-${i}`}>
                        <span className="font-mono text-[9px] text-primary mr-1">[{(ins.noveltyScore * 100).toFixed(0)}%]</span>
                        {ins.text}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {familyEntries.length > 0 && (
                <div data-testid="explored-territory">
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Explored Territory</p>
                  <div className="space-y-1">
                    {familyEntries.slice(0, 6).map(([fam, stats]: [string, any]) => (
                      <div key={fam} className="flex items-center gap-2">
                        <span className="text-[10px] text-muted-foreground w-20 truncate">{fam}</span>
                        <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${FAMILY_COLORS[fam] ?? "bg-primary"}`}
                            style={{ width: `${Math.max(5, ((stats?.count ?? 0) / maxCount) * 100)}%` }}
                          />
                        </div>
                        <span className="text-[10px] font-mono text-muted-foreground w-6 text-right">{stats?.count ?? 0}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {memory.abandonedStrategies.length > 0 && (
                <div data-testid="abandoned-strategies">
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Abandoned Paths</p>
                  <div className="flex flex-wrap gap-1">
                    {memory.abandonedStrategies.map(s => (
                      <Badge key={s} variant="secondary" className="text-[10px] bg-muted text-muted-foreground">{s}</Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function GNNActiveLearningCard() {
  const { data: gnnVersionData } = useQuery<{
    currentVersion: number;
    ensembleSize: number;
    latestMetrics: { r2: number; mae: number; rmse: number; datasetSize: number } | null;
    r2Trend: { version: number; r2: number }[];
    maeTrend: { version: number; mae: number }[];
    history: any[];
    uncertaintyMethods: string[];
  }>({ queryKey: ["/api/gnn/version-history"], refetchInterval: 30000 });

  const { data: alStats } = useQuery<{
    convergence: ActiveLearningStats;
    totalCycles: number;
    recentCycles: any[];
    avgUncertaintyTrend: { cycle: number; before: number; after: number; reductionPct: number }[];
    dftDatasetStats: { totalSize: number; bySource: Record<string, number>; growthHistory: { timestamp: number; size: number; source: string }[] };
  }>({ queryKey: ["/api/gnn/active-learning-stats"], refetchInterval: 30000 });

  const growthData = useMemo(() => {
    if (!alStats?.dftDatasetStats?.growthHistory?.length) return [];
    return alStats.dftDatasetStats.growthHistory.map(g => ({
      time: new Date(g.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      size: g.size,
    }));
  }, [alStats?.dftDatasetStats?.growthHistory]);

  return (
    <Card data-testid="card-gnn-active-learning">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Network className="h-4 w-4 text-primary" />
            GNN Active Learning
          </CardTitle>
          {gnnVersionData && (
            <div className="flex gap-1">
              {(gnnVersionData.uncertaintyMethods ?? []).map(m => (
                <Badge key={m} variant="secondary" className="text-[9px] px-1.5 py-0 border-0">{m.replace(/-/g, " ")}</Badge>
              ))}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {gnnVersionData?.latestMetrics ? (
            <div className="grid grid-cols-3 gap-2">
              <div className="p-2 bg-purple-50/50 dark:bg-purple-950/20 rounded-md border border-purple-200/50 dark:border-purple-800/30" data-testid="gnn-version">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">GNN v{gnnVersionData.currentVersion}</p>
                <p className="text-sm font-mono font-bold">{gnnVersionData.ensembleSize}-model</p>
              </div>
              <div className="p-2 bg-purple-50/50 dark:bg-purple-950/20 rounded-md border border-purple-200/50 dark:border-purple-800/30" data-testid="gnn-r2">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">R-squared</p>
                <p className="text-sm font-mono font-bold">{gnnVersionData.latestMetrics.r2 != null ? gnnVersionData.latestMetrics.r2.toFixed(4) : "--"}</p>
              </div>
              <div className="p-2 bg-purple-50/50 dark:bg-purple-950/20 rounded-md border border-purple-200/50 dark:border-purple-800/30" data-testid="gnn-mae">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">MAE</p>
                <p className="text-sm font-mono font-bold">{gnnVersionData.latestMetrics.mae != null ? `${gnnVersionData.latestMetrics.mae.toFixed(2)}K` : "--"}</p>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground italic" data-testid="gnn-no-data">GNN model initializing...</p>
          )}

          {alStats && (
            <div className="space-y-2">
              <div className="flex flex-wrap gap-3 text-xs">
                <div data-testid="al-total-dft"><span className="text-muted-foreground">DFT Runs:</span> <span className="font-mono font-medium">{alStats.convergence.totalDFTRuns}</span></div>
                <div data-testid="al-retrains"><span className="text-muted-foreground">Retrains:</span> <span className="font-mono font-medium">{alStats.convergence.modelRetrains}</span></div>
                <div data-testid="al-best-tc"><span className="text-muted-foreground">Best Tc:</span> <span className="font-mono font-medium">{(alStats.convergence.bestTcFromLoop ?? 0).toFixed(1)}K</span></div>
                <div data-testid="al-dataset"><span className="text-muted-foreground">Dataset:</span> <span className="font-mono font-medium">{gnnVersionData?.latestMetrics?.datasetSize ?? 0}</span></div>
              </div>

              {alStats.dftDatasetStats.totalSize > 0 && (
                <div className="text-xs" data-testid="al-dft-dataset">
                  <span className="text-muted-foreground">DFT Training:</span>{" "}
                  <span className="font-mono font-medium">{alStats.dftDatasetStats.totalSize} samples</span>
                  {Object.entries(alStats.dftDatasetStats.bySource).length > 0 && (
                    <span className="text-muted-foreground ml-1">
                      ({Object.entries(alStats.dftDatasetStats.bySource).map(([src, count]) => `${count} ${src}`).join(", ")})
                    </span>
                  )}
                </div>
              )}

              {alStats.totalCycles > 0 && (
                <div className="text-xs" data-testid="al-cycles">
                  <span className="text-muted-foreground">AL Cycles:</span>{" "}
                  <span className="font-mono font-medium">{alStats.totalCycles}</span>
                  {alStats.avgUncertaintyTrend.length > 0 && (
                    <span className="text-muted-foreground ml-1">
                      (unc. reduction: {alStats.avgUncertaintyTrend[alStats.avgUncertaintyTrend.length - 1]?.reductionPct?.toFixed(1) ?? "0"}%)
                    </span>
                  )}
                </div>
              )}
            </div>
          )}

          {growthData.length > 1 && (
            <div data-testid="gnn-dataset-growth">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">DFT Dataset Growth</p>
              <ResponsiveContainer width="100%" height={50}>
                <AreaChart data={growthData}>
                  <Area type="monotone" dataKey="size" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.15} strokeWidth={1.5} />
                  <Tooltip
                    contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "11px" }}
                    formatter={(v: any) => [v, "Samples"]}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {gnnVersionData?.r2Trend && gnnVersionData.r2Trend.filter(d => d.r2 != null).length > 1 && (
            <div data-testid="gnn-r2-trend">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">R-squared Trend</p>
              <ResponsiveContainer width="100%" height={50}>
                <LineChart data={gnnVersionData.r2Trend.filter(d => d.r2 != null)}>
                  <Line type="monotone" dataKey="r2" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} />
                  <Tooltip
                    contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "11px" }}
                    formatter={(v: any) => [v != null ? v.toFixed(4) : "--", "R-squared"]}
                    labelFormatter={(l: any) => `v${l}`}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

interface BenchmarkCompound {
  formula: string;
  name: string;
  family: string;
  textbook: {
    tc: number;
    lambda: number;
    omegaLog: number;
    crystalSystem: string;
    spaceGroup: string;
    yearDiscovered: number;
    pressureGpa: number;
    pairingMechanism: string;
    notes: string;
  };
  predicted: {
    xgboostTc: number;
    xgboostScore: number;
    gnnTc: number;
    gnnUncertainty: number;
    gnnLambda: number;
    ensembleTc: number;
    reasoning: string[];
  };
  accuracy: {
    tcErrorK: number;
    tcErrorPercent: number;
    lambdaError: number | null;
    rating: string;
  };
  computedAt: number;
}

function ReferenceBenchmarkCard() {
  const { data, isLoading } = useQuery<{ results: BenchmarkCompound[]; computedAt: number }>({
    queryKey: ["/api/reference-benchmark"],
    refetchInterval: 60000,
  });

  const ratingColor = (rating: string) => {
    if (rating === "excellent") return "text-emerald-600 dark:text-emerald-400";
    if (rating === "good") return "text-blue-600 dark:text-blue-400";
    if (rating === "fair") return "text-amber-600 dark:text-amber-400";
    return "text-red-600 dark:text-red-400";
  };

  const ratingBg = (rating: string) => {
    if (rating === "excellent") return "bg-emerald-500/10 border-emerald-500/20";
    if (rating === "good") return "bg-blue-500/10 border-blue-500/20";
    if (rating === "fair") return "bg-amber-500/10 border-amber-500/20";
    return "bg-red-500/10 border-red-500/20";
  };

  const familyBadge = (family: string) => {
    const colors: Record<string, string> = {
      Boride: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
      Cuprate: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
      A15: "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
    };
    return colors[family] ?? "bg-muted text-muted-foreground";
  };

  if (isLoading) {
    return (
      <Card data-testid="card-reference-benchmark">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Shield className="h-4 w-4 text-primary" />
            Reference Benchmark
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1, 2, 3].map(i => <Skeleton key={i} className="h-28" />)}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data?.results?.length) return null;

  const avgError = data.results.reduce((s, r) => s + r.accuracy.tcErrorPercent, 0) / data.results.length;

  return (
    <Card data-testid="card-reference-benchmark" className="col-span-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Shield className="h-4 w-4 text-primary" />
            Reference Benchmark: Known Superconductors
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs border-0 font-mono" data-testid="benchmark-avg-error">
              Avg Error: {avgError.toFixed(1)}%
            </Badge>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Comparing our physics engine predictions against experimentally verified textbook data
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {data.results.map((compound) => (
            <div
              key={compound.formula}
              className={`rounded-lg border p-4 space-y-3 ${ratingBg(compound.accuracy.rating)}`}
              data-testid={`benchmark-compound-${compound.formula}`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-bold text-base font-mono" data-testid={`benchmark-formula-${compound.formula}`}>
                    {compound.formula}
                  </h3>
                  <p className="text-xs text-muted-foreground">{compound.name}</p>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <Badge className={`text-[10px] ${familyBadge(compound.family)}`} data-testid={`benchmark-family-${compound.formula}`}>
                    {compound.family}
                  </Badge>
                  <span className="text-[10px] text-muted-foreground">{compound.textbook.yearDiscovered}</span>
                </div>
              </div>

              <div className="space-y-1.5">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Textbook Data</p>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                  <div className="flex justify-between" data-testid={`benchmark-textbook-tc-${compound.formula}`}>
                    <span className="text-muted-foreground">Tc</span>
                    <span className="font-mono font-semibold">{compound.textbook.tc} K</span>
                  </div>
                  <div className="flex justify-between" data-testid={`benchmark-textbook-lambda-${compound.formula}`}>
                    <span className="text-muted-foreground">Lambda</span>
                    <span className="font-mono font-semibold">{compound.textbook.lambda}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Omega_log</span>
                    <span className="font-mono">{compound.textbook.omegaLog} cm-1</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Structure</span>
                    <span className="font-mono text-[10px]">{compound.textbook.spaceGroup}</span>
                  </div>
                </div>
                <p className="text-[10px] text-muted-foreground italic">{compound.textbook.pairingMechanism}</p>
              </div>

              <div className="space-y-1.5 pt-2 border-t border-border/50">
                <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Our Prediction</p>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                  <div className="flex justify-between" data-testid={`benchmark-pred-ensemble-${compound.formula}`}>
                    <span className="text-muted-foreground">Ensemble Tc</span>
                    <span className="font-mono font-semibold">{compound.predicted.ensembleTc} K</span>
                  </div>
                  <div className="flex justify-between" data-testid={`benchmark-pred-lambda-${compound.formula}`}>
                    <span className="text-muted-foreground">GNN Lambda</span>
                    <span className="font-mono font-semibold">{compound.predicted.gnnLambda}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">XGBoost Tc</span>
                    <span className="font-mono">{compound.predicted.xgboostTc} K</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">GNN Tc</span>
                    <span className="font-mono">{compound.predicted.gnnTc} K</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Uncertainty</span>
                    <span className="font-mono text-muted-foreground">+/-{compound.predicted.gnnUncertainty} K</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Score</span>
                    <span className="font-mono">{compound.predicted.xgboostScore}</span>
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-border/50">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Accuracy</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className={`text-sm font-bold font-mono ${ratingColor(compound.accuracy.rating)}`} data-testid={`benchmark-rating-${compound.formula}`}>
                        {compound.accuracy.rating.toUpperCase()}
                      </span>
                      <span className="text-xs text-muted-foreground font-mono" data-testid={`benchmark-error-${compound.formula}`}>
                        {compound.accuracy.tcErrorK} K off ({compound.accuracy.tcErrorPercent}%)
                      </span>
                    </div>
                  </div>
                  {compound.accuracy.lambdaError !== null && (
                    <div className="text-right">
                      <p className="text-[10px] text-muted-foreground">Lambda Error</p>
                      <span className="text-xs font-mono">{compound.accuracy.lambdaError}</span>
                    </div>
                  )}
                </div>
                {compound.predicted.reasoning.length > 0 && (
                  <div className="mt-2 space-y-0.5">
                    {compound.predicted.reasoning.slice(0, 3).map((r, i) => (
                      <p key={i} className="text-[10px] text-muted-foreground leading-tight">{r}</p>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        <p className="text-[10px] text-muted-foreground mt-3 text-center italic" data-testid="benchmark-notes">
          Predictions computed at startup using XGBoost + GNN ensemble against experimentally verified values
        </p>
      </CardContent>
    </Card>
  );
}

function PhysicsUQCard() {
  const [formula, setFormula] = useState("MgB2");
  const [queryFormula, setQueryFormula] = useState("MgB2");
  const { data: uqData, isLoading } = useQuery<{
    formula: string;
    mean: number;
    std: number;
    ci95: [number, number];
    dominant_uncertainty_source: string;
    errorPropagation: {
      lambdaContribution: number;
      omegaLogContribution: number;
      muStarContribution: number;
    };
    partials: {
      dTc_dLambda: number;
      dTc_dOmegaLog: number;
      dTc_dMuStar: number;
    };
    mcSamples: number;
    mcMean: number;
    mcStd: number;
    analyticMean: number;
    analyticStd: number;
  }>({
    queryKey: ["/api/physics/tc-uq", queryFormula],
    queryFn: () => fetch(`/api/physics/tc-uq/${encodeURIComponent(queryFormula)}`).then(r => r.json()),
  });

  const contribs = uqData?.errorPropagation;
  const barData = contribs ? [
    { name: "\u03BB", value: Math.round(contribs.lambdaContribution * 100), fill: "hsl(var(--primary))" },
    { name: "\u03C9_log", value: Math.round(contribs.omegaLogContribution * 100), fill: "hsl(var(--chart-2))" },
    { name: "\u03BC*", value: Math.round(contribs.muStarContribution * 100), fill: "hsl(var(--chart-3))" },
  ] : [];

  return (
    <Card data-testid="card-physics-uq">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <Shield className="h-4 w-4 text-primary" />
          Physics Tc Uncertainty
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              className="flex-1 rounded-md border border-input bg-background px-3 text-sm min-h-9"
              value={formula}
              onChange={(e) => setFormula(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && formula.trim()) setQueryFormula(formula.trim()); }}
              placeholder="Enter formula..."
              data-testid="input-physics-uq-formula"
            />
          </div>

          {isLoading ? (
            <Skeleton className="h-24" />
          ) : uqData ? (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2">
                <div className="p-2 bg-muted/50 rounded-md" data-testid="physics-uq-mean">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Tc Mean</p>
                  <p className="text-sm font-mono font-bold">{uqData.mean}K</p>
                </div>
                <div className="p-2 bg-muted/50 rounded-md" data-testid="physics-uq-std">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Std Dev</p>
                  <p className="text-sm font-mono font-bold">{uqData.std}K</p>
                </div>
                <div className="p-2 bg-muted/50 rounded-md" data-testid="physics-uq-ci95">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">95% CI</p>
                  <p className="text-sm font-mono font-bold">[{uqData.ci95[0]}, {uqData.ci95[1]}]</p>
                </div>
              </div>

              <div data-testid="physics-uq-dominant">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Dominant Uncertainty Source</p>
                <Badge variant="secondary" className="text-xs border-0">
                  {uqData.dominant_uncertainty_source === "lambda" ? "\u03BB (coupling)" :
                   uqData.dominant_uncertainty_source === "omega_log" ? "\u03C9_log (phonon)" : "\u03BC* (Coulomb)"}
                </Badge>
              </div>

              <div data-testid="physics-uq-contributions">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5">Error Contributions</p>
                <div className="space-y-1">
                  {barData.map((item) => (
                    <div key={item.name} className="flex items-center gap-2">
                      <span className="text-[10px] text-muted-foreground w-10 font-mono">{item.name}</span>
                      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-primary"
                          style={{ width: `${Math.max(3, item.value)}%` }}
                        />
                      </div>
                      <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{item.value}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex flex-wrap gap-3 text-xs" data-testid="physics-uq-methods">
                <div><span className="text-muted-foreground">Analytic:</span> <span className="font-mono">{uqData.analyticMean}K \u00B1 {uqData.analyticStd}K</span></div>
                <div><span className="text-muted-foreground">MC ({uqData.mcSamples}):</span> <span className="font-mono">{uqData.mcMean}K \u00B1 {uqData.mcStd}K</span></div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground italic" data-testid="physics-uq-placeholder">
              Enter a formula to compute physics-based Tc uncertainty
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function DistortionDetectorCard() {
  const { data: distStats, isLoading } = useQuery<{
    totalAnalyzed: number;
    levelCounts: Record<string, number>;
    avgOverallScore: number;
    avgMeanDisplacement: number;
    avgStrainMagnitude: number;
    avgVolumeChangePct: number;
    symmetryBrokenCount: number;
    symmetryBrokenRate: number;
    jahnTellerCount: number;
    peierlsCount: number;
    cdwCount: number;
    changeTypeCounts: Record<string, number>;
    bondStats: { analyzedCount: number; avgBondVariance: number; avgDistortedFraction: number };
    octahedralStats: { analyzedCount: number; avgAngleVariance: number; distortedSiteRate: number; octDistLevels: Record<string, number> };
    phononStats: { analyzedCount: number; imaginaryCount: number; severityDistribution: Record<string, number> };
    recentDistortions: Array<{
      formula: string;
      level: string;
      score: number;
      meanDisp: number;
      strain: number;
      volChange: number;
      symmetryBroken: boolean;
      bondVariance: number | null;
      octAngleVar: number | null;
      phononUnstable: boolean;
    }>;
  }>({ queryKey: ["/api/distortion/stats"] });

  const levelColors: Record<string, string> = {
    none: "text-muted-foreground",
    small: "text-green-400",
    moderate: "text-yellow-400",
    large: "text-orange-400",
    severe: "text-red-400",
  };

  const mechanismCounts = distStats ? {
    ...(distStats.jahnTellerCount > 0 ? { "Jahn-Teller": distStats.jahnTellerCount } : {}),
    ...(distStats.peierlsCount > 0 ? { "Peierls": distStats.peierlsCount } : {}),
    ...(distStats.cdwCount > 0 ? { "CDW": distStats.cdwCount } : {}),
  } : {};

  return (
    <Card data-testid="card-distortion-detector">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-purple-400" />
          Distortion Detector
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
            <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
          </div>
        ) : !distStats || distStats.totalAnalyzed === 0 ? (
          <p className="text-xs text-muted-foreground">No distortion analyses recorded yet. Data populates after xTB relaxations.</p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-bold" data-testid="text-distortion-total">{distStats.totalAnalyzed}</div>
                <div className="text-[10px] text-muted-foreground">Analyzed</div>
              </div>
              <div>
                <div className="text-lg font-bold text-purple-400" data-testid="text-distortion-avg-score">{distStats.avgOverallScore.toFixed(2)}</div>
                <div className="text-[10px] text-muted-foreground">Avg Score</div>
              </div>
              <div>
                <div className="text-lg font-bold text-amber-400" data-testid="text-distortion-sym-broken">{distStats.symmetryBrokenCount}</div>
                <div className="text-[10px] text-muted-foreground">Sym Broken</div>
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-muted-foreground font-medium">Level Distribution</div>
              <div className="flex gap-1 flex-wrap">
                {Object.entries(distStats.levelCounts).filter(([, c]) => c > 0).map(([level, count]) => (
                  <span key={level} className={`text-[10px] px-1.5 py-0.5 rounded bg-muted ${levelColors[level] || "text-muted-foreground"}`} data-testid={`badge-distortion-level-${level}`}>
                    {level}: {count}
                  </span>
                ))}
              </div>
            </div>

            {Object.keys(mechanismCounts).length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Mechanisms Detected</div>
                <div className="flex gap-1 flex-wrap">
                  {Object.entries(mechanismCounts).map(([mech, count]) => (
                    <span key={mech} className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-300" data-testid={`badge-mechanism-${mech}`}>
                      {mech}: {count}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="grid grid-cols-3 gap-2">
              {distStats.bondStats.analyzedCount > 0 && (
                <div className="bg-muted/30 rounded p-1.5 text-center" data-testid="panel-bond-distortion">
                  <div className="text-[9px] text-muted-foreground font-medium">Bond Length</div>
                  <div className="text-xs font-bold text-blue-400">{distStats.bondStats.avgBondVariance.toFixed(4)}</div>
                  <div className="text-[9px] text-muted-foreground">avg variance (A^2)</div>
                  <div className="text-[9px] text-muted-foreground">{(distStats.bondStats.avgDistortedFraction * 100).toFixed(0)}% distorted</div>
                </div>
              )}
              {distStats.octahedralStats.analyzedCount > 0 && (
                <div className="bg-muted/30 rounded p-1.5 text-center" data-testid="panel-octahedral-distortion">
                  <div className="text-[9px] text-muted-foreground font-medium">Octahedral</div>
                  <div className="text-xs font-bold text-cyan-400">{distStats.octahedralStats.avgAngleVariance.toFixed(1)}</div>
                  <div className="text-[9px] text-muted-foreground">avg angle var (deg^2)</div>
                  <div className="text-[9px] text-muted-foreground">{(distStats.octahedralStats.distortedSiteRate * 100).toFixed(0)}% sites dist.</div>
                </div>
              )}
              {distStats.phononStats.analyzedCount > 0 && (
                <div className="bg-muted/30 rounded p-1.5 text-center" data-testid="panel-phonon-instability">
                  <div className="text-[9px] text-muted-foreground font-medium">Phonon</div>
                  <div className="text-xs font-bold text-red-400">{distStats.phononStats.imaginaryCount}</div>
                  <div className="text-[9px] text-muted-foreground">with imag. modes</div>
                  <div className="text-[9px] text-muted-foreground">of {distStats.phononStats.analyzedCount} checked</div>
                </div>
              )}
            </div>

            {distStats.recentDistortions.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Recent Analyses</div>
                <div className="space-y-1 max-h-36 overflow-y-auto">
                  {distStats.recentDistortions.slice(0, 6).map((a, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-distortion-${i}`}>
                      <span className="font-mono font-medium">{a.formula}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={levelColors[a.level] || ""}>{a.level}</span>
                        <span className="text-muted-foreground">
                          {a.strain.toFixed(3)}
                        </span>
                        {a.bondVariance !== null && a.bondVariance > 0.005 && (
                          <span className="text-blue-400 text-[9px]">BL:{a.bondVariance.toFixed(3)}</span>
                        )}
                        {a.octAngleVar !== null && a.octAngleVar > 10 && (
                          <span className="text-cyan-400 text-[9px]">Oct:{a.octAngleVar.toFixed(0)}</span>
                        )}
                        {a.phononUnstable && (
                          <span className="text-red-400 text-[9px]">imag</span>
                        )}
                        {a.symmetryBroken && (
                          <span className="text-amber-400 text-[9px]">sym</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function HeterostructureGeneratorCard() {
  const { data: heteroStats, isLoading } = useQuery<{
    totalGenerated: number;
    idealMismatchCount: number;
    workableMismatchCount: number;
    unlikelyMismatchCount: number;
    avgMismatch: number;
    avgInterfaceScore: number;
    substrateUsage: Record<string, number>;
    topCandidates: Array<{
      layer1: string;
      layer2: string;
      mismatch: number;
      mismatchQuality: string;
      interfaceScore: number;
      scEnhancement: number;
    }>;
    recentGenerations: Array<{
      layer1: string;
      layer2: string;
      orientation: string;
      mismatch: number;
      quality: string;
      atoms: number;
    }>;
  }>({ queryKey: ["/api/heterostructure/stats"], refetchInterval: 30000 });

  const qualityColors: Record<string, string> = {
    ideal: "text-green-400",
    workable: "text-yellow-400",
    unlikely: "text-red-400",
  };

  return (
    <Card data-testid="card-heterostructure-generator">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Layers className="h-4 w-4 text-teal-400" />
          Heterostructure Generator
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
            <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
          </div>
        ) : !heteroStats || heteroStats.totalGenerated === 0 ? (
          <p className="text-xs text-muted-foreground">No bilayer structures generated yet. Pairs SC candidates with substrate materials for interface-enhanced superconductivity.</p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-bold" data-testid="text-hetero-total">{heteroStats.totalGenerated}</div>
                <div className="text-[10px] text-muted-foreground">Generated</div>
              </div>
              <div>
                <div className="text-lg font-bold text-green-400" data-testid="text-hetero-ideal">{heteroStats.idealMismatchCount}</div>
                <div className="text-[10px] text-muted-foreground">Ideal Match</div>
              </div>
              <div>
                <div className="text-lg font-bold text-teal-400" data-testid="text-hetero-score">{heteroStats.avgInterfaceScore.toFixed(2)}</div>
                <div className="text-[10px] text-muted-foreground">Avg Score</div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-1">
              <div className="bg-muted/30 rounded p-1 text-center">
                <div className="text-[9px] text-muted-foreground">Ideal (&lt;3%)</div>
                <div className="text-xs font-bold text-green-400">{heteroStats.idealMismatchCount}</div>
              </div>
              <div className="bg-muted/30 rounded p-1 text-center">
                <div className="text-[9px] text-muted-foreground">Workable (3-7%)</div>
                <div className="text-xs font-bold text-yellow-400">{heteroStats.workableMismatchCount}</div>
              </div>
              <div className="bg-muted/30 rounded p-1 text-center">
                <div className="text-[9px] text-muted-foreground">Unlikely (&gt;7%)</div>
                <div className="text-xs font-bold text-red-400">{heteroStats.unlikelyMismatchCount}</div>
              </div>
            </div>

            {Object.keys(heteroStats.substrateUsage).length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Top Substrates</div>
                <div className="flex gap-1 flex-wrap">
                  {Object.entries(heteroStats.substrateUsage)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([sub, count]) => (
                      <span key={sub} className="text-[9px] px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-300" data-testid={`badge-substrate-${sub}`}>
                        {sub}: {count}
                      </span>
                    ))}
                </div>
              </div>
            )}

            {heteroStats.topCandidates.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Top Bilayer Candidates</div>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {heteroStats.topCandidates.slice(0, 5).map((c, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-hetero-top-${i}`}>
                      <span className="font-mono font-medium">{c.layer1}/{c.layer2}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={qualityColors[c.mismatchQuality] || "text-muted-foreground"}>
                          {(c.mismatch * 100).toFixed(1)}%
                        </span>
                        <span className="text-teal-400 text-[9px]">score:{c.interfaceScore.toFixed(2)}</span>
                        <span className="text-purple-400 text-[9px]">SC:{c.scEnhancement.toFixed(2)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {heteroStats.recentGenerations.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Recent Generations</div>
                <div className="space-y-1 max-h-24 overflow-y-auto">
                  {heteroStats.recentGenerations.slice(0, 4).map((g, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-hetero-recent-${i}`}>
                      <span className="font-mono font-medium">{g.layer1}/{g.layer2}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={qualityColors[g.quality] || "text-muted-foreground"}>
                          {g.quality}
                        </span>
                        <span className="text-muted-foreground">{g.orientation}</span>
                        <span className="text-muted-foreground">{g.atoms} atoms</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DisorderGeneratorCard() {
  const { data: disorderStats, isLoading } = useQuery<{
    totalGenerated: number;
    byType: Record<string, number>;
    avgDefectFraction: number;
    avgTcModifier: number;
    bestTcModifier: number;
    bestFormula: string;
    bestDisorderType: string;
    recentGenerations: Array<{
      base: string;
      type: string;
      element: string;
      fraction: number;
      defectCount: number;
      tcModifier: number;
    }>;
    topCandidates: Array<{
      base: string;
      type: string;
      element: string;
      fraction: number;
      tcModifier: number;
      formationEnergy: number;
    }>;
  }>({ queryKey: ["/api/disorder-generator/stats"], refetchInterval: 30000 });

  const { data: metricsStats } = useQuery<{
    totalAnalyzed: number;
    avgDisorderScore: number;
    maxDisorderScore: number;
    byClass: Record<string, number>;
    avgBondVariance: number;
    avgCoordinationVariance: number;
    avgLocalStrain: number;
    avgConfigEntropy: number;
    avgDosDisorderSignal: number;
  }>({ queryKey: ["/api/disorder-metrics/stats"], refetchInterval: 30000 });

  const { data: searchLimits } = useQuery<{
    maxVacancyFraction: number;
    maxSubstitutionFraction: number;
    maxInterstitialFraction: number;
    maxSiteMixingFraction: number;
    maxAmorphousFraction: number;
    maxDisorderTypes: number;
  }>({ queryKey: ["/api/disorder-generator/search-limits"], refetchInterval: 60000 });

  const typeColors: Record<string, string> = {
    vacancy: "text-red-400",
    substitution: "text-blue-400",
    interstitial: "text-green-400",
    "site-mixing": "text-purple-400",
    amorphous: "text-amber-400",
  };

  const typeLabels: Record<string, string> = {
    vacancy: "Vac",
    substitution: "Sub",
    interstitial: "Int",
    "site-mixing": "Mix",
    amorphous: "Amor",
  };

  return (
    <Card data-testid="card-disorder-generator">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Shuffle className="h-4 w-4 text-orange-400" />
          Disorder Generator
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
            <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
          </div>
        ) : !disorderStats || disorderStats.totalGenerated === 0 ? (
          <p className="text-xs text-muted-foreground">No disordered structures generated yet. Creates vacancy, substitution, interstitial, and site-mixing defect variants from base structures.</p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-bold" data-testid="text-disorder-total">{disorderStats.totalGenerated}</div>
                <div className="text-[10px] text-muted-foreground">Generated</div>
              </div>
              <div>
                <div className="text-lg font-bold text-orange-400" data-testid="text-disorder-best-tc">
                  {disorderStats.bestTcModifier > 0 ? `${((disorderStats.bestTcModifier - 1) * 100).toFixed(0)}%` : "0%"}
                </div>
                <div className="text-[10px] text-muted-foreground">Best Tc Boost</div>
              </div>
              <div>
                <div className="text-lg font-bold text-yellow-400" data-testid="text-disorder-avg-frac">
                  {(disorderStats.avgDefectFraction * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-muted-foreground">Avg Defect %</div>
              </div>
            </div>

            <div className="grid grid-cols-5 gap-1">
              {(["vacancy", "substitution", "interstitial", "site-mixing", "amorphous"] as const).map(t => (
                <div key={t} className="bg-muted/30 rounded p-1 text-center">
                  <div className="text-[9px] text-muted-foreground">{typeLabels[t]}</div>
                  <div className={`text-xs font-bold ${typeColors[t]}`} data-testid={`text-disorder-count-${t}`}>
                    {disorderStats.byType[t] || 0}
                  </div>
                </div>
              ))}
            </div>

            {disorderStats.bestFormula && (
              <div className="bg-muted/30 rounded p-2">
                <div className="text-[10px] text-muted-foreground font-medium">Best Candidate</div>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-xs font-mono font-medium" data-testid="text-disorder-best-formula">{disorderStats.bestFormula}</span>
                  <div className="flex items-center gap-1.5">
                    <span className={`text-[10px] ${typeColors[disorderStats.bestDisorderType] || "text-muted-foreground"}`}>
                      {disorderStats.bestDisorderType}
                    </span>
                    <span className="text-orange-400 text-[10px]">
                      +{((disorderStats.bestTcModifier - 1) * 100).toFixed(0)}% Tc
                    </span>
                  </div>
                </div>
              </div>
            )}

            {disorderStats.topCandidates.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Top Variants</div>
                <div className="space-y-1 max-h-28 overflow-y-auto">
                  {disorderStats.topCandidates.slice(0, 5).map((c, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-disorder-top-${i}`}>
                      <span className="font-mono font-medium">{c.base}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={typeColors[c.type] || "text-muted-foreground"}>{typeLabels[c.type] || c.type}</span>
                        <span className="text-muted-foreground">{c.element}</span>
                        <span className="text-muted-foreground">{(c.fraction * 100).toFixed(0)}%</span>
                        <span className="text-orange-400">Tc:{c.tcModifier.toFixed(2)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {disorderStats.recentGenerations.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Recent</div>
                <div className="space-y-1 max-h-20 overflow-y-auto">
                  {disorderStats.recentGenerations.slice(0, 4).map((g, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-disorder-recent-${i}`}>
                      <span className="font-mono font-medium">{g.base}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={typeColors[g.type] || "text-muted-foreground"}>{typeLabels[g.type] || g.type}</span>
                        <span className="text-muted-foreground">{g.element} {(g.fraction * 100).toFixed(0)}%</span>
                        <span className="text-muted-foreground">{g.defectCount} defects</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {metricsStats && metricsStats.totalAnalyzed > 0 && (
              <div className="space-y-2 border-t border-border/40 pt-2 mt-2">
                <div className="text-[10px] text-muted-foreground font-medium">Disorder Metrics</div>
                <div className="grid grid-cols-3 gap-1">
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[9px] text-muted-foreground">Avg Score</div>
                    <div className="text-xs font-bold text-cyan-400" data-testid="text-disorder-avg-score">{metricsStats.avgDisorderScore.toFixed(3)}</div>
                  </div>
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[9px] text-muted-foreground">Max Score</div>
                    <div className="text-xs font-bold text-orange-400" data-testid="text-disorder-max-score">{metricsStats.maxDisorderScore.toFixed(3)}</div>
                  </div>
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[9px] text-muted-foreground">Bond Var</div>
                    <div className="text-xs font-bold text-blue-400" data-testid="text-disorder-bond-var">{metricsStats.avgBondVariance.toFixed(3)}</div>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-1">
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[9px] text-muted-foreground">Coord Var</div>
                    <div className="text-xs font-bold text-green-400" data-testid="text-disorder-coord-var">{metricsStats.avgCoordinationVariance.toFixed(3)}</div>
                  </div>
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[9px] text-muted-foreground">Config Entropy</div>
                    <div className="text-xs font-bold text-purple-400" data-testid="text-disorder-config-entropy">{metricsStats.avgConfigEntropy.toFixed(3)}</div>
                  </div>
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[9px] text-muted-foreground">DOS Signal</div>
                    <div className="text-xs font-bold text-yellow-400" data-testid="text-disorder-dos-signal">{metricsStats.avgDosDisorderSignal.toFixed(3)}</div>
                  </div>
                </div>
                <div className="grid grid-cols-5 gap-1">
                  {(["perfect", "mild", "moderate", "strong", "amorphous"] as const).map(cls => (
                    <div key={cls} className="bg-muted/30 rounded p-1 text-center">
                      <div className="text-[8px] text-muted-foreground capitalize">{cls}</div>
                      <div className="text-[10px] font-bold" data-testid={`text-disorder-class-${cls}`}>{metricsStats.byClass[cls] || 0}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {searchLimits && (
              <div className="space-y-1 border-t border-border/40 pt-2 mt-2">
                <div className="text-[10px] text-muted-foreground font-medium">Search Limits</div>
                <div className="grid grid-cols-3 gap-1">
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[8px] text-muted-foreground">Vacancy</div>
                    <div className="text-[10px] font-bold text-red-400" data-testid="text-limit-vacancy">{(searchLimits.maxVacancyFraction * 100).toFixed(0)}%</div>
                  </div>
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[8px] text-muted-foreground">Substitution</div>
                    <div className="text-[10px] font-bold text-blue-400" data-testid="text-limit-substitution">{(searchLimits.maxSubstitutionFraction * 100).toFixed(0)}%</div>
                  </div>
                  <div className="bg-muted/30 rounded p-1 text-center">
                    <div className="text-[8px] text-muted-foreground">Max Types</div>
                    <div className="text-[10px] font-bold text-white" data-testid="text-limit-types">{searchLimits.maxDisorderTypes}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function InterfaceRelaxationCard() {
  const { data: relaxStats, isLoading } = useQuery<{
    totalRelaxations: number;
    xtbSuccesses: number;
    xtbFailures: number;
    avgCompositeScore: number;
    avgChargeTransfer: number;
    avgStrain: number;
    significantChargeTransferCount: number;
    optimalStrainCount: number;
    topInterfaces: Array<{
      film: string;
      substrate: string;
      compositeScore: number;
      chargePerAtom: number;
      strainPct: number;
      phononCoupling: number;
    }>;
    recentRelaxations: Array<{
      film: string;
      substrate: string;
      compositeScore: number;
      xtbConverged: boolean;
      wallTimeMs: number;
    }>;
    activeLearningSelections: number;
  }>({ queryKey: ["/api/interface-relaxation/stats"], refetchInterval: 30000 });

  return (
    <Card data-testid="card-interface-relaxation">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4 text-cyan-400" />
          Interface Physics
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
            <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
          </div>
        ) : !relaxStats || relaxStats.totalRelaxations === 0 ? (
          <p className="text-xs text-muted-foreground">No interface relaxations yet. Runs xTB optimization on bilayer structures to detect charge transfer, strain effects, and phonon coupling at interfaces.</p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-bold" data-testid="text-relax-total">{relaxStats.totalRelaxations}</div>
                <div className="text-[10px] text-muted-foreground">Relaxed</div>
              </div>
              <div>
                <div className="text-lg font-bold text-green-400" data-testid="text-relax-xtb">{relaxStats.xtbSuccesses}</div>
                <div className="text-[10px] text-muted-foreground">xTB OK</div>
              </div>
              <div>
                <div className="text-lg font-bold text-cyan-400" data-testid="text-relax-score">{relaxStats.avgCompositeScore.toFixed(3)}</div>
                <div className="text-[10px] text-muted-foreground">Avg Score</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="bg-muted/30 rounded p-1.5">
                <div className="text-[9px] text-muted-foreground">Charge Transfer</div>
                <div className="text-xs">
                  <span className="font-bold">{relaxStats.significantChargeTransferCount}</span>
                  <span className="text-muted-foreground"> significant ({'>'}0.05 e/atom)</span>
                </div>
                <div className="text-[9px] text-muted-foreground">avg: {relaxStats.avgChargeTransfer.toFixed(4)} e/atom</div>
              </div>
              <div className="bg-muted/30 rounded p-1.5">
                <div className="text-[9px] text-muted-foreground">Strain</div>
                <div className="text-xs">
                  <span className="font-bold text-green-400">{relaxStats.optimalStrainCount}</span>
                  <span className="text-muted-foreground"> optimal (1-4%)</span>
                </div>
                <div className="text-[9px] text-muted-foreground">avg: {relaxStats.avgStrain.toFixed(2)}%</div>
              </div>
            </div>

            {relaxStats.activeLearningSelections > 0 && (
              <div className="bg-cyan-500/10 rounded p-1.5 text-center">
                <span className="text-[10px] text-cyan-300">{relaxStats.activeLearningSelections} interfaces selected for active learning</span>
              </div>
            )}

            {relaxStats.topInterfaces.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Top Interface Candidates</div>
                <div className="space-y-1 max-h-36 overflow-y-auto">
                  {relaxStats.topInterfaces.slice(0, 5).map((t, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-interface-top-${i}`}>
                      <span className="font-mono font-medium">{t.film}/{t.substrate}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={t.compositeScore > 0.5 ? "text-green-400" : t.compositeScore > 0.3 ? "text-yellow-400" : "text-red-400"}>
                          {t.compositeScore.toFixed(3)}
                        </span>
                        <span className="text-cyan-400 text-[9px]">{t.chargePerAtom.toFixed(3)}e</span>
                        <span className={t.strainPct >= 1 && t.strainPct <= 4 ? "text-green-400 text-[9px]" : "text-muted-foreground text-[9px]"}>
                          {t.strainPct.toFixed(1)}%
                        </span>
                        <span className="text-purple-400 text-[9px]">ph:{t.phononCoupling.toFixed(2)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {relaxStats.recentRelaxations.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Recent Relaxations</div>
                <div className="space-y-1 max-h-24 overflow-y-auto">
                  {relaxStats.recentRelaxations.slice(0, 4).map((r, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-relax-recent-${i}`}>
                      <span className="font-mono font-medium">{r.film}/{r.substrate}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={r.compositeScore > 0.4 ? "text-green-400" : "text-muted-foreground"}>
                          {r.compositeScore.toFixed(3)}
                        </span>
                        <span className={r.xtbConverged ? "text-green-400" : "text-yellow-400"}>
                          {r.xtbConverged ? "xTB" : "est."}
                        </span>
                        <span className="text-muted-foreground">{(r.wallTimeMs / 1000).toFixed(1)}s</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function EnergyLandscapeCard() {
  const { data: landscapeStats, isLoading } = useQuery<{
    totalExplored: number;
    multipleMinima: number;
    multiMinimaRate: number;
    avgUniqueMinima: number;
    avgEnergySpread: number;
    recent: Array<{
      formula: string;
      uniqueMinima: number;
      multipleMinima: boolean;
      energySpread: number;
      distortionModes: boolean;
    }>;
  }>({ queryKey: ["/api/energy-landscape/stats"], refetchInterval: 30000 });

  return (
    <Card data-testid="card-energy-landscape">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Mountain className="h-4 w-4 text-emerald-400" />
          Energy Landscape Explorer
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
            <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
          </div>
        ) : !landscapeStats || landscapeStats.totalExplored === 0 ? (
          <p className="text-xs text-muted-foreground">No landscape explorations yet. Runs perturbation-reoptimization to detect multiple energy minima.</p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-bold" data-testid="text-landscape-total">{landscapeStats.totalExplored}</div>
                <div className="text-[10px] text-muted-foreground">Explored</div>
              </div>
              <div>
                <div className="text-lg font-bold text-emerald-400" data-testid="text-landscape-multi">{landscapeStats.multipleMinima}</div>
                <div className="text-[10px] text-muted-foreground">Multi-Minima</div>
              </div>
              <div>
                <div className="text-lg font-bold text-amber-400" data-testid="text-landscape-rate">{(landscapeStats.multiMinimaRate * 100).toFixed(0)}%</div>
                <div className="text-[10px] text-muted-foreground">Multi Rate</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="bg-muted/30 rounded p-1.5 text-center">
                <div className="text-[9px] text-muted-foreground font-medium">Avg Unique Minima</div>
                <div className="text-xs font-bold text-emerald-400">{landscapeStats.avgUniqueMinima.toFixed(1)}</div>
              </div>
              <div className="bg-muted/30 rounded p-1.5 text-center">
                <div className="text-[9px] text-muted-foreground font-medium">Avg Energy Spread</div>
                <div className="text-xs font-bold text-blue-400">{landscapeStats.avgEnergySpread.toFixed(5)} Eh</div>
              </div>
            </div>

            {landscapeStats.recent.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Recent Explorations</div>
                <div className="space-y-1 max-h-28 overflow-y-auto">
                  {landscapeStats.recent.slice(0, 5).map((r, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-landscape-${i}`}>
                      <span className="font-mono font-medium">{r.formula}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={r.multipleMinima ? "text-emerald-400" : "text-muted-foreground"}>
                          {r.uniqueMinima} minim{r.uniqueMinima !== 1 ? "a" : "um"}
                        </span>
                        <span className="text-muted-foreground">{r.energySpread.toFixed(4)} Eh</span>
                        {r.distortionModes && (
                          <span className="text-amber-400 text-[9px]">distortion modes</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DistortionClassifierCard() {
  const { data: classifierStats, isLoading } = useQuery<{
    trained: boolean;
    trainCount: number;
    accuracy: number;
    lastTrainedAt: number;
    weights: Record<string, number>;
    recentPredictions: Array<{
      formula: string;
      prediction: string;
      probability: number;
      confidence: number;
      topFeature: string;
    }>;
  }>({ queryKey: ["/api/distortion/classifier/stats"], refetchInterval: 30000 });

  const predColors: Record<string, string> = {
    "distorted": "text-red-400",
    "non-distorted": "text-green-400",
  };

  return (
    <Card data-testid="card-distortion-classifier">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Layers className="h-4 w-4 text-indigo-400" />
          ML Distortion Classifier
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
            <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
          </div>
        ) : !classifierStats ? (
          <p className="text-xs text-muted-foreground">ML classifier loading...</p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-bold" data-testid="text-classifier-samples">{classifierStats.trainCount}</div>
                <div className="text-[10px] text-muted-foreground">Train Samples</div>
              </div>
              <div>
                <div className="text-lg font-bold text-indigo-400" data-testid="text-classifier-accuracy">
                  {classifierStats.trained ? `${(classifierStats.accuracy * 100).toFixed(0)}%` : "--"}
                </div>
                <div className="text-[10px] text-muted-foreground">Accuracy</div>
              </div>
              <div>
                <div className={`text-lg font-bold ${classifierStats.trained ? "text-green-400" : "text-muted-foreground"}`} data-testid="text-classifier-status">
                  {classifierStats.trained ? "Trained" : "Pending"}
                </div>
                <div className="text-[10px] text-muted-foreground">Status</div>
              </div>
            </div>

            {classifierStats.trained && Object.keys(classifierStats.weights).length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Top Feature Weights</div>
                <div className="flex gap-1 flex-wrap">
                  {Object.entries(classifierStats.weights)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .slice(0, 5)
                    .map(([name, weight]) => (
                      <span key={name} className="text-[9px] px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300" data-testid={`badge-weight-${name}`}>
                        {name}: {weight.toFixed(2)}
                      </span>
                    ))}
                </div>
              </div>
            )}

            {classifierStats.recentPredictions.length > 0 && (
              <div className="space-y-1">
                <div className="text-[10px] text-muted-foreground font-medium">Recent Predictions</div>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {classifierStats.recentPredictions.slice(0, 6).map((p, i) => (
                    <div key={i} className="flex items-center justify-between text-[10px] bg-muted/50 rounded px-2 py-1" data-testid={`row-classifier-${i}`}>
                      <span className="font-mono font-medium">{p.formula}</span>
                      <div className="flex items-center gap-1.5">
                        <span className={predColors[p.prediction] || "text-muted-foreground"}>
                          {p.prediction}
                        </span>
                        <span className="text-muted-foreground">{(p.probability * 100).toFixed(0)}%</span>
                        <span className="text-[9px] text-muted-foreground">conf:{(p.confidence * 100).toFixed(0)}%</span>
                        <span className="text-[9px] text-indigo-300">top:{p.topFeature}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function FeedbackLoopCard() {
  const { data } = useQuery<{
    totalEvaluations: number;
    globalMeanAbsError: number;
    globalOverestimateRatio: number;
    familyCalibrations: { family: string; sampleCount: number; meanAbsError: number; overestimateRatio: number; calibrationFactor: number; stabilityAccuracy: number }[];
    recentErrors: { formula: string; predicted: number; actual: number; error: number }[];
    fitnessWeightEvolution: { cycle: number; weights: { predictedTc: number; stability: number; synthesis: number; novelty: number; uncertainty: number } }[];
    pillarDFTFeedback: { pillar: string; accuracy: number; total: number }[];
    explorationWeight: number;
    explorationSchedule: { maxWeight: number; minWeight: number; decayHalfLife: number; currentWeight: number };
    noveltySearch: { knownCompositions: number; vectorDimensions: number };
  }>({ queryKey: ["/api/surrogate-fitness/stats"], refetchInterval: 30000 });

  if (!data || data.totalEvaluations === 0) return null;

  return (
    <Card data-testid="card-feedback-loop">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <GitMerge className="h-4 w-4 text-primary" />
            Feedback Loop
          </CardTitle>
          <Badge variant="secondary" className="text-xs border-0" data-testid="feedback-eval-count">
            {data.totalEvaluations} evaluations
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-3 gap-2">
          <div className="p-2 bg-muted/40 rounded-md border border-border/30" data-testid="feedback-mae">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Mean Abs Error</p>
            <p className={`text-lg font-mono font-bold ${data.globalMeanAbsError > 30 ? "text-red-600 dark:text-red-400" : data.globalMeanAbsError > 15 ? "text-amber-600 dark:text-amber-400" : "text-emerald-600 dark:text-emerald-400"}`}>
              {data.globalMeanAbsError.toFixed(1)}K
            </p>
          </div>
          <div className="p-2 bg-muted/40 rounded-md border border-border/30" data-testid="feedback-overestimate">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Overestimate Rate</p>
            <p className={`text-lg font-mono font-bold ${data.globalOverestimateRatio > 0.5 ? "text-red-600 dark:text-red-400" : "text-emerald-600 dark:text-emerald-400"}`}>
              {(data.globalOverestimateRatio * 100).toFixed(0)}%
            </p>
          </div>
          <div className="p-2 bg-muted/40 rounded-md border border-border/30" data-testid="feedback-families">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Families Tracked</p>
            <p className="text-lg font-mono font-bold">{data.familyCalibrations.length}</p>
          </div>
        </div>
        {data.explorationWeight != null && (
          <div className="p-2 bg-muted/40 rounded-md border border-border/30" data-testid="feedback-exploration">
            <div className="flex items-center justify-between">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Exploration Weight</p>
              <span className={`text-xs font-mono font-medium ${data.explorationWeight > 0.15 ? "text-blue-600 dark:text-blue-400" : data.explorationWeight > 0.05 ? "text-amber-600 dark:text-amber-400" : "text-muted-foreground"}`}>
                {(data.explorationWeight * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500/80 rounded-full transition-all"
                style={{ width: `${Math.min(100, (data.explorationWeight / data.explorationSchedule.maxWeight) * 100)}%` }}
              />
            </div>
            <p className="text-[9px] text-muted-foreground mt-0.5">
              {data.explorationWeight > 0.15 ? "High exploration: prioritizing uncertain candidates" : data.explorationWeight > 0.05 ? "Moderate exploration" : "Low exploration: exploiting known chemistry"}
            </p>
          </div>
        )}
        {data.noveltySearch && (
          <div className="flex items-center justify-between text-xs" data-testid="novelty-search-stats">
            <span className="text-muted-foreground">Novelty Search DB</span>
            <span className="font-mono">{data.noveltySearch.knownCompositions} compositions ({data.noveltySearch.vectorDimensions}D vectors)</span>
          </div>
        )}
        {data.familyCalibrations.length > 0 && (
          <div className="space-y-1" data-testid="feedback-family-list">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Per-Family Calibration</p>
            <div className="space-y-1">
              {data.familyCalibrations.slice(0, 6).map(fc => (
                <div key={fc.family} className="flex items-center justify-between text-xs" data-testid={`feedback-family-${fc.family}`}>
                  <span className="font-medium truncate max-w-[100px]">{fc.family}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">{fc.sampleCount} samples</span>
                    <span className={`font-mono ${fc.meanAbsError > 30 ? "text-red-500" : fc.meanAbsError > 15 ? "text-amber-500" : "text-emerald-500"}`}>
                      {fc.meanAbsError.toFixed(1)}K err
                    </span>
                    <span className="font-mono text-muted-foreground">x{fc.calibrationFactor.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        {data.fitnessWeightEvolution.length > 0 && (
          <div className="space-y-1" data-testid="feedback-weight-evolution">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Current Fitness Weights</p>
            <div className="flex gap-1 h-4 rounded-full overflow-hidden">
              {(() => {
                const latest = data.fitnessWeightEvolution[data.fitnessWeightEvolution.length - 1].weights;
                return (
                  <>
                    <div className="bg-blue-500/80 rounded-l" style={{ width: `${latest.predictedTc * 100}%` }} title={`Tc: ${(latest.predictedTc * 100).toFixed(0)}%`} />
                    <div className="bg-emerald-500/80" style={{ width: `${latest.stability * 100}%` }} title={`Stability: ${(latest.stability * 100).toFixed(0)}%`} />
                    <div className="bg-amber-500/80" style={{ width: `${latest.synthesis * 100}%` }} title={`Synthesis: ${(latest.synthesis * 100).toFixed(0)}%`} />
                    <div className="bg-purple-500/80" style={{ width: `${latest.novelty * 100}%` }} title={`Novelty: ${(latest.novelty * 100).toFixed(0)}%`} />
                    <div className="bg-pink-500/80 rounded-r" style={{ width: `${latest.uncertainty * 100}%` }} title={`Uncertainty: ${(latest.uncertainty * 100).toFixed(0)}%`} />
                  </>
                );
              })()}
            </div>
            <div className="flex gap-2 text-[10px] flex-wrap">
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500/80 inline-block" />Tc</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500/80 inline-block" />Stability</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500/80 inline-block" />Synthesis</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500/80 inline-block" />Novelty</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-pink-500/80 inline-block" />Uncertainty</span>
            </div>
          </div>
        )}
        {data.pillarDFTFeedback && data.pillarDFTFeedback.length > 0 && (
          <div className="space-y-1" data-testid="feedback-pillar-accuracy">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Pillar DFT Accuracy</p>
            <div className="grid grid-cols-2 gap-1">
              {data.pillarDFTFeedback.slice(0, 8).map(pf => (
                <div key={pf.pillar} className="flex items-center justify-between text-[10px] px-1.5 py-0.5 bg-muted/30 rounded" data-testid={`pillar-accuracy-${pf.pillar}`}>
                  <span className="truncate max-w-[80px]">{pf.pillar}</span>
                  <span className={`font-mono ${pf.accuracy > 0.6 ? "text-emerald-500" : pf.accuracy > 0.4 ? "text-amber-500" : "text-red-500"}`}>
                    {(pf.accuracy * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery<Stats>({ queryKey: ["/api/stats"] });
  const { data: phases, isLoading: phasesLoading } = useQuery<LearningPhase[]>({ queryKey: ["/api/learning-phases"] });
  const { data: logs, isLoading: logsLoading } = useQuery<ResearchLog[]>({ queryKey: ["/api/research-logs"] });
  const { data: milestoneData } = useQuery<{ milestones: any[]; total: number }>({ queryKey: ["/api/milestones"] });
  const { data: dftStatus } = useQuery<{ total: number; dftEnrichedCount: number; breakdown: { high: number; medium: number; analytical: number } }>({ queryKey: ["/api/dft-status"] });
  const { data: bandCalcStats } = useQuery<{
    totalCalcs: number;
    succeeded: number;
    failed: number;
    avgWallTimeSeconds: number;
  }>({ queryKey: ["/api/dft-band-structure/stats"], refetchInterval: 30000 });
  const { data: bandAnalysisStats } = useQuery<{
    totalAnalyses: number;
    withPockets: number;
    withBandInversions: number;
    withVHS: number;
    withDiracCrossings: number;
    avgPocketCount: number;
  }>({ queryKey: ["/api/dft-band-analysis/stats"], refetchInterval: 30000 });
  const { data: engineMemory } = useQuery<EngineMemory>({ queryKey: ["/api/engine/memory"], refetchInterval: 60000 });
  const { data: scData } = useQuery<{ candidates: any[]; total: number }>({ queryKey: ["/api/superconductor-candidates"] });
  const { data: novelInsightData } = useQuery<{
    insights: { id: string; insightText: string; noveltyScore: number | null; category: string | null; phaseName: string; discoveredAt: string }[];
    total: number;
  }>({ queryKey: ["/api/novel-insights"], refetchInterval: 30000 });
  const { data: crossEngineStats } = useQuery<{
    totalFormulas: number;
    totalInsightsRecorded: number;
    engineCoverage: Record<string, number>;
    multiEngineFormulas: number;
    activePatterns: number;
    patternNames: string[];
  }>({ queryKey: ["/api/cross-engine/stats"], refetchInterval: 60000 });
  const { data: synthDiscStats } = useQuery<{
    totalDiscoveries: number;
    totalRoutes: number;
    avgFitness: number;
    bestFitness: number;
    bestFormula: string;
    engineUsage: Record<string, number>;
  }>({ queryKey: ["/api/synthesis-discovery/stats"], refetchInterval: 60000 });
  const { data: gaEvoStats } = useQuery<{
    mutationRate: number;
    generationsWithoutImprovement: number;
    totalAdaptations: number;
    stagnationThreshold: number;
    goodMotifCount: number;
    badMotifCount: number;
    formulaOutcomeCount: number;
    topGoodMotifs: { motif: string; score: number; count: number }[];
    topBadMotifs: { motif: string; penalty: number; count: number }[];
    eliteArchive: { formula: string; fitness: number; classification: string }[];
    eliteArchiveSize: number;
    structuralMotifs: {
      motifs: { name: string; weight: number; successes: number; failures: number; avgTc: number; successRate: number }[];
      totalMotifs: number;
      activeMotifs: number;
    };
  }>({ queryKey: ["/api/synthesis-discovery/ga-evolution"], refetchInterval: 30000 });
  const { data: genCompStats } = useQuery<{
    generators: { name: string; weight: number; discoveryRate: number; dftSuccesses: number; dftFailures: number; dftBestTc: number; pipelinePassRate: number }[];
    totalDFTSuccesses: number;
    totalDFTFailures: number;
    rebalanceCount: number;
  }>({ queryKey: ["/api/generator-competition/stats"], refetchInterval: 30000 });
  const { data: synthPlannerStats } = useQuery<{
    totalPlans: number;
    totalRoutes: number;
    avgFeasibility: number;
    methodBreakdown: Record<string, number>;
  }>({ queryKey: ["/api/synthesis-planner/stats"], refetchInterval: 60000 });
  const { data: heuristicStats } = useQuery<{
    totalGenerated: number;
    formulasProcessed: number;
    ruleHits: Record<string, number>;
    totalRules: number;
  }>({ queryKey: ["/api/heuristic-synthesis/stats"], refetchInterval: 60000 });
  const { data: mlSynthStats } = useQuery<{
    trained: boolean;
    trainingSize: number;
    featureImportance: Record<string, number>;
  }>({ queryKey: ["/api/ml-synthesis/stats"], refetchInterval: 30000 });
  const { data: retroStats } = useQuery<{
    totalAnalyzed: number;
    avgRoutesPerTarget: number;
    topMethods: Record<string, number>;
  }>({ queryKey: ["/api/retrosynthesis/stats"], refetchInterval: 60000 });
  const { data: synthesisGateStats } = useQuery<{
    totalEvaluated: number;
    totalRejected: number;
    totalPassed: number;
    rejectionRate: number;
    rejectionsByReason: Record<string, number>;
    classificationCounts: Record<string, number>;
    avgCompositeScore: number;
    recentRejections: Array<{ formula: string; score: number; reasons: string[]; at: number }>;
  }>({ queryKey: ["/api/synthesis-gate/stats"], refetchInterval: 60000 });
  const { data: reactionNetworkStats } = useQuery<{
    totalNetworksBuilt: number;
    totalNodesCreated: number;
    totalEdgesCreated: number;
    avgPathCost: number;
    methodBreakdown: Record<string, number>;
    familyBreakdown: Record<string, number>;
  }>({ queryKey: ["/api/synthesis/reaction-network/stats"], refetchInterval: 60000 });
  const { data: alStats } = useQuery<{
    convergence: ActiveLearningStats;
    totalCycles: number;
    recentCycles: any[];
    avgUncertaintyTrend: { cycle: number; before: number; after: number; reductionPct: number }[];
    dftDatasetStats: { totalSize: number; bySource: Record<string, number>; growthHistory: { timestamp: number; size: number; source: string }[] };
  }>({ queryKey: ["/api/gnn/active-learning-stats"], refetchInterval: 30000 });
  const { data: theoryReport } = useQuery<any>({ queryKey: ["/api/theory-report"], refetchInterval: 30000 });
  const ws = useWebSocket();

  const statsHistoryRef = useRef<Record<string, number[]>>({});

  useEffect(() => {
    if (!stats) return;
    const h = statsHistoryRef.current;
    const push = (key: string, val: number) => {
      if (!h[key]) h[key] = [];
      h[key].push(val);
      if (h[key].length > 30) h[key] = h[key].slice(-30);
    };
    push("elements", stats.elementsLearned);
    push("materials", stats.materialsIndexed);
    push("predictions", stats.predictionsGenerated);
    push("sc", stats.superconductorCandidates);
    push("synthesis", stats.synthesisProcesses);
    push("reactions", stats.chemicalReactions);
    push("progress", stats.overallProgress);
    push("milestones", milestoneData?.total ?? 0);
  }, [stats, milestoneData]);

  const getHistory = (key: string) => statsHistoryRef.current[key] ?? [];

  useEffect(() => {
    const relevantTypes = ["phaseUpdate", "progress", "prediction", "insight", "cycleEnd", "log", "strategyUpdate"];
    const hasRelevant = ws.messages.some((m) => relevantTypes.includes(m.type));
    if (hasRelevant) {
      queryClient.invalidateQueries({ queryKey: ["/api/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/learning-phases"] });
      queryClient.invalidateQueries({ queryKey: ["/api/research-logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/novel-predictions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/materials"] });
      queryClient.invalidateQueries({ queryKey: ["/api/engine/memory"] });
      queryClient.invalidateQueries({ queryKey: ["/api/dft-status"] });
      queryClient.invalidateQueries({ queryKey: ["/api/novel-insights"] });
    }
    const hasStrategy = ws.messages.some((m) => m.type === "strategyUpdate");
    if (hasStrategy) {
      queryClient.invalidateQueries({ queryKey: ["/api/research-strategy"] });
      queryClient.invalidateQueries({ queryKey: ["/api/research-strategy/history"] });
    }
    const hasMilestone = ws.messages.some((m) => m.type === "milestone");
    if (hasMilestone) {
      queryClient.invalidateQueries({ queryKey: ["/api/milestones"] });
    }
  }, [ws.messages.length]);

  const radarData = phases?.map(p => ({
    subject: p.name.split(" ").slice(0, 2).join(" "),
    value: p.progress,
    fullMark: 100,
  })) ?? [];

  const formatLogTime = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const sourceColors: Record<string, string> = {
    "NIST": "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
    "Materials Project": "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
    "OQMD": "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
    "AFLOW": "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
    "Internal": "bg-gray-100 text-gray-700 dark:bg-gray-950 dark:text-gray-300",
    "Internal ML Model": "bg-pink-100 text-pink-700 dark:bg-pink-950 dark:text-pink-300",
    "DFT Engine": "bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300",
    "ML Engine": "bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300",
    "SC Research": "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
    "Synthesis Engine": "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
    "Reaction Engine": "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300",
    "OpenAI NLP": "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300",
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Command Center</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Real-time overview of the MatSci-∞ learning system and materials research progress.
        </p>
      </div>

      <EngineControls
        engineState={ws.engineState}
        activeTasks={ws.activeTasks}
        connected={ws.connected}
        messages={ws.messages}
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {statsLoading ? (
          Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-28" />)
        ) : (
          <>
            <StatCard title="Elements Learned" value={stats?.elementsLearned ?? 0} icon={Atom} sub="of 118 known elements" history={getHistory("elements")} />
            <StatCard title="Materials Indexed" value={(stats?.materialsIndexed ?? 0).toLocaleString()} icon={Database} sub="from 4 scientific databases" history={getHistory("materials")} />
            <StatCard title="Predictions Made" value={stats?.predictionsGenerated ?? 0} icon={FlaskConical} sub="novel material candidates" history={getHistory("predictions")} />
            <StatCard title="SC Candidates" value={stats?.superconductorCandidates ?? 0} icon={Zap} sub="superconductor predictions" history={getHistory("sc")} />
            <StatCard title="Synthesis Paths" value={stats?.synthesisProcesses ?? 0} icon={Microscope} sub="lab creation processes" history={getHistory("synthesis")} />
            <StatCard title="Reactions Learned" value={stats?.chemicalReactions ?? 0} icon={BookOpen} sub="chemical reaction database" history={getHistory("reactions")} />
            <StatCard title="Overall Progress" value={`${(stats?.overallProgress ?? 0).toFixed(1)}%`} icon={Brain} sub="across 12 learning phases" history={getHistory("progress")} />
            <StatCard title="Active Phases" value={`${phases?.filter(p => p.status === "active").length ?? 0} / ${phases?.length ?? 12}`} icon={TrendingUp} sub="currently running" />
            <StatCard title="Milestones" value={milestoneData?.total ?? 0} icon={Star} sub="research milestones" history={getHistory("milestones")} />
            <StatCard title="DFT Enriched" value={dftStatus?.dftEnrichedCount ?? 0} icon={Cpu} sub={`${dftStatus?.breakdown?.high ?? 0} high / ${dftStatus?.breakdown?.medium ?? 0} medium confidence`} data-testid="stat-dft-enriched" />
            <StatCard title="Band Structure" value={bandCalcStats?.totalCalcs ?? 0} icon={Activity} sub={`${bandCalcStats?.succeeded ?? 0} converged${bandAnalysisStats?.totalAnalyses ? ` / ${bandAnalysisStats.withPockets} pockets / ${bandAnalysisStats.withBandInversions} inversions` : bandCalcStats?.failed ? ` / ${bandCalcStats.failed} failed` : ""}`} data-testid="stat-band-structure" />
          </>
        )}
      </div>

      <ReferenceBenchmarkCard />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card data-testid="active-learning-stats">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              Active Learning
            </CardTitle>
          </CardHeader>
          <CardContent>
            {(() => {
              const al = engineMemory?.autonomousLoopStats?.activeLearning;
              const invOpt = engineMemory?.autonomousLoopStats?.inverseOptimizer;
              const inverseBestTc = invOpt?.bestTcAcrossAll ?? 0;
              const loopBestTc = al?.bestTcFromLoop ?? 0;
              const memoryBestTc = engineMemory?.bestTc ?? 0;
              const unifiedBestTc = Math.max(memoryBestTc, loopBestTc, inverseBestTc);
              const bestSource = unifiedBestTc === memoryBestTc ? "Reconciled" : unifiedBestTc === loopBestTc ? "DFT Loop" : "Inverse Optimizer";
              const hasALData = al && (al.totalDFTRuns > 0 || al.modelRetrains > 0);

              return (
                <div className="space-y-3">
                  <div className={`p-2.5 rounded-md ${unifiedBestTc > 200 ? "bg-amber-500/10 border border-amber-500/20" : "bg-muted/50"}`} data-testid="al-unified-best-tc">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Best Tc (All Methods)</p>
                    <p className={`text-xl font-mono font-bold ${unifiedBestTc > 200 ? "text-amber-600 dark:text-amber-400" : ""}`}>
                      {unifiedBestTc > 0 ? `${Math.round(unifiedBestTc)}K` : "--"}
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">Source: {bestSource}</p>
                    {(inverseBestTc > 0 || loopBestTc > 0) && (
                      <div className="flex gap-3 mt-1 text-[10px] text-muted-foreground">
                        {inverseBestTc > 0 && <span>Inverse: {Math.round(inverseBestTc)}K</span>}
                        {loopBestTc > 0 && <span>DFT Loop: {Math.round(loopBestTc)}K</span>}
                        {memoryBestTc > 0 && memoryBestTc !== unifiedBestTc && <span>DB: {Math.round(memoryBestTc)}K</span>}
                      </div>
                    )}
                  </div>

                  {!hasALData ? (
                    <p className="text-sm text-muted-foreground italic" data-testid="active-learning-placeholder">
                      Active learning DFT stats will appear once the engine runs DFT cycles
                    </p>
                  ) : (() => {
                    const avgBefore = al?.avgUncertaintyBefore ?? 0;
                    const avgAfter = al?.avgUncertaintyAfter ?? 0;
                    const uncertaintyReduction = avgBefore > 0
                      ? ((avgBefore - avgAfter) / avgBefore * 100)
                      : 0;
                    return (
                      <>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-2.5 bg-muted/50 rounded-md" data-testid="al-dft-runs">
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">DFT Runs</p>
                            <p className="text-xl font-mono font-bold">{al?.totalDFTRuns ?? 0}</p>
                          </div>
                          <div className="p-2.5 bg-muted/50 rounded-md" data-testid="al-model-retrains">
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Model Retrains</p>
                            <p className="text-xl font-mono font-bold">{al?.modelRetrains ?? 0}</p>
                          </div>
                        </div>
                        <div className="space-y-1" data-testid="al-uncertainty-trend">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Uncertainty Trend</span>
                            <span className={`font-mono font-bold ${uncertaintyReduction > 0 ? "text-green-600 dark:text-green-400" : "text-muted-foreground"}`}>
                              {uncertaintyReduction > 0 ? `-${uncertaintyReduction.toFixed(1)}%` : "No change"}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px]">
                            <span className="text-muted-foreground">Before:</span>
                            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                              <div className="h-full rounded-full bg-amber-500" style={{ width: `${avgBefore * 100}%` }} />
                            </div>
                            <span className="font-mono w-8 text-right">{(avgBefore * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px]">
                            <span className="text-muted-foreground">After:</span>
                            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                              <div className="h-full rounded-full bg-green-500" style={{ width: `${avgAfter * 100}%` }} />
                            </div>
                            <span className="font-mono w-8 text-right">{(avgAfter * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </div>
              );
            })()}
          </CardContent>
        </Card>

        <Card data-testid="acquisition-functions">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Compass className="h-4 w-4 text-primary" />
              Acquisition Functions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {(() => {
              const cycles = alStats?.recentCycles ?? [];
              const latest = cycles.length > 0 ? cycles[cycles.length - 1] : null;
              if (!latest || !latest.tierBreakdown) {
                return (
                  <p className="text-sm text-muted-foreground italic" data-testid="acquisition-placeholder">
                    EI/UCB acquisition data will appear after the first active learning cycle
                  </p>
                );
              }
              const tb = latest.tierBreakdown;
              const total = latest.candidatesSelected || 1;
              const tierData = [
                { label: "EI-Best Tc", count: tb.bestTc ?? 0, color: "bg-blue-500" },
                { label: "UCB-Uncertain", count: tb.highUncertainty ?? 0, color: "bg-purple-500" },
                { label: "Pure Curiosity", count: tb.pureCuriosity ?? 0, color: "bg-cyan-500" },
                { label: "Pressure Expl.", count: tb.pressureExploration ?? 0, color: "bg-amber-500" },
                { label: "Random", count: tb.randomExploration ?? 0, color: "bg-gray-400" },
              ];
              const avgEI = cycles.length > 0
                ? cycles.reduce((s: number, c: any) => s + (c.topAcquisitionScore ?? 0), 0) / cycles.length
                : 0;
              return (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-2.5 bg-muted/50 rounded-md" data-testid="acq-top-score">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Top Acquisition</p>
                      <p className="text-xl font-mono font-bold">{(latest.topAcquisitionScore ?? 0).toFixed(3)}</p>
                    </div>
                    <div className="p-2.5 bg-muted/50 rounded-md" data-testid="acq-avg-score">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Top (Recent)</p>
                      <p className="text-xl font-mono font-bold">{avgEI.toFixed(3)}</p>
                    </div>
                  </div>
                  <div data-testid="acq-tier-breakdown">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5">Tier Breakdown (Cycle {latest.cycle})</p>
                    <div className="flex h-3 rounded-full overflow-hidden mb-2">
                      {tierData.filter(t => t.count > 0).map((t, i) => (
                        <div key={i} className={`${t.color} transition-all`} style={{ width: `${(t.count / total) * 100}%` }} title={`${t.label}: ${t.count}`} />
                      ))}
                    </div>
                    <div className="grid grid-cols-2 gap-1">
                      {tierData.map((t, i) => (
                        <div key={i} className="flex items-center gap-1.5 text-[10px]">
                          <div className={`w-2 h-2 rounded-full ${t.color}`} />
                          <span className="text-muted-foreground">{t.label}:</span>
                          <span className="font-mono font-medium">{t.count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  {latest.bestTcThisCycle > 0 && (
                    <div className="p-2.5 bg-muted/50 rounded-md" data-testid="acq-best-tc-cycle">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Best Tc This Cycle</p>
                      <p className="text-lg font-mono font-bold">{Math.round(latest.bestTcThisCycle)}K</p>
                      <p className="text-[10px] text-muted-foreground">{latest.topFormula}</p>
                    </div>
                  )}
                </div>
              );
            })()}
          </CardContent>
        </Card>

        <Card data-testid="hull-stability-summary">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Shield className="h-4 w-4 text-primary" />
              Hull Stability Gate
            </CardTitle>
          </CardHeader>
          <CardContent>
            {(() => {
              const loopStats = engineMemory?.autonomousLoopStats;
              const candidates = scData?.candidates ?? [];
              const withStability = candidates.filter((c: any) => c.stabilityScore != null);
              const stableCount = withStability.filter((c: any) => (c.stabilityScore ?? 0) >= 0.95).length;
              const nearHullCount = withStability.filter((c: any) => {
                const s = c.stabilityScore ?? 0;
                return s >= 0.5 && s < 0.95;
              }).length;
              const metastableCount = withStability.filter((c: any) => {
                const s = c.stabilityScore ?? 0;
                return s > 0 && s < 0.5;
              }).length;
              const totalScreened = loopStats?.totalScreened ?? 0;
              const totalPassed = loopStats?.totalPassed ?? 0;
              const rejected = totalScreened - totalPassed;

              if (totalScreened === 0 && withStability.length === 0) {
                return (
                  <p className="text-sm text-muted-foreground italic" data-testid="hull-placeholder">
                    Stability data will appear once the engine screens candidates
                  </p>
                );
              }

              return (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-2.5 bg-muted/50 rounded-md" data-testid="hull-screened">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Screened</p>
                      <p className="text-xl font-mono font-bold">{totalScreened}</p>
                    </div>
                    <div className="p-2.5 bg-muted/50 rounded-md" data-testid="hull-pass-rate">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Pass Rate</p>
                      <p className="text-xl font-mono font-bold">
                        {totalScreened > 0 ? `${((totalPassed / totalScreened) * 100).toFixed(1)}%` : "--"}
                      </p>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5">
                        <div className="h-2.5 w-2.5 rounded-sm bg-green-500" />
                        <span>Stable (hull ≤0.005)</span>
                      </div>
                      <span className="font-mono font-bold" data-testid="hull-stable-count">{stableCount}</span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5">
                        <div className="h-2.5 w-2.5 rounded-sm bg-yellow-500" />
                        <span>Near-hull (≤0.05)</span>
                      </div>
                      <span className="font-mono font-bold" data-testid="hull-nearhull-count">{nearHullCount}</span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5">
                        <div className="h-2.5 w-2.5 rounded-sm bg-orange-500" />
                        <span>Metastable (≤0.1)</span>
                      </div>
                      <span className="font-mono font-bold" data-testid="hull-metastable-count">{metastableCount}</span>
                    </div>
                    {rejected > 0 && (
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-1.5">
                          <div className="h-2.5 w-2.5 rounded-sm bg-red-500" />
                          <span>Rejected ({">"}0.1 eV/atom)</span>
                        </div>
                        <span className="font-mono font-bold text-red-500" data-testid="hull-rejected-count">{rejected}</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card data-testid="cross-engine-hub">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Network className="h-4 w-4 text-primary" />
              Cross-Engine Intelligence Hub
            </CardTitle>
          </CardHeader>
          <CardContent>
            {(!crossEngineStats || crossEngineStats.totalInsightsRecorded === 0) ? (
              <p className="text-sm text-muted-foreground italic" data-testid="cross-engine-placeholder">
                Cross-engine insights will accumulate as the engine runs analysis cycles
              </p>
            ) : (
              <div className="space-y-3">
                <div className="grid grid-cols-3 gap-2">
                  <div className="p-2 bg-muted/50 rounded-md" data-testid="ce-total-insights">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Insights</p>
                    <p className="text-lg font-mono font-bold">{crossEngineStats.totalInsightsRecorded}</p>
                  </div>
                  <div className="p-2 bg-muted/50 rounded-md" data-testid="ce-formulas">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Formulas</p>
                    <p className="text-lg font-mono font-bold">{crossEngineStats.totalFormulas}</p>
                  </div>
                  <div className="p-2 bg-muted/50 rounded-md" data-testid="ce-patterns">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Patterns</p>
                    <p className="text-lg font-mono font-bold">{crossEngineStats.activePatterns}</p>
                  </div>
                </div>
                {crossEngineStats.multiEngineFormulas > 0 && (
                  <div className="p-2 bg-primary/5 rounded-md" data-testid="ce-multi-engine">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Multi-Engine Convergence</p>
                    <p className="text-lg font-mono font-bold">{crossEngineStats.multiEngineFormulas} formulas</p>
                    <p className="text-[10px] text-muted-foreground">analyzed by 3+ engines simultaneously</p>
                  </div>
                )}
                <div className="space-y-1">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Engine Coverage</p>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(crossEngineStats.engineCoverage || {}).filter(([, count]) => count > 0).map(([engine, count]) => (
                      <Badge key={engine} variant="secondary" className="text-[10px] font-mono border-0" data-testid={`ce-engine-${engine}`}>
                        {engine}: {count}
                      </Badge>
                    ))}
                  </div>
                </div>
                {(crossEngineStats.patternNames ?? []).length > 0 && (
                  <div className="space-y-1">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Active Patterns</p>
                    <div className="space-y-0.5">
                      {crossEngineStats.patternNames.slice(0, 4).map((name, i) => (
                        <p key={i} className="text-xs text-muted-foreground" data-testid={`ce-pattern-${i}`}>{name}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <Card data-testid="synthesis-discovery">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <GitMerge className="h-4 w-4 text-primary" />
              Synthesis Discovery
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {synthPlannerStats && synthPlannerStats.totalPlans > 0 && (
                <>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sp-total-plans">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Plans</p>
                      <p className="text-lg font-mono font-bold">{synthPlannerStats.totalPlans}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sp-total-routes">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Routes</p>
                      <p className="text-lg font-mono font-bold">{synthPlannerStats.totalRoutes}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sp-avg-feasibility">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Feasibility</p>
                      <p className="text-lg font-mono font-bold">{(synthPlannerStats.avgFeasibility * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  {Object.keys(synthPlannerStats.methodBreakdown).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Methods Used</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(synthPlannerStats.methodBreakdown).sort(([, a], [, b]) => (b as number) - (a as number)).map(([method, count]) => (
                          <Badge key={method} variant="secondary" className="text-[10px] font-mono border-0" data-testid={`sp-method-${method}`}>
                            {method}: {count as number}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
              {synthDiscStats && synthDiscStats.totalDiscoveries > 0 && (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sd-total-runs">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Discovery Runs</p>
                      <p className="text-lg font-mono font-bold">{synthDiscStats.totalDiscoveries}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sd-best-fitness">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Best Fitness</p>
                      <p className="text-lg font-mono font-bold">{(synthDiscStats.bestFitness * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  {synthDiscStats.bestFormula && (
                    <div className="p-2 bg-primary/5 rounded-md" data-testid="sd-best-formula">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Best Candidate</p>
                      <p className="text-sm font-mono font-bold">{synthDiscStats.bestFormula}</p>
                    </div>
                  )}
                  {synthDiscStats.engineUsage && Object.keys(synthDiscStats.engineUsage).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Engine Contributions</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(synthDiscStats.engineUsage).filter(([, v]) => v > 0).map(([engine, count]) => (
                          <Badge key={engine} variant="secondary" className="text-[10px] font-mono border-0" data-testid={`sd-engine-${engine}`}>
                            {engine}: {count}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {gaEvoStats && gaEvoStats.formulaOutcomeCount > 0 && (
                    <div className="space-y-2 pt-1 border-t border-border/30" data-testid="ga-evolution-panel">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">GA Adaptive Evolution</p>
                      <div className="grid grid-cols-3 gap-1.5">
                        <div className="p-1.5 bg-muted/40 rounded" data-testid="ga-mutation-rate">
                          <p className="text-[9px] text-muted-foreground">Mutation Rate</p>
                          <p className={`text-sm font-mono font-bold ${gaEvoStats.mutationRate > 0.35 ? "text-amber-600 dark:text-amber-400" : "text-foreground"}`}>
                            {(gaEvoStats.mutationRate * 100).toFixed(0)}%
                          </p>
                        </div>
                        <div className="p-1.5 bg-muted/40 rounded" data-testid="ga-good-motifs">
                          <p className="text-[9px] text-muted-foreground">Good Motifs</p>
                          <p className="text-sm font-mono font-bold text-emerald-600 dark:text-emerald-400">{gaEvoStats.goodMotifCount}</p>
                        </div>
                        <div className="p-1.5 bg-muted/40 rounded" data-testid="ga-bad-motifs">
                          <p className="text-[9px] text-muted-foreground">Bad Motifs</p>
                          <p className="text-sm font-mono font-bold text-red-600 dark:text-red-400">{gaEvoStats.badMotifCount}</p>
                        </div>
                      </div>
                      {gaEvoStats.topGoodMotifs.length > 0 && (
                        <div className="space-y-0.5">
                          <p className="text-[9px] text-muted-foreground">Top Rewarded Motifs</p>
                          <div className="flex flex-wrap gap-1">
                            {gaEvoStats.topGoodMotifs.slice(0, 5).map(m => (
                              <Badge key={m.motif} variant="secondary" className="text-[9px] font-mono border-0 bg-emerald-100/50 dark:bg-emerald-900/30" data-testid={`ga-good-${m.motif}`}>
                                {m.motif} +{m.score.toFixed(2)}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {gaEvoStats.topBadMotifs.length > 0 && (
                        <div className="space-y-0.5">
                          <p className="text-[9px] text-muted-foreground">Top Penalized Motifs</p>
                          <div className="flex flex-wrap gap-1">
                            {gaEvoStats.topBadMotifs.slice(0, 5).map(m => (
                              <Badge key={m.motif} variant="secondary" className="text-[9px] font-mono border-0 bg-red-100/50 dark:bg-red-900/30" data-testid={`ga-bad-${m.motif}`}>
                                {m.motif} -{m.penalty.toFixed(2)}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {gaEvoStats.totalAdaptations > 0 && (
                        <p className="text-[9px] text-muted-foreground" data-testid="ga-adaptations">
                          {gaEvoStats.totalAdaptations} stagnation-triggered mutation boosts
                        </p>
                      )}
                      {gaEvoStats.eliteArchive && gaEvoStats.eliteArchive.length > 0 && (
                        <div className="space-y-0.5 pt-1 border-t border-border/20">
                          <p className="text-[9px] text-muted-foreground">Elite Archive (top {gaEvoStats.eliteArchiveSize} across all runs)</p>
                          <div className="space-y-0.5">
                            {gaEvoStats.eliteArchive.map((e, i) => (
                              <div key={i} className="flex items-center justify-between text-[10px]" data-testid={`elite-${i}`}>
                                <span className="font-mono truncate max-w-[120px]">{e.formula}</span>
                                <div className="flex items-center gap-1.5">
                                  <span className="font-mono font-medium">{(e.fitness * 100).toFixed(1)}%</span>
                                  <Badge variant="secondary" className={`text-[8px] border-0 px-1 py-0 ${e.classification === "practical" ? "bg-emerald-100/50 dark:bg-emerald-900/30" : e.classification === "experimental" ? "bg-amber-100/50 dark:bg-amber-900/30" : "bg-red-100/50 dark:bg-red-900/30"}`}>
                                    {e.classification}
                                  </Badge>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
              {gaEvoStats?.structuralMotifs && gaEvoStats.structuralMotifs.activeMotifs > 0 && (
                <div className="space-y-1.5 pt-2 border-t border-border/30" data-testid="structural-motif-panel">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Structural Motif Rewards</p>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground">{gaEvoStats.structuralMotifs.activeMotifs}/{gaEvoStats.structuralMotifs.totalMotifs} active</span>
                  </div>
                  <div className="space-y-0.5">
                    {gaEvoStats.structuralMotifs.motifs
                      .filter(m => m.successes + m.failures > 0)
                      .slice(0, 8)
                      .map((m) => (
                        <div key={m.name} className="flex items-center justify-between text-[10px]" data-testid={`motif-${m.name}`}>
                          <span className="font-mono truncate max-w-[100px]">{m.name}</span>
                          <div className="flex items-center gap-1.5">
                            <div className="w-12 bg-muted rounded-full h-1.5">
                              <div
                                className={`h-1.5 rounded-full ${m.weight > 1.3 ? "bg-emerald-500" : m.weight > 0.8 ? "bg-blue-500" : "bg-orange-500"}`}
                                style={{ width: `${Math.min(100, (m.weight / 3.0) * 100)}%` }}
                              />
                            </div>
                            <span className="font-mono w-8 text-right">{m.weight.toFixed(2)}</span>
                            <span className="text-muted-foreground w-10 text-right">{m.successes}W/{m.failures}L</span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
              {genCompStats && genCompStats.generators.some(g => g.dftSuccesses + g.dftFailures > 0) && (
                <div className="space-y-1.5 pt-2 border-t border-border/30" data-testid="generator-competition-panel">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Generator Competition (DFT-Verified)</p>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground">{genCompStats.totalDFTSuccesses} DFT wins / {genCompStats.totalDFTFailures} fails</span>
                    <span className="text-muted-foreground">({genCompStats.rebalanceCount} rebalances)</span>
                  </div>
                  <div className="space-y-0.5">
                    {genCompStats.generators
                      .filter(g => g.dftSuccesses + g.dftFailures > 0)
                      .map((g) => (
                        <div key={g.name} className="flex items-center justify-between text-[10px]" data-testid={`gen-comp-${g.name}`}>
                          <span className="font-mono truncate max-w-[90px]">{g.name.replace(/_/g, " ")}</span>
                          <div className="flex items-center gap-1.5">
                            <div className="w-10 bg-muted rounded-full h-1.5">
                              <div
                                className={`h-1.5 rounded-full ${g.discoveryRate > 0.5 ? "bg-emerald-500" : g.discoveryRate > 0.2 ? "bg-blue-500" : "bg-orange-500"}`}
                                style={{ width: `${Math.min(100, g.discoveryRate * 100)}%` }}
                              />
                            </div>
                            <span className="font-mono w-8 text-right">{(g.discoveryRate * 100).toFixed(0)}%</span>
                            <span className="font-mono w-12 text-right">{(g.weight * 100).toFixed(1)}%</span>
                            {g.dftBestTc > 0 && (
                              <span className="text-emerald-600 dark:text-emerald-400 font-mono w-10 text-right">{g.dftBestTc}K</span>
                            )}
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
              {heuristicStats && heuristicStats.totalGenerated > 0 && (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="hg-total">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Heuristic Routes</p>
                      <p className="text-lg font-mono font-bold">{heuristicStats.totalGenerated}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="hg-formulas">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Formulas Processed</p>
                      <p className="text-lg font-mono font-bold">{heuristicStats.formulasProcessed}</p>
                    </div>
                  </div>
                  {Object.keys(heuristicStats.ruleHits).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Rules Matched</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(heuristicStats.ruleHits).sort(([, a], [, b]) => (b as number) - (a as number)).map(([rule, count]) => (
                          <Badge key={rule} variant="secondary" className="text-[10px] font-mono border-0" data-testid={`hg-rule-${rule}`}>
                            {rule}: {count as number}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
              {mlSynthStats && mlSynthStats.trained && (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="ml-synth-trained">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">ML Predictor</p>
                      <p className="text-lg font-mono font-bold text-violet-500">Trained</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="ml-synth-training-size">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Training Size</p>
                      <p className="text-lg font-mono font-bold">{mlSynthStats.trainingSize}</p>
                    </div>
                  </div>
                  {Object.keys(mlSynthStats.featureImportance).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Top Features</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(mlSynthStats.featureImportance)
                          .sort(([, a], [, b]) => b - a)
                          .slice(0, 6)
                          .map(([feat, imp]) => (
                            <Badge key={feat} variant="secondary" className="text-[10px] font-mono border-0" data-testid={`ml-feat-${feat}`}>
                              {feat}: {(imp * 100).toFixed(0)}%
                            </Badge>
                          ))}
                      </div>
                    </div>
                  )}
                </>
              )}
              {retroStats && retroStats.totalAnalyzed > 0 && (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="retro-analyzed">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Retrosynthesis Analyzed</p>
                      <p className="text-lg font-mono font-bold">{retroStats.totalAnalyzed}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="retro-avg-routes">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Routes/Target</p>
                      <p className="text-lg font-mono font-bold">{retroStats.avgRoutesPerTarget.toFixed(1)}</p>
                    </div>
                  </div>
                  {Object.keys(retroStats.topMethods).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Decomposition Methods</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(retroStats.topMethods)
                          .sort(([, a], [, b]) => b - a)
                          .map(([method, count]) => (
                            <Badge key={method} variant="secondary" className="text-[10px] font-mono border-0" data-testid={`retro-method-${method}`}>
                              {method}: {count}
                            </Badge>
                          ))}
                      </div>
                    </div>
                  )}
                </>
              )}
              {reactionNetworkStats && reactionNetworkStats.totalNetworksBuilt > 0 && (
                <div className="space-y-2 pt-2 border-t border-border/30" data-testid="reaction-network-panel">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Reaction Network Graph</p>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="rn-networks">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Networks</p>
                      <p className="text-lg font-mono font-bold">{reactionNetworkStats.totalNetworksBuilt}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="rn-nodes">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Nodes</p>
                      <p className="text-lg font-mono font-bold">{reactionNetworkStats.totalNodesCreated}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="rn-edges">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Edges</p>
                      <p className="text-lg font-mono font-bold">{reactionNetworkStats.totalEdgesCreated}</p>
                    </div>
                  </div>
                  <div className="p-2 bg-muted/50 rounded-md" data-testid="rn-avg-cost">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Path Cost (Dijkstra)</p>
                    <p className="text-lg font-mono font-bold">{reactionNetworkStats.avgPathCost.toFixed(3)}</p>
                  </div>
                  {Object.keys(reactionNetworkStats.methodBreakdown).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Methods</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(reactionNetworkStats.methodBreakdown).sort(([, a], [, b]) => b - a).map(([method, count]) => (
                          <Badge key={method} variant="secondary" className="text-[10px] font-mono border-0" data-testid={`rn-method-${method}`}>
                            {method}: {count}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {Object.keys(reactionNetworkStats.familyBreakdown).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Families Analyzed</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(reactionNetworkStats.familyBreakdown).sort(([, a], [, b]) => b - a).map(([fam, count]) => (
                          <Badge key={fam} variant="secondary" className="text-[10px] font-mono border-0" data-testid={`rn-family-${fam}`}>
                            {fam}: {count}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              {synthesisGateStats && synthesisGateStats.totalEvaluated > 0 && (
                <div className="space-y-2 pt-2 border-t border-border/30" data-testid="synthesis-gate-panel">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Synthesis-First Gate (Hard Filter)</p>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sg-evaluated">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Evaluated</p>
                      <p className="text-lg font-mono font-bold">{synthesisGateStats.totalEvaluated}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sg-passed">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Passed</p>
                      <p className="text-lg font-mono font-bold text-emerald-600 dark:text-emerald-400">{synthesisGateStats.totalPassed}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sg-rejected">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Rejected</p>
                      <p className="text-lg font-mono font-bold text-red-600 dark:text-red-400">{synthesisGateStats.totalRejected}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sg-rejection-rate">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Rejection Rate</p>
                      <p className={`text-lg font-mono font-bold ${synthesisGateStats.rejectionRate > 0.5 ? "text-amber-600 dark:text-amber-400" : "text-foreground"}`}>
                        {(synthesisGateStats.rejectionRate * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="sg-avg-score">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg Composite</p>
                      <p className="text-lg font-mono font-bold">{(synthesisGateStats.avgCompositeScore * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  {Object.keys(synthesisGateStats.classificationCounts).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Route Classifications</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(synthesisGateStats.classificationCounts)
                          .sort(([, a], [, b]) => b - a)
                          .map(([cls, count]) => (
                            <Badge key={cls} variant="secondary" className={`text-[10px] font-mono border-0 ${cls === "one-pot" || cls === "trivial" ? "bg-emerald-100/50 dark:bg-emerald-900/30" : cls === "impractical" ? "bg-red-100/50 dark:bg-red-900/30" : ""}`} data-testid={`sg-class-${cls}`}>
                              {cls}: {count}
                            </Badge>
                          ))}
                      </div>
                    </div>
                  )}
                  {Object.keys(synthesisGateStats.rejectionsByReason).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Rejection Reasons</p>
                      <div className="space-y-0.5">
                        {Object.entries(synthesisGateStats.rejectionsByReason)
                          .sort(([, a], [, b]) => b - a)
                          .slice(0, 5)
                          .map(([reason, count]) => (
                            <div key={reason} className="flex items-center justify-between text-[10px]" data-testid={`sg-reason-${reason.slice(0, 20)}`}>
                              <span className="text-muted-foreground truncate max-w-[180px]">{reason.split(":")[0]}</span>
                              <span className="font-mono font-medium">{count}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                  {synthesisGateStats.recentRejections.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Recent Rejections</p>
                      <div className="space-y-0.5">
                        {synthesisGateStats.recentRejections.slice(-5).reverse().map((r, i) => (
                          <div key={i} className="flex items-center justify-between text-[10px]" data-testid={`sg-recent-${i}`}>
                            <span className="font-mono truncate max-w-[100px]">{r.formula}</span>
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-red-500">{(r.score * 100).toFixed(0)}%</span>
                              <span className="text-muted-foreground truncate max-w-[80px]">{r.reasons[0]?.split(":")[0] || ""}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              {(!synthDiscStats || synthDiscStats.totalDiscoveries === 0) && (!synthPlannerStats || synthPlannerStats.totalPlans === 0) && (!heuristicStats || heuristicStats.totalGenerated === 0) && !mlSynthStats?.trained && (!retroStats || retroStats.totalAnalyzed === 0) && (
                <p className="text-sm text-muted-foreground italic" data-testid="synth-disc-placeholder">
                  Synthesis routes will be planned as candidates progress through screening
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-4">
          <ThoughtFeed thoughts={ws.thoughts} tempo={ws.tempo} />

          <Card data-testid="card-research-events">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <FileText className="h-4 w-4 text-primary" />
                Recent Research Events
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <ScrollArea className="h-64">
                {logsLoading ? (
                  <div className="p-4 space-y-3">
                    {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-12" />)}
                  </div>
                ) : (
                  <div className="divide-y divide-border">
                    {logs?.slice(0, 12).map((log, i) => (
                      <div key={i} className="px-4 py-3 flex items-start gap-3" data-testid={`log-entry-${i}`}>
                        <div className="mt-0.5">
                          <Zap className="h-3.5 w-3.5 text-primary flex-shrink-0" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-sm font-medium break-words">{log.event}</span>
                            {log.dataSource && (
                              <span className={`text-xs px-1.5 py-0.5 rounded-sm font-medium ${sourceColors[log.dataSource] || "bg-muted text-muted-foreground"}`}>
                                {log.dataSource}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground mt-0.5 break-words line-clamp-3">{log.detail}</p>
                        </div>
                        {log.timestamp && (
                          <span className="text-xs text-muted-foreground font-mono flex-shrink-0 hidden sm:block">
                            {formatLogTime(log.timestamp as unknown as string)}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <StrategyCard />
          <ResearchMemoryCard />

          <Card data-testid="card-knowledge-radar">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-primary" />
                Knowledge Radar
              </CardTitle>
            </CardHeader>
            <CardContent>
              {phasesLoading ? (
                <Skeleton className="h-52 w-full" />
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="hsl(var(--border))" />
                    <PolarAngleAxis dataKey="subject" tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }} />
                    <Radar name="Progress" dataKey="value" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.2} strokeWidth={1.5} />
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "12px" }}
                      formatter={(v: any) => [`${v}%`, "Progress"]}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>

          <Card data-testid="card-data-sources">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Microscope className="h-4 w-4 text-primary" />
                Data Sources
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2.5">
                {[
                  { name: "NIST WebBook", status: "Synced", entries: "28K compounds", color: "bg-blue-500" },
                  { name: "Materials Project", status: "Synced", entries: "140K materials", color: "bg-purple-500" },
                  { name: "OQMD", status: "Synced", entries: "1M+ entries", color: "bg-orange-500" },
                  { name: "AFLOW", status: "Synced", entries: "3.5M alloys", color: "bg-green-500" },
                ].map((src) => (
                  <div key={src.name} className="flex items-center justify-between" data-testid={`source-${src.name.toLowerCase().replace(/ /g, "-")}`}>
                    <div className="flex items-center gap-2">
                      <div className={`h-2 w-2 rounded-full ${src.color}`} />
                      <span className="text-sm font-medium">{src.name}</span>
                    </div>
                    <div className="text-right">
                      <div className="text-xs font-mono text-primary">{src.entries}</div>
                      <div className="text-xs text-muted-foreground">{src.status}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <GNNActiveLearningCard />
          <PhysicsUQCard />
          <HeterostructureGeneratorCard />
          <DisorderGeneratorCard />
          <InterfaceRelaxationCard />
          <DistortionDetectorCard />
          <EnergyLandscapeCard />
          <DistortionClassifierCard />
          <FeedbackLoopCard />

          <Card data-testid="card-discovery-progress">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Compass className="h-4 w-4 text-primary" />
                Causal Discovery Progress
              </CardTitle>
            </CardHeader>
            <CardContent>
              {theoryReport ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-2 text-center">
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="disc-total-runs">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Discovery Runs</p>
                      <p className="text-lg font-mono font-bold">{theoryReport.summary?.totalDiscoveryRuns ?? 0}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="disc-total-rules">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Causal Rules</p>
                      <p className="text-lg font-mono font-bold">{theoryReport.summary?.totalRules ?? 0}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="disc-total-hypotheses">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Hypotheses</p>
                      <p className="text-lg font-mono font-bold">{theoryReport.summary?.totalHypotheses ?? 0}</p>
                    </div>
                    <div className="p-2 bg-muted/50 rounded-md" data-testid="disc-graph-size">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Graph Nodes</p>
                      <p className="text-lg font-mono font-bold">{theoryReport.summary?.graphNodes ?? 0}</p>
                    </div>
                  </div>

                  {theoryReport.causal?.categoryDiscoveryProgress && Object.keys(theoryReport.causal.categoryDiscoveryProgress).length > 0 && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground mb-2">Category Coverage</p>
                      <div className="space-y-2">
                        {Object.entries(theoryReport.causal.categoryDiscoveryProgress as Record<string, { variables: number; rulesDiscovered: number; coveragePercent: number }>).map(([cat, info]) => (
                          <div key={cat} data-testid={`disc-category-${cat}`}>
                            <div className="flex items-center justify-between text-xs mb-0.5">
                              <span className="capitalize font-medium">{cat}</span>
                              <span className="text-muted-foreground font-mono">{info.rulesDiscovered}/{info.variables} vars ({info.coveragePercent}%)</span>
                            </div>
                            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                              <div
                                className="h-full bg-primary rounded-full transition-all"
                                style={{ width: `${Math.min(info.coveragePercent, 100)}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {theoryReport.symbolic && (
                    <div className="border-t pt-2 mt-2">
                      <p className="text-xs font-medium text-muted-foreground mb-1.5">Symbolic Discovery</p>
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div className="p-1.5 bg-muted/50 rounded" data-testid="disc-sym-theories">
                          <p className="text-[9px] text-muted-foreground">Theories</p>
                          <p className="text-sm font-mono font-bold">{theoryReport.symbolic.totalTheories ?? 0}</p>
                        </div>
                        <div className="p-1.5 bg-muted/50 rounded" data-testid="disc-sym-best">
                          <p className="text-[9px] text-muted-foreground">Best Score</p>
                          <p className="text-sm font-mono font-bold">{(theoryReport.symbolic.bestScore ?? 0).toFixed(2)}</p>
                        </div>
                        <div className="p-1.5 bg-muted/50 rounded" data-testid="disc-sym-features">
                          <p className="text-[9px] text-muted-foreground">Features</p>
                          <p className="text-sm font-mono font-bold">{theoryReport.symbolic.featureLibrarySize ?? 0}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {theoryReport.summary?.topRecommendation && (
                    <div className="bg-green-50 dark:bg-green-950/30 rounded-md p-2 border border-green-200 dark:border-green-900" data-testid="disc-top-recommendation">
                      <p className="text-[10px] text-green-600 dark:text-green-400 font-medium uppercase tracking-wider mb-0.5">Top Recommendation</p>
                      <p className="text-xs text-green-800 dark:text-green-300">{theoryReport.summary.topRecommendation}</p>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground italic py-4" data-testid="discovery-progress-placeholder">
                  Discovery progress will appear after running causal and symbolic physics engines
                </p>
              )}
            </CardContent>
          </Card>

          <Card data-testid="card-learning-insights">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base flex items-center gap-2">
                  <BookOpen className="h-4 w-4 text-primary" />
                  Learning Insights
                </CardTitle>
                {(novelInsightData?.total ?? 0) > 0 && (
                  <Badge variant="secondary" className="text-xs border-0" data-testid="insight-total-count">
                    {novelInsightData?.total ?? 0} total
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-52">
                {(novelInsightData?.insights?.length ?? 0) > 0 ? (
                  <div className="space-y-2">
                    {(novelInsightData?.insights ?? []).slice(0, 10).map((insight) => (
                      <div key={insight.id} className="bg-muted/50 rounded-md px-3 py-2" data-testid={`insight-${insight.id}`}>
                        <p className="text-xs text-foreground leading-relaxed">{insight.insightText}</p>
                        <div className="flex items-center gap-2 mt-1.5">
                          {insight.category && (
                            <Badge variant="outline" className="text-[10px] h-4 px-1.5 border-0 bg-primary/10 text-primary">
                              {insight.category}
                            </Badge>
                          )}
                          {insight.noveltyScore != null && (
                            <span className="text-[10px] text-muted-foreground font-mono">
                              novelty: {(insight.noveltyScore * 100).toFixed(0)}%
                            </span>
                          )}
                          <span className="text-[10px] text-muted-foreground ml-auto">
                            {insight.phaseName}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground italic py-4" data-testid="insights-placeholder">
                    Insights will appear as the engine discovers novel patterns
                  </p>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
