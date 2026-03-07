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
  Cpu, Shield, Activity,
} from "lucide-react";
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip, LineChart, Line } from "recharts";
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

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery<Stats>({ queryKey: ["/api/stats"] });
  const { data: phases, isLoading: phasesLoading } = useQuery<LearningPhase[]>({ queryKey: ["/api/learning-phases"] });
  const { data: logs, isLoading: logsLoading } = useQuery<ResearchLog[]>({ queryKey: ["/api/research-logs"] });
  const { data: milestoneData } = useQuery<{ milestones: any[]; total: number }>({ queryKey: ["/api/milestones"] });
  const { data: dftStatus } = useQuery<{ total: number; dftEnrichedCount: number; breakdown: { high: number; medium: number; analytical: number } }>({ queryKey: ["/api/dft-status"] });
  const { data: engineMemory } = useQuery<EngineMemory>({ queryKey: ["/api/engine/memory"], refetchInterval: 15000 });
  const { data: scData } = useQuery<{ candidates: any[]; total: number }>({ queryKey: ["/api/superconductor-candidates"] });
  const { data: novelInsightData } = useQuery<{
    insights: { id: string; insightText: string; noveltyScore: number | null; category: string | null; phaseName: string; discoveredAt: string }[];
    total: number;
  }>({ queryKey: ["/api/novel-insights", "recent"], refetchInterval: 30000 });
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
      queryClient.invalidateQueries({ queryKey: ["/api/novel-insights", "recent"] });
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
          </>
        )}
      </div>

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
              const hasALData = al && (al.totalDFTRuns > 0 || al.modelRetrains > 0);

              return (
                <div className="space-y-3">
                  <div className={`p-2.5 rounded-md ${inverseBestTc > 200 ? "bg-amber-500/10 border border-amber-500/20" : "bg-muted/50"}`} data-testid="al-inverse-best-tc">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Inverse Optimizer Best Tc</p>
                    <p className={`text-xl font-mono font-bold ${inverseBestTc > 200 ? "text-amber-600 dark:text-amber-400" : ""}`}>
                      {inverseBestTc > 0 ? `${Math.round(inverseBestTc)}K` : "--"}
                    </p>
                  </div>

                  {!hasALData ? (
                    <p className="text-sm text-muted-foreground italic" data-testid="active-learning-placeholder">
                      Active learning DFT stats will appear once the engine runs DFT cycles
                    </p>
                  ) : (() => {
                    const uncertaintyReduction = al!.avgUncertaintyBefore > 0
                      ? ((al!.avgUncertaintyBefore - al!.avgUncertaintyAfter) / al!.avgUncertaintyBefore * 100)
                      : 0;
                    return (
                      <>
                        <div className="grid grid-cols-3 gap-3">
                          <div className="p-2.5 bg-muted/50 rounded-md" data-testid="al-dft-runs">
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">DFT Runs</p>
                            <p className="text-xl font-mono font-bold">{al!.totalDFTRuns}</p>
                          </div>
                          <div className="p-2.5 bg-muted/50 rounded-md" data-testid="al-model-retrains">
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Model Retrains</p>
                            <p className="text-xl font-mono font-bold">{al!.modelRetrains}</p>
                          </div>
                          <div className="p-2.5 bg-muted/50 rounded-md" data-testid="al-best-tc">
                            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Best Tc (Loop)</p>
                            <p className="text-xl font-mono font-bold">{Math.round(al!.bestTcFromLoop)}K</p>
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
                              <div className="h-full rounded-full bg-amber-500" style={{ width: `${al!.avgUncertaintyBefore * 100}%` }} />
                            </div>
                            <span className="font-mono w-8 text-right">{(al!.avgUncertaintyBefore * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px]">
                            <span className="text-muted-foreground">After:</span>
                            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                              <div className="h-full rounded-full bg-green-500" style={{ width: `${al!.avgUncertaintyAfter * 100}%` }} />
                            </div>
                            <span className="font-mono w-8 text-right">{(al!.avgUncertaintyAfter * 100).toFixed(0)}%</span>
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
        <div className="space-y-4">
          <ThoughtFeed thoughts={ws.thoughts} tempo={ws.tempo} />

          <Card>
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
                            <span className="text-sm font-medium truncate">{log.event}</span>
                            {log.dataSource && (
                              <span className={`text-xs px-1.5 py-0.5 rounded-sm font-medium ${sourceColors[log.dataSource] || "bg-muted text-muted-foreground"}`}>
                                {log.dataSource}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground mt-0.5 truncate">{log.detail}</p>
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

          <Card>
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

          <Card>
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

          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base flex items-center gap-2">
                  <BookOpen className="h-4 w-4 text-primary" />
                  Learning Insights
                </CardTitle>
                {(novelInsightData?.total ?? 0) > 0 && (
                  <Badge variant="secondary" className="text-xs border-0" data-testid="insight-total-count">
                    {novelInsightData!.total} total
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-52">
                {(novelInsightData?.insights?.length ?? 0) > 0 ? (
                  <div className="space-y-2">
                    {novelInsightData!.insights.slice(0, 10).map((insight) => (
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
