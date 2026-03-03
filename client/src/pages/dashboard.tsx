import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { queryClient } from "@/lib/queryClient";
import type { LearningPhase, ResearchLog } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Atom, Database, FlaskConical, Brain,
  TrendingUp, CheckCircle2, Clock, Loader2,
  Zap, BookOpen, Microscope, BarChart3, FileText
} from "lucide-react";
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from "recharts";
import { useWebSocket } from "@/hooks/use-websocket";
import { EngineControls } from "@/components/engine-controls";

interface Stats {
  elementsLearned: number;
  materialsIndexed: number;
  predictionsGenerated: number;
  overallProgress: number;
}

function PhaseStatusBadge({ status }: { status: string }) {
  if (status === "completed") return <Badge variant="secondary" className="text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950 border-0 text-xs"><CheckCircle2 className="h-3 w-3 mr-1" />Completed</Badge>;
  if (status === "active") return <Badge variant="secondary" className="text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950 border-0 text-xs"><Loader2 className="h-3 w-3 mr-1 animate-spin" />Active</Badge>;
  return <Badge variant="secondary" className="text-muted-foreground text-xs"><Clock className="h-3 w-3 mr-1" />Pending</Badge>;
}

function StatCard({ title, value, icon: Icon, sub }: { title: string; value: string | number; icon: any; sub?: string }) {
  return (
    <Card data-testid={`stat-card-${title.toLowerCase().replace(/ /g, "-")}`}>
      <CardHeader className="flex flex-row items-center justify-between gap-1 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <Icon className="h-4 w-4 text-primary" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold font-mono text-foreground">{value}</div>
        {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery<Stats>({ queryKey: ["/api/stats"] });
  const { data: phases, isLoading: phasesLoading } = useQuery<LearningPhase[]>({ queryKey: ["/api/learning-phases"] });
  const { data: logs, isLoading: logsLoading } = useQuery<ResearchLog[]>({ queryKey: ["/api/research-logs"] });
  const ws = useWebSocket();

  useEffect(() => {
    const relevantTypes = ["phaseUpdate", "progress", "prediction", "insight", "cycleEnd", "log"];
    const hasRelevant = ws.messages.some((m) => relevantTypes.includes(m.type));
    if (hasRelevant) {
      queryClient.invalidateQueries({ queryKey: ["/api/stats"] });
      queryClient.invalidateQueries({ queryKey: ["/api/learning-phases"] });
      queryClient.invalidateQueries({ queryKey: ["/api/research-logs"] });
      queryClient.invalidateQueries({ queryKey: ["/api/novel-predictions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/materials"] });
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

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {statsLoading ? (
          Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-28" />)
        ) : (
          <>
            <StatCard title="Elements Learned" value={stats?.elementsLearned ?? 0} icon={Atom} sub="of 118 known elements" />
            <StatCard title="Materials Indexed" value={(stats?.materialsIndexed ?? 0).toLocaleString()} icon={Database} sub="from 4 scientific databases" />
            <StatCard title="Predictions Made" value={stats?.predictionsGenerated ?? 0} icon={FlaskConical} sub="novel material candidates" />
            <StatCard title="Overall Progress" value={`${(stats?.overallProgress ?? 0).toFixed(1)}%`} icon={Brain} sub="across all learning phases" />
          </>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="md:col-span-2 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-primary" />
                Learning Pipeline
              </CardTitle>
            </CardHeader>
            <CardContent>
              {phasesLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-14" />)}
                </div>
              ) : (
                <div className="space-y-4">
                  {phases?.map((phase) => (
                    <div key={phase.id} data-testid={`phase-${phase.id}`} className="space-y-1.5">
                      <div className="flex items-center justify-between gap-2 flex-wrap">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-sm font-medium">{phase.name}</span>
                          <PhaseStatusBadge status={phase.status} />
                        </div>
                        <span className="text-xs font-mono text-muted-foreground">
                          {phase.itemsLearned.toLocaleString()} / {phase.totalItems.toLocaleString()}
                        </span>
                      </div>
                      <Progress value={phase.progress} className="h-2" />
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

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
              <CardTitle className="text-base flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-primary" />
                Learning Insights
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-44">
                {phasesLoading ? (
                  <div className="space-y-2">
                    {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-10" />)}
                  </div>
                ) : (
                  <div className="space-y-2">
                    {phases?.filter(p => p.insights?.length).flatMap(p => p.insights ?? []).slice(0, 6).map((insight, i) => (
                      <div key={i} className="text-xs text-muted-foreground bg-muted/50 rounded-md px-2 py-2 leading-relaxed">
                        {insight}
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

