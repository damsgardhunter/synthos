import { useQuery } from "@tanstack/react-query";
import type { LearningPhase, ResearchLog } from "@shared/schema";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CheckCircle2, Clock, Loader2, Zap, BookOpen, ArrowRight, BarChart3, FileText } from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";

function StatusIcon({ status }: { status: string }) {
  if (status === "completed") return <CheckCircle2 className="h-5 w-5 text-green-500" />;
  if (status === "active") return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
  return <Clock className="h-5 w-5 text-muted-foreground" />;
}

function PhaseCard({ phase, index }: { phase: LearningPhase; index: number }) {
  const insights = phase.insights ?? [];
  const isCompleted = phase.status === "completed";
  const isActive = phase.status === "active";

  return (
    <Card
      data-testid={`phase-card-${phase.id}`}
      className={`relative ${isActive ? "border-primary/40" : ""}`}
    >
      {isActive && (
        <div className="absolute inset-0 rounded-lg bg-primary/5 dark:bg-primary/10 pointer-events-none" />
      )}
      <CardHeader className="pb-3">
        <div className="flex items-start gap-3">
          <div className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-sm font-bold font-mono ${
            isCompleted ? "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300" :
            isActive ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300" :
            "bg-muted text-muted-foreground"
          }`}>
            {index + 1}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <StatusIcon status={phase.status} />
              <CardTitle className="text-base">{phase.name}</CardTitle>
              {isActive && (
                <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 border-0 text-xs animate-pulse">
                  ACTIVE
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground mt-1 leading-relaxed">{phase.description}</p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Progress</span>
            <span className="text-xs font-mono font-bold text-primary">{phase.progress.toFixed(1)}%</span>
          </div>
          <Progress value={phase.progress} className="h-2" />
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-muted-foreground">{phase.itemsLearned.toLocaleString()} learned</span>
            <span className="text-xs font-mono text-muted-foreground">{phase.totalItems.toLocaleString()} total</span>
          </div>
        </div>

        {insights.length > 0 && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2 flex items-center gap-1">
              <BookOpen className="h-3 w-3" />
              Key Insights
            </p>
            <div className="space-y-1.5">
              {insights.map((insight, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                  <ArrowRight className="h-3 w-3 text-primary flex-shrink-0 mt-0.5" />
                  <span className="leading-relaxed">{insight}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

const PROGRESS_HISTORY = [
  { day: "Day 1", atomic: 20, elements: 15, bonding: 0, materials: 0, prediction: 0, discovery: 0 },
  { day: "Day 3", atomic: 60, elements: 45, bonding: 5, materials: 0, prediction: 0, discovery: 0 },
  { day: "Day 7", atomic: 100, elements: 85, bonding: 20, materials: 5, prediction: 0, discovery: 0 },
  { day: "Day 14", atomic: 100, elements: 100, bonding: 45, materials: 15, prediction: 2, discovery: 0 },
  { day: "Day 21", atomic: 100, elements: 100, bonding: 65, materials: 30, prediction: 5, discovery: 0 },
  { day: "Today", atomic: 100, elements: 100, bonding: 78, materials: 42, prediction: 8, discovery: 2 },
];

export default function ResearchPipeline() {
  const { data: phases, isLoading: phasesLoading } = useQuery<LearningPhase[]>({
    queryKey: ["/api/learning-phases"],
  });
  const { data: logs, isLoading: logsLoading } = useQuery<ResearchLog[]>({
    queryKey: ["/api/research-logs"],
  });

  const logsByPhase: Record<string, ResearchLog[]> = {};
  logs?.forEach(log => {
    if (!logsByPhase[log.phase]) logsByPhase[log.phase] = [];
    logsByPhase[log.phase].push(log);
  });

  const sourceColors: Record<string, string> = {
    "NIST": "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
    "Materials Project": "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
    "OQMD": "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
    "AFLOW": "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
    "Internal": "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
    "Internal ML Model": "bg-pink-100 text-pink-700 dark:bg-pink-950 dark:text-pink-300",
    "DFT Engine": "bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300",
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <FileText className="h-6 w-6 text-primary" />
          Research Pipeline
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Full learning trajectory from subatomic particles to novel material discovery — tracking every phase of scientific understanding.
        </p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-primary" />
            Learning Progress Over Time
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={PROGRESS_HISTORY} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
              <YAxis tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "6px", fontSize: "11px" }}
                formatter={(v: any) => [`${v}%`]}
              />
              <Area type="monotone" dataKey="atomic" name="Subatomic" stroke="#6366f1" fill="#6366f120" strokeWidth={1.5} />
              <Area type="monotone" dataKey="elements" name="Elements" stroke="#8b5cf6" fill="#8b5cf620" strokeWidth={1.5} />
              <Area type="monotone" dataKey="bonding" name="Bonding" stroke="#06b6d4" fill="#06b6d420" strokeWidth={1.5} />
              <Area type="monotone" dataKey="materials" name="Materials" stroke="#10b981" fill="#10b98120" strokeWidth={1.5} />
              <Area type="monotone" dataKey="prediction" name="Prediction" stroke="#f59e0b" fill="#f59e0b20" strokeWidth={1.5} />
              <Area type="monotone" dataKey="discovery" name="Discovery" stroke="#ef4444" fill="#ef444420" strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 justify-center">
            {[
              { label: "Subatomic", color: "#6366f1" },
              { label: "Elements", color: "#8b5cf6" },
              { label: "Bonding", color: "#06b6d4" },
              { label: "Materials", color: "#10b981" },
              { label: "Prediction", color: "#f59e0b" },
              { label: "Discovery", color: "#ef4444" },
            ].map(item => (
              <div key={item.label} className="flex items-center gap-1.5">
                <div className="h-2 w-4 rounded-full" style={{ background: item.color }} />
                <span className="text-xs text-muted-foreground">{item.label}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <h2 className="text-base font-semibold">Learning Phases</h2>
          {phasesLoading ? (
            <div className="space-y-4">
              {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-48" />)}
            </div>
          ) : (
            <div className="space-y-4">
              {phases?.map((phase, i) => <PhaseCard key={phase.id} phase={phase} index={i} />)}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <h2 className="text-base font-semibold">Research Activity Log</h2>
          <Card>
            <CardContent className="p-0">
              <ScrollArea className="h-[calc(100vh-280px)]">
                {logsLoading ? (
                  <div className="p-4 space-y-3">
                    {Array.from({ length: 10 }).map((_, i) => <Skeleton key={i} className="h-16" />)}
                  </div>
                ) : (
                  <div className="divide-y divide-border">
                    {logs?.map((log, i) => (
                      <div key={i} className="px-4 py-3 flex items-start gap-3" data-testid={`log-row-${i}`}>
                        <div className="mt-1 flex-shrink-0">
                          <Zap className="h-3.5 w-3.5 text-primary" />
                        </div>
                        <div className="flex-1 min-w-0 space-y-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-sm font-medium">{log.event}</span>
                            {log.dataSource && (
                              <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${sourceColors[log.dataSource] ?? "bg-muted text-muted-foreground"}`}>
                                {log.dataSource}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground leading-relaxed">{log.detail}</p>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono text-muted-foreground/70">{log.phase}</span>
                            {log.timestamp && (
                              <span className="text-xs text-muted-foreground/70 font-mono">
                                · {new Date(log.timestamp as unknown as string).toLocaleDateString()}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                    {logs?.length === 0 && (
                      <div className="py-8 text-center text-muted-foreground text-sm">No research logs yet</div>
                    )}
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

